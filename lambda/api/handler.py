"""Front-door API for the AI video pipeline.

Every route except the phone page (GET /m, GET /m/{file}) requires a request signed with an authorized SSH
ed25519 key (see `_authenticate`). Routes:
  POST /jobs                     submit a job -> {job_id, status, avatars, instance}
                                 (optional meta: out_name, avatar_name, krea_lora, source; returned by the GETs)
  GET  /m, GET /m/{file}         the phone page (static, no signature; docs/specs/2026-11-27-phone-ui.md)
  GET  /jobs/{job_id}            job status / result -> presigned download URLs when done
  POST /jobs/{job_id}/cancel     cancel a queued or running job
  GET  /jobs                     your latest 100 jobs: summary, status, run/wait seconds
  POST /jobs/{job_id}/rerun      queue a copy of one of your jobs (same params and seed)
  POST /uploads                  presigned PUT URL for an input video (describe) or image (avatar, photos)
  GET  /status                   GPU instance state, queue depth, uptime hold, deployed models
  POST /wake                     start the GPU instance ahead of a batch (clears an uptime hold)
  POST /worker/restart           ask the render worker to restart between jobs (pulls the latest worker code)
  GET  /avatars                  your avatars and which one is the default
  POST /avatars                  save an avatar from an uploaded image or a generated candidate
  PUT  /avatars/default          make an existing avatar the default
  GET  /avatars/{name}           one avatar with its images (1-hour preview links)
  POST /avatars/{name}/images    add images to an avatar
  DELETE /avatars/{name}/images/{n}  remove image n (1-based)
  PUT  /avatars/{name}/main      make image n the main image (position 1)
  GET  /avatars/{name}/history   earlier versions of an avatar
  POST /avatars/{name}/restore   bring back an earlier version
  DELETE /avatars/{name}         {loras?}: delete an avatar (images, history, media); loras: its LoRAs too
  GET  /datasets                 your learning datasets (from their manifests)
  GET  /datasets/{name}          one dataset's manifest
  POST /datasets/{name}/uploads  presigned PUT URLs for dataset files (max 100 per call)
  PUT  /datasets/{name}          commit the manifest: checks files, deletes unlisted ones
  DELETE /datasets/{name}        delete a dataset and all its files
  GET  /loras                    trained LoRAs: versions, steps, checkpoints
  PUT  /loras/{name}/current     {version, step?}: the version/checkpoint renders use
  GET  /datasets/{name}/captions training captions per item (current = matches the dataset)
  PUT  /datasets/{name}/captions {items: {item: caption}}: edit captions before training

Video and story jobs star the default avatar unless the request says
"avatar": "none", or names up to 3 avatars ("a,b" or ["a", "b"]).

Also invoked every minute by EventBridge with {"waker": true}: starts the GPU
instance when jobs are waiting and it is stopped, and stops it if it has been
running longer than MAX_UPTIME_HOURS (then holds it off until the next job or
wake, so a stuck job cannot keep restarting it).
"""

import base64
import hashlib
import json
import os
import re
import secrets
import time
import uuid

import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.config import Config
from botocore.exceptions import ClientError
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

BUCKET = os.environ["BUCKET"]
QUEUE_URL = os.environ["QUEUE_URL"]
JOBS_TABLE = os.environ["JOBS_TABLE"]
INSTANCE_ID = os.environ["INSTANCE_ID"]
# LoRA trainer: empty strings when training_enabled = false in Terraform.
TRAINER_INSTANCE_ID = os.environ.get("TRAINER_INSTANCE_ID", "")
TRAIN_QUEUE_URL = os.environ.get("TRAIN_QUEUE_URL", "")
TRAIN_MAX_UPTIME_HOURS = float(os.environ.get("TRAIN_MAX_UPTIME_HOURS", "24"))
NONCES_TABLE = os.environ["NONCES_TABLE"]
AUTHORIZED_KEYS = os.environ["AUTHORIZED_KEYS"]  # newline-separated authorized_keys lines
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "30"))
MAX_UPTIME_HOURS = float(os.environ.get("MAX_UPTIME_HOURS", "8"))

URL_TTL = 3600
# Big uploads go as S3 multipart: the client PUTs MULTIPART_PART-sized parts in parallel to presigned part
# URLs, then POST /uploads/complete (or /uploads/abort). Smaller files keep one presigned PUT.
MULTIPART_MIN = 64 * 1024 ** 2
MULTIPART_PART = 32 * 1024 ** 2
JOBS_INDEX = "client-created"
JOB_HISTORY_LIMIT = 100
# Model repos the box runs with (same Terraform variables as its env file).
MODELS = json.loads(os.environ.get("MODELS", "{}"))
MAX_PROMPT_CHARS = 2000
WRITE_WORDS = (100, 2000)  # `write` story length
MAX_STORY_CHARS = 20000
MAX_AVATAR_DESCRIPTION_CHARS = 1000
VIDEO_EXTS = {"mp4", "mov", "webm", "mkv"}
IMAGE_EXTS = {"png", "jpg", "jpeg", "webp"}
AVATAR_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
MAX_AVATARS_PER_JOB = 3
MAX_AVATAR_IMAGES = 12
EDIT_REFERENCES = 3  # images per avatar the edit model gets (it takes at most 3 per call)
# Wan2.2 A14B: 81 frames (5 s at 16 fps) is its native length; longer clips run out of GPU memory.
MAX_CLIP_SECONDS = 5
MAX_PHOTOS = 8
MAX_SCENE_CHARS = 500
MAX_KREA_PROMPT_CHARS = 2000  # Krea 2 photo prompts (the full description)
PHOTO_ASPECTS = ("portrait", "landscape", "square")
DATASET_NAME = AVATAR_NAME
DATASET_KINDS = ("character", "style", "motion", "place")
DATASET_EXTS = VIDEO_EXTS | IMAGE_EXTS
MAX_DATASET_FILE_BYTES = 2 * 1024 ** 3
MAX_UPLOADS_PER_CALL = 100
MAX_DATASET_PATH = 512
VOICE_NAME = re.compile(r"^[a-z]{2}_[a-z0-9]{1,20}$")
CONTROL_KEY = "_control"  # row in the jobs table holding the uptime hold flag
TRAIN_CONTROL_KEY = "_control_trainer"  # the same for the trainer
TRAIN_STEPS = (100, 3000, 1250)  # min, max, default steps per expert
TRAIN_SPEEDS = ("standard", "fast")  # 832x480 / 512x288 training resolution
MAX_CAPTION_CHARS = 500
SIG_NAMESPACE = b"vidgen-api"
MAX_CLOCK_SKEW = 300

# SigV4 presigned URLs: the legacy SigV2 default signs Content-Type, so a client
# PUT that adds one (urllib sends application/x-www-form-urlencoded) gets a 403.
s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
sqs = boto3.client("sqs")
ec2 = boto3.client("ec2")
jobs = boto3.resource("dynamodb").Table(JOBS_TABLE)
nonces = boto3.resource("dynamodb").Table(NONCES_TABLE)


# ---------------------------------------------------------------- helpers

def _resp(status, body):
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body, default=str),
    }


# ---------------------------------------------------------------- auth
#
# The client signs this canonical string with `ssh-keygen -Y sign -n vidgen-api`:
#
#   vidgen-v1\n<METHOD>\n<path>\n<unix timestamp>\n<nonce>\n<sha256 hex of body>
#
# and sends x-vidgen-timestamp, x-vidgen-nonce and x-vidgen-signature (the
# armored SSHSIG, base64-encoded onto one line). A request is accepted only if
# the signing key is in AUTHORIZED_KEYS, the signature verifies, the timestamp
# is within MAX_CLOCK_SKEW and the nonce has not been seen before.

def _read_string(buf, off):
    n = int.from_bytes(buf[off:off + 4], "big")
    off += 4
    if n > len(buf) - off:
        raise ValueError("truncated SSH string")
    return buf[off:off + n], off + n


def _ssh_string(b):
    return len(b).to_bytes(4, "big") + b


def _load_authorized_keys():
    """Map wire-format public key blob -> name (the key's comment)."""
    keys = {}
    for line in AUTHORIZED_KEYS.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "ssh-ed25519":
            blob = base64.b64decode(parts[1])
            name = parts[2] if len(parts) > 2 else hashlib.sha256(blob).hexdigest()[:16]
            keys[blob] = name
    return keys


_authorized = _load_authorized_keys()


def _verify_sshsig(armored, message):
    """Verify an OpenSSH SSHSIG over message. Returns the signer's key blob."""
    body = "".join(l for l in armored.strip().splitlines() if not l.startswith("-----"))
    raw = base64.b64decode(body)
    if raw[:6] != b"SSHSIG" or int.from_bytes(raw[6:10], "big") != 1:
        raise ValueError("not an SSHSIG v1 signature")
    pubkey, off = _read_string(raw, 10)
    namespace, off = _read_string(raw, off)
    reserved, off = _read_string(raw, off)
    hash_alg, off = _read_string(raw, off)
    sig_blob, off = _read_string(raw, off)

    if namespace != SIG_NAMESPACE:
        raise ValueError("wrong signature namespace")
    if hash_alg not in (b"sha256", b"sha512"):
        raise ValueError("unsupported hash")
    if pubkey not in _authorized:
        raise ValueError("key not authorized")

    key_type, o = _read_string(pubkey, 0)
    key_bytes, _ = _read_string(pubkey, o)
    sig_type, o = _read_string(sig_blob, 0)
    sig, _ = _read_string(sig_blob, o)
    if key_type != b"ssh-ed25519" or sig_type != b"ssh-ed25519":
        raise ValueError("only ssh-ed25519 keys are supported")

    digest = hashlib.new(hash_alg.decode(), message).digest()
    signed = (b"SSHSIG" + _ssh_string(namespace) + _ssh_string(reserved)
              + _ssh_string(hash_alg) + _ssh_string(digest))
    Ed25519PublicKey.from_public_bytes(key_bytes).verify(sig, signed)
    return pubkey


def _authenticate(event, headers, raw_body):
    """Return the client name for a validly signed request, else None."""
    try:
        ts = int(headers["x-vidgen-timestamp"])
        nonce = headers["x-vidgen-nonce"]
        armored = base64.b64decode(headers["x-vidgen-signature"]).decode()
    except (KeyError, ValueError, UnicodeDecodeError):
        return None
    if abs(time.time() - ts) > MAX_CLOCK_SKEW or not (16 <= len(nonce) <= 64):
        return None

    message = "\n".join([
        "vidgen-v1",
        event["requestContext"]["http"]["method"],
        event["rawPath"],
        str(ts),
        nonce,
        hashlib.sha256(raw_body).hexdigest(),
    ]).encode()
    try:
        pubkey = _verify_sshsig(armored, message)
    except (ValueError, InvalidSignature) as e:
        print(json.dumps({"event": "auth_failed", "reason": str(e) or type(e).__name__}))
        return None

    # Single use: a captured request cannot be replayed.
    try:
        nonces.put_item(
            Item={"nonce": nonce, "expires_at": ts + 2 * MAX_CLOCK_SKEW},
            ConditionExpression="attribute_not_exists(nonce)",
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            print(json.dumps({"event": "auth_failed", "reason": "replayed nonce"}))
            return None
        raise
    return _authorized[pubkey]


def _instance(instance_id=INSTANCE_ID):
    return ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]


def _instance_state():
    return _instance()["State"]["Name"]


def _hold(key=CONTROL_KEY):
    return bool((jobs.get_item(Key={"job_id": key}).get("Item") or {}).get("hold"))


def _set_hold(value, key=CONTROL_KEY):
    jobs.put_item(Item={"job_id": key, "client": "_system", "hold": value})


def _queue_depth(queue_url=QUEUE_URL):
    attrs = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"],
    )["Attributes"]
    return int(attrs["ApproximateNumberOfMessages"]) + int(attrs["ApproximateNumberOfMessagesNotVisible"])


def _ensure_started(instance_id=INSTANCE_ID):
    """Start the instance if stopped. Returns its state after the attempt.

    A stopping instance cannot be started yet; the waker retries next minute.
    Spot capacity errors are reported, not raised, for the same reason.
    """
    state = _instance(instance_id)["State"]["Name"]
    if state != "stopped":
        return state
    try:
        ec2.start_instances(InstanceIds=[instance_id])
        return "pending"
    except ClientError as e:
        code = e.response["Error"]["Code"]
        print(json.dumps({"event": "start_failed", "code": code, "msg": str(e)}))
        return f"stopped ({code}, will retry)"


# ---------------------------------------------------------------- validation

def _ext(key):
    return key.rsplit(".", 1)[-1].lower() if "." in key else ""


def _own_upload(client, key, allowed_exts):
    if not key.startswith(f"uploads/{client}/") or _ext(key) not in allowed_exts:
        raise ValueError(f"upload_key from POST /uploads is required ({', '.join(sorted(allowed_exts))})")
    return key


def _voice(voice):
    if not VOICE_NAME.match(voice):
        raise ValueError("voice must look like af_heart, am_adam, bf_emma (Kokoro voice names)")
    return voice


MAX_TAKE_SECONDS = 60  # video: longer than MAX_CLIP_SECONDS chains 5 s segments
MAX_VARIATIONS = 4


DRAFT_SIZE = (832, 480)  # draft quality: render small, then RIFE + 2x upscale -> 1664x960


REALISM_LOOKS = ("pro", "phone", "candid", "portrait", "film")
SHOTS = ("stream", "pov")  # videos: a fixed webcam with a ring light | a handheld phone seen by its holder
# Video models (docs/specs/2026-11-26-helios-default.md). helios (the default): one take of any length in one
# run, one prompt, at 640x384 and 24 fps. wan: 5 s segments chained, with beats, plans and face re-anchoring.
VIDEO_MODELS = ("helios", "wan")
HELIOS_SIZE, HELIOS_FPS, HELIOS_SECONDS = (640, 384), 24, 30
# Krea 2 Turbo (text-to-image, own venv on the render box, content filter on every image).
AVATAR_MODELS = ("wan", "krea2")      # avatar candidates
PHOTO_MODELS = ("qwen", "krea2")      # qwen = Qwen-Image-Edit with the person; krea2 = no person
IMAGE_MODELS = ("auto", "wan", "krea2")  # first frame; auto = krea2 when it can be made, else wan


def _realism_params(body):
    """look preset plus natural_skin / candid / grain switches (video, story, photos)."""
    look = body.get("look") or "pro"
    if look not in REALISM_LOOKS:
        raise ValueError(f"look must be one of: {', '.join(REALISM_LOOKS)}")
    return {"look": look, **{k: body.get(k) is True for k in ("natural_skin", "candid", "grain")}}


def _video_model(body, allowed=True):
    """The video model of a `video` job: helios unless wan is asked for. Other job types refuse the key."""
    model = body.get("video_model")
    if not allowed:
        if model not in (None, ""):
            raise ValueError("video_model (helios, wan) is for videos; stories render with wan")
        return None
    model = "helios" if model in (None, "") else model
    if not isinstance(model, str) or model not in VIDEO_MODELS:
        raise ValueError(f"video_model must be one of: {', '.join(VIDEO_MODELS)}")
    return model


def _shot_param(body, allowed=True):
    """{"shot": "stream" | "pov"} for a video, {} when none is asked for. One setting, so the two can't be
    combined. Other job types refuse it instead of ignoring it."""
    shot = body.get("shot")
    if shot in (None, ""):
        return {}
    if not allowed:
        raise ValueError("shot (stream, pov) is for videos")
    if not isinstance(shot, str) or shot not in SHOTS:
        raise ValueError(f"shot must be one of: {', '.join(SHOTS)} (one at a time), or left out")
    return {"shot": shot}


def _enhance_params(body):
    """fps 32 (RIFE frame interpolation, default) or 16 (as rendered); upscale = 2x Real-ESRGAN.
    quality "draft" renders at 832x480 and upscales: about 2-3x faster, softer detail; "best" makes
    avatar keyframes with the full 40-step edit model (Wan itself keeps Lightning)."""
    fps = int(body.get("fps") or 32)
    if fps not in (16, 32):
        raise ValueError("fps must be 16 or 32")
    quality = body.get("quality") or "standard"
    if quality not in ("standard", "draft", "best"):
        raise ValueError("quality must be standard, draft or best")
    image_model = body.get("image_model") or "auto"
    if image_model not in IMAGE_MODELS:
        raise ValueError(f"image_model must be one of: {', '.join(IMAGE_MODELS)}")
    out = {"fps": fps, "upscale": bool(body.get("upscale", False)), "quality": quality, "image_model": image_model,
           **_realism_params(body)}
    if quality == "draft":
        out.update(width=DRAFT_SIZE[0], height=DRAFT_SIZE[1], upscale=True)
    return out


MAX_BEATS = 12
MAX_BEAT_CHARS = 500
SEGMENT_SECONDS = 5


def _parse_beats(beats, seconds):
    """A long take's timeline -> [{"at": s, "prompt": text}] or None (no beats).

    String form "0: walks in; 10: sits down" (`;` or newline separated, `SECONDS: text`), or a list of
    {"at", "prompt"}. Rules: only for takes over 5 s, first beat at 0, strictly increasing, each before the
    end, 1-12 beats, each text 1-500 chars."""
    if beats is None or (isinstance(beats, str) and not beats.strip()) or beats == []:
        return None
    if isinstance(beats, str):
        items = []
        for chunk in re.split(r"[;\n]", beats):
            if not chunk.strip():
                continue
            at, sep, text = chunk.partition(":")
            items.append({"at": at.strip(), "prompt": text.strip()} if sep else {"at": None, "prompt": chunk})
    elif isinstance(beats, list):
        items = [b if isinstance(b, dict) else {"at": None, "prompt": ""} for b in beats]
    else:
        raise ValueError("beats must be a string or a list of {at, prompt}")
    out = []
    for b in items:
        at = b.get("at")
        if isinstance(at, str) and re.fullmatch(r"\d{1,3}", at):
            at = int(at)
        if not isinstance(at, int) or isinstance(at, bool):
            raise ValueError("each beat must be 'SECONDS: text' (whole seconds), e.g. '0: walks in; 10: sits down'")
        text = b.get("prompt")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= MAX_BEAT_CHARS:
            raise ValueError(f"each beat text must be 1-{MAX_BEAT_CHARS} chars")
        out.append({"at": at, "prompt": text.strip()})
    if seconds <= SEGMENT_SECONDS:
        raise ValueError(f"beats work with takes over {SEGMENT_SECONDS} s")
    if not 1 <= len(out) <= MAX_BEATS:
        raise ValueError(f"beats: 1-{MAX_BEATS} beats")
    if out[0]["at"] != 0:
        raise ValueError("beats: the first beat must be at 0")
    if any(b["at"] <= a["at"] for a, b in zip(out, out[1:])):
        raise ValueError("beats: times must be strictly increasing")
    if out[-1]["at"] >= seconds:
        raise ValueError("beats: each beat's time must be less than seconds")
    # Segments start every 5 s; a beat takes effect at the first segment starting at or after it.
    count = -(-seconds // SEGMENT_SECONDS)
    starts = [-(-b["at"] // SEGMENT_SECONDS) for b in out]
    for b, k in zip(out, starts):
        if k >= count:
            raise ValueError(f"beats: the beat at {b['at']} s starts no segment (segments start every "
                             f"{SEGMENT_SECONDS} s; the last at {(count - 1) * SEGMENT_SECONDS} s)")
    if any(a == b for a, b in zip(starts, starts[1:])):
        raise ValueError(f"beats: two beats fall in the same segment (segments start every {SEGMENT_SECONDS} s: "
                         "0, 5, 10, ...); use times that are multiples of 5")
    return out


def _validate(client, body):
    """Return (job_type, params) or raise ValueError."""
    job_type = body.get("type")

    if job_type == "video":
        prompt = (body.get("prompt") or "").strip()
        if not prompt or len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt is required and must be <= {MAX_PROMPT_CHARS} chars")
        video_model = _video_model(body)
        helios = video_model == "helios"
        seconds = int(body["seconds"]) if body.get("seconds") is not None else (HELIOS_SECONDS if helios else 5)
        if not 1 <= seconds <= MAX_TAKE_SECONDS:
            raise ValueError(f"seconds must be 1-{MAX_TAKE_SECONDS} (over {MAX_CLIP_SECONDS} s is made of chained segments)")
        variations = int(body.get("variations") or 1)
        if not 1 <= variations <= MAX_VARIATIONS:
            raise ValueError(f"variations must be 2-{MAX_VARIATIONS}")
        if variations > 1 and seconds > MAX_CLIP_SECONDS and not helios:  # a Wan take over 5 s is a chain
            raise ValueError(f"variations work with clips up to {MAX_CLIP_SECONDS} s")
        if helios:  # one prompt, one take: nothing to plan, time or re-anchor
            for key, what in (("beats", "beats"), ("plan_only", "plan_only (planning)"), ("anchor_every", "anchor_every")):
                if body.get(key) not in (None, "", False):
                    raise ValueError(f"{what} needs video model wan (helios renders the whole take from one prompt)")
        beats = _parse_beats(body.get("beats"), seconds)
        plan_only = body.get("plan_only") is True
        if plan_only:  # the first frame and the planner's beats, to review before rendering
            if seconds <= 15:
                raise ValueError("plan_only works for takes over 15 s (shorter takes keep one prompt)")
            if beats:
                raise ValueError("plan_only writes the beats; leave beats out")
        anchor = body.get("anchor_every")
        if anchor is not None:  # re-anchor the avatar's face every N segments (one avatar; checked in create_job)
            if not isinstance(anchor, int) or isinstance(anchor, bool) or not 2 <= anchor <= 6:
                raise ValueError("anchor_every must be 2-6 (segments between re-anchored frames)")
            if seconds <= 10:
                raise ValueError("anchor_every works with takes over 10 s")
        return job_type, {
            **({"beats": beats} if beats else {}),
            **({"anchor_every": anchor} if anchor is not None else {}),
            **({"plan_only": True} if plan_only else {}),
            "prompt": prompt,
            "negative_prompt": body.get("negative_prompt", ""),
            "seconds": seconds,
            "width": int(body.get("width", 1280)),
            "height": int(body.get("height", 704)),
            "seed": body.get("seed"),
            "sound": bool(body.get("sound", True)),
            "variations": variations,
            **_enhance_params(body),
            **_shot_param(body),
            "video_model": video_model,
            # Helios renders at one size and speed, whatever was sent (a CLI always sends its Wan defaults).
            **({"width": HELIOS_SIZE[0], "height": HELIOS_SIZE[1], "fps": HELIOS_FPS, "quality": "standard"
                if (body.get("quality") or "standard") == "draft" else (body.get("quality") or "standard"),
                "upscale": bool(body.get("upscale", False))} if helios else {}),
        }

    if job_type == "describe":
        return job_type, {
            "upload_key": _own_upload(client, body.get("upload_key", ""), VIDEO_EXTS),
            "instructions": body.get("instructions", ""),
        }

    if job_type == "story":
        _shot_param(body, allowed=False)
        _video_model(body, allowed=False)
        story = (body.get("story") or "").strip()
        if not story or len(story) > MAX_STORY_CHARS:
            raise ValueError(f"story is required and must be <= {MAX_STORY_CHARS} chars")
        return job_type, {
            "story": story,
            "max_shots": max(1, min(int(body.get("max_shots", 6)), 12)),
            "seconds_per_shot": max(2, min(int(body.get("seconds_per_shot", 5)), MAX_CLIP_SECONDS)),
            "style": body.get("style", ""),
            "subtitles": bool(body.get("subtitles", True)),
            "narration": bool(body.get("narration", True)),
            "sound": bool(body.get("sound", True)),
            "voice": _voice(body.get("voice") or "af_heart"),
            **_enhance_params(body),
        }

    if job_type == "avatar":
        description = (body.get("description") or "").strip()
        if not description or len(description) > MAX_AVATAR_DESCRIPTION_CHARS:
            raise ValueError(f"description is required and must be <= {MAX_AVATAR_DESCRIPTION_CHARS} chars")
        model = body.get("model") or "wan"
        if model not in AVATAR_MODELS:
            raise ValueError(f"model must be one of: {', '.join(AVATAR_MODELS)}")
        return job_type, {
            "description": description,
            "candidates": max(1, min(int(body.get("candidates", 4)), 6)),
            "seed": body.get("seed"),
            "model": model,
        }

    if job_type == "photos":
        return job_type, _validate_photos(client, body)

    if job_type == "write":
        premise = (body.get("premise") or "").strip()
        if not premise or len(premise) > MAX_PROMPT_CHARS:
            raise ValueError(f"premise is required and must be <= {MAX_PROMPT_CHARS} chars")
        words = int(body.get("words") or 800)
        if not WRITE_WORDS[0] <= words <= WRITE_WORDS[1]:
            raise ValueError(f"words must be {WRITE_WORDS[0]}-{WRITE_WORDS[1]}")
        genre = str(body.get("genre") or "").strip()
        if len(genre) > 200:
            raise ValueError("genre must be <= 200 chars")
        seed = body.get("seed")
        return job_type, {"premise": premise, "words": words, "genre": genre,
                          "seed": None if seed is None else int(seed)}

    if job_type == "train":
        if not TRAIN_QUEUE_URL:
            raise ValueError("the trainer is not deployed (set training_enabled = true in Terraform)")
        name = (body.get("dataset") or "").strip().lower()
        manifest = _s3_json(_dataset_prefix(client, name) + "manifest.json")
        if not manifest:
            raise ValueError(f"dataset '{name}' not found; upload it first (vidgen dataset add)")
        lo, hi, default = TRAIN_STEPS
        steps = int(body.get("steps") or default)
        if not lo <= steps <= hi:
            raise ValueError(f"steps must be {lo}-{hi} per expert")
        speed = body.get("speed") or "standard"
        if speed not in TRAIN_SPEEDS:
            raise ValueError(f"speed must be one of: {', '.join(TRAIN_SPEEDS)}")
        return job_type, {"dataset": name, "steps": steps, "kind": manifest["kind"], "trigger": manifest["trigger"],
                          "speed": speed, "captions_only": bool(body.get("captions_only")),
                          "files": manifest["file_count"], "dataset_updated_at": manifest["updated_at"],
                          "fingerprint": hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()[:16]}

    if job_type == "lora_preview":
        return job_type, _validate_lora_preview(client, body)

    raise ValueError("type must be one of: video, describe, story, write, avatar, photos, train, lora_preview")


MAX_PREVIEW_STEPS = 6
PREVIEW_SCENE = "a person in a bright, simple room, medium shot, steady camera"


def _validate_lora_preview(client, body):
    """Short text-to-video clips of a LoRA's checkpoints, all with one prompt and seed, plus a
    baseline without the LoRA, so checkpoints can be compared before picking one."""
    name = str(body.get("name") or "").strip().lower()
    if not DATASET_NAME.match(name):
        raise ValueError("name: the LoRA to preview")
    pinned = body.get("version") not in (None, "")
    version, meta, _ = _lora_version(client, name, body.get("version") if pinned else None)
    if not meta:
        raise ValueError(f"LoRA '{name}' not found; see vidgen lora list")
    if "krea2" in (meta.get("models") or []):
        raise ValueError(f"'{name}' is a Krea 2 LoRA (images); previews are video clips of Wan LoRAs. "
                         "Try it with vidgen photos --model krea2 --lora NAME")
    available = _checkpoint_steps(meta)
    final = int(meta.get("steps_per_expert") or 0)
    steps = body.get("steps") or (available[-5:] or [final])
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_PREVIEW_STEPS:
        raise ValueError(f"steps: 1-{MAX_PREVIEW_STEPS} checkpoint steps")
    clips = []
    for step in sorted({int(x) for x in steps}):
        # The final step without a saved checkpoint (short runs) previews the final weights.
        high, low = _lora_keys(client, name, version, meta, None if step == final and step not in available else step)
        clips.append({"step": step, "high_key": high, "low_key": low})
    scene = (body.get("prompt") or "").strip() or PREVIEW_SCENE
    if len(scene) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt must be <= {MAX_PROMPT_CHARS} chars")
    weight = float(body.get("weight") or 1.0)
    if not LORA_WEIGHT[0] <= weight <= LORA_WEIGHT[1]:
        raise ValueError(f"weight must be {LORA_WEIGHT[0]}-{LORA_WEIGHT[1]}")
    seed = body.get("seed")
    return {"name": name, "version": int(version), "trigger": meta.get("trigger") or "", "clips": clips,
            "prompt": scene, "weight": weight, "seconds": 2, "width": 832, "height": 480,
            "seed": int(seed) if seed not in (None, "") else int.from_bytes(os.urandom(4), "big")}


PHOTO_SHOT_FIELDS = ("where", "wearing", "doing", "camera")


def _shot_fields(shot):
    """A photos scene given as fields (--where/--wearing/--doing/--camera): at least one, each 1-N chars."""
    if not isinstance(shot, dict) or not shot or set(shot) - set(PHOTO_SHOT_FIELDS):
        raise ValueError(f"shot takes any of: {', '.join(PHOTO_SHOT_FIELDS)}")
    out = {}
    for k, v in shot.items():
        v = str(v or "").strip()
        if len(v) > MAX_SCENE_CHARS:
            raise ValueError(f"shot {k} must be at most {MAX_SCENE_CHARS} chars")
        if v:
            out[k] = v
    if not out:
        raise ValueError(f"shot needs at least one of: {', '.join(PHOTO_SHOT_FIELDS)}")
    return out


def _series(series, scenes):
    """{"where", "wearing"?, "doing"?: [..], "anchor"?: bool}: every photo in one place and outfit."""
    if series is None:
        return None
    if not isinstance(series, dict) or set(series) - {"where", "wearing", "doing", "anchor"}:
        raise ValueError("series takes where, wearing, doing (a list) and anchor")
    if scenes:
        raise ValueError("give series or scenes, not both")
    out = {"anchor": series.get("anchor", True) is not False}
    for k in ("where", "wearing"):
        v = str(series.get(k) or "").strip()
        if len(v) > MAX_SCENE_CHARS:
            raise ValueError(f"series {k} must be at most {MAX_SCENE_CHARS} chars")
        if v:
            out[k] = v
    if "where" not in out:
        raise ValueError("a series needs where (the place)")
    doing = series.get("doing") or []
    if not isinstance(doing, list):
        raise ValueError("series doing must be a list of actions, one per photo")
    doing = [str(d).strip() for d in doing if str(d).strip()]
    if any(len(d) > MAX_SCENE_CHARS for d in doing):
        raise ValueError(f"each series action must be at most {MAX_SCENE_CHARS} chars")
    if doing:
        out["doing"] = doing
    return out


def _validate_photos(client, body):
    _shot_param(body, allowed=False)
    _video_model(body, allowed=False)
    scenes = body.get("scenes") or []
    if not isinstance(scenes, list) or len(scenes) > MAX_PHOTOS:
        raise ValueError(f"scenes must be a list of at most {MAX_PHOTOS}")
    clean = []
    # A Krea 2 prompt is the whole image description (often exact, see raw), so it may be longer.
    krea = (body.get("model") or ("krea2" if body.get("avatar") == "none" else "qwen")) == "krea2"
    for s in scenes:
        if not isinstance(s, dict) or len(s) != 1 or not ({"prompt", "location", "shot"} & s.keys()):
            raise ValueError('each scene must be {"prompt": str}, {"location": str} or {"shot": {...}}')
        (kind, text), = s.items()
        if kind == "shot":
            clean.append({"shot": _shot_fields(text)})
            continue
        text = str(text).strip()
        limit = MAX_KREA_PROMPT_CHARS if krea and kind == "prompt" else MAX_SCENE_CHARS
        if not text or len(text) > limit:
            raise ValueError(f"each {'Krea 2 prompt' if limit == MAX_KREA_PROMPT_CHARS else 'scene text'} "
                             f"must be 1-{limit} chars")
        clean.append({kind: text})
    series = _series(body.get("series"), clean)
    count = int(body.get("count") or len(clean) or 4)
    if not 1 <= count <= MAX_PHOTOS or count < len(clean):
        raise ValueError(f"count must be 1-{MAX_PHOTOS} and at least the number of scenes")
    if series and (count < 2 or len(series.get("doing", [])) > count):
        raise ValueError("a series needs count 2-8 and at most count actions")
    quality = body.get("quality") or "standard"
    if quality not in ("standard", "best"):
        raise ValueError("quality must be standard or best")
    aspect = body.get("aspect") or "portrait"
    if aspect not in PHOTO_ASPECTS:
        raise ValueError(f"aspect must be one of: {', '.join(PHOTO_ASPECTS)}")
    model = body.get("model") or ("krea2" if body.get("avatar") == "none" else "qwen")
    if model not in PHOTO_MODELS:
        raise ValueError(f"model must be one of: {', '.join(PHOTO_MODELS)}")
    raw = body.get("raw", False)
    if not isinstance(raw, bool):
        raise ValueError("raw must be true or false")
    if raw and model != "krea2":
        raise ValueError("raw (the prompt exactly as typed) is for Krea 2 photos: model krea2, avatar none")
    if raw and (not clean or series):
        raise ValueError("raw sends your own prompts as typed: give at least one prompt (extra photos repeat them)")
    if raw and any("shot" in s for s in clean):
        raise ValueError("raw sends typed text as-is: use a prompt, not the where/wearing/doing/camera fields")
    if model == "krea2":
        if body.get("upload_key"):
            raise ValueError("Krea 2 makes photos from text; use model qwen to place an uploaded photo")
        avatar_lora = None
        if body.get("avatar") not in (None, "", "none"):  # a person in Krea photos: only through their Krea LoRA
            avatar_name = str(body["avatar"]).strip().lower()
            avatar = _get_avatar(client, avatar_name) if AVATAR_NAME.match(avatar_name) else None
            if not avatar:
                raise ValueError(f"avatar '{avatar_name}' not found")
            kl = avatar.get("krea_lora") or {}
            if not kl.get("ready"):
                raise ValueError(f"avatar '{avatar_name}' has no Krea 2 LoRA yet (state {kl.get('state', 'none')}); "
                                 "use model qwen, or train it: vidgen avatar krea-train --name " + avatar_name)
            avatar_lora = {"name": kl["ready"]["lora"], "version": kl["ready"]["version"]}
        if series:
            raise ValueError("a series needs an avatar (photo 1 anchors the person); Krea 2 photos take scenes")
        if quality == "best":
            raise ValueError("best quality is for the Qwen photo model; Krea 2 Turbo has one speed")
        return {"subject": "avatar" if avatar_lora else "none", "model": "krea2", "scenes": clean, "count": count,
                "aspect": aspect, "seed": body.get("seed"), "quality": "standard", "raw": raw,
                **({"avatar": avatar_name, "avatar_lora": avatar_lora, "avatar_trigger": avatar_name}
                   if avatar_lora else {}), **_realism_params(body)}
    if body.get("avatar") == "none":
        raise ValueError("photos with model qwen need a person: an avatar or upload_key")
    if body.get("upload_key") and body.get("avatar"):
        raise ValueError("give upload_key or avatar, not both")
    if body.get("upload_key"):
        reference_key, subject = _own_upload(client, body["upload_key"], IMAGE_EXTS), "upload"
        reference_keys = [reference_key]
    else:
        avatars = _resolve_avatars(client, body.get("avatar"))
        if len(avatars) != 1:
            raise ValueError("give upload_key, or one avatar name (or set a default avatar)")
        reference_key, subject = avatars[0]["reference_key"], avatars[0]["name"]
        reference_keys = avatars[0]["reference_keys"]
    return {"reference_key": reference_key, "reference_keys": reference_keys, "subject": subject, "model": "qwen",
            "scenes": clean, "count": count, "aspect": aspect, "seed": body.get("seed"),
            "quality": quality, **_realism_params(body), **({"series": series} if series else {})}


# ---------------------------------------------------------------- avatars
#
# S3 layout, per client:
#   avatars/<client>/<name>/avatar.json      {name, description, created_at, reference_key,
#                                             images: [{key, added_at}, ...]}  (image 1 = main)
#   avatars/<client>/<name>/images/<id>.<ext>
#   avatars/<client>/<name>/history/<created_at>/...   earlier versions, same layout
#   avatars/<client>/_default.json           {name}
# Avatars saved before multi-image support have no `images`: they read as a one-image set
# [reference_key] (avatars/<client>/<name>/reference.<ext>). Images dropped from the set stay
# in S3 (history keeps a copy anyway); DELETE /avatars/{name} removes the whole folder.

def _s3_json(key):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def _put_json(key, data):
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(data).encode(), ContentType="application/json")


def _get_avatar(client, name):
    return _s3_json(f"avatars/{client}/{name}/avatar.json")


def _default_avatar_name(client):
    return (_s3_json(f"avatars/{client}/_default.json") or {}).get("name")


def _resolve_avatars(client, requested):
    """Avatar dicts for a video/story job (first = star), [] for none.

    requested: None (use the default), "none", "a,b" or ["a", "b"].
    Raises ValueError for unknown names or more than MAX_AVATARS_PER_JOB."""
    if requested == "none":
        return []
    if not requested:
        default = _default_avatar_name(client)
        names = [default] if default else []
    else:
        names = requested if isinstance(requested, list) else str(requested).split(",")
        names = list(dict.fromkeys(n.strip().lower() for n in names if n.strip()))
    if len(names) > MAX_AVATARS_PER_JOB:
        raise ValueError(f"at most {MAX_AVATARS_PER_JOB} avatars per job")
    avatars = []
    for name in names:
        avatar = _get_avatar(client, name)
        if avatar is None:
            raise ValueError(f"avatar '{name}' not found")
        avatar["reference_keys"] = _image_keys(avatar)[:EDIT_REFERENCES]
        avatars.append(avatar)
    return avatars


def _image_keys(avatar):
    """The avatar's image keys in order; older one-image avatars give [reference_key]."""
    return [i["key"] for i in avatar.get("images") or []] or [avatar["reference_key"]]


def _image_sources(client, body):
    """Upload keys from {"upload_keys": [...]} or {"upload_key": str}; raises ValueError."""
    keys = body.get("upload_keys")
    if keys is None:
        keys = [body["upload_key"]] if body.get("upload_key") else []
    if not isinstance(keys, list) or not 1 <= len(keys) <= MAX_AVATAR_IMAGES:
        raise ValueError(f"give 1-{MAX_AVATAR_IMAGES} image uploads (upload_keys)")
    return [_own_upload(client, str(k), IMAGE_EXTS) for k in keys]


def list_avatars(client):
    prefixes = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"avatars/{client}/", Delimiter="/").get("CommonPrefixes", [])
    avatars = []
    for p in prefixes:
        a = _s3_json(p["Prefix"] + "avatar.json")
        if a:
            avatars.append(dict({k: a[k] for k in ("name", "description", "created_at")},
                                images=len(_image_keys(a)), krea_lora=a.get("krea_lora") or {"state": "none"},
                                media=a.get("media") or {"state": "none"},
                                wan_lora=a.get("wan_lora") or {"state": "none"}))
    return _resp(200, {"default": _default_avatar_name(client), "avatars": avatars})


def create_avatar(client, body):
    """Save an avatar from {"upload_key"} or {"job_id", "candidate"}; same name replaces it."""
    name = (body.get("name") or "").strip().lower()
    if not AVATAR_NAME.match(name):
        return _resp(400, {"error": "name must be 1-32 chars: a-z, 0-9, - or _"})
    description = (body.get("description") or "").strip()

    if body.get("upload_keys") is not None or body.get("upload_key"):
        try:
            sources = _image_sources(client, body)
        except ValueError as e:
            return _resp(400, {"error": str(e)})
    elif body.get("job_id"):
        job = jobs.get_item(Key={"job_id": body["job_id"]}).get("Item")
        if not job or job.get("client") != client or job.get("type") != "avatar" or job.get("status") != "done":
            return _resp(400, {"error": "job_id must be one of your finished avatar jobs"})
        result = json.loads(job["result"])
        source = result.get(f"candidate_{int(body.get('candidate', 0))}_key")
        if not source:
            return _resp(400, {"error": "no such candidate in that job"})
        sources = [source]
        description = description or result.get("description", "")
    else:
        return _resp(400, {"error": "give upload_key, or job_id + candidate"})

    if len(description) > MAX_AVATAR_DESCRIPTION_CHARS:
        return _resp(400, {"error": f"description must be <= {MAX_AVATAR_DESCRIPTION_CHARS} chars"})
    krea_lora = body.get("krea_lora", True)
    if not isinstance(krea_lora, bool):
        return _resp(400, {"error": "krea_lora must be true or false"})
    wan_lora = body.get("wan_lora", True)
    if not isinstance(wan_lora, bool):
        return _resp(400, {"error": "wan_lora must be true or false"})

    _archive_avatar(client, name)
    _write_avatar(client, name, sources, description)
    av = _get_avatar(client, name)
    if (av or {}).get("wan_lora"):  # a new or replaced avatar: never the previous person's Wan LoRA
        _set_wan_lora(client, name, {k: v for k, v in av["wan_lora"].items() if k != "ready"})  # version stays: monotonic
    if not wan_lora:
        _set_wan_lora(client, name, dict(_wan_lora_of(client, name), state="skipped"))
    if krea_lora:
        try:
            kl = _queue_krea_lora(client, name, wan=wan_lora)
        except (ClientError, ValueError) as e:  # the avatar is saved either way
            kl = _set_krea_lora(client, name, dict(_krea_lora_of(client, name), state="failed",
                                                   error=f"could not queue the Krea LoRA build: {e}"))
            _set_media(client, name, {"state": "failed", "error": f"could not queue the avatar media: {e}"})
            if wan_lora:
                _set_wan_lora(client, name, dict(_wan_lora_of(client, name), state="failed",
                                                 error=f"could not queue the Wan LoRA build: {e}"))
    else:
        kl = _set_krea_lora(client, name, dict(_krea_lora_of(client, name), state="skipped"))
        try:  # no Krea LoRA, but every avatar gets its photos and clips (and, by default, a Wan LoRA)
            _queue_avatar_media(client, name, wan=wan_lora)
        except (ClientError, ValueError) as e:
            _set_media(client, name, {"state": "failed", "error": f"could not queue the avatar media: {e}"})
            if wan_lora:
                _set_wan_lora(client, name, dict(_wan_lora_of(client, name), state="failed",
                                                 error=f"could not queue the Wan LoRA build: {e}"))

    # make_default: true/false as asked; omitted = only if there is no default yet.
    make_default = body.get("make_default")
    is_default = bool(make_default) if make_default is not None else not _default_avatar_name(client)
    if is_default:
        _put_json(f"avatars/{client}/_default.json", {"name": name})
    saved = _get_avatar(client, name) or {}
    return _resp(200, {"name": name, "default": is_default, "images": len(sources), "krea_lora": kl,
                       "media": saved.get("media"), "wan_lora": saved.get("wan_lora")})


# ---------------------------------------------------------------- Krea 2 LoRA per avatar
#
# Every new avatar (unless krea_lora: false) gets a Krea 2 character LoRA: an avatar_photoset job on the
# render box makes and filters a photo set and queues a krea2 train job; the trainer registers the LoRA
# and marks it ready on avatar.json. The trigger word is the avatar's name.
# docs/specs/2026-11-24-krea-avatar-lora.md

AVATAR_DATASET_PHOTOS = 24


def _avatar_lora_name(name):
    """LoRA name for an avatar: avatar-<name>, shortened with a hash when it would pass 32 characters."""
    full = f"avatar-{name}"
    return full if len(full) <= 32 else f"avatar-{name[:16]}-{hashlib.sha1(name.encode()).hexdigest()[:6]}"


KREA_LORA_BUSY = ("queued", "photos", "training")


def _krea_lora_of(client, name):
    """The avatar's krea_lora record without the per-build fields (keeps the last ready link)."""
    kl = (_get_avatar(client, name) or {}).get("krea_lora") or {}
    return {k: kl[k] for k in ("ready", "lora", "trigger") if k in kl}


def _set_krea_lora(client, name, krea_lora):
    avatar = _get_avatar(client, name)
    avatar["krea_lora"] = krea_lora
    _put_json(f"avatars/{client}/{name}/avatar.json", avatar)
    return krea_lora


# ---------------------------------------------------------------- Wan LoRA per avatar
# The avatar_photoset job also writes a Wan dataset (photos + clips) and queues a Wan train job; the trainer
# registers it as avatar-<name>-wan and marks wan_lora ready. docs/specs/2026-11-24-avatar-wan-lora.md

WAN_LORA_BUSY = ("queued", "photos", "training")


def _avatar_wan_lora_name(name):
    """avatar-<name>-wan, shortened with a hash when it would pass 32 characters."""
    full = f"avatar-{name}-wan"
    return full if len(full) <= 32 else f"avatar-{name[:12]}-{hashlib.sha1(name.encode()).hexdigest()[:6]}-wan"


def _wan_lora_of(client, name):
    """The avatar's wan_lora record without the per-build fields (keeps the last ready link)."""
    wl = (_get_avatar(client, name) or {}).get("wan_lora") or {}
    return {k: wl[k] for k in ("ready", "lora", "trigger", "version") if k in wl}  # version: stays monotonic


def _set_wan_lora(client, name, wan_lora):
    avatar = _get_avatar(client, name)
    avatar["wan_lora"] = wan_lora
    _put_json(f"avatars/{client}/{name}/avatar.json", avatar)
    return wan_lora


def _wan_build(client, name, job_id):
    """Photoset-job params for a Wan LoRA build by job_id, with the avatar's wan_lora marked queued; {} (and
    wan_lora failed) when the name already holds a Krea LoRA."""
    lora = _avatar_wan_lora_name(name)
    kind = _lora_name_model(client, lora)
    if kind and kind != "t2v":
        _set_wan_lora(client, name, dict(_wan_lora_of(client, name), state="failed",
                                         error=f"LoRA name '{lora}' already holds a {kind} LoRA"))
        return {}
    prev = (_get_avatar(client, name) or {}).get("wan_lora") or {}
    version = max(_lora_versions(client, lora) + [int(prev.get("version") or 0)] + [0]) + 1
    _set_wan_lora(client, name, dict(_wan_lora_of(client, name), state="queued", lora=lora, version=version,
                                     trigger=name, job_ids=[job_id]))
    return {"wan_lora": True, "wan_lora_name": lora, "wan_version": version}


def avatar_wan_lora(client, name, body):
    """POST /avatars/{name}/wan-lora: build (a new version of) the avatar's Wan LoRA from its current media."""
    name = (name or "").strip().lower()
    avatar = _get_avatar(client, name) if AVATAR_NAME.match(name) else None
    if not avatar:
        return _resp(404, {"error": f"avatar '{name}' not found"})
    wl = avatar.get("wan_lora") or {}
    if wl.get("state") in WAN_LORA_BUSY and any(_job_live(j) for j in wl.get("job_ids") or []):
        return _resp(409, {"error": f"avatar '{name}' already has a Wan LoRA build {wl['state']}; wait for it"})
    if (avatar.get("media") or {}).get("state") != "ready":
        return _resp(409, {"error": f"avatar '{name}' has no ready media; make media first (vidgen avatar media)"})
    job_id = uuid.uuid4().hex
    try:
        params = {"avatar": name, "trigger": name, "avatar_created_at": avatar["created_at"], "krea_lora": False,
                  "wan_only": True, **_wan_build(client, name, job_id)}
        if not params.get("wan_lora"):
            return _resp(409, {"error": (_get_avatar(client, name).get("wan_lora") or {}).get("error")})
        _queue_job(client, "avatar_photoset", params, job_id=job_id)
    except (ClientError, ValueError) as e:
        _set_wan_lora(client, name, dict(_wan_lora_of(client, name), state="failed",
                                         error=f"could not queue the Wan LoRA build: {e}"))
        return _resp(502, {"error": f"could not queue the Wan LoRA build: {e}"})
    return _resp(202, {"name": name, "wan_lora": _get_avatar(client, name)["wan_lora"]})


def _queue_krea_lora(client, name, wan=False):
    """Queue a photo set + training for (a new version of) the avatar's Krea LoRA. The version counts the
    registered versions and the avatar's last build, so builds started before training finishes don't
    collide; the last ready LoRA stays linked (`ready`) until the new one is ready."""
    avatar = _get_avatar(client, name)
    prev = avatar.get("krea_lora") or {}
    lora = _avatar_lora_name(name)
    kind = _lora_name_model(client, lora)
    if kind and kind != "krea2":
        kl = _set_krea_lora(client, name, dict(_krea_lora_of(client, name), state="failed",
                                               error=f"LoRA name '{lora}' already holds a Wan LoRA"))
        _queue_avatar_media(client, name, wan=wan)  # no Krea LoRA, but the avatar still gets its media
        return kl
    version = max(_lora_versions(client, lora) + [int(prev.get("version") or 0)] + [0]) + 1
    params = {"avatar": name, "lora": lora, "version": version, "trigger": name, "count": AVATAR_DATASET_PHOTOS,
              "avatar_created_at": avatar["created_at"], "krea_lora": True}
    # The job id is on the avatar before the job is sent: a warm worker reads it as soon as the message lands.
    # The same job also makes the avatar's media (photos + clips), so the media record names it too.
    job_id = uuid.uuid4().hex
    params.update(_wan_build(client, name, job_id) if wan else {"wan_lora": False})
    kl = _set_krea_lora(client, name, dict(_krea_lora_of(client, name), state="queued", lora=lora,
                                           version=version, trigger=name, job_ids=[job_id]))
    _set_media(client, name, {"state": "queued", "job_id": job_id, "created_at": avatar["created_at"]})
    _queue_job(client, "avatar_photoset", params, job_id=job_id)
    return kl


# ---------------------------------------------------------------- avatar media (photos + 2 s clips)
# docs/specs/2026-11-24-avatar-media.md

MEDIA_BUSY = ("queued", "photos", "clips")


def _set_media(client, name, media):
    avatar = _get_avatar(client, name)
    avatar["media"] = media
    _put_json(f"avatars/{client}/{name}/avatar.json", avatar)
    return media


def _queue_avatar_media(client, name, media_only=False, wan=False):
    """Queue the avatar's photo set + clips without a Krea LoRA build (krea_lora false, or media for an
    existing avatar). The job id is on the avatar's media record before the job is sent."""
    avatar = _get_avatar(client, name)
    params = {"avatar": name, "trigger": name, "count": AVATAR_DATASET_PHOTOS,
              "avatar_created_at": avatar["created_at"], "krea_lora": False, "media_only": media_only}
    job_id = uuid.uuid4().hex
    params.update(_wan_build(client, name, job_id) if wan and not media_only else {"wan_lora": False})
    media = _set_media(client, name, {"state": "queued", "job_id": job_id, "created_at": avatar["created_at"]})
    _queue_job(client, "avatar_photoset", params, job_id=job_id)
    return media


def avatar_media_create(client, name, body):
    """POST /avatars/{name}/media: make (new) photos + clips for an existing avatar. Reuses the photos of its
    ready Krea LoRA dataset when there is one, so only the clips are rendered."""
    name = (name or "").strip().lower()
    avatar = _get_avatar(client, name) if AVATAR_NAME.match(name) else None
    if not avatar:
        return _resp(404, {"error": f"avatar '{name}' not found"})
    media = avatar.get("media") or {}
    if media.get("state") in MEDIA_BUSY and _job_live(media.get("job_id")):
        return _resp(409, {"error": f"avatar '{name}' already has media being made ({media['state']}); wait for it"})
    try:
        return _resp(202, {"name": name, "media": _queue_avatar_media(client, name, media_only=True)})
    except (ClientError, ValueError) as e:
        _set_media(client, name, {"state": "failed", "error": f"could not queue the avatar media: {e}"})
        return _resp(502, {"error": f"could not queue the avatar media: {e}"})


def _job_live(job_id):
    """Whether the job is still queued or running (a busy record whose job ended or vanished is stale)."""
    if not job_id:
        return False
    job = jobs.get_item(Key={"job_id": job_id}).get("Item")
    return bool(job) and job.get("status") in ("queued", "running")


MEDIA_FILE = re.compile(r"(photo_\d{2}\.png|clip_\d\.mp4|media\.json)")


def avatar_media_get(client, name):
    """GET /avatars/{name}/media: {state, files: [{name, size, url}]}; files only once the media is ready."""
    name = (name or "").strip().lower()
    avatar = _get_avatar(client, name) if AVATAR_NAME.match(name) else None
    if not avatar or not avatar.get("media"):
        return _resp(404, {"error": f"avatar '{name}' has no media"})
    media = avatar["media"]
    if media.get("state") != "ready":
        return _resp(200, {"state": media.get("state"), "files": []})
    prefix = f"avatars/{client}/{name}/media/"
    doc = _s3_json(prefix + "media.json") or {}
    entries = [(e.get("file"), e.get("size")) for e in (doc.get("photos") or []) + (doc.get("clips") or [])]
    entries.append(("media.json", None))
    files = [{"name": f, "size": size, "url": s3.generate_presigned_url(
                "get_object", Params={"Bucket": BUCKET, "Key": prefix + f}, ExpiresIn=URL_TTL)}
             for f, size in entries if isinstance(f, str) and MEDIA_FILE.fullmatch(f)]
    return _resp(200, {"state": "ready", "files": files})


def avatar_krea_lora(client, name, body):
    """POST /avatars/{name}/krea-lora: build (a new version of) the avatar's Krea 2 LoRA now."""
    name = (name or "").strip().lower()
    if not AVATAR_NAME.match(name) or not _get_avatar(client, name):
        return _resp(404, {"error": f"avatar '{name}' not found"})
    state = (_get_avatar(client, name).get("krea_lora") or {}).get("state")
    if state in KREA_LORA_BUSY:
        return _resp(409, {"error": f"avatar '{name}' already has a Krea LoRA build {state}; wait for it to finish"})
    try:
        return _resp(202, {"name": name, "krea_lora": _queue_krea_lora(client, name)})
    except (ClientError, ValueError) as e:
        _set_krea_lora(client, name, dict(_krea_lora_of(client, name), state="failed",
                                          error=f"could not queue the Krea LoRA build: {e}"))
        _set_media(client, name, {"state": "failed", "error": f"could not queue the avatar media: {e}"})
        return _resp(502, {"error": f"could not queue the Krea LoRA build: {e}"})


def _write_avatar(client, name, source_keys, description, created_at=None):
    """Save the avatar as exactly these images, in order. Images already in this avatar's
    images/ folder are kept as they are; anything else is copied in under a new id."""
    folder = f"avatars/{client}/{name}/images/"
    now = int(time.time())
    images = []
    for src in source_keys:
        key = src if src.startswith(folder) else f"{folder}{secrets.token_hex(4)}.{_ext(src)}"
        if key != src:
            s3.copy_object(Bucket=BUCKET, Key=key, CopySource={"Bucket": BUCKET, "Key": src})
        images.append({"key": key, "added_at": now})
    doc = {"name": name, "description": description, "reference_key": images[0]["key"],
           "images": images, "created_at": created_at or now}
    prev = _get_avatar(client, name) or {}
    if prev.get("krea_lora"):  # image edits and restore keep the Krea LoRA link (create_avatar sets it anew)
        doc["krea_lora"] = prev["krea_lora"]
    if prev.get("media"):  # and the media record
        doc["media"] = prev["media"]
    if prev.get("wan_lora"):  # and the Wan LoRA link
        doc["wan_lora"] = prev["wan_lora"]
    _put_json(f"avatars/{client}/{name}/avatar.json", doc)
    return images


def _archive_avatar(client, name):
    """Copy the current version (if any), with all its images, to
    avatars/<client>/<name>/history/<created_at>/."""
    current = _get_avatar(client, name)
    if not current:
        return
    base = f"avatars/{client}/{name}/history/{current['created_at']}/"
    images = []
    for key in _image_keys(current):
        dest = base + "images/" + key.rsplit("/", 1)[-1]
        s3.copy_object(Bucket=BUCKET, Key=dest, CopySource={"Bucket": BUCKET, "Key": key})
        images.append({"key": dest, "added_at": current["created_at"]})
    _put_json(base + "avatar.json", dict(current, reference_key=images[0]["key"], images=images))


def _previews(keys):
    return [{"n": n, "key": k.rsplit("/", 1)[-1], "preview": s3.generate_presigned_url(
        "get_object", Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=URL_TTL)}
        for n, k in enumerate(keys, 1)]


def get_avatar(client, name):
    avatar = _get_avatar(client, name)
    if not avatar:
        return _resp(404, {"error": f"avatar '{name}' not found"})
    return _resp(200, {"name": name, "description": avatar["description"],
                       "created_at": avatar["created_at"],
                       "default": _default_avatar_name(client) == name,
                       "images": _previews(_image_keys(avatar))})


def _change_images(client, name, new_keys):
    """Archive the current version, then save the avatar with new_keys (same description)."""
    avatar = _get_avatar(client, name)
    _archive_avatar(client, name)
    _write_avatar(client, name, new_keys, avatar["description"])
    return _resp(200, {"name": name, "images": len(new_keys)})


def add_avatar_images(client, name, body):
    avatar = _get_avatar(client, name)
    if not avatar:
        return _resp(404, {"error": f"avatar '{name}' not found"})
    try:
        sources = _image_sources(client, body)
    except ValueError as e:
        return _resp(400, {"error": str(e)})
    keys = _image_keys(avatar)
    if len(keys) + len(sources) > MAX_AVATAR_IMAGES:
        return _resp(400, {"error": f"an avatar holds at most {MAX_AVATAR_IMAGES} images "
                                    f"('{name}' has {len(keys)})"})
    return _change_images(client, name, keys + sources)


def _image_index(keys, raw):
    n = int(raw) if str(raw).isdigit() else 0
    return n - 1 if 1 <= n <= len(keys) else None


def remove_avatar_image(client, name, index):
    avatar = _get_avatar(client, name)
    if not avatar:
        return _resp(404, {"error": f"avatar '{name}' not found"})
    keys = _image_keys(avatar)
    i = _image_index(keys, index)
    if i is None:
        return _resp(404, {"error": f"no image {index} ('{name}' has {len(keys)})"})
    if len(keys) == 1:
        return _resp(400, {"error": "can't remove an avatar's only image; add another first"})
    return _change_images(client, name, keys[:i] + keys[i + 1:])


def set_main_avatar_image(client, name, body):
    avatar = _get_avatar(client, name)
    if not avatar:
        return _resp(404, {"error": f"avatar '{name}' not found"})
    keys = _image_keys(avatar)
    i = _image_index(keys, body.get("index"))
    if i is None:
        return _resp(404, {"error": f"no image {body.get('index')} ('{name}' has {len(keys)})"})
    if i == 0:
        return _resp(200, {"name": name, "images": len(keys)})
    return _change_images(client, name, [keys[i]] + keys[:i] + keys[i + 1:])


def avatar_history(client, name):
    if not _get_avatar(client, name):
        return _resp(404, {"error": f"avatar '{name}' not found"})
    prefixes = s3.list_objects_v2(
        Bucket=BUCKET, Prefix=f"avatars/{client}/{name}/history/", Delimiter="/"
    ).get("CommonPrefixes", [])
    versions = []
    for p in prefixes:
        a = _s3_json(p["Prefix"] + "avatar.json")
        if a:
            versions.append({"version": str(a["created_at"]), "description": a["description"],
                             "created_at": a["created_at"], "images": len(_image_keys(a)),
                             "preview": s3.generate_presigned_url(
                                 "get_object", Params={"Bucket": BUCKET, "Key": a["reference_key"]},
                                 ExpiresIn=URL_TTL)})
    return _resp(200, {"name": name, "versions": sorted(versions, key=lambda v: v["created_at"], reverse=True)})


def restore_avatar(client, name, body):
    version = str(body.get("version", ""))
    old = _s3_json(f"avatars/{client}/{name}/history/{version}/avatar.json") if version.isdigit() else None
    if not old:
        return _resp(404, {"error": f"no version {version!r} of avatar '{name}'"})
    _archive_avatar(client, name)  # restoring is itself undoable
    _write_avatar(client, name, _image_keys(old), old["description"])
    return _resp(200, {"name": name, "restored": version})


def set_default_avatar(client, body):
    name = (body.get("name") or "").strip().lower()
    if not _get_avatar(client, name):
        return _resp(404, {"error": f"avatar '{name}' not found"})
    _put_json(f"avatars/{client}/_default.json", {"name": name})
    return _resp(200, {"default": name})


def _avatar_live_job(client, name, avatar):
    """The id of a queued or running job that needs the avatar (builds its LoRAs or media, or renders with its
    images), or None."""
    kl, wl, media = (avatar.get(k) or {} for k in ("krea_lora", "wan_lora", "media"))
    for job_id in (kl.get("job_ids") or []) + (wl.get("job_ids") or []) + [media.get("job_id")]:
        if _job_live(job_id):
            return job_id
    items = jobs.query(IndexName=JOBS_INDEX, KeyConditionExpression=Key("client").eq(client),
                       ScanIndexForward=False, Limit=JOB_HISTORY_LIMIT).get("Items", [])
    for item in items:
        raw = item.get("params") or "{}"
        if item.get("status") in ("queued", "running") and (
                f'"avatars/{client}/{name}/' in raw or json.loads(raw).get("avatar") == name):
            return item["job_id"]
    return None


def _avatar_lora_names(client, name, avatar):
    """The LoRA names that belong to this avatar: avatar-<name> and avatar-<name>-wan, when their files
    (or training set) say they were made for it. Avatar 'x-wan' and avatar 'x' can meet on one name."""
    names = {_avatar_lora_name(name), _avatar_wan_lora_name(name)}
    names.update(l for l in ((avatar.get(k) or {}).get("lora") for k in ("krea_lora", "wan_lora")) if l)
    owned = []
    for lora in sorted(n for n in names if n.startswith("avatar-") and DATASET_NAME.match(n)):
        docs = [_s3_json(f"datasets/{client}/{lora}/manifest.json")]
        docs += [_s3_json(f"loras/{client}/{lora}/v{v}/lora.json") for v in _lora_versions(client, lora)]
        if any(d and d.get("avatar") == name for d in docs):
            owned.append(lora)
    return owned


def delete_avatar(client, name, body):
    """DELETE /avatars/{name} {loras?}: remove the avatar with its images, history and media; with
    loras: true also its Krea 2 and Wan LoRAs and their training photos. There is no undo."""
    name = (name or "").strip().lower()
    avatar = _get_avatar(client, name) if AVATAR_NAME.match(name) else None
    if not avatar:
        return _resp(404, {"error": f"avatar '{name}' not found"})
    with_loras = body.get("loras", False)
    if not isinstance(with_loras, bool):
        return _resp(400, {"error": "loras must be true or false"})
    live = _avatar_live_job(client, name, avatar)
    if live:
        return _resp(409, {"error": f"avatar '{name}' is in use by job {live} (queued or running); "
                                    "wait for it or cancel it, then delete"})
    loras = _avatar_lora_names(client, name, avatar) if with_loras else []
    keys = []
    for lora in loras:
        keys += list(_list_keys(f"loras/{client}/{lora}/")) + list(_list_keys(f"datasets/{client}/{lora}/"))
    doc = f"avatars/{client}/{name}/avatar.json"
    keys += [k for k in _list_keys(f"avatars/{client}/{name}/") if k != doc]
    was_default = _default_avatar_name(client) == name
    # avatar.json (and the default marker) go last: if S3 refuses anything the avatar is still listed, still
    # the default, and the delete can be repeated.
    last = [doc] + ([f"avatars/{client}/_default.json"] if was_default else [])
    failed = _delete_keys(keys) or _delete_keys(last)
    if failed:
        return _resp(502, {"error": f"could not delete {failed} of the files of avatar '{name}'; try again"})
    return _resp(200, {"name": name, "deleted": len(keys) + len(last), "loras": loras, "was_default": was_default})


# ---------------------------------------------------------------- datasets
#
#   datasets/<client>/<name>/raw/<relative path>   the files, uploaded straight from the client
#   datasets/<client>/<name>/manifest.json          {name, kind, trigger, files, ...}

def _dataset_prefix(client, name):
    if not DATASET_NAME.match(name):
        raise ValueError("dataset name: 1-32 characters, a-z 0-9 - _ (starting with a letter or digit)")
    return f"datasets/{client}/{name}/"


def _dataset_path(path):
    """A safe relative path inside the dataset, or ValueError."""
    if not isinstance(path, str) or not path or len(path) > MAX_DATASET_PATH:
        raise ValueError(f"file path must be 1-{MAX_DATASET_PATH} characters")
    if path.startswith(("/", "\\")) or "\\" in path or any(ord(c) < 32 for c in path):
        raise ValueError(f"bad file path: {path!r}")
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"bad file path: {path!r}")
    if _ext(path) not in DATASET_EXTS:
        raise ValueError(f"{path}: must be one of {', '.join(sorted(DATASET_EXTS))}")
    return path


def _dataset_size(size, path):
    if not isinstance(size, int) or not 1 <= size <= MAX_DATASET_FILE_BYTES:
        raise ValueError(f"{path}: size must be 1 B - 2 GB")
    return size


def _list_keys(prefix):
    """{key: size} for every object under prefix."""
    out = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            out[obj["Key"]] = obj["Size"]
    return out


def _delete_keys(keys):
    """Delete the keys; returns how many S3 refused."""
    keys, failed = list(keys), 0
    for i in range(0, len(keys), 1000):
        r = s3.delete_objects(Bucket=BUCKET, Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]], "Quiet": True})
        failed += len(r.get("Errors") or [])
    return failed


def dataset_uploads(client, name, body):
    try:
        prefix = _dataset_prefix(client, name)
        files = body.get("files")
        if not isinstance(files, list) or not 1 <= len(files) <= MAX_UPLOADS_PER_CALL:
            raise ValueError(f"files must be a list of 1-{MAX_UPLOADS_PER_CALL}")
        uploads = []
        for f in files:
            path = _dataset_path(f.get("path"))
            _dataset_size(f.get("size"), path)
            key = prefix + "raw/" + path
            uploads.append({"path": path, "key": key, **_upload_spec(key, f.get("size"))})
    except (ValueError, TypeError, AttributeError) as e:
        return _resp(400, {"error": str(e)})
    return _resp(200, {"uploads": uploads})


def commit_dataset(client, name, body):
    """Write the manifest after checking every listed file is in S3 with its size; files in
    S3 that the manifest no longer lists are deleted (the folder is the source of truth)."""
    try:
        prefix = _dataset_prefix(client, name)
        kind = body.get("kind")
        if kind not in DATASET_KINDS:
            raise ValueError(f"kind must be one of: {', '.join(DATASET_KINDS)}")
        trigger = (body.get("trigger") or f"vgn_{name}").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", trigger):
            raise ValueError("trigger: 1-40 characters, letters, digits, - _")
        files = body.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("files must be a non-empty list")
        clean, seen = [], set()
        for f in files:
            path = _dataset_path(f.get("path"))
            if path in seen:
                raise ValueError(f"duplicate file: {path}")
            seen.add(path)
            clean.append({"path": path, "size": _dataset_size(f.get("size"), path),
                          "mtime": int(f.get("mtime") or 0)})
        old = _s3_json(prefix + "manifest.json") or {}
        consent_at = old.get("consent_at")
        if kind == "character" and not consent_at:
            if body.get("consent") is not True:
                raise ValueError("character datasets need consent: confirm you have the consent of the person shown")
            consent_at = int(time.time())
    except (ValueError, TypeError, AttributeError) as e:
        return _resp(400, {"error": str(e)})

    stored = _list_keys(prefix + "raw/")
    for f in clean:
        size = stored.get(prefix + "raw/" + f["path"])
        if size != f["size"]:
            what = "missing" if size is None else f"{size} bytes in S3, {f['size']} listed"
            return _resp(400, {"error": f"{f['path']}: {what}; upload it first"})
    listed = {prefix + "raw/" + f["path"] for f in clean}
    removed = [k for k in stored if k not in listed]
    _delete_keys(removed)
    now = int(time.time())
    manifest = {"name": name, "kind": kind, "trigger": trigger, "files": clean, "file_count": len(clean),
                "total_bytes": sum(f["size"] for f in clean), "created_at": old.get("created_at", now),
                "updated_at": now}
    if consent_at:
        manifest["consent_at"] = consent_at
    _put_json(prefix + "manifest.json", manifest)
    return _resp(200, {k: v for k, v in manifest.items() if k != "files"} | {"removed": len(removed)})


def get_dataset(client, name):
    try:
        prefix = _dataset_prefix(client, name)
    except ValueError as e:
        return _resp(400, {"error": str(e)})
    manifest = _s3_json(prefix + "manifest.json")
    return _resp(200, manifest) if manifest else _resp(404, {"error": f"dataset '{name}' not found"})


def list_datasets(client):
    prefixes = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"datasets/{client}/", Delimiter="/").get("CommonPrefixes", [])
    out = []
    for p in prefixes:
        m = _s3_json(p["Prefix"] + "manifest.json")
        if m:
            out.append({k: v for k, v in m.items() if k != "files"})
    return _resp(200, {"datasets": sorted(out, key=lambda d: d["name"])})


MAX_RENDER_LORAS = 2
LORA_WEIGHT = (0.0, 1.5)


def _lora_version(client, name, version=None):
    """(version, lora.json) for a LoRA: the given version, or the one current.json points at."""
    base = f"loras/{client}/{name}/"
    current = _s3_json(base + "current.json") or {}
    version = int(version) if version not in (None, "") else current.get("version")
    meta = _s3_json(f"{base}v{version}/lora.json") if version else None
    return version, meta, current


def _lora_keys(client, name, version, meta, step):
    """S3 keys of the two expert files: final weights (step None) or a saved checkpoint."""
    base = f"loras/{client}/{name}/v{version}/"
    files = set(meta.get("files") or [])
    if step is None and meta.get("single_file"):  # an uploaded one-file LoRA serves both experts
        names = ("t2v_high.safetensors", "t2v_high.safetensors")
    elif step is None:
        names = ("t2v_high.safetensors", "t2v_low.safetensors")
    else:
        names = tuple(f"checkpoints/{e}-step{int(step):06d}.safetensors" for e in ("t2v_high", "t2v_low"))
    if not set(names) <= files:
        what = f"has no checkpoint at step {step}" if step is not None else "is missing its final weights"
        raise ValueError(f"LoRA '{name}' v{version} {what}; checkpoints: "
                         f"{', '.join(map(str, _checkpoint_steps(meta))) or 'none'}")
    return base + names[0], base + names[1]


def _checkpoint_steps(meta):
    return sorted({int(f.split("step")[-1].split(".")[0]) for f in meta.get("files", [])
                   if f.startswith("checkpoints/t2v_low")})


def _resolve_loras(client, requested):
    """Validated LoRAs for a video/story job, pinned at submit time (so a re-run renders the same)
    with the S3 keys of both expert files. Each item: {name, weight, version?, step?}; without
    version/step it uses current.json (version and optional checkpoint step). Also accepts a
    "name[:weight],..." string."""
    if not requested:
        return []
    if isinstance(requested, str):
        requested = [dict(zip(("name", "weight"), part.strip().split(":", 1)))
                     for part in requested.split(",") if part.strip()]
    if not isinstance(requested, list) or len(requested) > 2 * MAX_RENDER_LORAS:
        raise ValueError(f"at most {MAX_RENDER_LORAS} LoRAs per job (and {MAX_RENDER_LORAS} Krea 2 LoRAs)")
    out = []
    for item in requested:
        if not isinstance(item, dict):
            raise ValueError("each LoRA is {name, weight}")
        name = str(item.get("name") or "").strip().lower()
        if not DATASET_NAME.match(name):
            raise ValueError(f"LoRA name '{name}': 1-32 characters, a-z 0-9 - _")
        if name in (o["name"] for o in out):
            raise ValueError(f"LoRA '{name}' given twice")
        try:
            weight = float(item.get("weight") if item.get("weight") not in (None, "") else 1.0)
            pinned_version = item.get("version") not in (None, "")
            version, meta, current = _lora_version(client, name, item.get("version"))
            if item.get("step") not in (None, ""):
                step = int(item["step"])
            elif pinned_version:
                step = None  # a version without a step means its final weights
            else:
                step = current.get("step")
        except (TypeError, ValueError):
            raise ValueError(f"LoRA '{name}': weight, version and step must be numbers")
        if not LORA_WEIGHT[0] <= weight <= LORA_WEIGHT[1]:
            raise ValueError(f"LoRA '{name}': weight must be {LORA_WEIGHT[0]}-{LORA_WEIGHT[1]}")
        if not meta:
            raise ValueError(f"LoRA '{name}'{f' v{version}' if pinned_version else ''} not found; see vidgen lora list")
        if "krea2" in (meta.get("models") or []):  # Krea 2 image LoRA: one file, no checkpoints
            if item.get("step") not in (None, ""):
                raise ValueError(f"LoRA '{name}' is a Krea 2 LoRA; it has no checkpoint steps")
            out.append({"name": name, "version": int(version), "step": None, "weight": round(weight, 3),
                        "trigger": meta.get("trigger") or "", "model": "krea2",
                        "key": f"loras/{client}/{name}/v{int(version)}/{LORA_FILE['krea2']}"})
            continue
        if "t2v" not in (meta.get("models") or []):
            raise ValueError(f"LoRA '{name}' has no video (t2v) weights")
        high, low = _lora_keys(client, name, version, meta, step)
        out.append({"name": name, "version": int(version), "step": step, "weight": round(weight, 3), "model": "t2v",
                    "trigger": meta.get("trigger") or "", "high_key": high, "low_key": low})
    for model, label in (("t2v", "Wan"), ("krea2", "Krea 2")):
        if sum(o["model"] == model for o in out) > MAX_RENDER_LORAS:
            raise ValueError(f"at most {MAX_RENDER_LORAS} {label} LoRAs per job")
    return out


def _auto_first_frame(job_type, avatars):
    """The default first frame: Krea 2 whenever it can be made, else Wan. Videos: no avatar, or exactly one
    avatar with a ready Krea LoRA. Stories: always (shots whose avatars have no Krea LoRA keep the Qwen keyframe)."""
    if job_type == "story" or not avatars:
        return "krea2"
    ready = len(avatars) == 1 and ((avatars[0].get("krea_lora") or {}).get("ready"))
    return "krea2" if ready else "wan"


def _avatar_first_frame(client, job_type, params):
    """With image_model krea2 and avatars: a Krea first frame for an avatar that has a ready Krea LoRA (its
    LoRA added, its name as trigger). Video: exactly one avatar, and it needs the LoRA. Story: the first
    avatar with one; shots with just that avatar get Krea frames, others keep the Qwen keyframe."""
    if params.get("image_model") != "krea2" or not params.get("avatars"):
        return
    names = [a["name"] for a in params["avatars"]]
    ready = {}
    for n in names:
        kl = (_get_avatar(client, n) or {}).get("krea_lora") or {}
        if kl.get("ready"):
            ready[n] = kl["ready"]
    pick = next((n for n in names if n in ready), None) if job_type == "story" else \
        (names[0] if len(names) == 1 and names[0] in ready else None)
    if job_type == "video" and not pick:
        raise ValueError("a Krea 2 first frame with an avatar needs exactly one avatar with a ready Krea 2 LoRA "
                         "(vidgen avatar list); otherwise use avatar none, or the Qwen keyframe (image_model wan)")
    if pick:
        params["krea_avatar"], params["krea_avatar_lora"] = pick, ready[pick]["lora"]
        params["krea_loras"] = _with_avatar_lora(client, {"name": ready[pick]["lora"], "version": ready[pick]["version"]},
                                                 list(params.get("krea_loras") or []))


def _with_avatar_lora(client, avatar_lora, krea):
    """The avatar's Krea LoRA first, then the job's own Krea LoRAs (the avatar's one dropped if also picked by
    hand); with an avatar LoRA, one more Krea LoRA at most, so a job never loads more than 2."""
    own = [l for l in krea if l["name"] != avatar_lora["name"]]
    if len(own) > MAX_RENDER_LORAS - 1:
        raise ValueError(f"with an avatar's Krea 2 LoRA, add at most one more Krea 2 LoRA ({len(own)} given)")
    return _resolve_loras(client, [dict(avatar_lora, weight=1.0)]) + own


def _with_avatar_wan_lora(client, params):
    """A video starring an avatar with a ready Wan LoRA gets it first, at 1.0 (its final weights, pinned), within
    the Wan LoRA limit; not when the user already named it. A broken link is skipped (logged), never a 400."""
    avatars = params.get("avatars") or []
    ready = ((avatars[0].get("wan_lora") or {}).get("ready") or {}) if avatars else {}
    if not ready.get("lora") or any(l["name"] == ready["lora"] for l in params.get("loras") or []):
        return
    try:
        lora = _resolve_loras(client, [{"name": ready["lora"], "version": ready["version"], "weight": 1.0}])
    except ValueError as e:
        print(json.dumps({"event": "avatar_wan_lora_skipped", "avatar": avatars[0].get("name"), "error": str(e)}))
        return
    loras = lora + list(params.get("loras") or [])
    if sum(l["model"] == "t2v" for l in loras) > MAX_RENDER_LORAS:
        raise ValueError(f"at most {MAX_RENDER_LORAS} Wan LoRAs per job, counting the avatar's own Wan LoRA "
                         f"'{ready['lora']}' (send avatar_lora: false, or --no-avatar-lora, to leave it out)")
    params["loras"] = loras


def _job_loras(client, job_type, params, requested):
    """Route a job's LoRAs by kind: Wan ones to params["loras"] (video, story), Krea 2 ones to
    params["krea_loras"] (Krea photos; Krea first frames of videos and stories). Raises ValueError for
    a LoRA the job has no model for. Other job types ignore a loras field as before, except Krea LoRAs."""
    if job_type not in ("video", "story", "photos"):
        try:
            resolved = _resolve_loras(client, requested)
        except ValueError:
            return
        krea = [l for l in resolved if l["model"] == "krea2"]
        if krea:
            raise ValueError(f"Krea 2 LoRA '{krea[0]['name']}': {job_type} jobs have no Krea 2 step")
        return
    resolved = _resolve_loras(client, requested)
    wan = [l for l in resolved if l["model"] == "t2v"]
    krea = [l for l in resolved if l["model"] == "krea2"]
    if job_type in ("video", "story"):
        if krea and params.get("image_model") != "krea2":
            raise ValueError(f"Krea 2 LoRA '{krea[0]['name']}' needs --image-model krea2 (a Krea first frame)")
        params["loras"], params["krea_loras"] = wan, krea
    elif job_type == "photos":
        if resolved and params.get("model") != "krea2":
            raise ValueError("photos with the qwen model take no LoRAs; Krea 2 LoRAs need --model krea2")
        if wan:
            raise ValueError(f"Wan LoRA '{wan[0]['name']}' is for video; Krea 2 photos take Krea 2 LoRAs")
        if params.get("avatar_lora"):  # the avatar's own Krea LoRA comes first
            krea = _with_avatar_lora(client, params["avatar_lora"], krea)
        if krea:
            params["krea_loras"] = krea


# ---------------------------------------------------------------- uploaded LoRAs
#
# A LoRA file made elsewhere (e.g. downloaded) becomes a new version of a LoRA name, so it works
# like a trained one (--lora NAME, `lora use`). Two steps, because the API cannot copy GB-sized
# files within its timeout: POST .../uploads reserves version N and returns PUT links straight to
# loras/<client>/<name>/v<N>/t2v_high(.low).safetensors; PUT .../versions/<N> then checks the
# files' safetensors headers (ranged reads) and only then writes lora.json, which makes the
# version visible. A version without lora.json is ignored everywhere.

MAX_LORA_BYTES = 2 * 1024 ** 3
MAX_SAFETENSORS_HEADER = 100 * 1024 ** 2
LORA_DTYPES = {"F16", "BF16", "F32"}
WAN14B_DIM = 5120          # hidden size of the Wan 2.x 14B models the render box runs
SMALL_WAN_DIMS = {1536, 3072}  # Wan 1.3B / Wan 2.2 5B
OTHER_MODEL_KEYS = ("double_blocks", "single_blocks", "input_blocks", "output_blocks", "down_blocks",
                    "transformer_blocks", "joint_blocks")
LORA_KEY = re.compile(r"lora[._-]?(a|b|up|down)\b", re.IGNORECASE)
# Krea 2 Turbo (image model, own venv): 48 heads x 128 = 6144 wide. Its blocks (main and text-fusion)
# use attn wq/wk/wv/wo/gate (or qkv_proj / diffusers to_q...) and mlp gate/up/down; Wan uses
# self_attn/cross_attn/ffn (attn1/attn2 in diffusers naming), so the two can't be confused.
KREA2_DIM = 6144
WAN_MARKERS = re.compile(r"self_attn|cross_attn|ffn|attn1|attn2")
KREA2_MODULE = re.compile(r"attn[._]?(wq|wk|wv|wo|gate|qkv_proj|to_q|to_k|to_v|to_out)|mlp[._]?(gate|up|down)")
LORA_MODELS = ("t2v", "krea2")  # what a LoRA name is for: Wan video, or Krea 2 images
LORA_FILE = {"krea2": "krea2.safetensors"}
# File names a LoRA version may list (lora.json "files"): flat names, or one level under checkpoints/.
LORA_LOCAL_FILE = re.compile(r"(checkpoints/)?[A-Za-z0-9_.-]+\.safetensors")


def _krea2_layout(key):
    return bool(re.search(r"blocks[._]\d+[._]", key) and KREA2_MODULE.search(key))


def _lora_versions(client, name):
    base = f"loras/{client}/{name}/"
    return [int(p["Prefix"][len(base) + 1:-1])
            for p in s3.list_objects_v2(Bucket=BUCKET, Prefix=base + "v", Delimiter="/").get("CommonPrefixes", [])
            if p["Prefix"][len(base) + 1:-1].isdigit()]


def _upload_spec(key, size):
    """How the client uploads `key`: {"put_url"} for small or unknown sizes, else a multipart upload with
    presigned part URLs ({"multipart": {key, upload_id, part_size, part_urls}})."""
    if not isinstance(size, int) or isinstance(size, bool) or size < MULTIPART_MIN:
        return {"put_url": s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": key},
                                                     ExpiresIn=URL_TTL)}
    upload_id = s3.create_multipart_upload(Bucket=BUCKET, Key=key)["UploadId"]
    parts = -(-size // MULTIPART_PART)
    urls = [s3.generate_presigned_url("upload_part", Params={"Bucket": BUCKET, "Key": key, "UploadId": upload_id,
                                                             "PartNumber": n}, ExpiresIn=URL_TTL)
            for n in range(1, parts + 1)]
    return {"multipart": {"key": key, "upload_id": upload_id, "part_size": MULTIPART_PART, "part_urls": urls}}


def _multipart_key_ok(client, key):
    """Only the caller's own LoRA files and dataset files can be completed or aborted."""
    c = re.escape(client)
    return bool(re.fullmatch(rf"loras/{c}/[a-z0-9][a-z0-9_-]{{0,31}}/v[0-9]{{1,6}}/(krea2|t2v_high|t2v_low)\.safetensors",
                             key) or (re.fullmatch(rf"datasets/{c}/[a-z0-9][a-z0-9_/-]{{0,80}}/raw/[^\x00]+", key)
                                      and ".." not in key.split("/")))


def uploads_complete(client, body):
    """POST /uploads/complete {key, upload_id, parts: [{part, etag}]}: finish a multipart upload."""
    key, upload_id, parts = str(body.get("key") or ""), str(body.get("upload_id") or ""), body.get("parts")
    if not _multipart_key_ok(client, key) or not upload_id:
        return _resp(400, {"error": "not one of your uploads"})
    try:
        done = sorted(({"PartNumber": int(p["part"]), "ETag": str(p["etag"])} for p in parts),
                      key=lambda p: p["PartNumber"])
    except (TypeError, KeyError, ValueError):
        return _resp(400, {"error": "parts must be [{part, etag}]"})
    if not 1 <= len(done) <= 10000:
        return _resp(400, {"error": "parts must be a list of 1-10000"})
    try:
        s3.complete_multipart_upload(Bucket=BUCKET, Key=key, UploadId=upload_id, MultipartUpload={"Parts": done})
    except ClientError as e:
        return _resp(400, {"error": f"could not complete the upload: {e}"})
    return _resp(200, {"key": key, "parts": len(done)})


def uploads_abort(client, body):
    """POST /uploads/abort {key, upload_id}: drop an unfinished multipart upload (its parts are deleted)."""
    key, upload_id = str(body.get("key") or ""), str(body.get("upload_id") or "")
    if not _multipart_key_ok(client, key) or not upload_id:
        return _resp(400, {"error": "not one of your uploads"})
    try:
        s3.abort_multipart_upload(Bucket=BUCKET, Key=key, UploadId=upload_id)
    except ClientError as e:
        return _resp(400, {"error": f"could not abort the upload: {e}"})
    return _resp(200, {"key": key, "aborted": True})


def lora_uploads(client, name, body):
    """POST /loras/{name}/uploads {files: "single"|"pair", model: "t2v"|"krea2"} -> {version, model,
    urls: {high, low?}}. A Krea 2 LoRA is one file (krea2.safetensors); a name holds one model."""
    name = (name or "").strip().lower()
    if not DATASET_NAME.match(name):
        return _resp(400, {"error": "name must be 1-32 chars: a-z, 0-9, - or _"})
    files = body.get("files") or "single"
    if files not in ("single", "pair"):
        return _resp(400, {"error": "files must be single (one file for both experts) or pair (high + low)"})
    if name.startswith("avatar-"):
        return _resp(400, {"error": "names starting with avatar- are reserved for avatar LoRAs; pick another name"})
    model = body.get("model") or "t2v"
    if model not in LORA_MODELS:
        return _resp(400, {"error": f"model must be one of: {', '.join(LORA_MODELS)}"})
    if model == "krea2" and files == "pair":
        return _resp(400, {"error": "a Krea 2 LoRA is one file (files: single)"})
    existing = _lora_name_model(client, name)
    if existing and existing != model:
        what = {"t2v": "a Wan LoRA", "krea2": "a Krea 2 LoRA"}
        return _resp(400, {"error": f"'{name}' is {what[existing]}; use another name for {what[model]}"})
    sizes = body.get("sizes") or {}
    if not isinstance(sizes, dict) or any(not isinstance(v, int) or v > MAX_LORA_BYTES for v in sizes.values()):
        return _resp(400, {"error": f"sizes: bytes per file, at most {MAX_LORA_BYTES // 1024 ** 3} GB"})
    version = max(_lora_versions(client, name) or [0]) + 1
    base = f"loras/{client}/{name}/v{version}/"
    if model == "krea2":
        keys = {"high": base + LORA_FILE["krea2"]}
    else:
        keys = {part: f"{base}t2v_{part}.safetensors" for part in (("high", "low") if files == "pair" else ("high",))}
    urls, multipart = {}, {}
    for part, key in keys.items():  # with sizes (newer CLIs), big files get a parallel multipart upload
        spec = _upload_spec(key, sizes.get(part))
        if "put_url" in spec:
            urls[part] = spec["put_url"]
        else:
            multipart[part] = spec["multipart"]
    return _resp(200, {"version": version, "model": model, "urls": urls, "max_bytes": MAX_LORA_BYTES,
                       **({"multipart": multipart} if multipart else {})})


def _safetensors_header(key):
    """The JSON header of a .safetensors file in S3, read with two ranged GETs."""
    try:
        size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    except ClientError:
        raise ValueError("file not uploaded")
    if size > MAX_LORA_BYTES:
        raise ValueError(f"file is {size / 1024 ** 3:.1f} GB; the limit is {MAX_LORA_BYTES // 1024 ** 3} GB")
    first = s3.get_object(Bucket=BUCKET, Key=key, Range="bytes=0-7")["Body"].read()
    n = int.from_bytes(first, "little") if len(first) == 8 else 0
    if not 2 <= n <= min(MAX_SAFETENSORS_HEADER, size - 8):
        raise ValueError("not a .safetensors file")
    raw = s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes=8-{7 + n}")["Body"].read()
    try:
        header = json.loads(raw)
    except ValueError:
        raise ValueError("not a .safetensors file (bad header)")
    if not isinstance(header, dict):
        raise ValueError("not a .safetensors file (bad header)")
    for key, entry in header.items():  # untrusted: each tensor entry must be {dtype: str, shape: [int]}
        if key == "__metadata__":
            continue
        shape = entry.get("shape") if isinstance(entry, dict) else None
        if (not isinstance(entry.get("dtype") if isinstance(entry, dict) else None, str)
                or not isinstance(shape, list) or not all(isinstance(d, int) for d in shape)):
            raise ValueError("not a .safetensors file (bad header)")
    return header, size


def _s3_exists(key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError:
        return False


def _check_wan_lora(header):
    """Raise ValueError unless the header looks like a LoRA for the Wan 14B models."""
    keys = [k for k in header if k != "__metadata__"]
    lora = [k for k in keys if LORA_KEY.search(k)]
    if not lora:
        raise ValueError("not a LoRA (no lora_A/lora_B or lora_down/lora_up weights)")
    if any(marker in k for k in keys for marker in OTHER_MODEL_KEYS):
        raise ValueError("made for a different model (Flux, SD, SDXL, Qwen-Image, ...), not Wan")
    if not any(WAN_MARKERS.search(k) for k in keys) and any(_krea2_layout(k) for k in lora):
        raise ValueError("this is a Krea 2 LoRA (image model), not a Wan one; add it under a new name "
                         "(vidgen lora add detects the kind)")
    if not any(re.search(r"blocks[._]\d+[._]", k) and re.search(r"self_attn|cross_attn|attn1|attn2|ffn", k) for k in lora):
        raise ValueError("doesn't look like a Wan LoRA (no Wan attention blocks)")
    bad = sorted({str(header[k].get("dtype")) for k in lora if header[k].get("dtype") not in LORA_DTYPES})
    if bad:
        raise ValueError(f"unsupported weight type {', '.join(bad)} (needs F16, BF16 or F32)")
    dims = {d for k in lora for d in header[k].get("shape") or []}
    if WAN14B_DIM not in dims:
        if dims & SMALL_WAN_DIMS:
            raise ValueError("made for a smaller Wan model (1.3B or 5B); the render box runs the 14B models")
        raise ValueError("doesn't match the Wan 14B models (no 5120-wide layers)")
    return len(lora)


def _check_krea2_lora(header):
    """Raise ValueError unless the header looks like a LoRA for Krea 2 Turbo."""
    keys = [k for k in header if k != "__metadata__"]
    lora = [k for k in keys if LORA_KEY.search(k)]
    if not lora:
        raise ValueError("not a LoRA (no lora_A/lora_B or lora_down/lora_up weights)")
    if any(marker in k for k in keys for marker in OTHER_MODEL_KEYS):
        raise ValueError("made for a different model (Flux, SD, SDXL, Qwen-Image, ...), not Krea 2")
    if any(WAN_MARKERS.search(k) for k in keys):
        raise ValueError("this is a Wan LoRA (video model), not a Krea 2 one; add it under a new name "
                         "(vidgen lora add detects the kind)")
    if not any(_krea2_layout(k) for k in lora):
        raise ValueError("doesn't look like a Krea 2 LoRA (no Krea 2 attention or mlp blocks)")
    bad = sorted({str(header[k].get("dtype")) for k in lora if header[k].get("dtype") not in LORA_DTYPES})
    if bad:
        raise ValueError(f"unsupported weight type {', '.join(bad)} (needs F16, BF16 or F32)")
    if KREA2_DIM not in {d for k in lora for d in header[k].get("shape") or []}:
        raise ValueError(f"doesn't match Krea 2 (no {KREA2_DIM}-wide layers)")
    return len(lora)


def _lora_name_model(client, name):
    """The model ('t2v' or 'krea2') of a LoRA name's registered versions, or None for a new name."""
    for v in _lora_versions(client, name):
        meta = _s3_json(f"loras/{client}/{name}/v{v}/lora.json")
        if meta:
            return "krea2" if "krea2" in (meta.get("models") or []) else "t2v"
    return None


def register_lora_upload(client, name, version, body):
    """PUT /loras/{name}/versions/{version} {trigger?, description?, make_current?}: check the
    uploaded file(s) and make the version usable."""
    name = (name or "").strip().lower()
    if not DATASET_NAME.match(name) or not str(version).isdigit():
        return _resp(400, {"error": "bad LoRA name or version"})
    version = int(version)
    base = f"loras/{client}/{name}/v{version}/"
    if _s3_json(base + "lora.json"):
        return _resp(400, {"error": f"'{name}' v{version} already exists"})
    trigger = str(body.get("trigger") or "").strip()
    description = str(body.get("description") or "").strip()
    if len(trigger) > 64 or len(trigger.split()) > 3:
        return _resp(400, {"error": "trigger: a word or short phrase, at most 64 chars"})
    if len(description) > 500:
        return _resp(400, {"error": "description must be <= 500 chars"})
    # The reserve call put the file at krea2.safetensors (Krea 2) or t2v_high/low (Wan); exactly one kind.
    has_krea, has_wan = _s3_exists(base + LORA_FILE["krea2"]), _s3_exists(base + "t2v_high.safetensors")
    if has_krea and has_wan:
        return _resp(400, {"error": "this version has both a Krea 2 and a Wan file; reserve a new version "
                                    "and upload one kind"})
    if not has_krea and not has_wan:
        return _resp(400, {"error": "no file uploaded; PUT the file to the URL from POST .../uploads first"})
    model = "krea2" if has_krea else "t2v"
    existing = _lora_name_model(client, name)  # again here: an unregistered reservation can't see the other
    if existing and existing != model:
        what = {"t2v": "a Wan LoRA", "krea2": "a Krea 2 LoRA"}
        return _resp(400, {"error": f"'{name}' is {what[existing]}; use another name for {what[model]}"})
    files, sizes, tensors = [], {}, 0
    if model == "krea2":
        try:
            header, size = _safetensors_header(base + LORA_FILE["krea2"])
            tensors = _check_krea2_lora(header)
        except ValueError as e:
            return _resp(400, {"error": f"file: {e}"})
        files, sizes = [LORA_FILE["krea2"]], {LORA_FILE["krea2"]: size}
    else:
        for part in ("high", "low"):
            fname = f"t2v_{part}.safetensors"
            try:
                header, size = _safetensors_header(base + fname)
            except ValueError as e:
                if part == "low" and str(e) == "file not uploaded":
                    break
                return _resp(400, {"error": f"{part} file: {e}"})
            try:
                tensors += _check_wan_lora(header)
            except ValueError as e:
                return _resp(400, {"error": f"{part} file: {e}"})
            files.append(fname)
            sizes[fname] = size
    single = len(files) == 1
    meta = {"version": version, "source": "upload", "models": [model], "trigger": trigger,
            "description": description, "files": files, "single_file": single, "sizes": sizes,
            "lora_tensors": tensors, "created_at": int(time.time())}
    _put_json(base + "lora.json", meta)
    if body.get("make_current", True) is not False:
        _put_json(f"loras/{client}/{name}/current.json", {"version": version})
    return _resp(200, {"name": name, "version": version, "model": model, "single_file": single,
                       "trigger": trigger, "current": body.get("make_current", True) is not False})


def lora_files(client, name, version):
    """GET /loras/{name}/versions/{version}/files -> 1-hour download links for a registered version's
    final files, its checkpoints (listed apart) and lora.json, for the local copy (`vidgen lora pull`,
    the watcher). Only the caller's own LoRAs: keys are built from the signed-in client."""
    name = (name or "").strip().lower()
    if not DATASET_NAME.match(name) or not re.fullmatch(r"[0-9]{1,6}", str(version)):
        return _resp(400, {"error": "bad LoRA name or version"})
    base = f"loras/{client}/{name}/v{int(version)}/"
    meta = _s3_json(base + "lora.json")
    if not meta:
        return _resp(404, {"error": f"LoRA '{name}' v{int(version)} not found"})
    listed = [str(f) for f in meta.get("files") or []]
    if not all(LORA_LOCAL_FILE.fullmatch(f) for f in listed):
        return _resp(502, {"error": f"LoRA '{name}' v{int(version)} lists an invalid file name"})

    def link(fname):
        size = s3.head_object(Bucket=BUCKET, Key=base + fname)["ContentLength"]
        return {"url": s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": base + fname},
                                                 ExpiresIn=URL_TTL), "size": size}

    try:  # every listed file or an error: a dropped file would make the local copy look complete
        files = {f: link(f) for f in listed if not f.startswith("checkpoints/")}
        checkpoints = {f: link(f) for f in listed if f.startswith("checkpoints/")}
    except ClientError:
        return _resp(502, {"error": f"LoRA '{name}' v{int(version)}: a listed file is not readable; try again"})
    out = {"name": name, "version": int(version),
           "model": "krea2" if "krea2" in (meta.get("models") or []) else "t2v",
           "files": files, "checkpoints": checkpoints,
           "lora_json": s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": base + "lora.json"},
                                                  ExpiresIn=URL_TTL)}
    return _resp(200, out)


def set_lora_current(client, name, body):
    """PUT /loras/{name}/current {version, step?}: what renders with --lora NAME use."""
    name = (name or "").strip().lower()
    if not DATASET_NAME.match(name):
        return _resp(400, {"error": "bad LoRA name"})
    try:
        version = int(body.get("version"))
        step = int(body["step"]) if body.get("step") not in (None, "") else None
    except (TypeError, ValueError):
        return _resp(400, {"error": "version (and step) must be numbers"})
    _, meta, _ = _lora_version(client, name, version)
    if not meta:
        return _resp(404, {"error": f"LoRA '{name}' v{version} not found"})
    if "krea2" in (meta.get("models") or []):
        if step is not None:
            return _resp(400, {"error": f"'{name}' is a Krea 2 LoRA; it has no checkpoint steps"})
    else:
        try:
            _lora_keys(client, name, version, meta, step)
        except ValueError as e:
            return _resp(400, {"error": str(e)})
    s3.put_object(Bucket=BUCKET, Key=f"loras/{client}/{name}/current.json",
                  Body=json.dumps({"version": version, "step": step}).encode(), ContentType="application/json")
    return _resp(200, {"name": name, "version": version, "step": step})


def get_captions(client, name):
    """GET /datasets/{name}/captions: the trainer's (or your edited) caption per training item."""
    try:
        prefix = _dataset_prefix(client, (name or "").lower())
    except ValueError as e:
        return _resp(400, {"error": str(e)})
    caps = _s3_json(prefix + "captions.json")
    if not caps:
        return _resp(404, {"error": "no captions yet; run `vidgen train NAME --captions-only` (or a training) first"})
    manifest = _s3_json(prefix + "manifest.json") or {}
    current = hashlib.sha256(json.dumps(manifest.get("files", []), sort_keys=True).encode()).hexdigest()[:16]
    return _resp(200, dict(caps, current=caps.get("fingerprint") == current))


def put_captions(client, name, body):
    """PUT /datasets/{name}/captions {items: {item: caption}}: replace captions of existing items
    (the trainer uses them as-is while the dataset is unchanged)."""
    try:
        prefix = _dataset_prefix(client, (name or "").lower())
    except ValueError as e:
        return _resp(400, {"error": str(e)})
    caps = _s3_json(prefix + "captions.json")
    if not caps:
        return _resp(404, {"error": "no captions yet; run `vidgen train NAME --captions-only` first"})
    items = body.get("items")
    if not isinstance(items, dict) or not items:
        return _resp(400, {"error": "items: {item name: caption}"})
    unknown = sorted(set(items) - set(caps.get("items") or {}))
    if unknown:
        return _resp(400, {"error": f"unknown items: {', '.join(unknown[:5])}{' ...' if len(unknown) > 5 else ''}"})
    for k, v in items.items():
        if not isinstance(v, str) or not 1 <= len(v.strip()) <= MAX_CAPTION_CHARS:
            return _resp(400, {"error": f"{k}: caption must be 1-{MAX_CAPTION_CHARS} characters"})
    trigger = caps.get("trigger") or ""
    missing = [k for k, v in items.items() if trigger and trigger.lower() not in v.lower()]
    caps["items"] = dict(caps.get("items") or {}, **{k: v.strip() for k, v in items.items()})
    caps["edited_at"] = int(time.time())
    s3.put_object(Bucket=BUCKET, Key=prefix + "captions.json", Body=json.dumps(caps, indent=1).encode(),
                  ContentType="application/json")
    return _resp(200, {"updated": len(items), "without_trigger": missing})


def list_loras(client):
    """Trained LoRAs: per name the current version's lora.json summary and all versions."""
    base = f"loras/{client}/"
    out = []
    for p in s3.list_objects_v2(Bucket=BUCKET, Prefix=base, Delimiter="/").get("CommonPrefixes", []):
        name = p["Prefix"][len(base):-1]
        versions = []
        for v in s3.list_objects_v2(Bucket=BUCKET, Prefix=p["Prefix"], Delimiter="/").get("CommonPrefixes", []):
            meta = _s3_json(v["Prefix"] + "lora.json")
            if meta:
                versions.append({k: meta.get(k) for k in ("version", "steps_per_expert", "created_at", "train_seconds",
                                                          "trigger", "kind", "models", "source", "description")}
                                | {"checkpoints": sorted(int(f.split("step")[-1].split(".")[0])
                                                         for f in meta.get("files", []) if f.startswith("checkpoints/t2v_low"))})
        if versions:
            cur = _s3_json(p["Prefix"] + "current.json") or {}
            current = cur.get("version") or max(v["version"] for v in versions)
            out.append({"name": name, "current": current, "current_step": cur.get("step"),
                        "versions": sorted(versions, key=lambda v: v["version"])})
    return _resp(200, {"loras": out})


def delete_dataset(client, name):
    try:
        prefix = _dataset_prefix(client, name)
    except ValueError as e:
        return _resp(400, {"error": str(e)})
    keys = _list_keys(prefix)
    if not keys:
        return _resp(404, {"error": f"dataset '{name}' not found"})
    _delete_keys(keys)
    return _resp(200, {"deleted": len(keys)})


# ---------------------------------------------------------------- routes

# Per-job notes for the clients (not part of params, so the worker never sees them): the file name to save as,
# the avatar a candidate job is for and whether it gets a Krea LoRA, and which device sent it. Any client can read
# them back, so a job started on the phone downloads and picks on the Mac. docs/specs/2026-11-27-phone-ui.md
META_SOURCES = ("mac", "cli", "phone")
MAX_OUT_NAME = 128
# ASCII letters, digits, ". _ -" and spaces; starts with a letter or digit, doesn't end in "." or " ". No path
# parts, control or bidi characters, and short enough in bytes for any filesystem.
OUT_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._ -]{0,%d}[A-Za-z0-9_-])?" % (MAX_OUT_NAME - 2))
RERUN_SUFFIX = re.compile(r"(?:-rerun-[0-9a-f]{6})+$")


def _validate_meta(meta):
    if not isinstance(meta, dict):
        raise ValueError("meta must be an object")
    unknown = set(meta) - {"out_name", "avatar_name", "krea_lora", "source"}
    if unknown:
        raise ValueError(f"meta: unknown key(s) {', '.join(sorted(unknown))}")
    out = {}
    if meta.get("out_name") is not None:
        name = meta["out_name"]
        if not isinstance(name, str) or not OUT_NAME.fullmatch(name):
            raise ValueError(f"meta.out_name: a plain file name of 1-{MAX_OUT_NAME} chars: letters, digits, "
                             "spaces and . _ - (no folders, not starting with . or ending in . or a space)")
        out["out_name"] = name
    if meta.get("avatar_name") is not None:
        if not isinstance(meta["avatar_name"], str) or not AVATAR_NAME.fullmatch(meta["avatar_name"]):
            raise ValueError("meta.avatar_name: 1-32 chars, a-z 0-9 - _")
        out["avatar_name"] = meta["avatar_name"]
    if meta.get("krea_lora") is not None:
        if not isinstance(meta["krea_lora"], bool):
            raise ValueError("meta.krea_lora must be true or false")
        out["krea_lora"] = meta["krea_lora"]
    if meta.get("source") is not None:
        if meta["source"] not in META_SOURCES:
            raise ValueError(f"meta.source must be one of: {', '.join(META_SOURCES)}")
        out["source"] = meta["source"]
    return out


def _meta_of(item):
    try:
        meta = json.loads(item.get("meta") or "{}")
    except json.JSONDecodeError:
        return {}
    return meta if isinstance(meta, dict) else {}


def _rerun_meta(meta, new_job_id):
    """The original's meta for its re-run, saved next to it as <base>-rerun-<id6><ext>: a re-run of a re-run
    replaces the suffix, and the base is shortened so the name stays within MAX_OUT_NAME."""
    meta = dict(meta)
    if meta.get("out_name"):
        base, ext = os.path.splitext(meta["out_name"])
        suffix = f"-rerun-{new_job_id[:6]}{ext}"
        base = RERUN_SUFFIX.sub("", base)[:MAX_OUT_NAME - len(suffix)].rstrip(". ") or "job"
        meta["out_name"] = base + suffix
        if not OUT_NAME.fullmatch(meta["out_name"]):
            meta.pop("out_name")
    return meta


def create_job(client, body):
    try:
        meta = _validate_meta(body["meta"]) if body.get("meta") is not None else {}
        job_type, params = _validate(client, body)
        if job_type in ("video", "story"):
            params["avatars"] = _resolve_avatars(client, body.get("avatar"))
            if params.get("image_model") == "auto":
                params["image_model"] = _auto_first_frame(job_type, params["avatars"])
            if params.get("anchor_every") and len(params["avatars"]) != 1:
                raise ValueError("anchor_every needs exactly one avatar in the take")
        if job_type != "lora_preview":  # a preview names its LoRA in "name"
            _job_loras(client, job_type, params, body.get("loras"))
        if job_type == "video" and body.get("avatar_lora") is not False:
            _with_avatar_wan_lora(client, params)
        if job_type in ("video", "story"):
            _avatar_first_frame(client, job_type, params)
    except (ValueError, TypeError) as e:
        return _resp(400, {"error": str(e)})

    return _queue_job(client, job_type, params, **({"meta": meta} if meta else {}))


def _queue_job(client, job_type, params, rerun_of=None, job_id=None, meta=None):
    job_id = job_id or uuid.uuid4().hex
    now = int(time.time())
    item = {
        "job_id": job_id,
        "client": client,
        "type": job_type,
        "params": json.dumps(params),
        "status": "queued",
        "created_at": now,
        "expires_at": now + RETENTION_DAYS * 86400,
    }
    if rerun_of:
        item["rerun_of"] = rerun_of
    if meta:
        item["meta"] = json.dumps(_rerun_meta(meta, job_id) if rerun_of else meta)
    training = job_type == "train"
    if training and not TRAIN_QUEUE_URL:
        return _resp(400, {"error": "the trainer is not deployed (set training_enabled = true in Terraform)"})
    jobs.put_item(Item=item)
    sqs.send_message(QueueUrl=TRAIN_QUEUE_URL if training else QUEUE_URL, MessageBody=json.dumps({"job_id": job_id}))
    # A new job always gets its machine started.
    _set_hold(False, TRAIN_CONTROL_KEY if training else CONTROL_KEY)
    avatars = [a["name"] for a in params.get("avatars") or []]
    loras = [{k: l.get(k) for k in ("name", "version", "step", "weight", "trigger", "model")}
             for l in (params.get("loras") or []) + (params.get("krea_loras") or [])]
    return _resp(202, {"job_id": job_id, "type": job_type, "status": "queued", "avatars": avatars, "loras": loras,
                       "meta": _meta_of(item), "instance": _ensure_started(TRAINER_INSTANCE_ID if training else INSTANCE_ID)})


def get_job(client, job_id):
    item = jobs.get_item(Key={"job_id": job_id}).get("Item")
    if not item or item.get("client") != client:
        return _resp(404, {"error": "job not found"})

    out = {k: item[k] for k in ("job_id", "type", "status", "created_at") if k in item}
    out["meta"] = _meta_of(item)
    for k in ("progress", "error", "started_at", "finished_at", "cancel_requested"):
        if k in item:
            out[k] = item[k]
    if item.get("status") == "done":
        result = json.loads(item.get("result", "{}"))
        out["result"] = {k: v for k, v in result.items() if not k.endswith("_key")}
        out["downloads"] = {
            k[: -len("_key")]: s3.generate_presigned_url(
                "get_object", Params={"Bucket": BUCKET, "Key": v}, ExpiresIn=URL_TTL
            )
            for k, v in result.items()
            if k.endswith("_key")
        }
    return _resp(200, out)


def _num(value):
    return int(value) if value is not None else None


def _realism_summary(params):
    """Only the realism settings that differ from the defaults."""
    out = {k: True for k in ("natural_skin", "candid", "grain") if params.get(k)}
    if params.get("look", "pro") != "pro":
        out["look"] = params["look"]
    if params.get("shot"):
        out["shot"] = params["shot"]
    if params.get("image_model") == "krea2":
        out["first_frame"] = "krea2"
    if params.get("model") == "krea2":
        out["model"] = "krea2"
    return out


def _summary(job_type, params):
    """One line saying what a job was for, plus the settings worth showing in a list."""
    names = [a["name"] for a in params.get("avatars") or []]
    loras = [f"{l['name']}{'@' + str(l['step']) if l.get('step') else ''}:{l['weight']:g}"
             for l in params.get("loras") or []] + [f"{l['name']}:{l['weight']:g} (krea2)"
                                                     for l in params.get("krea_loras") or []]
    if job_type == "video":
        return params.get("prompt", ""), {"seconds": params.get("seconds"), "avatars": names, "loras": loras,
                                          "variations": params.get("variations", 1), "fps": params.get("fps", 16),
                                          "upscale": params.get("upscale", False),
                                          "quality": params.get("quality", "standard"), **_realism_summary(params),
                                          **({"beats": len(params["beats"])} if params.get("beats") else {}),
                                          **({"anchor_every": params["anchor_every"]}
                                             if params.get("anchor_every") else {}),
                                          **({"plan": True} if params.get("plan_only") else {}),
                                          **({"video_model": params["video_model"]} if params.get("video_model") else {})}
    if job_type == "story":
        return " ".join(params.get("story", "").split())[:120], {
            "shots": params.get("max_shots"), "seconds_per_shot": params.get("seconds_per_shot"),
            "avatars": names, "loras": loras, **_realism_summary(params),
            **({"quality": params["quality"]} if params.get("quality", "standard") != "standard" else {})}
    if job_type == "photos":
        extra_info = dict(_realism_summary(params),
                          **({"quality": "best"} if params.get("quality") == "best" else {}))
        if params.get("series"):
            ser = params["series"]
            text = f"series of {params.get('count')} at {ser['where']}" + (f", wearing {ser['wearing']}" if ser.get("wearing") else "")
            return text, {"count": params.get("count"), "aspect": params.get("aspect"), "subject": params.get("subject"),
                          "series": True, **extra_info}
        scenes = [", ".join(v.values()) if isinstance(v, dict) else v
                  for v in (next(iter(s.values())) for s in params.get("scenes", []))]
        extra = params.get("count", 0) - len(scenes)
        text = "; ".join(scenes + ([f"{extra} random location{'s' * (extra > 1)}"] if extra > 0 else []))
        return text, {"count": params.get("count"), "aspect": params.get("aspect"), "subject": params.get("subject"),
                      **({"loras": loras} if loras else {}), **({"raw": True} if params.get("raw") else {}),
                      **extra_info}
    if job_type == "avatar":
        return params.get("description", ""), {"candidates": params.get("candidates"),
                                               **({"model": "krea2"} if params.get("model") == "krea2" else {})}
    if job_type == "describe":
        return params.get("instructions") or "describe a video", {}
    if job_type == "avatar_photoset":
        return f"Krea LoRA photo set for avatar {params.get('avatar')}", {"count": params.get("count"),
                                                                         "lora": params.get("lora")}
    if job_type == "write":
        return params.get("premise", "")[:120], {"words": params.get("words"),
                                                **({"genre": params["genre"]} if params.get("genre") else {})}
    if job_type == "lora_preview":
        return f"preview {params.get('name')} v{params.get('version')}: {params.get('prompt', '')}"[:120], {
            "steps": [c["step"] for c in params.get("clips", [])]}
    if job_type == "train":
        return f"train {params.get('dataset')} ({params.get('kind')}, {params.get('files')} files)", {
            "steps": params.get("steps"), "trigger": params.get("trigger")}
    return "", {}


def list_jobs(client):
    """The caller's latest jobs, newest first, with how long each waited and ran."""
    items = jobs.query(IndexName=JOBS_INDEX, KeyConditionExpression=Key("client").eq(client),
                       ScanIndexForward=False, Limit=JOB_HISTORY_LIMIT).get("Items", [])
    out = []
    for item in items:
        created, started, finished = (_num(item.get(k)) for k in ("created_at", "started_at", "finished_at"))
        summary, details = _summary(item.get("type"), json.loads(item.get("params", "{}")))
        row = {"job_id": item["job_id"], "type": item.get("type"), "status": item.get("status"),
               "created_at": created, "started_at": started, "finished_at": finished,
               "run_seconds": finished - started if started and finished else None,
               "wait_seconds": started - created if started and created else None,
               "summary": summary, "details": details, "meta": _meta_of(item)}
        for k in ("error", "progress", "rerun_of"):
            if item.get(k):
                row[k] = item[k]
        out.append(row)
    return _resp(200, {"jobs": out})


def rerun_job(client, job_id):
    """Queue a copy of one of the caller's jobs with the same validated params."""
    item = jobs.get_item(Key={"job_id": job_id}).get("Item")
    if not item or item.get("client") != client:
        return _resp(404, {"error": "job not found"})
    meta = _meta_of(item)
    return _queue_job(client, item["type"], json.loads(item["params"]), rerun_of=job_id,
                      **({"meta": meta} if meta else {}))


def cancel_job(client, job_id):
    """Queued: cancelled at once. Running: flagged; the worker stops within ~15 s
    (between denoising steps) and marks it cancelled."""
    item = jobs.get_item(Key={"job_id": job_id}).get("Item")
    if not item or item.get("client") != client:
        return _resp(404, {"error": "job not found"})
    try:
        jobs.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #s = :c, cancel_requested = :t, finished_at = :now",
            ConditionExpression=Attr("status").eq("queued"),
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":c": "cancelled", ":t": True, ":now": int(time.time())},
        )
        return _resp(200, {"job_id": job_id, "status": "cancelled"})
    except ClientError as e:
        if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
    try:
        jobs.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET cancel_requested = :t",
            ConditionExpression=Attr("status").eq("running"),
            ExpressionAttributeValues={":t": True},
        )
        return _resp(202, {"job_id": job_id, "status": "running", "cancel_requested": True})
    except ClientError as e:
        if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
    return _resp(409, {"error": f"job is already {item['status']}"})


def create_upload(client, body):
    ext = (body.get("extension") or "mp4").lstrip(".").lower()
    if ext not in VIDEO_EXTS | IMAGE_EXTS:
        return _resp(400, {"error": f"extension must be one of {', '.join(sorted(VIDEO_EXTS | IMAGE_EXTS))}"})
    key = f"uploads/{client}/{uuid.uuid4().hex}.{ext}"
    url = s3.generate_presigned_url(
        "put_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=URL_TTL
    )
    return _resp(200, {"upload_key": key, "put_url": url})


WORKER_MODELS_KEY = "code/worker/models.py"
_CONSTANT = r'^{}\s*=\s*(?:os\.environ\.get\([^,]+,\s*)?"([^"]+)"'


def _models():
    """Deployed model repos plus what the worker code itself pins (Lightning, MMAudio)."""
    out = dict(MODELS)
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=WORKER_MODELS_KEY)
        code = obj["Body"].read().decode()
        find = lambda name: (re.search(_CONSTANT.format(name), code, re.M) or [None, None])[1]
        out["lightning"] = {"repo": find("LIGHTNING_REPO"), "revision": find("LIGHTNING_REVISION"),
                            "default_on": find("LIGHTNING") == "1"}
        out["mmaudio"] = find("MMAUDIO_VARIANT")
        if find("KREA_REPO"):
            out["krea2"] = {"repo": find("KREA_REPO"), "revision": find("KREA_REVISION"),
                            "filter": find("KREA_FILTER_REPO")}
        out["worker_code"] = {"version": obj["ETag"].strip('"')[:12],
                              "deployed_at": int(obj["LastModified"].timestamp())}
    except Exception as e:  # status must still answer if the code object is unreadable
        out["worker_code"] = {"error": type(e).__name__}
    return out


# How far the render machine is from taking a job: the box writes its own percent and line to this object
# (first boot at each step, the worker when it starts); GET /status adds the instance's state to it.
READY_KEY = "status/render.json"
FIRST_BOOT_MINUTES = 45  # a first boot without the model mirror, roughly


def _ready_report():
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=READY_KEY)["Body"].read())
    except Exception:  # none yet, or unreadable: the instance's state still says something
        return None


def _readiness(inst, report, now):
    """{"percent": 0-100, "text": one line} for the render machine."""
    state = inst["State"]["Name"]
    if state == "stopped":
        return {"percent": 0, "text": "Off. It starts when you send a job (or Wake): a few minutes to boot, "
                                      "then the first job loads its model."}
    if state == "pending":
        return {"percent": 5, "text": "Starting the machine."}
    if state != "running":
        return {"percent": 0, "text": "Stopping." if state in ("stopping", "shutting-down") else f"Machine is {state}."}
    launched = inst["LaunchTime"].timestamp()
    if isinstance(report, dict) and (report.get("updated_at") or 0) >= launched - 5:  # reported since this start
        percent = report.get("percent")
        percent = max(0, min(100, int(percent))) if isinstance(percent, (int, float)) and not isinstance(percent, bool) else 50
        return {"percent": percent, "text": " ".join(str(report.get("text") or "Working").split())[:200]}
    minutes = int((now - launched) / 60)
    disks = [d["Ebs"]["AttachTime"].timestamp() for d in inst.get("BlockDeviceMappings") or [] if d.get("Ebs", {}).get("AttachTime")]
    if disks and abs(launched - min(disks)) < 600:  # the disk is as new as this start: a first boot
        return {"percent": min(95, 5 + int(90 * minutes / FIRST_BOOT_MINUTES)),
                "text": f"First boot: installing and downloading the models, about {minutes} of ~{FIRST_BOOT_MINUTES} min "
                        "(an estimate; the box reports details once its worker starts)."}
    return {"percent": 50, "text": "Machine is up; starting the worker (about a minute)."}


def status():
    inst = _instance()
    out = {"instance": inst["State"]["Name"], "queued_jobs": _queue_depth(), "uptime_hold": _hold(),
           "instance_type": inst.get("InstanceType"), "models": _models(),
           "readiness": _readiness(inst, _ready_report(), time.time())}
    if inst["State"]["Name"] == "running":
        out["running_minutes"] = int((time.time() - inst["LaunchTime"].timestamp()) / 60)
    if TRAINER_INSTANCE_ID:
        t = _instance(TRAINER_INSTANCE_ID)
        out["trainer"] = {"instance": t["State"]["Name"], "instance_type": t.get("InstanceType"),
                          "queued_jobs": _queue_depth(TRAIN_QUEUE_URL), "uptime_hold": _hold(TRAIN_CONTROL_KEY)}
        if t["State"]["Name"] == "running":
            out["trainer"]["running_minutes"] = int((time.time() - t["LaunchTime"].timestamp()) / 60)
    else:
        out["trainer"] = None
    return _resp(200, out)


def wake():
    _set_hold(False)
    return _resp(200, {"instance": _ensure_started()})


# POST /worker/restart: a request row in the jobs table. The worker looks for it between jobs and exits, and
# systemd starts it again, which pulls the latest worker code. A render in progress is never cut off.
RESTART_KEY = "_restart_worker"


def restart_worker():
    state = _instance_state()
    if state != "running":
        return _resp(409, {"error": f"the render machine is {state}: its worker starts fresh, with the latest code, "
                                    "the next time the machine starts"})
    jobs.put_item(Item={"job_id": RESTART_KEY, "client": "_system", "requested_at": int(time.time())})
    return _resp(200, {"requested": True, "note": "the worker restarts when it is between jobs (within ~20 s when "
                                                  "idle, else after the running job) and is back in about a minute"})


def waker():
    _wake_or_cap(INSTANCE_ID, QUEUE_URL, MAX_UPTIME_HOURS, CONTROL_KEY, "gpu")
    if TRAINER_INSTANCE_ID:
        _wake_or_cap(TRAINER_INSTANCE_ID, TRAIN_QUEUE_URL, TRAIN_MAX_UPTIME_HOURS, TRAIN_CONTROL_KEY, "trainer")


def _wake_or_cap(instance_id, queue_url, max_hours, hold_key, label):
    inst = _instance(instance_id)
    state = inst["State"]["Name"]
    if state == "running":
        hours = (time.time() - inst["LaunchTime"].timestamp()) / 3600
        if hours > max_hours:
            # Hard cap, independent of the worker: covers a hung job or worker.
            ec2.stop_instances(InstanceIds=[instance_id])
            _set_hold(True, hold_key)
            print(json.dumps({"event": "max_uptime_stop", "machine": label, "hours": round(hours, 2)}))
        return
    if state == "stopped" and _queue_depth(queue_url) > 0 and not _hold(hold_key):
        print(json.dumps({"event": "waker", "machine": label, "instance": _ensure_started(instance_id)}))


# ---------------------------------------------------------------- phone page
# GET /m and GET /m/{file}: the static phone page, the only routes without a signature. They serve fixed files
# bundled with this function and touch no AWS service; every API call the page makes is signed in the browser.
# docs/specs/2026-11-27-phone-ui.md

PHONE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phone")
PHONE_FILES = {  # served name -> content type; nothing else under /m
    "app.js": "text/javascript; charset=utf-8",
    "sign.js": "text/javascript; charset=utf-8",
    "manifest.webmanifest": "application/manifest+json",
}
# Presigned links point at this bucket only (virtual-host style; with or without the region in the host name).
PHONE_S3 = f"https://{BUCKET}.s3.amazonaws.com https://{BUCKET}.s3.{os.environ.get('AWS_REGION', 'us-east-1')}.amazonaws.com"
PHONE_CSP = (f"default-src 'self'; img-src 'self' {PHONE_S3}; media-src 'self' {PHONE_S3}; "
             f"connect-src 'self' {PHONE_S3}; script-src 'self'; style-src 'self' 'unsafe-inline'; "
             "manifest-src 'self'; base-uri 'none'; form-action 'none'; object-src 'none'; worker-src 'none'; "
             "frame-ancestors 'none'")
PHONE_HEADERS = {"content-security-policy": PHONE_CSP, "cache-control": "no-cache", "x-content-type-options": "nosniff",
                 "referrer-policy": "no-referrer", "cross-origin-opener-policy": "same-origin"}


def phone_file(name):
    """The page (name None) or one of PHONE_FILES, with the page's fixed headers; 404 for anything else."""
    if name is None:
        fname, ctype = "m.html", "text/html; charset=utf-8"
    elif name in PHONE_FILES:
        fname, ctype = name, PHONE_FILES[name]
    else:
        return dict(_resp(404, {"error": "not found"}), headers={"content-type": "application/json", **PHONE_HEADERS})
    try:
        with open(os.path.join(PHONE_DIR, fname), encoding="utf-8") as f:
            body = f.read()
    except OSError as e:  # the bundle is missing the page (make build wasn't run)
        print(json.dumps({"event": "phone_page_missing", "file": fname, "error": str(e)}))
        return dict(_resp(500, {"error": "the phone page is not deployed"}),
                    headers={"content-type": "application/json", **PHONE_HEADERS})
    return {"statusCode": 200, "body": body, "headers": {"content-type": ctype, **PHONE_HEADERS}}


# ---------------------------------------------------------------- entrypoint

def handler(event, context):
    if event.get("waker"):
        waker()
        return {"ok": True}
    if event.get("routeKey") == "GET /m":
        return phone_file(None)
    if event.get("routeKey") == "GET /m/{file}":
        return phone_file((event.get("pathParameters") or {}).get("file"))

    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    raw = event.get("body") or ""
    raw = base64.b64decode(raw) if event.get("isBase64Encoded") else raw.encode()
    client = _authenticate(event, headers, raw)
    if client is None:
        return _resp(401, {"error": "unauthorized"})

    route = event.get("routeKey")
    try:
        body = json.loads(raw or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _resp(400, {"error": "body must be JSON"})

    if route == "POST /jobs":
        return create_job(client, body)
    params = event.get("pathParameters") or {}
    if route == "GET /jobs":
        return list_jobs(client)
    if route == "POST /jobs/{job_id}/rerun":
        return rerun_job(client, params["job_id"])
    if route == "GET /jobs/{job_id}":
        return get_job(client, params["job_id"])
    if route == "POST /jobs/{job_id}/cancel":
        return cancel_job(client, params["job_id"])
    if route == "POST /uploads":
        return create_upload(client, body)
    if route == "GET /status":
        return status()
    if route == "POST /wake":
        return wake()
    if route == "POST /worker/restart":
        return restart_worker()
    if route == "GET /avatars":
        return list_avatars(client)
    if route == "POST /avatars":
        return create_avatar(client, body)
    if route == "PUT /avatars/default":
        return set_default_avatar(client, body)
    if route == "GET /avatars/{name}":
        return get_avatar(client, params["name"].lower())
    if route == "DELETE /avatars/{name}":
        return delete_avatar(client, params["name"], body)
    if route == "POST /avatars/{name}/images":
        return add_avatar_images(client, params["name"].lower(), body)
    if route == "DELETE /avatars/{name}/images/{n}":
        return remove_avatar_image(client, params["name"].lower(), params["n"])
    if route == "PUT /avatars/{name}/main":
        return set_main_avatar_image(client, params["name"].lower(), body)
    if route == "GET /avatars/{name}/history":
        return avatar_history(client, params["name"].lower())
    if route == "POST /avatars/{name}/krea-lora":
        return avatar_krea_lora(client, params["name"], body)
    if route == "POST /avatars/{name}/wan-lora":
        return avatar_wan_lora(client, params["name"], body)
    if route == "POST /avatars/{name}/media":
        return avatar_media_create(client, params["name"], body)
    if route == "GET /avatars/{name}/media":
        return avatar_media_get(client, params["name"])
    if route == "POST /avatars/{name}/restore":
        return restore_avatar(client, params["name"].lower(), body)
    if route == "GET /datasets":
        return list_datasets(client)
    if route == "GET /loras":
        return list_loras(client)
    if route == "GET /datasets/{name}/captions":
        return get_captions(client, params.get("name"))
    if route == "PUT /datasets/{name}/captions":
        return put_captions(client, params.get("name"), body)
    if route == "PUT /loras/{name}/current":
        return set_lora_current(client, params.get("name"), body)
    if route == "POST /loras/{name}/uploads":
        return lora_uploads(client, params.get("name"), body)
    if route == "PUT /loras/{name}/versions/{version}":
        return register_lora_upload(client, params.get("name"), params.get("version"), body)
    if route == "POST /uploads/complete":
        return uploads_complete(client, body)
    if route == "POST /uploads/abort":
        return uploads_abort(client, body)
    if route == "GET /loras/{name}/versions/{version}/files":
        return lora_files(client, params.get("name"), params.get("version"))
    if route == "GET /datasets/{name}":
        return get_dataset(client, params["name"])
    if route == "POST /datasets/{name}/uploads":
        return dataset_uploads(client, params["name"], body)
    if route == "PUT /datasets/{name}":
        return commit_dataset(client, params["name"], body)
    if route == "DELETE /datasets/{name}":
        return delete_dataset(client, params["name"])
    return _resp(404, {"error": "unknown route"})
