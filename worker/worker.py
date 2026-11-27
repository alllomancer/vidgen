"""GPU worker: pulls jobs from SQS, runs them, uploads results to S3, and stops
its own EC2 instance after IDLE_MINUTES with nothing to do.

Each running job is watched by a JobWatch thread that keeps the SQS message
invisible, honours cancel requests and enforces JOB_TIMEOUT_MINUTES."""

import hashlib
import json
import os

# Before anything imports huggingface_hub: its Xet transfers at full speed (weights are tens of GB).
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import traceback
import urllib.request

import boto3
from botocore.exceptions import ClientError
from PIL import Image

import models
import helios
import krea
import mirror
import photos
import realism
import realism_check

REGION = os.environ["AWS_REGION"]
BUCKET = os.environ["BUCKET"]
QUEUE_URL = os.environ["QUEUE_URL"]


def load_settings(folder=os.path.dirname(os.path.abspath(__file__))):
    """settings.json next to the worker code (written by Terraform, refreshed on every worker start, so
    settings added after a box was created still reach it)."""
    try:
        with open(os.path.join(folder, "settings.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


TRAIN_QUEUE_URL = os.environ.get("TRAIN_QUEUE_URL") or load_settings().get("train_queue_url", "")  # "" = no trainer
JOBS_TABLE = os.environ["JOBS_TABLE"]
IDLE_SECONDS = int(os.environ.get("IDLE_MINUTES", "15")) * 60
JOB_TIMEOUT_SECONDS = int(os.environ.get("JOB_TIMEOUT_MINUTES", "360")) * 60
VISIBILITY_SECONDS = 1800
WATCH_SECONDS = 15
FINISHED = ("done", "failed", "cancelled")

s3 = boto3.client("s3", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION)
ec2 = boto3.client("ec2", region_name=REGION)
jobs = boto3.resource("dynamodb", region_name=REGION).Table(JOBS_TABLE)


WORK_ROOT = os.environ.get("WORK_ROOT", "/opt/ai-video/work")  # long takes' finished segments (resume)


WORK_MAX_AGE = 3 * 86400  # a job's saved segments outlive it only if it never came back (dead-lettered)


def take_work_dir(job_id):
    return os.path.join(WORK_ROOT, job_id)


def sweep_work_dirs():
    """Remove work folders untouched for WORK_MAX_AGE (jobs that were never delivered again)."""
    try:
        names = os.listdir(WORK_ROOT)
    except OSError:
        return
    for name in names:
        path = os.path.join(WORK_ROOT, name)
        try:
            if time.time() - os.path.getmtime(path) > WORK_MAX_AGE:
                shutil.rmtree(path, ignore_errors=True)
                log(event="work_dir_swept", job_id=name)
        except OSError:
            pass


def log(**fields):
    print(json.dumps(fields, default=str), flush=True)


# ---------------------------------------------------------------- self-stop

def _instance_id():
    token_req = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    token = urllib.request.urlopen(token_req, timeout=2).read().decode()
    id_req = urllib.request.Request(
        "http://169.254.169.254/latest/meta-data/instance-id",
        headers={"X-aws-ec2-metadata-token": token},
    )
    return urllib.request.urlopen(id_req, timeout=2).read().decode()


def _queue_empty():
    attrs = sqs.get_queue_attributes(
        QueueUrl=QUEUE_URL,
        AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"],
    )["Attributes"]
    return int(attrs["ApproximateNumberOfMessages"]) + int(attrs["ApproximateNumberOfMessagesNotVisible"]) == 0


# The render machine's readiness for the dashboard (GET /status reads it): a percent and one line.
READY_KEY = "status/render.json"


def report_ready(percent, text):
    """Best effort: a failed write never stops the worker."""
    try:
        s3.put_object(Bucket=BUCKET, Key=READY_KEY, ContentType="application/json",
                      Body=json.dumps({"percent": int(percent), "text": text, "updated_at": int(time.time())}))
    except Exception as e:  # noqa: BLE001
        log(event="report_ready_failed", error=f"{type(e).__name__}: {e}"[:200])


# A restart asked for from the dashboard (API: POST /worker/restart): a row in the jobs table with the time
# of the request. A request newer than this worker's start means "exit between jobs"; systemd then starts
# the worker again (Restart=on-failure), which pulls the latest code.
RESTART_KEY = "_restart_worker"
STARTED_AT = time.time()


def restart_requested():
    try:
        item = jobs.get_item(Key={"job_id": RESTART_KEY}).get("Item") or {}
        return float(item.get("requested_at") or 0) > STARTED_AT
    except Exception:  # noqa: BLE001 - a failed lookup never restarts the worker
        return False


def stop_self():
    instance_id = _instance_id()
    report_ready(0, "Stopping: no jobs for a while.")
    log(event="idle_stop", instance=instance_id)
    ec2.stop_instances(InstanceIds=[instance_id])


_preload = {"on": True}


def preload_idle():
    """Queue empty: load one more big model into RAM where the box has room for all of them (see
    models.preload_next). A load that fails is logged and not tried again until the worker restarts."""
    if not _preload["on"]:
        return
    start = time.time()
    try:
        name = models.preload_next()
    except Exception as e:
        _preload["on"] = False
        log(event="preload_failed", error=str(e)[:300])
        return
    if name:
        log(event="preloaded", model=name, seconds=round(time.time() - start, 1))
    else:
        _preload["on"] = False  # all loaded, or a box that holds one model: nothing more to do


# ---------------------------------------------------------------- job plumbing

class JobWatch(threading.Thread):
    """While a job runs: extend SQS visibility, poll for a cancel request and
    enforce the timeout. Cancel/timeout set models.CANCEL, which the models
    check between denoising steps."""

    def __init__(self, job_id, receipt):
        super().__init__(daemon=True)
        self.job_id = job_id
        self.receipt = receipt
        self.started = time.time()
        self.stopped = threading.Event()

    def run(self):
        last_extend = time.time()
        while not self.stopped.wait(WATCH_SECONDS):
            try:
                if time.time() - last_extend > VISIBILITY_SECONDS // 3:
                    sqs.change_message_visibility(
                        QueueUrl=QUEUE_URL, ReceiptHandle=self.receipt, VisibilityTimeout=VISIBILITY_SECONDS
                    )
                    last_extend = time.time()
                if time.time() - self.started > JOB_TIMEOUT_SECONDS:
                    models.request_cancel(f"timed out after {JOB_TIMEOUT_SECONDS // 60} min")
                item = jobs.get_item(Key={"job_id": self.job_id}, ProjectionExpression="cancel_requested")
                if item.get("Item", {}).get("cancel_requested"):
                    models.request_cancel("cancelled")
            except Exception:  # never let the watcher die mid-job
                traceback.print_exc()


def set_status(job_id, status, **extra):
    names = {"#s": "status"}
    values = {":s": status}
    expr = "SET #s = :s"
    for k, v in extra.items():
        names[f"#{k}"] = k
        values[f":{k}"] = v
        expr += f", #{k} = :{k}"
    jobs.update_item(
        Key={"job_id": job_id},
        UpdateExpression=expr,
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
    )


def progress(job_id, text):
    models.check_cancel()
    set_status(job_id, "running", progress=text)


def upload(local_path, client, job_id, name, content_type):
    key = f"outputs/{client}/{job_id}/{name}"
    s3.upload_file(local_path, BUCKET, key, ExtraArgs={"ContentType": content_type})
    return key


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


# ---------------------------------------------------------------- job types

def reference_keys(params):
    """Image keys for one avatar (or a photos job): reference_keys (multi-image avatars,
    first 3 images) or the single reference_key of older avatars and jobs."""
    return list(params.get("reference_keys") or [params["reference_key"]])


def load_references(keys, workdir, prefix):
    images = []
    for j, key in enumerate(keys):
        path = os.path.join(workdir, f"{prefix}_{j}" + os.path.splitext(key)[1])
        s3.download_file(BUCKET, key, path)
        images.append(Image.open(path).convert("RGB"))
    return images


def load_cast(params, workdir):
    """[(name, [PIL reference images], description)] for the job's avatars; image 1 first."""
    cast = []
    for i, avatar in enumerate(params.get("avatars") or []):
        refs = load_references(reference_keys(avatar), workdir, f"avatar_{i}")
        cast.append((avatar["name"], refs, avatar.get("description", "")))
    return cast


LORA_CACHE = "/opt/models/loras"


def prepare_loras(params):
    """Download the job's LoRA files (kept under LORA_CACHE per version, so a version is
    fetched once) and return what models.set_loras expects."""
    out = []
    for lora in params.get("loras") or []:
        local = {part: _cached_lora(lora[f"{part}_key"]) for part in ("high", "low")}
        out.append({"adapter": _adapter_name(lora), "weight": float(lora["weight"]),
                    "trigger": lora.get("trigger") or "", **local})
    return out


def prepare_krea_loras(params):
    """The job's Krea 2 LoRA files (one per LoRA, `key`), cached like Wan ones; what krea.generate expects."""
    return [{"path": _cached_lora(lora["key"]), "adapter": _adapter_name(lora), "weight": float(lora["weight"]),
             "trigger": lora.get("trigger") or ""} for lora in params.get("krea_loras") or []]


def _cached_lora(key):
    path = os.path.join(LORA_CACHE, key.split("/", 1)[1])  # <client>/<name>/v<N>/<file>
    if not os.path.realpath(path).startswith(os.path.realpath(LORA_CACHE) + os.sep):
        raise ValueError(f"LoRA key outside the LoRA cache: {key}")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s3.download_file(BUCKET, key, path + ".part")
        os.replace(path + ".part", path)
    return path


def _adapter_name(lora):
    adapter = "user_" + "".join(c if c.isalnum() else "_" for c in lora["name"]) + f"_v{lora['version']}"
    if lora.get("step"):
        adapter += f"_s{int(lora['step'])}"
    return adapter


def describe_cast(cast):
    return " ".join(f"{name}: {desc}." if desc else f"{name}." for name, _, desc in cast)


SOUND_CHUNK = 5     # longer takes get one MMAudio pass per 5 s (it is trained on 8 s clips)
SOUND_OVERLAP = 0.2


def sound_for_clip(path, prompt, seconds, seed, workdir, tag="sound"):
    """MMAudio sound for a clip; takes over SOUND_CHUNK s are done in overlapping chunks joined
    with a short crossfade, so every chunk stays within what the model was trained on."""
    out = os.path.join(workdir, f"{tag}.wav")
    if seconds <= SOUND_CHUNK + 3:
        return models.generate_sound(path, out, prompt, seconds, seed)
    chunks, start, n = [], 0.0, 0
    with models.sound_session():  # the sound model stays on the GPU for all the chunks
        while start < seconds - 0.01:
            length = min(SOUND_CHUNK + SOUND_OVERLAP, seconds - start)
            piece = os.path.join(workdir, f"{tag}_part{n}.mp4")
            ffmpeg("-ss", f"{start:.3f}", "-i", path, "-t", f"{length:.3f}", "-an", "-c:v", "libx264",
                   "-preset", "veryfast", "-crf", "18", piece)
            chunks.append(models.generate_sound(piece, os.path.join(workdir, f"{tag}_part{n}.wav"), prompt,
                                                length, None if seed is None else int(seed) + n))
            start += SOUND_CHUNK
            n += 1
    joined = chunks[0]
    for i, nxt in enumerate(chunks[1:], 1):
        mixed = os.path.join(workdir, f"{tag}_join{i}.wav")
        ffmpeg("-i", joined, "-i", nxt, "-filter_complex", f"acrossfade=d={SOUND_OVERLAP}:c1=tri:c2=tri", mixed)
        joined = mixed
    shutil.copy(joined, out)
    return out


# Helios videos (params["video_model"] == "helios", docs/specs/2026-11-26-helios-default.md): the first frame
# is made in Helios's 5:3 shape at twice its size, so the image models work at a size they are good at.
HELIOS_FRAME = (1280, 768)


def render_helios(job_id, params, args, seed, out, tag=""):
    """One Helios take written to `out`: the whole length in one run from one prompt (the same look, shot and
    natural-style wording as Wan), from the first frame when there is one, with the job's Wan LoRAs through
    their low-noise files, each at its weight. Grain and the 2x upscale work on the result; there are no
    segments to chain."""
    loras = [{"path": l["low"], "weight": l.get("weight", 1.0)} for l in models._JOB_LORAS if l.get("low")]
    prompt, _ = models.natural_prompts(models.with_triggers(args["prompt"]))
    frames = helios.generate(prompt, params["seconds"], seed=seed, image=args.get("image"), loras=loras,
                             progress=lambda msg: progress(job_id, msg + tag))
    if params.get("upscale"):
        import enhance

        progress(job_id, f"upscaling{tag}")
        frames = enhance.upscale(frames, models.check_cancel)
    effect = realism.frame_effect(params, seed)  # the phone look, or the plain grain switch
    models.write_video(frames, out, fps=helios.FPS,
                       grain=effect if callable(effect) else (realism.Grain(seed or 0) if effect else None))
    return out


def render_take(job, params, workdir, cast, seed, index=None, count=1):
    """One video (keyframe if there is a cast, then Wan, then optional sound). Returns
    (video path, keyframe path or None). `index` numbers the files of a variation."""
    job_id = job["job_id"]
    tag = f" {index}/{count}" if index else ""
    suffix = f"_{index}" if index else ""
    args = {k: params[k] for k in ("prompt", "negative_prompt", "width", "height")}
    styled = realism.with_style(params["prompt"], params)  # look / natural skin / candid; unchanged without
    args.update(prompt=styled, seconds=params["seconds"], seed=seed, fps=params.get("fps", 16),
                upscale=params.get("upscale", False), grain=realism.frame_effect(params, seed))
    if params.get("_work_dir"):  # long take: saved segments, so a re-delivered job resumes
        def segment_progress(text):  # best effort: a status write failing must not end a long take
            try:
                progress(job_id, text)
            except Exception as e:  # noqa: BLE001
                log(event="progress_failed", job_id=job_id, error=str(e)[:200])

        args.update(work_dir=params["_work_dir"], progress=segment_progress)
    timeline = params.get("_segment_prompts")  # long take: one prompt per 5 s segment
    if timeline:
        args["segment_prompts"] = [realism.with_style(t, params) for t in timeline]
    keyframe_path = None
    if params.get("_first_frame") is not None:  # Krea 2 first frame (no cast), made in run_video
        keyframe_path = os.path.join(workdir, f"keyframe{suffix}.png")
        params["_first_frame"].save(keyframe_path)
        args["image"] = params["_first_frame"]
    elif cast:
        keyframe = params.get("_keyframe")  # made before the planner (single takes)
        if keyframe is None:
            progress(job_id, f"placing avatar in scene{tag}")
            keyframe = models.make_keyframe(cast, styled, params["width"], params["height"], seed,
                                            best=params.get("quality") == "best")
        keyframe_path = os.path.join(workdir, f"keyframe{suffix}.png")
        keyframe.save(keyframe_path)
        args["image"] = keyframe
        args["prompt"] = f"{describe_cast(cast)} {styled}"
        if timeline:
            args["segment_prompts"] = [f"{describe_cast(cast)} {t}" for t in args["segment_prompts"]]
    if params.get("anchor_every") and len(cast) == 1:  # long take: restore the face every N segments
        refs, width, height = cast[0][1], params["width"], params["height"]
        args.update(anchor_every=int(params["anchor_every"]),
                    reanchor=lambda frame, n=0: models.reanchor(frame, refs, width, height, (seed or 0) + n))
    if timeline and params.get("beats") and "image" not in args:  # text-to-video start: beat 1 + the scene
        args["segment_prompts"][0] = realism.with_style(f"{params['prompt']} {timeline[0]}", params)
    progress(job_id, f"rendering video{tag}")
    name = f"video{suffix}"
    if params.get("video_model") == "helios":
        path = render_helios(job_id, params, args, seed, os.path.join(workdir, f"{name}.mp4"), tag)
    else:
        path = models.generate_video(os.path.join(workdir, f"{name}.mp4"), **args)
    if params.get("sound", True):
        progress(job_id, f"adding sound{tag}")
        sound = sound_for_clip(path, params["prompt"], params["seconds"], seed, workdir, f"sound_{name}")
        path = add_audio(path, sound, os.path.join(workdir, f"{name}_sound.mp4"))
    return path, keyframe_path


def run_video(job, params, workdir):
    client, job_id = job["client"], job["job_id"]
    cast = load_cast(params, workdir)
    count = int(params.get("variations") or 1)
    # No seed given: one derived from the job id, so a re-delivered take matches its saved segments.
    base = params["seed"] if params.get("seed") is not None else \
        int.from_bytes(hashlib.sha256(job_id.encode()).digest()[:4], "big")
    helios_job = params.get("video_model") == "helios"  # one run, one prompt: no segments, planner or resume
    if helios_job:
        params = dict(params, width=HELIOS_FRAME[0], height=HELIOS_FRAME[1])  # the size of the first frame
    work_dir = take_work_dir(job_id) if count == 1 and not helios_job and models.segment_count(params["seconds"]) > 1 \
        else None
    frames = krea_first_frames(job, params, cast, [base + i for i in range(count)])
    if count == 1:
        # The first frame comes first, so the planner can see the scene it is planning.
        keyframe_img, first_path = make_first_frame(job, params, cast, base, workdir, frames[0])
        if params.get("plan_only"):
            return plan_only_result(job, params, cast, base, first_path)
        timeline = None if helios_job else take_timeline(job, params, cast, image=first_path)
        path, keyframe = render_take(job, dict(params, _first_frame=frames[0], _keyframe=keyframe_img,
                                               _segment_prompts=timeline, _work_dir=work_dir), workdir, cast, base)
        result = {"video_key": upload(path, client, job_id, "video.mp4", "video/mp4"),
                  **({"video_model": "helios"} if helios_job else {})}
        if timeline:
            result["segment_prompts"] = timeline
        if keyframe:
            result["keyframe_key"] = upload(keyframe, client, job_id, "keyframe.png", "image/png")
        return result
    # Variations: same prompt, seeds base, base+1, ...; models stay loaded between takes.
    result = {"seeds": [base + i for i in range(count)], **({"video_model": "helios"} if helios_job else {})}
    for i in range(count):
        path, keyframe = render_take(job, dict(params, _first_frame=frames[i]), workdir, cast, base + i, i + 1, count)
        result[f"video_{i + 1}_key"] = upload(path, client, job_id, f"video_{i + 1}.mp4", "video/mp4")
        if keyframe:
            result[f"keyframe_{i + 1}_key"] = upload(keyframe, client, job_id, f"keyframe_{i + 1}.png", "image/png")
    return result


def make_first_frame(job, params, cast, seed, workdir, krea_frame=None):
    """A single take's first frame before anything else: the Krea frame when there is one, else the Qwen avatar
    keyframe when there is a cast, else none (text to video). Returns (keyframe image or None, saved path or None);
    the keyframe image is what render_take then animates (it isn't made twice)."""
    image, keyframe = krea_frame, None
    if image is None and cast:
        progress(job["job_id"], "placing avatar in scene")
        keyframe = image = models.make_keyframe(cast, realism.with_style(params["prompt"], params), params["width"],
                                                params["height"], seed, best=params.get("quality") == "best")
    if image is None:
        return None, None
    path = os.path.join(workdir, "first_frame.png")
    image.save(path)
    return keyframe, path


def plan_only_result(job, params, cast, seed, first_path):
    """A plan-only job: the planner's prompts (seeing the first frame) as editable beats every 5 s, the seed that
    makes the same first frame, and the frame itself; no video. A planner failure fails the job."""
    n = models.segment_count(params["seconds"])
    progress(job["job_id"], f"planning {n} segments")
    try:
        plan = models.plan_segments(params["prompt"], n, [(name, desc) for name, _, desc in cast],
                                    seconds=int(params["seconds"]), image=first_path)
    except models.JobCancelled:
        raise
    except Exception as e:  # noqa: BLE001 - shown to the user with what to do
        raise RuntimeError(f"the planner could not write a timeline ({type(e).__name__}); try again or write "
                           "beats yourself") from e
    if not models.valid_segment_plan(plan, n):
        raise RuntimeError("the planner could not write a timeline; try again or write beats yourself")
    # One line per beat; ";" would split a beat in the API's parser, so it becomes ",".
    beats = "\n".join(f"{models.SEGMENT_SECONDS * i}: {' '.join(p.replace(';', ',').split())}"
                      for i, p in enumerate(plan))
    result = {"plan_only": True, "segment_prompts": plan, "beats": beats, "seed": seed}
    if first_path:
        result["keyframe_key"] = upload(first_path, job["client"], job["job_id"], "keyframe.png", "image/png")
    return result


def take_timeline(job, params, cast=(), image=None):
    """A long take's prompt per 5 s segment, or None when every segment just repeats the prompt.
    From the user's beats; else, for takes of more than 3 segments, from the planner (VLM, before the
    video model loads). A planner failure is logged and the take keeps the user's prompt."""
    n = models.segment_count(params["seconds"])
    if n < 2:
        return None
    if params.get("beats"):
        return models.segment_prompts(params["prompt"], n, params["beats"])
    if n < models.PLANNED_MIN_SEGMENTS:
        return None
    saved = os.path.join(take_work_dir(job["job_id"]), "timeline.json")
    try:  # a re-delivered take keeps the timeline its saved segments were rendered with
        with open(saved) as f:
            plan = json.load(f)
        if models.valid_segment_plan(plan, n):
            return plan
    except (OSError, ValueError):
        pass
    progress(job["job_id"], f"planning {n} segments")
    try:
        plan = models.plan_segments(params["prompt"], n, [(name, desc) for name, _, desc in cast],
                                    seconds=int(params["seconds"]), image=image)
    except models.JobCancelled:
        raise
    except Exception as e:  # noqa: BLE001 - any planner failure: the take goes on with one prompt
        log(event="segment_plan_failed", job_id=job["job_id"], error=f"{type(e).__name__}: {e}"[:300])
        return None
    if not models.valid_segment_plan(plan, n):
        log(event="segment_plan_failed", job_id=job["job_id"], error="unusable plan")
        return None
    try:
        os.makedirs(os.path.dirname(saved), exist_ok=True)
        with open(saved + ".tmp", "w") as f:
            json.dump(plan, f)
        os.replace(saved + ".tmp", saved)
    except OSError as e:  # only resume needs it; the take goes on
        log(event="timeline_save_failed", job_id=job["job_id"], error=str(e)[:300])
    return plan


def krea_first_frames(job, params, cast, seeds):
    """With image_model krea2 and no avatar: one Krea 2 first frame per take (all in one Krea run,
    before Wan loads). [None, ...] otherwise."""
    if params.get("image_model") != "krea2" or (cast and not params.get("krea_avatar")):
        return [None] * len(seeds)
    prompt = realism.with_style(params["prompt"], params)
    frames, stats = krea.generate([{"prompt": prompt, "width": params["width"], "height": params["height"],
                                    "seed": s} for s in seeds], lambda m: progress(job["job_id"], m),
                                  loras=prepare_krea_loras(params))
    if any(f is None for f in frames):
        raise krea.FilteredOut("the first frame was flagged by the content filter; change the prompt")
    return [models.fit(f, params["width"], params["height"]) for f in frames]


def add_audio(video, audio, out):
    """Mux `audio` into `video` without re-encoding the picture."""
    ffmpeg("-i", video, "-i", audio, "-map", "0:v", "-map", "1:a",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", out)
    return out


def run_avatar(job, params, workdir):
    result = {"description": params["description"], "model": params.get("model") or "wan"}
    if result["model"] == "krea2":
        base = params["seed"] if params.get("seed") is not None else int.from_bytes(os.urandom(4), "big")
        prompt = models.avatar_prompt(params["description"])
        images, stats = krea.generate([{"prompt": prompt, "width": 896, "height": 1152, "seed": base + i}
                                       for i in range(params["candidates"])], lambda m: progress(job["job_id"], m))
        result["krea"] = stats
        images = [im for im in images if im is not None]  # flagged by the content filter
        if not images:
            raise krea.FilteredOut("every candidate was flagged by the content filter; change the description")
    else:
        images = models.generate_avatar_candidates(params["description"], params["candidates"], params.get("seed"))
    for i, image in enumerate(images):
        path = os.path.join(workdir, f"candidate_{i + 1}.png")
        image.save(path)
        result[f"candidate_{i + 1}_key"] = upload(path, job["client"], job["job_id"], f"candidate_{i + 1}.png", "image/png")
    return result


def run_photos(job, params, workdir):
    """Photorealistic stills of one person (uploaded image or avatar) in several scenes, or a
    series in one place (photos 2..N anchored on photo 1 unless anchor is off)."""
    client, job_id = job["client"], job["job_id"]
    if params.get("model") == "krea2":
        return run_photos_krea(job, params, workdir)
    references = load_references(reference_keys(params), workdir, "reference")
    width, height = photos.ASPECTS[params["aspect"]]
    base = params["seed"] if params.get("seed") is not None else int.from_bytes(os.urandom(4), "big")
    series = params.get("series")
    scenes = (photos.build_series(series, params["count"], base) if series
              else photos.build_scenes(params["scenes"], params["count"], base))
    anchor = bool(series and series.get("anchor", True))
    look = dict(look=params.get("look") or "pro", natural_skin=bool(params.get("natural_skin")),
                candid=bool(params.get("candid")))
    best = params.get("quality") == "best"
    result = {"seed": base, "aspect": params["aspect"], "scenes": [], "look": look["look"],
              "quality": "best" if best else "fast"}
    first = None
    for i, scene in enumerate(scenes, 1):
        progress(job_id, f"photo {i}/{len(scenes)}: {scene['text'][:80]}")
        anchored = anchor and first is not None and scene.get("anchor", True)
        refs = references[:photos_refs_limit(anchored)]
        prompt = photos.photo_prompt(scene["scene"], len(refs), scene.get("keep_outfit", True),
                                     anchor=anchored, **look)
        image = models.make_photo(refs + ([first] if anchored else []), prompt, photos.NEGATIVE,
                                  width, height, base + i, best=best)
        if first is None:
            first = image
        if params.get("grain"):
            image = photos.add_grain(image, base + i)
        path = os.path.join(workdir, f"photo_{i}.png")
        image.save(path)
        result[f"photo_{i}_key"] = upload(path, client, job_id, f"photo_{i}.png", "image/png")
        result["scenes"].append({"photo": i, "source": scene["source"], "text": scene["text"]})
    return result


def run_photos_krea(job, params, workdir):
    """Photos without an avatar: Krea 2 Turbo from text (scenes and random places), realism
    sentences and grain as for avatar photos. Flagged photos are left out and counted."""
    client, job_id = job["client"], job["job_id"]
    width, height = photos.ASPECTS[params["aspect"]]
    base = params["seed"] if params.get("seed") is not None else int.from_bytes(os.urandom(4), "big")
    scenes = photos.build_scenes(params["scenes"], params["count"], base)
    look = dict(look=params.get("look") or "pro", natural_skin=bool(params.get("natural_skin")),
                candid=bool(params.get("candid")))
    texts = [s["text"] if s["source"] in ("random", "location") else photos.text_scene(src)
             for s, src in zip(scenes, params["scenes"] + [None] * len(scenes))]
    raw = params.get("raw") is True  # the prompt exactly as typed: no look sentences, no trigger words
    if raw:  # only the user's own text: extra photos repeat the prompts (new seeds), no random places
        given = params["scenes"]
        scenes = [dict(source=next(iter(given[i % len(given)])), text=photos.text_scene(given[i % len(given)]))
                  for i in range(params["count"])]
        texts = [s["text"] for s in scenes]
        who = params.get("avatar_trigger")  # an avatar's name is the person, so it goes in even in raw mode
        if who:
            texts = [t if krea.has_word(t, who) else f"{who} {t}" for t in texts]
    images, stats = krea.generate([{"prompt": t if raw else photos.scene_prompt(t, **look), "width": width,
                                    "height": height, "seed": base + i} for i, t in enumerate(texts, 1)],
                                  lambda m: progress(job_id, m), loras=prepare_krea_loras(params),
                                  add_triggers=not raw)
    result = {"seed": base, "aspect": params["aspect"], "scenes": [], "look": look["look"], "model": "krea2",
              "krea": stats}
    n = 0
    for i, (image, text) in enumerate(zip(images, texts), 1):
        if image is None:
            continue
        n += 1
        if params.get("grain"):
            image = photos.add_grain(image, base + i)
        path = os.path.join(workdir, f"photo_{n}.png")
        image.save(path)
        result[f"photo_{n}_key"] = upload(path, client, job_id, f"photo_{n}.png", "image/png")
        result["scenes"].append({"photo": n, "source": scenes[i - 1]["source"], "text": text})
    if not n:
        raise krea.FilteredOut("every photo was flagged by the content filter; change the scenes")
    return result


def photos_refs_limit(anchored):
    """Images of the person per edit call: the model takes 3, one slot goes to the anchor photo."""
    return models.MAX_EDIT_IMAGES - 1 if anchored else models.MAX_EDIT_IMAGES


def run_write(job, params, workdir):
    """A short story from a premise (text only, the planner's model); story.txt can go to `story`."""
    seed = params["seed"] if params.get("seed") is not None else int.from_bytes(os.urandom(4), "big")
    progress(job["job_id"], "writing")
    story = models.write_story(params["premise"], params["words"], params.get("genre", ""), seed)
    path = os.path.join(workdir, "story.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{story['title']}\n\n{story['text']}\n")
    return {"title": story["title"], "words": len(story["text"].split()), "seed": seed,
            "story_key": upload(path, job["client"], job["job_id"], "story.txt", "text/plain; charset=utf-8")}


def run_describe(job, params, workdir):
    src = os.path.join(workdir, "input" + os.path.splitext(params["upload_key"])[1])
    s3.download_file(BUCKET, params["upload_key"], src)
    meta = models.describe_video(src, params.get("instructions", ""))
    return {"metadata": meta}


def _srt_time(seconds):
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def build_narration(shots, secs, voice, workdir):
    """One WAV for the whole story: each shot's line fitted into its slot.
    Lines longer than the slot are sped up (up to 1.5x), then trimmed."""
    fitted = []
    for i, shot in enumerate(shots):
        raw = os.path.join(workdir, f"line_{i:02}.wav")
        duration = models.narrate(shot["narration"], voice, raw)
        tempo = min(max(duration / (secs - 0.2), 1.0), 1.5)
        out = os.path.join(workdir, f"line_{i:02}_fit.wav")
        ffmpeg("-i", raw, "-af", f"atempo={tempo:.3f},apad,atrim=0:{secs}", "-ar", str(models.TTS_SAMPLE_RATE), out)
        fitted.append(out)
    concat_list = os.path.join(workdir, "audio_concat.txt")
    with open(concat_list, "w") as f:
        f.writelines(f"file '{p}'\n" for p in fitted)
    narration = os.path.join(workdir, "narration.wav")
    ffmpeg("-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", narration)
    return narration


def run_story(job, params, workdir):
    client, job_id = job["client"], job["job_id"]
    cast = load_cast(params, workdir)
    by_name = {name: (name, ref, desc) for name, ref, desc in cast}
    plan = models.plan_story(params["story"], params["max_shots"], params.get("style", ""),
                             [(name, desc) for name, _, desc in cast])
    shots = plan["shots"]
    secs = min(params["seconds_per_shot"], models.MAX_CLIP_SECONDS)
    progress(job_id, f"planned {len(shots)} shots")
    seed = int.from_bytes(os.urandom(4), "big")  # shared seed helps visual consistency

    # Draft quality renders smaller (and upscales); otherwise the models' default 1280x704.
    size = {k: params[k] for k in ("width", "height") if k in params}

    # All keyframes first, then all clips: few model loads instead of one per shot.
    keyframes, krea_loras_unused = story_first_frames(job, params, plan, shots, seed, size)
    for i, shot in enumerate(shots):
        shot_cast = [by_name[n] for n in shot.get("characters", [])]
        if shot_cast and keyframes[i] is None:  # None: no Krea frame for this shot
            progress(job_id, f"placing avatars, shot {i + 1}/{len(shots)}")
            keyframes[i] = models.make_keyframe(
                shot_cast, realism.with_style(f"{plan['visual_bible']} {shot['prompt']}", params), seed=seed,
                best=params.get("quality") == "best", **size)

    # Text-only shots first, then keyframed ones: T2V and I2V are separate 14B models,
    # so interleaving them would reload one per shot.
    clips = [os.path.join(workdir, f"shot_{i:02}.mp4") for i in range(len(shots))]
    order = sorted(range(len(shots)), key=lambda i: keyframes[i] is not None)
    # Burning in subtitles encodes the whole story again, so the shots are quick near-lossless intermediates.
    x264 = models.X264_INTERMEDIATE if params.get("subtitles", True) else None
    for n, i in enumerate(order):
        progress(job_id, f"rendering shot {n + 1}/{len(shots)}")
        shot = shots[i]
        shot_cast = [by_name[name] for name in shot.get("characters", [])]
        prompt = realism.with_style(f"{describe_cast(shot_cast)} {plan['visual_bible']} {shot['prompt']}".strip(),
                                    params)
        models.generate_video(clips[i], prompt=prompt, seconds=secs, seed=seed, image=keyframes[i],
                              fps=params.get("fps", 16), upscale=params.get("upscale", False),
                              grain=bool(params.get("grain")), x264=x264, **size)

    # Then all shot sounds: one more model load for the whole story.
    sound = None
    if params.get("sound", True):
        sounds = []
        with models.sound_session():  # the sound model stays on the GPU for all the shots
            for i, (clip, shot) in enumerate(zip(clips, shots)):
                progress(job_id, f"adding sound, shot {i + 1}/{len(shots)}")
                sounds.append(models.generate_sound(clip, os.path.join(workdir, f"sound_{i:02}.wav"),
                                                    shot["prompt"], secs, seed))
        sound_list = os.path.join(workdir, "sound_concat.txt")
        with open(sound_list, "w") as f:
            f.writelines(f"file '{s}'\n" for s in sounds)
        sound = os.path.join(workdir, "sound.wav")
        ffmpeg("-f", "concat", "-safe", "0", "-i", sound_list, "-c", "copy", sound)

    progress(job_id, "assembling")
    concat_list = os.path.join(workdir, "concat.txt")
    with open(concat_list, "w") as f:
        f.writelines(f"file '{c}'\n" for c in clips)
    video = os.path.join(workdir, "joined.mp4")
    ffmpeg("-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", video)

    srt = os.path.join(workdir, "subtitles.srt")
    with open(srt, "w") as f:
        for i, shot in enumerate(shots):
            f.write(f"{i + 1}\n{_srt_time(i * secs)} --> {_srt_time((i + 1) * secs)}\n{shot['narration']}\n\n")

    if params.get("subtitles", True):
        subbed = os.path.join(workdir, "subbed.mp4")
        ffmpeg("-i", video, "-vf", f"subtitles={srt}", "-c:v", "libx264", *models.X264_ARGS,
               "-pix_fmt", "yuv420p", subbed)
        video = subbed

    result = {}
    narration = None
    if params.get("narration", True):
        progress(job_id, "recording narration")
        narration = build_narration(shots, secs, params.get("voice", "af_heart"), workdir)
        result["narration_key"] = upload(narration, client, job_id, "narration.wav", "audio/wav")
    if sound:
        result["sound_key"] = upload(sound, client, job_id, "sound.wav", "audio/wav")

    track = None
    if narration and sound:
        # Narration on top at full volume, the generated sound underneath.
        track = os.path.join(workdir, "mix.wav")
        ffmpeg("-i", narration, "-i", sound, "-filter_complex",
               "[1:a]volume=0.3[bg];[0:a][bg]amix=inputs=2:duration=longest:normalize=0",
               "-ar", "44100", track)
    elif narration or sound:
        track = narration or sound
    if track:
        video = add_audio(video, track, os.path.join(workdir, "voiced.mp4"))

    plan_path = os.path.join(workdir, "plan.json")
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    result.update({
        "title": plan.get("title", ""),
        "shots": len(shots),
        "video_key": upload(video, client, job_id, "story.mp4", "video/mp4"),
        "subtitles_key": upload(srt, client, job_id, "subtitles.srt", "application/x-subrip"),
        "plan_key": upload(plan_path, client, job_id, "plan.json", "application/json"),
    })
    if krea_loras_unused:  # every shot had avatars, so no Krea first frame used the Krea LoRAs
        result["krea_loras_unused"] = True
    return result


def story_first_frames(job, params, plan, shots, seed, size):
    """With image_model krea2: Krea 2 first frames (with the job's Krea LoRAs) for the shots without
    avatars, in one Krea run. Returns (keyframes per shot, None where there is none; whether Krea LoRAs
    were given but no shot could use them)."""
    keyframes = [None] * len(shots)
    if params.get("image_model") != "krea2":
        return keyframes, False
    who = params.get("krea_avatar")  # shots with only this avatar: a Krea frame with their LoRA
    free = [i for i, shot in enumerate(shots) if not shot.get("characters") or (who and shot["characters"] == [who])]
    if not free:
        return keyframes, bool(params.get("krea_loras"))
    w, h = size.get("width", 1280), size.get("height", 704)
    # Shots with the avatar get its LoRA (and name); shots without characters run separately without it.
    with_avatar = [i for i in free if who and shots[i].get("characters") == [who]]
    without = [i for i in free if i not in with_avatar]
    plain = dict(params, krea_loras=[l for l in params.get("krea_loras") or []
                                     if l.get("name") != params.get("krea_avatar_lora")])
    for group, p in ((without, plain), (with_avatar, params)):
        if not group:
            continue
        frames, _ = krea.generate([{"prompt": realism.with_style(f"{plan['visual_bible']} {shots[i]['prompt']}", params),
                                    "width": w, "height": h, "seed": seed + i} for i in group],
                                  lambda m: progress(job["job_id"], m), loras=prepare_krea_loras(p))
        if any(f is None for f in frames):
            raise krea.FilteredOut("a shot's first frame was flagged by the content filter; change the story")
        for i, f in zip(group, frames):
            keyframes[i] = models.fit(f, w, h)
    return keyframes, False


def run_lora_preview(job, params, workdir):
    """Baseline + one short clip per checkpoint, same prompt and seed (text-to-video, no sound)."""
    client, job_id = job["client"], job["job_id"]
    lora = {k: params[k] for k in ("name", "version", "trigger")}
    args = dict(seconds=params["seconds"], width=params["width"], height=params["height"], seed=params["seed"],
                fps=16)  # previews compare checkpoints; no need to interpolate
    # The trigger word goes into every clip, the baseline too, so only the LoRA differs.
    prompt = f"{params['trigger']} {params['prompt']}".strip()
    result = {"name": params["name"], "version": params["version"], "weight": params["weight"],
              "seed": params["seed"], "prompt": prompt, "steps": []}
    models.set_loras([])
    progress(job_id, "baseline without the LoRA")
    path = models.generate_video(os.path.join(workdir, "preview_base.mp4"), prompt, **args)
    result["preview_base_key"] = upload(path, client, job_id, "preview_base.mp4", "video/mp4")
    for n, clip in enumerate(params["clips"], 1):
        progress(job_id, f"checkpoint {clip['step']} ({n}/{len(params['clips'])})")
        models.set_loras(prepare_loras({"loras": [dict(lora, step=clip["step"], weight=params["weight"],
                                                         high_key=clip["high_key"], low_key=clip["low_key"])]}))
        name = f"preview_s{clip['step']:05d}"
        path = models.generate_video(os.path.join(workdir, name + ".mp4"), prompt, **args)
        result[name + "_key"] = upload(path, client, job_id, name + ".mp4", "video/mp4")
        result["steps"].append(clip["step"])
    return result


# ---------------------------------------------------------------- Krea 2 LoRA per avatar: the photo set
AVATAR_DATASET_MIN = 16  # photos that must pass the content filter to train
AVATAR_LORA_STEPS = 960  # Krea 2 RAW training steps (spike 2026-11-24: likeness from ~240, 960 at ~3.4 s/step)
JOB_RETENTION_DAYS = 30


def _s3_json(key):
    """The JSON document at key, or None when it doesn't exist; other errors raise."""
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except KeyError:  # test stand-in for a missing key
        return None
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise


def _current_build(client, name, job_id, created_at):
    """(avatar, krea_lora) while this job is still the avatar's current Krea LoRA build, else None (the
    avatar was replaced, or a newer build started)."""
    avatar = _s3_json(f"avatars/{client}/{name}/avatar.json")
    kl = (avatar or {}).get("krea_lora") or {}
    if not avatar or job_id not in (kl.get("job_ids") or []):
        return None
    if created_at is not None and avatar.get("created_at") != created_at:
        return None
    return avatar, kl


def _update_build(client, name, job_id, created_at, **fields):
    """Update the avatar's krea_lora from a fresh read, only while this job is its current build."""
    cur = _current_build(client, name, job_id, created_at)
    if not cur:
        return False
    avatar, kl = cur
    avatar["krea_lora"] = dict(kl, **fields)
    s3.put_object(Bucket=BUCKET, Key=f"avatars/{client}/{name}/avatar.json", Body=json.dumps(avatar).encode(),
                  ContentType="application/json")
    return True


def _current_media(client, name, job_id, created_at):
    """(avatar, media) while this job is still the one making the avatar's media, else None."""
    avatar = _s3_json(f"avatars/{client}/{name}/avatar.json")
    media = (avatar or {}).get("media") or {}
    if not avatar or media.get("job_id") != job_id:
        return None
    if created_at is not None and avatar.get("created_at") != created_at:
        return None
    return avatar, media


def _update_media(client, name, job_id, created_at, **fields):
    """Update the avatar's media record from a fresh read, only while this job owns it."""
    cur = _current_media(client, name, job_id, created_at)
    if not cur:
        return False
    avatar, media = cur
    avatar["media"] = dict(media, **fields)
    s3.put_object(Bucket=BUCKET, Key=f"avatars/{client}/{name}/avatar.json", Body=json.dumps(avatar).encode(),
                  ContentType="application/json")
    return True


def _one_person(kept, job_id):
    """Keep the photos that show exactly one person (the VLM counts); returns (kept, dropped)."""
    progress(job_id, "checking each photo shows one person")
    ok = [(p, scene) for p, scene in kept if models.count_people(p) == 1]
    return ok, len(kept) - len(ok)


def _make_photo_set(job_id, params, workdir, avatar, trigger, purpose):
    """The avatar's photo set: one Qwen photo per KREA_DATASET_PROMPTS entry at 1024x1024, content-filtered.
    Returns [(path, scene)] of the kept photos and the number made; too few kept raises FilteredOut."""
    keys = [i["key"] for i in avatar.get("images") or []] or [avatar["reference_key"]]
    refs = load_references(keys[:3], workdir, "avatar")
    base = int.from_bytes(os.urandom(4), "big")
    prompts = photos.KREA_DATASET_PROMPTS[:int(params.get("count") or len(photos.KREA_DATASET_PROMPTS))]
    paths = []
    for i, scene in enumerate(prompts, 1):
        progress(job_id, f"photo {i}/{len(prompts)} {purpose}")
        image = models.make_photo(refs, photos.photo_prompt(scene, len(refs), keep_outfit=False), photos.NEGATIVE,
                                  1024, 1024, base + i)
        path = os.path.join(workdir, f"{trigger}_{i:02d}.png")
        image.save(path)
        paths.append(path)
    progress(job_id, "content filter on the photo set")
    scores = krea.score_images(paths, lambda m: progress(job_id, m))
    kept = [(p, scene) for p, scene, sc in zip(paths, prompts, scores) if not krea.flagged(sc)]
    if len(kept) < AVATAR_DATASET_MIN:
        raise krea.FilteredOut(f"the photo set was flagged by the content filter ({len(paths) - len(kept)} of "
                               f"{len(paths)} photos)")
    kept, people_dropped = _one_person(kept, job_id)
    if len(kept) < AVATAR_DATASET_MIN:
        raise krea.FilteredOut(f"only {len(kept)} of {len(paths)} photos show exactly one person; use reference "
                               "images of just this person")
    return kept, len(paths), people_dropped


AVATAR_CLIPS = 4


class MediaFailed(Exception):
    pass


def _avatar_media(client, name, job_id, created, kept, workdir, trigger):
    """The avatar's media: the kept photos stored first, then four 2 s clips from kept photos (positions 0,
    n/4, n/2, 3n/4), content-filtered together after rendering (one filter process for all 12 stills), stored with
    media.json under avatars/<client>/<name>/media/, and the media record set ready. Raises MediaFailed
    (after marking the media failed) when no clip is left; the photos stay stored."""
    stale = {"media": "skipped: the avatar was replaced or newer media was requested"}
    if not _update_media(client, name, job_id, created, state="clips"):
        return stale
    prefix = f"avatars/{client}/{name}/media/"
    doc = {"created_at": int(time.time()), "job_id": job_id, "photos": [], "clips": []}
    for i, (path, scene) in enumerate(kept, 1):
        fname = f"photo_{i:02d}.png"
        s3.upload_file(path, BUCKET, prefix + fname, ExtraArgs={"ContentType": "image/png"})
        doc["photos"].append({"file": fname, "scene": scene, "size": os.path.getsize(path)})
    n = len(kept)
    picks = [kept[i * n // AVATAR_CLIPS][0] for i in range(AVATAR_CLIPS)]
    base = int.from_bytes(os.urandom(4), "big")
    rendered = []
    for i, (photo, motion) in enumerate(zip(picks, photos.AVATAR_CLIP_MOTIONS), 1):
        if not _current_media(client, name, job_id, created):  # replaced meanwhile: stop using the GPU
            return stale
        progress(job_id, f"clip {i}/{AVATAR_CLIPS} of the avatar")
        prompt = f"{trigger} {motion}, natural light, static camera"
        out = os.path.join(workdir, f"clip_{i}.mp4")
        stills = models.render_clip(photo, prompt, out, base + i)
        rendered.append(({"path": out, "prompt": prompt, "motion": motion, "seed": base + i,
                          "photo": os.path.basename(photo)}, stills))
    progress(job_id, "content filter on the clips")
    scores = krea.score_images([p for _, stills in rendered for p in stills])
    clips, dropped = [], 0
    for k, (clip, stills) in enumerate(rendered):
        if any(krea.flagged(sc) for sc in scores[k * 3:k * 3 + len(stills)]) or \
                models.count_people(stills[len(stills) // 2]) != 1:  # flagged, or not just this one person
            dropped += 1
        else:
            clips.append(clip)
    if not clips:
        _update_media(client, name, job_id, created, state="failed", photos=n, clips=0,
                      error=f"all {AVATAR_CLIPS} clips were flagged by the content filter")
        raise MediaFailed(f"all {AVATAR_CLIPS} clips were flagged by the content filter")
    if not _current_media(client, name, job_id, created):
        return stale
    for i, c in enumerate(clips, 1):
        fname = f"clip_{i}.mp4"
        s3.upload_file(c["path"], BUCKET, prefix + fname, ExtraArgs={"ContentType": "video/mp4"})
        doc["clips"].append({"file": fname, "size": os.path.getsize(c["path"]),
                             **{k: c[k] for k in ("prompt", "motion", "seed", "photo")}})
    s3.put_object(Bucket=BUCKET, Key=prefix + "media.json", Body=json.dumps(doc).encode(),
                  ContentType="application/json")
    _update_media(client, name, job_id, created, state="ready", photos=n, clips=len(clips),
                  preview="clip_1.mp4", updated_at=doc["created_at"], error=None)
    return {"clips": len(clips), "clips_filtered": dropped, "_clips": [(c["path"], c["motion"]) for c in clips]}


def _dataset_photos(client, lora, version, workdir, trigger):
    """The kept photos of an avatar's Krea dataset (datasets/<client>/<lora>/v<N>/), downloaded:
    [(path, scene)] in dataset order, or None when the dataset isn't there."""
    prefix = f"datasets/{client}/{lora}/v{int(version)}/"
    manifest = _s3_json(prefix + "manifest.json")
    if not manifest or not manifest.get("files"):
        return None
    items = (_s3_json(prefix + "captions.json") or {}).get("items") or {}
    kept = []
    for i, f in enumerate(manifest["files"], 1):
        path = os.path.join(workdir, f"ds_{i:02d}_{os.path.basename(f['path'])}")
        s3.download_file(BUCKET, prefix + "raw/" + f["path"], path)
        caption = items.get(f"p{i:04d}.png") or ""
        kept.append((path, caption[len(trigger) + 2:] if caption.startswith(f"{trigger}, ") else caption))
    return kept


def _media_photoset(job, params, workdir):
    """An avatar_photoset job without a Krea LoRA build: the photo set for the avatar's media only."""
    client, job_id = job["client"], job["job_id"]
    name, trigger, created = params["avatar"], params.get("trigger") or params["avatar"], params.get("avatar_created_at")
    cur = _current_media(client, name, job_id, created)
    if not cur:
        return {"skipped": "the avatar was replaced or newer media was requested"}
    try:
        _update_media(client, name, job_id, created, state="photos")
        kl = cur[0].get("krea_lora") or {}
        # Reuse the photos of a Krea LoRA that is ready for this avatar (not one kept from a replaced avatar).
        ready = (kl.get("ready") or {}) if params.get("media_only") and kl.get("state") == "ready" else {}
        kept = None
        if ready.get("lora") and ready.get("version"):
            try:
                kept = _dataset_photos(client, ready["lora"], ready["version"], workdir, trigger)
            except (ClientError, OSError, KeyError, ValueError) as e:  # unreadable dataset: make new photos
                log(event="media_dataset_unreadable", job_id=job_id, error=str(e)[:200])
        made, people_dropped = len(kept or []), 0
        if kept:  # the reused photos get the one-person check too (sets made before it existed)
            kept, people_dropped = _one_person(kept, job_id)
            if len(kept) < AVATAR_DATASET_MIN:
                kept = None
        if not kept:  # no usable Krea dataset to reuse: make the photo set
            kept, made, people_dropped = _make_photo_set(job_id, params, workdir, cur[0], trigger, "of the avatar")
        if not _current_media(client, name, job_id, created):
            return {"skipped": "the avatar was replaced or newer media was requested"}
        result = {"kept": len(kept), "filtered": made - len(kept) - people_dropped, "people_dropped": people_dropped,
                  **_avatar_media(client, name, job_id, created, kept, workdir, trigger)}
        clips = result.pop("_clips", [])
        return dict(result, **_maybe_wan(client, name, job_id, created, params, kept, clips, trigger))
    except Exception as e:
        _update_media(client, name, job_id, created, state="failed", error=str(e)[:500])
        _update_wan(client, name, job_id, created, state="failed", error=str(e)[:500])
        raise


def run_avatar_photoset(job, params, workdir):
    """Qwen photos of the avatar for its Krea 2 LoRA: one per KREA_DATASET_PROMPTS entry at 1024x1024, each
    checked by the Krea content filter (flagged ones dropped; fewer than AVATAR_DATASET_MIN -> fail), written
    as the dataset datasets/<client>/<lora>/v<N>/ (manifest, raw/, captions.json starting with the trigger =
    the avatar name), then a krea2 train job is queued for the trainer. Every avatar write checks that this
    job is still the avatar's current build. docs/specs/2026-11-24-krea-avatar-lora.md"""
    if params.get("wan_only"):  # an on-demand Wan LoRA build from the avatar's stored media
        return _wan_only(job, params, workdir)
    if params.get("krea_lora") is False:  # media only (no Krea LoRA for this avatar, or media for an existing one)
        return _media_photoset(job, params, workdir)
    client, job_id = job["client"], job["job_id"]
    name, lora, trigger, version = params["avatar"], params["lora"], params["trigger"], int(params["version"])
    created = params.get("avatar_created_at")
    if not TRAIN_QUEUE_URL:
        _update_media(client, name, job_id, created, state="failed", error="the trainer is not deployed")
        _update_wan(client, name, job_id, created, state="failed", error="the trainer is not deployed")
        raise RuntimeError("the trainer is not deployed (training_enabled in Terraform); can't train the Krea LoRA")
    cur = _current_build(client, name, job_id, created)
    if not cur:
        return {"skipped": "the avatar was replaced or a newer Krea LoRA build started"}
    avatar, kl = cur
    train_id = _train_job_id(job_id)
    existing = jobs.get_item(Key={"job_id": train_id}).get("Item")
    if existing:  # redelivered after the train job was written: make sure it's linked and sent, nothing else
        _update_build(client, name, job_id, created, state="training",
                      job_ids=[j for j in kl.get("job_ids") or [] if j != train_id] + [train_id])
        if existing.get("status") == "queued":  # a duplicate message is harmless: the trainer skips finished jobs
            sqs.send_message(QueueUrl=TRAIN_QUEUE_URL, MessageBody=json.dumps({"job_id": train_id}))
        result = {"skipped": "the train job is already queued", "train_job_id": train_id}
        media = (_current_media(client, name, job_id, created) or (None, {}))[1]
        if media.get("state") == "ready" and params.get("wan_lora") and \
                not jobs.get_item(Key={"job_id": _wan_train_job_id(job_id)}).get("Item"):
            try:  # the media is done but the Wan build never started: build it from the stored media
                stored = _media_files(client, name, workdir)
                kept_m = _one_person(stored[0], job_id)[0] if stored else []
                if len(kept_m) < AVATAR_DATASET_MIN:
                    raise RuntimeError(f"only {len(kept_m)} stored photos show exactly one person")
                result.update(_maybe_wan(client, name, job_id, created, params, kept_m, stored[1], trigger))
            except models.JobCancelled:
                _update_wan(client, name, job_id, created, state="failed", error="cancelled")
                raise
            except Exception as e:  # noqa: BLE001 - the Krea build is queued; only the Wan build failed
                _update_wan(client, name, job_id, created, state="failed", error=str(e)[:500])
                result["wan_error"] = str(e)[:500]
        if media and media.get("state") != "ready":  # the media wasn't finished: make it from this build's photos
            try:
                kept = _dataset_photos(client, lora, version, workdir, trigger)
                if kept:
                    kept, _ = _one_person(kept, job_id)
                if kept and len(kept) >= AVATAR_DATASET_MIN:
                    result.update(_avatar_media(client, name, job_id, created, kept, workdir, trigger))
                    clips = result.pop("_clips", [])
                    result.update(_maybe_wan(client, name, job_id, created, params, kept, clips, trigger))
            except models.JobCancelled:
                raise
            except Exception as e:  # noqa: BLE001 - as in _photoset: media never fails the Krea build
                _update_media(client, name, job_id, created, state="failed", error=str(e)[:500])
                result["media_error"] = str(e)[:500]
        return result
    try:
        return _photoset(job, params, workdir, avatar, client, job_id, name, lora, trigger, version, created)
    except Exception as e:  # the photo set failed: the Krea build and the media and Wan builds it was for
        _update_build(client, name, job_id, created, state="failed", error=str(e)[:500])
        _update_media(client, name, job_id, created, state="failed", error=str(e)[:500])
        _update_wan(client, name, job_id, created, state="failed", error=str(e)[:500])
        raise


AVATAR_WAN_STEPS = 1000  # per expert; spike 2026-11-24: likeness held through 50 s, ~54 min for both experts


def _current_wan(client, name, job_id, created_at):
    """(avatar, wan_lora) while this job is still the avatar's current Wan LoRA build, else None."""
    avatar = _s3_json(f"avatars/{client}/{name}/avatar.json")
    wl = (avatar or {}).get("wan_lora") or {}
    if not avatar or job_id not in (wl.get("job_ids") or []):
        return None
    if created_at is not None and avatar.get("created_at") != created_at:
        return None
    return avatar, wl


def _update_wan(client, name, job_id, created_at, **fields):
    cur = _current_wan(client, name, job_id, created_at)
    if not cur:
        return False
    avatar, wl = cur
    avatar["wan_lora"] = dict(wl, **fields)
    s3.put_object(Bucket=BUCKET, Key=f"avatars/{client}/{name}/avatar.json", Body=json.dumps(avatar).encode(),
                  ContentType="application/json")
    return True


def _wan_train_job_id(photoset_job_id):
    return uuid.uuid5(uuid.NAMESPACE_URL, f"vidgen-wan-train:{photoset_job_id}").hex


def _wan_step(client, name, job_id, created, params, kept, clips, trigger):
    """The avatar's Wan LoRA build: kept photos + clips as datasets/<client>/<wan lora>/v<N>/ (captions keyed as
    the trainer names items: photo i -> p{i:04d}.png, clip at manifest index i -> s{i:04d}_00.mp4), then one
    t2v train job (id derived from this job), linked on the avatar before it is sent."""
    lora, version = params["wan_lora_name"], int(params["wan_version"])
    cur = _current_wan(client, name, job_id, created)
    if not cur:
        return {"wan": "skipped: the avatar was replaced or a newer Wan LoRA build started"}
    train_id = _wan_train_job_id(job_id)
    existing = jobs.get_item(Key={"job_id": train_id}).get("Item")
    if existing:  # redelivered after the train job was written: link and (if still queued) re-send only
        _update_wan(client, name, job_id, created, state="training",
                    job_ids=[j for j in cur[1].get("job_ids") or [] if j != train_id] + [train_id])
        if existing.get("status") == "queued":
            sqs.send_message(QueueUrl=TRAIN_QUEUE_URL, MessageBody=json.dumps({"job_id": train_id}))
        return {"wan_train_job_id": train_id}
    if not TRAIN_QUEUE_URL:
        raise RuntimeError("the trainer is not deployed; can't train the Wan LoRA")
    progress(job_id, "writing the Wan LoRA dataset")
    dataset = f"{lora}/v{version}"
    prefix = f"datasets/{client}/{dataset}/"
    files, items = [], {}
    for i, (path, scene) in enumerate(kept, 1):
        fname = f"p_{i:02d}.png"
        s3.upload_file(path, BUCKET, prefix + "raw/" + fname, ExtraArgs={"ContentType": "image/png"})
        files.append({"path": fname, "size": os.path.getsize(path)})
        items[f"p{i:04d}.png"] = f"{trigger}, {scene}"
    for k, (path, motion) in enumerate(clips, 1):
        fname = f"c_{k}.mp4"
        s3.upload_file(path, BUCKET, prefix + "raw/" + fname, ExtraArgs={"ContentType": "video/mp4"})
        files.append({"path": fname, "size": os.path.getsize(path)})
        items[f"s{len(kept) + k:04d}_00.mp4"] = f"{trigger}, {motion}, natural light"
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    now = int(time.time())
    manifest = {"kind": "character", "trigger": trigger, "files": files, "file_count": len(files),
                "total_bytes": sum(f["size"] for f in files), "updated_at": now, "source": "avatar", "avatar": name}
    captions = {"trigger": trigger, "fingerprint": fingerprint, "edited_at": now, "source": "avatar", "items": items}
    for key, doc in (("manifest.json", manifest), ("captions.json", captions)):
        s3.put_object(Bucket=BUCKET, Key=prefix + key, Body=json.dumps(doc).encode(), ContentType="application/json")
    if not _current_wan(client, name, job_id, created):
        return {"wan": "skipped: the avatar was replaced or a newer Wan LoRA build started"}
    tparams = {"dataset": dataset, "model": "t2v", "lora": lora, "version": version, "trigger": trigger,
               "avatar": name, "avatar_created_at": created, "steps": AVATAR_WAN_STEPS, "kind": "character",
               "speed": "standard", "captions_only": False, "files": len(files), "dataset_updated_at": now,
               "fingerprint": fingerprint}
    try:
        jobs.put_item(Item={"job_id": train_id, "client": client, "type": "train", "params": json.dumps(tparams),
                            "status": "queued", "created_at": now, "expires_at": now + JOB_RETENTION_DAYS * 86400},
                      ConditionExpression="attribute_not_exists(job_id)")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
            raise
    cur = _current_wan(client, name, job_id, created)
    if cur:
        _update_wan(client, name, job_id, created, state="training",
                    job_ids=[j for j in cur[1].get("job_ids") or [] if j != train_id] + [train_id])
    sqs.send_message(QueueUrl=TRAIN_QUEUE_URL, MessageBody=json.dumps({"job_id": train_id}))
    jobs.put_item(Item={"job_id": "_control_trainer", "client": "_system", "hold": False})
    return {"wan_train_job_id": train_id, "wan_dataset": prefix}


def _media_files(client, name, workdir):
    """The avatar's stored media, downloaded: ([(photo path, scene)], [(clip path, motion)]), or None."""
    prefix = f"avatars/{client}/{name}/media/"
    doc = _s3_json(prefix + "media.json")
    if not doc or not doc.get("photos"):
        return None
    out = []
    for kind, label in (("photos", "scene"), ("clips", "motion")):
        items = []
        for e in doc.get(kind) or []:
            fname = os.path.basename(str(e.get("file") or ""))
            if not fname:
                continue
            path = os.path.join(workdir, f"m_{fname}")
            s3.download_file(BUCKET, prefix + fname, path)
            items.append((path, e.get(label) or ""))
        out.append(items)
    return out[0], out[1]


def _wan_only(job, params, workdir):
    """An on-demand Wan LoRA build (POST /avatars/{name}/wan-lora): the avatar's stored photos (one-person
    checked) and clips become a new Wan dataset version; no new photos or clips."""
    client, job_id = job["client"], job["job_id"]
    name, trigger, created = params["avatar"], params.get("trigger") or params["avatar"], params.get("avatar_created_at")
    if not _current_wan(client, name, job_id, created):
        return {"skipped": "the avatar was replaced or a newer Wan LoRA build started"}
    try:
        _update_wan(client, name, job_id, created, state="photos")
        media = _media_files(client, name, workdir)
        if not media:
            raise RuntimeError("the avatar has no stored media to train on; make media first")
        kept, clips = media
        kept, people_dropped = _one_person(kept, job_id)
        if len(kept) < AVATAR_DATASET_MIN:
            raise krea.FilteredOut(f"only {len(kept)} of the avatar's media photos show exactly one person")
        return dict(_wan_step(client, name, job_id, created, params, kept, clips, trigger),
                    people_dropped=people_dropped)
    except Exception as e:
        _update_wan(client, name, job_id, created, state="failed", error=str(e)[:500])
        raise


def _maybe_wan(client, name, job_id, created, params, kept, clips, trigger):
    """The Wan step when this build asks for it; a failure marks the Wan LoRA failed, never the job."""
    if not params.get("wan_lora") or params.get("media_only"):
        return {}
    try:
        return _wan_step(client, name, job_id, created, params, kept, clips, trigger)
    except models.JobCancelled:
        _update_wan(client, name, job_id, created, state="failed", error="cancelled")
        raise
    except Exception as e:  # noqa: BLE001 - the Krea build and the media are already done
        _update_wan(client, name, job_id, created, state="failed", error=str(e)[:500])
        return {"wan_error": str(e)[:500]}


def _train_job_id(photoset_job_id):
    """One train job per photoset job: derived from its id, so a redelivered job finds it."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"vidgen-krea-train:{photoset_job_id}").hex


def _photoset(job, params, workdir, avatar, client, job_id, name, lora, trigger, version, created):
    _update_build(client, name, job_id, created, state="photos")
    _update_media(client, name, job_id, created, state="photos")
    try:
        kept, made, people_dropped = _make_photo_set(job_id, params, workdir, avatar, trigger, "for the Krea LoRA")
    except krea.FilteredOut as e:
        raise krea.FilteredOut(f"{e}; no Krea LoRA for this avatar") from e
    if not _current_build(client, name, job_id, created):
        return {"skipped": "the avatar was replaced or a newer Krea LoRA build started"}
    dataset = f"{lora}/v{version}"
    prefix = f"datasets/{client}/{dataset}/"
    files = []
    for p, _ in kept:
        s3.upload_file(p, BUCKET, prefix + "raw/" + os.path.basename(p), ExtraArgs={"ContentType": "image/png"})
        files.append({"path": os.path.basename(p), "size": os.path.getsize(p)})
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    now = int(time.time())
    manifest = {"kind": "character", "trigger": trigger, "files": files, "file_count": len(files),
                "total_bytes": sum(f["size"] for f in files), "updated_at": now, "source": "avatar", "avatar": name}
    captions = {"trigger": trigger, "fingerprint": fingerprint, "edited_at": now, "source": "avatar",
                "items": {f"p{i:04d}.png": f"{trigger}, {scene}" for i, (_, scene) in enumerate(kept, 1)}}
    for key, doc in (("manifest.json", manifest), ("captions.json", captions)):
        s3.put_object(Bucket=BUCKET, Key=prefix + key, Body=json.dumps(doc).encode(), ContentType="application/json")
    if not _current_build(client, name, job_id, created):
        return {"skipped": "the avatar was replaced or a newer Krea LoRA build started"}
    # One train job per photoset job: the id is derived from this job's id and the put is conditional. The
    # avatar records the train job before its message is sent, so the trainer's updates always match.
    train_id = _train_job_id(job_id)
    tparams = {"dataset": dataset, "model": "krea2", "lora": lora, "version": version, "trigger": trigger,
               "avatar": name, "avatar_created_at": created, "steps": AVATAR_LORA_STEPS, "kind": "character",
               "speed": "standard", "captions_only": False, "files": len(files), "dataset_updated_at": now,
               "fingerprint": fingerprint}
    try:
        jobs.put_item(Item={"job_id": train_id, "client": client, "type": "train", "params": json.dumps(tparams),
                            "status": "queued", "created_at": now, "expires_at": now + JOB_RETENTION_DAYS * 86400},
                      ConditionExpression="attribute_not_exists(job_id)")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
            raise
    cur = _current_build(client, name, job_id, created)
    if cur:
        _update_build(client, name, job_id, created, state="training",
                      job_ids=[j for j in cur[1].get("job_ids") or [] if j != train_id] + [train_id])
    sqs.send_message(QueueUrl=TRAIN_QUEUE_URL, MessageBody=json.dumps({"job_id": train_id}))
    jobs.put_item(Item={"job_id": "_control_trainer", "client": "_system", "hold": False})  # as the API does
    result = {"kept": len(kept), "filtered": made - len(kept) - people_dropped, "people_dropped": people_dropped,
              "train_job_id": train_id, "dataset": prefix}
    clips = []
    if _current_media(client, name, job_id, created):  # the same job makes the avatar's media
        try:
            result.update(_avatar_media(client, name, job_id, created, kept, workdir, trigger))
            clips = result.pop("_clips", [])
        except models.JobCancelled:
            _update_media(client, name, job_id, created, state="failed", error="cancelled")
            raise
        except Exception as e:  # noqa: BLE001 - media must not fail the Krea build; its training is queued
            _update_media(client, name, job_id, created, state="failed", error=str(e)[:500])
            result["media_error"] = str(e)[:500]
    result.update(_maybe_wan(client, name, job_id, created, params, kept, clips, trigger))
    return result


RUNNERS = {"avatar_photoset": run_avatar_photoset, "lora_preview": run_lora_preview, "video": run_video, "describe": run_describe, "story": run_story, "write": run_write, "avatar": run_avatar,
           "photos": run_photos}


def process(job_id, receipt):
    job = jobs.get_item(Key={"job_id": job_id}).get("Item")
    if not job or job["status"] in FINISHED:
        log(event="skip", job_id=job_id, reason="missing or finished")
        return
    if job.get("cancel_requested"):
        set_status(job_id, "cancelled", finished_at=int(time.time()))
        log(event="cancelled", job_id=job_id, before_start=True)
        if job.get("type") == "avatar_photoset":  # release the avatar's build/media records this job owned
            p = json.loads(job.get("params") or "{}")
            if p.get("avatar"):
                _update_build(job["client"], p["avatar"], job_id, p.get("avatar_created_at"),
                              state="failed", error="cancelled")
                _update_media(job["client"], p["avatar"], job_id, p.get("avatar_created_at"),
                              state="failed", error="cancelled")
                _update_wan(job["client"], p["avatar"], job_id, p.get("avatar_created_at"),
                            state="failed", error="cancelled")
        return

    params = json.loads(job["params"])
    log(event="start", job_id=job_id, type=job["type"])
    set_status(job_id, "running", started_at=int(time.time()))
    workdir = tempfile.mkdtemp(prefix=f"job-{job_id}-")
    models.CANCEL.clear()
    watch = JobWatch(job_id, receipt)
    watch.start()
    try:
        models.set_loras(prepare_loras(params) if job["type"] in ("video", "story") else [])
        result = RUNNERS[job["type"]](job, params, workdir)
        if params.get("realism_check") and job["type"] in realism_check.TYPES and not params.get("plan_only"):
            result["realism"] = realism_check.check(
                job["type"], params, result, workdir, download=lambda key, path: s3.download_file(BUCKET, key, path),
                judge=models.judge_realism, progress=lambda msg: progress(job_id, msg))
            log(event="realism_check", job_id=job_id, score=result["realism"].get("score"),
                error=result["realism"].get("error"))
        if params.get("loras"):
            result["loras"] = [{k: l.get(k) for k in ("name", "version", "step", "weight")} for l in params["loras"]]
        set_status(job_id, "done", result=json.dumps(result), finished_at=int(time.time()))
        log(event="done", job_id=job_id, seconds=int(time.time() - watch.started))
    except models.JobCancelled as e:
        status = "cancelled" if str(e) == "cancelled" else "failed"
        set_status(job_id, status, error=str(e), finished_at=int(time.time()))
        log(event=status, job_id=job_id, reason=str(e))
    except Exception as e:
        # Model and input errors are deterministic, so fail instead of retrying.
        traceback.print_exc()
        set_status(job_id, "failed", error=str(e)[:1000], finished_at=int(time.time()))
    finally:
        models.set_loras([])
        watch.stopped.set()
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree(take_work_dir(job_id), ignore_errors=True)  # saved segments: the job has ended


# ---------------------------------------------------------------- main loop

def mirror_backfill(hf_home=None):
    """Queue for the model mirror every weights folder that is complete on this disk (mirror.py uploads only
    what the bucket lacks, in the background, and skips mirrored folders after one small request). This is
    how a running box seeds the mirror, so the next new disk restores from the bucket instead of Hugging
    Face. Never stops the worker."""
    try:
        hub = os.path.join(hf_home or os.environ.get("HF_HOME", "/opt/models"), "hub")
        units = []
        for name in sorted(os.listdir(models.BF16_DIR)) if os.path.isdir(models.BF16_DIR) else []:
            if os.path.exists(os.path.join(models.BF16_DIR, name, "model_index.json")):  # a finished conversion
                units.append((f"bf16/{name}", os.path.join(models.BF16_DIR, name)))
        # The fp32 Wan downloads are not mirrored: their bf16 copies are.
        fp32 = {"models--" + r.replace("/", "--") for r in (models.VIDEO_MODEL, models.I2V_MODEL)}
        for name in sorted(os.listdir(hub)) if os.path.isdir(hub) else []:
            folder = os.path.join(hub, name)
            if not name.startswith("models--") or name in fp32:
                continue
            if any(f.endswith(".incomplete") for _, _, fs in os.walk(folder) for f in fs):
                continue  # a download in progress or cut off
            units.append((f"hub/{name}", folder))
        if os.path.isdir(models.MMAUDIO_DIR) and mirror.local_files(models.MMAUDIO_DIR)[0]:
            units.append(("mmaudio", models.MMAUDIO_DIR))
        for unit, folder in ((krea.MODEL_UNIT, krea.MODEL_DIR), (krea.FILTER_UNIT, krea.FILTER_DIR),
                             (helios.MODEL_UNIT, helios.MODEL_DIR)):
            if os.path.exists(os.path.join(folder, ".complete")):
                units.append((unit, folder))
        for unit, folder in units:
            mirror.push_async(unit, folder)
        log(event="mirror_backfill", units=[u for u, _ in units])
    except Exception as e:
        log(event="mirror_backfill", error=f"{type(e).__name__}: {e}")


def mirror_busy():
    """Uploads to the model mirror still queued or running: the box doesn't stop itself in the middle (an
    interrupted file starts again from zero). The API's max-uptime guard still applies."""
    return mirror.busy()


def main():
    log(event="worker_up", idle_seconds=IDLE_SECONDS, job_timeout_seconds=JOB_TIMEOUT_SECONDS)
    sweep_work_dirs()
    mirror_backfill()
    report_ready(100, "Ready: the worker is up and takes jobs (the first job of each kind loads its model).")
    last_work = time.time()
    while True:
        if restart_requested():  # between jobs only: a render in progress is never cut off
            log(event="restart_requested")
            report_ready(60, "Restarting the worker (pulling the latest code); back in about a minute.")
            raise SystemExit(1)  # non-zero, so systemd starts it again
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=VISIBILITY_SECONDS,
        )
        messages = resp.get("Messages", [])
        if not messages:
            if time.time() - last_work > IDLE_SECONDS and _queue_empty() and not mirror_busy():
                stop_self()
                time.sleep(300)  # the instance is going down; don't poll meanwhile
            else:
                preload_idle()
            continue

        message = messages[0]
        try:
            process(json.loads(message["Body"])["job_id"], message["ReceiptHandle"])
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=message["ReceiptHandle"])
        finally:
            last_work = time.time()


if __name__ == "__main__":
    main()
