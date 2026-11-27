#!/usr/bin/env python3
"""Command-line client for the AI video pipeline. Standard library only;
runs on macOS, Linux and Windows.

Every request is signed with your SSH ed25519 key via `ssh-keygen -Y sign`;
only keys listed in Terraform's authorized_ssh_public_keys are accepted.

Submitting a job returns immediately; you do not need to stay connected while
it renders. Results wait in S3 (outputs/, kept 30 days). A background watcher
checks every 45 seconds and downloads finished jobs to the download folder.

Configure once (saved to ~/.vidgen/config.json; VIDGEN_* env vars override it):
    vidgen.py configure --url $(terraform -chdir=terraform output -raw api_url) \\
        --ssh-key ~/.ssh/vidgen_ed25519.pub --download-dir ~/Movies/vidgen
    vidgen.py watcher install      # background check every 45 s

Avatars: video and story jobs star your default avatar so every clip shows the
same character. The first video/story run without one asks you to set it up;
`vidgen.py avatar refresh` replaces it (the old version is kept in history).

Examples:
    vidgen.py video "walking through a night market, handheld camera" -o market.mp4
    vidgen.py video "two friends sharing an umbrella" --avatar star,robot
    vidgen.py video "dancing in a neon club, full body" --avatar dan --lora jerk:0.9
    vidgen.py story story.txt --shots 6 --style "watercolor animation" --voice bf_emma
    vidgen.py photos --image me.jpg --prompt "reading at a Paris café, golden hour" --count 4 -o trip
    vidgen.py photos --avatar dan --where "on a rainy rooftop" --wearing "a navy suit" --doing "walking" --camera "low angle"
    vidgen.py write "a lighthouse keeper finds a message dated tomorrow" --words 600 -o lighthouse.txt
    vidgen.py story ~/Movies/vidgen/lighthouse.txt   # then film it
    vidgen.py describe market.mp4
    vidgen.py jobs                 # what is pending / downloaded
    vidgen.py ui                   # dashboard: status, models, history, learning themes
    vidgen.py dataset add singing ~/Movies/singing --kind motion   # upload a learning folder
    vidgen.py train singing --steps 1250   # train a LoRA on the trainer machine (~9 h)
    vidgen.py cancel 3f9a          # job id or a unique prefix of it
    vidgen.py smoke                # end-to-end test of every model on the GPU
"""

import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
    import msvcrt

CHECK_INTERVAL = 45
SIG_NAMESPACE = "vidgen-api"
STATE_DIR = os.path.expanduser("~/.vidgen")
CONFIG_FILE = os.path.join(STATE_DIR, "config.json")
JOBS_FILE = os.path.join(STATE_DIR, "jobs.json")
LOCK_FILE = os.path.join(STATE_DIR, "check.lock")
LOG_FILE = os.path.join(STATE_DIR, "watcher.log")
PID_FILE = os.path.join(STATE_DIR, "watcher.pid")
DISMISSED_FILE = os.path.join(STATE_DIR, "dismissed.json")  # {job_id: when}: jobs `jobs forget` dropped
PENDING = ("queued", "running")
FORGET_AFTER_DAYS = 30  # outputs expire in S3 after this anyway
DEFAULT_AVATAR_NAME = "star"
IMAGE_EXTS = {"png", "jpg", "jpeg", "webp"}
SCRIPT = os.path.abspath(__file__)

LAUNCHD_LABEL = "com.vidgen.watcher"
LAUNCHD_PLIST = os.path.expanduser(f"~/Library/LaunchAgents/{LAUNCHD_LABEL}.plist")
SYSTEMD_DIR = os.path.expanduser("~/.config/systemd/user")
WINDOWS_STARTUP = os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs\Startup")


class ApiError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def notify(title, msg):
    """Desktop notification where the platform has a simple way; else just the log."""
    if shutil.which("osascript"):
        cmd = ["osascript", "-e", f"display notification {json.dumps(msg)} with title {json.dumps(title)}"]
    elif shutil.which("notify-send"):
        cmd = ["notify-send", title, msg]
    else:
        return
    subprocess.run(cmd, capture_output=True)


# ---------------------------------------------------------------- config

def load_config():
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def setting(name, default=None):
    """VIDGEN_<NAME> env var, else ~/.vidgen/config.json, else default."""
    return os.environ.get(f"VIDGEN_{name.upper()}") or load_config().get(name) or default


def save_config(**values):
    os.makedirs(STATE_DIR, exist_ok=True)
    config = load_config()
    config.update({k: v for k, v in values.items() if v})
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    return config


def download_dir():
    d = os.path.expanduser(setting("download_dir", "~/Movies/vidgen"))
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- signed API

def sign(method, path, body):
    """Signature headers for one request (format documented in lambda/api/handler.py)."""
    key = os.path.expanduser(setting("ssh_key", "~/.ssh/vidgen_ed25519.pub"))
    env = dict(os.environ)
    # Background watchers on Linux may start without the login session's agent.
    if "SSH_AUTH_SOCK" not in env and setting("ssh_auth_sock"):
        env["SSH_AUTH_SOCK"] = setting("ssh_auth_sock")
    ts = str(int(time.time()))
    nonce = secrets.token_hex(16)
    message = "\n".join(["vidgen-v1", method, path, ts, nonce, hashlib.sha256(body).hexdigest()])
    r = subprocess.run(
        ["ssh-keygen", "-Y", "sign", "-q", "-f", key, "-n", SIG_NAMESPACE],
        input=message.encode(), capture_output=True, env=env,
    )
    if r.returncode != 0:
        raise ApiError(f"signing failed ({key}): {r.stderr.decode().strip()}. "
                       "If the key has a passphrase, load it first: ssh-add ~/.ssh/vidgen_ed25519")
    return {
        "x-vidgen-timestamp": ts,
        "x-vidgen-nonce": nonce,
        "x-vidgen-signature": base64.b64encode(r.stdout).decode(),
    }


def api(method, path, body=None):
    base = setting("url")
    if not base:
        raise ApiError("API URL not configured; run: vidgen.py configure --url <api_url>")
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(base.rstrip("/") + path, data=data or None, method=method, headers={
        "content-type": "application/json",
        **sign(method, path, data),
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise ApiError(f"HTTP {e.code}: {e.read().decode()}", e.code)
    except urllib.error.URLError as e:
        raise ApiError(f"network error: {e.reason}")


# ---------------------------------------------------------------- transfers
# Big files move as parallel pieces: downloads as byte ranges of the presigned GET, uploads as S3 multipart
# parts (the API presigns them). One connection rarely fills a long-distance link; several do.
DOWNLOAD_PARALLEL_MIN = 32 * 1024 ** 2  # below this, one stream
DOWNLOAD_CHUNK = 16 * 1024 ** 2
TRANSFER_WORKERS = 8
PART_RETRIES = 3


def _urlopen(req, timeout=None):
    return urllib.request.urlopen(req, timeout=timeout)


def download(url, path):
    return download_file(url, path)


def download_file(url, path):
    """Download via a .part file so a half-finished file never looks complete; big files in parallel ranges."""
    from concurrent.futures import ThreadPoolExecutor

    tmp = path + ".part"
    size = None
    try:  # probe: a 1-byte range gives the size, and whether ranges work at all
        with _urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60) as r:
            cr = r.headers.get("Content-Range") or ""
            if getattr(r, "status", 200) == 206 and "/" in cr and cr.rsplit("/", 1)[1].isdigit():
                size = int(cr.rsplit("/", 1)[1])
    except urllib.error.HTTPError:
        size = None
    if size is None or size < DOWNLOAD_PARALLEL_MIN:
        with _urlopen(urllib.request.Request(url), timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(tmp, path)
        return
    with open(tmp, "wb") as f:
        f.truncate(size)
    pieces = [(a, min(a + DOWNLOAD_CHUNK, size) - 1) for a in range(0, size, DOWNLOAD_CHUNK)]

    def fetch(piece):
        a, b = piece
        for attempt in range(1, PART_RETRIES + 1):
            try:
                with _urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"}), timeout=120) as r:
                    data = r.read()
                if len(data) != b - a + 1:
                    raise OSError(f"short read for bytes {a}-{b}")
                fd = os.open(tmp, os.O_WRONLY)
                try:
                    os.pwrite(fd, data, a)
                finally:
                    os.close(fd)
                return
            except (urllib.error.URLError, OSError):
                if attempt == PART_RETRIES:
                    raise

    try:
        with ThreadPoolExecutor(TRANSFER_WORKERS) as pool:
            list(pool.map(fetch, pieces))
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)


def upload_multipart(spec, path, on_bytes=None):
    """Upload `path` as the multipart upload the API started (spec: key, upload_id, part_size, part_urls):
    parts in parallel (each retried), then POST /uploads/complete; on failure the upload is aborted."""
    from concurrent.futures import ThreadPoolExecutor

    size, part = os.path.getsize(path), int(spec["part_size"])
    urls = spec["part_urls"]
    if -(-size // part) != len(urls):
        raise ApiError(f"{path}: {len(urls)} parts offered for {size} bytes; start the upload again")

    def send(n):
        a = (n - 1) * part
        with open(path, "rb") as f:
            f.seek(a)
            data = f.read(min(part, size - a))
        for attempt in range(1, PART_RETRIES + 1):
            try:
                req = urllib.request.Request(urls[n - 1], data=data, method="PUT",
                                             headers={"Content-Length": str(len(data))})
                with _urlopen(req, timeout=600) as r:
                    etag = r.headers.get("ETag")
                if not etag:
                    raise OSError("no ETag in the response")
                if on_bytes:
                    on_bytes(len(data))
                return {"part": n, "etag": etag}
            except (urllib.error.URLError, OSError) as e:
                if attempt == PART_RETRIES:
                    raise ApiError(f"upload of {os.path.basename(path)} part {n} failed: {e}")

    try:
        with ThreadPoolExecutor(TRANSFER_WORKERS) as pool:
            parts = list(pool.map(send, range(1, len(urls) + 1)))
    except BaseException:
        try:
            api("POST", "/uploads/abort", {"key": spec["key"], "upload_id": spec["upload_id"]})
        except ApiError:
            pass  # unfinished parts are also removed by the bucket after a day
        raise
    api("POST", "/uploads/complete", {"key": spec["key"], "upload_id": spec["upload_id"], "parts": parts})


def upload_file(path, allowed_exts=None):
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    if allowed_exts and ext not in allowed_exts:
        raise ApiError(f"{path}: must be one of {', '.join(sorted(allowed_exts))}")
    if not os.path.isfile(path):
        raise ApiError(f"{path}: file not found")
    up = api("POST", "/uploads", {"extension": ext})
    with open(path, "rb") as f:
        req = urllib.request.Request(up["put_url"], data=f.read(), method="PUT")
        urllib.request.urlopen(req, timeout=300).close()
    return up["upload_key"]


# ---------------------------------------------------------------- local job list

class locked:
    """Exclusive lock so the watcher and a manual check never run at once."""

    def __enter__(self):
        os.makedirs(STATE_DIR, exist_ok=True)
        self.f = open(LOCK_FILE, "a+")
        if fcntl:
            fcntl.flock(self.f, fcntl.LOCK_EX)
        else:
            while True:
                try:
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:  # LK_LOCK gives up after ~10 s; keep waiting
                    pass
        return self

    def __exit__(self, *exc):
        if fcntl:
            fcntl.flock(self.f, fcntl.LOCK_UN)
        else:
            self.f.seek(0)
            msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        self.f.close()


def load_jobs():
    try:
        with open(JOBS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_jobs(jobs):
    os.makedirs(STATE_DIR, exist_ok=True)
    cutoff = time.time() - FORGET_AFTER_DAYS * 86400
    jobs = {k: v for k, v in jobs.items() if v["status"] in PENDING or v["submitted_at"] > cutoff}
    tmp = JOBS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(jobs, f, indent=2)
    os.replace(tmp, JOBS_FILE)


def resolve_job_id(prefix):
    """A full job id, or a unique prefix of one in the local job list."""
    if len(prefix) == 32:
        return prefix
    matches = [j for j in load_jobs() if j.startswith(prefix)]
    if len(matches) != 1:
        raise ApiError(f"'{prefix}' matches {len(matches)} local jobs; give more of the id")
    return matches[0]


def local_out(job_type, job_id, out_name=None):
    """Where the watcher saves a job's results: a file, or a folder for photos."""
    if job_type == "photos":
        return os.path.abspath(os.path.join(download_dir(), out_name or f"photos-{job_id}"))
    if job_type == "lora_preview":
        return os.path.abspath(os.path.join(download_dir(), out_name or f"lora-preview-{job_id}"))
    out = os.path.abspath(os.path.join(download_dir(), out_name or f"{job_id}.mp4"))
    if job_type in ("video", "story") and not os.path.splitext(out)[1]:  # "m1" -> m1.mp4, so it opens as a video
        out += ".mp4"
    if job_type == "describe":
        return os.path.splitext(out)[0] + ".json"
    if job_type == "write":
        return os.path.splitext(out)[0] + ".txt"
    return out


def register_job(job_id, job_type, out, extra=None):
    """Add a submitted job to the local list so the watcher downloads it."""
    with locked():
        jobs = load_jobs()
        jobs[job_id] = {"type": job_type, "out": out, "status": "queued", "submitted_at": time.time(),
                        **(extra or {})}
        save_jobs(jobs)


# Same rule as the API's meta.out_name (lambda/api/handler.py OUT_NAME): only plain names are shared with other
# devices; a CLI name with folders stays local.
OUT_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._ -]{0,126}[A-Za-z0-9_-])?")


def job_meta(source, out_name=None, extra=None):
    """The job's meta for the API (docs/specs/2026-11-27-phone-ui.md): where it came from, the file name to save
    as when it is a plain name, and for avatar candidates the avatar name and krea_lora."""
    meta = {"source": source}
    if out_name and OUT_NAME.fullmatch(out_name):
        meta["out_name"] = out_name
    for k in ("avatar_name", "krea_lora"):
        if (extra or {}).get(k) is not None:
            meta[k] = extra[k]
    return meta


def server_meta(job_id):
    """The server's meta for a job (e.g. started on the phone); {} if it can't be read."""
    try:
        meta = api("GET", f"/jobs/{job_id}").get("meta")
    except (ApiError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def submit(body, out_name, extra=None, quiet=False):
    r = api("POST", "/jobs", dict(body, meta=job_meta("cli", out_name, extra)))
    job_id = r["job_id"]
    out = local_out(body["type"], job_id, out_name)
    register_job(job_id, body["type"], out, extra)

    if not quiet:
        starring = f", starring {', '.join(r['avatars'])}" if r.get("avatars") else ""
        print(f"job {job_id} queued (GPU instance: {r['instance']}){starring}")
        for lora in r.get("loras") or []:
            trigger = f", trigger word '{lora['trigger']}' added to prompts" if lora.get("trigger") else ""
            step = f" step {lora['step']}" if lora.get("step") else ""
            kind = " (Krea 2, first frame/photos)" if lora.get("model") == "krea2" else ""
            print(f"LoRA {lora['name']} v{lora['version']}{step} at {lora['weight']:g}{kind}{trigger}")
        if body["type"] == "train":
            print("training runs on the trainer machine; follow it with `vidgen.py jobs` or in `vidgen.py ui`")
        else:
            print(f"it will be downloaded to {out}{os.sep if body['type'] == 'photos' else ''} when finished")
        if not watcher_installed():
            print("note: background watcher not installed; run `vidgen.py watcher install` or `vidgen.py watch`")
    return job_id


# ---------------------------------------------------------------- checking

def video_target(out, ext):
    """Where a video really goes: with its extension when the name had none (jobs saved before names got
    one), and never onto a folder of that name (then name-2.mp4, name-3.mp4, ...)."""
    if not os.path.splitext(out)[1]:
        out += ext or ".mp4"
    base, ext = os.path.splitext(out)
    n = 2
    while os.path.isdir(out):
        out, n = f"{base}-{n}{ext}", n + 1
    return out


def permanent_download_error(e):
    """Errors retrying can't fix (a folder in the way, no permission): stop instead of re-downloading forever."""
    return isinstance(e, (IsADirectoryError, NotADirectoryError, PermissionError, FileExistsError))


def fetch_results(local, remote):
    out = local["out"]
    if local["type"] == "describe":
        with open(out, "w") as f:
            json.dump(remote.get("result", {}).get("metadata", {}), f, indent=2)
        return [out]

    if local["type"] == "write":  # story.txt to the requested path
        url = remote.get("downloads", {}).get("story")
        if url:
            download(url, out)
        return [out] if url else []
    if local["type"] == "photos":
        return fetch_photos(out, remote)
    if local["type"] == "lora_preview":
        return fetch_previews(out, remote)
    if local["type"] == "train":  # the LoRA stays in S3 for the render box; nothing to download
        return []

    # "video" goes to the requested path; everything else (subtitles, plan,
    # narration, keyframe, avatar candidates) lands next to it as <base>.<name><ext>.
    saved = []
    result = remote.get("result") or {}
    result = json.loads(result) if isinstance(result, str) else result
    video_url = (remote.get("downloads") or {}).get("video")
    if video_url:
        out = video_target(out, os.path.splitext(urllib.parse.urlparse(video_url).path)[1])
    base = os.path.splitext(out)[0]
    for name, url in sorted(remote.get("downloads", {}).items()):
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1]
        take = re.fullmatch(r"(video|keyframe)_(\d+)", name)  # variations: <out>_1.mp4, <out>_1.keyframe.png
        if take:
            path = f"{base}_{take[2]}{ext}" if take[1] == "video" else f"{base}_{take[2]}.keyframe{ext}"
        else:
            path = out if name == "video" else f"{base}.{name}{ext}"
        download(url, path)
        saved.append(path)
    if result.get("plan_only") and result.get("beats"):  # a reviewed plan: one "SECONDS: text" per line
        path = f"{base}.beats.txt"
        with open(path, "w") as f:
            f.write(result["beats"].rstrip("\n") + "\n")
        saved.append(path)
    return saved


def plan_render_command(args, seed, beats_path):
    """The `vidgen video` command that renders a reviewed plan: same prompt, length and avatar, its seed (the same
    first frame) and the beats file."""
    import shlex

    if seed is None:
        raise ApiError("the plan has no seed; render with --beats-file only (the first frame may differ)")
    parts = ["vidgen", "video", args.prompt, "--video-model", "wan", "--seconds", str(args.seconds), "--seed", str(seed),
             "--beats-file", beats_path]
    if getattr(args, "no_avatar", False):
        parts.append("--no-avatar")
    elif getattr(args, "avatar", None):
        parts += ["--avatar", args.avatar]
    if getattr(args, "image_model", "auto") not in (None, "auto"):
        parts += ["--image-model", args.image_model]
    if getattr(args, "shot", None):
        parts += ["--shot", args.shot]
    # the options that shape the first frame or the look, when not the defaults
    if getattr(args, "negative", ""):
        parts += ["--negative", args.negative]
    if (getattr(args, "width", 1280), getattr(args, "height", 704)) != (1280, 704):
        parts += ["--width", str(args.width), "--height", str(args.height)]
    parts += [flag for flag, on in (("--best", getattr(args, "best", False)), ("--draft", getattr(args, "draft", False)),
                                    ("--upscale", getattr(args, "upscale", False))) if on]
    if getattr(args, "fps", 32) != 32:
        parts += ["--fps", str(args.fps)]
    if os.name == "nt":
        import subprocess
        return subprocess.list2cmdline(parts)
    return " ".join(shlex.quote(p) for p in parts)


PLAN_POLL_SECONDS = 15


def wait_for_plan(job_id, args):
    """`vidgen plan`: wait for the plan-only job (Ctrl-C leaves it to the watcher), save <name>.beats.txt and the
    first frame, print the beats and the command that renders them."""
    last = None
    try:
        while True:
            job = api("GET", f"/jobs/{job_id}")
            status, prog = job.get("status"), job.get("progress")
            if (status, prog) != last:
                print(f"  {status}{': ' + prog if prog else ''}", flush=True)
                last = (status, prog)
            if status in ("done", "failed", "cancelled"):
                break
            time.sleep(PLAN_POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nleft to the watcher: the beats and first frame will be saved to the download folder")
        return
    if status != "done":
        raise ApiError(f"plan {status}: {job.get('error') or ''}".strip())
    with locked():
        jobs = load_jobs()
        local = jobs.get(job_id) or {"type": "video", "out": local_out("video", job_id, args.out)}
        saved = fetch_results(local, job)
        if job_id in jobs:
            jobs[job_id]["status"] = "downloaded"
            save_jobs(jobs)
    result = job.get("result") or {}
    result = json.loads(result) if isinstance(result, str) else result
    beats_path = next((p for p in saved if p.endswith(".beats.txt")), "")
    print("\nbeats (edit the file, then render):\n" + result.get("beats", ""))
    for p in saved:
        print(f"saved {p}")
    print("\nrender with:\n  " + plan_render_command(args, result.get("seed"), beats_path))


def fetch_photos(folder, remote):
    """photo_<n>.png into the job's folder, plus prompts.txt saying what each one shows."""
    os.makedirs(folder, exist_ok=True)
    saved = []
    for name, url in sorted(remote.get("downloads", {}).items()):
        path = os.path.join(folder, name + os.path.splitext(urllib.parse.urlparse(url).path)[1])
        download(url, path)
        saved.append(path)
    lines = [f"photo_{s['photo']}.png  [{s['source']}]  {s['text']}" for s in remote.get("result", {}).get("scenes", [])]
    prompts = os.path.join(folder, "prompts.txt")
    with open(prompts, "w") as f:
        f.write("\n".join(lines) + "\n")
    return saved + [prompts]


def fetch_previews(folder, remote):
    """preview_base.mp4 and preview_s<step>.mp4 into the job's folder, plus about.txt."""
    os.makedirs(folder, exist_ok=True)
    saved = []
    for name, url in sorted(remote.get("downloads", {}).items()):
        path = os.path.join(folder, name + os.path.splitext(urllib.parse.urlparse(url).path)[1])
        download(url, path)
        saved.append(path)
    r = remote.get("result", {})
    about = os.path.join(folder, "about.txt")
    with open(about, "w") as f:
        f.write(f"LoRA {r.get('name')} v{r.get('version')}, weight {r.get('weight')}, seed {r.get('seed')}\n"
                f"prompt: {r.get('prompt')}\n"
                "preview_base.mp4 = same prompt and seed without the LoRA\n")
    return saved + [about]


def read_prompts_file(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


SHOT_KEYS = ("where", "wearing", "doing", "camera")


def read_shots_file(path):
    """Photo shots, one block per photo (blank line between blocks), `key: value` lines with
    keys where/wearing/doing/camera, `#` comments -> [{"shot": {...}}]."""
    shots, current = [], {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except (OSError, UnicodeDecodeError) as e:
        raise ApiError(f"cannot read {path}: {e}")
    for n, line in enumerate(lines + [""], 1):
        text = line.strip()
        if text.startswith("#"):
            continue
        if not text:
            if current:
                shots.append({"shot": current})
                current = {}
            continue
        key, sep, value = text.partition(":")
        key = key.strip().lower()
        if not sep or key not in SHOT_KEYS:
            raise ApiError(f"{path}:{n}: expected 'where:', 'wearing:', 'doing:' or 'camera:', got: {text[:60]}")
        if key in current:
            raise ApiError(f"{path}:{n}: '{key}' given twice in one shot")
        if value.strip():
            current[key] = value.strip()
    if not shots:
        raise ApiError(f"{path}: no shots found")
    return shots


# Jobs started on another device (the phone) that the watcher downloads too. docs/specs/2026-11-27-phone-ui.md
ADOPT_DAYS = 7
ADOPT_TYPES = ("video", "photos", "story", "write", "describe", "avatar", "lora_preview", "train")
ADOPT_STATUSES = ("queued", "running", "done")


def load_dismissed():
    try:
        with open(DISMISSED_FILE) as f:
            d = json.load(f)
    except (FileNotFoundError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def _name_taken(out, taken):
    """True when out, or a file a download writes next to it (<base>.<x>, <base>_<n>.mp4), exists or is the
    destination of another job."""
    if out in taken or os.path.exists(out):
        return True
    base = os.path.splitext(out)[0]
    if base != out and (base in taken or os.path.exists(base)):
        return True
    folder, stem = os.path.split(base)
    try:
        return any(f.startswith((stem + ".", stem + "_")) for f in os.listdir(folder))
    except OSError:
        return False


def adopted_out(job_type, job_id, server_name, taken=()):
    """Where an adopted job saves: the server's name only when it is a plain name (it came from another device),
    never over an existing file or another job's destination (taken), and always inside the download folder."""
    name = server_name if isinstance(server_name, str) and OUT_NAME.fullmatch(server_name) else None
    out = local_out(job_type, job_id, name)
    taken = set(taken)
    if name and _name_taken(out, taken):
        base, ext = os.path.splitext(out)
        for tag in (job_id[:6], job_id[:12], job_id):  # the short tag unless another job already has it
            out = f"{base}-{tag}{ext}"
            if not _name_taken(out, taken):
                break
    root = os.path.abspath(download_dir())
    if os.path.commonpath([root, out]) != root or os.path.dirname(out) != root:
        out = local_out(job_type, job_id)
    return out


def adopt_remote(jobs):
    """Register the server's recent jobs that this computer doesn't know (started on another device)."""
    dismissed = load_dismissed()
    cutoff = time.time() - ADOPT_DAYS * 86400
    r = api("GET", "/jobs")
    rows = r.get("jobs") if isinstance(r, dict) else None
    taken = {j.get("out") for j in jobs.values() if isinstance(j, dict)}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):  # server data: skip anything malformed rather than stop the pass
            continue
        job_id, job_type, created = row.get("job_id"), row.get("type"), row.get("created_at")
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id) or job_id in jobs \
                or job_id in dismissed:
            continue
        if not isinstance(created, (int, float)) or isinstance(created, bool) or created < cutoff:
            continue
        if job_type not in ADOPT_TYPES or row.get("status") not in ADOPT_STATUSES:
            continue
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        extra = {k: meta[k] for k in ("avatar_name", "krea_lora") if meta.get(k) is not None}
        if not (isinstance(extra.get("avatar_name"), str) and OUT_NAME.fullmatch(extra["avatar_name"])):
            extra.pop("avatar_name", None)
        if not isinstance(extra.get("krea_lora"), bool):
            extra.pop("krea_lora", None)
        out = adopted_out(job_type, job_id, meta.get("out_name"), taken)
        taken.add(out)
        jobs[job_id] = {"type": job_type, "out": out, "status": "queued", "submitted_at": created,
                        "adopted": True, **extra}
        log(f"{job_id}: started on {meta.get('source') or 'another device'}; downloading it here too")


def forget_jobs(prefixes):
    """`vidgen jobs forget ID...`: drop jobs from the local list and never adopt them again."""
    with locked():
        jobs = load_jobs()
        dismissed = load_dismissed()
        ids = [resolve_job_id(p) for p in prefixes]  # all resolve, or nothing changes
        for job_id in ids:
            if not re.fullmatch(r"[0-9a-f]{32}", job_id):
                raise ApiError(f"'{job_id}' is not a job id")
        for job_id in ids:
            jobs.pop(job_id, None)
            dismissed[job_id] = int(time.time())
            print(f"forgot job {job_id}")
        cutoff = time.time() - 2 * FORGET_AFTER_DAYS * 86400  # past the server's history by then
        dismissed = {k: v for k, v in dismissed.items() if v > cutoff}
        os.makedirs(STATE_DIR, exist_ok=True)
        tmp = DISMISSED_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dismissed, f, indent=2)
        os.replace(tmp, DISMISSED_FILE)
        save_jobs(jobs)


def check_once():
    """One pass over pending jobs. Returns how many are still pending."""
    with locked():
        jobs = load_jobs()
        try:
            adopt_remote(jobs)
        except Exception as e:  # noqa: BLE001 - whatever the server sent, the local jobs still get checked
            log(f"listing jobs from other devices failed, will retry: {e}")
        for job_id in [jid for jid, j in jobs.items() if j["status"] in PENDING]:
            local = jobs[job_id]
            try:
                remote = api("GET", f"/jobs/{job_id}")
            except ApiError as e:
                if e.status == 404:
                    local["status"] = "lost"
                    log(f"{job_id}: not found on server, giving up")
                else:
                    log(f"{job_id}: check failed, will retry: {e}")
                continue

            status = remote["status"]
            if status == "done":
                try:
                    saved = fetch_results(local, remote)
                except (OSError, urllib.error.URLError) as e:
                    if permanent_download_error(e):
                        local["download_errors"] = local.get("download_errors", 0) + 1
                        if local["download_errors"] >= 3:  # retrying can't fix it: stop re-downloading
                            local["status"], local["error"] = "download_failed", str(e)
                            log(f"{job_id}: download failed, giving up: {e}")
                            notify("Download failed", os.path.basename(local.get("out", job_id)))
                            continue
                    log(f"{job_id}: download failed, will retry: {e}")
                    continue
                try:
                    extra = save_realism(local, remote)
                    if extra:
                        saved = saved + [extra]
                except OSError as e:
                    log(f"{job_id}: couldn't save the AI-look verdict: {e}")
                local["status"] = "downloaded"
                local["files"] = saved
                local.pop("progress", None)
                log(f"{job_id}: downloaded {', '.join(saved)}")
                if local["type"] == "avatar":
                    log(f"{job_id}: pick one with: vidgen.py avatar pick {job_id} <1-{len(saved)}>")
                    notify("Avatar candidates ready", f"vidgen avatar pick {job_id[:8]} <number>")
                elif local["type"] == "photos":
                    notify("Photos ready", os.path.basename(local["out"]))
                elif local["type"] == "lora_preview":
                    log(f"{job_id}: pick a checkpoint with: vidgen.py lora use NAME --version N --step S")
                    notify("LoRA previews ready", os.path.basename(local["out"]))
                elif local["type"] == "train":
                    r = remote.get("result", {})
                    if r.get("captions_only"):
                        log(f"{job_id}: {r.get('captions')} captions ready; review: vidgen.py dataset captions NAME --edit")
                        notify("Captions ready", f"{r.get('captions')} captions")
                    else:
                        log(f"{job_id}: LoRA {r.get('lora')} v{r.get('version')} trained "
                            f"({r.get('steps_per_expert')} steps per expert, {r.get('checkpoints')} checkpoints)")
                        notify("Training done", f"{r.get('lora')} v{r.get('version')}")
                        if r.get("preview_job_id") and r["preview_job_id"] not in jobs:
                            pid = r["preview_job_id"]
                            jobs[pid] = {"type": "lora_preview", "out": adopted_out("lora_preview", pid, f"lora-preview-{r.get('lora')}-v{r.get('version')}",
                                                                                   {j.get("out") for j in jobs.values()}),
                                         "status": "queued", "submitted_at": time.time()}
                            log(f"{pid}: checkpoint previews queued on the render box; they download when ready")
                elif local["type"] == "write":
                    title = remote.get("result", {}).get("title", "")
                    notify("Story written", f"{title} ({os.path.basename(local['out'])})")
                    log(f"{job_id}: film it with: vidgen story {local['out']}")
                else:
                    notify("Video ready", os.path.basename(saved[0]) if saved else job_id)
            elif status in ("failed", "cancelled"):
                local["status"] = status
                local["error"] = remote.get("error", "")
                local.pop("progress", None)
                log(f"{job_id}: {status}: {local['error']}".rstrip(": "))
                if status == "failed":
                    notify("Video job failed", f"{job_id[:8]}: {local['error'][:100]}")
            else:
                prog = remote.get("progress", "") + (" (cancelling)" if remote.get("cancel_requested") else "")
                if (status, prog) != (local["status"], local.get("progress")):
                    log(f"{job_id}: {status} {prog}".rstrip())
                local["status"] = status
                local["progress"] = prog
            local["checked_at"] = time.time()

        save_jobs(jobs)
        pending = sum(1 for j in jobs.values() if j["status"] in PENDING)
    maybe_sync_loras()  # outside the jobs lock: a big LoRA download never blocks job bookkeeping
    maybe_sync_avatar_media()
    return pending


def watch(interval, until_done):
    while True:
        try:
            remaining = check_once()
        except ApiError as e:  # e.g. config missing; keep the loop alive
            log(f"check failed: {e}")
            remaining = 1
        if until_done and remaining == 0:
            return
        time.sleep(interval)


def redirect_output(path):
    """Background runs (pythonw on Windows has no console) log to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a", buffering=1)
    sys.stdout = sys.stderr = f


# ---------------------------------------------------------------- background watcher
#
# macOS: launchd agent, `check` every 45 s.
# Linux: systemd user timer, `check` every 45 s.
# Windows: Task Scheduler cannot repeat more often than every minute, so a
#          Startup-folder launcher runs `watch` (a 45 s loop) under pythonw.

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _pythonw():
    candidate = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    return candidate if os.path.exists(candidate) else sys.executable


def watcher_installed():
    if sys.platform == "darwin":
        return os.path.exists(LAUNCHD_PLIST)
    if sys.platform == "win32":
        return os.path.exists(os.path.join(WINDOWS_STARTUP, "vidgen-watcher.cmd"))
    return os.path.exists(os.path.join(SYSTEMD_DIR, "vidgen-watcher.timer"))


def watcher_install(interval):
    if not setting("url"):
        raise ApiError("run `vidgen.py configure --url ...` first")
    # Save what the background process needs; it won't see this shell's env.
    save_config(url=setting("url"), ssh_key=setting("ssh_key"), download_dir=setting("download_dir"),
                ssh_auth_sock=os.environ.get("SSH_AUTH_SOCK"))

    if sys.platform == "darwin":
        import plistlib
        plist = {
            "Label": LAUNCHD_LABEL,
            "ProgramArguments": [sys.executable, SCRIPT, "check"],
            "StartInterval": interval,
            "RunAtLoad": True,
            "EnvironmentVariables": {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin"},
            "StandardOutPath": LOG_FILE,
            "StandardErrorPath": LOG_FILE,
            "ProcessType": "Background",
        }
        subprocess.run(["launchctl", "unload", LAUNCHD_PLIST], capture_output=True)
        os.makedirs(os.path.dirname(LAUNCHD_PLIST), exist_ok=True)
        with open(LAUNCHD_PLIST, "wb") as f:
            plistlib.dump(plist, f)
        subprocess.run(["launchctl", "load", LAUNCHD_PLIST], check=True)

    elif sys.platform == "win32":
        launcher = os.path.join(WINDOWS_STARTUP, "vidgen-watcher.cmd")
        cmdline = f'"{_pythonw()}" "{SCRIPT}" watch --interval {interval} --log "{LOG_FILE}"'
        _write(launcher, f'@echo off\r\nstart "" {cmdline}\r\n')
        watcher_uninstall(quiet=True, keep_launcher=True)
        subprocess.Popen([_pythonw(), SCRIPT, "watch", "--interval", str(interval), "--log", LOG_FILE],
                         creationflags=0x00000008 | 0x08000000)  # DETACHED_PROCESS | CREATE_NO_WINDOW

    else:
        if not shutil.which("systemctl"):
            raise ApiError("no systemd here; run `vidgen.py watch` in a terminal or tmux instead")
        _write(os.path.join(SYSTEMD_DIR, "vidgen-watcher.service"), f"""[Unit]
Description=vidgen: download finished video jobs

[Service]
Type=oneshot
ExecStart="{sys.executable}" "{SCRIPT}" check --log "{LOG_FILE}"
""")
        _write(os.path.join(SYSTEMD_DIR, "vidgen-watcher.timer"), f"""[Unit]
Description=vidgen: check for finished jobs every {interval} s

[Timer]
OnBootSec=30s
OnUnitActiveSec={interval}s
AccuracySec=5s

[Install]
WantedBy=timers.target
""")
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "--now", "vidgen-watcher.timer"], check=True)

    print(f"watcher installed: checks every {interval}s, log at {LOG_FILE}")


def watcher_uninstall(quiet=False, keep_launcher=False):
    if sys.platform == "darwin":
        subprocess.run(["launchctl", "unload", LAUNCHD_PLIST], capture_output=True)
        if os.path.exists(LAUNCHD_PLIST):
            os.remove(LAUNCHD_PLIST)
    elif sys.platform == "win32":
        launcher = os.path.join(WINDOWS_STARTUP, "vidgen-watcher.cmd")
        if not keep_launcher and os.path.exists(launcher):
            os.remove(launcher)
        if os.path.exists(PID_FILE):
            pid = open(PID_FILE).read().strip()
            subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
            os.remove(PID_FILE)
    else:
        subprocess.run(["systemctl", "--user", "disable", "--now", "vidgen-watcher.timer"], capture_output=True)
        for unit in ("vidgen-watcher.timer", "vidgen-watcher.service"):
            path = os.path.join(SYSTEMD_DIR, unit)
            if os.path.exists(path):
                os.remove(path)
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    if not quiet:
        print("watcher removed")


def watcher_status():
    print(f"watcher: {'installed' if watcher_installed() else 'not installed'} ({sys.platform})")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            print("".join(f.readlines()[-10:]), end="")


# ---------------------------------------------------------------- avatar

MAX_AVATAR_IMAGES = 12


def upload_images(paths):
    if not 1 <= len(paths) <= MAX_AVATAR_IMAGES:
        raise ApiError(f"give 1-{MAX_AVATAR_IMAGES} images")
    keys = []
    for i, path in enumerate(paths, 1):
        print(f"uploading image {i}/{len(paths)}: {os.path.basename(path)}")
        keys.append(upload_file(os.path.expanduser(path), IMAGE_EXTS))
    return keys


def avatar_from_image(name, images, description, make_default=True, krea_lora=True):
    """images: one path or a list of paths of the same person (the first is the main image).
    krea_lora: also build the avatar's Krea 2 LoRA (~1.5 h GPU)."""
    images = [images] if isinstance(images, str) else list(images)
    keys = upload_images(images)
    r = api("POST", "/avatars", {"name": name, "upload_keys": keys, "description": description,
                                 "make_default": make_default, "krea_lora": krea_lora})
    if krea_lora and (r or {}).get("krea_lora", {}).get("state") == "queued":
        print(f"its Krea 2 LoRA is being built (~1.5 h; trigger word '{name}'); see vidgen.py avatar list")
    print(f"avatar '{name}' saved with {len(keys)} image{'s' if len(keys) > 1 else ''}"
          + (" and set as default" if make_default else ""))
    if len(keys) == 1:
        print(f"tip: more photos of the same person improve likeness: vidgen.py avatar add --name {name} FILE...")


def avatar_generate(name, description, candidates, model="wan", krea_lora=True):
    out = f"avatar-{name}-{int(time.time())}.png"
    job_id = submit({"type": "avatar", "description": description, "candidates": candidates, "model": model}, out,
                    {"avatar_name": name, "krea_lora": krea_lora})
    print(f"{candidates} candidate images are being generated. When the watcher has downloaded them,")
    print(f"look at them and pick one:  vidgen.py avatar pick {job_id} <1-{candidates}>")
    return job_id


def avatar_setup(name, image=None, description=None, candidates=4, model="wan", krea_lora=True):
    """Interactive unless image/description are given (image: a path or a list of paths). Returns True when an
    avatar is ready now, False when candidates are still being generated."""
    if image is None and description is None:
        if not sys.stdin.isatty():
            raise ApiError("no avatar set up; run `vidgen.py avatar refresh` in a terminal, or pass --no-avatar")
        print("The avatar is the character who stars in every video, so clips stay consistent.")
        print("Use a fictional or AI-generated character, or yourself; never a real person without their consent.")
        image = input("Path to an avatar image (png/jpg/webp), or press Enter to generate one from a description: ").strip()
        prompt = ("Short description of the character (look, hair, clothing); it helps keep them consistent: "
                  if image else "Describe the character (look, hair, clothing, age range): ")
        description = input(prompt).strip()
    if image:
        avatar_from_image(name, image, description or "", krea_lora=krea_lora)
        return True
    if not description:
        raise ApiError("a description is needed to generate an avatar")
    avatar_generate(name, description, candidates, model, krea_lora=krea_lora)
    return False


def ensure_avatar(args):
    """Before a video/story job: make sure an avatar exists unless opted out."""
    if args.no_avatar or args.avatar:
        return True
    if api("GET", "/avatars")["default"]:
        return True
    print("No avatar set up yet.")
    if avatar_setup(DEFAULT_AVATAR_NAME):
        return True
    print("Your video was not submitted yet. Run the same command again after picking an avatar,")
    print("or add --no-avatar to submit it without one.")
    return False


def avatar_command(args):
    a = args.args
    if args.action == "list":
        r = api("GET", "/avatars")
        if not r["avatars"]:
            print("no avatars yet; run: vidgen.py avatar refresh")
        for av in r["avatars"]:
            mark = "*" if av["name"] == r["default"] else " "
            count = av.get("images", 1)
            kl = (av.get("krea_lora") or {}).get("state", "none")
            md = (av.get("media") or {}).get("state", "none")
            wl = (av.get("wan_lora") or {}).get("state", "none")
            print(f"{mark} {av['name']:16} {count:2} image{'s' if count != 1 else ' '}  Krea LoRA: {kl:9} "
                  f"Wan LoRA: {wl:9} media: {md:7} {av['description']}")
    elif args.action == "refresh":
        name = args.name or api("GET", "/avatars")["default"] or DEFAULT_AVATAR_NAME
        avatar_setup(name, args.image, args.describe, args.candidates, args.model,
                     krea_lora=not getattr(args, "no_krea_lora", False))
    elif args.action == "pick":
        if len(a) != 2 or not a[1].isdigit():
            raise ApiError("usage: vidgen.py avatar pick <job_id> <candidate number>")
        job_id = resolve_job_id(a[0])
        local = load_jobs().get(job_id, {})
        if not local.get("avatar_name"):  # started on another device: the server has its avatar name
            local = dict(server_meta(job_id), **local)
        name = args.name or local.get("avatar_name") or DEFAULT_AVATAR_NAME
        krea_lora = local.get("krea_lora", True) and not getattr(args, "no_krea_lora", False)
        api("POST", "/avatars", {"name": name, "job_id": job_id, "candidate": int(a[1]), "make_default": True,
                                 "krea_lora": krea_lora})
        print(f"avatar '{name}' saved from candidate {a[1]} and set as default")
    elif args.action == "krea-train":
        if not args.name:
            raise ApiError("usage: vidgen.py avatar krea-train --name <avatar>")
        r = api("POST", f"/avatars/{args.name.lower()}/krea-lora")
        kl = r["krea_lora"]
        print(f"Krea 2 LoRA for '{r['name']}' {kl['state']}: {kl['lora']} v{kl.get('version')} "
              f"(~1.5 h; trigger word '{r['name']}'); see vidgen.py avatar list")
    elif args.action == "wan-train":
        name = args.name or (a[0] if a else None)
        if not name:
            raise ApiError("usage: vidgen.py avatar wan-train --name <avatar>")
        r = api("POST", f"/avatars/{name.lower()}/wan-lora", {})
        wl = r["wan_lora"]
        print(f"Wan LoRA for '{r['name']}' {wl['state']}: {wl.get('lora')} v{wl.get('version')} from its media "
              f"(~1 h on the trainer; added to its videos automatically when ready); see vidgen.py avatar list")
    elif args.action == "media":
        name = args.name or (a[0] if a else None)
        if not name:
            raise ApiError("usage: vidgen.py avatar media --name <avatar>")
        r = api("POST", f"/avatars/{name.lower()}/media", {})
        print(f"media for '{r['name']}' {r['media']['state']}: photos and four 2 s clips (~6-8 min when the "
              f"avatar has a Krea photo set, ~25 min otherwise); copied to "
              f"{os.path.join(download_dir(), 'avatars', r['name'])} by the watcher")
    elif args.action == "use":
        if not args.name:
            raise ApiError("usage: vidgen.py avatar use --name <avatar>")
        api("PUT", "/avatars/default", {"name": args.name})
        print(f"default avatar is now '{args.name}'")
    elif args.action == "history":
        name = args.name or api("GET", "/avatars")["default"] or DEFAULT_AVATAR_NAME
        r = api("GET", f"/avatars/{name}/history")
        if not r["versions"]:
            print(f"'{name}' has no earlier versions")
        for v in r["versions"]:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(v["created_at"]))
            print(f"{v['version']}  {when}  {v.get('images', 1)} image(s)  {v['description']}\n"
                  f"    preview (1 h): {v['preview']}")
    elif args.action == "add":
        if not a:
            raise ApiError("usage: vidgen.py avatar add [--name NAME] FILE...")
        name = args.name or api("GET", "/avatars")["default"] or DEFAULT_AVATAR_NAME
        r = api("POST", f"/avatars/{name}/images", {"upload_keys": upload_images(a)})
        print(f"added {len(a)} image{'s' if len(a) > 1 else ''} to '{name}' (now {r['images']}; "
              "the previous set is in history)")
    elif args.action == "images":
        name = args.name or api("GET", "/avatars")["default"] or DEFAULT_AVATAR_NAME
        r = api("GET", f"/avatars/{name}")
        print(f"'{name}': {len(r['images'])} image(s); photos and keyframes use the first "
              f"{min(3, len(r['images']))}, and image 1 alone in shots with several avatars")
        for img in r["images"]:
            print(f"  {img['n']:2}  {'main ' if img['n'] == 1 else '     '}preview (1 h): {img['preview']}")
    elif args.action in ("remove", "main"):
        if len(a) != 1 or not a[0].isdigit():
            raise ApiError(f"usage: vidgen.py avatar {args.action} [--name NAME] <image number>   "
                           "(numbers: avatar images)")
        name = args.name or api("GET", "/avatars")["default"] or DEFAULT_AVATAR_NAME
        if args.action == "remove":
            r = api("DELETE", f"/avatars/{name}/images/{a[0]}")
            print(f"removed image {a[0]} from '{name}' (now {r['images']}; the previous set is in history)")
        else:
            api("PUT", f"/avatars/{name}/main", {"index": int(a[0])})
            print(f"image {a[0]} is now the main image of '{name}'")
    elif args.action == "delete":
        name = (args.name or (a[0] if a else "")).lower()
        if not name:  # never the default by omission: deleting can't be undone
            raise ApiError("usage: vidgen.py avatar delete --name <avatar> [--with-loras] [--yes]")
        what = "images, earlier versions and media" + (", and its LoRAs with their training photos"
                                                       if args.with_loras else "")
        if not args.yes and not _ask(f"Delete avatar '{name}' from the server ({what})? This can't be undone."):
            print("nothing deleted")
            return
        r = api("DELETE", f"/avatars/{name}", {"loras": args.with_loras})
        print(f"avatar '{name}' deleted ({r['deleted']} files"
              + (f"; LoRAs: {', '.join(r['loras'])}" if r.get("loras") else "") + ")"
              + ("; it was the default: pick another with vidgen.py avatar use --name N" if r.get("was_default") else ""))
    elif args.action == "restore":
        if len(a) != 1:
            raise ApiError("usage: vidgen.py avatar restore <version> [--name NAME]   (versions: avatar history)")
        name = args.name or api("GET", "/avatars")["default"] or DEFAULT_AVATAR_NAME
        api("POST", f"/avatars/{name}/restore", {"version": a[0]})
        print(f"avatar '{name}' restored to version {a[0]} (the replaced version is in history)")


# ---------------------------------------------------------------- CLI

def show_jobs():
    jobs = load_jobs()
    if not jobs:
        print("no jobs")
        return
    for job_id, j in sorted(jobs.items(), key=lambda kv: kv[1]["submitted_at"]):
        when = time.strftime("%m-%d %H:%M", time.localtime(j["submitted_at"]))
        detail = j.get("error") or j.get("progress") or j["out"]
        print(f"{when}  {job_id[:12]}  {j['type']:8}  {j['status']:10}  {detail}")


def cancel(prefix):
    job_id = resolve_job_id(prefix)
    r = api("POST", f"/jobs/{job_id}/cancel")
    if r["status"] == "cancelled":
        print(f"job {job_id} cancelled")
    else:
        print(f"job {job_id} is running; it will stop within ~15 s and show as cancelled")


def _ask(question):
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def dataset_command(args):
    import datasets
    import ui
    if args.action == "captions":
        return captions_command(args)
    if args.action == "list":
        found = api("GET", "/datasets")["datasets"]
        if not found:
            print("no datasets yet; upload one with: vidgen.py dataset add NAME FOLDER --kind KIND")
        for d in found:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(d["updated_at"]))
            print(f"{d['name']:16} {d['kind']:10} {d['file_count']:5} files  {datasets.human_bytes(d['total_bytes']):>9}  "
                  f"updated {when}  trigger {d['trigger']}")
        return
    if args.action == "rm":
        if not (args.yes or _ask(f"Delete dataset '{args.name}' and all its uploaded files from S3?")):
            print("nothing deleted")
            return
        r = api("DELETE", f"/datasets/{args.name}")
        print(f"deleted {r['deleted']} object(s)")
        if args.name in ui.load_themes():
            ui.remove_theme(args.name)
        return

    # add: register/update the local theme, then sync the folder
    if not args.folder or not args.kind:
        raise ApiError("dataset add needs NAME FOLDER --kind character|style|motion|place")
    theme = ui.upsert_theme(args.name, args.kind, args.folder, args.trigger)
    for line in theme["scan"]["problems"] + theme["scan"]["warnings"]:
        print(f"  note: {line}")
    p = datasets.plan(args.name, theme["folder"])
    for rel in p["skipped"]["too_big"]:
        print(f"  skipped (over 2 GB): {rel}")
    print(f"{len(p['files'])} files ({datasets.human_bytes(p['total_bytes'])}); to upload: {len(p['upload'])} "
          f"({datasets.human_bytes(p['upload_bytes'])}); to delete from the dataset: {len(p['remove'])}")
    allow_remove = False
    if p["remove"]:
        for rel in p["remove"][:20]:
            print(f"  no longer in the folder: {rel}")
        if len(p["remove"]) > 20:
            print(f"  ... and {len(p['remove']) - 20} more")
        allow_remove = args.yes or _ask(f"Delete these {len(p['remove'])} file(s) from the dataset?")
        if not allow_remove:
            print("nothing uploaded; restore the files or confirm the deletion")
            return
    consent = False
    if args.kind == "character" and not p["has_consent"]:
        consent = args.yes or _ask("Do you have the consent of the person shown to train on their likeness?")
        if not consent:
            print("nothing uploaded: character datasets need the consent of the person shown")
            return
    last = [0.0]

    def progress(st):
        if time.time() - last[0] > 2 or st["files_done"] == st["files_total"]:
            last[0] = time.time()
            print(f"  {st['files_done']}/{st['files_total']} files, {datasets.human_bytes(st['bytes_done'])} "
                  f"of {datasets.human_bytes(st['bytes_total'])}", flush=True)
    result = datasets.sync(args.name, theme["folder"], args.kind, theme["trigger"], consent=consent,
                           allow_remove=allow_remove, progress=progress, the_plan=p)
    ui.record_upload(args.name, result)
    print(f"dataset {args.name}: {result['file_count']} files, {datasets.human_bytes(result['total_bytes'])} "
          f"(uploaded {result['uploaded']}, deleted {result['removed']}); trigger {result['trigger']}")


# Measured on the L40S trainer, 2026-11-18: standard 17.6 s/step steady (18.1 incl. overhead), fast 6.0.
SECONDS_PER_STEP = {"standard": 18.1, "fast": 6.0}


def captions_command(args):
    """dataset captions NAME [--edit]: show, or edit in $EDITOR, the training captions."""
    caps = api("GET", f"/datasets/{args.name}/captions")
    items = caps.get("items") or {}
    if not caps.get("current"):
        print("note: these captions were made for an older version of the dataset; the next training captions again")
    if not args.edit:
        for name, text in items.items():
            print(f"{name}\t{text}")
        edited = time.strftime("%Y-%m-%d %H:%M", time.localtime(caps["edited_at"])) if caps.get("edited_at") else "never"
        print(f"\n{len(items)} captions, trigger {caps.get('trigger')}, edited {edited}. Edit with --edit.")
        return
    import tempfile
    header = ("# One caption per line: <item><TAB><caption>. Keep the trigger word "
              f"'{caps.get('trigger')}' in each caption.\n# Describe setting, action, camera and light; "
              "leave out what the LoRA should learn. Lines starting with # are ignored.\n")
    fd, path = tempfile.mkstemp(suffix=".tsv", prefix=f"captions-{args.name}-")
    with os.fdopen(fd, "w") as f:
        f.write(header + "".join(f"{k}\t{v}\n" for k, v in items.items()))
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "nano")
    subprocess.call([*editor.split(), path])
    changed = {}
    for n, line in enumerate(open(path), 1):
        if not line.strip() or line.startswith("#"):
            continue
        name, sep, text = line.rstrip("\n").partition("\t")
        if not sep:
            raise ApiError(f"{path}:{n}: expected <item><TAB><caption>")
        if items.get(name.strip()) != text.strip():
            changed[name.strip()] = text.strip()
    if not changed:
        print("no changes")
        return
    r = api("PUT", f"/datasets/{args.name}/captions", {"items": changed})
    print(f"saved {r['updated']} caption(s); the next training of {args.name} uses them")
    if r.get("without_trigger"):
        print(f"warning: no trigger word in: {', '.join(r['without_trigger'][:8])}")
    os.remove(path)


MAX_LORA_BYTES = 2 * 1024 ** 3


def put_file(url, path, label=""):
    """Stream a (large) file to a presigned PUT URL without reading it all into memory."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        req = urllib.request.Request(url, data=f, method="PUT", headers={"Content-Length": str(size)})
        if label:
            print(f"uploading {label}: {os.path.basename(path)} ({size / 1024 ** 2:.0f} MB)", flush=True)
        urllib.request.urlopen(req, timeout=3600).close()


# Which model a LoRA file is for, from its header: Krea 2 (image model) or Wan (video, "t2v").
# Same rules as the API's _check_krea2_lora / _check_wan_lora (lambda/api/handler.py; the Lambda
# bundle holds only handler.py, so this is a copy and tests/test_api.py checks they agree). Anything
# that isn't Krea is sent as Wan, so the API's full Wan checks give the reason when it's neither.
LORA_KEY = re.compile(r"lora[._-]?(a|b|up|down)\b", re.IGNORECASE)
WAN_MARKERS = re.compile(r"self_attn|cross_attn|ffn|attn1|attn2")
KREA2_MODULE = re.compile(r"attn[._]?(wq|wk|wv|wo|gate|qkv_proj|to_q|to_k|to_v|to_out)|mlp[._]?(gate|up|down)")
MAX_SAFETENSORS_HEADER = 100 * 1024 ** 2
OTHER_MODEL_KEYS = ("double_blocks", "single_blocks", "input_blocks", "output_blocks", "down_blocks",
                    "transformer_blocks", "joint_blocks")


def read_safetensors_header(path):
    """The JSON header of a local .safetensors file (reads only the header)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if not 2 <= n <= min(MAX_SAFETENSORS_HEADER, os.path.getsize(path) - 8):
            raise ApiError(f"{path}: not a .safetensors file")
        try:
            header = json.loads(f.read(n))
        except ValueError:
            raise ApiError(f"{path}: not a .safetensors file (bad header)")
    if not isinstance(header, dict):
        raise ApiError(f"{path}: not a .safetensors file (bad header)")
    return header


def lora_kind(header):
    keys = [k for k in header if k != "__metadata__"]
    if any(marker in k for k in keys for marker in OTHER_MODEL_KEYS):  # refuse before a long upload
        raise ApiError("made for a different model (Flux, SD, SDXL, Qwen-Image, ...), not Wan or Krea 2")
    lora = [k for k in keys if LORA_KEY.search(k)]
    krea = any(re.search(r"blocks[._]\d+[._]", k) and KREA2_MODULE.search(k) for k in lora)
    return "krea2" if krea and not any(WAN_MARKERS.search(k) for k in keys) else "t2v"


def lora_upload(name, high, low=None, trigger="", description="", make_current=True):
    """Add a LoRA file made elsewhere as a new version of NAME. The kind comes from the file: a Krea 2
    LoRA is one file; for Wan one file serves both Wan 2.2 experts, or with two, `high` goes to the
    high-noise expert and `low` to the low-noise one."""
    paths = [os.path.expanduser(p) for p in ([high, low] if low else [high])]
    for p in paths:
        if not p.lower().endswith(".safetensors"):
            raise ApiError(f"{p}: must be a .safetensors file")
        if not os.path.isfile(p):
            raise ApiError(f"{p}: file not found")
        if os.path.getsize(p) > MAX_LORA_BYTES:
            raise ApiError(f"{p}: larger than 2 GB")
    kinds = [lora_kind(read_safetensors_header(p)) for p in paths]
    if "krea2" in kinds and len(paths) > 1:
        raise ApiError("Krea 2 LoRAs are one file; use --file")
    model = kinds[0]
    sizes = {part: os.path.getsize(path) for part, path in zip(("high", "low"), paths)}
    r = api("POST", f"/loras/{name.lower()}/uploads", {"files": "pair" if low else "single", "model": model,
                                                       "sizes": sizes})
    for part, path in zip(("high", "low"), paths):
        label = f"{part}-noise file" if low else ("Krea 2 LoRA" if model == "krea2" else "LoRA")
        if part in (r.get("multipart") or {}):  # big file: parallel parts
            print(f"uploading {label}: {os.path.basename(path)} ({sizes[part] / 1024 ** 2:.0f} MB, "
                  f"{len(r['multipart'][part]['part_urls'])} parts in parallel)", flush=True)
            upload_multipart(r["multipart"][part], path)
        else:
            put_file(r["urls"][part], path, label)
    return api("PUT", f"/loras/{name.lower()}/versions/{r['version']}",
               {"trigger": trigger, "description": description, "make_current": make_current})


# Local copies of every LoRA (trained, uploaded, avatar): <download dir>/loras/<name>/v<N>/. `.files.json`
# records the final files' sizes after a complete download, so complete versions need no server call.
LORA_SYNC_EVERY = 600  # seconds between the watcher's LoRA syncs
LORA_RECORD = ".files.json"
LORA_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
LORA_LOCAL_FILE = re.compile(r"(checkpoints/)?[A-Za-z0-9_.-]+\.safetensors")  # same rule as the API


def _lora_dir(name, version):
    return os.path.join(download_dir(), "loras", name, f"v{int(version)}")


def _lora_complete(d):
    try:
        with open(os.path.join(d, LORA_RECORD)) as f:
            sizes = json.load(f)
    except (OSError, ValueError):
        return False
    return bool(sizes) and all(os.path.isfile(os.path.join(d, n)) and os.path.getsize(os.path.join(d, n)) == s
                               for n, s in sizes.items())


def lora_sync(names=None, checkpoints=False):
    """Download every registered LoRA version not yet copied here (or only `names`); checkpoints only
    when asked. Returns how many versions got new files. One sync at a time (a second one is skipped);
    a version that fails is logged and retried next time, the others still sync."""
    import fcntl

    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, "lora_sync.lock"), "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log("another LoRA sync is running; skipping")
            return 0
        changed_versions = 0
        for lora in api("GET", "/loras")["loras"]:
            if names and lora["name"] not in names:
                continue
            if not LORA_NAME_RE.fullmatch(str(lora.get("name"))):
                log(f"LoRA name {lora.get('name')!r} from the server is not valid; skipped")
                continue
            for v in lora["versions"]:
                try:
                    changed_versions += _sync_version(lora["name"], int(v["version"]), checkpoints)
                except Exception as e:  # noqa: BLE001 - network, disk, server: retried next sync
                    log(f"LoRA {lora['name']} v{v['version']}: copy failed, will retry: {type(e).__name__}: {e}")
        return changed_versions


def _sync_version(name, version, checkpoints):
    d = _lora_dir(name, version)
    if not checkpoints and _lora_complete(d):
        return 0
    info = api("GET", f"/loras/{name}/versions/{version}/files")
    want = dict(info["files"], **(info["checkpoints"] if checkpoints else {}))
    root = os.path.realpath(d) + os.sep
    for fname in want:
        if not LORA_LOCAL_FILE.fullmatch(fname) or not os.path.realpath(os.path.join(d, fname)).startswith(root):
            raise ValueError(f"file name {fname!r} from the server is not allowed")
    changed = False
    for fname, link in want.items():
        path = os.path.join(d, fname)
        if os.path.isfile(path) and os.path.getsize(path) == link["size"]:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            download(link["url"], path)
        except BaseException:
            if os.path.exists(path + ".part"):  # never leave a half file (or fill the disk with it)
                os.remove(path + ".part")
            raise
        changed = True
    if changed or not os.path.isfile(os.path.join(d, "lora.json")):
        download(info["lora_json"], os.path.join(d, "lora.json"))
    with open(os.path.join(d, LORA_RECORD), "w") as f:  # only after every final file is here
        json.dump({n: link["size"] for n, link in info["files"].items()}, f)
    if changed:
        log(f"LoRA {name} v{version}: copied to {d}")
    return int(changed)


def maybe_sync_loras():
    """From the watcher: sync LoRAs at most every LORA_SYNC_EVERY seconds; failures are logged and retried."""
    stamp = os.path.join(STATE_DIR, "lora_sync.json")
    try:
        with open(stamp) as f:
            last = json.load(f).get("at", 0)
    except (OSError, ValueError):
        last = 0
    if time.time() - last < LORA_SYNC_EVERY:
        return
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(stamp, "w") as f:
        json.dump({"at": time.time()}, f)
    try:
        lora_sync()
    except Exception as e:  # noqa: BLE001 - the watcher must keep running whatever the sync hits
        log(f"LoRA sync failed, will retry: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- avatar media copies
# Every avatar's photos and 2 s clips are copied to <download dir>/avatars/<name>/ by the watcher.
# docs/specs/2026-11-24-avatar-media.md
AVATAR_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
AVATAR_MEDIA_FILE = re.compile(r"(photo_\d{2}\.png|clip_\d\.mp4|media\.json)")  # same rule as the API


def avatar_media_sync():
    """Copy every avatar's ready media not yet here (or changed: new media, or a file of another size).
    Returns how many avatars got new files. One sync at a time; an avatar that fails is retried next time."""
    import fcntl

    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, "avatar_media_sync.lock"), "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log("another avatar media sync is running; skipping")
            return 0
        changed = 0
        for av in api("GET", "/avatars")["avatars"]:
            media = av.get("media") or {}
            if media.get("state") != "ready":
                continue
            if not AVATAR_NAME_RE.fullmatch(str(av.get("name"))):
                log(f"avatar name {av.get('name')!r} from the server is not valid; skipped")
                continue
            try:
                changed += _sync_avatar_media(av["name"], media.get("updated_at"))
            except Exception as e:  # noqa: BLE001 - network, disk, server: retried next sync
                log(f"avatar {av['name']} media: copy failed, will retry: {type(e).__name__}: {e}")
        return changed


def _sync_avatar_media(name, updated_at):
    """Copy one avatar's media. New media (another updated_at, or none given) is fetched in full; the
    same media only fills in missing or wrong-sized files. Files the server no longer lists are removed."""
    d = os.path.join(download_dir(), "avatars", name)
    record_path = os.path.join(d, LORA_RECORD)
    try:
        with open(record_path) as f:
            record = json.load(f)
    except (OSError, ValueError):
        record = {}
    files = record.get("files") or {}
    same = updated_at is not None and record.get("updated_at") == updated_at
    if same and files and all(os.path.isfile(os.path.join(d, n)) and
                              (s is None or os.path.getsize(os.path.join(d, n)) == s) for n, s in files.items()):
        return 0
    info = api("GET", f"/avatars/{name}/media")
    os.makedirs(d, exist_ok=True)
    got, new = {}, False
    for f in info.get("files") or []:
        fname, size = str(f.get("name")), f.get("size")
        if not AVATAR_MEDIA_FILE.fullmatch(fname):
            log(f"avatar {name} media: file name {fname!r} from the server is not allowed; skipped")
            continue
        path = os.path.join(d, fname)
        if same and os.path.isfile(path) and (size is None or os.path.getsize(path) == size):
            got[fname] = size
            continue
        try:
            download(f["url"], path)
        except BaseException:
            if os.path.exists(path + ".part"):
                os.remove(path + ".part")
            raise
        got[fname], new = size, True
    for old in os.listdir(d):  # e.g. clip_4.mp4 after a remake that kept 3 clips
        if AVATAR_MEDIA_FILE.fullmatch(old) and old not in got:
            os.remove(os.path.join(d, old))
            new = True
    with open(record_path, "w") as f:  # only after every file is here
        json.dump({"updated_at": updated_at, "files": got}, f)
    if new:
        log(f"avatar {name}: media copied to {d}")
    return int(new)


def maybe_sync_avatar_media():
    """From the watcher: sync avatar media at most every LORA_SYNC_EVERY seconds; failures are logged."""
    stamp = os.path.join(STATE_DIR, "avatar_media_sync.json")
    try:
        with open(stamp) as f:
            last = json.load(f).get("at", 0)
    except (OSError, ValueError):
        last = 0
    if time.time() - last < LORA_SYNC_EVERY:
        return
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(stamp, "w") as f:
        json.dump({"at": time.time()}, f)
    try:
        avatar_media_sync()
    except Exception as e:  # noqa: BLE001 - the watcher must keep running whatever the sync hits
        log(f"avatar media sync failed, will retry: {type(e).__name__}: {e}")


def local_lora_versions():
    """{(name, version): bytes on disk} for the complete local LoRA copies (lora.json and .part files not
    counted)."""
    root, out = os.path.join(download_dir(), "loras"), {}
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if not os.path.isdir(os.path.join(root, name)):
            continue
        for vdir in os.listdir(os.path.join(root, name)):
            if not (vdir.startswith("v") and vdir[1:].isdigit()) or not _lora_complete(os.path.join(root, name, vdir)):
                continue
            total = 0
            for dirpath, _, files in os.walk(os.path.join(root, name, vdir)):
                total += sum(os.path.getsize(os.path.join(dirpath, f)) for f in files
                             if f.endswith(".safetensors"))
            out[(name, int(vdir[1:]))] = total
    return out


def lora_command(args):
    if args.action not in ("list", "pull") and not args.name:
        raise ApiError(f"lora {args.action} needs a NAME")
    if args.action == "pull":
        if args.name and args.name.lower() not in {l["name"] for l in api("GET", "/loras")["loras"]}:
            raise ApiError(f"no LoRA named '{args.name}'; see vidgen.py lora list")
        n = lora_sync(names=[args.name.lower()] if args.name else None, checkpoints=args.checkpoints)
        print(f"{n} LoRA version(s) copied to {os.path.join(download_dir(), 'loras')}" if n
              else f"all LoRAs are already in {os.path.join(download_dir(), 'loras')}")
        return
    if args.action == "add":
        if bool(args.file) == bool(args.high):
            raise ApiError("lora add needs --file FILE (one file for both experts) or --high FILE --low FILE")
        if args.high and not args.low:
            raise ApiError("--high needs --low too (or use --file for a single-file LoRA)")
        r = lora_upload(args.name, args.file or args.high, args.low, args.trigger or "", args.description or "",
                        not args.keep_current)
        what = ("Krea 2 LoRA, for Krea photos and first frames" if r.get("model") == "krea2"
                else "one file, both experts" if r["single_file"] else "high + low")
        print(f"added {r['name']} v{r['version']} ({what})"
              + (f", trigger '{r['trigger']}'" if r["trigger"] else "")
              + ("; --lora " + r["name"] + " now uses it" if r["current"] else ""))
        return
    if args.action == "use":
        if args.version is None:
            raise ApiError("lora use needs --version N (and optionally --step S)")
        r = api("PUT", f"/loras/{args.name.lower()}/current", {"version": args.version, "step": args.step})
        what = f"step {r['step']}" if r.get("step") else "final weights"
        print(f"--lora {r['name']} now renders with v{r['version']} {what}")
        return
    if args.action == "preview":
        body = {"type": "lora_preview", "name": args.name.lower(), "version": args.version,
                "prompt": args.prompt or "", "seed": args.seed}
        if args.steps:
            body["steps"] = [int(s) for s in args.steps.split(",") if s.strip()]
        submit(body, args.out)
        return
    found = api("GET", "/loras")["loras"]
    if not found:
        print("no LoRAs yet; train one (vidgen.py train NAME) or add a file (vidgen.py lora add NAME --file F)")
    local = local_lora_versions()
    for l in found:
        for v in l["versions"]:
            current = v["version"] == l["current"]
            mark = "*" if current else " "
            here = "  [on this Mac]" if (l["name"], v["version"]) in local else ""
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(v["created_at"]))
            pick = l.get("current_step") if current else None
            cps = ",".join(f"[{c}]" if c == pick else str(c) for c in v["checkpoints"]) or "-"
            using = f"  using step {pick}" if pick else ("  using final" if current else "")
            if v.get("source") in ("upload", "avatar"):
                kind = "krea2" if "krea2" in (v.get("models") or []) else \
                    ("wan" if v.get("source") == "avatar" else "uploaded")
                note = (f"  avatar {v.get('trigger') or l['name']}" if v.get("source") == "avatar"
                        else ("  " + v["description"] if v.get("description") else ""))
                print(f"{mark} {l['name']:16} v{v['version']:<3} {kind:10} {when}  "
                      f"trigger {v['trigger'] or '-'}{note}{here}")
                continue
            print(f"{mark} {l['name']:16} v{v['version']:<3} {v['kind'] or '':10} {v['steps_per_expert']} steps/expert  "
                  f"checkpoints {cps}  {when}  trigger {v['trigger']}{using}{here}")
    if local:
        print(f"local copies: {len(local)} version(s), {sum(local.values()) / 1024 ** 3:.1f} GB in "
              f"{os.path.join(download_dir(), 'loras')}")


def avatar_flags(parser, what):
    parser.add_argument("--avatar", help=f"star these avatars instead of the default (comma-separated, max 3)")
    parser.add_argument("--no-avatar", action="store_true", help=f"no avatar in this {what}")
    parser.add_argument("--no-sound", action="store_true", help=f"no generated ambient sound in this {what}")
    parser.add_argument("--fps", type=int, choices=[16, 32], default=32,
                        help="32 (default) doubles the frames with RIFE for smoother motion; 16 = as rendered")
    parser.add_argument("--upscale", action="store_true", help="2x Real-ESRGAN upscale (2560x1408, slower, ~3x bigger files)")
    parser.add_argument("--image-model", choices=["auto", "wan", "krea2"], default="auto",
                        help=f"first frame of {what} shots: auto (default) = Krea 2 Turbo when it can be made (no avatar, "
                             "or the avatar's ready Krea LoRA), else Wan; krea2 = always Krea (sharper scenes, ~1-2 min "
                             "more; its content filter can refuse a frame); wan = as rendered")
    parser.add_argument("--draft", action="store_true",
                        help="faster draft: render at 832x480, then 32 fps + 2x upscale to 1664x960 (softer detail)")
    parser.add_argument("--lora", action="append", default=[], metavar="NAME[@[vN][-]STEP][:WEIGHT]",
                        help="apply a trained LoRA (weight 0-1.5, default 1.0), e.g. jerk:0.9, jerk@750, "
                             "jerk@v1-750:0.8; repeat for a second one")


LOOKS = ("pro", "phone", "candid", "portrait", "film")
SHOTS = ("stream", "pov")


def shot_flag(parser):
    """--shot on video: how the clip is filmed and lit (one or none)."""
    parser.add_argument("--shot", choices=SHOTS,
                        help="stream: a streaming session (fixed webcam, ring light on the face); "
                             "pov: looks shot on a handheld phone (shaky, soft low resolution, grainy, the phone's "
                             "light; the phone and the person filming are not in the picture). Adds wording to "
                             "the prompt; pov also shakes, softens and grains the finished frames")


VIDEO_MODELS = ("helios", "wan")
HELIOS_SECONDS, WAN_SECONDS = 30, 5  # the server's default lengths


def video_model_flag(parser):
    """--video-model on video: helios (the server's default) or wan."""
    parser.add_argument("--video-model", choices=VIDEO_MODELS,
                        help="helios (default): the whole clip in one take from one prompt, 640x384 at 24 fps, "
                             "30 s unless --seconds says otherwise. wan: 5 s segments chained, 1280x704, "
                             "with --beats, plans and --anchor-every")


def video_model_body(args, planning=False):
    """What the job says about the model and length: only what was asked for (the server has the defaults).
    A plan writes beats, which are wan's."""
    model = "wan" if planning else getattr(args, "video_model", None)
    seconds = getattr(args, "seconds", None)
    return {**({"video_model": model} if model else {}), **({"seconds": seconds} if seconds is not None else {})}


def video_seconds(args, planning=False):
    """The length this job will have: --seconds, else the model's default."""
    if getattr(args, "seconds", None) is not None:
        return args.seconds
    return WAN_SECONDS if planning or getattr(args, "video_model", None) == "wan" else HELIOS_SECONDS


def shot_body(args):
    return {"shot": args.shot} if getattr(args, "shot", None) else {}


def realism_flags(parser, what):
    """--look, --natural-skin, --candid, --grain, --best on video, story and photos."""
    parser.add_argument("--look", choices=LOOKS, default="pro",
                        help="pro (default, clean 50mm look), phone (casual smartphone), candid (35mm documentary), "
                             "portrait (85mm, soft background), film (35mm film colours and grain)")
    parser.add_argument("--natural-skin", action="store_true", help="visible skin texture, no retouched look")
    parser.add_argument("--candid", action="store_true", help="unposed; people don't look at the camera")
    parser.add_argument("--grain", action="store_true", help="add real film grain to the finished result")
    best = ("full 40-step image model (about 3x slower per photo), so the anti-plastic-skin settings apply"
            if what == "photos" else
            "make avatar keyframes with the full 40-step image model (about +1.5 min per keyframe); "
            "the video model is unchanged")
    parser.add_argument("--best", action="store_true", help=f"best quality: {best}")
    parser.add_argument("--realism-check", action="store_true",
                        help="after the render, have the vision model rate how AI-made it looks and suggest a prompt "
                             "and settings that look more real (about 1-2 min more; a critique, not a detector)")


def photos_body(args):
    """The photos job from the command line: a series, or scenes (fields, prompts, shots file)."""
    extra = {"aspect": args.aspect, "seed": args.seed, "quality": "best" if args.best else "standard",
             **realism_body(args)}
    if args.no_avatar or args.model == "krea2":  # Krea 2: no person, or an avatar through its Krea 2 LoRA
        avatar = None if args.no_avatar else getattr(args, "avatar", None)
        extra.update(model="krea2", avatar=avatar or "none")
    if getattr(args, "raw", False):  # Krea 2 gets the prompt exactly as typed
        if extra.get("model") != "krea2":
            raise ApiError("--raw is for Krea 2 photos; add --model krea2 (photos without a person)")
        extra["raw"] = True
    if getattr(args, "lora", None):  # Krea 2 LoRAs; the qwen edit model takes none
        if extra.get("model") != "krea2":
            raise ApiError("--lora on photos is for Krea 2 LoRAs; add --model krea2 (photos without a person)")
        extra["loras"] = parse_loras(args.lora)
    if args.series:
        if not args.where:
            raise ApiError("--series needs --where (the place)")
        if args.scenes or args.prompts_file or args.shots or args.camera:
            raise ApiError("--series takes --where, --wearing and --doing only (the camera is varied for you)")
        series = {"where": args.where, "anchor": not args.no_anchor}
        if args.wearing:
            series["wearing"] = args.wearing
        if args.doing:
            series["doing"] = args.doing
        return {"type": "photos", "scenes": [], "series": series, "count": args.count or max(4, len(args.doing)),
                **extra}
    if len(args.doing) > 1:
        raise ApiError("--doing more than once needs --series (one action per photo)")
    scenes = args.scenes + [{"prompt": t} for t in (read_prompts_file(args.prompts_file) if args.prompts_file else [])]
    shot = {k: v for k, v in (("where", args.where), ("wearing", args.wearing),
                              ("doing", args.doing[0] if args.doing else None), ("camera", args.camera)) if v}
    if shot:
        scenes = [{"shot": shot}] + scenes
    if args.shots:
        scenes += read_shots_file(args.shots)
    if len(scenes) > 8:
        raise ApiError(f"{len(scenes)} photos requested; at most 8 per job")
    return {"type": "photos", "scenes": scenes, "count": args.count, **extra}


def realism_body(args):
    body = {"look": args.look, "natural_skin": args.natural_skin, "candid": args.candid, "grain": args.grain}
    if getattr(args, "realism_check", False):
        body["realism_check"] = True
    return body


def save_realism(local, remote):
    """The AI-look verdict of a downloaded job: logged, and saved as realism.json next to its files (in a photos
    folder; <base>.realism.json beside a video). docs/specs/2026-11-27-realism-check.md"""
    verdict = (remote.get("result") or {}).get("realism")
    if not isinstance(verdict, dict):
        return None
    out = local["out"]
    path = os.path.join(out, "realism.json") if os.path.isdir(out) else os.path.splitext(out)[0] + ".realism.json"
    with open(path, "w") as f:
        json.dump(verdict, f, indent=2)
    if isinstance(verdict.get("score"), int):
        signs = "; ".join(str(s) for s in verdict.get("signs") or [])
        log(f"AI look {verdict['score']}/10{': ' + signs if signs else ''} (suggestion in {os.path.basename(path)})")
    else:
        log(f"AI-look check: {verdict.get('error', 'no verdict')}")
    return path


def video_quality(args):
    if args.best and args.draft:
        raise ApiError("give --draft or --best, not both")
    return "best" if args.best else "draft" if args.draft else "standard"


LORA_SPEC = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_-]*)(?:@(?:v(?P<version>\d+))?(?:-?(?P<step>\d+))?)?(?::(?P<weight>[^:]+))?$")


BEAT_LINE = re.compile(r"\s*(\d{1,3})\s*:\s*(.*\S)\s*$")


def _beat(text, where):
    m = BEAT_LINE.match(text)
    if not m:
        raise ApiError(f"{where}expected 'SECONDS: text' (whole seconds), e.g. '10: sits down', got: {text.strip()[:60]}")
    return {"at": int(m.group(1)), "prompt": m.group(2)}


def parse_beats(text):
    """--beats "0: walks in; 10: sits down" -> [{"at": 0, "prompt": "walks in"}, ...] (the API checks the rules)."""
    return [_beat(chunk, "beats: ") for chunk in re.split(r"[;\n]", text) if chunk.strip()]


def read_beats_file(path):
    """--beats-file: one 'SECONDS: text' per line (the text may contain ';'), '#' comments, blank lines ignored."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except (OSError, UnicodeDecodeError) as e:
        raise ApiError(f"cannot read {path}: {e}")
    beats = [_beat(line, f"{path}:{n}: ") for n, line in enumerate(lines, 1)
             if line.strip() and not line.strip().startswith("#")]
    if not beats:
        raise ApiError(f"{path}: no beats found")
    return beats


def video_beats(args):
    if args.beats and args.beats_file:
        raise ApiError("give --beats or --beats-file, not both")
    if args.beats_file:
        return read_beats_file(args.beats_file)
    return parse_beats(args.beats) if args.beats else None


SEGMENT_MINUTES = 5  # render time of one 5 s segment on the render box, roughly


def take_estimate(seconds):
    """Rough render time of a long single take: ~5 min per 5 s segment, plus smoothing and sound."""
    segments = -(-int(seconds) // 5)
    return (f"a {seconds} s take is {segments} segments of 5 s: at least {segments * SEGMENT_MINUTES + 5} min "
            f"to render (~{SEGMENT_MINUTES} min per segment plus smoothing and sound; more with --best or --upscale)")


def parse_loras(values):
    """--lora NAME[@[vN][-]STEP][:WEIGHT]: jerk, jerk:0.9, jerk@750, jerk@v1, jerk@v1-750:0.8.
    Without @ the job uses the pick set with `lora use`; the API checks names and ranges."""
    out = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            m = LORA_SPEC.match(part)
            if not m or part.endswith("@"):
                raise ApiError(f"--lora {part}: expected NAME[@[vN][-]STEP][:WEIGHT], e.g. jerk@v1-750:0.8")
            try:
                item = {"name": m["name"].lower(), "weight": float(m["weight"]) if m["weight"] else 1.0}
            except ValueError:
                raise ApiError(f"--lora {part}: weight must be a number")
            if m["version"]:
                item["version"] = int(m["version"])
            if m["step"]:
                item["step"] = int(m["step"])
            out.append(item)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("configure", help="save API URL, key and download folder to ~/.vidgen/config.json")
    c.add_argument("--url")
    c.add_argument("--ssh-key", help="public key path to sign via ssh-agent (or private key path)")
    c.add_argument("--download-dir")

    sub.add_parser("status", help="GPU instance state, queue depth, uptime hold")
    sub.add_parser("wake", help="start the GPU instance ahead of a batch")
    sub.add_parser("restart-worker", help="ask the render worker to restart between jobs (it pulls the latest "
                                          "worker code); the render machine must be running")
    jb = sub.add_parser("jobs", help="list local jobs and where their files went; `jobs forget ID...` drops jobs "
                                     "(also ones from the phone; a job not downloaded yet needs its full id) so the watcher doesn't download them")
    jb.add_argument("action", nargs="?", choices=["forget"])
    jb.add_argument("ids", nargs="*")
    ch = sub.add_parser("check", help="one pass: download any finished jobs (what the watcher runs)")
    ch.add_argument("--log", help=argparse.SUPPRESS)

    w = sub.add_parser("watch", help="check in the foreground until you press Ctrl-C")
    w.add_argument("--interval", type=int, default=CHECK_INTERVAL)
    w.add_argument("--until-done", action="store_true", help="exit once nothing is pending")
    w.add_argument("--log", help=argparse.SUPPRESS)

    wt = sub.add_parser("watcher", help="manage the background watcher (macOS, Linux, Windows)")
    wt.add_argument("action", choices=["install", "uninstall", "status"])
    wt.add_argument("--interval", type=int, default=CHECK_INTERVAL)

    k = sub.add_parser("cancel", help="cancel a queued or running job")
    k.add_argument("job_id", help="job id or unique prefix")

    v = sub.add_parser("video", aliases=["plan"],
                       help="text to video; `plan` (same options, takes over 15 s): first frame + editable beats, no video")
    v.add_argument("prompt")
    v.add_argument("--negative", default="")
    v.add_argument("--seconds", type=int, help="clip length, 1-60 s (default 30 with helios, 5 with wan; "
                                                "a wan take over 5 s chains 5 s segments, ~5 min each)")
    v.add_argument("--variations", type=int, default=1, help="2-4 takes of the same prompt (seeds N, N+1, ...); up to 5 s")
    v.add_argument("--width", type=int, default=1280)
    v.add_argument("--height", type=int, default=704)
    v.add_argument("--seed", type=int)
    v.add_argument("--beats", help='long takes: what happens when, e.g. "0: a man walks into a sunny kitchen; '
                                   '10: he sits down at the table" - each beat a full description, times in '
                                   "multiples of 5 (segments start every 5 s)")
    v.add_argument("--beats-file", help="the beats from a file, one 'SECONDS: text' per line")
    v.add_argument("--no-avatar-lora", action="store_true",
                   help="don't add the avatar's own Wan LoRA (added by default when it has a ready one)")
    v.add_argument("--anchor-every", type=int, metavar="N",
                   help="takes over 10 s with one avatar: restore the avatar's face every N segments (2-6)")
    v.add_argument("-o", "--out", help="file name inside the download folder (default <job_id>.mp4)")
    avatar_flags(v, "video")
    realism_flags(v, "video")
    shot_flag(v)
    video_model_flag(v)

    d = sub.add_parser("describe", help="title/description/tags for a video file")
    d.add_argument("file")
    d.add_argument("--instructions", default="")
    d.add_argument("-o", "--out", help="file name inside the download folder (default <job_id>.json)")

    w = sub.add_parser("write", help="write a short story from a premise (text only); film it with `story`",
                       description="Uses the story planner's model on the render machine (no video). The story "
                                   "downloads as a .txt you can edit, then: vidgen story FILE")
    w.add_argument("premise", help="what the story is about, or - to read it from stdin")
    w.add_argument("--words", type=int, default=800, help="about how long, 100-2000 (default 800)")
    w.add_argument("--genre", default="", help="genre and tone, e.g. 'cozy mystery', 'bittersweet, for kids'")
    w.add_argument("--seed", type=int, help="same premise + seed = the same story")
    w.add_argument("-o", "--out", help="file name inside the download folder (default <job_id>.txt)")

    s = sub.add_parser("story", help="text story to narrated multi-shot video")
    s.add_argument("file", help="text file with the story, or - for stdin")
    s.add_argument("--shots", type=int, default=6)
    s.add_argument("--seconds-per-shot", type=int, default=5, help="2-5 s")
    s.add_argument("--style", default="")
    s.add_argument("--no-subtitles", action="store_true")
    s.add_argument("--no-narration", action="store_true", help="no spoken narration")
    s.add_argument("--voice", default="af_heart", help="Kokoro voice, e.g. af_heart, am_adam, bf_emma, bm_george")
    s.add_argument("-o", "--out", help="file name inside the download folder (default <job_id>.mp4)")
    avatar_flags(s, "story")
    realism_flags(s, "story")

    ph = sub.add_parser("photos", help="photorealistic photos of one person in several places",
                        description="Each --prompt / --location is one photo, in the order given; --count tops up "
                                    "with random locations. Max 8 photos per job.")
    ph.add_argument("--image", help="photo of the person (png/jpg/webp); default: your default avatar")
    ph.add_argument("--avatar", help="use this saved avatar instead of --image")
    ph.add_argument("--prompt", dest="scenes", action="append", type=lambda t: {"prompt": t}, default=[], metavar="TEXT",
                    help="free-form scene for one photo: place, action, mood, lighting (repeatable)")
    ph.add_argument("--location", dest="scenes", action="append", type=lambda t: {"location": t}, metavar="PLACE",
                    help="just a place for one photo (repeatable)")
    ph.add_argument("--where", help="one photo built from fields: the place (e.g. 'on a rainy Tokyo rooftop at dusk')")
    ph.add_argument("--wearing", help="what they wear in that photo (replaces the avatar's outfit)")
    ph.add_argument("--doing", action="append", default=[],
                    help="their action or pose (e.g. 'leaning on the railing, laughing'); with --series, "
                         "repeat it for one action per photo")
    ph.add_argument("--camera", help="camera position and framing (e.g. 'low angle, wide lens', 'from behind')")
    ph.add_argument("--shots", help="file with one shot per block: where:/wearing:/doing:/camera: lines, "
                                    "blank line between photos")
    ph.add_argument("--prompts-file", help="one prompt per line (blank lines and # comments skipped)")
    ph.add_argument("--count", type=int, help="total photos, 1-8; extras are random locations "
                                               "(default: number of prompts/locations, or 4)")
    ph.add_argument("--no-avatar", action="store_true",
                    help="photos without a person (scenes, places, objects), made by Krea 2 Turbo from text")
    ph.add_argument("--model", choices=["qwen", "krea2"], default="qwen",
                    help="qwen (default): places the avatar/photo; krea2: text-to-image without a person "
                         "(same as --no-avatar); its content filter drops flagged images")
    ph.add_argument("--series", action="store_true",
                    help="every photo in the --where place (and --wearing outfit), pose and camera varied; "
                         "--count 2-8 (default 4)")
    ph.add_argument("--no-anchor", action="store_true",
                    help="series: don't give photo 1 to the later photos as a reference (keeps more likeness, "
                         "the place drifts more)")
    ph.add_argument("--aspect", choices=["portrait", "landscape", "square"], default="portrait")
    ph.add_argument("--seed", type=int)
    ph.add_argument("--lora", action="append", default=[], metavar="NAME[:WEIGHT]",
                    help="Krea 2 LoRA for --model krea2 photos (repeatable, max 2; vidgen lora list)")
    ph.add_argument("--raw", action="store_true",
                    help="with --model krea2: send each --prompt to Krea 2 exactly as typed (no look/realism sentences, no LoRA trigger words; extra --count photos repeat your prompts, no random places)")
    ph.add_argument("-o", "--out", help="folder name inside the download folder (default photos-<job_id>)")
    realism_flags(ph, "photos")

    a = sub.add_parser("avatar", help="the characters who star in your videos",
                       description="list | refresh | add FILE... | images | remove <n> | main <n> | "
                                   "pick <job_id> <n> | use --name N | history | restore <version> | "
                                   "krea-train --name N | media --name N (photos + 2 s clips) | "
                                   "wan-train --name N (Wan LoRA from its media) | "
                                   "delete --name N [--with-loras] (no undo). "
                                   "An avatar holds up to 12 images of the same person; photos and "
                                   "keyframes use the first 3 (best: full-length, face close-up, another angle).")
    a.add_argument("action", choices=["list", "refresh", "add", "images", "remove", "main", "pick", "use",
                                      "history", "restore", "krea-train", "media", "wan-train", "delete"])
    a.add_argument("--with-loras", action="store_true",
                   help="delete: also delete the avatar's own LoRAs and their training photos")
    a.add_argument("--yes", action="store_true", help="delete: don't ask")
    a.add_argument("--no-krea-lora", action="store_true",
                   help="refresh/pick: don't build this avatar's Krea 2 LoRA (it is built by default, ~1.5 h GPU)")
    a.add_argument("args", nargs="*", help="add: image files; remove/main: image number; "
                                           "pick: <job_id> <n>; restore: <version>")
    a.add_argument("--name", help=f"avatar name (default: current default, or '{DEFAULT_AVATAR_NAME}')")
    a.add_argument("--image", action="append",
                   help="refresh: use this image instead of asking (repeat for several images of the same person)")
    a.add_argument("--describe", help="refresh: description (with --image) or generate candidates from it")
    a.add_argument("--candidates", type=int, default=4, help="refresh: how many to generate (1-6)")
    a.add_argument("--model", choices=["wan", "krea2"], default="wan",
                   help="refresh: model for generated candidates: wan (one video frame, default) or krea2 "
                        "(Krea 2 Turbo image model, sharper; images flagged by its content filter are dropped)")

    tr = sub.add_parser("train", help="train a LoRA from an uploaded dataset on the trainer machine")
    tr.add_argument("name", help="dataset name (upload it first with `dataset add`)")
    tr.add_argument("--steps", type=int, default=1250, help="steps per expert, 100-3000 (1250 is ~12 h in total at standard speed)")
    tr.add_argument("--speed", choices=["standard", "fast"], default="standard",
                    help="fast trains at 512x288 instead of 832x480: several times faster, less fine detail")
    tr.add_argument("--captions-only", action="store_true",
                    help="only cut and caption the dataset (minutes), then review with `dataset captions NAME --edit`")

    lo = sub.add_parser("lora", help="LoRAs: list | use | preview | add | pull (copy them to this computer)",
                        description="list: versions and checkpoints (* = what --lora NAME renders with). "
                                    "use NAME --version N [--step S]: pick a version or checkpoint. "
                                    "preview NAME: short clips of each checkpoint with the same seed. "
                                    "add NAME --file F | --high F --low F: upload a Wan 14B LoRA made elsewhere "
                                    "as a new version (checked before it can be used).")
    lo.add_argument("action", choices=["list", "use", "preview", "add", "pull"])
    lo.add_argument("--checkpoints", action="store_true", help="pull: also copy the checkpoints")
    lo.add_argument("name", nargs="?")
    lo.add_argument("--version", type=int)
    lo.add_argument("--step", type=int, help="use: a checkpoint step (default: final weights)")
    lo.add_argument("--steps", help="preview: comma-separated checkpoint steps (default: up to the last 5)")
    lo.add_argument("--prompt", help="preview: the scene (the trigger word is added)")
    lo.add_argument("--seed", type=int)
    lo.add_argument("-o", "--out", help="preview: folder name inside the download folder")
    lo.add_argument("--file", help="add: one .safetensors LoRA for both Wan 2.2 experts (e.g. a Wan 2.1/2.2 14B LoRA)")
    lo.add_argument("--high", help="add: the high-noise expert file of a Wan 2.2 pair")
    lo.add_argument("--low", help="add: the low-noise expert file of a Wan 2.2 pair")
    lo.add_argument("--trigger", help="add: the LoRA's trigger word, if it has one (added to prompts)")
    lo.add_argument("--description", help="add: a note shown in lists")
    lo.add_argument("--keep-current", action="store_true", help="add: don't switch --lora NAME to the new version")

    ds = sub.add_parser("dataset", help="learning datasets: upload a folder of clips/photos for the trainer",
                        description="add NAME FOLDER --kind K | list | rm NAME | captions NAME [--edit]")
    ds.add_argument("action", choices=["add", "list", "rm", "captions"])
    ds.add_argument("name", nargs="?")
    ds.add_argument("--edit", action="store_true", help="captions: edit them in $EDITOR")
    ds.add_argument("folder", nargs="?")
    ds.add_argument("--kind", choices=["character", "style", "motion", "place"])
    ds.add_argument("--trigger", help="trigger word (default vgn_<name>)")
    ds.add_argument("--yes", action="store_true", help="don't ask (deletions, consent)")

    u = sub.add_parser("ui", help="local dashboard: server status, models, job history, learning themes")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--no-browser", action="store_true", help="don't open the browser")

    sm = sub.add_parser("smoke", help="end-to-end test of every model on the GPU (~30-60 min)")
    sm.add_argument("--skip-cancel", action="store_true", help="skip the cancel test")

    args = p.parse_args()
    if getattr(args, "log", None):
        redirect_output(args.log)

    try:
        if args.cmd == "configure":
            config = save_config(url=args.url, ssh_key=args.ssh_key, download_dir=args.download_dir)
            print(json.dumps(config, indent=2))
        elif args.cmd == "status":
            print(json.dumps(api("GET", "/status"), indent=2))
        elif args.cmd == "wake":
            print(json.dumps(api("POST", "/wake"), indent=2))
        elif args.cmd == "restart-worker":
            print(api("POST", "/worker/restart")["note"])
        elif args.cmd == "jobs":
            if args.action == "forget":
                if not args.ids:
                    raise ApiError("usage: vidgen.py jobs forget ID...")
                forget_jobs(args.ids)
            else:
                show_jobs()
        elif args.cmd == "check":
            check_once()
        elif args.cmd == "watch":
            os.makedirs(STATE_DIR, exist_ok=True)
            with open(PID_FILE, "w") as f:
                f.write(str(os.getpid()))
            watch(args.interval, args.until_done)
        elif args.cmd == "watcher":
            {"install": lambda: watcher_install(args.interval),
             "uninstall": watcher_uninstall,
             "status": watcher_status}[args.action]()
        elif args.cmd == "cancel":
            cancel(args.job_id)
        elif args.cmd == "avatar":
            avatar_command(args)
        elif args.cmd == "train":
            body = {"type": "train", "dataset": args.name, "steps": args.steps, "speed": args.speed,
                    "captions_only": args.captions_only}
            if args.captions_only:
                print(f"captioning {args.name} on the trainer (no training); then: vidgen.py dataset captions {args.name} --edit")
            else:
                hours = args.steps * 2 * SECONDS_PER_STEP[args.speed] / 3600
                print(f"training {args.name} ({args.speed}): {args.steps} steps per expert x 2 experts, about {hours:.1f} h "
                      f"(~${hours * 2:.0f} at spot prices); checkpoint previews are queued automatically at the end")
            submit(body, f"train-{args.name}")
        elif args.cmd == "lora":
            lora_command(args)
        elif args.cmd == "dataset":
            if args.action != "list" and not args.name:
                raise ApiError(f"dataset {args.action} needs a NAME")
            dataset_command(args)
        elif args.cmd == "ui":
            import ui
            ui.serve(args.port, open_browser=not args.no_browser)
        elif args.cmd == "smoke":
            import smoke
            sys.exit(smoke.run(skip_cancel=args.skip_cancel))
        elif args.cmd in ("video", "plan"):
            beats = video_beats(args)
            planning = args.cmd == "plan"
            seconds = video_seconds(args, planning)
            wan = planning or args.video_model == "wan"
            if planning and (seconds <= 15 or beats or args.variations > 1):
                raise ApiError("plan: a take over 15 s, without --beats or --variations (it writes the beats)")
            if wan and seconds > 15 and not planning:
                print(take_estimate(seconds), flush=True)
            if ensure_avatar(args):
                body = {
                    "type": "video", "prompt": args.prompt, "negative_prompt": args.negative,
                    **video_model_body(args, planning), "width": args.width, "height": args.height, "seed": args.seed,
                    "variations": args.variations,
                    "avatar": "none" if args.no_avatar else args.avatar, "sound": not args.no_sound,
                    "loras": parse_loras(args.lora), "fps": args.fps, "upscale": args.upscale,
                    "quality": video_quality(args), "image_model": None if args.image_model == "auto" else args.image_model, **realism_body(args),
                    **shot_body(args),
                    **({"beats": beats} if beats else {}),
                    **({"anchor_every": args.anchor_every} if args.anchor_every else {}),
                    **({"avatar_lora": False} if args.no_avatar_lora else {}),
                    **({"plan_only": True} if planning else {}),
                }
                job_id = submit(body, args.out)
                if planning:
                    wait_for_plan(job_id, args)
        elif args.cmd == "photos":
            if args.image and args.avatar:
                raise ApiError("give --image or --avatar, not both")
            body = photos_body(args)
            if body.get("model") == "krea2":
                if args.image:
                    raise ApiError("Krea 2 draws from text; drop --image (or use the default model), or give "
                                   "--avatar NAME for an avatar with a Krea 2 LoRA")
            elif args.image:
                body["upload_key"] = upload_file(args.image, {"png", "jpg", "jpeg", "webp"})
            elif args.avatar:
                body["avatar"] = args.avatar
            elif not api("GET", "/avatars")["default"]:
                raise ApiError("no default avatar; give --image FILE or --avatar NAME")
            submit(body, args.out)
        elif args.cmd == "write":
            premise = sys.stdin.read() if args.premise == "-" else args.premise
            if not 100 <= args.words <= 2000:
                raise ApiError("--words must be 100-2000")
            submit({"type": "write", "premise": premise.strip(), "words": args.words, "genre": args.genre,
                    "seed": args.seed}, args.out)
        elif args.cmd == "describe":
            key = upload_file(args.file)
            submit({"type": "describe", "upload_key": key, "instructions": args.instructions}, args.out)
        elif args.cmd == "story":
            # Read the story first: stdin may be the story itself, not a terminal.
            story = sys.stdin.read() if args.file == "-" else open(args.file).read()
            if ensure_avatar(args):
                submit({
                    "type": "story", "story": story, "max_shots": args.shots,
                    "seconds_per_shot": args.seconds_per_shot, "style": args.style,
                    "subtitles": not args.no_subtitles, "narration": not args.no_narration,
                    "sound": not args.no_sound, "voice": args.voice, "avatar": "none" if args.no_avatar else args.avatar,
                    "loras": parse_loras(args.lora), "fps": args.fps, "upscale": args.upscale,
                    "quality": video_quality(args), "image_model": None if args.image_model == "auto" else args.image_model, **realism_body(args),
                }, args.out)
    except ApiError as e:
        sys.exit(str(e))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
