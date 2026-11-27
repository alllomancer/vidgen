"""Local dashboard for the AI video pipeline: server status and model versions, job
history with run times (and re-run), and learning themes (folders of clips for the
upcoming trainer). Standard library only, like vidgen.py.

    vidgen.py ui [--port 8765] [--no-browser]

The page is served on 127.0.0.1 only. API calls go through vidgen.api(), so they are
signed with your SSH key exactly like the CLI. Requests must carry the right Host header
(against DNS rebinding) and /api/* calls the per-session token embedded in the page
(against other sites in your browser).
"""

import base64
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import datasets
import vidgen

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
THEMES_FILE = os.path.join(vidgen.STATE_DIR, "themes.json")
THEME_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")  # same rule as dataset names in the API
CLIP_EXTS = {"mp4", "mov", "webm", "mkv"}
PHOTO_EXTS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_BYTES = 2 * 1024 ** 3
LONG_CLIP_SECONDS = 60
# Clips needed per theme kind (docs/specs/2026-11-09-lora-training.md).
KINDS = {
    "character": {"min": 10, "max": 40, "photos": True, "hint": "varied angles, light, clothes; the person clearly visible"},
    "style": {"min": 30, "max": 100, "photos": False, "hint": "many different subjects, all in the same look"},
    "motion": {"min": 20, "max": 60, "photos": False, "hint": "different people/places doing the same motion"},
    "place": {"min": 15, "max": 40, "photos": True, "hint": "different times of day and angles"},
}


# ---------------------------------------------------------------- themes

def load_themes():
    try:
        with open(THEMES_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_themes(themes):
    os.makedirs(vidgen.STATE_DIR, exist_ok=True)
    tmp = THEMES_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(themes, f, indent=2)
    os.replace(tmp, THEMES_FILE)


def _duration(path):
    if not vidgen.shutil.which("ffprobe"):
        return None
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, timeout=30)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def scan_folder(folder, kind):
    """Counts and problems for a learning folder, checked against the kind's guidance."""
    rule = KINDS[kind]
    clips, photos, other, too_big, long_clips = [], [], 0, [], []
    seconds = 0.0
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lstrip(".").lower()
            if ext in CLIP_EXTS:
                clips.append(path)
                d = _duration(path)
                if d:
                    seconds += d
                    if d > LONG_CLIP_SECONDS:
                        long_clips.append(os.path.relpath(path, folder))
            elif ext in PHOTO_EXTS:
                photos.append(path)
            else:
                other += 1
                continue
            if os.path.getsize(path) > MAX_FILE_BYTES:
                too_big.append(os.path.relpath(path, folder))

    problems, warnings = [], []
    n = len(clips)
    if n < rule["min"]:
        problems.append(f"{n} clips; {kind} needs {rule['min']}-{rule['max']}")
    elif n > rule["max"]:
        warnings.append(f"{n} clips; more than {rule['max']} adds training time with little gain")
    if too_big:
        problems.append(f"{len(too_big)} file(s) over 2 GB: {', '.join(too_big[:3])}")
    if long_clips:
        warnings.append(f"{len(long_clips)} clip(s) over {LONG_CLIP_SECONDS} s will be cut into pieces")
    if photos and not rule["photos"]:
        warnings.append(f"{len(photos)} photo(s) are ignored for {kind} themes")
    if other:
        warnings.append(f"{other} unsupported file(s) skipped")
    return {"clips": n, "photos": len(photos), "minutes": round(seconds / 60, 1) if seconds else None,
            "problems": problems, "warnings": warnings, "ready": not problems,
            "fingerprint": datasets.fingerprint(datasets.local_files(folder)[0]),
            "scanned_at": int(time.time())}


def add_theme(body):
    name = (body.get("name") or "").strip().lower()
    kind = body.get("kind")
    folder = os.path.abspath(os.path.expanduser((body.get("folder") or "").strip()))
    if not THEME_NAME.match(name):
        raise ValueError("name: 1-32 characters, a-z 0-9 - _ (starting with a letter or digit)")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of: {', '.join(KINDS)}")
    if not os.path.isdir(folder):
        raise ValueError(f"folder not found: {folder}")
    themes = load_themes()
    if name in themes:
        raise ValueError(f"a theme named '{name}' already exists")
    trigger = (body.get("trigger") or "").strip() or f"vgn_{name}"
    themes[name] = {"name": name, "kind": kind, "folder": folder, "trigger": trigger,
                    "created_at": int(time.time()), "scan": scan_folder(folder, kind)}
    save_themes(themes)
    return themes[name]


def rescan_theme(name):
    themes = load_themes()
    if name not in themes:
        raise KeyError(name)
    t = themes[name]
    t["scan"] = scan_folder(t["folder"], t["kind"]) if os.path.isdir(t["folder"]) else {
        "clips": 0, "photos": 0, "minutes": None, "problems": [f"folder not found: {t['folder']}"],
        "warnings": [], "ready": False, "scanned_at": int(time.time())}
    save_themes(themes)
    return t


def remove_theme(name, remote=False):
    """Forget a theme; with remote, also delete its uploaded dataset (all files) from S3."""
    themes = load_themes()
    if name not in themes:
        raise KeyError(name)
    if remote:
        try:
            vidgen.api("DELETE", f"/datasets/{name}")
        except vidgen.ApiError as e:
            if e.status != 404:
                raise
    themes.pop(name)
    save_themes(themes)


def upsert_theme(name, kind, folder, trigger=None):
    """Register a theme, or update its kind/folder/trigger (used by `vidgen dataset add`)."""
    themes = load_themes()
    if name in themes:
        t = themes[name]
        t.update(kind=kind, folder=os.path.abspath(folder), trigger=trigger or t["trigger"])
        t["scan"] = scan_folder(t["folder"], kind)
        save_themes(themes)
        return t
    return add_theme({"name": name, "kind": kind, "folder": folder, "trigger": trigger})


def record_upload(name, result):
    themes = load_themes()
    if name in themes:
        themes[name]["upload"] = {"at": result["updated_at"], "files": result["file_count"],
                                  "bytes": result["total_bytes"], "uploaded": result["uploaded"],
                                  "removed": result.get("removed", 0), "fingerprint": result["fingerprint"]}
        save_themes(themes)


# Uploads running in this process: {theme name: progress dict}.
UPLOADS = {}
UPLOADS_LOCK = threading.Lock()


def theme_view(t):
    """A theme plus upload state for the page."""
    t = dict(t)
    up = t.get("upload")
    t["changed_since_upload"] = bool(up) and up.get("fingerprint") != (t.get("scan") or {}).get("fingerprint")
    with UPLOADS_LOCK:
        if t["name"] in UPLOADS:
            t["upload_progress"] = dict(UPLOADS[t["name"]])
    return t


def upload_plan(name):
    t = load_themes().get(name)
    if not t:
        raise KeyError(name)
    p = datasets.plan(name, t["folder"])
    return {"upload_files": len(p["upload"]), "upload_bytes": p["upload_bytes"], "total_files": len(p["files"]),
            "remove": p["remove"][:50], "remove_count": len(p["remove"]),
            "needs_consent": t["kind"] == "character" and not p["has_consent"],
            "too_big": p["skipped"]["too_big"][:10]}


def start_upload(name, consent=False, allow_remove=False):
    t = load_themes().get(name)
    if not t:
        raise KeyError(name)
    with UPLOADS_LOCK:
        if UPLOADS.get(name, {}).get("state") == "uploading":
            raise ValueError(f"{name} is already uploading")
        UPLOADS[name] = {"state": "uploading", "files_done": 0, "files_total": 0, "bytes_done": 0, "bytes_total": 0}

    def progress(p):
        with UPLOADS_LOCK:
            UPLOADS[name].update(p)

    def run():
        try:
            result = datasets.sync(name, t["folder"], t["kind"], t["trigger"], consent=consent,
                                   allow_remove=allow_remove, progress=progress)
            record_upload(name, result)
            rescan_theme(name)
            with UPLOADS_LOCK:
                UPLOADS.pop(name, None)
        except (vidgen.ApiError, datasets.NeedsConfirmation, OSError) as e:
            with UPLOADS_LOCK:
                UPLOADS[name].update(state="failed", error=str(e))

    threading.Thread(target=run, daemon=True).start()
    return {"started": True}


def pick_folder():
    """Native folder picker (macOS). None if cancelled."""
    if sys.platform != "darwin":
        raise NotImplementedError("folder picker is macOS only; type the path instead")
    out = subprocess.run(["osascript", "-e", 'POSIX path of (choose folder with prompt '
                          '"Choose a folder of clips for learning")'], capture_output=True, text=True)
    return out.stdout.strip().rstrip("/") or None if out.returncode == 0 else None


def pick_file():
    """Native file picker for a .safetensors LoRA (macOS). None if cancelled."""
    if sys.platform != "darwin":
        raise NotImplementedError("file picker is macOS only; type the path instead")
    out = subprocess.run(["osascript", "-e", 'POSIX path of (choose file with prompt '
                          '"Choose a .safetensors LoRA file")'], capture_output=True, text=True)
    return out.stdout.strip() or None if out.returncode == 0 else None


LORA_NAME = THEME_NAME


def add_lora(body):
    """Learning > Add a LoRA file: upload a local .safetensors (or a high/low pair) as a new version."""
    name = str(body.get("name") or "").strip().lower()
    if not LORA_NAME.match(name):
        raise ValueError("name must be 1-32 chars: a-z, 0-9, - or _ (starting with a letter or digit)")
    high = str(body.get("file") or "").strip()
    low = str(body.get("low") or "").strip() or None
    if not high:
        raise ValueError("choose a .safetensors file")
    for path in [high] + ([low] if low else []):
        if not os.path.isabs(os.path.expanduser(path)):
            raise ValueError(f"{path}: give the full path")
    return vidgen.lora_upload(name, high, low, str(body.get("trigger") or "").strip(),
                              str(body.get("description") or "").strip(), body.get("make_current", True) is not False)


# ---------------------------------------------------------------- jobs

def job_history():
    remote = vidgen.api("GET", "/jobs")["jobs"]
    local = vidgen.load_jobs()
    for job in remote:
        mine = local.get(job["job_id"])
        if mine:
            job["local"] = {"out": mine["out"], "status": mine["status"], "files": mine.get("files", []),
                            "avatar_name": mine.get("avatar_name")}
    return {"jobs": remote}


def rerun(job_id):
    r = vidgen.api("POST", f"/jobs/{job_id}/rerun")
    old = vidgen.load_jobs().get(job_id)
    server_name = (r.get("meta") or {}).get("out_name")  # the server names it: <name>-rerun-<id6>.ext
    if server_name:  # checked like any name from the server: plain, inside the download folder, no clobbering
        out = vidgen.adopted_out(r["type"], r["job_id"], server_name)
    else:  # an older job with no meta: next to the original, the same way
        out_name = None
        if old:
            base, ext = os.path.splitext(os.path.basename(old["out"]))
            out_name = f"{base}-rerun-{r['job_id'][:6]}{ext}"
        out = vidgen.local_out(r["type"], r["job_id"], out_name)
    vidgen.register_job(r["job_id"], r["type"], out, {"rerun_of": job_id})
    return r


def train_theme(name, steps, speed="standard"):
    """Queue a train job for an uploaded theme and register it so `vidgen jobs`/History follow it."""
    if name not in load_themes():
        raise KeyError(name)
    if speed not in ("standard", "fast"):
        raise ValueError("speed must be standard or fast")
    r = vidgen.api("POST", "/jobs", {"type": "train", "dataset": name, "steps": int(steps), "speed": speed,
                                     "meta": vidgen.job_meta("mac", f"train-{name}")})
    vidgen.register_job(r["job_id"], "train", vidgen.local_out("train", r["job_id"], f"train-{name}"))
    return r


CREATE_TYPES = ("video", "photos", "story", "write", "avatar")
MAX_STORY_TEXT = 20000  # same limit as the API's story text
AVATAR_NAME = THEME_NAME  # same rule as the API: 1-32 chars, a-z 0-9 - _
MAX_AVATAR_IMAGE_BYTES = 25 * 1024 ** 2
MAX_BODY_BYTES = 450 * 1024 ** 2  # 12 images of 25 MB as base64, plus room
IMAGE_MAGIC = {"png": (b"\x89PNG\r\n\x1a\n",), "jpg": (b"\xff\xd8\xff",), "webp": (b"RIFF",)}


def _image_ext(data):
    for ext, magics in IMAGE_MAGIC.items():
        if any(data.startswith(m) for m in magics) and (ext != "webp" or data[8:12] == b"WEBP"):
            return ext
    return None


def create_avatar(body):
    """Dashboard: create (or replace) an avatar from 1-12 browser-picked photos of one person.
    body: {name, description, make_default, images: [base64, ...]} in order (the first is the main image).
    The bytes are checked, written to a temp folder and uploaded the same way as `avatar refresh --image`."""
    name = str(body.get("name") or "").strip().lower()
    if not AVATAR_NAME.match(name):
        raise ValueError("name must be 1-32 chars: a-z, 0-9, - or _ (starting with a letter or digit)")
    images = body.get("images")
    if not isinstance(images, list) or not 1 <= len(images) <= vidgen.MAX_AVATAR_IMAGES:
        raise ValueError(f"give 1-{vidgen.MAX_AVATAR_IMAGES} images")
    description = str(body.get("description") or "").strip()
    with tempfile.TemporaryDirectory(prefix="vidgen-avatar-") as tmp:
        paths = []
        for i, b64 in enumerate(images, 1):
            try:
                data = base64.b64decode(str(b64), validate=True)
            except ValueError:
                raise ValueError(f"image {i}: not valid image data")
            if len(data) > MAX_AVATAR_IMAGE_BYTES:
                raise ValueError(f"image {i}: larger than {MAX_AVATAR_IMAGE_BYTES // 1024 ** 2} MB")
            ext = _image_ext(data)
            if not ext:
                raise ValueError(f"image {i}: must be a png, jpg or webp image")
            path = os.path.join(tmp, f"{i:02d}.{ext}")
            with open(path, "wb") as f:
                f.write(data)
            paths.append(path)
        keys = [vidgen.upload_file(p, vidgen.IMAGE_EXTS) for p in paths]
    return vidgen.api("POST", "/avatars", {"name": name, "upload_keys": keys, "description": description,
                                           "make_default": body.get("make_default") is True,
                                           "krea_lora": body.get("krea_lora") is not False})




def create_job(body):
    """Submit a video/photos/story job from the Create tab and register it for the watcher."""
    job = body.get("job") or {}
    if job.get("type") not in CREATE_TYPES:
        raise ValueError(f"type must be one of: {', '.join(CREATE_TYPES)}")
    job = {k: v for k, v in job.items() if v is not None}
    out = (body.get("out") or "").strip() or None
    if out and (os.sep in out or out.startswith(".")):
        raise ValueError("file name: a plain name, no folders")
    extra = None
    if job["type"] == "avatar":  # candidates download as avatar-<name>-....candidate_N.png; Pick saves one
        name = str(body.get("avatar_name") or "").strip().lower()
        if not THEME_NAME.match(name):
            raise ValueError("avatar name must be 1-32 chars: a-z, 0-9, - or _")
        out = out or f"avatar-{name}-{int(time.time())}.png"
        extra = {"avatar_name": name, "krea_lora": body.get("krea_lora") is not False}
    r = vidgen.api("POST", "/jobs", dict(job, meta=vidgen.job_meta("mac", out, extra)))
    vidgen.register_job(r["job_id"], job["type"], vidgen.local_out(job["type"], r["job_id"], out), extra)
    return r


def pick_candidate(body):
    """Save candidate N of a downloaded avatar job as the avatar it was generated for."""
    job_id = str(body.get("job_id") or "")
    local = vidgen.load_jobs().get(job_id)
    if not local or local.get("type") != "avatar":
        raise KeyError(job_id)
    if not local.get("avatar_name"):  # started on another device: the server has its avatar name
        local = dict(vidgen.server_meta(job_id), **local)
    name = local.get("avatar_name") or vidgen.DEFAULT_AVATAR_NAME
    return vidgen.api("POST", "/avatars", {"name": name, "job_id": job_id, "candidate": int(body.get("candidate")),
                                           "make_default": body.get("make_default") is True,
                                           "krea_lora": local.get("krea_lora", True) is not False})


def avatar_krea_lora(name):
    """Build (a new version of) the avatar's Krea 2 LoRA now."""
    name = str(name or "").strip().lower()
    if not THEME_NAME.match(name):
        raise ValueError("avatar name must be 1-32 chars: a-z, 0-9, - or _")
    return vidgen.api("POST", f"/avatars/{name}/krea-lora")


def get_job(job_id):
    """One job's status, progress, result and download links (the dashboard's "Plan first" waits on it)."""
    job_id = str(job_id or "")
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("job id must be 32 hex characters")
    return vidgen.api("GET", f"/jobs/{job_id}")


def cancel_job(job_id):
    """Cancel a queued job, or ask a running one to stop (as `vidgen cancel` does)."""
    job_id = str(job_id or "")
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("job id must be 32 hex characters")
    return vidgen.api("POST", f"/jobs/{job_id}/cancel")


PREVIEW_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
                 ".mp4": "video/mp4"}


def job_file(job_id, index):
    """(path, content type) of a downloaded job's index-th file, for the History previews. Only files the
    watcher saved for that job, and only images and mp4 video; anything else is not found."""
    job = vidgen.load_jobs().get(str(job_id)) or {}
    files = job.get("files") or []
    if job.get("status") != "downloaded" or not 0 <= index < len(files):
        raise KeyError(f"{job_id}/{index}")
    path = files[index]
    content_type = PREVIEW_TYPES.get(os.path.splitext(path)[1].lower())
    if not content_type or not os.path.isfile(path):
        raise KeyError(f"{job_id}/{index}")
    return path, content_type


def delete_avatar(name, loras=False):
    """Delete an avatar on the server (images, earlier versions, media); loras: also its own LoRAs and their
    training photos. Media already copied to the download folder stays."""
    name = str(name or "").strip().lower()
    if not THEME_NAME.match(name):
        raise ValueError("avatar name must be 1-32 chars: a-z, 0-9, - or _")
    return vidgen.api("DELETE", f"/avatars/{name}", {"loras": bool(loras)})


def set_default_avatar(body):
    """Make an existing avatar the default (the one videos and stories star when none is picked)."""
    name = str(body.get("name") or "").strip().lower()
    if not THEME_NAME.match(name):
        raise ValueError("avatar name must be 1-32 chars: a-z, 0-9, - or _")
    return vidgen.api("PUT", "/avatars/default", {"name": name})


def get_avatar(name):
    """One avatar with 1-hour preview links to its images (the Avatars tab's thumbnails)."""
    name = str(name or "").strip().lower()
    if not THEME_NAME.match(name):
        raise ValueError("avatar name must be 1-32 chars: a-z, 0-9, - or _")
    return vidgen.api("GET", f"/avatars/{name}")


def avatar_wan_lora(name):
    """Build (a new version of) the avatar's Wan LoRA from its media now."""
    name = str(name or "").strip().lower()
    if not THEME_NAME.match(name):
        raise ValueError("avatar name must be 1-32 chars: a-z, 0-9, - or _")
    return vidgen.api("POST", f"/avatars/{name}/wan-lora", {})


def avatar_media(name, method):
    """GET: the avatar's media state and file links; POST: make (new) photos + 2 s clips for it."""
    name = str(name or "").strip().lower()
    if not THEME_NAME.match(name):
        raise ValueError("avatar name must be 1-32 chars: a-z, 0-9, - or _")
    return vidgen.api(method, f"/avatars/{name}/media", {} if method == "POST" else None)


def list_loras():
    """GET /loras, with `local: true` on versions copied to this computer (<download dir>/loras/)."""
    r = vidgen.api("GET", "/loras")
    local = vidgen.local_lora_versions()
    for lora in r.get("loras") or []:
        for v in lora.get("versions") or []:
            v["local"] = (lora["name"], v["version"]) in local
    return r


def use_lora(name, body):
    """Pick the version/checkpoint that --lora NAME renders with."""
    return vidgen.api("PUT", f"/loras/{name}/current", {"version": body.get("version"), "step": body.get("step")})


def preview_lora(name, body):
    """Queue checkpoint previews; the watcher downloads them into lora-preview-<job_id>/."""
    req = {"type": "lora_preview", "name": name, "version": body.get("version")}
    if body.get("steps"):
        req["steps"] = body["steps"]
    r = vidgen.api("POST", "/jobs", dict(req, meta=vidgen.job_meta("mac")))
    vidgen.register_job(r["job_id"], "lora_preview", vidgen.local_out("lora_preview", r["job_id"]))
    return r


def written_story(job_id):
    """Text of a downloaded `write` job, to load into the Story form."""
    job = vidgen.load_jobs().get(job_id)
    if not job or job.get("type") != "write" or job.get("status") != "downloaded":
        raise KeyError(job_id)
    with open(job["out"], encoding="utf-8") as f:
        text = f.read(MAX_STORY_TEXT + 1)
    if len(text) > MAX_STORY_TEXT:
        raise ValueError("the story is longer than a story video takes (20000 characters)")
    return {"text": text, "path": job["out"]}


def reveal(path):
    """Show a downloaded file in Finder / the file manager."""
    known = {p for j in vidgen.load_jobs().values() for p in [j["out"], *j.get("files", [])]}
    if path not in known or not os.path.exists(path):
        raise ValueError("not a downloaded job file")
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", path], check=False)
    elif sys.platform == "win32":
        subprocess.run(["explorer", "/select,", path], check=False)
    else:
        subprocess.run(["xdg-open", os.path.dirname(path)], check=False)


# ---------------------------------------------------------------- server

def make_handler(port, token):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet; errors still reach the page
            pass

        def _head(self, status, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             # S3 presigned links: plan first frames and avatar thumbnails (img), avatar preview
                             # clips (media). blob: downloaded job files, which the page fetches with its token.
                             "default-src 'self'; img-src 'self' blob: https://*.amazonaws.com; "
                             "media-src 'self' blob: https://*.amazonaws.com; style-src 'self' 'unsafe-inline'; "
                             "script-src 'self' 'unsafe-inline'")

        def _send(self, status, body, content_type="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
            self._head(status, content_type)
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path, content_type):
            try:
                f = open(path, "rb")
            except OSError:
                raise KeyError(os.path.basename(path))
            with f:
                self._head(200, content_type)
                self.send_header("Content-Length", str(os.fstat(f.fileno()).st_size))
                self.end_headers()
                try:
                    shutil.copyfileobj(f, self.wfile)
                except (BrokenPipeError, ConnectionResetError):  # the page stopped loading it (row closed, reload)
                    pass

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY_BYTES:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def _guard(self):
            if self.headers.get("Host") not in allowed_hosts:
                self._send(403, {"error": "bad host"})
                return False
            if self.path.startswith("/api/") and not secrets.compare_digest(self.headers.get("X-UI-Token", ""), token):
                self._send(403, {"error": "bad token"})
                return False
            return True

        def _route(self, method):
            if not self._guard():
                return
            path = self.path.split("?")[0]
            try:
                if method == "GET" and path == "/":
                    with open(PAGE, "rb") as f:
                        page = f.read().replace(b"__UI_TOKEN__", token.encode())
                    return self._send(200, page, "text/html; charset=utf-8")
                if method == "GET" and path == "/api/status":
                    return self._send(200, vidgen.api("GET", "/status"))
                if method == "POST" and path == "/api/wake":
                    return self._send(200, vidgen.api("POST", "/wake"))
                if method == "POST" and path == "/api/worker/restart":
                    return self._send(200, vidgen.api("POST", "/worker/restart"))
                if method == "GET" and path == "/api/jobs":
                    return self._send(200, job_history())
                m = re.fullmatch(r"/api/jobs/([0-9a-f]{32})", path)
                if method == "GET" and m:
                    return self._send(200, get_job(m.group(1)))
                m = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/story", path)
                if method == "GET" and m:
                    return self._send(200, written_story(m.group(1)))
                m = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/rerun", path)
                if method == "POST" and m:
                    return self._send(202, rerun(m.group(1)))
                m = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/cancel", path)
                if method == "POST" and m:
                    r = cancel_job(m.group(1))
                    return self._send(200 if r.get("status") == "cancelled" else 202, r)
                m = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/files/(\d{1,3})", path)
                if method == "GET" and m:
                    return self._send_file(*job_file(m.group(1), int(m.group(2))))
                if method == "POST" and path == "/api/create":
                    return self._send(202, create_job(self._body()))
                if method == "GET" and path == "/api/avatars":
                    return self._send(200, vidgen.api("GET", "/avatars"))
                if method == "POST" and path == "/api/avatars/pick":
                    return self._send(200, pick_candidate(self._body()))
                if method == "POST" and path == "/api/avatars":
                    return self._send(201, create_avatar(self._body()))
                if method == "PUT" and path == "/api/avatars/default":
                    return self._send(200, set_default_avatar(self._body()))
                if method == "GET" and path == "/api/loras":
                    return self._send(200, list_loras())
                m = re.fullmatch(r"/api/avatars/([a-z0-9][a-z0-9_-]{0,31})", path)
                if method == "GET" and m:
                    return self._send(200, get_avatar(m.group(1)))
                if method == "DELETE" and m:
                    return self._send(200, delete_avatar(
                        m.group(1), loras="loras=1" in self.path.partition("?")[2].split("&")))
                m = re.fullmatch(r"/api/avatars/([a-z0-9][a-z0-9_-]{0,31})/krea-lora", path)
                if method == "POST" and m:
                    return self._send(202, avatar_krea_lora(m.group(1)))
                m = re.fullmatch(r"/api/avatars/([a-z0-9][a-z0-9_-]{0,31})/wan-lora", path)
                if method == "POST" and m:
                    return self._send(202, avatar_wan_lora(m.group(1)))
                m = re.fullmatch(r"/api/avatars/([a-z0-9][a-z0-9_-]{0,31})/media", path)
                if m and method in ("GET", "POST"):
                    return self._send(202 if method == "POST" else 200, avatar_media(m.group(1), method))
                m = re.fullmatch(r"/api/loras/([a-z0-9][a-z0-9_-]{0,31})/(current|preview)", path)
                if m and method == "PUT" and m.group(2) == "current":
                    return self._send(200, use_lora(m.group(1), self._body()))
                if m and method == "POST" and m.group(2) == "preview":
                    return self._send(202, preview_lora(m.group(1), self._body()))
                if method == "POST" and path == "/api/reveal":
                    reveal(self._body().get("path", ""))
                    return self._send(200, {"ok": True})
                if method == "GET" and path == "/api/themes":
                    themes = sorted(load_themes().values(), key=lambda t: t["created_at"])
                    return self._send(200, {"themes": [theme_view(t) for t in themes], "kinds": KINDS})
                if method == "POST" and path == "/api/themes":
                    return self._send(201, add_theme(self._body()))
                m = re.fullmatch(r"/api/themes/([a-z0-9][a-z0-9_-]{0,31})(/rescan|/upload-plan|/upload|/train)?", path)
                if m and method == "POST" and m.group(2) == "/rescan":
                    return self._send(200, rescan_theme(m.group(1)))
                if m and method == "POST" and m.group(2) == "/upload-plan":
                    return self._send(200, upload_plan(m.group(1)))
                if m and method == "POST" and m.group(2) == "/upload":
                    body = self._body()
                    return self._send(202, start_upload(m.group(1), consent=body.get("consent") is True,
                                                        allow_remove=body.get("allow_remove") is True))
                if m and method == "POST" and m.group(2) == "/train":
                    body = self._body()
                    return self._send(202, train_theme(m.group(1), body.get("steps", 1250), body.get("speed") or "standard"))
                if m and method == "DELETE" and not m.group(2):
                    remove_theme(m.group(1), remote="remote=1" in self.path.partition("?")[2].split("&"))
                    return self._send(200, {"ok": True})
                if method == "POST" and path == "/api/pick-folder":
                    return self._send(200, {"folder": pick_folder()})
                if method == "POST" and path == "/api/pick-file":
                    return self._send(200, {"file": pick_file()})
                if method == "POST" and path == "/api/loras":
                    return self._send(201, add_lora(self._body()))
                return self._send(404, {"error": "not found"})
            except NotImplementedError as e:
                return self._send(501, {"error": str(e)})
            except KeyError as e:
                return self._send(404, {"error": f"not found: {e}"})
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            except vidgen.ApiError as e:
                return self._send(502, {"error": str(e)})

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def do_PUT(self):
            self._route("PUT")

        def do_DELETE(self):
            self._route("DELETE")

    return Handler


def serve(port=8765, open_browser=True):
    token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(port, token))
    url = f"http://127.0.0.1:{port}/"
    print(f"vidgen UI on {url}  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    serve(int(os.environ.get("VIDGEN_UI_PORT", "8765")), open_browser="--no-browser" not in sys.argv)
