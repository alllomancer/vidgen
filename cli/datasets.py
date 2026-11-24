"""Learning datasets: sync a local folder to S3 (datasets/<client>/<name>/) for the trainer.
Used by `vidgen.py dataset ...` and the dashboard's Upload button. Standard library only.

A sync uploads only files that are new or changed (path, size, mtime) since the last
upload, then commits the manifest; files no longer in the folder are deleted from the
dataset, which needs explicit confirmation (allow_remove). Files go straight to S3 with
presigned PUT URLs; the API never handles file bytes.
"""

import hashlib
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import vidgen

DATASET_EXTS = {"mp4", "mov", "webm", "mkv", "png", "jpg", "jpeg", "webp"}
MAX_FILE_BYTES = 2 * 1024 ** 3
URL_BATCH = 100  # API limit per POST /datasets/{name}/uploads
PUT_RETRIES = 3


class NeedsConfirmation(Exception):
    """The sync would delete files from the dataset (or needs consent) and was not allowed to."""


def local_files(folder):
    """(files, skipped): files = [{path, size, mtime}] sorted by path, path relative with '/'."""
    files, skipped = [], {"too_big": [], "unsupported": 0}
    for root, dirs, names in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(names):
            if name.startswith("."):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, folder).replace(os.sep, "/")
            if os.path.splitext(name)[1].lstrip(".").lower() not in DATASET_EXTS:
                skipped["unsupported"] += 1
                continue
            st = os.stat(full)
            if st.st_size > MAX_FILE_BYTES:
                skipped["too_big"].append(rel)
                continue
            if st.st_size == 0:  # empty files cannot be training data
                skipped["unsupported"] += 1
                continue
            files.append({"path": rel, "size": st.st_size, "mtime": int(st.st_mtime)})
    return sorted(files, key=lambda f: f["path"]), skipped


def fingerprint(files):
    """Stable id of a file list: changes when any path, size or mtime changes."""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda f: f["path"]):
        h.update(f"{f['path']}\0{f['size']}\0{f['mtime']}\n".encode())
    return h.hexdigest()[:16]


def remote_manifest(name):
    try:
        return vidgen.api("GET", f"/datasets/{name}")
    except vidgen.ApiError as e:
        if e.status == 404:
            return None
        raise


def plan(name, folder):
    """What a sync would do, without doing it."""
    files, skipped = local_files(folder)
    remote = remote_manifest(name)
    have = {f["path"]: f for f in (remote or {}).get("files", [])}
    upload = [f for f in files if (have.get(f["path"]) or {}).get("size") != f["size"]
              or have[f["path"]].get("mtime") != f["mtime"]]
    local_paths = {f["path"] for f in files}
    remove = sorted(p for p in have if p not in local_paths)
    return {"files": files, "skipped": skipped, "remote": remote, "upload": upload, "remove": remove,
            "upload_bytes": sum(f["size"] for f in upload), "total_bytes": sum(f["size"] for f in files),
            "fingerprint": fingerprint(files), "has_consent": bool((remote or {}).get("consent_at"))}


class _Progress:
    """File-like wrapper that reports bytes read (urllib streams the body from it)."""

    def __init__(self, f, on_bytes):
        self.f, self.on_bytes = f, on_bytes

    def read(self, n=-1):
        chunk = self.f.read(n if n and n > 0 else 1 << 20)
        if chunk:
            self.on_bytes(len(chunk))
        return chunk


FILE_WORKERS = 4  # dataset files uploaded at once


def _upload(u, path, size, on_bytes):
    """One dataset file: a parallel multipart upload when the API offers one (big files), else one PUT."""
    if u.get("multipart"):
        return vidgen.upload_multipart(u["multipart"], path, on_bytes)
    return _put(u["put_url"], path, size, on_bytes)


def _put(url, path, size, on_bytes):
    for attempt in range(1, PUT_RETRIES + 1):
        sent = 0

        def count(n):
            nonlocal sent
            sent += n
            on_bytes(n)
        try:
            with open(path, "rb") as f:
                req = urllib.request.Request(url, data=_Progress(f, count), method="PUT",
                                             headers={"Content-Length": str(size)})
                urllib.request.urlopen(req, timeout=600).close()
            return
        except (urllib.error.URLError, OSError) as e:
            on_bytes(-sent)  # this attempt's bytes do not count
            if attempt == PUT_RETRIES:
                raise vidgen.ApiError(f"upload of {path} failed: {e}")


def sync(name, folder, kind, trigger=None, consent=False, allow_remove=False, progress=None, the_plan=None):
    """Upload new/changed files and commit the manifest. progress(dict) is called as it goes."""
    p = the_plan or plan(name, folder)
    if not p["files"]:
        raise vidgen.ApiError(f"no clips or photos in {folder}")
    if p["remove"] and not allow_remove:
        raise NeedsConfirmation(f"{len(p['remove'])} file(s) would be deleted from the dataset")
    if kind == "character" and not consent and not p["has_consent"]:
        raise NeedsConfirmation("character datasets need the consent of the person shown")
    state = {"files_done": 0, "files_total": len(p["upload"]), "bytes_done": 0, "bytes_total": p["upload_bytes"]}

    def tick(n=0, file_done=False):
        state["bytes_done"] += n
        state["files_done"] += file_done
        if progress:
            progress(dict(state))

    tick()
    for i in range(0, len(p["upload"]), URL_BATCH):
        batch = p["upload"][i:i + URL_BATCH]
        urls = vidgen.api("POST", f"/datasets/{name}/uploads",
                          {"files": [{"path": f["path"], "size": f["size"]} for f in batch]})["uploads"]
        def one(pair):  # several files at once; big ones also in parallel parts
            f, u = pair
            _upload(u, os.path.join(folder, *f["path"].split("/")), f["size"], tick)
            tick(file_done=True)

        with ThreadPoolExecutor(FILE_WORKERS) as pool:
            list(pool.map(one, zip(batch, urls)))
    result = vidgen.api("PUT", f"/datasets/{name}", {"kind": kind, "trigger": trigger, "consent": bool(consent),
                                                     "files": p["files"]})
    result.update(uploaded=len(p["upload"]), uploaded_bytes=p["upload_bytes"], fingerprint=p["fingerprint"])
    return result


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
