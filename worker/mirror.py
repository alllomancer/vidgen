"""Model mirror: a copy of the weight folders in the project's bucket (s3://<bucket>/models/<unit>/).

A machine with a new disk restores its weights from here instead of downloading them from Hugging Face
(and, on the render box, instead of converting Wan to bf16 again); whatever a machine has on disk that the
mirror lacks is uploaded in the background. Used by the render box and the trainer (Terraform ships this
one file to both). See docs/specs/2026-11-25-model-mirror.md.

A unit is one folder. Its manifest.json ({"files": {path: size}, "links": {path: symlink target}}) is
written last, so a half-uploaded unit is never restored. Restores are checked by size.

    mirror.py pull UNIT DIR             exit 0 when the folder was restored, 1 when the mirror has no such unit
    mirror.py pull-all ROOT [--skip P]  restore every unit into ROOT/<unit> (first boot)

MODEL_MIRROR=0 turns it off: pull finds nothing and push does nothing.
"""

import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait

PREFIX = "models/"
MANIFEST = "manifest.json"
PART = 64 * 1024 * 1024  # one ranged GET
CHUNK = 4 * 1024 * 1024  # one read from its body
THREADS = 32
TRIES = 3  # per part
SKIP_DIRS = (".cache", ".locks")
SKIP_SUFFIXES = (".incomplete", ".lock", ".part")
SKIP_NAMES = (".complete",)

_s3 = None
_pushes = queue.Queue()
_pusher = None
_lock = threading.Lock()


class _Failed(Exception):
    """A restore that could not be finished (the caller falls back to Hugging Face)."""


def log(**fields):
    print(json.dumps(fields, default=str), flush=True)


def enabled():
    return os.environ.get("MODEL_MIRROR", "1") != "0" and bool(os.environ.get("BUCKET"))


def client():
    global _s3
    if _s3 is None:
        import boto3
        from botocore.config import Config

        _s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"),
                           config=Config(max_pool_connections=THREADS + 16))
    return _s3


def _bucket():
    return os.environ["BUCKET"]


def _safe(rel):
    """A relative path that stays inside its folder (unit names and manifest paths come from S3)."""
    return bool(rel) and not rel.startswith("/") and all(p not in ("", ".", "..") for p in rel.split("/"))


def _link_ok(local_dir, rel, target):
    """A symlink whose target stays inside the unit's folder."""
    if not _safe(rel) or not isinstance(target, str) or os.path.isabs(target):
        return False
    root = os.path.normpath(local_dir)
    dest = os.path.normpath(os.path.join(root, os.path.dirname(rel), target))
    return dest.startswith(root + os.sep)


def local_files(local_dir):
    """({relative path: size}, {relative path: symlink target}) of a folder, without caches, locks and
    unfinished downloads."""
    files, links = {}, {}
    for d, dirs, names in os.walk(local_dir):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for n in list(dirs) + names:
            p = os.path.join(d, n)
            rel = os.path.relpath(p, local_dir).replace(os.sep, "/")
            if os.path.islink(p):
                links[rel] = os.readlink(p)
            elif n in names and n not in SKIP_NAMES and not n.endswith(SKIP_SUFFIXES):
                files[rel] = os.path.getsize(p)
        dirs[:] = [x for x in dirs if not os.path.islink(os.path.join(d, x))]
    return files, links


def _manifest(unit):
    """The unit's manifest, or None when there is none (or it can't be read)."""
    try:
        body = client().get_object(Bucket=_bucket(), Key=f"{PREFIX}{unit}/{MANIFEST}")["Body"].read()
        man = json.loads(body)
        files, links = man["files"], man.get("links") or {}
        if not all(isinstance(s, int) and s >= 0 for s in files.values()):
            raise ValueError("sizes")
        return {"files": files, "links": links}
    except Exception as e:  # no such key, no access, not JSON: there is no usable mirror of this unit
        log(event="mirror_manifest", unit=unit, found=False, reason=type(e).__name__)
        return None


def _stats(unit, source, nbytes, t0, **extra):
    secs = round(time.time() - t0, 1)
    return {"unit": unit, "source": source, "bytes": nbytes, "seconds": secs,
            "mb_per_s": round(nbytes / 1e6 / max(secs, 0.1)), **extra}


# ---------------------------------------------------------------- restore

def _fetch(unit, local_dir, need, progress):
    """Download `need` ([(relative path, size)]) with parallel ranged GETs into <file>.part, then rename."""
    s3, bucket, total = client(), _bucket(), sum(s for _, s in need)
    fds, parts, done, stop = {}, [], [0], threading.Event()

    def get(rel, off, n):
        for attempt in range(TRIES):
            got = 0
            try:
                if stop.is_set():
                    return
                body = s3.get_object(Bucket=bucket, Key=f"{PREFIX}{unit}/{rel}",
                                     Range=f"bytes={off}-{off + n - 1}")["Body"]
                while not stop.is_set():
                    chunk = body.read(CHUNK)
                    if not chunk:
                        break
                    os.pwrite(fds[rel], chunk, off + got)
                    got += len(chunk)
                    with _lock:
                        done[0] += len(chunk)
                if got == n or stop.is_set():
                    return
                raise OSError(f"{rel}: got {got} of {n} bytes at {off}")
            except Exception:  # a dropped connection mid-file: this part again, from its start
                with _lock:
                    done[0] -= got
                if attempt == TRIES - 1:
                    raise

    def tell():
        if progress:
            progress(f"restoring {unit} from the model mirror: {done[0] / 1e9:.1f}/{total / 1e9:.1f} GB")

    try:
        for rel, size in need:
            path = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.lexists(path):
                os.remove(path)
            if size == 0:
                open(path, "wb").close()
                continue
            fds[rel] = os.open(path + ".part", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.ftruncate(fds[rel], size)
            parts += [(rel, off, min(PART, size - off)) for off in range(0, size, PART)]
        tell()
        with ThreadPoolExecutor(THREADS) as pool:
            pending = {pool.submit(get, *p) for p in parts}
            try:
                while pending:
                    finished, pending = wait(pending, timeout=2, return_when=FIRST_EXCEPTION)
                    for f in finished:
                        if f.exception() is not None:
                            raise _Failed(f"{type(f.exception()).__name__}: {f.exception()}")
                    tell()
            finally:
                stop.set()  # on an error or a cancel the running parts end at their next read
                for f in pending:
                    f.cancel()
    except BaseException:
        _close(fds)
        for rel, size in need:
            if size and os.path.exists(os.path.join(local_dir, rel) + ".part"):
                os.remove(os.path.join(local_dir, rel) + ".part")
        raise
    _close(fds)
    for rel, size in need:
        if size:
            os.rename(os.path.join(local_dir, rel) + ".part", os.path.join(local_dir, rel))
    return total


def _close(fds):
    for rel in list(fds):
        os.close(fds.pop(rel))


def pull(unit, local_dir, progress=None):
    """Restore a unit into local_dir. Returns {"source": "mirror", "bytes", "seconds", ...}, or None when
    the mirror has no usable copy (the caller then downloads from Hugging Face). Only an exception raised by
    `progress` (a cancel) leaves this function."""
    if not enabled() or not _safe(unit):
        return None
    t0 = time.time()
    man = _manifest(unit)
    if man is None:
        return None
    if not all(_safe(r) for r in man["files"]) or not all(_link_ok(local_dir, r, t) for r, t in man["links"].items()):
        log(event="mirror_pull", unit=unit, ok=False, reason="a path in the manifest leaves the folder")
        return None

    def have(rel, size):
        p = os.path.join(local_dir, rel)
        return os.path.isfile(p) and not os.path.islink(p) and os.path.getsize(p) == size

    need = [(rel, size) for rel, size in sorted(man["files"].items()) if not have(rel, size)]
    try:
        nbytes = _fetch(unit, local_dir, need, progress) if need else 0
        for rel, target in man["links"].items():
            path = os.path.join(local_dir, rel)
            if os.path.islink(path) and os.readlink(path) == target:
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.lexists(path):
                os.remove(path)
            os.symlink(target, path)
    except (_Failed, OSError) as e:
        log(event="mirror_pull", unit=unit, ok=False, reason=str(e)[:300])
        return None
    stats = _stats(unit, "mirror", nbytes, t0, files=len(need))
    log(event="mirror_pull", ok=True, **stats)
    return stats


def units():
    """Every unit in the mirror that has a manifest."""
    out, token = [], None
    while True:
        kw = {"ContinuationToken": token} if token else {}
        page = client().list_objects_v2(Bucket=_bucket(), Prefix=PREFIX, **kw)
        out += [o["Key"][len(PREFIX):-len(MANIFEST) - 1] for o in page.get("Contents") or []
                if o["Key"].endswith("/" + MANIFEST)]
        if not page.get("IsTruncated"):
            return sorted(out)
        token = page["NextContinuationToken"]


def pull_all(local_root, skip=(), progress=None):
    """Restore every mirrored unit into local_root/<unit>, except units starting with a `skip` prefix.
    Returns [(unit, stats)] of the ones restored."""
    if not enabled():
        return []
    try:
        names = [u for u in units() if _safe(u) and not u.startswith(tuple(skip))]
    except Exception as e:
        log(event="mirror_pull_all", ok=False, reason=type(e).__name__)
        return []
    done = [(u, pull(u, os.path.join(local_root, u), progress)) for u in names]
    return [(u, st) for u, st in done if st]


# ---------------------------------------------------------------- upload

def _upload(path, key):
    try:
        from boto3.s3.transfer import TransferConfig

        cfg = TransferConfig(max_concurrency=16, multipart_chunksize=PART, multipart_threshold=PART)
    except ImportError:
        cfg = None
    client().upload_file(path, _bucket(), key, Config=cfg)


def _remote_sizes(unit):
    out, token, prefix = {}, None, f"{PREFIX}{unit}/"
    while True:
        kw = {"ContinuationToken": token} if token else {}
        page = client().list_objects_v2(Bucket=_bucket(), Prefix=prefix, **kw)
        out.update({o["Key"][len(prefix):]: o["Size"] for o in page.get("Contents") or []})
        if not page.get("IsTruncated"):
            return out
        token = page["NextContinuationToken"]


def _listing(local_dir, files):
    """What a push would mirror: (files, links), or None when a listed file is missing."""
    if files is None:
        return local_files(local_dir)
    paths = {rel: os.path.join(local_dir, rel) for rel in files}
    if not all(os.path.isfile(p) for p in paths.values()):
        return None
    return {rel: os.path.getsize(p) for rel, p in paths.items()}, {}


def push(unit, local_dir, files=None, _listed=None):
    """Upload what the mirror lacks of a unit (files: the unit's relative paths when the folder holds more
    than the unit), then its manifest. Returns stats, or None when nothing could be mirrored. Never raises:
    the mirror is a convenience, the weights on disk are what the machine uses."""
    if not enabled() or not _safe(unit):
        return None
    t0 = time.time()
    try:
        listed = _listed or _listing(local_dir, files)
        if not listed or not listed[0]:
            return None
        mine, links = listed
        man = _manifest(unit) or {"files": {}, "links": {}}
        if all(man["files"].get(r) == s for r, s in mine.items()) and \
                all(man["links"].get(r) == t for r, t in links.items()):
            return _stats(unit, "disk", 0, t0, uploaded=0)  # the mirror already holds all of it
        remote = _remote_sizes(unit)
        todo = [(rel, size) for rel, size in sorted(mine.items()) if remote.get(rel) != size]
        for rel, _ in todo:
            _upload(os.path.join(local_dir, rel), f"{PREFIX}{unit}/{rel}")
        # Kept: what another machine sharing this unit mirrored (its objects are still there).
        merged = {"files": {**man["files"], **mine}, "links": {**man["links"], **links}}
        client().put_object(Bucket=_bucket(), Key=f"{PREFIX}{unit}/{MANIFEST}",
                            Body=json.dumps(merged, indent=1, sort_keys=True).encode(), ContentType="application/json")
    except Exception as e:
        log(event="mirror_push", unit=unit, ok=False, reason=f"{type(e).__name__}: {str(e)[:200]}")
        return None
    stats = _stats(unit, "disk", sum(s for _, s in todo), t0, uploaded=len(todo))
    log(event="mirror_push", ok=True, **stats)
    return stats


def _push_loop():
    while True:
        args = _pushes.get()
        try:
            push(*args[:3], _listed=args[3])
        finally:
            _pushes.task_done()


def push_async(unit, local_dir, files=None):
    """push() on one background thread (pushes run one after another, so they never crowd the disk). The
    file list is taken now, before anything else writes into the folder."""
    global _pusher
    if not enabled() or not _safe(unit):
        return
    try:
        listed = _listing(local_dir, files)
    except OSError as e:
        log(event="mirror_push", unit=unit, ok=False, reason=type(e).__name__)
        return
    if not listed or not listed[0]:
        return
    with _lock:
        if _pusher is None or not _pusher.is_alive():
            _pusher = threading.Thread(target=_push_loop, daemon=True, name="mirror-push")
            _pusher.start()
    _pushes.put((unit, local_dir, files, listed))


def wait_pushes():
    """Block until the queued pushes are done (tests)."""
    _pushes.join()


def main(argv):
    say = lambda msg: print(msg, flush=True)  # noqa: E731
    if len(argv) == 3 and argv[0] == "pull":
        return 0 if pull(argv[1], argv[2], say) else 1
    if argv[:1] == ["pull-all"] and len(argv) in (2, 4) and (len(argv) == 2 or argv[2] == "--skip"):
        pull_all(argv[1], skip=tuple(argv[3:]), progress=say)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
