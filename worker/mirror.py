"""Model mirror: a copy of the weight folders in the project's bucket (s3://<bucket>/models/<unit>/).

A machine with a new disk restores its weights from here instead of downloading them from Hugging Face
(and, on the render box, instead of converting Wan to bf16 again); whatever a machine has on disk that the
mirror lacks is uploaded in the background. Used by the render box and the trainer (Terraform ships this
one file to both). See docs/specs/2026-11-25-model-mirror.md.

A unit is one folder. Its manifest (.mirror-manifest.json: {"files": {path: size}, "links": {path: symlink
target}}) is written last, so a half-uploaded unit is never restored. Restores are checked by size.

The manifest is data from the bucket and this runs as root, so a restore never trusts it: paths must stay
inside the unit's folder, also through symlinks (links are leaves, their targets never pass through another
link, and every write checks where its folder really is on disk).

    mirror.py pull UNIT DIR             exit 0 when the folder was restored, 1 when the mirror has no such unit
    mirror.py pull-all ROOT [--skip P]  restore every unit into ROOT/<unit> (first boot)

MODEL_MIRROR=0 turns it off: pull finds nothing and push does nothing.
"""

import json
import os
import queue
import shutil
import sys
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait

PREFIX = "models/"
MANIFEST = ".mirror-manifest.json"
PART = 64 * 1024 * 1024  # one ranged GET
CHUNK = 4 * 1024 * 1024  # one read from its body
THREADS = 32
TRIES = 3  # per part
MAX_FILES = 20000
DISK_RESERVE = 5 * 1024 ** 3  # a restore never fills the disk to the last GB
SKIP_DIRS = (".cache", ".locks")
SKIP_SUFFIXES = (".incomplete", ".lock", ".part")
SKIP_NAMES = (".complete", MANIFEST)

_s3 = None
_pushes = queue.Queue()
_pusher = None
_lock = threading.Lock()


class _Failed(Exception):
    """A restore that could not be finished (the caller falls back to Hugging Face)."""


class _ProgressRaised(Exception):
    """The caller's progress callback raised (a job cancel): the one thing pull() lets through."""


class _NoManifest(Exception):
    """The unit has no manifest in the bucket."""


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
    return (isinstance(rel, str) and bool(rel) and not rel.startswith("/") and "\x00" not in rel
            and all(p not in ("", ".", "..") for p in rel.split("/")))


def _under(rel, links):
    """Does a path lie below one of the links (a link used as a folder)?"""
    parts = rel.split("/")
    return any("/".join(parts[:i]) in links for i in range(1, len(parts)))


def _link_ok(rel, target, links):
    """A symlink that is a leaf and whose target stays inside the unit without passing through, or ending
    at, another link (so what the path says is where it really goes)."""
    if not _safe(rel) or not isinstance(target, str) or not target or target.startswith("/") or "\x00" in target:
        return False
    at = rel.split("/")[:-1]
    for comp in target.split("/"):
        if comp in ("", "."):
            continue
        if comp == "..":
            if not at:
                return False  # above the unit's folder
            at.pop()
        else:
            at.append(comp)
            if "/".join(at) in links:
                return False
    return bool(at)


def _valid(man):
    """Is this a manifest we would restore? (Shape, sizes, and paths that stay inside the folder.)"""
    if not isinstance(man, dict):
        return False
    files, links = man.get("files"), man.get("links", {})
    if not isinstance(files, dict) or not isinstance(links, dict) or len(files) + len(links) > MAX_FILES:
        return False
    if not all(_safe(r) and type(s) is int and s >= 0 and r not in links and not _under(r, links)
               for r, s in files.items()):
        return False
    return all(not _under(r, links) and _link_ok(r, t, links) for r, t in links.items())


def _inside(local_dir, path):
    """Is the folder a path would be written into really inside local_dir on this disk (whatever symlinks
    are already there)?"""
    root = os.path.realpath(local_dir)
    parent = os.path.realpath(os.path.dirname(path))
    return parent == root or parent.startswith(root + os.sep)


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


def _missing(e):
    """Is this S3 error "no such object"?"""
    code = str(((getattr(e, "response", None) or {}).get("Error") or {}).get("Code"))
    return code in ("NoSuchKey", "404")


def _read_manifest(unit):
    """The unit's manifest as stored. Raises _NoManifest when there is none, ValueError when it is not one
    we would restore, and the S3 error for anything else (throttling, no access)."""
    try:
        body = client().get_object(Bucket=_bucket(), Key=f"{PREFIX}{unit}/{MANIFEST}")["Body"].read()
    except Exception as e:
        if _missing(e):
            raise _NoManifest(unit) from None
        raise
    try:
        man = json.loads(body)
    except (ValueError, TypeError) as e:
        raise ValueError("not JSON") from e
    if not _valid(man):
        raise ValueError("not a usable manifest")
    return {"files": man["files"], "links": man.get("links", {})}


def _stats(unit, source, nbytes, t0, **extra):
    secs = round(time.time() - t0, 1)
    return {"unit": unit, "source": source, "bytes": nbytes, "seconds": secs,
            "mb_per_s": round(nbytes / 1e6 / max(secs, 0.1)), **extra}


# ---------------------------------------------------------------- restore

def _write_all(fd, data, pos):
    """pwrite until every byte is on disk (a full disk returns a short count before it raises)."""
    view, done = memoryview(data), 0
    while done < len(view):
        n = os.pwrite(fd, view[done:], pos + done)
        if n <= 0:
            raise OSError("short write")
        done += n


def _fetch(unit, local_dir, need, progress):
    """Download `need` ([(relative path, size)]) with parallel ranged GETs into <file>.part, then rename."""
    s3, bucket, total = client(), _bucket(), sum(s for _, s in need)
    parts, done, stop = [], [0], threading.Event()
    sizes = dict(need)

    def get(rel, off, n):
        fd = os.open(os.path.join(local_dir, rel) + ".part", os.O_WRONLY)
        try:
            for attempt in range(TRIES):
                got, body = 0, None
                try:
                    if stop.is_set():
                        return
                    resp = s3.get_object(Bucket=bucket, Key=f"{PREFIX}{unit}/{rel}", Range=f"bytes={off}-{off + n - 1}")
                    body = resp["Body"]
                    whole = str(resp.get("ContentRange") or "").rpartition("/")[2]
                    if whole.isdigit() and int(whole) != sizes[rel]:  # the object is not the one the manifest lists
                        raise _Failed(f"{rel}: {whole} bytes in the bucket, {sizes[rel]} in the manifest")
                    while not stop.is_set():
                        chunk = body.read(min(CHUNK, n - got) or 1)
                        if not chunk:
                            break
                        if got + len(chunk) > n:
                            raise OSError(f"{rel}: more than {n} bytes at {off}")
                        _write_all(fd, chunk, off + got)
                        got += len(chunk)
                        with _lock:
                            done[0] += len(chunk)
                        if got == n:
                            break
                    if got == n or stop.is_set():
                        return
                    raise OSError(f"{rel}: got {got} of {n} bytes at {off}")
                except Exception as e:  # a dropped connection mid-file: this part again, from its start
                    with _lock:
                        done[0] -= got
                    if attempt == TRIES - 1 or isinstance(e, _Failed):
                        raise
                    stop.wait(attempt + 1)
                finally:
                    if body is not None and hasattr(body, "close"):
                        body.close()
        finally:
            os.close(fd)

    def tell():
        if progress:
            try:
                progress(f"restoring {unit} from the model mirror: {done[0] / 1e9:.1f}/{total / 1e9:.1f} GB")
            except BaseException as e:
                raise _ProgressRaised() from e

    try:
        for rel, size in need:
            path = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not _inside(local_dir, path):
                raise _Failed(f"{rel}: its folder is outside {local_dir}")
            if os.path.lexists(path):
                os.remove(path)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path if size == 0 else path + ".part", flags, 0o644)
            try:
                os.ftruncate(fd, size)
            finally:
                os.close(fd)
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
        for rel, size in need:
            if size:
                os.rename(os.path.join(local_dir, rel) + ".part", os.path.join(local_dir, rel))
    except BaseException:
        for rel, size in need:
            part = os.path.join(local_dir, rel) + ".part"
            if size and os.path.lexists(part) and _inside(local_dir, part):
                os.remove(part)
        raise
    return total


def _restore(unit, local_dir, man, progress):
    def have(rel, size):
        p = os.path.join(local_dir, rel)
        return os.path.isfile(p) and not os.path.islink(p) and os.path.getsize(p) == size and _inside(local_dir, p)

    need = [(rel, size) for rel, size in sorted(man["files"].items()) if not have(rel, size)]
    total = sum(s for _, s in need)
    if need:
        at = local_dir
        while not os.path.isdir(at):
            at = os.path.dirname(at) or "/"
        if total > shutil.disk_usage(at).free - DISK_RESERVE:
            raise _Failed(f"{total / 1e9:.0f} GB to restore, not enough free disk")
        _fetch(unit, local_dir, need, progress)
    for rel, target in sorted(man["links"].items()):
        path = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not _inside(local_dir, path):
            raise _Failed(f"{rel}: its folder is outside {local_dir}")
        if os.path.islink(path) and os.readlink(path) == target:
            continue
        if os.path.lexists(path):
            os.remove(path)
        os.symlink(target, path)
    return total, len(need)


def pull(unit, local_dir, progress=None):
    """Restore a unit into local_dir. Returns {"source": "mirror", "bytes", "seconds", ...}, or None when
    the mirror has no usable copy (the caller then downloads from Hugging Face). Only an exception raised by
    `progress` (a cancel) leaves this function."""
    if not enabled() or not _safe(unit):
        return None
    t0 = time.time()
    try:
        man = _read_manifest(unit)
        nbytes, nfiles = _restore(unit, local_dir, man, progress)
    except _ProgressRaised as p:
        raise p.__cause__ from None
    except _NoManifest:
        log(event="mirror_pull", unit=unit, ok=False, reason="not in the mirror")
        return None
    except Exception as e:
        log(event="mirror_pull", unit=unit, ok=False, reason=f"{type(e).__name__}: {str(e)[:300]}")
        return None
    stats = _stats(unit, "mirror", nbytes, t0, files=nfiles)
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
        token = page.get("NextContinuationToken")
        if not page.get("IsTruncated") or not token:
            return sorted(out)


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
        token = page.get("NextContinuationToken")
        if not page.get("IsTruncated") or not token:
            return out


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
    the mirror is a convenience, the weights on disk are what the machine uses.

    Two machines pushing the same unit at the same moment can each write a manifest without the other's
    new files; the objects stay, and that machine's next start adds them back."""
    if not enabled() or not _safe(unit):
        return None
    t0 = time.time()
    try:
        listed = _listed or _listing(local_dir, files)
        if not listed or not listed[0]:
            return None
        mine, links = listed
        try:
            man = _read_manifest(unit)
        except (_NoManifest, ValueError):  # none yet, or one nobody could restore: start it again
            man = {"files": {}, "links": {}}
        # Any other error (throttling, a timeout) stops here: writing a manifest without knowing the
        # current one would drop what another machine mirrored.
        if all(man["files"].get(r) == s for r, s in mine.items()) and \
                all(man["links"].get(r) == t for r, t in links.items()):
            return _stats(unit, "disk", 0, t0, uploaded=0)  # the mirror already holds all of it
        # Kept: what another machine sharing this unit mirrored (its objects are still there).
        merged = {"files": {**man["files"], **mine}, "links": {**man["links"], **links}}
        if not _valid(merged):
            raise ValueError("this folder's layout can't be mirrored (a file below a symlink, or a link out of it)")
        remote = _remote_sizes(unit)
        todo = [(rel, size) for rel, size in sorted(mine.items()) if remote.get(rel) != size]
        for rel, _ in todo:
            _upload(os.path.join(local_dir, rel), f"{PREFIX}{unit}/{rel}")
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
