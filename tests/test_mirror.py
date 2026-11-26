"""Model mirror (worker/mirror.py): restore weight folders from the bucket, upload what it lacks."""
import io
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("worker")
os.environ.pop("MODEL_MIRROR", None)
import mirror  # noqa: E402


class AccessDenied(Exception):
    response = {"Error": {"Code": "AccessDenied"}}  # the shape of botocore's ClientError


class S3:
    """In-memory bucket with the calls the mirror makes; counts them."""

    def __init__(self):
        self.objects, self.gets, self.uploads, self.calls, self.fail_upload, self.deny = {}, [], [], 0, None, False
        self.throttle = None

    def get_object(self, Bucket, Key, Range=None):
        self.calls += 1
        if self.deny:
            raise PermissionError("AccessDenied")
        if self.throttle and self.throttle in Key:
            raise TimeoutError("SlowDown")
        if Key not in self.objects:
            # What S3 answers these machines: their ListBucket grant is scoped by prefix, which a GET
            # can't satisfy, so a missing key is "access denied", not "no such key".
            raise AccessDenied(Key)
        data = self.objects[Key]
        if Range:
            a, b = (int(x) for x in Range.split("=")[1].split("-"))
            assert a <= b < len(data), f"bad range {Range} for {len(data)} bytes"
            data = data[a:b + 1]
        self.gets.append((Key, Range))
        return {"Body": io.BytesIO(data)}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.calls += 1
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()
        self.uploads.append(Key)

    def upload_file(self, Filename, Bucket, Key, Config=None):
        self.calls += 1
        if self.fail_upload and self.fail_upload in Key:
            raise OSError("upload failed")
        with open(Filename, "rb") as f:
            self.objects[Key] = f.read()
        self.uploads.append(Key)

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        self.calls += 1
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + 2]  # tiny pages: the mirror must follow them
        out = {"Contents": [{"Key": k, "Size": len(self.objects[k])} for k in page],
               "IsTruncated": start + 2 < len(keys)}
        if out["IsTruncated"]:
            out["NextContinuationToken"] = str(start + 2)
        return out


def fresh():
    s3 = S3()
    mirror._s3 = s3
    return s3


def write(root, rel, data):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)


def tree(root):
    out = {}
    for d, _, names in os.walk(root):
        for n in names:
            p = os.path.join(d, n)
            rel = os.path.relpath(p, root)
            out[rel] = ("link", os.readlink(p)) if os.path.islink(p) else open(p, "rb").read()
    return out


mirror.PART = 5  # bytes: every file below spans several ranged GETs
BIG = bytes(range(256)) * 3

# ---- the local listing skips caches, locks and unfinished files; symlinks are links (criterion 8)
src = tempfile.mkdtemp()
write(src, "blobs/abc", BIG)
write(src, "blobs/empty", b"")
write(src, "refs/main", b"rev1")
os.makedirs(os.path.join(src, "snapshots/rev1"))
os.symlink("../../blobs/abc", os.path.join(src, "snapshots/rev1/model.safetensors"))
for junk in (".cache/huggingface/x.metadata", ".locks/a.lock", "blobs/def.incomplete", "blobs/x.lock",
             "blobs/y.part", ".complete"):
    write(src, junk, b"junk")
files, links = mirror.local_files(src)
assert files == {"blobs/abc": len(BIG), "blobs/empty": 0, "refs/main": 4}, files
assert links == {"snapshots/rev1/model.safetensors": "../../blobs/abc"}, links

# ---- push uploads the files, then the manifest last (criterion 5)
s3 = fresh()
st = mirror.push("hub/models--a--b", src)
assert st["uploaded"] == 3 and st["bytes"] == len(BIG) + 4, st
assert s3.uploads[-1] == "models/hub/models--a--b/.mirror-manifest.json", s3.uploads
assert sorted(s3.uploads[:-1]) == ["models/hub/models--a--b/blobs/abc", "models/hub/models--a--b/blobs/empty",
                                   "models/hub/models--a--b/refs/main"]
man = json.loads(s3.objects["models/hub/models--a--b/.mirror-manifest.json"])
assert man == {"files": files, "links": links}, man

# ---- pull restores files, sizes, content and symlinks (criterion 1); 0-byte files without a GET (10)
dst = tempfile.mkdtemp()
seen = []
st = mirror.pull("hub/models--a--b", dst, progress=seen.append)
assert st["source"] == "mirror" and st["bytes"] == len(BIG) + 4 and st["seconds"] >= 0, st
want = {k: v for k, v in tree(src).items() if k in files or k in links}
assert tree(dst) == want, sorted(tree(dst))
assert open(os.path.join(dst, "snapshots/rev1/model.safetensors"), "rb").read() == BIG  # the link resolves
assert not any(k.endswith("blobs/empty") for k, _ in s3.gets)
assert len([1 for k, r in s3.gets if k.endswith("blobs/abc")]) == -(-len(BIG) // 5)  # ranged parts
assert seen, "progress is reported"

# ---- a second pull keeps files of the right size (criterion 2); a changed size is fetched again
s3.gets.clear()
st = mirror.pull("hub/models--a--b", dst)
assert st["bytes"] == 0 and [k for k, _ in s3.gets] == ["models/hub/models--a--b/.mirror-manifest.json"], s3.gets
write(dst, "refs/main", b"stale-longer")
st = mirror.pull("hub/models--a--b", dst)
assert st["bytes"] == 4 and open(os.path.join(dst, "refs/main"), "rb").read() == b"rev1"

# ---- no manifest, or no access: None, nothing left behind (criterion 3)
empty = tempfile.mkdtemp()
assert mirror.pull("hub/models--none", empty) is None and os.listdir(empty) == []
s3.deny = True
assert mirror.pull("hub/models--a--b", empty) is None and os.listdir(empty) == []
s3.deny = False

# ---- a file that arrives with another size: None, not kept under its name, no .part (criterion 4)
s3.objects["models/hub/models--a--b/refs/main"] = b"r"  # the manifest says 4 bytes
bad = tempfile.mkdtemp()
assert mirror.pull("hub/models--a--b", bad) is None
assert "refs/main" not in tree(bad) and not [k for k in tree(bad) if k.endswith(".part")], tree(bad).keys()
s3.objects["models/hub/models--a--b/refs/main"] = b"rev1"

# ---- a part whose connection drops is fetched again (twice at most), not the whole unit given up
flaky = {"left": 2}
real_get = s3.get_object


def dropping(Bucket, Key, Range=None):
    if Range and Key.endswith("blobs/abc") and Range.startswith("bytes=10-") and flaky["left"]:
        flaky["left"] -= 1
        raise ConnectionError("reset")
    return real_get(Bucket=Bucket, Key=Key, Range=Range)


s3.get_object = dropping
again = tempfile.mkdtemp()
assert mirror.pull("hub/models--a--b", again)["bytes"] == len(BIG) + 4 and flaky["left"] == 0
assert open(os.path.join(again, "blobs/abc"), "rb").read() == BIG
flaky["left"] = 5  # more drops than retries: the restore is given up
gave_up = tempfile.mkdtemp()
assert mirror.pull("hub/models--a--b", gave_up) is None
assert not [k for k in tree(gave_up) if k.endswith(".part")] and "blobs/abc" not in tree(gave_up)
s3.get_object = real_get

# ---- a body that ends early is a failed part, not a file with a hole
class Short(io.BytesIO):
    def read(self, n=-1):
        return super().read(max(1, (n if n > 0 else 2) // 2)) if self.tell() < 2 else b""


def short_body(Bucket, Key, Range=None):
    out = real_get(Bucket=Bucket, Key=Key, Range=Range)
    return {"Body": Short(out["Body"].read())} if Range and Key.endswith("blobs/abc") else out


s3.get_object = short_body
holed = tempfile.mkdtemp()
assert mirror.pull("hub/models--a--b", holed) is None and "blobs/abc" not in tree(holed)
s3.get_object = real_get

# ---- a write the disk only half takes is finished, not skipped
real_pwrite = os.pwrite
os.pwrite = lambda fd, data, pos: real_pwrite(fd, bytes(data[:max(1, len(data) // 2)]), pos)
halves = tempfile.mkdtemp()
assert mirror.pull("hub/models--a--b", halves) and open(os.path.join(halves, "blobs/abc"), "rb").read() == BIG
os.pwrite = real_pwrite

# ---- the object in the bucket is not the size the manifest lists: not restored
def other_size(Bucket, Key, Range=None):
    out = real_get(Bucket=Bucket, Key=Key, Range=Range)
    return dict(out, ContentRange=f"bytes 0-4/{len(BIG) + 50}") if Range and Key.endswith("blobs/abc") else out


s3.get_object = other_size
assert mirror.pull("hub/models--a--b", tempfile.mkdtemp()) is None
s3.get_object = real_get

# ---- a cancel raised by progress reaches the caller
class Cancelled(Exception):
    pass


def cancel(_msg):
    raise Cancelled()


try:
    mirror.pull("hub/models--a--b", tempfile.mkdtemp(), progress=cancel)
    raise AssertionError("the cancel was swallowed")
except Cancelled:
    pass
# ... also in the middle of the download: the running parts stop and no .part file stays
import threading  # noqa: E402
import time  # noqa: E402

slow_get = s3.get_object


def slow(Bucket, Key, Range=None):
    if Range:
        time.sleep(0.05)
    return slow_get(Bucket=Bucket, Key=Key, Range=Range)


calls = []


def cancel_later(msg):
    calls.append(msg)
    if len(calls) > 1:
        raise Cancelled()


s3.get_object = slow
mirror.THREADS, mid = 2, tempfile.mkdtemp()
real_wait = mirror.wait
mirror.wait = lambda fs, timeout, return_when: real_wait(fs, timeout=0.01, return_when=return_when)
try:
    mirror.pull("hub/models--a--b", mid, progress=cancel_later)
    raise AssertionError("the cancel was swallowed")
except Cancelled:
    pass
assert not [k for k in tree(mid) if k.endswith(".part")] and "blobs/abc" not in tree(mid), tree(mid).keys()
assert not [t for t in threading.enumerate() if t.name.startswith("ThreadPoolExecutor")], "download threads ended"
s3.get_object, mirror.THREADS, mirror.wait = slow_get, 32, real_wait

# ---- push again: the mirror already holds everything, so nothing is uploaded or rewritten (criterion 6)
s3.uploads.clear()
st = mirror.push("hub/models--a--b", src)
assert st["uploaded"] == 0 and s3.uploads == [], (st, s3.uploads)
# ... also when this machine has fewer files than the mirror (two machines share a unit)
os.remove(os.path.join(src, "blobs/empty"))
assert mirror.push("hub/models--a--b", src)["uploaded"] == 0 and s3.uploads == []
# a new or changed file: only that one is uploaded, and the manifest is rewritten after it
write(src, "blobs/new", b"12345678")
st = mirror.push("hub/models--a--b", src)
assert s3.uploads == ["models/hub/models--a--b/blobs/new", "models/hub/models--a--b/.mirror-manifest.json"], s3.uploads
merged = json.loads(s3.objects["models/hub/models--a--b/.mirror-manifest.json"])
assert merged["files"] == {"blobs/abc": len(BIG), "blobs/empty": 0, "refs/main": 4, "blobs/new": 8}, \
    "what another machine mirrored (blobs/empty) stays in the manifest"
assert merged["links"] == links

# ---- the manifest can't be read right now (throttling): nothing is uploaded and it is not rewritten,
# so what another machine mirrored is never dropped from it
before = dict(s3.objects)
s3.uploads.clear()
s3.throttle = ".mirror-manifest.json"
write(src, "blobs/newer", b"abc")
assert mirror.push("hub/models--a--b", src) is None and s3.uploads == [] and s3.objects == before
s3.throttle = None
os.remove(os.path.join(src, "blobs/newer"))
# a folder's own .mirror-manifest.json is never mirrored as a file
write(src, ".mirror-manifest.json", b"{}")
assert ".mirror-manifest.json" not in mirror.local_files(src)[0]
os.remove(os.path.join(src, ".mirror-manifest.json"))

# ---- an explicit file list (the trainer's folders hold more than one unit)
s3 = fresh()
tr = tempfile.mkdtemp()
write(tr, "raw.safetensors", b"raw-weights")
write(tr, "other/big.bin", b"not part of the unit")
mirror.push("trainer/krea2", tr, files=["raw.safetensors"])
assert json.loads(s3.objects["models/trainer/krea2/.mirror-manifest.json"]) == {"files": {"raw.safetensors": 11}, "links": {}}
assert "models/trainer/krea2/other/big.bin" not in s3.objects
# a listed file that is missing: nothing is mirrored
s3 = fresh()
assert mirror.push("trainer/krea2", tr, files=["raw.safetensors", "turbo.safetensors"]) is None and not s3.objects

# ---- a failed upload: no manifest, no exception (criterion 7)
s3 = fresh()
s3.fail_upload = "blobs/new"
assert mirror.push("hub/models--a--b", src) is None
assert "models/hub/models--a--b/.mirror-manifest.json" not in s3.objects
assert mirror.pull("hub/models--a--b", tempfile.mkdtemp()) is None  # so the half-uploaded unit is never used

# ---- push_async runs pushes one after another in the background
s3 = fresh()
mirror.push_async("hub/models--a--b", src)
mirror.push_async("trainer/krea2", tr, files=["raw.safetensors"])
mirror.wait_pushes()
assert {"models/hub/models--a--b/.mirror-manifest.json", "models/trainer/krea2/.mirror-manifest.json"} <= set(s3.objects)

# ---- busy(): pushes queued or running (the machines don't stop themselves meanwhile)
assert mirror.busy() is False
hold = threading.Event()
real_push = mirror.push
mirror.push = lambda *a, **k: hold.wait(5)
mirror.push_async("hub/models--a--b", src)
assert mirror.busy() is True
hold.set()
mirror.wait_pushes()
mirror.push = real_push
assert mirror.busy() is False

# ---- pull_all restores every unit with a manifest into <root>/<unit>, except skipped prefixes
root = tempfile.mkdtemp()
done = mirror.pull_all(root, skip=("trainer/",))
assert [u for u, _ in done] == ["hub/models--a--b"], done
assert os.path.exists(os.path.join(root, "hub/models--a--b/blobs/abc")) and not os.path.exists(os.path.join(root, "trainer"))

# ---- MODEL_MIRROR=0: no S3 call at all (criterion 9)
s3 = fresh()
os.environ["MODEL_MIRROR"] = "0"
assert mirror.pull("hub/models--a--b", tempfile.mkdtemp()) is None
assert mirror.push("hub/models--a--b", src) is None
mirror.push_async("hub/models--a--b", src)
mirror.wait_pushes()
assert mirror.pull_all(tempfile.mkdtemp()) == [] and s3.calls == 0, s3.calls
os.environ.pop("MODEL_MIRROR")

# ---- unit names can't leave models/ or the local folder
s3 = fresh()
for bad_unit in ("../code/worker", "/abs", "a/../../b", ""):
    assert mirror.pull(bad_unit, tempfile.mkdtemp()) is None and mirror.push(bad_unit, src) is None
assert s3.calls == 0
s3.objects["models/evil/.mirror-manifest.json"] = json.dumps({"files": {"../../escape": 1}, "links": {}}).encode()
s3.objects["models/evil/../../escape"] = b"x"
out = tempfile.mkdtemp()
assert mirror.pull("evil", os.path.join(out, "unit")) is None and not os.path.exists(os.path.join(out, "escape"))
s3.objects["models/evil2/.mirror-manifest.json"] = json.dumps({"files": {}, "links": {"../../l": "/etc/passwd"}}).encode()
assert mirror.pull("evil2", os.path.join(out, "unit2")) is None and not os.path.lexists(os.path.join(out, "l"))

# symlink chains: a link through another link, a file below a link, an absolute or empty target, a bad shape
def poisoned(name, man):
    s3.objects[f"models/{name}/.mirror-manifest.json"] = json.dumps(man).encode()
    where = os.path.join(out, name)
    assert mirror.pull(name, where) is None, name
    return where


poisoned("chain", {"files": {}, "links": {"a/b/c": "..", "r": "a/b/c/../.."}})
assert not os.path.lexists(os.path.join(out, "chain", "r"))
s3.objects["models/below/r/payload"] = b"x"
poisoned("below", {"files": {"r/payload": 1}, "links": {"r": "sub"}})
poisoned("abs", {"files": {}, "links": {"l": "/etc"}})
poisoned("up", {"files": {}, "links": {"l": ".."}})
for shape in ({"files": {}, "links": ["x"]}, {"files": [], "links": {}}, {"files": {"a": True}}, {"files": {"a": -1}},
              {"files": {"a\u0000b": 1}}, {"files": {"a": 1.5}}, ["files"], {"links": {}}):
    poisoned("shape", shape)
# a symlink already on the disk (left by anything) is not written through
pre = os.path.join(out, "pre")
os.makedirs(pre)
os.symlink(out, os.path.join(pre, "sub"))
s3.objects["models/pre/sub/planted"] = b"x"
s3.objects["models/pre/.mirror-manifest.json"] = json.dumps({"files": {"sub/planted": 1}, "links": {}}).encode()
assert mirror.pull("pre", pre) is None and not os.path.exists(os.path.join(out, "planted"))
s3.objects["models/pre/.mirror-manifest.json"] = json.dumps({"files": {}, "links": {"sub/l": "x"}}).encode()
assert mirror.pull("pre", pre) is None and not os.path.lexists(os.path.join(out, "l"))
# a restore larger than the free disk is refused before anything is written
s3.objects["models/huge/.mirror-manifest.json"] = json.dumps({"files": {"w": 10 ** 18}, "links": {}}).encode()
assert mirror.pull("huge", os.path.join(out, "huge")) is None and not os.path.exists(os.path.join(out, "huge", "w.part"))
# a manifest nobody could restore doesn't block a push: it is started again from this disk
s3.objects["models/trainer/krea2/.mirror-manifest.json"] = b"not json"
assert mirror.push("trainer/krea2", tr, files=["raw.safetensors"])["uploaded"] == 1
assert json.loads(s3.objects["models/trainer/krea2/.mirror-manifest.json"])["files"] == {"raw.safetensors": 11}
# a folder that can't be restored safely (a link out of it) is not mirrored at all
odd = tempfile.mkdtemp()
write(odd, "w", b"1")
os.symlink("/etc", os.path.join(odd, "l"))
assert mirror.push("odd", odd) is None and not [k for k in s3.objects if k.startswith("models/odd/")]

# ---- the command line used by first-boot setup: exit 0 when restored, 1 when not
env = dict(os.environ, MODEL_MIRROR="0")
r = subprocess.run([sys.executable, "-B", os.path.join(_stubs.ROOT, "worker", "mirror.py"), "pull", "x", out],
                   env=env, capture_output=True, text=True)
assert r.returncode == 1, r
r = subprocess.run([sys.executable, "-B", os.path.join(_stubs.ROOT, "worker", "mirror.py"), "pull-all", out,
                    "--skip", "trainer/"], env=env, capture_output=True, text=True)
assert r.returncode == 0, r
r = subprocess.run([sys.executable, "-B", os.path.join(_stubs.ROOT, "worker", "mirror.py"), "nonsense"],
                   env=env, capture_output=True, text=True)
assert r.returncode == 2, r

print("mirror tests passed")
