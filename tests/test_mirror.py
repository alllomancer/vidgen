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


class S3:
    """In-memory bucket with the calls the mirror makes; counts them."""

    def __init__(self):
        self.objects, self.gets, self.uploads, self.calls, self.fail_upload, self.deny = {}, [], [], 0, None, False

    def get_object(self, Bucket, Key, Range=None):
        self.calls += 1
        if self.deny:
            raise PermissionError("AccessDenied")
        if Key not in self.objects:
            raise KeyError(Key)
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
assert s3.uploads[-1] == "models/hub/models--a--b/manifest.json", s3.uploads
assert sorted(s3.uploads[:-1]) == ["models/hub/models--a--b/blobs/abc", "models/hub/models--a--b/blobs/empty",
                                   "models/hub/models--a--b/refs/main"]
man = json.loads(s3.objects["models/hub/models--a--b/manifest.json"])
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
assert st["bytes"] == 0 and [k for k, _ in s3.gets] == ["models/hub/models--a--b/manifest.json"], s3.gets
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
assert s3.uploads == ["models/hub/models--a--b/blobs/new", "models/hub/models--a--b/manifest.json"], s3.uploads
assert "blobs/new" in json.loads(s3.objects["models/hub/models--a--b/manifest.json"])["files"]

# ---- an explicit file list (the trainer's folders hold more than one unit)
s3 = fresh()
tr = tempfile.mkdtemp()
write(tr, "raw.safetensors", b"raw-weights")
write(tr, "other/big.bin", b"not part of the unit")
mirror.push("trainer/krea2", tr, files=["raw.safetensors"])
assert json.loads(s3.objects["models/trainer/krea2/manifest.json"]) == {"files": {"raw.safetensors": 11}, "links": {}}
assert "models/trainer/krea2/other/big.bin" not in s3.objects
# a listed file that is missing: nothing is mirrored
s3 = fresh()
assert mirror.push("trainer/krea2", tr, files=["raw.safetensors", "turbo.safetensors"]) is None and not s3.objects

# ---- a failed upload: no manifest, no exception (criterion 7)
s3 = fresh()
s3.fail_upload = "blobs/new"
assert mirror.push("hub/models--a--b", src) is None
assert "models/hub/models--a--b/manifest.json" not in s3.objects
assert mirror.pull("hub/models--a--b", tempfile.mkdtemp()) is None  # so the half-uploaded unit is never used

# ---- push_async runs pushes one after another in the background
s3 = fresh()
mirror.push_async("hub/models--a--b", src)
mirror.push_async("trainer/krea2", tr, files=["raw.safetensors"])
mirror.wait_pushes()
assert {"models/hub/models--a--b/manifest.json", "models/trainer/krea2/manifest.json"} <= set(s3.objects)

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
s3.objects["models/evil/manifest.json"] = json.dumps({"files": {"../../escape": 1}, "links": {}}).encode()
s3.objects["models/evil/../../escape"] = b"x"
out = tempfile.mkdtemp()
assert mirror.pull("evil", os.path.join(out, "unit")) is None and not os.path.exists(os.path.join(out, "escape"))
s3.objects["models/evil2/manifest.json"] = json.dumps({"files": {}, "links": {"../../l": "/etc/passwd"}}).encode()
assert mirror.pull("evil2", os.path.join(out, "unit2")) is None and not os.path.lexists(os.path.join(out, "l"))

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
