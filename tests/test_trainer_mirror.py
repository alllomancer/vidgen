"""Trainer and the model mirror: Krea 2 training files from the mirror first, the rest from Hugging Face in
parallel at pinned revisions, fetched in the background from the start of a Krea job, timing in the result,
and the backfill at trainer start (docs/specs/2026-11-25-model-mirror.md, criteria 14-18)."""
import json
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("trainer", "worker")
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
import mirror  # noqa: E402
import trainer  # noqa: E402

assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
trainer.Progress = lambda job_id: (lambda *a, **k: None)
trainer._hf_token = lambda: "tok"

# every Krea training file is pinned to a commit
assert [(r, f) for r, f, _ in trainer.KREA_FILES] == [
    ("krea/Krea-2-Raw", "raw.safetensors"), ("krea/Krea-2-Turbo", "turbo.safetensors"),
    ("Comfy-Org/Qwen-Image", "split_files/vae/qwen_image_vae.safetensors"),
    ("Comfy-Org/Qwen3-VL", "text_encoders/qwen3vl_4b_bf16.safetensors")]
assert all(len(rev) == 40 and set(rev) <= set("0123456789abcdef") for _, _, rev in trainer.KREA_FILES)
assert trainer.KREA_RAW.endswith("/raw.safetensors") and trainer.KREA_TE.endswith("text_encoders/qwen3vl_4b_bf16.safetensors")

pulls, pushes, hf = [], [], []
in_mirror = []          # files a pull restores
hf_gate = threading.Event()
hf_gate.set()
hf_live = {"now": 0, "max": 0}
lock = threading.Lock()


def put(rel, data=b"w" * 10):
    p = os.path.join(trainer.KREA_DIR, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)


def pull(unit, local_dir, progress=None):
    pulls.append((unit, local_dir))
    if not in_mirror:
        return None
    for rel in in_mirror:
        put(rel)
    return {"unit": unit, "source": "mirror", "bytes": 10 * len(in_mirror), "seconds": 1.0, "mb_per_s": 900}


def hf_download(repo, filename, local_dir, token, revision):
    with lock:
        hf_live["now"] += 1
        hf_live["max"] = max(hf_live["max"], hf_live["now"])
    try:
        assert hf_gate.wait(10)
        time.sleep(0.05)  # long enough for the other files' downloads to overlap this one
        hf.append((repo, filename, token, revision))
        assert local_dir == trainer.KREA_DIR
        put(filename)
    finally:
        with lock:
            hf_live["now"] -= 1


mirror.pull, trainer._hf_download = pull, hf_download
mirror.push_async = lambda unit, local_dir, files=None: pushes.append((unit, local_dir, files))
ALL = [f for _, f, _ in trainer.KREA_FILES]


def reset(mirrored=()):
    trainer.KREA_DIR = tempfile.mkdtemp()
    in_mirror[:] = mirrored
    for l in (pulls, pushes, hf):
        l.clear()
    hf_live.update(now=0, max=0)
    trainer._KREA_FETCH.update(thread=None, result=None, error=None)


def make_job():
    p = dict(model="krea2", dataset="avatar-ava", lora="avatar-ava", version=1, trigger="ava", avatar="ava", steps=960)
    return trainer.Job({"job_id": "train1", "client": "m", "params": json.dumps(p)}, tempfile.mkdtemp())


# ---- the mirror has the unit: Hugging Face is never asked (criterion 14); timing is kept for the result (18)
reset(mirrored=ALL)
job = make_job()
job.krea_models()
assert pulls == [("trainer/krea2", trainer.KREA_DIR)] and hf == [] and pushes == [], (pulls, hf, pushes)
assert job.state["krea_download"] == {"source": "mirror", "bytes": 40, "seconds": job.state["krea_download"]["seconds"]}
# everything on disk: nothing is fetched and nothing recorded
pulls.clear()
job2 = make_job()
job2.krea_models()
assert pulls == [] and hf == [] and "krea_download" not in job2.state

# ---- not mirrored: all four from Hugging Face at once, each at its pin, then queued for the mirror (15)
reset()
job = make_job()
job.krea_models()
assert sorted(hf) == sorted((r, f, "tok", rev) for r, f, rev in trainer.KREA_FILES), hf
assert hf_live["max"] == 4, f"downloads overlapped {hf_live['max']} at a time"
assert pushes == [("trainer/krea2", trainer.KREA_DIR, ALL)], pushes
assert job.state["krea_download"]["source"] == "huggingface" and job.state["krea_download"]["bytes"] == 40

# ---- the mirror restored some files: only the others come from Hugging Face
reset(mirrored=ALL[:2])
job = make_job()
job.krea_models()
assert sorted(f for _, f, _, _ in hf) == sorted(ALL[2:]) and pushes, (hf, pushes)
assert job.state["krea_download"] == {"source": "mirror+huggingface", "bytes": 40,
                                      "seconds": job.state["krea_download"]["seconds"]}

# ---- no access to Krea 2 RAW: the license message, whatever else failed; nothing queued for the mirror
reset()


def gated(repo, filename, local_dir, token, revision):
    raise PermissionError("403 gated")


trainer._hf_download = gated
try:
    make_job().krea_models()
    raise AssertionError("no RAW access must fail")
except RuntimeError as e:
    assert "Krea-2-Raw" in str(e) and "license" in str(e), e
assert pushes == []
trainer._hf_download = hf_download

# ---- a Krea job starts the fetch before prepare; krea_models waits for it (16); one fetch at a time (17)
reset()
real_exists = os.path.exists
trainer.os.path.exists = lambda p: True if p.endswith("krea2_train_network.py") else real_exists(p)
hf_gate.clear()  # downloads hang until released
job = make_job()
order = []


def fetching():
    t = trainer._KREA_FETCH["thread"]
    return bool(t and t.is_alive())


for phase in ("prepare", "caption", "krea_env", "krea_cache", "krea_train", "krea_upload", "krea_link"):
    setattr(job, phase, (lambda name: lambda *a: order.append((name, fetching())))(phase))
job.krea_current = lambda: True
job.state["result"] = {"lora": "avatar-ava", "version": 1}
real_models = job.krea_models


def models_phase():
    # by now the fetch has been running since before `prepare`; a second request joins it
    first = trainer._KREA_FETCH["thread"]
    assert first.is_alive() and trainer.krea_fetch_start() is first
    threading.Timer(0.2, hf_gate.set).start()
    real_models()
    order.append(("krea_models", fetching()))


job.krea_models = models_phase
job.run_all()
assert order[0] == ("prepare", True), order  # the fetch is already running
assert [o[1] for o in order[:4]] == [True, True, True, False], order  # ... until krea_models has waited for it
assert [o[0] for o in order] == ["prepare", "caption", "krea_env", "krea_models", "krea_cache", "krea_train",
                                 "krea_upload", "krea_link"], order
assert len(pulls) == 1 and len(hf) == 4 and job.state["krea_download"]["source"] == "huggingface"
assert all(real_exists(os.path.join(trainer.KREA_DIR, f)) for f in ALL)

# a redelivered job whose models phase is done doesn't start a fetch; nor does a trainer without Krea support
reset()
job = make_job()
job.state["done"].append("krea_models")
job.krea_prefetch()
assert trainer._KREA_FETCH["thread"] is None and pulls == []
trainer.os.path.exists = real_exists
make_job().krea_prefetch()
assert trainer._KREA_FETCH["thread"] is None and pulls == []

# an error in the background fetch is raised by krea_models, and the next job tries again
reset()
trainer._hf_download = gated
trainer.krea_fetch_start().join(10)
trainer._hf_download = hf_download
job = make_job()
job.krea_models()  # the failed attempt is not reported as this job's: it fetches again
assert len(hf) == 4 and job.state["krea_download"]["source"] == "huggingface"
reset()
trainer._hf_download = gated
job = make_job()
job.krea_prefetch = lambda: None
try:
    job.krea_models()
    raise AssertionError("the fetch error was lost")
except RuntimeError as e:
    assert "Krea-2-Raw" in str(e)
trainer._hf_download = hf_download

# a cancel while waiting for the fetch stops the job (the download itself goes on for the next job)
reset()
hf_gate.clear()


class Stop(Exception):
    pass


job = make_job()
job.progress = lambda *a, **k: (_ for _ in ()).throw(Stop()) if trainer._KREA_FETCH["thread"] else None
trainer.FETCH_POLL = 0.05
try:
    job.krea_models()
    raise AssertionError("the cancel was swallowed")
except Stop:
    pass
hf_gate.set()
trainer._KREA_FETCH["thread"].join(10)

# ---- the result carries the download timing (criterion 18)
uploaded, store = {}, {}


class S3:
    class exceptions:
        NoSuchKey = KeyError

    def get_object(self, Bucket, Key):
        raise KeyError(Key)

    def put_object(self, Bucket, Key, Body, ContentType=None):
        store[Key] = Body

    def upload_file(self, path, Bucket, Key, ExtraArgs=None):
        uploaded[Key] = True


trainer.s3 = S3()
job = make_job()
os.makedirs(os.path.join(job.jd, "out"), exist_ok=True)
open(os.path.join(job.jd, "out", "krea2_r0.safetensors"), "w").write("lora")
job.state.update(krea2={"final": "krea2_r0.safetensors"}, krea_download={"source": "mirror", "bytes": 40, "seconds": 1.0})
job.krea_upload()
assert job.state["result"]["krea_download"] == {"source": "mirror", "bytes": 40, "seconds": 1.0}
job = make_job()
os.makedirs(os.path.join(job.jd, "out"), exist_ok=True)
open(os.path.join(job.jd, "out", "krea2_r0.safetensors"), "w").write("lora")
job.state.update(krea2={"final": "krea2_r0.safetensors"})
job.krea_upload()
assert "krea_download" not in job.state["result"], "no download, nothing reported"

# ---- trainer start: what is on this disk is queued for the mirror
reset()
root = tempfile.mkdtemp()
trainer.MODELS, trainer.KREA_DIR = root, os.path.join(root, "krea2")
hub = os.path.join(root, "hf", "hub", "models--q")
os.makedirs(os.path.join(hub, "blobs"))
open(os.path.join(hub, "blobs", "aa"), "w").write("w")
trainer.mirror_backfill(hf_home=os.path.join(root, "hf"))
assert pushes == [("trainer/wan", root, trainer.WAN_FILES), ("trainer/krea2", trainer.KREA_DIR, ALL),
                  ("hub/models--q", hub, None)], pushes
assert trainer.WAN_FILES == ["split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors",
                             "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors",
                             "split_files/vae/wan_2.1_vae.safetensors", "models_t5_umt5-xxl-enc-bf16.pth"]
# an unfinished caption-model download is not mirrored; a mirror problem never stops the trainer
open(os.path.join(hub, "blobs", "bb.incomplete"), "w").write("w")
pushes.clear()
trainer.mirror_backfill(hf_home=os.path.join(root, "hf"))
assert [u for u, _, _ in pushes] == ["trainer/wan", "trainer/krea2"], pushes


def broken(*a, **k):
    raise OSError("disk")


mirror.push_async = broken
trainer.mirror_backfill(hf_home=os.path.join(root, "hf"))

print("trainer mirror tests passed")
