"""Render box and the model mirror: Krea weights restored from the mirror first, timing logged, and the
backfill of what is on disk at worker start (docs/specs/2026-11-25-model-mirror.md, criteria 11-13)."""
import contextlib
import io
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("worker")
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
os.environ.pop("MODEL_MIRROR", None)
import krea  # noqa: E402
import mirror  # noqa: E402
import models  # noqa: E402
import worker  # noqa: E402

# Hugging Face's Xet transfers run in high-performance mode on this box
assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"

hf_calls, pulls, pushes = [], [], []


def snapshot_download(repo, revision=None, local_dir=None, allow_patterns=None, token=None):
    hf_calls.append((repo, token))
    os.makedirs(os.path.join(local_dir, "transformer"), exist_ok=True)
    with open(os.path.join(local_dir, "transformer", "w.safetensors"), "wb") as f:
        f.write(b"x" * 1000)


sys.modules["huggingface_hub"] = types.SimpleNamespace(snapshot_download=snapshot_download)
krea._hf_token = lambda: "tok"
mirror.push_async = lambda unit, local_dir, files=None: pushes.append((unit, local_dir))
in_mirror = {}


def pull(unit, local_dir, progress=None):
    pulls.append(unit)
    if unit not in in_mirror:
        return None
    os.makedirs(local_dir, exist_ok=True)
    with open(os.path.join(local_dir, "restored"), "wb") as f:
        f.write(b"m")
    return {"unit": unit, "source": "mirror", "bytes": in_mirror[unit], "seconds": 2.0, "mb_per_s": 500}


mirror.pull = pull


def run(fn):
    """fn(), with the JSON lines it printed."""
    out = io.StringIO()
    said = []
    with contextlib.redirect_stdout(out):
        fn(said.append)
    return [json.loads(l) for l in out.getvalue().splitlines() if l.startswith("{")], said


def fresh_dirs():
    root = tempfile.mkdtemp()
    krea.MODEL_DIR, krea.FILTER_DIR = os.path.join(root, "krea-2-turbo", "abc"), os.path.join(root, "nsfw-filter", "def")
    return root


assert krea.MODEL_UNIT == f"krea-2-turbo/{models.KREA_REVISION[:12]}"
assert krea.FILTER_UNIT == f"nsfw-filter/{models.KREA_FILTER_REVISION[:12]}"

# ---- the mirror has the weights: no Hugging Face download, marker written, timing logged (criterion 11)
fresh_dirs()
in_mirror.update({krea.MODEL_UNIT: 36_000_000_000, krea.FILTER_UNIT: 350_000_000})
lines, said = run(krea.ensure_weights)
assert hf_calls == [] and pulls == [krea.MODEL_UNIT, krea.FILTER_UNIT], (hf_calls, pulls)
assert os.path.exists(os.path.join(krea.MODEL_DIR, ".complete")) and os.path.exists(os.path.join(krea.FILTER_DIR, ".complete"))
dl = [l for l in lines if l.get("event") == "model_download"]
assert [(l["unit"], l["source"], l["bytes"]) for l in dl] == [(krea.MODEL_UNIT, "mirror", 36_000_000_000),
                                                             (krea.FILTER_UNIT, "mirror", 350_000_000)], dl
assert all({"seconds", "mb_per_s"} <= set(l) for l in dl)
assert pushes == [], "what came from the mirror is not uploaded back"
# with the markers in place nothing is asked again
pulls.clear()
lines, _ = run(krea.ensure_weights)
assert pulls == [] and hf_calls == [] and lines == []

# ---- not in the mirror: Hugging Face as before, marker, a push queued, source logged (criterion 12)
fresh_dirs()
in_mirror.clear()
lines, said = run(krea.ensure_weights)
assert [r for r, _ in hf_calls] == [models.KREA_REPO, models.KREA_FILTER_REPO] and hf_calls[0][1] == "tok", hf_calls
assert any("downloading Krea 2 Turbo" in s for s in said), said
assert os.path.exists(os.path.join(krea.MODEL_DIR, ".complete"))
assert pushes == [(krea.MODEL_UNIT, krea.MODEL_DIR), (krea.FILTER_UNIT, krea.FILTER_DIR)], pushes
dl = [l for l in lines if l.get("event") == "model_download"]
assert [(l["unit"], l["source"], l["bytes"]) for l in dl] == [(krea.MODEL_UNIT, "huggingface", 1000),
                                                             (krea.FILTER_UNIT, "huggingface", 1000)], dl

# ---- a failed Hugging Face download leaves no marker and queues nothing
fresh_dirs()
pushes.clear()


def boom(*a, **k):
    raise RuntimeError("401")


sys.modules["huggingface_hub"] = types.SimpleNamespace(snapshot_download=boom)
try:
    krea.ensure_weights()
    raise AssertionError("the download error was swallowed")
except RuntimeError:
    pass
assert not os.path.exists(os.path.join(krea.MODEL_DIR, ".complete")) and pushes == []

# ---- worker start: everything complete on disk is queued for the mirror (criterion 13)
root = fresh_dirs()


def put(rel, data=b"w"):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)


models.BF16_DIR, models.MMAUDIO_DIR = os.path.join(root, "bf16"), os.path.join(root, "mmaudio")
models.VIDEO_MODEL, models.I2V_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
put("bf16/Wan2.2-T2V-A14B-Diffusers/model_index.json")
put("bf16/Wan2.2-I2V-A14B-Diffusers.tmp/transformer/half")       # a conversion that never finished
put("hub/models--Qwen--Qwen2.5-VL-7B-Instruct/blobs/aa")
put("hub/models--Qwen--Qwen-Image-Edit-2511/blobs/bb")
put("hub/models--org--half/blobs/cc.incomplete")                 # a download that never finished
put("hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/blobs/fp32")  # fp32 source: the bf16 copy is mirrored instead
put("hub/models--Wan-AI--Wan2.2-I2V-A14B-Diffusers/blobs/fp32")
put("hub/.locks/x.lock")
put("mmaudio/weights/mmaudio_large_44k_v2.pth")
put("krea-2-turbo/abc/.complete")
put("nsfw-filter/def/model.safetensors")                         # no marker: not complete
pushes.clear()
worker.mirror_backfill(hf_home=root)
assert sorted(pushes) == sorted([
    ("bf16/Wan2.2-T2V-A14B-Diffusers", os.path.join(root, "bf16/Wan2.2-T2V-A14B-Diffusers")),
    ("hub/models--Qwen--Qwen2.5-VL-7B-Instruct", os.path.join(root, "hub/models--Qwen--Qwen2.5-VL-7B-Instruct")),
    ("hub/models--Qwen--Qwen-Image-Edit-2511", os.path.join(root, "hub/models--Qwen--Qwen-Image-Edit-2511")),
    ("mmaudio", os.path.join(root, "mmaudio")),
    (krea.MODEL_UNIT, krea.MODEL_DIR),
]), pushes

# nothing on disk yet (a brand-new box), or an unreadable folder: no error, nothing queued
pushes.clear()
models.BF16_DIR, models.MMAUDIO_DIR = "/nonexistent/bf16", "/nonexistent/mmaudio"
krea.MODEL_DIR = krea.FILTER_DIR = "/nonexistent/k"
worker.mirror_backfill(hf_home="/nonexistent")
assert pushes == []


def broken(*a, **k):
    raise OSError("disk")


mirror.push_async = broken
models.BF16_DIR = os.path.join(root, "bf16")
worker.mirror_backfill(hf_home=root)  # a mirror problem never stops the worker

# the box doesn't stop itself while uploads to the mirror are running
mirror.busy = lambda: True
assert worker.mirror_busy() is True
mirror.busy = lambda: False
assert worker.mirror_busy() is False

print("worker mirror tests passed")
