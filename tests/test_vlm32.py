"""The Qwen3-VL-32B critic (worker/vlm32.py, worker/vlm_judge.py) with the subprocess faked: the request it writes,
the GPU freed first, answers read back, failures raised. docs/specs/2026-11-28-realism-retry.md"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("worker")
import krea  # noqa: E402
import models  # noqa: E402
import vlm32  # noqa: E402
import vlm_judge  # noqa: E402

# ---- {stepN} placeholders
assert vlm_judge.fill("findings: {step0}; then {step1}; {step7}", ["A", "B"]) == "findings: A; then B; {step7}"

# ---- pinned, ungated, Apache-2.0 weights from the mirror or Hugging Face
assert vlm32.REVISION == "0cfaf48183f594c314753d30a4c4974bc75f3ccb" and vlm32.REPO == "Qwen/Qwen3-VL-32B-Instruct"
assert vlm32.UNIT == "qwen3-vl-32b/0cfaf48183f5" and vlm32.MODEL_DIR.endswith(vlm32.UNIT)

events = []
krea.ensure_env = lambda progress=None: events.append("env")
vlm32.ensure_weights = lambda progress=None: events.append("weights")
models.release_all = lambda: events.append("release")
models.check_cancel = lambda: None


class FakeProc:
    def __init__(self, cmd, answers, code=0, error=None):
        self.cmd, self.code = cmd, code
        req = json.load(open(cmd[-1]))
        events.append(("request", req))
        result = {"error": error} if error else {"answers": answers(req["steps"]), "load_seconds": 40.0, "seconds": 9.0}
        json.dump(result, open(req["result"], "w"))
        self.stdout = io.StringIO("".join(json.dumps({"step": i + 1}) + "\n" for i in range(len(req["steps"]))))

    def wait(self):
        return self.code

    def poll(self):
        return self.code


steps = [{"images": ["a.png"], "text": "judge"}, {"images": ["a.png"], "text": "rewrite {step0}", "sample": True, "seed": 2}]
vlm32.subprocess.Popen = lambda cmd, **kw: FakeProc(cmd, lambda st: [f"answer {i}" for i in range(len(st))])
progress = []
assert vlm32.critic(steps, progress.append) == ["answer 0", "answer 1"]
req = [e for e in events if isinstance(e, tuple)][-1][1]
assert req["model_dir"] == vlm32.MODEL_DIR and req["steps"] == steps
assert events[:3] == ["env", "weights", "release"], "Krea venv, weights, then the GPU freed before the 32B model loads"
assert progress[-1] == "AI-look critic: step 2/2"

# ---- the GPU must really be free after release_all(): else a clear error, no 32B load
vlm32._free_gpu_bytes = lambda: 20 * 2 ** 30
try:
    vlm32.critic(steps)
    raise AssertionError("must refuse")
except RuntimeError as e:
    assert "only 20 GiB of GPU memory free" in str(e), e
vlm32._free_gpu_bytes = lambda: None
assert "torchvision==0.29.1+cu130" in open(os.path.join(os.path.dirname(vlm32.__file__), "krea_requirements.txt")).read()

# ---- failures raise (the check turns them into an error, the job stays done)
for proc in (lambda cmd, **kw: FakeProc(cmd, None, code=1, error="OutOfMemoryError: CUDA out of memory"),
             lambda cmd, **kw: FakeProc(cmd, lambda st: ["only one"])):
    vlm32.subprocess.Popen = proc
    try:
        vlm32.critic(steps)
        raise AssertionError("must raise")
    except RuntimeError as e:
        assert "out of memory" in str(e) or "critic failed" in str(e), e

print("test_vlm32: ok")
