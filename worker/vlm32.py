"""The AI-look critic on Qwen3-VL-32B: weights on first use (model mirror, else Hugging Face; ~66 GB, Apache-2.0,
ungated), run in the Krea venv by vlm_judge.py, which has the transformers it needs. Each call frees the GPU (the
32B model needs ~70 GB of it), loads the model once and answers every step of one try.
docs/specs/2026-11-28-realism-retry.md (critic upgrade)
"""
import json
import os
import subprocess
import tempfile

import krea
import models

REPO = "Qwen/Qwen3-VL-32B-Instruct"
REVISION = "0cfaf48183f594c314753d30a4c4974bc75f3ccb"  # 2025-10-21, Apache-2.0
MODEL_DIR = f"/opt/models/qwen3-vl-32b/{REVISION[:12]}"
UNIT = f"qwen3-vl-32b/{REVISION[:12]}"  # its copy in the model mirror
FILES = ["*.json", "*.safetensors", "merges.txt", "vocab.json", "tokenizer*"]
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_judge.py")
MIN_FREE_GPU = 72 * 2 ** 30  # 32B in bf16 (~66 GB) plus the pictures' tokens


def _free_gpu_bytes():
    """Free GPU memory after release_all(), or None where it can't be read."""
    try:
        import torch

        free = torch.cuda.mem_get_info()[0]
    except Exception:  # noqa: BLE001 - no CUDA here (tests): don't block
        return None
    return free if isinstance(free, int) else None


def ensure_weights(progress=lambda msg: None):
    def download():
        from huggingface_hub import snapshot_download

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        snapshot_download(REPO, revision=REVISION, local_dir=MODEL_DIR, allow_patterns=FILES)

    krea._ensure(UNIT, MODEL_DIR, download, progress, "downloading the AI-look critic (first use, ~66 GB)")


def critic(steps, progress=lambda msg: None):
    """Answers for steps ([{images, text, max_new_tokens?, sample?, seed?}]; text may use {stepN}), in order."""
    krea.ensure_env(progress)  # the Krea venv has transformers with Qwen3-VL
    ensure_weights(progress)
    models.release_all()  # the 32B model needs the GPU to itself; the render model loads again afterwards
    free = _free_gpu_bytes()
    if free is not None and free < MIN_FREE_GPU:
        raise RuntimeError(f"only {free / 2 ** 30:.0f} GiB of GPU memory free for the AI-look critic "
                           f"(needs ~{MIN_FREE_GPU / 2 ** 30:.0f}); something still holds the GPU")
    work = tempfile.mkdtemp(prefix="vlm-")
    req = {"model_dir": MODEL_DIR, "result": os.path.join(work, "result.json"), "steps": steps}
    with open(os.path.join(work, "request.json"), "w") as f:
        json.dump(req, f)
    progress("AI-look critic: loading")
    proc = subprocess.Popen([os.path.join(krea.VENV, "bin", "python"), SCRIPT, os.path.join(work, "request.json")],
                            stdout=subprocess.PIPE, text=True)
    try:
        for line in proc.stdout:
            if line.startswith("{"):
                progress(f"AI-look critic: step {json.loads(line).get('step')}/{len(steps)}")
            models.check_cancel()
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    try:
        with open(req["result"]) as f:
            result = json.load(f)
    except (OSError, ValueError):
        result = {}
    if code != 0 or not isinstance(result.get("answers"), list) or len(result["answers"]) != len(steps):
        raise RuntimeError(result.get("error") or f"the AI-look critic failed (exit {code}); see the worker log")
    print(json.dumps({"event": "critic", "load_seconds": result.get("load_seconds"),
                      "seconds": result.get("seconds"), "steps": len(steps)}), flush=True)
    return [str(a) for a in result["answers"]]
