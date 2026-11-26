"""Validate the render worker's stack on a g7e (docs/specs/2026-11-25-render-on-g7e.md, criterion 4): every
model through the worker's own loaders, with big models held on the GPU. Run from the worker folder with the
worker venv and its env (VIDEO_MODEL, I2V_MODEL, VLM_MODEL, EDIT_MODEL, EDIT_LIGHTNING, HF_HOME):

    cd /opt/ai-video/worker && /opt/ai-video/venv/bin/python -u /opt/trial/validate_g7e.py OUT_DIR [step ...]

Steps (default: all, in this order): t2v edit i2v sound vlm krea. Each writes its output to OUT_DIR and one
JSON line; results.json has them all. A failing step is recorded with its error and the rest still run.
"""
import json
import os
import sys
import time
import traceback

import torch

import models

OUT = sys.argv[1]
STEPS = sys.argv[2:] or ["t2v", "edit", "i2v", "sound", "vlm", "krea"]
PROMPT = ("A woman in a green raincoat walks along a wet city street at dusk, shop lights reflecting on the "
          "pavement, steady camera, natural light, photorealistic")
os.makedirs(OUT, exist_ok=True)
results = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "archs": torch.cuda.get_arch_list(),
           "gpu_gb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1),
           "gpu_hold": models.GPU_HOLD, "resident_limit": models.RESIDENT_LIMIT, "steps": {}}
print(json.dumps({k: v for k, v in results.items() if k != "steps"}), flush=True)


def step(name, fn):
    if name not in STEPS:
        return None
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        extra = fn() or {}
        rec = {"ok": True, "seconds": round(time.time() - t0, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1), **extra}
    except Exception as e:
        traceback.print_exc()
        rec = {"ok": False, "seconds": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {str(e)[:400]}"}
    results["steps"][name] = rec
    print(json.dumps({"step": name, **rec}), flush=True)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    return rec


def t2v():
    # Two passes of the production 5 s segment: the first includes the load, the second is the steady time.
    out = os.path.join(OUT, "t2v.mp4")
    models.generate_video(out, PROMPT, seconds=5, seed=7, fps=32, raw_path=os.path.join(OUT, "t2v_raw.mp4"))
    t1 = time.time()
    models.generate_video(os.path.join(OUT, "t2v_2.mp4"), PROMPT, seconds=5, seed=8, fps=16)
    return {"second_pass_seconds": round(time.time() - t1, 1), "bytes": os.path.getsize(out)}


def edit():
    from PIL import Image

    import imageio

    frame = Image.fromarray(imageio.get_reader(os.path.join(OUT, "t2v_raw.mp4")).get_data(0))
    frame.save(os.path.join(OUT, "ref.png"))
    key = models.make_keyframe([("ava", [frame], "a woman in a green raincoat")], "standing in a bright cafe, medium shot",
                               seed=7)
    key.save(os.path.join(OUT, "keyframe.png"))
    t1 = time.time()
    models.make_keyframe([("ava", [frame], "a woman in a green raincoat")], "on a ferry deck at dusk, medium shot",
                         seed=8).save(os.path.join(OUT, "keyframe_2.png"))
    return {"second_pass_seconds": round(time.time() - t1, 1), "size": list(key.size)}


def i2v():
    from PIL import Image

    key = Image.open(os.path.join(OUT, "keyframe.png"))
    out = os.path.join(OUT, "i2v.mp4")
    models.generate_video(out, "She looks up and smiles, steady camera", seconds=5, seed=7, image=key, fps=16)
    t1 = time.time()
    models.generate_video(os.path.join(OUT, "i2v_2.mp4"), "She turns and walks away, steady camera", seconds=5, seed=8,
                          image=key, fps=16)
    return {"second_pass_seconds": round(time.time() - t1, 1), "bytes": os.path.getsize(out)}


def sound():
    # Next to the Wan pipeline that is still held on the GPU.
    out = os.path.join(OUT, "sound.wav")
    models.generate_sound(os.path.join(OUT, "t2v_raw.mp4"), out, prompt="rain, footsteps on wet pavement", seconds=5, seed=7)
    return {"bytes": os.path.getsize(out), "loaded": list(models._loaded)}


def vlm():
    meta = models.describe_video(os.path.join(OUT, "t2v.mp4"))
    return {"title": str(meta.get("title"))[:80] if isinstance(meta, dict) else str(meta)[:80]}


def krea_step():
    import krea

    images, stats = krea.generate([{"prompt": "a lighthouse on a rocky coast at sunrise, photorealistic", "width": 896,
                                    "height": 1152, "seed": 7}])
    if images[0] is None:
        raise RuntimeError("the content filter dropped the image")
    images[0].save(os.path.join(OUT, "krea.png"))
    return {"krea": {k: stats[k] for k in ("load_seconds", "generate_seconds", "filter_seconds")}}


step("t2v", t2v)
step("edit", edit)
step("i2v", i2v)
step("sound", sound)
step("vlm", vlm)
step("krea", krea_step)
failed = [n for n, r in results["steps"].items() if not r["ok"]]
print(json.dumps({"done": True, "failed": failed}), flush=True)
sys.exit(1 if failed else 0)
