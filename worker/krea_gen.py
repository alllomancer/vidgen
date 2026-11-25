"""Krea 2 Turbo text-to-image, with the content filter its license requires.

Runs in its own venv (/opt/krea-venv, worker/krea_requirements.txt): Krea 2 needs newer
diffusers/transformers than the Wan and Qwen pipelines are pinned to. The worker (krea.py) calls

    /opt/krea-venv/bin/python krea_gen.py REQUEST.json

Score mode (`krea.score_images`): {"mode": "score", "filter_dir", "images": [paths], "result"} -> {"scores"}.

REQUEST: {"model_dir", "filter_dir", "threshold", "loras": [{"path", "adapter", "weight"}],
"items": [{"prompt", "width", "height", "seed", "out"}]}
Writes REQUEST.json's "result" path: {"load_seconds", "items": [{"out", "seed", "filtered",
"nsfw_score", "rerolled", "generate_seconds", "filter_seconds"}]}, or {"error"} and exit 2 when
loading fails (e.g. a LoRA diffusers can't map). LoRAs change what is generated, never the filter:
every image is checked by the
filter (Falconsai/nsfw_image_detection); a flagged image is re-rolled once with another seed and
dropped if flagged again. There is no switch to skip the filter. Prints one JSON line per image
as progress.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

STEPS, GUIDANCE = 8, 0.0      # Turbo: 8 steps, CFG off (model card)
REROLL_OFFSET = 1_000_003     # seed for the one re-roll of a flagged image


def prefetch(folder, threads=8, chunk=64 << 20):
    """Read the weights into the page cache in parallel (one reader gets ~1/3 of the disk's speed)."""
    files = [os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs
             if f.endswith((".safetensors", ".bin"))]
    jobs = [(p, off) for p in files for off in range(0, os.path.getsize(p), chunk)]

    def read(job):
        path, off = job
        with open(path, "rb") as f:
            return len(os.pread(f.fileno(), chunk, off))

    with ThreadPoolExecutor(threads) as pool:
        return sum(pool.map(read, jobs))


class LoraLoadError(RuntimeError):
    """A job's LoRA can't be used; the message is shown as the job's error (no internal detail)."""


def _peft_available():
    from diffusers.utils import is_peft_available
    return is_peft_available()


def _read_lora(path):
    from safetensors.torch import load_file  # tensors only, never pickle
    return load_file(path)


KOHYA_PREFIX = "lora_unet_"
KOHYA_MODULE = re.compile(r"^(?:txtfusion_(layerwise_blocks|refiner_blocks)_(\d+)|blocks_(\d+))_(attn|mlp)_([a-z]+)$")
# The standalone Linear layers musubi-tuner's networks.lora_krea2 also trains (all 264 Linear layers), by their
# reference names, which diffusers' Krea 2 converter maps (img_in, final_layer, time/text embeddings).
KOHYA_STANDALONE = {"first": "first", "last_linear": "last.linear", "tmlp_0": "tmlp.0", "tmlp_2": "tmlp.2",
                    "tproj_1": "tproj.1", "txtfusion_projector": "txtfusion.projector", "txtmlp_1": "txtmlp.1",
                    "txtmlp_3": "txtmlp.3"}


def kohya_to_reference(state_dict):
    """kohya/musubi-tuner Krea 2 LoRAs (`lora_unet_blocks_3_attn_wq.lora_down.weight`, `.lora_up`, `.alpha`)
    in the reference layout diffusers 0.41 converts (`diffusion_model.blocks.3.attn.wq.lora_A.weight`, `lora_B`),
    with alpha/rank folded into lora_B. Its own converter only knows the reference/ComfyUI/AI-Toolkit layouts and
    silently loads nothing from kohya keys. Any other layout is returned unchanged."""
    if not any(k.startswith(KOHYA_PREFIX) for k in state_dict):
        return state_dict
    out = {}
    for module in sorted({k.split(".", 1)[0] for k in state_dict}):
        flat = module[len(KOHYA_PREFIX):] if module.startswith(KOHYA_PREFIX) else None
        m = KOHYA_MODULE.match(flat) if flat else None
        if m:
            tf_block, tf_idx, idx, kind, sub = m.groups()
            dotted = f"txtfusion.{tf_block}.{tf_idx}.{kind}.{sub}" if tf_block else f"blocks.{idx}.{kind}.{sub}"
        elif flat in KOHYA_STANDALONE:
            dotted = KOHYA_STANDALONE[flat]
        else:
            raise ValueError(f"unknown Krea 2 LoRA module {module}")
        down, up = state_dict.get(f"{module}.lora_down.weight"), state_dict.get(f"{module}.lora_up.weight")
        if down is None or up is None:
            raise ValueError(f"Krea 2 LoRA module {module} needs both lora_down and lora_up")
        alpha = state_dict.get(f"{module}.alpha")
        scale = float(alpha) / down.shape[0] if alpha is not None else 1.0
        out[f"diffusion_model.{dotted}.lora_A.weight"] = down
        out[f"diffusion_model.{dotted}.lora_B.weight"] = up * scale
    return out


def apply_loras(pipe, loras):
    """Load each LoRA file as an adapter and set their weights. Files are read with safetensors and
    kohya/musubi ones converted first (kohya_to_reference); Krea2LoraLoaderMixin maps the rest. Needs peft
    in the Krea venv (krea_requirements.txt)."""
    if not loras:
        return pipe
    if not _peft_available():
        raise LoraLoadError("Krea 2 LoRAs need peft in the Krea venv (krea_requirements.txt); it isn't installed")
    lora = None
    try:
        for lora in loras:
            pipe.load_lora_weights(kohya_to_reference(_read_lora(lora["path"])), adapter_name=lora["adapter"])
        lora = None
        pipe.set_adapters([l["adapter"] for l in loras], adapter_weights=[float(l["weight"]) for l in loras])
    except Exception as e:  # a file in a layout diffusers can't map, or adapters it can't combine
        names = lora["adapter"] if lora else ", ".join(l["adapter"] for l in loras)
        print(f"LoRA load failed ({names}): {type(e).__name__}: {e}", file=sys.stderr, flush=True)  # worker log
        raise LoraLoadError(f"LoRA {names} could not be loaded ({type(e).__name__}); it may use a layout or "
                            "model version Krea 2 Turbo doesn't support") from e
    return pipe


def load_filter(filter_dir, device=0):
    from transformers import pipeline as hf_pipeline
    return hf_pipeline("image-classification", model=filter_dir, device=device)


def load(model_dir, filter_dir, loras=()):
    import torch
    from diffusers import Krea2Pipeline

    pipe = apply_loras(Krea2Pipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16), loras)
    try:
        pipe.to("cuda")  # ~36 GB: fits the 48 GB L40S when nothing else is loaded
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        pipe.enable_model_cpu_offload()
    return pipe, load_filter(filter_dir)


def nsfw_score(nsfw, image):
    """The filter's NSFW probability; a filter without an "nsfw" label is an error, never a pass."""
    for r in nsfw(image):
        if r["label"].lower() == "nsfw":
            return r["score"]
    raise ValueError("the content filter returned no 'nsfw' label")


def run_score(req):
    """Score mode: the filter alone over existing images (e.g. an avatar's training photos). On the CPU
    (a small ViT, well under a second per image), so the worker's loaded models keep the GPU."""
    from PIL import Image

    nsfw = load_filter(req["filter_dir"], device="cpu")
    scores = [round(nsfw_score(nsfw, Image.open(p).convert("RGB")), 4) for p in req["images"]]
    with open(req["result"], "w") as f:
        json.dump({"scores": scores}, f)


def main(request_path):
    with open(request_path) as f:
        req = json.load(f)
    if req.get("mode") == "score":
        return run_score(req)
    import torch

    t0 = time.time()
    prefetch(req["model_dir"])
    try:
        pipe, nsfw = load(req["model_dir"], req["filter_dir"], req.get("loras") or [])
    except LoraLoadError as e:  # reported by krea.generate as the job's error; anything else propagates
        with open(req["result"], "w") as f:
            json.dump({"error": str(e)}, f)
        sys.exit(2)
    out = {"load_seconds": round(time.time() - t0, 1), "items": []}
    threshold = float(req.get("threshold", 0.5))
    for item in req["items"]:
        rec = {"out": item["out"], "seed": int(item["seed"]), "filtered": False, "rerolled": False,
               "generate_seconds": 0.0, "filter_seconds": 0.0}
        for attempt in range(2):
            seed = int(item["seed"]) + (REROLL_OFFSET if attempt else 0)
            t = time.time()
            image = pipe(item["prompt"], width=int(item["width"]), height=int(item["height"]),
                         num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                         generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            rec["generate_seconds"] += round(time.time() - t, 2)
            t = time.time()
            score = nsfw_score(nsfw, image)
            rec["filter_seconds"] += round(time.time() - t, 3)
            rec["nsfw_score"], rec["seed"] = round(score, 4), seed
            if score < threshold:
                image.save(item["out"])
                break
            rec["rerolled"] = True
        else:
            rec["filtered"] = True
        out["items"].append(rec)
        print(json.dumps({k: rec[k] for k in ("out", "filtered", "generate_seconds", "filter_seconds")}), flush=True)
    with open(req["result"], "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main(sys.argv[1])
