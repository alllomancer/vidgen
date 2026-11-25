"""Benchmark one Wan2.2 T2V 5 s segment the way the render worker renders it (worker/models.py _load_wan:
fp32 VAE with tiling/slicing, bf16 experts, Lightning LoRAs, 4 steps, no CFG), with the two experts either
swapped on and off the GPU (--mode offload, what the worker does on 48 GB) or kept there (--mode resident,
possible on 96 GB). See docs/specs/2026-11-25-g7e-benchmark.md.

    python bench_wan.py --model PATH_OR_REPO --mode resident|offload --out result.json [--runs 3]
"""
import argparse
import json
import statistics
import time

import torch
from diffusers import AutoencoderKLWan, WanPipeline

LIGHTNING_REPO = "lightx2v/Wan2.2-Lightning"
LIGHTNING_REVISION = "18bccf8884ec0a078eed79785eb4ef13ea16ce1e"  # worker/models.py
LIGHTNING_DIR = "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1"
PROMPT = ("A woman in a green raincoat walks along a wet city street at dusk, shop lights reflecting on the "
          "pavement, steady camera, natural light, photorealistic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=("resident", "offload"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--frames", type=int, default=81)
    a = ap.parse_args()

    result = {"mode": a.mode, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "gpu_gb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1),
              "size": [a.width, a.height, a.frames], "runs": []}
    t0 = time.time()
    try:
        vae = AutoencoderKLWan.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.float32)
        vae.enable_tiling()
        vae.enable_slicing()
        pipe = WanPipeline.from_pretrained(a.model, vae=vae, torch_dtype=torch.bfloat16)
        for name, target in (("high_noise_model", "lightning"), ("low_noise_model", "lightning_2")):
            pipe.load_lora_weights(LIGHTNING_REPO, revision=LIGHTNING_REVISION, adapter_name=target,
                                   weight_name=f"{LIGHTNING_DIR}/{name}.safetensors",
                                   load_into_transformer_2=target == "lightning_2")
        pipe.set_adapters(["lightning", "lightning_2"], adapter_weights=[1.0, 1.0])
        if a.mode == "offload":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        result["load_seconds"] = round(time.time() - t0, 1)

        ends = []

        def on_step(p, step, timestep, kw):
            ends.append(time.time())
            return kw

        for i in range(a.runs + 1):  # the first is a warm-up and is reported apart
            del ends[:]
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            frames = pipe(prompt=PROMPT, width=a.width, height=a.height, num_frames=a.frames,
                          num_inference_steps=4, guidance_scale=1.0, guidance_scale_2=1.0,
                          generator=torch.Generator("cpu").manual_seed(7), callback_on_step_end=on_step).frames[0]
            marks = [start] + ends + [time.time()]
            run = {"steps": [round(b - x, 2) for x, b in zip(marks[:-2], marks[1:-1])],
                   "finish": round(marks[-1] - marks[-2], 2), "seconds": round(marks[-1] - start, 2),
                   "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1), "frames": len(frames)}
            print(json.dumps({"run": i, **run}), flush=True)
            if i == 0:
                result["warmup"] = run
            else:
                result["runs"].append(run)
        result["median_seconds"] = statistics.median(r["seconds"] for r in result["runs"])
        result["median_steps_seconds"] = statistics.median(sum(r["steps"]) for r in result["runs"])
        result["median_finish_seconds"] = statistics.median(r["finish"] for r in result["runs"])
    except Exception as e:  # e.g. out of memory in resident mode: reported, not hidden
        result["error"] = f"{type(e).__name__}: {str(e)[:500]}"
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"done": a.mode, "median_seconds": result.get("median_seconds"), "error": result.get("error")}),
          flush=True)


if __name__ == "__main__":
    main()
