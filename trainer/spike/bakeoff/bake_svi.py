"""Bake-off A: SVI 2.0 Pro on Wan 2.2 I2V A14B, following the project's inference_svi_2.0_pro.py (same
pipeline, anchor image, previous-clip latents, overlap), with the step count, guidance and an optional
Lightning LoRA exposed. Run in the SVI repo with its venv:
  python bake_svi.py REF.png PROMPTS.json OUT_DIR TAG --clips 6 --steps 4 --cfg 1.0 [--lightning HIGH LOW]"""
import argparse
import json
import os
import time

import imageio
import numpy as np
import torch
from PIL import Image

from diffsynth.pipelines.wan_video_svi_pro import ModelConfig, WanVideoSviProPipeline

ap = argparse.ArgumentParser()
ap.add_argument("ref"); ap.add_argument("prompts"); ap.add_argument("out"); ap.add_argument("tag")
ap.add_argument("--clips", type=int, default=6)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--cfg", type=float, default=4.0)
ap.add_argument("--shift", type=float, default=5.0)
ap.add_argument("--lightning", nargs=2, default=None)
ap.add_argument("--width", type=int, default=832)
ap.add_argument("--height", type=int, default=480)
a = ap.parse_args()
L = "models/Stable-Video-Infinity/version-2.0/SVI_Wan2.2-I2V-A14B_%s_noise_lora_v2.0_pro.safetensors"
NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
OVERLAP, MOTION_FRAMES, MOTION_LATENT = 5, 4, 1  # README: --num_overlap_frame 5 --num_motion_latent 1; script default 4 motion frames

os.makedirs(a.out, exist_ok=True)
t0 = time.time()
pipe = WanVideoSviProPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda", model_configs=[
    ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
    ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
    ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
    ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu")],
    redirect_common_files=False)
pipe.load_lora(pipe.dit, L % "high", alpha=1)
pipe.load_lora(pipe.dit2, L % "low", alpha=1)
if a.lightning:
    pipe.load_lora(pipe.dit, a.lightning[0], alpha=1)
    pipe.load_lora(pipe.dit2, a.lightning[1], alpha=1)
load = round(time.time() - t0, 1)
prompts = json.load(open(a.prompts))["beats"]
anchor = Image.open(a.ref).convert("RGB").resize((a.width, a.height))
frames, current, prev, per_clip = [], anchor, None, []
torch.cuda.reset_peak_memory_stats()
t1 = time.time()
for i in range(a.clips):
    c0 = time.time()
    out = pipe(prompt=prompts[i % len(prompts)], negative_prompt=NEG, seed=i * 42, tiled=False, height=a.height,
               width=a.width, input_image=current, num_frames=81, anchor=anchor, prev_last_latent=prev,
               num_motion_latent=MOTION_LATENT, cfg_scale=a.cfg, num_inference_steps=a.steps, sigma_shift=a.shift)
    clip = out["video"]
    prev = out["prev_last_latent"]
    clip = clip if isinstance(clip, list) else [clip]
    frames.extend(clip if i == 0 else clip[OVERLAP:])
    current = clip[-MOTION_FRAMES:]
    per_clip.append(round(time.time() - c0, 1))
    print(json.dumps({"clip": i + 1, "seconds": per_clip[-1], "frames": len(frames)}), flush=True)
imageio.mimwrite(os.path.join(a.out, f"{a.tag}.mp4"), [np.asarray(f) for f in frames], fps=16, codec="libx264",
                 quality=8, macro_block_size=None)
rec = {"tag": a.tag, "what": f"SVI 2.0 Pro on Wan 2.2 I2V ({a.steps} steps, cfg {a.cfg}{', Lightning LoRA' if a.lightning else ''})",
       "size": [a.width, a.height], "fps": 16, "clips": a.clips, "video_seconds": round(len(frames) / 16, 1),
       "seconds": round(time.time() - t1, 1), "per_clip_seconds": per_clip, "load_seconds": load,
       "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
json.dump(rec, open(os.path.join(a.out, f"{a.tag}.json"), "w"))
print(json.dumps(rec), flush=True)
