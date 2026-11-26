"""Bake-off reference: today's long take (a chain of 5 s Wan 2.2 segments through the worker's own
generate_video), 30 s from the shared first frame with the shared beats. Run in /opt/ai-video/worker with
the worker venv:  python bake_ref.py REF.png PROMPTS.json OUT_DIR WIDTH HEIGHT TAG"""
import json
import os
import sys
import time

import torch
from PIL import Image

import models

ref, prompts, out, w, h, tag = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
os.makedirs(out, exist_ok=True)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
models.generate_video(os.path.join(out, f"{tag}.mp4"), prompts["beats"][0], seconds=30, seed=7, image=Image.open(ref),
                      fps=16, width=w, height=h, segment_prompts=prompts["beats"])
rec = {"tag": tag, "what": "today's chain (Wan 2.2 I2V, Lightning 4 steps, last-frame chaining)", "size": [w, h],
       "fps": 16, "seconds": round(time.time() - t0, 1), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1),
       "gpu_hold": models.GPU_HOLD}
json.dump(rec, open(os.path.join(out, f"{tag}.json"), "w"))
print(json.dumps(rec), flush=True)
