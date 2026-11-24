"""Trainer trial, last step: can the production render code use a musubi-tuner LoRA?

Loads the T2V pipeline exactly as the worker does (bf16 + Lightning), adds the trial
LoRA to the low-noise expert (transformer_2), and renders the same 2 s prompt with
and without it. Writes render.json with load details and timings.

    cd /opt/ai-video/worker && python render_test.py <trial out dir>
"""

import json
import os
import sys
import time

# The worker's models.py, not /opt/trial/models/ (the weights folder next to this script).
sys.path.insert(0, "/opt/ai-video/worker")
import models  # noqa: E402

out = sys.argv[1]
report = {}
prompt = "vgn_trial, a woman walking along a quiet beach at sunset"

t = time.time()
# Through the worker's cache, so generate_video below renders with this same pipe.
pipe = models._swap_to("t2v", models._load_t2v)
report["load_t2v_s"] = round(time.time() - t)


def lora_modules(name):
    return sum(1 for n, _ in pipe.transformer_2.named_modules() if n.endswith(f"lora_A.{name}"))


loaded = None
for candidate in ("trial_comfy.safetensors", "trial.safetensors"):
    path = os.path.join(out, candidate)
    if not os.path.exists(path):
        continue
    try:
        pipe.load_lora_weights(out, weight_name=candidate, adapter_name="trial", load_into_transformer_2=True)
        report["lora_modules_in_transformer_2"] = lora_modules("trial")
        loaded = candidate
        break
    except Exception as e:  # record why, try the next format
        report[f"load_error_{candidate}"] = f"{type(e).__name__}: {e}"[:500]
report["loaded_file"] = loaded
if not loaded or not report.get("lora_modules_in_transformer_2"):
    json.dump(report, open(os.path.join(out, "render.json"), "w"), indent=2)
    sys.exit("LoRA did not load into transformer_2")


def render(name, adapters, weights):
    pipe.set_adapters(adapters, adapter_weights=weights)
    t = time.time()
    models.generate_video(os.path.join(out, f"{name}.mp4"), prompt, seconds=2, seed=1)
    report[f"render_{name}_s"] = round(time.time() - t)


render("with_lora", ["lightning", "lightning_2", "trial"], [1.0, 1.0, 1.0])
render("baseline", ["lightning", "lightning_2"], [1.0, 1.0])
json.dump(report, open(os.path.join(out, "render.json"), "w"), indent=2)
print(json.dumps(report, indent=2))
