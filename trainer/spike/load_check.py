"""Spike part 2, last step: do the resumed and the extended LoRAs load in the render code?

Loads the T2V pipeline as the worker does (bf16 + Lightning), loads each converted LoRA
into the low-noise expert under its own adapter name, counts its modules, then renders
one 2 s clip with the extended LoRA. Writes part2.json.

    cd /opt/ai-video/worker && python load_check.py <part-2 out dir>
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
pipe = models._swap_to("t2v", models._load_t2v)

for name in ("resume", "extend"):
    try:
        pipe.load_lora_weights(out, weight_name=f"{name}_comfy.safetensors", adapter_name=name,
                               load_into_transformer_2=True)
        report[f"{name}_modules"] = sum(1 for n, _ in pipe.transformer_2.named_modules()
                                        if n.endswith(f"lora_A.{name}"))
    except Exception as e:
        report[f"{name}_error"] = f"{type(e).__name__}: {e}"[:500]

ok = all(report.get(f"{n}_modules") for n in ("resume", "extend"))
if report.get("extend_modules"):
    pipe.set_adapters(["lightning", "lightning_2", "extend"], adapter_weights=[1.0, 1.0, 1.0])
    t = time.time()
    models.generate_video(os.path.join(out, "extend.mp4"), "vgn_trial, a woman walking along a quiet beach at sunset",
                          seconds=2, seed=1)
    report["render_extend_s"] = round(time.time() - t)

json.dump(report, open(os.path.join(out, "part2.json"), "w"), indent=2)
print(json.dumps(report, indent=2))
sys.exit(0 if ok else "a LoRA did not load")
