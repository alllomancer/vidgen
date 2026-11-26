"""End-to-end check of Helios video jobs through the worker's own job code (docs/specs/2026-11-26-helios-default.md,
criterion 14): the first frame (Krea with the avatar's LoRA), Helios with her Wan LoRA, the shot option's
frame effect, sound. Progress and uploads are kept local; everything else is the worker's real path.
  cd /opt/ai-video/worker && PYTHONPATH=. python -u validate_helios.py JOBS.json OUT_DIR [job ...]"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import helios
import models
import worker

jobs, out = json.load(open(sys.argv[1])), sys.argv[2]
only = sys.argv[3:]
os.makedirs(out, exist_ok=True)
worker.progress = lambda job_id, text: print(f"progress [{job_id}]: {text}", flush=True)
results = {"helios_python": helios.PYTHON, "gpu_hold": models.GPU_HOLD, "jobs": {}}
for name, params in jobs.items():
    if only and name not in only:
        continue

    def upload(path, client, job_id, fname, ctype, _name=name):
        shutil.copy(path, os.path.join(out, f"{_name}.{fname}"))
        return f"{_name}.{fname}"

    worker.upload = upload
    t0 = time.time()
    try:
        models.set_loras(worker.prepare_loras(params))
        res = worker.run_video({"client": "master", "job_id": f"validate-{name}"}, params, tempfile.mkdtemp())
        rec = {"ok": True, "seconds": round(time.time() - t0, 1), "result": res}
    except Exception as e:
        traceback.print_exc()
        rec = {"ok": False, "seconds": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {str(e)[:600]}"}
    finally:
        models.set_loras([])
    results["jobs"][name] = rec
    print(json.dumps({"job": name, **rec}), flush=True)
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
failed = [n for n, r in results["jobs"].items() if not r["ok"]]
print(json.dumps({"done": True, "failed": failed}), flush=True)
sys.exit(1 if failed else 0)
