"""Helios's own infer_helios.py with several LoRAs (runs in the Helios venv, started by worker/helios.py from
the Helios repository folder).

infer_helios.py loads one LoRA (--lora_path, as adapter "default") and switches it on at weight 1. This runs
that script unchanged, but when it switches its LoRA on, the job's other LoRAs are loaded too and every one
gets its own weight. HELIOS_LORAS: JSON [{"path", "weight"}, ...]; the first is the script's --lora_path.
"""

import json
import os
import sys


def patch(pipeline_cls, loras):
    """Make pipeline_cls.set_adapters(["default"], ...) (the script switching its one LoRA on) load the other
    LoRAs and set all the weights. Later calls pass through."""
    original = pipeline_cls.set_adapters

    def set_adapters(self, adapter_names, adapter_weights=None, *args, **kwargs):
        if list(adapter_names) != ["default"] or getattr(self, "_job_loras_set", False):
            return original(self, adapter_names, adapter_weights, *args, **kwargs)
        self._job_loras_set = True
        names, weights = ["default"], [float(loras[0]["weight"])]
        for n, lora in enumerate(loras[1:], 2):
            self.load_lora_weights(lora["path"], adapter_name=f"lora{n}")
            names.append(f"lora{n}")
            weights.append(float(lora["weight"]))
        print(json.dumps({"event": "helios_loras", "adapters": names, "weights": weights}), flush=True)
        return original(self, names, adapter_weights=weights)

    pipeline_cls.set_adapters = set_adapters


def main():
    import runpy

    # The Helios repository's package is called `helios`, like the worker's module: its folder (the working
    # directory) goes first, and the worker's folder (this file's) leaves the path.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [os.getcwd()] + [p for p in sys.path if p and os.path.abspath(p) != here]
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline

    patch(HeliosPipeline, json.loads(os.environ["HELIOS_LORAS"]))
    sys.argv[0] = "infer_helios.py"
    runpy.run_path("infer_helios.py", run_name="__main__")


if __name__ == "__main__":
    main()
