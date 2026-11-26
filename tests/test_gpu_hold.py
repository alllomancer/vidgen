"""Big models held on a large GPU (models.GPU_HOLD): on a GPU with room (>= 80 GB) a pipeline stays wholly
on the GPU instead of moving its parts on and off for every pass (docs/specs/2026-11-25-render-on-g7e.md,
criteria 1-3)."""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("worker")
os.environ.pop("GPU_HOLD", None)
import models  # noqa: E402

torch = sys.modules["torch"]
GB = 1024 ** 3


def hold_for(gpu_gb, env=None):
    """models._gpu_hold() on a GPU of that size (None: no GPU to ask), with GPU_HOLD set to env."""
    props = mock.MagicMock(total_memory=gpu_gb * GB) if gpu_gb is not None else None
    torch.cuda.get_device_properties = (lambda i: props) if props else mock.MagicMock(side_effect=RuntimeError("no GPU"))
    os.environ.pop("GPU_HOLD", None)
    if env is not None:
        os.environ["GPU_HOLD"] = env
    try:
        return models._gpu_hold()
    finally:
        os.environ.pop("GPU_HOLD", None)


# ---- on at 80 GB and above, off below, off when the GPU can't be asked; the env setting overrides (3)
assert hold_for(96) is True and hold_for(80) is True
assert hold_for(48) is False and hold_for(79) is False and hold_for(None) is False
assert hold_for(48, "1") is True and hold_for(96, "0") is False
assert hold_for(96, "yes") is True, "anything but 0 or 1 is ignored"
assert models.GPU_HOLD is False, "the test machine has no GPU: nothing changes by default"


class Pipe:
    def __init__(self):
        self.calls = []

    def enable_model_cpu_offload(self):
        self.calls.append("offload")

    def to(self, device):
        self.calls.append(("to", device))
        return self


# ---- off: offload hooks, as before (1)
models.KEEP_RAM_WEIGHTS = False
p = Pipe()
assert models._offload(p) is p and p.calls == ["offload"]

# ---- on: the whole pipeline goes to the GPU, no offload hooks (2)
models.GPU_HOLD = True
p = Pipe()
assert models._offload(p) is p and p.calls == [("to", "cuda")]

# ---- on: one big model at a time; loading one drops the others and the VLM first (2)
models._loaded.clear()
loads = []


def loader(name):
    return lambda: loads.append(name) or f"<{name}>"


models.RESIDENT_LIMIT = 3  # plenty of RAM: without holding, all three would stay loaded
assert models._swap_to("vlm", loader("vlm")) == "<vlm>"
assert models._swap_to("t2v", loader("t2v")) == "<t2v>" and list(models._loaded) == ["t2v"]
assert models._swap_to("i2v", loader("i2v")) == "<i2v>" and list(models._loaded) == ["i2v"]
assert models._swap_to("edit", loader("edit")) == "<edit>" and list(models._loaded) == ["edit"]
assert models._swap_to("edit", loader("edit")) == "<edit>" and loads == ["vlm", "t2v", "i2v", "edit"], "kept while it is the one in use"
assert models._swap_to("vlm", loader("vlm")) == "<vlm>" and list(models._loaded) == ["vlm"]
# an idle box doesn't load the other big models ahead: each load would push the held one off the GPU
models._loaded.clear()
models._swap_to("t2v", loader("t2v"))
models.PRELOAD_MODELS = True
assert models.preload_next() is None and list(models._loaded) == ["t2v"]

# ---- off again: the big models share RAM as before (1)
models.GPU_HOLD = False
models._loaded.clear()
for n in ("t2v", "i2v", "edit"):
    models._swap_to(n, loader(n))
assert list(models._loaded) == ["t2v", "i2v", "edit"]
models._swap_to("vlm", loader("vlm"))
assert list(models._loaded) == ["t2v", "i2v", "edit", "vlm"]
models._swap_to("t2v", loader("t2v"))
assert "vlm" not in models._loaded, "the VLM always leaves the GPU for another model"

print("gpu hold tests passed")
