"""Offloaded components keep their RAM copy (models.keep_ram_weights). Needs torch; skipped without it.
Run: make test, or with torch: python3 -B tests/test_offload.py."""
import importlib.util
import os
import sys

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("numpy") is None:
    print("offload tests skipped: torch is not installed here (the render box has it)")
    sys.exit(0)

try:  # before the stubs go in: accelerate looks for boto3 while it is imported
    from accelerate import cpu_offload_with_hook
    from accelerate.hooks import remove_hook_from_module
except ImportError:
    cpu_offload_with_hook = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("worker", keep=("torch", "torch.nn", "torch.nn.functional", "numpy"))
import torch  # noqa: E402

import models  # noqa: E402


def net():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.BatchNorm1d(16), torch.nn.Linear(16, 4))
    return m.eval()


def pointers(m):
    return [p.data_ptr() for p in m.parameters()]


x = torch.randn(3, 8)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
if device:  # moving needs a GPU (CUDA on the render box, Apple's on a Mac)
    plain = net()
    where = pointers(plain)
    plain.to(device), plain.to("cpu")
    assert pointers(plain) != where, "the usual .to(cpu) copies the weights back"

    m = models.keep_ram_weights(net())
    want, before, where = m(x), [p.detach().clone() for p in m.parameters()], pointers(m)
    assert m.to(device) is m and all(p.device.type == device for p in m.parameters())
    m.to(device)  # accelerate calls .to(device) before every forward: the copies must survive it
    assert m.to("cpu") is m and all(t.device.type == "cpu" for t in list(m.parameters()) + list(m.buffers()))
    assert all(torch.equal(a, b) for a, b in zip(m.parameters(), before)), "the same weights are back"
    assert pointers(m) == where, "the same memory: nothing was copied"
    assert torch.equal(m(x), want)
    m.to(torch.device(device), None), m.to("cpu", None)  # the way a diffusers pipeline calls it
    assert pointers(m) == where and torch.equal(m(x), want)
    m.to("cpu")  # already in RAM: nothing to do
    assert pointers(m) == where

    # a parameter added later (a LoRA) is kept as well; a deleted one is forgotten
    m.add_module("lora", torch.nn.Linear(4, 4))
    extra, extra_at = m.lora.weight.detach().clone(), m.lora.weight.data_ptr()
    m.to(device), m.to("cpu")
    assert torch.equal(m.lora.weight, extra) and m.lora.weight.data_ptr() == extra_at
    del m.lora
    m.to(device), m.to("cpu")
    assert pointers(m) == where and torch.equal(m(x), want)

    # a parameter swapped while on the GPU (another LoRA under the same name) is not mixed up with the old one
    m.add_module("lora", torch.nn.Linear(4, 4))
    m.to(device)
    m.lora = torch.nn.Linear(4, 4).to(device)
    swapped = m.lora.weight.detach().cpu().clone()
    m.to(device), m.to("cpu")
    assert m.lora.weight.device.type == "cpu" and torch.equal(m.lora.weight, swapped) and pointers(m)[:len(where)] == where
    del m.lora

    # a dtype change makes new weights: no stale copy comes back afterwards
    m.to(device), m.half(), m.to("cpu")
    assert all(p.dtype == torch.float16 and p.device.type == "cpu" for p in m.parameters())
    assert all(torch.equal(p, b.half()) for p, b in zip(m.parameters(), before))
    m.to(device), m.to(dtype=torch.float32), m.to("cpu")
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in m.parameters())

    if cpu_offload_with_hook:  # the real offload hooks, two models taking turns
        a, b = models.keep_ram_weights(net()), models.keep_ram_weights(net())
        plain_a, plain_b = a(x), b(x)
        where = pointers(a), pointers(b)
        _, hook_a = cpu_offload_with_hook(a, device)
        _, hook_b = cpu_offload_with_hook(b, device, prev_module_hook=hook_a)
        for _ in range(2):  # two "pipeline calls"
            out_a = a(x)
            assert next(a.parameters()).device.type == device
            out_b = b(x)  # b's turn: a goes back to RAM
            assert next(a.parameters()).device.type == "cpu" and next(b.parameters()).device.type == device
            hook_b.offload()
            assert (pointers(a), pointers(b)) == where, "back in RAM without a copy"
            assert torch.allclose(out_a.cpu(), plain_a, atol=1e-5) and torch.allclose(out_b.cpu(), plain_b, atol=1e-5)
            # diffusers takes the hooks off after every call and puts new ones on; accelerate then deletes
            # a `to` set on the module itself, so the weight keeping must not live there
            remove_hook_from_module(a, recurse=True), remove_hook_from_module(b, recurse=True)
            _, hook_a = cpu_offload_with_hook(a, device)
            _, hook_b = cpu_offload_with_hook(b, device, prev_module_hook=hook_a)
    print(f"offload: weight moves checked on {device}")
else:
    print("offload: no GPU here, so only the wiring is checked")

# _offload wraps the big components of a pipeline and switches model offload on
calls = []


class Pipe:
    def __init__(self):
        self.text_encoder, self.transformer, self.transformer_2, self.vae = net(), net(), None, net()

    def enable_model_cpu_offload(self):
        calls.append("offload")


pipe = models._offload(Pipe())
assert calls == ["offload"]
Plain = torch.nn.Sequential
assert all(type(m) is not Plain and isinstance(m, Plain) and type(m).__name__ == "Sequential"
           for m in (pipe.text_encoder, pipe.transformer)), "wrapped, under the same class name"
assert type(pipe.vae) is Plain, "the VAE is small: left alone"
models.KEEP_RAM_WEIGHTS = False
assert type(models._offload(Pipe()).transformer) is Plain, "KEEP_RAM_WEIGHTS=0: plain offload"
models.KEEP_RAM_WEIGHTS = True

print("offload tests passed")
