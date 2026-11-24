"""Post-processing for rendered clips: RIFE frame interpolation (16 -> 32 fps) and a 2x
Real-ESRGAN upscale. Both work on uint8 frame arrays (N, H, W, 3) and run after Wan has
moved off the GPU (cpu offload), so they fit next to it on the 48 GB card.

Weights are downloaded once to /opt/models/enhance and checked against pinned hashes:
  RIFE 4.9 (rife49.pth, MIT), arch from Practical-RIFE via ComfyUI-Frame-Interpolation (MIT)
  RealESRGAN_x2plus.pth (BSD-3), loaded with spandrel (MIT), no basicsr needed
"""

import hashlib
import os
import urllib.request

import numpy as np
import torch

ENHANCE_DIR = "/opt/models/enhance"
WEIGHTS = {
    "rife49.pth": ("https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife49.pth",
                   "e55fd00f3cc184e3c65961f4bb827a9da022e78eed36b055242c0ac30000d533"),
    "RealESRGAN_x2plus.pth": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                              "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"),
}
RIFE_ARCH = "4.7"  # rife47/rife49 share the 4.7 architecture
_models = {}


def _weights(name):
    path = os.path.join(ENHANCE_DIR, name)
    url, sha = WEIGHTS[name]
    if not os.path.exists(path):
        os.makedirs(ENHANCE_DIR, exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        digest = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
        if digest != sha:
            os.remove(tmp)
            raise RuntimeError(f"{name}: sha256 {digest} does not match the pinned {sha}")
        os.replace(tmp, path)
    return path


def _rife():
    if "rife" not in _models:
        from rife_arch import IFNet

        net = IFNet(arch_ver=RIFE_ARCH)
        net.load_state_dict(torch.load(_weights("rife49.pth"), map_location="cpu", weights_only=False))
        _models["rife"] = net.half().eval()
    return _models["rife"]


def _upscaler():
    if "esrgan" not in _models:
        from spandrel import ModelLoader

        _models["esrgan"] = ModelLoader().load_from_file(_weights("RealESRGAN_x2plus.pth")).model.half().eval()
    return _models["esrgan"]


def _to_tensor(frames):
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float().div(255)


def _to_uint8(t):
    return (t.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy())


@torch.inference_mode()
def interpolate(frames, check=lambda: None):
    """2x frames: one RIFE middle frame between each pair (N -> 2N-1). uint8 in, uint8 out."""
    n, h, w, _ = frames.shape
    if n < 2:
        return frames
    net = _rife().to("cuda")
    try:
        out = [frames[0]]
        for i in range(n - 1):
            check()
            pair = _to_tensor(frames[i:i + 2]).to("cuda", torch.float16)
            t = torch.full((1, 1, 1, 1), 0.5, device="cuda", dtype=torch.float16)
            # IFNet pads to multiples of 64 itself; crop back to the clip size.
            mid = net(pair[0:1], pair[1:2], t, [8, 4, 2, 1], training=False, fastmode=True, ensemble=True)[:, :, :h, :w]
            out += [_to_uint8(mid.float())[0], frames[i + 1]]
        return np.stack(out)
    finally:
        net.to("cpu")
        torch.cuda.empty_cache()


@torch.inference_mode()
def upscale(frames, check=lambda: None):
    """2x every frame with Real-ESRGAN x2plus. uint8 in, uint8 out."""
    net = _upscaler().to("cuda")
    try:
        out = []
        for f in frames:
            check()
            x = _to_tensor(f[None]).to("cuda", torch.float16)
            out.append(_to_uint8(net(x).float())[0])
        return np.stack(out)
    finally:
        net.to("cpu")
        torch.cuda.empty_cache()
