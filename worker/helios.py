"""Helios for the worker: long video in one take (docs/specs/2026-11-26-helios-default.md).

Helios (PKU-YuanGroup/Helios, Apache-2.0) generates a clip of any length in 33-frame chunks from one prompt,
from a first frame (image-to-video) or from text alone. It needs newer libraries than the Wan/Qwen venv, so it
runs in its own venv and its own process, like Krea 2: this module installs both on first use, then runs
Helios's own infer_helios.py with the settings that tested clean on 2026-11-26:

  640x384 at 24 fps (other sizes jump at the chunk boundaries), distilled 2-2-2 steps, no guidance, one Wan
  LoRA through its low-noise file, the attention library off (it has no build for the g7e's GPU).

First use: /opt/helios-venv from helios_requirements.txt, the Helios repository at a pinned commit in
/opt/helios, and ~80 GB of weights (from the model mirror when it has them, else Hugging Face). The repository
is third-party code that runs as root on the box: bump REPO_COMMIT only after reviewing the diff.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import imageio
import numpy as np

import krea
import models

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/PKU-YuanGroup/Helios.git"
REPO_COMMIT = "f1a8aadd7c42df29968b9ed73b5c3bc6ae2980c4"  # main on 2026-11-01; what the tests ran
MODEL_REPO = "BestWishYsh/Helios-Distilled"
MODEL_REVISION = "b991c0379a018f4de3227d95468237f56066f5bb"
VENV = "/opt/helios-venv"
REPO_DIR = "/opt/helios"
REQUIREMENTS = os.path.join(HERE, "helios_requirements.txt")
MODEL_DIR = f"/opt/models/helios/{MODEL_REVISION[:12]}"
MODEL_UNIT = f"helios/{MODEL_REVISION[:12]}"  # its copy in the model mirror (mirror.py)
# Not transformer_ode/ (57 GB): another sampler's weights, unused by this path.
MODEL_FILES = ["model_index.json", "modular_model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*",
               "transformer/*", "vae/*", "guider/*", "README.md", "LICENSE*"]
# Helios asks for Python 3.11; the box's own python is used when there is no 3.11.
PYTHON = os.environ.get("HELIOS_PYTHON") or shutil.which("python3.11") or sys.executable
SIZE, FPS, CHUNK = (640, 384), 24, 33  # the size it renders cleanly at; frames per second; frames per chunk
TAIL = 30  # lines of Helios's output kept for an error message


def frames_for(seconds):
    """Frames to ask Helios for: whole chunks that cover the length."""
    return -(-int(seconds) * FPS // CHUNK) * CHUNK


def _env_key():
    with open(REQUIREMENTS, "rb") as f:
        return hashlib.sha256(f.read() + REPO_COMMIT.encode() + PYTHON.encode()).hexdigest()[:16]


def ensure_env(progress=lambda msg: None):
    """The venv and the repository at the pinned commit; redone when the requirements or the pin change."""
    marker = os.path.join(VENV, ".ready")
    want = _env_key()
    if os.path.exists(marker) and open(marker).read().strip() == want:
        return
    progress("installing Helios (first use, a few minutes)")
    if not os.path.exists(os.path.join(VENV, "bin", "python")):
        subprocess.run([PYTHON, "-m", "venv", VENV], check=True)
    pip = os.path.join(VENV, "bin", "pip")
    subprocess.run([pip, "install", "-q", "--upgrade", "pip", "wheel"], check=True)
    subprocess.run([pip, "install", "-q", "-r", REQUIREMENTS], check=True)
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
    else:
        subprocess.run(["git", "-C", REPO_DIR, "fetch", "-q", "origin"], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", "-q", "--detach", REPO_COMMIT], check=True)
    with open(marker, "w") as f:
        f.write(want)


def ensure_weights(progress=lambda msg: None):
    def download():
        from huggingface_hub import snapshot_download

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=MODEL_DIR, allow_patterns=MODEL_FILES)

    krea._ensure(MODEL_UNIT, MODEL_DIR, download, progress, "downloading Helios (first use, ~80 GB)")


def generate(prompt, seconds, seed=None, image=None, lora=None, progress=lambda msg: None):
    """One Helios take: `seconds` of video at 640x384, 24 fps, as uint8 frames (seconds x 24 of them).
    image: a PIL first frame (image-to-video; fitted to 640x384), or None for text-to-video.
    lora: the path of one Wan LoRA file (the low-noise one), or None."""
    ensure_env(progress)
    ensure_weights(progress)
    models.release_all()  # Helios needs ~45 GB of the GPU to itself
    work = tempfile.mkdtemp(prefix="helios-")
    out = os.path.join(work, "out")
    cmd = [os.path.join(VENV, "bin", "python"), "-u", "infer_helios.py",
           "--base_model_path", MODEL_DIR, "--transformer_path", MODEL_DIR,
           "--sample_type", "i2v" if image is not None else "t2v",
           "--num_frames", str(frames_for(seconds)), "--fps", str(FPS),
           "--width", str(SIZE[0]), "--height", str(SIZE[1]), "--prompt", prompt,
           "--guidance_scale", "1.0", "--is_enable_stage2", "--pyramid_num_inference_steps_list", "2", "2", "2",
           "--is_amplify_first_chunk", "--disable_flash_attention", "--output_folder", out]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    if image is not None:
        first = os.path.join(work, "first_frame.png")
        models.fit(image, *SIZE).save(first)
        cmd += ["--image_path", first, "--image_noise_sigma_min", "0.111", "--image_noise_sigma_max", "0.135"]
    if lora:
        cmd += ["--lora_path", lora]
    progress(f"Helios: loading, then {seconds} s of video in one take")
    started = time.time()
    proc = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="1"))
    tail = []
    try:
        try:
            for line in proc.stdout:  # progress bars end in \r: only whole lines arrive here
                line = line.strip()
                if line:
                    tail = (tail + [line[-300:]])[-TAIL:]
                models.check_cancel()
            if proc.wait() != 0:
                raise RuntimeError(f"Helios failed (exit {proc.returncode}): " + " | ".join(tail[-8:]))
        finally:
            if proc.poll() is None:
                proc.kill()
        clips = sorted(f for f in os.listdir(out) if f.endswith(".mp4")) if os.path.isdir(out) else []
        if not clips:
            raise RuntimeError("Helios finished without a video: " + " | ".join(tail[-8:]))
        frames = np.stack([f for f in imageio.get_reader(os.path.join(out, clips[0]))])
        frames = frames[:int(seconds) * FPS]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({"event": "helios", "seconds": int(seconds), "frames": len(frames), "lora": bool(lora),
                      "first_frame": image is not None, "render_seconds": round(time.time() - started, 1),
                      "memory": next((l for l in reversed(tail) if "Max memory" in l), None)}), flush=True)
    return frames
