"""Krea 2 Turbo for the worker: install its venv and weights on first use, then run krea_gen.py.

Krea 2 is text-to-image only (no reference images), so it is used where no avatar has to be
kept: avatar candidates, photos without an avatar, and the first frame of videos / story shots
without avatars (Wan image-to-video animates it). Its license (Krea 2 Community License) requires
a content filter: krea_gen.py checks every image and drops flagged ones; this module only passes
results on.

First use: creates /opt/krea-venv from krea_requirements.txt (a few minutes) and gets the weights
(~36 GB): from the model mirror in the bucket when it has them (mirror.py), else the gated download
with the Hugging Face token in SSM parameter /ai-video/hf_token. Both are kept on the instance's
disk, so later jobs only load the model.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import boto3

import mirror
import models

# Hugging Face's Xet transfers at full speed (more connections, bigger buffers): these are 30+ GB downloads.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = "/opt/krea-venv"
REQUIREMENTS = os.path.join(HERE, "krea_requirements.txt")
GENERATOR = os.path.join(HERE, "krea_gen.py")
MODEL_DIR = f"/opt/models/krea-2-turbo/{models.KREA_REVISION[:12]}"
FILTER_DIR = f"/opt/models/nsfw-filter/{models.KREA_FILTER_REVISION[:12]}"
# Their copies in the model mirror (mirror.py): restored from there first, Hugging Face second.
MODEL_UNIT = f"krea-2-turbo/{models.KREA_REVISION[:12]}"
FILTER_UNIT = f"nsfw-filter/{models.KREA_FILTER_REVISION[:12]}"
HF_TOKEN_PARAM = os.environ.get("HF_TOKEN_PARAM", "/ai-video/hf_token")
NSFW_THRESHOLD = 0.5  # license-required filter (Krea 2 Community License 4.2); a score is 0-1, so >= 1 disables it
# Skip the 26 GB single-file copy of the transformer and the sample images: the diffusers
# folders hold the same weights.
MODEL_FILES = ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*",
               "LICENSE*", "README.md"]
FILTER_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]


class FilteredOut(Exception):
    """Every image the job needed was flagged by the content filter."""


def _requirements_hash():
    with open(REQUIREMENTS, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def ensure_env(progress=lambda msg: None):
    """Create or update /opt/krea-venv when krea_requirements.txt changed."""
    marker = os.path.join(VENV, ".requirements")
    want = _requirements_hash()
    if os.path.exists(marker) and open(marker).read().strip() == want:
        return
    progress("installing Krea 2 (first use, a few minutes)")
    if not os.path.exists(os.path.join(VENV, "bin", "python")):
        subprocess.run([sys.executable, "-m", "venv", VENV], check=True)
    subprocess.run([os.path.join(VENV, "bin", "pip"), "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([os.path.join(VENV, "bin", "pip"), "install", "-q", "-r", REQUIREMENTS], check=True)
    with open(marker, "w") as f:
        f.write(want)


def _hf_token():
    ssm = boto3.client("ssm", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    try:
        return ssm.get_parameter(Name=HF_TOKEN_PARAM, WithDecryption=True)["Parameter"]["Value"]
    except Exception as e:
        raise RuntimeError(f"Krea 2 needs a Hugging Face token with access to {models.KREA_REPO} in SSM parameter "
                           f"{HF_TOKEN_PARAM} (see RUNBOOK): {type(e).__name__}") from e


def _ensure(unit, local_dir, download, progress, what):
    """A weights folder on disk: from the model mirror when it has it, else download() from Hugging Face
    (then queued for the mirror). The .complete marker is written only once the folder is whole. Logs where
    the weights came from and how fast (event model_download)."""
    done = os.path.join(local_dir, ".complete")
    if os.path.exists(done):
        return
    t0 = time.time()
    stats = mirror.pull(unit, local_dir, progress)
    if not stats:
        if what:
            progress(what)
        download()
        secs = round(time.time() - t0, 1)
        nbytes = sum(mirror.local_files(local_dir)[0].values())
        stats = {"unit": unit, "source": "huggingface", "bytes": nbytes, "seconds": secs,
                 "mb_per_s": round(nbytes / 1e6 / max(secs, 0.1))}
    open(done, "w").close()
    if stats["source"] != "mirror":
        mirror.push_async(unit, local_dir)
    print(json.dumps({"event": "model_download", **stats}), flush=True)


def ensure_weights(progress=lambda msg: None):
    def download():
        from huggingface_hub import snapshot_download

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        snapshot_download(models.KREA_REPO, revision=models.KREA_REVISION, local_dir=MODEL_DIR,
                          allow_patterns=MODEL_FILES, token=_hf_token())

    _ensure(MODEL_UNIT, MODEL_DIR, download, progress, "downloading Krea 2 Turbo (first use, ~36 GB)")
    ensure_filter(progress)


def ensure_filter(progress=lambda msg: None):
    """The content filter's weights (small; ungated)."""
    def download():
        from huggingface_hub import snapshot_download

        snapshot_download(models.KREA_FILTER_REPO, revision=models.KREA_FILTER_REVISION, local_dir=FILTER_DIR,
                          allow_patterns=FILTER_FILES)

    _ensure(FILTER_UNIT, FILTER_DIR, download, progress, None)


def flagged(score):
    return score >= NSFW_THRESHOLD


def score_images(paths, progress=lambda msg: None):
    """The content filter's NSFW score for each image (same filter and threshold as Krea images; no Krea
    model is loaded). Used to keep flagged photos out of avatar LoRA training sets. The filter runs on the
    CPU in its own process, so the worker's loaded models stay where they are (no reload afterwards)."""
    ensure_env(progress)
    ensure_filter(progress)
    import shutil

    work = tempfile.mkdtemp(prefix="krea-score-")
    try:
        req = {"mode": "score", "filter_dir": FILTER_DIR, "images": list(paths),
               "result": os.path.join(work, "result.json")}
        with open(os.path.join(work, "request.json"), "w") as f:
            json.dump(req, f)
        progress(f"content filter: {len(req['images'])} image{'s' * (len(req['images']) != 1)}")
        proc = subprocess.Popen([os.path.join(VENV, "bin", "python"), GENERATOR, os.path.join(work, "request.json")],
                                stdout=None, text=True)
        if proc.wait() != 0:
            raise RuntimeError(f"content filter failed (exit {proc.returncode}); see the worker log")
        with open(req["result"]) as f:
            scores = json.load(f).get("scores") or []
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if len(scores) != len(req["images"]) or not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in scores):
        raise RuntimeError(f"content filter returned {len(scores)} scores for {len(req['images'])} images")
    return scores


def has_word(text, word):
    """Whole-word, case-insensitive: 'al' is not in 'also'."""
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text, re.IGNORECASE) is not None


def with_triggers(prompt, loras):
    """The LoRAs' trigger words in front of the prompt, unless it already has them (as whole words)."""
    missing = [l["trigger"] for l in loras if l.get("trigger") and not has_word(prompt, l["trigger"])]
    return " ".join(missing + [prompt]).strip()


def generate(items, progress=lambda msg: None, loras=(), add_triggers=True):
    """items: [{"prompt", "width", "height", "seed"}]; loras: [{"path", "adapter", "weight", "trigger"}]
    (worker.prepare_krea_loras); add_triggers=False sends the prompts unchanged (raw photos). Returns
    (images, stats): images[i] is a PIL image, or None when the filter flagged it twice; stats has
    load/generate/filter seconds and the LoRA adapters."""
    from PIL import Image

    ensure_env(progress)
    ensure_weights(progress)
    models.release_all()  # Krea needs the GPU and ~40 GB of RAM to itself
    work = tempfile.mkdtemp(prefix="krea-")
    req = {"model_dir": MODEL_DIR, "filter_dir": FILTER_DIR, "threshold": NSFW_THRESHOLD,
           "result": os.path.join(work, "result.json"),
           "loras": [{k: l[k] for k in ("path", "adapter", "weight")} for l in loras],
           "items": [dict(it, prompt=with_triggers(it["prompt"], loras) if add_triggers else it["prompt"],
                          width=int(it["width"]) // 16 * 16, height=int(it["height"]) // 16 * 16,
                          out=os.path.join(work, f"{i:02d}.png")) for i, it in enumerate(items)]}
    with open(os.path.join(work, "request.json"), "w") as f:
        json.dump(req, f)
    progress(f"Krea 2: loading, then {len(items)} image{'s' * (len(items) > 1)}")
    proc = subprocess.Popen([os.path.join(VENV, "bin", "python"), GENERATOR, os.path.join(work, "request.json")],
                            stdout=subprocess.PIPE, text=True)
    done = 0
    try:
        for line in proc.stdout:  # one JSON line per finished image
            if line.startswith("{"):
                done += 1
                progress(f"Krea 2: image {done}/{len(items)}")
            models.check_cancel()
        if proc.wait() != 0:
            try:
                with open(req["result"]) as f:
                    reason = json.load(f).get("error")
            except (OSError, ValueError):
                reason = None
            raise RuntimeError(reason or f"Krea 2 generation failed (exit {proc.returncode}); see the worker log")
    finally:
        if proc.poll() is None:
            proc.kill()
    with open(req["result"]) as f:
        result = json.load(f)
    images = [None if r["filtered"] else Image.open(r["out"]).convert("RGB") for r in result["items"]]
    stats = {"load_seconds": result["load_seconds"],
             "generate_seconds": round(sum(r["generate_seconds"] for r in result["items"]), 1),
             "filter_seconds": round(sum(r["filter_seconds"] for r in result["items"]), 2),
             "filtered": sum(r["filtered"] for r in result["items"]),
             "rerolled": sum(r["rerolled"] for r in result["items"]),
             "seeds": [r["seed"] for r in result["items"]], "loras": [l["adapter"] for l in loras]}
    print(json.dumps({"event": "krea", **stats}), flush=True)
    return images, stats
