"""Model wrappers. Only one model is resident on the GPU at a time; switching
unloads the other, which keeps the whole pipeline inside one 48 GB card.

Models:
  t2v    Wan2.2-T2V-A14B: text-to-video, and single frames (avatar candidates)
  i2v    Wan2.2-I2V-A14B: image-to-video from an avatar keyframe
  edit   Qwen-Image-Edit (EDIT_MODEL, 2509 or 2511): puts 1-3 avatars into each shot's scene (keyframe)
  audio  MMAudio large_44k_v2: ambient sound and Foley generated from a clip
  vlm    Qwen2.5-VL: video descriptions and story shot lists
  tts    Kokoro-82M: story narration, runs on CPU so it never swaps the GPU

Long-running calls check CANCEL between denoising steps, so a cancelled or
timed-out job stops within one step instead of running to the end.
"""

import gc
import json
import os
import re
import threading
import time

# Must be set before CUDA initialises; reduces fragmentation on the 48 GB card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image, ImageOps

VIDEO_MODEL = os.environ["VIDEO_MODEL"]
I2V_MODEL = os.environ.get("I2V_MODEL", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
VLM_MODEL = os.environ["VLM_MODEL"]
EDIT_MODEL = os.environ["EDIT_MODEL"]
VIDEO_FPS = 16  # Wan2.2 A14B native frame rate
# 81 frames, the A14B native length. 10 s (161 frames) runs out of memory on the 48 GB GPU.
MAX_CLIP_SECONDS = 5
# Wan2.2-Lightning (Apache-2.0): 4-step distilled LoRAs for both A14B experts, ~10x faster
# denoising. VIDEO_LIGHTNING=0 in /etc/ai-video.env switches back to the full 40-step model.
LIGHTNING = os.environ.get("VIDEO_LIGHTNING", "1") == "1"
LIGHTNING_REPO = "lightx2v/Wan2.2-Lightning"
LIGHTNING_REVISION = "18bccf8884ec0a078eed79785eb4ef13ea16ce1e"
TTS_SAMPLE_RATE = 24000  # Kokoro output

CANCEL = threading.Event()
_cancel_reason = {"text": "cancelled"}


class JobCancelled(Exception):
    pass


def request_cancel(reason):
    _cancel_reason["text"] = reason
    CANCEL.set()


def check_cancel():
    if CANCEL.is_set():
        raise JobCancelled(_cancel_reason["text"])


def _on_step(pipe, step, timestep, callback_kwargs):
    """diffusers callback_on_step_end: abort between denoising steps."""
    check_cancel()
    return callback_kwargs


def _resident_limit():
    """How many big models stay loaded at once. Each Wan/edit pipeline holds ~55-65 GB
    of CPU RAM between runs, so 3 fit on a 256 GB g6e.8xlarge and 1 on a 4xlarge."""
    if os.environ.get("RESIDENT_MODELS"):
        return max(1, int(os.environ["RESIDENT_MODELS"]))
    with open("/proc/meminfo") as f:
        total_gb = int(f.readline().split()[1]) / 1024 ** 2
    return 3 if total_gb >= 200 else 1


RESIDENT_LIMIT = _resident_limit()
# The VLM is loaded straight onto the GPU; every other model offloads to CPU RAM between
# runs, so only the VLM has to be evicted before something else runs.
GPU_RESIDENT = {"vlm"}
_loaded = {}  # name -> model, least recently used first


def _swap_to(name, loader):
    # The VLM sits on the GPU, so it can't stay loaded while anything else runs.
    evict = [n for n in _loaded if n in GPU_RESIDENT and n != name]
    if name not in _loaded and name not in GPU_RESIDENT:
        cpu_models = [n for n in _loaded if n not in GPU_RESIDENT]
        evict += cpu_models[:max(0, len(cpu_models) - RESIDENT_LIMIT + 1)]
    for n in evict:
        del _loaded[n]
    if evict:
        gc.collect()
        torch.cuda.empty_cache()
    if name in _loaded:
        _loaded[name] = _loaded.pop(name)  # mark as most recently used
        return _loaded[name]
    print(f"loading {name} (resident: {', '.join(_loaded) or 'none'}; limit {RESIDENT_LIMIT})", flush=True)
    _loaded[name] = loader()
    return _loaded[name]


def release_all():
    """Drop every loaded model (CPU and GPU), e.g. before Krea 2 runs in its own process."""
    _loaded.clear()
    gc.collect()
    torch.cuda.empty_cache()


# Krea 2 Turbo (krea.py / krea_gen.py, own venv): text-to-image for avatar candidates, photos without
# an avatar and first frames. Krea 2 Community License; gated (HF token in SSM). The license requires
# a content filter: Falconsai/nsfw_image_detection (Apache-2.0) checks every image.
KREA_REPO = "krea/Krea-2-Turbo"
KREA_REVISION = "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"
KREA_FILTER_REPO = "Falconsai/nsfw_image_detection"
KREA_FILTER_REVISION = "96cb0d0342c7afb80cab76ecc58b265fa44da256"


def _generator(seed):
    return torch.Generator("cuda").manual_seed(int(seed)) if seed is not None else None


def fit(image, width, height):
    """Scale and centre-crop to exactly width x height."""
    return ImageOps.fit(image.convert("RGB"), (width, height), Image.LANCZOS)


# ---------------------------------------------------------------- video

BF16_DIR = "/opt/models/bf16"


def bf16_path(repo):
    """Where convert_bf16.py keeps the local bf16 copy of a Wan pipeline repo."""
    return os.path.join(BF16_DIR, repo.split("/")[-1])


PREFETCH_THREADS = 8
PREFETCH_CHUNK = 64 << 20


def prefetch(folder, threads=PREFETCH_THREADS):
    """Read a model folder's weight files into the page cache with parallel readers.

    Diffusers maps the .safetensors files and pulls pages in as each expert moves to the GPU:
    one slow stream (~315 MB/s measured on the gp3 root disk). Several readers reach the
    disk's 1000 MB/s, so a cold expert (28.6 GB) takes ~30 s instead of ~90 s."""
    from concurrent.futures import ThreadPoolExecutor

    files = [os.path.join(d, f) for d, _, names in os.walk(folder) for f in names
             if f.endswith((".safetensors", ".bin", ".pth"))]
    jobs = [(f, off) for f in files for off in range(0, os.path.getsize(f), PREFETCH_CHUNK)]

    def read(job):
        path, off = job
        fd = os.open(path, os.O_RDONLY)
        try:
            return len(os.pread(fd, PREFETCH_CHUNK, off))
        finally:
            os.close(fd)

    t0 = time.time()
    with ThreadPoolExecutor(threads) as pool:
        total = sum(pool.map(read, jobs))
    secs = max(time.time() - t0, 0.001)
    print(f"prefetched {folder}: {total / 1e9:.1f} GB in {secs:.0f} s ({total / 1e6 / secs:.0f} MB/s)", flush=True)
    return total


def _model_folder(model):
    """Where a model's files are on disk: the path itself, or the Hugging Face cache snapshot of a repo
    id. None when it isn't downloaded yet (from_pretrained will fetch it; nothing to warm)."""
    if os.path.isdir(model):
        return model
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model, local_files_only=True)
    except Exception:  # not in the cache, or the cache can't be read: loading works without the warm-up
        return None


def prefetch_model(model):
    """prefetch() for a model given as a path or repo id; a warm-up only, so it never fails the load."""
    try:
        folder = _model_folder(model)
        return prefetch(folder) if folder else 0
    except OSError as e:
        print(f"prefetch of {model} skipped: {e}", flush=True)
        return 0


def _load_wan(pipeline_cls, repo, lightning_dir):
    """A Wan2.2 A14B pipeline: two 14B experts (~28 GB each in bf16), so components
    move to the GPU only while they run; the 128 GB of RAM holds the rest."""
    from diffusers import AutoencoderKLWan

    # The Hugging Face repos store the experts in fp32; prefer the local bf16 copy.
    local = bf16_path(repo)
    if os.path.exists(os.path.join(local, "model_index.json")):
        repo = local
        prefetch(local)  # warm the page cache in parallel before diffusers maps the files
    print(f"loading {repo}", flush=True)
    vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
    # Decoding all frames at 1280x704 in one pass overflows the GPU (the fp32 VAE is the peak).
    vae.enable_tiling()
    vae.enable_slicing()
    pipe = pipeline_cls.from_pretrained(repo, vae=vae, torch_dtype=torch.bfloat16)
    if LIGHTNING:
        # One LoRA per expert: high-noise into transformer, low-noise into transformer_2.
        for name, target in (("high_noise_model", "lightning"), ("low_noise_model", "lightning_2")):
            pipe.load_lora_weights(
                LIGHTNING_REPO, revision=LIGHTNING_REVISION, adapter_name=target,
                weight_name=f"{lightning_dir}/{name}.safetensors",
                load_into_transformer_2=target == "lightning_2",
            )
        pipe.set_adapters(["lightning", "lightning_2"], adapter_weights=[1.0, 1.0])
    pipe.enable_model_cpu_offload()
    return pipe


# User LoRAs (trained by the trainer, `--lora` on video/story jobs) for the current job:
# [{"adapter", "high", "low", "weight", "trigger"}], set by the worker around each job.
# They load next to Lightning: the high-noise file into `transformer`, the low-noise one
# into `transformer_2`, and stay loaded while the Wan model is resident and the next job
# asks for the same version.
_JOB_LORAS = []


def set_loras(loras):
    _JOB_LORAS[:] = loras


def _apply_loras(pipe):
    """Make the pipe's adapters match _JOB_LORAS: drop others, load missing, set weights."""
    want = {l["adapter"]: l for l in _JOB_LORAS}
    have = getattr(pipe, "_user_loras", set())
    drop = [name for adapter in have - set(want) for name in (adapter, adapter + "_2")]
    if drop:
        pipe.delete_adapters(drop)
    for adapter, lora in want.items():
        if adapter not in have:
            print(f"loading LoRA {adapter}", flush=True)
            try:
                pipe.load_lora_weights(lora["high"], adapter_name=adapter)
                pipe.load_lora_weights(lora["low"], adapter_name=adapter + "_2", load_into_transformer_2=True)
            except Exception as e:  # an uploaded file in a layout diffusers can't map
                for name in (adapter, adapter + "_2"):
                    try:
                        pipe.delete_adapters([name])
                    except Exception:
                        pass
                raise RuntimeError(f"LoRA {adapter} could not be loaded: {str(e)[:300]}. "
                                   "It may use a layout or model version this pipeline doesn't support.") from e
    pipe._user_loras = set(want)
    names, weights = (["lightning", "lightning_2"], [1.0, 1.0]) if LIGHTNING else ([], [])
    for adapter, lora in want.items():
        names += [adapter, adapter + "_2"]
        weights += [lora["weight"]] * 2
    if names:
        pipe.set_adapters(names, adapter_weights=weights)
    return pipe


def with_triggers(prompt):
    """The job's LoRA trigger words in front of the prompt, unless it already has them."""
    missing = [l["trigger"] for l in _JOB_LORAS
               if l.get("trigger") and l["trigger"].lower() not in prompt.lower()]
    return " ".join(missing + [prompt]).strip()


def _load_t2v():
    from diffusers import WanPipeline

    return _load_wan(WanPipeline, VIDEO_MODEL, "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1")


def _load_i2v():
    from diffusers import WanImageToVideoPipeline

    return _load_wan(WanImageToVideoPipeline, I2V_MODEL, "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1")


def _sampling():
    """Denoising settings: Lightning is distilled for 4 steps without CFG, so the
    negative prompt has no effect there; the full model uses 40 steps with CFG."""
    if LIGHTNING:
        return dict(num_inference_steps=4, guidance_scale=1.0, guidance_scale_2=1.0)
    # guidance_scale for the high-noise expert, guidance_scale_2 for the low-noise one.
    return dict(num_inference_steps=40, guidance_scale=4.0, guidance_scale_2=3.0)


# Appended to every clip prompt for a natural, true-to-life look. The camera defaults
# to a locked-off tripod unless the prompt asks for a camera move.
NATURAL_STYLE = ("Photorealistic, natural lighting, true-to-life colors, subtle natural motion, "
                 "real-world physics, shot on a 35mm lens, fine film grain.")
TRIPOD = "Static tripod shot, locked-off camera, no camera movement."
CAMERA_MOVE = re.compile(r"\b(handheld|pans?|panning|tilts?|tilting|dolly|tracking|push[- ]in|pull[- ]back|"
                         r"zooms?|zooming|drone|aerial|orbit(?:ing)?|crane|steadicam|following|follows)\b", re.I)
NATURAL_NEGATIVE = ("cartoon, anime, CGI, 3d render, plastic skin, oversaturated, blurry, distorted face, "
                    "deformed hands, extra fingers, text, watermark, flicker")
TRIPOD_NEGATIVE = "camera shake, shaky camera, camera movement, zoom"


def natural_prompts(prompt, negative_prompt=""):
    """(prompt, negative) with the natural-look base added; tripod unless a camera move is named."""
    tripod = not CAMERA_MOVE.search(prompt)
    prompt = prompt.strip()
    if prompt and prompt[-1] not in ".!?":
        prompt += "."
    prompt = " ".join(p for p in (TRIPOD if tripod else "", prompt, NATURAL_STYLE) if p)
    negative = ", ".join(n for n in (negative_prompt.strip(), NATURAL_NEGATIVE,
                                     TRIPOD_NEGATIVE if tripod else "") if n)
    return prompt, negative


SEGMENT_SECONDS = 5
MAX_TAKE_SECONDS = 60  # longer single takes chain 5 s segments (each continues from the last frame)
OUTPUT_FPS = (16, 32)  # 32 = RIFE doubles the frames, same duration


def _to_uint8_frames(frames):
    import numpy as np

    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0, 1) * 255).round().astype("uint8")
    return frames


# Text-encoder outputs of recent prompts. Variations and avatar candidates repeat one prompt, as do the
# segments of a long take without beats; a hit skips moving the ~11 GB text encoder to the GPU and back.
# Keyed per pipeline; LoRAs only touch the transformers, so they don't change the embeddings.
EMBED_CACHE_SIZE = 16
_embed_cache = {}  # (model, prompt, negative, cfg) -> (embeds, negative embeds or None) on the CPU, oldest first


def _prompt_embeds(name, pipe, prompt, negative_prompt, cfg):
    """`prompt_embeds`/`negative_prompt_embeds` kwargs for a Wan pipeline call, encoded once per prompt.
    The same encoding the pipeline does itself (512 tokens; the negative only with CFG)."""
    key = (name, prompt, negative_prompt, cfg)
    if key in _embed_cache:
        _embed_cache[key] = _embed_cache.pop(key)  # mark as most recently used
    else:
        with torch.no_grad():
            embeds, negative = pipe.encode_prompt(prompt=prompt, negative_prompt=negative_prompt,
                                                  do_classifier_free_guidance=cfg, max_sequence_length=512)
        _embed_cache[key] = (embeds.to("cpu"), negative.to("cpu") if negative is not None else None)
        for old in list(_embed_cache)[:-EMBED_CACHE_SIZE]:
            del _embed_cache[old]
    embeds, negative = _embed_cache[key]
    return dict(prompt_embeds=embeds.to("cuda"),
                negative_prompt_embeds=negative.to("cuda") if negative is not None else None)


def _render_segment(prompt, negative_prompt, num_frames, width, height, seed, image):
    """One Wan pass (<= 81 frames): T2V, or I2V from `image`. Returns uint8 (N, H, W, 3)."""
    sampling = _sampling()
    kwargs = dict(width=width, height=height, num_frames=num_frames, **sampling, generator=_generator(seed),
                  output_type="np", callback_on_step_end=_on_step)
    cfg = sampling["guidance_scale"] > 1.0
    if image is not None:
        i2v = _apply_loras(_swap_to("i2v", _load_i2v))
        kwargs.update(_prompt_embeds("i2v", i2v, prompt, negative_prompt, cfg))
        frames = i2v(image=fit(image, width, height), **kwargs).frames[0]
    else:
        t2v = _apply_loras(_swap_to("t2v", _load_t2v))
        kwargs.update(_prompt_embeds("t2v", t2v, prompt, negative_prompt, cfg))
        frames = t2v(**kwargs).frames[0]
    return _to_uint8_frames(frames)


# ---------------------------------------------------------------- colour matching (long takes)
# Each segment after the first is pulled towards segment 1's LAB mean/std, ramping up to COLOUR_STRENGTH
# on its last frame, before that frame conditions the next segment, so colour drift never accumulates.
COLOUR_STRENGTH = 0.7
COLOUR_MIN_GAIN, COLOUR_MAX_GAIN = 0.7, 1.4
COLOUR_MATCH_OVER_SECONDS = 10
_SRGB_TO_XYZ = ((0.4124564, 0.3575761, 0.1804375), (0.2126729, 0.7151522, 0.0721750),
                (0.0193339, 0.1191920, 0.9503041))
_D65 = (0.95047, 1.0, 1.08883)


def _rgb8_to_lab(frames):
    """uint8 RGB (..., 3) -> float32 CIE LAB (D65)."""
    import numpy as np

    rgb = np.asarray(frames, dtype=np.float32) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.asarray(_SRGB_TO_XYZ, dtype=np.float32).T / np.asarray(_D65, dtype=np.float32)
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def _lab_to_rgb8(lab):
    """float CIE LAB (D65) -> uint8 RGB, clipped to the sRGB gamut."""
    import numpy as np

    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    d = 6 / 29
    xyz = np.where(f > d, f ** 3, 3 * d * d * (f - 4 / 29)) * np.asarray(_D65, dtype=np.float32)
    lin = np.clip(xyz @ np.linalg.inv(np.asarray(_SRGB_TO_XYZ, dtype=np.float32)).T, 0, 1)
    rgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)


def colour_stats(frames):
    """(mean, std) of each LAB channel over a segment, from a subsample of frames and pixels."""
    import numpy as np

    frames = np.asarray(frames)
    sample = _rgb8_to_lab(frames[::4, ::4, ::4]).reshape(-1, 3)
    return sample.mean(axis=0), sample.std(axis=0)


def match_colours(frames, reference, strength=COLOUR_STRENGTH):
    """Pull a segment's colours towards `reference` (colour_stats of segment 1).

    The drift is measured on the segment's last quarter and the correction ramps from nothing on frame 0
    (which continues the already corrected conditioning frame, so the seam has no step) to `strength` on
    the last frame (which conditions the next segment). Contrast change is clamped to COLOUR_MIN_GAIN -
    COLOUR_MAX_GAIN so flat frames (fades, plain walls) are not blown up. Returns `frames` itself when
    there is nothing to correct."""
    import numpy as np

    frames = np.asarray(frames)
    ref_mean, ref_std = reference
    mean, std = colour_stats(frames[-max(1, len(frames) // 4):])
    full_scale = np.clip((std + strength * (ref_std - std)) / np.maximum(std, 1e-3), COLOUR_MIN_GAIN, COLOUR_MAX_GAIN)
    shift = strength * (ref_mean - mean)
    if strength <= 0 or (np.all(np.abs(shift) < 0.25) and np.all(np.abs(full_scale - 1) < 0.005)):
        return frames
    out = frames.copy()
    last = max(1, len(frames) - 1)
    for i in range(1, len(frames)):  # frame by frame: a 5 s segment in float LAB would be several GB
        w = i / last  # 0 at the seam -> 1 at the last frame
        scale = 1 + w * (full_scale - 1)
        out[i] = _lab_to_rgb8((_rgb8_to_lab(frames[i]) - mean) * scale + mean + w * shift)
    return out


RESUME_VERSION = 1  # bump when segment rendering changes, so older saved segments are not spliced in


def _save_segment(path, frames):
    import numpy as np

    tmp = path[:-len(".npz")] + ".tmp.npz"
    np.savez(tmp, frames=np.asarray(frames))
    os.replace(tmp, path)


def _load_segment(path):
    import numpy as np

    with np.load(path) as z:
        return z["frames"]


def _save_progress(work_dir, key, n, frames):
    """Segment n is finished: its frames, then segments.json saying how many are done (written last, so a
    crash in between leaves the count at the previous segment)."""
    _save_segment(os.path.join(work_dir, f"segment_{n:02d}.npz"), frames)
    tmp = os.path.join(work_dir, "segments.json.tmp")
    with open(tmp, "w") as f:
        json.dump({"key": key, "done": n + 1}, f)
    os.replace(tmp, os.path.join(work_dir, "segments.json"))


def _load_segments(work_dir, key):
    """The finished segments of an earlier run of this take (same settings), else [] and a clean folder."""
    os.makedirs(work_dir, exist_ok=True)
    try:
        with open(os.path.join(work_dir, "segments.json")) as f:
            state = json.load(f)
    except (OSError, ValueError):
        state = {}
    parts = []
    if state.get("key") == key:
        for n in range(int(state.get("done") or 0)):
            path = os.path.join(work_dir, f"segment_{n:02d}.npz")
            try:
                parts.append(_load_segment(path))
            except Exception:  # noqa: BLE001 - a damaged file: render from here
                break
    if not parts:
        for name in os.listdir(work_dir):
            if name.startswith("segment"):
                os.remove(os.path.join(work_dir, name))
    return parts


# px: soften the frame each later segment starts from. Off: live check 2026-11-25 showed 0.8 px didn't reduce the
# halo and over-sharpened textures; kept switchable while the halo's cause is investigated.
CONDITION_BLUR = 0


def soften(image, radius=CONDITION_BLUR):
    """A lightly blurred copy of a PIL image (for the frame a segment starts from; see CONDITION_BLUR)."""
    from PIL import ImageFilter

    return image.filter(ImageFilter.GaussianBlur(radius)) if radius else image


def transfer_colours(frame, reference):
    """One uint8 frame moved fully onto `reference` LAB stats (gain clamped). Used on a re-anchored frame, so
    the image edit's colour shift doesn't start the next segment off the take's colours."""
    import numpy as np

    ref_mean, ref_std = reference
    mean, std = colour_stats(np.asarray(frame)[None])
    scale = np.clip(ref_std / np.maximum(std, 1e-3), COLOUR_MIN_GAIN, COLOUR_MAX_GAIN)
    if np.all(np.abs(ref_mean - mean) < 0.25) and np.all(np.abs(scale - 1) < 0.005):
        return frame
    return _lab_to_rgb8((_rgb8_to_lab(frame) - mean) * scale + ref_mean)


def np_frame(image):
    import numpy as np

    return np.asarray(image.convert("RGB"))


def segment_count(seconds):
    """How many 5 s Wan passes a take of `seconds` needs."""
    return -(-max(1, min(int(seconds), MAX_TAKE_SECONDS)) // SEGMENT_SECONDS)


def segment_prompts(prompt, n, beats=None):
    """One prompt per 5 s segment. With beats ([{at, prompt}], first at 0), segment i (starting at 5*i s)
    takes the last beat with at <= 5*i; without, every segment gets `prompt`."""
    if not beats:
        return [prompt] * n
    return [[b["prompt"] for b in beats if b["at"] <= SEGMENT_SECONDS * i][-1] for i in range(n)]


def generate_video(out_path, prompt, negative_prompt="", seconds=5, width=1280, height=704,
                   seed=None, image=None, fps=32, upscale=False, raw_path=None, grain=False,
                   segment_prompts=None, work_dir=None, progress=None, anchor_every=None, reanchor=None):
    """Text-to-video, or image-to-video when `image` (a PIL keyframe) is given.

    Up to MAX_TAKE_SECONDS: segments of 5 s, each after the first rendered I2V from the previous
    segment's last frame (seed + n). fps 32 doubles the frames with RIFE; upscale doubles the
    size with Real-ESRGAN. `raw_path` also gets the 16 fps clip (for the sound model).
    grain: seeded film grain on the final frames (not on the raw clip).
    segment_prompts: one prompt per segment (a long take's timeline) instead of `prompt` for all; each gets
    the trigger words and natural-look text.
    work_dir: finished segments are saved there (segment_NN.npz + segments.json); a re-run with the same
    settings loads them and renders only the rest. progress(text) gets "segment N/M" for long takes.
    anchor_every k + reanchor(frame, n) -> frame: the conditioning frames of segments k, 2k, ... (1-based) are
    re-anchored (the avatar's face restored) before those segments render."""
    import numpy as np

    import enhance

    texts = list(segment_prompts or [prompt])
    styled = [natural_prompts(with_triggers(t), negative_prompt or "") for t in texts]
    seconds = max(1, min(int(seconds), MAX_TAKE_SECONDS))
    total = seconds * VIDEO_FPS // 4 * 4 + 1  # Wan needs 4k + 1 frames per pass
    sizes, done = [], 0  # frames per Wan pass; a continuation repeats the previous last frame
    while done < total:
        num = min(SEGMENT_SECONDS * VIDEO_FPS + 1, (total - done + (1 if sizes else 0) + 2) // 4 * 4 + 1)
        done += num - (1 if sizes else 0)
        sizes.append(num)
    colour = seconds > COLOUR_MATCH_OVER_SECONDS  # long takes: every segment matched to segment 1
    key = {"v": RESUME_VERSION, "seed": seed, "total": total, "width": width, "height": height,
           "image": image is not None, "prompts": [p for p, _ in styled], "negatives": [g for _, g in styled],
           "loras": [[l.get("adapter"), l.get("weight")] for l in _JOB_LORAS],
           "colour": [COLOUR_STRENGTH, COLOUR_MIN_GAIN, COLOUR_MAX_GAIN] if colour else None,
           "blur": CONDITION_BLUR if seconds > SEGMENT_SECONDS else 0, "anchor_every": anchor_every if reanchor else None}
    parts = _load_segments(work_dir, key) if work_dir else []
    cond, reference = image, None
    if parts:
        cond = Image.fromarray(parts[-1][-1])
        reference = colour_stats(parts[0]) if colour else None
    for n in range(len(parts), len(sizes)):
        if progress and len(sizes) > 1:
            progress(f"segment {n + 1}/{len(sizes)}")
        num = sizes[n]
        anchored = bool(reanchor and anchor_every and n and (n + 1) % anchor_every == 0)
        if anchored:
            before = np_frame(cond)
            cond = reanchor(cond, n)
            if colour:  # keep the take's colours across the edit
                cond = Image.fromarray(transfer_colours(np_frame(cond), colour_stats([before])))
        # The halo: soften what the next segment starts from (not a freshly re-anchored frame: no build-up there).
        if n and seconds > SEGMENT_SECONDS and CONDITION_BLUR and not anchored:
            cond = soften(cond, CONDITION_BLUR)
        seg_prompt, seg_negative = styled[min(n, len(styled) - 1)]
        frames = _render_segment(seg_prompt, seg_negative, num, width, height,
                                 None if seed is None else int(seed) + n, cond)
        if n:
            frames = frames[1:]
        if colour and n:
            frames = match_colours(frames, reference)
        elif colour:
            reference = colour_stats(frames)
        parts.append(frames)
        if work_dir:
            try:
                _save_progress(work_dir, key, n, frames)
            except OSError as e:  # e.g. a full disk: the take goes on, only resume is lost
                print(json.dumps({"event": "segment_save_failed", "segment": n, "error": str(e)[:200]}), flush=True)
                work_dir = None
        cond = Image.fromarray(frames[-1])
    frames = np.concatenate(parts)[:total]
    if raw_path:
        write_video(frames, raw_path, fps=VIDEO_FPS)
    if fps == 32:
        frames = enhance.interpolate(frames, check_cancel)
    if upscale:
        frames = enhance.upscale(frames, check_cancel)
    import realism

    write_video(frames, out_path, fps=fps, grain=realism.Grain(seed or 0) if grain else None)
    return out_path


# High-quality H.264: imageio's default (quality 5) is visibly blocky in gradients and skin.
X264_ARGS = ["-crf", "16", "-preset", "slow"]


def write_video(frames, out_path, fps=VIDEO_FPS, grain=None):
    import imageio
    import numpy as np

    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=None, pixelformat="yuv420p",
                            macro_block_size=16, ffmpeg_params=X264_ARGS) as writer:
        for frame in frames:
            if isinstance(frame, Image.Image):
                frame = np.asarray(frame.convert("RGB"))
            elif frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).round().astype("uint8")
            writer.append_data(grain(frame) if grain else frame)


def avatar_prompt(description):
    """Avatar candidate prompt (Wan or Krea 2): neutral, well-lit, full figure."""
    return (f"Full-body portrait photo of {description}. Standing, facing the camera, "
            "plain light grey studio background, soft even lighting, sharp focus, high detail.")


def generate_avatar_candidates(description, count, seed=None):
    """Single-frame Wan renders of the avatar: neutral, well-lit, full figure,
    which gives the keyframe editor a clean reference to work from."""
    import numpy as np

    t2v = _apply_loras(_swap_to("t2v", _load_t2v))  # the worker clears job LoRAs; this drops leftovers
    prompt = avatar_prompt(description)
    base = seed if seed is not None else int.from_bytes(os.urandom(4), "big")
    sampling = _sampling()
    images = []
    for i in range(count):
        frame = t2v(
            # the same prompt for every candidate: encoded once
            **_prompt_embeds("t2v", t2v, prompt, "blurry, distorted face, extra limbs, text, watermark",
                             sampling["guidance_scale"] > 1.0),
            width=704,
            height=1280,
            num_frames=1,
            **sampling,
            generator=_generator(base + i),
            output_type="np",
            callback_on_step_end=_on_step,
        ).frames[0][0]
        images.append(Image.fromarray((np.clip(frame, 0, 1) * 255).round().astype("uint8")))
    return images


# ---------------------------------------------------------------- keyframes

# Qwen-Image-Edit-2511-Lightning (Apache-2.0): 4- or 8-step distilled LoRA, no CFG.
# EDIT_LIGHTNING=4|8 in /etc/ai-video.env turns it on (only for a 2511 EDIT_MODEL, which needs
# diffusers >= 0.37 for its zero_cond_t option); 0 keeps the full 40-step model with CFG.
EDIT_LIGHTNING = int(os.environ.get("EDIT_LIGHTNING", "0") or 0)
EDIT_LIGHTNING_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
EDIT_LIGHTNING_REVISION = "107d238d68b8387e32843d0b4a2e24ceb0d3ad86"


# Qwen-Image-Edit takes at most 3 input images per call.
MAX_EDIT_IMAGES = 3


def _as_list(images):
    return list(images) if isinstance(images, (list, tuple)) else [images]


def _edit_lightning():
    return EDIT_LIGHTNING if EDIT_LIGHTNING in (4, 8) and "2511" in EDIT_MODEL else 0


def _edit_sampling(best=False):
    steps = 0 if best else _edit_lightning()
    if steps:
        return dict(num_inference_steps=steps, true_cfg_scale=1.0)
    return dict(num_inference_steps=40, true_cfg_scale=4.0)


def _load_edit():
    from diffusers import QwenImageEditPlusPipeline

    prefetch_model(EDIT_MODEL)  # ~58 GB: warm the page cache in parallel, as for Wan
    pipe = QwenImageEditPlusPipeline.from_pretrained(EDIT_MODEL, torch_dtype=torch.bfloat16)
    steps = _edit_lightning()
    if steps:
        pipe.load_lora_weights(
            EDIT_LIGHTNING_REPO, revision=EDIT_LIGHTNING_REVISION, adapter_name="lightning",
            weight_name=f"Qwen-Image-Edit-2511-Lightning-{steps}steps-V1.0-bf16.safetensors",
        )
    # ~58 GB of weights: components move to the GPU only while they run.
    pipe.enable_model_cpu_offload()
    return pipe


class _full_model:
    """With best=True and Lightning loaded: the adapter is off inside the block (40-step model, true CFG)."""

    def __init__(self, pipe, best):
        self.pipe, self.off = pipe, bool(best and _edit_lightning())

    def __enter__(self):
        if self.off:
            self.pipe.disable_lora()

    def __exit__(self, *exc):
        if self.off:
            self.pipe.enable_lora()


def make_keyframe(cast, scene, width=1280, height=704, seed=None, best=False):
    """First frame of a clip with every avatar in `cast` placed in `scene`.

    cast: list of (name, [PIL reference images], description), 1-3 entries. One avatar gets
    up to MAX_EDIT_IMAGES of its images; with 2-3 avatars each gets its image 1, because the
    model takes at most 3 input images.
    """
    pipe = _swap_to("edit", _load_edit)
    if len(cast) == 1:
        images = _as_list(cast[0][1])[:MAX_EDIT_IMAGES]
        who = "Keep this exact character identical: same face, hair, skin tone, body shape and outfit."
        if len(images) > 1:
            who = (f"Images 1-{len(images)} all show the same single character from different views. "
                   f"{who} Show only this one character, once.")
    else:
        images = [_as_list(refs)[0] for _, refs, _ in cast]
        roles = "; ".join(f"image {i + 1} is {name}" for i, (name, _, _) in enumerate(cast))
        who = (f"Keep each of these characters exactly identical to their image ({roles}): "
               "same faces, hair, skin tones, body shapes and outfits. Show all of them together.")
    with _full_model(pipe, best):
        image = pipe(
            image=[ref.convert("RGB") for ref in images],
            prompt=f"{who} Scene: {scene}. Wide cinematic frame, natural lighting, photorealistic.",
            negative_prompt=" ",
            # Compose at the clip's size. Without it the pipeline matches the reference's aspect
            # (a tall full-body avatar), and fit() then crops the head off for a 1280x704 clip.
            width=width,
            height=height,
            **_edit_sampling(best),
            generator=_generator(seed),
            callback_on_step_end=_on_step,
        ).images[0]
    return fit(image, width, height)


CLIP_FRAMES = 33  # 2 s at 16 fps (4k + 1 for Wan); also the trainer's clip length


def render_clip(image, prompt, out_path, seed=None, width=832, height=832):
    """A 2 s Wan I2V clip from `image` (a path or PIL image): 16 fps, no smoothing/upscale/sound/grain.
    Writes `out_path` and three stills (first, middle, last frame) for the content filter; returns their paths."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    text, negative = natural_prompts(with_triggers(prompt))
    frames = _render_segment(text, negative, CLIP_FRAMES, width, height, seed, fit(image, width, height))
    write_video(frames, out_path, fps=VIDEO_FPS)
    stills = []
    for k, i in enumerate((0, len(frames) // 2, len(frames) - 1)):
        path = f"{out_path}.{k}.png"
        Image.fromarray(frames[i]).save(path)
        stills.append(path)
    return stills


def reanchor(frame, references, width, height, seed=None):
    """A long take's conditioning frame with the avatar's face and hair restored from `references`
    (its images), everything else kept: composition, pose, clothes, background, light."""
    pipe = _swap_to("edit", _load_edit)
    refs = _as_list(references)[:MAX_EDIT_IMAGES - 1]
    image = pipe(
        image=[frame.convert("RGB")] + [ref.convert("RGB") for ref in refs],
        prompt=(f"Image 1 is a frame from a video. Images 2-{len(refs) + 1} show the same person. Keep image 1 "
                "exactly as it is: same composition, framing, pose, expression, clothes, background, lighting "
                "and colours. Only restore this person's face and hair so they match the person in the other "
                "images. Photorealistic, no other changes."),
        negative_prompt=" ",
        width=width,
        height=height,
        **_edit_sampling(False),
        generator=_generator(seed),
        callback_on_step_end=_on_step,
    ).images[0]
    return fit(image, width, height)


def make_photo(references, prompt, negative_prompt, width, height, seed=None, best=False):
    """One photorealistic still of the person in `references` (a PIL image, or a list of
    images of the same person; the first MAX_EDIT_IMAGES are used) as `prompt` describes.
    best: the full 40-step model with true CFG (the Lightning adapter is switched off for this
    call), so the negative prompt applies; ~3x slower."""
    pipe = _swap_to("edit", _load_edit)
    with _full_model(pipe, best):
        image = pipe(
            image=[ref.convert("RGB") for ref in _as_list(references)[:MAX_EDIT_IMAGES]],
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            **_edit_sampling(best),
            generator=_generator(seed),
            callback_on_step_end=_on_step,
        ).images[0]
    return fit(image, width, height)


# ---------------------------------------------------------------- sound

MMAUDIO_VARIANT = "large_44k_v2"
MMAUDIO_DIR = os.environ.get("MMAUDIO_DIR", "/opt/models/mmaudio")


def _load_audio():
    import dataclasses
    from pathlib import Path

    from mmaudio.eval_utils import all_model_cfg
    from mmaudio.model.networks import get_my_mmaudio
    from mmaudio.model.utils.features_utils import FeaturesUtils

    # MMAudio's default checkpoint paths are relative to the working directory.
    base = Path(MMAUDIO_DIR)
    cfg = all_model_cfg[MMAUDIO_VARIANT]
    cfg = dataclasses.replace(
        cfg,
        model_path=base / "weights" / cfg.model_path.name,
        vae_path=base / "ext_weights" / cfg.vae_path.name,
        synchformer_ckpt=base / "ext_weights" / cfg.synchformer_ckpt.name,
    )
    for path in (cfg.model_path, cfg.vae_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    cfg.download_if_needed()

    net = get_my_mmaudio(cfg.model_name).to(torch.bfloat16).eval()
    net.load_weights(torch.load(cfg.model_path, map_location="cpu", weights_only=True))
    features = FeaturesUtils(
        tod_vae_ckpt=cfg.vae_path,
        synchformer_ckpt=cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=cfg.mode,
        bigvgan_vocoder_ckpt=cfg.bigvgan_16k_path,
        need_vae_encoder=False,
    ).to(torch.bfloat16).eval()
    return cfg, net, features


# MMAudio is small (a few GB), so it stays in RAM beside whichever big model is loaded
# and visits the GPU only while it runs. Going through _swap_to would evict the 14B
# video model after every clip and force a reload on the next job.
_audio = {}


def _audio_models():
    if not _audio:
        print("loading audio", flush=True)
        _audio["models"] = _load_audio()
    return _audio["models"]


@torch.inference_mode()
def generate_sound(video_path, out_path, prompt="", seconds=5, seed=None):
    """Ambient sound and Foley for a silent clip, synchronised to its frames and
    steered by `prompt`. Writes a 44.1 kHz WAV of the clip's length."""
    cfg, net, features = _audio_models()
    check_cancel()
    net.to("cuda")
    features.to("cuda")
    try:
        return _generate_sound(cfg, net, features, video_path, out_path, prompt, seconds, seed)
    finally:
        net.to("cpu")
        features.to("cpu")
        torch.cuda.empty_cache()


def _generate_sound(cfg, net, features, video_path, out_path, prompt, seconds, seed):
    import soundfile as sf
    from mmaudio.eval_utils import generate, load_video
    from mmaudio.model.flow_matching import FlowMatching

    info = load_video(video_path, float(seconds))
    seq_cfg = cfg.seq_cfg
    seq_cfg.duration = info.duration_sec
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    rng = torch.Generator("cuda")
    rng.manual_seed(int(seed) if seed is not None else int.from_bytes(os.urandom(4), "big"))
    audio = generate(
        info.clip_frames.unsqueeze(0),
        info.sync_frames.unsqueeze(0),
        [prompt],
        negative_text=["speech, talking, music"],
        feature_utils=features,
        net=net,
        fm=FlowMatching(min_sigma=0, inference_mode="euler", num_steps=25),
        rng=rng,
        cfg_strength=4.5,
    ).float().cpu()[0]
    sf.write(out_path, audio.numpy().T, seq_cfg.sampling_rate)
    return out_path


# ---------------------------------------------------------------- narration

_tts = {}


def narrate(text, voice, out_path):
    """Speak `text` with a Kokoro voice (e.g. af_heart, am_adam, bf_emma) to a WAV.
    The voice's first letter is its language (a = US English, b = UK English...)."""
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    lang = voice[0]
    if lang not in _tts:
        _tts[lang] = KPipeline(lang_code=lang, device="cpu")
    chunks = [audio for _, _, audio in _tts[lang](text, voice=voice)]
    audio = np.concatenate(chunks) if chunks else np.zeros(TTS_SAMPLE_RATE // 10, dtype="float32")
    sf.write(out_path, audio, TTS_SAMPLE_RATE)
    return len(audio) / TTS_SAMPLE_RATE


# ---------------------------------------------------------------- vision-language

def _load_vlm():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    prefetch_model(VLM_MODEL)  # ~16 GB
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(VLM_MODEL)
    return model, processor


def _chat(messages, max_new_tokens=1024, **generate):
    from qwen_vl_utils import process_vision_info

    check_cancel()
    model, processor = _swap_to("vlm", _load_vlm)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text], images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs
    ).to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, **generate)
    out = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True)[0]


def _extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"model did not return JSON: {text[:300]}")
    return json.loads(match.group(0))


COUNT_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
               "nine": 9, "ten": 10}


def _parse_count(answer):
    """The number of people in a VLM answer: the first integer, else a number word; "none"/"no one" -> 0;
    -1 when it can't be read."""
    text = (answer or "").strip().lower()
    m = re.search(r"\d+", text)
    if m:
        return int(m.group(0))
    if re.search(r"\b(none|nobody|no one|no people|zero)\b", text):
        return 0
    if re.search(r"\b(a single|single|just one|only one|alone)\b", text):
        return 1
    m = re.search(r"\b(" + "|".join(COUNT_WORDS) + r")\b", text)
    return COUNT_WORDS[m.group(1)] if m else -1


def count_people(image_path):
    """How many people the VLM sees in a photo (partial people included); -1 when its answer can't be read.
    Avatar photo sets keep only photos with exactly one person."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": f"file://{image_path}"},
        {"type": "text", "text": "How many people are visible in this photo? Count partial people and people "
                                 "in the background too. Answer with a number only."}]}]
    return _parse_count(_chat(messages, max_new_tokens=8))


def describe_video(video_path, instructions=""):
    prompt = (
        "Watch this video and write metadata for publishing it. Respond with JSON only: "
        '{"title": str (<= 80 chars), "description": str (2-4 sentences), "tags": [str, ...] (5-10)}.'
    )
    if instructions:
        prompt += f"\nAdditional instructions: {instructions}"
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": f"file://{video_path}", "max_pixels": 448 * 448, "fps": 1.0},
            {"type": "text", "text": prompt},
        ],
    }]
    return _extract_json(_chat(messages, max_new_tokens=512))


def write_story(premise, words=800, genre="", seed=None):
    """An original short story from a premise, written to be read aloud and filmed with `story`.
    Returns {"title", "text"}. Sampled (temperature 0.8), so the seed picks the version."""
    import torch

    if seed is not None:
        torch.manual_seed(int(seed))
    prompt = f"""Write an original short story of about {words} words.
Premise: {premise}
{"Genre and tone: " + genre if genre else ""}
It will be read aloud as narration and turned into a short video, so: concrete, visual scenes;
few characters with names; a clear beginning, middle and end; no chapter headings or lists.
Start with one line "Title: <title>", then a blank line, then the story in paragraphs."""
    text = _chat([{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                 min(4096, int(words * 1.8) + 100), do_sample=True, temperature=0.8, top_p=0.9,
                 repetition_penalty=1.05).strip()
    return split_title(text)


def split_title(text):
    """'Title: X' (or a short first line) as the title; the rest is the story."""
    first, _, rest = text.partition("\n")
    line = first.strip().strip("*#").strip()
    if line.lower().startswith("title:"):
        return {"title": line[6:].strip().strip('"') or "Untitled", "text": rest.strip()}
    if len(line) <= 80 and rest.strip():
        return {"title": line.strip('"'), "text": rest.strip()}
    return {"title": "Untitled", "text": text.strip()}


MAX_SEGMENT_PROMPT_CHARS = 500
PLANNED_MIN_SEGMENTS = 4  # takes of up to 3 segments (15 s) keep one prompt


def valid_segment_plan(plan, n):
    """n non-empty strings of at most MAX_SEGMENT_PROMPT_CHARS each."""
    return (isinstance(plan, list) and len(plan) == n
            and all(isinstance(p, str) and 0 < len(p.strip()) <= MAX_SEGMENT_PROMPT_CHARS for p in plan))


def _first_json_object(text):
    """The first {...} in `text` that parses as a JSON object (models add notes before or after)."""
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            obj, _ = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"model did not return JSON: {text[:300]}")


def plan_segments(prompt, n, cast=(), seconds=None, image=None):
    """A long take's timeline: n prompts, one per continuous 5 s segment, written by the VLM from the
    user's prompt. Same subject, place, light and camera style; only the action moves on.
    cast: (name, description) pairs, referred to by name. Raises ValueError on unusable output."""
    names = ", ".join(name for name, _ in cast)
    rules = (f"The people are {names}; refer to them by these names and do not describe their looks."
             if names else "Keep the same subject, described the same way, in every prompt.")
    seconds = seconds or n * SEGMENT_SECONDS
    last = seconds - (n - 1) * SEGMENT_SECONDS
    ask = f"""This is one continuous {seconds}-second camera take, rendered as {n} consecutive segments of 5 seconds
(the last segment is {last} seconds). Each segment continues exactly from the last frame of the one before it.
Write {n} short video prompts, one per segment, in order. Every prompt names the same subject,
place, lighting and camera style as the take; only the action moves on, a little at a time,
continuing from the previous segment. No cuts, no new locations, no jumps in time.
Each prompt is one or two sentences, at most 400 characters.
{rules}
Respond with JSON only: {{"segments": [str, ...]}} with exactly {n} strings.

TAKE: {prompt}"""
    content = [{"type": "text", "text": ask}]
    if image:  # the take's opening frame: the plan keeps to what is in it
        content = [{"type": "image", "image": f"file://{image}"},
                   {"type": "text", "text": "This is the opening frame of the take. Every prompt must keep to what is "
                                            "in this frame: the same people (no one else appears), place, light and "
                                            "camera; no new animals, objects, crowds or effects.\n\n" + ask}]
    out = _first_json_object(_chat([{"role": "user", "content": content}], min(4096, 160 * n + 200)))
    plan = [p.strip() if isinstance(p, str) else p for p in out.get("segments") or []]
    if not valid_segment_plan(plan, n):
        raise ValueError(f"planner returned {len(plan)} usable segment prompts, wanted {n}")
    return plan


def plan_story(story, max_shots=6, style="", cast=()):
    """Shot list for a story.

    cast: list of (name, description). The first is the star. Each shot lists
    which cast members appear; their looks come from reference images, so
    shot prompts describe only setting, action, camera and lighting.
    """
    if cast:
        roster = "\n".join(f"- {name}: {desc}" for name, desc in cast)
        cast_rules = f"""CAST (use these names exactly):
{roster}
{cast[0][0]} is the star and should appear in most shots. Every shot lists in "characters"
which cast members appear (at most 3; [] for a shot with none of them). In shot prompts,
refer to cast members by name and do not describe their appearance; describe only the
setting, their action, camera movement and lighting."""
        shot_schema = '{"prompt": str, "narration": str, "characters": [str]}'
    else:
        cast_rules = "Each shot prompt describes subject, action, camera movement and lighting."
        shot_schema = '{"prompt": str, "narration": str}'

    prompt = f"""Turn this story into a shot list for a short video of at most {max_shots} shots.
Respond with JSON only:
{{"title": str,
  "visual_bible": str,   // one paragraph fixing setting and look, reused in every shot
  "shots": [{shot_schema}]}}
Each shot is one continuous 5-second camera shot. Narration is one short sentence
that will be both spoken and shown as a subtitle.
{cast_rules}
{"Visual style: " + style if style else ""}

STORY:
{story}"""
    plan = _extract_json(_chat([{"role": "user", "content": [{"type": "text", "text": prompt}]}], 2048))
    plan["shots"] = plan.get("shots", [])[:max_shots]
    if not plan["shots"]:
        raise ValueError("model returned no shots")
    names = {name for name, _ in cast}
    for shot in plan["shots"]:
        shot["characters"] = [c for c in shot.get("characters", []) if c in names][:3]
    return plan
