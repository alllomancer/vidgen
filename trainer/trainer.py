"""LoRA trainer: pulls `train` jobs from the train queue, trains a T2V LoRA (low-noise then
high-noise expert) with musubi-tuner, uploads it to loras/<client>/<name>/v<N>/, and stops
its own instance after IDLE_MINUTES with nothing to do.

A job's work folder lives on this instance's EBS volume (/opt/trainer/jobs/<job_id>) and
records finished phases in state.json, so after a spot interruption the redelivered job
skips what is done and resumes training from the last saved state (every 250 steps).
See docs/specs/2026-11-09-lora-training.md, "Trainer: step 2".
"""

import json
import os

# Before anything imports huggingface_hub: its Xet transfers at full speed (the Krea files are ~66 GB).
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import re
import shutil
import signal
import subprocess
import threading
import time
import traceback
import urllib.request

from concurrent.futures import ThreadPoolExecutor

import boto3

import mirror
import plan

REGION = os.environ["AWS_REGION"]
BUCKET = os.environ["BUCKET"]
QUEUE_URL = os.environ["TRAIN_QUEUE_URL"]
JOBS_TABLE = os.environ["JOBS_TABLE"]
VLM_MODEL = os.environ["VLM_MODEL"]
IDLE_SECONDS = int(os.environ.get("IDLE_MINUTES", "15")) * 60
JOB_TIMEOUT_SECONDS = int(os.environ.get("JOB_TIMEOUT_MINUTES", "960")) * 60
ROOT = "/opt/trainer"
VENV_PY = f"{ROOT}/venv/bin/python"
# Captioning has its own venv: musubi-tuner pins transformers 5.x, while Qwen2.5-VL captioning
# uses the render worker's known-good transformers 4.55.4 stack.
CAPTION_PY = f"{ROOT}/caption-venv/bin/python"
MUSUBI = f"{ROOT}/musubi-tuner/src/musubi_tuner"
MODELS = f"{ROOT}/models"
DIT = {"t2v_low": f"{MODELS}/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors",
       "t2v_high": f"{MODELS}/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors"}
VAE = f"{MODELS}/split_files/vae/wan_2.1_vae.safetensors"
T5 = f"{MODELS}/models_t5_umt5-xxl-enc-bf16.pth"
# The Wan training files, relative to MODELS: their unit in the model mirror (mirror.py).
WAN_FILES = [os.path.relpath(p, MODELS) for p in (DIT["t2v_low"], DIT["t2v_high"], VAE, T5)]
BASE_MODEL = "Wan-AI/Wan2.2-T2V-A14B (ComfyUI fp16 repackage)"
VISIBILITY_SECONDS = 1800
WATCH_SECONDS = 15
PROGRESS_EVERY = 20
FINISHED = ("done", "failed", "cancelled")
EXPERT_LABEL = {"t2v_low": "low-noise expert", "t2v_high": "high-noise expert"}
# Training resolution per speed preset (`vidgen train --speed`); both use 33-frame segments.
RESOLUTION = {"standard": [832, 480], "fast": [512, 288]}
# Checkpoint previews queued on the render box when a training finishes (same as the API's
# `lora preview` defaults).
PREVIEW_SCENE = "a person in a bright, simple room, medium shot, steady camera"
PREVIEW_CHECKPOINTS = 5
# Krea 2 avatar LoRAs (docs/specs/2026-11-24-krea-avatar-lora.md; settings from the 2026-11-24 spike,
# trainer/spike/run_krea_trial.sh): musubi-tuner trains on Krea 2 RAW, samples on Turbo (what renders use).
KREA_DIR = f"{MODELS}/krea2"
KREA_RAW = f"{KREA_DIR}/raw.safetensors"
KREA_TURBO = f"{KREA_DIR}/turbo.safetensors"
KREA_VAE = f"{KREA_DIR}/split_files/vae/qwen_image_vae.safetensors"
KREA_TE = f"{KREA_DIR}/text_encoders/qwen3vl_4b_bf16.safetensors"
# (repo, file, commit): pinned, so a rebuilt trainer trains on the same weights. Turbo's pin is the render
# box's (worker/models.py KREA_REVISION). Bump deliberately.
KREA_FILES = [("krea/Krea-2-Raw", "raw.safetensors", "6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603"),
              ("krea/Krea-2-Turbo", "turbo.safetensors", "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"),
              ("Comfy-Org/Qwen-Image", "split_files/vae/qwen_image_vae.safetensors",
               "1f12b17be14c89b026c51a91d67c32f84bb047bc"),
              ("Comfy-Org/Qwen3-VL", "text_encoders/qwen3vl_4b_bf16.safetensors",
               "02f0d3eefb59528799653d16735818801177ef1f")]
KREA_UNIT = "trainer/krea2"  # their copy in the model mirror
FETCH_POLL = 5  # seconds between looks at a running fetch (and at a cancel)
# musubi-tuner's Krea 2 scripts need torch >= 2.6 (the Wan venv keeps 2.5.1): own venv, built on first use.
KREA_VENV = f"{ROOT}/krea-venv"
KREA_TORCH = ["torch==2.8.0", "torchvision==0.23.0", "--index-url", "https://download.pytorch.org/whl/cu126"]
KREA_SAVE_EVERY = 240
HF_TOKEN_PARAM = os.environ.get("HF_TOKEN_PARAM", "/ai-video/hf_token")

s3 = boto3.client("s3", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION)
ec2 = boto3.client("ec2", region_name=REGION)
jobs = boto3.resource("dynamodb", region_name=REGION).Table(JOBS_TABLE)

CANCEL = threading.Event()
CANCEL_REASON = ["cancelled"]
CURRENT = {"proc": None}


class Cancelled(Exception):
    pass


def log(**fields):
    print(json.dumps(fields, default=str), flush=True)


# ---------------------------------------------------------------- job record

def set_status(job_id, status, **extra):
    names, values, expr = {"#s": "status"}, {":s": status}, "SET #s = :s"
    for k, v in extra.items():
        names[f"#{k}"] = k
        values[f":{k}"] = v
        expr += f", #{k} = :{k}"
    jobs.update_item(Key={"job_id": job_id}, UpdateExpression=expr,
                     ExpressionAttributeNames=names, ExpressionAttributeValues=values)


class Progress:
    """Throttled progress text on the job record."""

    def __init__(self, job_id):
        self.job_id, self.last = job_id, 0.0

    def __call__(self, text, force=False):
        if CANCEL.is_set():
            raise Cancelled(CANCEL_REASON[0])
        if force or time.time() - self.last > PROGRESS_EVERY:
            self.last = time.time()
            set_status(self.job_id, "running", progress=text)


def request_cancel(reason):
    CANCEL_REASON[0] = reason
    CANCEL.set()
    proc = CURRENT["proc"]
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)  # the whole group: accelerate spawns the trainer


class JobWatch(threading.Thread):
    """Keeps the SQS message invisible, honours cancel requests and the timeout."""

    def __init__(self, job_id, receipt):
        super().__init__(daemon=True)
        self.job_id, self.receipt = job_id, receipt
        self.started = time.time()
        self.stopped = threading.Event()

    def run(self):
        last_extend = time.time()
        while not self.stopped.wait(WATCH_SECONDS):
            try:
                if time.time() - last_extend > VISIBILITY_SECONDS // 3:
                    sqs.change_message_visibility(QueueUrl=QUEUE_URL, ReceiptHandle=self.receipt,
                                                  VisibilityTimeout=VISIBILITY_SECONDS)
                    last_extend = time.time()
                if time.time() - self.started > JOB_TIMEOUT_SECONDS:
                    request_cancel(f"timed out after {JOB_TIMEOUT_SECONDS // 60} min")
                item = jobs.get_item(Key={"job_id": self.job_id}, ProjectionExpression="cancel_requested")
                if item.get("Item", {}).get("cancel_requested"):
                    request_cancel("cancelled")
            except Exception:
                traceback.print_exc()


# ---------------------------------------------------------------- subprocesses

STEP_LINE = re.compile(r"steps:\s+\d+%.*?\|\s*(\d+)/(\d+)\s*\[.*?([\d.]+)(s/it|it/s)")


def run(cmd, log_path, on_steps=None):
    """Run a command in its own process group, tee output to log_path, report musubi
    progress bars via on_steps(done, total, seconds_per_step). Raises on failure/cancel."""
    with open(log_path, "ab") as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        CURRENT["proc"] = proc
        tail, buf = [], b""
        try:
            while chunk := proc.stdout.read1(4096):
                logf.write(chunk)
                buf += chunk
                *lines, buf = re.split(rb"[\r\n]", buf)
                for raw in lines:
                    line = raw.decode(errors="replace").strip()
                    if not line:
                        continue
                    tail = (tail + [line])[-25:]
                    m = STEP_LINE.search(line)
                    if m and on_steps:
                        rate = float(m[3]) if m[4] == "s/it" else 1 / max(float(m[3]), 1e-6)
                        on_steps(int(m[1]), int(m[2]), rate)
                    elif line.startswith("CAPTION") and on_steps:
                        done, total = line.split()[1].split("/")
                        on_steps(int(done), int(total), None)
        finally:
            proc.wait()
            CURRENT["proc"] = None
    if CANCEL.is_set():
        raise Cancelled(CANCEL_REASON[0])
    if proc.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[1] if len(cmd) > 1 else cmd[0])} failed "
                           f"(exit {proc.returncode}): " + " | ".join(tail[-6:]))


def ffprobe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------- phases

class Job:
    def __init__(self, job, jd):
        self.job, self.jd = job, jd
        self.params = json.loads(job["params"])
        self.client, self.name = job["client"], self.params["dataset"]
        self.state_path = os.path.join(jd, "state.json")
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {"done": []}
        self.progress = Progress(job["job_id"])
        self.speed = self.params.get("speed") or "standard"
        self.captions_key = f"datasets/{self.client}/{self.name}/captions.json"
        for d in ("raw", "items", "cache", "out"):
            os.makedirs(os.path.join(jd, d), exist_ok=True)

    def save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp, self.state_path)

    def phase(self, name, fn):
        if name in self.state["done"]:
            return
        fn()
        self.state["done"].append(name)
        self.save()

    # -- prepare: manifest files -> 5 s segments (max 12 per source) and photos
    def prepare(self):
        prefix = f"datasets/{self.client}/{self.name}/"
        manifest = json.loads(s3.get_object(Bucket=BUCKET, Key=prefix + "manifest.json")["Body"].read())
        self.state["manifest"] = {k: manifest[k] for k in ("kind", "trigger", "file_count", "total_bytes", "updated_at")}
        items, raw = os.path.join(self.jd, "items"), os.path.join(self.jd, "raw")
        segments = photos = 0
        files = manifest["files"]
        for i, f in enumerate(files, 1):
            self.progress(f"preparing clips {i}/{len(files)}")
            ext = os.path.splitext(f["path"])[1].lower()
            local = os.path.join(raw, f"src{i:04d}{ext}")
            s3.download_file(BUCKET, prefix + "raw/" + f["path"], local)
            if ext in (".png", ".jpg", ".jpeg", ".webp"):
                shutil.move(local, os.path.join(items, f"p{i:04d}{ext}"))
                photos += 1
                continue
            duration = ffprobe_duration(local)
            for j, start in enumerate(plan.segment_plan(duration)):
                length = min(plan.SEGMENT_SECONDS, duration - start)
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{length:.2f}",
                                "-i", local, "-an", "-vf", "scale='min(832,iw)':-2,fps=16",
                                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                                os.path.join(items, f"s{i:04d}_{j:02d}.mp4")], check=True)
                segments += 1
            os.remove(local)
        if segments + photos < 5:
            raise RuntimeError(f"dataset too small: {segments + photos} usable items (need >= 5)")
        self.state.update(segments=segments, photos=photos)

    def caption(self):
        items = os.path.join(self.jd, "items")
        total = self.state["segments"] + self.state["photos"]
        reused = self.apply_saved_captions(items)
        self.progress(f"captioning {reused}/{total}" + (f" ({reused} from saved captions)" if reused else ""), force=True)
        if reused < total:  # all captions saved: no need to load the captioning model
            run([CAPTION_PY if os.path.exists(CAPTION_PY) else VENV_PY, f"{ROOT}/code/caption.py", items,
                 self.state["manifest"]["trigger"], self.state["manifest"]["kind"], VLM_MODEL],
                os.path.join(self.jd, "caption.log"), on_steps=lambda d, t, _: self.progress(f"captioning {d}/{t}"))
        self.save_captions(items)

    def _saved_captions(self):
        try:
            return json.loads(s3.get_object(Bucket=BUCKET, Key=self.captions_key)["Body"].read())
        except s3.exceptions.NoSuchKey:
            return None
        except Exception as e:  # unreadable or no permission: caption from scratch
            log(event="captions_unreadable", error=str(e)[:200])
            return None

    def apply_saved_captions(self, items):
        """Write saved (possibly hand-edited) captions as the items' .txt when they belong to this
        exact dataset; caption.py then only captions the rest."""
        saved = self._saved_captions()
        if not saved or saved.get("fingerprint") != self.params.get("fingerprint"):
            if saved:
                log(event="captions_stale", note="dataset changed since the saved captions; captioning again")
            return 0
        n = 0
        for name, text in (saved.get("items") or {}).items():
            path = os.path.join(items, name)
            if os.path.exists(path) and text.strip():
                with open(os.path.splitext(path)[0] + ".txt", "w") as f:
                    f.write(text.strip())
                n += 1
        self.state["edited_captions_at"] = saved.get("edited_at")
        return n

    def save_captions(self, items):
        """captions.json for review (`vidgen dataset captions NAME --edit`) and reuse next time."""
        out = {}
        for n in sorted(os.listdir(items)):
            base, ext = os.path.splitext(n)
            txt = os.path.join(items, base + ".txt")
            if ext.lower() != ".txt" and os.path.exists(txt):
                out[n] = open(txt).read().strip()
        body = {"fingerprint": self.params.get("fingerprint"), "trigger": self.state["manifest"]["trigger"],
                "items": out, "updated_at": int(time.time()), "edited_at": self.state.get("edited_captions_at")}
        s3.put_object(Bucket=BUCKET, Key=self.captions_key, Body=json.dumps(body, indent=1).encode(),
                      ContentType="application/json")
        self.state["captions"] = len(out)

    def dataset_toml(self):
        items, cache = os.path.join(self.jd, "items"), os.path.join(self.jd, "cache")
        width, height = RESOLUTION[self.speed]
        text = f"""[general]
resolution = [{width}, {height}]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false
"""
        if self.state.get("segments", 1):  # photo-only sets (an avatar's photos) get no empty video block
            text += f"""
[[datasets]]
video_directory = "{items}"
cache_directory = "{cache}/video"
target_frames = [33]
frame_extraction = "uniform"
frame_sample = 2
"""
        if self.state["photos"]:
            os.makedirs(os.path.join(self.jd, "photos"), exist_ok=True)
            for n in os.listdir(items):  # musubi wants images in their own dataset folder
                if n.startswith("p"):
                    shutil.move(os.path.join(items, n), os.path.join(self.jd, "photos", n))
            text += f"""
[[datasets]]
image_directory = "{self.jd}/photos"
cache_directory = "{cache}/image"
"""
        path = os.path.join(self.jd, "dataset.toml")
        with open(path, "w") as f:
            f.write(text)
        return path

    def cache(self):
        self.progress("caching latents and text", force=True)
        toml = self.dataset_toml()
        logp = os.path.join(self.jd, "cache.log")
        run([VENV_PY, f"{MUSUBI}/wan_cache_latents.py", "--dataset_config", toml, "--vae", VAE], logp)
        run([VENV_PY, f"{MUSUBI}/wan_cache_text_encoder_outputs.py", "--dataset_config", toml, "--t5", T5,
             "--batch_size", "16"], logp)

    def train(self, expert):
        target = self.params["steps"]
        st = self.state.setdefault(expert, {"runs": []})
        out = os.path.join(self.jd, "out")
        done, state_dir = plan.resume_point(expert, st["runs"], os.listdir(out))
        if done >= target and st.get("final"):
            return
        run_no = len(st["runs"])
        st["runs"].append({"run": run_no, "base": done})
        self.save()
        lo, hi = plan.TIMESTEPS[expert]
        cmd = [f"{ROOT}/venv/bin/accelerate", "launch", "--num_cpu_threads_per_process", "1",
               "--mixed_precision", "fp16", f"{MUSUBI}/wan_train_network.py",
               "--task", "t2v-A14B", "--dit", DIT[expert], "--dataset_config", os.path.join(self.jd, "dataset.toml"),
               "--sdpa", "--mixed_precision", "fp16", "--fp8_base", "--optimizer_type", "adamw8bit",
               "--learning_rate", "2e-4", "--gradient_checkpointing", "--max_data_loader_n_workers", "2",
               "--persistent_data_loader_workers", "--network_module", "networks.lora_wan",
               "--network_dim", "32", "--network_alpha", "16", "--timestep_sampling", "shift",
               "--discrete_flow_shift", "3.0", "--min_timestep", str(lo), "--max_timestep", str(hi),
               "--max_train_steps", str(target - done), "--save_every_n_steps", str(plan.SAVE_EVERY),
               "--save_state", "--seed", "42", "--output_dir", out, "--output_name", f"{expert}_r{run_no}"]
        if state_dir:
            cmd += ["--resume", os.path.join(out, state_dir)]
        note = f" (resumed at {done})" if done else ""
        other_left = target if expert == "t2v_low" else 0

        def on_steps(rel, _total, rate):
            total_done = done + rel
            left = plan.eta_seconds(target - total_done, other_left, rate or 12.7)
            self.progress(f"{EXPERT_LABEL[expert]} {total_done}/{target}{note} · ~{left // 3600} h {left % 3600 // 60} min left")
        self.progress(f"{EXPERT_LABEL[expert]} {done}/{target}{note}", force=True)
        run(cmd, os.path.join(self.jd, f"{expert}.log"), on_steps=on_steps)
        final = f"{expert}_r{run_no}.safetensors"
        if not os.path.exists(os.path.join(out, final)):
            raise RuntimeError(f"training finished without {final}")
        st["final"] = final
        self.save()

    def convert(self, src, dst):
        run([VENV_PY, f"{MUSUBI}/convert_lora.py", "--input", src, "--output", dst, "--target", "other"],
            os.path.join(self.jd, "convert.log"))

    def avatar_wan(self):
        """An avatar's Wan LoRA build (docs/specs/2026-11-24-avatar-wan-lora.md): uploaded under its own LoRA name
        and version, linked on the avatar, no auto previews."""
        return self.params.get("model") == "t2v" and bool(self.params.get("avatar")) and bool(self.params.get("lora"))

    def wan_current(self):
        """Is this train job still the avatar's current Wan LoRA build?"""
        try:
            avatar = json.loads(s3.get_object(Bucket=BUCKET, Key=f"avatars/{self.client}/{self.params['avatar']}/"
                                                                  "avatar.json")["Body"].read())
        except s3.exceptions.NoSuchKey:
            return False
        created = self.params.get("avatar_created_at")
        return (self.job["job_id"] in ((avatar.get("wan_lora") or {}).get("job_ids") or [])
                and (created is None or avatar.get("created_at") in (None, created)))

    def wan_link(self):
        r = self.state["result"]
        _update_avatar_lora(self.client, self.params["avatar"], self.job["job_id"], "wan_lora", state="ready",
                            version=r["version"], ready={"lora": r["lora"], "version": r["version"]}, error=None)

    def upload(self):
        out = os.path.join(self.jd, "out")
        if self.avatar_wan() and not self.wan_current():  # replaced while training: write nothing
            self.state["result"] = {"skipped": "the avatar was replaced or a newer Wan LoRA build started"}
            return
        lora_name = self.params["lora"] if self.avatar_wan() else self.name
        base = f"loras/{self.client}/{lora_name}/"
        if self.avatar_wan() and "version" not in self.state:  # the avatar build picked its version
            self.state["version"] = int(self.params["version"])
            self.save()
        if "version" not in self.state:
            keys = [o["Key"] for p in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=base)
                    for o in p.get("Contents", [])]
            self.state["version"] = plan.next_version(keys, base)
            self.save()
        version = self.state["version"]
        prefix = f"{base}v{version}/"
        self.progress(f"uploading v{version}", force=True)
        files = []
        for expert in plan.EXPERTS:
            conv = os.path.join(out, f"{expert}.converted.safetensors")
            self.convert(os.path.join(out, self.state[expert]["final"]), conv)
            s3.upload_file(conv, BUCKET, prefix + f"{expert}.safetensors")
            files.append(f"{expert}.safetensors")
            for name in sorted(os.listdir(out)):
                abs_name = plan.checkpoint_name(name, self.state[expert]["runs"])
                if abs_name and abs_name.startswith(expert):
                    conv = os.path.join(out, abs_name + ".converted")
                    self.convert(os.path.join(out, name), conv)
                    s3.upload_file(conv, BUCKET, prefix + "checkpoints/" + abs_name)
                    files.append("checkpoints/" + abs_name)
        m = self.state["manifest"]
        lora = {"name": lora_name, "kind": m["kind"], "trigger": m["trigger"], "version": version,
                "models": ["t2v"], "steps_per_expert": self.params["steps"], "rank": 32,
                "speed": self.speed, "resolution": RESOLUTION[self.speed],
                "base": {"t2v": BASE_MODEL, "dit": {e: os.path.basename(p) for e, p in DIT.items()}},
                "dataset": {"fingerprint": self.params.get("fingerprint"), "files": m["file_count"],
                            "segments": self.state["segments"], "photos": self.state["photos"]},
                "files": files, "job_id": self.job["job_id"], "created_at": int(time.time()),
                "train_seconds": int(time.time() - int(self.job.get("started_at") or time.time())),
                **({"avatar": self.params["avatar"], "source": "avatar"} if self.avatar_wan() else {})}
        s3.put_object(Bucket=BUCKET, Key=prefix + "lora.json", Body=json.dumps(lora, indent=2).encode(),
                      ContentType="application/json")
        if not self.avatar_wan() or self.wan_current():  # a replaced avatar build never becomes current
            s3.put_object(Bucket=BUCKET, Key=base + "current.json", Body=json.dumps({"version": version}).encode(),
                          ContentType="application/json")
        self.state["files"] = files
        self.state["result"] = {"lora": lora_name, "version": version, "prefix": prefix, "speed": self.speed,
                                "steps_per_expert": self.params["steps"], "trigger": m["trigger"],
                                "segments": self.state["segments"], "photos": self.state["photos"],
                                "checkpoints": sum(f.startswith("checkpoints/") for f in files)}

    def queue_preview(self):
        """Queue a `lora_preview` job on the render box for the last checkpoints, so the clips
        are ready when the training is. A failure here never fails the training."""
        try:
            cfg = json.loads(s3.get_object(Bucket=BUCKET, Key="code/trainer/render.json")["Body"].read())
            r = self.state["result"]
            prefix = r["prefix"]
            steps = sorted(int(f.split("step")[-1].split(".")[0]) for f in self.state["files"]
                           if f.startswith("checkpoints/t2v_low"))
            clips = [{"step": st, "high_key": f"{prefix}checkpoints/t2v_high-step{st:06d}.safetensors",
                      "low_key": f"{prefix}checkpoints/t2v_low-step{st:06d}.safetensors"} for st in steps]
            final = int(self.params["steps"])
            if final not in steps:  # e.g. a short run with no checkpoint yet: preview the final weights
                clips.append({"step": final, "high_key": f"{prefix}t2v_high.safetensors",
                              "low_key": f"{prefix}t2v_low.safetensors"})
            clips = clips[-PREVIEW_CHECKPOINTS:]
            steps = [c["step"] for c in clips]
            if not clips:
                return
            params = {"name": self.name, "version": r["version"], "trigger": r.get("trigger") or "", "clips": clips,
                      "prompt": PREVIEW_SCENE, "weight": 1.0, "seconds": 2, "width": 832, "height": 480,
                      "seed": int.from_bytes(os.urandom(4), "big")}
            job_id, now = os.urandom(16).hex(), int(time.time())
            jobs.put_item(Item={"job_id": job_id, "client": self.client, "type": "lora_preview",
                                "params": json.dumps(params), "status": "queued", "created_at": now,
                                "expires_at": now + int(cfg.get("retention_days", 30)) * 86400,
                                "auto_of": self.job["job_id"]})
            sqs.send_message(QueueUrl=cfg["queue_url"], MessageBody=json.dumps({"job_id": job_id}))
            r["preview_job_id"] = job_id
            log(event="preview_queued", job_id=self.job["job_id"], preview_job_id=job_id, steps=steps)
        except Exception as e:
            self.state["result"]["preview_error"] = str(e)[:300]
            log(event="preview_failed", error=str(e)[:300])

    # ---------------------------------------------------------------- Krea 2 avatar LoRAs
    def krea_current(self):
        """Is this train job still the avatar's current Krea LoRA build (not replaced, no newer build)?"""
        try:
            avatar = json.loads(s3.get_object(Bucket=BUCKET, Key=f"avatars/{self.client}/{self.params['avatar']}/"
                                                                  "avatar.json")["Body"].read())
        except s3.exceptions.NoSuchKey:
            return False
        created = self.params.get("avatar_created_at")
        return (self.job["job_id"] in ((avatar.get("krea_lora") or {}).get("job_ids") or [])
                and (created is None or avatar.get("created_at") in (None, created)))

    def krea_env(self):
        if not os.path.exists(f"{MUSUBI}/krea2_train_network.py"):  # before any big download
            raise RuntimeError("this trainer's musubi-tuner has no Krea 2 support; bump musubi_commit and rebuild it")
        marker = os.path.join(KREA_VENV, ".ready")
        if os.path.exists(marker):
            return
        self.progress("installing the Krea 2 training venv (first Krea job, a few minutes)", force=True)
        logp = os.path.join(self.jd, "krea_env.log")
        run(["python3", "-m", "venv", KREA_VENV], logp)
        run([f"{KREA_VENV}/bin/pip", "install", "-q", "--upgrade", "pip", "wheel"], logp)
        run([f"{KREA_VENV}/bin/pip", "install", "-q", *KREA_TORCH], logp)
        run([f"{KREA_VENV}/bin/pip", "install", "-q", "-e", f"{ROOT}/musubi-tuner"], logp)
        with open(marker, "w") as f:
            f.write("ok")

    def krea_prefetch(self):
        """Start getting the Krea models now, in the background, so the download overlaps prepare, caption
        and the venv build instead of following them. Only on a trainer whose musubi-tuner has Krea support
        (krea_env reports the other case) and while this job still needs them."""
        if "krea_models" not in self.state["done"] and os.path.exists(f"{MUSUBI}/krea2_train_network.py"):
            krea_fetch_start()

    def krea_models(self):
        if not _krea_missing() and _KREA_FETCH["thread"] is None:
            return
        self.progress("getting the Krea 2 models (first Krea job, ~66 GB)", force=True)
        stats = krea_fetch_wait(self.progress)
        if stats:
            self.state["krea_download"] = {k: stats[k] for k in ("source", "bytes", "seconds")}

    def krea_dataset_toml(self):
        items, cache = os.path.join(self.jd, "items"), os.path.join(self.jd, "cache")
        path = os.path.join(self.jd, "dataset.toml")
        with open(path, "w") as f:
            f.write(f"""[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "{items}"
cache_directory = "{cache}/image"
num_repeats = 1
""")
        return path

    def krea_cache(self):
        self.progress("caching latents and text", force=True)
        toml, logp = self.krea_dataset_toml(), os.path.join(self.jd, "cache.log")
        py = f"{KREA_VENV}/bin/python"
        run([py, f"{MUSUBI}/krea2_cache_latents.py", "--dataset_config", toml, "--vae", KREA_VAE], logp)
        run([py, f"{MUSUBI}/krea2_cache_text_encoder_outputs.py", "--dataset_config", toml, "--text_encoder", KREA_TE,
             "--batch_size", "4"], logp)

    def krea_train(self):
        target, trigger = int(self.params["steps"]), self.params["trigger"]
        st = self.state.setdefault("krea2", {"runs": []})
        out = os.path.join(self.jd, "out")
        os.makedirs(out, exist_ok=True)
        done, state_dir = plan.resume_point("krea2", st["runs"], os.listdir(out))
        if done >= target and st.get("final"):
            return
        run_no = len(st["runs"])
        st["runs"].append({"run": run_no, "base": done})
        self.save()
        samples = os.path.join(self.jd, "samples.txt")
        with open(samples, "w") as f:
            for i, scene in enumerate(("close-up portrait, soft window light",
                                       "full body, standing on a city street at night",
                                       "sitting in a cafe reading a book, medium shot")):
                f.write(f"{trigger}, {scene} --w 1024 --h 1024 --s 8 --l 1 --d {42 + i}\n")
        cmd = [f"{KREA_VENV}/bin/accelerate", "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16",
               f"{MUSUBI}/krea2_train_network.py", "--dit", KREA_RAW, "--vae", KREA_VAE, "--text_encoder", KREA_TE,
               "--dataset_config", os.path.join(self.jd, "dataset.toml"), "--sdpa", "--mixed_precision", "bf16",
               "--timestep_sampling", "shift", "--weighting_scheme", "none", "--discrete_flow_shift", "2.5",
               "--optimizer_type", "adamw8bit", "--learning_rate", "1e-4", "--gradient_checkpointing",
               "--max_data_loader_n_workers", "2", "--persistent_data_loader_workers",
               "--network_module", "networks.lora_krea2", "--network_dim", "32", "--network_alpha", "32",
               "--max_train_steps", str(target - done), "--save_every_n_steps", str(KREA_SAVE_EVERY), "--save_state",
               "--sample_prompts", samples, "--sample_every_n_steps", str(KREA_SAVE_EVERY), "--turbo_dit", KREA_TURBO,
               "--seed", "42", "--output_dir", out, "--output_name", f"krea2_r{run_no}"]
        if state_dir:
            cmd += ["--resume", os.path.join(out, state_dir)]

        def on_steps(rel, _total, rate):
            left = int((target - done - rel) * (rate or 3.5))
            self.progress(f"Krea 2 LoRA {done + rel}/{target} · ~{left // 60} min left")
        self.progress(f"Krea 2 LoRA {done}/{target}" + (f" (resumed at {done})" if done else ""), force=True)
        run(cmd, os.path.join(self.jd, "krea2.log"), on_steps=on_steps)
        final = f"krea2_r{run_no}.safetensors"
        if not os.path.exists(os.path.join(out, final)):
            raise RuntimeError(f"training finished without {final}")
        st["final"] = final
        self.save()

    def krea_upload(self):
        out, lora, version = os.path.join(self.jd, "out"), self.params["lora"], int(self.params["version"])
        base = f"loras/{self.client}/{lora}/"
        prefix = f"{base}v{version}/"
        self.progress(f"uploading {lora} v{version}", force=True)
        s3.upload_file(os.path.join(out, self.state["krea2"]["final"]), BUCKET, prefix + "krea2.safetensors")
        sample_dir = os.path.join(out, "sample")
        samples = sorted(os.listdir(sample_dir)) if os.path.isdir(sample_dir) else []
        for n in samples:
            s3.upload_file(os.path.join(sample_dir, n), BUCKET, prefix + "samples/" + n,
                           ExtraArgs={"ContentType": "image/png"})
        m = self.state.get("manifest") or {}
        meta = {"name": lora, "version": version, "source": "avatar", "avatar": self.params.get("avatar"),
                "models": ["krea2"], "trigger": self.params["trigger"], "files": ["krea2.safetensors"],
                "single_file": True, "steps": int(self.params["steps"]), "rank": 32, "resolution": [1024, 1024],
                "base": {"train": "krea/Krea-2-Raw", "render": "krea/Krea-2-Turbo"},
                "dataset": {"fingerprint": self.params.get("fingerprint"), "files": m.get("file_count")},
                "samples": [f"samples/{n}" for n in samples], "job_id": self.job["job_id"],
                "created_at": int(time.time()),
                "train_seconds": int(time.time() - int(self.job.get("started_at") or time.time()))}
        s3.put_object(Bucket=BUCKET, Key=prefix + "lora.json", Body=json.dumps(meta, indent=2).encode(),
                      ContentType="application/json")
        if self.krea_current():  # a build replaced meanwhile is kept as a version, but never made current
            s3.put_object(Bucket=BUCKET, Key=base + "current.json", Body=json.dumps({"version": version}).encode(),
                          ContentType="application/json")
        self.state["result"] = {"lora": lora, "version": version, "prefix": prefix, "model": "krea2",
                                "trigger": self.params["trigger"], "avatar": self.params.get("avatar"),
                                "photos": self.state.get("photos"), "steps": int(self.params["steps"])}
        if self.state.get("krea_download"):  # this job fetched the models: where from and how long it took
            self.state["result"]["krea_download"] = self.state["krea_download"]

    def krea_link(self):
        r = self.state["result"]
        _update_avatar_krea_lora(self.client, self.params["avatar"], self.job["job_id"], state="ready",
                                 version=r["version"], ready={"lora": r["lora"], "version": r["version"]}, error=None)

    def run_all(self):
        if self.params.get("model") == "krea2":
            if not self.krea_current():
                return {"skipped": "the avatar was replaced or a newer Krea LoRA build started"}
            self.krea_prefetch()
            self.phase("prepare", self.prepare)
            self.phase("caption", self.caption)  # the photo set's captions are saved: nothing to caption
            self.phase("krea_env", self.krea_env)
            self.phase("krea_models", self.krea_models)
            self.phase("krea_cache", self.krea_cache)
            self.phase("krea_train", self.krea_train)
            self.phase("krea_upload", self.krea_upload)
            self.phase("krea_link", self.krea_link)
            return self.state["result"]
        if self.avatar_wan() and not self.wan_current():
            return {"skipped": "the avatar was replaced or a newer Wan LoRA build started"}
        self.phase("prepare", self.prepare)
        self.phase("caption", self.caption)
        if self.params.get("captions_only"):
            return {"captions_only": True, "captions": self.state.get("captions", 0),
                    "segments": self.state["segments"], "photos": self.state["photos"]}
        self.phase("cache", self.cache)
        self.phase("t2v_low", lambda: self.train("t2v_low"))
        if self.avatar_wan() and not self.wan_current():  # replaced meanwhile: don't spend the second expert
            return {"skipped": "the avatar was replaced or a newer Wan LoRA build started"}
        self.phase("t2v_high", lambda: self.train("t2v_high"))
        self.phase("upload", self.upload)
        if self.avatar_wan():
            if self.state.get("result", {}).get("skipped"):
                return self.state["result"]
            self.phase("wan_link", self.wan_link)
        else:
            self.phase("preview", self.queue_preview)
        return self.state["result"]


def _hf_token():
    ssm = boto3.client("ssm", region_name=REGION)
    return ssm.get_parameter(Name=HF_TOKEN_PARAM, WithDecryption=True)["Parameter"]["Value"]


def _hf_download(repo, filename, local_dir, token, revision):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_dir=local_dir, token=token)


def _krea_missing():
    return [(r, f, rev) for r, f, rev in KREA_FILES if not os.path.exists(os.path.join(KREA_DIR, f))]


def krea_fetch():
    """The Krea 2 training files on this disk: from the model mirror when it has them, the rest from
    Hugging Face, all files at once (then queued for the mirror). Returns where they came from, the bytes
    and the seconds; logs them as model_download."""
    t0 = time.time()
    wanted = _krea_missing()
    restored = mirror.pull(KREA_UNIT, KREA_DIR)
    missing = _krea_missing()
    if missing:
        token = _hf_token()
        with ThreadPoolExecutor(len(missing)) as pool:
            jobs_ = [(repo, pool.submit(_hf_download, repo, f, KREA_DIR, token, rev)) for repo, f, rev in missing]
        failed = [(repo, fut.exception()) for repo, fut in jobs_ if fut.exception() is not None]
        for repo, e in failed:
            if repo == "krea/Krea-2-Raw":
                raise RuntimeError("no access to krea/Krea-2-Raw: accept its license on huggingface.co with the "
                                   f"account of the SSM token ({type(e).__name__})") from e
        if failed:
            raise failed[0][1]
        mirror.push_async(KREA_UNIT, KREA_DIR, files=[f for _, f, _ in KREA_FILES])
    secs = round(time.time() - t0, 1)
    nbytes = sum(os.path.getsize(os.path.join(KREA_DIR, f)) for _, f, _ in wanted)
    source = "+".join(x for x in ("mirror" if restored else "", "huggingface" if missing else "") if x)
    stats = {"unit": KREA_UNIT, "source": source, "bytes": nbytes, "seconds": secs,
             "mb_per_s": round(nbytes / 1e6 / max(secs, 0.1))}
    log(event="model_download", **stats)
    return stats


# One fetch at a time in this process: a Krea job starts it early (Job.krea_prefetch), its krea_models
# phase waits for it, and a job that arrives while one runs joins it.
_KREA_FETCH = {"thread": None, "result": None, "error": None}
_FETCH_LOCK = threading.Lock()


def krea_fetch_start():
    """Start the fetch in the background unless one is running or nothing is missing. Returns its thread."""
    with _FETCH_LOCK:
        t = _KREA_FETCH["thread"]
        if t is not None and t.is_alive():
            return t
        if not _krea_missing():
            return None

        def work():
            try:
                _KREA_FETCH["result"] = krea_fetch()
            except BaseException as e:  # raised to the job that waits for it
                _KREA_FETCH["error"] = e

        _KREA_FETCH.update(result=None, error=None)
        t = _KREA_FETCH["thread"] = threading.Thread(target=work, daemon=True, name="krea-fetch")
        t.start()
        return t


def krea_fetch_wait(progress):
    """Wait for the fetch (starting it if needed) and return its stats, or raise its error. `progress` is
    called while waiting, so a cancel stops the job; the download itself goes on for the next job."""
    krea_fetch_start()  # an earlier attempt that failed is tried again; a running or finished one is kept
    t = _KREA_FETCH["thread"]
    while t is not None and t.is_alive():
        t.join(FETCH_POLL)
        progress("getting the Krea 2 models (first Krea job, ~66 GB)")
    with _FETCH_LOCK:
        result, error = _KREA_FETCH["result"], _KREA_FETCH["error"]
        _KREA_FETCH.update(thread=None, result=None, error=None)
    if error is not None:
        raise error
    return result


def mirror_backfill(hf_home=None):
    """Queue for the model mirror the weights that are complete on this disk (mirror.py uploads only what
    the bucket lacks, in the background): the Wan and Krea training files and the caption model. This is how
    a running trainer seeds the mirror for the next new disk. Never stops the trainer."""
    try:
        mirror.push_async("trainer/wan", MODELS, files=WAN_FILES)  # skipped while a file is missing
        mirror.push_async(KREA_UNIT, KREA_DIR, files=[f for _, f, _ in KREA_FILES])
        name = "models--" + VLM_MODEL.replace("/", "--")
        folder = os.path.join(hf_home or os.environ.get("HF_HOME", "/opt/models"), "hub", name)
        if os.path.isdir(folder) and not any(f.endswith(".incomplete") for _, _, fs in os.walk(folder) for f in fs):
            mirror.push_async(f"hub/{name}", folder)
    except Exception as e:
        log(event="mirror_backfill", error=f"{type(e).__name__}: {e}")


def _update_avatar_krea_lora(client, name, job_id, **fields):
    return _update_avatar_lora(client, name, job_id, "krea_lora", **fields)


def _update_avatar_lora(client, name, job_id, record, **fields):
    """Set fields on the avatar's krea_lora / wan_lora record, only while this job is still the avatar's
    current build (a field set to None is removed)."""
    key = f"avatars/{client}/{name}/avatar.json"
    try:
        avatar = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except s3.exceptions.NoSuchKey:
        return False
    kl = avatar.get(record) or {}
    if job_id not in (kl.get("job_ids") or []):
        return False
    kl.update(fields)
    for k in [k for k, v in fields.items() if v is None]:
        kl.pop(k)
    avatar[record] = kl
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(avatar).encode(), ContentType="application/json")
    return True


def krea_failed(job, error):
    """An avatar LoRA build (Krea or Wan) failed: show it on the avatar (its last ready LoRA stays linked)."""
    params = json.loads(job.get("params") or "{}")
    record = {"krea2": "krea_lora", "t2v": "wan_lora"}.get(params.get("model"))
    if record and params.get("avatar") and (record == "krea_lora" or params.get("lora")):
        try:
            _update_avatar_lora(job["client"], params["avatar"], job["job_id"], record, state="failed",
                                error=str(error)[:500])
        except Exception as e:  # never hide the job's own error
            log(event="krea_failed_update_error", error=str(e)[:200])


# ---------------------------------------------------------------- loop

def process(job_id, receipt):
    job = jobs.get_item(Key={"job_id": job_id}).get("Item")
    if not job or job["status"] in FINISHED:
        log(event="skip", job_id=job_id)
        return
    if job.get("cancel_requested"):
        set_status(job_id, "cancelled", finished_at=int(time.time()))
        return
    jd = os.path.join(ROOT, "jobs", job_id)
    resumed = os.path.isdir(jd)
    extra = {} if resumed and job.get("started_at") else {"started_at": int(time.time())}
    set_status(job_id, "running", progress="resuming after an interruption" if resumed else "starting", **extra)
    job.update(extra)
    log(event="start", job_id=job_id, resumed=resumed)
    CANCEL.clear()
    watch = JobWatch(job_id, receipt)
    watch.start()
    keep = False
    try:
        result = Job(job, jd).run_all()
        set_status(job_id, "done", result=json.dumps(result), finished_at=int(time.time()), progress="")
        log(event="done", job_id=job_id, seconds=int(time.time() - watch.started))
    except Cancelled as e:
        status = "cancelled" if str(e) == "cancelled" else "failed"
        set_status(job_id, status, error=str(e), finished_at=int(time.time()))
        krea_failed(job, e)  # cancelled or timed out: the avatar's build can be retried
    except Exception as e:
        traceback.print_exc()
        set_status(job_id, "failed", error=str(e)[:1000], finished_at=int(time.time()))
        krea_failed(job, e)
    except BaseException:  # shutting down mid-job (spot stop): keep the folder to resume
        keep = True
        raise
    finally:
        watch.stopped.set()
        if not keep:
            shutil.rmtree(jd, ignore_errors=True)


def _instance_id():
    token = urllib.request.urlopen(urllib.request.Request(
        "http://169.254.169.254/latest/api/token", method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}), timeout=2).read().decode()
    return urllib.request.urlopen(urllib.request.Request(
        "http://169.254.169.254/latest/meta-data/instance-id",
        headers={"X-aws-ec2-metadata-token": token}), timeout=2).read().decode()


def _queue_empty():
    a = sqs.get_queue_attributes(QueueUrl=QUEUE_URL, AttributeNames=[
        "ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"])["Attributes"]
    return int(a["ApproximateNumberOfMessages"]) + int(a["ApproximateNumberOfMessagesNotVisible"]) == 0


def main():
    log(event="trainer_up", idle_seconds=IDLE_SECONDS, job_timeout_seconds=JOB_TIMEOUT_SECONDS)
    mirror_backfill()
    last_work = time.time()
    while True:
        msgs = sqs.receive_message(QueueUrl=QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=20,
                                   VisibilityTimeout=VISIBILITY_SECONDS).get("Messages", [])
        if not msgs:
            if time.time() - last_work > IDLE_SECONDS and _queue_empty():
                log(event="idle_stop")
                ec2.stop_instances(InstanceIds=[_instance_id()])
                time.sleep(300)
            continue
        m = msgs[0]
        try:
            process(json.loads(m["Body"])["job_id"], m["ReceiptHandle"])
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=m["ReceiptHandle"])
        finally:
            last_work = time.time()


if __name__ == "__main__":
    main()
