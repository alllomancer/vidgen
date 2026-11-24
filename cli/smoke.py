"""End-to-end smoke test: runs one small job through every model and code
path on the real GPU instance and reports pass/fail with timings.

    vidgen.py smoke            (or: make smoke)

Steps, in queue order (the worker runs one job at a time). The order groups jobs by
model, because on a g6e.4xlarge each switch between the big models reloads ~60 GB
(~5 min): three Wan T2V jobs and the story share one T2V load, then the avatar video
needs the keyframe and I2V models once each.
  1. avatar     2 candidate portraits (Wan T2V, single frame)
  2. video      2 s text-to-video, no avatar (Wan T2V + MMAudio sound)
  3. cancel     a 5 s video cancelled once it is running (cancel path)
  4. story      2 shots x 2 s, no avatars (Qwen2.5-VL plan, Wan T2V, Kokoro narration,
                subtitles, sound mix, ffmpeg assembly)
  5. avatar-vid 2 s video starring two avatars (multi-avatar Qwen-Image-Edit keyframe +
                Wan I2V + sound)
  6. photos     3 photos of smoke-a: one --prompt, one --location, one random location
                (Qwen-Image-Edit; queued just before avatar-vid so they share one edit-model load)
  7. krea       2 photos without a person, one --prompt and one --location (Krea 2 Turbo in
                its own venv + content filter); queued after the edit-model jobs
  8. describe   metadata for the step-2 video (Qwen2.5-VL video input)

Avatars "smoke-a" and "smoke-b" are created without touching your default.
Outputs go to <download dir>/smoke-<timestamp>/. Takes ~30-40 min on a warm
instance; the GPU stops itself afterwards as usual.
"""

import os
import struct
import subprocess
import threading
import time

import vidgen

POLL = 15
MIN_VIDEO_BYTES = 10_000
MAX_NETWORK_ERRORS = 20  # consecutive failed polls (~5 min) before giving up


def _wait(job_id, label, until=("done", "failed", "cancelled"), timeout=3 * 3600):
    start, last, errors = time.time(), None, 0
    while time.time() - start < timeout:
        # A dropped poll is a client network blip, not a failed job: the job keeps running.
        try:
            job = vidgen.api("GET", f"/jobs/{job_id}")
            errors = 0
        except vidgen.ApiError as e:
            if not str(e).startswith("network error") or errors >= MAX_NETWORK_ERRORS:
                raise
            errors += 1
            vidgen.log(f"  {label}: {e}; retrying ({errors}/{MAX_NETWORK_ERRORS})")
            time.sleep(POLL)
            continue
        line = f"{job['status']} {job.get('progress', '')}".strip()
        if line != last:
            vidgen.log(f"  {label}: {line}")
            last = line
        if job["status"] in until:
            return job
        time.sleep(POLL)
    raise TimeoutError(f"{label} still {last} after {timeout // 60} min")


def _probe(path):
    """(duration seconds, has audio) via ffprobe if installed, else (None, None)."""
    if not vidgen.shutil.which("ffprobe"):
        return None, None
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", path],
        capture_output=True, text=True,
    )
    info = vidgen.json.loads(out.stdout or "{}")
    duration = float(info.get("format", {}).get("duration", 0) or 0)
    return duration, any(s.get("codec_type") == "audio" for s in info.get("streams", []))


class Results:
    def __init__(self):
        self.rows = []

    def add(self, step, ok, seconds, detail):
        self.rows.append((step, ok, seconds, detail))
        vidgen.log(f"{'PASS' if ok else 'FAIL'} {step}: {detail}")

    def report(self):
        print("\n" + "=" * 72)
        for step, ok, seconds, detail in self.rows:
            secs = f"{seconds / 60:5.1f} min" if seconds is not None else "        -"
            print(f"{'PASS' if ok else 'FAIL'}  {step:11} {secs}  {detail}")
        failed = sum(1 for r in self.rows if not r[1])
        print("=" * 72)
        print("all passed" if not failed else f"{failed} step(s) failed; see the worker log for details:\n"
              "  journalctl -u ai-video-worker -n 300   (on the instance, via terraform output ssh_command)")
        return 1 if failed else 0


def _cancel_when_running(job_id, out):
    """Background thread: cancel the job the moment it is running. It runs while the main thread
    waits on other steps, because the worker may reach this job (and finish it) before them."""
    try:
        job = _wait(job_id, "cancel", until=("running", "done", "failed", "cancelled"))
        if job["status"] == "running":
            try:
                vidgen.api("POST", f"/jobs/{job_id}/cancel")
            except vidgen.ApiError as e:
                if e.status != 409:  # 409: it finished between the poll and the cancel
                    raise
            job = _wait(job_id, "cancel", timeout=600)
        out["job"] = job
    except Exception as e:  # reported as a FAIL row, never a crash of the whole run
        out["error"] = f"{type(e).__name__}: {e}"


def _job_seconds(job):
    if job.get("started_at") and job.get("finished_at"):
        return int(job["finished_at"]) - int(job["started_at"])
    return None


def _download_all(job, folder, base):
    saved = {}
    for name, url in job.get("downloads", {}).items():
        ext = os.path.splitext(vidgen.urllib.parse.urlparse(url).path)[1]
        path = os.path.join(folder, f"{base}.{name}{ext}")
        vidgen.download(url, path)
        saved[name] = path
    return saved


def _check_video(results, step, job, folder, expect_seconds=None, expect_audio=False):
    if job["status"] != "done":
        results.add(step, False, _job_seconds(job), f"{job['status']}: {job.get('error', '')}")
        return None
    files = _download_all(job, folder, step)
    video = files.get("video")
    if not video or os.path.getsize(video) < MIN_VIDEO_BYTES:
        results.add(step, False, _job_seconds(job), "no video, or video suspiciously small")
        return None
    duration, has_audio = _probe(video)
    problems = []
    if duration is not None and expect_seconds and abs(duration - expect_seconds) > 1.5:
        problems.append(f"duration {duration:.1f}s, expected ~{expect_seconds}s")
    if expect_audio and has_audio is False:
        problems.append("no audio track")
    detail = f"{os.path.getsize(video) // 1024} KB" + (f", {duration:.1f}s" if duration else "")
    detail += f", files: {', '.join(sorted(files))}"
    results.add(step, not problems, _job_seconds(job), "; ".join(problems) or detail)
    return video


PHOTO_SIZE = (896, 1152)  # portrait
PHOTO_SCENES = [{"prompt": "reading a newspaper at a sidewalk café, morning light"},
                {"location": "a white sand beach at golden hour"}]


def _png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def _check_photos(results, job, folder):
    if job["status"] != "done":
        results.add("photos", False, _job_seconds(job), f"{job['status']}: {job.get('error', '')}")
        return
    files = _download_all(job, folder, "photos")
    sources = [s["source"] for s in job.get("result", {}).get("scenes", [])]
    sizes = [_png_size(files[k]) for k in sorted(files)]
    problems = []
    if len(files) != 3:
        problems.append(f"{len(files)} photos, expected 3")
    if sources != ["prompt", "location", "random"]:
        problems.append(f"scene sources {sources}")
    if any(size != PHOTO_SIZE for size in sizes):
        problems.append(f"sizes {sizes}, expected {PHOTO_SIZE}")
    random_place = job.get("result", {}).get("scenes", [{}] * 3)[-1].get("text")
    results.add("photos", not problems, _job_seconds(job),
                "; ".join(problems) or f"3 x {PHOTO_SIZE[0]}x{PHOTO_SIZE[1]}, random: {random_place!r}")


KREA_SCENES = [{"prompt": "a steaming cup of coffee on a wooden table by a rainy window"},
               {"location": "a misty pine forest at dawn"}]


def _check_krea(results, job, folder):
    if job["status"] != "done":
        results.add("krea", False, _job_seconds(job), f"{job['status']}: {job.get('error', '')}")
        return
    files = _download_all(job, folder, "krea")
    result = job.get("result", {})
    sources = [s["source"] for s in result.get("scenes", [])]
    sizes = [_png_size(files[k]) for k in sorted(files)]
    filtered = (result.get("krea") or {}).get("filtered", 0)
    problems = []
    if len(files) != 2:
        problems.append(f"{len(files)} photos, expected 2 ({filtered} flagged by the content filter)")
    if sources != ["prompt", "location"]:
        problems.append(f"scene sources {sources}")
    if any(size != PHOTO_SIZE for size in sizes):
        problems.append(f"sizes {sizes}, expected {PHOTO_SIZE}")
    results.add("krea", not problems, _job_seconds(job),
                "; ".join(problems) or f"2 x {PHOTO_SIZE[0]}x{PHOTO_SIZE[1]}, model {result.get('model')!r}")


def run(skip_cancel=False):
    folder = os.path.join(vidgen.download_dir(), f"smoke-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(folder)
    results = Results()
    vidgen.log(f"smoke test starting; outputs in {folder}")
    vidgen.log(f"instance: {vidgen.api('GET', '/status')}")

    def submit(body):
        return vidgen.api("POST", "/jobs", body)["job_id"]

    # Phase 1: everything that runs on Wan T2V, queued together so it shares one model load.
    avatar_job = submit({"type": "avatar", "candidates": 2, "seed": 1234,
                         "description": "a smoke-test character: a man in his 40s with a grey beard, yellow raincoat"})
    video_job = submit({"type": "video", "prompt": "a paper boat floating on a calm pond, slow pan",
                        "seconds": 2, "avatar": "none", "seed": 1})
    cancel_job = None if skip_cancel else submit({"type": "video", "prompt": "a slow timelapse of clouds",
                                                  "seconds": 5, "avatar": "none"})
    cancel_out = {}
    cancel_thread = None
    if cancel_job:
        cancel_thread = threading.Thread(target=_cancel_when_running, args=(cancel_job, cancel_out), daemon=True)
        cancel_thread.start()
    story_job = submit({
        "type": "story", "max_shots": 2, "seconds_per_shot": 2, "avatar": "none",
        "narration": True, "subtitles": True, "voice": "af_heart",
        "story": "Two old friends meet at a train station after many years. They laugh and walk out into the rain together.",
    })

    # Avatars from the two candidates (not made default).
    job = _wait(avatar_job, "avatar")
    if job["status"] == "done" and len(job.get("downloads", {})) >= 2:
        _download_all(job, folder, "avatar")
        for name, n in (("smoke-a", 1), ("smoke-b", 2)):
            vidgen.api("POST", "/avatars", {"name": name, "job_id": avatar_job, "candidate": n,
                                            "make_default": False})
        results.add("avatar", True, _job_seconds(job), "2 candidates; saved as smoke-a, smoke-b")
        # Phase 2: the keyframe + I2V path, with both avatars in one shot (multi-avatar keyframe).
        # Photos first: the edit model it loads stays resident for avatar-vid's keyframe.
        photos_job = submit({"type": "photos", "avatar": "smoke-a", "scenes": PHOTO_SCENES, "count": 3, "seed": 3})
        avatar_video_job = submit({"type": "video", "prompt": "two friends walking along a beach at sunset, tracking shot",
                                   "seconds": 2, "avatar": "smoke-a,smoke-b", "seed": 2})
    else:
        results.add("avatar", False, _job_seconds(job), f"{job['status']}: {job.get('error', 'no candidates')}")
        avatar_video_job = photos_job = None

    # Krea 2 runs in its own process and unloads the other models, so it goes after the edit-model jobs.
    krea_job = submit({"type": "photos", "model": "krea2", "avatar": "none", "scenes": KREA_SCENES, "count": 2,
                       "seed": 4})

    plain_video = _check_video(results, "video", _wait(video_job, "video"), folder, expect_seconds=2,
                               expect_audio=True)
    if plain_video:
        key = vidgen.upload_file(plain_video)
        describe_job = submit({"type": "describe", "upload_key": key})
    else:
        describe_job = None

    if cancel_thread:
        cancel_thread.join(timeout=3 * 3600 + 600)
        job = cancel_out.get("job")
        if job:
            results.add("cancel", job["status"] == "cancelled", None,
                        f"ended as {job['status']}" + (f" ({job.get('error')})" if job.get("error") else "")
                        + ("; it finished before the cancel reached it" if job["status"] == "done" else ""))
        else:
            results.add("cancel", False, None, cancel_out.get("error", "still waiting after the timeout"))

    _check_video(results, "story", _wait(story_job, "story"), folder, expect_seconds=4, expect_audio=True)

    if avatar_video_job:
        _check_video(results, "avatar-vid", _wait(avatar_video_job, "avatar-vid"), folder, expect_seconds=2,
                     expect_audio=True)
    else:
        results.add("avatar-vid", False, None, "skipped: no avatars")

    if photos_job:
        _check_photos(results, _wait(photos_job, "photos"), folder)
    else:
        results.add("photos", False, None, "skipped: no avatars")

    _check_krea(results, _wait(krea_job, "krea"), folder)

    if describe_job:
        job = _wait(describe_job, "describe")
        meta = job.get("result", {}).get("metadata", {})
        ok = job["status"] == "done" and bool(meta.get("title")) and bool(meta.get("description"))
        if ok:
            with open(os.path.join(folder, "describe.json"), "w") as f:
                vidgen.json.dump(meta, f, indent=2)
        results.add("describe", ok, _job_seconds(job), f"title: {meta.get('title')!r}" if ok else
                    f"{job['status']}: {job.get('error', 'missing title/description')}")
    else:
        results.add("describe", False, None, "skipped: no video to describe")

    vidgen.log(f"outputs in {folder}; look at them, as the test only checks files exist and have sane lengths")
    return results.report()
