"""AI-look check: after a render, a vision-language model critiques how AI-made the output looks and suggests a
prompt and realism settings likely to look more real. A critique of visible artifacts, not a detector.
docs/specs/2026-11-27-realism-check.md

The GPU part is models.judge_realism (one Qwen2.5-VL call); everything here is plain Python + ffmpeg so it can be
tested anywhere. check() never raises for a failed check: the job still succeeds with {"error": ...}.
"""
import json
import os
import re
import subprocess
from collections import Counter

import realism

TYPES = ("video", "photos", "story")
FRAMES = {"video": 4, "story": 6}
FRAME_SIDE = 768
MAX_SIGNS = 5
MAX_SIGN_CHARS = 80
MAX_PROMPT_CHARS = 2000
NOTE = "An AI critique of visible artifacts, not a detector."
OPTION_KEYS = ("look", "natural_skin", "candid", "grain", "shot")


def frame_times(duration, n):
    """n moments spread over the clip, away from the cut-in and cut-out: 1/2n, 3/2n, ... of the duration."""
    return [round(duration * (2 * i + 1) / (2 * n), 3) for i in range(n)]


def _duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, timeout=60)
    return float(out.stdout.strip())


def extract_frames(video, n, workdir, name):
    """n still frames (png, long side FRAME_SIDE) from a video."""
    paths = []
    for i, t in enumerate(frame_times(_duration(video), n)):
        path = os.path.join(workdir, f"{name}_{i}.png")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(t), "-i", video,
                        "-frames:v", "1", "-vf", f"scale='if(gt(iw,ih),{FRAME_SIDE},-2)':'if(gt(iw,ih),-2,{FRAME_SIDE})'",
                        path], check=True)
        paths.append(path)
    return paths


def _allowed(kind):
    shots = f"; shot (videos only): one of {', '.join(realism.SHOTS)} or omit" if kind == "video" else ""
    return (f"look: one of {', '.join(realism.LOOKS)}; natural_skin, candid, grain: true or false{shots}")


def instruction(kind, prompt, current):
    """The question for the VLM. kind: photo | video | story; prompt: what was asked for (None for stories)."""
    what = {"photo": "this photo", "video": "these frames from one video clip, in order",
            "story": "these frames from a short film, in order"}[kind]
    asked = f'It was generated from this prompt:\n"""{prompt}"""\n' if prompt else ""
    rewrite = ('"prompt": the prompt rewritten so the result looks like real, unretouched '
               + ("photography" if kind == "photo" else "footage")
               + " (keep the subject, place and action; change what causes the giveaways), "
               if kind != "story" else "")
    return (
        f"You are reviewing {what}, made by an image/video generator. Judge how obviously AI-generated it looks to "
        "a careful viewer, from what is visible: skin and hair texture, hands and fingers, eyes and teeth, text and "
        "logos, lighting and shadows, reflections, backgrounds, colour, symmetry and anything implausible.\n"
        f"{asked}Its realism settings were: {json.dumps(current)}.\n"
        "Answer with JSON only: {"
        '"score": integer 0-10 (0-2 could pass as real, 3-5 some giveaways, 6-8 clearly generated, 9-10 obviously '
        'AI), "signs": up to 5 short phrases naming the visible giveaways (most important first), '
        f"{rewrite}"
        f'"options": realism settings to use next time ({_allowed(kind)})'
        "}."
    )


def _json_object(text):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        raise ValueError("no JSON in the answer")
    out = json.loads(m.group(0))
    if not isinstance(out, dict):
        raise ValueError("the answer is not an object")
    return out


def clean_options(options, kind):
    out = {}
    if not isinstance(options, dict):
        return out
    if options.get("look") in realism.LOOKS:
        out["look"] = options["look"]
    for k in ("natural_skin", "candid", "grain"):
        if isinstance(options.get(k), bool):
            out[k] = options[k]
    if kind == "video" and options.get("shot") in realism.SHOTS:
        out["shot"] = options["shot"]
    return out


def clean(answer, kind):
    """The VLM's answer as {score, signs, prompt?, options}; ValueError when it can't be read."""
    raw = _json_object(answer)
    score = raw.get("score")
    if isinstance(score, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", score):
        score = float(score)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("no score in the answer")
    out = {"score": max(0, min(10, int(round(score))))}
    signs = raw.get("signs") if isinstance(raw.get("signs"), list) else []
    out["signs"] = [re.sub(r"\s+", " ", s).strip()[:MAX_SIGN_CHARS] for s in signs if isinstance(s, str) and s.strip()][:MAX_SIGNS]
    if kind != "story" and isinstance(raw.get("prompt"), str) and raw["prompt"].strip():
        out["prompt"] = raw["prompt"].strip()[:MAX_PROMPT_CHARS]
    out["options"] = clean_options(raw.get("options"), kind)
    return out


def aggregate_photos(per_photo):
    """The job's verdict from each photo's: the worst score, the most frequent signs, options any photo asked for
    (a setting turned on by one photo stays on; the look most photos asked for)."""
    signs = Counter(s.lower() for p in per_photo for s in p["signs"])
    first_spelling = {}
    for p in per_photo:
        for s in p["signs"]:
            first_spelling.setdefault(s.lower(), s)
    options = {}
    looks = Counter(p["options"]["look"] for p in per_photo if "look" in p["options"])
    if looks:
        options["look"] = looks.most_common(1)[0][0]
    for k in ("natural_skin", "candid", "grain"):
        votes = [p["options"][k] for p in per_photo if k in p["options"]]
        if votes:
            options[k] = any(votes)
    return {"score": max(p["score"] for p in per_photo),
            "signs": [first_spelling[s] for s, _ in signs.most_common(MAX_SIGNS)], "options": options}


def current_settings(params, kind):
    out = {k: params.get(k) for k in ("look", "natural_skin", "candid", "grain") if k in params}
    if kind == "video" and params.get("shot"):
        out["shot"] = params["shot"]
    return out


def check(job_type, params, result, workdir, download, judge, progress=lambda msg: None):
    """result.realism for a finished job. download(key, path) fetches an output; judge(images, text) asks the VLM.
    Never raises (except cancellation, which judge raises): a failed check is {"error": ...}."""
    try:
        return _check(job_type, params, result, workdir, download, judge, progress)
    except Exception as e:  # noqa: BLE001 - the render is done; the check is extra
        if type(e).__name__ == "JobCancelled":
            raise
        reason = "the check couldn't read the model's answer" if isinstance(e, (ValueError, json.JSONDecodeError)) \
            else f"the check failed: {str(e)[:200]}"
        return {"error": reason, "checked": job_type, "note": NOTE}


def _check(job_type, params, result, workdir, download, judge, progress):
    folder = os.path.join(workdir, "realism")
    os.makedirs(folder, exist_ok=True)
    progress("checking how AI it looks")
    if job_type == "photos":
        per = []
        scenes = {s.get("photo"): s.get("text", "") for s in result.get("scenes") or []}
        keys = sorted(((int(m.group(1)), v) for k, v in result.items()
                       if (m := re.fullmatch(r"photo_(\d+)_key", k))), key=lambda kv: kv[0])
        if not keys:
            raise RuntimeError("no photos to check")
        for n, key in keys:
            progress(f"checking how AI it looks: photo {n}/{len(keys)}")
            path = os.path.join(folder, f"photo_{n}.png")
            download(key, path)
            scene = scenes.get(n, "")
            verdict = clean(judge([path], instruction("photo", scene, current_settings(params, "photos"))), "photo")
            per.append({"photo": n, "score": verdict["score"], "signs": verdict["signs"],
                        "scene": verdict.get("prompt", scene), "options": verdict["options"]})
        out = aggregate_photos(per)
        out.update(scenes=[p["scene"] for p in per], photos=[{k: p[k] for k in ("photo", "score", "signs")} for p in per])
    else:
        key = result.get("video_key")
        if not key:
            raise RuntimeError("no video to check")
        video = os.path.join(folder, "video.mp4")
        download(key, video)
        frames = extract_frames(video, FRAMES[job_type], folder, "frame")
        prompt = params.get("prompt") if job_type == "video" else None
        out = clean(judge(frames, instruction(job_type, prompt, current_settings(params, job_type))), job_type)
    out.update(checked=job_type, note=NOTE)
    return out
