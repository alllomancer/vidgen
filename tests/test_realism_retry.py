"""Retry until it looks real: the loop (stops, prompts, seeds, files, skip) and the API's checks.
docs/specs/2026-11-28-realism-retry.md"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("worker", os.path.join("lambda", "api"))
import realism_check as rc  # noqa: E402
import handler as h  # noqa: E402


def fake(job_type, scores, verdict_extra=None):
    """run/check/copy/seeds fakes: try n scores scores[n-1] and suggests "better n"."""
    runs, copies, n = [], [], [0]

    def run(p):
        runs.append(p)
        n[0] += 1
        if job_type == "photos":
            return {"photo_1_key": "outputs/m/j/photo_1.png", "scenes": [{"photo": 1, "text": "x"}], "seed": p["seed"]}
        return {"video_key": "outputs/m/j/video.mp4", "keyframe_key": "outputs/m/j/keyframe.png", "seed": p["seed"]}

    def check_one(p, res):
        s = scores[n[0] - 1]
        if isinstance(s, str):
            return {"skipped": s} if s.startswith("skipped") else {"error": s}
        v = {"score": s, "signs": [f"sign {n[0]}"]}
        v.update({"prompt": f"better {n[0]}"} if job_type == "video" else {"scenes": [f"better {n[0]}"]})
        return v

    def copy(key, name):
        copies.append((key, name))
        return f"outputs/m/j/{name}"

    seeds = iter(range(100, 200))
    return run, check_one, copy, (lambda: next(seeds)), runs, copies


progress = []
# ---- photos: 7 5 6 3 then six without a better score: stops after try 10, keeps try 4
scores = [7, 5, 6, 3, 4, 4, 4, 4, 4, 4, 1]
run, check_one, copy, seeds, runs, copies = fake("photos", scores)
params = {"scenes": [{"shot": {"where": "a cafe"}}], "count": 1, "realism_check": True, "realism_retries": 25,
          "reference_keys": ["avatars/m/dana/a.png"], "loras": [{"name": "x"}]}
out = rc.retry("photos", params, run, check_one, copy, progress.append, seeds)
assert len(runs) == 10 and out["realism_retry"]["tries"] == 10 and out["realism_retry"]["best_try"] == 4, out["realism_retry"]
assert out["realism_retry"]["stopped"] == "no better score in 6 tries"
assert out["photo_1_key"] == "outputs/m/j/attempt_4_photo_1.png" and out["realism"]["score"] == 3
assert [a["score"] for a in out["realism_retry"]["attempts"]] == scores[:10]
assert all(f"attempt_{i}_key" in out for i in range(1, 11)) and out["attempt_7_key"] == "outputs/m/j/attempt_7_photo_1.png"
# try 1 as asked; then a new seed and the best attempt's scene each time; avatar and LoRAs never change
assert runs[0]["scenes"] == [{"shot": {"where": "a cafe"}}]
assert [r["scenes"][0]["prompt"] for r in runs[1:]] == ["better 1", "better 2", "better 2", "better 4", "better 4",
                                                         "better 4", "better 4", "better 4", "better 4"]
assert len({r["seed"] for r in runs}) == 10
assert all(r["reference_keys"] == params["reference_keys"] and r["loras"] == params["loras"] for r in runs)
assert all("realism_check" not in r and "realism_retries" not in r for r in runs), "the loop checks; the render doesn't"
assert progress[0] == "try 1/25" and progress[4] == "try 5/25 · best 3/10", progress
# a seed given with the job: try 1 uses it
run, check_one, copy, seeds, runs, copies = fake("photos", [0])
rc.retry("photos", dict(params, seed=42), run, check_one, copy, progress.append, seeds)
assert runs[0]["seed"] == 42

# ---- video: stops at score 0; the clip and its keyframe are copied per try
run, check_one, copy, seeds, runs, copies = fake("video", [6, 2, 0, 5])
out = rc.retry("video", {"prompt": "a boat", "seconds": 5, "realism_check": True, "realism_retries": 25},
               run, check_one, copy, progress.append, seeds)
assert len(runs) == 3 and out["realism_retry"]["stopped"] == "score 0" and out["realism_retry"]["best_try"] == 3
assert [r["prompt"] for r in runs] == ["a boat", "better 1", "better 2"]
assert out["video_key"] == "outputs/m/j/attempt_3_video.mp4" and out["keyframe_key"] == "outputs/m/j/attempt_3_keyframe.png"
assert ("outputs/m/j/keyframe.png", "attempt_2_keyframe.png") in copies
assert out["realism_retry"]["attempts"][1] == {"try": 2, "score": 2, "signs": ["sign 2"], "prompt": "better 1", "seed": 101}

# ---- all tries used
run, check_one, copy, seeds, runs, copies = fake("video", [5, 4, 3])
out = rc.retry("video", {"prompt": "p", "realism_retries": 3}, run, check_one, copy, progress.append, seeds)
assert len(runs) == 3 and out["realism_retry"]["stopped"] == "3 tries" and out["realism"]["score"] == 3

# ---- a skipped (sexual) or failed check stops the loop and keeps the best so far
run, check_one, copy, seeds, runs, copies = fake("photos", [4, "skipped: the AI-look check doesn't critique sexual images"])
out = rc.retry("photos", params, run, check_one, copy, progress.append, seeds)
assert len(runs) == 2 and out["realism_retry"]["stopped"].startswith("skipped") and out["photo_1_key"].endswith("attempt_1_photo_1.png")
run, check_one, copy, seeds, runs, copies = fake("photos", ["skipped: x"])
out = rc.retry("photos", params, run, check_one, copy, progress.append, seeds)
assert len(runs) == 1 and out["realism"] == {"skipped": "skipped: x"} and out["photo_1_key"].endswith("attempt_1_photo_1.png")
run, check_one, copy, seeds, runs, copies = fake("video", [3, "the check failed: boom"])
out = rc.retry("video", {"prompt": "p", "realism_retries": 25}, run, check_one, copy, progress.append, seeds)
assert len(runs) == 2 and out["realism_retry"]["stopped"] == "the check failed: boom" and out["realism"]["score"] == 3

# ---- a render failing on a later try keeps the best so far; on try 1 it fails the job
run, check_one, copy, seeds, runs, copies = fake("photos", [5, 3, 9])
def failing(p, real=run):
    if len(runs) == 2:
        runs.append(p)
        raise RuntimeError("the content filter flagged the photo")
    return real(p)
out = rc.retry("photos", params, failing, check_one, copy, progress.append, seeds)
assert out["realism_retry"]["stopped"] == "try 3 failed: the content filter flagged the photo", out["realism_retry"]
assert out["realism_retry"]["best_try"] == 2 and out["photo_1_key"].endswith("attempt_2_photo_1.png")
try:
    rc.retry("photos", params, lambda p: (_ for _ in ()).throw(RuntimeError("OOM")), check_one, copy, progress.append, seeds)
    raise AssertionError("a first try that fails fails the job")
except RuntimeError:
    pass


class JobCancelled(Exception):
    pass


# ---- cancel / time-out after a scored try: the best is kept; before any: the job is cancelled
run, check_one, copy, seeds, runs, copies = fake("video", [6, 4, 5])
def cancel_on_3(p, real=run):
    if len(runs) == 2:
        raise JobCancelled("cancelled")
    return real(p)
out = rc.retry("video", {"prompt": "p", "realism_retries": 25}, cancel_on_3, check_one, copy, progress.append, seeds)
assert out["realism_retry"]["stopped"] == "cancelled" and out["realism"]["score"] == 4
run, check_one, copy, seeds, runs, copies = fake("video", [6, 4])
def check_cancel_on_2(p, res, real=check_one):
    if len(runs) == 2:
        raise JobCancelled("timed out after 360 min")
    return real(p, res)
out = rc.retry("video", {"prompt": "p", "realism_retries": 25}, run, check_cancel_on_2, copy, progress.append, seeds)
assert out["realism_retry"]["stopped"] == "cancelled" and out["realism_retry"]["best_try"] == 1
try:
    rc.retry("video", {"prompt": "p", "realism_retries": 25}, lambda p: (_ for _ in ()).throw(JobCancelled("cancelled")),
             check_one, copy, progress.append, seeds)
    raise AssertionError("cancelling the first try cancels the job")
except JobCancelled:
    pass

# ---- every progress line of a try carries "try n/N · best x/10 · "; cleared at the end
labels = []
run, check_one, copy, seeds, runs, copies = fake("photos", [4, 2, 0])
rc.retry("photos", params, run, check_one, copy, progress.append, seeds, label=labels.append)
assert labels == ["try 1/25 · ", "try 2/25 · best 4/10 · ", "try 3/25 · best 2/10 · ", ""], labels

# ---- API: which jobs may retry
for body in [{"type": "video", "prompt": "a boat", "realism_check": True, "realism_retries": 26},
             {"type": "video", "prompt": "a boat", "realism_check": True, "realism_retries": 0.5},
             {"type": "video", "prompt": "a boat", "realism_retries": 5},
             {"type": "video", "prompt": "a boat", "realism_check": True, "realism_retries": 5, "seconds": 10},
             {"type": "video", "prompt": "a boat", "realism_check": True, "realism_retries": 5, "variations": 2},
             {"type": "video", "prompt": "a boat", "realism_check": True, "realism_retries": 5, "plan_only": True},
             {"type": "story", "story": "x", "realism_check": True, "realism_retries": 5},
             {"type": "write", "premise": "x", "realism_retries": 5}]:
    try:
        h._realism_retries(dict(body))
        raise AssertionError(body)
    except ValueError:
        pass
b = {"type": "video", "prompt": "a boat", "realism_check": True, "realism_retries": 25}
assert h._realism_retries(b) == 25 and b["seconds"] == 5, "a video with retries is a 5 s clip"
assert h._realism_retries({"type": "photos"}) is None
queued = []
h._queue_job = lambda client, job_type, params, **kw: queued.append(params) or h._resp(202, {})
h._resolve_avatars = lambda client, requested: [{"name": "dana", "reference_key": "k", "reference_keys": ["k"]}]
for extra, ok in [({"scenes": [{"prompt": "x"}], "count": 1}, True), ({"scenes": [{"prompt": "x"}, {"prompt": "y"}]}, False),
                  ({"scenes": [{"prompt": "x"}], "count": 3}, False),
                  ({"series": {"where": "a cafe"}, "count": 2}, False)]:
    r = h.create_job("m", dict({"type": "photos", "avatar": "dana", "realism_check": True, "realism_retries": 25}, **extra))
    assert (r["statusCode"] == 202) is ok, (extra, r)
assert queued[-1]["realism_retries"] == 25 and queued[-1]["count"] == 1

print("test_realism_retry: ok")
