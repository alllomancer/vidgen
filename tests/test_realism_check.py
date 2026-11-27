"""AI-look check logic (worker/realism_check.py): frames, the question, reading the answer, photo verdicts and
failures, with the VLM and S3 faked. docs/specs/2026-11-27-realism-check.md"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("worker")
import realism_check as rc  # noqa: E402

# ---- frame times
assert rc.frame_times(8.0, 4) == [1.0, 3.0, 5.0, 7.0]  # 12.5 / 37.5 / 62.5 / 87.5 %
assert rc.frame_times(12.0, 6) == [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]

# ---- the question names the prompt, the settings and only allowed values
q = rc.instruction("video", "a woman walks in the rain", {"look": "pro"})
assert "a woman walks in the rain" in q and '"look": "pro"' in q and "stream, pov" in q and '"prompt"' in q
assert "shot" not in rc.instruction("photo", "x", {}) and '"prompt"' not in rc.instruction("story", None, {})

# ---- reading answers
a = rc.clean('Sure! {"score": 7.6, "signs": ["waxy skin", "  too-perfect\\nlight ", "", 5, "' + "x" * 200 + '",'
             '"a", "b", "c"], "prompt": "candid phone photo of a woman in the rain", '
             '"options": {"look": "phone", "natural_skin": true, "grain": "yes", "shot": "pov", "evil": 1}}', "photo")
assert a["score"] == 8 and a["signs"][:2] == ["waxy skin", "too-perfect light"] and len(a["signs"]) == 5, a
assert len(a["signs"][2]) == 80 and a["prompt"].startswith("candid")
assert a["options"] == {"look": "phone", "natural_skin": True}, "shot is for videos; bad values dropped"
assert rc.clean('{"score": 3, "options": {"shot": "pov", "look": "neon"}}', "video")["options"] == {"shot": "pov"}
assert rc.clean('{"score": 12}', "video")["score"] == 10 and rc.clean('{"score": -4}', "video")["score"] == 0
assert "prompt" not in rc.clean('{"score": 2, "prompt": "x"}', "story")
for bad in ["no json here", '{"signs": []}', '{"score": true}', '["score", 3]', '{"score": "high"}']:
    try:
        rc.clean(bad, "video")
        raise AssertionError(bad)
    except ValueError:
        pass

# ---- photos: worst score, most frequent signs, options any photo asked for
agg = rc.aggregate_photos([
    {"score": 3, "signs": ["Waxy skin"], "options": {"look": "candid", "grain": False}},
    {"score": 8, "signs": ["waxy skin", "warped fingers"], "options": {"look": "phone", "grain": True}},
    {"score": 5, "signs": ["warped fingers", "waxy skin"], "options": {"look": "phone"}}])
assert agg == {"score": 8, "signs": ["Waxy skin", "warped fingers"], "options": {"look": "phone", "grain": True}}, agg

# ---- check(): photos job end to end with fakes
tmp = tempfile.mkdtemp()
store = {"outputs/m/j/photo_1.png": b"png1", "outputs/m/j/photo_2.png": b"png2", "outputs/m/j/video.mp4": None}


def download(key, path):
    if store.get(key) is None:
        raise RuntimeError(f"missing {key}")
    open(path, "wb").write(store[key])


asked = []


def judge(images, text):
    asked.append((list(images), text))
    n = len(asked)
    return json.dumps({"score": 2 + n * 3, "signs": [f"sign {n}"], "prompt": f"better scene {n}",
                       "options": {"natural_skin": n == 2}})


result = {"photo_1_key": "outputs/m/j/photo_1.png", "photo_2_key": "outputs/m/j/photo_2.png",
          "scenes": [{"photo": 1, "text": "in a cafe"}, {"photo": 2, "text": "on a beach"}]}
out = rc.check("photos", {"look": "pro", "natural_skin": False}, result, tmp, download, judge)
assert [len(i) for i, _ in asked] == [1, 1] and "in a cafe" in asked[0][1] and "on a beach" in asked[1][1]
assert out["score"] == 8 and out["scenes"] == ["better scene 1", "better scene 2"], out
assert out["photos"] == [{"photo": 1, "score": 5, "signs": ["sign 1"]}, {"photo": 2, "score": 8, "signs": ["sign 2"]}]
assert out["options"] == {"natural_skin": True} and out["checked"] == "photos" and out["note"] == rc.NOTE

# ---- check(): a video, frames taken at the spec's moments (real ffmpeg when available)
if shutil.which("ffmpeg") and shutil.which("ffprobe"):
    clip = os.path.join(tmp, "src.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=10:duration=2", "-pix_fmt", "yuv420p", clip], check=True)
    store["outputs/m/j/video.mp4"] = open(clip, "rb").read()
    asked.clear()
    out = rc.check("video", {"prompt": "a boat", "look": "pro", "shot": "stream"}, {"video_key": "outputs/m/j/video.mp4"},
                   tempfile.mkdtemp(), download, lambda images, text: asked.append((images, text)) or
                   '{"score": 6, "signs": ["too clean"], "prompt": "a boat, shaky phone video", "options": {"shot": "pov"}}')
    assert len(asked[0][0]) == 4 and all(os.path.getsize(p) > 0 for p in asked[0][0])
    assert '"shot": "stream"' in asked[0][1] and "a boat" in asked[0][1]
    assert out == {"score": 6, "signs": ["too clean"], "prompt": "a boat, shaky phone video", "options": {"shot": "pov"},
                   "checked": "video", "note": rc.NOTE}, out
    out = rc.check("story", {"look": "film"}, {"video_key": "outputs/m/j/video.mp4"}, tempfile.mkdtemp(), download,
                   lambda images, text: asked.append((images, text)) or '{"score": 1, "prompt": "x"}')
    assert len(asked[-1][0]) == 6 and "prompt" not in out and out["score"] == 1

# ---- failures: never raised, the job keeps its result
assert rc.check("photos", {}, {}, tmp, download, judge)["error"].startswith("the check failed")
out = rc.check("photos", {}, result, tmp, download, lambda images, text: "I can't tell")
assert out == {"error": "the check couldn't read the model's answer", "checked": "photos", "note": rc.NOTE}, out
out = rc.check("video", {}, {"video_key": "outputs/m/j/nope.mp4"}, tmp, download, judge)
assert out["error"].startswith("the check failed"), out


class JobCancelled(Exception):
    pass


try:
    rc.check("photos", {}, result, tmp, download, lambda images, text: (_ for _ in ()).throw(JobCancelled("cancelled")))
    raise AssertionError("cancelling must stop the job")
except JobCancelled:
    pass

print("test_realism_check: ok")
