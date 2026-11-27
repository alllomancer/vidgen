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

# ---- the question: the prompt, the settings, every check; the suggestion goes in the prompt, no vidgen settings
q = rc.instruction("video", "a woman walks in the rain", {"look": "pro"})
assert "a woman walks in the rain" in q and '"look": "pro"' in q and '"prompt"' in q and '"checks"' in q
assert all(f'"{k}": 0|1|2' in q for k in rc.CHECKS) and '"options"' not in q and '"score"' not in q
assert "camera or phone" in q and "photorealistic" in q  # concrete camera words in; buzzwords named as banned
assert '"prompt"' not in rc.instruction("story", None, {})

# ---- the score comes from the checks: 3 per clear giveaway, 1 per slight one, up to 10
assert rc.score_from_checks({}) == 0 and rc.score_from_checks({"skin": 1}) == 1 and rc.score_from_checks({"hands": 2}) == 3
assert rc.score_from_checks({"skin": 2, "hands": 1, "polish": 2}) == 7
assert rc.score_from_checks({k: 2 for k in rc.CHECKS}) == 10
assert rc.score_from_checks({"skin": "2", "hands": True, "face": 7, "text": -1, "light": float("nan")}) == 6

# ---- reading answers
a = rc.clean('Sure! {"checks": {"skin": 2, "light": 1, "colour": 0}, "signs": ["waxy skin", "  too-perfect\\nlight ", '
             '"", 5, "' + "x" * 200 + '", "a", "b", "c"], "prompt": "A hyperrealistic 8k photo of a woman in the rain, '
             'masterpiece, shot on a phone at dusk, wet hair, street light from the left", "options": {"look": "phone"}}',
             "photo")
assert a["score"] == 4 and a["checks"]["skin"] == 2 and a["checks"]["hands"] == 0, a
assert a["signs"][:2] == ["waxy skin", "too-perfect light"] and len(a["signs"]) == 5 and len(a["signs"][2]) == 80
assert a["prompt"] == "A photo of a woman in the rain, shot on a phone at dusk, wet hair, street light from the left", a
assert "options" not in a, "no vidgen settings are suggested"
assert rc.clean('{"score": 3}', "video")["score"] == 3, "an answer with only a score still reads"
assert rc.clean('{"score": 12}', "video")["score"] == 10 and rc.clean('{"score": -4}', "video")["score"] == 0
assert "prompt" not in rc.clean('{"checks": {"skin": 1}, "prompt": "x"}', "story")
assert "prompt" not in rc.clean('{"checks": {"skin": 1}, "prompt": "8k, masterpiece"}', "photo"), "only buzzwords: none"
for bad in ["no json here", '{"signs": []}', '{"score": true}', '["score", 3]', '{"score": "high"}',
            '{"score": Infinity}', '{"score": NaN}', '{"score": 3,}', '{"checks": {"nope": 2}}']:
    try:
        rc.clean(bad, "video")
        raise AssertionError(bad)
    except rc.BadAnswer:
        pass

# ---- photos: worst score, most frequent signs
agg = rc.aggregate_photos([{"score": 3, "signs": ["Waxy skin"]}, {"score": 8, "signs": ["waxy skin", "warped fingers"]},
                           {"score": 5, "signs": ["warped fingers", "waxy skin"]}])
assert agg == {"score": 8, "signs": ["Waxy skin", "warped fingers"]}, agg

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
    return json.dumps({"checks": {"skin": 2, "hands": n}, "signs": [f"sign {n}"], "prompt": f"better scene {n}"})


result = {"photo_1_key": "outputs/m/j/photo_1.png", "photo_2_key": "outputs/m/j/photo_2.png",
          "scenes": [{"photo": 1, "text": "in a cafe"}, {"photo": 2, "text": "on a beach"}]}
out = rc.check("photos", {"look": "pro", "natural_skin": False}, result, tmp, download, judge)
assert [len(i) for i, _ in asked] == [1, 1] and "in a cafe" in asked[0][1] and "on a beach" in asked[1][1]
assert out["score"] == 6 and out["scenes"] == ["better scene 1", "better scene 2"], out
assert out["photos"] == [{"photo": 1, "score": 4, "signs": ["sign 1"]}, {"photo": 2, "score": 6, "signs": ["sign 2"]}]
assert "options" not in out and out["checked"] == "photos" and out["note"] == rc.NOTE

# ---- check(): a video, frames taken at the spec's moments (real ffmpeg when available)
if shutil.which("ffmpeg") and shutil.which("ffprobe"):
    clip = os.path.join(tmp, "src.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=10:duration=2", "-pix_fmt", "yuv420p", clip], check=True)
    store["outputs/m/j/video.mp4"] = open(clip, "rb").read()
    asked.clear()
    out = rc.check("video", {"prompt": "a boat", "look": "pro", "shot": "stream"}, {"video_key": "outputs/m/j/video.mp4"},
                   tempfile.mkdtemp(), download, lambda images, text: asked.append((images, text)) or
                   '{"checks": {"background": 2, "polish": 2}, "signs": ["too clean"], "prompt": "a boat, shaky phone video"}')
    assert len(asked[0][0]) == 4 and all(os.path.getsize(p) > 0 for p in asked[0][0])
    assert '"shot": "stream"' in asked[0][1] and "a boat" in asked[0][1]
    assert out["score"] == 6 and out["signs"] == ["too clean"] and out["prompt"] == "a boat, shaky phone video", out
    assert out["checks"]["background"] == 2 and out["checked"] == "video" and out["note"] == rc.NOTE and "options" not in out
    # variations: each take judged, the worst counts
    store["outputs/m/j/video_2.mp4"] = store["outputs/m/j/video.mp4"]
    scores = iter([3, 8])
    asked.clear()
    out = rc.check("video", {"prompt": "a boat"}, {"video_1_key": "outputs/m/j/video.mp4", "video_2_key": "outputs/m/j/video_2.mp4"},
                   tempfile.mkdtemp(), download, lambda images, text: asked.append(images) or
                   json.dumps({"score": next(scores), "signs": ["s"], "prompt": "p"}))
    assert len(asked) == 2 and out["score"] == 8 and out["takes"] == [{"take": 1, "score": 3, "signs": ["s"]},
                                                                      {"take": 2, "score": 8, "signs": ["s"]}], out
    # a 2 s video with a longer audio track: frames still come from the picture
    long_audio = os.path.join(tmp, "audio.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=2",
                    "-f", "lavfi", "-i", "sine=duration=6", "-pix_fmt", "yuv420p", long_audio], check=True)
    frames = rc.extract_frames(long_audio, 4, tempfile.mkdtemp(), "f")
    assert len(frames) == 4 and all(os.path.getsize(p) > 0 for p in frames)
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

# ---- the render box may read back what it wrote (the check downloads outputs/...): the fake S3 above can't catch
# a missing permission, so check the worker's IAM policy itself
import re  # noqa: E402

iam = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "terraform", "iam.tf")).read()
worker = iam[iam.index('data "aws_iam_policy_document" "worker"'):]
worker = worker[:worker.index("\ndata \"aws_iam_policy_document\"", 1) if "\ndata \"aws_iam_policy_document\"" in worker[1:] else len(worker)]
reads_outputs = [b for b in re.findall(r"statement \{(.*?)\n  \}", worker, re.S)
                 if "/outputs/*" in b and '"s3:GetObject"' in b]
assert reads_outputs, "the worker role must be allowed s3:GetObject on outputs/* (the AI-look check reads them back)"

print("test_realism_check: ok")
