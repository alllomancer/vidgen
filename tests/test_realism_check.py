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

clean_screen = lambda paths: False  # noqa: E731 - the content filter finds nothing


def as_critic(judge):
    """A critic (steps -> answers) from a one-call fake judge(images, text) whose JSON may carry a "prompt": the
    judge step gets the fake's answer, the rewrite step the "prompt" from it, a compare step "same"."""
    def critic(steps):
        answers = []
        for st in steps:
            if st["text"].startswith("You are reviewing"):
                answers.append(judge(st["images"], st["text"]))
            elif st["text"].startswith("You are improving"):
                try:
                    answers.append(json.dumps({"prompt": rc._json_object(answers[0]).get("prompt") or ""}))
                except rc.BadAnswer:
                    answers.append("{}")
            else:
                answers.append('{"more_real": "same"}')
        return answers
    return critic


def chk(job_type, params, result, workdir, download, judge, **kw):
    out = rc.check(job_type, params, result, workdir, download, as_critic(judge), **kw)
    out.pop("_images", None)
    return out

# ---- frame times
assert rc.frame_times(8.0, 4) == [1.0, 3.0, 5.0, 7.0]  # 12.5 / 37.5 / 62.5 / 87.5 %
assert rc.frame_times(12.0, 6) == [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]

# ---- the steps: judge (checks, no rewrite), rewrite (prompt only, with the judge's answer and the history), compare
q = rc.judge_text("video", "a woman walks in the rain", {"look": "pro"})
assert "a woman walks in the rain" in q and '"look": "pro"' in q and '"checks"' in q and '"prompt"' not in q
assert all(k in q for k in rc.CHECKS) and "0|1|2" not in q, "no template the model could copy as-is"
assert '"options"' not in q and '"score"' not in q
w = rc.rewrite_text("photo", "a woman in a cafe", 0, [
    {"try": 1, "score": 7, "signs": ["waxy skin"], "prompt": "a woman in a cafe"},
    {"try": 2, "score": 4, "signs": ["too clean"], "prompt": "a candid phone photo of a woman in a cafe", "best": True}])
assert "{step0}" in w and '"prompt"' in w and "camera or phone" in w and "photorealistic" in w
assert "try 1: 7/10; giveaways: waxy skin" in w and "try 2: 4/10 (best so far)" in w and "candid phone photo" in w
assert "Earlier tries" not in rc.rewrite_text("photo", "x", 0, ())
c = rc.compare_text("video", 4)
assert "first 4 images (A)" in c and "last 4 images (B)" in c and '"more_real"' in c
assert '"""' not in q + w + c, "the prompt is quoted between lines, never in triple quotes"

# ---- one critic call per attempt: judge, sampled rewrite with the history, compare with the best
steps_seen = []
def critic(steps):
    steps_seen.append(steps)
    return ['{"checks": {"skin": 2, "polish": 1}, "signs": ["waxy skin"]}',
            '{"prompt": "A candid 8k phone photo of a woman in a cafe, window light"}',
            '{"more_real": "B", "why": "skin texture"}'][:len(steps)]
v = rc._assess("photo", "a woman in a cafe", {"look": "pro"}, ["new.png"], critic, ["best.png"],
               [{"try": 1, "score": 5, "signs": ["x"], "prompt": "p"}])
st = steps_seen[-1]
assert [x["images"] for x in st] == [["new.png"], ["new.png"], ["new.png", "best.png"]], st  # 1 earlier try: new first
assert st[0].get("sample") is not True and st[1]["sample"] is True and "{step0}" in st[1]["text"] and "try 1: 5/10" in st[1]["text"]
assert v == {"score": 4, "checks": dict({k: 0 for k in rc.CHECKS}, skin=2, polish=1), "signs": ["waxy skin"],
             "prompt": "A candid phone photo of a woman in a cafe, window light", "compared": "A"}, v  # "B" = the best, shown second
# with an even number of earlier tries the best comes first and "B" means the new attempt
v = rc._assess("photo", "x", {}, ["new.png"], critic, ["best.png"], [{"try": 1}, {"try": 2}])
assert steps_seen[-1][2]["images"] == ["best.png", "new.png"] and v["compared"] == "B"
assert "first image (A) and the second image (B)" in steps_seen[-1][2]["text"]
# user or model text can't pose as a placeholder
assert "{step0}" not in rc.judge_text("photo", "a {step0} trick", {}) and "{ step0}" in rc.judge_text("photo", "a {step0} trick", {})
assert rc.compare_text("video", 4, 3).startswith("The first 4 images (A) and the last 3 images (B)")
v = rc._assess("story", None, {}, ["f.png"], critic, None, ())
assert len(steps_seen[-1]) == 1 and "prompt" not in v, "a story: judged only"
assert rc._compared('{"more_real": "a"}') == "A" and rc._compared('{"more_real": "same"}') == "SAME"
assert rc._compared("dunno") is None and rc._rewritten("dunno") is None

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
# answers a small model really gives: fenced, with text around, a second object, cut off by the token limit
fenced = 'Here is my review:\n```json\n{"checks": {"skin": 2, "hands": 1}, "signs": ["waxy skin"]}\n```\nHope it helps {:)}'
assert rc.clean(fenced, "photo")["score"] == 4
cut = '{"checks": {"skin": 2, "polish": 2}, "signs": ["waxy skin", "too smooth"], "prompt": "A candid phone photo of'
got = rc.clean(cut, "photo")
assert got["score"] == 6 and got["prompt"] == "A candid phone photo of", got
assert rc.clean('{"checks": {"skin": 1, "hands": 2', "photo")["score"] == 4, "cut inside checks"
assert rc.clean('{"checks": {"skin": 1}, "signs": ["a", "b"', "photo")["signs"] == ["a", "b"]
try:
    rc.clean("The image looks quite real, I would say mostly fine.", "photo")
    raise AssertionError("prose only")
except rc.BadAnswer as e:
    assert e.answer.startswith("The image looks quite real")
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
out = chk("photos", {"look": "pro", "natural_skin": False}, result, tmp, download, judge, screen=clean_screen)
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
    out = chk("video", {"prompt": "a boat", "look": "pro", "shot": "stream"}, {"video_key": "outputs/m/j/video.mp4"},
                   tempfile.mkdtemp(), download, lambda images, text: asked.append((images, text)) or
                   '{"checks": {"background": 2, "polish": 2}, "signs": ["too clean"], "prompt": "a boat, shaky phone video"}', screen=clean_screen)
    assert len(asked[0][0]) == 4 and all(os.path.getsize(p) > 0 for p in asked[0][0])
    assert '"shot": "stream"' in asked[0][1] and "a boat" in asked[0][1]
    assert out["score"] == 6 and out["signs"] == ["too clean"] and out["prompt"] == "a boat, shaky phone video", out
    assert out["checks"]["background"] == 2 and out["checked"] == "video" and out["note"] == rc.NOTE and "options" not in out
    # variations: each take judged, the worst counts
    store["outputs/m/j/video_2.mp4"] = store["outputs/m/j/video.mp4"]
    scores = iter([3, 8])
    asked.clear()
    out = chk("video", {"prompt": "a boat"}, {"video_1_key": "outputs/m/j/video.mp4", "video_2_key": "outputs/m/j/video_2.mp4"},
                   tempfile.mkdtemp(), download, lambda images, text: asked.append(images) or
                   json.dumps({"score": next(scores), "signs": ["s"], "prompt": "p"}), screen=clean_screen)
    seen_v = []
    chk("video", {"prompt": "a boat"}, {"video_1_key": "outputs/m/j/video.mp4", "video_2_key": "outputs/m/j/video_2.mp4"},
             tempfile.mkdtemp(), download, lambda images, text: '{"score": 1}', screen=lambda p: seen_v.append(len(p)) or False)
    assert seen_v == [8], "4 frames of each of the 2 takes, screened together before judging"
    assert len(asked) == 2 and out["score"] == 8 and out["takes"] == [{"take": 1, "score": 3, "signs": ["s"]},
                                                                      {"take": 2, "score": 8, "signs": ["s"]}], out
    # a 2 s video with a longer audio track: frames still come from the picture
    long_audio = os.path.join(tmp, "audio.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=2",
                    "-f", "lavfi", "-i", "sine=duration=6", "-pix_fmt", "yuv420p", long_audio], check=True)
    frames = rc.extract_frames(long_audio, 4, tempfile.mkdtemp(), "f")
    assert len(frames) == 4 and all(os.path.getsize(p) > 0 for p in frames)
    out = chk("story", {"look": "film"}, {"video_key": "outputs/m/j/video.mp4"}, tempfile.mkdtemp(), download,
                   lambda images, text: asked.append((images, text)) or '{"score": 1, "prompt": "x"}', screen=clean_screen)
    assert len(asked[-1][0]) == 6 and "prompt" not in out and out["score"] == 1

# ---- failures: never raised, the job keeps its result
assert chk("photos", {}, {}, tmp, download, judge, screen=clean_screen)["error"].startswith("the check failed")
out = chk("photos", {}, result, tmp, download, lambda images, text: "I can't tell,\n  sorry", screen=clean_screen)
assert out == {"error": "the check couldn't read the model's answer", "checked": "photos", "note": rc.NOTE,
               "answer": "I can't tell, sorry"}, out
assert len(chk("photos", {}, result, tmp, download, lambda images, text: "x" * 5000, screen=clean_screen)["answer"]) == 400
out = chk("video", {}, {"video_key": "outputs/m/j/nope.mp4"}, tmp, download, judge, screen=clean_screen)
assert out["error"].startswith("the check failed"), out


class JobCancelled(Exception):
    pass


try:
    chk("photos", {}, result, tmp, download, lambda images, text: (_ for _ in ()).throw(JobCancelled("cancelled")),
             screen=clean_screen)
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

# ---- the prompt is quoted between lines, never in triple quotes (the model copied them into its JSON)
assert "---\na woman in a cafe\n---" in rc.judge_text("photo", "a woman in a cafe", {})
tq = '```json { "checks": {"skin": 2, "hands": 1}, "signs": ["smooth skin"], "prompt": """A woman reads on a bed,\nwindow light from the left"""} ```'
got = rc.clean(tq, "photo")
assert got["score"] == 4 and got["prompt"].startswith("A woman reads on a bed") and "window light" in got["prompt"], got
cut_tq = '{"checks": {"skin": 1}, "signs": ["x"], "prompt": """Ja is lying on the bed holding a small book'
assert rc.clean(cut_tq, "photo")["prompt"] == "Ja is lying on the bed holding a small book"

# ---- sexual content is never critiqued: screened before the model sees anything
asked.clear()
for params, res, screen in [({"prompt": "she is naked on the bed"}, result, clean_screen),
                            ({}, dict(result, scenes=[{"photo": 1, "text": "topless on a beach"}]), clean_screen),
                            ({"prompt": "a cafe"}, result, lambda paths: True)]:
    out = chk("photos", params, res, tempfile.mkdtemp(), download, judge, screen=screen)
    assert out == {"skipped": rc.SKIPPED, "checked": "photos", "note": rc.NOTE}, out
assert not asked, "the model was never asked"
# the filter itself failing: skipped too (fail closed)
def broken(paths):
    raise RuntimeError("filter missing")
out = chk("photos", {}, result, tempfile.mkdtemp(), download, judge, screen=broken)
assert out["skipped"].startswith("skipped: the content filter couldn't run") and not asked, out
# the screen sees every picture: all photos, and every frame of every take
seen = []
chk("photos", {}, result, tempfile.mkdtemp(), download, judge, screen=lambda paths: seen.append(len(paths)) or False)
assert seen == [2], seen
# a rewrite that turns out sexual is dropped: skipped
out = chk("photos", {}, result, tempfile.mkdtemp(), download,
               lambda images, text: '{"checks": {"skin": 1}, "signs": ["x"], "prompt": "make him nude"}', screen=clean_screen)
assert out["skipped"] == rc.SKIPPED, out
# no filter given: a programming error, not a silent unscreened check
try:
    chk("photos", {}, result, tmp, download, judge)
    raise AssertionError("screen is required")
except ValueError:
    pass

print("test_realism_check: ok")
