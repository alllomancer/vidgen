"""Worker unit tests (no GPU: model calls are stubbed). Run: make test."""
import os
import sys
import tempfile
import types
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("worker")
# A list-based numpy stand-in, enough for the frame bookkeeping in generate_video.
sys.modules["numpy"] = types.SimpleNamespace(concatenate=lambda parts: [f for p in parts for f in p],
                                             asarray=lambda x: x, stack=list, uint8="uint8")
import models  # noqa: E402
import photos  # noqa: E402
import worker  # noqa: E402

worker.WORK_ROOT = tempfile.mkdtemp()  # long takes' saved segments/timelines, never /opt in tests


class Img:
    def __init__(self, n):
        self.n = n

    def convert(self, mode):
        return self


# ---------------------------------------------------------------- photos and keyframes: references
calls = []


class EditPipe:
    def __call__(self, **kw):
        calls.append(kw)
        return mock.MagicMock(images=[Img("out")])


real_swap = models._swap_to
models._swap_to = lambda name, loader: EditPipe()
models.fit = lambda im, w, h: im
refs = [Img(i) for i in range(5)]
models.make_photo(refs, "p", "n", 896, 1152, 1)
assert [i.n for i in calls[-1]["image"]] == [0, 1, 2], "photos use the first 3 references"
models.make_keyframe([("a", refs, "")], "pier")
assert [i.n for i in calls[-1]["image"]] == [0, 1, 2] and calls[-1]["width"] == 1280
models.make_keyframe([("a", [Img(10), Img(11)], ""), ("b", [Img(20)], ""), ("c", [Img(30)], "")], "pier")
assert [i.n for i in calls[-1]["image"]] == [10, 20, 30], "2-3 avatars: image 1 each"
shot = photos.build_scenes([{"shot": {"where": "on a rooftop", "wearing": "a suit", "camera": "low angle"}}], 1)[0]
assert shot["scene"] == "in a natural relaxed pose, on a rooftop, wearing a suit. Camera: low angle", shot["scene"]
prompt = photos.photo_prompt(shot["scene"], 3, shot["keep_outfit"])
assert prompt.startswith("Images 1-3") and "replacing the outfit" in prompt
assert "body shape and outfit." in photos.photo_prompt("x")

# best quality: Lightning off for the call (40 steps, true CFG), back on afterwards
lora_log = []
EditPipe.disable_lora = lambda self: lora_log.append("off")
EditPipe.enable_lora = lambda self: lora_log.append("on")
models.EDIT_LIGHTNING, models.EDIT_MODEL = 8, "Qwen/Qwen-Image-Edit-2511"
models.make_photo(refs, "p", "n", 896, 1152, 1)
assert calls[-1]["num_inference_steps"] == 8 and calls[-1]["true_cfg_scale"] == 1.0 and not lora_log
models.make_photo(refs, "p", "n", 896, 1152, 1, best=True)
assert calls[-1]["num_inference_steps"] == 40 and calls[-1]["true_cfg_scale"] == 4.0 and lora_log == ["off", "on"]
models.make_keyframe([("a", refs, "")], "pier", best=True)
assert calls[-1]["num_inference_steps"] == 40 and lora_log == ["off", "on", "off", "on"]

# ---------------------------------------------------------------- run_photos: series anchored on photo 1
class Out(Img):
    def save(self, path):
        open(path, "w").write("png")


shots = []
worker.progress = lambda *a: None
worker.upload = lambda path, client, job_id, name, ctype: f"outputs/{name}"
worker.load_references = lambda keys, workdir, prefix: [Img(i) for i in range(len(keys))]
real_make_photo = models.make_photo
models.make_photo = lambda refs, prompt, neg, w, h, seed, best=False: shots.append((refs, prompt, best)) or Out(f"p{len(shots)}")
grained = []
photos.add_grain = lambda im, seed: grained.append(seed) or im
pp = {"reference_keys": ["a", "b", "c"], "aspect": "portrait", "seed": 5, "count": 4, "scenes": [],
      "series": {"where": "a Lisbon café", "doing": ["reading"], "anchor": True},
      "look": "phone", "natural_skin": True, "candid": False, "grain": True, "quality": "best"}
res = worker.run_photos({"client": "m", "job_id": "j"}, pp, tempfile.mkdtemp())
plan = photos.build_series(pp["series"], 4, 5)
assert [len(r) for r, _, _ in shots] == [3, 3, 3, 3]
assert [i.n for i in shots[0][0]] == [0, 1, 2], "photo 1: 3 avatar images"
for (r, prompt, _), sc in zip(shots[1:], plan[1:]):
    if sc["anchor"]:
        assert [getattr(i, "n", None) for i in r] == [0, 1, "p1"] and "Image 3 is an earlier photo" in prompt
    else:
        assert [i.n for i in r] == [0, 1, 2] and "earlier photo" not in prompt, "far shots: no anchor image"
assert "earlier photo" not in shots[0][1]
assert all("in a Lisbon café" in p for _, p, _ in shots) and shots[0][1].count("reading, full body") == 1
assert all(b for _, _, b in shots) and "smartphone" in shots[0][1] and photos.NATURAL_SKIN in shots[0][1]
assert grained == [6, 7, 8, 9] and res["look"] == "phone" and res["quality"] == "best" and len(res["scenes"]) == 4
shots.clear()
worker.run_photos({"client": "m", "job_id": "j"}, dict(pp, series=dict(pp["series"], anchor=False), grain=False),
                  tempfile.mkdtemp())
assert [len(r) for r, _, _ in shots] == [3, 3, 3, 3] and not any("earlier photo" in p for _, p, _ in shots)
shots.clear()
worker.run_photos({"client": "m", "job_id": "j"}, {"reference_keys": ["a"], "aspect": "square", "seed": 1, "count": 1,
                                                   "scenes": [{"prompt": "on a train"}]}, tempfile.mkdtemp())
assert shots[0][1] == photos.photo_prompt("on a train") and shots[0][2] is False, "no options: unchanged prompt"
models.make_photo = real_make_photo

# ---------------------------------------------------------------- user LoRAs next to Lightning
class Pipe:
    def __init__(self):
        self.log = []

    def load_lora_weights(self, path, adapter_name, load_into_transformer_2=False):
        self.log.append(("load", path, adapter_name, load_into_transformer_2))

    def delete_adapters(self, names):
        self.log.append(("delete", sorted(names)))

    def set_adapters(self, names, adapter_weights):
        self.log.append(("set", names, adapter_weights))


pipe = Pipe()
J = {"adapter": "user_jerk_v1_s750", "high": "/h", "low": "/l", "weight": 0.9, "trigger": "vgn_jerk"}
models.set_loras([J])
models._apply_loras(pipe)
assert ("load", "/l", "user_jerk_v1_s750_2", True) in pipe.log
assert pipe.log[-1] == ("set", ["lightning", "lightning_2", "user_jerk_v1_s750", "user_jerk_v1_s750_2"], [1.0, 1.0, 0.9, 0.9])
pipe.log.clear()
models._apply_loras(pipe)
assert [e[0] for e in pipe.log] == ["set"], "same LoRA: no reload"
pipe.log.clear()
models.set_loras([])
models._apply_loras(pipe)
assert pipe.log[0] == ("delete", ["user_jerk_v1_s750", "user_jerk_v1_s750_2"])
models.set_loras([J])
assert models.with_triggers("dancing") == "vgn_jerk dancing" and models.with_triggers("VGN_JERK x") == "VGN_JERK x"
models.set_loras([])

worker.LORA_CACHE = tempfile.mkdtemp()
downloads = []
worker.s3 = mock.MagicMock(download_file=lambda b, k, p: downloads.append(k) or open(p, "w").write("w"))
lora_params = {"loras": [{"name": "jerk", "version": 1, "step": 750, "weight": 0.9, "trigger": "t",
                          "high_key": "loras/m/jerk/v1/checkpoints/h", "low_key": "loras/m/jerk/v1/checkpoints/l"}]}
assert worker.prepare_loras(lora_params)[0]["adapter"] == "user_jerk_v1_s750"
worker.prepare_loras(lora_params)
assert len(downloads) == 2, "downloaded once per version/step"

# ---------------------------------------------------------------- generate_video: segments, fps, upscale
segments = []


seg_prompts = []


def fake_segment(prompt, negative, num, width, height, seed, image):
    segments.append((num, seed, image))
    seg_prompts.append(prompt)
    return [f"f{len(segments)}_{i}" for i in range(num)]


written = []
models._render_segment = fake_segment
models._to_uint8_frames = lambda f: f
models.Image = mock.MagicMock(fromarray=lambda f: f"img:{f}")
models.write_video = lambda frames, path, fps=16, grain=None: written.append((path, len(frames), fps, grain))
sys.modules["enhance"] = types.SimpleNamespace(
    interpolate=lambda frames, check: [x for f in frames for x in (f, f)][:-1],
    upscale=lambda frames, check: [f + "@2x" for f in frames])

matched = []
models.colour_stats = lambda frames: ("stats", frames[0])
models.match_colours = lambda frames, ref: matched.append((frames[0], ref)) or [f + "~" for f in frames]
models.generate_video("/o.mp4", "lighthouse", seconds=12, seed=7)
assert [s[0] for s in segments] == [81, 81, 33] and [s[1] for s in segments] == [7, 8, 9]
assert segments[1][2] == "img:f1_80", "each segment starts from the last frame"
assert segments[2][2] == "img:f2_80~", "over 10 s: the colour-matched last frame conditions the next segment"
assert matched == [("f2_1", ("stats", "f1_0")), ("f3_1", ("stats", "f1_0"))], "segments 2+ matched to segment 1"
assert written[-1] == ("/o.mp4", 2 * 193 - 1, 32, None), written[-1]
matched.clear(), segments.clear()
models.generate_video("/o.mp4", "x", seconds=10, seed=7)
assert matched == [] and segments[1][2] == "img:f1_80", "10 s or less: no colour matching"
# resume: finished segments saved in the work folder; a re-delivered take renders only the rest
import json as _jsonr  # noqa: E402
models._save_segment = lambda path, frames: _jsonr.dump(list(frames), open(path, "w"))
models._load_segment = lambda path: _jsonr.load(open(path))
captured = []
real_write_video = models.write_video
models.write_video = lambda frames, path, fps=16, grain=None: captured.append(list(frames))
msgs = []


def seeded_segment(prompt, negative, num, width, height, seed, image):  # frames depend on the seed only
    segments.append((num, seed, image))
    return [f"s{seed}_{i}" for i in range(num)]


real_fake = models._render_segment
models._render_segment = seeded_segment
segments.clear()
models.generate_video("/o.mp4", "x", seconds=50, fps=16, seed=1, work_dir=tempfile.mkdtemp(), progress=msgs.append)
full = captured[-1]
assert msgs == [f"segment {i}/10" for i in range(1, 11)], msgs
assert len(segments) == 10 and len(full) == 801
wd = tempfile.mkdtemp()


def dies_after_six(*a):
    if len(segments) == 6:
        raise RuntimeError("spot interruption")
    return seeded_segment(*a)


segments.clear()
models._render_segment = dies_after_six
_stubs.expect_error(models.generate_video, "/o.mp4", "x", "", 50, 1280, 704, 1, None, 16, False, None, False, None,
                    wd, exc=RuntimeError)
assert sorted(f for f in os.listdir(wd) if f.endswith(".npz")) == [f"segment_{i:02d}.npz" for i in range(6)]
assert _jsonr.load(open(os.path.join(wd, "segments.json")))["done"] == 6
models._render_segment = seeded_segment
segments.clear(), msgs.clear()
models.generate_video("/o.mp4", "x", seconds=50, fps=16, seed=1, work_dir=wd, progress=msgs.append)
assert [s[1] for s in segments] == [7, 8, 9, 10], "only segments 7-10 rendered"
assert captured[-1] == full, "same frames as an uninterrupted take"
assert msgs[0] == "segment 7/10" and msgs[-1] == "segment 10/10", msgs
segments.clear()
models.generate_video("/o.mp4", "x", seconds=50, fps=16, seed=2, work_dir=wd)
assert len(segments) == 10, "saved segments of different settings are not reused"
segments.clear()
models._JOB_LORAS[:] = [{"adapter": "user_x_v1", "weight": 0.8}]
models.generate_video("/o.mp4", "x", seconds=50, fps=16, seed=2, work_dir=wd)
models._JOB_LORAS[:] = []
assert len(segments) == 10, "other LoRAs: not reused"
real_save = models._save_segment
models._save_segment = lambda path, frames: (_ for _ in ()).throw(OSError(28, "No space left on device"))
segments.clear()
models.generate_video("/o.mp4", "x", seconds=50, fps=16, seed=3, work_dir=tempfile.mkdtemp())
assert len(segments) == 10 and len(captured[-1]) == 801, "a full disk only loses resume"
models._save_segment = real_save
# re-anchoring: the conditioning frames of segments k, 2k, ... (1-based) go through reanchor()
anchored = []
segments.clear()
transfers = []
models.transfer_colours = lambda frame, stats: transfers.append(stats) or f"{frame}#"
models.np_frame = lambda image: image  # PIL <-> array are identity in these fakes
models.generate_video("/o.mp4", "x", seconds=50, fps=16, seed=1, anchor_every=3,
                      reanchor=lambda frame, n=0: anchored.append((frame, n)) or f"anchored:{frame}")
assert anchored == [("img:s2_80~", 2), ("img:s5_80~", 5), ("img:s8_80~", 8)], "segments 3, 6, 9 start from re-anchored frames"
assert segments[2][2] == "img:anchored:img:s2_80~#" and segments[3][2] == "img:s3_80~"
assert transfers == [("stats", "img:s2_80~"), ("stats", "img:s5_80~"), ("stats", "img:s8_80~")], \
    "the anchored frame takes the colours of the frame it replaces"
anchored.clear(), segments.clear()
models.generate_video("/o.mp4", "x", seconds=50, fps=16, seed=1)
assert anchored == [], "no anchoring without anchor_every"
models.write_video = real_write_video
models._render_segment = real_fake
segments.clear()
segments.clear()
models.generate_video("/o.mp4", "x", seconds=50, fps=16, seed=1)
assert len(segments) == 10 and written[-1][:3] == ("/o.mp4", 801, 16), (len(segments), written[-1])
assert [s[1] for s in segments] == list(range(1, 11)), "seed + n per segment"
# beats -> one prompt per 5 s segment; trigger words in every segment
BEATS3 = [{"at": 0, "prompt": "walks in"}, {"at": 10, "prompt": "sits down"}, {"at": 30, "prompt": "looks up"}]
assert models.segment_prompts("p", 10, BEATS3) == ["walks in"] * 2 + ["sits down"] * 4 + ["looks up"] * 4
assert models.segment_prompts("p", 3, None) == ["p"] * 3
segments.clear(), seg_prompts.clear()
models._JOB_LORAS[:] = [{"trigger": "zx9"}]
models.generate_video("/o.mp4", "base", seconds=50, fps=16, seed=1,
                      segment_prompts=models.segment_prompts("base", 10, BEATS3))
models._JOB_LORAS[:] = []
assert len(seg_prompts) == 10 and all("zx9" in p for p in seg_prompts), seg_prompts
assert "walks in" in seg_prompts[1] and "sits down" in seg_prompts[2] and "looks up" in seg_prompts[9]
assert "base" not in seg_prompts[0], "with segment prompts, the single prompt is not used"
segments.clear()
models.generate_video("/o.mp4", "x", seconds=5, fps=16, raw_path="/raw.mp4")
assert [s[0] for s in segments] == [81] and written[-2][:3] == ("/raw.mp4", 81, 16) and written[-1][:3] == ("/o.mp4", 81, 16)
segments.clear()
models.generate_video("/o.mp4", "x", seconds=2, fps=16, upscale=True, image="key")
assert segments[0][2] == "key" and written[-1][:3] == ("/o.mp4", 33, 16)
import realism  # noqa: E402
real_grain, realism.Grain = realism.Grain, lambda seed: ("grain", seed)
models.generate_video("/o.mp4", "x", seconds=2, fps=16, seed=4, grain=True, raw_path="/raw.mp4")
assert written[-1][3] == ("grain", 4) and written[-2][3] is None, "grain on the final clip only"
realism.Grain = real_grain

# ---------------------------------------------------------------- run_video: variations and sound
worker.progress = lambda *a: None
worker.upload = lambda path, client, job_id, name, ctype: f"outputs/{name}"
worker.load_cast = lambda params, workdir: []
real_sound_for_clip = worker.sound_for_clip
takes, sounds = [], []
models.generate_video = lambda out, **kw: takes.append(kw) or out
worker.sound_for_clip = lambda path, prompt, seconds, seed, workdir, tag="sound": sounds.append((seconds, seed)) or "/s.wav"
worker.add_audio = lambda v, a, out: out
base = {"prompt": "p", "negative_prompt": "", "width": 1280, "height": 704, "seconds": 5, "fps": 32,
        "upscale": False, "sound": True}
res = worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=10, variations=3), tempfile.mkdtemp())
assert res["seeds"] == [10, 11, 12] and [t["seed"] for t in takes] == [10, 11, 12]
assert {"video_1_key", "video_2_key", "video_3_key"} <= set(res) and [s[1] for s in sounds] == [10, 11, 12]
takes.clear()
res = worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=None, seconds=12), tempfile.mkdtemp())
assert takes[0]["seconds"] == 12 and takes[0]["fps"] == 32 and "video_key" in res
assert takes[0]["prompt"] == "p" and takes[0]["grain"] is False, "no realism options: prompt unchanged"
assert "segment_prompts" not in res and takes[0].get("segment_prompts") in (None, ["p"] * 3)
# beats: one styled prompt per segment, listed in the result
takes.clear()
res = worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=1, seconds=50, beats=BEATS3, look="film"),
                       tempfile.mkdtemp())
sp = takes[0]["segment_prompts"]
assert len(sp) == 10 and sp[0].startswith("p walks in") and sp[1].startswith("walks in") and sp[9].startswith("looks up"), sp
assert sp[1] != "walks in", "the look text is added to each segment"
assert res["segment_prompts"] == ["walks in"] * 2 + ["sits down"] * 4 + ["looks up"] * 4
# automatic timeline: the planner writes one prompt per segment for takes of more than 3 segments
plans = []
real_plan_segments = models.plan_segments


def good_plan(prompt, n, cast=(), seconds=None):
    plans.append((prompt, n, list(cast), seconds))
    return [f"step {i}" for i in range(n)]


def boom(prompt, n, cast=(), seconds=None):
    raise ValueError("model did not return JSON")


models.plan_segments = good_plan
takes.clear()
worker.WORK_ROOT = tempfile.mkdtemp()
res = worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=1, seconds=50), tempfile.mkdtemp())
assert plans == [("p", 10, [], 50)] and res["segment_prompts"] == [f"step {i}" for i in range(10)], (plans, res)
assert takes[0]["segment_prompts"] == [f"step {i}" for i in range(10)]
for bad in (lambda p, n, cast=(), seconds=None: ["a"] * (n - 1),
            lambda p, n, cast=(), seconds=None: ["a"] * (n - 1) + ["  "],
            lambda p, n, cast=(), seconds=None: ["a"] * (n - 1) + ["x" * 501],
            lambda p, n, cast=(), seconds=None: "a" * n, boom):
    models.plan_segments = bad
    takes.clear()
    worker.WORK_ROOT = tempfile.mkdtemp()
    res = worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=1, seconds=50), tempfile.mkdtemp())
    assert "segment_prompts" not in res and takes[0].get("segment_prompts") is None, "planner failure: user's prompt"
# resume: the take gets a work folder, a seed fixed by the job, and keeps its planned timeline
worker.WORK_ROOT = tempfile.mkdtemp()
plans.clear(), takes.clear()
models.plan_segments = good_plan
worker.run_video({"client": "m", "job_id": "lt1"}, dict(base, seed=None, seconds=50), tempfile.mkdtemp())
worker.run_video({"client": "m", "job_id": "lt1"}, dict(base, seed=None, seconds=50), tempfile.mkdtemp())
assert len(plans) == 1, "a re-delivered take reuses its timeline"
assert takes[0]["work_dir"] == os.path.join(worker.WORK_ROOT, "lt1") and callable(takes[0]["progress"])
assert takes[0]["seed"] == takes[1]["seed"], "no seed given: the same seed on re-delivery"
assert takes[1]["segment_prompts"] == takes[0]["segment_prompts"]
takes.clear()
worker.run_video({"client": "m", "job_id": "j5"}, dict(base, seed=1, seconds=5), tempfile.mkdtemp())
assert takes[0].get("work_dir") is None, "5 s clips: no work folder"
plans.clear()
worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=1, seconds=15), tempfile.mkdtemp())
worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=1, seconds=50, beats=BEATS3), tempfile.mkdtemp())
assert plans == [], "no planner for takes of up to 15 s or with the user's beats"
models.plan_segments = lambda p, n, cast=(), seconds=None: (_ for _ in ()).throw(models.JobCancelled("stop"))
_stubs.expect_error(worker.run_video, {"client": "m", "job_id": "j"}, dict(base, seed=1, seconds=50),
                    tempfile.mkdtemp(), exc=models.JobCancelled)
models.plan_segments = real_plan_segments
# plan_segments itself: asks the VLM for n prompts, refuses a wrong count
real_chat = models._chat
asked = []
models._chat = lambda messages, max_new_tokens=1024, **k: asked.append(messages[0]["content"][0]["text"]) or \
    '{"segments": ["a", "b", "c", "d"]}'
assert models.plan_segments("a dog runs", 4, [("mira", "a woman")]) == ["a", "b", "c", "d"]
assert "a dog runs" in asked[0] and "4" in asked[0] and "mira" in asked[0]
_stubs.expect_error(models.plan_segments, "a dog runs", 5, exc=ValueError)
models._chat = lambda messages, max_new_tokens=1024, **k: asked.append(messages[0]["content"][0]["text"]) or \
    'Sure {here}: {"segments": ["a", "b", "c", "d"]} Note: {done}'
assert models.plan_segments("a dog runs", 4, seconds=18) == ["a", "b", "c", "d"], "first JSON object, prose around it"
assert "18-second" in asked[-1] and "last segment" in asked[-1]
models._chat = real_chat
# realism: look/candid on the Wan prompt and the keyframe scene, best keyframe, grain
takes.clear()
kf = []
worker.load_cast = lambda params, workdir: [("dana", [Img(0)], "d")]
worker.describe_cast = lambda cast: "dana:"
models.make_keyframe = lambda cast, scene, w, h, seed, best=False: kf.append((scene, best)) or Out("kf")
worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=1, look="film", candid=True, grain=True,
                                                      quality="best", sound=False), tempfile.mkdtemp())
styled = realism.with_style("p", {"look": "film", "candid": True})
assert kf == [(styled, True)] and takes[0]["prompt"] == f"dana: {styled}" and takes[0]["grain"] is True
takes.clear()
worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=1, seconds=15, beats=BEATS3[:2], sound=False),
                 tempfile.mkdtemp())
assert takes[0]["segment_prompts"] == ["dana: walks in"] * 2 + ["dana: sits down"], "the avatar in every segment"
assert "reanchor" not in takes[0]
takes.clear()
reanchors = []
models.reanchor = lambda frame, refs, w, h, seed=None: reanchors.append((frame, refs, w, h, seed)) or "re"
worker.run_video({"client": "m", "job_id": "ja"}, dict(base, seed=1, seconds=50, anchor_every=3, sound=False),
                 tempfile.mkdtemp())
assert takes[0]["anchor_every"] == 3 and takes[0]["reanchor"]("frame", 4) == "re" and reanchors[-1][4] == 5
assert takes[0]["reanchor"]("frame") == "re" and reanchors[-1][4] == 1
assert reanchors[0][0] == "frame" and reanchors[0][1][0].n == 0 and reanchors[0][2:4] == (1280, 704), "the avatar's images"
worker.load_cast = lambda params, workdir: []
takes.clear()
worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=1, seconds=15, beats=BEATS3[:2], sound=False),
                 tempfile.mkdtemp())
assert takes[0]["segment_prompts"] == ["p walks in", "walks in", "sits down"], "no first image: the scene in segment 1"
worker.load_cast = lambda params, workdir: []

# Long takes: sound in overlapping 5 s chunks, joined with crossfades.
gen, cmds = [], []
models.generate_sound = lambda path, out, prompt, seconds, seed: gen.append((round(seconds, 2), seed)) or out
worker.ffmpeg = lambda *args: cmds.append(args)
worker.shutil = mock.MagicMock()
d = tempfile.mkdtemp()
real_sound_for_clip("/v.mp4", "p", 5, 3, d)
assert gen == [(5, 3)] and not cmds, "short clips: one pass"
gen.clear()
real_sound_for_clip("/v.mp4", "p", 12, 3, d)
assert gen == [(5.2, 3), (5.2, 4), (2, 5)], gen
assert sum("acrossfade" in " ".join(c) for c in cmds) == 2
# Prefetch reads every weight file fully, in parallel chunks.
pdir = tempfile.mkdtemp()
os.makedirs(os.path.join(pdir, "transformer"))
open(os.path.join(pdir, "transformer", "a.safetensors"), "wb").write(b"x" * 1000)
open(os.path.join(pdir, "b.bin"), "wb").write(b"y" * 10)
open(os.path.join(pdir, "readme.md"), "w").write("skip")
models.PREFETCH_CHUNK = 64
assert models.prefetch(pdir, threads=3) == 1010
# write: title split, story.txt, sampling settings
assert models.split_title("Title: The Keeper\n\nShe climbed.") == {"title": "The Keeper", "text": "She climbed."}
assert models.split_title("**The Keeper**\n\nShe climbed.")["title"] == "The Keeper"
assert models.split_title("One long paragraph with no title line at all.")["title"] == "Untitled"
chats = []
models._chat = lambda messages, max_new_tokens=1024, **gen: chats.append((messages, max_new_tokens, gen)) or \
    "Title: The Keeper\n\nShe climbed the stairs."
sys.modules.setdefault("torch", types.SimpleNamespace())
sys.modules["torch"].manual_seed = lambda s: None
saved = {}
worker.upload = lambda path, client, job_id, name, ctype: saved.update(text=open(path).read(), name=name) or f"outputs/{name}"
r = worker.run_write({"client": "m", "job_id": "j"}, {"premise": "a lighthouse keeper", "words": 600, "genre": "cozy", "seed": 3},
                     tempfile.mkdtemp())
assert r == {"title": "The Keeper", "words": 4, "seed": 3, "story_key": "outputs/story.txt"}, r
assert saved == {"text": "The Keeper\n\nShe climbed the stairs.\n", "name": "story.txt"}
prompt_text = chats[-1][0][0]["content"][0]["text"]
assert "about 600 words" in prompt_text and "Genre and tone: cozy" in prompt_text and chats[-1][1] == 1180
assert chats[-1][2]["do_sample"] is True and "write" in worker.RUNNERS
# ---------------------------------------------------------------- Krea 2: filter in the generator
import json as _json  # noqa: E402
import krea  # noqa: E402
import krea_gen  # noqa: E402


class FakeImage:
    def __init__(self, seed):
        self.seed = seed

    def save(self, path):
        open(path, "w").write(str(self.seed))


class FakeKreaPipe:
    def __call__(self, prompt, width, height, num_inference_steps, guidance_scale, generator):
        assert (num_inference_steps, guidance_scale) == (8, 0.0) and width % 16 == 0
        return types.SimpleNamespace(images=[FakeImage(generator.seed)])


flagged = {11, 12, 12 + krea_gen.REROLL_OFFSET}  # 11 passes on re-roll; 12 is flagged twice
sys.modules["torch"].Generator = lambda dev: types.SimpleNamespace(manual_seed=lambda s: types.SimpleNamespace(seed=s))
krea_gen.prefetch = lambda d: 0
krea_gen.load = lambda m, f, loras=(): (FakeKreaPipe(), lambda im: [{"label": "nsfw", "score": 0.9 if im.seed in flagged else 0.01},
                                                         {"label": "normal", "score": 0.1}])
kd = tempfile.mkdtemp()
req = {"model_dir": kd, "filter_dir": kd, "threshold": 0.5, "result": os.path.join(kd, "r.json"),
       "items": [{"prompt": "p", "width": 896, "height": 1152, "seed": sd, "out": os.path.join(kd, f"{sd}.png")}
                 for sd in (10, 11, 12)]}
open(os.path.join(kd, "req.json"), "w").write(_json.dumps(req))
krea_gen.main(os.path.join(kd, "req.json"))
out = _json.load(open(req["result"]))["items"]
assert [(o["filtered"], o["rerolled"]) for o in out] == [(False, False), (False, True), (True, True)], out
assert os.path.exists(os.path.join(kd, "10.png")) and os.path.exists(os.path.join(kd, "11.png"))
assert not os.path.exists(os.path.join(kd, "12.png")), "a twice-flagged image is never written"
assert out[1]["seed"] == 11 + krea_gen.REROLL_OFFSET and all("filter_seconds" in o for o in out)
# The license-required filter must be able to flag: scores are 0-1, so a threshold >= 1 turns it off.
assert 0 < krea.NSFW_THRESHOLD < 1, krea.NSFW_THRESHOLD


# Krea 2 LoRAs: loaded as adapters, weights set; a file diffusers can't map gives a named error
class LoraPipe(FakeKreaPipe):
    def __init__(self):
        self.calls = []

    def load_lora_weights(self, state_dict, adapter_name):
        assert isinstance(state_dict, dict), "a state dict read with safetensors, never a path diffusers opens"
        self.calls.append(("load", state_dict, adapter_name))

    def set_adapters(self, names, adapter_weights):
        self.calls.append(("set", names, adapter_weights))


MOODY = {"path": "/c/m/moody/v1/krea2.safetensors", "adapter": "user_moody_v1", "weight": 0.8}
krea_gen._peft_available = lambda: True


class T:  # a stand-in tensor: shape and scaling are all the conversion uses
    def __init__(self, shape, scale=1.0):
        self.shape, self.scale = shape, scale

    def __mul__(self, k):
        return T(self.shape, self.scale * k)

    def __float__(self):
        return float(self.shape[0])  # an alpha "tensor": T([16]) means alpha 16


# kohya/musubi layout (lora_unet_ flattened names, lora_down/up, alpha) -> the reference layout diffusers converts
kohya = {"lora_unet_blocks_3_attn_wq.lora_down.weight": T([32, 6144]),
         "lora_unet_blocks_3_attn_wq.lora_up.weight": T([6144, 32]),
         "lora_unet_blocks_3_attn_wq.alpha": T([16]),
         "lora_unet_txtfusion_refiner_blocks_1_mlp_gate.lora_down.weight": T([8, 2560]),
         "lora_unet_txtfusion_refiner_blocks_1_mlp_gate.lora_up.weight": T([6144, 8])}
ref = krea_gen.kohya_to_reference(kohya)
assert set(ref) == {"diffusion_model.blocks.3.attn.wq.lora_A.weight", "diffusion_model.blocks.3.attn.wq.lora_B.weight",
                    "diffusion_model.txtfusion.refiner_blocks.1.mlp.gate.lora_A.weight",
                    "diffusion_model.txtfusion.refiner_blocks.1.mlp.gate.lora_B.weight"}, set(ref)
assert ref["diffusion_model.blocks.3.attn.wq.lora_B.weight"].scale == 0.5, "alpha 16 / rank 32 folded into B"
assert ref["diffusion_model.blocks.3.attn.wq.lora_A.weight"].scale == 1.0
assert ref["diffusion_model.txtfusion.refiner_blocks.1.mlp.gate.lora_B.weight"].scale == 1.0, "no alpha: scale 1"
# musubi-tuner's networks.lora_krea2 also trains the 8 standalone Linear layers (as trained live 2026-11-24)
standalone = {"first": "first", "last_linear": "last.linear", "tmlp_0": "tmlp.0", "tmlp_2": "tmlp.2", "tproj_1": "tproj.1",
              "txtfusion_projector": "txtfusion.projector", "txtmlp_1": "txtmlp.1", "txtmlp_3": "txtmlp.3"}
musubi = {}
for flat in standalone:
    musubi.update({f"lora_unet_{flat}.lora_down.weight": T([32, 64]), f"lora_unet_{flat}.lora_up.weight": T([6144, 32]),
                   f"lora_unet_{flat}.alpha": T([32])})
ref2 = krea_gen.kohya_to_reference(musubi)
assert set(ref2) == {f"diffusion_model.{d}.lora_{x}.weight" for d in standalone.values() for x in "AB"}, sorted(ref2)
already = {"diffusion_model.blocks.0.attn.wq.lora_A.weight": T([4, 6144])}
assert krea_gen.kohya_to_reference(already) is already, "other layouts go to diffusers unchanged"
for bad in ({"lora_unet_blocks_0_attn_wq.lora_down.weight": T([4, 6144])},  # no lora_up
            {"lora_unet_mystery_0_x.lora_down.weight": T([4, 6144]), "lora_unet_mystery_0_x.lora_up.weight": T([6144, 4])}):
    try:
        krea_gen.kohya_to_reference(bad)
        raise AssertionError(f"must refuse {bad}")
    except ValueError:
        pass
# apply_loras reads the file, converts, and hands diffusers the converted dict
read = []
krea_gen._read_lora = lambda path: read.append(path) or dict(kohya)
lp = LoraPipe()
krea_gen.apply_loras(lp, [MOODY])
assert read == [MOODY["path"]] and lp.calls[0][0] == "load" and lp.calls[0][2] == "user_moody_v1"
assert set(lp.calls[0][1]) == set(ref) and lp.calls[1] == ("set", ["user_moody_v1"], [0.8]), lp.calls
lp = LoraPipe()
krea_gen.apply_loras(lp, [])
assert lp.calls == [], "no LoRAs: the pipe is untouched"


class BadLoraPipe(LoraPipe):
    def load_lora_weights(self, state_dict, adapter_name):
        raise KeyError("blocks.0.attn.zz under /opt/models/loras/m/moody")


class BadAdaptersPipe(LoraPipe):
    def set_adapters(self, names, adapter_weights):
        raise ValueError("adapter mismatch")


for bad in (BadLoraPipe(), BadAdaptersPipe()):  # load or weights: one named error, no internal detail
    try:
        krea_gen.apply_loras(bad, [MOODY])
        raise AssertionError("an unusable LoRA must fail")
    except krea_gen.LoraLoadError as e:
        assert "LoRA user_moody_v1 could not be loaded" in str(e) and "/opt/models" not in str(e), e
krea_gen._peft_available = lambda: False  # the Krea venv without peft: say so, not "unsupported layout"
try:
    krea_gen.apply_loras(LoraPipe(), [MOODY])
    raise AssertionError("missing peft must fail")
except krea_gen.LoraLoadError as e:
    assert "peft" in str(e), e
assert krea_gen.apply_loras(LoraPipe(), []) is not None, "no LoRAs: peft isn't needed"
krea_gen._peft_available = lambda: True
# With a LoRA loaded the filter still checks every image: seed 12 is flagged twice and dropped.
stub_load, seen_loras = krea_gen.load, []
krea_gen.load = lambda m, f, loras=(): seen_loras.append(list(loras)) or stub_load(m, f)
kd2 = tempfile.mkdtemp()
req2 = dict(req, result=os.path.join(kd2, "r.json"), loras=[MOODY],
            items=[{"prompt": "p", "width": 896, "height": 1152, "seed": sd, "out": os.path.join(kd2, f"{sd}.png")}
                   for sd in (10, 12)])
open(os.path.join(kd2, "req.json"), "w").write(_json.dumps(req2))
krea_gen.main(os.path.join(kd2, "req.json"))
out2 = _json.load(open(req2["result"]))["items"]
assert seen_loras == [[MOODY]] and [o["filtered"] for o in out2] == [False, True], (seen_loras, out2)
assert not os.path.exists(os.path.join(kd2, "12.png")) and all("nsfw_score" in o for o in out2)
# A load error ends the generator with the message in the result file, for krea.generate to report
krea_gen.load = lambda m, f, loras=(): (_ for _ in ()).throw(krea_gen.LoraLoadError("LoRA user_moody_v1 could not be loaded"))
try:
    krea_gen.main(os.path.join(kd2, "req.json"))
    raise AssertionError("a load error must exit non-zero")
except SystemExit as e:
    assert e.code == 2
assert "could not be loaded" in _json.load(open(req2["result"]))["error"]
# Other failures (CUDA out of memory, ...) are not reported as LoRA errors: they propagate
os.remove(req2["result"])
krea_gen.load = lambda m, f, loras=(): (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
try:
    krea_gen.main(os.path.join(kd2, "req.json"))
    raise AssertionError("an unrelated error must propagate")
except RuntimeError as e:
    assert "out of memory" in str(e) and not os.path.exists(req2["result"])
krea_gen.load = stub_load

# Score mode: the filter alone over existing images (avatar photo sets), no Krea pipeline load
loaded_krea = []
krea_gen.load = lambda *a, **k: loaded_krea.append(1) or (_ for _ in ()).throw(AssertionError("no Krea load"))
krea_gen.load_filter = lambda f: (lambda im: [{"label": "nsfw", "score": 0.9 if "bad" in im.name else 0.02}])
sys.modules["PIL"].Image.open = lambda path: types.SimpleNamespace(name=os.path.basename(path), convert=lambda m: types.SimpleNamespace(name=os.path.basename(path)))
kd3 = tempfile.mkdtemp()
sreq = {"mode": "score", "filter_dir": kd3, "result": os.path.join(kd3, "s.json"),
        "images": [os.path.join(kd3, n) for n in ("ok1.png", "bad2.png", "ok3.png")]}
open(os.path.join(kd3, "sreq.json"), "w").write(_json.dumps(sreq))
krea_gen.main(os.path.join(kd3, "sreq.json"))
assert _json.load(open(sreq["result"]))["scores"] == [0.02, 0.9, 0.02] and not loaded_krea
krea_gen.load = stub_load

# krea.generate: runs the generator process, turns results into images (None = filtered) and stats
krea.ensure_env = lambda progress=None: None
krea.ensure_weights = lambda progress=None: None
released = []
models.release_all = lambda: released.append(1)


class FakeProc:
    def __init__(self, cmd, stdout, text):
        self.req = _json.load(open(cmd[-1]))
        items = [{"out": it["out"], "seed": it["seed"], "filtered": it["seed"] == 2, "rerolled": it["seed"] == 2,
                  "generate_seconds": 1.5, "filter_seconds": 0.02, "nsfw_score": 0.0} for it in self.req["items"]]
        for it in items:
            if not it["filtered"]:
                open(it["out"], "w").write("png")
        _json.dump({"load_seconds": 40.0, "items": items}, open(self.req["result"], "w"))
        self.stdout = iter(['{"out": 1}\n'] * len(items))
        self.returncode = 0

    def wait(self):
        return 0

    def poll(self):
        return 0


krea.subprocess = types.SimpleNamespace(Popen=FakeProc, PIPE=-1, run=None)
sys.modules["PIL"].Image.open = lambda path: types.SimpleNamespace(convert=lambda m: f"img:{os.path.basename(path)}")
msgs = []
ims, st = krea.generate([{"prompt": "a", "width": 1280, "height": 705, "seed": 1},
                         {"prompt": "b", "width": 1280, "height": 704, "seed": 2}], msgs.append)
assert ims == ["img:00.png", None] and st["filtered"] == 1 and st["load_seconds"] == 40.0 and released
assert st["generate_seconds"] == 3.0 and st["filter_seconds"] == 0.04, st
assert any("image 2/2" in m for m in msgs)
# LoRAs go into the request (path, adapter, weight); triggers are put in front of prompts without them
procs = []
krea.subprocess = types.SimpleNamespace(Popen=lambda *a, **k: procs.append(FakeProc(*a, **k)) or procs[-1], PIPE=-1)
ims, st = krea.generate([{"prompt": "a cat", "width": 896, "height": 1152, "seed": 1},
                         {"prompt": "MDY a dog", "width": 896, "height": 1152, "seed": 3}],
                        loras=[dict(MOODY, trigger="mdy")])
assert procs[-1].req["loras"] == [MOODY], procs[-1].req["loras"]
assert [it["prompt"] for it in procs[-1].req["items"]] == ["mdy a cat", "MDY a dog"]
assert st["loras"] == ["user_moody_v1"]
ims, st = krea.generate([{"prompt": "a cat", "width": 896, "height": 1152, "seed": 1}])
assert procs[-1].req["loras"] == [] and procs[-1].req["items"][0]["prompt"] == "a cat" and st["loras"] == []
krea.generate([{"prompt": "a cat", "width": 896, "height": 1152, "seed": 1}], loras=[dict(MOODY, trigger="mdy")],
              add_triggers=False)
assert procs[-1].req["items"][0]["prompt"] == "a cat" and procs[-1].req["loras"] == [MOODY], "raw: no trigger added"


class FailingProc(FakeProc):
    def __init__(self, cmd, stdout, text):
        self.req = _json.load(open(cmd[-1]))
        _json.dump({"error": "LoRA user_moody_v1 could not be loaded: bad keys"}, open(self.req["result"], "w"))
        self.stdout, self.returncode = iter([]), 2

    def wait(self):
        return 2


krea.subprocess = types.SimpleNamespace(Popen=FailingProc, PIPE=-1)
try:
    krea.generate([{"prompt": "a", "width": 896, "height": 1152, "seed": 1}], loras=[MOODY])
    raise AssertionError("a failed generator must raise")
except RuntimeError as e:
    assert "LoRA user_moody_v1 could not be loaded: bad keys" in str(e), e
krea.subprocess = types.SimpleNamespace(Popen=FakeProc, PIPE=-1, run=None)
# krea.score_images: runs the score mode as a subprocess with the filter only (no Krea weights download)
weights_asked = []
krea.ensure_weights = lambda progress=None: weights_asked.append("krea")
krea.ensure_filter = lambda progress=None: weights_asked.append("filter")


class ScoreProc(FakeProc):
    def __init__(self, cmd, stdout, text):
        self.req = _json.load(open(cmd[-1]))
        assert self.req["mode"] == "score" and self.req["filter_dir"] == krea.FILTER_DIR
        _json.dump({"scores": [0.1 * i for i in range(len(self.req["images"]))]}, open(self.req["result"], "w"))
        self.stdout, self.returncode = iter([]), 0


krea.subprocess = types.SimpleNamespace(Popen=ScoreProc, PIPE=-1)
scores = krea.score_images(["/a.png", "/b.png", "/c.png"])
assert scores == [0.0, 0.1, 0.2] and weights_asked == ["filter"], (scores, weights_asked)
assert krea.flagged(0.6) and not krea.flagged(0.3) and krea.flagged(krea.NSFW_THRESHOLD)


class ShortProc(ScoreProc):  # a result missing scores must never pass unscored photos
    def __init__(self, cmd, stdout, text):
        self.req = _json.load(open(cmd[-1]))
        _json.dump({"scores": [0.1]}, open(self.req["result"], "w"))
        self.stdout, self.returncode = iter([]), 0


krea.subprocess = types.SimpleNamespace(Popen=ShortProc, PIPE=-1)
try:
    krea.score_images(["/a.png", "/b.png"])
    raise AssertionError("a short score list must fail")
except RuntimeError as e:
    assert "scores" in str(e)
try:  # a filter without an "nsfw" label is an error, not a pass
    krea_gen.nsfw_score(lambda im: [{"label": "safe", "score": 0.99}], None)
    raise AssertionError("missing nsfw label must fail")
except ValueError:
    pass
krea.subprocess = types.SimpleNamespace(Popen=FakeProc, PIPE=-1, run=None)
krea.ensure_weights = lambda progress=None: None
# prepare_krea_loras: the job's Krea LoRA files, downloaded once per version into the LoRA cache
downloads.clear()
kp = {"krea_loras": [{"name": "moody", "version": 1, "weight": 0.8, "trigger": "mdy",
                      "key": "loras/m/moody/v1/krea2.safetensors"}]}
got = worker.prepare_krea_loras(kp)
assert got == [{"path": os.path.join(worker.LORA_CACHE, "m/moody/v1/krea2.safetensors"), "adapter": "user_moody_v1",
                "weight": 0.8, "trigger": "mdy"}], got
worker.prepare_krea_loras(kp)
assert downloads == ["loras/m/moody/v1/krea2.safetensors"] and worker.prepare_krea_loras({}) == []
try:  # a key can never place a file outside the LoRA cache
    worker.prepare_krea_loras({"krea_loras": [dict(kp["krea_loras"][0], key="loras/../../etc/x.safetensors")]})
    raise AssertionError("a key escaping the cache must be refused")
except ValueError as e:
    assert "LoRA key" in str(e), e

# jobs use Krea where there is no person to keep
calls_k = []
real_generate = krea.generate
krea.generate = lambda items, progress=None, loras=(), add_triggers=True: calls_k.append(items) or ([Out(f"k{i}") for i in range(len(items))],
                                                                        {"filtered": 0})
worker.progress = lambda *a: None
worker.upload = lambda path, client, job_id, name, ctype: f"outputs/{name}"
r = worker.run_avatar({"client": "m", "job_id": "j"}, {"description": "a sailor", "candidates": 3, "seed": 5,
                                                       "model": "krea2"}, tempfile.mkdtemp())
assert [it["seed"] for it in calls_k[-1]] == [5, 6, 7] and "a sailor" in calls_k[-1][0]["prompt"]
assert r["model"] == "krea2" and {"candidate_1_key", "candidate_3_key"} <= set(r)
krea.generate = lambda items, progress=None, loras=(), add_triggers=True: calls_k.append(items) or ([None] * len(items), {"filtered": len(items)})
expect_filtered = False
try:
    worker.run_avatar({"client": "m", "job_id": "j"}, {"description": "x", "candidates": 2, "model": "krea2"}, tempfile.mkdtemp())
except krea.FilteredOut:
    expect_filtered = True
assert expect_filtered, "all candidates flagged: the job fails with the filter's reason"
krea.generate = lambda items, progress=None, loras=(), add_triggers=True: calls_k.append(items) or ([Out("a"), None, Out("c")], {"filtered": 1})
r = worker.run_photos({"client": "m", "job_id": "j"}, {"model": "krea2", "aspect": "landscape", "seed": 1, "count": 3,
                                                       "scenes": [{"prompt": "a misty harbour at dawn"},
                                                                  {"shot": {"where": "a library", "camera": "wide"}}],
                                                       "look": "film", "grain": False}, tempfile.mkdtemp())
pr = [it["prompt"] for it in calls_k[-1]]
assert pr[0].startswith("a misty harbour at dawn.") and "35mm film" in pr[0] and pr[1].startswith("in a library. Camera: wide")
assert "Keep this exact person" not in " ".join(pr) and calls_k[-1][0]["width"] == 1152
assert [sc["photo"] for sc in r["scenes"]] == [1, 2] and set(k for k in r if k.endswith("_key")) == {"photo_1_key", "photo_2_key"}
assert r["krea"]["filtered"] == 1
# Krea photos with a LoRA: the job's krea_loras reach krea.generate
seen_kl = []
krea.generate = lambda items, progress=None, loras=(), add_triggers=True: seen_kl.append(list(loras)) or ([Out("a")], {"filtered": 0})
worker.run_photos({"client": "m", "job_id": "j"}, {"model": "krea2", "aspect": "portrait", "seed": 1, "count": 1,
                                                   "scenes": [{"prompt": "a quiet street"}], **kp}, tempfile.mkdtemp())
assert [l["adapter"] for l in seen_kl[-1]] == ["user_moody_v1"], seen_kl
# raw: each prompt goes to Krea exactly as typed, and krea.generate adds no trigger words
seen_raw = []
krea.generate = lambda items, progress=None, loras=(), add_triggers=True: seen_raw.append((items, add_triggers)) or (
    [Out("a"), Out("b")], {"filtered": 0})
worker.run_photos({"client": "m", "job_id": "j"}, {"model": "krea2", "aspect": "portrait", "seed": 1, "count": 2,
                                                   "raw": True, "look": "portrait", "natural_skin": True,
                                                   "scenes": [{"prompt": "a red fox in snow, wide shot"},
                                                              {"location": "a misty harbour"}], **kp}, tempfile.mkdtemp())
items, add = seen_raw[-1]
assert [it["prompt"] for it in items] == ["a red fox in snow, wide shot", "a misty harbour"] and add is False, seen_raw
krea.generate = lambda items, progress=None, loras=(), add_triggers=True: seen_raw.append((items, add_triggers)) or (
    [Out(str(i)) for i in range(len(items))], {"filtered": 0})
r = worker.run_photos({"client": "m", "job_id": "j"}, {"model": "krea2", "aspect": "portrait", "seed": 1, "count": 3,
                                                       "raw": True, "scenes": [{"prompt": "a red fox"}]}, tempfile.mkdtemp())
assert [it["prompt"] for it in seen_raw[-1][0]] == ["a red fox"] * 3, "extra photos repeat the prompt, no random places"
assert len({it["seed"] for it in seen_raw[-1][0]}) == 3 and [s["source"] for s in r["scenes"]] == ["prompt"] * 3
worker.run_photos({"client": "m", "job_id": "j"}, {"model": "krea2", "aspect": "portrait", "seed": 1, "count": 1,
                                                   "scenes": [{"prompt": "a red fox"}]}, tempfile.mkdtemp())
assert seen_raw[-1][0][0]["prompt"].startswith("a red fox.") and "Photorealistic" in seen_raw[-1][0][0]["prompt"]
assert seen_raw[-1][1] is True, "default: look sentences and triggers as before"
models.fit = lambda im, w, h: im
krea.generate = lambda items, progress=None, loras=(), add_triggers=True: calls_k.append(items) or ([Out(f"f{it['seed']}") for it in items], {})
worker.load_cast = lambda params, workdir: []
takes.clear()
worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=20, variations=2, image_model="krea2", look="phone",
                                                      sound=False), tempfile.mkdtemp())
assert [it["seed"] for it in calls_k[-1]] == [20, 21] and "smartphone" in calls_k[-1][0]["prompt"]
assert [t["image"].n for t in takes] == ["f20", "f21"], "each take starts from its Krea frame"
takes.clear()
n_before = len(calls_k)
worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=20, sound=False), tempfile.mkdtemp())
assert len(calls_k) == n_before and "image" not in takes[0], "default: no Krea, text-to-video as before"
# Krea LoRAs reach the Krea first frame only; Wan's set_loras never sees them
seen_kl.clear()
krea.generate = lambda items, progress=None, loras=(), add_triggers=True: seen_kl.append(list(loras)) or ([Out("f")] * len(items), {})
takes.clear()
worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=20, image_model="krea2", sound=False, **kp),
                 tempfile.mkdtemp())
assert [l["adapter"] for l in seen_kl[-1]] == ["user_moody_v1"] and takes[0]["image"].n == "f"
assert worker.prepare_loras(dict(kp)) == [], "Wan loading takes params['loras'] only"
# story: Krea first frames (with the LoRAs) for shots without avatars; all-avatar plans leave them unused
shots = [{"prompt": "a pier", "characters": []}, {"prompt": "two friends", "characters": ["dana"]}]
plan = {"visual_bible": "rainy port town."}
seen_kl.clear()
frames, unused = worker.story_first_frames({"client": "m", "job_id": "j"}, dict(image_model="krea2", **kp),
                                           plan, shots, 7, {"width": 1280, "height": 704})
assert frames[0].n == "f" and frames[1] is None and unused is False
assert [l["adapter"] for l in seen_kl[-1]] == ["user_moody_v1"]
seen_kl.clear()
frames, unused = worker.story_first_frames({"client": "m", "job_id": "j"}, dict(image_model="krea2", **kp),
                                           plan, [shots[1]], 7, {})
assert frames == [None] and unused is True and seen_kl == []
frames, unused = worker.story_first_frames({"client": "m", "job_id": "j"}, {"image_model": "wan"}, plan, shots, 7, {})
assert frames == [None, None] and unused is False and seen_kl == []
# same items as before the refactor: free shots only, seed + shot index, default 1280x704, then fit
seen_items, fits = [], []
krea.generate = lambda items, progress=None, loras=(), add_triggers=True: seen_items.append(items) or ([Out("f")] * len(items), {})
models.fit = lambda im, w, h: fits.append((w, h)) or im
shots3 = [{"prompt": "a pier", "characters": []}, shots[1], {"prompt": "gulls", "characters": []}]
worker.story_first_frames({"client": "m", "job_id": "j"}, {"image_model": "krea2"}, plan, shots3, 7, {})
assert [(it["seed"], it["width"], it["height"]) for it in seen_items[-1]] == [(7, 1280, 704), (9, 1280, 704)]
assert seen_items[-1][0]["prompt"].startswith("rainy port town. a pier") and fits == [(1280, 704)] * 2
models.fit = lambda im, w, h: im
krea.generate = lambda items, progress=None, loras=(), add_triggers=True: ([None] * len(items), {"filtered": len(items)})
try:
    worker.story_first_frames({"client": "m", "job_id": "j"}, {"image_model": "krea2"}, plan, shots, 7, {})
    raise AssertionError("a filtered first frame must fail the story")
except krea.FilteredOut as e:
    assert "change the story" in str(e)
# an avatar's Krea LoRA: raw photos get the avatar's name in front (only when missing); a video with
# krea_avatar gets a Krea first frame (no Qwen keyframe); story shots with only that avatar too
seen_raw.clear()
krea.generate = lambda items, progress=None, loras=(), add_triggers=True: seen_raw.append((items, add_triggers)) or (
    [Out(str(i)) for i in range(len(items))], {"filtered": 0})
worker.run_photos({"client": "m", "job_id": "j"}, {"model": "krea2", "aspect": "portrait", "seed": 1, "count": 2, "raw": True,
                                                   "avatar_trigger": "ready1",
                                                   "scenes": [{"prompt": "on a ferry"}, {"prompt": "ready1 on a pier"}]},
                  tempfile.mkdtemp())
assert [it["prompt"] for it in seen_raw[-1][0]] == ["ready1 on a ferry", "ready1 on a pier"], seen_raw[-1]
worker.run_photos({"client": "m", "job_id": "j"}, {"model": "krea2", "aspect": "portrait", "seed": 1, "count": 1, "raw": True,
                                                   "avatar_trigger": "al", "scenes": [{"prompt": "also on a ferry"}]},
                  tempfile.mkdtemp())
assert seen_raw[-1][0][0]["prompt"] == "al also on a ferry", "whole-word match: 'al' is not in 'also'"
assert krea.with_triggers("also here", [{"trigger": "al"}]) == "al also here"
cast_calls = []
worker.load_cast = lambda params, workdir: [("ready1", ["ref"], "")]
models.make_keyframe = lambda *a, **k: cast_calls.append(a) or Out("kf")
takes.clear()
worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=20, image_model="krea2", sound=False,
                                                      krea_avatar="ready1", avatars=[{"name": "ready1"}]), tempfile.mkdtemp())
assert takes[0]["image"].n == "0" and not cast_calls, "Krea frame with the avatar's LoRA, no Qwen keyframe"
shots3 = [{"prompt": "a pier", "characters": []}, {"prompt": "ready1 walks", "characters": ["ready1"]},
          {"prompt": "two friends", "characters": ["ready1", "dana"]}]
seen_raw.clear()
frames, unused = worker.story_first_frames({"client": "m", "job_id": "j"}, {"image_model": "krea2", "krea_avatar": "ready1"},
                                           plan, shots3, 7, {})
assert frames[0] is not None and frames[1] is not None and frames[2] is None, frames
# shots without characters never get the avatar's LoRA or name: a separate Krea run without it
seen_lr = []
krea.generate = lambda items, progress=None, loras=(), add_triggers=True: seen_lr.append(([it["prompt"] for it in items], [l["adapter"] for l in loras])) or (
    [Out(str(i)) for i in range(len(items))], {"filtered": 0})
worker.story_first_frames({"client": "m", "job_id": "j"},
                          {"image_model": "krea2", "krea_avatar": "ready1", "krea_avatar_lora": "avatar-ready1",
                           "krea_loras": [{"name": "avatar-ready1", "version": 1, "weight": 1.0, "trigger": "ready1",
                                           "key": "loras/m/avatar-ready1/v1/krea2.safetensors"},
                                          {"name": "moody", "version": 1, "weight": 0.8, "trigger": "mdy",
                                           "key": "loras/m/moody/v1/krea2.safetensors"}]}, plan, shots3, 7, {})
by_adapters = {tuple(a): p for p, a in seen_lr}
assert ("user_moody_v1",) in by_adapters and ("user_avatar_ready1_v1", "user_moody_v1") in by_adapters, seen_lr
assert "a pier" in by_adapters[("user_moody_v1",)][0], "scenery shot runs without the avatar LoRA"
worker.load_cast = lambda params, workdir: []
krea.generate = real_generate

# ---------------------------------------------------------------- avatar_photoset: photo set -> filter -> dataset -> train job
import photos as _photos  # noqa: E402
assert len(_photos.KREA_DATASET_PROMPTS) == 24 and len(set(_photos.KREA_DATASET_PROMPTS)) == 24
import re as _re  # noqa: E402
assert not any(_re.search(r"\b(he|him|his|she|her)\b", x) for x in _photos.KREA_DATASET_PROMPTS), "neutral wording"
assert not any("yellow raincoat" in x for x in _photos.KREA_DATASET_PROMPTS), "no outfit from one avatar"


class FakeS3W:
    def __init__(self, objects):
        self.objects, self.uploaded = dict(objects), {}

    def get_object(self, Bucket, Key):
        import io
        if Key not in self.objects:
            raise KeyError(Key)  # stands in for NoSuchKey
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def upload_file(self, path, Bucket, Key, ExtraArgs=None):
        self.uploaded[Key] = open(path, "rb").read()

    def download_file(self, Bucket, Key, path):
        open(path, "wb").write(b"img")


avatar_doc = {"name": "ava", "description": "a sailor", "reference_key": "avatars/m/ava/images/1.png", "created_at": 77,
              "images": [{"key": f"avatars/m/ava/images/{i}.png"} for i in range(1, 5)],
              "krea_lora": {"state": "queued", "lora": "avatar-ava", "version": 1, "trigger": "ava", "job_ids": ["pj"]}}
fs3 = worker.s3 = FakeS3W({"avatars/m/ava/avatar.json": _json.dumps(avatar_doc).encode()})
made, refs_used, put_jobs, sent_msgs = [], [], [], []
worker.load_references = lambda keys, workdir, prefix: refs_used.append(list(keys)) or [Out("r1"), Out("r2"), Out("r3")]
models.make_photo = lambda references, prompt, neg, w, h, seed, best=False: made.append((prompt, w, h, seed, len(references))) or Out(f"p{seed}")
class FakeClientError(Exception):  # botocore is stubbed in tests; the worker catches ClientError by .response
    def __init__(self, response, op):
        super().__init__(op)
        self.response = response


worker.ClientError = FakeClientError
existing_ids = set()
job_items = {}


def put_item(Item, ConditionExpression=None):
    if ConditionExpression and Item["job_id"] in existing_ids:
        raise worker.ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")
    existing_ids.add(Item["job_id"])
    job_items[Item["job_id"]] = Item
    put_jobs.append(Item)


worker.jobs = mock.MagicMock(put_item=put_item, get_item=lambda Key: {"Item": job_items[Key["job_id"]]}
                             if Key["job_id"] in job_items else {})
worker.sqs = mock.MagicMock(send_message=lambda QueueUrl, MessageBody: sent_msgs.append((QueueUrl, MessageBody)))
worker.TRAIN_QUEUE_URL = "tq"
krea.score_images = lambda paths, progress=None: [0.9 if i in (3, 7) else 0.01 for i in range(len(paths))]
pparams = {"avatar": "ava", "lora": "avatar-ava", "version": 1, "trigger": "ava", "count": 24, "avatar_created_at": 77}
r = worker.run_avatar_photoset({"client": "m", "job_id": "pj"}, dict(pparams), tempfile.mkdtemp())
assert refs_used[0] == [f"avatars/m/ava/images/{i}.png" for i in (1, 2, 3)], "the avatar's first 3 images"
assert len(made) == 24 and all((w, h) == (1024, 1024) and n == 3 for _, w, h, _, n in made)
assert _photos.KREA_DATASET_PROMPTS[0] in made[0][0] and len({s for _, _, _, s, _ in made}) == 24
assert r["kept"] == 22 and r["filtered"] == 2, r
manifest = _json.loads(fs3.objects["datasets/m/avatar-ava/v1/manifest.json"])
assert manifest["file_count"] == 22 and manifest["trigger"] == "ava" and manifest["kind"] == "character"
assert [f["path"] for f in manifest["files"]][:2] == ["ava_01.png", "ava_02.png"] and "ava_04.png" not in [f["path"] for f in manifest["files"]]
assert "datasets/m/avatar-ava/v1/raw/ava_01.png" in fs3.uploaded and len([k for k in fs3.uploaded if "/raw/" in k]) == 22
caps = _json.loads(fs3.objects["datasets/m/avatar-ava/v1/captions.json"])
fp = __import__("hashlib").sha256(_json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()[:16]
assert caps["fingerprint"] == fp and caps["items"]["p0001.png"].startswith("ava, ") and len(caps["items"]) == 22
tj = [j for j in put_jobs if j.get("type") == "train"][-1]
assert any(j.get("job_id") == "_control_trainer" and j.get("hold") is False for j in put_jobs), "trainer hold cleared"
tparams = _json.loads(tj["params"])
assert tj["type"] == "train" and tj["status"] == "queued" and tj["client"] == "m"
assert tparams["model"] == "krea2" and tparams["dataset"] == "avatar-ava/v1" and tparams["lora"] == "avatar-ava"
assert tparams["version"] == 1 and tparams["trigger"] == "ava" and tparams["avatar"] == "ava" and tparams["fingerprint"] == fp
assert sent_msgs[-1] == ("tq", _json.dumps({"job_id": tj["job_id"]})) and r["train_job_id"] == tj["job_id"]
kl = _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["krea_lora"]
assert kl["state"] == "training" and kl["job_ids"] == ["pj", tj["job_id"]], kl
# redelivery after a restart (the train job exists, its message maybe lost): no new photos, no second
# train job; the avatar is repaired and the message re-sent while the train job is still queued
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(avatar_doc).encode()
n_train = len([j for j in put_jobs if j.get("type") == "train"])
n_msgs, n_made = len(sent_msgs), len(made)
r = worker.run_avatar_photoset({"client": "m", "job_id": "pj"}, dict(pparams), tempfile.mkdtemp())
assert "skipped" in r and len(made) == n_made and len([j for j in put_jobs if j.get("type") == "train"]) == n_train
assert len(sent_msgs) == n_msgs + 1 and sent_msgs[-1][1] == _json.dumps({"job_id": tj["job_id"]})
kl = _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["krea_lora"]
assert kl["state"] == "training" and tj["job_id"] in kl["job_ids"], kl
job_items[tj["job_id"]] = dict(job_items[tj["job_id"]], status="running")
worker.run_avatar_photoset({"client": "m", "job_id": "pj"}, dict(pparams), tempfile.mkdtemp())
assert len(sent_msgs) == n_msgs + 1, "no re-send once the trainer has it"
# the avatar replaced while the photos render: nothing uploaded or queued, the newer build untouched
newer = dict(avatar_doc, created_at=99, krea_lora=dict(avatar_doc["krea_lora"], job_ids=["newer"], version=2))
real_make_photo = models.make_photo


def replace_midway(*a, **k):
    fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(newer).encode()
    return real_make_photo(*a, **k)


fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(avatar_doc).encode()
models.make_photo = replace_midway
fs3.uploaded.clear()
avatar_doc2 = dict(avatar_doc, krea_lora=dict(avatar_doc["krea_lora"], job_ids=["pj2"]))
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(avatar_doc2).encode()
r = worker.run_avatar_photoset({"client": "m", "job_id": "pj2"}, dict(pparams), tempfile.mkdtemp())
assert "skipped" in r and not fs3.uploaded, r
assert _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["krea_lora"]["job_ids"] == ["newer"], "newer build untouched"
models.make_photo = real_make_photo
# any error after the photos started marks the build failed (so it can be retried)
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(avatar_doc, krea_lora=dict(avatar_doc["krea_lora"], job_ids=["pj3"]))).encode()
krea.score_images = lambda paths, progress=None: (_ for _ in ()).throw(RuntimeError("content filter failed (exit 1)"))
try:
    worker.run_avatar_photoset({"client": "m", "job_id": "pj3"}, dict(pparams), tempfile.mkdtemp())
    raise AssertionError("must raise")
except RuntimeError:
    pass
kl = _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["krea_lora"]
assert kl["state"] == "failed" and "content filter failed" in kl["error"], kl
# too many flagged: the job fails, no dataset, no train job, avatar marked failed
krea.score_images = lambda paths, progress=None: [0.9] * 10 + [0.01] * (len(paths) - 10)
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(avatar_doc).encode()
n_jobs = len(put_jobs)
pparams["version"] = 3
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(avatar_doc, krea_lora=dict(avatar_doc["krea_lora"], job_ids=["pj4"]))).encode()
try:
    worker.run_avatar_photoset({"client": "m", "job_id": "pj4"}, dict(pparams), tempfile.mkdtemp())
    raise AssertionError("must fail when fewer than 16 photos pass the filter")
except krea.FilteredOut as e:
    assert "flagged by the content filter" in str(e)
assert len(put_jobs) == n_jobs and "datasets/m/avatar-ava/v3/manifest.json" not in fs3.objects
pparams["version"] = 1
kl = _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["krea_lora"]
assert kl["state"] == "failed" and "content filter" in kl["error"]
# no trainer deployed: a clear failure
worker.TRAIN_QUEUE_URL = ""
try:
    worker.run_avatar_photoset({"client": "m", "job_id": "pj"}, dict(pparams), tempfile.mkdtemp())
    raise AssertionError("no trainer: must fail")
except RuntimeError as e:
    assert "trainer" in str(e)
assert "avatar_photoset" in worker.RUNNERS
# the train queue URL also comes from settings.json next to the code (Terraform; boxes keep old user data)
sd = tempfile.mkdtemp()
open(os.path.join(sd, "settings.json"), "w").write(_json.dumps({"train_queue_url": "https://sqs/train"}))
assert worker.load_settings(sd).get("train_queue_url") == "https://sqs/train"
assert worker.load_settings(tempfile.mkdtemp()) == {}
# ---------------------------------------------------------------- process: the long-take work folder
import shutil as _shutil  # noqa: E402
worker.shutil = _shutil  # an earlier test mocked it
worker.WORK_ROOT = tempfile.mkdtemp()
worker.JobWatch = lambda job_id, receipt: mock.MagicMock(started=0)
statuses = []
worker.set_status = lambda job_id, status, **kw: statuses.append(status)
real_runners = dict(worker.RUNNERS)
for outcome in ("done", "failed", "cancelled"):
    def runner(job, params, workdir, outcome=outcome):
        os.makedirs(worker.take_work_dir(job["job_id"]))
        open(os.path.join(worker.take_work_dir(job["job_id"]), "segment_00.npz"), "w").write("x")
        if outcome == "failed":
            raise RuntimeError("boom")
        if outcome == "cancelled":
            raise models.JobCancelled("cancelled")
        return {}
    worker.RUNNERS["video"] = runner
    job_items["lt-" + outcome] = {"job_id": "lt-" + outcome, "status": "queued", "type": "video",
                                  "client": "m", "params": _json.dumps({"seconds": 50})}
    worker.process("lt-" + outcome, "r")
    assert not os.path.exists(worker.take_work_dir("lt-" + outcome)), f"work folder removed when {outcome}"
assert {"done", "failed", "cancelled"} <= set(statuses), statuses
worker.RUNNERS.update(real_runners)
assert worker.take_work_dir("abc") == os.path.join(worker.WORK_ROOT, "abc")
import time as _time  # noqa: E402
os.makedirs(worker.take_work_dir("old")), os.makedirs(worker.take_work_dir("new"))
os.utime(worker.take_work_dir("old"), (_time.time() - 4 * 86400,) * 2)
worker.sweep_work_dirs()
assert not os.path.exists(worker.take_work_dir("old")) and os.path.exists(worker.take_work_dir("new"))
worker.WORK_ROOT = os.path.join(tempfile.mkdtemp(), "missing")
worker.sweep_work_dirs()  # no folder yet: nothing to do
print("worker tests passed")
