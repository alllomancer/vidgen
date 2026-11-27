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

real_render_segment = models._render_segment  # most tests stub it; the prompt-embedding test needs the real one
real_generate_sound = models.generate_sound

# Encoder settings: X264_ARGS unless a clip is an intermediate (x264=...)
models.write_video([], "/x.mp4")
assert sys.modules["imageio"].get_writer.call_args.kwargs["ffmpeg_params"] == models.X264_ARGS
models.write_video([], "/x.mp4", x264=models.X264_INTERMEDIATE)
assert sys.modules["imageio"].get_writer.call_args.kwargs["ffmpeg_params"] == ["-crf", "10", "-preset", "veryfast"]

# Pipeline calls log where their time went: one JSON line with the seconds per denoising step
import contextlib as _contextlib  # noqa: E402
import io as _io  # noqa: E402
import json as _jsont  # noqa: E402

logged = _io.StringIO()
with _contextlib.redirect_stdout(logged):
    out = models._timed("t2v", lambda **kw: [models._on_step(None, i, 0, {}) for i in range(4)] and kw, width=8)
line = _jsont.loads(logged.getvalue())
assert out == {"width": 8} and line["event"] == "pipeline_timing" and line["call"] == "t2v"
assert len(line["steps"]) == 4 and set(line) == {"event", "call", "steps", "finish", "seconds"}


# a cancelled or failed call: the pipeline's components go back to RAM (its own clean-up was skipped)
class Cancelled:
    freed = 0

    def __call__(self, **kw):
        raise models.JobCancelled("cancelled")

    def maybe_free_model_hooks(self):
        self.freed += 1


cancelled = Cancelled()
_stubs.expect_error(models._timed, "t2v", cancelled, exc=models.JobCancelled)
assert cancelled.freed == 1
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


nexts = []  # the `then` hint of each pass: the model to warm up for the next one


def fake_segment(prompt, negative, num, width, height, seed, image, then=None):
    nexts.append(then)
    segments.append((num, seed, image))
    seg_prompts.append(prompt)
    return [f"f{len(segments)}_{i}" for i in range(num)]


written = []
models._render_segment = fake_segment
models._to_uint8_frames = lambda f: f
models.Image = mock.MagicMock(fromarray=lambda f: f"img:{f}")
models.soften = lambda image, radius=None: image  # identity here; the halo test below marks it
models.write_video = lambda frames, path, fps=16, grain=None, x264=None: written.append((path, len(frames), fps, grain))
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
assert nexts == ["i2v", "i2v", None], "every pass but the last says I2V runs next (a no-op unless this one is T2V)"
matched.clear(), segments.clear()
models.generate_video("/o.mp4", "x", seconds=10, seed=7)
assert matched == [] and segments[1][2] == "img:f1_80", "10 s or less: no colour matching"
# resume: finished segments saved in the work folder; a re-delivered take renders only the rest
import json as _jsonr  # noqa: E402
models._save_segment = lambda path, frames: _jsonr.dump(list(frames), open(path, "w"))
models._load_segment = lambda path: _jsonr.load(open(path))
captured = []
real_write_video = models.write_video
models.write_video = lambda frames, path, fps=16, grain=None, x264=None: captured.append(list(frames))
msgs = []


def seeded_segment(prompt, negative, num, width, height, seed, image, then=None):  # frames depend on the seed only
    nexts.append(then)
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


def dies_after_six(*a, **kw):
    if len(segments) == 6:
        raise RuntimeError("spot interruption")
    return seeded_segment(*a, **kw)


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
assert nexts[-10:] == ["i2v", "edit", "i2v", "i2v", "edit", "i2v", "i2v", "edit", "i2v", None], \
    "the pass before a re-anchored one says the edit model runs next"
assert segments[2][2] == "img:anchored:img:s2_80~#" and segments[3][2] == "img:s3_80~"
assert transfers == [("stats", "img:s2_80~"), ("stats", "img:s5_80~"), ("stats", "img:s8_80~")], \
    "the anchored frame takes the colours of the frame it replaces"
anchored.clear(), segments.clear()
models.generate_video("/o.mp4", "x", seconds=50, fps=16, seed=1)
assert anchored == [], "no anchoring without anchor_every"
# halo: segments 2+ of a long take start from a softened copy of the previous last frame
softened = []
models.soften = lambda image, radius=None: softened.append((image, radius)) or f"soft:{image}"
assert models.CONDITION_BLUR == 0, "off by default (live check 2026-11-25: it didn't reduce the halo)"
segments.clear()
models.generate_video("/o.mp4", "x", seconds=50, fps=16, seed=1)
assert not softened, "blur 0: nothing softened"
real_blur = models.CONDITION_BLUR
models.CONDITION_BLUR = 0.8  # the mechanism, when switched on
segments.clear(), captured.clear()
models.generate_video("/o.mp4", "x", seconds=50, fps=16, seed=1)
assert segments[0][2] is None and all(str(s[2]).startswith("soft:img:") for s in segments[1:]), segments[:3]
assert len(softened) == 9 and all(r == 0.8 for _, r in softened)
assert not any(str(f).startswith("soft:") for f in captured[-1]), "the frames you see are never softened"
segments.clear(), softened.clear()
models.generate_video("/o.mp4", "x", seconds=5, fps=16, seed=1, image="key")
assert segments[0][2] == "key" and not softened, "5 s: one segment, nothing softened"
# the blur is part of the resume key: segments saved with another radius are not reused
wdb = tempfile.mkdtemp()
segments.clear()
models.generate_video("/o.mp4", "x", seconds=50, fps=16, seed=1, work_dir=wdb)
models.CONDITION_BLUR = 1.2
segments.clear()
models.generate_video("/o.mp4", "x", seconds=50, fps=16, seed=1, work_dir=wdb)
assert len(segments) == 10, "new blur: rendered again"
softened.clear(), segments.clear()
models.CONDITION_BLUR = 0.8
models.generate_video("/o.mp4", "x", seconds=50, fps=16, seed=1, anchor_every=3, reanchor=lambda frame, n=0: f"a:{frame}")
assert len(softened) == 6 and all(not str(img).startswith("a:") for img, _ in softened), "re-anchored frames aren't softened"
models.CONDITION_BLUR = real_blur
models.soften = lambda image, radius=None: image
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


def good_plan(prompt, n, cast=(), seconds=None, image=None):
    plans.append((prompt, n, list(cast), seconds))
    return [f"step {i}" for i in range(n)]


def boom(prompt, n, cast=(), seconds=None, image=None):
    raise ValueError("model did not return JSON")


models.plan_segments = good_plan
takes.clear()
worker.WORK_ROOT = tempfile.mkdtemp()
res = worker.run_video({"client": "m", "job_id": "j"}, dict(base, seed=1, seconds=50), tempfile.mkdtemp())
assert plans == [("p", 10, [], 50)] and res["segment_prompts"] == [f"step {i}" for i in range(10)], (plans, res)
assert takes[0]["segment_prompts"] == [f"step {i}" for i in range(10)]
for bad in (lambda p, n, cast=(), seconds=None, image=None: ["a"] * (n - 1),
            lambda p, n, cast=(), seconds=None, image=None: ["a"] * (n - 1) + ["  "],
            lambda p, n, cast=(), seconds=None, image=None: ["a"] * (n - 1) + ["x" * 501],
            lambda p, n, cast=(), seconds=None, image=None: "a" * n, boom):
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
models.plan_segments = lambda p, n, cast=(), seconds=None, image=None: (_ for _ in ()).throw(models.JobCancelled("stop"))
_stubs.expect_error(worker.run_video, {"client": "m", "job_id": "j"}, dict(base, seed=1, seconds=50),
                    tempfile.mkdtemp(), exc=models.JobCancelled)
# the first frame is made before the planner, which sees it; the render reuses it (one keyframe)
events, plan_images = [], []
models.plan_segments = lambda p, n, cast=(), seconds=None, image=None: events.append("plan") or plan_images.append(image) \
    or [f"s{i}" for i in range(n)]
worker.load_cast = lambda params, workdir: [("dana", [Img(0)], "d")]
real_mk = models.make_keyframe
models.make_keyframe = lambda cast, scene, w, h, seed, best=False: events.append("kf") or Out("kf")
real_gv = models.generate_video
models.generate_video = lambda out, **kw: events.append("video") or takes.append(kw) or out
takes.clear()
worker.WORK_ROOT = tempfile.mkdtemp()
res = worker.run_video({"client": "m", "job_id": "ff1"}, dict(base, seed=1, seconds=50, sound=False), tempfile.mkdtemp())
assert events == ["kf", "plan", "video"], events
assert plan_images[-1] and plan_images[-1].endswith("first_frame.png") and takes[0]["image"].n == "kf"
assert "keyframe_key" in res and res["segment_prompts"] == [f"s{i}" for i in range(10)]
worker.load_cast = lambda params, workdir: []
events.clear()
worker.run_video({"client": "m", "job_id": "ff2"}, dict(base, seed=1, seconds=50, sound=False, image_model="wan"),
                 tempfile.mkdtemp())
assert events == ["plan", "video"] and plan_images[-1] is None, "no first frame: the planner works from text"
# plan only: the first frame and the plan, no video or sound; beats every 5 s and the seed to render them with
events.clear(), takes.clear()
worker.load_cast = lambda params, workdir: [("dana", [Img(0)], "d")]
res = worker.run_video({"client": "m", "job_id": "pl1"}, dict(base, seed=4, seconds=50, plan_only=True), tempfile.mkdtemp())
assert events == ["kf", "plan"] and not takes, events
assert res["plan_only"] is True and res["seed"] == 4 and res["segment_prompts"] == [f"s{i}" for i in range(10)]
assert res["beats"].splitlines() == [f"{5 * i}: s{i}" for i in range(10)] and res["keyframe_key"] == "outputs/keyframe.png"
models.plan_segments = lambda p, n, cast=(), seconds=None, image=None: [f"she turns; step {i}" for i in range(n)]
res = worker.run_video({"client": "m", "job_id": "pl3"}, dict(base, seed=4, seconds=50, plan_only=True), tempfile.mkdtemp())
assert ";" not in res["beats"] and res["beats"].splitlines()[1] == "5: she turns, step 1", res["beats"]
models.plan_segments = lambda p, n, cast=(), seconds=None, image=None: (_ for _ in ()).throw(ValueError("no JSON"))
_stubs.expect_error(worker.run_video, {"client": "m", "job_id": "pl2"}, dict(base, seed=4, seconds=50, plan_only=True),
                    tempfile.mkdtemp(), contains="could not write a timeline", exc=RuntimeError)
worker.load_cast = lambda params, workdir: []
models.make_keyframe, models.generate_video = real_mk, real_gv
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
# with the first frame: the image comes first, with the keep-to-the-frame rule
asked_full = []
models._chat = lambda messages, max_new_tokens=1024, **k: asked_full.append(messages) or '{"segments": ["a", "b", "c", "d"]}'
models.plan_segments("a dog runs", 4, image="/w/first_frame.png")
content = asked_full[-1][0]["content"]
assert content[0] == {"type": "image", "image": "file:///w/first_frame.png"}, content[0]
assert "opening frame" in content[1]["text"] and "no one else appears" in content[1]["text"]
models.plan_segments("a dog runs", 4)
assert asked_full[-1][0]["content"][0]["type"] == "text", "no first frame: text only"
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
open_sessions = []
models.generate_sound = lambda path, out, prompt, seconds, seed: open_sessions.append(models._audio.get("sessions")) or out
real_sound_for_clip("/v.mp4", "p", 12, 3, d)
assert open_sessions == [1, 1, 1] and models._audio["sessions"] == 0, "all chunks inside one sound session"


# The sound model goes to the GPU once per sound_session (a long take's chunks, a story's shots), not per call
class Mover:
    def __init__(self):
        self.moves = []

    def to(self, device):
        self.moves.append(device)


net, feats = Mover(), Mover()
models._audio.clear()
with models.sound_session():
    pass
assert not models._audio.get("models"), "opening a session loads nothing"
models._audio["models"] = ("cfg", net, feats)
during = []
models._generate_sound = lambda cfg, n, f, path, out, prompt, seconds, seed: during.append(list(n.moves)) or out
assert real_generate_sound("/v.mp4", "/a.wav", "p", 5, 1) == "/a.wav"
assert net.moves == feats.moves == ["cuda", "cpu"] and during == [["cuda"]], "a single call: there and back"
net.moves.clear(), feats.moves.clear(), during.clear()
with models.sound_session():
    for _ in range(3):
        real_generate_sound("/v.mp4", "/a.wav", "p", 5, 1)
    assert net.moves == ["cuda"], "it stays on the GPU between the calls"
assert net.moves == feats.moves == ["cuda", "cpu"] and during == [["cuda"]] * 3
net.moves.clear(), feats.moves.clear()
models._generate_sound = lambda *a: (_ for _ in ()).throw(RuntimeError("out of memory"))
try:
    with models.sound_session():
        real_generate_sound("/v.mp4", "/a.wav", "p", 5, 1)
except RuntimeError:
    pass
assert net.moves == ["cuda", "cpu"] and not models._audio.get("on_gpu") and models._audio["sessions"] == 0, \
    "a failed call still leaves the GPU free"
models._audio.clear()
# Prefetch reads every weight file fully, in parallel chunks.
pdir = tempfile.mkdtemp()
os.makedirs(os.path.join(pdir, "transformer"))
open(os.path.join(pdir, "transformer", "a.safetensors"), "wb").write(b"x" * 1000)
open(os.path.join(pdir, "b.bin"), "wb").write(b"y" * 10)
open(os.path.join(pdir, "readme.md"), "w").write("skip")
models.PREFETCH_CHUNK = 64
assert models.prefetch(pdir, threads=3) == 1010
# A model given as a repo id is warmed from its Hugging Face cache snapshot; not downloaded yet = nothing to warm
sys.modules["huggingface_hub"] = types.SimpleNamespace(
    snapshot_download=lambda repo, local_files_only: pdir if repo == "org/cached" and local_files_only
    else (_ for _ in ()).throw(FileNotFoundError(repo)))
assert models._model_folder(pdir) == pdir and models._model_folder("org/missing") is None
assert models.prefetch_model("org/cached") == 1010 and models.prefetch_model("org/missing") == 0
del sys.modules["huggingface_hub"]
# the edit model and the VLM are warmed before they load (Wan does it in _load_wan)
warmed = []
real_prefetch_model, models.prefetch_model = models.prefetch_model, warmed.append
sys.modules["diffusers"] = sys.modules["transformers"] = mock.MagicMock()
models._load_edit()
models._load_vlm()
assert warmed == [models.EDIT_MODEL, models.VLM_MODEL], warmed
del sys.modules["diffusers"], sys.modules["transformers"]
# warm-ahead: the next model's files are read in the background, one warm-up at a time, never for a loaded model
import threading as _threading  # noqa: E402

gate = _threading.Event()
warmed.clear()
models.prefetch_model = lambda source: (gate.wait(5), warmed.append(source))
kept_loaded = dict(models._loaded)
models._loaded.clear(), models._warming.clear()
thread = models.warm_ahead("i2v")
assert thread is not None and models.warm_ahead("edit") is None, "one warm-up at a time"
gate.set(), thread.join()
assert warmed == [models.I2V_MODEL], "no local bf16 copy here: the repo id (its Hugging Face cache folder)"
models._loaded["edit"] = object()
assert models.warm_ahead("edit") is None, "already loaded: nothing to read"
del models._loaded["edit"]
gate.clear(), warmed.clear()
thread = models.warm_ahead("edit")
_threading.Timer(0.05, gate.set).start()
models._warm(models.EDIT_MODEL)  # a load waits for the warm-up of its own files, then reads (quick: cached)
assert not thread.is_alive() and warmed == [models.EDIT_MODEL] * 2 and not models._warming
models._loaded.update(kept_loaded)
models.prefetch_model = real_prefetch_model
# a Wan model loads from its local bf16 copy when there is one
models.BF16_DIR = tempfile.mkdtemp()
assert models._wan_source("Wan-AI/Wan2.2-I2V-A14B-Diffusers") == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
os.makedirs(models.bf16_path("Wan-AI/Wan2.2-I2V-A14B-Diffusers"))
open(os.path.join(models.bf16_path("Wan-AI/Wan2.2-I2V-A14B-Diffusers"), "model_index.json"), "w").write("{}")
assert models._wan_source("Wan-AI/Wan2.2-I2V-A14B-Diffusers") == models.bf16_path("Wan-AI/Wan2.2-I2V-A14B-Diffusers")


# Prompt embeddings: a prompt is encoded once per pipeline and reused (variations, candidates, long takes)
class Emb:  # a tensor stand-in that remembers where it was moved
    def __init__(self, text, device="cuda"):
        self.text, self.device = text, device

    def to(self, device):
        return Emb(self.text, device)


class WanPipe:
    def __init__(self):
        self.encoded, self.calls = [], []

    def encode_prompt(self, prompt, negative_prompt, do_classifier_free_guidance, max_sequence_length):
        self.encoded.append((prompt, negative_prompt, do_classifier_free_guidance, max_sequence_length))
        return Emb(prompt), Emb(negative_prompt) if do_classifier_free_guidance else None

    def __call__(self, **kw):
        self.calls.append(kw)
        return mock.MagicMock(frames=[["frame"]])


wan = {"t2v": WanPipe(), "i2v": WanPipe()}
kept = (models._swap_to, models._apply_loras, models._generator, models.LIGHTNING)
models._swap_to = lambda name, loader: wan[name]
models._apply_loras = lambda pipe: pipe
models._generator = lambda seed: seed
models.LIGHTNING = True
for s in (1, 2):
    real_render_segment("a cat", "blurry", 81, 1280, 704, s, None)
assert wan["t2v"].encoded == [("a cat", "blurry", False, 512)], "same prompt, two seeds: one encode, at 512 tokens"
kw = wan["t2v"].calls[-1]
assert "prompt" not in kw and "negative_prompt" not in kw and kw["generator"] == 2
assert (kw["prompt_embeds"].text, kw["prompt_embeds"].device) == ("a cat", "cuda")
assert kw["negative_prompt_embeds"] is None, "Lightning runs without CFG: no negative embedding"
assert all(e.device == "cpu" for e, _ in models._embed_cache.values()), "cached off the GPU"
real_render_segment("a dog", "blurry", 81, 1280, 704, 1, None)
real_render_segment("a cat", "blurry", 81, 1280, 704, 1, Img(0))
assert len(wan["t2v"].encoded) == 2 and wan["i2v"].encoded == [("a cat", "blurry", False, 512)], \
    "a new prompt and the other pipeline are encoded for themselves"
assert "image" in wan["i2v"].calls[-1] and wan["i2v"].calls[-1]["prompt_embeds"].text == "a cat"
models.LIGHTNING = False  # the full model uses CFG: the negative is encoded too, under its own cache entry
real_render_segment("a cat", "blurry", 81, 1280, 704, 1, None)
assert wan["t2v"].encoded[-1] == ("a cat", "blurry", True, 512)
assert wan["t2v"].calls[-1]["negative_prompt_embeds"].text == "blurry" and wan["t2v"].calls[-1]["guidance_scale"] == 4.0
models.LIGHTNING = True
for i in range(models.EMBED_CACHE_SIZE + 4):  # bounded: the oldest prompts go
    real_render_segment(f"prompt {i}", "", 81, 1280, 704, 1, None)
assert len(models._embed_cache) == models.EMBED_CACHE_SIZE and ("t2v", "a cat", "blurry", False) not in models._embed_cache
# avatar candidates: one encode for all of them
sys.modules["numpy"].clip = lambda f, lo, hi: mock.MagicMock()
n_enc = len(wan["t2v"].encoded)
assert len(models.generate_avatar_candidates("a tall woman", 3, seed=5)) == 3
assert len(wan["t2v"].encoded) == n_enc + 1 and [c["generator"] for c in wan["t2v"].calls[-3:]] == [5, 6, 7]
assert "prompt" not in wan["t2v"].calls[-1] and wan["t2v"].calls[-1]["num_frames"] == 1
del sys.modules["numpy"].clip
models._swap_to, models._apply_loras, models._generator, models.LIGHTNING = kept
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
filter_devices = []
krea_gen.load_filter = lambda f, device=0: filter_devices.append(device) or (
    lambda im: [{"label": "nsfw", "score": 0.9 if "bad" in im.name else 0.02}])
sys.modules["PIL"].Image.open = lambda path: types.SimpleNamespace(name=os.path.basename(path), convert=lambda m: types.SimpleNamespace(name=os.path.basename(path)))
kd3 = tempfile.mkdtemp()
sreq = {"mode": "score", "filter_dir": kd3, "result": os.path.join(kd3, "s.json"),
        "images": [os.path.join(kd3, n) for n in ("ok1.png", "bad2.png", "ok3.png")]}
open(os.path.join(kd3, "sreq.json"), "w").write(_json.dumps(sreq))
krea_gen.main(os.path.join(kd3, "sreq.json"))
assert _json.load(open(sreq["result"]))["scores"] == [0.02, 0.9, 0.02] and not loaded_krea
assert filter_devices == ["cpu"], "score mode keeps off the GPU: the worker's models stay loaded beside it"
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
released.clear()
scores = krea.score_images(["/a.png", "/b.png", "/c.png"])
assert scores == [0.0, 0.1, 0.2] and weights_asked == ["filter"], (scores, weights_asked)
assert not released, "the filter runs on the CPU: scoring must not unload the worker's models"
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

# story: with burned-in subtitles the whole story is encoded again, so its shots are quick intermediates
takes.clear()
models.plan_story = lambda story, max_shots, style, cast: {"title": "T", "visual_bible": "port.", "shots": [
    {"prompt": "a pier", "narration": "One.", "characters": []}, {"prompt": "a boat", "narration": "Two.", "characters": []}]}
story = dict(story="s", max_shots=2, seconds_per_shot=5, sound=False, narration=False)
worker.run_story({"client": "m", "job_id": "st"}, dict(story), tempfile.mkdtemp())
assert [t["x264"] for t in takes] == [models.X264_INTERMEDIATE] * 2 and models.X264_INTERMEDIATE != models.X264_ARGS
takes.clear()
worker.run_story({"client": "m", "job_id": "st"}, dict(story, subtitles=False), tempfile.mkdtemp())
assert [t["x264"] for t in takes] == [None, None], "no subtitles: the shots are the final encode"

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
# one-person check: the VLM's answer parsed to a count
real_chat_cp = models._chat
for answer, want in [("1", 1), ("There are 2 people.", 2), ("One.", 1), ("two", 2), ("none", 0), ("no one", 0),
                     ("A single person.", 1), ("I can't tell", -1)]:
    models._chat = lambda messages, max_new_tokens=1024, a=answer, **k: a
    assert models.count_people("/x.png") == want, (answer, models.count_people("/x.png"))
asked_cp = []
models._chat = lambda messages, max_new_tokens=1024, **k: asked_cp.append(messages) or "1"
models.count_people("/p/x.png")
assert asked_cp[0][0]["content"][0] == {"type": "image", "image": "file:///p/x.png"}
assert "how many people" in asked_cp[0][0]["content"][1]["text"].lower()
models._chat = real_chat_cp
people = {}  # photo basename -> people count (default 1)
models.count_people = lambda path: people.get(os.path.basename(path), 1)
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
clip_calls = []


def fake_clip(image, prompt, out_path, seed=None, width=832, height=832):
    clip_calls.append((os.path.basename(image), prompt, seed, width, height,
                       len([j for j in put_jobs if j.get("type") == "train"])))
    open(out_path, "w").write("mp4")
    stills = [f"{out_path}.{k}.png" for k in range(3)]
    for sp in stills:
        open(sp, "w").write("png")
    return stills


models.render_clip = fake_clip
# krea_lora false: the photo set for the avatar's media only; no Krea dataset, no train job, no trainer needed
krea.score_images = lambda paths, progress=None: [0.01] * len(paths)
media_doc = dict(avatar_doc, krea_lora={"state": "skipped"}, media={"state": "queued", "job_id": "pm", "created_at": 77})
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(media_doc).encode()
n_jobs, n_msgs = len(put_jobs), len(sent_msgs)
fs3.uploaded.clear(), made.clear()
mparams = {"avatar": "ava", "trigger": "ava", "count": 24, "avatar_created_at": 77, "krea_lora": False, "media_only": False}
r = worker.run_avatar_photoset({"client": "m", "job_id": "pm"}, dict(mparams), tempfile.mkdtemp())
assert len(made) == 24 and r["kept"] == 24, r
assert len(put_jobs) == n_jobs and len(sent_msgs) == n_msgs and not any(k.startswith("datasets/") for k in fs3.uploaded)
assert _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["krea_lora"] == {"state": "skipped"}
# a stale media job (the avatar got a newer one) does nothing
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(media_doc, media=dict(media_doc["media"], job_id="newer"))).encode()
made.clear()
r = worker.run_avatar_photoset({"client": "m", "job_id": "pm"}, dict(mparams), tempfile.mkdtemp())
assert "skipped" in r and not made, r
# clips: four 2 s clips after the train job is queued, content-filtered, stored with the photos as media
flag_clip = {"clip_2"}  # its middle frame is flagged
krea.score_images = lambda paths, progress=None: [
    0.9 if any(c in p for c in flag_clip) and p.endswith(".1.png") else 0.01 for p in paths]
kdoc = dict(avatar_doc, krea_lora=dict(avatar_doc["krea_lora"], job_ids=["pk"]),
            media={"state": "queued", "job_id": "pk", "created_at": 77})
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(kdoc).encode()
fs3.uploaded.clear(), clip_calls.clear()
worker.TRAIN_QUEUE_URL = "tq"
n_train = len([j for j in put_jobs if j.get("type") == "train"])
r = worker.run_avatar_photoset({"client": "m", "job_id": "pk"}, dict(pparams, version=5), tempfile.mkdtemp())
assert len(clip_calls) == 4 and all(c[5] == n_train + 1 for c in clip_calls), "the train job is queued before the clips"
assert [c[0] for c in clip_calls] == ["ava_01.png", "ava_07.png", "ava_13.png", "ava_19.png"], clip_calls
assert all(c[1].startswith("ava ") and m in c[1] for c, m in zip(clip_calls, _photos.AVATAR_CLIP_MOTIONS))
assert len({c[2] for c in clip_calls}) == 4 and all(c[3:5] == (832, 832) for c in clip_calls)
media_keys = sorted(k for k in fs3.uploaded if k.startswith("avatars/m/ava/media/"))
assert media_keys == sorted([f"avatars/m/ava/media/photo_{i:02d}.png" for i in range(1, 25)] +
                            [f"avatars/m/ava/media/clip_{i}.mp4" for i in (1, 2, 3)]), media_keys
mj = _json.loads(fs3.objects["avatars/m/ava/media/media.json"])
assert len(mj["clips"]) == 3 and len(mj["photos"]) == 24 and mj["clips"][0]["file"] == "clip_1.mp4"
av = _json.loads(fs3.objects["avatars/m/ava/avatar.json"])
assert av["media"]["state"] == "ready" and av["media"]["photos"] == 24 and av["media"]["clips"] == 3
assert av["media"]["preview"] == "clip_1.mp4" and av["krea_lora"]["state"] == "training", av
assert r["clips"] == 3 and r["clips_filtered"] == 1, r
# every clip flagged: media failed, the Krea build unaffected and the job not failed
flag_clip = {"clip_1", "clip_2", "clip_3", "clip_4"}
clip_calls.clear()
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(kdoc, krea_lora=dict(kdoc["krea_lora"], job_ids=["pk2"]),
                                                            media=dict(kdoc["media"], job_id="pk2"))).encode()
fs3.uploaded.clear()
r = worker.run_avatar_photoset({"client": "m", "job_id": "pk2"}, dict(pparams, version=6), tempfile.mkdtemp())
av = _json.loads(fs3.objects["avatars/m/ava/avatar.json"])
assert av["media"]["state"] == "failed" and "content filter" in av["media"]["error"] and av["krea_lora"]["state"] == "training"
assert len([k for k in fs3.uploaded if "/media/photo_" in k]) == 24, "photos are stored even when every clip is flagged"
# the clips are filtered in one call after all four are rendered (one filter process, not four)
score_calls = []
real_score = krea.score_images
krea.score_images = lambda paths, progress=None: score_calls.append(list(paths)) or [0.01] * len(paths)
clip_calls.clear()
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(kdoc, krea_lora=dict(kdoc["krea_lora"], job_ids=["pk5"]),
                                                            media=dict(kdoc["media"], job_id="pk5"))).encode()
worker.run_avatar_photoset({"client": "m", "job_id": "pk5"}, dict(pparams, version=8), tempfile.mkdtemp())
assert [len(c) for c in score_calls] == [24, 12], [len(c) for c in score_calls]
krea.score_images = real_score
# a photo-set failure on the Krea path fails the media too (it must not stay busy and block POST /media)
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(kdoc, krea_lora=dict(kdoc["krea_lora"], job_ids=["pk6"]),
                                                            media=dict(kdoc["media"], job_id="pk6"))).encode()
real_mp = models.make_photo
models.make_photo = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("edit model crashed"))
_stubs.expect_error(worker.run_avatar_photoset, {"client": "m", "job_id": "pk6"}, dict(pparams, version=9),
                    tempfile.mkdtemp(), exc=RuntimeError)
models.make_photo = real_mp
av = _json.loads(fs3.objects["avatars/m/ava/avatar.json"])
assert av["media"]["state"] == "failed" and av["krea_lora"]["state"] == "failed", av
worker.TRAIN_QUEUE_URL = ""
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(kdoc, krea_lora=dict(kdoc["krea_lora"], job_ids=["pk7"]),
                                                            media=dict(kdoc["media"], job_id="pk7"))).encode()
_stubs.expect_error(worker.run_avatar_photoset, {"client": "m", "job_id": "pk7"}, dict(pparams, version=9),
                    tempfile.mkdtemp(), exc=RuntimeError)
assert _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["media"]["state"] == "failed", "no trainer: media failed too"
worker.TRAIN_QUEUE_URL = "tq"
# a cancelled job is cancelled, not "done with media failed"
def cancel_clip(*a, **k):
    raise models.JobCancelled("cancelled")
models.render_clip = cancel_clip
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(kdoc, krea_lora=dict(kdoc["krea_lora"], job_ids=["pk8"]),
                                                            media=dict(kdoc["media"], job_id="pk8"))).encode()
_stubs.expect_error(worker.run_avatar_photoset, {"client": "m", "job_id": "pk8"}, dict(pparams, version=10),
                    tempfile.mkdtemp(), exc=models.JobCancelled)
models.render_clip = fake_clip
# media only (krea_lora false) also gets its clips
flag_clip = set()
clip_calls.clear()
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(media_doc, media=dict(media_doc["media"], job_id="pm2"))).encode()
r = worker.run_avatar_photoset({"client": "m", "job_id": "pm2"}, dict(mparams), tempfile.mkdtemp())
assert len(clip_calls) == 4 and _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["media"]["state"] == "ready"
# Wan LoRA: after the media, the kept photos + clips become a Wan dataset and one Wan train job is queued
wparams = dict(pparams, version=30, wan_lora=True, wan_lora_name="avatar-ava-wan", wan_version=1)
wdoc = dict(kdoc, krea_lora=dict(kdoc["krea_lora"], job_ids=["pw1"]), media=dict(kdoc["media"], job_id="pw1"),
            wan_lora={"state": "queued", "lora": "avatar-ava-wan", "version": 1, "trigger": "ava", "job_ids": ["pw1"]})
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(wdoc).encode()
fs3.uploaded.clear(), clip_calls.clear()
n_put = len(put_jobs)
r = worker.run_avatar_photoset({"client": "m", "job_id": "pw1"}, dict(wparams), tempfile.mkdtemp())
wm = _json.loads(fs3.objects["datasets/m/avatar-ava-wan/v1/manifest.json"])
assert wm["file_count"] == 28 and wm["kind"] == "character" and wm["source"] == "avatar" and wm["trigger"] == "ava", wm
assert [f["path"] for f in wm["files"]][:2] == ["p_01.png", "p_02.png"] and wm["files"][24]["path"] == "c_1.mp4"
assert "datasets/m/avatar-ava-wan/v1/raw/c_4.mp4" in fs3.uploaded and "datasets/m/avatar-ava-wan/v1/raw/p_24.png" in fs3.uploaded
wc = _json.loads(fs3.objects["datasets/m/avatar-ava-wan/v1/captions.json"])
assert wc["items"]["p0001.png"].startswith("ava, ") and wc["items"]["s0025_00.mp4"].startswith("ava, ")
assert _photos.AVATAR_CLIP_MOTIONS[0] in wc["items"]["s0025_00.mp4"] and len(wc["items"]) == 28
new = put_jobs[n_put:]
assert [j.get("type") for j in new if j.get("type") == "train"] == ["train", "train"], "Krea train, then Wan train"
wj = [j for j in new if j.get("type") == "train"][-1]
wp = _json.loads(wj["params"])
assert wp["model"] == "t2v" and wp["lora"] == "avatar-ava-wan" and wp["version"] == 1 and wp["steps"] == 1000
assert wp["dataset"] == "avatar-ava-wan/v1" and wp["avatar"] == "ava" and wp["trigger"] == "ava" and wp["kind"] == "character"
assert wp["fingerprint"] == wc["fingerprint"] and wp["files"] == 28 and wp["speed"] == "standard"
assert sent_msgs[-1] == ("tq", _json.dumps({"job_id": wj["job_id"]})) and r["wan_train_job_id"] == wj["job_id"]
wl = _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["wan_lora"]
assert wl["state"] == "training" and wj["job_id"] in wl["job_ids"], wl
# redelivered (both train jobs exist): no second Wan train job; re-sent while still queued
n_train_all = len([j for j in put_jobs if j.get("type") == "train"])
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(wdoc).encode()
r = worker.run_avatar_photoset({"client": "m", "job_id": "pw1"}, dict(wparams), tempfile.mkdtemp())
assert len([j for j in put_jobs if j.get("type") == "train"]) == n_train_all, "no duplicate train jobs"
assert sent_msgs[-1] == ("tq", _json.dumps({"job_id": wj["job_id"]}))
assert _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["wan_lora"]["state"] == "training"
# wan_lora false: no Wan dataset or train job
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(wdoc, krea_lora=dict(kdoc["krea_lora"], job_ids=["pw2"]),
                                                            media=dict(kdoc["media"], job_id="pw2"))).encode()
n_train_all = len([j for j in put_jobs if j.get("type") == "train"])
worker.run_avatar_photoset({"client": "m", "job_id": "pw2"}, dict(pparams, version=31, wan_lora=False), tempfile.mkdtemp())
assert len([j for j in put_jobs if j.get("type") == "train"]) == n_train_all + 1, "only the Krea train job"
# a Wan dataset failure marks the Wan LoRA failed, the Krea build and media unaffected, the job not failed
real_up = fs3.upload_file
fs3.upload_file = lambda path, B, K, ExtraArgs=None: (_ for _ in ()).throw(OSError("disk")) if "-wan/" in K else real_up(path, B, K, ExtraArgs)
wdoc3 = dict(wdoc, krea_lora=dict(kdoc["krea_lora"], job_ids=["pw3"]), media=dict(kdoc["media"], job_id="pw3"),
             wan_lora=dict(wdoc["wan_lora"], job_ids=["pw3"], version=2))
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(wdoc3).encode()
r = worker.run_avatar_photoset({"client": "m", "job_id": "pw3"}, dict(wparams, version=32, wan_version=2), tempfile.mkdtemp())
fs3.upload_file = real_up
av = _json.loads(fs3.objects["avatars/m/ava/avatar.json"])
assert av["wan_lora"]["state"] == "failed" and av["media"]["state"] == "ready" and av["krea_lora"]["state"] == "training", av
assert "wan_error" in r
# no Krea LoRA (krea_lora false) but a Wan one: the media-only path builds it too
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(media_doc, media=dict(media_doc["media"], job_id="pw4"),
    wan_lora={"state": "queued", "lora": "avatar-ava-wan", "version": 3, "trigger": "ava", "job_ids": ["pw4"]})).encode()
r = worker.run_avatar_photoset({"client": "m", "job_id": "pw4"}, dict(mparams, wan_lora=True, wan_lora_name="avatar-ava-wan",
                                                                       wan_version=3), tempfile.mkdtemp())
assert "datasets/m/avatar-ava-wan/v3/manifest.json" in fs3.objects and r.get("wan_train_job_id"), r
# on demand (wan_only): the Wan dataset comes from the avatar's stored media; no new photos or clips
fs3.objects["avatars/m/ava/media/media.json"] = _json.dumps({
    "photos": [{"file": f"photo_{i:02d}.png", "scene": f"scene {i}"} for i in range(1, 21)],
    "clips": [{"file": f"clip_{i}.mp4", "motion": _photos.AVATAR_CLIP_MOTIONS[i - 1]} for i in (1, 2, 3)]}).encode()
people["m_photo_03.png"] = 2
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(media_doc, media={"state": "ready", "job_id": "old"},
    wan_lora={"state": "queued", "lora": "avatar-ava-wan", "version": 4, "trigger": "ava", "job_ids": ["po1"]})).encode()
made.clear(), clip_calls.clear()
oparams = {"avatar": "ava", "trigger": "ava", "avatar_created_at": 77, "krea_lora": False, "wan_only": True,
           "wan_lora": True, "wan_lora_name": "avatar-ava-wan", "wan_version": 4}
r = worker.run_avatar_photoset({"client": "m", "job_id": "po1"}, dict(oparams), tempfile.mkdtemp())
om = _json.loads(fs3.objects["datasets/m/avatar-ava-wan/v4/manifest.json"])
assert not made and not clip_calls and om["file_count"] == 19 + 3 and r.get("wan_train_job_id"), (r, om["file_count"])
oc = _json.loads(fs3.objects["datasets/m/avatar-ava-wan/v4/captions.json"])
assert oc["items"]["p0001.png"] == "ava, scene 1" and "s0020_00.mp4" in oc["items"]
assert _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["media"] == {"state": "ready", "job_id": "old"}, "media untouched"
people.clear()
# a redelivered build whose media is done but whose Wan build never started builds it from the stored media
made.clear(), clip_calls.clear()
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(kdoc, krea_lora=dict(kdoc["krea_lora"], job_ids=["pw1"]),
    media=dict(kdoc["media"], job_id="pw1", state="ready"),
    wan_lora={"state": "queued", "lora": "avatar-ava-wan", "version": 6, "trigger": "ava", "job_ids": ["pw1"]})).encode()
gone = job_items.pop(worker._wan_train_job_id("pw1"))  # the Krea train job exists, the Wan one never got written
existing_ids.discard(gone["job_id"])
r = worker.run_avatar_photoset({"client": "m", "job_id": "pw1"}, dict(wparams, wan_version=6), tempfile.mkdtemp())
assert r.get("wan_train_job_id") == gone["job_id"] and not made and not clip_calls, r
assert _json.loads(fs3.objects["datasets/m/avatar-ava-wan/v6/manifest.json"])["file_count"] == 20 + 3
# no stored media: the Wan build fails with the reason
del fs3.objects["avatars/m/ava/media/media.json"]
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(media_doc, wan_lora={"state": "queued", "job_ids": ["po2"]})).encode()
_stubs.expect_error(worker.run_avatar_photoset, {"client": "m", "job_id": "po2"}, dict(oparams, wan_version=5),
                    tempfile.mkdtemp(), contains="media", exc=RuntimeError)
assert _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["wan_lora"]["state"] == "failed"
# one-person check: photos that don't show exactly one person are dropped before any dataset or media
people.update({f"ava_{i:02d}.png": 2 for i in range(1, 7)})  # 6 photos with two people
clip_calls.clear(), fs3.uploaded.clear()
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(kdoc, krea_lora=dict(kdoc["krea_lora"], job_ids=["pp1"]),
                                                            media=dict(kdoc["media"], job_id="pp1"))).encode()
r = worker.run_avatar_photoset({"client": "m", "job_id": "pp1"}, dict(pparams, version=20), tempfile.mkdtemp())
assert r["kept"] == 18 and r["people_dropped"] == 6, r
ds = _json.loads(fs3.objects["datasets/m/avatar-ava/v20/manifest.json"])
assert ds["file_count"] == 18 and "ava_01.png" not in [f["path"] for f in ds["files"]]
assert len([k for k in fs3.uploaded if "/media/photo_" in k]) == 18
people.update({f"ava_{i:02d}.png": 2 for i in range(1, 13)})  # 12 of 24: too few left
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(kdoc, krea_lora=dict(kdoc["krea_lora"], job_ids=["pp2"]),
                                                            media=dict(kdoc["media"], job_id="pp2"),
                                                            wan_lora={"state": "queued", "job_ids": ["pp2"]})).encode()
_stubs.expect_error(worker.run_avatar_photoset, {"client": "m", "job_id": "pp2"}, dict(pparams, version=21),
                    tempfile.mkdtemp(), contains="exactly one person", exc=krea.FilteredOut)
assert "datasets/m/avatar-ava/v21/manifest.json" not in fs3.objects
av = _json.loads(fs3.objects["avatars/m/ava/avatar.json"])
assert av["krea_lora"]["state"] == "failed" and av["media"]["state"] == "failed"
assert av["wan_lora"]["state"] == "failed", "the Wan build is released too (not stuck queued)"
# the same on the media-only path (no Krea LoRA)
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(media_doc, media=dict(media_doc["media"], job_id="pp2m"),
                                                            wan_lora={"state": "queued", "job_ids": ["pp2m"]})).encode()
_stubs.expect_error(worker.run_avatar_photoset, {"client": "m", "job_id": "pp2m"}, dict(mparams, wan_lora=True,
                    wan_lora_name="avatar-ava-wan", wan_version=9), tempfile.mkdtemp(), exc=krea.FilteredOut)
assert _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["wan_lora"]["state"] == "failed"
people.clear()
# a clip whose middle frame shows two people is dropped
people["clip_2.mp4.1.png"] = 2
clip_calls.clear()
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(kdoc, krea_lora=dict(kdoc["krea_lora"], job_ids=["pp3"]),
                                                            media=dict(kdoc["media"], job_id="pp3"))).encode()
r = worker.run_avatar_photoset({"client": "m", "job_id": "pp3"}, dict(pparams, version=22), tempfile.mkdtemp())
assert r["clips"] == 3 and r["clips_filtered"] == 1, r
people.clear()
# media for an existing avatar with a ready Krea LoRA: its dataset photos are reused, no new photos
made.clear(), clip_calls.clear(), fs3.uploaded.clear()
n_train_before = len([j for j in put_jobs if j.get("type") == "train"])
ready_doc = dict(media_doc, krea_lora={"state": "ready", "ready": {"lora": "avatar-ava", "version": 1}},
                 media={"state": "queued", "job_id": "pm3", "created_at": 77})
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(ready_doc).encode()
r = worker.run_avatar_photoset({"client": "m", "job_id": "pm3"}, dict(mparams, media_only=True), tempfile.mkdtemp())
n_ds = len(_json.loads(fs3.objects["datasets/m/avatar-ava/v1/manifest.json"])["files"])
assert not made and r["kept"] == n_ds and len(clip_calls) == 4, r
mj = _json.loads(fs3.objects["avatars/m/ava/media/media.json"])
assert len(mj["photos"]) == n_ds and mj["photos"][0]["scene"] and not mj["photos"][0]["scene"].startswith("ava, ")
# reused dataset photos go through the one-person check too
people["ds_01_ava_01.png"] = 3
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(ready_doc, media=dict(ready_doc["media"], job_id="pm3b"))).encode()
r = worker.run_avatar_photoset({"client": "m", "job_id": "pm3b"}, dict(mparams, media_only=True), tempfile.mkdtemp())
assert r["kept"] == n_ds - 1 and r["people_dropped"] == 1, r
people.clear()
assert len([j for j in put_jobs if j.get("type") == "train"]) == n_train_before, "media only: no train job"
# a replaced avatar (Krea state skipped, old ready link kept) does not reuse the old person's photos
made.clear(), clip_calls.clear()
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(ready_doc, krea_lora=dict(ready_doc["krea_lora"], state="skipped"),
                                                            media=dict(ready_doc["media"], job_id="pm5"))).encode()
worker.run_avatar_photoset({"client": "m", "job_id": "pm5"}, dict(mparams, media_only=True), tempfile.mkdtemp())
assert len(made) == 24, "photos made for the current avatar"
# an unreadable dataset falls back to new photos
made.clear()
real_dl_w = fs3.download_file
fs3.download_file = lambda b, k, p: (_ for _ in ()).throw(FakeClientError({"Error": {"Code": "404"}}, "GetObject"))
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(ready_doc, media=dict(ready_doc["media"], job_id="pm6"))).encode()
worker.run_avatar_photoset({"client": "m", "job_id": "pm6"}, dict(mparams, media_only=True), tempfile.mkdtemp())
fs3.download_file = real_dl_w
assert len(made) == 24
# media for an avatar without a Krea dataset: photos are made
clip_calls.clear(), made.clear()
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(ready_doc, krea_lora={"state": "skipped"},
                                                            media=dict(ready_doc["media"], job_id="pm4"))).encode()
worker.run_avatar_photoset({"client": "m", "job_id": "pm4"}, dict(mparams, media_only=True), tempfile.mkdtemp())
assert len(made) == 24 and len(clip_calls) == 4
# a redelivered Krea build whose train job exists still makes the media, from its own dataset's photos
made.clear(), clip_calls.clear()
redo = dict(kdoc, krea_lora=dict(kdoc["krea_lora"], job_ids=["pk"]), media=dict(kdoc["media"], job_id="pk", state="clips"))
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(redo).encode()
n_jobs = len(put_jobs)
r = worker.run_avatar_photoset({"client": "m", "job_id": "pk"}, dict(pparams, version=5), tempfile.mkdtemp())
assert not made and len(clip_calls) == 4 and len(put_jobs) == n_jobs, r
assert _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["media"]["state"] == "ready"
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
# ---------------------------------------------------------------- process: the AI-look check after the render
done_results = []
worker.set_status = lambda job_id, status, **kw: statuses.append(status) or (
    done_results.append(_json.loads(kw["result"])) if status == "done" else None)
checks = []
real_check = worker.realism_check.check
worker.realism_check.check = lambda job_type, params, result, workdir, download, judge, progress, screen: checks.append(
    (job_type, sorted(result))) or {"score": 4, "signs": ["x"], "checked": job_type}
worker.RUNNERS["photos"] = lambda job, params, workdir: {"photo_1_key": "k"}
worker.RUNNERS["write"] = lambda job, params, workdir: {"story_key": "s"}
for jid, jtype, params in [("rc1", "photos", {"realism_check": True}), ("rc2", "photos", {}),
                           ("rc3", "write", {"realism_check": True})]:
    job_items[jid] = {"job_id": jid, "status": "queued", "type": jtype, "client": "m", "params": _json.dumps(params)}
    worker.process(jid, "r")
assert checks == [("photos", ["photo_1_key"])], "only flagged photos/video/story jobs are checked"
assert done_results[-3]["realism"]["score"] == 4 and "realism" not in done_results[-2] and "realism" not in done_results[-1]
worker.realism_check.check = real_check  # the real one: a failing judge still leaves the job done
real_score_images = worker.krea.score_images
screened = []
worker.krea.score_images = lambda paths, progress=None: screened.append(list(paths)) or [0.0] * len(paths)
worker.models.judge_realism = lambda images, text: "not json"
worker.s3 = mock.MagicMock(download_file=lambda b, k, p: open(p, "wb").write(b"x"))
job_items["rc4"] = {"job_id": "rc4", "status": "queued", "type": "photos", "client": "m",
                    "params": _json.dumps({"realism_check": True})}
worker.process("rc4", "r")
assert statuses[-1] == "done" and done_results[-1]["realism"]["error"] == "the check couldn't read the model's answer"
assert screened and screened[-1][0].endswith("photo_1.png"), "the hook runs the content filter on the outputs"
worker.krea.score_images = lambda paths, progress=None: [0.99] * len(paths)  # flagged: skipped, model not asked
worker.models.judge_realism = lambda images, text: (_ for _ in ()).throw(AssertionError("asked about a flagged image"))
job_items["rc5"] = dict(job_items["rc4"], job_id="rc5")
worker.process("rc5", "r")
assert statuses[-1] == "done" and done_results[-1]["realism"]["skipped"].startswith("skipped:"), done_results[-1]
worker.krea.score_images = real_score_images
worker.RUNNERS.update(real_runners)
assert worker.take_work_dir("abc") == os.path.join(worker.WORK_ROOT, "abc")
# a photo-set job cancelled before it (re)starts releases the avatar's Krea build and media (no stuck "photos")
worker.s3 = fs3
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(
    avatar_doc, krea_lora=dict(avatar_doc["krea_lora"], state="photos", job_ids=["pc"]),
    media={"state": "photos", "job_id": "pc", "created_at": 77})).encode()
job_items["pc"] = {"job_id": "pc", "status": "running", "type": "avatar_photoset", "client": "m",
                   "cancel_requested": True, "params": _json.dumps(dict(pparams))}
worker.process("pc", "r")
av = _json.loads(fs3.objects["avatars/m/ava/avatar.json"])
assert av["krea_lora"]["state"] == "failed" and av["krea_lora"]["error"] == "cancelled", av["krea_lora"]
assert av["media"]["state"] == "failed" and av["media"]["error"] == "cancelled"
fs3.objects["avatars/m/ava/avatar.json"] = _json.dumps(dict(avatar_doc, wan_lora={"state": "queued", "job_ids": ["pc2"]})).encode()
job_items["pc2"] = dict(job_items["pc"], job_id="pc2")
worker.process("pc2", "r")
assert _json.loads(fs3.objects["avatars/m/ava/avatar.json"])["wan_lora"]["state"] == "failed", "cancel releases Wan"
import time as _time  # noqa: E402
os.makedirs(worker.take_work_dir("old")), os.makedirs(worker.take_work_dir("new"))
os.utime(worker.take_work_dir("old"), (_time.time() - 4 * 86400,) * 2)
worker.sweep_work_dirs()
assert not os.path.exists(worker.take_work_dir("old")) and os.path.exists(worker.take_work_dir("new"))
worker.WORK_ROOT = os.path.join(tempfile.mkdtemp(), "missing")
worker.sweep_work_dirs()  # no folder yet: nothing to do

# An idle worker loads the big models that are not in RAM yet, one per empty poll, where all three fit
kept = (models._swap_to, models.RESIDENT_LIMIT, models.PRELOAD_MODELS, dict(models._loaded))
swaps = []
models._swap_to = lambda name, loader: (swaps.append((name, loader)), models._loaded.update({name: object()}))
models._loaded.clear()
models.RESIDENT_LIMIT = 1
assert models.preload_next() is None and not swaps, "a box that holds one model: nothing is loaded ahead"
models.RESIDENT_LIMIT = 3
models._loaded["i2v"] = object()
assert [models.preload_next() for _ in range(3)] == ["t2v", "edit", None], "one per call, loaded ones skipped"
assert swaps == [("t2v", models._load_t2v), ("edit", models._load_edit)]
models._loaded.clear()
models.PRELOAD_MODELS = False
assert models.preload_next() is None and len(swaps) == 2, "PRELOAD_MODELS=0: off"
models.PRELOAD_MODELS = True
logged = _io.StringIO()
with _contextlib.redirect_stdout(logged):
    for _ in range(5):
        worker.preload_idle()
events = [_jsont.loads(line) for line in logged.getvalue().splitlines()]
assert [(e["event"], e["model"]) for e in events] == [("preloaded", m) for m in ("t2v", "i2v", "edit")]
assert len(swaps) == 5 and not worker._preload["on"], "all loaded: the idle loop stops asking"


def no_room(name, loader):
    raise MemoryError("no room")


worker._preload["on"] = True
models._loaded.clear()
models._swap_to = no_room
logged = _io.StringIO()
with _contextlib.redirect_stdout(logged):
    worker.preload_idle(), worker.preload_idle()
events = [_jsont.loads(line) for line in logged.getvalue().splitlines()]
assert [e["event"] for e in events] == ["preload_failed"], "a failed load is logged once and not retried"
models._swap_to, models.RESIDENT_LIMIT, models.PRELOAD_MODELS = kept[:3]
models._loaded.clear(), models._loaded.update(kept[3])
print("worker tests passed")
