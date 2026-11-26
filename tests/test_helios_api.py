"""Helios as the default video model (docs/specs/2026-11-26-helios-default.md, criteria 1-5, 11): the API's
defaults and refusals, and the CLI's flag."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install(os.path.join("lambda", "api"), "cli")
import handler as h  # noqa: E402
import vidgen  # noqa: E402

assert h.VIDEO_MODELS == ("helios", "wan")

# ---- no model named: Helios, at its own size and frame rate, 30 s unless a length is sent (criterion 1)
_, v = h._validate("m", {"type": "video", "prompt": "x"})
assert (v["video_model"], v["width"], v["height"], v["fps"], v["seconds"]) == ("helios", 640, 384, 24, 30), v
# what an existing CLI sends (its Wan defaults) doesn't change that, except the length it asked for
_, v = h._validate("m", {"type": "video", "prompt": "x", "width": 1280, "height": 704, "fps": 32, "seconds": 12})
assert (v["video_model"], v["width"], v["height"], v["fps"], v["seconds"]) == ("helios", 640, 384, 24, 12), v
_, v = h._validate("m", {"type": "video", "prompt": "x", "video_model": "helios", "seconds": 60, "upscale": True})
assert v["seconds"] == 60 and v["upscale"] is True and (v["width"], v["height"]) == (640, 384)
# draft is Wan's quick mode; Helios renders at one size and speed
_, v = h._validate("m", {"type": "video", "prompt": "x", "quality": "draft"})
assert v["quality"] == "standard" and (v["width"], v["height"]) == (640, 384) and v["upscale"] is False, v
# several takes of a long clip are fine with Helios (each is one run)
_, v = h._validate("m", {"type": "video", "prompt": "x", "variations": 3})
assert v["variations"] == 3 and v["seconds"] == 30
# the look and shot settings still apply
_, v = h._validate("m", {"type": "video", "prompt": "x", "shot": "pov", "look": "film", "grain": True})
assert (v["shot"], v["look"], v["grain"]) == ("pov", "film", True)

# ---- Wan: exactly as before (criterion 2)
_, w = h._validate("m", {"type": "video", "prompt": "x", "video_model": "wan"})
assert (w["video_model"], w["width"], w["height"], w["fps"], w["seconds"]) == ("wan", 1280, 704, 32, 5), w
_, w = h._validate("m", {"type": "video", "prompt": "x", "video_model": "wan", "seconds": 50,
                         "beats": "0: walks in; 10: sits down", "anchor_every": 3})
assert len(w["beats"]) == 2 and w["anchor_every"] == 3
_stubs.expect_error(h._validate, "m", {"type": "video", "prompt": "x", "video_model": "wan", "variations": 2,
                                        "seconds": 10}, contains="variations")
_, w = h._validate("m", {"type": "video", "prompt": "x", "video_model": "wan", "quality": "draft"})
assert w["quality"] == "draft" and w["upscale"] is True

# ---- an unknown model, or the key on a story / photos job (criterion 3)
for bad in ("sora", "Helios", ["wan"], True):
    _stubs.expect_error(h._validate, "m", {"type": "video", "prompt": "x", "video_model": bad}, contains="video_model")
_stubs.expect_error(h._validate, "m", {"type": "story", "story": "once", "video_model": "helios"}, contains="video_model")
_stubs.expect_error(h._validate_photos, "m", {"avatar": "none", "model": "krea2", "scenes": ["a pier"],
                                               "video_model": "wan"}, contains="video_model")
_, st = h._validate("m", {"type": "story", "story": "once"})
assert "video_model" not in st, "stories stay on Wan"

# ---- Helios takes one prompt for one take: beats, plans and re-anchoring are Wan's (criterion 4)
for extra, word in (({"beats": "0: walks in; 10: sits down"}, "beats"), ({"plan_only": True, "seconds": 30}, "plan"),
                    ({"anchor_every": 3}, "anchor_every")):
    try:
        h._validate("m", {"type": "video", "prompt": "x", "seconds": 30, **extra})
        raise AssertionError(f"{word} was accepted for Helios")
    except ValueError as e:
        assert word in str(e) and "wan" in str(e).lower(), e

# ---- Helios takes the job's Wan LoRAs like Wan does (the worker's runner loads them all): no extra limit
assert not hasattr(h, "_check_helios_loras"), "the one-LoRA refusal is gone"

# ---- History shows the model
assert h._summary("video", {"prompt": "x", "seconds": 30, "video_model": "helios"})[1]["video_model"] == "helios"
assert h._summary("video", {"prompt": "x", "seconds": 5, "video_model": "wan"})[1]["video_model"] == "wan"
assert "video_model" not in h._summary("video", {"prompt": "x", "seconds": 5})[1], "jobs from before this change"

# ---- CLI: --video-model, and the length only when asked for (criterion 11)
parser = argparse.ArgumentParser()
vidgen.video_model_flag(parser)
parser.add_argument("--seconds", type=int)
assert vidgen.video_model_body(parser.parse_args([])) == {}
assert vidgen.video_model_body(parser.parse_args(["--video-model", "wan"])) == {"video_model": "wan"}
assert vidgen.video_model_body(parser.parse_args(["--video-model", "helios", "--seconds", "45"])) == {
    "video_model": "helios", "seconds": 45}
assert vidgen.video_model_body(parser.parse_args(["--seconds", "12"])) == {"seconds": 12}
assert vidgen.VIDEO_MODELS == h.VIDEO_MODELS
# planning writes beats, which are Wan's
assert vidgen.video_model_body(parser.parse_args([]), planning=True) == {"video_model": "wan"}

print("helios api tests passed")
