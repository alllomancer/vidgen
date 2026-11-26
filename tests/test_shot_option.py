"""The two shot options for videos: "stream" (a fixed webcam, ring light) and "pov" (a handheld phone seen
by the person holding it, phone light). One setting, so only one can be on; they add wording to the prompt.
API validation, worker wording and the CLI flag."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("worker", os.path.join("lambda", "api"), "cli")
import handler as h  # noqa: E402
import models  # noqa: E402
import realism  # noqa: E402
import vidgen  # noqa: E402

# ---------------------------------------------------------------- worker: the wording
assert tuple(realism.SHOTS) == ("stream", "pov")
stream, pov = realism.style_text({"shot": "stream"}), realism.style_text({"shot": "pov"})
for word in ("webcam", "never moves", "ring light"):
    assert word in stream, (word, stream)
for word in ("handheld", "point of view", "phone", "camera light", "moves"):
    assert word in pov, (word, pov)
assert stream != pov
# no option, or one the worker doesn't know (an older job): the prompt is unchanged
assert realism.with_style("a fox", {}) == "a fox" and realism.with_style("a fox", {"shot": ""}) == "a fox"
assert realism.with_style("a fox", {"shot": "drone"}) == "a fox"
# it combines with the look and the other switches, shot first
both = realism.style_text({"shot": "stream", "look": "film", "natural_skin": True})
assert both.startswith(realism.SHOTS["stream"]) and realism.VIDEO_LOOKS["film"] in both and realism.NATURAL_SKIN in both
assert realism.with_style("she waves", {"shot": "pov"}).startswith("she waves ")
# the phone option replaces the default tripod sentence (a handheld camera); the webcam keeps it
p, neg = models.natural_prompts(realism.with_style("she waves", {"shot": "pov"}))
assert models.TRIPOD not in p and models.TRIPOD_NEGATIVE not in neg, p
p, neg = models.natural_prompts(realism.with_style("she waves", {"shot": "stream"}))
assert p.startswith(models.TRIPOD) and models.TRIPOD_NEGATIVE in neg, p
p, _ = models.natural_prompts(realism.with_style("she waves", {}))
assert p.startswith(models.TRIPOD)

# ---------------------------------------------------------------- API: one validated setting, videos only
_, v = h._validate("m", {"type": "video", "video_model": "wan", "prompt": "x", "shot": "stream"})
assert v["shot"] == "stream"
_, v = h._validate("m", {"type": "video", "video_model": "wan", "prompt": "x", "shot": "pov", "look": "film"})
assert v["shot"] == "pov" and v["look"] == "film"
for none in ({}, {"shot": None}, {"shot": ""}):
    _, v = h._validate("m", {"type": "video", "video_model": "wan", "prompt": "x", **none})
    assert "shot" not in v, v
for bad in ("both", "stream,pov", ["stream", "pov"], True, "phone"):
    _stubs.expect_error(h._validate, "m", {"type": "video", "video_model": "wan", "prompt": "x", "shot": bad}, contains="shot")
# a plan carries it, so the first frame and the render match
_, v = h._validate("m", {"type": "video", "video_model": "wan", "prompt": "x", "seconds": 30, "plan_only": True, "shot": "pov"})
assert v["shot"] == "pov" and v["plan_only"] is True
# stories and photos don't take it (it would be ignored silently otherwise)
_stubs.expect_error(h._validate, "m", {"type": "story", "story": "once", "shot": "stream"}, contains="shot")
_stubs.expect_error(h._validate_photos, "m", {"avatar": "none", "model": "krea2", "scenes": ["a pier"], "shot": "pov"},
                    contains="shot")
_, st = h._validate("m", {"type": "story", "story": "once"})
assert "shot" not in st
# History shows it
assert h._summary("video", {"prompt": "x", "seconds": 5, "shot": "stream"})[1]["shot"] == "stream"
assert "shot" not in h._summary("video", {"prompt": "x", "seconds": 5})[1]

# ---------------------------------------------------------------- CLI: --shot stream|pov on video
parser = argparse.ArgumentParser()
vidgen.shot_flag(parser)
assert vidgen.shot_body(parser.parse_args([])) == {}
assert vidgen.shot_body(parser.parse_args(["--shot", "stream"])) == {"shot": "stream"}
assert vidgen.shot_body(parser.parse_args(["--shot", "pov"])) == {"shot": "pov"}
try:
    with open(os.devnull, "w") as devnull:
        sys.stderr = devnull
        parser.parse_args(["--shot", "both"])
    raise AssertionError("an unknown shot was accepted")
except SystemExit:
    pass
finally:
    sys.stderr = sys.__stderr__
assert vidgen.SHOTS == h.SHOTS == tuple(realism.SHOTS), "the three lists must match"
# a plan's "render with" command keeps the shot, so the render matches the planned first frame
ns = argparse.Namespace(prompt="a walk", seconds=30, no_avatar=True, shot="pov")
assert "--shot pov" in vidgen.plan_render_command(ns, 7, "walk.beats.txt")
ns.shot = None
assert "--shot" not in vidgen.plan_render_command(ns, 7, "walk.beats.txt")

print("shot option tests passed")
