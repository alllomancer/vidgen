"""CLI side of the AI-look check: --realism-check on video/photos/story, and the watcher saving the verdict next to
the downloaded files. docs/specs/2026-11-27-realism-check.md"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("cli")
import vidgen  # noqa: E402

logged = []
vidgen.log = logged.append
import argparse  # noqa: E402

for what in ("video", "story", "photos"):  # the shared realism flags of all three commands
    parser = argparse.ArgumentParser()
    vidgen.realism_flags(parser, what)
    assert vidgen.realism_body(parser.parse_args(["--realism-check"]))["realism_check"] is True
    assert "realism_check" not in vidgen.realism_body(parser.parse_args([])), "absent unless asked"
src = open(vidgen.__file__).read()
assert src.count("realism_flags(") >= 4, "video, story and photos all take the realism flags"

tmp = tempfile.mkdtemp()
video = os.path.join(tmp, "boat.mp4")
open(video, "w").close()
verdict = {"score": 7, "signs": ["waxy skin", "too-perfect light"], "prompt": "p", "note": "n"}
path = vidgen.save_realism({"out": video}, {"result": {"realism": verdict}})
assert path == os.path.join(tmp, "boat.realism.json") and json.load(open(path)) == verdict
assert logged[-1] == "AI look 7/10: waxy skin; too-perfect light (suggestion in boat.realism.json)", logged
folder = os.path.join(tmp, "cafe")
os.makedirs(folder)
assert vidgen.save_realism({"out": folder}, {"result": {"realism": {"error": "the check failed: x"}}}) == \
    os.path.join(folder, "realism.json")
assert logged[-1] == "AI-look check: the check failed: x"
assert vidgen.save_realism({"out": video}, {"result": {}}) is None
assert vidgen.save_realism({"out": video}, {}) is None

print("test_realism_cli: ok")
