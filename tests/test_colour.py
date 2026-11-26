"""Colour matching between long-take segments (needs numpy; skipped without it). Run: make test,
or with numpy: python3 -B tests/test_colour.py."""
import importlib.util
import os
import sys

if importlib.util.find_spec("numpy") is None:
    print("colour tests skipped: numpy is not installed here (the render box has it)")
    sys.exit(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install("worker", keep=("numpy",))
import numpy as np  # noqa: E402

import models  # noqa: E402

rng = np.random.default_rng(0)
ref = rng.integers(60, 200, size=(6, 32, 48, 3), dtype=np.uint8)

# RGB -> LAB -> RGB is lossless up to rounding
back = models._lab_to_rgb8(models._rgb8_to_lab(ref))
assert back.dtype == np.uint8 and np.abs(back.astype(int) - ref).max() <= 1

# equal statistics: frames come back unchanged
stats = models.colour_stats(ref)
steady = np.repeat(ref[:1], 6, axis=0)
assert models.match_colours(steady, models.colour_stats(steady)) is steady

# a darker, bluer segment: the correction ramps from none on frame 0 (it continues the already corrected
# conditioning frame, so no step at the seam) to COLOUR_STRENGTH on the last frame
shifted = np.clip(ref.astype(int) * [0.7, 0.75, 0.95] - 10, 0, 255).astype(np.uint8)
s_mean, s_std = models.colour_stats(shifted)
r_mean, r_std = stats
out = models.match_colours(shifted, stats)
assert out.shape == shifted.shape and out.dtype == np.uint8
threads, models.COLOUR_THREADS = models.COLOUR_THREADS, 1
assert np.array_equal(models.match_colours(shifted, stats), out), "frames in parallel: the same pixels as one at a time"
models.COLOUR_THREADS = threads
assert np.array_equal(out[0], shifted[0]), "frame 0 untouched: no step at the seam"


tail_mean, _ = models.colour_stats(shifted[-max(1, len(shifted) // 4):])  # the drift is measured at the end


def moved(i):
    m, _ = models.colour_stats(out[i:i + 1])
    m0, _ = models.colour_stats(shifted[i:i + 1])
    return (m - m0) / (r_mean - tail_mean)


last = len(out) - 1
assert np.all(np.abs(moved(last) - models.COLOUR_STRENGTH) < 0.1), moved(last)
mid = moved(last // 2 + 1) if last % 2 else moved(last // 2)
assert np.all(np.abs(mid - models.COLOUR_STRENGTH * (last // 2 + last % 2) / last) < 0.1), mid
assert models.match_colours(shifted, stats, strength=0) is shifted
# a near-flat segment (fade to black) is not blown up: contrast change clamped
flat = np.clip(10 + rng.integers(-1, 2, size=(6, 32, 48, 3)), 0, 255).astype(np.uint8)
f_out = models.match_colours(flat, stats)
assert f_out[-1].astype(float).std() <= flat[-1].astype(float).std() * models.COLOUR_MAX_GAIN * 1.3 + 1, \
    (f_out[-1].std(), flat[-1].std())
# transfer_colours: a re-anchored frame takes the colours of the frame it replaces (full strength, one frame)
edited = np.clip(ref[0].astype(int) * 1.2 + 15, 0, 255).astype(np.uint8)
target = models.colour_stats(ref[:1])
moved_frame = models.transfer_colours(edited, target)
mm, _ = models.colour_stats(moved_frame[None])
assert np.all(np.abs(mm - target[0]) < 1.5), (mm, target[0])
f0 = ref[0]
assert models.transfer_colours(f0, target) is f0, "same colours: unchanged"
# saved segments (resume) round-trip as real .npz files; other settings or a damaged file start clean
import tempfile  # noqa: E402
wd = tempfile.mkdtemp()
key = {"seed": 1, "total": 801}
models._save_progress(wd, key, 0, ref)
models._save_progress(wd, key, 1, shifted)
got = models._load_segments(wd, key)
assert len(got) == 2 and np.array_equal(got[0], ref) and np.array_equal(got[1], shifted)
assert not any(f.endswith(".tmp.npz") or f.endswith(".tmp") for f in os.listdir(wd))
open(os.path.join(wd, "segment_01.npz"), "wb").write(b"broken")
assert len(models._load_segments(wd, key)) == 1, "a damaged segment is rendered again"
assert models._load_segments(wd, dict(key, seed=2)) == [] and not any(f.startswith("segment_") for f in os.listdir(wd))
print("colour and segment-file tests passed")
