"""Pure planning logic for train jobs (no AWS, no GPU), so it can be tested anywhere:

  segment_plan     which 5 s pieces to cut from each source clip (at most 12 per source)
  resume_point     where training stands after any number of spot interruptions
  checkpoint_name  absolute step numbers for checkpoints saved by resumed runs
  next_version     the next LoRA version number for a theme

musubi-tuner restarts its step counter when it resumes from saved state (measured in the
spike), so every resumed run gets its own output name `<expert>_r<k>` and a recorded `base`
(steps already done before it). A state folder `<expert>_r<k>-step<rel>-state` therefore
stands for `base_k + rel` steps in total.
"""

import re

SEGMENT_SECONDS = 5.0
MAX_SEGMENTS_PER_SOURCE = 12
MIN_CLIP_SECONDS = 1.0
SAVE_EVERY = 250
EXPERTS = ("t2v_low", "t2v_high")
# Timestep ranges (of 1000) each expert is trained on; Wan2.2 T2V switches experts at 875.
TIMESTEPS = {"t2v_low": (0, 875), "t2v_high": (875, 1000)}

STATE_DIR = re.compile(r"^(?P<expert>t2v_low|t2v_high|krea2)_r(?P<run>\d+)-step(?P<step>\d+)-state$")
CHECKPOINT = re.compile(r"^(?P<expert>t2v_low|t2v_high|krea2)_r(?P<run>\d+)-step(?P<step>\d+)\.safetensors$")


def segment_plan(duration):
    """Start times (seconds) of the segments to cut from a clip of `duration` seconds.

    Clips up to one segment long are used whole ([0.0], cut length = duration); shorter than
    MIN_CLIP_SECONDS are skipped ([]). Longer clips give floor(duration / 5) candidate
    segments, of which at most MAX_SEGMENTS_PER_SOURCE are kept, spread evenly over the clip.
    """
    if duration is None or duration < MIN_CLIP_SECONDS:
        return []
    possible = int(duration // SEGMENT_SECONDS)
    if possible <= 1:
        return [0.0]
    keep = min(possible, MAX_SEGMENTS_PER_SOURCE)
    if keep == possible:
        return [i * SEGMENT_SECONDS for i in range(possible)]
    # Evenly spaced segment indices across all possible ones, first and last included.
    idx = [round(i * (possible - 1) / (keep - 1)) for i in range(keep)]
    return [i * SEGMENT_SECONDS for i in idx]


def resume_point(expert, runs, names):
    """(steps_done, state_dir_name or None) for `expert`.

    runs: [{"run": k, "base": steps done before run k}] recorded for this expert.
    names: entries of the output folder (state folders and checkpoint files).
    """
    base = {r["run"]: r["base"] for r in runs}
    best = (max(base.values(), default=0), None)
    for name in names:
        m = STATE_DIR.match(name)
        if not m or m["expert"] != expert or int(m["run"]) not in base:
            continue
        total = base[int(m["run"])] + int(m["step"])
        if total > best[0] or (total == best[0] and best[1] is None):
            best = (total, name)
    return best


def checkpoint_name(name, runs):
    """`t2v_low-step000750.safetensors` for a checkpoint file saved by any run, or None."""
    m = CHECKPOINT.match(name)
    base = {r["run"]: r["base"] for r in runs}
    if not m or int(m["run"]) not in base:
        return None
    return f"{m['expert']}-step{base[int(m['run'])] + int(m['step']):06d}.safetensors"


def next_version(existing_keys, prefix):
    """1 + the highest v<N> under prefix (keys like '<prefix>v3/lora.json')."""
    found = [int(m.group(1)) for k in existing_keys if (m := re.match(re.escape(prefix) + r"v(\d+)/", k))]
    return max(found, default=0) + 1


def eta_seconds(steps_left_low, steps_left_high, seconds_per_step=12.7):
    return int((steps_left_low + steps_left_high) * seconds_per_step)
