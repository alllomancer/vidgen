# Long clips (30 s): bake-off of today's chain, SVI 2.0 Pro and Helios

Jira: N/A
Date: 2026-11-26
Status: Results recorded; inconclusive on quality (see Limits)
Last reviewed: 2026-11-26

## Question

The user wants clips of at least 30 s. Today a long take is a chain of 5 s Wan 2.2 segments, each continuing
from the previous one's last frame. Two alternatives were tried on a g7e (96 GB GPU), same first frame, same
six 5 s beats (`trainer/spike/bakeoff/prompts.json`), 30 s each:

- **A. SVI 2.0 Pro** (vita-epfl/Stable-Video-Infinity, branch `svi_wan22`, commit bab0662): two LoRAs on the
  Wan 2.2 I2V A14B model; clips continue from the previous clip's latents plus a fixed anchor image.
- **B. Helios** (PKU-YuanGroup/Helios, commit f1a8aad, Helios-Distilled): a 14B model that generates long
  video in one run, in 33-frame chunks.

## Results

| Run | Output | Render time | Peak GPU | What the frames show |
|---|---|---|---|---|
| Today's chain (worker `generate_video`, Lightning 4 steps, models held on the GPU) | 30.1 s, 832x480, 16 fps | 261 s | 69.4 GB | Both people of the first frame stay; the actions follow the beats; clothing drifts towards the prompt |
| A. SVI 2.0 Pro + Lightning LoRA, 4 steps, no guidance | 28.8 s, 832x480, 16 fps | 156 s (25 s per clip) | 68.9 GB | **Noise only: not a usable result** |
| A. SVI 2.0 Pro, 20 steps, guidance 4 (2 clips) | 9.8 s, 832x480, 16 fps | 450 s (225 s per clip) | 68.9 GB | Replaces the first frame's people with the prompt's subject; a ghosted double image at the 5 s seam |
| B. Helios distilled, image-to-video | 30.2 s, 640x384, 24 fps | 188 s including model load | 44.9 GB | One person carries out the whole sequence in one continuous take; the second person of the first frame is dropped; the room changes over the take |
| B. Helios distilled at 832x480 | - | - | - | Failed: that size is not accepted (tensor size mismatch) |

Clips: `s3://<bucket>/outputs/trial-g7e/20261126-*-run3/bakeoff/` (and `...-run2/bakeoff/` for the chain).

## Limits (why this does not decide the question)

- **The first frame did not match the prompts.** It shows a man and a woman in business clothes; every beat
  describes "a woman in a green raincoat". Each method resolved the conflict differently (the chain kept the
  frame, SVI with guidance followed the text, Helios kept one person), so identity can't be compared.
- **SVI's practical setting failed.** With the Lightning LoRA at 4 steps the output is noise. Not yet known
  whether the LoRA does not load correctly in DiffSynth, or needs other step/shift settings (community
  guides use 6 steps, shift 8, guidance 1.5). Without a few-step setting SVI costs ~225 s per 5 s clip.
- One seed per run; Helios only at its default size.

## What it does show

- Helios really produces a 30 s take in one run, in about 3 minutes, in 45 GB of GPU memory (it would fit
  the current 48 GB L40S), at low resolution.
- Today's chain on a g7e with models held on the GPU takes 261 s for 30 s at 832x480.
- SVI 2.0 Pro works with full guidance but is about 5x slower than the chain per clip.

## Next step if continued

A fair rerun: a first frame with one person that matches the beats (an avatar keyframe), a working few-step
SVI setting, Helios at a size it accepts (e.g. 832x448 or 896x512), two seeds each.
