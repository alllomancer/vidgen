# Krea 2 LoRAs — decision log

Spec: docs/specs/2026-11-23-krea2-lora.md

## D1 — [2026-11-23] — Task 1: Server accepts Krea 2 LoRA uploads

**Status**: Accepted
**Decision**: "One kind per name" is checked at reserve and again at register, and a version holding both a Krea and a Wan file is refused.
**Reason**: Reserving writes nothing, so an unregistered reservation is invisible to a second one of the other kind (code review finding).
**Alternatives considered**: A reservation marker object per version; more S3 writes and IAM paths for a single-user tool.

## D2 — [2026-11-23] — Task 1: Server accepts Krea 2 LoRA uploads

**Status**: Accepted
**Decision**: The safetensors header check rejects tensor entries that aren't `{dtype: str, shape: [int]}` with a 400 "bad header".
**Reason**: Malformed untrusted headers raised TypeError/AttributeError and returned 500 (security review, low).
**Alternatives considered**: Catching broad exceptions in each checker; less precise.

## D3 — [2026-11-23] — Task 1: Server accepts Krea 2 LoRA uploads

**Status**: Accepted
**Decision**: The presigned upload URL staying writable for its 1 h TTL after registration (content swap after the checks) is left as is for Krea, same as Wan, and raised as a separate follow-up.
**Reason**: It is a pre-existing design gap of the upload flow (security review, medium); fixing it (versioned objects or copy-on-register) changes the Wan path too and is out of this spec's scope.
**Alternatives considered**: Fixing only the Krea path; inconsistent and still leaves Wan open.

## D4 — [2026-11-23] — Task 2: Worker loads Krea 2 LoRAs

**Status**: Superseded by D6
**Decision**: Pin `peft==0.17.1` in the Krea venv and fail with a specific "needs peft" message when it is missing.
**Reason**: The Krea venv had no peft (`is_peft_available()` False on the box), so every LoRA load would have failed as "unsupported layout" (code review finding); 0.17.1 matches the Wan venv and resolves without other changes (pip dry-run on the box).
**Alternatives considered**: Merging LoRA weights manually without peft; reimplements what Krea2LoraLoaderMixin already does.

## D5 — [2026-11-23] — Task 2: Worker loads Krea 2 LoRAs

**Status**: Accepted
**Decision**: Only `LoraLoadError` (load or set_adapters) becomes the job's error, with a fixed message and the detail in the worker log; other failures (e.g. CUDA OOM) propagate as before, and LoRAs load with `use_safetensors=True`.
**Reason**: Catching every RuntimeError mislabelled OOM as a LoRA problem and leaked internal paths (code and security reviews).
**Alternatives considered**: Returning the raw diffusers message; leaks cache paths and module details.

## D6 — [2026-11-23] — Task 7: Docs, deploy and live check

**Status**: Accepted (supersedes D4)
**Decision**: Pin `peft==0.21.3` in the Krea venv instead of 0.17.1.
**Reason**: Live, peft 0.17.1 imported `HybridCache` from transformers (removed in 5.x), and because diffusers imports peft whenever it is installed, every Krea job failed at `from diffusers import Krea2Pipeline`; 0.21.3 imports cleanly there (checked on the box from a temp dir; the pip dry-run only adds peft). A pip dry-run checks resolution, not imports, which is why D4 missed it.
**Alternatives considered**: Uninstalling peft (drops LoRA support); pinning transformers back (would break Krea2Pipeline, which needs transformers >= 4.57 / Qwen3VL).

## D7 — [2026-11-23] — Task 7: Docs, deploy and live check

**Status**: Accepted
**Decision**: Read Krea LoRA files with safetensors in `krea_gen` and convert kohya/musubi keys to the reference layout (`diffusion_model.blocks.N.attn.wq.lora_A/B.weight`, alpha/rank folded into lora_B) before `load_lora_weights`; diffusers maps the rest.
**Reason**: diffusers 0.41's Krea 2 converter only runs for `diffusion_model.`/`base_model.model.` keys and silently loads nothing from kohya keys (live failure); reading the file ourselves also removes any pickle fallback.
**Alternatives considered**: Converting at upload time on the API (would need torch in the Lambda); mapping straight to diffusers module names (duplicates diffusers' own maps).

## D8 — [2026-11-23] — Follow-up: exact prompts for Krea photos

**Status**: Accepted
**Decision**: `raw: true` on Krea photos (`vidgen photos --model krea2 --raw`, dashboard "Exact prompt") sends each prompt word for word: no look/realism sentences, no LoRA trigger words; extra photos repeat the prompts with new seeds instead of random places. The content filter is unchanged.
**Reason**: Every Krea photo prompt got a look sentence ("Photorealistic photograph, 50mm lens … realistic skin texture …") that pulled scenes toward face close-ups. Live check: "a red fox walking through deep snow in a pine forest, wide shot" with `--raw` gave the wide scene as written (2 images, 0 filtered).
**Alternatives considered**: Rewording the default look sentences; would change every existing photo job's output.

## D9 — [2026-11-24] — Spike: Krea 2 character LoRA trained from an avatar

**Status**: Accepted
**Decision**: A Krea 2 character LoRA trained with musubi-tuner (pinned `f8a1b037`, `krea2_train_network.py`, `networks.lora_krea2`, rank/alpha 32, lr 1e-4, flow shift 2.5, 1024 px) on Krea 2 RAW from 24 Qwen photos of avatar `smoke-a` keeps the identity on Turbo; the converter now also maps the 8 standalone Linear layers musubi-tuner trains (`first`, `last_linear`, `tmlp_0/2`, `tproj_1`, `txtmlp_1/3`, `txtfusion_projector`).
**Reason**: Live run (`trainer/spike/run_krea_trial.sh` on the render box, worker paused): 960 steps at ~3.45 s/step on the L40S (55 min), loss ~0.07; Turbo samples looked like the avatar from step 240. Rendered as `smka v1` with an exact prompt not in the training set: with the LoRA both seeds show the same man matching smoke-a, without it two different generic men. All 264 layers mapped on the box (0 missing, 0 shape mismatches). Bias seen: his default shirt is blue, as in many training photos. musubi-tuner's doc had an outdated VAE path (now `Comfy-Org/Qwen-Image`).
**Alternatives considered**: Image-to-image from the avatar photo (keeps colours/composition, not the face).

## D10 — [2026-11-24] — Task 1 (avatar LoRAs plan): Download every LoRA locally

**Status**: Accepted
**Decision**: The watcher's `check` syncs LoRAs in-process after the jobs pass (at most every 10 min, one sync at a time via a lock, per-version failures logged and retried, every error caught); a long download delays that check's next job poll.
**Reason**: launchd starts `vidgen check` every 45 s and skips while one is running, so a background thread would die with the process; a LoRA copy takes ~10-30 s, so the delay is small.
**Alternatives considered**: A separate launchd job for LoRA sync (more install surface on three OSes).

## D11 — [2026-11-24] — Task 4 (avatar LoRAs plan): avatar_photoset job

**Status**: Accepted
**Decision**: The worker reads the train queue URL from `code/worker/settings.json` (Terraform-managed, synced on every worker start) when the env var is missing.
**Reason**: The render box keeps its first-boot user data (`user_data_replace_on_change = false`), so a new env var would never reach the existing box.
**Alternatives considered**: Replacing the instance (loses its disk); `sqs:GetQueueUrl` by name (couples the worker to naming).

## D12 — [2026-11-24] — Tasks 4-5 (avatar LoRAs plan): build state on avatar.json

**Status**: Accepted
**Decision**: A build is "current" while its job id is in `avatar.json` `krea_lora.job_ids` (and the avatar's `created_at` matches); the API writes the photoset job id before queueing, the worker writes the train job id before sending its message, every writer re-reads and checks before writing, and the train job id is derived from the photoset job id (uuid5) with a conditional put.
**Reason**: Reviews found races (warm worker skipping a fresh build, redelivery duplicating training, failures leaving the avatar busy forever); these orderings close them without a lock service.
**Alternatives considered**: S3 conditional writes with ETags on avatar.json; not done — the remaining read-modify-write window is milliseconds with one user (accepted risk).

## D13 — [2026-11-24] — Task 5 (avatar LoRAs plan): Krea training on the trainer

**Status**: Accepted
**Decision**: The trainer builds its own Krea venv (`/opt/trainer/krea-venv`, torch 2.8/cu126 + the same musubi-tuner checkout) and downloads the Krea models (~66 GB) on the first Krea job, instead of changing the trainer's user data.
**Reason**: musubi-tuner's Krea scripts need torch >= 2.6 while the Wan venv keeps 2.5.1, and the existing trainer never re-runs user data.
**Alternatives considered**: Upgrading the Wan venv's torch (risks the working Wan training).
