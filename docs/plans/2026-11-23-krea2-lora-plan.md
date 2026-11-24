# Krea 2 LoRAs — Implementation Plan

Spec: docs/specs/2026-11-23-krea2-lora.md
Workspace: current folder (~/Documents/private/myproject; not a git repository, so no commits or SHAs)
Jira: N/A

## Progress

- [x] Task 1: Server accepts Krea 2 LoRA uploads — reviewed; review fixes applied (no git, no SHA)
- [x] Task 2: Worker loads Krea 2 LoRAs, filter still on every image — reviewed; review fixes applied (peft pin, error boundary)
- [x] Task 3: CLI uploads and lists Krea 2 LoRAs — reviewed (approve); advisories applied
- [x] Task 4: Krea photos render with --lora — reviewed (no blocking); advisories applied
- [x] Task 5: Video and story first frames use Krea LoRAs — reviewed (approve); test advisory applied
- [x] Task 6: Dashboard shows and picks Krea LoRAs — reviewed (approve); verified in the browser preview
- [x] Task 7: Docs, deploy and live check — deployed; live photos with the LoRA pass (two live fixes: peft pin, kohya conversion)

## Tasks

### Task 1: Server accepts Krea 2 LoRA uploads

**What**: In `handler.py` add `KREA2_DIM = 6144` and `_check_krea2_lora(header)` (LoRA keys; no
`OTHER_MODEL_KEYS` and no `self_attn`/`cross_attn`/`ffn`; `blocks[._]N[._]` with
`attn[._]?(wq|wk|wv|wo|qkv_proj|gate)` or `mlp`, main and `txtfusion_*_blocks`; dtype in
`LORA_DTYPES`; a 6144 dim). `lora_uploads` takes `model: t2v|krea2` (default `t2v`): `krea2` returns
one URL to `v<N>/krea2.safetensors`, `files: pair` with `krea2` is a 400, and a name whose registered
versions are the other kind is a 400 ("'NAME' is a Wan LoRA; use another name for a Krea 2 LoRA" and
the reverse). `register_lora_upload` finds `krea2.safetensors` first and writes `lora.json` with
`models: ["krea2"]`, `files: ["krea2.safetensors"]`, `single_file: true`; a Krea file under `t2v`
names fails with "this is a Krea 2 LoRA" and a Wan file under `krea2` with "this is a Wan LoRA".
`GET /loras` returns each version's `models`. IAM: allow PutObject `loras/*/krea2.safetensors`.
**Files**: lambda/api/handler.py, terraform/iam.tf, tests/test_api.py
**Depends on**: none
**Verify**: `make test` passes with new cases in test_api.py: a synthetic header copied from the
user's file layout (main + txtfusion blocks, BF16/F32, 6144) passes `_check_krea2_lora` and fails
`_check_wan_lora` with "this is a Krea 2 LoRA"; each refusal (no LoRA keys, Wan keys, no 6144, bad
dtype, pair, kind clash) returns 400; a registered Krea version lists with `models: ["krea2"]`;
`terraform -chdir=terraform validate` passes.
**Parallelizable**: yes — does not share files with Task 2

### Task 2: Worker loads Krea 2 LoRAs, filter still on every image

**What**: `worker.prepare_krea_loras(params)` downloads `params["krea_loras"]` (`key`) into
`LORA_CACHE` and returns `[{"path", "adapter", "weight", "trigger"}]` (adapter named like Wan's,
`user_<name>_v<N>`). `krea.generate(items, progress, loras=())` puts `loras` (path, adapter, weight)
in request.json, prepends missing trigger words to each prompt, and adds adapter names to `stats`.
`krea_gen.load(model_dir, filter_dir, loras)` calls `pipe.load_lora_weights(path, adapter_name=...)`
per LoRA and `pipe.set_adapters(names, adapter_weights=...)`; a load error raises
"LoRA <adapter> could not be loaded: ..." (exit non-zero, the message in stderr is surfaced by
`krea.generate`). `run_photos_krea` passes `prepare_krea_loras(params)`. The filter loop in
`krea_gen.main` is untouched.
**Files**: worker/worker.py, worker/krea.py, worker/krea_gen.py, tests/test_worker.py
**Depends on**: none
**Verify**: `make test` passes with new cases in test_worker.py: with a stub pipeline that records
calls, `krea_gen.main` on a request with one LoRA calls `load_lora_weights` then `set_adapters` with
the weight, and a stub filter flagging one image still drops it (re-rolled, then filtered) — the
filter runs on every image with a LoRA loaded; `krea.generate(..., loras=[...])` writes them into
request.json and prepends the trigger; a failing `load_lora_weights` gives the named error;
`prepare_krea_loras` downloads once per version.
**Parallelizable**: yes — does not share files with Task 1

### Task 3: CLI uploads and lists Krea 2 LoRAs

**What**: In `vidgen.py` add `lora_kind(header)` (`t2v` | `krea2` | None, same rules as Task 1) and
`read_safetensors_header(path)`. `lora_upload` detects the kind before any API call: Krea with
`--high/--low` is refused ("Krea 2 LoRAs are one file; use --file"), an unknown kind is refused with
the server's reasons, and the reserve call sends `model`. `lora list` shows `krea2` as the kind.
**Files**: cli/vidgen.py, tests/test_trainer_cli.py, tests/test_api.py
**Depends on**: Task 1
**Verify**: `make test` passes: `lora_upload` with a Krea header file reserves with `model: krea2`
and one URL; a Krea pair is refused with no API call; `lora list` prints `krea2`; an agreement test
runs `vidgen.lora_kind` and `handler._check_krea2_lora`/`_check_wan_lora` over the same headers
(Krea, Wan single, Wan pair file, Flux-like, non-LoRA) and they agree.

### Task 4: Krea photos render with --lora

**What**: `_resolve_loras` returns `model` per item and, for Krea, `key` instead of
`high_key`/`low_key`; Krea items refuse a `step`; at most 2 per kind. In `create_job`, photos with
`model: krea2` resolve `loras` into `params["krea_loras"]` (a Wan LoRA is a 400 "Wan LoRA 'x' is for
video; Krea 2 photos take Krea 2 LoRAs"); photos with `model: qwen` and avatar jobs refuse any LoRA.
Job records/summaries list each LoRA with `model`. CLI `photos --lora NAME[:WEIGHT]` (repeatable),
refused locally with `--model qwen`.
**Files**: lambda/api/handler.py, cli/vidgen.py, tests/test_api.py, tests/test_trainer_cli.py
**Depends on**: Task 1, Task 2, Task 3
**Verify**: `make test` passes: a krea2 photos job with `loras: "moody:0.8"` queues with
`krea_loras: [{name, version, weight: 0.8, model: krea2, key: .../krea2.safetensors}]`; qwen photos,
avatar job, Wan LoRA on krea photos, step on Krea, 3 Krea LoRAs and weight 2.0 each return 400 with
the named reason; `vidgen photos --model krea2 --lora moody:0.8` sends `loras`.

### Task 5: Video and story first frames use Krea LoRAs

**What**: In `create_job` for video/story, split resolved LoRAs: Wan into `params["loras"]`
(unchanged), Krea into `params["krea_loras"]`; Krea LoRAs need `image_model: krea2` (400 "Krea 2 LoRA
'x' needs --image-model krea2 (a Krea first frame)"). describe/write/lora_preview refuse Krea LoRAs
(`lora preview` refuses a Krea name). Worker: `krea_first_frames` and the story first-frame path pass
`prepare_krea_loras(params)` to `krea.generate`; Wan `set_loras` keeps getting only Wan LoRAs. A story
whose shots all have avatars records `krea_loras_unused: true` in the result.
**Files**: lambda/api/handler.py, worker/worker.py, tests/test_api.py, tests/test_worker.py
**Depends on**: Task 4
**Note (during implementation)**: the API split for video/story (`_job_loras`, image_model check) landed in Task 4 with the shared routing helper; this task keeps the worker side, the story `krea_loras_unused` flag and the `lora_preview` refusal.
**Verify**: `make test` passes: a video with `image_model: krea2`, `avatar: none` and one Wan + one Krea
LoRA queues with each in its own list; the worker test sees the Krea LoRA only in the Krea request and
the Wan LoRA only in `set_loras`; a video without `image_model: krea2` and a `lora_preview` of a Krea
name return 400; the all-avatar story sets `krea_loras_unused`.

### Task 6: Dashboard shows and picks Krea LoRAs

**What**: The LoRA pickers (`.lora-select`) list only the kinds a form can use: video/story forms
list Wan, plus Krea when "Krea first frame" is chosen; the photos form gets a LoRA picker shown with
the Krea model only. Learning > uploaded LoRAs shows the kind; the Add-a-LoRA hint says Wan 14B or
Krea 2 files are accepted (detected from the file). `ui.py` passes the photos LoRA through.
**Files**: cli/ui.html, cli/ui.py, tests/test_trainer_cli.py
**Depends on**: Task 4, Task 5
**Verify**: `make test` passes (ui.py builds the photos body with `loras`); `vidgen ui` in the
browser preview: the photos form shows the picker only with Krea 2 selected and lists only Krea LoRAs;
the video form lists Krea LoRAs only with the Krea first frame on; no console errors.

### Task 7: Docs, deploy and live check

**What**: RUNBOOK (LoRA section, command tables, `lora add` accepts Krea 2 files, `--lora` routing by
kind, refusals) and README (LoRA paragraph); spec "Update: live check" section. Deploy: `make plan`
(review: Lambda, IAM policy, worker files), `make apply`, restart the worker. Live: `vidgen lora add
<name> --file ~/Movies/lora/kera2_Unlocked_V3.safetensors`, then `vidgen photos --model krea2 --lora
<name> --prompt "..." --count 2 --seed N` and the same without `--lora`.
**Files**: RUNBOOK.md, README.md, docs/specs/2026-11-23-krea2-lora.md
**Depends on**: Task 1, Task 2, Task 3, Task 4, Task 5, Task 6
**Verify**: `make test` passes; the plan changes only the API Lambda, the API IAM policy and the worker
code objects; `lora list` shows the new name as `krea2`; the worker log shows `loading LoRA user_..._v1`
in the Krea run and the job result lists filter stats; the two photo sets differ visibly with the same
seed.
