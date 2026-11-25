# AI video pipeline

Self-hosted text-to-video, video descriptions, story-to-video and photos of a person in many places on one AWS
GPU spot instance that **runs only while there is work** and stops itself
when idle.

```
client (vidgen.py)
   │  request signed with your SSH key
   ▼
API Gateway (HTTP API) ──► Lambda "api" ──► DynamoDB (job records)
                              │      └────► SQS (job queue)
                              └─ starts ──► EC2 g6e.4xlarge spot (L40S 48 GB)
EventBridge 1/min ─► Lambda "waker" ──┘        │ pulls jobs from SQS, honours cancel + timeout
     (restarts after interruptions,            │ writes results to S3
      stops it after max_uptime_hours)         └ stops itself after IDLE_MINUTES idle

laptop: vidgen watcher (launchd / systemd timer / Windows startup) checks every 45 s, downloads results
```

| Purpose | Model | Notes |
|---|---|---|
| Short videos (text) | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 1280×704, 16 fps; single takes up to 60 s as chained 5 s segments (per-segment timeline, colour matching, resume); also renders avatar candidates |
| Short videos (avatar shots) | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | animates the avatar keyframe |
| Speed-up | `lightx2v/Wan2.2-Lightning` | 4-step distilled LoRAs for both A14B models (no CFG); `VIDEO_LIGHTNING=0` on the box switches back to 40 steps |
| Sound | `hkchengrex/MMAudio` (`large_44k_v2`) | ambient sound and Foley generated from each clip; **weights are CC-BY-NC 4.0 (non-commercial)**, `--no-sound` turns it off |
| Video descriptions | `Qwen/Qwen2.5-VL-7B-Instruct` | title, description and tags as JSON |
| Story → video | Qwen2.5-VL plans the shots, Wan renders each one | clips joined with ffmpeg, burned-in subtitles, narration mixed over generated sound |
| Avatar keyframes | `Qwen/Qwen-Image-Edit-2511` + 8-step Lightning LoRA | places 1–3 avatars into each shot's scene; Wan then animates from that frame |
| Photos of a person | `Qwen/Qwen-Image-Edit-2511` + 8-step Lightning LoRA | `vidgen photos`: one person (uploaded photo or avatar) in up to 8 scenes you describe or random locations, ~1 MP photorealistic stills |
| Narration | `hexgrad/Kokoro-82M` | local text-to-speech on CPU, English voices |

**Avatars:** a default character stars in every video and story so clips stay consistent. Up to 3 can share a scene (`--avatar a,b`). Each avatar is a set of up to 12 reference images of the same person, either generated from a description (Wan renders candidates and you pick one) or uploaded; `vidgen avatar add` adds more photos later, and photos and keyframes use the first 3. The first video run without one asks you to set it up. `vidgen avatar refresh` replaces it, and earlier versions stay in `vidgen avatar history`. Every new avatar also gets a Krea 2 character LoRA (24 generated photos, trained on the trainer machine; trigger word = the avatar name; `--no-krea-lora` to skip), so Krea photos and Krea first frames can show that person; all LoRAs are copied to your computer (`<download dir>/loras/`). Every avatar also comes with media: its photo set and four 2 s clips of the person moving, copied to `<download dir>/avatars/<name>/` (`vidgen avatar media --name N` for existing avatars; see [docs/specs/2026-11-24-avatar-media.md](docs/specs/2026-11-24-avatar-media.md)). Long takes can be planned first (`vidgen plan`, dashboard "Plan first"): the first frame and editable per-segment beats, then rendered with the same seed. Each avatar also gets a Wan character LoRA (`avatar-<name>-wan`, trained on its photos and clips, ~1 h) that videos starring it use automatically, so the face holds over long takes; photo sets keep only photos with exactly one person ([docs/specs/2026-11-24-avatar-wan-lora.md](docs/specs/2026-11-24-avatar-wan-lora.md)). See [docs/specs/2026-11-24-krea-avatar-lora.md](docs/specs/2026-11-24-krea-avatar-lora.md).

**Photos:** `vidgen photos --image me.jpg --prompt "..." --location "..." --count 6` makes photorealistic stills of the person in each scene: free-form prompts and places in the order given, then random locations from a curated list of 50. It keeps face, hair, skin tone, body and outfit. A sharp, well-lit photo of one person works best.

**Dashboard:** `vidgen ui` opens a local page (127.0.0.1 only, signed with your SSH key like the CLI) with the server's state and the exact models deployed, your job history with waiting and render time per job (progress bar and time left while a job runs, Cancel, Re-run, and previews of the downloaded photos and videos), the Create tab for videos, photos and stories, the Avatars tab (each avatar with its Krea LoRA, Wan LoRA and media, and the forms for a new one), and the Learning panel where you mark folders of clips as themes and upload them (`vidgen dataset add` does the same) to your private bucket for the trainer.

**Model mirror:** both machines keep a copy of their model weights in your bucket (`models/`) and a machine with a new disk restores from it instead of downloading from Hugging Face; it fills itself in the background ([spec](docs/specs/2026-11-25-model-mirror.md), RUNBOOK "Model mirror").

**Training your own themes:** teach the generators a person, style or motion from your own clips with LoRA adapters, trained on a separate machine and passed to the render box through S3. A theme can be trained further later (more steps, or added clips); each round is a new version you can switch back from. The design is in [docs/specs/2026-11-09-lora-training.md](docs/specs/2026-11-09-lora-training.md); a viability trial lives in `trainer/spike/`. Upload a folder (`vidgen dataset add`), then `vidgen train NAME` (or Train in the dashboard) trains a T2V LoRA on the trainer machine (~9 h for 1250 steps per expert). Then render with it: `vidgen video "..." --lora NAME[:WEIGHT]` (also `story`, with or without an avatar). LoRA files made elsewhere can be added with `vidgen lora add`: Wan 14B LoRAs for video, and Krea 2 LoRAs for Krea photos and Krea first frames (`--lora` sends each to its model; the content filter still checks every Krea image; [spec](docs/specs/2026-11-23-krea2-lora.md)).

**Look and quality:** every clip prompt gets a natural-look base (photorealistic, natural light, 35 mm look) and a static tripod camera unless the prompt names a move, plus a default negative prompt (full 40-step mode only). Clips are encoded with x264 CRF 16.

**Cost guards:** jobs can be cancelled (`vidgen cancel`), a single job times out after `job_timeout_minutes`, and the waker stops the machine after `max_uptime_hours` no matter what it's doing.

You can change the models through `video_model` / `i2v_model` / `vlm_model` / `edit_model`.

## How on-demand works

- **Start:** `POST /jobs` queues the job and calls `StartInstances` if the instance is stopped. A start takes about 1–2 min to boot, plus about 1 min to load the model.
- **Stop:** the worker calls `StopInstances` on itself once the queue has been empty for `idle_minutes` (default 15).
- **Spot interruption:** the instance is created as a *persistent* spot request with `instance_interruption_behavior = "stop"`. When AWS reclaims capacity, the instance stops instead of terminating, so the disk (and the downloaded models) survive. The in-flight SQS message becomes visible again. The waker Lambda restarts the instance once capacity returns, and the job reruns.
- **No capacity:** if `StartInstances` fails (e.g. `InsufficientInstanceCapacity`), the job stays queued and the waker retries every minute. `vidgen.py status` shows the state.
- **Cost while idle:** only the EBS root volume (~$40/month for 500 GB gp3, plus ~$35/month for 1000 MB/s throughput) plus negligible Lambda, API Gateway, SQS and DynamoDB charges.
- **Network:** the instance runs in the account's existing VPC (found by its `Name` tag, `vpc_name`) in the private subnet of `availability_zone` (`Visibility` tag), with no public IP, because the organisation's SCPs forbid creating VPCs and public IPs. Egress for model downloads goes through the VPC's transit gateway. The security group has no inbound rules.
- **S3:** the bucket has encryption, a TLS-only policy and `BucketOwnerEnforced`, but no bucket-level public access block (the SCP denies setting one); nothing grants public access.

## Authentication

There are no API keys. Every request is **signed with your SSH ed25519 key** (`ssh-keygen -Y sign`), and the Lambda verifies it against `authorized_ssh_public_keys`:

- The signature covers the method, path, timestamp, a nonce and a hash of the body. Tampered requests fail.
- Requests older than 5 minutes are rejected, and so is any reuse of a nonce.
- Only your public key is stored in AWS. The private key never leaves your laptop, and ssh-agent works.
- The same key is the only one allowed to SSH into the GPU worker, which has no open ports; SSH goes through SSM.

## Deploy and use

The [RUNBOOK.md](RUNBOOK.md) covers setup, daily use, limits, troubleshooting and operations. In short:

```bash
ssh-keygen -t ed25519 -C dan -f ~/.ssh/vidgen_ed25519
cp terraform/terraform.tfvars.example terraform/terraform.tfvars   # paste the .pub in
make plan      # review
make apply

python3 cli/vidgen.py configure --url "$(terraform -chdir=terraform output -raw api_url)" \
  --ssh-key ~/.ssh/vidgen_ed25519.pub --download-dir ~/Movies/vidgen
make smoke                                 # end-to-end test of every model on the GPU
python3 cli/vidgen.py watcher install      # downloads finished jobs every 45 s
python3 cli/vidgen.py video "walking through a night market at dusk" -o market.mp4   # returns at once
```

## Tuning

- `instance_type`: defaults to `g6e.4xlarge` (one L40S, 128 GB RAM): cheapest, with good spot capacity, but it holds one big model at a time, so each switch between T2V, I2V and the avatar keyframe model reloads ~60 GB (~5 min). Larger sizes keep all three loaded and suit batches of avatar videos or stories; see [Choosing the instance](RUNBOOK.md#choosing-the-instance). Changing it replaces the spot instance and its disk. The Wan2.2 A14B video models (two ~28 GB experts each) run on the one 48 GB GPU with CPU offload, which also needs this RAM.
- `availability_zone`: a stopped spot instance can only restart in its own AZ. If one AZ keeps failing for lack of capacity, move it (that replaces the instance).
- Speed: Wan2.2-Lightning renders in 4 steps (`VIDEO_LIGHTNING=1`, the default); the full model uses 40 steps with CFG for maximum quality (`VIDEO_LIGHTNING=0` in `/etc/ai-video.env`, then restart the worker). Both A14B models are kept as local bf16 copies (`worker/convert_bf16.py`), halving load time. MMAudio stays in RAM beside the video model so sound doesn't force a reload. A model's weights stay in RAM while it is on the GPU, so handing the GPU to the next one copies nothing back (`KEEP_RAM_WEIGHTS=0` switches that off). The worker log says where each render's time went (`pipeline_timing` and `video_timing` lines).
- `root_volume_throughput`: 1000 MB/s by default; model loads read ~60 GB, so this bounds cold-start and model-switch time.
