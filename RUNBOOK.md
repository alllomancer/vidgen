# Runbook — AI video pipeline

How to deploy, use and operate the pipeline, and what its limits are.
Architecture is in [README.md](README.md).

---

## 1. One-time setup

### 1.1 Prerequisites

**For deploying (once):**

| Tool | Check |
|---|---|
| Terraform ≥ 1.6 | `terraform version` |
| AWS CLI v2 + credentials for the target account | `aws sts get-caller-identity` |
| Session Manager plugin (SSH/shell into the worker) | `session-manager-plugin --version` |
| Python 3.10+ with pip (Lambda build) | `python3 --version` |

**For using it, on any machine you'll send jobs from (macOS, Linux or Windows):**

| Tool | Check |
|---|---|
| Python 3.10+ (the CLI is standard library only; nothing to install) | `python3 --version` / `py --version` |
| OpenSSH ≥ 8.1 (signs every request). Built into macOS, Linux and Windows 10/11 ("OpenSSH Client" optional feature) | `ssh -V` |

### 1.2 Create your signing key

Use a dedicated **ed25519** key for this system, with a passphrase:

```bash
ssh-keygen -t ed25519 -C dan -f ~/.ssh/vidgen_ed25519
```

The comment (`dan`) becomes your client name in job records and S3 paths. Load the key into your ssh-agent so the CLI and the background watcher can sign without asking for the passphrase:

| OS | Load the key |
|---|---|
| macOS | `ssh-add --apple-use-keychain ~/.ssh/vidgen_ed25519` once. After each reboot: `ssh-add --apple-load-keychain` |
| Linux | `ssh-add ~/.ssh/vidgen_ed25519` in each login session. For the background watcher, use an agent with a fixed socket (e.g. `systemctl --user enable --now ssh-agent`, plus `export SSH_AUTH_SOCK=$XDG_RUNTIME_DIR/ssh-agent.socket` in your profile) |
| Windows (PowerShell, admin once) | `Get-Service ssh-agent \| Set-Service -StartupType Automatic; Start-Service ssh-agent`, then as yourself `ssh-add $HOME\.ssh\vidgen_ed25519` (persists across reboots) |

### 1.3 Configure and deploy

```bash
cd ~/Documents/private/myproject
cp terraform/terraform.tfvars.example terraform/terraform.tfvars
# paste the contents of ~/.ssh/vidgen_ed25519.pub into authorized_ssh_public_keys
make plan        # builds the Lambda bundle, runs terraform plan, saves tfplan
# review the plan
make apply       # applies exactly the reviewed tfplan
```

### 1.4 First boot (30–45 min, one time only)

The instance launches immediately. It installs packages and gets the model weights (video, description, avatar-keyframe and voice models): from the [model mirror](#model-mirror) in your bucket when an earlier box filled it, else about 310 GB from Hugging Face plus the Wan conversion. Watch it with:

```bash
eval "$(terraform -chdir=terraform output -raw ssh_command)"
sudo tail -f /var/log/ai-video-setup.log      # setup
journalctl -u ai-video-worker -f              # worker once setup is done
```

`{"event": "worker_up", ...}` means it's ready. It stops itself `idle_minutes` (15) later, and from then on starts only when you send a job.

### 1.5 Configure the CLI

Settings are saved to `~/.vidgen/config.json`; `VIDGEN_URL` / `VIDGEN_SSH_KEY` / `VIDGEN_DOWNLOAD_DIR` env vars override them.

```bash
python3 cli/vidgen.py configure \
  --url "$(terraform -chdir=terraform output -raw api_url)" \
  --ssh-key ~/.ssh/vidgen_ed25519.pub \
  --download-dir ~/Movies/vidgen
```

- `--ssh-key` takes the **.pub** path, which means "sign through ssh-agent".
- Add a shortcut:
  - macOS/Linux: `alias vidgen="python3 $HOME/Documents/private/myproject/cli/vidgen.py"`
  - Windows: `function vidgen { py C:\path\to\cli\vidgen.py @args }` in your PowerShell `$PROFILE`

### 1.6 Install the background watcher

```bash
vidgen watcher install
```

It checks every **45 seconds** and downloads finished jobs to your download folder. When nothing is pending, it doesn't contact the server, except for the LoRA copy: at most every 10 minutes it lists your LoRAs and copies any new version to `<download dir>/loras/<name>/v<N>/` (see "Krea 2 LoRA per avatar").

| OS | How it runs |
|---|---|
| macOS | launchd agent running `vidgen check` every 45 s. Survives logout and reboot. Shows a notification when a video is ready |
| Linux | systemd user timer (`~/.config/systemd/user/vidgen-watcher.timer`) running `vidgen check` every 45 s. Notifications via `notify-send` if installed. With no systemd (e.g. some WSL setups), run `vidgen watch` in tmux instead |
| Windows | Task Scheduler can't repeat more often than once a minute. Instead, a launcher in your Startup folder runs `vidgen watch` (a 45 s loop) hidden under `pythonw` at every login, and the install starts it right away. No notifications; check `vidgen jobs` or the log |

| Command | Does |
|---|---|
| `vidgen watcher status` | installed or not, plus the last 10 log lines |
| `vidgen watcher uninstall` | remove it |
| log file | `~/.vidgen/watcher.log` |
| `vidgen watch` | the same loop in the foreground (Ctrl-C to stop) |

If your key isn't loaded in the agent, the log shows `signing failed … will retry`. The watcher catches up once the key is loaded.

### 1.7 Smoke test (after first boot, and after any model or worker change)

```bash
make smoke          # same as: vidgen smoke
```

This runs one small job through every model and code path on the real GPU and prints a PASS/FAIL table:

Jobs are queued grouped by model, so a g6e.4xlarge loads each big model only once: everything that uses Wan T2V first, then photos and the keyframe + I2V job (sharing one edit-model load), then describe.

| Step | Exercises |
|---|---|
| avatar | Wan T2V-A14B single-frame candidates; saves avatars `smoke-a` and `smoke-b` (your default is untouched) |
| video | Wan T2V-A14B text-to-video (2 s) + MMAudio sound |
| cancel | cancelling a running job (a 5 s clip) |
| story | Qwen2.5-VL shot planning, Wan T2V shots (no avatars), Kokoro narration, subtitles, sound mix, ffmpeg assembly |
| photos | Qwen-Image-Edit-2511 (Lightning 8 steps): 3 portrait photos of `smoke-a` (one prompt, one location, one random location); checks count, order and 896×1152 size |
| avatar-vid | two-avatar Qwen-Image-Edit keyframe → Wan I2V-A14B image-to-video + sound |
| krea | Krea 2 Turbo (own venv, content filter): 2 photos without a person (one prompt, one location); checks count, order and 896×1152 size |
| describe | Qwen2.5-VL video understanding |

- **Time:** ~30–60 min including cold start. It runs in the foreground, so leave the terminal open.
- **Outputs:** `<download dir>/smoke-<timestamp>/`. **Open them and look**: the test only checks that files exist and have sane lengths, not that the video looks right.
- **Exit code:** non-zero if any step failed. The failing step's error is in the table. The full traceback is in the worker log (`journalctl -u ai-video-worker -n 300`).
- `--skip-cancel` drops the cancel step.

---

## 2. Daily use

**Submitting is fire-and-forget.** Each command queues the job, prints its ID and where the result will be saved, and exits. The GPU machine starts if needed, renders the job and stores the result in S3 for 30 days. The watcher downloads it within 45 s of finishing. If your laptop is asleep or offline, the next check after you're back online catches up.

`-o name.mp4` picks the file name inside the download folder (default `<job_id>.mp4`). Extra files land next to it as `name.<kind>.<ext>`.

### Jobs

```bash
vidgen jobs              # local list: queued / running (with progress) / downloaded / failed / cancelled
vidgen cancel 3f9a       # job id or a unique prefix
vidgen status            # instance state, queue depth, minutes running, uptime hold
vidgen wake              # start the machine now (e.g. before a batch); also clears an uptime hold
```

**Cancelling:**
- A queued job is cancelled immediately.
- A running job stops within ~15 seconds, between denoising steps, and shows as `cancelled`. Nothing is uploaded.
- The machine then moves on to the next job, or stops after the idle timeout.

### Avatars (your stars)

Every `video` and `story` stars your **default avatar**, so the same character appears in every clip.

**Krea 2 LoRA per avatar** (on by default): creating or replacing an avatar also builds a Krea 2 character LoRA for it, so Krea photos and Krea first frames can show that person.
- **How**: the render box makes 24 varied Qwen photos of the avatar (framing, light, places, outfits), checks each with the Krea content filter (flagged ones dropped; fewer than 16 left and there is no LoRA), and writes them as `datasets/<client>/avatar-<name>/v<N>/`; the trainer machine trains a rank-32 LoRA on Krea 2 RAW (960 steps, ~55 min) and registers it as `avatar-<name>` (`krea2`, samples under `samples/`). About 1.5 h of GPU and ~$3 per avatar. The first Krea build on a trainer disk also gets ~66 GB of Krea models (from the [model mirror](#model-mirror) when it has them, else Hugging Face with the four files in parallel; started in the background as soon as the job starts) and builds its Krea venv.
- **Trigger word = the avatar name**: write the name in prompts ("dana reading on a park bench"); when it's missing it is put in front, also with `--raw`. Common words or very short names are weaker triggers.
- **State**: `vidgen avatar list` and the dashboard show `queued`, `photos`, `training`, `ready`, `failed` (with the reason) or `skipped`. A rebuild keeps the last ready LoRA in use until the new one is ready; one build at a time per avatar. Image edits and restores keep the link.
- **Opt out / on demand**: `--no-krea-lora` on `avatar refresh`/`pick` (dashboard: untick "Also train a Krea 2 LoRA"); `vidgen avatar krea-train --name N` builds one (or a new version) later, e.g. for avatars made before this. Existing avatars are not built automatically.
- **Using it**: `vidgen photos --model krea2 --avatar N --prompt "N on a ferry at dusk"` (dashboard: Photos, the avatar, "Krea 2 with this avatar's Krea LoRA"); videos/stories with `--image-model krea2` and that avatar get a Krea first frame with its LoRA (videos: exactly one avatar; stories: shots with just that avatar). An avatar without a ready LoRA is refused for Krea with its state.
- **Content filter**: as for every Krea image; avatars whose photo sets are flagged get no Krea LoRA. Use it only for people who agreed: a likeness LoRA can put them in any image.
- **Needs** `training_enabled = true` and access to `krea/Krea-2-Raw` (accept its license on huggingface.co with the account of the SSM token).

**Avatar media (photos + 2 s clips), for every avatar:** creating or replacing an avatar also makes its media, with or without a Krea LoRA.
- **What**: the photo set above (16-24 photos that pass the content filter) and four 2 s clips (33 frames at 16 fps, 832x832, no sound): Wan animates four of the photos with a gentle motion (a slow head turn and smile, a step towards the camera, a look around, a soft laugh). Each clip's first, middle and last frame go through the content filter; flagged clips are dropped (photos are kept even when every clip is). The first clip is the preview. About 6-8 min on the render box on top of the photo set. They are also training material for the planned avatar Wan LoRA.
- **Where**: stored as `avatars/<client>/<name>/media/` (`photo_NN.png`, `clip_N.mp4`, `media.json` with the prompts, motions and seeds) and copied by the watcher to `<download dir>/avatars/<name>/` (every 10 min, changed files only).
- **State**: `vidgen avatar list` and the dashboard show `queued`, `photos`, `clips`, `ready` (photo and clip counts) or `failed` (with the reason). A media failure never affects the Krea LoRA.
- **Existing avatars**: `vidgen avatar media --name N` (dashboard: "Make media" / "Remake media"; "Play clip" plays the preview). With a ready Krea LoRA its photo set is reused (clips only, ~6-8 min); otherwise the photos are made first (~25 min). One media job at a time per avatar.

**Wan LoRA per avatar** (on by default): every new avatar also gets a Wan character LoRA, so videos keep the person's face, also over long takes (docs/specs/2026-11-24-avatar-wan-lora.md).
- **How**: after the avatar's media is stored, its kept photos and clips become `datasets/<client>/avatar-<name>-wan/v<N>/` (captions filled in) and the trainer trains a T2V LoRA on both experts (1000 steps, ~1 h, after the Krea LoRA); it is registered as `avatar-<name>-wan`, trigger word = the avatar name.
- **One person per photo**: every photo set (Krea, media, Wan) is checked by the planner model; photos that don't show exactly one person are dropped, and fewer than 16 left fails the set ("use reference images of just this person"). Avatars whose reference photos look very different, or show other people, can fail here: use photos of just this person, similar in look.
- **Used automatically**: a `video` starring the avatar gets its ready Wan LoRA first at weight 1.0 (counted in the 2 Wan LoRAs per job); `--no-avatar-lora` (dashboard: untick "Avatar's Wan LoRA") leaves it out; `--lora avatar-<name>-wan:0.8` sets another weight. Stories don't use it yet. Re-running an old job reuses the LoRA version it was made with.
- **State / on demand**: `vidgen avatar list` and the dashboard show `queued`, `photos`, `training`, `ready` or `failed`; `--no-wan-lora` on create skips it; `vidgen avatar wan-train --name N` (dashboard "Train Wan LoRA") builds one from the avatar's stored media (needs ready media), e.g. for avatars made before this. Replacing an avatar drops the previous person's Wan LoRA link.
- **Local copies of every LoRA**: Wan and Krea, trained, uploaded or avatar LoRAs are copied to `<download dir>/loras/<name>/v<N>/` by the watcher (final files and `lora.json`; `vidgen lora pull [NAME] [--checkpoints]` on demand). `vidgen lora list` marks them `[on this Mac]` and shows the total size.

**First run:** if no avatar exists, `vidgen video …` / `vidgen story …` asks before submitting:
- **Image path** (png/jpg/webp) plus a short description of the look. The avatar is saved and your job is submitted.
- **Press Enter** to generate one from a description instead. 4 candidate portraits are rendered and downloaded as `avatar-star-….candidate_N.png`. Your job is *not* submitted yet: pick a candidate with `vidgen avatar pick <job_id> <N>`, then rerun your command.

| Command | Does |
|---|---|
| `vidgen avatar list` | your avatars; `*` marks the default |
| `vidgen avatar refresh [--name N]` | set up or replace an avatar (default name `star`) and make it the default. Asks for an image or a description. Non-interactive: `--image me.png --describe "…"`, or `--describe "…" --candidates 6` to generate |
| `vidgen avatar refresh --name N --image a.jpg --image b.jpg …` | create (or replace) an avatar from several photos of the same person, up to 12; the first is the main image |
| `vidgen avatar add [--name N] FILE…` | add photos to an existing avatar (up to 12 in total); the previous set goes into history |
| `vidgen avatar images [--name N]` | the avatar's images in order, with 1-hour preview links |
| `vidgen avatar main [--name N] <n>` | make image n the main image (moves it to position 1) |
| `vidgen avatar remove [--name N] <n>` | remove image n (an avatar keeps at least one); undo with `avatar restore` |
| `vidgen avatar pick <job_id> <N>` | save generated candidate N as the avatar |
| `vidgen avatar use --name N` | switch the default to another saved avatar |
| `vidgen avatar history [--name N]` | earlier versions, with 1-hour preview links |
| `vidgen avatar restore <version> [--name N]` | bring an earlier version back. The version it replaces goes into history too, so a restore can be undone |
| `vidgen avatar delete --name N [--with-loras] [--yes]` | delete the avatar from the server: images, earlier versions and media. **No undo.** Refused while a job that uses it is queued or running. `--with-loras` also deletes its own LoRAs (`avatar-N`, `avatar-N-wan`) and their training photos; without it they stay in your LoRA list. Files already copied to your download folder stay |
| `--avatar a,b` on `video`/`story` | star up to **3** avatars together in this job (the first is the lead) |
| `--no-avatar` on `video`/`story` | no avatar in this job |

**How it works:**
- For each clip, Qwen-Image-Edit places the avatars from their reference images into the shot's scene, and Wan animates from that frame.
- The edit model takes at most 3 images per call. A shot or photo with one avatar gets that avatar's first 3 images; a shot with 2–3 avatars gets each one's main image (image 1). Jobs record which images they used, so a re-run uses the same ones.
- Video jobs also save that frame as `<name>.keyframe.png`.
- In stories, the planner decides per shot which of the job's avatars appear. A shot with none of them is rendered from text alone.

**A good set of images** (only the first 3 are used, so order matters): image 1 full-length or waist-up (the main image, also used alone in shots with several avatars), image 2 a sharp face close-up, image 3 another angle. Extra images beyond 3 are kept, and `avatar main` / `avatar remove` reorder the set.

**A good reference image:**
- one person, full body or waist-up, facing the camera
- even light and a plain background
- the outfit you want in every video
- keep the description to look and clothing

**Consent:** use fictional or AI-generated characters, or yourself. Never use a real person's likeness without their consent.

### Video models: Helios (default) and Wan

`vidgen video` renders with **Helios** unless you ask for Wan (`--video-model wan`; dashboard: Create → Video → **Video model**). Stories always use Wan. Design and test results: [docs/specs/2026-11-26-helios-default.md](docs/specs/2026-11-26-helios-default.md).

| | Helios (default) | Wan (`--video-model wan`) |
|---|---|---|
| How a clip is made | the whole length in one take, from one prompt | 5 s segments, each continuing from the last frame |
| Length | 30 s unless `--seconds` says otherwise (1–60) | 5 s unless `--seconds` says otherwise (1–60) |
| Size and frame rate | 640×384 at 24 fps, always (`--upscale` doubles it to 1280×768) | 1280×704 at 16 or 32 fps |
| Render time on a g7e | about 3.5 min for 30 s, 5 min for 60 s (plus first frame and sound) | about 1.75–2 min per 5 s segment |
| Avatars | one person from the first frame; with two in the frame it kept one in a test | 1–3 |
| LoRAs | the job's Wan LoRAs (as many as Wan takes), each at its weight, through their low-noise files only | several, both halves |
| Beats, `vidgen plan`, `--anchor-every`, Smooth, Draft | no (the API says to use Wan) | yes |
| Variations | any length | takes up to 5 s |

- **Why 640×384**: at other sizes (512×512, 640×640, 384×640) Helios jumps at every 33rd frame; at its own size it doesn't (measured 2026-11-26). The first frame is made at 1280×768 and fitted.
- **Longer is not always better**: in a dark scene, coloured speckle built up in the background from about 20 s and was heavy by 60 s; a bright kitchen scene was clean at 30 s. If a long Helios take degrades, shorten it or use Wan.
- **The avatar's Wan LoRA matters**: without it the person drifts to someone else within ~15 s. Only its low-noise file is used; the high-noise file broke the colours in a test.
- **Several LoRAs**: Helios's own script takes one LoRA at full weight; with more, or another weight, the worker runs the same script through `worker/helios_run.py`, which loads them all and sets each weight (log line `helios_loras`). Your LoRAs were trained for Wan 2.2, so each is already off its home model here: stacked, they may come out weaker or stranger than on Wan. Lower a weight, or use `--video-model wan`, if a combination doesn't hold.
- **Look options** (`--look`, `--shot`, `--grain`, …) apply to both models.
- **Box**: Helios has its own venv (`/opt/helios-venv`, from `worker/helios_requirements.txt`), its repository at a pinned commit in `/opt/helios`, and ~80 GB of weights in `/opt/models/helios/`; all three are prepared at first boot (or on the first Helios job) and the weights are in the [model mirror](#model-mirror). Log line: `{"event": "helios", ...}`.

### Short video from text

```bash
vidgen video "walking through a night market, handheld camera, warm lantern light" -o market.mp4          # Helios, 30 s
vidgen video "walking through a night market, handheld camera" --video-model wan --seconds 5 -o market5.mp4
vidgen video "two friends sharing an umbrella in the rain" --avatar star,robot
vidgen video "a red fox in fresh snow" --no-avatar
```

| Option | Default | Allowed |
|---|---|---|
| `--video-model` | helios | `helios` or `wan` (see the table above) |
| `--seconds` | 30 (Helios), 5 (Wan) | 1–60. Wan: up to 5 s is one pass; longer takes chain 5 s segments (see **Long single takes** below) |
| `--width` / `--height` | 1280 × 704 | Wan only: stick to 1280×704 (landscape) or 704×1280 (vertical/shorts). Helios is always 640×384 |
| `--negative` | none | things to avoid, e.g. `"blurry, text, watermark"` |
| `--seed` | random | set to reproduce a result |
| `--no-sound` | sound on | skip the generated ambient sound/Foley (MMAudio); the clip comes back silent |

**Base prompt:** the worker adds a natural-look style to every clip (photorealistic, natural light, true-to-life colours, 35 mm look) and a default negative prompt (cartoon, CGI, plastic skin, warped hands, text…). The camera defaults to a **static tripod** shot; naming a move in your prompt (pan, handheld, tracking, dolly, drone, zoom, orbit…) turns the tripod default off. `--negative` is added to the default list. The base lives in `NATURAL_STYLE` / `NATURAL_NEGATIVE` in `worker/models.py`.

**Speed mode:** video is rendered with Wan2.2-Lightning, 4-step distilled LoRAs for both A14B experts (Apache-2.0), about 10× faster denoising than the full 40-step model. Lightning runs without CFG, so `--negative` and the default negative prompt have no effect. For maximum quality, set `VIDEO_LIGHTNING=0` in `/etc/ai-video.env` on the box and `sudo systemctl restart ai-video-worker`.

**Prompt tips:**
- Describe one continuous shot: setting, action, camera movement and lighting.
- With an avatar, don't describe their looks; the reference image covers that.
- For cuts, use `story`.


**How it is filmed (`--shot`, videos only; one or none):** `--shot stream` makes it look like a streaming session (a fixed webcam that never moves, the person facing it, a ring light on the face). `--shot pov` makes it look shot on a phone by someone holding it: the wording asks for amateur handheld phone footage (the phone and the person filming are not in the picture, so don't write them into your prompt), and the finished frames get a slow hand shake, a softer low-resolution image and grain, whatever the model did with the wording; the usual "static tripod" sentence is left out. `stream` only adds wording. Dashboard: Create → Video → Realism → **Streaming session** / **Shot on a phone** (ticking one unticks the other). Plans carry it (`vidgen plan ... --shot pov`), and the "render with" command repeats it. Stories and photos refuse it.

### Long single takes with Wan (up to 60 s)

This section is about `--video-model wan` (beats, plans and re-anchoring are Wan's). With Helios a long take is just `--seconds N`.

```bash
vidgen video "a man in a sunny kitchen makes coffee" --seconds 50
vidgen video "a man in a sunny kitchen" --seconds 30 --beats "0: he walks in and fills the kettle; 10: he sits down at the table; 20: he looks out of the window"
vidgen video "mirabel on a pier at sunset" --seconds 50 --avatar mirabel --anchor-every 3
```

A take over 5 s is a chain of 5 s Wan segments; each segment starts from the previous one's last frame, so it is one continuous shot.

- **Timeline:** each segment has its own prompt. With `--beats` (or `--beats-file F`, one `SECONDS: text` per line) segment *i* (starting at 5×*i* s) uses the last beat at or before its start. Rules: 1-12 beats, the first at 0, increasing, times in multiples of 5 and before the end, each 1-500 chars. Write each beat as a full description of the shot; with no avatar and no Krea first frame, segment 1 also gets your prompt. Without beats, takes over 15 s get a timeline written by the planner model (same subject, place, light and camera; only the action moves on); if the planner fails, every segment uses your prompt. The job result lists `segment_prompts`.
- **Plan first:** `vidgen plan "PROMPT" --seconds 50 [same options as video]` makes the first frame and the planner's beats (no video; ~2 min plus the render box starting), saves `<name>.beats.txt` and `<name>.keyframe.png`, and prints the command that renders them: `vidgen video "PROMPT" --seconds 50 --seed S --beats-file <name>.beats.txt` (the seed gives the same first frame). Edit the beats file in between. Dashboard: "Plan first" (takes over 15 s) fills the Beats box and the Seed field and shows the first frame.
- **The planner sees the first frame:** the first frame (Krea frame or avatar keyframe) is made before planning and shown to the planner, which keeps every segment to what is in it (no new people, animals, objects or effects). Without a first frame (Wan first frame, no avatar) it plans from the text.
- **Glow around people at golden hour:** a bright outline around hair and shoulders comes from backlight (the low sun behind the person), and grows as the camera moves in; it is not built up by chaining (tested: a front-lit 30 s take stays clean). For close or approaching shots, write side or front light ("the low sun to her left", "soft daylight on her face"); the negative prompt has no effect with Lightning.
- **Colour:** takes over 10 s pull each segment towards segment 1's colours, ramping from none at the seam to 70% on its last frame, so colours don't drift.
- **Face re-anchoring:** `--anchor-every N` (2-6; takes over 10 s with exactly one avatar) restores the avatar's face and hair on the frame that starts segments N, 2N, ... with the image editor. Off by default; each anchor costs ~2-4 min on the g6e.4xlarge (Wan is swapped out and back in) and may show at that seam.
- **Resume:** finished segments are saved in `/opt/ai-video/work/<job_id>/` on the render box. If the box is interrupted (spot) the job comes back after the box restarts and renders only the missing segments; the folder is removed when the job ends (folders left by jobs that never came back are swept after 3 days).
- **Progress and time:** the job shows "segment N/M". Roughly 5 min per segment plus smoothing and sound: about 55 min for 50 s, more with `--best` or `--upscale`. Variations work only with takes of up to 5 s.
- **Limits:** motion can change pace at the 5 s seams (each segment continues from a single frame); memory for a 60 s take with 32 fps and upscale is high (check on the box before relying on it).

### Krea 2 (second image model)

Krea 2 Turbo (open weights, Krea 2 Community License) is a text-to-image model, so it's offered where no particular person has to be kept:

```bash
vidgen avatar refresh --name mira --describe "a woman in her 30s, short dark hair, green raincoat" --model krea2
vidgen photos --no-avatar --prompt "a misty harbour at dawn, fishing boats" --count 4 --look film
vidgen video "waves rolling onto a black sand beach at sunset" --no-avatar --image-model krea2
vidgen story my_story.txt --image-model krea2      # first frames for the shots without avatars
```

- Avatar photos, series and keyframes stay on Qwen-Image-Edit (Krea 2 can't take reference images); video stays on Wan. With `--image-model krea2` Krea 2 makes the first frame and Wan animates it.
- **Content filter** (required by the license): every Krea 2 image is checked by `Falconsai/nsfw_image_detection`; a flagged image is re-rolled once with another seed and dropped if flagged again. Job results and the worker log show how many were dropped and the seconds for loading, generating and filtering. It can't be turned off.
- **Setup, once:** accept the terms at https://huggingface.co/krea/Krea-2-Turbo, create a read token, and store it (replace the value):
  `aws ssm put-parameter --region us-east-1 --name /ai-video/hf_token --type SecureString --value "hf_…"`. The first Krea job then installs its own Python environment (`/opt/krea-venv`, a few minutes) and gets ~36 GB of weights (from the [model mirror](#model-mirror) when it has them, else Hugging Face); later jobs only load it (~1-2 min) and the worker reloads Wan/Qwen afterwards.
- **License:** commercial use of the model and its outputs is allowed only for organisations under $1M annual revenue; above that you need a Krea enterprise license. Check before using it for work.
- Dashboard: Photos → Avatar "no person: Krea 2"; Video/Story → Realism → "First frame without an avatar"; Create → Avatar → **Generate candidates** (model Krea 2 or Wan), then **Pick N** in History.

### Writing a story (`vidgen write`)

```bash
vidgen write "A lighthouse keeper finds a message in a bottle addressed to her, dated tomorrow." --words 600 -o lighthouse
vidgen write "two rival food-truck owners stuck in a blizzard" --genre "romantic comedy" --seed 4
vidgen story ~/Movies/vidgen/lighthouse.txt --shots 6        # then film it (edit the .txt first if you like)
```

- Text only: the story planner's model (Qwen2.5-VL-7B) on the render machine writes it; no video is made. About a minute, plus loading the model if the machine was idle.
- Downloads as `<name>.txt` (title line, blank line, story). `--words` 100-2000 (default 800; lengths are approximate, often 20-40% off either way), `--genre` sets genre and tone, `--seed` repeats a version.
- Dashboard: **Create → Write**; when it has downloaded, **Use in Story** in History puts the text into the Story form.
- It's a 7B model: fine for short, simple stories; expect plain prose rather than polished writing.

### Video from a text story

```bash
vidgen story my_story.txt --shots 6 --seconds-per-shot 5 --style "watercolor animation" -o story.mp4
vidgen story my_story.txt --avatar star,robot --voice bm_george
cat my_story.txt | vidgen story - --no-narration
```

Downloaded files:

| File | Contents |
|---|---|
| `story.mp4` | shots stitched together, with burned-in subtitles, spoken narration and generated ambient sound underneath |
| `story.subtitles.srt` | the subtitles on their own |
| `story.narration.wav` | the narration track on its own |
| `story.sound.wav` | the generated ambient sound on its own |
| `story.plan.json` | the shot list: prompts, narration lines, which avatars are in each shot |

**Narration:**
- Each shot has one narration line, spoken by a local Kokoro voice.
- A line too long for its shot is sped up (at most 1.5×), then cut off. Keep stories so that one short sentence per shot carries them.
- Voices: `af_heart` (default), `af_bella`, `af_nicole`, `am_adam`, `am_michael` (US English), and `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis` (UK English).
- `--no-narration` leaves only the ambient sound; add `--no-sound` too for a silent video. `--no-subtitles` removes the burned-in text.

**Sound:** every `video` and `story` clip gets ambient sound and Foley generated from its frames by MMAudio, steered by the shot's prompt (in stories it sits at 30 % under the narration). It adds roughly 1 min per job for the model swap plus a few seconds per clip; the first sound job after this change also downloads ~4 GB of weights. `--no-sound` turns it off.

**A shot came out wrong?** Take its prompt from `plan.json` and regenerate just that clip with `vidgen video … --avatar <the shot's characters>`.

### Photos of a person in different places

```bash
vidgen photos --image me.jpg \
  --prompt "reading a newspaper at a Paris café, golden hour, shallow depth of field" \
  --location "a Tokyo street at night" --count 4 -o trip
vidgen photos --avatar dana --count 6 --aspect landscape     # 6 random locations
vidgen photos --prompts-file scenes.txt                       # one prompt per line, # comments allowed
vidgen photos --avatar dan --where "on a rainy Tokyo rooftop at dusk" --wearing "a navy wool suit" \
  --doing "walking toward the camera with an umbrella" --camera "low angle, wide lens"   # one photo from fields
```

- **Who:** `--image FILE` (png/jpg/webp) or `--avatar NAME`; with neither, your default avatar. Use a sharp, well-lit photo of one person.
- **Where:** each `--prompt` (free-form: place, action, mood, light, clothing) or `--location` (just a place) is one photo, in the order given; `--prompts-file` lines come after them. `--count` tops up with random locations from a curated list of 50 (no repeats); `--seed` makes the picks and renders repeatable.
- **Shot files:** `--shots FILE` makes one photo per block of `where:` / `wearing:` / `doing:` / `camera:` lines (blank line between photos, `#` comments, max 8 per job); errors name the line.
- **Fields:** `--where`, `--wearing`, `--doing`, `--camera` (any of them) build one photo, placed first. `--wearing` replaces the avatar's outfit (face, hair, skin tone and body are still kept); without it the outfit stays. `--camera` gets its own sentence ("Camera: …"): angle, height, distance, lens, e.g. `from behind`, `overhead`, `close-up at eye level`. A `--where` that starts with a preposition ("on", "at", "by", …) is used as written, otherwise "in" is put in front.
- **Look:** photorealistic (full-frame 50 mm look, natural light matching the scene, real skin texture). The person's face, hair, skin tone, body and outfit are kept unless your prompt asks for something else.
- **Output:** a folder `<download dir>/<-o name or photos-<job_id>>/` with `photo_1.png … photo_N.png` and `prompts.txt` (which prompt or place each photo came from). `--aspect portrait` (896×1152, default), `landscape` (1152×896) or `square` (1024×1024).
- **Time:** ~40 s per photo on `g6e.4xlarge` (Qwen-Image-Edit-2511 with the 8-step Lightning LoRA), plus 3–5 min to load the edit model when it is not already loaded.
- **Series in one place:** `--series --where "…" [--wearing "…"] [--doing "…" --doing "…"] --count 2-8` (default 4): every photo in that place and outfit, pose and camera varied for you (your `--doing` actions are used first, in order). Photo 1 is a full-body shot that shows the place. Close and medium shots get it as a reference (for the place's look, light and outfit only, not its framing), which leaves 2 of the avatar's images for them instead of 3; wide, side and from-behind shots don't, so their camera really moves. `--no-anchor` turns that off (a bit more likeness, the background drifts more).
  ```bash
  vidgen photos --avatar dan --series --where "a sunny café terrace in Lisbon" --wearing "a linen shirt" \
      --doing "reading a menu" --doing "pouring coffee" --count 6 --look phone --natural-skin
  ```
- **Realism** (photos, video and story): `--look pro|phone|candid|portrait|film` (pro is the default, unchanged look; phone and candid usually read as the most real), `--natural-skin` (skin texture, no retouched look), `--candid` (unposed, not looking at the camera), `--grain` (real film grain added to the finished photo or every video frame), `--best`. For photos `--best` runs the full 40-step model with CFG, so the anti-plastic-skin negative prompt applies: ~2.2 min per photo instead of ~45 s. For video and story `--best` only makes the avatar keyframes with the full model (~+1.5 min each); the video model keeps Lightning. `--best` and `--draft` exclude each other.
- **Speed mode:** like video, photos and avatar keyframes use a distilled Lightning LoRA (Apache-2.0) without CFG, so the negative prompt has no effect. `EDIT_LIGHTNING` in `/etc/ai-video.env` on the box picks it: `8` (default), `4` (~30 s per photo, slightly harsher colours) or `0` (full 40-step model with CFG, ~2.2 min per photo); then `sudo systemctl restart ai-video-worker`. Terraform variable: `edit_lightning` (only used at first boot).

### Dashboard (`vidgen ui`)

```bash
vidgen ui                  # opens http://127.0.0.1:8765; Ctrl-C to stop
vidgen ui --port 9000 --no-browser
```

- **Server**: instance state and type, uptime, queue, uptime hold, Wake button; the deployed models (T2V, I2V, edit, VLM), Lightning repo and revision, MMAudio variant, and the worker code version and deploy time. Refreshes every 15 s.
- **History**: your jobs from the last 30 days, newest first: prompt and settings, status, **Waited** (queued to started) and **Render time** (started to finished, including model loads), totals and average. Filter by type or search prompts. **Re-run** queues the same job again (same settings and seed) and the watcher downloads it next to the original; **Show** reveals a downloaded file.
- **Restart worker**: next to Wake GPU, while the machine is running. It asks the render worker to restart between jobs (within ~20 s when idle, else after the running job; a render in progress is never cut off); the worker then pulls the latest worker code and is back in about a minute. Use it after `make apply` when worker code changed. `vidgen restart-worker` does the same. It can't revive a worker that is hung or not running: for that, `sudo systemctl restart ai-video-worker` on the box.
- **Ready to render**: under the render server's state, a bar with a percent and one line saying how far the machine is from taking a job: off, starting, the first-boot step it is on (packages, models, converting Wan, Helios), or ready. First boot and the worker write it to `status/render.json`; `vidgen status` shows the same as `readiness`. A box set up before this existed reports nothing during its first boot, so the line is an estimate from the time so far until its worker starts.
- **Photo viewer**: in a job's preview, **Full screen** under a photo (or a click on it) opens it full screen; **Previous** / **Next** or the arrow keys go through all of the job's photos, **Close** or Esc leaves.
- **Create**: forms for video, photos and story (submitted like CLI jobs and downloaded by the watcher), each with a **Realism** section (look, natural skin, candid, film grain, best); Photos also has **Series in one place**. **Write** writes a story from a premise; **Use in Story** in History loads it into the Story form. And **Avatar**: name, optional description and 1-12 photos of the same person (png/jpg/webp, up to 25 MB each). Thumbnails show the order; ◀ ▶ reorder and ✕ removes, and photo 1 is the main image. "Make it my default avatar" is off unless you have no avatar yet. An existing name is replaced (its previous version goes into history). Same as `vidgen avatar refresh --name N --image … --image …`. Each avatar card has **Make default** and **Delete**: Delete asks first, removes the avatar's images, earlier versions and media from the server (no undo; refused while a job that uses it is queued or running) and can also delete its own LoRAs and their training photos (unticked by default). Same as `vidgen avatar delete`.
- **Learning**: mark a folder (Choose… opens the macOS folder picker) as a theme: name, kind (character, style, motion, place) and trigger word. The folder is scanned against the clips each kind needs and problems are listed; themes live in `~/.vidgen/themes.json`. **Upload** copies the theme's clips and photos to your private S3 bucket (see "Learning datasets" below) with a progress bar; the card then shows "up to date" or "changed since upload". Train arrives with the trainer.
- **Security**: the page is only served on 127.0.0.1, rejects other Host headers, and its API calls need a per-session token, so other websites cannot use it.

### Learning datasets (`vidgen dataset`)

```bash
vidgen dataset add singing ~/Movies/singing --kind motion     # same as Upload in the dashboard
vidgen dataset add dana ~/Movies/dana --kind character --trigger vgn_dana
vidgen dataset list
vidgen dataset rm singing                                      # deletes every uploaded file
```

- **What goes up**: every mp4, mov, webm, mkv, png, jpg, jpeg and webp under the folder (subfolders too, hidden files skipped), 1 B - 2 GB each, to `s3://<bucket>/datasets/<client>/<name>/raw/`, plus a `manifest.json` (kind, trigger word, file list) the trainer will train from.
- **It's a sync**: running it again uploads only new or changed files. Files you removed from the folder are listed and deleted from the dataset only after you confirm (`--yes` skips the questions).
- **Consent**: the first upload of a **character** dataset asks you to confirm you have the consent of the person shown; the API refuses it otherwise.
- **Retention**: no automatic expiry yet; a dataset stays until `vidgen dataset rm` or Remove in the dashboard (which asks first). Datasets contain faces: delete the ones you no longer need.
- **How much**: see the per-kind guidance in the dashboard (e.g. motion 20-60 clips of 2-5 s). A long video counts as one clip for variety; it is cut into segments only at training time.

### Training a LoRA (`vidgen train`)

```bash
vidgen dataset add singing ~/Movies/singing --kind motion   # upload first (or Upload in the dashboard)
vidgen train singing                  # 1250 steps per expert, ~9 h, ~$18 on spot
vidgen train singing --steps 600      # quick first try, ~4.5 h
vidgen lora list                      # trained versions, steps, checkpoints
```

- **Where it runs**: the trainer machine (`training_enabled = true` in `terraform.tfvars`), a separate spot g6e.4xlarge. The render box is not involved; both can run at once. The trainer starts when a train job is queued and stops itself 15 min after its queue is empty. Its very first start installs musubi-tuner and gets ~85 GB of weights (from the [model mirror](#model-mirror) when it has them, else Hugging Face, ~35 min).
- **What a job does**: downloads the dataset from its manifest, cuts clips into 5 s segments (at most 12 per source video, spread over the clip; photos are used as single frames), captions every segment with Qwen2.5-VL (trigger word first), caches latents and text, trains the T2V **low-noise** then **high-noise** expert (fp16 + fp8 base, rank 32), converts and uploads. Progress shows in the dashboard (card and History) and `vidgen jobs`, e.g. "low-noise expert 600/1250 · ~6 h 40 min left".
- **Result**: `s3://<bucket>/loras/<client>/<name>/v<N>/`: `t2v_low.safetensors`, `t2v_high.safetensors`, `checkpoints/` every 250 steps, `lora.json`. Each training makes a new version; `current.json` points at the newest.
- **Rendering with it**: `vidgen video "..." --lora NAME[:WEIGHT]` (also `story`; repeat `--lora` for a second one, max 2; weight 0–1.5, default 1.0). The API pins the current version when the job is submitted (a re-run uses the same one), the worker downloads both expert files once per version to `/opt/models/loras/` and loads them next to Lightning (high-noise into `transformer`, low-noise into `transformer_2`). The LoRA's trigger word is put in front of every clip prompt when missing. The job result lists the LoRAs used.
- **Adding a LoRA file made elsewhere**: `vidgen lora add NAME --file style.safetensors [--trigger WORD]` (one file, used for both Wan 2.2 experts: Wan 2.1 14B and single-file 2.2 LoRAs) or `--high H.safetensors --low L.safetensors` (a Wan 2.2 pair). It becomes a new version of NAME, set as current unless `--keep-current`, and works like a trained one (`--lora NAME`, pickers, `lora use`). Before it can be used it's checked: a real .safetensors LoRA for Wan, with the 14B models' layer sizes (Wan 1.3B/5B, Flux, SD, SDXL and Qwen-Image LoRAs are refused with the reason), at most 2 GB. Dashboard: Learning → **Add a LoRA file** (Choose… picks the file; it's uploaded from your Mac). Refused uploads stay in S3 invisibly (the API can't delete).
- **Krea 2 LoRAs** (image model): `vidgen lora add NAME --file krea_style.safetensors [--trigger WORD]` detects a Krea 2 LoRA from the file (musubi-tuner/kohya `lora_unet_blocks_N_attn_*` or original `krea-ai/krea-2` names; 6144-wide layers) and stores it as one file (`v<N>/krea2.safetensors`, `lora.json` `models: ["krea2"]`); `lora list` shows `krea2`. A name holds one kind: a Krea file under a Wan name (or the reverse) is refused, so pick another name. Krea 2 LoRAs are one file (no `--high/--low`) and have no checkpoints (`lora use NAME --version N` works, `--step` and `lora preview` don't).
  - **Where they apply**: the same `--lora` list is split by kind. Krea 2 LoRAs go to the Krea 2 step: `vidgen photos --model krea2 --lora NAME[:WEIGHT]` (photos without a person), and the Krea first frame of `video`/`story` with `--image-model krea2` (stories: shots without avatars; if every shot has avatars the result says `krea_loras_unused`). Wan LoRAs in the same list still go to Wan. Max 2 per kind, weight 0–1.5, the trigger word is put in front of each Krea prompt. Refused with a reason: a Krea LoRA on qwen photos, avatars, or a video/story without `--image-model krea2`; a Wan LoRA on Krea photos.
  - **Exact prompt** (`vidgen photos --model krea2 --raw --prompt "..."`, dashboard "Exact prompt"): Krea 2 gets each prompt word for word. Normally every Krea photo prompt gets a look sentence ("Photorealistic photograph, 50mm lens … realistic skin texture …", plus portrait/skin wording with those options) and LoRA trigger words, which pull it toward people and faces. With `--raw` nothing is added: write the framing yourself ("wide shot", "full body", "no people") and type the LoRA's trigger word if it needs one. Extra photos (`--count`) repeat your prompts with new seeds instead of random places. `--look`/`--natural-skin`/`--candid` are ignored; `--grain` still applies after generation.
  - **Content filter**: unchanged. Every Krea 2 image, with or without a LoRA, is checked; a LoRA that pushes images toward what the filter flags gets those images dropped (the job fails if all are).
  - **Box**: LoRA loading needs `peft` in `/opt/krea-venv` (pinned in `worker/krea_requirements.txt`; changing that file reinstalls the venv on the next Krea job, a few minutes). A file diffusers can't map fails the job with "LoRA <adapter> could not be loaded (<error type>)"; the detail is in the worker log. LoRAs trained on Krea 2 Raw may look weaker on Turbo (8 steps, no CFG).
  - Dashboard: Video, Story and Photos take several LoRAs ("+ Add LoRA": one row per LoRA with its weight; up to 2 Wan + 2 Krea 2, or 1 more Krea 2 next to an avatar's Krea LoRA). Learning → Add a LoRA file takes Krea 2 files too; Create → Photos with Avatar "no person" shows a Krea 2 LoRA picker; Video/Story pickers list Krea 2 LoRAs once "First frame" is Krea 2.
- **Picking a checkpoint**: `vidgen lora use NAME --version N [--step S]` sets what `--lora NAME` renders with (checkpoints every 250 steps; no `--step` = final weights); `lora list` shows the pick. One job can override it: `--lora NAME@750`, `NAME@v1`, `NAME@v1-750:0.8`. The dashboard's Learning tab has the same buttons per version.
- **Previews**: `vidgen lora preview NAME [--version N] [--steps 500,1000] [--prompt TEXT]` renders a 2 s 832x480 clip per checkpoint (default: the last 5) plus one without the LoRA, same prompt (trigger word included) and seed, into `lora-preview-<job_id>/`. About 1 min per clip once the model is loaded (~8 min cold). Also the Preview button in the dashboard.
- **With an avatar**: avatar shots are animated by the I2V model, while the trainer makes T2V LoRAs. The T2V LoRA is loaded into I2V as well (same layer shapes); how well it carries over is still to be measured. If the face drifts, lower the weight (e.g. `:0.6`).
- **Spot interruptions**: the job's folder stays on the trainer's disk with saved state every 250 steps; when capacity returns the job resumes where it stopped ("resumed at 500") instead of starting over.
- **Cancel**: `vidgen cancel <job>` stops it within ~15 s; nothing is uploaded.
- **Limits**: steps 100-3000 per expert; T2V only for now (I2V later); a train job times out after `train_job_timeout_minutes` (960) and the waker stops the trainer after `train_max_uptime_hours` (24).
- **Trainer logs**: `terraform output -raw trainer_ssm_shell`, then `journalctl -u ai-video-trainer -n 200` and `/opt/trainer/jobs/<job_id>/*.log` while a job runs.

### Description for a video

```bash
vidgen describe market.mp4 --instructions "audience: YouTube Shorts, playful tone"
```

This saves `<job_id>.json` (or `-o name.json`) with `title`, `description` and `tags`.

### Raw HTTP (no CLI)

Every request needs three `x-vidgen-*` headers, signed over:

```
vidgen-v1\n<METHOD>\n<path>\n<unix-ts>\n<nonce>\n<sha256-hex-of-body>
```

The signing command is `ssh-keygen -Y sign -n vidgen-api -f <key>`. The armored output, base64-encoded, goes in `x-vidgen-signature`. See `sign()` in `cli/vidgen.py`. The routes are listed at the top of `lambda/api/handler.py`.

---

## 3. Limits

### Jobs

| Limit | Value |
|---|---|
| Video length | 1–60 s per take, 16 fps rendered (32 fps after smoothing); over 5 s is a chain of 5 s Wan passes (81 frames each), stories stay at up to 5 s per shot |
| Prompt / story length | 2,000 / 20,000 chars |
| Story shots | 1–12, each 2–5 s, so at most 1 min |
| Photos per job | 1–8; each prompt/location 1–500 chars; one person per job |
| Avatars per job | at most 3 (keyframe model limit). The more in one frame, the weaker each one's likeness |
| Describe input | mp4, mov, webm or mkv. Sampled at 1 fps, ≤448×448. Keep it under ~2 min to avoid running out of GPU memory |
| Concurrency | **one job at a time**. Queue order is roughly FIFO, but not guaranteed |
| Job timeout | `job_timeout_minutes` (360). A longer job is stopped and marked `failed: timed out` |
| Cancel latency | queued: immediate. Running: ~15 s (one check, then the current denoising step) |
| Failed jobs | not retried |
| Interrupted jobs (spot) | rerun from the start. After 3 interruptions the job goes to the dead-letter queue |
| Narration | English voices only (other Kokoro languages need extra packages not installed). Long lines are sped up at most 1.5×, then cut |
| Avatar consistency | strong for face, hair and outfit, but not pixel-perfect. Small details (jewellery, logos, hands) can vary |
| Avatar storage | any number of avatars. Names: 1–32 chars (`a–z 0–9 - _`). Avatars and their history **never expire** |

### Timing

Measured with `make smoke` on `g6e.4xlarge` (Wan2.2 A14B, bf16). Each step includes any model load it triggered, so the order matters:

| Smoke step | Full 40-step (2026-11-08) | Lightning (2026-11-08) | Lightning, model-grouped order (2026-11-09) | + photos step (2026-11-09) |
|---|---|---|---|---|
| avatar (2 candidates, incl. first T2V load) | 8.1 min | 6.0 min | 6.0 min | 5.8 min |
| video (2 s + sound) | 16.6 min | 3.2 min | 1.9 min | 3.1 min |
| story (2 shots × 2 s) | — | 19.4 min (two avatars) | 6.3 min (no avatars) | 5.0 min |
| avatar-vid (keyframe + 2 s + sound) | 26.2 min | 11.5 min (one avatar) | 12.6 min (two avatars) | 8.8 min |
| photos (3 × 896×1152, incl. edit-model load) | — | — | — | 9.9 min |
| describe | 1.1 min | 1.1 min | 1.0 min | 0.1 min |
| cancel | — | failed (10 s clip, out of memory) | passed (5 s clip) | passed |
| **whole run** | — | 43.8 min | **31.2 min** | **34.3 min** |

The queue is not strictly FIFO, so the worker may still run a job out of the planned order (on 2026-11-09 the cancel test ran last and reloaded T2V for ~3 min before it was cancelled).

| Step | Expect |
|---|---|
| Machine off → first job starts | 2–4 min, plus the first model load |
| Rendering a 2 s clip (Lightning) | ~1.5 min; most of a 4-step render is moving the two 14B experts on and off the GPU |
| Sound | ~10–20 s per clip (MMAudio stays loaded) |
| Switching models | on `g6e.4xlarge`: ~5 min per switch (reload ~60 GB from disk); on 8xlarge/16xlarge all three stay in RAM, and an idle worker loads the ones not used yet (`preloaded` in the worker log) |
| Describe a 1-minute video | ~1 min |
| Photos | ~40 s per photo (Lightning 8 steps), plus the edit-model load |
| Smoke test | ~34 min on a warm `g6e.4xlarge` (jobs are queued grouped by model) |
| Machine stops after the last job | `idle_minutes` (15) |

### Cost guards

| Guard | What it does |
|---|---|
| Idle stop | the worker stops the machine after 15 min with an empty queue |
| Job timeout | stops any single job after 6 h |
| **Max uptime** | the waker Lambda (outside the machine) stops it after `max_uptime_hours` (8) of continuous running, *whatever it's doing*. It then **holds**: no automatic restart until you submit a job or run `vidgen wake`, so a stuck job can't keep restarting it. `vidgen status` shows `uptime_hold: true` |

### Access and security

| Limit | Value |
|---|---|
| Who can call the API | only holders of a private key listed in `authorized_ssh_public_keys` (ed25519 only) |
| Request freshness | signature timestamp within ±5 min. Keep your clock synced |
| Replay | each nonce is single-use |
| Rate limit | 10 requests/s, bursts of 20 |
| Result links | presigned, valid for **1 hour**. Anyone holding a link can download that file until it expires |
| Uploads | each key can only use its own uploads |
| Data retention | outputs, uploads and job records: 30 days. Avatars: kept until you delete them |
| Open ports on the GPU machine | none. SSH goes through SSM |

**What the SSH key does *not* protect against:** anyone with enough IAM permissions in this AWS account can bypass the API (SSM, SQS, S3, the Lambda). Keep this in an AWS account only you control.

### Infrastructure

| Limit | Notes |
|---|---|
| Spot capacity | a stopped spot instance can only restart in its own AZ. If that AZ has no `g6e` capacity, jobs wait |
| Instance size | the render box is a g7e (one RTX PRO 6000, 96 GB VRAM), `g7e.4xlarge` (128 GB RAM) by default; Terraform refuses other families. See [Moving the render box to a g7e](#moving-the-render-box-to-a-g7e). The trainer stays on `g6e.4xlarge` |
| Disk | 500 GB; weights ~210 GB after first boot converts the two Wan2.2 A14B models to local bf16 copies in `/opt/models/bf16` (the fp32 downloads are ~118 GB each and are deleted) |
| Model licences | Wan2.2, Qwen2.5-VL, Qwen-Image-Edit-2511 (and its Lightning LoRA) and Kokoro are Apache-2.0. **MMAudio weights (generated sound) are CC-BY-NC 4.0: videos with sound are for non-commercial use only**; use `--no-sound` for anything commercial. Check again if you swap models |
| Content | no moderation layer; you're responsible for what is generated. No sexual content, no real people's likenesses without consent, nothing involving minors |

---

## 4. Operations

### Health and logs

| What | Command |
|---|---|
| Instance, queue, hold | `vidgen status` |
| End-to-end check | `make smoke` |
| API, auth and waker log (`auth_failed`, `max_uptime_stop`, `start_failed`) | `aws logs tail /aws/lambda/ai-video-api --follow` |
| Worker log (`start`/`done`/`cancelled`/`failed` events with seconds; `pipeline_timing` per model call with seconds per step, `video_timing` per video with seconds for render, colour, interpolate, upscale and encode) | `eval "$(terraform -chdir=terraform output -raw ssh_command)"` → `journalctl -u ai-video-worker -f` |
| Watcher log | `~/.vidgen/watcher.log` |
| GPU use | on the box: `nvidia-smi` |
| Dead-letter queue | `aws sqs get-queue-attributes --queue-url $(aws sqs get-queue-url --queue-name ai-video-jobs-dlq --query QueueUrl --output text) --attribute-names ApproximateNumberOfMessages` |
| Spot price | `aws ec2 describe-spot-price-history --instance-types g6e.4xlarge --availability-zone us-east-1a --product-descriptions Linux/UNIX --max-items 1` |

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `HTTP 401 unauthorized` | Check the Lambda log. `key not authorized`: key not in tfvars. `InvalidSignature`: something changed the body or path. Nothing logged: clock skew or missing headers |
| `signing failed …` | key not loaded in the agent; see 1.2 |
| `stopped (InsufficientInstanceCapacity, will retry)` | no spot capacity. The job waits and the waker retries every minute. If it lasts hours, change `availability_zone` (replaces the instance; models re-download) |
| `uptime_hold: true`, jobs not starting | the 8 h cap fired. Check the worker log for what ran so long, then `vidgen wake` |
| Job `failed: timed out after 360 min` | the job hit `job_timeout_minutes`. Split the story, or raise the limit |
| Job stuck `queued`, instance `running` | worker down: `systemctl status ai-video-worker`, `journalctl -u ai-video-worker -n 200` |
| `CUDA out of memory` | describe input too long, or too many seconds/too high a resolution. Avatar jobs on an instance smaller than 4xlarge also do this. Clips over 5 s are clamped for this reason |
| Smoke step fails right after a deploy | usually a library version or model API mismatch. The traceback in the worker log names the call; fix it in `worker/models.py` or `worker/requirements.txt`, `make plan && make apply`, restart the worker, rerun `make smoke` |
| Watcher not downloading | `vidgen watcher status`. On Linux, check `systemctl --user list-timers`; on Windows, check that the Startup-folder launcher exists and `pythonw` is running |

### Model mirror

A copy of the weights both machines use is kept in your bucket under `models/`, so a machine with a new disk (another instance type or AZ, spot ↔ on-demand) restores them from S3 instead of downloading from Hugging Face and converting Wan to bf16 again. Design: [docs/specs/2026-11-25-model-mirror.md](docs/specs/2026-11-25-model-mirror.md).

- **What it holds**: render box: the bf16 Wan copies, the Hugging Face cache folders (description/planner model, Qwen-Image-Edit, Lightning), MMAudio, Krea 2 Turbo and its content filter. Trainer (all under `trainer/`): the Wan training files (`trainer/wan/<pins>`), the Krea 2 training files (`trainer/krea2/<pins>`; the folder name comes from the pinned commits, so a bumped pin starts a new folder) and its own copy of the caption model. About 420 GB, roughly $10 a month in S3.
- **It fills itself**: every time the worker or the trainer starts, it queues what is complete on its disk and uploads only what the bucket lacks, in the background (one folder at a time; `{"event": "mirror_push", ...}` in the log). The first start after this is deployed uploads everything (the render box reads ~210 GB off its disk while it works); a stop in the middle is fine, the next start continues. A folder counts only once its manifest (`.mirror-manifest.json`, written last) is there.
- **It is used**: at first boot (render box: every folder in the mirror; trainer: Wan files and caption model) and on the first Krea job of a disk (both machines). Anything not in the mirror comes from Hugging Face as before and is then uploaded.
- **How fast it was**: `{"event": "model_download", "unit": ..., "source": "mirror" | "huggingface", "bytes": ..., "seconds": ..., "mb_per_s": ...}` in the worker/trainer log; a Krea LoRA build that fetched the models also has `krea_download` in its job result. Restores are checked by file size.
- **Check it**: `aws s3 ls s3://<bucket>/models/ --recursive --summarize | tail -2` (size), `aws s3 ls s3://<bucket>/models/ --recursive | grep mirror-manifest` (the folders that are complete).
- **Remove a folder** (an old Krea revision, a model you no longer use): `aws s3 rm --recursive s3://<bucket>/models/<folder>/`. A machine that still has it on disk uploads it again at its next start.
- **Turn it off** on a machine: `MODEL_MIRROR=0` in `/etc/ai-video.env` (render box) or `/etc/ai-video-trainer.env` (trainer), then restart the service.
- **Access**: only the two machine roles, each to its own folders (the trainer to `models/trainer/` only), so neither machine restores anything the other wrote. The Krea weights are gated on Hugging Face: this copy is for these machines only.
- **Idle stop**: a machine doesn't stop itself while it is uploading to the mirror (or, on the trainer, while the Krea models are still downloading); the max-uptime guard still applies.

### Common changes

| Task | How |
|---|---|
| **Rotate your key** | new key → replace the line in `terraform.tfvars` → `make plan && make apply`. The API switches immediately; the worker's SSH keys update on its next start |
| Add another laptop | add another `ssh-ed25519 … comment` line. It gets its own client name, jobs and avatars |
| Idle timeout / job timeout | `idle_minutes` / `job_timeout_minutes`, then apply. Both live in `/etc/ai-video.env`, which is written at first boot: on the box, edit that file and `sudo systemctl restart ai-video-worker` |
| Max uptime | `max_uptime_hours`, then apply. Takes effect immediately (Lambda) |
| Update worker code | edit `worker/` → `make plan && make apply` → pulled on the worker's next start: **Restart worker** in the dashboard / `vidgen restart-worker` (or `sudo systemctl restart ai-video-worker` on the box) → `make smoke` |
| Swap models | change `video_model` / `i2v_model` / `vlm_model` / `edit_model` (`edit_lightning` needs a 2511 edit model). On the box: update `/etc/ai-video.env`, `huggingface-cli download <repo>` (for a Wan video model, then `sudo systemctl stop ai-video-worker` and `cd /opt/ai-video/worker && sudo -E /opt/ai-video/venv/bin/python convert_bf16.py --delete-source` with the env loaded), restart the worker, `make smoke`. A different model family also needs code changes |
| Pause everything | `aws events disable-rule --name ai-video-waker`, then stop the instance |
| Tear down | `terraform -chdir=terraform destroy` (empty the S3 bucket first if Terraform refuses) |

### Trainer viability trial (LoRA training spike)

`trainer/spike/run_trial.sh` checks, on the render box, that LoRA training works before the trainer is built (see `docs/specs/2026-11-09-lora-training.md`). Run it only when the queue is empty; it pauses the render worker and resumes it at the end.

```bash
aws s3 cp trainer/spike/ s3://$(terraform -chdir=terraform output -raw bucket)/code/trainer-spike/ --recursive
# on the box (ssm_shell), as root:
aws s3 cp s3://$BUCKET/code/trainer-spike/ /opt/trial/ --recursive
nohup bash /opt/trial/run_trial.sh master 200 &
```

- **Pinned:** musubi-tuner is third-party code running as root on the box, so the script checks out a fixed commit (`MUSUBI_COMMIT`, currently `f8a1b037`). Bump it only after reviewing the diff.
- **What it does:** installs musubi-tuner in `/opt/trial/venv` (the worker venv is untouched), downloads the ComfyUI-format Wan2.2 T2V low-noise expert, VAE and umT5 (~40 GB into `/opt/trial/models`), trains a 200-step LoRA on the 12 most recent output clips, then renders the same 2 s prompt with and without the LoRA through the production loader (Lightning on).
- **Results:** `s3://<bucket>/outputs/<client>/trial-<timestamp>/`: `timings.txt` (time per phase, peak GPU memory, train speed, musubi-tuner commit), `render.json` (LoRA load details), `with_lora.mp4`, `baseline.mp4`, `trial.log`, `trial.safetensors`.
- **Clean up** afterwards if the trainer will not run on this box: `rm -rf /opt/trial` (frees ~45 GB).

### Choosing the instance

The render box is a **g7e** (one RTX PRO 6000, 96 GB of GPU memory); Terraform refuses any other family for it. Helios and a whole Wan or image-edit pipeline stay on that GPU, and the worker's and Krea's torch builds are the ones for it (they were validated there, not on the older L40S boxes).

| Instance | RAM | Notes |
|---|---|---|
| `g7e.4xlarge` (default) | 128 GB | what the tests ran on |
| `g7e.8xlarge` | 256 GB | more page cache: switching between big models reloads faster |

Capacity changes by the hour. On 2026-11-26 spot `g7e.4xlarge` never launched in us-east-1a/b/d; on-demand `g7e.4xlarge` usually did and twice did not; spot `g7e.8xlarge` launched twice when nothing else did. `make g7e` (next section) tries them in order by really launching. The trainer is a separate machine and stays on `g6e.4xlarge`.

### Moving the render box to a g7e

A g7e has a 96 GB GPU (RTX PRO 6000) against the L40S's 48 GB: both Wan experts stay on it, and a 5 s segment
took 104 s in the benchmark against ~240 s on the g6e that evening ([benchmark](docs/specs/2026-11-25-g7e-benchmark.md),
[design](docs/specs/2026-11-25-render-on-g7e.md)).

```bash
make g7e ARGS=--dry-run     # only report what could be launched right now
make g7e                    # find capacity and move the render box there
```

- **What it does**: checks first (AWS credentials; the worker's torch can run on that GPU; no job queued or running), says whether the [model mirror](#model-mirror) holds the weights, then looks for capacity by launching a probe instance and terminating it at once: spot `g7e.4xlarge` in each zone, then spot `g7e.8xlarge`, then on-demand `g7e.4xlarge`, then on-demand `g7e.8xlarge`. The first that starts wins. It writes `instance_type`, `on_demand` and `availability_zone` to `terraform/terraform.tfvars` and runs build, plan and apply.
- **It replaces the render box and its disk without asking again.** Jobs sent while the new box boots wait in the queue. First boot restores the weights from the mirror when it has them; **the first move has no mirror yet** (the old box never ran this code), so that first boot downloads everything from Hugging Face and converts Wan (~45 min), and the new box then fills the mirror. Everything in your working tree is deployed with it, as with `make apply`.
- **If the apply fails** (the capacity went in between) it tries the next option. If none works it puts `terraform.tfvars` back; the old box may already be gone then: `make plan && make apply` brings it back.
- **`--force`** goes on although a job is queued or the worker's torch pin looks too old.
- **Cost**: a probe is a few cents. On-demand `g7e.4xlarge` is about $4.00/h (spot ~$1.40-2.20 when there is any); the box still stops itself when idle.
- **On the box**: with 80 GB or more of GPU memory the worker keeps one big model wholly on the GPU (`GPU_HOLD`; `GPU_HOLD=0` in `/etc/ai-video.env` goes back to moving parts on and off). Only one big model is loaded at a time then, so switching models reloads from disk.
- **Another size or zone later**: run `make g7e` again. Going back to a g6e would mean taking out the g7e-only check in `terraform/variables.tf` and the torch pins in `worker/requirements.txt` and `worker/krea_requirements.txt` (those builds were only tried on the g7e).

### Cost model

- **Idle (machine off):** about $60/month for the 750 GB disk. Everything else costs cents.
- **Working:** `g6e.4xlarge` spot price × hours running, plus ~15 min of idle time after each burst.
- **Worst case:** bounded by the max-uptime cap, 8 h per start.

## 5. Command reference

Every command in one place. `vidgen` stands for `python3 ~/Documents/private/myproject/cli/vidgen.py`
(see 1.5 for an alias). Run any command with `--help` for the full flag list.

### Setup and machine

| Command | What it does |
|---|---|
| `vidgen configure --url URL --ssh-key ~/.ssh/vidgen_ed25519.pub --download-dir DIR` | save the API URL, signing key and download folder to `~/.vidgen/config.json` |
| `vidgen status` | GPU instance state, queue depth, uptime hold, deployed models |
| `vidgen wake` | start the GPU instance ahead of a batch (also clears an uptime hold) |
| `vidgen restart-worker` | ask the render worker to restart between jobs and pull the latest worker code (machine must be running) |
| `vidgen smoke [--skip-cancel]` | end-to-end test of every model on the GPU (same as `make smoke`) |
| `vidgen ui [--port 8765] [--no-browser]` | local dashboard: status, models, history, a Create tab (video, photos, story, avatar from photos), learning themes, LoRA checkpoints |

### Jobs and downloads

| Command | What it does |
|---|---|
| `vidgen jobs` | local list of jobs: pending, downloaded, where the files went |
| `vidgen cancel JOB_ID` | cancel a queued or running job (a unique prefix of the id is enough) |
| `vidgen check` | one pass: download any finished jobs (what the watcher runs) |
| `vidgen watch [--interval 45] [--until-done]` | check in the foreground until Ctrl-C (or until nothing is pending) |
| `vidgen watcher install\|uninstall\|status [--interval 45]` | background watcher (macOS, Linux, Windows) |

### Video and story

| Command | What it does |
|---|---|
| `vidgen video "PROMPT" [-o out.mp4]` | clip from text with Helios (30 s in one take unless `--seconds`), starring your default avatar |
| `  --video-model wan` | render with Wan instead: 5 s segments chained, with beats, plans, several avatars and LoRAs |
| `  --seconds 1-60 --width 1280 --height 704 --seed N --negative "TEXT"` | length, size, repeatable seed, negative prompt (no effect with Lightning) |
| `  --beats "0: walks in; 10: sits down" / --beats-file F` | long takes (over 5 s): what happens when; each 5 s segment follows the last beat before it (1-12 beats, first at 0, times in multiples of 5, each a full description). Without beats, takes over 15 s get a timeline written by the planner |
| `vidgen plan "PROMPT" --seconds N` | first frame + editable beats for a long take (no video); prints the render command with `--seed` and `--beats-file` |
| `  --anchor-every N` | takes over 10 s with exactly one avatar: before segments N, 2N, ... the frame the next segment starts from gets the avatar's face and hair restored (Qwen edit; on a g6e.4xlarge each anchor swaps Wan out and back in, ~2-4 min each); off by default |
| `  --variations 2-4` | several takes of the same prompt (seeds N, N+1, ...), saved as `<out>_1.mp4` ...; clips up to 5 s |
| `  --draft` | faster draft: render at 832x480, then 32 fps + 2x upscale to 1664x960 (~1.9x faster; the same seed gives a different scene than full quality) |
| `  --fps 32\|16 --upscale` | 32 fps (default, RIFE frame interpolation) or 16 as rendered; 2x Real-ESRGAN upscale to 2560x1408 |
| `  --avatar a,b \| --no-avatar` | star up to 3 named avatars instead of the default, or none |
| `  --no-sound` | no generated ambient sound (MMAudio is non-commercial) |
| `  --lora NAME[@[vN][-]STEP][:WEIGHT]` | apply a trained LoRA (max 2; weight 0-1.5): `jerk:0.9`, `jerk@750`, `jerk@v1-750:0.8` |
| `  --model krea2` (avatar refresh), `--no-avatar` (photos), `--image-model krea2` (video, story) | Krea 2 Turbo text-to-image where no person is kept; content filter on every image |
| `  --lora KREA_NAME[:WEIGHT]` with `photos --model krea2` or `--image-model krea2` | a Krea 2 LoRA for the Krea image (max 2; mixed with Wan LoRAs, each goes to its model) |
| `vidgen write "PREMISE"\|- [--words N] [--genre G] [--seed S] [-o NAME]` | write a short story (text only) to `NAME.txt` |
| `vidgen story FILE\|- [-o out.mp4]` | narrated multi-shot video from a text story (`-` reads stdin) |
| `  --shots 6 --seconds-per-shot 2-5 --style "TEXT" --voice af_heart` | shot count, length, look, Kokoro voice (af_heart, am_adam, bf_emma, bm_george, ...) |
| `  --no-subtitles --no-narration --no-sound --avatar a,b --no-avatar --lora ... --fps --upscale` | as for `video` |
| `vidgen describe FILE [--instructions "TEXT"] [-o out.json]` | title, description and tags for a video file |

### Photos

| Command | What it does |
|---|---|
| `vidgen photos [--avatar NAME \| --image FILE] [-o FOLDER]` | photorealistic photos of one person (default: your default avatar) |
| `  --series --where … [--wearing …] [--doing … ×N] [--no-anchor] --count 2-8` | a series in one place and outfit, pose and camera varied |
| `  --look pro\|phone\|candid\|portrait\|film --natural-skin --candid --grain --best` | realism options (also on `video` and `story`) |
| `  --shot stream\|pov` (video, plan) | filmed as a streaming session (fixed webcam, ring light), or looking shot on a phone (handheld shake, soft low resolution, grain; no phone in the picture); one or none |
| `  --prompt "TEXT"` (repeatable) | one photo per free-form scene |
| `  --location "PLACE"` (repeatable) | one photo per place |
| `  --where "..." --wearing "..." --doing "..." --camera "..."` | one photo built from fields; `--wearing` replaces the outfit |
| `  --shots FILE` | one photo per block of `where:`/`wearing:`/`doing:`/`camera:` lines |
| `  --prompts-file FILE` | one prompt per line |
| `  --count 1-8 --aspect portrait\|landscape\|square --seed N` | top up with random locations, size, repeatable seed |

### Avatars

| Command | What it does |
|---|---|
| `vidgen avatar list` | your avatars, image counts, the default (*), Krea LoRA state |
| `vidgen avatar krea-train --name N` | build (a new version of) the avatar's Krea 2 LoRA (~1.5 h GPU) |
| `vidgen avatar wan-train --name N` | build (a new version of) the avatar's Wan LoRA from its media (~1 h trainer); attached to its videos when ready |
| `  --no-avatar-lora` (video) | leave out the avatar's own Wan LoRA |
| `vidgen avatar media --name N` | photos + four 2 s clips of the avatar (made for every new avatar; this is for existing ones), copied to `<download dir>/avatars/N/` |
| `  --no-krea-lora` (refresh, pick) | don't build the avatar's Krea 2 LoRA |
| `vidgen avatar refresh --name N --image a.jpg [--image b.jpg ...] --describe "..."` | create or replace an avatar from 1-12 images (becomes the default) |
| `vidgen avatar refresh --name N --describe "..." [--candidates 4]` | generate candidates from a description, then `pick` |
| `vidgen avatar pick JOB_ID N` | save candidate N of an avatar job |
| `vidgen avatar add --name N FILE...` | add images to an avatar (max 12 in total) |
| `vidgen avatar images --name N` | list its images in order, with preview links (first 3 are used) |
| `vidgen avatar main --name N <n>` | make image n the main image |
| `vidgen avatar remove --name N <n>` | remove image n (an avatar keeps at least one) |
| `vidgen avatar use --name N` | make N the default avatar |
| `vidgen avatar history --name N` | earlier versions (every change saves one) |
| `vidgen avatar restore --name N VERSION` | bring back an earlier version |
| `vidgen avatar delete --name N [--with-loras]` | delete an avatar (no undo); `--with-loras`: its own LoRAs too |

### Learning datasets, training and LoRAs

| Command | What it does |
|---|---|
| `vidgen dataset add NAME FOLDER --kind character\|style\|motion\|place [--trigger WORD] [--yes]` | upload new or changed clips/photos of a theme (asks before deleting removed files) |
| `vidgen dataset list` | uploaded datasets with counts |
| `vidgen dataset rm NAME [--yes]` | delete a dataset from S3 |
| `vidgen train NAME [--steps 100-3000] [--speed standard\|fast]` | train a LoRA on the trainer machine (1250 steps per expert is ~12.5 h at standard; fast trains at 512x288). Checkpoint previews are queued automatically when it finishes |
| `vidgen train NAME --captions-only` | only cut and caption the dataset (minutes), to review the captions first |
| `vidgen dataset captions NAME [--edit]` | show the training captions, or edit them in `$EDITOR`; the next training uses them while the dataset is unchanged |
| `vidgen lora list` | versions, checkpoints, trigger words; `*` and `using ...` show what `--lora NAME` renders with; `[on this Mac]` = local copy |
| `vidgen lora pull [NAME] [--checkpoints]` | copy LoRAs to `<download dir>/loras/` now (the watcher does it every 10 min) |
| `vidgen lora add NAME --file F \| --high F --low F [--trigger W] [--description D] [--keep-current]` | upload a Wan 14B or Krea 2 LoRA file made elsewhere as a new version (kind detected from the file, checked first) |
| `vidgen lora use NAME --version N [--step S]` | pick the version or checkpoint renders use (no `--step` = final weights) |
| `vidgen lora preview NAME [--version N] [--steps 500,1000] [--prompt "SCENE"] [--seed N] [-o FOLDER]` | 2 s clip per checkpoint plus one without the LoRA, same prompt and seed |

### Deploy and infrastructure (from the project folder)

| Command | What it does |
|---|---|
| `make build` | build the Lambda bundle (needed before a plan, or Lambda changes don't show) |
| `make plan` | build + `terraform plan`, saved to `tfplan` |
| `make apply` | apply exactly the reviewed `tfplan` |
| `make output` | Terraform outputs (API URL, instance ids, shell commands) |
| `make smoke` | end-to-end GPU test |
| `make clean` | remove build files |
| `make test` | unit tests (no cloud, no GPU) |
| `terraform -chdir=terraform output -raw ssm_shell` | command for a root shell on the render box (`trainer_ssm_shell` for the trainer) |
| `eval "$(terraform -chdir=terraform output -raw ssh_command)"` | SSH to the render box through SSM |
| `aws logs tail /aws/lambda/ai-video-api --follow` | API and waker log |
| `aws events disable-rule --name ai-video-waker` | pause automatic starts (then stop the instance) |
| `terraform -chdir=terraform destroy` | tear everything down |

### On the render box (root shell)

| Command | What it does |
|---|---|
| `journalctl -u ai-video-worker -f` | follow the worker log |
| `systemctl status ai-video-worker` / `sudo systemctl restart ai-video-worker` | worker state / restart (pulls the latest worker code from S3) |
| `nvidia-smi` | GPU use and memory |
| `sudo nano /etc/ai-video.env` | runtime settings: `VIDEO_LIGHTNING`, `EDIT_LIGHTNING`, `KEEP_RAM_WEIGHTS` (`0` = models copy their weights back from the GPU as before), `PRELOAD_MODELS` (`0` = an idle 8xlarge/16xlarge no longer loads the big models ahead), `IDLE_MINUTES`, models (restart after editing) |
| `ls /opt/models/loras/` | downloaded LoRA files (per client/name/version) |
| `df -h /` | disk space (models live on the 500 GB root disk) |
| `journalctl -u ai-video-worker \| grep -E 'model_download\|mirror_'` | where weights came from and how fast; what was uploaded to the [model mirror](#model-mirror) |

### On the trainer (root shell)

| Command | What it does |
|---|---|
| `journalctl -u ai-video-trainer -f` | follow training |
| `tail -f /var/log/ai-video-trainer-setup.log` | first-boot setup log |
| `ls /opt/trainer/jobs/<job_id>/` | a job's work folder (state, logs, checkpoints while training) |
