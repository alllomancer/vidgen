#!/usr/bin/env bash
# Krea 2 character-LoRA trial (spike; see docs/adr/krea2-lora.md and RUNBOOK "Krea 2 LoRAs").
#
# Runs ON the render box as root, after its queue is empty:
#   1. pauses the render worker (so it neither takes jobs nor stops the machine)
#   2. installs musubi-tuner (pinned commit, Krea 2 support) in its own venv (/opt/kreatrial/venv)
#   3. downloads Krea 2 RAW (training) + Turbo (samples), Qwen-Image VAE, Qwen3-VL-4B (~60 GB, kept)
#   4. fetches images + captions staged under s3://$BUCKET/code/trainer-spike/krea-data/,
#      caches latents and text-encoder outputs
#   5. trains a rank-32 LoRA on RAW, saving checkpoints with Turbo samples (musubi-tuner's
#      recommended RAW-train -> Turbo-infer workflow)
#   6. uploads LoRAs, samples, timings and the log to s3://$BUCKET/outputs/<client>/krea-trial-<ts>/
#      and resumes the worker
#
# Usage: run_krea_trial.sh <client> <trigger> [epochs] [save_every_epochs]
set -uo pipefail

CLIENT=${1:?client}
TRIGGER=${2:?trigger word}
EPOCHS=${3:-40}
SAVE_EVERY=${4:-10}
set -a; . /etc/ai-video.env; set +a
TS=$(date -u +%Y%m%d-%H%M%S)
T=/opt/kreatrial
OUT=$T/out-$TS
DEST="s3://$BUCKET/outputs/$CLIENT/krea-trial-$TS"
mkdir -p $T/models $T/data $T/cache $OUT
exec > >(tee -a $OUT/trial.log) 2>&1

finish() {
  echo "== finishing (exit $1)"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
  df -h /opt | tail -1 >> $OUT/timings.txt
  for f in $OUT/*; do [ -f "$f" ] && aws s3 cp "$f" "$DEST/$(basename "$f")" --only-show-errors || true; done
  for f in $OUT/sample/*; do [ -f "$f" ] && aws s3 cp "$f" "$DEST/sample/$(basename "$f")" --only-show-errors || true; done
  systemctl start ai-video-worker   # resumes jobs; it stops the machine after IDLE_MINUTES idle
  echo "results: $DEST/"
}
trap 'finish $?' EXIT

phase_start=0
phase() {
  local now; now=$(date +%s)
  [ -n "${PHASE:-}" ] && echo "$PHASE $((now - phase_start))s" | tee -a $OUT/timings.txt
  PHASE=$1; phase_start=$now; echo "== $PHASE"
}
fail() { echo "FAIL in $PHASE: $*"; echo "FAIL $PHASE: $*" >> $OUT/timings.txt; exit 1; }

phase pause-worker
systemctl stop ai-video-worker
nvidia-smi --query-gpu=name,memory.total --format=csv
df -h /opt | tail -1

phase install
if [ ! -x $T/venv/bin/python ]; then
  python3 -m venv $T/venv || fail venv
  $T/venv/bin/pip install -q --upgrade pip wheel
  # musubi-tuner wants torch >= 2.6; cu126 matches the box's driver (as the Krea venv does).
  $T/venv/bin/pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 || fail torch
fi
# Pinned: third-party code that runs as root on this box (same commit as the trainer). Bump after review.
MUSUBI_COMMIT=f8a1b03794a49239a3539015075f5123d6c07d66
[ -d $T/musubi-tuner ] || git clone -q https://github.com/kohya-ss/musubi-tuner.git $T/musubi-tuner || fail clone
git -C $T/musubi-tuner fetch -q origin $MUSUBI_COMMIT 2>/dev/null || true
git -C $T/musubi-tuner checkout -q --detach $MUSUBI_COMMIT || fail "checkout musubi-tuner $MUSUBI_COMMIT"
(cd $T/musubi-tuner && $T/venv/bin/pip install -q -e .) || fail "pip install musubi-tuner"
echo "musubi-tuner $(git -C $T/musubi-tuner rev-parse HEAD)" | tee -a $OUT/timings.txt
M=$T/musubi-tuner/src/musubi_tuner
[ -f $M/krea2_train_network.py ] || fail "this musubi-tuner has no krea2_train_network.py"

phase download-weights
HF_TOKEN=$(aws ssm get-parameter --name "${HF_TOKEN_PARAM:-/ai-video/hf_token}" --with-decryption \
           --query Parameter.Value --output text) || fail "read the Hugging Face token from SSM"
export HF_TOKEN HF_HUB_ENABLE_HF_TRANSFER=0
HF="$T/venv/bin/hf"; [ -x "$HF" ] || HF="$T/venv/bin/huggingface-cli"
# Krea 2 RAW is gated separately from Turbo: fail early with a clear message if the token can't read it.
$HF download krea/Krea-2-Raw README.md --local-dir $T/models/raw >/dev/null 2>&1 \
  || fail "no access to krea/Krea-2-Raw: accept its license on huggingface.co with the account of the SSM token"
$HF download krea/Krea-2-Raw raw.safetensors --local-dir $T/models/raw >/dev/null || fail "download Krea 2 RAW"
$HF download krea/Krea-2-Turbo turbo.safetensors --local-dir $T/models/turbo >/dev/null || fail "download Krea 2 Turbo"
$HF download Comfy-Org/Qwen-Image split_files/vae/qwen_image_vae.safetensors --local-dir $T/models >/dev/null || fail "download VAE"
$HF download Comfy-Org/Qwen3-VL text_encoders/qwen3vl_4b_bf16.safetensors --local-dir $T/models >/dev/null || fail "download text encoder"
unset HF_TOKEN
RAW=$T/models/raw/raw.safetensors
TURBO=$T/models/turbo/turbo.safetensors
VAE=$T/models/split_files/vae/qwen_image_vae.safetensors
TE=$T/models/text_encoders/qwen3vl_4b_bf16.safetensors
ls -la $RAW $TURBO $VAE $TE

phase dataset
rm -rf $T/data/* $T/cache/*
aws s3 cp "s3://$BUCKET/code/trainer-spike/krea-data/" $T/data/ --recursive --only-show-errors || fail "fetch dataset"
N=$(ls $T/data/*.png 2>/dev/null | wc -l)
[ "$N" -ge 8 ] || fail "need at least 8 images, found $N"
for f in $T/data/*.png; do [ -f "${f%.png}.txt" ] || fail "missing caption for $(basename $f)"; done
echo "images $N" | tee -a $OUT/timings.txt
cat > $T/dataset.toml <<EOF
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "$T/data"
cache_directory = "$T/cache"
num_repeats = 1
EOF
# Samples on Turbo (8 steps, CFG off), as the render box will use the LoRA.
cat > $T/samples.txt <<EOF
$TRIGGER, a man with a grey beard, close-up portrait, soft window light --w 1024 --h 1024 --s 8 --l 1 --d 42
$TRIGGER, a man with a grey beard, full body, standing on a city street at night, wearing a black coat --w 1024 --h 1024 --s 8 --l 1 --d 43
$TRIGGER, a man with a grey beard, sitting in a cafe reading a book, medium shot --w 1024 --h 1024 --s 8 --l 1 --d 44
EOF

phase cache
$T/venv/bin/python $M/krea2_cache_latents.py --dataset_config $T/dataset.toml --vae $VAE || fail "cache latents"
$T/venv/bin/python $M/krea2_cache_text_encoder_outputs.py --dataset_config $T/dataset.toml --text_encoder $TE \
  --batch_size 4 || fail "cache text encoder outputs"

phase train
$T/venv/bin/accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 $M/krea2_train_network.py \
  --dit $RAW --vae $VAE --text_encoder $TE --dataset_config $T/dataset.toml \
  --sdpa --mixed_precision bf16 \
  --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.5 \
  --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --network_module networks.lora_krea2 --network_dim 32 --network_alpha 32 \
  --max_train_epochs $EPOCHS --save_every_n_epochs $SAVE_EVERY --seed 42 \
  --sample_prompts $T/samples.txt --sample_every_n_epochs $SAVE_EVERY --turbo_dit $TURBO \
  --output_dir $OUT --output_name krea-trial || fail "train"
nvidia-smi --query-gpu=memory.used --format=csv | tee -a $OUT/timings.txt
ls -la $OUT
phase done
