#!/usr/bin/env bash
# Trainer viability trial (docs/specs/2026-11-09-lora-training.md, "Spike first").
#
# Runs ON the render box as root, after its queue is empty:
#   1. pauses the render worker (so it neither takes jobs nor stops the machine)
#   2. installs musubi-tuner in its own venv (/opt/trial/venv; the worker venv is untouched)
#   3. downloads the ComfyUI-format Wan2.2 T2V low-noise expert, VAE and umT5 (~40 GB)
#   4. builds a tiny dataset from recent output clips, caches latents + text embeddings
#   5. trains a 200-step LoRA on the low-noise expert (fp8 base), timing every phase
#   6. loads the LoRA in diffusers next to Lightning and renders a 2 s clip with and without it
#   7. uploads results to s3://$BUCKET/outputs/<client>/trial-<ts>/ and resumes the worker
#
# Usage: run_trial.sh <client> [steps]
set -uo pipefail

CLIENT=${1:?client}
STEPS=${2:-200}
set -a; . /etc/ai-video.env; set +a
TS=$(date -u +%Y%m%d-%H%M%S)
T=/opt/trial
OUT=$T/out-$TS
DEST="s3://$BUCKET/outputs/$CLIENT/trial-$TS"
mkdir -p $T/models $T/data $T/cache $OUT
exec > >(tee -a $OUT/trial.log) 2>&1

finish() {
  echo "== finishing (exit $1)"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
  df -h /opt | tail -1 >> $OUT/timings.txt
  # Per-file cp: the worker role may write outputs/ but not list it (which sync needs).
  for f in $OUT/*; do aws s3 cp "$f" "$DEST/$(basename "$f")" --only-show-errors || true; done
  systemctl start ai-video-worker   # resumes jobs; it stops the machine after IDLE_MINUTES idle
  echo "results: $DEST/"
}
trap 'finish $?' EXIT

phase_start=0
phase() {  # phase <name>: closes the previous phase's timing and opens a new one
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
  $T/venv/bin/pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 || fail torch
fi
# Pinned: third-party code that runs as root on this box. Bump deliberately after review.
MUSUBI_COMMIT=f8a1b03794a49239a3539015075f5123d6c07d66
[ -d $T/musubi-tuner ] || git clone -q https://github.com/kohya-ss/musubi-tuner.git $T/musubi-tuner || fail clone
git -C $T/musubi-tuner fetch -q origin $MUSUBI_COMMIT 2>/dev/null || true
git -C $T/musubi-tuner checkout -q --detach $MUSUBI_COMMIT || fail "checkout musubi-tuner $MUSUBI_COMMIT"
(cd $T/musubi-tuner && $T/venv/bin/pip install -q -e .) || fail "pip install musubi-tuner"
echo "musubi-tuner $(git -C $T/musubi-tuner rev-parse HEAD)" | tee -a $OUT/timings.txt
# Newer checkouts keep the scripts in src/musubi_tuner/, older ones at the top level.
M=$T/musubi-tuner/src/musubi_tuner; [ -f $M/wan_train_network.py ] || M=$T/musubi-tuner

phase download-weights
HF=/opt/ai-video/venv/bin/huggingface-cli
$HF download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors \
  split_files/vae/wan_2.1_vae.safetensors --local-dir $T/models >/dev/null || fail "download comfy weights"
$HF download Wan-AI/Wan2.1-T2V-14B models_t5_umt5-xxl-enc-bf16.pth --local-dir $T/models >/dev/null || fail "download t5"
DIT=$T/models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors
VAE=$T/models/split_files/vae/wan_2.1_vae.safetensors
T5=$T/models/models_t5_umt5-xxl-enc-bf16.pth
ls -la $DIT $VAE $T5

phase dataset
rm -rf $T/data/* $T/cache/*
# Clips staged by the operator under code/trainer-spike/data/ (the worker role can read
# code/*, but by design cannot list or read outputs/).
aws s3 cp "s3://$BUCKET/code/trainer-spike/data/" $T/data/ --recursive --exclude "*" --include "*.mp4" --only-show-errors \
  || fail "download clips from code/trainer-spike/data/"
ls $T/data/*.mp4 > $OUT/clips.txt 2>/dev/null
[ "$(wc -l < $OUT/clips.txt)" -ge 4 ] || fail "need at least 4 clips in code/trainer-spike/data/, found $(wc -l < $OUT/clips.txt)"
i=0
while read -r clip; do
  i=$((i + 1))
  echo "vgn_trial, a short cinematic video clip with natural lighting." > "${clip%.mp4}.txt"
done < $OUT/clips.txt
echo "clips: $i" | tee -a $OUT/timings.txt
cat > $T/dataset.toml <<EOF
[general]
resolution = [832, 480]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "$T/data"
cache_directory = "$T/cache"
target_frames = [17, 33]
frame_extraction = "head"
EOF

phase cache-latents
$T/venv/bin/python $M/wan_cache_latents.py --dataset_config $T/dataset.toml --vae $VAE || fail cache-latents

phase cache-text
$T/venv/bin/python $M/wan_cache_text_encoder_outputs.py --dataset_config $T/dataset.toml --t5 $T5 --batch_size 16 \
  || fail cache-text

train() {  # train <extra args...>
  # The ComfyUI-format expert is fp16, and musubi-tuner requires fp16 mixed precision for it.
  $T/venv/bin/accelerate launch --num_cpu_threads_per_process 1 --mixed_precision fp16 $M/wan_train_network.py \
    --task t2v-A14B --dit $DIT --dataset_config $T/dataset.toml --sdpa --mixed_precision fp16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_wan --network_dim 32 --network_alpha 16 \
    --timestep_sampling shift --discrete_flow_shift 3.0 --min_timestep 0 --max_timestep 875 \
    --max_train_steps $STEPS --save_every_n_steps $STEPS --seed 42 \
    --output_dir $OUT --output_name trial "$@"
}

phase train
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 30 > $OUT/gpu_mem.txt &
MEMLOG=$!
if ! train; then
  grep -q "out of memory" $OUT/trial.log || { kill $MEMLOG; fail "train (not out of memory; see trial.log)"; }
  echo "== plain fp8 run ran out of memory; retrying with block swap"
  echo "train: retried with --blocks_to_swap 20" >> $OUT/timings.txt
  train --blocks_to_swap 20 || { kill $MEMLOG; fail train; }
fi
kill $MEMLOG
echo "peak GPU memory: $(sort -n $OUT/gpu_mem.txt | tail -1 | tr -d ' ')" | tee -a $OUT/timings.txt
grep -oE '[0-9.]+(s/it|it/s)' $OUT/trial.log | tail -1 | sed 's/^/train speed: /' | tee -a $OUT/timings.txt
ls -la $OUT/*.safetensors

phase convert
$T/venv/bin/python $M/convert_lora.py --input $OUT/trial.safetensors --output $OUT/trial_comfy.safetensors --target other \
  || echo "convert_lora failed; render test will try the raw file only"

phase render-test
(cd /opt/ai-video/worker && /opt/ai-video/venv/bin/python -u $T/render_test.py $OUT) || fail render-test

phase done
