#!/usr/bin/env bash
# Trainer spike, part 2: "training further" (docs/specs/2026-11-09-lora-training.md).
#
# Reuses /opt/trial from part 1 (venv, pinned musubi-tuner, weights, caches) on the render box:
#   resume  train 60 steps with --save_state, then --resume from that state up to 90 steps
#   extend  new run initialised from part 1's LoRA (--network_weights) on the part-1 clips plus
#           extra clips from code/trainer-spike/data2/, at the lower "extend" learning rate
#   load    convert both, load each into the production T2V pipeline next to Lightning,
#           render one 2 s clip with the extended LoRA
#
# Usage: run_part2.sh <client> <part-1 out dir with trial.safetensors>
set -uo pipefail

CLIENT=${1:?client}
BASE=${2:?part-1 out dir}
set -a; . /etc/ai-video.env; set +a
TS=$(date -u +%Y%m%d-%H%M%S)
T=/opt/trial
OUT=$T/part2-$TS
DEST="s3://$BUCKET/outputs/$CLIENT/trial-part2-$TS"
mkdir -p $OUT $T/data2 $T/cache2
exec > >(tee -a $OUT/part2.log) 2>&1

finish() {
  echo "== finishing (exit $1)"
  # Top-level files only (the state folders stay on the box); per-file cp, no list permission needed.
  for f in $OUT/*; do [ -f "$f" ] && aws s3 cp "$f" "$DEST/$(basename "$f")" --only-show-errors; done
  systemctl start ai-video-worker
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
losses() {  # losses <log>: first and last reported average loss
  local all; all=$(tr '\r' '\n' < "$1" | grep -oE 'avr_loss=[0-9.]+' | cut -d= -f2)
  echo "first $(echo "$all" | head -1) last $(echo "$all" | tail -1)"
}

M=$T/musubi-tuner/src/musubi_tuner; [ -f $M/wan_train_network.py ] || M=$T/musubi-tuner
DIT=$T/models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors
VAE=$T/models/split_files/vae/wan_2.1_vae.safetensors
T5=$T/models/models_t5_umt5-xxl-enc-bf16.pth
[ -f $BASE/trial.safetensors ] || { PHASE=check; fail "no $BASE/trial.safetensors"; }

phase pause-worker
systemctl stop ai-video-worker
echo "musubi-tuner $(git -C $T/musubi-tuner rev-parse HEAD)" | tee -a $OUT/timings.txt

phase dataset2
rm -rf $T/data2/* $T/cache2/*
aws s3 cp "s3://$BUCKET/code/trainer-spike/data2/" $T/data2/ --recursive --exclude "*" --include "*.mp4" --only-show-errors \
  || fail "download clips from code/trainer-spike/data2/"
for clip in $T/data2/*.mp4; do echo "vgn_trial, a short cinematic video clip with natural lighting." > "${clip%.mp4}.txt"; done
echo "extra clips: $(ls $T/data2/*.mp4 | wc -l)" | tee -a $OUT/timings.txt
# Part-1 dataset plus the new clips, as two datasets sharing the same buckets.
sed -n '1,/^\[\[datasets\]\]/p' $T/dataset.toml | head -n -1 > $T/dataset2.toml
cat >> $T/dataset2.toml <<EOF
[[datasets]]
video_directory = "$T/data"
cache_directory = "$T/cache"
target_frames = [17, 33]
frame_extraction = "head"

[[datasets]]
video_directory = "$T/data2"
cache_directory = "$T/cache2"
target_frames = [17, 33]
frame_extraction = "head"
EOF
cat $T/dataset2.toml

phase cache2
$T/venv/bin/python $M/wan_cache_latents.py --dataset_config $T/dataset2.toml --vae $VAE --skip_existing || fail cache-latents
$T/venv/bin/python $M/wan_cache_text_encoder_outputs.py --dataset_config $T/dataset2.toml --t5 $T5 --batch_size 16 \
  --skip_existing || fail cache-text

train() {  # train <dataset toml> <output name> <lr> <log> <extra args...>
  local cfg=$1 name=$2 lr=$3 log=$4; shift 4
  $T/venv/bin/accelerate launch --num_cpu_threads_per_process 1 --mixed_precision fp16 $M/wan_train_network.py \
    --task t2v-A14B --dit $DIT --dataset_config $cfg --sdpa --mixed_precision fp16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate $lr --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_wan --network_dim 32 --network_alpha 16 \
    --timestep_sampling shift --discrete_flow_shift 3.0 --min_timestep 0 --max_timestep 875 \
    --seed 42 --output_dir $OUT --output_name $name "$@" 2>&1 | tee $OUT/$log
  return ${PIPESTATUS[0]}
}

phase resume-base
train $T/dataset.toml resume 2e-4 resume_base.log --max_train_steps 60 --save_every_n_steps 60 --save_state \
  || fail "60-step run with --save_state"
STATE=$(ls -d $OUT/resume*-state 2>/dev/null | sort | tail -1)
[ -n "$STATE" ] || fail "no saved state folder"
echo "state: $(basename $STATE) ($(du -sh $STATE | cut -f1))" | tee -a $OUT/timings.txt
echo "resume-base loss: $(losses $OUT/resume_base.log)" | tee -a $OUT/timings.txt

phase resume
train $T/dataset.toml resume 2e-4 resume.log --max_train_steps 90 --save_every_n_steps 90 --resume $STATE \
  || fail "--resume to 90 steps"
# Proof the step counter continued: the progress bar of the resumed run starts at 60/90.
FIRST=$(tr '\r' '\n' < $OUT/resume.log | grep -oE '[0-9]+/90 ' | head -1)
LAST=$(tr '\r' '\n' < $OUT/resume.log | grep -oE '[0-9]+/90 ' | tail -1)
echo "resume progress: first ${FIRST:-?} last ${LAST:-?}" | tee -a $OUT/timings.txt
echo "resume loss: $(losses $OUT/resume.log)" | tee -a $OUT/timings.txt
ls -la $OUT/*.safetensors

phase extend
train $T/dataset2.toml extend 5e-5 extend.log --max_train_steps 100 --save_every_n_steps 100 \
  --network_weights $BASE/trial.safetensors || fail "extend run from part-1 weights"
echo "extend loss: $(losses $OUT/extend.log)" | tee -a $OUT/timings.txt
echo "part-1 from-scratch loss: $(losses $BASE/trial.log)" | tee -a $OUT/timings.txt

phase convert
for name in resume extend; do
  $T/venv/bin/python $M/convert_lora.py --input $OUT/$name.safetensors --output $OUT/${name}_comfy.safetensors \
    --target other || fail "convert $name"
done

phase load-check
(cd /opt/ai-video/worker && /opt/ai-video/venv/bin/python -u $T/load_check.py $OUT) || fail load-check

phase done
