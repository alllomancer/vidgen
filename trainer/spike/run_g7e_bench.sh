#!/bin/bash
# g7e benchmark on a temporary box (docs/specs/2026-11-25-g7e-benchmark.md). Run as root:
#   bash run_g7e_bench.sh <bucket> <timestamp>
# Builds its own venv (torch for the Blackwell GPU, the worker's diffusers pins), downloads Wan2.2 T2V,
# runs bench_wan.py with both experts on the GPU and with offloading, uploads results to
# s3://<bucket>/outputs/trial-g7e/<timestamp>/.
set -uo pipefail
BUCKET=$1; TS=$2
T=/opt/trial; OUT=$T/out; DEST="s3://$BUCKET/outputs/trial-g7e/$TS"
mkdir -p $OUT; cd $T
exec > >(tee $OUT/trial.log) 2>&1
export HF_HOME=/opt/models HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 DEBIAN_FRONTEND=noninteractive
upload() { aws s3 cp $OUT "$DEST/" --recursive --only-show-errors; }
trap upload EXIT
step() { echo "== $(date -u +%H:%M:%S) $*"; }

step "gpu"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
step "venv"
apt-get update -qq && apt-get install -y -qq python3-venv python3-pip >/dev/null
python3 -m venv $T/venv
$T/venv/bin/pip install -q --upgrade pip wheel
$T/venv/bin/pip install -q torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
$T/venv/bin/pip install -q diffusers==0.37.1 transformers==4.55.4 accelerate==1.6.0 peft==0.17.1 \
  "huggingface_hub[cli,hf_transfer]==0.34.4" ftfy==6.3.1 sentencepiece==0.2.0 safetensors
$T/venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
step "download"
t0=$(date +%s)
$T/venv/bin/huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers >/dev/null
echo "download_seconds $(( $(date +%s) - t0 )) size $(du -sh /opt/models/hub | cut -f1)"
for mode in resident offload; do
  step "bench $mode"
  $T/venv/bin/python -u $T/bench_wan.py --model Wan-AI/Wan2.2-T2V-A14B-Diffusers --mode $mode --out $OUT/g7e_$mode.json
  upload
done
step "done"
