#!/bin/bash
# g7e box, run 3: Krea venv check on this GPU, then the bake-off's SVI 2.0 Pro and Helios runs.
# Reports to s3://.../outputs/trial-g7e/20261126-075952-run3/status.log every minute and TERMINATES ITSELF when the script ends.
set -uo pipefail
B=ai-video-842676012528-us-east-1; DEST=s3://$B/outputs/trial-g7e/20261126-075952-run3; SRC=s3://$B/code/g7e-trial
T=/opt/trial; K=/opt/bake; OUT=$T/out; BOUT=$K/out
mkdir -p $T $K $OUT $BOUT /opt/ai-video/worker /opt/models
export AWS_DEFAULT_REGION=us-east-1 AWS_REGION=us-east-1 DEBIAN_FRONTEND=noninteractive HOME=/root
export HF_HOME=/opt/models HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1
exec > $T/main.log 2>&1
step() { echo "== $(date -u +%H:%M:%S) $*"; }
status() { { date -u +%H:%M:%S; df -h / | tail -1; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; grep -aE "^==|exit|^\{|FAILED|Error|Traceback" $T/main.log | tail -70 | cut -c1-600; echo "--- tail"; tail -c 1500 $T/main.log | tr '\r' '\n' | tail -12 | cut -c1-300; } > $T/status.log 2>&1; aws s3 cp $T/status.log $DEST/status.log --only-show-errors; }
( while true; do status; sleep 60; done ) &
upload() { aws s3 cp $OUT $DEST/validate/ --recursive --only-show-errors; aws s3 cp $BOUT $DEST/bakeoff/ --recursive --only-show-errors; aws s3 cp $T/main.log $DEST/main.log --only-show-errors; }
finish() { step "finished: uploading and terminating this box"; upload; status; shutdown -h now; }
trap finish EXIT

step "code and packages"
aws s3 sync $SRC/worker/ /opt/ai-video/worker/ --only-show-errors
aws s3 cp $SRC/ $T/ --recursive --exclude "worker/*" --only-show-errors
apt-get update -qq && apt-get install -y -qq ffmpeg python3-venv python3-pip git software-properties-common >/dev/null
add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1; apt-get update -qq; apt-get install -y -qq python3.11 python3.11-venv python3.11-dev libopenmpi-dev >/dev/null; echo "python3.11 exit $?"
REF=$T/ref.png; cp $REF $BOUT/ref.png; cp $T/prompts.json $BOUT/

step "environments (worker, SVI, Helios)"
sed -e 's/^torch==2.5.1$/--extra-index-url https:\/\/download.pytorch.org\/whl\/cu128\ntorch==2.9.0+cu128/' -e 's/^torchvision==0.20.1$/torchvision==0.24.0+cu128/' -e 's/^torchaudio==2.5.1$/torchaudio==2.9.0+cu128/' -e 's/^torchcodec==0.1.1$/torchcodec==0.8.0/' -e 's/^tensordict==0.6.2$/tensordict==0.10.0/' /opt/ai-video/worker/requirements.txt > $T/requirements.candidate.txt
( python3 -m venv /opt/ai-video/venv && /opt/ai-video/venv/bin/pip install -q --upgrade pip wheel && /opt/ai-video/venv/bin/pip install -q -r $T/requirements.candidate.txt; echo "worker pip exit $?" ) > $T/venv_worker.log 2>&1 &
cd $K
git clone -q -b svi_wan22 https://github.com/vita-epfl/Stable-Video-Infinity.git svi; git clone -q --depth=1 https://github.com/PKU-YuanGroup/Helios.git helios
echo "svi $(git -C svi rev-parse --short HEAD) helios $(git -C helios rev-parse --short HEAD)"
printf 'def visualize_video_tensor(*args, **kwargs):\n    return None\n' > $K/svi/diffsynth/utils/visualize_video_tensor.py  # a debug helper the repo imports but does not ship
( python3 -m venv svi-venv && svi-venv/bin/pip install -q --upgrade pip wheel && svi-venv/bin/pip install -q torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 && (cd svi && ../svi-venv/bin/pip install -q -e .) && svi-venv/bin/pip install -q imageio imageio-ffmpeg hf_transfer hf_xet; echo "svi pip exit $?" ) > $T/venv_svi.log 2>&1 &
( python3.11 -m venv helios-venv && helios-venv/bin/pip install -q --upgrade pip wheel && helios-venv/bin/pip install -q torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128 && (cd helios && ../helios-venv/bin/pip install -q -r requirements.txt); echo "helios pip exit $?"; helios-venv/bin/pip install -q hf_transfer hf_xet ) > $T/venv_helios.log 2>&1 &
wait %2 %3 %4 2>/dev/null; wait
tail -1 $T/venv_worker.log; tail -2 $T/venv_svi.log | cut -c1-300; grep "exit" $T/venv_helios.log | cut -c1-300
PY=/opt/ai-video/venv/bin/python; HF=/opt/ai-video/venv/bin/huggingface-cli

step "quick checks on this GPU, before any large download"
cd $K/svi && $K/svi-venv/bin/python -c "import torch; from diffsynth.pipelines.wan_video_svi_pro import ModelConfig, WanVideoSviProPipeline; x=torch.ones(4, device='cuda'); print('svi import ok', torch.__version__, float(torch.cat([x, x]).sum()))"; echo "svi check exit $?"
cd $K/helios && $K/helios-venv/bin/python infer_helios.py --help >/dev/null; echo "helios check exit $?"
$K/helios-venv/bin/python -c "import torch; x=torch.ones(4, device='cuda'); print('helios torch ok', torch.__version__, float(torch.cat([x, x]).sum()))"

step "model downloads"
M=$K/svi/models
( $HF download vita-video-gen/svi-model --local-dir $M/Stable-Video-Infinity --include "version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors" "version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors" >/dev/null; echo "svi lora exit $?"
  $HF download Wan-AI/Wan2.2-I2V-A14B --local-dir $M/Wan-AI/Wan2.2-I2V-A14B --include "high_noise_model/*" "low_noise_model/*" "models_t5_umt5-xxl-enc-bf16.pth" "Wan2.1_VAE.pth" >/dev/null; echo "svi wan exit $?"
  $HF download Wan-AI/Wan2.1-T2V-1.3B --local-dir $M/Wan-AI/Wan2.1-T2V-1.3B --include "google/*" >/dev/null; echo "svi tokenizer exit $?"
  $HF download lightx2v/Wan2.2-Lightning --revision 18bccf8884ec0a078eed79785eb4ef13ea16ce1e --local-dir $K/lightning --include "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/*" >/dev/null; echo "lightning exit $?"
  du -sh $M; echo "svi models done" ) > $K/dl_svi.log 2>&1 &

step "Krea 2 on this GPU with torch 2.14.1+cu130 in its venv"
cat > /etc/ai-video.env <<'ENV'
AWS_REGION=us-east-1
AWS_DEFAULT_REGION=us-east-1
VIDEO_MODEL=Wan-AI/Wan2.2-T2V-A14B-Diffusers
I2V_MODEL=Wan-AI/Wan2.2-I2V-A14B-Diffusers
VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
EDIT_MODEL=Qwen/Qwen-Image-Edit-2511
EDIT_LIGHTNING=8
HF_HOME=/opt/models
HOME=/root
HF_HUB_ENABLE_HF_TRANSFER=1
HF_XET_HIGH_PERFORMANCE=1
MODEL_MIRROR=0
ENV
sed -i -e 's#whl/cu126#whl/cu130#' -e 's/^torch==2.14.1+cu126$/torch==2.14.1+cu130/' /opt/ai-video/worker/krea_requirements.txt; grep -n "cu1\|^torch" /opt/ai-video/worker/krea_requirements.txt
( set -a; . /etc/ai-video.env; set +a; cd /opt/ai-video/worker; PYTHONPATH=/opt/ai-video/worker $PY -u $T/validate_g7e.py $OUT krea; echo "krea validate exit $?"; /opt/krea-venv/bin/python -c "import torch; print('krea venv', torch.__version__, torch.cuda.get_arch_list())"; /opt/krea-venv/bin/pip freeze > $OUT/krea_pip_freeze.txt )
upload

PROMPT="$($PY -c "import json;print(json.load(open('$T/prompts.json'))['single'])")"
step "bake-off: Helios distilled, 640x384 (its default), 30 s"
cd $K/helios; t0=$(date +%s)
HF_HOME=$K/hf $K/helios-venv/bin/python -u infer_helios.py --base_model_path BestWishYsh/Helios-Distilled --transformer_path BestWishYsh/Helios-Distilled --sample_type i2v --num_frames 720 --fps 24 --image_path $REF --image_noise_sigma_min 0.111 --image_noise_sigma_max 0.135 --prompt "$PROMPT" --guidance_scale 1.0 --is_enable_stage2 --pyramid_num_inference_steps_list 2 2 2 --is_amplify_first_chunk --disable_flash_attention --output_folder $BOUT/helios_640x384; echo "helios 640 exit $? in $(( $(date +%s) - t0 )) s (includes its model download and load)"; upload
step "bake-off: Helios distilled, 640x384 again (model on disk: load + render time)"
t0=$(date +%s)
HF_HOME=$K/hf $K/helios-venv/bin/python -u infer_helios.py --base_model_path BestWishYsh/Helios-Distilled --transformer_path BestWishYsh/Helios-Distilled --sample_type i2v --num_frames 720 --fps 24 --seed 7 --image_path $REF --image_noise_sigma_min 0.111 --image_noise_sigma_max 0.135 --prompt "$PROMPT" --guidance_scale 1.0 --is_enable_stage2 --pyramid_num_inference_steps_list 2 2 2 --is_amplify_first_chunk --disable_flash_attention --output_folder $BOUT/helios_640x384_b; echo "helios 640 second exit $? in $(( $(date +%s) - t0 )) s"; upload
step "bake-off: Helios distilled, 832x480"
t0=$(date +%s)
HF_HOME=$K/hf $K/helios-venv/bin/python -u infer_helios.py --base_model_path BestWishYsh/Helios-Distilled --transformer_path BestWishYsh/Helios-Distilled --sample_type i2v --num_frames 720 --fps 24 --height 480 --width 832 --image_path $REF --image_noise_sigma_min 0.111 --image_noise_sigma_max 0.135 --prompt "$PROMPT" --guidance_scale 1.0 --is_enable_stage2 --pyramid_num_inference_steps_list 2 2 2 --is_amplify_first_chunk --disable_flash_attention --output_folder $BOUT/helios_832x480; echo "helios 832 exit $? in $(( $(date +%s) - t0 )) s"; upload
nvidia-smi --query-gpu=memory.used --format=csv,noheader

step "bake-off: SVI 2.0 Pro + Lightning (4 steps), 6 clips"
until grep -q "svi models done" $K/dl_svi.log; do sleep 15; done; cat $K/dl_svi.log
cd $K/svi
LD=$K/lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1
DIFFSYNTH_DOWNLOAD_SOURCE=huggingface $K/svi-venv/bin/python -u $T/bake_svi.py $REF $T/prompts.json $BOUT svi_pro_lightning_4step --clips 6 --steps 4 --cfg 1.0 --lightning $LD/high_noise_model.safetensors $LD/low_noise_model.safetensors; echo "svi lightning exit $?"; upload
step "bake-off: SVI 2.0 Pro full guidance sample (2 clips, 20 steps, cfg 4)"
DIFFSYNTH_DOWNLOAD_SOURCE=huggingface $K/svi-venv/bin/python -u $T/bake_svi.py $REF $T/prompts.json $BOUT svi_pro_cfg4_20step_10s --clips 2 --steps 20 --cfg 4.0; echo "svi full exit $?"; upload
step "all done"
