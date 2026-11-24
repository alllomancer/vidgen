#!/bin/bash
# First-boot setup for the LoRA trainer (training_enabled = true). Runs once; later boots
# only run the systemd unit, which re-syncs trainer code from S3 before starting.
set -euxo pipefail
exec > >(tee /var/log/ai-video-trainer-setup.log) 2>&1

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ffmpeg python3-venv python3-pip git

mkdir -p /opt/trainer/jobs /opt/trainer/models /opt/models
cat > /etc/ai-video-trainer.env <<'EOF'
AWS_REGION=${region}
AWS_DEFAULT_REGION=${region}
BUCKET=${bucket}
TRAIN_QUEUE_URL=${train_queue_url}
JOBS_TABLE=${jobs_table}
IDLE_MINUTES=${idle_minutes}
JOB_TIMEOUT_MINUTES=${job_timeout}
VLM_MODEL=${vlm_model}
HF_HOME=/opt/models
HOME=/root
PYTHONUNBUFFERED=1
EOF

python3 -m venv /opt/trainer/venv
/opt/trainer/venv/bin/pip install --upgrade pip wheel
/opt/trainer/venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# Pinned: third-party code that runs as root on this machine. Bump deliberately after review.
git clone https://github.com/kohya-ss/musubi-tuner.git /opt/trainer/musubi-tuner
git -C /opt/trainer/musubi-tuner checkout --detach ${musubi_commit}
/opt/trainer/venv/bin/pip install -e /opt/trainer/musubi-tuner
# The training venv keeps exactly what musubi-tuner pins (transformers 5.x, huggingface_hub 1.x).
/opt/trainer/venv/bin/pip install boto3
# Captioning (Qwen2.5-VL) gets its own venv with the render worker's known-good stack.
python3 -m venv /opt/trainer/caption-venv
/opt/trainer/caption-venv/bin/pip install --upgrade pip wheel
/opt/trainer/caption-venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
/opt/trainer/caption-venv/bin/pip install "transformers==4.55.4" "accelerate==1.6.0" "huggingface_hub==0.34.4" \
  "qwen-vl-utils[decord]==0.0.11" av

aws s3 sync "s3://${bucket}/code/trainer/" /opt/trainer/code/ --delete --region ${region}

# Weights: ComfyUI-format Wan2.2 T2V experts (fp16), Wan2.1 VAE, umT5, and Qwen2.5-VL for captions.
# `hf` replaced `huggingface-cli` (which no longer works in huggingface_hub 1.x).
HF=/opt/trainer/venv/bin/hf
$HF download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors \
  split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
  split_files/vae/wan_2.1_vae.safetensors --local-dir /opt/trainer/models
$HF download Wan-AI/Wan2.1-T2V-14B models_t5_umt5-xxl-enc-bf16.pth --local-dir /opt/trainer/models
HF_HOME=/opt/models $HF download "${vlm_model}"

cat > /etc/systemd/system/ai-video-trainer.service <<'EOF'
[Unit]
Description=AI video LoRA trainer
After=network-online.target
Wants=network-online.target

[Service]
EnvironmentFile=/etc/ai-video-trainer.env
ExecStartPre=/bin/sh -c 'aws s3 sync s3://${bucket}/code/trainer/ /opt/trainer/code/ --delete'
ExecStart=/opt/trainer/venv/bin/python -u /opt/trainer/code/trainer.py
WorkingDirectory=/opt/trainer/code
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now ai-video-trainer.service
