#!/bin/bash
# First-boot setup for the GPU worker. Runs once, on the instance's first
# launch. Every later boot only runs the systemd unit, which re-syncs worker
# code from S3 before starting.
set -euxo pipefail
exec > >(tee /var/log/ai-video-setup.log) 2>&1

# The dashboard's "ready to render" line: a percent and one line, read by GET /status (worker.py READY_KEY).
ready() {
  printf '{"percent": %s, "text": "%s", "updated_at": %s}' "$1" "$2" "$(date +%s)" |
    aws s3 cp - "s3://${bucket}/status/render.json" --region ${region} --content-type application/json >/dev/null 2>&1 || true
}

export DEBIAN_FRONTEND=noninteractive
ready 3 "First boot: installing system packages"
apt-get update
apt-get install -y ffmpeg python3-venv python3-pip fonts-dejavu-core espeak-ng git

mkdir -p /opt/ai-video /opt/models
cat > /etc/ai-video.env <<'EOF'
AWS_REGION=${region}
AWS_DEFAULT_REGION=${region}
BUCKET=${bucket}
QUEUE_URL=${queue_url}
JOBS_TABLE=${jobs_table}
IDLE_MINUTES=${idle_minutes}
VIDEO_MODEL=${video_model}
I2V_MODEL=${i2v_model}
VLM_MODEL=${vlm_model}
EDIT_MODEL=${edit_model}
EDIT_LIGHTNING=${edit_lightning}
JOB_TIMEOUT_MINUTES=${job_timeout}
HF_HOME=/opt/models
# systemd sets no HOME; PyTorch needs one to cache compiled kernels between runs.
HOME=/root
HF_HUB_ENABLE_HF_TRANSFER=1
HF_XET_HIGH_PERFORMANCE=1
EOF

ready 8 "First boot: installing the Python environment"
python3 -m venv /opt/ai-video/venv
/opt/ai-video/venv/bin/pip install --upgrade pip wheel

aws s3 sync "s3://${bucket}/code/worker/" /opt/ai-video/worker/ --delete --region ${region}
/opt/ai-video/venv/bin/pip install -r /opt/ai-video/worker/requirements.txt

# Weights, so the first job does not wait for them. First from the model mirror in the bucket (every
# folder an earlier box uploaded, bf16 Wan copies included; worker/mirror.py), then whatever is still
# missing from Hugging Face: the commands below find mirrored files in place and skip them.
set -a; . /etc/ai-video.env; set +a
ready 15 "First boot: restoring models from the model mirror"
/opt/ai-video/venv/bin/python /opt/ai-video/worker/mirror.py pull-all /opt/models --skip trainer/ || true
ready 25 "First boot: downloading the Wan video models that are not in the mirror"
# The fp32 Wan repos are only needed to make the bf16 copies (convert_bf16.py below).
[ -f "/opt/models/bf16/$(basename "${video_model}")/model_index.json" ] || /opt/ai-video/venv/bin/huggingface-cli download "${video_model}"
[ -f "/opt/models/bf16/$(basename "${i2v_model}")/model_index.json" ] || /opt/ai-video/venv/bin/huggingface-cli download "${i2v_model}"
ready 45 "First boot: downloading the description and image-edit models"
/opt/ai-video/venv/bin/huggingface-cli download "${vlm_model}"
/opt/ai-video/venv/bin/huggingface-cli download "${edit_model}"
ready 60 "First boot: downloading the sound model"
/opt/ai-video/venv/bin/python - <<'PY'
import dataclasses
from pathlib import Path
from mmaudio.eval_utils import all_model_cfg
base = Path("/opt/models/mmaudio")
cfg = all_model_cfg["large_44k_v2"]
cfg = dataclasses.replace(cfg, model_path=base / "weights" / cfg.model_path.name,
                          vae_path=base / "ext_weights" / cfg.vae_path.name,
                          synchformer_ckpt=base / "ext_weights" / cfg.synchformer_ckpt.name)
for p in (cfg.model_path, cfg.vae_path):
    p.parent.mkdir(parents=True, exist_ok=True)
cfg.download_if_needed()
PY
ready 68 "First boot: converting the Wan video models"
# The Wan repos store the 14B experts in fp32; keep a bf16 copy and drop the fp32 download.
(cd /opt/ai-video/worker && /opt/ai-video/venv/bin/python convert_bf16.py --delete-source)
ready 82 "First boot: preparing Helios (its environment and weights)"
# Helios, the default video model: its venv, its pinned repository and its weights (worker/helios.py; the
# worker does the same on a first Helios job, so a failure here only moves the wait there).
(cd /opt/ai-video/worker && /opt/ai-video/venv/bin/python -c "import helios; helios.ensure_env(print); helios.ensure_weights(print)") || echo "Helios setup failed; the worker retries on the first Helios job"

cat > /etc/systemd/system/ai-video-worker.service <<'EOF'
[Unit]
Description=AI video worker
After=network-online.target
Wants=network-online.target

[Service]
EnvironmentFile=/etc/ai-video.env
ExecStartPre=/bin/sh -c 'aws s3 sync s3://${bucket}/code/worker/ /opt/ai-video/worker/ --delete'
ExecStartPre=/bin/sh -c 'aws s3 cp s3://${bucket}/code/authorized_keys /home/ubuntu/.ssh/authorized_keys && chown ubuntu:ubuntu /home/ubuntu/.ssh/authorized_keys && chmod 600 /home/ubuntu/.ssh/authorized_keys'
ExecStartPre=/opt/ai-video/venv/bin/pip install -q -r /opt/ai-video/worker/requirements.txt
ExecStart=/opt/ai-video/venv/bin/python -u /opt/ai-video/worker/worker.py
WorkingDirectory=/opt/ai-video/worker
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

ready 96 "First boot: starting the worker"
systemctl daemon-reload
systemctl enable --now ai-video-worker.service
