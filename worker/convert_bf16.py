"""One-time conversion of the Wan2.2 A14B pipelines to a local bf16 copy.

The Hugging Face repos store both 14B experts in fp32 (~54 GB each), which the worker
casts to bf16 on every load. A bf16 copy halves the disk read and the load-time
memory spike. The VAE stays fp32, as the worker uses it.

    /opt/ai-video/venv/bin/python convert_bf16.py [--delete-source]

Stop the worker first: a pipeline needs ~65 GB of RAM while it is converted.
--delete-source removes the fp32 download once its copy is complete.
"""

import os
import shutil
import sys

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

from models import BF16_DIR, bf16_path


def convert(repo, pipeline_cls, delete_source):
    dst = bf16_path(repo)
    if os.path.exists(os.path.join(dst, "model_index.json")):
        print(f"{repo}: already converted at {dst}", flush=True)
    else:
        print(f"{repo}: converting to {dst}", flush=True)
        vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
        pipe = pipeline_cls.from_pretrained(repo, vae=vae, torch_dtype=torch.bfloat16)
        tmp = dst + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        pipe.save_pretrained(tmp, safe_serialization=True, max_shard_size="10GB")
        os.rename(tmp, dst)
        del pipe, vae
        print(f"{repo}: done", flush=True)
    if delete_source:
        cache = os.path.join(os.environ.get("HF_HOME", "/opt/models"), "hub",
                             "models--" + repo.replace("/", "--"))
        if os.path.isdir(cache):
            shutil.rmtree(cache)
            print(f"{repo}: removed fp32 download {cache}", flush=True)


if __name__ == "__main__":
    os.makedirs(BF16_DIR, exist_ok=True)
    delete = "--delete-source" in sys.argv
    convert(os.environ["VIDEO_MODEL"], WanPipeline, delete)
    convert(os.environ.get("I2V_MODEL", "Wan-AI/Wan2.2-I2V-A14B-Diffusers"), WanImageToVideoPipeline, delete)
