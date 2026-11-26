variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Name prefix for every resource."
  type        = string
  default     = "ai-video"
}

variable "availability_zone" {
  description = "AZ for the GPU subnet. A stopped spot instance can only restart in its own AZ, so pick one with good capacity for instance_type."
  type        = string
  default     = "us-east-1a"
}

variable "vpc_name" {
  type        = string
  description = "Value of the Name tag of the existing VPC to place the worker in"
  default     = "dev-use1-sat-dev-S-thruster-general-purpose-vpc"
}

variable "subnet_visibility_tag" {
  type        = string
  description = "Value of the Visibility tag of the subnet to use in availability_zone"
  default     = "private"
}

variable "instance_type" {
  description = <<-EOT
    GPU instance type of the render box: a g7e (one RTX PRO 6000, 96 GB VRAM). Helios, the default video
    model, and a whole Wan or image-edit pipeline stay on the GPU there (no moving parts on and off), and the
    worker's and Krea's torch builds are the ones for that GPU. g7e.4xlarge has 128 GB of RAM, g7e.8xlarge
    256 GB. `make g7e` finds a size, market and zone with capacity and sets this, on_demand and
    availability_zone (RUNBOOK "Moving the render box to a g7e"). Changing it replaces the instance and its
    disk: first boot restores the models from the model mirror, or downloads them (~45 min).
  EOT
  type        = string
  default     = "g7e.4xlarge"

  validation {
    condition     = startswith(var.instance_type, "g7e.")
    error_message = "The render box runs on g7e only (g7e.4xlarge, g7e.8xlarge): run `make g7e` to pick one with capacity."
  }
}

variable "on_demand" {
  description = "Run the render box on-demand instead of spot: for a type with no spot capacity, at the on-demand price (g6e.8xlarge ~$4.5/h). Changing this replaces the instance and its disk."
  type        = bool
  default     = false
}

variable "spot_max_price" {
  description = "Max hourly spot price in USD. Empty string = cap at on-demand price."
  type        = string
  default     = ""
}

variable "root_volume_gb" {
  description = "Root EBS volume size. Holds the OS, the Python envs and model weights (~330 GB: the bf16 Wan models, the edit and description models, Helios, Krea 2). A first boot without the model mirror also holds both fp32 Wan downloads (~240 GB) until they are converted."
  type        = number
  default     = 750
}

variable "root_volume_throughput" {
  description = "Root volume throughput in MB/s (gp3, needs iops >= 4x this). Model loads read ~60 GB, so this sets cold-start and model-switch time."
  type        = number
  default     = 1000
}

variable "idle_minutes" {
  description = "Worker stops the instance after this many minutes with an empty queue."
  type        = number
  default     = 15
}

variable "video_model" {
  description = "Hugging Face repo for text-to-video and avatar candidate frames (diffusers format)."
  type        = string
  default     = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

  # A Hugging Face repo id: it is put into root shell commands at first boot.
  validation {
    condition     = can(regex("^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$", var.video_model))
    error_message = "video_model must be a Hugging Face repo id like org/name."
  }
}

variable "i2v_model" {
  description = "Hugging Face repo for image-to-video from an avatar keyframe (diffusers format)."
  type        = string
  default     = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

  # A Hugging Face repo id: it is put into root shell commands at first boot.
  validation {
    condition     = can(regex("^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$", var.i2v_model))
    error_message = "i2v_model must be a Hugging Face repo id like org/name."
  }
}

variable "vlm_model" {
  description = "Hugging Face repo for the vision-language model used for descriptions and story shot lists."
  type        = string
  default     = "Qwen/Qwen2.5-VL-7B-Instruct"

  # A Hugging Face repo id: it is put into root shell commands at first boot.
  validation {
    condition     = can(regex("^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$", var.vlm_model))
    error_message = "vlm_model must be a Hugging Face repo id like org/name."
  }
}

variable "edit_model" {
  description = "Hugging Face repo for the image-edit model that places 1-3 avatars into each shot (diffusers format)."
  type        = string
  default     = "Qwen/Qwen-Image-Edit-2511"

  # A Hugging Face repo id: it is put into root shell commands at first boot.
  validation {
    condition     = can(regex("^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$", var.edit_model))
    error_message = "edit_model must be a Hugging Face repo id like org/name."
  }
}

variable "edit_lightning" {
  description = "Qwen-Image-Edit-2511-Lightning steps for photos and keyframes: 8 (default), 4 (faster, slightly harsher), or 0 (full 40-step model with CFG)."
  type        = number
  default     = 8
  validation {
    condition     = contains([0, 4, 8], var.edit_lightning)
    error_message = "edit_lightning must be 0, 4 or 8."
  }
}

variable "job_timeout_minutes" {
  description = "A job running longer than this is stopped and marked failed (timed out)."
  type        = number
  default     = 360
}

variable "max_uptime_hours" {
  description = "Hard cap: the waker stops the GPU instance after this many hours running, whatever the worker is doing, and holds it off until the next job or wake."
  type        = number
  default     = 8
}

variable "output_retention_days" {
  description = "Days to keep generated outputs and uploads in S3."
  type        = number
  default     = 30
}

variable "authorized_ssh_public_keys" {
  description = <<-EOT
    SSH public keys (authorized_keys format, ssh-ed25519 only) allowed to call
    the API and to SSH into the GPU worker. The comment field (e.g. "dan") is
    used as the client name, so give each key a distinct comment.
  EOT
  type        = list(string)

  validation {
    condition     = length(var.authorized_ssh_public_keys) > 0 && alltrue([for k in var.authorized_ssh_public_keys : can(regex("^ssh-ed25519 [A-Za-z0-9+/=]+( .+)?$", trimspace(k)))])
    error_message = "Provide at least one key, each a single 'ssh-ed25519 AAAA... comment' line."
  }
}

# ---------------------------------------------------------------- LoRA trainer (training.tf)

variable "training_enabled" {
  description = "Create the LoRA trainer: a second spot GPU instance, the train queue and its role. Off by default."
  type        = bool
  default     = false
}

variable "train_instance_type" {
  description = "Trainer instance type. g6e.4xlarge (L40S 48 GB) trains a T2V expert at ~12.7 s/step (measured); H100 spot was not available in this account."
  type        = string
  default     = "g6e.4xlarge"
}

variable "train_availability_zone" {
  description = "AZ for the trainer's subnet. Empty = the same as availability_zone. Set it to keep the trainer where it is when the render box moves to another AZ."
  type        = string
  default     = ""
}

variable "train_root_volume_gb" {
  description = "Trainer disk: ~85 GB of weights plus a job's segments, caches and saved states."
  type        = number
  default     = 300
}

variable "train_job_timeout_minutes" {
  description = "A train job longer than this is stopped and marked failed (1250 steps x 2 experts is ~9 h)."
  type        = number
  default     = 960
}

variable "train_max_uptime_hours" {
  description = "Hard cap on one continuous trainer uptime, enforced by the waker (the render box keeps max_uptime_hours)."
  type        = number
  default     = 24
}

variable "musubi_commit" {
  description = "Pinned musubi-tuner commit (third-party code running as root on the trainer). Bump only after review."
  type        = string
  default     = "f8a1b03794a49239a3539015075f5123d6c07d66"
}
