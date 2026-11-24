terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.6"
    }
  }

  # Uncomment and fill in to keep state remotely.
  # backend "s3" {
  #   bucket = "my-terraform-state"
  #   key    = "ai-video-pipeline/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.region

  # The org's FinOps automation tags resources (FinOpsC7NStoppedDate, FinOpsAmiId, ...); not ours to remove.
  ignore_tags {
    key_prefixes = ["FinOps"]
  }

  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
    }
  }
}
