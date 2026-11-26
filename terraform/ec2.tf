# GPU worker. A persistent spot instance whose interruption behaviour is
# "stop", so it can be stopped and started like an on-demand instance while
# keeping its root volume (OS, Python env, model weights). With
# var.on_demand it is a plain on-demand instance: same lifecycle, no
# interruptions, on-demand price.
#
# Lifecycle:
#   - The API Lambda (and a 1-minute EventBridge "waker") starts it when jobs
#     are queued.
#   - The worker stops its own instance after var.idle_minutes with an empty
#     queue.
#   - If AWS reclaims spot capacity, the instance is stopped; the in-flight job
#     returns to the queue and the waker restarts the instance once capacity is
#     back.

data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

resource "aws_instance" "gpu" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnet.gpu.id
  vpc_security_group_ids = [aws_security_group.worker.id]

  # Private subnet: no public IP (the org SCP denies it); egress is via the VPC's transit gateway.
  associate_public_ip_address = false
  iam_instance_profile        = aws_iam_instance_profile.worker.name

  # Spot unless var.on_demand: an on-demand box has no market options at all.
  dynamic "instance_market_options" {
    for_each = var.on_demand ? [] : [1]
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "persistent"
        instance_interruption_behavior = "stop"
        max_price                      = var.spot_max_price == "" ? null : var.spot_max_price
      }
    }
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    iops                  = 6000
    throughput            = var.root_volume_throughput
    encrypted             = true
    delete_on_termination = true
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  user_data_replace_on_change = false
  user_data = templatefile("${path.module}/templates/user_data.sh.tpl", {
    region         = var.region
    bucket         = aws_s3_bucket.data.id
    queue_url      = aws_sqs_queue.jobs.url
    jobs_table     = aws_dynamodb_table.jobs.name
    idle_minutes   = var.idle_minutes
    video_model    = var.video_model
    i2v_model      = var.i2v_model
    vlm_model      = var.vlm_model
    edit_model     = var.edit_model
    edit_lightning = var.edit_lightning
    job_timeout    = var.job_timeout_minutes
  })

  tags = {
    Name = var.render_instance_name
    Role = "gpu-worker"
  }

  # Worker code lands in S3 before the first boot tries to sync it.
  depends_on = [aws_s3_object.worker_code]

  lifecycle {
    # A newer DLAMI should not replace the instance (and its downloaded models). user_data only
    # runs on first boot, so a template change must not stop/start the box; change
    # /etc/ai-video.env on the box instead (RUNBOOK).
    # FinOps* tags from the org's automation are ignored provider-wide (versions.tf).
    ignore_changes = [ami, user_data]
  }
}
