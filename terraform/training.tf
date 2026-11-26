# LoRA trainer (docs/specs/2026-11-09-lora-training.md, "Trainer: step 2").
# Everything here exists only with training_enabled = true.
#
# A second persistent spot GPU instance (stop on interruption, like the render box) that
# pulls `train` jobs from its own queue. The API queues train jobs there and the waker
# starts it; it stops itself after idle_minutes with an empty queue. It never touches the
# render box, and the two can run at the same time.

locals {
  training = var.training_enabled ? 1 : 0
}

# ---------------------------------------------------------------- queue

resource "aws_sqs_queue" "train_dlq" {
  count                     = local.training
  name                      = "${var.project}-train-dlq"
  message_retention_seconds = 1209600
  sqs_managed_sse_enabled   = true
}

resource "aws_sqs_queue" "train" {
  count                      = local.training
  name                       = "${var.project}-train"
  message_retention_seconds  = 345600 # 4 days: a queued training waits for spot capacity
  receive_wait_time_seconds  = 20
  visibility_timeout_seconds = 1800 # the trainer extends it while a job runs
  sqs_managed_sse_enabled    = true

  # Redelivery only happens when the trainer stops mid-job (spot interruption); it then
  # resumes from saved state. Failed jobs are deleted by the trainer, not redelivered.
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.train_dlq[0].arn
    maxReceiveCount     = 6
  })
}

# ---------------------------------------------------------------- code

resource "aws_s3_object" "trainer_code" {
  for_each = var.training_enabled ? toset([
    for f in fileset("${path.module}/../trainer", "**") : f
    if !strcontains(f, "__pycache__") && !startswith(f, "spike/")
  ]) : toset([])

  bucket = aws_s3_bucket.data.id
  key    = "code/trainer/${each.value}"
  source = "${path.module}/../trainer/${each.value}"
  etag   = filemd5("${path.module}/../trainer/${each.value}")
}

# The model mirror module is the render worker's file: one source, shipped to both machines.
resource "aws_s3_object" "trainer_mirror" {
  count = local.training

  bucket = aws_s3_bucket.data.id
  key    = "code/trainer/mirror.py"
  source = "${path.module}/../worker/mirror.py"
  etag   = filemd5("${path.module}/../worker/mirror.py")
}

# Where the trainer queues the checkpoint previews when a training finishes. A file rather
# than the env: the trainer's env file is only written at first boot.
resource "aws_s3_object" "trainer_render_config" {
  count        = local.training
  bucket       = aws_s3_bucket.data.id
  key          = "code/trainer/render.json"
  content      = jsonencode({ queue_url = aws_sqs_queue.jobs.url, retention_days = var.output_retention_days })
  content_type = "application/json"
}

# ---------------------------------------------------------------- role

resource "aws_iam_role" "trainer" {
  count              = local.training
  name               = "${var.project}-trainer"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "trainer_ssm" {
  count      = local.training
  role       = aws_iam_role.trainer[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "trainer" {
  count = local.training

  statement {
    sid       = "Queue"
    actions   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:ChangeMessageVisibility", "sqs:GetQueueAttributes"]
    resources = [aws_sqs_queue.train[0].arn]
  }

  statement {
    sid       = "Jobs"
    # PutItem: the checkpoint-preview job queued when a training finishes.
    actions   = ["dynamodb:GetItem", "dynamodb:UpdateItem", "dynamodb:PutItem"]
    resources = [aws_dynamodb_table.jobs.arn]
  }

  statement {
    sid       = "QueuePreview"
    actions   = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.jobs.arn]
  }

  statement {
    sid       = "WriteCaptions"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.data.arn}/datasets/*/captions.json"]
  }

  statement {
    sid       = "ReadCodeAndDatasets"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.data.arn}/code/trainer/*", "${aws_s3_bucket.data.arn}/datasets/*"]
  }

  # Krea 2 avatar LoRAs: the gated Krea 2 RAW download needs the HF token; the build's state lives on
  # avatar.json (docs/specs/2026-11-24-krea-avatar-lora.md).
  statement {
    sid       = "HfToken"
    actions   = ["ssm:GetParameter"]
    resources = ["arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter/ai-video/hf_token"]
  }

  statement {
    sid       = "AvatarKreaLoraState"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.data.arn}/avatars/*/avatar.json"]
  }

  statement {
    sid       = "WriteLoras"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.data.arn}/loras/*"]
  }

  statement {
    sid       = "List"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["code/trainer/*", "loras/*", "models/trainer/*"]
    }
  }

  # Model mirror (docs/specs/2026-11-25-model-mirror.md): the trainer restores its weights from the bucket
  # and uploads what the mirror lacks. Only its own folders: neither machine restores what the other wrote.
  statement {
    sid       = "ModelMirror"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"]
    resources = ["${aws_s3_bucket.data.arn}/models/trainer/*"]
  }

  statement {
    sid       = "StopSelf"
    actions   = ["ec2:StopInstances"]
    resources = ["arn:aws:ec2:${var.region}:${data.aws_caller_identity.current.account_id}:instance/*"]
    condition {
      test     = "StringEquals"
      variable = "aws:ResourceTag/Project"
      values   = [var.project]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:ResourceTag/Role"
      values   = ["lora-trainer"]
    }
  }
}

resource "aws_iam_role_policy" "trainer" {
  count  = local.training
  name   = "trainer"
  role   = aws_iam_role.trainer[0].id
  policy = data.aws_iam_policy_document.trainer[0].json
}

resource "aws_iam_instance_profile" "trainer" {
  count = local.training
  name  = "${var.project}-trainer"
  role  = aws_iam_role.trainer[0].name
}

# ---------------------------------------------------------------- instance

resource "aws_instance" "trainer" {
  count                       = local.training
  ami                         = data.aws_ami.dlami.id
  instance_type               = var.train_instance_type
  subnet_id                   = data.aws_subnet.trainer.id
  vpc_security_group_ids      = [aws_security_group.worker.id]
  associate_public_ip_address = false # private subnet; the org SCP denies public IPs
  iam_instance_profile        = aws_iam_instance_profile.trainer[0].name

  instance_market_options {
    market_type = "spot"
    spot_options {
      spot_instance_type             = "persistent"
      instance_interruption_behavior = "stop" # keeps the disk, so a job resumes from saved state
      max_price                      = var.spot_max_price == "" ? null : var.spot_max_price
    }
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.train_root_volume_gb
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
  user_data = templatefile("${path.module}/templates/trainer_user_data.sh.tpl", {
    region          = var.region
    bucket          = aws_s3_bucket.data.id
    train_queue_url = aws_sqs_queue.train[0].url
    jobs_table      = aws_dynamodb_table.jobs.name
    idle_minutes    = var.idle_minutes
    job_timeout     = var.train_job_timeout_minutes
    vlm_model       = var.vlm_model
    musubi_commit   = var.musubi_commit
  })

  tags = {
    Name = var.trainer_instance_name
    Role = "lora-trainer"
  }

  depends_on = [aws_s3_object.trainer_code, aws_s3_object.trainer_mirror]

  lifecycle {
    # user_data only runs on first boot; a template change must not stop/start a running trainer.
    # FinOps* tags from the org's automation are ignored provider-wide (versions.tf).
    ignore_changes = [ami, user_data]
  }
}
