# ---------------------------------------------------------------- GPU worker

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "worker" {
  name               = "${var.project}-worker"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "worker_ssm" {
  role       = aws_iam_role.worker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "worker" {
  # Krea 2 Turbo is gated on Hugging Face: the worker reads your token from this SecureString
  # (created by hand, never in Terraform state; see RUNBOOK "Krea 2").
  statement {
    sid       = "HfToken"
    actions   = ["ssm:GetParameter"]
    resources = ["arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter/ai-video/hf_token"]
  }

  statement {
    sid = "Queue"
    actions = [
      "sqs:ReceiveMessage",
      "sqs:DeleteMessage",
      "sqs:ChangeMessageVisibility",
      "sqs:GetQueueAttributes",
    ]
    resources = [aws_sqs_queue.jobs.arn]
  }

  statement {
    sid       = "Jobs"
    actions   = ["dynamodb:GetItem", "dynamodb:UpdateItem"]
    resources = [aws_dynamodb_table.jobs.arn]
  }

  statement {
    sid       = "ListBucket"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["code/*", "uploads/*", "models/*"]
    }
  }

  # Model mirror (docs/specs/2026-11-25-model-mirror.md): the render box restores its weights from the
  # bucket and uploads what the mirror lacks. Its own units only; models/trainer/ belongs to the trainer.
  statement {
    sid     = "ModelMirror"
    actions = ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"]
    resources = [
      "${aws_s3_bucket.data.arn}/models/hub/*",
      "${aws_s3_bucket.data.arn}/models/bf16/*",
      "${aws_s3_bucket.data.arn}/models/mmaudio/*",
      "${aws_s3_bucket.data.arn}/models/krea-2-turbo/*",
      "${aws_s3_bucket.data.arn}/models/nsfw-filter/*",
      "${aws_s3_bucket.data.arn}/models/helios/*",
    ]
  }

  statement {
    sid     = "ReadCodeUploadsAvatars"
    actions = ["s3:GetObject"]
    resources = [
      "${aws_s3_bucket.data.arn}/code/*",
      "${aws_s3_bucket.data.arn}/uploads/*",
      "${aws_s3_bucket.data.arn}/avatars/*",
      # Trained LoRAs for video/story jobs that ask for them (--lora).
      "${aws_s3_bucket.data.arn}/loras/*",
    ]
  }

  # Avatar media for an existing avatar reuses the photos of its Krea LoRA dataset.
  statement {
    sid       = "ReadAvatarDatasets"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.data.arn}/datasets/*/avatar-*"]
  }

  statement {
    sid       = "WriteOutputs"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.data.arn}/outputs/*"]
  }

  # The dashboard's "ready to render" line: first boot and the worker write it, GET /status reads it.
  statement {
    sid       = "WriteReadiness"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.data.arn}/status/render.json"]
  }

  # Krea 2 avatar LoRAs: the avatar_photoset job writes the avatar's dataset, updates its krea_lora
  # state, and queues the train job (docs/specs/2026-11-24-krea-avatar-lora.md).
  statement {
    sid     = "AvatarLoraDataset"
    actions = ["s3:PutObject"]
    resources = [
      "${aws_s3_bucket.data.arn}/datasets/*/avatar-*",
      "${aws_s3_bucket.data.arn}/avatars/*/avatar.json",
      # Avatar media: photos, 2 s clips and media.json (docs/specs/2026-11-24-avatar-media.md).
      "${aws_s3_bucket.data.arn}/avatars/*/media/*",
    ]
  }

  statement {
    sid       = "QueueTrainJob"
    actions   = ["dynamodb:PutItem"]
    resources = [aws_dynamodb_table.jobs.arn]
  }

  dynamic "statement" {
    for_each = var.training_enabled ? [1] : []
    content {
      sid       = "SendTrainQueue"
      actions   = ["sqs:SendMessage"]
      resources = [aws_sqs_queue.train[0].arn]
    }
  }

  # Lets the worker stop itself when idle. Scoped by tag so it cannot touch
  # any other instance in the account.
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
      values   = ["gpu-worker"]
    }
  }
}

resource "aws_iam_role_policy" "worker" {
  name   = "worker"
  role   = aws_iam_role.worker.id
  policy = data.aws_iam_policy_document.worker.json
}

resource "aws_iam_instance_profile" "worker" {
  name = "${var.project}-worker"
  role = aws_iam_role.worker.name
}

# ---------------------------------------------------------------- API Lambda

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "api" {
  name               = "${var.project}-api"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "api_logs" {
  role       = aws_iam_role.api.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "api" {
  statement {
    sid       = "Queue"
    actions   = ["sqs:SendMessage", "sqs:GetQueueAttributes"]
    resources = [aws_sqs_queue.jobs.arn]
  }

  statement {
    sid       = "Jobs"
    actions   = ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:UpdateItem"]
    resources = [aws_dynamodb_table.jobs.arn]
  }

  statement {
    sid       = "JobHistory"
    actions   = ["dynamodb:Query"]
    resources = ["${aws_dynamodb_table.jobs.arn}/index/client-created"]
  }

  # GET /status reports what the deployed worker code pins (Lightning, MMAudio).
  statement {
    sid       = "ReadWorkerModels"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.data.arn}/code/worker/models.py"]
  }

  # Presigned URLs carry the Lambda's permissions. avatars/ is written by
  # copying an upload or a generated candidate into it.
  statement {
    sid     = "ObjectsReadWrite"
    actions = ["s3:GetObject", "s3:PutObject"]
    resources = [
      "${aws_s3_bucket.data.arn}/outputs/*",
      "${aws_s3_bucket.data.arn}/uploads/*",
      "${aws_s3_bucket.data.arn}/avatars/*",
    ]
  }

  # The render machine's readiness report (written by the box), shown by GET /status.
  statement {
    sid       = "ReadReadiness"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.data.arn}/status/render.json"]
  }

  # DELETE /avatars/{name}: the avatar's folder and the default marker, and (when asked) the avatar's own
  # LoRAs. Uploaded LoRAs cannot be named avatar-*, so no other LoRA is in reach.
  statement {
    sid     = "DeleteAvatars"
    actions = ["s3:DeleteObject"]
    resources = [
      "${aws_s3_bucket.data.arn}/avatars/*",
      "${aws_s3_bucket.data.arn}/loras/*/avatar-*",
    ]
  }

  # Learning datasets: presigned uploads (PutObject), manifest check/commit, delete.
  statement {
    sid       = "Datasets"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.data.arn}/datasets/*"]
  }

  statement {
    sid       = "ListDatasets"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["datasets/*"]
    }
  }

  statement {
    sid       = "ListAvatars"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["avatars/*"]
    }
  }

  statement {
    sid       = "Nonces"
    actions   = ["dynamodb:PutItem"]
    resources = [aws_dynamodb_table.nonces.arn]
  }

  statement {
    sid       = "DescribeInstance"
    actions   = ["ec2:DescribeInstances"]
    resources = ["*"]
  }

  statement {
    sid       = "StartStopWorker"
    actions   = ["ec2:StartInstances", "ec2:StopInstances"]
    resources = [aws_instance.gpu.arn]
  }

  # LoRA trainer, only with training_enabled.
  dynamic "statement" {
    for_each = var.training_enabled ? [1] : []
    content {
      sid       = "TrainQueue"
      actions   = ["sqs:SendMessage", "sqs:GetQueueAttributes"]
      resources = [aws_sqs_queue.train[0].arn]
    }
  }

  dynamic "statement" {
    for_each = var.training_enabled ? [1] : []
    content {
      sid       = "StartStopTrainer"
      actions   = ["ec2:StartInstances", "ec2:StopInstances"]
      resources = [aws_instance.trainer[0].arn]
    }
  }

  # Train jobs check the dataset manifest; GET /loras reads lora.json files.
  statement {
    sid       = "ReadLoras"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.data.arn}/loras/*"]
  }

  # PUT /loras/{name}/current picks the version/checkpoint renders use; nothing else is writable.
  statement {
    sid       = "PickLora"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.data.arn}/loras/*/current.json"]
  }

  # Uploaded LoRAs: presigned PUTs to a new version's t2v_high/low.safetensors (Wan) or
  # krea2.safetensors (Krea 2), then lora.json once the files pass the header checks.
  statement {
    sid     = "UploadLoras"
    actions = ["s3:PutObject"]
    resources = [
      "${aws_s3_bucket.data.arn}/loras/*/t2v_high.safetensors",
      "${aws_s3_bucket.data.arn}/loras/*/t2v_low.safetensors",
      "${aws_s3_bucket.data.arn}/loras/*/krea2.safetensors",
      "${aws_s3_bucket.data.arn}/loras/*/lora.json",
    ]
  }

  # Parallel multipart uploads of big LoRA and dataset files (create/complete use PutObject; abort is its own).
  statement {
    sid     = "AbortMultipartUploads"
    actions = ["s3:AbortMultipartUpload"]
    resources = [
      "${aws_s3_bucket.data.arn}/loras/*/t2v_high.safetensors",
      "${aws_s3_bucket.data.arn}/loras/*/t2v_low.safetensors",
      "${aws_s3_bucket.data.arn}/loras/*/krea2.safetensors",
      "${aws_s3_bucket.data.arn}/datasets/*",
    ]
  }

  statement {
    sid       = "ListLoras"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["loras/*"]
    }
  }
}

resource "aws_iam_role_policy" "api" {
  name   = "api"
  role   = aws_iam_role.api.id
  policy = data.aws_iam_policy_document.api.json
}
