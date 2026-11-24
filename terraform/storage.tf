data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------- S3

# Holds worker code (code/), client uploads (uploads/) and results (outputs/).
resource "aws_s3_bucket" "data" {
  bucket = "${var.project}-${data.aws_caller_identity.current.account_id}-${var.region}"
}

resource "aws_s3_bucket_ownership_controls" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_policy" "data_tls_only" {
  bucket = aws_s3_bucket.data.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "expire-outputs"
    status = "Enabled"
    filter { prefix = "outputs/" }
    expiration { days = var.output_retention_days }
  }

  rule {
    id     = "expire-uploads"
    status = "Enabled"
    filter { prefix = "uploads/" }
    expiration { days = var.output_retention_days }
  }

  # Multipart uploads a client never completed or aborted (laptop closed mid-upload): parts deleted.
  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload { days_after_initiation = 1 }
  }
}

# Worker code is shipped through S3 and re-synced on every boot, so
# `terraform apply` + next start is all a code change needs.
resource "aws_s3_object" "worker_code" {
  # Local bytecode caches are never shipped.
  for_each = toset([for f in fileset("${path.module}/../worker", "**") : f if !strcontains(f, "__pycache__")])

  bucket = aws_s3_bucket.data.id
  key    = "code/worker/${each.value}"
  source = "${path.module}/../worker/${each.value}"
  etag   = filemd5("${path.module}/../worker/${each.value}")
}

# Settings the worker reads next to its code (refreshed on every worker start, unlike user data).
resource "aws_s3_object" "worker_settings" {
  bucket       = aws_s3_bucket.data.id
  key          = "code/worker/settings.json"
  content_type = "application/json"
  content = jsonencode({
    train_queue_url = var.training_enabled ? aws_sqs_queue.train[0].url : ""
  })
}

# ---------------------------------------------------------------- SQS

resource "aws_sqs_queue" "jobs_dlq" {
  name                      = "${var.project}-jobs-dlq"
  message_retention_seconds = 1209600
}

resource "aws_sqs_queue" "jobs" {
  name                      = "${var.project}-jobs"
  message_retention_seconds = 86400
  receive_wait_time_seconds = 20
  # Longest expected job (a multi-shot story) plus headroom. The worker also
  # extends visibility while a job runs.
  visibility_timeout_seconds = 1800
  sqs_managed_sse_enabled    = true

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.jobs_dlq.arn
    maxReceiveCount     = 3
  })
}

# ---------------------------------------------------------------- DynamoDB

resource "aws_dynamodb_table" "jobs" {
  name         = "${var.project}-jobs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "job_id"

  attribute {
    name = "job_id"
    type = "S"
  }

  attribute {
    name = "client"
    type = "S"
  }

  attribute {
    name = "created_at"
    type = "N"
  }

  # Job history per client, newest first (GET /jobs).
  global_secondary_index {
    name            = "client-created"
    hash_key        = "client"
    range_key       = "created_at"
    projection_type = "ALL"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = true
  }
}

# Each signed request carries a single-use nonce; this table rejects replays.
resource "aws_dynamodb_table" "nonces" {
  name         = "${var.project}-nonces"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "nonce"

  attribute {
    name = "nonce"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

# The worker copies this into ~ubuntu/.ssh/authorized_keys on every boot, so
# key changes reach the instance on its next start.
resource "aws_s3_object" "authorized_keys" {
  bucket  = aws_s3_bucket.data.id
  key     = "code/authorized_keys"
  content = join("\n", [for k in var.authorized_ssh_public_keys : trimspace(k)])
}
