# Always-on, pay-per-request front door. API Gateway HTTP API -> Lambda.
# The Lambda verifies the request's SSH signature, records the job, queues it and starts
# the GPU instance if it is stopped. The GPU box itself has no inbound access.

# Built by `make build` (handler + Linux wheels for `cryptography`).
data "archive_file" "api" {
  type        = "zip"
  source_dir  = "${path.module}/.build/api_src"
  output_path = "${path.module}/.build/api.zip"
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${var.project}-api"
  retention_in_days = 30
}

resource "aws_lambda_function" "api" {
  function_name    = "${var.project}-api"
  role             = aws_iam_role.api.arn
  runtime          = "python3.12"
  handler          = "handler.handler"
  filename         = data.archive_file.api.output_path
  source_code_hash = data.archive_file.api.output_base64sha256
  timeout          = 15
  memory_size      = 256

  environment {
    variables = {
      BUCKET           = aws_s3_bucket.data.id
      QUEUE_URL        = aws_sqs_queue.jobs.url
      JOBS_TABLE       = aws_dynamodb_table.jobs.name
      INSTANCE_ID      = aws_instance.gpu.id
      NONCES_TABLE     = aws_dynamodb_table.nonces.name
      AUTHORIZED_KEYS  = join("\n", [for k in var.authorized_ssh_public_keys : trimspace(k)])
      RETENTION_DAYS   = tostring(var.output_retention_days)
      MAX_UPTIME_HOURS = tostring(var.max_uptime_hours)
      # LoRA trainer (empty when training_enabled = false).
      TRAINER_INSTANCE_ID    = var.training_enabled ? aws_instance.trainer[0].id : ""
      TRAIN_QUEUE_URL        = var.training_enabled ? aws_sqs_queue.train[0].url : ""
      TRAIN_MAX_UPTIME_HOURS = tostring(var.train_max_uptime_hours)
      # Same variables as the GPU box's env file, reported by GET /status.
      MODELS = jsonencode({
        video = var.video_model
        i2v   = var.i2v_model
        edit  = var.edit_model
        vlm   = var.vlm_model
      })
    }
  }

  depends_on = [aws_cloudwatch_log_group.api]
}

resource "aws_apigatewayv2_api" "main" {
  name          = var.project
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "api" {
  api_id                 = aws_apigatewayv2_api.main.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "routes" {
  for_each = toset([
    "POST /jobs",
    "GET /jobs",
    "POST /jobs/{job_id}/rerun",
    "GET /jobs/{job_id}",
    "POST /jobs/{job_id}/cancel",
    "POST /uploads",
    "GET /status",
    "POST /wake",
    "GET /avatars",
    "POST /avatars",
    "PUT /avatars/default",
    "GET /avatars/{name}",
    "POST /avatars/{name}/images",
    "DELETE /avatars/{name}/images/{n}",
    "PUT /avatars/{name}/main",
    "GET /avatars/{name}/history",
    "POST /avatars/{name}/restore",
    "POST /avatars/{name}/krea-lora",
    "POST /avatars/{name}/media",
    "POST /avatars/{name}/wan-lora",
    "GET /avatars/{name}/media",
    "GET /datasets",
    "GET /datasets/{name}",
    "POST /datasets/{name}/uploads",
    "PUT /datasets/{name}",
    "DELETE /datasets/{name}",
    "GET /loras",
    "PUT /loras/{name}/current",
    "POST /loras/{name}/uploads",
    "PUT /loras/{name}/versions/{version}",
    "GET /loras/{name}/versions/{version}/files",
    "POST /uploads/complete",
    "POST /uploads/abort",
    "GET /datasets/{name}/captions",
    "PUT /datasets/{name}/captions",
  ])

  api_id    = aws_apigatewayv2_api.main.id
  route_key = each.value
  target    = "integrations/${aws_apigatewayv2_integration.api.id}"
}

resource "aws_cloudwatch_log_group" "apigw" {
  name              = "/aws/apigateway/${var.project}"
  retention_in_days = 30
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = "$default"
  auto_deploy = true

  # Cheap brute-force / runaway-client protection.
  default_route_settings {
    throttling_burst_limit = 20
    throttling_rate_limit  = 10
  }

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.apigw.arn
    format = jsonencode({
      requestId = "$context.requestId"
      ip        = "$context.identity.sourceIp"
      route     = "$context.routeKey"
      status    = "$context.status"
      latency   = "$context.responseLatency"
    })
  }
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowApiGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*"
}

# ---------------------------------------------------------------- Waker

# Every minute: if jobs are waiting and the instance is stopped, start it.
# Covers spot interruptions, a start that failed for lack of capacity, and a
# job that arrived while the instance was mid-stop. Also enforces
# max_uptime_hours on a running instance.
resource "aws_cloudwatch_event_rule" "waker" {
  name                = "${var.project}-waker"
  schedule_expression = "rate(1 minute)"
}

resource "aws_cloudwatch_event_target" "waker" {
  rule  = aws_cloudwatch_event_rule.waker.name
  arn   = aws_lambda_function.api.arn
  input = jsonencode({ waker = true })
}

resource "aws_lambda_permission" "waker" {
  statement_id  = "AllowEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.waker.arn
}
