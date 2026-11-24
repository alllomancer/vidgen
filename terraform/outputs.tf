output "api_url" {
  description = "Base URL for the CLI / HTTP clients."
  value       = aws_apigatewayv2_api.main.api_endpoint
}


output "instance_id" {
  value = aws_instance.gpu.id
}

output "bucket" {
  value = aws_s3_bucket.data.id
}

output "ssh_command" {
  description = "SSH to the worker over SSM (no open ports). Needs the session-manager-plugin; instance must be running."
  value       = "ssh -i ~/.ssh/vidgen_ed25519 -o ProxyCommand='aws ssm start-session --region ${var.region} --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p' ubuntu@${aws_instance.gpu.id}"
}

output "ssm_shell" {
  description = "Open a shell on the worker (instance must be running)."
  value       = "aws ssm start-session --region ${var.region} --target ${aws_instance.gpu.id}"
}

output "trainer_instance_id" {
  value = var.training_enabled ? aws_instance.trainer[0].id : null
}

output "trainer_ssm_shell" {
  value = var.training_enabled ? "aws ssm start-session --region ${var.region} --target ${aws_instance.trainer[0].id}" : null
}
