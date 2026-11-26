# Uses the account's existing VPC and a public subnet in the chosen AZ (the
# organisation's SCPs block creating a new VPC). The instance gets a public IP
# for outbound traffic only (model downloads, AWS APIs); the security group
# allows no inbound traffic. Shell access is through SSM Session Manager.

data "aws_vpc" "main" {
  filter {
    name   = "tag:Name"
    values = [var.vpc_name]
  }
}

data "aws_subnet" "gpu" {
  vpc_id            = data.aws_vpc.main.id
  availability_zone = var.availability_zone

  filter {
    name   = "tag:Visibility"
    values = [var.subnet_visibility_tag]
  }
}

# The trainer keeps its own AZ when the render box moves: moving replaces an instance and its disk.
data "aws_subnet" "trainer" {
  vpc_id            = data.aws_vpc.main.id
  availability_zone = var.train_availability_zone == "" ? var.availability_zone : var.train_availability_zone

  filter {
    name   = "tag:Visibility"
    values = [var.subnet_visibility_tag]
  }
}

resource "aws_security_group" "worker" {
  name        = "${var.project}-worker"
  description = "GPU worker: no inbound, all outbound"
  vpc_id      = data.aws_vpc.main.id

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
