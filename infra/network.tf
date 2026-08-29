resource "aws_vpc" "pilot" {
  cidr_block           = "10.84.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "${local.name}-vpc" }
}

resource "aws_internet_gateway" "pilot" {
  vpc_id = aws_vpc.pilot.id
  tags   = { Name = "${local.name}-igw" }
}

resource "aws_subnet" "app" {
  for_each = {
    a = { cidr = "10.84.0.0/24", az = "${var.aws_region}a" }
    b = { cidr = "10.84.1.0/24", az = "${var.aws_region}b" }
  }

  vpc_id                  = aws_vpc.pilot.id
  cidr_block              = each.value.cidr
  availability_zone       = each.value.az
  map_public_ip_on_launch = true

  tags = { Name = "${local.name}-app-${each.key}" }
}

resource "aws_subnet" "database" {
  for_each = {
    a = { cidr = "10.84.10.0/24", az = "${var.aws_region}a" }
    b = { cidr = "10.84.11.0/24", az = "${var.aws_region}b" }
  }

  vpc_id            = aws_vpc.pilot.id
  cidr_block        = each.value.cidr
  availability_zone = each.value.az

  tags = { Name = "${local.name}-db-${each.key}" }
}

resource "aws_route_table" "app" {
  vpc_id = aws_vpc.pilot.id
  tags   = { Name = "${local.name}-app" }
}

resource "aws_route" "app_internet" {
  route_table_id         = aws_route_table.app.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.pilot.id
}

resource "aws_route_table_association" "app" {
  for_each = aws_subnet.app

  subnet_id      = each.value.id
  route_table_id = aws_route_table.app.id
}

resource "aws_route_table" "database" {
  vpc_id = aws_vpc.pilot.id
  tags   = { Name = "${local.name}-database-private" }
}

resource "aws_route_table_association" "database" {
  for_each = aws_subnet.database

  subnet_id      = each.value.id
  route_table_id = aws_route_table.database.id
}

resource "aws_security_group" "task_data" {
  name        = "${local.name}-task-data"
  description = "Additional task SG used only as the RDS source identity"
  vpc_id      = aws_vpc.pilot.id

  tags = { Name = "${local.name}-task-data" }
}

resource "aws_security_group" "database" {
  name        = "${local.name}-database"
  description = "Private PostgreSQL reachable only from pilot tasks"
  vpc_id      = aws_vpc.pilot.id

  tags = { Name = "${local.name}-database" }
}

resource "aws_vpc_security_group_ingress_rule" "database_from_tasks" {
  security_group_id            = aws_security_group.database.id
  referenced_security_group_id = aws_security_group.task_data.id
  ip_protocol                  = "tcp"
  from_port                    = 5432
  to_port                      = 5432
  description                  = "PostgreSQL from annotation tasks only"
}

resource "aws_vpc_security_group_egress_rule" "tasks_to_database" {
  security_group_id            = aws_security_group.task_data.id
  referenced_security_group_id = aws_security_group.database.id
  ip_protocol                  = "tcp"
  from_port                    = 5432
  to_port                      = 5432
  description                  = "PostgreSQL to the private annotation database"
}

resource "aws_vpc_security_group_egress_rule" "tasks_https" {
  security_group_id = aws_security_group.task_data.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
  description       = "HTTPS for governed S3 and ECS task startup dependencies"
}

resource "aws_vpc_security_group_egress_rule" "tasks_dns_udp" {
  security_group_id = aws_security_group.task_data.id
  cidr_ipv4         = "${cidrhost(aws_vpc.pilot.cidr_block, 2)}/32"
  ip_protocol       = "udp"
  from_port         = 53
  to_port           = 53
  description       = "DNS through the VPC resolver"
}

resource "aws_vpc_security_group_egress_rule" "tasks_dns_tcp" {
  security_group_id = aws_security_group.task_data.id
  cidr_ipv4         = "${cidrhost(aws_vpc.pilot.cidr_block, 2)}/32"
  ip_protocol       = "tcp"
  from_port         = 53
  to_port           = 53
  description       = "TCP DNS fallback through the VPC resolver"
}
