resource "aws_db_subnet_group" "pilot" {
  name       = local.name
  subnet_ids = [for subnet in aws_subnet.database : subnet.id]
}

resource "aws_db_parameter_group" "pilot" {
  name   = local.name
  family = "postgres18"

  parameter {
    name         = "rds.force_ssl"
    value        = "1"
    apply_method = "immediate"
  }
}

resource "aws_db_instance" "postgres" {
  identifier = local.name

  engine         = "postgres"
  engine_version = "18.6"
  instance_class = var.postgres_instance_class

  db_name  = "annotation"
  username = "annotationadmin"
  port     = 5432

  manage_master_user_password   = true
  master_user_secret_kms_key_id = aws_kms_key.pilot.arn

  allocated_storage     = var.postgres_allocated_storage_gib
  max_allocated_storage = max(100, var.postgres_allocated_storage_gib)
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id            = aws_kms_key.pilot.arn

  db_subnet_group_name   = aws_db_subnet_group.pilot.name
  parameter_group_name   = aws_db_parameter_group.pilot.name
  vpc_security_group_ids = [aws_security_group.database.id]
  publicly_accessible    = false
  multi_az               = var.postgres_multi_az

  backup_retention_period   = 14
  backup_window             = "05:00-06:00"
  maintenance_window        = "sun:06:30-sun:07:30"
  copy_tags_to_snapshot     = true
  deletion_protection       = true
  skip_final_snapshot       = false
  final_snapshot_identifier = "${local.name}-final"

  auto_minor_version_upgrade      = true
  apply_immediately               = false
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  lifecycle {
    prevent_destroy = true
  }
}
