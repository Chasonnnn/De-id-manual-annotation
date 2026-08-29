resource "aws_kms_key" "pilot" {
  description             = "Encryption key for ${local.name}"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_kms_alias" "pilot" {
  name          = "alias/${local.name}"
  target_key_id = aws_kms_key.pilot.key_id
}

resource "aws_ecr_repository" "app" {
  name                 = local.name
  image_tag_mutability = "IMMUTABLE"

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.pilot.arn
  }

  image_scanning_configuration {
    scan_on_push = true
  }

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_secretsmanager_secret" "database_url" {
  name                    = "${local.name}/database-url"
  description             = "Complete sslmode=require DATABASE_URL for the hosted app"
  kms_key_id              = aws_kms_key.pilot.arn
  recovery_window_in_days = 30

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_secretsmanager_secret" "initial_admin_password" {
  name                    = "${local.name}/initial-admin-password"
  description             = "One-time initial administrator password; remove from service after bootstrap"
  kms_key_id              = aws_kms_key.pilot.arn
  recovery_window_in_days = 30

  lifecycle {
    prevent_destroy = true
  }
}
