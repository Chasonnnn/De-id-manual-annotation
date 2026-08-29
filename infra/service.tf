resource "aws_ecs_cluster" "pilot" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${local.name}"
  retention_in_days = 30

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_ecs_express_gateway_service" "app" {
  count = var.enable_express_service ? 1 : 0

  service_name            = local.service_name
  cluster                 = aws_ecs_cluster.pilot.name
  execution_role_arn      = aws_iam_role.execution.arn
  infrastructure_role_arn = aws_iam_role.infrastructure.arn
  task_role_arn           = aws_iam_role.task.arn

  cpu                   = "512"
  memory                = "1024"
  health_check_path     = "/api/health"
  wait_for_steady_state = true

  network_configuration {
    subnets         = [for subnet in aws_subnet.app : subnet.id]
    security_groups = [aws_security_group.task_data.id]
  }

  primary_container {
    image          = var.container_image
    container_port = 8000

    aws_logs_configuration {
      log_group         = aws_cloudwatch_log_group.app.name
      log_stream_prefix = "app"
    }

    environment {
      name  = "HOSTED_ALLOWED_HOSTS"
      value = local.express_host_pattern
    }

    environment {
      name  = "HOSTED_COOKIE_SECURE"
      value = "true"
    }

    environment {
      name  = "HOSTED_S3_BUCKET"
      value = var.governed_bucket_name
    }

    environment {
      name  = "HOSTED_S3_RAW_PREFIXES"
      value = join(",", var.raw_read_prefixes)
    }

    environment {
      name  = "HOSTED_S3_REFERENCE_PREFIXES"
      value = join(",", var.processed_read_prefixes)
    }

    secret {
      name       = "DATABASE_URL"
      value_from = aws_secretsmanager_secret.database_url.arn
    }

    dynamic "environment" {
      for_each = local.bootstrap_enabled ? {
        INITIAL_ADMIN_EMAIL        = var.bootstrap_admin_email
        INITIAL_ADMIN_DISPLAY_NAME = var.bootstrap_admin_display_name
      } : {}
      content {
        name  = environment.key
        value = environment.value
      }
    }

    dynamic "secret" {
      for_each = local.bootstrap_enabled ? [1] : []
      content {
        name       = "INITIAL_ADMIN_PASSWORD"
        value_from = aws_secretsmanager_secret.initial_admin_password.arn
      }
    }
  }

  scaling_target {
    auto_scaling_metric       = "AVERAGE_CPU"
    auto_scaling_target_value = 60
    min_task_count            = 1
    max_task_count            = 2
  }

  lifecycle {
    prevent_destroy = true

    precondition {
      condition     = can(regex("@sha256:[0-9a-f]{64}$", var.container_image))
      error_message = "Express requires an immutable image digest."
    }
  }

  depends_on = [
    aws_iam_role_policy.execution,
    aws_iam_role_policy_attachment.infrastructure,
    aws_iam_role_policy.dataset_read,
    aws_iam_role_policy.staging_write,
    aws_iam_role_policy.ground_truth_write,
  ]
}
