output "ecr_repository_url" {
  value       = aws_ecr_repository.app.repository_url
  description = "Push the immutable application image here before phase two."
}

output "database_endpoint" {
  value       = aws_db_instance.postgres.endpoint
  description = "Private RDS endpoint used to populate the DATABASE_URL secret."
}

output "database_master_secret_arn" {
  value       = aws_db_instance.postgres.master_user_secret[0].secret_arn
  description = "RDS-managed master credential secret; do not expose its value."
}

output "database_url_secret_arn" {
  value = aws_secretsmanager_secret.database_url.arn
}

output "initial_admin_password_secret_arn" {
  value = aws_secretsmanager_secret.initial_admin_password.arn
}

output "express_service_arn" {
  value = var.enable_express_service ? aws_ecs_express_gateway_service.app[0].service_arn : null
}

output "express_url" {
  value = var.enable_express_service ? format(
    "%s/",
    trimsuffix(aws_ecs_express_gateway_service.app[0].ingress_paths[0].endpoint, "/"),
  ) : null
  description = "Stable only while the protected Express service is retained."
}
