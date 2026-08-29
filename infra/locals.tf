locals {
  name                 = "deid-annotation-pilot"
  service_name         = "deid-annotation-pilot"
  express_host_pattern = "*.ecs.${var.aws_region}.on.aws"

  tags = {
    Project     = local.name
    Environment = "pilot"
    ManagedBy   = "terraform"
    Owner       = var.owner
    ExpiresOn   = var.expiration_date
    Repository  = "De-id-manual-annotation"
  }

  read_prefixes = concat(var.raw_read_prefixes, var.processed_read_prefixes)
  read_objects = [
    for prefix in local.read_prefixes :
    "arn:aws:s3:::${var.governed_bucket_name}/${trimprefix(prefix, "/")}*"
  ]
  list_prefixes = [
    for prefix in local.read_prefixes :
    "${trimprefix(prefix, "/")}*"
  ]

  bootstrap_enabled = var.bootstrap_admin_email != null
}

check "express_inputs" {
  assert {
    condition = !var.enable_express_service || can(regex(
      "@sha256:[0-9a-f]{64}$",
      var.container_image,
    ))
    error_message = "container_image must use an immutable @sha256 digest before Express is enabled."
  }
}

check "bootstrap_inputs" {
  assert {
    condition = (
      (var.bootstrap_admin_email == null && var.bootstrap_admin_display_name == null) ||
      (var.bootstrap_admin_email != null && var.bootstrap_admin_display_name != null)
    )
    error_message = "bootstrap admin email and display name must be set together."
  }
}
