variable "aws_region" {
  description = "AWS Region for the isolated pilot stack."
  type        = string
  default     = "us-east-1"
}

variable "expected_account_id" {
  description = "Twelve-digit AWS account ID allowed for this stack."
  type        = string

  validation {
    condition     = can(regex("^[0-9]{12}$", var.expected_account_id))
    error_message = "expected_account_id must be a twelve-digit AWS account ID."
  }
}

variable "owner" {
  description = "Human owner recorded on every supported resource."
  type        = string
}

variable "expiration_date" {
  description = "Pilot review date in YYYY-MM-DD form."
  type        = string

  validation {
    condition     = can(regex("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", var.expiration_date))
    error_message = "expiration_date must use YYYY-MM-DD."
  }
}

variable "governed_bucket_name" {
  description = "Existing governed dataset bucket. This stack never creates or owns it."
  type        = string
}

variable "raw_read_prefixes" {
  description = "Exact raw prefixes the task may list and read, without leading slash."
  type        = list(string)
  default = [
    "governed/raw/Saga/",
    "governed/raw/Saga-MultiModel/",
  ]
}

variable "processed_read_prefixes" {
  description = "Exact processed-reference prefixes the task may list and read."
  type        = list(string)
  default = [
    "governed/processed/Saga/",
    "governed/processed/Saga-MultiModel/",
  ]
}

variable "staging_write_prefix" {
  description = "Manual-annotation staging prefix. No delete permission is granted."
  type        = string
  default     = "workflows/manual-annotation/"
}

variable "ground_truth_write_prefix" {
  description = "Approved ground-truth release prefix. No delete permission is granted."
  type        = string
  default     = "governed/GT/"
}

variable "container_image" {
  description = "Immutable ECR image URI with an @sha256 digest. Required only when enabling Express."
  type        = string
  default     = ""
}

variable "enable_express_service" {
  description = "Second-phase gate. Keep false until the image and secret values exist."
  type        = bool
  default     = false
}

variable "bootstrap_admin_email" {
  description = "Optional first admin Cornell email. Null disables bootstrap variables."
  type        = string
  default     = null
}

variable "bootstrap_admin_display_name" {
  description = "Display name paired with bootstrap_admin_email."
  type        = string
  default     = null
}

variable "postgres_instance_class" {
  description = "RDS instance class for the pilot."
  type        = string
  default     = "db.t4g.micro"
}

variable "postgres_allocated_storage_gib" {
  description = "Initial encrypted gp3 storage allocation."
  type        = number
  default     = 20

  validation {
    condition     = var.postgres_allocated_storage_gib >= 20
    error_message = "RDS allocated storage must be at least 20 GiB."
  }
}

variable "postgres_multi_az" {
  description = "Enable Multi-AZ RDS. False is the lower-cost two-week pilot default."
  type        = bool
  default     = false
}

variable "alert_email" {
  description = "Email subscribed to health alarms and the isolated stack budget."
  type        = string

  validation {
    condition     = can(regex("^[^@[:space:]]+@[^@[:space:]]+$", var.alert_email))
    error_message = "alert_email must be a valid email address."
  }
}

variable "monthly_budget_usd" {
  description = "Monthly cost budget for resources tagged to this pilot."
  type        = number
  default     = 100

  validation {
    condition     = var.monthly_budget_usd > 0
    error_message = "monthly_budget_usd must be positive."
  }
}
