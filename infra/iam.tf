data "aws_iam_policy_document" "ecs_tasks_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "ecs_infrastructure_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

data "aws_iam_policy_document" "execution" {
  statement {
    sid       = "EcrAuthorization"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "PullPilotImage"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [aws_ecr_repository.app.arn]
  }

  statement {
    sid    = "WritePilotLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["${aws_cloudwatch_log_group.app.arn}:*"]
  }

  statement {
    sid     = "ReadPilotSecrets"
    effect  = "Allow"
    actions = ["secretsmanager:GetSecretValue"]
    resources = [
      aws_secretsmanager_secret.database_url.arn,
      aws_secretsmanager_secret.initial_admin_password.arn,
    ]
  }

  statement {
    sid       = "DecryptPilotSecrets"
    effect    = "Allow"
    actions   = ["kms:Decrypt"]
    resources = [aws_kms_key.pilot.arn]
  }
}

resource "aws_iam_role_policy" "execution" {
  name   = "${local.name}-execution"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution.json
}

resource "aws_iam_role" "infrastructure" {
  name               = "${local.name}-infrastructure"
  assume_role_policy = data.aws_iam_policy_document.ecs_infrastructure_trust.json
}

resource "aws_iam_role_policy_attachment" "infrastructure" {
  role       = aws_iam_role.infrastructure.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices"
}

resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

data "aws_iam_policy_document" "dataset_read" {
  statement {
    sid       = "ReadGovernedBucketLocation"
    effect    = "Allow"
    actions   = ["s3:GetBucketLocation"]
    resources = ["arn:aws:s3:::${var.governed_bucket_name}"]
  }

  statement {
    sid    = "ListOnlyApprovedDatasetPrefixes"
    effect = "Allow"
    actions = [
      "s3:ListBucket",
      "s3:ListBucketVersions",
    ]
    resources = ["arn:aws:s3:::${var.governed_bucket_name}"]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = local.list_prefixes
    }
  }

  statement {
    sid    = "ReadOnlyApprovedDatasetObjects"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:GetObjectAttributes",
      "s3:GetObjectTagging",
      "s3:GetObjectVersion",
      "s3:GetObjectVersionTagging",
    ]
    resources = local.read_objects
  }
}

resource "aws_iam_role_policy" "dataset_read" {
  name   = "${local.name}-dataset-read"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.dataset_read.json
}

data "aws_iam_policy_document" "staging_write" {
  statement {
    sid    = "WriteManualAnnotationStaging"
    effect = "Allow"
    actions = [
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts",
      "s3:PutObject",
      "s3:PutObjectTagging",
    ]
    resources = [
      "arn:aws:s3:::${var.governed_bucket_name}/${trimprefix(var.staging_write_prefix, "/")}*"
    ]
  }
}

resource "aws_iam_role_policy" "staging_write" {
  name   = "${local.name}-staging-write"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.staging_write.json
}

data "aws_iam_policy_document" "ground_truth_write" {
  statement {
    sid    = "WriteApprovedGroundTruthReleases"
    effect = "Allow"
    actions = [
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts",
      "s3:PutObject",
      "s3:PutObjectTagging",
    ]
    resources = [
      "arn:aws:s3:::${var.governed_bucket_name}/${trimprefix(var.ground_truth_write_prefix, "/")}*"
    ]
  }
}

resource "aws_iam_role_policy" "ground_truth_write" {
  name   = "${local.name}-ground-truth-write"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.ground_truth_write.json
}
