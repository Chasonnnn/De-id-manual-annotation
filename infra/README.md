# AWS pilot infrastructure

This directory defines the isolated `deid-annotation-pilot` deployment.

## Provider boundary

Terraform is locked to `1.15.8`, matching the repository Mise toolchain. The HashiCorp AWS provider is locked to `6.61.0`. That provider has a first-class `aws_ecs_express_gateway_service` resource, so this stack does not use AWSCC or a CloudFormation shim. AWSCC `1.98.0` was reviewed as the fallback because its resources are generated from CloudFormation registry schemas, but adding a second provider would widen the dependency and state surface without adding capability.

Official references:

- [AWS provider Express service](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/ecs_express_gateway_service)
- [AWS CloudFormation Express service](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-ecs-expressgatewayservice.html)
- [ECS Express managed resources and networking](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-work.html)
- [AWSCC provider model](https://registry.terraform.io/providers/hashicorp/awscc/latest/docs)
- [Terraform lifecycle protection](https://developer.hashicorp.com/terraform/language/meta-arguments/lifecycle)
- [RDS backup retention and point-in-time recovery](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_WorkingWithAutomatedBackups.BackupRetention.html)

## Isolation and security

- Every supported resource receives `Project=deid-annotation-pilot`, environment, owner, expiration, repository, and Terraform tags.
- A dedicated VPC contains two public application subnets and two private database subnets. Public application subnets are required for the managed public Express URL; AWS documents that private subnets produce an internal load balancer.
- Express creates and maintains its load-balancer ingress. The supplied task security group has no user-managed inbound rule; its outbound rules allow only PostgreSQL to the private database, DNS to the VPC resolver, and HTTPS for governed S3 plus ECS task-start dependencies. The database accepts `5432` only from that task group.
- RDS PostgreSQL 18.6 is private, encrypted, SSL-required, protected from deletion, backed up for fourteen days, and configured for point-in-time recovery. Multi-AZ is optional because it roughly increases pilot database cost.
- ECR is encrypted, immutable, scan-on-push, and protected from destruction.
- Secrets Manager containers are created without secret versions. No password or connection string is stored in Terraform configuration or state.
- The task role can list/read only configured governed raw and processed prefixes, including exact object versions. Staging and GT writes are separate IAM policies. None grants `s3:DeleteObject` or access to the other project namespace.
- Express, RDS, KMS, ECR, Secrets Manager, and the application log group use `prevent_destroy`. Removing a protected block from configuration also removes Terraform's lifecycle guard, so teardown requires a reviewed code change and verified exports.

## Two-phase apply design

The safe default is `enable_express_service=false`.

Phase one creates networking, RDS, ECR, secret containers, IAM, monitoring, and the budget. An authorized operator then:

1. Builds one tested `linux/amd64,linux/arm64` image index, pushes it to the emitted ECR repository, and records its `@sha256` URI.
2. Retrieves the RDS-managed username and password through an approved secure channel.
3. Combines those credentials with the emitted `database_endpoint` output and writes a complete `postgresql+psycopg://...?...sslmode=require` value into the database URL secret. The RDS-managed secret does not supply host or port fields.
4. If bootstrapping, writes a unique password of at least twelve characters into the initial-admin secret.
5. Sets the immutable image URI, optional bootstrap identity, and `enable_express_service=true` for phase two.
6. Logs in as the bootstrap admin, then removes all three bootstrap settings from the service while retaining the human admin account.

Terraform intentionally does not create secret versions because secret values would be retained in Terraform state.

## Static validation

No AWS credentials are needed for formatting or validation:

```bash
mise exec -- terraform -chdir=infra fmt -check -recursive
mise exec -- terraform -chdir=infra init -backend=false -lockfile=readonly
mise exec -- terraform -chdir=infra validate
```

`init` downloads only the exactly locked provider. It must not be followed by `plan` during static review.

## Apply gates

No plan or apply is authorized until all gates are explicitly approved:

1. Verify the named AWS identity, account ID, and `us-east-1` through the repository auth guard.
2. Confirm no `deid-annotation-pilot` resource or Express service name already exists.
3. Create an encrypted, versioned remote-state bucket and a distinct state key outside this stack. Supply `backend.hcl`; never use local state for an apply.
4. Review a saved plan generated with `-out`; require zero deletes and zero replacements for every pilot update.
5. Confirm the governed bucket and exact raw, processed, staging, and GT prefixes.
6. Confirm the alert email subscriptions and the tag-filtered budget.
7. Complete backend, frontend, container, PostgreSQL concurrency, and two-account authorization tests.
8. Populate secret values before enabling Express; never pass them through `-var`, shell arguments, committed files, or plan output.
9. Confirm the Express plan preserves the exact service name and infrastructure role. Both are replacement-sensitive, and replacement is blocked during the pilot.
10. Apply only the reviewed saved plan, then verify TLS, `/api/health`, secure cookies, CSRF, role isolation, import, autosave, assignment, export, RDS backups, alarms, and the stable URL.

Example backend configuration, deliberately excluded from Git:

```hcl
bucket       = "replace-with-dedicated-state-bucket"
key          = "deid-annotation-pilot/terraform.tfstate"
region       = "us-east-1"
encrypt      = true
use_lockfile = true
```

## Pilot retention and teardown

Do not set `enable_express_service=false`, rename the service, change the infrastructure role, or remove the Express resource during the two-week pilot. Terraform will reject those changes because `prevent_destroy` is enabled. The AWS-generated URL remains tied to retention of that service; AWS does not publish a minimum URL-retention SLA after deletion.

Teardown is a separate, explicitly approved operation after annotation exports and GT promotion have been verified. It requires a final snapshot decision, retained-backup cost review, disabling AWS-side RDS deletion protection, and a temporary reviewed removal of Terraform lifecycle guards. No generic `terraform destroy` runbook is provided.
