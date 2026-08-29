# Hosted deployment

## AWS target

Use Amazon ECS Express Mode with an explicit service name. Express returns a managed HTTPS endpoint after service creation:

```text
https://<generated-id>.ecs.<region>.on.aws/
```

App Runner is not recommended for a new deployment because AWS has closed it to new customers and directs new workloads to ECS Express Mode.

Keep the same Express service for the entire pilot and update it in place. Do not delete or recreate it until at least fourteen days after deployment and after all annotation exports have been verified. The pilot treats service retention as the URL-stability control.

Use a dedicated prefix such as `deid-annotation-pilot` for every resource in the shared AWS account. Apply project, environment, owner, and expiration tags. Use dedicated IAM roles, security groups, database, secrets, ECR repository, log groups, and budget alarms.

## Container configuration

- Image: immutable ECR tag or digest
- Container port: `8000`
- Health path: `/api/health`
- CPU and memory: set explicitly
- Public hosts: `HOSTED_ALLOWED_HOSTS=*.ecs.<region>.on.aws`; Starlette restricts this wildcard to Express subdomains in that Region

Required secret:

```text
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require
```

The first deployment may bootstrap one administrator:

```text
INITIAL_ADMIN_EMAIL=<admin>@cornell.edu
INITIAL_ADMIN_DISPLAY_NAME=Admin
INITIAL_ADMIN_PASSWORD=<at-least-12-characters>
```

All three bootstrap values must be present together. Store the password in Secrets Manager. Remove all three from the service after the first successful admin login.

`HOSTED_COOKIE_SECURE=true` and `HOSTED_STATIC_DIR=/app/frontend-dist` are set in the image.

Build one immutable image index for both task architectures because Express may
schedule either platform:

```bash
docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile.hosted -t <ecr-repository>:<immutable-tag> --push .
```

Record the registry digest and deploy the `@sha256` URI. Monitoring and operator links must use the actual endpoint returned in `ingress_paths`, not a hostname inferred from the service name.

The service receives the governed read boundary explicitly:

```text
HOSTED_S3_BUCKET=nto-contextshift-deid
HOSTED_S3_RAW_PREFIXES=governed/raw/Saga/,governed/raw/Saga-MultiModel/
HOSTED_S3_REFERENCE_PREFIXES=governed/processed/Saga/,governed/processed/Saga-MultiModel/
```

All three values are required together. The application fails startup on partial or overlapping configuration and uses a read-only S3 client.

## Network and database

- Use a private PostgreSQL RDS instance in the same VPC.
- Allow database port `5432` only from the application task security group.
- Allow task traffic only from the load balancer security group.
- Use encrypted storage, encrypted connections, backups, and point-in-time recovery.
- The application creates its initial schema at startup, so the database user needs schema DDL privileges for this first pilot.

The RDS-managed master secret supplies the username and password. Build the
application `DATABASE_URL` with those credentials plus the Terraform
`database_endpoint` output; do not expect the managed secret to contain host or
port fields. Rotate the application secret, then force a new service deployment
so replacement tasks read its current version.

## IAM and secrets

- ECS task-execution role with `AmazonECSTaskExecutionRolePolicy`
- Least-privilege access to the specific Secrets Manager values and KMS key
- ECS Express infrastructure role with `AmazonECSInfrastructureRoleforExpressGatewayServices`
- Dedicated application task role with version-aware reads limited to `governed/raw/` and `governed/processed/`
- Application writes limited to `workflows/manual-annotation/` and approved release paths under `governed/GT/`
- No S3 delete permission and no access to `projects/annotation-project/`, model, artifact, or evaluation prefixes
- Deployment identity limited to the dedicated resource prefix and required `iam:PassRole` targets

## Session import format

The admin API and CLI use an explicit manifest. Each session declares its raw adapter, exact S3 VersionId, and optional processed-reference adapter. Saga sessions use `saga_zip_transcript` plus an archive member ending in `<session-id>/transcript.json`; Saga MultiModel sessions use `saga_multimodel_transcript` against `<session>/transcript.json`. Processed references use `cascade_prediction_export`. Raw-only imports set both `reference` and `reference_format` to `null`.

```json
{
  "name": "Saga batch 02 pilot",
  "documents": [
    {
      "external_id": "<session-uuid>",
      "filename": "<session-uuid>.txt",
      "label_set": ["NAME", "URL"],
      "raw_format": "saga_zip_transcript",
      "reference_format": "cascade_prediction_export",
      "raw": {
        "bucket": "nto-contextshift-deid",
        "key": "governed/raw/Saga/2026-05-18_10batches_914/nto-transcripts-batch-02.zip",
        "version_id": "<exact-version-id>",
        "archive_member": "batch-02/<session-uuid>/transcript.json"
      },
      "reference": {
        "bucket": "nto-contextshift-deid",
        "key": "governed/processed/Saga/batch-02-04/2026-08-23_max_privacy/exports/<session-uuid>.json",
        "version_id": "<exact-version-id>"
      }
    }
  ]
}
```

Planning performs metadata-only discovery and returns a manifest digest plus exact source identities without database writes or transcript content. Apply requires that digest and source list, downloads only the declared sources, verifies decoded raw text against the processed export, validates spans and labels, and writes the batch atomically and idempotently.

## Release gates

- Verify the AWS account ID and Region before creating resources.
- Confirm the dedicated prefix does not collide with either existing project.
- Run backend, frontend, production build, container, and PostgreSQL concurrency tests.
- Verify login, authorization isolation, assignment, save/reload, conflict, completion, reopen, progress, import, and export in the deployed UI.
- Confirm secure cookies, CSRF rejection, host validation, TLS, backup policy, log retention, and budget alarms.

## Pilot cleanup

Export and verify annotations before teardown. Delete the Express service, RDS instance and unneeded snapshots, ECR images, Secrets Manager values, dedicated IAM roles, security groups, alarms, and retained CloudWatch log groups. AWS charges can continue while retained databases, snapshots, load balancers, public IPv4 addresses, logs, or secrets remain.
