# Hosted Annotation MVP Workplan

## Decisions

- The product is a hosted annotation application with a web client and a command-line client.
- Both clients use the same authenticated human identity, permissions, domain operations, and audit records.
- Accounts are restricted to explicitly created `@cornell.edu` users. Public registration and Cornell SSO are excluded from the pilot.
- Admins can create, activate, deactivate, and reactivate annotators; import annotation batches; assign, rebalance, reassign, and reopen sessions; inspect progress; and build releases.
- Annotators can see only assigned sessions, edit manual annotations, and mark work complete.
- Raw and processed source objects remain immutable in S3. Live annotation state is revisioned in PostgreSQL.
- Completed annotations are staged under `workflows/manual-annotation/` before admin-approved publication to `governed/GT/`.
- Amazon ECS Express Mode hosts the pilot through one explicitly named service. The AWS-managed URL remains the shared URL for at least fourteen days by retaining and updating that service in place.
- Experiment execution, model inference, evaluation dashboards, legacy local-file migration, and the separate `projects/annotation-project/` namespace are excluded.

## Current state

| Area | State |
|---|---|
| Hosted FastAPI application and PostgreSQL repository | Implemented and revalidated locally; PostgreSQL concurrency gate remains environment-dependent |
| Cornell-domain email/password login | Implemented locally |
| Annotator visibility, revisioned autosave, conflicts, completion, and reopen | Implemented locally |
| Basic admin users, assignment, progress, import, and export endpoints | Implemented locally |
| Three-panel web workspace and local QA server | Implemented locally |
| Backend and frontend dependency modernization | Implemented and unit/build validated locally |
| Repository-local Mise toolchain and cross-platform lock | Implemented |
| Ponytail whole-repository audit | Approved reductions applied; security and validation controls retained |
| Legacy experiment and local-file removal | Completed in the working tree; regression and hosted-container gates pass locally |
| CLI | Account, session, status, exact-manifest batch import, single/bulk assignment, and filtered audit commands implemented locally; high-level import discovery and release commands remain |
| User activation, deactivation, and session revocation | Backend, activation web flow, and CLI implemented and unit-validated locally |
| S3 manifest import and provenance | Governed read-only runtime, plan/apply API and CLI, exact Saga ZIP and Saga MultiModel adapters, optional processed references, checksums, provenance, and idempotent atomic import implemented locally; high-level prefix-to-manifest generation remains |
| Audit events and bulk assignment | Deterministic preview, atomic idempotent apply, lifecycle audit coverage, filtered API, and CLI implemented locally |
| Web account lifecycle | Reset, explicit deactivate with unassign/reassign, reactivate, account-state visibility, and one-time activation links implemented locally |
| Release staging and ground-truth publication | Not implemented |
| AWS infrastructure and deployment | Isolated Terraform scaffold formatted and validated locally; no AWS plan or apply performed |

## Architecture

```text
Web client ──────────────┐
                        ├─ Hosted application ─ PostgreSQL
Command-line client ────┘          │
                                   ├─ read: governed/raw/
                                   ├─ read: governed/processed/
                                   ├─ write: workflows/manual-annotation/
                                   └─ approved write: governed/GT/
```

The hosted application owns authentication, authorization, imports, assignment state, annotation revisions, progress, audit records, exports, and release validation. Web and CLI code do not reproduce those rules.

Two source adapters satisfy one batch-import interface:

- Saga archive adapter for versioned batch ZIP objects.
- Saga MultiModel adapter for versioned per-session object trees.

One credential-store interface has a macOS Keychain adapter for the CLI and an in-memory adapter for tests. There is no plaintext-file credential fallback.

## Workstream 0: Baseline, modernization, and simplification

### Work

1. Preserve unrelated working-tree changes and inventory the intentional hosted rewrite.
2. Add repository-local `mise.toml` and `mise.lock` from the approved global versions.
3. Research official release notes and compatibility guidance, then upgrade every direct backend and frontend dependency to current stable versions.
4. Adopt useful current framework features and configurations; do not perform version-only churn or add speculative architecture.
5. Align Python, Node.js, uv, container images, manifests, and verification commands.
6. Retain npm because this repository already has `package-lock.json`; document the compatibility exception rather than silently changing package managers.
7. Run a Ponytail whole-repository audit and separately triage its read-only findings. Apply only cuts that preserve the hosted product contract.
8. Add single entry points for local development and full verification.
9. Finish the intentional deletion of experiment, evaluation, local-model, and local-session code.

### Gate

- No runtime import, route, UI text, build input, or dependency references removed legacy features.
- Backend tests, frontend tests, production build, React Doctor, lock checks, and container build pass from the reproducible toolchain.
- Modernization notes identify version changes, adopted features, rejected features, and compatibility evidence.
- Ponytail findings have explicit apply or defer dispositions; the audit itself changes no code.

## Workstream 1: Human account lifecycle

### Work

1. Replace the active boolean contract with explicit account states: `pending_activation`, `active`, and `deactivated`.
2. Make admin user creation accept Cornell email and display name without accepting a permanent password.
3. Generate a single-use activation token, store only its hash, expire it after twenty-four hours, and return the activation URL once.
4. Let the annotator set the initial password through the activation page.
5. Add admin deactivation, reactivation, and password-reset operations.
6. Revoke all login sessions on deactivation, password change, or password reset.
7. Reject deactivation when incomplete assignments exist unless the request explicitly unassigns or reassigns them.
8. Keep account deletion out of scope.

### Tests first

- Non-Cornell and duplicate emails fail explicitly.
- Activation tokens are single-use, hashed, expiring, and bound to one account.
- Deactivated users cannot log in and existing sessions stop working immediately.
- Incomplete work cannot become orphaned through a default deactivation.
- Annotators cannot call account-management operations.

### Gate

- An admin can create, activate, deactivate, reactivate, and reset one test annotator without direct database access.

## Workstream 2: CLI-first administration

### Interface

```text
annotationctl login --url URL --email EMAIL
annotationctl whoami
annotationctl logout

annotationctl users list
annotationctl users create
annotationctl users deactivate
annotationctl users reactivate
annotationctl users reset-password

annotationctl batches plan-import
annotationctl batches apply-import
annotationctl sessions list
annotationctl status

annotationctl assignments preview
annotationctl assignments apply
annotationctl assignments reassign
annotationctl assignments reopen

annotationctl releases build
annotationctl releases verify
annotationctl releases publish
annotationctl audit list
```

Every read command supports human-readable output and `--json`. Every mutation accepts a caller-supplied mutation ID or creates one and prints it.

### Authentication

1. Prompt for the existing account password through the terminal without echo.
2. Use the normal login endpoint and normal human admin role.
3. Store only the returned short-lived session and CSRF material in macOS Keychain, keyed by application URL and email.
4. Never place a password or session token in arguments, environment variables, files, logs, or command output.
5. Return explicit login-expired errors; do not silently reauthenticate or downgrade behavior.

### Tests first

- Password prompts never enter argument parsing or diagnostic output.
- Keychain writes occur only after successful login.
- CLI writes include the correct CSRF value.
- `--json` output is stable and contains no credentials or transcript content unless the command explicitly requests a session.
- HTTP errors preserve status and actionable server details.
- Admin and annotator CLI permissions match the web application.

### Gate

- After one interactive admin login, all supported administration can be completed through the CLI without browser-only operations.

## Workstream 3: S3 batch catalog and import

### Source locations

```text
s3://nto-contextshift-deid/governed/raw/Saga/<drop-id>/
s3://nto-contextshift-deid/governed/raw/Saga-MultiModel/<drop-id>/
s3://nto-contextshift-deid/governed/processed/<dataset>/<cohort>/<run-id>/
```

### Batch manifest

Each imported session records:

- dataset, cohort, and canonical session ID;
- raw bucket, key, VersionId, object checksum, and Saga archive member when applicable;
- processed-reference bucket, key, VersionId, and checksum;
- imported raw transcript snapshot and reference snapshot;
- import actor, timestamp, and immutable batch ID.

### Work

1. Add a read-only S3 catalog operation that lists eligible versioned source drops and processed runs without returning raw transcript contents.
2. Implement Saga archive and Saga MultiModel source adapters.
3. Implement exact-manifest plan/apply API and CLI commands. Planning performs metadata discovery and validation without body reads or database writes.
4. Add a high-level prefix discovery command that generates the exact canonical manifest for large drops.
5. Require canonical IDs, stable reference offsets, allowed labels, unique batch membership, VersionIds, and checksums.
6. Allow raw-only imports when processed output is not yet available; when a reference is declared, reject the batch on any missing, duplicate, ambiguous, or text-mismatched reference.
7. Snapshot validated transcript and optional reference data into encrypted PostgreSQL in one transaction.
8. Preserve the S3 provenance alongside the snapshot.
9. Make apply idempotent by manifest digest and mutation ID.

### Pilot

1. Import a small Saga sample from batches 02–04 against the existing 300-session processed run.
2. Import the four current Saga MultiModel sessions.
3. Compare displayed raw transcript and reference spans against their exact source versions.
4. Expand to the approved full batch only after the pilot gate passes.

### Gate

- Repeating an import cannot create duplicates.
- Imported text and offsets are identical across CLI, database, and web views.
- No local corpus directory or persistent runtime file is created.

## Workstream 4: Assignment, progress, and audit

### Work

1. Add filterable session queries by dataset, batch, annotator, state, and last activity.
2. Add batch and per-annotator progress summaries with totals and completion percentages.
3. Add balanced bulk assignment with server-side preview and atomic apply.
4. Keep the first strategy deterministic equal-count distribution; capacity weighting is deferred until requested.
5. Require expected current assignment state to prevent a stale bulk plan from overwriting newer work.
6. Make bulk apply idempotent and all-or-nothing.
7. Record audit events for account, import, assignment, reopen, annotation completion, export, and publication mutations.
8. Store actor, action, target identifiers, before/after state, mutation ID, timestamp, result, and optional reason. Do not store raw transcript or annotation text in audit events.

### Tests first

- Distribution is deterministic, balanced, and never assigns inactive users.
- Concurrent or stale assignment plans fail without partial application.
- Progress matches source assignment rows after assign, save, complete, reopen, and deactivate flows.
- Audit records are written in the same transaction as successful mutations and contain no sensitive text.

### Gate

- The CLI can answer who owns every session, current completion by batch and person, inactive work, and every administrative change.

## Workstream 5: Annotation workspace and admin web UI

### Work

1. Retain the session sidebar and Raw Transcript, Manual Annotation, and Reference panels.
2. Remove every local-file, experiment, evaluation, and local-model control.
3. Add only the required shadcn/ui primitives for login, tabs, tables, badges, progress, dialogs, selects, and notifications.
4. Keep web administration focused on users, assignments, progress, and read-only release status. Batch import and export execution belong to the CLI.
5. Keep the same server operations as the CLI; the UI must not implement separate assignment or import rules.
6. Preserve revisioned autosave, visible saving/saved/conflict/error states, reload recovery, and completed-work locking.
7. Add keyboard and screen-reader checks for annotation and administration flows.

### Gate

- A two-account visible QA pass confirms admin-all and annotator-assigned-only visibility.
- Hard reload recovers the last acknowledged annotation revision.
- No removed feature or local-file language appears in the built application.

## Workstream 6: Export and ground-truth release

### Locations

```text
s3://nto-contextshift-deid/workflows/manual-annotation/
  batches/<batch-id>/manifest.json
  exports/<release-id>/sessions/<session-id>.json
  exports/<release-id>/RELEASE.json
  exports/<release-id>/CHECKSUMS.txt

s3://nto-contextshift-deid/governed/GT/<dataset>/<release-id>/
```

### Work

1. Build immutable per-session exports only from completed annotations.
2. Include source VersionIds and checksums, annotation revision, annotator identity, completion timestamp, schema version, and export digest.
3. Write and verify the staging export before publication is available.
4. Require an explicit admin publication command naming the verified release digest.
5. Convert the staging schema to the canonical ground-truth schema and verify exact object inventory and checksums after upload.
6. Never overwrite a prior release and never grant object deletion to the application role.
7. Mark publication complete only after post-write verification succeeds.

### Gate

- A staged pilot release can be rebuilt byte-identically.
- An incorrect digest, incomplete session, schema error, or existing release path fails closed.

## Workstream 7: AWS infrastructure

### Target

- AWS context: `cu-nto-research`.
- Region: `us-east-1`, colocated with the governed S3 bucket.
- Resource prefix: `deid-annotation-pilot`.
- Hosting: one Amazon ECS Express Mode service with an explicit immutable service name.
- Database: private encrypted PostgreSQL RDS with backups and point-in-time recovery.
- Container registry: dedicated ECR repository with immutable image identifiers.
- Secrets: dedicated Secrets Manager entries for database and one-time bootstrap configuration.
- Observability: dedicated CloudWatch logs, health alarms, and budget alarms.
- Infrastructure: repository-managed Terraform using the approved Mise version.

### Isolation

1. Plan and apply only resources carrying the dedicated prefix and project tags.
2. Use dedicated execution, infrastructure, and application task roles.
3. Permit the task role to read only approved raw and processed prefixes and write only manual-annotation staging and approved ground-truth release prefixes.
4. Deny S3 deletion and deny unrelated `projects/annotation-project/`, model, artifact, and evaluation prefixes.
5. Restrict PostgreSQL to the application security group; it is not publicly reachable.
6. Keep the existing two projects in the AWS account untouched.

### Express URL contract

1. Create the service once with an explicit name.
2. Record the returned `https://<generated-id>.ecs.<region>.on.aws/` URL.
3. Restrict application host validation to `*.ecs.<region>.on.aws` and point monitoring at the exact returned hostname.
4. Update images and configuration on the existing service.
5. Prohibit service deletion or replacement until the pilot has ended and verified exports exist.
6. Monitor the URL before and after every deployment.

### Gate

- Terraform plan contains only named pilot resources.
- Guarded AWS identity and Region are verified immediately before apply.
- No resource collision or modification appears for the two existing projects.
- The same Express URL passes health and login checks after an in-place deployment.

## Workstream 8: Deployment and bootstrap

### Work

1. Build and test the production container locally.
2. Scan dependency locks and the final image for known high-severity vulnerabilities.
3. Push an immutable image digest to the dedicated ECR repository.
4. Apply infrastructure through the guarded AWS context after reviewing the exact plan.
5. Bootstrap the first Cornell admin from Secrets Manager.
6. Verify the first login, then remove bootstrap credentials from service configuration and deploy a new revision.
7. Log in through `annotationctl` as the same admin and verify `whoami`.
8. Run database schema migration and backup checks.
9. Execute deployed smoke tests without transcript content in logs.

### Gate

- Health, secure cookies, CSRF, host validation, TLS, CLI authentication, and database connectivity pass against the AWS URL.
- The bootstrap secret is no longer attached to the running service.

## Workstream 9: Live acceptance and pilot operations

### Acceptance

1. Create two test annotators through the CLI and activate them.
2. Import the Saga and Saga MultiModel pilot manifests from S3.
3. Preview and apply a balanced assignment.
4. Confirm each annotator sees only assigned sessions.
5. Annotate, autosave, reload, complete, reopen, and reassign test sessions.
6. Verify CLI and web progress agree.
7. Build and verify a staging export.
8. Publish a deliberately small approved ground-truth release and verify S3 inventory, versions, and checksums.
9. Confirm audit attribution for every mutation.
10. Revoke one user and confirm immediate access loss without orphaned work.

### Two-week operation

1. Keep the Express service and RDS instance intact for at least fourteen days from deployment.
2. Use only in-place Express service updates.
3. Check health, failed logins, save failures, 5xx responses, database storage, backups, and annotator inactivity daily during active annotation.
4. Export and verify work before any infrastructure teardown.
5. Review retained RDS snapshots, ECR images, load balancers, public IPv4 addresses, logs, and secrets before claiming the pilot has no remaining cost.

## Dependency order and parallel work

```text
Baseline
  ├─ Account lifecycle ─ CLI ───────────────┐
  ├─ S3 import ──────────────┐              │
  ├─ Assignment/audit ───────┼─ Web admin ──┼─ Acceptance
  └─ AWS infrastructure ─────┘              │
                 Export/release ────────────┘
```

Account contracts, import manifests, audit events, and release schemas are defined before parallel implementation begins. Subagents may implement independent modules and tests after those interfaces are fixed. One integration owner resolves shared database and route changes.

## Final release gates

- Backend test suite passes against PostgreSQL.
- Frontend tests and production build pass.
- CLI tests pass with fake credential storage and live local HTTP acceptance.
- React Doctor reports no unresolved issues.
- Container build and non-root runtime pass.
- S3 import, assignment concurrency, annotation revision, export reproducibility, and publication fail-closed tests pass.
- Terraform format, validate, and reviewed plan pass.
- Deployed two-account web and CLI acceptance passes.
- Backup, restore, log redaction, least-privilege IAM, and cost-alarm checks pass.
- The AWS URL is recorded and unchanged after one in-place deployment.
