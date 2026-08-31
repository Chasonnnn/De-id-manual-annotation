# De-ID Manual Annotation

Hosted annotation application for assigning transcript sessions, creating manual PII annotations, viewing supplied reference annotations, and tracking completion.

## Scope

- Admin-created email/password accounts
- Admin visibility across all sessions
- Annotator visibility limited to assigned sessions
- Raw Transcript, Manual Annotation, and Reference panels
- Revisioned PostgreSQL saves with conflict detection and retry
- Assignment, progress, import, export, completion, and reopen controls

Experiment execution, model inference, evaluation dashboards, local-file sessions, public registration, and Cornell SSO are not included.

## Local development

Requirements: PostgreSQL and Mise. Repository-local tool versions are pinned in `mise.toml` and `mise.lock` for macOS ARM64 and Linux x64.

For an isolated, disposable local database bound only to loopback:

```bash
docker compose -f compose.dev.yaml up -d --wait
```

The development container uses PostgreSQL trust authentication only on `127.0.0.1:55433`, stores its database in memory, and must never be used for hosted data.

```bash
mise install
DATABASE_URL='postgresql+psycopg://annotation@127.0.0.1:55433/annotation' \
HOSTED_COOKIE_SECURE=false \
HOSTED_ALLOWED_HOSTS=localhost,127.0.0.1 \
./run.sh --install
```

Subsequent runs can omit `--install`. The frontend is served at `http://localhost:5173` and proxies `/api` to `http://localhost:8000`.

## Command-line administration

Run the CLI from `backend/`. Login prompts for the password without echo and stores the resulting human session in macOS Keychain.

```bash
uv run annotationctl login --url http://localhost:8000 --email admin@example.com
uv run annotationctl whoami
uv run annotationctl users create --email annotator@example.com
uv run annotationctl sessions list
uv run annotationctl status
```

User creation prints one single-use activation URL. Account, assignment, and reset commands are listed by `uv run annotationctl --help`.

## Verification

```bash
cd backend && uv run pytest
cd frontend && npm test && npm run build
docker build -f Dockerfile.hosted -t deid-annotation-hosted .
```

The CLI-first implementation plan is documented in [docs/hosted-mvp-plan.md](docs/hosted-mvp-plan.md). AWS Express deployment and S3 import contracts are documented in [docs/hosted-deployment.md](docs/hosted-deployment.md).
