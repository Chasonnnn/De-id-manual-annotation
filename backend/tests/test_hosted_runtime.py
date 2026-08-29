from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hosted_app.runtime import (
    RuntimeConfigurationError,
    RuntimeSettings,
    add_frontend,
    add_host_validation,
)


def test_runtime_requires_an_explicit_postgresql_database() -> None:
    with pytest.raises(RuntimeConfigurationError, match="DATABASE_URL is required"):
        RuntimeSettings.from_environment({})

    with pytest.raises(RuntimeConfigurationError, match="PostgreSQL"):
        RuntimeSettings.from_environment({"DATABASE_URL": "sqlite:///local.db"})


def test_runtime_parses_explicit_cookie_and_bootstrap_configuration() -> None:
    settings = RuntimeSettings.from_environment(
        {
            "DATABASE_URL": "postgresql+psycopg://app:secret@db/annotations",
            "HOSTED_COOKIE_SECURE": "false",
            "HOSTED_ALLOWED_HOSTS": "annotations.example.edu,localhost",
            "INITIAL_ADMIN_EMAIL": "admin@example.edu",
            "INITIAL_ADMIN_DISPLAY_NAME": "Admin",
            "INITIAL_ADMIN_PASSWORD": "correct horse battery staple",
        }
    )

    assert settings.database_url.startswith("postgresql+psycopg://")
    assert settings.cookie_secure is False
    assert settings.allowed_hosts == ("annotations.example.edu", "localhost")
    assert settings.initial_admin_email == "admin@example.edu"
    assert settings.initial_admin_display_name == "Admin"
    assert settings.initial_admin_password == "correct horse battery staple"
    assert settings.allowed_email_domains == ("cornell.edu",)


def test_runtime_parses_one_governed_s3_bucket_and_explicit_source_prefixes() -> None:
    settings = RuntimeSettings.from_environment(
        {
            "DATABASE_URL": "postgresql+psycopg://app:secret@db/annotations",
            "HOSTED_S3_BUCKET": "nto-contextshift-deid",
            "HOSTED_S3_RAW_PREFIXES": (
                "governed/raw/Saga/,governed/raw/Saga-MultiModel/"
            ),
            "HOSTED_S3_REFERENCE_PREFIXES": (
                "governed/processed/Saga/,governed/processed/Saga-MultiModel/"
            ),
        }
    )

    assert settings.s3_catalog_config is not None
    assert settings.s3_catalog_config.bucket == "nto-contextshift-deid"
    assert settings.s3_catalog_config.raw_prefixes == (
        "governed/raw/Saga/",
        "governed/raw/Saga-MultiModel/",
    )
    assert settings.s3_catalog_config.reference_prefixes == (
        "governed/processed/Saga/",
        "governed/processed/Saga-MultiModel/",
    )


def test_runtime_rejects_partial_or_unsafe_s3_configuration() -> None:
    base = {"DATABASE_URL": "postgresql+psycopg://app:secret@db/annotations"}
    with pytest.raises(RuntimeConfigurationError, match="must all be set together"):
        RuntimeSettings.from_environment(
            {**base, "HOSTED_S3_BUCKET": "nto-contextshift-deid"}
        )
    with pytest.raises(RuntimeConfigurationError, match="must not include"):
        RuntimeSettings.from_environment(
            {
                **base,
                "HOSTED_S3_BUCKET": "nto-contextshift-deid",
                "HOSTED_S3_RAW_PREFIXES": "governed/raw/Saga/",
                "HOSTED_S3_REFERENCE_PREFIXES": "governed/GT/",
            }
        )


def test_runtime_rejects_partial_bootstrap_configuration() -> None:
    with pytest.raises(RuntimeConfigurationError, match="all be set together"):
        RuntimeSettings.from_environment(
            {
                "DATABASE_URL": "postgresql+psycopg://app:secret@db/annotations",
                "INITIAL_ADMIN_EMAIL": "admin@example.edu",
            }
        )


def test_runtime_rejects_an_unrestricted_or_empty_host_allowlist() -> None:
    for allowed_hosts in ("*", ", ,"):
        with pytest.raises(RuntimeConfigurationError, match="HOSTED_ALLOWED_HOSTS"):
            RuntimeSettings.from_environment(
                {
                    "DATABASE_URL": "postgresql+psycopg://app:secret@db/annotations",
                    "HOSTED_ALLOWED_HOSTS": allowed_hosts,
                }
            )


def test_alb_health_check_bypasses_host_validation_only_for_health_probe() -> None:
    app = FastAPI()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/private")
    def private() -> dict[str, str]:
        return {"status": "private"}

    add_host_validation(app, ("annotation-pilot.ecs.us-east-1.on.aws",))
    client = TestClient(app)
    alb_headers = {
        "Host": "10.0.1.42:8000",
        "User-Agent": "ELB-HealthChecker/2.0",
    }

    assert client.get("/api/health", headers=alb_headers).status_code == 200
    assert client.get("/api/private", headers=alb_headers).status_code == 400
    assert (
        client.get(
            "/api/health",
            headers={"Host": "10.0.1.42:8000", "User-Agent": "browser"},
        ).status_code
        == 400
    )


def test_frontend_supports_client_routes_without_shadowing_the_api(tmp_path) -> None:
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    (frontend_dir / "index.html").write_text("<main>annotation app</main>")
    assets_dir = frontend_dir / "assets"
    assets_dir.mkdir()
    (assets_dir / "app.js").write_text("console.log('annotation app')")
    app = FastAPI()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    add_frontend(app, frontend_dir)
    client = TestClient(app)

    assert client.get("/api/health").json() == {"status": "ok"}
    client_route = client.get(
        "/sessions/session-001",
        headers={"Accept": "text/html"},
    )
    assert client_route.status_code == 200
    assert client_route.text == "<main>annotation app</main>"
    assert client.get("/assets/missing.js").status_code == 404
