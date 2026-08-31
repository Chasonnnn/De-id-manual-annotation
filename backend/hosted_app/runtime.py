from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import boto3
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from sqlmodel import Session, create_engine
from starlette.types import ASGIApp, Receive, Scope, Send

from .api import create_hosted_app
from .auth import AuthManager
from .database import create_schema
from .domain import UserState
from .repository import HostedRepository
from .s3_import import Boto3S3ReadAdapter, S3CatalogConfig, S3CatalogError


class RuntimeConfigurationError(RuntimeError):
    pass


class AlbHealthCheckHostMiddleware:
    """Normalize only ECS/ALB health probes before TrustedHostMiddleware."""

    def __init__(self, app: ASGIApp, replacement_host: str) -> None:
        self.app = app
        self.replacement_host = replacement_host.encode("ascii")

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] == "http"
            and scope["method"] == "GET"
            and scope["path"] == "/api/health"
        ):
            headers = list(scope.get("headers", []))
            user_agent = next(
                (value for name, value in headers if name.lower() == b"user-agent"),
                b"",
            )
            if user_agent == b"ELB-HealthChecker/2.0":
                scope = dict(scope)
                scope["headers"] = [
                    (name, self.replacement_host if name.lower() == b"host" else value)
                    for name, value in headers
                ]
        await self.app(scope, receive, send)


def add_host_validation(app: FastAPI, allowed_hosts: tuple[str, ...]) -> None:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(allowed_hosts),
    )
    app.add_middleware(
        AlbHealthCheckHostMiddleware,
        replacement_host=allowed_hosts[0],
    )


def add_frontend(app: FastAPI, static_dir: str | Path) -> None:
    frontend_dir = Path(static_dir).resolve()
    if not frontend_dir.is_dir() or not (frontend_dir / "index.html").is_file():
        raise RuntimeConfigurationError(
            "HOSTED_STATIC_DIR must contain the built frontend index.html"
        )
    app.frontend("/", directory=frontend_dir, check_dir=True)


def _parse_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeConfigurationError(f"{name} must be true or false")


def _is_allowed_host_pattern(host: str) -> bool:
    if "*" not in host:
        return True
    suffix = host.removeprefix("*.")
    return host.startswith("*.") and host.count("*") == 1 and suffix.count(".") >= 3


@dataclass(frozen=True)
class RuntimeSettings:
    database_url: str
    cookie_secure: bool = True
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1")
    initial_admin_email: str | None = None
    initial_admin_display_name: str | None = None
    initial_admin_password: str | None = None
    static_dir: str | None = None
    s3_catalog_config: S3CatalogConfig | None = None

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> RuntimeSettings:
        database_url = environment.get("DATABASE_URL", "").strip()
        if not database_url:
            raise RuntimeConfigurationError("DATABASE_URL is required")
        if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
            raise RuntimeConfigurationError("DATABASE_URL must use PostgreSQL")

        bootstrap_values = {
            "email": environment.get("INITIAL_ADMIN_EMAIL", "").strip(),
            "display_name": environment.get("INITIAL_ADMIN_DISPLAY_NAME", "").strip(),
            "password": environment.get("INITIAL_ADMIN_PASSWORD", ""),
        }
        configured_count = sum(bool(value) for value in bootstrap_values.values())
        if configured_count not in (0, 3):
            raise RuntimeConfigurationError(
                "INITIAL_ADMIN_EMAIL, INITIAL_ADMIN_DISPLAY_NAME, and "
                "INITIAL_ADMIN_PASSWORD must all be set together"
            )

        allowed_hosts = tuple(
            host.strip()
            for host in environment.get(
                "HOSTED_ALLOWED_HOSTS",
                "localhost,127.0.0.1",
            ).split(",")
            if host.strip()
        )
        if not allowed_hosts or any(
            not _is_allowed_host_pattern(host) for host in allowed_hosts
        ):
            raise RuntimeConfigurationError(
                "HOSTED_ALLOWED_HOSTS must contain explicit hostnames or scoped "
                "leading wildcards"
            )

        s3_values = {
            "bucket": environment.get("HOSTED_S3_BUCKET", "").strip(),
            "raw_prefixes": environment.get("HOSTED_S3_RAW_PREFIXES", "").strip(),
            "reference_prefixes": environment.get(
                "HOSTED_S3_REFERENCE_PREFIXES", ""
            ).strip(),
        }
        s3_configured_count = sum(bool(value) for value in s3_values.values())
        if s3_configured_count not in (0, 3):
            raise RuntimeConfigurationError(
                "HOSTED_S3_BUCKET, HOSTED_S3_RAW_PREFIXES, and "
                "HOSTED_S3_REFERENCE_PREFIXES must all be set together"
            )
        s3_catalog_config = None
        if s3_configured_count == 3:
            try:
                s3_catalog_config = S3CatalogConfig(
                    bucket=s3_values["bucket"],
                    raw_prefixes=tuple(
                        prefix.strip()
                        for prefix in s3_values["raw_prefixes"].split(",")
                        if prefix.strip()
                    ),
                    reference_prefixes=tuple(
                        prefix.strip()
                        for prefix in s3_values["reference_prefixes"].split(",")
                        if prefix.strip()
                    ),
                )
            except S3CatalogError as error:
                raise RuntimeConfigurationError(str(error)) from error

        return cls(
            database_url=database_url,
            cookie_secure=_parse_bool(
                environment.get("HOSTED_COOKIE_SECURE", "true"),
                name="HOSTED_COOKIE_SECURE",
            ),
            allowed_hosts=allowed_hosts,
            initial_admin_email=bootstrap_values["email"] or None,
            initial_admin_display_name=bootstrap_values["display_name"] or None,
            initial_admin_password=bootstrap_values["password"] or None,
            static_dir=environment.get("HOSTED_STATIC_DIR") or None,
            s3_catalog_config=s3_catalog_config,
        )


def build_runtime_app(settings: RuntimeSettings) -> FastAPI:
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    create_schema(engine)
    repository = HostedRepository(lambda: Session(engine))
    auth = AuthManager(repository)

    if settings.initial_admin_email is not None:
        existing = repository.get_user_by_email(settings.initial_admin_email)
        if existing is None:
            auth.bootstrap_admin(
                settings.initial_admin_email,
                settings.initial_admin_password or "",
                display_name=settings.initial_admin_display_name or "",
            )
        elif existing.role != "admin" or existing.state != UserState.ACTIVE:
            raise RuntimeConfigurationError(
                "INITIAL_ADMIN_EMAIL belongs to a non-admin or inactive account"
            )

    s3_reader = None
    if settings.s3_catalog_config is not None:
        s3_reader = Boto3S3ReadAdapter(boto3.client("s3"))
    app = create_hosted_app(
        repository=repository,
        auth=auth,
        cookie_secure=settings.cookie_secure,
        s3_reader=s3_reader,
        s3_catalog_config=settings.s3_catalog_config,
    )
    add_host_validation(app, settings.allowed_hosts)
    if settings.static_dir is not None:
        add_frontend(app, settings.static_dir)
    return app
