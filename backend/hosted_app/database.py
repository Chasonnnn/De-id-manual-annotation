from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Column, DateTime, UniqueConstraint
from sqlalchemy.types import TypeDecorator
from sqlmodel import Field, SQLModel

from .domain import AssignmentState, Role, UserState


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid4())


class UTCDateTime(TypeDecorator[datetime]):
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self, value: datetime | None, dialect: Any
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("timezone-aware datetime required")
        return value.astimezone(UTC)

    def process_result_value(
        self, value: datetime | None, dialect: Any
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: str = Field(default_factory=new_id, primary_key=True)
    email: str = Field(index=True, unique=True)
    display_name: str
    role: Role
    password_hash: str | None = None
    state: UserState = UserState.ACTIVE
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class BootstrapGate(SQLModel, table=True):
    __tablename__ = "bootstrap_gate"

    id: int = Field(default=1, primary_key=True)


class LoginSession(SQLModel, table=True):
    __tablename__ = "login_sessions"

    token_hash: str = Field(primary_key=True)
    user_id: str = Field(foreign_key="users.id", index=True)
    expires_at: datetime = Field(sa_column=Column(UTCDateTime(), nullable=False))
    created_at: datetime = Field(default_factory=utc_now)


class ActivationToken(SQLModel, table=True):
    __tablename__ = "activation_tokens"

    token_hash: str = Field(primary_key=True)
    user_id: str = Field(foreign_key="users.id", unique=True, index=True)
    expires_at: datetime = Field(sa_column=Column(UTCDateTime(), nullable=False))
    created_at: datetime = Field(default_factory=utc_now)


class Batch(SQLModel, table=True):
    __tablename__ = "batches"

    id: str = Field(default_factory=new_id, primary_key=True)
    name: str
    created_by: str = Field(foreign_key="users.id", index=True)
    manifest_digest: str | None = Field(default=None, index=True)
    import_mutation_id: str | None = Field(default=None, unique=True, index=True)
    created_at: datetime = Field(default_factory=utc_now)


class SessionFolder(SQLModel, table=True):
    __tablename__ = "session_folders"

    id: str = Field(default_factory=new_id, primary_key=True)
    name: str = Field(unique=True, index=True)
    created_by: str = Field(foreign_key="users.id", index=True)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Document(SQLModel, table=True):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "batch_id", "external_id", name="uq_documents_batch_external_id"
        ),
    )

    id: str = Field(default_factory=new_id, primary_key=True)
    batch_id: str = Field(foreign_key="batches.id", index=True)
    external_id: str = Field(index=True)
    filename: str
    raw_text: str
    label_set: list[str] = Field(sa_column=Column(JSON, nullable=False))
    reference_spans: list[dict[str, Any]] | None = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    raw_source: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    reference_source: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    created_at: datetime = Field(default_factory=utc_now)


class SessionFolderMembership(SQLModel, table=True):
    __tablename__ = "session_folder_memberships"

    document_id: str = Field(foreign_key="documents.id", primary_key=True)
    folder_id: str = Field(foreign_key="session_folders.id", index=True)
    updated_by: str = Field(foreign_key="users.id")
    updated_at: datetime = Field(default_factory=utc_now)


class Assignment(SQLModel, table=True):
    __tablename__ = "assignments"

    id: str = Field(default_factory=new_id, primary_key=True)
    document_id: str = Field(foreign_key="documents.id", unique=True, index=True)
    assignee_id: str = Field(foreign_key="users.id", index=True)
    assigned_by: str = Field(foreign_key="users.id")
    state: AssignmentState = AssignmentState.ASSIGNED
    assigned_at: datetime = Field(default_factory=utc_now)
    last_activity_at: datetime | None = None
    completed_at: datetime | None = None


class Annotation(SQLModel, table=True):
    __tablename__ = "annotations"

    document_id: str = Field(foreign_key="documents.id", primary_key=True)
    spans: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    revision: int = 0
    updated_by: str = Field(foreign_key="users.id")
    updated_at: datetime = Field(default_factory=utc_now)


class AnnotationRevision(SQLModel, table=True):
    __tablename__ = "annotation_revisions"
    __table_args__ = (
        UniqueConstraint("document_id", "revision"),
        UniqueConstraint("mutation_id"),
    )

    id: str = Field(default_factory=new_id, primary_key=True)
    document_id: str = Field(foreign_key="documents.id", index=True)
    revision: int
    mutation_id: str = Field(index=True)
    spans: list[dict[str, Any]] = Field(sa_column=Column(JSON, nullable=False))
    author_id: str = Field(foreign_key="users.id")
    created_at: datetime = Field(default_factory=utc_now)


class BulkAssignmentMutation(SQLModel, table=True):
    __tablename__ = "bulk_assignment_mutations"

    mutation_id: str = Field(primary_key=True)
    plan_digest: str = Field(index=True)
    assignment_ids: list[str] = Field(sa_column=Column(JSON, nullable=False))
    actor_id: str = Field(foreign_key="users.id", index=True)
    created_at: datetime = Field(default_factory=utc_now)


class AuditEvent(SQLModel, table=True):
    __tablename__ = "audit_events"

    id: str = Field(default_factory=new_id, primary_key=True)
    actor_id: str = Field(foreign_key="users.id", index=True)
    action: str = Field(index=True)
    target_type: str = Field(index=True)
    target_id: str = Field(index=True)
    before_metadata: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    after_metadata: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    mutation_id: str | None = Field(default=None, index=True)
    occurred_at: datetime = Field(default_factory=utc_now, index=True)
    result: str = Field(index=True)
    reason: str | None = None


def create_schema(engine: Any) -> None:
    SQLModel.metadata.create_all(engine)
