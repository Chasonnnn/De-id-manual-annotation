from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    ADMIN = "admin"
    ANNOTATOR = "annotator"


class UserState(StrEnum):
    PENDING_ACTIVATION = "pending_activation"
    ACTIVE = "active"
    DEACTIVATED = "deactivated"


class AssignmentState(StrEnum):
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class RepositoryError(Exception):
    """Base class for repository failures that HTTP handlers map explicitly."""


class NotFound(RepositoryError):
    pass


class Forbidden(RepositoryError):
    pass


class VisibilityDenied(NotFound):
    """Resource is intentionally indistinguishable from a missing resource."""


class RevisionConflict(RepositoryError):
    def __init__(self, current_revision: int) -> None:
        super().__init__(
            f"expected revision is stale; current revision is {current_revision}"
        )
        self.current_revision = current_revision


class CompletedLocked(RepositoryError):
    pass


class DuplicateExternalId(RepositoryError):
    pass


class InvalidReference(RepositoryError):
    pass


class InvalidAssignee(RepositoryError):
    pass


class InvalidAccountAction(RepositoryError):
    pass


class ImportMutationConflict(RepositoryError):
    pass


class DuplicateSelection(RepositoryError):
    pass


class BulkPlanStale(RepositoryError):
    pass


class BulkMutationConflict(RepositoryError):
    pass


@dataclass(frozen=True)
class DocumentImport:
    external_id: str
    filename: str
    raw_text: str
    label_set: list[str]
    reference_spans: list[dict[str, Any]] | None
    raw_source: dict[str, Any] | None = None
    reference_source: dict[str, Any] | None = None


@dataclass(frozen=True)
class ImportedBatch:
    batch_id: str
    imported_count: int


@dataclass(frozen=True)
class DocumentProvenance:
    document_id: str
    raw_source: dict[str, Any] | None
    reference_source: dict[str, Any] | None


@dataclass(frozen=True)
class ManualAnnotationExport:
    document_id: str
    external_id: str
    filename: str
    manual_annotations: list[dict[str, Any]]
    annotation_revision: int
    assignee_id: str | None
    assignment_state: AssignmentState | None
    updated_at: datetime | None


@dataclass(frozen=True)
class SaveResult:
    revision: int
    spans: list[dict[str, Any]]
    assignment_state: AssignmentState


@dataclass(frozen=True)
class DocumentDetail:
    id: str
    batch_id: str
    external_id: str
    filename: str
    raw_text: str
    label_set: list[str]
    reference_spans: list[dict[str, Any]] | None
    assignment_id: str | None
    assignee_id: str | None
    assignment_state: AssignmentState | None
    manual_spans: list[dict[str, Any]]
    revision: int


@dataclass(frozen=True)
class Progress:
    total: int
    unassigned: int
    assigned: int
    in_progress: int
    completed: int
    by_annotator: list[dict[str, Any]]


@dataclass(frozen=True)
class LoginSessionRecord:
    token_hash: str
    user_id: str
    expires_at: datetime


@dataclass(frozen=True)
class ActivationTokenRecord:
    token_hash: str
    expires_at: datetime


@dataclass(frozen=True)
class BulkAssignmentItem:
    document_id: str
    assignee_id: str


@dataclass(frozen=True)
class DocumentAssignmentPrecondition:
    document_id: str
    assignment_id: str | None
    assignee_id: str | None
    state: AssignmentState | None
    revision: int


@dataclass(frozen=True)
class AnnotatorPrecondition:
    user_id: str
    state: UserState


@dataclass(frozen=True)
class BulkAssignmentPlan:
    plan_digest: str
    assignments: list[BulkAssignmentItem]
    document_preconditions: list[DocumentAssignmentPrecondition]
    annotator_preconditions: list[AnnotatorPrecondition]


@dataclass(frozen=True)
class BulkAssignmentResult:
    plan_digest: str
    mutation_id: str
    assignment_ids: list[str]


AuditMetadataValue = str | int | list[str] | None


@dataclass(frozen=True)
class AuditRecord:
    id: str
    actor_id: str
    action: str
    target_type: str
    target_id: str
    before_metadata: dict[str, AuditMetadataValue]
    after_metadata: dict[str, AuditMetadataValue]
    mutation_id: str | None
    occurred_at: datetime
    result: str
    reason: str | None
