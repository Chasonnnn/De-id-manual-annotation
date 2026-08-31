from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from .auth import AuthenticatedPrincipal
from .domain import (
    BulkMutationConflict,
    BulkPlanStale,
    DuplicateSelection,
    Forbidden,
    InvalidAssignee,
    NotFound,
)
from .repository import HostedRepository


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


Identifier = Annotated[str, Field(min_length=1, max_length=200)]


class BulkAssignmentPreviewRequest(RequestModel):
    document_ids: list[Identifier] = Field(min_length=1)
    annotator_ids: list[Identifier] = Field(min_length=1)


class BulkAssignmentApplyRequest(BulkAssignmentPreviewRequest):
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    mutation_id: str = Field(min_length=1, max_length=200)


class BulkAssignmentItemResponse(BaseModel):
    document_id: str
    assignee_id: str


class DocumentPreconditionResponse(BaseModel):
    document_id: str
    assignment_id: str | None
    assignee_id: str | None
    state: Literal["assigned", "in_progress", "completed"] | None
    revision: int


class AnnotatorPreconditionResponse(BaseModel):
    user_id: str
    state: Literal["pending_activation", "active"]


class BulkAssignmentPreviewResponse(BaseModel):
    plan_digest: str
    assignments: list[BulkAssignmentItemResponse]
    document_preconditions: list[DocumentPreconditionResponse]
    annotator_preconditions: list[AnnotatorPreconditionResponse]


class BulkAssignmentApplyResponse(BaseModel):
    plan_digest: str
    mutation_id: str
    assignment_ids: list[str]


AuditMetadataValue = str | int | list[str] | None


class AuditEventResponse(BaseModel):
    id: str
    actor_id: str
    action: str
    target_type: str
    target_id: str
    before_metadata: dict[str, AuditMetadataValue]
    after_metadata: dict[str, AuditMetadataValue]
    mutation_id: str | None
    occurred_at: datetime
    result: Literal["success"]
    reason: str | None


def create_bulk_audit_router(
    *,
    repository: HostedRepository,
    current_principal: Callable[..., AuthenticatedPrincipal],
    require_csrf: Callable[..., None],
) -> APIRouter:
    router = APIRouter(prefix="/api/admin", tags=["admin"])

    def require_admin(principal: AuthenticatedPrincipal) -> None:
        if principal.role != "admin":
            raise HTTPException(status_code=403, detail="admin role required")

    @router.post(
        "/assignments/bulk/preview",
        response_model=BulkAssignmentPreviewResponse,
        dependencies=[Depends(require_csrf)],
    )
    def preview_bulk_assignment(
        body: BulkAssignmentPreviewRequest,
        principal: Annotated[
            AuthenticatedPrincipal,
            Depends(current_principal),
        ],
    ) -> BulkAssignmentPreviewResponse:
        require_admin(principal)
        try:
            plan = repository.preview_balanced_assignment(
                admin_id=principal.id,
                document_ids=body.document_ids,
                annotator_ids=body.annotator_ids,
            )
        except DuplicateSelection as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except InvalidAssignee as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except NotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return BulkAssignmentPreviewResponse(
            plan_digest=plan.plan_digest,
            assignments=[
                BulkAssignmentItemResponse(
                    document_id=item.document_id,
                    assignee_id=item.assignee_id,
                )
                for item in plan.assignments
            ],
            document_preconditions=[
                DocumentPreconditionResponse(
                    document_id=item.document_id,
                    assignment_id=item.assignment_id,
                    assignee_id=item.assignee_id,
                    state=item.state,
                    revision=item.revision,
                )
                for item in plan.document_preconditions
            ],
            annotator_preconditions=[
                AnnotatorPreconditionResponse(
                    user_id=item.user_id,
                    state=item.state,
                )
                for item in plan.annotator_preconditions
            ],
        )

    @router.post(
        "/assignments/bulk/apply",
        response_model=BulkAssignmentApplyResponse,
        dependencies=[Depends(require_csrf)],
    )
    def apply_bulk_assignment(
        body: BulkAssignmentApplyRequest,
        principal: Annotated[
            AuthenticatedPrincipal,
            Depends(current_principal),
        ],
    ) -> BulkAssignmentApplyResponse:
        require_admin(principal)
        try:
            result = repository.apply_balanced_assignment(
                admin_id=principal.id,
                document_ids=body.document_ids,
                annotator_ids=body.annotator_ids,
                plan_digest=body.plan_digest,
                mutation_id=body.mutation_id,
            )
        except DuplicateSelection as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except InvalidAssignee as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except NotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (BulkPlanStale, BulkMutationConflict) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return BulkAssignmentApplyResponse(
            plan_digest=result.plan_digest,
            mutation_id=result.mutation_id,
            assignment_ids=result.assignment_ids,
        )

    @router.get("/audit", response_model=list[AuditEventResponse])
    def list_audit_events(
        principal: Annotated[
            AuthenticatedPrincipal,
            Depends(current_principal),
        ],
        actor_id: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
        action: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
        target_type: Annotated[str | None, Query(min_length=1, max_length=200)] = (
            None
        ),
        target_id: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
        mutation_id: Annotated[str | None, Query(min_length=1, max_length=200)] = (
            None
        ),
        result: Annotated[Literal["success"] | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> list[AuditEventResponse]:
        require_admin(principal)
        try:
            events = repository.list_audit_events(
                admin_id=principal.id,
                actor_id=actor_id,
                action=action,
                target_type=target_type,
                target_id=target_id,
                mutation_id=mutation_id,
                result=result,
                limit=limit,
            )
        except Forbidden as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        return [
            AuditEventResponse.model_validate(event, from_attributes=True)
            for event in events
        ]

    return router
