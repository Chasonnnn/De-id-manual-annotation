from datetime import datetime
from typing import Annotated, Literal

from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import IntegrityError

from .auth import (
    AuthenticatedPrincipal,
    AuthenticationRequired,
    AuthManager,
    EmailNotAllowed,
    InvalidActivationToken,
    InvalidCredentials,
    WeakPassword,
)
from .bulk_audit_api import create_bulk_audit_router
from .domain import (
    CompletedLocked,
    DocumentImport,
    DuplicateExternalId,
    Forbidden,
    ImportMutationConflict,
    InvalidAccountAction,
    InvalidAssignee,
    InvalidReference,
    NotFound,
    RevisionConflict,
)
from .repository import HostedRepository
from .s3_import import (
    AmbiguousSourcePair,
    CanonicalReferenceJsonDecoder,
    CatalogObject,
    DisallowedSource,
    GovernedS3Catalog,
    GovernedS3Importer,
    InvalidSourceFormat,
    ManifestDigestMismatch,
    MissingSourcePair,
    RawSourceFormat,
    ReferenceSourceFormat,
    S3CatalogConfig,
    S3CatalogError,
    S3ImportError,
    S3ImportManifest,
    S3ManifestDocument,
    S3ObjectRef,
    S3ReadAdapter,
    SourceIntegrityError,
    Utf8TranscriptDecoder,
    canonical_manifest_digest,
    resolve_manifest_sources,
)

SESSION_COOKIE = "annotation_session"
CSRF_COOKIE = "annotation_csrf"


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginRequest(RequestModel):
    email: str
    password: str


class HealthResponse(BaseModel):
    status: Literal["ok"]


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    role: Literal["admin", "annotator"]
    state: Literal["pending_activation", "active", "deactivated"]


class SessionSummaryResponse(BaseModel):
    id: str
    external_id: str
    filename: str
    assignment_id: str | None
    assignment_state: Literal["assigned", "in_progress", "completed"] | None
    assignee_id: str | None
    assignee_name: str | None


class WorkspaceResponse(BaseModel):
    sessions: list[SessionSummaryResponse]


class CanonicalSpan(RequestModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    label: str = Field(min_length=1)
    text: str

    @model_validator(mode="after")
    def validate_range(self) -> CanonicalSpan:
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        return self


class AssignmentResponse(BaseModel):
    id: str
    assignee_id: str
    assignee_name: str
    state: Literal["assigned", "in_progress", "completed"]


class DocumentResponse(BaseModel):
    id: str
    external_id: str
    filename: str
    raw_text: str
    label_set: list[str]
    reference_annotations: list[CanonicalSpan] | None
    manual_annotations: list[CanonicalSpan]
    annotation_revision: int
    assignment: AssignmentResponse | None


class SaveAnnotationsRequest(RequestModel):
    spans: list[CanonicalSpan]
    expected_revision: int = Field(ge=0)
    mutation_id: str = Field(min_length=1, max_length=200)


class SaveAnnotationsResponse(BaseModel):
    revision: int
    spans: list[CanonicalSpan]


class AssignmentStateResponse(BaseModel):
    assignment_id: str
    state: Literal["assigned", "in_progress", "completed"]


class AdminProgressTotals(BaseModel):
    unassigned: int
    assigned: int
    in_progress: int
    completed: int
    total: int


class AnnotatorProgressResponse(BaseModel):
    user_id: str
    display_name: str
    email: str
    assigned: int
    in_progress: int
    completed: int


class AdminProgressResponse(BaseModel):
    totals: AdminProgressTotals
    annotators: list[AnnotatorProgressResponse]


class AssignDocumentRequest(RequestModel):
    assignee_id: str


class AssignmentIdResponse(BaseModel):
    assignment_id: str


class CreateUserRequest(RequestModel):
    email: str = Field(min_length=3)
    display_name: str = Field(min_length=1, max_length=120)
    role: Literal["annotator"]


class ActivationRequest(RequestModel):
    token: str = Field(min_length=1)
    password: str


class ActivationResponse(BaseModel):
    user: UserResponse
    activation_url: str
    activation_expires_at: str


class EmptyRequest(RequestModel):
    pass


class UnassignIncompleteRequest(RequestModel):
    action: Literal["unassign"]


class ReassignIncompleteRequest(RequestModel):
    action: Literal["reassign"]
    assignee_id: str = Field(min_length=1)


IncompleteAssignmentAction = Annotated[
    UnassignIncompleteRequest | ReassignIncompleteRequest,
    Field(discriminator="action"),
]


class DeactivateUserRequest(RequestModel):
    incomplete_assignments: IncompleteAssignmentAction


class ImportSessionRequest(RequestModel):
    external_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    raw_text: str
    label_set: list[str] = Field(min_length=1)
    reference_annotations: list[CanonicalSpan] | None = None


class ImportBatchRequest(RequestModel):
    name: str = Field(min_length=1, max_length=200)
    sessions: list[ImportSessionRequest] = Field(min_length=1)


class ImportBatchResponse(BaseModel):
    batch_id: str
    imported_count: int


class S3ObjectRefRequest(RequestModel):
    bucket: str = Field(min_length=1)
    key: str = Field(min_length=1)
    version_id: str | None = None
    archive_member: str | None = None


class S3ManifestDocumentRequest(RequestModel):
    external_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    label_set: list[str] = Field(min_length=1)
    raw_format: RawSourceFormat
    reference_format: ReferenceSourceFormat | None
    raw: S3ObjectRefRequest
    reference: S3ObjectRefRequest | None


class S3ImportManifestRequest(RequestModel):
    name: str = Field(min_length=1, max_length=200)
    documents: list[S3ManifestDocumentRequest] = Field(min_length=1)


class PlanS3ImportRequest(RequestModel):
    manifest: S3ImportManifestRequest


class S3SourceIdentityResponse(RequestModel):
    external_id: str
    kind: Literal["raw", "reference"]
    format: RawSourceFormat | ReferenceSourceFormat
    archive_member: str | None
    bucket: str
    key: str
    version_id: str | None
    content_length: int
    etag: str | None
    last_modified: datetime | None
    server_checksum_algorithm: str | None
    server_checksum_value: str | None
    checksum_type: str | None


class PlanS3ImportResponse(BaseModel):
    manifest_digest: str
    sources: list[S3SourceIdentityResponse]


class ApplyS3ImportRequest(RequestModel):
    manifest: S3ImportManifestRequest
    expected_manifest_digest: str = Field(min_length=64, max_length=64)
    expected_sources: list[S3SourceIdentityResponse] = Field(min_length=1)
    mutation_id: str = Field(min_length=1, max_length=200)


class ExportSessionResponse(BaseModel):
    document_id: str
    external_id: str
    filename: str
    manual_annotations: list[CanonicalSpan]
    annotation_revision: int
    assignee_id: str | None
    assignment_state: Literal["assigned", "in_progress", "completed"] | None
    updated_at: str | None


class ExportResponse(BaseModel):
    sessions: list[ExportSessionResponse]


def _user_response(principal: AuthenticatedPrincipal) -> UserResponse:
    return UserResponse(
        id=principal.id,
        email=principal.email,
        display_name=principal.display_name,
        role=principal.role,
        state=principal.state,
    )


def _s3_manifest(body: S3ImportManifestRequest) -> S3ImportManifest:
    def source(item: S3ObjectRefRequest) -> S3ObjectRef:
        return S3ObjectRef(
            bucket=item.bucket,
            key=item.key,
            version_id=item.version_id,
            archive_member=item.archive_member,
        )

    manifest = S3ImportManifest(
        name=body.name,
        documents=tuple(
            S3ManifestDocument(
                external_id=item.external_id,
                filename=item.filename,
                label_set=tuple(item.label_set),
                raw=source(item.raw),
                reference=source(item.reference)
                if item.reference is not None
                else None,
                raw_format=item.raw_format,
                reference_format=item.reference_format,
            )
            for item in body.documents
        ),
    )
    return manifest


def _s3_source_response(
    *,
    external_id: str,
    format: RawSourceFormat | ReferenceSourceFormat,
    archive_member: str | None,
    source: CatalogObject,
) -> S3SourceIdentityResponse:
    return S3SourceIdentityResponse(
        external_id=external_id,
        kind=source.kind,
        format=format,
        archive_member=archive_member,
        bucket=source.bucket,
        key=source.key,
        version_id=source.version_id,
        content_length=source.content_length,
        etag=source.etag,
        last_modified=source.last_modified,
        server_checksum_algorithm=source.server_checksum_algorithm,
        server_checksum_value=source.server_checksum_value,
        checksum_type=source.checksum_type,
    )


def _resolved_s3_source_responses(
    resolved: tuple[
        tuple[S3ManifestDocument, CatalogObject, CatalogObject | None], ...
    ],
) -> list[S3SourceIdentityResponse]:
    sources: list[S3SourceIdentityResponse] = []
    for document, raw, reference in resolved:
        sources.append(
            _s3_source_response(
                external_id=document.external_id,
                format=document.raw_format,
                archive_member=document.raw.archive_member,
                source=raw,
            )
        )
        if reference is not None:
            if document.reference_format is None:
                raise AssertionError("resolved reference must declare its format")
            sources.append(
                _s3_source_response(
                    external_id=document.external_id,
                    format=document.reference_format,
                    archive_member=None,
                    source=reference,
                )
            )
    return sources


def create_hosted_app(
    *,
    repository: HostedRepository,
    auth: AuthManager,
    cookie_secure: bool,
    s3_reader: S3ReadAdapter | None = None,
    s3_catalog_config: S3CatalogConfig | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Hosted Annotation Tool",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; object-src 'none'; base-uri 'self'; "
            "frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def get_current_principal(
        annotation_session: Annotated[
            str | None,
            Cookie(alias=SESSION_COOKIE),
        ] = None,
    ) -> AuthenticatedPrincipal:
        if not annotation_session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )
        try:
            return auth.authenticate(annotation_session)
        except AuthenticationRequired as error:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(error),
            ) from error

    CurrentPrincipal = Annotated[
        AuthenticatedPrincipal,
        Depends(get_current_principal),
    ]

    def require_csrf(
        annotation_session: Annotated[
            str | None,
            Cookie(alias=SESSION_COOKIE),
        ] = None,
        annotation_csrf: Annotated[
            str | None,
            Cookie(alias=CSRF_COOKIE),
        ] = None,
        x_csrf_token: Annotated[
            str | None,
            Header(alias="X-CSRF-Token"),
        ] = None,
    ) -> None:
        if (
            not annotation_session
            or not annotation_csrf
            or not x_csrf_token
            or x_csrf_token != annotation_csrf
            or not auth.validate_csrf(annotation_session, x_csrf_token)
        ):
            raise HTTPException(status_code=403, detail="CSRF validation failed")

    def require_admin(principal: AuthenticatedPrincipal) -> None:
        if principal.role != "admin":
            raise HTTPException(status_code=403, detail="admin role required")

    app.include_router(
        create_bulk_audit_router(
            repository=repository,
            current_principal=get_current_principal,
            require_csrf=require_csrf,
        )
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/api/auth/login", response_model=UserResponse)
    def login(body: LoginRequest, response: Response) -> UserResponse:
        try:
            result = auth.login(body.email, body.password)
        except InvalidCredentials as error:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(error),
            ) from error
        response.set_cookie(
            key=SESSION_COOKIE,
            value=result.token,
            expires=result.expires_at,
            httponly=True,
            secure=cookie_secure,
            samesite="lax",
            path="/",
        )
        response.set_cookie(
            key=CSRF_COOKIE,
            value=result.csrf_token,
            expires=result.expires_at,
            httponly=False,
            secure=cookie_secure,
            samesite="lax",
            path="/",
        )
        return _user_response(result.principal)

    @app.post("/api/auth/activate", response_model=UserResponse)
    def activate(body: ActivationRequest) -> UserResponse:
        try:
            return _user_response(auth.activate(body.token, body.password))
        except WeakPassword as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except InvalidActivationToken as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/auth/me", response_model=UserResponse)
    def me(principal: CurrentPrincipal) -> UserResponse:
        return _user_response(principal)

    @app.post(
        "/api/auth/logout",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[Depends(require_csrf)],
    )
    def logout(
        response: Response,
        principal: CurrentPrincipal,
        annotation_session: Annotated[
            str | None,
            Cookie(alias=SESSION_COOKIE),
        ] = None,
    ) -> None:
        del principal
        if annotation_session:
            auth.logout(annotation_session)
        response.delete_cookie(
            key=SESSION_COOKIE,
            path="/",
            secure=cookie_secure,
            httponly=True,
            samesite="lax",
        )
        response.delete_cookie(
            key=CSRF_COOKIE,
            path="/",
            secure=cookie_secure,
            httponly=False,
            samesite="lax",
        )

    @app.get("/api/workspace", response_model=WorkspaceResponse)
    def workspace(principal: CurrentPrincipal) -> WorkspaceResponse:
        assignee_names = {principal.id: principal.display_name}
        if principal.role == "admin":
            assignee_names = {
                user.id: user.display_name
                for user in repository.list_users(admin_id=principal.id)
            }
        sessions: list[SessionSummaryResponse] = []
        for document in repository.list_visible_documents(principal.id):
            detail = repository.get_document(document.id, user_id=principal.id)
            sessions.append(
                SessionSummaryResponse(
                    id=detail.id,
                    external_id=detail.external_id,
                    filename=detail.filename,
                    assignment_id=detail.assignment_id,
                    assignment_state=detail.assignment_state,
                    assignee_id=detail.assignee_id,
                    assignee_name=assignee_names.get(detail.assignee_id),
                )
            )
        return WorkspaceResponse(sessions=sessions)

    def assignee_names_for(principal: AuthenticatedPrincipal) -> dict[str, str]:
        if principal.role == "admin":
            return {
                user.id: user.display_name
                for user in repository.list_users(admin_id=principal.id)
            }
        return {principal.id: principal.display_name}

    @app.get("/api/documents/{document_id}", response_model=DocumentResponse)
    def get_document(
        document_id: str,
        principal: CurrentPrincipal,
    ) -> DocumentResponse:
        try:
            detail = repository.get_document(document_id, user_id=principal.id)
        except NotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        assignment = None
        if detail.assignment_id and detail.assignee_id and detail.assignment_state:
            assignment = AssignmentResponse(
                id=detail.assignment_id,
                assignee_id=detail.assignee_id,
                assignee_name=assignee_names_for(principal).get(
                    detail.assignee_id,
                    "Unknown annotator",
                ),
                state=detail.assignment_state,
            )
        return DocumentResponse(
            id=detail.id,
            external_id=detail.external_id,
            filename=detail.filename,
            raw_text=detail.raw_text,
            label_set=detail.label_set,
            reference_annotations=detail.reference_spans,
            manual_annotations=detail.manual_spans,
            annotation_revision=detail.revision,
            assignment=assignment,
        )

    @app.put(
        "/api/documents/{document_id}/annotations",
        response_model=SaveAnnotationsResponse,
        dependencies=[Depends(require_csrf)],
    )
    def save_annotations(
        document_id: str,
        body: SaveAnnotationsRequest,
        principal: CurrentPrincipal,
    ) -> SaveAnnotationsResponse:
        try:
            detail = repository.get_document(document_id, user_id=principal.id)
        except NotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        normalized: list[dict[str, object]] = []
        for span in body.spans:
            if span.end > len(detail.raw_text):
                raise HTTPException(
                    status_code=422, detail="span is outside transcript"
                )
            if span.label not in detail.label_set:
                raise HTTPException(status_code=422, detail="span label is not allowed")
            if detail.raw_text[span.start : span.end] != span.text:
                raise HTTPException(
                    status_code=422, detail="span text does not match transcript"
                )
            normalized.append(span.model_dump())
        try:
            result = repository.save_annotations(
                document_id=document_id,
                user_id=principal.id,
                spans=normalized,
                expected_revision=body.expected_revision,
                mutation_id=body.mutation_id,
            )
        except RevisionConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except CompletedLocked as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except NotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return SaveAnnotationsResponse(revision=result.revision, spans=result.spans)

    @app.post(
        "/api/assignments/{assignment_id}/complete",
        response_model=AssignmentStateResponse,
        dependencies=[Depends(require_csrf)],
    )
    def complete_assignment(
        assignment_id: str,
        principal: CurrentPrincipal,
    ) -> AssignmentStateResponse:
        try:
            assignment = repository.complete_assignment(
                assignment_id,
                user_id=principal.id,
            )
        except NotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return AssignmentStateResponse(
            assignment_id=assignment.id,
            state=assignment.state,
        )

    @app.post(
        "/api/admin/assignments/{assignment_id}/reopen",
        response_model=AssignmentStateResponse,
        dependencies=[Depends(require_csrf)],
    )
    def reopen_assignment(
        assignment_id: str,
        principal: CurrentPrincipal,
    ) -> AssignmentStateResponse:
        require_admin(principal)
        try:
            assignment = repository.reopen_assignment(
                assignment_id,
                admin_id=principal.id,
            )
        except Forbidden as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except NotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return AssignmentStateResponse(
            assignment_id=assignment.id,
            state=assignment.state,
        )

    @app.get("/api/admin/users", response_model=list[UserResponse])
    def admin_users(principal: CurrentPrincipal) -> list[UserResponse]:
        require_admin(principal)
        return [
            UserResponse(
                id=user.id,
                email=user.email,
                display_name=user.display_name,
                role=user.role,
                state=user.state,
            )
            for user in repository.list_users(admin_id=principal.id)
        ]

    @app.post(
        "/api/admin/users",
        response_model=ActivationResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_csrf)],
    )
    def create_admin_user(
        body: CreateUserRequest,
        principal: CurrentPrincipal,
    ) -> ActivationResponse:
        require_admin(principal)
        try:
            invitation = auth.invite_user(
                email=body.email,
                display_name=body.display_name.strip(),
                role=body.role,
                admin_id=principal.id,
            )
        except EmailNotAllowed as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except IntegrityError as error:
            raise HTTPException(
                status_code=409, detail="email already exists"
            ) from error
        return ActivationResponse(
            user=_user_response(invitation.principal),
            activation_url=f"/activate#token={invitation.token}",
            activation_expires_at=invitation.expires_at.isoformat(),
        )

    @app.post(
        "/api/admin/users/{user_id}/reset-password",
        response_model=ActivationResponse,
        dependencies=[Depends(require_csrf)],
    )
    def reset_admin_user_password(
        user_id: str,
        body: EmptyRequest,
        principal: CurrentPrincipal,
    ) -> ActivationResponse:
        del body
        require_admin(principal)
        try:
            invitation = auth.reset_password(user_id, admin_id=principal.id)
        except NotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return ActivationResponse(
            user=_user_response(invitation.principal),
            activation_url=f"/activate#token={invitation.token}",
            activation_expires_at=invitation.expires_at.isoformat(),
        )

    @app.post(
        "/api/admin/users/{user_id}/deactivate",
        response_model=UserResponse,
        dependencies=[Depends(require_csrf)],
    )
    def deactivate_admin_user(
        user_id: str,
        body: DeactivateUserRequest,
        principal: CurrentPrincipal,
    ) -> UserResponse:
        require_admin(principal)
        action = body.incomplete_assignments
        try:
            user = repository.deactivate_user(
                user_id=user_id,
                admin_id=principal.id,
                incomplete_action=action.action,
                reassign_to_id=(
                    action.assignee_id
                    if isinstance(action, ReassignIncompleteRequest)
                    else None
                ),
            )
        except NotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except InvalidAssignee as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return UserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            role=user.role,
            state=user.state,
        )

    @app.post(
        "/api/admin/users/{user_id}/reactivate",
        response_model=UserResponse,
        dependencies=[Depends(require_csrf)],
    )
    def reactivate_admin_user(
        user_id: str,
        body: EmptyRequest,
        principal: CurrentPrincipal,
    ) -> UserResponse:
        del body
        require_admin(principal)
        try:
            user = repository.reactivate_user(
                user_id=user_id,
                admin_id=principal.id,
            )
        except NotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except InvalidAccountAction as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return UserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            role=user.role,
            state=user.state,
        )

    @app.post(
        "/api/admin/s3-imports/plan",
        response_model=PlanS3ImportResponse,
        dependencies=[Depends(require_csrf)],
    )
    def plan_admin_s3_import(
        body: PlanS3ImportRequest,
        principal: CurrentPrincipal,
    ) -> PlanS3ImportResponse:
        require_admin(principal)
        if s3_reader is None or s3_catalog_config is None:
            raise HTTPException(
                status_code=503,
                detail="governed S3 import is not configured",
            )
        try:
            manifest = _s3_manifest(body.manifest)
            catalog = GovernedS3Catalog(s3_reader, s3_catalog_config).catalog()
            resolved = resolve_manifest_sources(
                manifest=manifest,
                catalog=catalog,
                config=s3_catalog_config,
            )
        except (
            AmbiguousSourcePair,
            DisallowedSource,
            InvalidSourceFormat,
            MissingSourcePair,
            ValueError,
        ) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except S3CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return PlanS3ImportResponse(
            manifest_digest=canonical_manifest_digest(manifest),
            sources=_resolved_s3_source_responses(resolved),
        )

    @app.post(
        "/api/admin/s3-imports/apply",
        response_model=ImportBatchResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_csrf)],
    )
    def apply_admin_s3_import(
        body: ApplyS3ImportRequest,
        principal: CurrentPrincipal,
    ) -> ImportBatchResponse:
        require_admin(principal)
        if s3_reader is None or s3_catalog_config is None:
            raise HTTPException(
                status_code=503,
                detail="governed S3 import is not configured",
            )
        try:
            manifest = _s3_manifest(body.manifest)
            digest = canonical_manifest_digest(manifest)
            if digest != body.expected_manifest_digest:
                raise ManifestDigestMismatch(
                    "expected manifest digest does not match canonical manifest"
                )
            retry = repository.resolve_import_retry(
                admin_id=principal.id,
                mutation_id=body.mutation_id,
                manifest_digest=digest,
            )
            if retry is not None:
                return ImportBatchResponse(
                    batch_id=retry.batch_id,
                    imported_count=retry.imported_count,
                )
            catalog = GovernedS3Catalog(s3_reader, s3_catalog_config).catalog()
            resolved = resolve_manifest_sources(
                manifest=manifest,
                catalog=catalog,
                config=s3_catalog_config,
            )
            current_sources = _resolved_s3_source_responses(resolved)
            if current_sources != body.expected_sources:
                raise SourceIntegrityError("S3 catalog changed after import planning")
            result = GovernedS3Importer(
                reader=s3_reader,
                config=s3_catalog_config,
                repository=repository,
                raw_decoder=Utf8TranscriptDecoder(),
                reference_decoder=CanonicalReferenceJsonDecoder(),
            ).import_manifest(
                manifest=manifest,
                catalog=catalog,
                created_by=principal.id,
                expected_manifest_digest=body.expected_manifest_digest,
                mutation_id=body.mutation_id,
            )
        except (ManifestDigestMismatch, SourceIntegrityError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except MissingSourcePair as error:
            raise HTTPException(
                status_code=409,
                detail="S3 catalog changed after import planning",
            ) from error
        except (DuplicateExternalId, ImportMutationConflict) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (
            AmbiguousSourcePair,
            DisallowedSource,
            InvalidSourceFormat,
            InvalidReference,
            ValueError,
        ) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except (S3CatalogError, S3ImportError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return ImportBatchResponse(
            batch_id=result.batch_id,
            imported_count=result.imported_count,
        )

    @app.post(
        "/api/admin/batches/import",
        response_model=ImportBatchResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_csrf)],
    )
    def import_admin_batch(
        body: ImportBatchRequest,
        principal: CurrentPrincipal,
    ) -> ImportBatchResponse:
        require_admin(principal)
        documents = [
            DocumentImport(
                external_id=item.external_id.strip(),
                filename=item.filename.strip(),
                raw_text=item.raw_text,
                label_set=item.label_set,
                reference_spans=(
                    [span.model_dump() for span in item.reference_annotations]
                    if item.reference_annotations is not None
                    else None
                ),
            )
            for item in body.sessions
        ]
        try:
            result = repository.import_batch(
                name=body.name.strip(),
                created_by=principal.id,
                documents=documents,
            )
        except DuplicateExternalId as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return ImportBatchResponse(
            batch_id=result.batch_id,
            imported_count=result.imported_count,
        )

    @app.get("/api/admin/export", response_model=ExportResponse)
    def export_admin_annotations(principal: CurrentPrincipal) -> ExportResponse:
        require_admin(principal)
        exports = repository.export_manual_annotations(admin_id=principal.id)
        return ExportResponse(
            sessions=[
                ExportSessionResponse(
                    document_id=item.document_id,
                    external_id=item.external_id,
                    filename=item.filename,
                    manual_annotations=item.manual_annotations,
                    annotation_revision=item.annotation_revision,
                    assignee_id=item.assignee_id,
                    assignment_state=item.assignment_state,
                    updated_at=(
                        item.updated_at.isoformat()
                        if item.updated_at is not None
                        else None
                    ),
                )
                for item in exports
            ]
        )

    @app.get("/api/admin/progress", response_model=AdminProgressResponse)
    def admin_progress(principal: CurrentPrincipal) -> AdminProgressResponse:
        require_admin(principal)
        progress = repository.progress(admin_id=principal.id)
        return AdminProgressResponse(
            totals=AdminProgressTotals(
                unassigned=progress.unassigned,
                assigned=progress.assigned,
                in_progress=progress.in_progress,
                completed=progress.completed,
                total=progress.total,
            ),
            annotators=[
                AnnotatorProgressResponse(**item) for item in progress.by_annotator
            ],
        )

    @app.put(
        "/api/admin/documents/{document_id}/assignment",
        response_model=AssignmentIdResponse,
        dependencies=[Depends(require_csrf)],
    )
    def assign_document(
        document_id: str,
        body: AssignDocumentRequest,
        principal: CurrentPrincipal,
    ) -> AssignmentIdResponse:
        require_admin(principal)
        try:
            assignment = repository.assign_document(
                document_id=document_id,
                assignee_id=body.assignee_id,
                assigned_by_id=principal.id,
            )
        except NotFound as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except InvalidAssignee as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return AssignmentIdResponse(assignment_id=assignment.id)

    return app
