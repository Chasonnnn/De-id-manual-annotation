from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from .database import (
    ActivationToken,
    Annotation,
    AnnotationRevision,
    Assignment,
    AuditEvent,
    Batch,
    BootstrapGate,
    BulkAssignmentMutation,
    Document,
    LoginSession,
    SessionFolder,
    SessionFolderMembership,
    User,
    utc_now,
)
from .domain import (
    ActivationTokenRecord,
    AnnotatorPrecondition,
    AssignmentState,
    AuditRecord,
    BulkAssignmentItem,
    BulkAssignmentPlan,
    BulkAssignmentResult,
    BulkMutationConflict,
    BulkPlanStale,
    DocumentAssignmentPrecondition,
    DocumentDetail,
    DocumentImport,
    DocumentProvenance,
    DuplicateExternalId,
    DuplicateFolderName,
    DuplicateSelection,
    FolderAssignmentResult,
    FolderProgress,
    Forbidden,
    ImportedBatch,
    ImportMutationConflict,
    InvalidAccountAction,
    InvalidAssignee,
    InvalidReference,
    LoginSessionRecord,
    ManualAnnotationExport,
    NotFound,
    Progress,
    RevisionConflict,
    Role,
    SaveResult,
    UserState,
    VisibilityDenied,
)


class HostedRepository:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    @staticmethod
    def _assignment_for_update(*, document_id: str):
        return (
            select(Assignment)
            .where(Assignment.document_id == document_id)
            .with_for_update()
        )

    @staticmethod
    def _assignment_by_id_for_update(assignment_id: str):
        return (
            select(Assignment).where(Assignment.id == assignment_id).with_for_update()
        )

    @staticmethod
    def _assignments_for_documents(document_ids: list[str], *, lock: bool):
        statement = select(Assignment).where(Assignment.document_id.in_(document_ids))
        return statement.with_for_update() if lock else statement

    def create_user(
        self,
        *,
        email: str,
        password_hash: str,
        role: Role,
        display_name: str | None = None,
    ) -> User:
        with self._session_factory() as session:
            normalized_email = email.strip().lower()
            user = User(
                email=normalized_email,
                display_name=display_name or normalized_email.split("@", 1)[0],
                password_hash=password_hash,
                role=role,
                state=UserState.ACTIVE,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return user

    def create_pending_user_with_activation(
        self,
        *,
        email: str,
        display_name: str,
        role: str,
        activation: ActivationTokenRecord,
        admin_id: str,
    ) -> User:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            user = User(
                email=email,
                display_name=display_name,
                role=Role(role),
                password_hash=None,
                state=UserState.PENDING_ACTIVATION,
            )
            session.add(user)
            session.flush()
            session.add(
                ActivationToken(
                    token_hash=activation.token_hash,
                    user_id=user.id,
                    expires_at=activation.expires_at,
                )
            )
            self._record_audit(
                session,
                actor_id=admin_id,
                action="account.created",
                target_type="user",
                target_id=user.id,
                before_metadata={"state": None},
                after_metadata={"state": UserState.PENDING_ACTIVATION.value},
            )
            session.commit()
            session.refresh(user)
            return user

    def activate_user(
        self,
        *,
        token_hash: str,
        password_hash: str,
        now: datetime,
    ) -> User | None:
        with self._session_factory() as session:
            activation = session.exec(
                select(ActivationToken)
                .where(ActivationToken.token_hash == token_hash)
                .with_for_update()
            ).one_or_none()
            if activation is None:
                return None
            user = session.get(User, activation.user_id)
            if (
                user is None
                or user.state != UserState.PENDING_ACTIVATION
                or activation.expires_at <= now
            ):
                session.delete(activation)
                session.commit()
                return None
            user.password_hash = password_hash
            user.state = UserState.ACTIVE
            user.updated_at = utc_now()
            session.add(user)
            session.delete(activation)
            self._record_audit(
                session,
                actor_id=user.id,
                action="account.activated",
                target_type="user",
                target_id=user.id,
                before_metadata={"state": UserState.PENDING_ACTIVATION.value},
                after_metadata={"state": UserState.ACTIVE.value},
            )
            session.commit()
            session.refresh(user)
            return user

    def reset_user_password(
        self,
        *,
        user_id: str,
        activation: ActivationTokenRecord,
        admin_id: str,
    ) -> User:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            user = session.get(User, user_id)
            if user is None or not (
                user.role == Role.ANNOTATOR
                or (user.role == Role.ADMIN and user.id == admin_id)
            ):
                raise NotFound("account not found")
            session.exec(
                delete(ActivationToken).where(ActivationToken.user_id == user_id)
            )
            session.exec(delete(LoginSession).where(LoginSession.user_id == user_id))
            before_state = user.state
            user.password_hash = None
            user.state = UserState.PENDING_ACTIVATION
            user.updated_at = utc_now()
            session.add(user)
            session.add(
                ActivationToken(
                    token_hash=activation.token_hash,
                    user_id=user_id,
                    expires_at=activation.expires_at,
                )
            )
            self._record_audit(
                session,
                actor_id=admin_id,
                action="account.password_reset",
                target_type="user",
                target_id=user_id,
                before_metadata={"state": before_state.value},
                after_metadata={"state": UserState.PENDING_ACTIVATION.value},
            )
            session.commit()
            session.refresh(user)
            return user

    def deactivate_user(
        self,
        *,
        user_id: str,
        admin_id: str,
        incomplete_action: str,
        reassign_to_id: str | None,
    ) -> User:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            user = session.get(User, user_id)
            if user is None or user.role != Role.ANNOTATOR:
                raise NotFound("annotator not found")
            before_user_state = user.state
            incomplete = list(
                session.exec(
                    select(Assignment)
                    .where(
                        Assignment.assignee_id == user_id,
                        Assignment.state != AssignmentState.COMPLETED,
                    )
                    .with_for_update()
                ).all()
            )
            if incomplete_action == "reassign":
                reassignee = session.get(User, reassign_to_id)
                if (
                    reassignee is None
                    or reassignee.id == user_id
                    or reassignee.role != Role.ANNOTATOR
                    or reassignee.state != UserState.ACTIVE
                ):
                    raise InvalidAssignee(
                        "assignee must be a different active annotator"
                    )
                for assignment in incomplete:
                    before_assignment = {
                        "assignee_id": assignment.assignee_id,
                        "state": assignment.state.value,
                    }
                    assignment.assignee_id = reassignee.id
                    assignment.assigned_by = admin_id
                    assignment.state = AssignmentState.ASSIGNED
                    assignment.assigned_at = utc_now()
                    assignment.last_activity_at = None
                    assignment.completed_at = None
                    session.add(assignment)
                    self._record_audit(
                        session,
                        actor_id=admin_id,
                        action="assignment.reassigned",
                        target_type="assignment",
                        target_id=assignment.id,
                        before_metadata=before_assignment,
                        after_metadata={
                            "assignee_id": reassignee.id,
                            "state": AssignmentState.ASSIGNED.value,
                        },
                    )
            elif incomplete_action == "unassign":
                for assignment in incomplete:
                    self._record_audit(
                        session,
                        actor_id=admin_id,
                        action="assignment.unassigned",
                        target_type="assignment",
                        target_id=assignment.id,
                        before_metadata={
                            "assignee_id": assignment.assignee_id,
                            "state": assignment.state.value,
                        },
                        after_metadata={
                            "assignee_id": None,
                            "state": "unassigned",
                        },
                    )
                    session.delete(assignment)
            else:
                raise InvalidAccountAction(
                    "explicit unassign or reassign action required"
                )
            user.state = UserState.DEACTIVATED
            user.updated_at = utc_now()
            session.add(user)
            session.exec(delete(LoginSession).where(LoginSession.user_id == user_id))
            session.exec(
                delete(ActivationToken).where(ActivationToken.user_id == user_id)
            )
            self._record_audit(
                session,
                actor_id=admin_id,
                action="account.deactivated",
                target_type="user",
                target_id=user_id,
                before_metadata={
                    "assignment_ids": sorted(item.id for item in incomplete),
                    "state": before_user_state.value,
                },
                after_metadata={
                    "reassignee_id": reassign_to_id,
                    "state": UserState.DEACTIVATED.value,
                },
            )
            session.commit()
            session.refresh(user)
            return user

    def reactivate_user(self, *, user_id: str, admin_id: str) -> User:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            user = session.get(User, user_id)
            if user is None or user.role != Role.ANNOTATOR:
                raise NotFound("annotator not found")
            if user.state == UserState.PENDING_ACTIVATION or not user.password_hash:
                raise InvalidAccountAction(
                    "pending account must be activated with its activation link"
                )
            user.state = UserState.ACTIVE
            user.updated_at = utc_now()
            session.add(user)
            self._record_audit(
                session,
                actor_id=admin_id,
                action="account.reactivated",
                target_type="user",
                target_id=user_id,
                before_metadata={"state": UserState.DEACTIVATED.value},
                after_metadata={"state": UserState.ACTIVE.value},
            )
            session.commit()
            session.refresh(user)
            return user

    def import_batch(
        self,
        *,
        name: str,
        created_by: str,
        documents: list[DocumentImport],
        manifest_digest: str | None = None,
        mutation_id: str | None = None,
    ) -> ImportedBatch:
        with self._session_factory() as session:
            self._require_admin(session, created_by)
            if (manifest_digest is None) != (mutation_id is None):
                raise ValueError(
                    "manifest_digest and mutation_id must be supplied together"
                )
            if mutation_id is not None:
                existing = session.exec(
                    select(Batch).where(Batch.import_mutation_id == mutation_id)
                ).one_or_none()
                if existing is not None:
                    return self._resolve_existing_import(
                        session,
                        existing=existing,
                        manifest_digest=manifest_digest,
                    )
            external_ids: set[str] = set()
            for document in documents:
                if document.external_id in external_ids:
                    raise DuplicateExternalId(
                        f"duplicate external id {document.external_id!r} in batch"
                    )
                external_ids.add(document.external_id)
                self._validate_document_import(document)
            batch = Batch(
                name=name,
                created_by=created_by,
                manifest_digest=manifest_digest,
                import_mutation_id=mutation_id,
            )
            session.add(batch)
            imported_documents = [
                Document(
                    batch_id=batch.id,
                    external_id=document.external_id,
                    filename=document.filename,
                    raw_text=document.raw_text,
                    label_set=document.label_set,
                    reference_spans=document.reference_spans,
                    raw_source=document.raw_source,
                    reference_source=document.reference_source,
                )
                for document in documents
            ]
            session.add_all(imported_documents)
            self._record_audit(
                session,
                actor_id=created_by,
                action="batch.imported",
                target_type="batch",
                target_id=batch.id,
                before_metadata={"state": None},
                after_metadata={
                    "batch_id": batch.id,
                    "imported_count": len(imported_documents),
                    "manifest_digest": manifest_digest,
                    "state": "imported",
                },
                mutation_id=mutation_id,
            )
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                if mutation_id is None:
                    raise
                existing = session.exec(
                    select(Batch).where(Batch.import_mutation_id == mutation_id)
                ).one_or_none()
                if existing is None:
                    raise
                return self._resolve_existing_import(
                    session,
                    existing=existing,
                    manifest_digest=manifest_digest,
                )
            return ImportedBatch(
                batch_id=batch.id,
                imported_count=len(imported_documents),
            )

    def resolve_import_retry(
        self,
        *,
        admin_id: str,
        mutation_id: str,
        manifest_digest: str,
    ) -> ImportedBatch | None:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            existing = session.exec(
                select(Batch).where(Batch.import_mutation_id == mutation_id)
            ).one_or_none()
            if existing is None:
                return None
            return self._resolve_existing_import(
                session,
                existing=existing,
                manifest_digest=manifest_digest,
            )

    def preview_balanced_assignment(
        self,
        *,
        admin_id: str,
        document_ids: list[str],
        annotator_ids: list[str],
    ) -> BulkAssignmentPlan:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            return self._build_balanced_plan(
                session,
                document_ids=document_ids,
                annotator_ids=annotator_ids,
                lock=False,
            )

    def apply_balanced_assignment(
        self,
        *,
        admin_id: str,
        document_ids: list[str],
        annotator_ids: list[str],
        plan_digest: str,
        mutation_id: str,
    ) -> BulkAssignmentResult:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            self._validate_bulk_selection(document_ids, annotator_ids)
            if not mutation_id:
                raise ValueError("mutation id is required")
            previous = self._resolve_existing_bulk_mutation(
                session,
                admin_id=admin_id,
                mutation_id=mutation_id,
                plan_digest=plan_digest,
            )
            if previous is not None:
                return previous

            plan = self._build_balanced_plan(
                session,
                document_ids=document_ids,
                annotator_ids=annotator_ids,
                lock=True,
            )
            previous = self._resolve_existing_bulk_mutation(
                session,
                admin_id=admin_id,
                mutation_id=mutation_id,
                plan_digest=plan_digest,
            )
            if previous is not None:
                return previous
            if plan.plan_digest != plan_digest:
                raise BulkPlanStale("assignment plan is stale; preview again")

            assignment_ids: list[str] = []
            existing_by_document = {
                assignment.document_id: assignment
                for assignment in session.exec(
                    self._assignments_for_documents(
                        [item.document_id for item in plan.assignments],
                        lock=True,
                    )
                ).all()
            }
            now = utc_now()
            for item in plan.assignments:
                assignment = existing_by_document.get(item.document_id)
                if assignment is None:
                    assignment = Assignment(
                        document_id=item.document_id,
                        assignee_id=item.assignee_id,
                        assigned_by=admin_id,
                        assigned_at=now,
                    )
                    session.add(assignment)
                    session.flush()
                    self._record_audit(
                        session,
                        actor_id=admin_id,
                        action="assignment.assigned",
                        target_type="assignment",
                        target_id=assignment.id,
                        before_metadata={
                            "assignee_id": None,
                            "state": "unassigned",
                        },
                        after_metadata={
                            "assignee_id": item.assignee_id,
                            "state": AssignmentState.ASSIGNED.value,
                        },
                        mutation_id=mutation_id,
                    )
                elif assignment.assignee_id != item.assignee_id:
                    before = {
                        "assignee_id": assignment.assignee_id,
                        "state": assignment.state.value,
                    }
                    assignment.assignee_id = item.assignee_id
                    assignment.assigned_by = admin_id
                    assignment.state = AssignmentState.ASSIGNED
                    assignment.assigned_at = now
                    assignment.last_activity_at = None
                    assignment.completed_at = None
                    session.add(assignment)
                    self._record_audit(
                        session,
                        actor_id=admin_id,
                        action="assignment.reassigned",
                        target_type="assignment",
                        target_id=assignment.id,
                        before_metadata=before,
                        after_metadata={
                            "assignee_id": item.assignee_id,
                            "state": AssignmentState.ASSIGNED.value,
                        },
                        mutation_id=mutation_id,
                    )
                assignment_ids.append(assignment.id)

            result = BulkAssignmentResult(
                plan_digest=plan.plan_digest,
                mutation_id=mutation_id,
                assignment_ids=sorted(assignment_ids),
            )
            session.add(
                BulkAssignmentMutation(
                    mutation_id=mutation_id,
                    plan_digest=plan.plan_digest,
                    assignment_ids=result.assignment_ids,
                    actor_id=admin_id,
                )
            )
            self._record_audit(
                session,
                actor_id=admin_id,
                action="assignment.bulk_applied",
                target_type="assignment_plan",
                target_id=plan.plan_digest,
                before_metadata={"assignment_ids": [], "state": "previewed"},
                after_metadata={
                    "assignment_ids": result.assignment_ids,
                    "state": "applied",
                },
                mutation_id=mutation_id,
            )
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                previous_result = self._resolve_existing_bulk_mutation(
                    session,
                    admin_id=admin_id,
                    mutation_id=mutation_id,
                    plan_digest=plan_digest,
                )
                if previous_result is None:
                    raise
                return previous_result
            return result

    def assign_document(
        self, *, document_id: str, assignee_id: str, assigned_by_id: str
    ) -> Assignment:
        with self._session_factory() as session:
            self._require_admin(session, assigned_by_id)
            if session.get(Document, document_id) is None:
                raise NotFound("document not found")
            assignee = session.get(User, assignee_id)
            if (
                assignee is None
                or assignee.state != UserState.ACTIVE
                or not (
                    assignee.role == Role.ANNOTATOR
                    or (assignee.role == Role.ADMIN and assignee.id == assigned_by_id)
                )
            ):
                raise InvalidAssignee(
                    "assignee must be an active annotator or self-assigning admin"
                )
            assignment = session.exec(
                self._assignment_for_update(document_id=document_id)
            ).one_or_none()
            if assignment is None:
                assignment = Assignment(
                    document_id=document_id,
                    assignee_id=assignee_id,
                    assigned_by=assigned_by_id,
                )
                action = "assignment.assigned"
                before_metadata = {"assignee_id": None, "state": "unassigned"}
            elif assignment.assignee_id == assignee_id:
                return assignment
            else:
                action = "assignment.reassigned"
                before_metadata = {
                    "assignee_id": assignment.assignee_id,
                    "state": assignment.state.value,
                }
                assignment.assignee_id = assignee_id
                assignment.assigned_by = assigned_by_id
                assignment.state = AssignmentState.ASSIGNED
                assignment.assigned_at = utc_now()
                assignment.last_activity_at = None
                assignment.completed_at = None
            session.add(assignment)
            self._record_audit(
                session,
                actor_id=assigned_by_id,
                action=action,
                target_type="assignment",
                target_id=assignment.id,
                before_metadata=before_metadata,
                after_metadata={
                    "assignee_id": assignee_id,
                    "state": AssignmentState.ASSIGNED.value,
                },
            )
            session.commit()
            session.refresh(assignment)
            return assignment

    def list_assignments(self, *, admin_id: str) -> list[Assignment]:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            return list(session.exec(select(Assignment).order_by(Assignment.id)).all())

    def list_users(self, *, admin_id: str) -> list[User]:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            return list(session.exec(select(User).order_by(User.email)).all())

    def create_folder(self, *, name: str, admin_id: str) -> FolderProgress:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            folder = SessionFolder(name=name.strip(), created_by=admin_id)
            session.add(folder)
            self._record_audit(
                session,
                actor_id=admin_id,
                action="folder.created",
                target_type="session_folder",
                target_id=folder.id,
                before_metadata={"state": None},
                after_metadata={"folder_id": folder.id, "state": "active"},
            )
            try:
                session.commit()
            except IntegrityError as error:
                session.rollback()
                raise DuplicateFolderName("folder name already exists") from error
            return FolderProgress(
                id=folder.id,
                name=folder.name,
                session_count=0,
                unassigned=0,
                assigned=0,
                in_progress=0,
                completed=0,
            )

    def list_folders(self, *, admin_id: str) -> list[FolderProgress]:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            folders = list(
                session.exec(select(SessionFolder).order_by(SessionFolder.name)).all()
            )
            memberships = list(session.exec(select(SessionFolderMembership)).all())
            assignments = {
                assignment.document_id: assignment
                for assignment in session.exec(select(Assignment)).all()
            }
            document_ids_by_folder: dict[str, list[str]] = {
                folder.id: [] for folder in folders
            }
            for membership in memberships:
                if membership.folder_id in document_ids_by_folder:
                    document_ids_by_folder[membership.folder_id].append(
                        membership.document_id
                    )
            result: list[FolderProgress] = []
            for folder in folders:
                document_ids = document_ids_by_folder[folder.id]
                state_counts = {
                    state: sum(
                        assignments.get(document_id) is not None
                        and assignments[document_id].state == state
                        for document_id in document_ids
                    )
                    for state in AssignmentState
                }
                result.append(
                    FolderProgress(
                        id=folder.id,
                        name=folder.name,
                        session_count=len(document_ids),
                        unassigned=sum(
                            document_id not in assignments
                            for document_id in document_ids
                        ),
                        assigned=state_counts[AssignmentState.ASSIGNED],
                        in_progress=state_counts[AssignmentState.IN_PROGRESS],
                        completed=state_counts[AssignmentState.COMPLETED],
                    )
                )
            return result

    def move_documents_to_folder(
        self,
        *,
        folder_id: str,
        document_ids: list[str],
        admin_id: str,
    ) -> FolderProgress:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            folder = session.get(SessionFolder, folder_id)
            if folder is None:
                raise NotFound("folder not found")
            documents = list(
                session.exec(
                    select(Document).where(Document.id.in_(document_ids))
                ).all()
            )
            if len(documents) != len(set(document_ids)):
                raise NotFound("document not found")
            now = utc_now()
            for document_id in document_ids:
                membership = session.get(SessionFolderMembership, document_id)
                if membership is None:
                    membership = SessionFolderMembership(
                        document_id=document_id,
                        folder_id=folder_id,
                        updated_by=admin_id,
                        updated_at=now,
                    )
                else:
                    membership.folder_id = folder_id
                    membership.updated_by = admin_id
                    membership.updated_at = now
                session.add(membership)
            folder.updated_at = now
            session.add(folder)
            self._record_audit(
                session,
                actor_id=admin_id,
                action="folder.sessions_moved",
                target_type="session_folder",
                target_id=folder_id,
                before_metadata={"document_ids": sorted(document_ids)},
                after_metadata={
                    "document_ids": sorted(document_ids),
                    "folder_id": folder_id,
                },
            )
            session.commit()
        return next(
            folder
            for folder in self.list_folders(admin_id=admin_id)
            if folder.id == folder_id
        )

    def assign_folder(
        self,
        *,
        folder_id: str,
        assignee_id: str,
        admin_id: str,
    ) -> FolderAssignmentResult:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            if session.get(SessionFolder, folder_id) is None:
                raise NotFound("folder not found")
            assignee = session.get(User, assignee_id)
            if (
                assignee is None
                or assignee.state != UserState.ACTIVE
                or not (
                    assignee.role == Role.ANNOTATOR
                    or (assignee.role == Role.ADMIN and assignee.id == admin_id)
                )
            ):
                raise InvalidAssignee(
                    "assignee must be an active annotator or self-assigning admin"
                )
            document_ids = [
                membership.document_id
                for membership in session.exec(
                    select(SessionFolderMembership).where(
                        SessionFolderMembership.folder_id == folder_id
                    )
                ).all()
            ]
            assignments_by_document = {
                assignment.document_id: assignment
                for assignment in session.exec(
                    self._assignments_for_documents(document_ids, lock=True)
                ).all()
            }
            assignment_ids: list[str] = []
            now = utc_now()
            for document_id in document_ids:
                assignment = assignments_by_document.get(document_id)
                if assignment is None:
                    assignment = Assignment(
                        document_id=document_id,
                        assignee_id=assignee_id,
                        assigned_by=admin_id,
                        assigned_at=now,
                    )
                    action = "assignment.assigned"
                    before_metadata = {"assignee_id": None, "state": "unassigned"}
                elif assignment.assignee_id == assignee_id:
                    assignment_ids.append(assignment.id)
                    continue
                else:
                    action = "assignment.reassigned"
                    before_metadata = {
                        "assignee_id": assignment.assignee_id,
                        "state": assignment.state.value,
                    }
                    assignment.assignee_id = assignee_id
                    assignment.assigned_by = admin_id
                    assignment.state = AssignmentState.ASSIGNED
                    assignment.assigned_at = now
                    assignment.last_activity_at = None
                    assignment.completed_at = None
                session.add(assignment)
                session.flush()
                assignment_ids.append(assignment.id)
                self._record_audit(
                    session,
                    actor_id=admin_id,
                    action=action,
                    target_type="assignment",
                    target_id=assignment.id,
                    before_metadata=before_metadata,
                    after_metadata={
                        "assignee_id": assignee_id,
                        "folder_id": folder_id,
                        "state": AssignmentState.ASSIGNED.value,
                    },
                )
            session.commit()
            return FolderAssignmentResult(
                folder_id=folder_id,
                assignment_ids=sorted(assignment_ids),
            )

    def list_visible_documents(self, user_id: str) -> list[Document]:
        with self._session_factory() as session:
            user = session.get(User, user_id)
            if user is None:
                return []
            statement = select(Document).order_by(Document.created_at, Document.id)
            if user.role == Role.ANNOTATOR:
                statement = statement.join(Assignment).where(
                    Assignment.assignee_id == user_id
                )
            return list(session.exec(statement).all())

    def get_document(self, document_id: str, *, user_id: str) -> DocumentDetail:
        with self._session_factory() as session:
            user = session.get(User, user_id)
            document = session.get(Document, document_id)
            if user is None or document is None:
                raise NotFound("document not found")
            assignment = session.exec(
                select(Assignment).where(Assignment.document_id == document_id)
            ).one_or_none()
            if user.role == Role.ANNOTATOR and (
                assignment is None or assignment.assignee_id != user_id
            ):
                raise VisibilityDenied("document not found")
            annotation = session.get(Annotation, document_id)
            membership = session.get(SessionFolderMembership, document_id)
            folder = (
                session.get(SessionFolder, membership.folder_id)
                if membership is not None
                else None
            )
            return DocumentDetail(
                id=document.id,
                batch_id=document.batch_id,
                folder_id=folder.id if folder else None,
                folder_name=folder.name if folder else None,
                external_id=document.external_id,
                filename=document.filename,
                raw_text=document.raw_text,
                label_set=document.label_set,
                reference_spans=document.reference_spans,
                assignment_id=assignment.id if assignment else None,
                assignee_id=assignment.assignee_id if assignment else None,
                assignment_state=assignment.state if assignment else None,
                manual_spans=annotation.spans if annotation else [],
                revision=annotation.revision if annotation else 0,
            )

    def get_document_provenance(
        self, document_id: str, *, admin_id: str
    ) -> DocumentProvenance:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            document = session.get(Document, document_id)
            if document is None:
                raise NotFound("document not found")
            return DocumentProvenance(
                document_id=document.id,
                raw_source=document.raw_source,
                reference_source=document.reference_source,
            )

    def save_annotations(
        self,
        *,
        document_id: str,
        user_id: str,
        spans: list[dict[str, Any]],
        expected_revision: int,
        mutation_id: str,
    ) -> SaveResult:
        with self._session_factory() as session:
            user = session.get(User, user_id)
            assignment = session.exec(
                self._assignment_for_update(document_id=document_id)
            ).one_or_none()
            is_active_admin = (
                user is not None
                and user.role == Role.ADMIN
                and user.state == UserState.ACTIVE
            )
            if not is_active_admin and (
                assignment is None or assignment.assignee_id != user_id
            ):
                raise VisibilityDenied("document not found")

            prior_revision = session.exec(
                select(AnnotationRevision).where(
                    AnnotationRevision.mutation_id == mutation_id
                )
            ).one_or_none()
            if prior_revision is not None:
                if (
                    prior_revision.document_id != document_id
                    or prior_revision.author_id != user_id
                ):
                    raise RevisionConflict(self._current_revision(session, document_id))
                return SaveResult(
                    revision=prior_revision.revision,
                    spans=prior_revision.spans,
                    assignment_state=assignment.state if assignment else None,
                )

            annotation = session.get(Annotation, document_id)
            current_revision = annotation.revision if annotation else 0
            if expected_revision != current_revision:
                raise RevisionConflict(current_revision)

            next_revision = current_revision + 1
            now = utc_now()
            if annotation is None:
                annotation = Annotation(
                    document_id=document_id,
                    spans=spans,
                    revision=next_revision,
                    updated_by=user_id,
                    updated_at=now,
                )
            else:
                annotation.spans = spans
                annotation.revision = next_revision
                annotation.updated_by = user_id
                annotation.updated_at = now
            revision = AnnotationRevision(
                document_id=document_id,
                revision=next_revision,
                mutation_id=mutation_id,
                spans=spans,
                author_id=user_id,
                created_at=now,
            )
            session.add(annotation)
            session.add(revision)
            if assignment:
                assignment.state = AssignmentState.IN_PROGRESS
                assignment.completed_at = None
                assignment.last_activity_at = now
                session.add(assignment)
            try:
                session.commit()
            except IntegrityError as error:
                session.rollback()
                current_revision = self._current_revision(session, document_id)
                raise RevisionConflict(current_revision) from error
            return SaveResult(
                revision=next_revision,
                spans=spans,
                assignment_state=assignment.state if assignment else None,
            )

    def complete_assignment(self, assignment_id: str, *, user_id: str) -> Assignment:
        with self._session_factory() as session:
            user = session.get(User, user_id)
            assignment = session.exec(
                self._assignment_by_id_for_update(assignment_id)
            ).one_or_none()
            is_active_admin = (
                user is not None
                and user.role == Role.ADMIN
                and user.state == UserState.ACTIVE
            )
            if assignment is None or (
                assignment.assignee_id != user_id and not is_active_admin
            ):
                raise VisibilityDenied("assignment not found")
            if assignment.state != AssignmentState.COMPLETED:
                before_state = assignment.state
                now = utc_now()
                assignment.state = AssignmentState.COMPLETED
                assignment.last_activity_at = now
                assignment.completed_at = now
                session.add(assignment)
                self._record_audit(
                    session,
                    actor_id=user_id,
                    action="assignment.completed",
                    target_type="assignment",
                    target_id=assignment.id,
                    before_metadata={"state": before_state.value},
                    after_metadata={"state": AssignmentState.COMPLETED.value},
                )
                session.commit()
                session.refresh(assignment)
            return assignment

    def reopen_assignment(self, assignment_id: str, *, admin_id: str) -> Assignment:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            assignment = session.exec(
                self._assignment_by_id_for_update(assignment_id)
            ).one_or_none()
            if assignment is None:
                raise NotFound("assignment not found")
            before_state = assignment.state
            assignment.state = AssignmentState.IN_PROGRESS
            assignment.completed_at = None
            assignment.last_activity_at = utc_now()
            session.add(assignment)
            self._record_audit(
                session,
                actor_id=admin_id,
                action="assignment.reopened",
                target_type="assignment",
                target_id=assignment.id,
                before_metadata={"state": before_state.value},
                after_metadata={"state": AssignmentState.IN_PROGRESS.value},
            )
            session.commit()
            session.refresh(assignment)
            return assignment

    def progress(self, *, admin_id: str) -> Progress:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            documents = list(session.exec(select(Document)).all())
            assignments = list(session.exec(select(Assignment)).all())
            assigned_user_ids = {assignment.assignee_id for assignment in assignments}
            users = {
                user.id: user
                for user in session.exec(select(User)).all()
                if user.role == Role.ANNOTATOR
                or (user.role == Role.ADMIN and user.id in assigned_user_ids)
            }
            counts_by_user: dict[str, dict[str, Any]] = {
                user.id: {
                    "user_id": user.id,
                    "email": user.email,
                    "display_name": user.display_name,
                    "assigned": 0,
                    "in_progress": 0,
                    "completed": 0,
                    "total": 0,
                }
                for user in users.values()
            }
            for assignment in assignments:
                user = users.get(assignment.assignee_id)
                if user is None:
                    continue
                counts = counts_by_user[user.id]
                counts[assignment.state.value] += 1
                counts["total"] += 1
            state_counts = {
                state: sum(assignment.state == state for assignment in assignments)
                for state in AssignmentState
            }
            return Progress(
                total=len(documents),
                unassigned=len(documents) - len(assignments),
                assigned=state_counts[AssignmentState.ASSIGNED],
                in_progress=state_counts[AssignmentState.IN_PROGRESS],
                completed=state_counts[AssignmentState.COMPLETED],
                by_annotator=sorted(
                    counts_by_user.values(), key=lambda item: item["email"]
                ),
            )

    def export_manual_annotations(
        self, *, admin_id: str
    ) -> list[ManualAnnotationExport]:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            documents = list(
                session.exec(
                    select(Document).order_by(Document.external_id, Document.id)
                ).all()
            )
            assignments = {
                assignment.document_id: assignment
                for assignment in session.exec(select(Assignment)).all()
            }
            annotations = {
                annotation.document_id: annotation
                for annotation in session.exec(select(Annotation)).all()
            }
            exports = [
                ManualAnnotationExport(
                    document_id=document.id,
                    external_id=document.external_id,
                    filename=document.filename,
                    manual_annotations=(
                        annotations[document.id].spans
                        if document.id in annotations
                        else []
                    ),
                    annotation_revision=(
                        annotations[document.id].revision
                        if document.id in annotations
                        else 0
                    ),
                    assignee_id=(
                        assignments[document.id].assignee_id
                        if document.id in assignments
                        else None
                    ),
                    assignment_state=(
                        assignments[document.id].state
                        if document.id in assignments
                        else None
                    ),
                    updated_at=(
                        annotations[document.id].updated_at
                        if document.id in annotations
                        else None
                    ),
                )
                for document in documents
            ]
            self._record_audit(
                session,
                actor_id=admin_id,
                action="annotations.exported",
                target_type="annotation_collection",
                target_id="all",
                before_metadata={"state": "stored"},
                after_metadata={
                    "document_ids": sorted(document.id for document in documents),
                    "state": "exported",
                },
            )
            session.commit()
            return exports

    def list_audit_events(
        self,
        *,
        admin_id: str,
        actor_id: str | None = None,
        action: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        mutation_id: str | None = None,
        result: str | None = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        with self._session_factory() as session:
            self._require_admin(session, admin_id)
            statement = select(AuditEvent)
            filters = (
                (AuditEvent.actor_id, actor_id),
                (AuditEvent.action, action),
                (AuditEvent.target_type, target_type),
                (AuditEvent.target_id, target_id),
                (AuditEvent.mutation_id, mutation_id),
                (AuditEvent.result, result),
            )
            for column, value in filters:
                if value is not None:
                    statement = statement.where(column == value)
            events = session.exec(
                statement.order_by(
                    AuditEvent.occurred_at.desc(), AuditEvent.id.desc()
                ).limit(limit)
            ).all()
            return [
                AuditRecord(
                    id=event.id,
                    actor_id=event.actor_id,
                    action=event.action,
                    target_type=event.target_type,
                    target_id=event.target_id,
                    before_metadata=event.before_metadata,
                    after_metadata=event.after_metadata,
                    mutation_id=event.mutation_id,
                    occurred_at=event.occurred_at,
                    result=event.result,
                    reason=event.reason,
                )
                for event in events
            ]

    def _build_balanced_plan(
        self,
        session: Session,
        *,
        document_ids: list[str],
        annotator_ids: list[str],
        lock: bool,
    ) -> BulkAssignmentPlan:
        self._validate_bulk_selection(document_ids, annotator_ids)

        canonical_document_ids = sorted(document_ids)
        canonical_annotator_ids = sorted(annotator_ids)
        document_statement = select(Document).where(
            Document.id.in_(canonical_document_ids)
        )
        user_statement = select(User).where(User.id.in_(canonical_annotator_ids))
        assignment_statement = self._assignments_for_documents(
            canonical_document_ids,
            lock=lock,
        )
        annotation_statement = select(Annotation).where(
            Annotation.document_id.in_(canonical_document_ids)
        )
        if lock:
            document_statement = document_statement.with_for_update()
            user_statement = user_statement.with_for_update()
            annotation_statement = annotation_statement.with_for_update()

        documents = list(session.exec(document_statement).all())
        if {document.id for document in documents} != set(canonical_document_ids):
            raise NotFound("document not found")
        users = list(session.exec(user_statement).all())
        if {user.id for user in users} != set(canonical_annotator_ids):
            raise InvalidAssignee("assignee must be an invited annotator")
        if any(
            user.role != Role.ANNOTATOR
            or user.state not in (UserState.PENDING_ACTIVATION, UserState.ACTIVE)
            for user in users
        ):
            raise InvalidAssignee("assignee must be an invited annotator")

        assignments_by_document = {
            assignment.document_id: assignment
            for assignment in session.exec(assignment_statement).all()
        }
        revisions_by_document = {
            annotation.document_id: annotation.revision
            for annotation in session.exec(annotation_statement).all()
        }
        assignments = [
            BulkAssignmentItem(
                document_id=document_id,
                assignee_id=canonical_annotator_ids[
                    index % len(canonical_annotator_ids)
                ],
            )
            for index, document_id in enumerate(canonical_document_ids)
        ]
        document_preconditions = []
        for document_id in canonical_document_ids:
            current = assignments_by_document.get(document_id)
            document_preconditions.append(
                DocumentAssignmentPrecondition(
                    document_id=document_id,
                    assignment_id=current.id if current else None,
                    assignee_id=current.assignee_id if current else None,
                    state=current.state if current else None,
                    revision=revisions_by_document.get(document_id, 0),
                )
            )
        user_states = {user.id: user.state for user in users}
        annotator_preconditions = [
            AnnotatorPrecondition(user_id=user_id, state=user_states[user_id])
            for user_id in canonical_annotator_ids
        ]
        canonical_payload = {
            "annotators": [
                {"state": item.state.value, "user_id": item.user_id}
                for item in annotator_preconditions
            ],
            "assignments": [
                {
                    "assignee_id": item.assignee_id,
                    "document_id": item.document_id,
                }
                for item in assignments
            ],
            "documents": [
                {
                    "assignee_id": item.assignee_id,
                    "assignment_id": item.assignment_id,
                    "document_id": item.document_id,
                    "state": item.state.value if item.state else None,
                    "revision": item.revision,
                }
                for item in document_preconditions
            ],
            "version": 1,
        }
        plan_digest = hashlib.sha256(
            json.dumps(
                canonical_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return BulkAssignmentPlan(
            plan_digest=plan_digest,
            assignments=assignments,
            document_preconditions=document_preconditions,
            annotator_preconditions=annotator_preconditions,
        )

    @staticmethod
    def _validate_bulk_selection(
        document_ids: list[str], annotator_ids: list[str]
    ) -> None:
        if not document_ids or not annotator_ids:
            raise ValueError("documents and annotators are required")
        if len(set(document_ids)) != len(document_ids):
            raise DuplicateSelection("duplicate document id in selection")
        if len(set(annotator_ids)) != len(annotator_ids):
            raise DuplicateSelection("duplicate annotator id in selection")

    @staticmethod
    def _record_audit(
        session: Session,
        *,
        actor_id: str,
        action: str,
        target_type: str,
        target_id: str,
        before_metadata: dict[str, Any],
        after_metadata: dict[str, Any],
        mutation_id: str | None = None,
        result: str = "success",
        reason: str | None = None,
    ) -> None:
        HostedRepository._validate_audit_metadata(before_metadata)
        HostedRepository._validate_audit_metadata(after_metadata)
        session.add(
            AuditEvent(
                actor_id=actor_id,
                action=action,
                target_type=target_type,
                target_id=target_id,
                before_metadata=before_metadata,
                after_metadata=after_metadata,
                mutation_id=mutation_id,
                result=result,
                reason=reason,
            )
        )

    @staticmethod
    def _validate_audit_metadata(metadata: dict[str, Any]) -> None:
        allowed_keys = {
            "assignee_id",
            "assignment_id",
            "assignment_ids",
            "batch_id",
            "document_ids",
            "folder_id",
            "imported_count",
            "manifest_digest",
            "reassignee_id",
            "revision",
            "state",
            "user_id",
        }
        if not set(metadata).issubset(allowed_keys):
            raise ValueError("audit metadata contains a disallowed field")
        for value in metadata.values():
            if value is None or isinstance(value, (str, int)):
                continue
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                continue
            raise ValueError("audit metadata contains a disallowed value")

    @staticmethod
    def _current_revision(session: Session, document_id: str) -> int:
        annotation = session.get(Annotation, document_id)
        return annotation.revision if annotation else 0

    @staticmethod
    def _require_admin(session: Session, user_id: str) -> User:
        user = session.get(User, user_id)
        if user is None or user.role != Role.ADMIN or user.state != UserState.ACTIVE:
            raise Forbidden("admin role required")
        return user

    @staticmethod
    def _resolve_existing_bulk_mutation(
        session: Session,
        *,
        admin_id: str,
        mutation_id: str,
        plan_digest: str,
    ) -> BulkAssignmentResult | None:
        previous = session.get(BulkAssignmentMutation, mutation_id)
        if previous is None:
            return None
        if previous.plan_digest != plan_digest or previous.actor_id != admin_id:
            raise BulkMutationConflict(
                "bulk mutation id was already used for another plan"
            )
        return BulkAssignmentResult(
            plan_digest=previous.plan_digest,
            mutation_id=previous.mutation_id,
            assignment_ids=previous.assignment_ids,
        )

    @staticmethod
    def _resolve_existing_import(
        session: Session,
        *,
        existing: Batch,
        manifest_digest: str | None,
    ) -> ImportedBatch:
        if existing.manifest_digest != manifest_digest:
            raise ImportMutationConflict(
                "import mutation id was already used with different content"
            )
        document_count = len(
            session.exec(
                select(Document.id).where(Document.batch_id == existing.id)
            ).all()
        )
        return ImportedBatch(
            batch_id=existing.id,
            imported_count=document_count,
        )

    @staticmethod
    def _validate_document_import(document: DocumentImport) -> None:
        required_keys = {"start", "end", "label", "text"}
        for index, span in enumerate(document.reference_spans or []):
            if not isinstance(span, dict) or set(span) != required_keys:
                raise InvalidReference(
                    f"reference span {index} must contain start, end, label, and text"
                )
            start = span["start"]
            end = span["end"]
            label = span["label"]
            text = span["text"]
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start < 0
                or end <= start
                or end > len(document.raw_text)
            ):
                raise InvalidReference(f"reference span {index} has invalid offsets")
            if not isinstance(label, str) or label not in document.label_set:
                raise InvalidReference(f"reference span {index} has an unknown label")
            if not isinstance(text, str) or document.raw_text[start:end] != text:
                raise InvalidReference(f"reference text does not match span {index}")

    def bootstrap_admin(
        self, *, email: str, display_name: str, password_hash: str
    ) -> User | None:
        with self._session_factory() as session:
            if session.exec(select(User.id).limit(1)).first() is not None:
                return None
            session.add(BootstrapGate())
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                return None
            normalized_email = email.strip().lower()
            user = User(
                email=normalized_email,
                display_name=display_name,
                password_hash=password_hash,
                role=Role.ADMIN,
                state=UserState.ACTIVE,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return user

    def get_user_by_email(self, email: str) -> User | None:
        with self._session_factory() as session:
            return session.exec(
                select(User).where(User.email == email.strip().lower())
            ).one_or_none()

    def get_user_by_id(self, user_id: str) -> User | None:
        with self._session_factory() as session:
            return session.get(User, user_id)

    def create_login_session(self, login_session: LoginSessionRecord) -> None:
        with self._session_factory() as session:
            session.add(
                LoginSession(
                    token_hash=login_session.token_hash,
                    user_id=login_session.user_id,
                    expires_at=login_session.expires_at,
                )
            )
            session.commit()

    def get_login_session(self, token_hash: str) -> LoginSession | None:
        with self._session_factory() as session:
            return session.get(LoginSession, token_hash)

    def delete_login_session(self, token_hash: str) -> None:
        with self._session_factory() as session:
            session.exec(
                delete(LoginSession).where(LoginSession.token_hash == token_hash)
            )
            session.commit()

    def replace_password_and_delete_sessions(
        self, *, user_id: str, password_hash: str
    ) -> None:
        with self._session_factory() as session:
            user = session.get(User, user_id)
            if user is None:
                raise NotFound("user not found")
            user.password_hash = password_hash
            user.updated_at = utc_now()
            session.add(user)
            session.exec(delete(LoginSession).where(LoginSession.user_id == user_id))
            session.commit()
