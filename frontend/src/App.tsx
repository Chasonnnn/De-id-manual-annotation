import { useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { ENTITY_TYPES, type CanonicalSpan } from "./hosted/types";
import ManualAnnotationPane from "./components/ManualAnnotationPane";
import TranscriptRows from "./components/TranscriptRows";
import CodebookPanel from "./components/CodebookPanel";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  ApiError,
  applyBulkAssignment,
  assignFolder,
  completeAssignment,
  createAdminFolder,
  createAdminUser,
  deactivateAdminUser,
  getAdminProgress,
  getAdminUsers,
  getCurrentUser,
  getDocument,
  getWorkspace,
  login,
  logout,
  moveSessionsToFolder,
  previewBulkAssignment,
  reactivateAdminUser,
  resetAdminUserPassword,
  saveAnnotations,
  type IncompleteAssignmentAction,
} from "./hosted/api";
import type {
  AdminProgress,
  HostedDocument,
  HostedUser,
  SaveAnnotationsRequest,
  SaveStatus,
  SessionSummary,
} from "./hosted/types";

interface PendingSave extends SaveAnnotationsRequest {
  documentId: string;
}

interface QueuedSave {
  documentId: string;
  mutation_id: string;
  spans: CanonicalSpan[];
  expected_revision?: number;
}

interface LocalDraft {
  documentId: string;
  mutation_id: string;
  spans: CanonicalSpan[];
  expected_revision: number;
  saved_at: string;
}

const DRAFT_KEY_PREFIX = "deid_annotation_draft:";

function draftKey(documentId: string): string {
  return `${DRAFT_KEY_PREFIX}${documentId}`;
}

function persistDraft(draft: LocalDraft): void {
  try {
    localStorage.setItem(draftKey(draft.documentId), JSON.stringify(draft));
  } catch {
    // The in-memory save queue remains authoritative when storage is unavailable.
  }
}

function readDraft(documentId: string): LocalDraft | null {
  try {
    const raw = localStorage.getItem(draftKey(documentId));
    if (!raw) return null;
    const draft = JSON.parse(raw) as LocalDraft;
    if (draft.documentId !== documentId || !Array.isArray(draft.spans)) return null;
    return draft;
  } catch {
    return null;
  }
}

function clearDraft(documentId: string): void {
  try {
    localStorage.removeItem(draftKey(documentId));
  } catch {
    // A stale draft can be safely replaced by the next edit.
  }
}

type RecoveryCopyState =
  | { status: "idle" | "copying" | "copied"; error: null }
  | { status: "error"; error: string };

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed.";
}

function accountStateLabel(state: HostedUser["state"]): string {
  if (state === "pending_activation") return "Pending activation";
  if (state === "deactivated") return "Deactivated";
  return "Active";
}

function sessionReviewState(session: SessionSummary): "new" | "started" | "complete" {
  if (session.assignment_state === "completed") return "complete";
  return session.manual_annotation_count > 0 ? "started" : "new";
}

function sessionReviewLabel(state: ReturnType<typeof sessionReviewState>): string {
  if (state === "complete") return "Complete";
  if (state === "started") return "Started";
  return "New";
}

function sessionCountLabel(count: number): string {
  return `${count} ${count === 1 ? "session" : "sessions"}`;
}

function conflictCurrentRevision(message: string | null): number | undefined {
  const match = message?.match(/current revision is (\d+)/i);
  return match ? Number(match[1]) : undefined;
}

function useUnsavedChangesGuard(active: boolean) {
  useEffect(() => {
    if (!active) return;
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = true;
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [active]);
}

function LoginScreen({ onAuthenticated }: { onAuthenticated: (user: HostedUser) => Promise<void> }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const user = await login(email.trim(), password);
      setPassword("");
      await onAuthenticated(user);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="login-page">
      <Card className="login-card">
        <form className="contents" onSubmit={handleSubmit}>
          <div className="brand-mark" aria-hidden="true">D</div>
          <h1>Sign in</h1>
          {error && <div className="form-error" role="alert">{error}</div>}
          <label htmlFor="login-email">Email</label>
          <Input
            id="login-email"
            name="email"
            type="email"
            autoComplete="username"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
          <label htmlFor="login-password">Password</label>
          <Input
            id="login-password"
            name="password"
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
          <Button className="primary-button" type="submit" disabled={submitting}>
            {submitting ? "Signing in…" : "Sign in"}
          </Button>
        </form>
      </Card>
    </main>
  );
}

function SessionSidebar({
  user,
  sessions,
  selectedId,
  saveStatus,
  onSelect,
  onCodebook,
  onAdmin,
  onLogout,
}: {
  user: HostedUser;
  sessions: SessionSummary[];
  selectedId: string | null;
  saveStatus: SaveStatus;
  onSelect: (id: string) => void;
  onCodebook: () => void;
  onAdmin: () => void;
  onLogout: () => void;
}) {
  const [query, setQuery] = useState("");
  const [collapsedFolderIds, setCollapsedFolderIds] = useState<string[]>([]);
  const visible = sessions.filter((session) =>
    `${session.external_id} ${session.filename} ${session.folder_name ?? "Unfiled"}`
      .toLowerCase()
      .includes(query.trim().toLowerCase()),
  );
  const groups = visible.reduce<Array<{
    id: string;
    name: string;
    sessions: SessionSummary[];
  }>>((result, session) => {
    const id = session.folder_id ?? "unfiled";
    const existing = result.find((group) => group.id === id);
    if (existing) {
      existing.sessions.push(session);
    } else {
      result.push({
        id,
        name: session.folder_name ?? "Unfiled",
        sessions: [session],
      });
    }
    return result;
  }, []).sort((left, right) => {
    if (left.id === "unfiled") return 1;
    if (right.id === "unfiled") return -1;
    return left.name.localeCompare(right.name);
  });
  const searching = query.trim().length > 0;
  const collapsedFolderIdSet = new Set(collapsedFolderIds);
  const navigationLocked = saveStatus !== "saved";
  const navigationBlockedByFailure = saveStatus === "error" || saveStatus === "conflict";

  function toggleFolder(folderId: string) {
    setCollapsedFolderIds((current) => current.includes(folderId)
      ? current.filter((id) => id !== folderId)
      : [...current, folderId]);
  }

  return (
    <aside className="hosted-sidebar">
      <div className="sidebar-brand">
        <span className="brand-mark small" aria-hidden="true">D</span>
        <span>De-ID Annotation</span>
      </div>
      <div className="sidebar-section-title">Folders</div>
      <label className="visually-hidden" htmlFor="session-search">Search sessions</label>
      <Input
        id="session-search"
        className="session-search"
        type="search"
        placeholder="Search sessions"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      <nav className="session-list" aria-label="Sessions">
        {groups.map((group) => {
          const expanded = searching || !collapsedFolderIdSet.has(group.id);
          return (
            <section className="session-folder" key={group.id}>
              <button
                className="session-folder-toggle"
                type="button"
                aria-expanded={expanded}
                aria-label={`${group.name}, ${sessionCountLabel(group.sessions.length)}`}
                onClick={() => toggleFolder(group.id)}
              >
                <span>{group.name}</span>
                <small>{group.sessions.length}</small>
                <b>{expanded ? "Hide" : "Show"}</b>
              </button>
              {expanded && group.sessions.map((session) => {
                const reviewState = sessionReviewState(session);
                return (
                  <Button
                    variant="ghost"
                    className={session.id === selectedId ? "session-row active" : "session-row"}
                    type="button"
                    key={session.id}
                    disabled={navigationBlockedByFailure}
                    aria-disabled={navigationLocked}
                    onClick={() => {
                      if (!navigationLocked) onSelect(session.id);
                    }}
                  >
                    <span className="session-title">{session.external_id}</span>
                    <span className={`state-dot ${reviewState}`} aria-hidden="true" />
                    <span className="session-state">{sessionReviewLabel(reviewState)}</span>
                  </Button>
                );
              })}
            </section>
          );
        })}
        {visible.length === 0 && <p className="sidebar-empty">No sessions</p>}
      </nav>
      <div className="sidebar-tools">
        <Button className="sidebar-tool-button" variant="outline" type="button" onClick={onCodebook}>
          Codebook
        </Button>
        {user.role === "admin" && (
          <Button className="admin-button sidebar-tool-button" variant="outline" type="button" disabled={saveStatus !== "saved"} onClick={onAdmin}>
            Manage assignments
          </Button>
        )}
      </div>
      <div className="account-block">
        <span>{user.email}</span>
        <small>{user.role === "admin" ? "Admin" : "Annotator"}</small>
        <Button variant="ghost" type="button" disabled={saveStatus !== "saved"} onClick={onLogout}>Sign out</Button>
      </div>
    </aside>
  );
}

function CodebookDialog({ onClose }: { onClose: () => void }) {
  return (
    <div className="codebook-dialog-layer">
      <dialog
        open
        className="codebook-dialog"
        aria-labelledby="codebook-dialog-title"
        aria-modal="true"
        onKeyDown={(event) => {
          if (event.key === "Escape") onClose();
        }}
      >
        <header className="codebook-dialog-header">
          <div>
            <h2 id="codebook-dialog-title">Annotation codebook</h2>
            <span>{ENTITY_TYPES.length} types</span>
          </div>
          <Button
            variant="outline"
            type="button"
            autoFocus
            aria-label="Close codebook"
            onClick={onClose}
          >
            Close
          </Button>
        </header>
        <CodebookPanel />
      </dialog>
    </div>
  );
}

function WorkspacePanels({
  document,
  spans,
  saveStatus,
  saveError,
  savedAt,
  completing,
  hasNextSession,
  recoveryJson,
  recoveryCopyState,
  onSpansChange,
  onComplete,
  onNextSession,
  onRetrySave,
  onCopyRecovery,
}: {
  document: HostedDocument;
  spans: CanonicalSpan[];
  saveStatus: SaveStatus;
  saveError: string | null;
  savedAt: Date | null;
  completing: boolean;
  hasNextSession: boolean;
  recoveryJson: string | null;
  recoveryCopyState: RecoveryCopyState;
  onSpansChange: (spans: CanonicalSpan[]) => void;
  onComplete: () => void;
  onNextSession: () => void;
  onRetrySave: () => void;
  onCopyRecovery: () => void;
}) {
  const completed = document.assignment?.state === "completed";
  const [comparisonMode, setComparisonMode] = useState(false);
  const [completionReviewOpen, setCompletionReviewOpen] = useState(false);
  const manualScrollRef = useRef<HTMLDivElement>(null);
  const referenceScrollRef = useRef<HTMLDivElement>(null);
  const scrollSyncLockRef = useRef(false);
  const statusLabel = saveStatus === "saving"
    ? "Saving…"
    : saveStatus === "conflict"
      ? "Conflict"
      : saveStatus === "error"
        ? "Save failed"
        : "Saved";
  const savedTime = savedAt?.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });

  function synchronizeScroll(source: HTMLDivElement, target: HTMLDivElement | null) {
    if (!target || scrollSyncLockRef.current) return;
    scrollSyncLockRef.current = true;
    target.scrollTop = source.scrollTop;
    window.requestAnimationFrame(() => {
      scrollSyncLockRef.current = false;
    });
  }

  return (
    <div className="workspace-shell">
      <header className="workspace-header">
        <div className="session-heading">
          <h1>{document.external_id}</h1>
        </div>
        <div className="workspace-actions">
          <div
            className={`save-state ${saveStatus}`}
            role={saveStatus === "conflict" || saveStatus === "error" ? "alert" : "status"}
          >
            <span className="save-state-dot" aria-hidden="true" />
            <span className="save-copy">
              <strong>{statusLabel}</strong>
              {saveStatus === "saved" && savedTime && <small>at {savedTime}</small>}
              {saveError && <small>{saveError}</small>}
            </span>
          </div>
          {saveStatus === "error" && (
              <Button className="primary-button compact" size="compact" type="button" onClick={onRetrySave}>
                Retry save
              </Button>
          )}
          <Button
            variant="outline"
            className="compact header-secondary"
            size="compact"
            type="button"
            disabled={!hasNextSession || saveStatus !== "saved" || completing}
            onClick={onNextSession}
          >
            Next session
          </Button>
          {document.assignment !== null && (
            <Button
              className="primary-button compact"
              size="compact"
              type="button"
              disabled={completed || completing || saveStatus !== "saved"}
              onClick={() => setCompletionReviewOpen(true)}
            >
              {completed ? "Completed" : completing ? "Completing…" : "Review & complete"}
            </Button>
          )}
        </div>
      </header>
      {saveStatus === "conflict" && recoveryJson && (
        <div className="locked-note conflict-recovery">
          <strong>Server data was not overwritten.</strong>
          <span>Your local annotations remain unsaved. Copy the recovery JSON before refreshing.</span>
          <Button
            variant="outline"
            size="compact"
            type="button"
            disabled={recoveryCopyState.status === "copying"}
            onClick={onCopyRecovery}
          >
            {recoveryCopyState.status === "copying" ? "Copying…" : "Copy recovery JSON"}
          </Button>
          {recoveryCopyState.status === "copied" && <span role="status">Recovery JSON copied.</span>}
          {recoveryCopyState.status === "error" && <span role="alert">{recoveryCopyState.error}</span>}
        </div>
      )}
      <div className="comparison-toolbar">
        <label>
          <input
            type="checkbox"
            checked={comparisonMode}
            disabled={document.reference_annotations === null}
            onChange={(event) => setComparisonMode(event.target.checked)}
          />
          Comparison
        </label>
      </div>
      <div className="panel-viewport">
        <div className="panel-grid">
        <section className="hosted-panel manual-panel">
          <div className="panel-heading">
            <h2>Manual annotation</h2>
            <span>{spans.length} spans</span>
          </div>
          <ManualAnnotationPane
            text={document.raw_text}
            spans={spans}
            labels={[...ENTITY_TYPES]}
            comparisonMode={comparisonMode}
            referenceSpans={document.reference_annotations ?? []}
            scrollRef={manualScrollRef}
            onScroll={(event) => synchronizeScroll(event.currentTarget, referenceScrollRef.current)}
            onSpansChange={onSpansChange}
          />
        </section>
        <section className="hosted-panel reference-panel">
          <div className="panel-heading">
            <h2>Reference</h2>
            <span>{document.reference_annotations?.length ?? 0} spans</span>
          </div>
          <div
            className="hosted-panel-body transcript-body"
            ref={referenceScrollRef}
            onScroll={(event) => synchronizeScroll(event.currentTarget, manualScrollRef.current)}
          >
            {document.reference_annotations === null ? (
              <div className="reference-empty">
                <span aria-hidden="true">—</span>
                <p>Reference annotations are not available for this session.</p>
              </div>
            ) : (
              <TranscriptRows
                text={document.raw_text}
                spans={document.reference_annotations}
                comparisonSpans={spans}
                comparisonMode={comparisonMode}
              />
            )}
          </div>
        </section>
        </div>
      </div>
      {completionReviewOpen && !completed && (
          <dialog open className="completion-dialog" aria-labelledby="completion-title">
            <h2 id="completion-title">Complete this session?</h2>
            <dl>
              <div><dt>Manual annotations</dt><dd>{spans.length}</dd></div>
              <div><dt>Reference annotations</dt><dd>{document.reference_annotations?.length ?? 0}</dd></div>
              <div><dt>Save status</dt><dd>{savedTime ? `Saved at ${savedTime}` : "Saved"}</dd></div>
            </dl>
            <div className="completion-actions">
              <Button variant="outline" type="button" onClick={() => setCompletionReviewOpen(false)}>
                Keep editing
              </Button>
              <Button
                className="primary-button"
                type="button"
                disabled={completing || saveStatus !== "saved"}
                onClick={() => {
                  setCompletionReviewOpen(false);
                  onComplete();
                }}
              >
                Complete session
              </Button>
            </div>
          </dialog>
      )}
    </div>
  );
}

interface AccountControlsState {
  busy: { userId: string; kind: "reset" | "deactivate" | "reactivate" } | null;
  deactivation: {
    userId: string;
    choice: "" | "unassign" | "reassign";
    reassignToId: string;
  } | null;
  error: string | null;
  notice: string | null;
  oneTimeLink: { url: string; label: string } | null;
}

const idleAccountControls: AccountControlsState = {
  busy: null,
  deactivation: null,
  error: null,
  notice: null,
  oneTimeLink: null,
};

function AnnotatorAccounts({
  accounts,
  onAccountChanged,
}: {
  accounts: HostedUser[];
  onAccountChanged: () => Promise<void>;
}) {
  const [controls, setControls] = useState<AccountControlsState>(idleAccountControls);
  const activeAnnotators = accounts.filter(
    (user) => user.role === "annotator" && user.state === "active",
  );
  const anyAccountActionBusy = controls.busy !== null;

  async function handleResetPassword(annotator: HostedUser) {
    setControls({ ...idleAccountControls, busy: { userId: annotator.id, kind: "reset" } });
    try {
      const activation = await resetAdminUserPassword(annotator.id);
      await onAccountChanged();
      setControls({
        ...idleAccountControls,
        notice: `Password reset started for ${annotator.email}.`,
        oneTimeLink: {
          url: activation.activation_url,
          label: `Password reset link for ${annotator.email}`,
        },
      });
    } catch (caught) {
      setControls({ ...idleAccountControls, error: errorMessage(caught) });
    }
  }

  async function handleReactivate(annotator: HostedUser) {
    setControls({ ...idleAccountControls, busy: { userId: annotator.id, kind: "reactivate" } });
    try {
      await reactivateAdminUser(annotator.id);
      await onAccountChanged();
      setControls({ ...idleAccountControls, notice: `${annotator.email} was reactivated.` });
    } catch (caught) {
      setControls({ ...idleAccountControls, error: errorMessage(caught) });
    }
  }

  async function handleDeactivate(event: FormEvent<HTMLFormElement>, annotator: HostedUser) {
    event.preventDefault();
    const deactivation = controls.deactivation;
    if (!deactivation?.choice || (deactivation.choice === "reassign" && !deactivation.reassignToId)) return;
    const incompleteAssignments: IncompleteAssignmentAction = deactivation.choice === "unassign"
      ? { action: "unassign" }
      : { action: "reassign", assignee_id: deactivation.reassignToId };
    setControls({ ...idleAccountControls, deactivation, busy: { userId: annotator.id, kind: "deactivate" } });
    try {
      await deactivateAdminUser(annotator.id, incompleteAssignments);
      await onAccountChanged();
      setControls({ ...idleAccountControls, notice: `${annotator.email} was deactivated.` });
    } catch (caught) {
      setControls({ ...idleAccountControls, deactivation, error: errorMessage(caught) });
    }
  }

  return (
    <section className="admin-card" aria-labelledby="accounts-heading">
      <h2 id="accounts-heading">Accounts</h2>
      {controls.error && <div className="form-error" role="alert">{controls.error}</div>}
      {controls.notice && (
        <div className="admin-notice" role="status">
          {controls.notice}
          {controls.oneTimeLink && <> <a href={controls.oneTimeLink.url}>{controls.oneTimeLink.label}</a></>}
        </div>
      )}
      {accounts.length === 0 ? (
        <p className="admin-empty">No accounts.</p>
      ) : (
        <div className="account-list">
          {accounts.map((annotator) => {
            const isBusy = controls.busy?.userId === annotator.id;
            const isDeactivating = controls.deactivation?.userId === annotator.id;
            const reassignCandidates = activeAnnotators.filter((candidate) => candidate.id !== annotator.id);
            return (
              <article key={annotator.id}>
                <div className="account-summary">
                  <div>
                    <strong>{annotator.email}</strong>
                    <small>{annotator.role === "admin" ? "Admin" : "Annotator"}</small>
                  </div>
                  <span className={`account-state ${annotator.state}`}>{accountStateLabel(annotator.state)}</span>
                  <div className="account-actions">
                    {annotator.role === "annotator" && (
                      annotator.state === "deactivated" ? (
                        <Button variant="outline" size="compact" disabled={anyAccountActionBusy} onClick={() => void handleReactivate(annotator)}>
                          {isBusy && controls.busy?.kind === "reactivate" ? "Reactivating…" : `Reactivate ${annotator.email}`}
                        </Button>
                      ) : (
                        <>
                        <Button variant="outline" size="compact" disabled={anyAccountActionBusy} onClick={() => void handleResetPassword(annotator)}>
                          {isBusy && controls.busy?.kind === "reset" ? "Resetting…" : `Reset password for ${annotator.email}`}
                        </Button>
                        <Button
                          className="danger-button"
                          variant="outline"
                          size="compact"
                          disabled={anyAccountActionBusy || isDeactivating}
                          onClick={() => setControls({
                            ...idleAccountControls,
                            deactivation: { userId: annotator.id, choice: "", reassignToId: "" },
                          })}
                        >
                          {`Deactivate ${annotator.email}`}
                        </Button>
                        </>
                      )
                    )}
                  </div>
                </div>
                {isDeactivating && controls.deactivation && (
                  <form className="deactivation-form" onSubmit={(event) => void handleDeactivate(event, annotator)}>
                    <fieldset disabled={anyAccountActionBusy}>
                      <legend>{`Unfinished work for ${annotator.email}`}</legend>
                      <label>
                        <input
                          type="radio"
                          name={`deactivation-${annotator.id}`}
                          checked={controls.deactivation.choice === "unassign"}
                          onChange={() => setControls((current) => ({
                            ...current,
                            deactivation: { userId: annotator.id, choice: "unassign", reassignToId: "" },
                          }))}
                        />
                        Unassign unfinished work
                      </label>
                      <label>
                        <input
                          type="radio"
                          name={`deactivation-${annotator.id}`}
                          checked={controls.deactivation.choice === "reassign"}
                          onChange={() => setControls((current) => ({
                            ...current,
                            deactivation: { userId: annotator.id, choice: "reassign", reassignToId: "" },
                          }))}
                        />
                        Reassign unfinished work
                      </label>
                      {controls.deactivation.choice === "reassign" && (
                        <label className="reassign-select">
                          Reassign to
                          <select
                            required
                            value={controls.deactivation.reassignToId}
                            onChange={(event) => setControls((current) => ({
                              ...current,
                              deactivation: { userId: annotator.id, choice: "reassign", reassignToId: event.target.value },
                            }))}
                          >
                            <option value="">Select an active annotator</option>
                            {reassignCandidates.map((candidate) => (
                              <option key={candidate.id} value={candidate.id}>{candidate.email}</option>
                            ))}
                          </select>
                        </label>
                      )}
                    </fieldset>
                    <div className="deactivation-actions">
                      <Button
                        className="danger-button"
                        type="submit"
                        size="compact"
                        disabled={!controls.deactivation.choice
                          || (controls.deactivation.choice === "reassign" && !controls.deactivation.reassignToId)
                          || anyAccountActionBusy}
                      >
                        {isBusy ? "Deactivating…" : "Confirm deactivation"}
                      </Button>
                      <Button variant="outline" size="compact" disabled={anyAccountActionBusy} onClick={() => setControls(idleAccountControls)}>Cancel</Button>
                    </div>
                  </form>
                )}
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}

function AdminView({
  currentAdmin,
  sessions,
  onAssigned,
}: {
  currentAdmin: HostedUser;
  sessions: SessionSummary[];
  onAssigned: () => Promise<void>;
}) {
  const [progress, setProgress] = useState<AdminProgress | null>(null);
  const [annotators, setAnnotators] = useState<HostedUser[]>([]);
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [folderAssigneeId, setFolderAssigneeId] = useState("");
  const [selectedAssigneeId, setSelectedAssigneeId] = useState("");
  const [moveTargetFolderId, setMoveTargetFolderId] = useState("");
  const [newFolderName, setNewFolderName] = useState("");
  const [creatingFolder, setCreatingFolder] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [oneTimeLink, setOneTimeLink] = useState<{ url: string; label: string } | null>(null);
  const [newEmail, setNewEmail] = useState("");
  const [creatingUser, setCreatingUser] = useState(false);

  const activeAnnotators = annotators.filter((user) => user.state === "active");
  const invitedAnnotators = annotators.filter(
    (user) => user.state === "active" || user.state === "pending_activation",
  );
  const assignmentCandidates = [currentAdmin, ...activeAnnotators];
  const folders = progress?.folders ?? [];
  const unfiledSessions = sessions.filter((session) => !session.folder_id);
  const selectedFolder = folders.find((folder) => folder.id === selectedFolderId) ?? null;
  const selectedFolderSessions = selectedFolder
    ? sessions.filter((session) => session.folder_id === selectedFolder.id)
    : selectedFolderId === "unfiled"
      ? unfiledSessions
      : [];
  const selectedDocumentIdSet = new Set(selectedDocumentIds);

  const loadAdminData = useCallback(async () => {
    try {
      const [nextProgress, users] = await Promise.all([getAdminProgress(), getAdminUsers()]);
      setProgress(nextProgress);
      setAnnotators(users.filter((user) => user.role === "annotator"));
    } catch (caught) {
      setError(errorMessage(caught));
    }
  }, []);

  useEffect(() => {
    void loadAdminData();
  }, [loadAdminData]);

  useEffect(() => {
    if (!progress) return;
    if (selectedFolderId && (selectedFolderId === "unfiled" || progress.folders.some((folder) => folder.id === selectedFolderId))) return;
    setSelectedFolderId(progress.folders[0]?.id ?? "unfiled");
  }, [progress, selectedFolderId]);

  async function handleAssignFolder(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedFolder || !folderAssigneeId) return;
    setSubmitting(true);
    setError(null);
    setNotice(null);
    try {
      await assignFolder(selectedFolder.id, folderAssigneeId);
      await Promise.all([loadAdminData(), onAssigned()]);
      setNotice(`${selectedFolder.name} was assigned.`);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleCreateFolder(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const name = newFolderName.trim();
    if (!name) return;
    setCreatingFolder(true);
    setError(null);
    setNotice(null);
    try {
      const created = await createAdminFolder(name);
      await loadAdminData();
      setSelectedFolderId(created.id);
      setNewFolderName("");
      setNotice(`${created.name} was created.`);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setCreatingFolder(false);
    }
  }

  async function handleMoveSessions(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!moveTargetFolderId || selectedDocumentIds.length === 0) return;
    setSubmitting(true);
    setError(null);
    setNotice(null);
    try {
      const target = await moveSessionsToFolder(moveTargetFolderId, selectedDocumentIds);
      await Promise.all([loadAdminData(), onAssigned()]);
      setSelectedDocumentIds([]);
      setNotice(`${selectedDocumentIds.length} session${selectedDocumentIds.length === 1 ? "" : "s"} moved to ${target.name}.`);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleAssignSelected(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedAssigneeId || selectedDocumentIds.length === 0) return;
    const assignee = annotators.find((user) => user.id === selectedAssigneeId);
    if (!assignee) return;
    setSubmitting(true);
    setError(null);
    setNotice(null);
    try {
      const preview = await previewBulkAssignment(selectedDocumentIds, [selectedAssigneeId]);
      const result = await applyBulkAssignment(
        selectedDocumentIds,
        [selectedAssigneeId],
        preview.plan_digest,
        crypto.randomUUID(),
      );
      await Promise.all([loadAdminData(), onAssigned()]);
      setSelectedDocumentIds([]);
      setSelectedAssigneeId("");
      setNotice(
        `${result.assignment_ids.length} session${result.assignment_ids.length === 1 ? "" : "s"} assigned to ${assignee.email}.`,
      );
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleCreateUser(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setCreatingUser(true);
    setError(null);
    setNotice(null);
    setOneTimeLink(null);
    try {
      const activation = await createAdminUser({
        email: newEmail.trim(),
        role: "annotator",
      });
      setNewEmail("");
      await Promise.all([loadAdminData(), onAssigned()]);
      setNotice("Annotator created.");
      setOneTimeLink({ url: activation.activation_url, label: "Activation link" });
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setCreatingUser(false);
    }
  }

  return (
    <main className="admin-view">
      <header>
        <p className="eyebrow">Administration</p>
        <h1>Assignments and progress</h1>
      </header>
      {error && <div className="form-error" role="alert">{error}</div>}
      {notice && (
        <div className="admin-notice" role="status">
          {notice}
          {oneTimeLink && <> <a href={oneTimeLink.url}>{oneTimeLink.label}</a></>}
        </div>
      )}
      {progress && (
        <section className="progress-cards" aria-label="Progress overview">
          {([
            ["Total", progress.totals.total],
            ["Unassigned", progress.totals.unassigned],
            ["Assigned", progress.totals.assigned],
            ["In progress", progress.totals.in_progress],
            ["Completed", progress.totals.completed],
          ] as const).map(([label, value]) => (
            <article key={label}><span>{label}</span><strong>{value}</strong></article>
          ))}
        </section>
      )}
      <Card className="admin-card folder-management-card">
        <div className="folder-management-heading">
          <h2>Session folders</h2>
          <form className="new-folder-form" onSubmit={handleCreateFolder}>
            <label className="visually-hidden" htmlFor="new-folder-name">Folder name</label>
            <Input
              id="new-folder-name"
              placeholder="Folder name"
              required
              value={newFolderName}
              onChange={(event) => setNewFolderName(event.target.value)}
            />
            <Button variant="outline" type="submit" disabled={creatingFolder || !newFolderName.trim()}>
              {creatingFolder ? "Creating…" : "New folder"}
            </Button>
          </form>
        </div>
        <div className="folder-management-layout">
          <nav className="admin-folder-list" aria-label="Session folders">
            {folders.map((folder) => (
              <button
                className={selectedFolderId === folder.id ? "active" : ""}
                type="button"
                key={folder.id}
                aria-label={`${folder.name}, ${sessionCountLabel(folder.session_count)}`}
                aria-pressed={selectedFolderId === folder.id}
                onClick={() => {
                  setSelectedFolderId(folder.id);
                  setSelectedDocumentIds([]);
                }}
              >
                <span>{folder.name}</span>
                <small>{folder.session_count}</small>
              </button>
            ))}
            <button
              className={selectedFolderId === "unfiled" ? "active" : ""}
              type="button"
              aria-label={`Unfiled, ${sessionCountLabel(unfiledSessions.length)}`}
              aria-pressed={selectedFolderId === "unfiled"}
              onClick={() => {
                setSelectedFolderId("unfiled");
                setSelectedDocumentIds([]);
              }}
            >
              <span>Unfiled</span>
              <small>{unfiledSessions.length}</small>
            </button>
          </nav>
          <section className="folder-detail" aria-live="polite">
            <div className="folder-detail-header">
              <div>
                <h3>{selectedFolder?.name ?? "Unfiled"}</h3>
                <span>{sessionCountLabel(selectedFolderSessions.length)}</span>
              </div>
              {selectedFolder && (
                <form className="folder-assignment-form" onSubmit={handleAssignFolder}>
                  <label htmlFor="folder-assignee">Folder assignee</label>
                  <select id="folder-assignee" required value={folderAssigneeId} onChange={(event) => setFolderAssigneeId(event.target.value)}>
                    <option value="">Select an assignee</option>
                    {assignmentCandidates.map((candidate) => (
                      <option key={candidate.id} value={candidate.id}>
                        {candidate.id === currentAdmin.id ? `${candidate.email} (you)` : candidate.email}
                      </option>
                    ))}
                  </select>
                  <Button className="primary-button" type="submit" disabled={submitting || !folderAssigneeId}>
                    {submitting ? "Assigning…" : "Assign folder"}
                  </Button>
                </form>
              )}
            </div>
            <div className="folder-session-list">
              {selectedFolderSessions.map((session) => (
                <label key={session.id}>
                  <input
                    type="checkbox"
                    aria-label={`Select ${session.external_id}`}
                    checked={selectedDocumentIdSet.has(session.id)}
                    onChange={(event) => setSelectedDocumentIds((current) => event.target.checked
                      ? [...current, session.id]
                      : current.filter((id) => id !== session.id))}
                  />
                  <span>{session.external_id}</span>
                  <small>{session.assignment_id ? "Assigned" : "Unassigned"}</small>
                  <b>{sessionReviewLabel(sessionReviewState(session))}</b>
                </label>
              ))}
              {selectedFolderSessions.length === 0 && <p className="admin-empty">No sessions.</p>}
            </div>
            <div className="folder-selected-actions">
              <span>{selectedDocumentIds.length} selected</span>
              <form className="selected-assignment-form" onSubmit={handleAssignSelected}>
                <label htmlFor="selected-assignee">Selected assignee</label>
                <select id="selected-assignee" required value={selectedAssigneeId} onChange={(event) => setSelectedAssigneeId(event.target.value)}>
                  <option value="">Select an assignee</option>
                  {invitedAnnotators.map((candidate) => (
                    <option key={candidate.id} value={candidate.id}>{candidate.email}</option>
                  ))}
                </select>
                <Button className="primary-button" type="submit" disabled={submitting || selectedDocumentIds.length === 0 || !selectedAssigneeId}>
                  {submitting ? "Assigning…" : "Assign selected"}
                </Button>
              </form>
              <form className="folder-move-form" onSubmit={handleMoveSessions}>
                <label htmlFor="move-target-folder">Move to folder</label>
                <select id="move-target-folder" required value={moveTargetFolderId} onChange={(event) => setMoveTargetFolderId(event.target.value)}>
                  <option value="">Select a folder</option>
                  {folders.map((folder) => folder.id === selectedFolderId
                    ? null
                    : <option key={folder.id} value={folder.id}>{folder.name}</option>)}
                </select>
                <Button variant="outline" type="submit" disabled={submitting || selectedDocumentIds.length === 0 || !moveTargetFolderId}>
                  Move selected
                </Button>
              </form>
            </div>
          </section>
        </div>
      </Card>
      <Card className="admin-card">
        <h2>Add annotator</h2>
        <form className="stacked-form" onSubmit={handleCreateUser}>
          <label htmlFor="new-user-email">Email</label>
          <Input id="new-user-email" type="email" autoComplete="off" required value={newEmail} onChange={(event) => setNewEmail(event.target.value)} />
          <Button className="primary-button" type="submit" disabled={creatingUser}>
            {creatingUser ? "Creating…" : "Create annotator"}
          </Button>
        </form>
      </Card>
      <AnnotatorAccounts
        accounts={[currentAdmin, ...annotators]}
        onAccountChanged={async () => {
          await Promise.all([loadAdminData(), onAssigned()]);
        }}
      />
      {progress && progress.annotators.length > 0 && (
        <section className="admin-card">
          <h2>Annotator progress</h2>
          <div className="annotator-list">
            {progress.annotators.map((annotator) => (
              <article key={annotator.user_id}>
                <div><strong>{annotator.email}</strong></div>
                <span>{annotator.completed} completed · {annotator.in_progress} in progress · {annotator.assigned} assigned</span>
              </article>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}

export default function App() {
  const [checkingAuth, setCheckingAuth] = useState(true);
  const [user, setUser] = useState<HostedUser | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [document, setDocument] = useState<HostedDocument | null>(null);
  const [spans, setSpans] = useState<CanonicalSpan[]>([]);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("saved");
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<Date | null>(null);
  const [documentLoadState, setDocumentLoadState] = useState<"idle" | "loading">("idle");
  const [completing, setCompleting] = useState(false);
  const [adminOpen, setAdminOpen] = useState(false);
  const [codebookOpen, setCodebookOpen] = useState(false);
  const [appError, setAppError] = useState<string | null>(null);
  const [recoveryCopyState, setRecoveryCopyState] = useState<RecoveryCopyState>({ status: "idle", error: null });
  const revisionRef = useRef(0);
  const saveInFlightRef = useRef(false);
  const pendingSaveRef = useRef<PendingSave | null>(null);
  const queuedSaveRef = useRef<QueuedSave | null>(null);
  const documentRef = useRef<HostedDocument | null>(null);
  const documentLoadRequestRef = useRef(0);
  const recoveryCopyRequestRef = useRef(0);

  useUnsavedChangesGuard(saveStatus !== "saved");

  const loadWorkspace = useCallback(async () => {
    const workspace = await getWorkspace();
    setSessions(workspace.sessions);
  }, []);

  useEffect(() => {
    let active = true;
    void getCurrentUser()
      .then(async (currentUser) => {
        if (!active) return;
        setUser(currentUser);
        await loadWorkspace();
      })
      .catch((caught) => {
        if (active && !(caught instanceof ApiError && caught.status === 401)) {
          setAppError(errorMessage(caught));
        }
      })
      .finally(() => {
        if (active) setCheckingAuth(false);
      });
    return () => { active = false; };
  }, [loadWorkspace]);

  async function handleAuthenticated(currentUser: HostedUser) {
    const workspace = await getWorkspace();
    setSessions(workspace.sessions);
    setUser(currentUser);
  }

  async function handleLogout() {
    try {
      await logout();
      documentLoadRequestRef.current += 1;
      setUser(null);
      setSessions([]);
      setDocument(null);
      setSelectedId(null);
      setSavedAt(null);
      setAdminOpen(false);
      setAppError(null);
    } catch (caught) {
      setAppError(`Sign out failed. Your session is still active. ${errorMessage(caught)}`);
    }
  }

  async function handleSelect(id: string) {
    if (saveInFlightRef.current || pendingSaveRef.current || queuedSaveRef.current) return;
    const requestId = documentLoadRequestRef.current + 1;
    documentLoadRequestRef.current = requestId;
    setSelectedId(id);
    setAdminOpen(false);
    setDocumentLoadState("loading");
    setAppError(null);
    try {
      const nextDocument = await getDocument(id);
      if (documentLoadRequestRef.current !== requestId) return;
      documentRef.current = nextDocument;
      revisionRef.current = nextDocument.annotation_revision;
      pendingSaveRef.current = null;
      const draft = readDraft(nextDocument.id);
      const canRestoreDraft = draft !== null
        && (
          user?.role === "admin"
          || nextDocument.assignment?.assignee_id === user?.id
        );
      queuedSaveRef.current = canRestoreDraft ? {
        documentId: draft.documentId,
        spans: draft.spans.map((span) => ({ ...span })),
        mutation_id: draft.mutation_id,
        expected_revision: draft.expected_revision,
      } : null;
      saveInFlightRef.current = false;
      setDocument(nextDocument);
      setSpans(canRestoreDraft ? draft.spans : nextDocument.manual_annotations);
      setSaveStatus("saved");
      setSaveError(null);
      setSavedAt(canRestoreDraft ? null : new Date());
      recoveryCopyRequestRef.current += 1;
      setRecoveryCopyState({ status: "idle", error: null });
      if (canRestoreDraft) void flushSave();
    } catch (caught) {
      if (documentLoadRequestRef.current !== requestId) return;
      setAppError(errorMessage(caught));
      setDocument(null);
    } finally {
      if (documentLoadRequestRef.current === requestId) {
        setDocumentLoadState("idle");
      }
    }
  }

  const flushSave = useCallback(async (retryFailed = false) => {
    const activeDocument = documentRef.current;
    if (!activeDocument || saveInFlightRef.current) return;

    let pending = pendingSaveRef.current;
    if (pending && !retryFailed) return;
    if (!pending) {
      const queued = queuedSaveRef.current;
      if (!queued || queued.documentId !== activeDocument.id) return;
      queuedSaveRef.current = null;
      pending = {
        documentId: queued.documentId,
        spans: queued.spans,
        expected_revision: queued.expected_revision ?? revisionRef.current,
        mutation_id: queued.mutation_id,
      };
      pendingSaveRef.current = pending;
    }
    if (pending.documentId !== activeDocument.id) return;

    saveInFlightRef.current = true;
    setSaveStatus("saving");
    setSaveError(null);
    recoveryCopyRequestRef.current += 1;
    setRecoveryCopyState({ status: "idle", error: null });
    try {
      const saved = await saveAnnotations(activeDocument.id, {
        spans: pending.spans,
        expected_revision: pending.expected_revision,
        mutation_id: pending.mutation_id,
      });
      if (pendingSaveRef.current === pending) pendingSaveRef.current = null;
      revisionRef.current = saved.revision;
      if (documentRef.current?.id === activeDocument.id) {
        if (documentRef.current.assignment && saved.assignment_state) {
          documentRef.current = {
            ...documentRef.current,
            assignment: {
              ...documentRef.current.assignment,
              state: saved.assignment_state,
            },
          };
          setDocument(documentRef.current);
        }
        setSessions((current) => current.map((session) =>
          session.id === activeDocument.id
            ? {
                ...session,
                assignment_state: saved.assignment_state ?? session.assignment_state,
                manual_annotation_count: saved.spans.length,
              }
            : session,
        ));
        if (queuedSaveRef.current === null) setSpans(saved.spans);
      }
      saveInFlightRef.current = false;
      if (queuedSaveRef.current !== null) {
        await flushSave();
      } else {
        clearDraft(activeDocument.id);
        setSavedAt(new Date());
        setSaveStatus("saved");
      }
    } catch (caught) {
      saveInFlightRef.current = false;
      if (caught instanceof ApiError && caught.status === 409) {
        setSaveStatus("conflict");
        setSaveError(errorMessage(caught));
      } else {
        setSaveStatus("error");
        setSaveError(errorMessage(caught));
      }
    }
  }, []);

  const handleSpansChange = useCallback((nextSpans: CanonicalSpan[]) => {
    setSpans(nextSpans);
    recoveryCopyRequestRef.current += 1;
    setRecoveryCopyState({ status: "idle", error: null });
    const activeDocument = documentRef.current;
    if (!activeDocument) return;
    const queued: QueuedSave = {
      documentId: activeDocument.id,
      spans: nextSpans.map((span) => ({ ...span })),
      mutation_id: crypto.randomUUID(),
    };
    queuedSaveRef.current = queued;
    persistDraft({
      ...queued,
      expected_revision: revisionRef.current,
      saved_at: new Date().toISOString(),
    });
    void flushSave();
  }, [flushSave]);

  async function handleComplete() {
    if (
      !document?.assignment ||
      saveStatus !== "saved" ||
      saveInFlightRef.current ||
      pendingSaveRef.current ||
      queuedSaveRef.current
    ) return;
    setCompleting(true);
    setAppError(null);
    try {
      await completeAssignment(document.assignment.id);
      const completedDocument: HostedDocument = {
        ...document,
        assignment: { ...document.assignment, state: "completed" },
      };
      documentRef.current = completedDocument;
      setDocument(completedDocument);
      setSessions((current) => current.map((session) =>
        session.id === document.id ? { ...session, assignment_state: "completed" } : session,
      ));
    } catch (caught) {
      setAppError(errorMessage(caught));
    } finally {
      setCompleting(false);
    }
  }

  const pendingRecovery = pendingSaveRef.current;
  const recoveryJson = saveStatus === "conflict" && document && pendingRecovery
    ? JSON.stringify({
        document_id: pendingRecovery.documentId,
        expected_revision: pendingRecovery.expected_revision,
        ...(conflictCurrentRevision(saveError) === undefined
          ? {}
          : { current_revision: conflictCurrentRevision(saveError) }),
        spans,
      })
    : null;
  const selectedIndex = selectedId === null
    ? -1
    : sessions.findIndex((session) => session.id === selectedId);
  const nextSession = selectedIndex >= 0 ? sessions[selectedIndex + 1] ?? null : null;

  async function handleCopyRecovery() {
    if (!recoveryJson) return;
    const requestId = recoveryCopyRequestRef.current + 1;
    recoveryCopyRequestRef.current = requestId;
    setRecoveryCopyState({ status: "copying", error: null });
    try {
      if (!navigator.clipboard?.writeText) {
        throw new Error("Clipboard access is unavailable.");
      }
      await navigator.clipboard.writeText(recoveryJson);
      if (recoveryCopyRequestRef.current !== requestId) return;
      setRecoveryCopyState({ status: "copied", error: null });
    } catch (caught) {
      if (recoveryCopyRequestRef.current !== requestId) return;
      setRecoveryCopyState({
        status: "error",
        error: `Recovery JSON was not copied. ${errorMessage(caught)}`,
      });
    }
  }

  if (checkingAuth) return <div className="app-loading" role="status">Loading…</div>;
  if (!user) return <LoginScreen onAuthenticated={handleAuthenticated} />;

  return (
    <div className="hosted-app">
      <SessionSidebar
        user={user}
        sessions={sessions}
        selectedId={selectedId}
        saveStatus={saveStatus}
        onSelect={(id) => { void handleSelect(id); }}
        onCodebook={() => setCodebookOpen(true)}
        onAdmin={() => {
          documentLoadRequestRef.current += 1;
          setAdminOpen(true);
          setSelectedId(null);
          setDocument(null);
        }}
        onLogout={() => { void handleLogout(); }}
      />
      <div className="hosted-main">
        {appError && <div className="app-error" role="alert">{appError}</div>}
        {adminOpen && user.role === "admin" ? (
          <AdminView
            currentAdmin={user}
            sessions={sessions}
            onAssigned={loadWorkspace}
          />
        ) : documentLoadState === "loading" ? (
          <div className="app-loading" role="status">Loading session…</div>
        ) : document ? (
          <WorkspacePanels
            document={document}
            spans={spans}
            saveStatus={saveStatus}
            saveError={saveError}
            savedAt={savedAt}
            completing={completing}
            hasNextSession={nextSession !== null}
            recoveryJson={recoveryJson}
            recoveryCopyState={recoveryCopyState}
            onSpansChange={handleSpansChange}
            onComplete={() => { void handleComplete(); }}
            onNextSession={() => {
              if (nextSession) void handleSelect(nextSession.id);
            }}
            onRetrySave={() => { void flushSave(true); }}
            onCopyRecovery={() => { void handleCopyRecovery(); }}
          />
        ) : (
          <div className="workspace-empty">
            <div aria-hidden="true">↖</div>
            <h1>Select a session</h1>
          </div>
        )}
      </div>
      {codebookOpen && <CodebookDialog onClose={() => setCodebookOpen(false)} />}
    </div>
  );
}
