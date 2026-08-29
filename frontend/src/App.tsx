import { useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import type { CanonicalSpan } from "./hosted/types";
import AnnotatedText from "./components/AnnotatedText";
import ManualAnnotationPane from "./components/ManualAnnotationPane";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  ApiError,
  assignSession,
  completeAssignment,
  createAdminUser,
  deactivateAdminUser,
  getAdminProgress,
  getAdminUsers,
  getCurrentUser,
  getDocument,
  getWorkspace,
  login,
  logout,
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
  onAdmin,
  onLogout,
}: {
  user: HostedUser;
  sessions: SessionSummary[];
  selectedId: string | null;
  saveStatus: SaveStatus;
  onSelect: (id: string) => void;
  onAdmin: () => void;
  onLogout: () => void;
}) {
  const [query, setQuery] = useState("");
  const visible = sessions.filter((session) =>
    `${session.external_id} ${session.filename}`.toLowerCase().includes(query.trim().toLowerCase()),
  );

  return (
    <aside className="hosted-sidebar">
      <div className="sidebar-brand">
        <span className="brand-mark small" aria-hidden="true">D</span>
        <span>De-ID Annotation</span>
      </div>
      <div className="sidebar-section-title">Sessions</div>
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
        {visible.map((session) => (
          <Button
            variant="ghost"
            className={session.id === selectedId ? "session-row active" : "session-row"}
            type="button"
            key={session.id}
            disabled={saveStatus !== "saved"}
            onClick={() => onSelect(session.id)}
          >
            <span className="session-title">{session.external_id}</span>
            <span className={`state-dot ${session.assignment_state ?? "unassigned"}`} aria-hidden="true" />
            <span className="session-state">
              {session.assignment_state === "in_progress"
                ? "In progress"
                : session.assignment_state === "completed"
                  ? "Complete"
                  : session.assignment_state === "assigned"
                    ? "Assigned"
                    : "Unassigned"}
            </span>
          </Button>
        ))}
        {visible.length === 0 && <p className="sidebar-empty">No sessions</p>}
      </nav>
      {user.role === "admin" && (
        <Button className="admin-button" variant="outline" type="button" disabled={saveStatus !== "saved"} onClick={onAdmin}>
          Manage assignments
        </Button>
      )}
      <div className="account-block">
        <span>{user.display_name}</span>
        <small>{user.role === "admin" ? "Admin" : "Annotator"}</small>
        <Button variant="ghost" type="button" disabled={saveStatus !== "saved"} onClick={onLogout}>Sign out</Button>
      </div>
    </aside>
  );
}

function WorkspacePanels({
  document,
  spans,
  saveStatus,
  saveError,
  completing,
  readOnly,
  recoveryJson,
  recoveryCopyState,
  onSpansChange,
  onComplete,
  onRetrySave,
  onCopyRecovery,
}: {
  document: HostedDocument;
  spans: CanonicalSpan[];
  saveStatus: SaveStatus;
  saveError: string | null;
  completing: boolean;
  readOnly: boolean;
  recoveryJson: string | null;
  recoveryCopyState: RecoveryCopyState;
  onSpansChange: (spans: CanonicalSpan[]) => void;
  onComplete: () => void;
  onRetrySave: () => void;
  onCopyRecovery: () => void;
}) {
  const completed = document.assignment?.state === "completed";
  const statusLabel = saveStatus === "saving"
    ? "Saving"
    : saveStatus === "conflict"
      ? "Conflict"
      : saveStatus === "error"
        ? "Save failed"
        : "Saved";

  return (
    <div className="workspace-shell">
      <header className="workspace-header">
        <div>
          <p className="eyebrow">Session</p>
          <h1>{document.external_id}</h1>
        </div>
        {!readOnly && (
          <div className="workspace-actions">
            <div
              className={`save-state ${saveStatus}`}
              role={saveStatus === "conflict" || saveStatus === "error" ? "alert" : "status"}
            >
              <span aria-hidden="true" />
              {statusLabel}{saveError ? `: ${saveError}` : ""}
            </div>
            {saveStatus === "error" && (
              <Button className="primary-button compact" size="compact" type="button" onClick={onRetrySave}>
                Retry save
              </Button>
            )}
            <Button
              className="primary-button compact"
              size="compact"
              type="button"
              disabled={completed || completing || saveStatus !== "saved"}
              onClick={onComplete}
            >
              {completed ? "Completed" : completing ? "Completing…" : "Mark complete"}
            </Button>
          </div>
        )}
      </header>
      {!readOnly && saveStatus === "conflict" && recoveryJson && (
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
      <div className="panel-grid">
        <section className="hosted-panel raw-panel">
          <h2>Raw Transcript</h2>
          <div className="hosted-panel-body transcript-body">
            <AnnotatedText text={document.raw_text} spans={[]} />
          </div>
        </section>
        <section className="hosted-panel manual-panel">
          <h2>Manual Annotation</h2>
          {completed || readOnly ? (
            <div className="locked-note">
              {readOnly ? "Admin view is read-only." : "This completed session is read-only until an admin reopens it."}
            </div>
          ) : (
            <ManualAnnotationPane
              text={document.raw_text}
              spans={spans}
              labels={document.label_set}
              onSpansChange={onSpansChange}
            />
          )}
          {(completed || readOnly) && (
            <div className="hosted-panel-body transcript-body">
              <AnnotatedText text={document.raw_text} spans={spans} />
            </div>
          )}
        </section>
        <section className="hosted-panel reference-panel">
          <h2>Reference</h2>
          <div className="hosted-panel-body transcript-body">
            {document.reference_annotations === null ? (
              <div className="reference-empty">
                <span aria-hidden="true">—</span>
                <p>Reference annotations are not available for this session.</p>
              </div>
            ) : (
              <AnnotatedText text={document.raw_text} spans={document.reference_annotations} />
            )}
          </div>
        </section>
      </div>
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
  annotators,
  onAccountChanged,
}: {
  annotators: HostedUser[];
  onAccountChanged: () => Promise<void>;
}) {
  const [controls, setControls] = useState<AccountControlsState>(idleAccountControls);
  const activeAnnotators = annotators.filter((user) => user.state === "active");
  const anyAccountActionBusy = controls.busy !== null;

  async function handleResetPassword(annotator: HostedUser) {
    setControls({ ...idleAccountControls, busy: { userId: annotator.id, kind: "reset" } });
    try {
      const activation = await resetAdminUserPassword(annotator.id);
      await onAccountChanged();
      setControls({
        ...idleAccountControls,
        notice: `Password reset started for ${annotator.display_name}.`,
        oneTimeLink: {
          url: activation.activation_url,
          label: `Password reset link for ${annotator.display_name}`,
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
      setControls({ ...idleAccountControls, notice: `${annotator.display_name} was reactivated.` });
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
      setControls({ ...idleAccountControls, notice: `${annotator.display_name} was deactivated.` });
    } catch (caught) {
      setControls({ ...idleAccountControls, deactivation, error: errorMessage(caught) });
    }
  }

  return (
    <section className="admin-card" aria-labelledby="annotator-accounts-heading">
      <h2 id="annotator-accounts-heading">Annotator accounts</h2>
      {controls.error && <div className="form-error" role="alert">{controls.error}</div>}
      {controls.notice && (
        <div className="admin-notice" role="status">
          {controls.notice}
          {controls.oneTimeLink && <> <a href={controls.oneTimeLink.url}>{controls.oneTimeLink.label}</a></>}
        </div>
      )}
      {annotators.length === 0 ? (
        <p className="admin-empty">No annotator accounts.</p>
      ) : (
        <div className="account-list">
          {annotators.map((annotator) => {
            const isBusy = controls.busy?.userId === annotator.id;
            const isDeactivating = controls.deactivation?.userId === annotator.id;
            const reassignCandidates = activeAnnotators.filter((candidate) => candidate.id !== annotator.id);
            return (
              <article key={annotator.id}>
                <div className="account-summary">
                  <div>
                    <strong>{annotator.display_name}</strong>
                    <small>{annotator.email}</small>
                  </div>
                  <span className={`account-state ${annotator.state}`}>{accountStateLabel(annotator.state)}</span>
                  <div className="account-actions">
                    {annotator.state === "deactivated" ? (
                      <Button variant="outline" size="compact" disabled={anyAccountActionBusy} onClick={() => void handleReactivate(annotator)}>
                        {isBusy && controls.busy?.kind === "reactivate" ? "Reactivating…" : `Reactivate ${annotator.display_name}`}
                      </Button>
                    ) : (
                      <>
                        <Button variant="outline" size="compact" disabled={anyAccountActionBusy} onClick={() => void handleResetPassword(annotator)}>
                          {isBusy && controls.busy?.kind === "reset" ? "Resetting…" : `Reset password for ${annotator.display_name}`}
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
                          {`Deactivate ${annotator.display_name}`}
                        </Button>
                      </>
                    )}
                  </div>
                </div>
                {isDeactivating && controls.deactivation && (
                  <form className="deactivation-form" onSubmit={(event) => void handleDeactivate(event, annotator)}>
                    <fieldset disabled={anyAccountActionBusy}>
                      <legend>{`Unfinished work for ${annotator.display_name}`}</legend>
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
                              <option key={candidate.id} value={candidate.id}>{candidate.display_name}</option>
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
  const [documentId, setDocumentId] = useState("");
  const [assigneeId, setAssigneeId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [oneTimeLink, setOneTimeLink] = useState<{ url: string; label: string } | null>(null);
  const [newEmail, setNewEmail] = useState("");
  const [newDisplayName, setNewDisplayName] = useState("");
  const [creatingUser, setCreatingUser] = useState(false);

  const activeAnnotators = annotators.filter((user) => user.state === "active");
  const assignmentCandidates = [currentAdmin, ...activeAnnotators];

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

  async function handleAssign(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setNotice(null);
    try {
      await assignSession({
        document_id: documentId,
        assignee_id: assigneeId,
      });
      await Promise.all([loadAdminData(), onAssigned()]);
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
        display_name: newDisplayName.trim(),
        role: "annotator",
      });
      setNewEmail("");
      setNewDisplayName("");
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
      <Card className="admin-card">
        <h2>Assign session</h2>
        <form className="assignment-form" onSubmit={handleAssign}>
          <label htmlFor="assignment-session">Session</label>
          <select id="assignment-session" required value={documentId} onChange={(event) => setDocumentId(event.target.value)}>
            <option value="">Select a session</option>
            {sessions.map((session) => <option key={session.id} value={session.id}>{session.external_id}</option>)}
          </select>
          <label htmlFor="assignment-annotator">Assignee</label>
          <select id="assignment-annotator" required value={assigneeId} onChange={(event) => setAssigneeId(event.target.value)}>
            <option value="">Select an assignee</option>
            {assignmentCandidates.map((candidate) => (
              <option key={candidate.id} value={candidate.id}>
                {candidate.id === currentAdmin.id ? `${candidate.display_name} (you)` : candidate.display_name}
              </option>
            ))}
          </select>
          <Button className="primary-button" type="submit" disabled={submitting}>
            {submitting ? "Assigning…" : "Assign session"}
          </Button>
        </form>
      </Card>
      <Card className="admin-card">
        <h2>Add annotator</h2>
        <form className="stacked-form" onSubmit={handleCreateUser}>
          <label htmlFor="new-user-email">Email</label>
          <Input id="new-user-email" type="email" autoComplete="off" required value={newEmail} onChange={(event) => setNewEmail(event.target.value)} />
          <label htmlFor="new-user-name">Display name</label>
          <Input id="new-user-name" required value={newDisplayName} onChange={(event) => setNewDisplayName(event.target.value)} />
          <Button className="primary-button" type="submit" disabled={creatingUser}>
            {creatingUser ? "Creating…" : "Create annotator"}
          </Button>
        </form>
      </Card>
      <AnnotatorAccounts
        annotators={annotators}
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
                <div><strong>{annotator.display_name}</strong><small>{annotator.email}</small></div>
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
  const [documentLoadState, setDocumentLoadState] = useState<"idle" | "loading">("idle");
  const [completing, setCompleting] = useState(false);
  const [adminOpen, setAdminOpen] = useState(false);
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
      queuedSaveRef.current = null;
      saveInFlightRef.current = false;
      setDocument(nextDocument);
      setSpans(nextDocument.manual_annotations);
      setSaveStatus("saved");
      setSaveError(null);
      recoveryCopyRequestRef.current += 1;
      setRecoveryCopyState({ status: "idle", error: null });
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
        expected_revision: revisionRef.current,
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
        if (documentRef.current.assignment?.state === "assigned") {
          documentRef.current = {
            ...documentRef.current,
            assignment: { ...documentRef.current.assignment, state: "in_progress" },
          };
          setDocument(documentRef.current);
          setSessions((current) => current.map((session) =>
            session.id === activeDocument.id ? { ...session, assignment_state: "in_progress" } : session,
          ));
        }
        if (queuedSaveRef.current === null) setSpans(saved.spans);
      }
      saveInFlightRef.current = false;
      if (queuedSaveRef.current !== null) {
        await flushSave();
      } else {
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
    queuedSaveRef.current = {
      documentId: activeDocument.id,
      spans: nextSpans.map((span) => ({ ...span })),
      mutation_id: crypto.randomUUID(),
    };
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

  const canEditDocument = user !== null && document?.assignment?.assignee_id === user.id;
  const pendingRecovery = pendingSaveRef.current;
  const recoveryJson = saveStatus === "conflict" && canEditDocument && document && pendingRecovery
    ? JSON.stringify({
        document_id: pendingRecovery.documentId,
        expected_revision: pendingRecovery.expected_revision,
        ...(conflictCurrentRevision(saveError) === undefined
          ? {}
          : { current_revision: conflictCurrentRevision(saveError) }),
        spans,
      })
    : null;

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
          <AdminView currentAdmin={user} sessions={sessions} onAssigned={loadWorkspace} />
        ) : documentLoadState === "loading" ? (
          <div className="app-loading" role="status">Loading session…</div>
        ) : document ? (
          <WorkspacePanels
            document={document}
            spans={spans}
            saveStatus={saveStatus}
            saveError={saveError}
            completing={completing}
            readOnly={!canEditDocument}
            recoveryJson={recoveryJson}
            recoveryCopyState={recoveryCopyState}
            onSpansChange={handleSpansChange}
            onComplete={() => { void handleComplete(); }}
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
    </div>
  );
}
