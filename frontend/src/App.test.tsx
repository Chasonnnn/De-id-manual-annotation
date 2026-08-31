import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import * as api from "./hosted/api";

vi.mock("./hosted/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./hosted/api")>();
  return {
    ...actual,
    getCurrentUser: vi.fn(),
    login: vi.fn(),
    logout: vi.fn(),
    getWorkspace: vi.fn(),
    getDocument: vi.fn(),
    saveAnnotations: vi.fn(),
    completeAssignment: vi.fn(),
    getAdminProgress: vi.fn(),
    getAdminFolders: vi.fn(),
    getAdminUsers: vi.fn(),
    createAdminFolder: vi.fn(),
    moveSessionsToFolder: vi.fn(),
    assignFolder: vi.fn(),
    assignSession: vi.fn(),
    createAdminUser: vi.fn(),
    deactivateAdminUser: vi.fn(),
    reactivateAdminUser: vi.fn(),
    resetAdminUserPassword: vi.fn(),
  };
});

vi.mock("./components/ManualAnnotationPane", () => ({
  default: ({ onSpansChange }: { onSpansChange: (spans: Array<Record<string, unknown>>) => void }) => (
    <div aria-label="Manual annotation editor">
      <button type="button" onClick={() => onSpansChange([{ start: 0, end: 5, label: "NAME", text: "Alice" }])}>
        Add test annotation
      </button>
      <button type="button" onClick={() => onSpansChange([
        { start: 0, end: 5, label: "NAME", text: "Alice" },
        { start: 10, end: 13, label: "NAME", text: "Bob" },
      ])}>
        Add another annotation
      </button>
    </div>
  ),
}));

const annotator = {
  id: "user-1",
  email: "annotator@cornell.edu",
  display_name: "Ada Annotator",
  role: "annotator" as const,
  state: "active" as const,
};

const admin = {
  id: "admin-1",
  email: "admin@cornell.edu",
  display_name: "Admin User",
  role: "admin" as const,
  state: "active" as const,
};

const sessions = [
  {
    id: "doc-1",
    external_id: "Session 001",
    filename: "session-001.json",
    folder_id: "folder-1",
    folder_name: "August intake",
    assignment_id: "assignment-1",
    assignment_state: "assigned" as const,
    manual_annotation_count: 0,
    assignee_id: annotator.id,
    assignee_name: annotator.display_name,
  },
  {
    id: "doc-2",
    external_id: "Session 002",
    filename: "session-002.json",
    folder_id: "folder-1",
    folder_name: "August intake",
    assignment_id: "assignment-2",
    assignment_state: "completed" as const,
    manual_annotation_count: 0,
    assignee_id: annotator.id,
    assignee_name: annotator.display_name,
  },
];

const folder = {
  id: "folder-1",
  name: "August intake",
  session_count: 2,
  unassigned: 0,
  assigned: 1,
  in_progress: 0,
  completed: 1,
};

const document = {
  id: "doc-1",
  external_id: "Session 001",
  filename: "session-001.json",
  raw_text: "Alice met Bob.",
  label_set: ["NAME"],
  reference_annotations: null,
  manual_annotations: [],
  annotation_revision: 0,
  assignment: {
    id: "assignment-1",
    assignee_id: annotator.id,
    assignee_name: annotator.display_name,
    state: "assigned" as const,
  },
};

function mockAuthenticated(user: typeof annotator | typeof admin = annotator) {
  vi.mocked(api.getCurrentUser).mockResolvedValue(user);
  vi.mocked(api.getWorkspace).mockResolvedValue({ sessions });
  vi.mocked(api.getDocument).mockResolvedValue(document);
  vi.mocked(api.logout).mockResolvedValue(undefined);
  vi.mocked(api.completeAssignment).mockResolvedValue({ assignment_id: "assignment-1", state: "completed" });
}

describe("hosted annotation app", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    vi.mocked(api.getCurrentUser).mockRejectedValue(new api.ApiError(401, "Not authenticated"));
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("logs in with email and password", async () => {
    vi.mocked(api.login).mockResolvedValue(annotator);
    vi.mocked(api.getWorkspace).mockResolvedValue({ sessions });

    render(<App />);

    await screen.findByRole("heading", { name: "Sign in" });
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: annotator.email } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "correct horse battery staple" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

    await screen.findByText("Session 001");
    expect(api.login).toHaveBeenCalledWith(annotator.email, "correct horse battery staple");
    expect(screen.queryByText("Experiments")).toBeNull();
    expect(screen.queryByText("Models")).toBeNull();
    expect(screen.queryByText("Evaluation")).toBeNull();
    expect(screen.queryByText("Prompt Lab")).toBeNull();
    expect(screen.queryByText("Methods Lab")).toBeNull();
  });

  it("keeps the authenticated workspace when logout cannot be confirmed", async () => {
    mockAuthenticated();
    vi.mocked(api.logout).mockRejectedValue(new api.ApiError(503, "The server did not confirm logout."));

    render(<App />);
    await screen.findByText("Session 001");
    fireEvent.click(screen.getByRole("button", { name: "Sign out" }));

    expect((await screen.findByRole("alert")).textContent).toContain("Your session is still active");
    expect(screen.getByText("Session 001")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Sign out" })).toBeTruthy();
  });

  it("combines the raw transcript and manual annotation into a two-panel workspace", async () => {
    mockAuthenticated();

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));

    expect(await screen.findByRole("heading", { name: "Manual annotation" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Reference" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Raw Transcript" })).toBeNull();
    expect(screen.getByText("Reference annotations are not available for this session.")).toBeTruthy();
    expect(screen.getByRole("checkbox", { name: "Comparison" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Manage assignments" })).toBeNull();
  });

  it("uses blue New, yellow Started, and green Complete session states", async () => {
    mockAuthenticated();
    vi.mocked(api.getWorkspace).mockResolvedValue({
      sessions: [
        sessions[0]!,
        { ...sessions[0]!, id: "doc-started", external_id: "Session Started", manual_annotation_count: 2 },
        sessions[1]!,
      ],
    });

    render(<App />);

    expect(await screen.findByText("New")).toBeTruthy();
    expect(screen.getByText("Started")).toBeTruthy();
    expect(screen.getByText("Complete")).toBeTruthy();
    expect(window.document.querySelector(".state-dot.new")).toBeTruthy();
    expect(window.document.querySelector(".state-dot.started")).toBeTruthy();
    expect(window.document.querySelector(".state-dot.complete")).toBeTruthy();
  });

  it("lets users collapse and reopen session folders in the workspace sidebar", async () => {
    mockAuthenticated();

    render(<App />);

    const folder = await screen.findByRole("button", { name: "August intake, 2 sessions" });
    expect(folder.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByText("Session 001")).toBeTruthy();
    expect(screen.getByText("Session 002")).toBeTruthy();

    fireEvent.click(folder);

    expect(folder.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByText("Session 001")).toBeNull();
    expect(screen.queryByText("Session 002")).toBeNull();

    fireEvent.click(folder);
    expect(screen.getByText("Session 001")).toBeTruthy();
    expect(screen.getByText("Session 002")).toBeTruthy();
  });

  it("opens the next session without redundant header metadata", async () => {
    mockAuthenticated();
    vi.mocked(api.getDocument).mockImplementation(async (id) => ({
      ...document,
      id,
      external_id: id === "doc-2" ? "Session 002" : "Session 001",
    }));

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));
    expect(await screen.findByRole("heading", { name: "Session 001" })).toBeTruthy();
    expect(screen.queryByText("1 of 2 assigned")).toBeNull();
    expect(window.document.querySelector(".session-heading .eyebrow")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Next session" }));

    await waitFor(() => expect(api.getDocument).toHaveBeenLastCalledWith("doc-2"));
    expect(await screen.findByRole("heading", { name: "Session 002" })).toBeTruthy();
  });

  it("lets an admin edit a session assigned to an annotator", async () => {
    mockAuthenticated(admin);

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));

    expect(await screen.findByLabelText("Manual annotation editor")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Review & complete" })).toBeTruthy();
    expect(screen.queryByText(/Read-only/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Retry save" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Copy recovery JSON" })).toBeNull();
  });

  it("lets an admin annotate an unassigned session without showing completion", async () => {
    mockAuthenticated(admin);
    vi.mocked(api.getWorkspace).mockResolvedValue({
      sessions: [{
        ...sessions[0]!,
        assignment_id: null,
        assignment_state: null,
        assignee_id: null,
        assignee_name: null,
      }],
    });
    vi.mocked(api.getDocument).mockResolvedValue({ ...document, assignment: null });

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));

    expect(await screen.findByLabelText("Manual annotation editor")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Review & complete" })).toBeNull();
  });

  it("never turns a visible session into a read-only workspace", async () => {
    mockAuthenticated();
    vi.mocked(api.getWorkspace).mockResolvedValue({
      sessions: [{
        ...sessions[0]!,
        assignee_id: "user-2",
        assignee_name: "Other Annotator",
      }],
    });
    vi.mocked(api.getDocument).mockResolvedValue({
      ...document,
      reference_annotations: [
        { start: 0, end: 5, label: "NAME", text: "Alice" },
      ],
      assignment: {
        ...document.assignment,
        assignee_id: "user-2",
        assignee_name: "Other Annotator",
      },
    });

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));

    expect(await screen.findByLabelText("Manual annotation editor")).toBeTruthy();
    expect(screen.queryByText(/Read-only/)).toBeNull();
    expect(screen.queryByRole("button", { name: "NAME: Alice" })).toBeNull();
  });

  it("lets an admin edit a session assigned to their own account", async () => {
    mockAuthenticated(admin);
    vi.mocked(api.getWorkspace).mockResolvedValue({
      sessions: [{ ...sessions[0]!, assignee_id: admin.id, assignee_name: admin.display_name }],
    });
    vi.mocked(api.getDocument).mockResolvedValue({
      ...document,
      assignment: {
        ...document.assignment,
        assignee_id: admin.id,
        assignee_name: admin.display_name,
      },
    });

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));

    expect(await screen.findByLabelText("Manual annotation editor")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Review & complete" })).toBeTruthy();
    expect(screen.queryByText(/Read-only/)).toBeNull();
  });

  it("saves every annotation snapshot with a revision and mutation id", async () => {
    mockAuthenticated();
    let resolveSave: ((value: Awaited<ReturnType<typeof api.saveAnnotations>>) => void) | undefined;
    vi.mocked(api.saveAnnotations).mockImplementation(() => new Promise((resolve) => { resolveSave = resolve; }));

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));
    fireEvent.click(await screen.findByRole("button", { name: "Add test annotation" }));

    expect(screen.getByRole("status").textContent).toContain("Saving");
    expect(api.saveAnnotations).toHaveBeenCalledWith(
      "doc-1",
      expect.objectContaining({
        expected_revision: 0,
        mutation_id: expect.any(String),
        spans: [{ start: 0, end: 5, label: "NAME", text: "Alice" }],
      }),
    );

    resolveSave?.({ revision: 1, spans: [{ start: 0, end: 5, label: "NAME", text: "Alice" }] });
    expect(await screen.findByText("Saved")).toBeTruthy();
  });

  it("renders the save indicator dot separately from its status copy", async () => {
    mockAuthenticated();

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));

    const status = (await screen.findByText("Saved")).closest('[role="status"]');
    expect(status).not.toBeNull();
    expect(status?.firstElementChild?.classList.contains("save-state-dot")).toBe(true);
    expect(status?.lastElementChild?.classList.contains("save-copy")).toBe(true);
  });

  it("keeps the manual editor available while showing the codebook", async () => {
    mockAuthenticated();

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));
    fireEvent.click(await screen.findByRole("tab", { name: "Codebook" }));

    expect(screen.getByText("Manual annotation")).toBeTruthy();
    expect(screen.getByText("Examples are synthetic and contain no session data.")).toBeTruthy();
    expect(screen.getByRole("list", { name: "NAME examples" }).children).toHaveLength(2);
    expect(screen.getByRole("tab", { name: "Codebook" }).getAttribute("aria-selected")).toBe("true");
  });

  it("guards browser unload only while annotation changes are unsaved", async () => {
    mockAuthenticated();
    let resolveSave: ((value: Awaited<ReturnType<typeof api.saveAnnotations>>) => void) | undefined;
    vi.mocked(api.saveAnnotations).mockImplementation(() => new Promise((resolve) => { resolveSave = resolve; }));

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));

    const savedUnload = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(savedUnload);
    expect(savedUnload.defaultPrevented).toBe(false);

    fireEvent.click(await screen.findByRole("button", { name: "Add test annotation" }));
    await screen.findByText("Saving…");
    const unsavedUnload = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(unsavedUnload);
    expect(unsavedUnload.defaultPrevented).toBe(true);

    resolveSave?.({ revision: 1, spans: [{ start: 0, end: 5, label: "NAME", text: "Alice" }] });
    await screen.findByText("Saved");
    const acknowledgedUnload = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(acknowledgedUnload);
    expect(acknowledgedUnload.defaultPrevented).toBe(false);
  });

  it("keeps a local draft until the server confirms the save", async () => {
    mockAuthenticated();
    let resolveSave: ((value: Awaited<ReturnType<typeof api.saveAnnotations>>) => void) | undefined;
    vi.mocked(api.saveAnnotations).mockImplementation(() => new Promise((resolve) => { resolveSave = resolve; }));

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));
    fireEvent.click(await screen.findByRole("button", { name: "Add test annotation" }));

    expect(localStorage.getItem("deid_annotation_draft:doc-1")).toContain('"label":"NAME"');

    resolveSave?.({ revision: 1, spans: [{ start: 0, end: 5, label: "NAME", text: "Alice" }] });
    await screen.findByText("Saved");
    expect(localStorage.getItem("deid_annotation_draft:doc-1")).toBeNull();
  });

  it("shows a conflict and blocks completion after a stale save", async () => {
    mockAuthenticated();
    vi.mocked(api.saveAnnotations).mockRejectedValue(new api.ApiError(409, "This session was updated elsewhere."));

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));
    fireEvent.click(await screen.findByRole("button", { name: "Add test annotation" }));

    expect((await screen.findByRole("alert")).textContent).toContain("Conflict");
    expect((screen.getByRole("button", { name: "Review & complete" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("copies compact local recovery data without overwriting a conflicting server revision", async () => {
    mockAuthenticated();
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    vi.mocked(api.saveAnnotations).mockRejectedValue(
      new api.ApiError(409, "expected revision is stale; current revision is 4"),
    );

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));
    fireEvent.click(await screen.findByRole("button", { name: "Add test annotation" }));

    expect(await screen.findByText("Server data was not overwritten.")).toBeTruthy();
    const conflictUnload = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(conflictUnload);
    expect(conflictUnload.defaultPrevented).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Copy recovery JSON" }));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(JSON.stringify({
      document_id: "doc-1",
      expected_revision: 0,
      current_revision: 4,
      spans: [{ start: 0, end: 5, label: "NAME", text: "Alice" }],
    })));
    expect(await screen.findByText("Recovery JSON copied.")).toBeTruthy();
    expect(api.saveAnnotations).toHaveBeenCalledTimes(1);
  });

  it("reports clipboard denial without claiming recovery data was copied", async () => {
    mockAuthenticated();
    const writeText = vi.fn().mockRejectedValue(new Error("Clipboard permission denied."));
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    vi.mocked(api.saveAnnotations).mockRejectedValue(
      new api.ApiError(409, "expected revision is stale; current revision is 2"),
    );

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));
    fireEvent.click(await screen.findByRole("button", { name: "Add test annotation" }));
    fireEvent.click(await screen.findByRole("button", { name: "Copy recovery JSON" }));

    expect(await screen.findByText("Recovery JSON was not copied. Clipboard permission denied.")).toBeTruthy();
    expect(screen.queryByText("Recovery JSON copied.")).toBeNull();
  });

  it("does not claim a stale recovery payload was copied after local annotations change", async () => {
    mockAuthenticated();
    let resolveCopy: (() => void) | undefined;
    const writeText = vi.fn().mockImplementation(() => new Promise<void>((resolve) => { resolveCopy = resolve; }));
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    vi.mocked(api.saveAnnotations).mockRejectedValue(
      new api.ApiError(409, "expected revision is stale; current revision is 2"),
    );

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));
    fireEvent.click(await screen.findByRole("button", { name: "Add test annotation" }));
    fireEvent.click(await screen.findByRole("button", { name: "Copy recovery JSON" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledOnce());

    fireEvent.click(screen.getByRole("button", { name: "Add another annotation" }));
    await act(async () => { resolveCopy?.(); });

    expect(screen.queryByText("Recovery JSON copied.")).toBeNull();
    expect(screen.getByRole("button", { name: "Copy recovery JSON" })).toBeTruthy();
  });

  it("replays the exact failed save before allowing completion", async () => {
    mockAuthenticated();
    vi.mocked(api.saveAnnotations)
      .mockRejectedValueOnce(new api.ApiError(503, "The save response was lost."))
      .mockResolvedValueOnce({
        revision: 1,
        spans: [{ start: 0, end: 5, label: "NAME", text: "Alice" }],
      });

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));
    fireEvent.click(await screen.findByRole("button", { name: "Add test annotation" }));

    expect((await screen.findByRole("alert")).textContent).toContain("Save failed");
    const completion = screen.getByRole("button", { name: "Review & complete" }) as HTMLButtonElement;
    expect(completion.disabled).toBe(true);

    const firstRequest = vi.mocked(api.saveAnnotations).mock.calls[0];
    fireEvent.click(screen.getByRole("button", { name: "Retry save" }));

    await waitFor(() => expect(api.saveAnnotations).toHaveBeenCalledTimes(2));
    expect(vi.mocked(api.saveAnnotations).mock.calls[1]).toEqual(firstRequest);
    expect(await screen.findByText("Saved")).toBeTruthy();
    expect(completion.disabled).toBe(false);
  });

  it("keeps newer edits queued behind an explicitly retried save", async () => {
    mockAuthenticated();
    const firstSpans = [{ start: 0, end: 5, label: "NAME", text: "Alice" }];
    const latestSpans = [
      ...firstSpans,
      { start: 10, end: 13, label: "NAME", text: "Bob" },
    ];
    vi.mocked(api.saveAnnotations)
      .mockRejectedValueOnce(new api.ApiError(503, "The save response was lost."))
      .mockResolvedValueOnce({ revision: 1, spans: firstSpans })
      .mockResolvedValueOnce({ revision: 2, spans: latestSpans });

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));
    fireEvent.click(await screen.findByRole("button", { name: "Add test annotation" }));
    await screen.findByRole("button", { name: "Retry save" });

    fireEvent.click(screen.getByRole("button", { name: "Add another annotation" }));
    expect(api.saveAnnotations).toHaveBeenCalledTimes(1);
    const failedRequest = vi.mocked(api.saveAnnotations).mock.calls[0]!;

    fireEvent.click(screen.getByRole("button", { name: "Retry save" }));

    await waitFor(() => expect(api.saveAnnotations).toHaveBeenCalledTimes(3));
    expect(vi.mocked(api.saveAnnotations).mock.calls[1]).toEqual(failedRequest);
    expect(vi.mocked(api.saveAnnotations).mock.calls[2]).toEqual([
      "doc-1",
      expect.objectContaining({
        expected_revision: 1,
        mutation_id: expect.not.stringMatching(failedRequest[1].mutation_id),
        spans: latestSpans,
      }),
    ]);
    expect(await screen.findByText("Saved")).toBeTruthy();
  });

  it("requires a review checkpoint before completing an assignment", async () => {
    mockAuthenticated();

    render(<App />);
    fireEvent.click(await screen.findByText("Session 001"));
    fireEvent.click(await screen.findByRole("button", { name: "Review & complete" }));

    expect(api.completeAssignment).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog", { name: "Complete this session?" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Complete session" }));

    await waitFor(() => expect(api.completeAssignment).toHaveBeenCalledWith("assignment-1"));
    expect(await screen.findByText("Completed")).toBeTruthy();
  });

  it("keeps a completed session editable and completed after saving", async () => {
    mockAuthenticated();
    vi.mocked(api.getDocument).mockResolvedValue({
      ...document,
      id: "doc-2",
      external_id: "Session 002",
      assignment: {
        ...document.assignment,
        id: "assignment-2",
        state: "completed",
      },
    });
    vi.mocked(api.saveAnnotations).mockResolvedValue({
      revision: 1,
      spans: [{ start: 0, end: 5, label: "NAME", text: "Alice" }],
    });

    render(<App />);
    fireEvent.click(await screen.findByText("Session 002"));

    expect(await screen.findByLabelText("Manual annotation editor")).toBeTruthy();
    expect(screen.queryByText(/Read-only/)).toBeNull();
    expect(screen.getByRole("button", { name: "Completed" })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Add test annotation" }));

    await waitFor(() => expect(api.saveAnnotations).toHaveBeenCalledWith(
      "doc-2",
      expect.objectContaining({
        expected_revision: 0,
        spans: [{ start: 0, end: 5, label: "NAME", text: "Alice" }],
      }),
    ));
    expect(window.document.querySelector(".state-dot.complete")).toBeTruthy();
  });

  it("lets an admin track progress and assign a complete folder", async () => {
    mockAuthenticated(admin);
    vi.mocked(api.getAdminProgress).mockResolvedValue({
      totals: { unassigned: 1, assigned: 1, in_progress: 2, completed: 4, total: 8 },
      annotators: [{ user_id: annotator.id, display_name: annotator.display_name, email: annotator.email, assigned: 1, in_progress: 2, completed: 4 }],
      folders: [folder],
    });
    vi.mocked(api.getAdminUsers).mockResolvedValue([annotator]);
    vi.mocked(api.assignFolder).mockResolvedValue({ folder_id: folder.id, assignment_ids: ["assignment-1", "assignment-2"] });

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Manage assignments" }));

    const progress = await screen.findByLabelText("Progress overview");
    expect(within(progress).getByText("8")).toBeTruthy();
    expect(within(progress).getByText("4")).toBeTruthy();

    const folderList = screen.getByRole("navigation", { name: "Session folders" });
    expect(within(folderList).getByRole("button", { name: "August intake, 2 sessions" })).toBeTruthy();
    expect(await screen.findByRole("option", { name: `${admin.email} (you)` })).toBeTruthy();

    fireEvent.change(screen.getByLabelText("Folder assignee"), { target: { value: annotator.id } });
    fireEvent.click(screen.getByRole("button", { name: "Assign folder" }));

    await waitFor(() => expect(api.assignFolder).toHaveBeenCalledWith(folder.id, annotator.id));
  });

  it("uses bound emails as account identities and shows the role separately", async () => {
    mockAuthenticated(admin);
    vi.mocked(api.getAdminProgress).mockResolvedValue({
      totals: { unassigned: 0, assigned: 1, in_progress: 0, completed: 0, total: 1 },
      annotators: [],
      folders: [folder],
    });
    vi.mocked(api.getAdminUsers).mockResolvedValue([admin, annotator]);

    render(<App />);
    const accountBlock = (await screen.findByRole("button", { name: "Sign out" })).parentElement;
    expect(accountBlock).not.toBeNull();
    expect(within(accountBlock!).getByText(admin.email)).toBeTruthy();
    expect(within(accountBlock!).getByText("Admin")).toBeTruthy();

    fireEvent.click(await screen.findByRole("button", { name: "Manage assignments" }));

    const accounts = await screen.findByRole("region", { name: "Accounts" });
    expect(within(accounts).getByText(admin.email)).toBeTruthy();
    expect(within(accounts).getByText(annotator.email)).toBeTruthy();
    expect(within(accounts).queryByRole("button", { name: /Edit name/ })).toBeNull();
  });

  it("lets an admin create an annotator while batch operations stay CLI-only", async () => {
    mockAuthenticated(admin);
    vi.mocked(api.getAdminProgress).mockResolvedValue({
      totals: { unassigned: 0, assigned: 1, in_progress: 0, completed: 0, total: 1 },
      annotators: [],
      folders: [folder],
    });
    vi.mocked(api.getAdminUsers).mockResolvedValue([]);
    vi.mocked(api.createAdminUser).mockResolvedValue({
      user: { ...annotator, state: "pending_activation" },
      activation_url: "/activate#token=one-time-token",
      activation_expires_at: "2026-08-29T20:00:00Z",
    });

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Manage assignments" }));
    await screen.findByRole("heading", { name: "Add annotator" });

    fireEvent.change(screen.getByLabelText("Email"), { target: { value: annotator.email } });
    fireEvent.click(screen.getByRole("button", { name: "Create annotator" }));

    await waitFor(() => expect(api.createAdminUser).toHaveBeenCalledWith({
      email: annotator.email,
      role: "annotator",
    }));
    expect(screen.queryByLabelText("Display name")).toBeNull();
    expect((await screen.findByRole("link", { name: "Activation link" })).getAttribute("href"))
      .toBe("/activate#token=one-time-token");
    expect(screen.queryByLabelText("Initial password")).toBeNull();
    expect(screen.queryByRole("heading", { name: "Import sessions" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Export annotations" })).toBeNull();
  });

  it("shows account states and exposes a password-reset link only after success", async () => {
    mockAuthenticated(admin);
    const pending = { ...annotator, id: "user-2", email: "pending@cornell.edu", display_name: "Pat Pending", state: "pending_activation" as const };
    const deactivated = { ...annotator, id: "user-3", email: "former@cornell.edu", display_name: "Dee Activated", state: "deactivated" as const };
    vi.mocked(api.getAdminProgress).mockResolvedValue({
      totals: { unassigned: 0, assigned: 0, in_progress: 0, completed: 0, total: 0 },
      annotators: [],
      folders: [],
    });
    vi.mocked(api.getAdminUsers).mockResolvedValue([admin, annotator, pending, deactivated]);
    vi.mocked(api.resetAdminUserPassword).mockResolvedValue({
      user: { ...annotator, state: "pending_activation" },
      activation_url: "/activate#token=reset-token",
      activation_expires_at: "2026-08-29T20:00:00Z",
    });

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Manage assignments" }));

    const accounts = await screen.findByRole("region", { name: "Accounts" });
    expect(within(accounts).getAllByText("Active")).toHaveLength(2);
    expect(within(accounts).getByText("Pending activation")).toBeTruthy();
    expect(within(accounts).getByText("Deactivated")).toBeTruthy();
    expect(within(accounts).getByText(admin.email)).toBeTruthy();
    expect(screen.queryByRole("link", { name: /password reset link/i })).toBeNull();

    fireEvent.click(within(accounts).getByRole("button", { name: `Reset password for ${annotator.email}` }));

    expect((await screen.findByRole("link", { name: `Password reset link for ${annotator.email}` })).getAttribute("href"))
      .toBe("/activate#token=reset-token");
    expect(api.resetAdminUserPassword).toHaveBeenCalledWith(annotator.id);
  });

  it("requires an explicit unfinished-work choice before deactivating an annotator", async () => {
    mockAuthenticated(admin);
    const grace = { ...annotator, id: "user-2", email: "grace@cornell.edu", display_name: "Grace Annotator" };
    vi.mocked(api.getAdminProgress).mockResolvedValue({
      totals: { unassigned: 0, assigned: 2, in_progress: 0, completed: 0, total: 2 },
      annotators: [],
      folders: [],
    });
    vi.mocked(api.getAdminUsers).mockResolvedValue([annotator, grace]);
    vi.mocked(api.deactivateAdminUser).mockResolvedValue({ ...annotator, state: "deactivated" });

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Manage assignments" }));
    fireEvent.click(await screen.findByRole("button", { name: `Deactivate ${annotator.email}` }));

    const confirm = screen.getByRole("button", { name: "Confirm deactivation" });
    expect((confirm as HTMLButtonElement).disabled).toBe(true);
    expect(api.deactivateAdminUser).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("radio", { name: "Reassign unfinished work" }));
    expect((confirm as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Reassign to"), { target: { value: grace.id } });
    expect((confirm as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(confirm);

    await waitFor(() => expect(api.deactivateAdminUser).toHaveBeenCalledWith(annotator.id, {
      action: "reassign",
      assignee_id: grace.id,
    }));
    expect(await screen.findByText(`${annotator.email} was deactivated.`)).toBeTruthy();
  });

  it("can explicitly unassign unfinished work when deactivating an annotator", async () => {
    mockAuthenticated(admin);
    vi.mocked(api.getAdminProgress).mockResolvedValue({
      totals: { unassigned: 0, assigned: 1, in_progress: 0, completed: 0, total: 1 },
      annotators: [],
      folders: [],
    });
    vi.mocked(api.getAdminUsers).mockResolvedValue([annotator]);
    vi.mocked(api.deactivateAdminUser).mockResolvedValue({ ...annotator, state: "deactivated" });

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Manage assignments" }));
    fireEvent.click(await screen.findByRole("button", { name: `Deactivate ${annotator.email}` }));
    fireEvent.click(screen.getByRole("radio", { name: "Unassign unfinished work" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm deactivation" }));

    await waitFor(() => expect(api.deactivateAdminUser).toHaveBeenCalledWith(annotator.id, {
      action: "unassign",
    }));
  });

  it("reactivates a deactivated annotator and reports account-action failures", async () => {
    mockAuthenticated(admin);
    const deactivated = { ...annotator, state: "deactivated" as const };
    vi.mocked(api.getAdminProgress).mockResolvedValue({
      totals: { unassigned: 0, assigned: 0, in_progress: 0, completed: 0, total: 0 },
      annotators: [],
      folders: [],
    });
    vi.mocked(api.getAdminUsers).mockResolvedValue([deactivated]);
    vi.mocked(api.reactivateAdminUser).mockRejectedValue(new api.ApiError(409, "Account cannot be reactivated."));

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Manage assignments" }));
    fireEvent.click(await screen.findByRole("button", { name: `Reactivate ${annotator.email}` }));

    expect((await screen.findByRole("alert")).textContent).toContain("Account cannot be reactivated.");
    expect(screen.queryByRole("link", { name: /activation|password reset/i })).toBeNull();
  });
});
