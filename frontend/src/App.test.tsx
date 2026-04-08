import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import {
  deleteFolderDocument,
  getAgentCredentialStatus,
  getAgentMethods,
  getDocument,
  getFolder,
  getMetricsDashboard,
  listFolders,
  listDocuments,
} from "./api/client";

vi.mock("./api/client", () => ({
  createFolder: vi.fn(),
  createFolderSample: vi.fn(),
  deleteFolder: vi.fn(),
  deleteFolderDocument: vi.fn(),
  deleteDocument: vi.fn(),
  exportGroundTruth: vi.fn(),
  exportSession: vi.fn(),
  getAgentCredentialStatus: vi.fn(),
  getFolder: vi.fn(),
  getAgentProgress: vi.fn(),
  getAgentMethods: vi.fn().mockResolvedValue([]),
  getDocument: vi.fn(),
  getMetricsDashboard: vi.fn(),
  getMetrics: vi.fn(),
  ingestSessionFile: vi.fn(),
  listFolders: vi.fn().mockResolvedValue([]),
  listDocuments: vi.fn().mockResolvedValue([]),
  pruneEmptyFolderDocs: vi.fn(),
  runAgent: vi.fn(),
  updateManualAnnotations: vi.fn(),
}));

describe("App", () => {
  const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});

  const methodFixture = {
    id: "doc-method",
    filename: "doc-method.json",
    raw_text: "Tutor: Hello Chloe.",
    utterances: [],
    pre_annotations: [],
    label_set: ["NAME"],
    manual_annotations: [],
    agent_annotations: [],
    agent_outputs: {
      rule: [],
      llm: [],
      llm_runs: {},
      llm_run_metadata: {},
      methods: {
        "presidio-lite+extended-v2::anthropic.claude-4.6-sonnet": [
          { start: 13, end: 18, label: "NAME", text: "Chloe" },
        ],
        "presidio-lite+extended-v2::google.gemini-3.1-pro-preview": [
          { start: 13, end: 18, label: "NAME", text: "Chloe" },
        ],
      },
      method_run_metadata: {
        "presidio-lite+extended-v2::anthropic.claude-4.6-sonnet": {
          mode: "method",
          updated_at: "2026-03-17T18:45:00Z",
          method_id: "presidio-lite+extended-v2",
          model: "anthropic.claude-4.6-sonnet",
        },
        "presidio-lite+extended-v2::google.gemini-3.1-pro-preview": {
          mode: "method",
          updated_at: "2026-03-17T18:57:00Z",
          method_id: "presidio-lite+extended-v2",
          model: "google.gemini-3.1-pro-preview",
        },
      },
    },
    agent_run_warnings: [],
    agent_run_metrics: {
      llm_confidence: null,
      chunk_diagnostics: [],
    },
    status: "pending",
  };

  const agentFixture = {
    id: "doc-agent",
    filename: "doc-agent.json",
    raw_text: "Tutor: Hi Chloe.",
    utterances: [],
    pre_annotations: [],
    label_set: ["NAME"],
    manual_annotations: [],
    agent_annotations: [],
    agent_outputs: {
      rule: [],
      llm: [],
      llm_runs: {
        older_run: [{ start: 10, end: 15, label: "NAME", text: "Chloe" }],
        newer_run: [{ start: 10, end: 15, label: "NAME", text: "Chloe" }],
      },
      llm_run_metadata: {
        older_run: {
          mode: "llm",
          updated_at: "2026-03-17T18:45:00Z",
          model: "anthropic.claude-4.6-sonnet",
        },
        newer_run: {
          mode: "llm",
          updated_at: "2026-03-17T18:57:00Z",
          model: "google.gemini-3.1-pro-preview",
        },
      },
      methods: {},
      method_run_metadata: {},
    },
    agent_run_warnings: [],
    agent_run_metrics: {
      llm_confidence: null,
      chunk_diagnostics: [],
    },
    status: "pending",
  };

  beforeEach(() => {
    consoleError.mockClear();
    vi.mocked(listDocuments).mockResolvedValue([]);
    vi.mocked(getAgentCredentialStatus).mockResolvedValue({
      has_api_key: false,
      api_key_sources: [],
      has_api_base: false,
      api_base_sources: [],
    });
    vi.mocked(getMetricsDashboard).mockResolvedValue(null as never);
    vi.mocked(getDocument).mockRejectedValue(new Error("not configured"));
    vi.mocked(getAgentMethods).mockResolvedValue([
      {
        id: "default",
        label: "Default",
        description: "Default method",
        requires_presidio: false,
        uses_llm: true,
        supports_verify_override: false,
        prompt_templates: [],
        available: true,
        unavailable_reason: null,
      },
    ]);
  });

  afterEach(() => {
    cleanup();
  });

  it("renders the workspace without hitting the error boundary", async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Workspace")).toBeTruthy();
    });

    expect(screen.queryByText("Something went wrong")).toBeNull();
    expect(screen.getByText("Select a document or upload a file to begin")).toBeTruthy();
  });

  it("defaults the method pane to the latest saved method output when one exists", async () => {
    vi.mocked(listDocuments).mockResolvedValue([
      {
        id: methodFixture.id,
        filename: methodFixture.filename,
        display_name: methodFixture.id,
        status: "pending",
      },
    ]);
    vi.mocked(getDocument).mockResolvedValue(methodFixture as never);

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(methodFixture.id)).toBeTruthy();
    });

    fireEvent.click(screen.getByText(methodFixture.id));

    await waitFor(() => {
      expect(getDocument).toHaveBeenCalledWith(methodFixture.id);
    });

    fireEvent.click(screen.getByRole("button", { name: "+Methods" }));

    await waitFor(() => {
      expect(screen.getByLabelText("View:")).toBeTruthy();
    });

    const methodView = screen.getByLabelText("View:") as HTMLSelectElement;
    expect(methodView.value).toBe("presidio-lite+extended-v2::google.gemini-3.1-pro-preview");
    expect(screen.queryByText("No method annotations yet for the selected method.")).toBeNull();
  });

  it("defaults the agent pane to the latest saved llm run when one exists", async () => {
    vi.mocked(listDocuments).mockResolvedValue([
      {
        id: agentFixture.id,
        filename: agentFixture.filename,
        display_name: agentFixture.id,
        status: "pending",
      },
    ]);
    vi.mocked(getDocument).mockResolvedValue(agentFixture as never);

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(agentFixture.id)).toBeTruthy();
    });

    fireEvent.click(screen.getByText(agentFixture.id));

    await waitFor(() => {
      expect(getDocument).toHaveBeenCalledWith(agentFixture.id);
    });

    fireEvent.click(screen.getByRole("button", { name: "+Agent" }));

    await waitFor(() => {
      expect(screen.getByLabelText("Run:")).toBeTruthy();
    });

    const agentView = screen.getByLabelText("View:") as HTMLSelectElement;
    const agentRun = screen.getByLabelText("Run:") as HTMLSelectElement;
    expect(agentView.value).toBe("llm");
    expect(agentRun.value).toBe("newer_run");
  });

  it("deletes a folder transcript through the dedicated folder-doc flow", async () => {
    vi.mocked(listDocuments).mockResolvedValue([]);
    vi.mocked(listFolders).mockResolvedValue([
      {
        id: "folder-1",
        name: "Folder 1",
        kind: "manual",
        parent_folder_id: null,
        merged_doc_id: null,
        doc_count: 1,
        child_folder_count: 0,
        source_filename: null,
        source_folder_id: null,
        sample_size: null,
        sample_seed: null,
        created_at: "2026-04-08T00:00:00Z",
      },
    ]);
    vi.mocked(getFolder).mockResolvedValue({
      id: "folder-1",
      name: "Folder 1",
      kind: "manual",
      parent_folder_id: null,
      merged_doc_id: null,
      doc_count: 1,
      child_folder_count: 0,
      source_filename: null,
      source_folder_id: null,
      sample_size: null,
      sample_seed: null,
      created_at: "2026-04-08T00:00:00Z",
      doc_ids: ["child-1"],
      child_folder_ids: [],
      documents: [
        {
          id: "child-1",
          filename: "child-1.json",
          display_name: "child-1",
          status: "pending",
        },
      ],
      child_folders: [],
    } as never);
    vi.mocked(deleteFolderDocument).mockResolvedValue({
      deleted: true,
      folder_id: "folder-1",
      doc_id: "child-1",
      updated_folder_ids: ["folder-1"],
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Folder 1")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "Expand folder" }));

    await waitFor(() => {
      expect(screen.getByText("child-1")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "Delete transcript child-1" }));

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "Delete Transcript" }));

    await waitFor(() => {
      expect(deleteFolderDocument).toHaveBeenCalledWith("folder-1", "child-1");
    });
  });
});
