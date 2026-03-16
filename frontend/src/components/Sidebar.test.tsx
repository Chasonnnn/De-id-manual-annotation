import type { ComponentProps } from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import Sidebar from "./Sidebar";
import type { DocumentSummary, FolderDetail, FolderSummary } from "../types";

describe("Sidebar", () => {
  afterEach(() => {
    cleanup();
  });

  const documents: DocumentSummary[] = [
    {
      id: "doc-merged",
      filename: "batch.jsonl",
      display_name: "batch.jsonl",
      status: "pending",
    },
  ];
  const importFolder: FolderSummary = {
    id: "folder-import",
    name: "batch",
    kind: "import",
    parent_folder_id: null,
    merged_doc_id: "doc-merged",
    doc_count: 2,
    child_folder_count: 0,
    source_filename: "batch.jsonl",
    source_folder_id: null,
    sample_size: null,
    sample_seed: null,
    created_at: "2026-03-09T12:00:00Z",
  };
  const sampleFolder: FolderSummary = {
    id: "folder-sample",
    name: "Sample 1",
    kind: "sample",
    parent_folder_id: null,
    merged_doc_id: null,
    doc_count: 1,
    child_folder_count: 0,
    source_filename: null,
    source_folder_id: "folder-import",
    sample_size: 1,
    sample_seed: 7,
    created_at: "2026-03-09T12:01:00Z",
  };
  const manualFolder: FolderSummary = {
    id: "folder-manual",
    name: "Manual Set",
    kind: "manual",
    parent_folder_id: null,
    merged_doc_id: null,
    doc_count: 0,
    child_folder_count: 0,
    source_filename: null,
    source_folder_id: null,
    sample_size: null,
    sample_seed: null,
    created_at: "2026-03-09T12:02:00Z",
  };
  const folders: FolderSummary[] = [importFolder, sampleFolder, manualFolder];

  const folderDetailsById: Record<string, FolderDetail> = {
    "folder-import": {
      ...importFolder,
      doc_ids: ["doc-child-1", "doc-child-2"],
      child_folder_ids: [],
      documents: [
        {
          id: "doc-child-1",
          filename: "batch_record_0001.json",
          display_name: "Session alpha",
          status: "pending",
        },
        {
          id: "doc-child-2",
          filename: "batch_record_0002.json",
          display_name: "Session beta",
          status: "reviewed",
        },
      ],
      child_folders: [],
    },
    "folder-sample": {
      ...sampleFolder,
      doc_ids: ["doc-child-2"],
      child_folder_ids: [],
      documents: [
        {
          id: "doc-child-2",
          filename: "batch_record_0002.json",
          display_name: "Session beta",
          status: "reviewed",
        },
      ],
      child_folders: [],
    },
    "folder-manual": {
      ...manualFolder,
      doc_ids: [],
      child_folder_ids: [],
      documents: [],
      child_folders: [],
    },
  };

  function renderSidebar(overrides?: Partial<ComponentProps<typeof Sidebar>>) {
    return render(
      <Sidebar
        documents={documents}
        folders={folders}
        folderDetailsById={folderDetailsById}
        selectedId={null}
        onSelect={vi.fn()}
        onIngestFiles={vi.fn()}
        onDelete={vi.fn()}
        onCreateFolder={vi.fn()}
        onCreateFolderSample={vi.fn()}
        onDeleteFolder={vi.fn()}
        onPruneFolder={vi.fn()}
        onExportSession={vi.fn()}
        exportSourceOptions={[{ value: "manual", label: "Manual annotations" }]}
        {...overrides}
      />,
    );
  }

  it("renders top-level documents separately from folder children and lets child docs be selected", async () => {
    const onSelect = vi.fn();
    renderSidebar({ onSelect });

    expect(
      screen.getByText("Drop transcript/session bundle/ground truth here or click to add"),
    ).toBeTruthy();
    expect(screen.getByText("Top-Level Documents")).toBeTruthy();
    expect(screen.getByText("batch.jsonl")).toBeTruthy();
    expect(screen.getByText("Folders")).toBeTruthy();
    expect(screen.getByText("batch")).toBeTruthy();
    expect(screen.getByText("Manual Set")).toBeTruthy();
    expect(screen.getByText("Sample 1")).toBeTruthy();
    expect(screen.queryByText("Session alpha")).toBeNull();
    expect(screen.queryByText("Session beta")).toBeNull();

    const expandButtons = screen.getAllByRole("button", { name: "Expand folder" });
    fireEvent.click(expandButtons[0]!);

    expect(await screen.findByText("Session alpha")).toBeTruthy();
    expect(screen.getByText("Session beta")).toBeTruthy();

    fireEvent.click(screen.getByText("Session alpha"));

    expect(onSelect).toHaveBeenCalledWith("doc-child-1");
  });

  it("creates top-level folders and nested subfolders from prompt actions", () => {
    const onCreateFolder = vi.fn();

    renderSidebar({ onCreateFolder });

    fireEvent.click(screen.getByRole("button", { name: "New Folder" }));
    fireEvent.change(screen.getByRole("textbox", { name: "New Folder" }), {
      target: { value: "Research Batch" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create" }));

    const subfolderButtons = screen.getAllByRole("button", { name: "New" });
    fireEvent.click(subfolderButtons[0]!);
    fireEvent.change(screen.getByRole("textbox", { name: "New Subfolder" }), {
      target: { value: "Reviewed" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create" }));

    expect(onCreateFolder).toHaveBeenNthCalledWith(1, "Research Batch", null);
    expect(onCreateFolder).toHaveBeenNthCalledWith(2, "Reviewed", "folder-import");
  });

  it("triggers the folder prune action", () => {
    const onPruneFolder = vi.fn();
    renderSidebar({ onPruneFolder });

    fireEvent.click(screen.getAllByRole("button", { name: "Prune Empty" })[0]!);

    expect(onPruneFolder).toHaveBeenCalledWith("folder-import");
  });

  it("passes the selected import conflict policy to ingest", () => {
    const onIngestFiles = vi.fn();
    const view = renderSidebar({ onIngestFiles });
    const file = new File(["{}"], "bundle.json", { type: "application/json" });

    fireEvent.change(screen.getByLabelText("Import Conflicts"), {
      target: { value: "keep_current" },
    });
    fireEvent.change(view.container.querySelector("#sidebar-ingest-file")!, {
      target: { files: [file] },
    });

    expect(onIngestFiles).toHaveBeenCalledWith([file], "keep_current");
  });

  it("accepts txt transcripts in the ingest picker", () => {
    const view = renderSidebar();
    const input = view.container.querySelector("#sidebar-ingest-file");

    expect(input?.getAttribute("accept")).toBe(".json,.jsonl,.txt,.zip");
  });

  it("creates a sample using the prompt dialog", () => {
    const onCreateFolderSample = vi.fn();
    renderSidebar({ onCreateFolderSample });

    fireEvent.click(screen.getAllByRole("button", { name: "Sample" })[0]!);
    fireEvent.change(screen.getByRole("textbox", { name: "Create Sample" }), {
      target: { value: "3" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create Sample" }));

    expect(onCreateFolderSample).toHaveBeenCalledWith("folder-import", 3);
  });
});
