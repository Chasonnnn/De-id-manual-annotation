import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import Sidebar from "./Sidebar";
import type { DocumentSummary, FolderDetail, FolderSummary, SessionProfile } from "../types";

describe("Sidebar", () => {
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
    child_folder_count: 1,
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
    parent_folder_id: "folder-import",
    merged_doc_id: null,
    doc_count: 1,
    child_folder_count: 0,
    source_filename: null,
    source_folder_id: "folder-import",
    sample_size: 1,
    sample_seed: 7,
    created_at: "2026-03-09T12:01:00Z",
  };
  const folders: FolderSummary[] = [importFolder, sampleFolder];

  const folderDetailsById: Record<string, FolderDetail> = {
    "folder-import": {
      ...importFolder,
      doc_ids: ["doc-child-1", "doc-child-2"],
      child_folder_ids: ["folder-sample"],
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
      child_folders: [sampleFolder],
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
  };

  const sessionProfile: SessionProfile = {
    project_name: "",
    author: "",
  };

  it("renders top-level documents separately from folder children and lets child docs be selected", async () => {
    const onSelect = vi.fn();

    render(
      <Sidebar
        documents={documents}
        folders={folders}
        folderDetailsById={folderDetailsById}
        selectedId={null}
        onSelect={onSelect}
        onUpload={vi.fn()}
        onDelete={vi.fn()}
        onCreateFolderSample={vi.fn()}
        onDeleteFolder={vi.fn()}
        onExportSession={vi.fn()}
        onImportSession={vi.fn()}
        exportSourceOptions={[{ value: "manual", label: "Manual annotations" }]}
        sessionProfile={sessionProfile}
        onSessionProfileChange={vi.fn()}
        onSaveSessionProfile={vi.fn()}
      />,
    );

    expect(screen.getByText("Top-Level Documents")).toBeTruthy();
    expect(screen.getByText("batch.jsonl")).toBeTruthy();
    expect(screen.getByText("Folders")).toBeTruthy();
    expect(screen.getByText("batch")).toBeTruthy();
    expect(await screen.findByText("Session alpha")).toBeTruthy();
    expect(screen.getByText("Session beta")).toBeTruthy();
    expect(screen.getByText("Sample 1")).toBeTruthy();

    fireEvent.click(screen.getByText("Session alpha"));

    expect(onSelect).toHaveBeenCalledWith("doc-child-1");
  });
});
