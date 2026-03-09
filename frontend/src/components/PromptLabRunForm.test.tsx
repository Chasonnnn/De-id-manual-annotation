import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import PromptLabRunForm from "./PromptLabRunForm";
import type { DocumentSummary, FolderSummary } from "../types";

describe("PromptLabRunForm", () => {
  const documents: DocumentSummary[] = [
    { id: "doc-1", filename: "doc-1.txt", display_name: "doc-1.txt", status: "reviewed" },
  ];
  const folders: FolderSummary[] = [
    {
      id: "folder-1",
      name: "Imported sessions",
      kind: "import",
      parent_folder_id: null,
      merged_doc_id: "doc-1",
      doc_count: 3,
      child_folder_count: 0,
      source_filename: "batch.jsonl",
      source_folder_id: null,
      sample_size: null,
      sample_seed: null,
      created_at: "2026-03-09T12:00:00Z",
    },
  ];

  it("uses the provided concurrency max for the input and validation", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);

    render(
      <PromptLabRunForm
        documents={documents}
        folders={[]}
        selectedDocumentId="doc-1"
        methods={[]}
        onRun={onRun}
        running={false}
        concurrencyMax={12}
      />,
    );

    await waitFor(() => {
      expect((screen.getAllByLabelText("doc-1.txt")[0] as HTMLInputElement).checked).toBe(true);
    });

    const concurrencyInput = screen.getByLabelText("Concurrency") as HTMLInputElement;
    expect(concurrencyInput.max).toBe("12");
    expect(concurrencyInput.value).toBe("10");

    fireEvent.change(concurrencyInput, { target: { value: "13" } });
    fireEvent.click(screen.getByRole("button", { name: "Run Prompt Lab" }));

    expect(await screen.findByText("Concurrency must be 1 to 12")).toBeTruthy();
    expect(onRun).not.toHaveBeenCalled();
  });

  it("defaults prompt lab chunk mode to off", async () => {
    render(
      <PromptLabRunForm
        documents={documents}
        folders={[]}
        selectedDocumentId="doc-1"
        methods={[]}
        onRun={vi.fn()}
        running={false}
        concurrencyMax={12}
      />,
    );

    await waitFor(() => {
      expect((screen.getAllByLabelText("doc-1.txt")[0] as HTMLInputElement).checked).toBe(true);
    });

    expect((screen.getByLabelText("Chunk Mode") as HTMLSelectElement).value).toBe("off");
  });

  it("defaults prompt lab match mode to overlap", async () => {
    render(
      <PromptLabRunForm
        documents={documents}
        folders={[]}
        selectedDocumentId="doc-1"
        methods={[]}
        onRun={vi.fn()}
        running={false}
        concurrencyMax={12}
      />,
    );

    await waitFor(() => {
      expect((screen.getAllByLabelText("doc-1.txt")[0] as HTMLInputElement).checked).toBe(true);
    });

    expect((screen.getByLabelText("Match") as HTMLSelectElement).value).toBe("overlap");
  });

  it("submits selected folder ids separately from explicit doc ids", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);

    const view = render(
      <PromptLabRunForm
        documents={documents}
        folders={folders}
        selectedDocumentId={null}
        methods={[]}
        onRun={onRun}
        running={false}
        concurrencyMax={12}
      />,
    );
    const scoped = within(view.container);

    fireEvent.click(scoped.getByLabelText("Imported sessions (3 docs)"));
    fireEvent.click(scoped.getByRole("button", { name: "Run Prompt Lab" }));

    await waitFor(() => {
      expect(onRun).toHaveBeenCalledWith(
        expect.objectContaining({
          doc_ids: [],
          folder_ids: ["folder-1"],
        }),
      );
    });
    expect(scoped.getByText(/Requests:/).textContent).toContain("3");
  });
});
