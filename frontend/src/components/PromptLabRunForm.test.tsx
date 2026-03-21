import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import PromptLabRunForm from "./PromptLabRunForm";
import type { DocumentSummary, FolderSummary } from "../types";

describe("PromptLabRunForm", () => {
  afterEach(() => {
    cleanup();
  });

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
      expect((screen.getAllByLabelText("doc-1")[0] as HTMLInputElement).checked).toBe(true);
    });

    const concurrencyInput = screen.getByLabelText("Concurrency") as HTMLInputElement;
    expect(concurrencyInput.max).toBe("12");
    expect(concurrencyInput.value).toBe("12");

    fireEvent.change(concurrencyInput, { target: { value: "13" } });
    fireEvent.click(screen.getByRole("button", { name: "Run Prompt Lab" }));

    expect(await screen.findByText("Concurrency must be 1 to 12")).toBeTruthy();
    expect(onRun).not.toHaveBeenCalled();
  });

  it("does not show API key or base URL overrides", async () => {
    render(
      <PromptLabRunForm
        documents={documents}
        folders={[]}
        selectedDocumentId="doc-1"
        methods={[]}
        onRun={vi.fn()}
        running={false}
        concurrencyMax={16}
      />,
    );

    await waitFor(() => {
      expect((screen.getAllByLabelText("doc-1")[0] as HTMLInputElement).checked).toBe(true);
    });

    expect(screen.queryByLabelText("API Key (optional override)")).toBeNull();
    expect(screen.queryByLabelText("LiteLLM Base URL (optional override)")).toBeNull();
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
      expect((screen.getAllByLabelText("doc-1")[0] as HTMLInputElement).checked).toBe(true);
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
      expect((screen.getAllByLabelText("doc-1")[0] as HTMLInputElement).checked).toBe(true);
    });

    expect((screen.getByLabelText("Match") as HTMLSelectElement).value).toBe("overlap");
  });

  it("defaults the prompt lab preset method bundle to audited", async () => {
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
      expect((screen.getAllByLabelText("doc-1")[0] as HTMLInputElement).checked).toBe(true);
    });

    expect((screen.getByLabelText("Preset Bundle") as HTMLSelectElement).value).toBe("audited");
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

  it("submits the selected prompt lab preset method bundle", async () => {
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
      expect((screen.getAllByLabelText("doc-1")[0] as HTMLInputElement).checked).toBe(true);
    });

    fireEvent.change(screen.getByLabelText("Preset Bundle"), {
      target: { value: "legacy" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run Prompt Lab" }));

    await waitFor(() => {
      expect(onRun).toHaveBeenCalledWith(
        expect.objectContaining({
          runtime: expect.objectContaining({
            method_bundle: "legacy",
          }),
        }),
      );
    });
  });

  it("submits the test prompt lab preset method bundle", async () => {
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
      expect((screen.getAllByLabelText("doc-1")[0] as HTMLInputElement).checked).toBe(true);
    });

    fireEvent.change(screen.getByLabelText("Preset Bundle"), {
      target: { value: "test" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run Prompt Lab" }));

    await waitFor(() => {
      expect(onRun).toHaveBeenCalledWith(
        expect.objectContaining({
          runtime: expect.objectContaining({
            method_bundle: "test",
          }),
        }),
      );
    });
  });

  it("submits the v2+post-process prompt lab preset method bundle", async () => {
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
      expect((screen.getAllByLabelText("doc-1")[0] as HTMLInputElement).checked).toBe(true);
    });

    fireEvent.change(screen.getByLabelText("Preset Bundle"), {
      target: { value: "v2+post-process" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run Prompt Lab" }));

    await waitFor(() => {
      expect(onRun).toHaveBeenCalledWith(
        expect.objectContaining({
          runtime: expect.objectContaining({
            method_bundle: "v2+post-process",
          }),
        }),
      );
    });
  });

  it("submits the v2 prompt lab preset method bundle", async () => {
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
      expect((screen.getAllByLabelText("doc-1")[0] as HTMLInputElement).checked).toBe(true);
    });

    fireEvent.change(screen.getByLabelText("Preset Bundle"), {
      target: { value: "v2" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run Prompt Lab" }));

    await waitFor(() => {
      expect(onRun).toHaveBeenCalledWith(
        expect.objectContaining({
          runtime: expect.objectContaining({
            method_bundle: "v2",
          }),
        }),
      );
    });
  });

  it("shows internal document ids in the prompt lab document picker", async () => {
    const jsonlDocuments: DocumentSummary[] = [
      {
        id: "476838c9_line0",
        filename: "DeID_GT_UPchieve_math_1000transcripts (2).record-0001.json",
        display_name: "Session 16592",
        status: "reviewed",
      },
    ];

    render(
      <PromptLabRunForm
        documents={jsonlDocuments}
        folders={[]}
        selectedDocumentId="476838c9_line0"
        methods={[]}
        onRun={vi.fn()}
        running={false}
        concurrencyMax={12}
      />,
    );

    await waitFor(() => {
      expect(
        (screen.getAllByLabelText("476838c9_line0")[0] as HTMLInputElement).checked,
      ).toBe(true);
    });
    expect(screen.queryByText("Session 16592")).toBeNull();
  });
});
