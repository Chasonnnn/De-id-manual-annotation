import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import MethodsLabRunForm from "./MethodsLabRunForm";
import type { DocumentSummary, FolderSummary } from "../types";

describe("MethodsLabRunForm", () => {
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
      <MethodsLabRunForm
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
    expect(concurrencyInput.value).toBe("10");

    fireEvent.change(concurrencyInput, { target: { value: "13" } });
    fireEvent.click(screen.getByRole("button", { name: "Run Methods Lab" }));

    expect(await screen.findByText("Concurrency must be 1 to 12")).toBeTruthy();
    expect(onRun).not.toHaveBeenCalled();
  });

  it("defaults methods lab chunk mode to off", async () => {
    render(
      <MethodsLabRunForm
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

  it("defaults methods lab match mode to overlap", async () => {
    render(
      <MethodsLabRunForm
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

  it("defaults the methods lab method bundle to audited", async () => {
    render(
      <MethodsLabRunForm
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

    expect((screen.getByLabelText("Method Bundle") as HTMLSelectElement).value).toBe("audited");
  });

  it("submits selected folder ids separately from explicit doc ids", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);

    const view = render(
      <MethodsLabRunForm
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
    fireEvent.click(scoped.getByRole("button", { name: "Run Methods Lab" }));

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

  it("submits the selected match mode", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);

    render(
      <MethodsLabRunForm
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

    fireEvent.change(screen.getByLabelText("Match"), {
      target: { value: "boundary" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run Methods Lab" }));

    await waitFor(() => {
      expect(onRun).toHaveBeenCalledWith(
        expect.objectContaining({
          runtime: expect.objectContaining({
            match_mode: "boundary",
          }),
        }),
      );
    });
  });

  it("lets the user compare against pre-annotations", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);

    render(
      <MethodsLabRunForm
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

    fireEvent.change(screen.getByLabelText("Reference"), {
      target: { value: "pre" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run Methods Lab" }));

    await waitFor(() => {
      expect(onRun).toHaveBeenCalledWith(
        expect.objectContaining({
          runtime: expect.objectContaining({
            reference_source: "pre",
            fallback_reference_source: "pre",
          }),
        }),
      );
    });
  });

  it("submits the selected methods lab method bundle", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);

    render(
      <MethodsLabRunForm
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

    fireEvent.change(screen.getByLabelText("Method Bundle"), {
      target: { value: "legacy" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run Methods Lab" }));

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

  it("submits the test methods lab method bundle", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);

    render(
      <MethodsLabRunForm
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

    fireEvent.change(screen.getByLabelText("Method Bundle"), {
      target: { value: "test" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run Methods Lab" }));

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

  it("submits the v2+post-process methods lab method bundle", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);

    render(
      <MethodsLabRunForm
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

    fireEvent.change(screen.getByLabelText("Method Bundle"), {
      target: { value: "v2+post-process" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run Methods Lab" }));

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

  it("submits the v2 methods lab method bundle", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);

    render(
      <MethodsLabRunForm
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

    fireEvent.change(screen.getByLabelText("Method Bundle"), {
      target: { value: "v2" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run Methods Lab" }));

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

  it("shows internal document ids in the methods lab document picker", async () => {
    const jsonlDocuments: DocumentSummary[] = [
      {
        id: "476838c9_line0",
        filename: "DeID_GT_UPchieve_math_1000transcripts (2).record-0001.json",
        display_name: "Session 16592",
        status: "reviewed",
      },
    ];

    render(
      <MethodsLabRunForm
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
