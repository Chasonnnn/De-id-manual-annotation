import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import MethodsLabTab from "./MethodsLabTab";
import type {
  DocumentSummary,
  MethodsLabDocResult,
  MethodsLabRunDetail,
  MethodsLabRunSummary,
} from "../types";

const clientMocks = vi.hoisted(() => ({
  createMethodsLabRun: vi.fn(),
  deleteMethodsLabRun: vi.fn(),
  getAgentMethods: vi.fn(),
  getExperimentLimits: vi.fn(),
  getMethodsLabDocResult: vi.fn(),
  getMethodsLabRun: vi.fn(),
  listMethodsLabRuns: vi.fn(),
  stopMethodsLabRun: vi.fn(),
}));

vi.mock("../api/client", () => clientMocks);

vi.mock("./MethodsLabRunForm", () => ({
  default: ({ concurrencyMax }: { concurrencyMax: number }) => (
    <div data-testid="methods-lab-run-form">{`run form max=${concurrencyMax}`}</div>
  ),
}));

vi.mock("./MethodsLabMatrix", () => ({
  default: ({ selectedCellId }: { selectedCellId: string | null }) => (
    <div data-testid="methods-lab-matrix">{selectedCellId ?? "no-cell"}</div>
  ),
}));

vi.mock("./MethodsLabCellDetail", () => ({
  default: ({
    detail,
    loading,
  }: {
    detail: MethodsLabDocResult | null;
    loading: boolean;
  }) => (
    <div data-testid="methods-lab-cell-detail">
      {loading ? "loading" : detail ? `${detail.run_id}:${detail.cell_id}:${detail.doc_id}` : "no-detail"}
    </div>
  ),
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function cloneValue<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function makeRunSummary(id: string, name: string): MethodsLabRunSummary {
  return {
    id,
    name,
    status: "completed",
    cancellable: false,
    created_at: "2026-03-08T00:00:00Z",
    started_at: "2026-03-08T00:00:00Z",
    finished_at: "2026-03-08T00:01:00Z",
    doc_count: 1,
    method_count: 1,
    model_count: 1,
    total_tasks: 1,
    completed_tasks: 1,
    failed_tasks: 0,
  };
}

function makeRunDetail(
  summary: MethodsLabRunSummary,
  {
    docId,
    cellId,
    modelId,
    modelLabel,
    methodId,
    methodLabel,
  }: {
    docId: string;
    cellId: string;
    modelId: string;
    modelLabel: string;
    methodId: string;
    methodLabel: string;
  },
): MethodsLabRunDetail {
  return {
    ...summary,
    doc_ids: [docId],
    methods: [
      {
        id: methodId,
        label: methodLabel,
        method_id: methodId,
        method_verify_override: null,
      },
    ],
    models: [
      {
        id: modelId,
        label: modelLabel,
        model: modelLabel,
        reasoning_effort: "none",
        anthropic_thinking: false,
        anthropic_thinking_budget_tokens: null,
      },
    ],
    runtime: {
      temperature: 0,
      match_mode: "exact",
      label_profile: "simple",
      label_projection: "native",
      chunk_mode: "auto",
      chunk_size_chars: 10000,
    },
    concurrency: 1,
    warnings: [],
    errors: [],
    matrix: {
      models: [{ id: modelId, label: modelLabel }],
      methods: [{ id: methodId, label: methodLabel }],
      cells: [
        {
          id: cellId,
          model_id: modelId,
          model_label: modelLabel,
          method_id: methodId,
          method_label: methodLabel,
          status: "completed",
          total_docs: 1,
          completed_docs: 1,
          failed_docs: 0,
          error_count: 0,
          micro: { precision: 1, recall: 1, f1: 1, tp: 1, fp: 0, fn: 0 },
          per_label: {},
          mean_confidence: null,
        },
      ],
      available_labels: [],
    },
    progress: {
      total_tasks: 1,
      completed_tasks: 1,
      failed_tasks: 0,
    },
  };
}

function makeDocResult(
  run: MethodsLabRunDetail,
  {
    cellId,
    docId,
  }: {
    cellId: string;
    docId: string;
  },
): MethodsLabDocResult {
  return {
    run_id: run.id,
    cell_id: cellId,
    doc_id: docId,
    status: "completed",
    error: null,
    warnings: [],
    reference_source_used: "manual",
    reference_spans: [],
    hypothesis_spans: [],
    metrics: null,
    llm_confidence: null,
    transcript_text: "Example transcript",
    document: { id: docId, filename: `${docId}.txt` },
    model: run.models[0],
    method: run.methods[0],
  };
}

describe("MethodsLabTab", () => {
  const documents: DocumentSummary[] = [
    { id: "doc-1", filename: "doc-1.txt", status: "reviewed" },
  ];

  const codexSummary = makeRunSummary("5bcf6ed4", "methods_lab__codex");
  const geminiSummary = makeRunSummary("run-gemini", "methods_lab__gemini");
  const codexDetail = makeRunDetail(codexSummary, {
    docId: "doc-1",
    cellId: "model_1__method_1",
    modelId: "model_1",
    modelLabel: "Codex",
    methodId: "method_1",
    methodLabel: "Default",
  });
  const geminiDetail = makeRunDetail(geminiSummary, {
    docId: "doc-1",
    cellId: "gemini_pro__method_1",
    modelId: "gemini_pro",
    modelLabel: "Gemini Pro",
    methodId: "method_1",
    methodLabel: "Default",
  });

  beforeEach(() => {
    clientMocks.createMethodsLabRun.mockReset();
    clientMocks.deleteMethodsLabRun.mockReset();
    clientMocks.getAgentMethods.mockReset();
    clientMocks.getExperimentLimits.mockReset();
    clientMocks.getMethodsLabDocResult.mockReset();
    clientMocks.getMethodsLabRun.mockReset();
    clientMocks.listMethodsLabRuns.mockReset();
    clientMocks.stopMethodsLabRun.mockReset();
    clientMocks.getAgentMethods.mockResolvedValue([]);
    clientMocks.getExperimentLimits.mockResolvedValue({
      prompt_lab_default_concurrency: 10,
      prompt_lab_max_concurrency: 16,
      methods_lab_default_concurrency: 10,
      methods_lab_max_concurrency: 16,
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("switches runs without requesting a stale method cell id", async () => {
    clientMocks.listMethodsLabRuns.mockResolvedValue([codexSummary, geminiSummary]);
    clientMocks.getMethodsLabRun.mockImplementation(async (runId: string) => {
      if (runId === codexSummary.id) return codexDetail;
      if (runId === geminiSummary.id) return geminiDetail;
      throw new Error(`Unknown run ${runId}`);
    });
    clientMocks.getMethodsLabDocResult.mockImplementation(
      async (runId: string, cellId: string, docId: string) => {
        if (runId === codexDetail.id && cellId === codexDetail.matrix.cells[0]?.id && docId === "doc-1") {
          return makeDocResult(codexDetail, { cellId, docId });
        }
        if (runId === geminiDetail.id && cellId === geminiDetail.matrix.cells[0]?.id && docId === "doc-1") {
          return makeDocResult(geminiDetail, { cellId, docId });
        }
        throw new Error("404: Methods Lab cell not found");
      },
    );

    render(
      <MethodsLabTab
        documents={documents}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    await screen.findByText("5bcf6ed4:model_1__method_1:doc-1");

    fireEvent.click(screen.getByRole("button", { name: /methods_lab__gemini/i }));

    await screen.findByText("run-gemini:gemini_pro__method_1:doc-1");

    expect(screen.queryByText(/404: Methods Lab cell not found/i)).toBeNull();
    expect(clientMocks.getMethodsLabDocResult).not.toHaveBeenCalledWith(
      "run-gemini",
      "model_1__method_1",
      "doc-1",
    );
  });

  it("loads experiment limits and passes the methods concurrency max to the form", async () => {
    clientMocks.listMethodsLabRuns.mockResolvedValue([]);
    clientMocks.getExperimentLimits.mockResolvedValue({
      prompt_lab_default_concurrency: 10,
      prompt_lab_max_concurrency: 16,
      methods_lab_default_concurrency: 10,
      methods_lab_max_concurrency: 12,
    });

    render(
      <MethodsLabTab
        documents={documents}
        selectedDocumentId="doc-1"
        onSelectDocument={() => {}}
      />,
    );

    expect(await screen.findByText("run form max=12")).toBeTruthy();
    expect(clientMocks.getExperimentLimits).toHaveBeenCalledTimes(1);
  });

  it("clears stale detail before loading the next run after deleting the selected run", async () => {
    const geminiDetailDeferred = deferred<MethodsLabRunDetail>();
    let currentRuns = [codexSummary, geminiSummary];
    clientMocks.listMethodsLabRuns.mockImplementation(async () => currentRuns);
    clientMocks.getMethodsLabRun.mockImplementation((runId: string) => {
      if (runId === codexSummary.id) return Promise.resolve(codexDetail);
      if (runId === geminiSummary.id) return geminiDetailDeferred.promise;
      throw new Error(`Unknown run ${runId}`);
    });
    clientMocks.getMethodsLabDocResult.mockImplementation(
      async (runId: string, cellId: string, docId: string) => {
        if (runId === codexDetail.id && cellId === codexDetail.matrix.cells[0]?.id) {
          return makeDocResult(codexDetail, { cellId, docId });
        }
        if (runId === geminiDetail.id && cellId === geminiDetail.matrix.cells[0]?.id) {
          return makeDocResult(geminiDetail, { cellId, docId });
        }
        throw new Error("404: Methods Lab cell not found");
      },
    );
    clientMocks.deleteMethodsLabRun.mockImplementation(async () => {
      currentRuns = [geminiSummary];
      return { ok: true, id: codexSummary.id };
    });
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(
      <MethodsLabTab
        documents={documents}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    await screen.findByText("5bcf6ed4:model_1__method_1:doc-1");

    fireEvent.click(screen.getAllByRole("button", { name: "Delete" })[0]!);

    await screen.findByText("Create or select a Methods Lab run.");

    geminiDetailDeferred.resolve(geminiDetail);

    await screen.findByText("run-gemini:gemini_pro__method_1:doc-1");

    expect(screen.queryByText(/404: Methods Lab cell not found/i)).toBeNull();
    await waitFor(() => {
      expect(clientMocks.getMethodsLabDocResult).toHaveBeenCalledWith(
        "run-gemini",
        "gemini_pro__method_1",
        "doc-1",
      );
    });
  });

  it("stops an active run and refreshes its status", async () => {
    const runningSummary: MethodsLabRunSummary = {
      ...codexSummary,
      status: "running",
      cancellable: true,
      finished_at: null,
      total_tasks: 2,
      completed_tasks: 1,
    };
    const cancellingDetail: MethodsLabRunDetail = {
      ...codexDetail,
      ...runningSummary,
      status: "cancelling",
    };

    clientMocks.listMethodsLabRuns
      .mockResolvedValueOnce([runningSummary])
      .mockResolvedValueOnce([{ ...runningSummary, status: "cancelling" }]);
    clientMocks.getMethodsLabRun
      .mockResolvedValueOnce({ ...codexDetail, ...runningSummary })
      .mockResolvedValueOnce(cancellingDetail);
    clientMocks.getMethodsLabDocResult.mockResolvedValue(
      makeDocResult(codexDetail, { cellId: codexDetail.matrix.cells[0]!.id, docId: "doc-1" }),
    );
    clientMocks.stopMethodsLabRun.mockResolvedValue({
      ok: true,
      id: runningSummary.id,
      status: "cancelling",
    });
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(
      <MethodsLabTab
        documents={documents}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    await screen.findByText("5bcf6ed4:model_1__method_1:doc-1");

    fireEvent.click(screen.getByRole("button", { name: "Stop" }));

    await waitFor(() => {
      expect(clientMocks.stopMethodsLabRun).toHaveBeenCalledWith("5bcf6ed4");
    });
    await waitFor(() => {
      expect(clientMocks.listMethodsLabRuns).toHaveBeenCalledTimes(2);
      expect(clientMocks.getMethodsLabRun).toHaveBeenCalledTimes(2);
    });
  });

  it("keeps the current doc detail visible while polling an active run", async () => {
    const runningSummary: MethodsLabRunSummary = {
      ...codexSummary,
      status: "running",
      cancellable: true,
      finished_at: null,
      total_tasks: 2,
      completed_tasks: 1,
    };
    const runningDetail: MethodsLabRunDetail = {
      ...codexDetail,
      ...runningSummary,
      matrix: {
        ...codexDetail.matrix,
        cells: [
          {
            ...codexDetail.matrix.cells[0]!,
            status: "running",
            total_docs: 2,
            completed_docs: 1,
          },
        ],
      },
      progress: {
        total_tasks: 2,
        completed_tasks: 1,
        failed_tasks: 0,
      },
    };
    const secondDocResult = deferred<MethodsLabDocResult>();
    let pollCallback: (() => void) | null = null;

    clientMocks.listMethodsLabRuns.mockResolvedValue([runningSummary]);
    clientMocks.getMethodsLabRun.mockImplementation(
      async () => cloneValue(runningDetail),
    );
    clientMocks.getMethodsLabDocResult
      .mockResolvedValueOnce(
        makeDocResult(runningDetail, {
          cellId: runningDetail.matrix.cells[0]!.id,
          docId: "doc-1",
        }),
      )
      .mockImplementationOnce(() => secondDocResult.promise);
    vi.spyOn(window, "setInterval").mockImplementation((handler) => {
      pollCallback = typeof handler === "function" ? (handler as () => void) : null;
      return 1 as unknown as number;
    });
    vi.spyOn(window, "clearInterval").mockImplementation(() => {});

    render(
      <MethodsLabTab
        documents={documents}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    await screen.findByText("5bcf6ed4:model_1__method_1:doc-1");

    await act(async () => {
      pollCallback?.();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(clientMocks.getMethodsLabDocResult).toHaveBeenCalledTimes(2);
    expect(screen.queryByText("loading")).toBeNull();
    expect(screen.getByText("5bcf6ed4:model_1__method_1:doc-1")).toBeTruthy();

    secondDocResult.resolve(
      makeDocResult(runningDetail, {
        cellId: runningDetail.matrix.cells[0]!.id,
        docId: "doc-1",
      }),
    );

    await waitFor(() => {
      expect(screen.getByText("5bcf6ed4:model_1__method_1:doc-1")).toBeTruthy();
    });
  });
});
