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
  getExperimentDiagnostics: vi.fn(),
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
  default: ({
    selectedCellId,
    experimentDiagnostics,
    run,
  }: {
    selectedCellId: string | null;
    experimentDiagnostics?: { api_base_host?: string | null } | null;
    run: { diagnostics?: { effective_worker_count?: number } };
  }) => (
    <div data-testid="methods-lab-matrix">
      {selectedCellId ?? "no-cell"}|{experimentDiagnostics?.api_base_host ?? "no-host"}|
      {run.diagnostics?.effective_worker_count ?? "no-effective"}
    </div>
  ),
}));

vi.mock("./MethodsLabCellDetail", () => ({
  default: ({
    cell,
    detail,
    run,
    loading,
    onRerunErrorDocs,
    rerunningErrorDocs,
  }: {
    cell: { error_count: number } | null;
    detail: MethodsLabDocResult | null;
    run?: { doc_ids?: string[] };
    loading: boolean;
    onRerunErrorDocs?: () => void | Promise<void>;
    rerunningErrorDocs?: boolean;
  }) => (
    <div data-testid="methods-lab-cell-detail">
      {loading ? "loading" : detail ? `${detail.run_id}:${detail.cell_id}:${detail.doc_id}` : "no-detail"}
      <div data-testid="methods-lab-doc-options">
        {(run?.doc_ids ?? []).map((docId) => `${docId}:${docId}`).join("|")}
      </div>
      {cell ? (
        <button
          type="button"
          disabled={cell.error_count === 0 || rerunningErrorDocs}
          onClick={() => void onRerunErrorDocs?.()}
        >
          Re-run error docs
        </button>
      ) : null}
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

function makeRunSummary(
  id: string,
  name: string,
  methodBundle: "legacy" | "audited" | "test" = "audited",
): MethodsLabRunSummary {
  return {
    id,
    name,
    status: "completed",
    method_bundle: methodBundle,
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
    folder_ids: [],
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
      reference_source: "manual",
      fallback_reference_source: "pre",
      method_bundle: summary.method_bundle,
      chunk_mode: "auto",
      chunk_size_chars: 10000,
    },
    concurrency: 1,
    warnings: [],
    errors: [],
    diagnostics: {
      requested_concurrency: 16,
      effective_worker_count: 1,
      max_allowed_concurrency: 16,
      total_tasks: 1,
      clamped_by_task_count: true,
      clamped_by_server_cap: false,
      api_base_host: "api.ai.it.cornell.edu",
    },
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
    { id: "doc-1", filename: "doc-1.txt", display_name: "doc-1.txt", status: "reviewed" },
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
    clientMocks.getExperimentDiagnostics.mockReset();
    clientMocks.getExperimentLimits.mockReset();
    clientMocks.getMethodsLabDocResult.mockReset();
    clientMocks.getMethodsLabRun.mockReset();
    clientMocks.listMethodsLabRuns.mockReset();
    clientMocks.stopMethodsLabRun.mockReset();
    clientMocks.getAgentMethods.mockResolvedValue([]);
    clientMocks.getExperimentDiagnostics.mockResolvedValue({
      resolved_api_base: "https://api.ai.it.cornell.edu",
      api_base_host: "api.ai.it.cornell.edu",
      prompt_lab_max_concurrency: 16,
      methods_lab_max_concurrency: 16,
      gateway_catalog: {
        reachable: true,
        model_count: 189,
        error: null,
        checked_at: "2026-03-10T00:00:00Z",
      },
    });
    clientMocks.getExperimentLimits.mockResolvedValue({
      prompt_lab_default_concurrency: 16,
      prompt_lab_max_concurrency: 16,
      methods_lab_default_concurrency: 16,
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
        folders={[]}
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
      prompt_lab_default_concurrency: 16,
      prompt_lab_max_concurrency: 16,
      methods_lab_default_concurrency: 16,
      methods_lab_max_concurrency: 12,
    });

    render(
      <MethodsLabTab
        documents={documents}
        folders={[]}
        selectedDocumentId="doc-1"
        onSelectDocument={() => {}}
      />,
    );

    expect(await screen.findByText("run form max=12")).toBeTruthy();
    expect(clientMocks.getExperimentLimits).toHaveBeenCalledTimes(1);
  });

  it("shows the method bundle in run history", async () => {
    const legacySummary = makeRunSummary("legacy-run", "methods_lab__legacy", "legacy");
    const legacyDetail = makeRunDetail(legacySummary, {
      docId: "doc-1",
      cellId: "model_1__method_1",
      modelId: "model_1",
      modelLabel: "Codex",
      methodId: "method_1",
      methodLabel: "Default",
    });
    clientMocks.listMethodsLabRuns.mockResolvedValue([legacySummary]);
    clientMocks.getMethodsLabRun.mockResolvedValue(legacyDetail);
    clientMocks.getMethodsLabDocResult.mockResolvedValue(
      makeDocResult(legacyDetail, {
        cellId: legacyDetail.matrix.cells[0]!.id,
        docId: "doc-1",
      }),
    );

    render(
      <MethodsLabTab
        documents={documents}
        folders={[]}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    expect(await screen.findByText(/Legacy methods/i)).toBeTruthy();
  });

  it("shows the test method bundle in run history", async () => {
    const testSummary = makeRunSummary("test-run", "methods_lab__test", "test");
    const testDetail = makeRunDetail(testSummary, {
      docId: "doc-1",
      cellId: "model_1__method_1",
      modelId: "model_1",
      modelLabel: "Codex",
      methodId: "method_1",
      methodLabel: "Default",
    });
    clientMocks.listMethodsLabRuns.mockResolvedValue([testSummary]);
    clientMocks.getMethodsLabRun.mockResolvedValue(testDetail);
    clientMocks.getMethodsLabDocResult.mockResolvedValue(
      makeDocResult(testDetail, {
        cellId: testDetail.matrix.cells[0]!.id,
        docId: "doc-1",
      }),
    );

    render(
      <MethodsLabTab
        documents={documents}
        folders={[]}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    expect(await screen.findByText(/Test methods/i)).toBeTruthy();
  });

  it("loads experiment diagnostics and passes them to the selected run view", async () => {
    clientMocks.listMethodsLabRuns.mockResolvedValue([codexSummary]);
    clientMocks.getMethodsLabRun.mockResolvedValue(codexDetail);
    clientMocks.getMethodsLabDocResult.mockResolvedValue(
      makeDocResult(codexDetail, {
        cellId: codexDetail.matrix.cells[0]!.id,
        docId: "doc-1",
      }),
    );

    render(
      <MethodsLabTab
        documents={documents}
        folders={[]}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    expect(
      await screen.findByText("model_1__method_1|api.ai.it.cornell.edu|1"),
    ).toBeTruthy();
    expect(clientMocks.getExperimentDiagnostics).toHaveBeenCalledTimes(1);
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
        folders={[]}
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
        folders={[]}
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
        folders={[]}
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

  it("re-runs only the error docs for the selected method cell", async () => {
    const errorRunDetail: MethodsLabRunDetail = {
      ...codexDetail,
      status: "completed_with_errors",
      doc_count: 2,
      total_tasks: 2,
      completed_tasks: 2,
      failed_tasks: 1,
      doc_ids: ["doc-1", "doc-2"],
      matrix: {
        ...codexDetail.matrix,
        cells: [
          {
            ...codexDetail.matrix.cells[0]!,
            status: "completed_with_errors",
            total_docs: 2,
            completed_docs: 1,
            failed_docs: 1,
            error_count: 1,
          },
        ],
      },
      progress: {
        total_tasks: 2,
        completed_tasks: 2,
        failed_tasks: 1,
      },
    };
    const rerunSummary = makeRunSummary("rerun-ml", "Methods error rerun");
    const rerunDetail: MethodsLabRunDetail = {
      ...makeRunDetail(rerunSummary, {
        docId: "doc-2",
        cellId: errorRunDetail.matrix.cells[0]!.id,
        modelId: errorRunDetail.models[0]!.id!,
        modelLabel: errorRunDetail.models[0]!.label,
        methodId: errorRunDetail.methods[0]!.id!,
        methodLabel: errorRunDetail.methods[0]!.label,
      }),
      name: "Methods error rerun",
    };

    clientMocks.listMethodsLabRuns
      .mockResolvedValueOnce([codexSummary])
      .mockResolvedValueOnce([codexSummary, rerunSummary]);
    clientMocks.getMethodsLabRun.mockImplementation(async (runId: string) => {
      if (runId === codexSummary.id) return cloneValue(errorRunDetail);
      if (runId === rerunSummary.id) return cloneValue(rerunDetail);
      throw new Error(`Unknown run ${runId}`);
    });
    clientMocks.getMethodsLabDocResult.mockImplementation(
      async (runId: string, cellId: string, docId: string) => {
        if (runId === errorRunDetail.id && cellId === errorRunDetail.matrix.cells[0]!.id) {
          return {
            ...makeDocResult(errorRunDetail, { cellId, docId }),
            status: docId === "doc-2" ? "failed" : "completed",
            error: docId === "doc-2" ? "timeout" : null,
          };
        }
        if (runId === rerunDetail.id && cellId === rerunDetail.matrix.cells[0]!.id && docId === "doc-2") {
          return makeDocResult(rerunDetail, { cellId, docId });
        }
        throw new Error("404: Methods Lab cell not found");
      },
    );
    clientMocks.createMethodsLabRun.mockResolvedValue(rerunDetail);

    render(
      <MethodsLabTab
        documents={documents}
        folders={[]}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    await screen.findByText("5bcf6ed4:model_1__method_1:doc-1");

    fireEvent.click(screen.getByRole("button", { name: "Re-run error docs" }));

    await waitFor(() => {
      expect(clientMocks.createMethodsLabRun).toHaveBeenCalledWith(
        expect.objectContaining({
          doc_ids: ["doc-2"],
          folder_ids: [],
          methods: [errorRunDetail.methods[0]],
          models: [errorRunDetail.models[0]],
          runtime: expect.objectContaining(errorRunDetail.runtime),
          concurrency: errorRunDetail.concurrency,
        }),
      );
    });
    await screen.findByText("rerun-ml:model_1__method_1:doc-2");
  });

  it("passes internal document ids to the Methods Lab detail selector", async () => {
    const jsonlDocuments: DocumentSummary[] = [
      {
        id: "476838c9_line0",
        filename: "DeID_GT_UPchieve_math_1000transcripts (2).record-0001.json",
        display_name: "Session 16592",
        status: "reviewed",
      },
    ];
    const jsonlSummary = makeRunSummary("run-jsonl", "methods_lab__jsonl");
    const jsonlDetail = makeRunDetail(jsonlSummary, {
      docId: "476838c9_line0",
      cellId: "model_1__method_1",
      modelId: "model_1",
      modelLabel: "Claude Sonnet 4.6",
      methodId: "method_1",
      methodLabel: "Regex + LLM Extended v2",
    });
    clientMocks.listMethodsLabRuns.mockResolvedValue([jsonlSummary]);
    clientMocks.getMethodsLabRun.mockResolvedValue(jsonlDetail);
    clientMocks.getMethodsLabDocResult.mockResolvedValue(
      makeDocResult(jsonlDetail, {
        cellId: jsonlDetail.matrix.cells[0]!.id,
        docId: "476838c9_line0",
      }),
    );

    render(
      <MethodsLabTab
        documents={jsonlDocuments}
        folders={[]}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    await screen.findByText("run-jsonl:model_1__method_1:476838c9_line0");
    expect(screen.getByTestId("methods-lab-doc-options").textContent).toContain(
      "476838c9_line0:476838c9_line0",
    );
  });
});
