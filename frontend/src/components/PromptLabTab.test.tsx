import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import PromptLabTab from "./PromptLabTab";
import type {
  DocumentSummary,
  PromptLabDocResult,
  PromptLabRunDetail,
  PromptLabRunSummary,
} from "../types";

const clientMocks = vi.hoisted(() => ({
  createPromptLabRun: vi.fn(),
  deletePromptLabRun: vi.fn(),
  getAgentMethods: vi.fn(),
  getExperimentDiagnostics: vi.fn(),
  getExperimentLimits: vi.fn(),
  getPromptLabDocResult: vi.fn(),
  getPromptLabRun: vi.fn(),
  listPromptLabRuns: vi.fn(),
  stopPromptLabRun: vi.fn(),
}));

vi.mock("../api/client", () => clientMocks);

vi.mock("./PromptLabRunForm", () => ({
  default: ({ concurrencyMax }: { concurrencyMax: number }) => (
    <div data-testid="prompt-lab-run-form">{`run form max=${concurrencyMax}`}</div>
  ),
}));

vi.mock("./PromptLabMatrix", () => ({
  default: ({
    selectedCellId,
    experimentDiagnostics,
    run,
  }: {
    selectedCellId: string | null;
    experimentDiagnostics?: { api_base_host?: string | null } | null;
    run: { diagnostics?: { effective_worker_count?: number } };
  }) => (
    <div data-testid="prompt-lab-matrix">
      {selectedCellId ?? "no-cell"}|{experimentDiagnostics?.api_base_host ?? "no-host"}|
      {run.diagnostics?.effective_worker_count ?? "no-effective"}
    </div>
  ),
}));

vi.mock("./PromptLabCellDetail", () => ({
  default: ({
    cell,
    detail,
    loading,
    onRerunErrorDocs,
    rerunningErrorDocs,
  }: {
    cell: { error_count: number } | null;
    detail: PromptLabDocResult | null;
    loading: boolean;
    onRerunErrorDocs?: () => void | Promise<void>;
    rerunningErrorDocs?: boolean;
  }) => (
    <div data-testid="prompt-lab-cell-detail">
      {loading ? "loading" : detail ? `${detail.run_id}:${detail.cell_id}:${detail.doc_id}` : "no-detail"}
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
): PromptLabRunSummary {
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
    prompt_count: 1,
    model_count: 1,
    total_tasks: 1,
    completed_tasks: 1,
    failed_tasks: 0,
  };
}

function makeRunDetail(
  summary: PromptLabRunSummary,
  {
    docId,
    cellId,
    modelId,
    modelLabel,
    promptId,
    promptLabel,
  }: {
    docId: string;
    cellId: string;
    modelId: string;
    modelLabel: string;
    promptId: string;
    promptLabel: string;
  },
): PromptLabRunDetail {
  return {
    ...summary,
    doc_ids: [docId],
    folder_ids: [],
    prompts: [
      {
        id: promptId,
        label: promptLabel,
        variant_type: "prompt",
        system_prompt: `${promptLabel} prompt`,
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
      label_profile: "simple",
      label_projection: "native",
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
      prompts: [{ id: promptId, label: promptLabel }],
      cells: [
        {
          id: cellId,
          model_id: modelId,
          model_label: modelLabel,
          prompt_id: promptId,
          prompt_label: promptLabel,
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
  run: PromptLabRunDetail,
  {
    cellId,
    docId,
  }: {
    cellId: string;
    docId: string;
  },
): PromptLabDocResult {
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
    prompt: run.prompts[0],
  };
}

describe("PromptLabTab", () => {
  const documents: DocumentSummary[] = [
    { id: "doc-1", filename: "doc-1.txt", display_name: "doc-1.txt", status: "reviewed" },
  ];

  const openAiSummary = makeRunSummary("15fad00a", "prompt_lab__prompt_text_core__openai");
  const claudeSummary = makeRunSummary("21ee460f", "prompt_lab__prompt_text_core__claude");
  const openAiDetail = makeRunDetail(openAiSummary, {
    docId: "doc-1",
    cellId: "codex_xhigh__baseline_raw",
    modelId: "codex_xhigh",
    modelLabel: "codex_xhigh",
    promptId: "baseline_raw",
    promptLabel: "baseline_raw",
  });
  const claudeDetail = makeRunDetail(claudeSummary, {
    docId: "doc-1",
    cellId: "claude_thinking_off__baseline_raw",
    modelId: "claude_thinking_off",
    modelLabel: "claude_thinking_off",
    promptId: "baseline_raw",
    promptLabel: "baseline_raw",
  });

  beforeEach(() => {
    clientMocks.createPromptLabRun.mockReset();
    clientMocks.deletePromptLabRun.mockReset();
    clientMocks.getAgentMethods.mockReset();
    clientMocks.getExperimentDiagnostics.mockReset();
    clientMocks.getExperimentLimits.mockReset();
    clientMocks.getPromptLabDocResult.mockReset();
    clientMocks.getPromptLabRun.mockReset();
    clientMocks.listPromptLabRuns.mockReset();
    clientMocks.stopPromptLabRun.mockReset();
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

  it("switches runs without requesting a stale cell id", async () => {
    clientMocks.listPromptLabRuns.mockResolvedValue([openAiSummary, claudeSummary]);
    clientMocks.getPromptLabRun.mockImplementation(async (runId: string) => {
      if (runId === openAiSummary.id) return openAiDetail;
      if (runId === claudeSummary.id) return claudeDetail;
      throw new Error(`Unknown run ${runId}`);
    });
    clientMocks.getPromptLabDocResult.mockImplementation(
      async (runId: string, cellId: string, docId: string) => {
        if (runId === openAiDetail.id && cellId === openAiDetail.matrix.cells[0]?.id && docId === "doc-1") {
          return makeDocResult(openAiDetail, { cellId, docId });
        }
        if (runId === claudeDetail.id && cellId === claudeDetail.matrix.cells[0]?.id && docId === "doc-1") {
          return makeDocResult(claudeDetail, { cellId, docId });
        }
        throw new Error("404: Prompt Lab cell not found");
      },
    );

    render(
      <PromptLabTab
        documents={documents}
        folders={[]}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    await screen.findByText("15fad00a:codex_xhigh__baseline_raw:doc-1");

    fireEvent.click(screen.getByRole("button", { name: /prompt_lab__prompt_text_core__claude/i }));

    await screen.findByText("21ee460f:claude_thinking_off__baseline_raw:doc-1");

    expect(screen.queryByText(/404: Prompt Lab cell not found/i)).toBeNull();
    expect(clientMocks.getPromptLabDocResult).not.toHaveBeenCalledWith(
      "21ee460f",
      "codex_xhigh__baseline_raw",
      "doc-1",
    );
  });

  it("loads experiment limits and passes the prompt concurrency max to the form", async () => {
    clientMocks.listPromptLabRuns.mockResolvedValue([]);
    clientMocks.getExperimentLimits.mockResolvedValue({
      prompt_lab_default_concurrency: 10,
      prompt_lab_max_concurrency: 12,
      methods_lab_default_concurrency: 10,
      methods_lab_max_concurrency: 16,
    });

    render(
      <PromptLabTab
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
    const legacySummary = makeRunSummary("legacy-run", "prompt_lab__legacy", "legacy");
    const legacyDetail = makeRunDetail(legacySummary, {
      docId: "doc-1",
      cellId: "model_1__prompt_1",
      modelId: "model_1",
      modelLabel: "Codex",
      promptId: "prompt_1",
      promptLabel: "Preset",
    });
    clientMocks.listPromptLabRuns.mockResolvedValue([legacySummary]);
    clientMocks.getPromptLabRun.mockResolvedValue(legacyDetail);
    clientMocks.getPromptLabDocResult.mockResolvedValue(
      makeDocResult(legacyDetail, {
        cellId: legacyDetail.matrix.cells[0]!.id,
        docId: "doc-1",
      }),
    );

    render(
      <PromptLabTab
        documents={documents}
        folders={[]}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    expect(await screen.findByText(/Legacy methods/i)).toBeTruthy();
  });

  it("shows the test method bundle in run history", async () => {
    const testSummary = makeRunSummary("test-run", "prompt_lab__test", "test");
    const testDetail = makeRunDetail(testSummary, {
      docId: "doc-1",
      cellId: "model_1__prompt_1",
      modelId: "model_1",
      modelLabel: "Codex",
      promptId: "prompt_1",
      promptLabel: "Preset",
    });
    clientMocks.listPromptLabRuns.mockResolvedValue([testSummary]);
    clientMocks.getPromptLabRun.mockResolvedValue(testDetail);
    clientMocks.getPromptLabDocResult.mockResolvedValue(
      makeDocResult(testDetail, {
        cellId: testDetail.matrix.cells[0]!.id,
        docId: "doc-1",
      }),
    );

    render(
      <PromptLabTab
        documents={documents}
        folders={[]}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    expect(await screen.findByText(/Test methods/i)).toBeTruthy();
  });

  it("loads experiment diagnostics and passes them to the selected run view", async () => {
    clientMocks.listPromptLabRuns.mockResolvedValue([openAiSummary]);
    clientMocks.getPromptLabRun.mockResolvedValue(openAiDetail);
    clientMocks.getPromptLabDocResult.mockResolvedValue(
      makeDocResult(openAiDetail, {
        cellId: openAiDetail.matrix.cells[0]!.id,
        docId: "doc-1",
      }),
    );

    render(
      <PromptLabTab
        documents={documents}
        folders={[]}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    expect(
      await screen.findByText("codex_xhigh__baseline_raw|api.ai.it.cornell.edu|1"),
    ).toBeTruthy();
    expect(clientMocks.getExperimentDiagnostics).toHaveBeenCalledTimes(1);
  });

  it("keeps the current doc detail visible while polling an active run", async () => {
    const runningSummary: PromptLabRunSummary = {
      ...openAiSummary,
      status: "running",
      cancellable: true,
      finished_at: null,
      total_tasks: 2,
      completed_tasks: 1,
    };
    const runningDetail: PromptLabRunDetail = {
      ...openAiDetail,
      ...runningSummary,
      matrix: {
        ...openAiDetail.matrix,
        cells: [
          {
            ...openAiDetail.matrix.cells[0]!,
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
    const secondDocResult = deferred<PromptLabDocResult>();
    let pollCallback: (() => void) | null = null;

    clientMocks.listPromptLabRuns.mockResolvedValue([runningSummary]);
    clientMocks.getPromptLabRun.mockImplementation(
      async () => cloneValue(runningDetail),
    );
    clientMocks.getPromptLabDocResult
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
      <PromptLabTab
        documents={documents}
        folders={[]}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    await screen.findByText("15fad00a:codex_xhigh__baseline_raw:doc-1");

    await act(async () => {
      pollCallback?.();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(clientMocks.getPromptLabDocResult).toHaveBeenCalledTimes(2);
    expect(screen.queryByText("loading")).toBeNull();
    expect(screen.getByText("15fad00a:codex_xhigh__baseline_raw:doc-1")).toBeTruthy();

    secondDocResult.resolve(
      makeDocResult(runningDetail, {
        cellId: runningDetail.matrix.cells[0]!.id,
        docId: "doc-1",
      }),
    );

    await waitFor(() => {
      expect(screen.getByText("15fad00a:codex_xhigh__baseline_raw:doc-1")).toBeTruthy();
    });
  });

  it("clears stale detail before loading the next run after deleting the selected run", async () => {
    const claudeDetailDeferred = deferred<PromptLabRunDetail>();
    let currentRuns = [openAiSummary, claudeSummary];
    clientMocks.listPromptLabRuns.mockImplementation(async () => currentRuns);
    clientMocks.getPromptLabRun.mockImplementation((runId: string) => {
      if (runId === openAiSummary.id) return Promise.resolve(openAiDetail);
      if (runId === claudeSummary.id) return claudeDetailDeferred.promise;
      throw new Error(`Unknown run ${runId}`);
    });
    clientMocks.getPromptLabDocResult.mockImplementation(
      async (runId: string, cellId: string, docId: string) => {
        if (runId === openAiDetail.id && cellId === openAiDetail.matrix.cells[0]?.id) {
          return makeDocResult(openAiDetail, { cellId, docId });
        }
        if (runId === claudeDetail.id && cellId === claudeDetail.matrix.cells[0]?.id) {
          return makeDocResult(claudeDetail, { cellId, docId });
        }
        throw new Error("404: Prompt Lab cell not found");
      },
    );
    clientMocks.deletePromptLabRun.mockImplementation(async () => {
      currentRuns = [claudeSummary];
      return { ok: true, id: openAiSummary.id };
    });
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(
      <PromptLabTab
        documents={documents}
        folders={[]}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    await screen.findByText("15fad00a:codex_xhigh__baseline_raw:doc-1");

    fireEvent.click(screen.getAllByRole("button", { name: "Delete" })[0]!);

    await screen.findByText("Create or select a Prompt Lab run.");

    claudeDetailDeferred.resolve(claudeDetail);

    await screen.findByText("21ee460f:claude_thinking_off__baseline_raw:doc-1");

    expect(screen.queryByText(/404: Prompt Lab cell not found/i)).toBeNull();
    await waitFor(() => {
      expect(clientMocks.getPromptLabDocResult).toHaveBeenCalledWith(
        "21ee460f",
        "claude_thinking_off__baseline_raw",
        "doc-1",
      );
    });
  });

  it("stops an active run and refreshes its status", async () => {
    const runningSummary: PromptLabRunSummary = {
      ...openAiSummary,
      status: "running",
      cancellable: true,
      finished_at: null,
      total_tasks: 2,
      completed_tasks: 1,
    };
    const cancellingDetail: PromptLabRunDetail = {
      ...openAiDetail,
      ...runningSummary,
      status: "cancelling",
    };

    clientMocks.listPromptLabRuns
      .mockResolvedValueOnce([runningSummary])
      .mockResolvedValueOnce([{ ...runningSummary, status: "cancelling" }]);
    clientMocks.getPromptLabRun
      .mockResolvedValueOnce({ ...openAiDetail, ...runningSummary })
      .mockResolvedValueOnce(cancellingDetail);
    clientMocks.getPromptLabDocResult.mockResolvedValue(
      makeDocResult(openAiDetail, { cellId: openAiDetail.matrix.cells[0]!.id, docId: "doc-1" }),
    );
    clientMocks.stopPromptLabRun.mockResolvedValue({
      ok: true,
      id: runningSummary.id,
      status: "cancelling",
    });
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(
      <PromptLabTab
        documents={documents}
        folders={[]}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    await screen.findByText("15fad00a:codex_xhigh__baseline_raw:doc-1");

    fireEvent.click(screen.getByRole("button", { name: "Stop" }));

    await waitFor(() => {
      expect(clientMocks.stopPromptLabRun).toHaveBeenCalledWith("15fad00a");
    });
    await waitFor(() => {
      expect(clientMocks.listPromptLabRuns).toHaveBeenCalledTimes(2);
      expect(clientMocks.getPromptLabRun).toHaveBeenCalledTimes(2);
    });
  });

  it("re-runs only the error docs for the selected prompt cell", async () => {
    const errorRunDetail: PromptLabRunDetail = {
      ...openAiDetail,
      status: "completed_with_errors",
      doc_count: 2,
      total_tasks: 2,
      completed_tasks: 2,
      failed_tasks: 1,
      doc_ids: ["doc-1", "doc-2"],
      matrix: {
        ...openAiDetail.matrix,
        cells: [
          {
            ...openAiDetail.matrix.cells[0]!,
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
    const rerunSummary = makeRunSummary("rerun-1", "Prompt error rerun");
    const rerunDetail: PromptLabRunDetail = {
      ...makeRunDetail(rerunSummary, {
        docId: "doc-2",
        cellId: errorRunDetail.matrix.cells[0]!.id,
        modelId: errorRunDetail.models[0]!.id!,
        modelLabel: errorRunDetail.models[0]!.label,
        promptId: errorRunDetail.prompts[0]!.id!,
        promptLabel: errorRunDetail.prompts[0]!.label,
      }),
      name: "Prompt error rerun",
    };

    clientMocks.listPromptLabRuns
      .mockResolvedValueOnce([openAiSummary])
      .mockResolvedValueOnce([openAiSummary, rerunSummary]);
    clientMocks.getPromptLabRun.mockImplementation(async (runId: string) => {
      if (runId === openAiSummary.id) return cloneValue(errorRunDetail);
      if (runId === rerunSummary.id) return cloneValue(rerunDetail);
      throw new Error(`Unknown run ${runId}`);
    });
    clientMocks.getPromptLabDocResult.mockImplementation(
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
        throw new Error("404: Prompt Lab cell not found");
      },
    );
    clientMocks.createPromptLabRun.mockResolvedValue(rerunDetail);

    render(
      <PromptLabTab
        documents={documents}
        folders={[]}
        selectedDocumentId={null}
        onSelectDocument={vi.fn()}
      />,
    );

    await screen.findByText("15fad00a:codex_xhigh__baseline_raw:doc-1");

    fireEvent.click(screen.getByRole("button", { name: "Re-run error docs" }));

    await waitFor(() => {
      expect(clientMocks.createPromptLabRun).toHaveBeenCalledWith(
        expect.objectContaining({
          doc_ids: ["doc-2"],
          folder_ids: [],
          prompts: [errorRunDetail.prompts[0]],
          models: [errorRunDetail.models[0]],
          runtime: expect.objectContaining(errorRunDetail.runtime),
          concurrency: errorRunDetail.concurrency,
        }),
      );
    });
    await screen.findByText("rerun-1:codex_xhigh__baseline_raw:doc-2");
  });
});
