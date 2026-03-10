import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createPromptLabRun,
  deletePromptLabRun,
  getAgentMethods,
  getExperimentDiagnostics,
  getExperimentLimits,
  getPromptLabDocResult,
  getPromptLabRun,
  listPromptLabRuns,
  stopPromptLabRun,
} from "../api/client";
import { formatMethodBundleLabel } from "../experimentDisplay";
import { DEFAULT_EXPERIMENT_LIMITS } from "../types";
import type {
  AgentMethodOption,
  DocumentSummary,
  ExperimentDiagnostics,
  ExperimentLimits,
  FolderSummary,
  PromptLabDocResult,
  PromptLabRunCreateRequest,
  PromptLabRunDetail,
  PromptLabRunSummary,
} from "../types";
import { formatExperimentModelLabel, getExperimentModelLabelById } from "../modelDisplay";
import PromptLabCellDetail from "./PromptLabCellDetail";
import PromptLabMatrix from "./PromptLabMatrix";
import PromptLabRunForm from "./PromptLabRunForm";

interface Props {
  documents: DocumentSummary[];
  folders: FolderSummary[];
  selectedDocumentId: string | null;
  onSelectDocument: (docId: string) => void;
}

type PromptLabToastKind = "success" | "error" | "info";

interface PromptLabToast {
  id: string;
  kind: PromptLabToastKind;
  message: string;
}

function isActiveStatus(status: string): boolean {
  return status === "queued" || status === "running" || status === "cancelling";
}

function isErrorDocStatus(status: PromptLabDocResult["status"]): boolean {
  return status === "failed" || status === "unavailable";
}

function readSessionStorageValue(key: string): string | undefined {
  try {
    const value = sessionStorage.getItem(key)?.trim() ?? "";
    return value || undefined;
  } catch {
    return undefined;
  }
}

function usePromptLabTabController(onSelectDocument: (docId: string) => void) {
  const [setupCollapsed, setSetupCollapsed] = useState(false);
  const [runs, setRuns] = useState<PromptLabRunSummary[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [deletingRunId, setDeletingRunId] = useState<string | null>(null);
  const [stoppingRunId, setStoppingRunId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<PromptLabRunDetail | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [creatingRun, setCreatingRun] = useState(false);
  const [rerunningErrorCellId, setRerunningErrorCellId] = useState<string | null>(null);

  const [selectedCellId, setSelectedCellId] = useState<string | null>(null);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [docDetail, setDocDetail] = useState<PromptLabDocResult | null>(null);
  const [docLoading, setDocLoading] = useState(false);
  const [methodCatalog, setMethodCatalog] = useState<AgentMethodOption[]>([]);
  const [experimentLimits, setExperimentLimits] = useState<ExperimentLimits>(
    DEFAULT_EXPERIMENT_LIMITS,
  );
  const [experimentDiagnostics, setExperimentDiagnostics] = useState<ExperimentDiagnostics | null>(
    null,
  );
  const [toasts, setToasts] = useState<PromptLabToast[]>([]);
  const previousRunStatusRef = useRef<Record<string, string>>({});
  const previousCellCompletedRef = useRef<Record<string, number>>({});
  const previousCellErrorsRef = useRef<Record<string, number>>({});
  const previousDocStatusRef = useRef<Record<string, string>>({});
  const latestRunDetailRequestRef = useRef<string | null>(null);
  const latestDocRequestRef = useRef<string | null>(null);
  const latestDocDetailRef = useRef<PromptLabDocResult | null>(null);

  const pushToast = useCallback((kind: PromptLabToastKind, message: string) => {
    const id = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    setToasts((prev) => [...prev, { id, kind, message }]);
    window.setTimeout(() => {
      setToasts((prev) => prev.filter((item) => item.id !== id));
    }, 4500);
  }, []);

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const clearDocDetailState = useCallback(() => {
    latestDocRequestRef.current = null;
    setDocDetail(null);
    setDocLoading(false);
  }, []);

  const clearSelectedDocState = useCallback(() => {
    setSelectedCellId(null);
    setSelectedDocId(null);
    clearDocDetailState();
  }, [clearDocDetailState]);

  const beginDocLoad = useCallback((shouldStart: boolean) => {
    if (shouldStart) {
      setDocLoading(true);
    }
  }, []);

  const handleDocLoadError = useCallback(
    (requestKey: string, active: boolean, error: unknown) => {
      if (!active || latestDocRequestRef.current !== requestKey) {
        return;
      }
      setDocDetail(null);
      setRunError(String(error));
    },
    [],
  );

  const finishDocLoad = useCallback((requestKey: string, active: boolean) => {
    if (!active || latestDocRequestRef.current !== requestKey) {
      return;
    }
    setDocLoading(false);
  }, []);

  const refreshRuns = useCallback(async () => {
    setRunsLoading(true);
    try {
      const next = await listPromptLabRuns();
      setRuns(next);
      setSelectedRunId((current) => {
        if (next.length === 0) {
          setSetupCollapsed(false);
          return null;
        }
        if (!current || !next.some((run) => run.id === current)) {
          setSetupCollapsed(true);
          return next[0]?.id ?? null;
        }
        return current;
      });
    } catch (e: unknown) {
      setRunError(String(e));
    } finally {
      setRunsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshRuns();
  }, [refreshRuns]);

  useEffect(() => {
    getAgentMethods()
      .then(setMethodCatalog)
      .catch(() => setMethodCatalog([]));
  }, []);

  useEffect(() => {
    getExperimentLimits()
      .then(setExperimentLimits)
      .catch(() => setExperimentLimits(DEFAULT_EXPERIMENT_LIMITS));
  }, []);

  useEffect(() => {
    getExperimentDiagnostics()
      .then(setExperimentDiagnostics)
      .catch(() => setExperimentDiagnostics(null));
  }, []);

  useEffect(() => {
    latestRunDetailRequestRef.current = selectedRunId;
    latestDocRequestRef.current = null;
    setRunDetail(null);
    clearSelectedDocState();
    setRunError(null);
  }, [clearSelectedDocState, selectedRunId]);

  const refreshRunDetail = useCallback(async () => {
    if (!selectedRunId) {
      latestRunDetailRequestRef.current = null;
      setRunDetail(null);
      return;
    }
    const requestRunId = selectedRunId;
    latestRunDetailRequestRef.current = requestRunId;
    try {
      const next = await getPromptLabRun(requestRunId);
      if (latestRunDetailRequestRef.current !== requestRunId) {
        return;
      }
      setRunDetail(next);
      setRunError(null);
    } catch (e: unknown) {
      if (latestRunDetailRequestRef.current !== requestRunId) {
        return;
      }
      setRunError(String(e));
    }
  }, [selectedRunId]);

  const loadSelectedDocDetail = useCallback(
    async (
      requestKey: string,
      runId: string,
      cellId: string,
      docId: string,
      onStale: () => boolean,
    ) => {
      const detail = await getPromptLabDocResult(runId, cellId, docId);
      if (onStale() || latestDocRequestRef.current !== requestKey) {
        return;
      }
      setDocDetail(detail);
      setRunError(null);
      const key = `${detail.run_id}::${detail.cell_id}::${detail.doc_id}`;
      const previousStatus = previousDocStatusRef.current[key];
      if (previousStatus && previousStatus !== detail.status) {
        const filename = detail.document.filename || detail.document.id;
        const modelLabel = formatExperimentModelLabel(detail.model, detail.model?.label ?? "model");
        const promptLabel = detail.prompt?.label ?? "prompt";
        if (detail.status === "completed") {
          pushToast(
            "success",
            `Doc completed: ${filename} (${modelLabel} × ${promptLabel}).`,
          );
        } else if (detail.status === "cancelled") {
          pushToast(
            "info",
            `Doc cancelled: ${filename} (${modelLabel} × ${promptLabel}).`,
          );
        } else if (detail.status === "failed" || detail.status === "unavailable") {
          pushToast(
            "error",
            `Doc failed: ${filename} (${modelLabel} × ${promptLabel}).`,
          );
        }
      }
      previousDocStatusRef.current[key] = detail.status;
      onSelectDocument(detail.doc_id);
    },
    [onSelectDocument, pushToast],
  );

  useEffect(() => {
    void refreshRunDetail();
  }, [refreshRunDetail]);

  useEffect(() => {
    if (!runDetail) return;
    const runId = runDetail.id;
    const previousRunStatus = previousRunStatusRef.current[runId];
    if (previousRunStatus && previousRunStatus !== runDetail.status) {
      if (runDetail.status === "completed") {
        pushToast(
          "success",
          `Prompt Lab run completed: ${runDetail.name}.`,
        );
      } else if (runDetail.status === "cancelled") {
        pushToast("info", `Prompt Lab run stopped: ${runDetail.name}.`);
      } else if (runDetail.status === "completed_with_errors" || runDetail.status === "failed") {
        pushToast(
          "error",
          `Prompt Lab run ended with issues: ${runDetail.name} (${runDetail.status}).`,
        );
      }
    }
    previousRunStatusRef.current[runId] = runDetail.status;

    for (const cell of runDetail.matrix.cells) {
      const modelLabel = getExperimentModelLabelById(
        runDetail.models,
        cell.model_id,
        cell.model_label,
      );
      const cellKey = `${runId}::${cell.id}`;
      const previousCompleted = previousCellCompletedRef.current[cellKey];
      if (
        typeof previousCompleted === "number" &&
        cell.completed_docs > previousCompleted
      ) {
        const delta = cell.completed_docs - previousCompleted;
        pushToast(
          "info",
          `Cell ${modelLabel} × ${cell.prompt_label}: +${delta} doc(s) completed (${cell.completed_docs}/${cell.total_docs}).`,
        );
      }
      previousCellCompletedRef.current[cellKey] = cell.completed_docs;

      const previousErrors = previousCellErrorsRef.current[cellKey];
      if (
        typeof previousErrors === "number" &&
        cell.error_count > previousErrors
      ) {
        const delta = cell.error_count - previousErrors;
        pushToast(
          "error",
          `Cell ${modelLabel} × ${cell.prompt_label}: +${delta} new error(s) (${cell.error_count} total).`,
        );
      }
      previousCellErrorsRef.current[cellKey] = cell.error_count;
    }
  }, [pushToast, runDetail]);

  useEffect(() => {
    latestDocDetailRef.current = docDetail;
  }, [docDetail]);

  useEffect(() => {
    if (!runDetail) return;
    const firstCellId = runDetail.matrix.cells[0]?.id ?? null;
    if (!selectedCellId || !runDetail.matrix.cells.some((cell) => cell.id === selectedCellId)) {
      setSelectedCellId(firstCellId);
    }
    if (!selectedDocId || !runDetail.doc_ids.includes(selectedDocId)) {
      setSelectedDocId(runDetail.doc_ids[0] ?? null);
    }
  }, [runDetail, selectedCellId, selectedDocId]);

  useEffect(() => {
    if (!runDetail || !isActiveStatus(runDetail.status)) return;
    const timer = window.setInterval(() => {
      void Promise.all([refreshRuns(), refreshRunDetail()]);
    }, 2000);
    return () => window.clearInterval(timer);
  }, [runDetail, refreshRunDetail, refreshRuns]);

  const hasValidDocSelection = Boolean(
    runDetail &&
      selectedCellId &&
      selectedDocId &&
      runDetail.matrix.cells.some((cell) => cell.id === selectedCellId) &&
      runDetail.doc_ids.includes(selectedDocId),
  );

  useEffect(() => {
    if (!runDetail || !selectedCellId || !selectedDocId || !hasValidDocSelection) {
      clearDocDetailState();
      return;
    }
    const requestKey = `${runDetail.id}::${selectedCellId}::${selectedDocId}`;
    const currentDocDetail = latestDocDetailRef.current;
    const hasVisibleDetailForSelection =
      currentDocDetail?.run_id === runDetail.id &&
      currentDocDetail.cell_id === selectedCellId &&
      currentDocDetail.doc_id === selectedDocId;
    const previousRequestKey = latestDocRequestRef.current;
    latestDocRequestRef.current = requestKey;
    let active = true;
    beginDocLoad(previousRequestKey !== requestKey || !hasVisibleDetailForSelection);
    void loadSelectedDocDetail(
      requestKey,
      runDetail.id,
      selectedCellId,
      selectedDocId,
      () => !active,
    )
      .catch((e: unknown) => handleDocLoadError(requestKey, active, e))
      .finally(() => finishDocLoad(requestKey, active));
    return () => {
      active = false;
    };
  }, [
    beginDocLoad,
    clearDocDetailState,
    finishDocLoad,
    handleDocLoadError,
    hasValidDocSelection,
    loadSelectedDocDetail,
    runDetail,
    selectedCellId,
    selectedDocId,
  ]);

  const selectedCell = useMemo(
    () => runDetail?.matrix.cells.find((cell) => cell.id === selectedCellId) ?? null,
    [runDetail, selectedCellId],
  );

  const handleToggleSetup = useCallback(() => {
    setSetupCollapsed((prev) => !prev);
  }, []);

  const handleSelectRun = useCallback((runId: string) => {
    setSelectedRunId(runId);
    setSetupCollapsed(true);
  }, []);

  const createAndSelectRun = useCallback(async (payload: PromptLabRunCreateRequest) => {
    const created = await createPromptLabRun(payload);
    setSetupCollapsed(true);
    setSelectedRunId(created.id);
    setRunDetail(created);
    setSelectedCellId(created.matrix.cells[0]?.id ?? null);
    setSelectedDocId(created.doc_ids[0] ?? null);
    setRunError(null);
    const list = await listPromptLabRuns();
    setRuns(list);
    return created;
  }, []);

  const handleCreateRun = async (payload: PromptLabRunCreateRequest) => {
    setCreatingRun(true);
    try {
      await createAndSelectRun(payload);
    } finally {
      setCreatingRun(false);
    }
  };

  const handleRerunErrorDocs = useCallback(async () => {
    if (!runDetail || !selectedCell || isActiveStatus(runDetail.status)) {
      return;
    }
    const modelLabel = getExperimentModelLabelById(
      runDetail.models,
      selectedCell.model_id,
      selectedCell.model_label,
    );
    if (selectedCell.error_count === 0) {
      pushToast(
        "info",
        `No error docs to re-run for ${modelLabel} × ${selectedCell.prompt_label}.`,
      );
      return;
    }

    const selectedPrompt = runDetail.prompts.find((prompt) => prompt.id === selectedCell.prompt_id);
    const selectedModel = runDetail.models.find((model) => model.id === selectedCell.model_id);
    if (!selectedPrompt || !selectedModel) {
      const message = `Could not resolve the selected Prompt Lab cell configuration for ${modelLabel} × ${selectedCell.prompt_label}.`;
      setRunError(message);
      pushToast("error", message);
      return;
    }

    setRerunningErrorCellId(selectedCell.id);
    try {
      const docResults = await Promise.all(
        runDetail.doc_ids.map((docId) => getPromptLabDocResult(runDetail.id, selectedCell.id, docId)),
      );
      const errorDocIds = docResults
        .filter((result) => isErrorDocStatus(result.status))
        .map((result) => result.doc_id);
      if (errorDocIds.length === 0) {
        pushToast(
          "info",
          `No error docs remain for ${modelLabel} × ${selectedCell.prompt_label}.`,
        );
        return;
      }
      const created = await createAndSelectRun({
        name: `${runDetail.name} · ${modelLabel} × ${selectedCell.prompt_label} · error docs`,
        doc_ids: errorDocIds,
        folder_ids: [],
        prompts: [selectedPrompt],
        models: [selectedModel],
        runtime: {
          ...runDetail.runtime,
          api_key: readSessionStorageValue("prompt_lab_api_key"),
          api_base:
            readSessionStorageValue("prompt_lab_api_base") ?? runDetail.runtime.api_base,
        },
        concurrency: runDetail.concurrency,
      });
      pushToast("success", `Created Prompt Lab error-doc rerun: ${created.name}.`);
    } catch (e: unknown) {
      const message = String(e);
      setRunError(message);
      pushToast(
        "error",
        `Failed to re-run Prompt Lab error docs for ${modelLabel} × ${selectedCell.prompt_label}.`,
      );
    } finally {
      setRerunningErrorCellId(null);
    }
  }, [createAndSelectRun, pushToast, runDetail, selectedCell]);

  const handleDeleteRun = useCallback(
    async (run: PromptLabRunSummary) => {
      const confirmed = window.confirm(`Delete Prompt Lab run "${run.name}"?`);
      if (!confirmed) return;

      setDeletingRunId(run.id);
      try {
        await deletePromptLabRun(run.id);
        const next = await listPromptLabRuns();
        setRuns(next);
        const nextSelectedRunId =
          selectedRunId === run.id || !next.some((item) => item.id === selectedRunId)
            ? (next[0]?.id ?? null)
            : selectedRunId;
        setSelectedRunId(nextSelectedRunId);
        if (!nextSelectedRunId) {
          setSetupCollapsed(false);
        }
        if (selectedRunId === run.id) {
          setRunDetail(null);
          clearSelectedDocState();
        }
        setRunError(null);
        pushToast("success", `Deleted Prompt Lab run: ${run.name}.`);
      } catch (e: unknown) {
        const message = String(e);
        setRunError(message);
        pushToast("error", `Failed to delete Prompt Lab run: ${run.name}.`);
      } finally {
        setDeletingRunId(null);
      }
    },
    [clearSelectedDocState, pushToast, selectedRunId],
  );

  const handleStopRun = useCallback(
    async (run: PromptLabRunSummary) => {
      const confirmed = window.confirm(`Stop Prompt Lab run "${run.name}"?`);
      if (!confirmed) return;

      setStoppingRunId(run.id);
      try {
        await stopPromptLabRun(run.id);
        await Promise.all([
          refreshRuns(),
          selectedRunId === run.id ? refreshRunDetail() : Promise.resolve(),
        ]);
        setRunError(null);
        pushToast("info", `Stop requested for Prompt Lab run: ${run.name}.`);
      } catch (e: unknown) {
        const message = String(e);
        setRunError(message);
        pushToast("error", `Failed to stop Prompt Lab run: ${run.name}.`);
      } finally {
        setStoppingRunId(null);
      }
    },
    [pushToast, refreshRunDetail, refreshRuns, selectedRunId],
  );

  return {
    setupCollapsed,
    runs,
    runsLoading,
    deletingRunId,
    stoppingRunId,
    selectedRunId,
    runDetail,
    runError,
    creatingRun,
    rerunningErrorCellId,
    selectedCellId,
    selectedDocId,
    docDetail,
    docLoading,
    methodCatalog,
    experimentLimits,
    experimentDiagnostics,
    toasts,
    selectedCell,
    dismissToast,
    refreshRuns,
    handleCreateRun,
    handleDeleteRun,
    handleRerunErrorDocs,
    handleSelectRun,
    handleStopRun,
    handleToggleSetup,
    setSelectedCellId,
    setSelectedDocId,
  };
}

export default function PromptLabTab({
  documents,
  folders,
  selectedDocumentId,
  onSelectDocument,
}: Props) {
  const {
    setupCollapsed,
    runs,
    runsLoading,
    deletingRunId,
    stoppingRunId,
    selectedRunId,
    runDetail,
    runError,
    creatingRun,
    rerunningErrorCellId,
    selectedCellId,
    selectedDocId,
    docDetail,
    docLoading,
    methodCatalog,
    experimentLimits,
    experimentDiagnostics,
    toasts,
    selectedCell,
    dismissToast,
    refreshRuns,
    handleCreateRun,
    handleDeleteRun,
    handleRerunErrorDocs,
    handleSelectRun,
    handleStopRun,
    handleToggleSetup,
    setSelectedCellId,
    setSelectedDocId,
  } = usePromptLabTabController(onSelectDocument);

  return (
    <div className="prompt-lab-tab">
        <PromptLabRunForm
          documents={documents}
          folders={folders}
          selectedDocumentId={selectedDocumentId}
          methods={methodCatalog}
        concurrencyMax={experimentLimits.prompt_lab_max_concurrency}
        onRun={handleCreateRun}
        running={creatingRun}
        collapsed={setupCollapsed}
        onToggleCollapsed={handleToggleSetup}
      />

      <div className="prompt-lab-body">
        <aside className="prompt-lab-history">
          <div className="prompt-lab-history-header">
            <h3>Run History</h3>
            <button type="button" onClick={() => void refreshRuns()} disabled={runsLoading}>
              {runsLoading ? "..." : "Refresh"}
            </button>
          </div>
          <ul>
            {runs.map((run) => (
              <li key={run.id} className="prompt-lab-history-item">
                <button
                  type="button"
                  className={`prompt-lab-history-select ${selectedRunId === run.id ? "active" : ""}`}
                  onClick={() => handleSelectRun(run.id)}
                >
                  <div className="prompt-lab-history-name">{run.name}</div>
                  <div className="prompt-lab-history-meta">
                    {run.status} · {run.completed_tasks}/{run.total_tasks} ·{" "}
                    {formatMethodBundleLabel(run.method_bundle)}
                  </div>
                </button>
                <div className="prompt-lab-history-actions">
                  {run.cancellable && isActiveStatus(run.status) && (
                    <button
                      type="button"
                      className="prompt-lab-history-stop"
                      onClick={() => void handleStopRun(run)}
                      disabled={stoppingRunId === run.id || deletingRunId === run.id || run.status === "cancelling"}
                    >
                      {stoppingRunId === run.id || run.status === "cancelling" ? "Stopping..." : "Stop"}
                    </button>
                  )}
                  <button
                    type="button"
                    className="prompt-lab-history-delete"
                    onClick={() => void handleDeleteRun(run)}
                    disabled={deletingRunId === run.id || stoppingRunId === run.id}
                  >
                    {deletingRunId === run.id ? "..." : "Delete"}
                  </button>
                </div>
              </li>
            ))}
            {runs.length === 0 && <li className="prompt-lab-empty">No runs yet.</li>}
          </ul>
        </aside>

        <section className="prompt-lab-main">
          {runError && <div className="prompt-lab-error">{runError}</div>}
          {runDetail ? (
            <>
              {(runDetail.errors ?? []).length > 0 && (
                <div className="prompt-lab-error">
                  {runDetail.errors.join(" | ")}
                </div>
              )}
              {(runDetail.warnings ?? []).length > 0 && (
                <div className="prompt-lab-warning">
                  {runDetail.warnings.join(" | ")}
                </div>
              )}
              <PromptLabMatrix
                run={runDetail}
                experimentDiagnostics={experimentDiagnostics}
                selectedCellId={selectedCellId}
                onSelectCell={setSelectedCellId}
              />
              <PromptLabCellDetail
                run={runDetail}
                cell={selectedCell}
                onSelectCell={setSelectedCellId}
                selectedDocId={selectedDocId}
                onSelectDoc={setSelectedDocId}
                detail={docDetail}
                loading={docLoading}
                canRerunErrorDocs={!isActiveStatus(runDetail.status)}
                rerunningErrorDocs={rerunningErrorCellId === selectedCell?.id}
                onRerunErrorDocs={handleRerunErrorDocs}
              />
            </>
          ) : (
            <div className="prompt-lab-empty-state">Create or select a Prompt Lab run.</div>
          )}
        </section>
      </div>
      {toasts.length > 0 && (
        <div className="prompt-lab-toast-stack" aria-live="polite" aria-atomic="false">
          {toasts.map((toast) => (
            <button
              key={toast.id}
              type="button"
              className={`prompt-lab-toast ${toast.kind}`}
              onClick={() => dismissToast(toast.id)}
              aria-label={`Dismiss notification: ${toast.message}`}
            >
              {toast.message}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
