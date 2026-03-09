import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createMethodsLabRun,
  deleteMethodsLabRun,
  getAgentMethods,
  getExperimentLimits,
  getMethodsLabDocResult,
  getMethodsLabRun,
  listMethodsLabRuns,
  stopMethodsLabRun,
} from "../api/client";
import { DEFAULT_EXPERIMENT_LIMITS } from "../types";
import type {
  AgentMethodOption,
  DocumentSummary,
  ExperimentLimits,
  MethodsLabDocResult,
  MethodsLabRunCreateRequest,
  MethodsLabRunDetail,
  MethodsLabRunSummary,
} from "../types";
import MethodsLabCellDetail from "./MethodsLabCellDetail";
import MethodsLabMatrix from "./MethodsLabMatrix";
import MethodsLabRunForm from "./MethodsLabRunForm";

interface Props {
  documents: DocumentSummary[];
  selectedDocumentId: string | null;
  onSelectDocument: (docId: string) => void;
}

type MethodsLabToastKind = "success" | "error" | "info";

interface MethodsLabToast {
  id: string;
  kind: MethodsLabToastKind;
  message: string;
}

function isActiveStatus(status: string): boolean {
  return status === "queued" || status === "running" || status === "cancelling";
}

export default function MethodsLabTab({
  documents,
  selectedDocumentId,
  onSelectDocument,
}: Props) {
  const [runs, setRuns] = useState<MethodsLabRunSummary[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [deletingRunId, setDeletingRunId] = useState<string | null>(null);
  const [stoppingRunId, setStoppingRunId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<MethodsLabRunDetail | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [creatingRun, setCreatingRun] = useState(false);

  const [selectedCellId, setSelectedCellId] = useState<string | null>(null);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [docDetail, setDocDetail] = useState<MethodsLabDocResult | null>(null);
  const [docLoading, setDocLoading] = useState(false);
  const [methodCatalog, setMethodCatalog] = useState<AgentMethodOption[]>([]);
  const [experimentLimits, setExperimentLimits] = useState<ExperimentLimits>(
    DEFAULT_EXPERIMENT_LIMITS,
  );
  const [toasts, setToasts] = useState<MethodsLabToast[]>([]);
  const previousRunStatusRef = useRef<Record<string, string>>({});
  const previousCellCompletedRef = useRef<Record<string, number>>({});
  const previousCellErrorsRef = useRef<Record<string, number>>({});
  const previousDocStatusRef = useRef<Record<string, string>>({});
  const latestRunDetailRequestRef = useRef<string | null>(null);
  const latestDocRequestRef = useRef<string | null>(null);
  const latestDocDetailRef = useRef<MethodsLabDocResult | null>(null);

  const pushToast = useCallback((kind: MethodsLabToastKind, message: string) => {
    const id = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    setToasts((prev) => [...prev, { id, kind, message }]);
    window.setTimeout(() => {
      setToasts((prev) => prev.filter((item) => item.id !== id));
    }, 4500);
  }, []);

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const clearSelectedDocState = useCallback(() => {
    latestDocRequestRef.current = null;
    setSelectedCellId(null);
    setSelectedDocId(null);
    setDocDetail(null);
    setDocLoading(false);
  }, []);

  const refreshRuns = useCallback(async () => {
    setRunsLoading(true);
    try {
      const next = await listMethodsLabRuns();
      setRuns(next);
      setSelectedRunId((current) => {
        if (next.length === 0) {
          return null;
        }
        if (!current || !next.some((run) => run.id === current)) {
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
      const next = await getMethodsLabRun(requestRunId);
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

  useEffect(() => {
    void refreshRunDetail();
  }, [refreshRunDetail]);

  useEffect(() => {
    if (!runDetail) return;
    const runId = runDetail.id;
    const previousRunStatus = previousRunStatusRef.current[runId];
    if (previousRunStatus && previousRunStatus !== runDetail.status) {
      if (runDetail.status === "completed") {
        pushToast("success", `Methods Lab run completed: ${runDetail.name}.`);
      } else if (runDetail.status === "cancelled") {
        pushToast("info", `Methods Lab run stopped: ${runDetail.name}.`);
      } else if (runDetail.status === "completed_with_errors" || runDetail.status === "failed") {
        pushToast(
          "error",
          `Methods Lab run ended with issues: ${runDetail.name} (${runDetail.status}).`,
        );
      }
    }
    previousRunStatusRef.current[runId] = runDetail.status;

    for (const cell of runDetail.matrix.cells) {
      const cellKey = `${runId}::${cell.id}`;
      const previousCompleted = previousCellCompletedRef.current[cellKey];
      if (typeof previousCompleted === "number" && cell.completed_docs > previousCompleted) {
        const delta = cell.completed_docs - previousCompleted;
        pushToast(
          "info",
          `Cell ${cell.model_label} × ${cell.method_label}: +${delta} doc(s) completed (${cell.completed_docs}/${cell.total_docs}).`,
        );
      }
      previousCellCompletedRef.current[cellKey] = cell.completed_docs;

      const previousErrors = previousCellErrorsRef.current[cellKey];
      if (typeof previousErrors === "number" && cell.error_count > previousErrors) {
        const delta = cell.error_count - previousErrors;
        pushToast(
          "error",
          `Cell ${cell.model_label} × ${cell.method_label}: +${delta} new error(s) (${cell.error_count} total).`,
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

  useEffect(() => {
    if (!runDetail || !selectedCellId || !selectedDocId) {
      latestDocRequestRef.current = null;
      setDocDetail(null);
      setDocLoading(false);
      return;
    }
    if (!runDetail.matrix.cells.some((cell) => cell.id === selectedCellId)) {
      latestDocRequestRef.current = null;
      setDocDetail(null);
      setDocLoading(false);
      return;
    }
    if (!runDetail.doc_ids.includes(selectedDocId)) {
      latestDocRequestRef.current = null;
      setDocDetail(null);
      setDocLoading(false);
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
    if (previousRequestKey !== requestKey || !hasVisibleDetailForSelection) {
      setDocLoading(true);
    }
    getMethodsLabDocResult(runDetail.id, selectedCellId, selectedDocId)
      .then((detail) => {
        if (!active || latestDocRequestRef.current !== requestKey) {
          return;
        }
        setDocDetail(detail);
        setRunError(null);
        const key = `${detail.run_id}::${detail.cell_id}::${detail.doc_id}`;
        const previousStatus = previousDocStatusRef.current[key];
        if (previousStatus && previousStatus !== detail.status) {
          const filename = detail.document.filename || detail.document.id;
          const modelLabel = detail.model?.label ?? detail.model?.model ?? "model";
          const methodLabel = detail.method?.label ?? "method";
          if (detail.status === "completed") {
            pushToast(
              "success",
              `Doc completed: ${filename} (${modelLabel} × ${methodLabel}).`,
            );
          } else if (detail.status === "cancelled") {
            pushToast(
              "info",
              `Doc cancelled: ${filename} (${modelLabel} × ${methodLabel}).`,
            );
          } else if (detail.status === "failed" || detail.status === "unavailable") {
            pushToast(
              "error",
              `Doc failed: ${filename} (${modelLabel} × ${methodLabel}).`,
            );
          }
        }
        previousDocStatusRef.current[key] = detail.status;
        onSelectDocument(detail.doc_id);
      })
      .catch((e: unknown) => {
        if (!active || latestDocRequestRef.current !== requestKey) {
          return;
        }
        setDocDetail(null);
        setRunError(String(e));
      })
      .finally(() => {
        if (!active || latestDocRequestRef.current !== requestKey) {
          return;
        }
        setDocLoading(false);
      });
    return () => {
      active = false;
    };
  }, [onSelectDocument, pushToast, runDetail, selectedCellId, selectedDocId]);

  const handleCreateRun = async (payload: MethodsLabRunCreateRequest) => {
    setCreatingRun(true);
    try {
      const created = await createMethodsLabRun(payload);
      setSelectedRunId(created.id);
      setRunDetail(created);
      setSelectedCellId(created.matrix.cells[0]?.id ?? null);
      setSelectedDocId(created.doc_ids[0] ?? null);
      setRunError(null);
      const list = await listMethodsLabRuns();
      setRuns(list);
    } finally {
      setCreatingRun(false);
    }
  };

  const handleDeleteRun = useCallback(
    async (run: MethodsLabRunSummary) => {
      const confirmed = window.confirm(`Delete Methods Lab run "${run.name}"?`);
      if (!confirmed) return;

      setDeletingRunId(run.id);
      try {
        await deleteMethodsLabRun(run.id);
        const next = await listMethodsLabRuns();
        setRuns(next);
        const nextSelectedRunId =
          selectedRunId === run.id || !next.some((item) => item.id === selectedRunId)
            ? (next[0]?.id ?? null)
            : selectedRunId;
        setSelectedRunId(nextSelectedRunId);
        if (selectedRunId === run.id) {
          setRunDetail(null);
          clearSelectedDocState();
        }
        setRunError(null);
        pushToast("success", `Deleted Methods Lab run: ${run.name}.`);
      } catch (e: unknown) {
        const message = String(e);
        setRunError(message);
        pushToast("error", `Failed to delete Methods Lab run: ${run.name}.`);
      } finally {
        setDeletingRunId(null);
      }
    },
    [clearSelectedDocState, pushToast, selectedRunId],
  );

  const handleStopRun = useCallback(
    async (run: MethodsLabRunSummary) => {
      const confirmed = window.confirm(`Stop Methods Lab run "${run.name}"?`);
      if (!confirmed) return;

      setStoppingRunId(run.id);
      try {
        await stopMethodsLabRun(run.id);
        await Promise.all([
          refreshRuns(),
          selectedRunId === run.id ? refreshRunDetail() : Promise.resolve(),
        ]);
        setRunError(null);
        pushToast("info", `Stop requested for Methods Lab run: ${run.name}.`);
      } catch (e: unknown) {
        const message = String(e);
        setRunError(message);
        pushToast("error", `Failed to stop Methods Lab run: ${run.name}.`);
      } finally {
        setStoppingRunId(null);
      }
    },
    [pushToast, refreshRunDetail, refreshRuns, selectedRunId],
  );

  const selectedCell = useMemo(
    () => runDetail?.matrix.cells.find((cell) => cell.id === selectedCellId) ?? null,
    [runDetail, selectedCellId],
  );

  return (
    <div className="prompt-lab-tab">
      <MethodsLabRunForm
        documents={documents}
        selectedDocumentId={selectedDocumentId}
        methods={methodCatalog}
        concurrencyMax={experimentLimits.methods_lab_max_concurrency}
        onRun={handleCreateRun}
        running={creatingRun}
        forceCollapsed={Boolean(runDetail)}
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
                  onClick={() => setSelectedRunId(run.id)}
                >
                  <div className="prompt-lab-history-name">{run.name}</div>
                  <div className="prompt-lab-history-meta">
                    {run.status} · {run.completed_tasks}/{run.total_tasks}
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
                <div className="prompt-lab-error">{runDetail.errors.join(" | ")}</div>
              )}
              {(runDetail.warnings ?? []).length > 0 && (
                <div className="prompt-lab-warning">{runDetail.warnings.join(" | ")}</div>
              )}
              <MethodsLabMatrix
                run={runDetail}
                selectedCellId={selectedCellId}
                onSelectCell={setSelectedCellId}
              />
              <MethodsLabCellDetail
                run={runDetail}
                cell={selectedCell}
                onSelectCell={setSelectedCellId}
                selectedDocId={selectedDocId}
                onSelectDoc={setSelectedDocId}
                detail={docDetail}
                loading={docLoading}
              />
            </>
          ) : (
            <div className="prompt-lab-empty-state">Create or select a Methods Lab run.</div>
          )}
        </section>
      </div>
      {toasts.length > 0 && (
        <div className="prompt-lab-toast-stack" aria-live="polite" aria-atomic="false">
          {toasts.map((toast) => (
            <div
              key={toast.id}
              className={`prompt-lab-toast ${toast.kind}`}
              onClick={() => dismissToast(toast.id)}
              role="status"
            >
              {toast.message}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
