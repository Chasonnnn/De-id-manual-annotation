import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createMethodsLabRun,
  getAgentMethods,
  getMethodsLabDocResult,
  getMethodsLabRun,
  listMethodsLabRuns,
} from "../api/client";
import type {
  AgentMethodOption,
  DocumentSummary,
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
  return status === "queued" || status === "running";
}

export default function MethodsLabTab({
  documents,
  selectedDocumentId,
  onSelectDocument,
}: Props) {
  const [runs, setRuns] = useState<MethodsLabRunSummary[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<MethodsLabRunDetail | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [creatingRun, setCreatingRun] = useState(false);

  const [selectedCellId, setSelectedCellId] = useState<string | null>(null);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [docDetail, setDocDetail] = useState<MethodsLabDocResult | null>(null);
  const [docLoading, setDocLoading] = useState(false);
  const [methodCatalog, setMethodCatalog] = useState<AgentMethodOption[]>([]);
  const [toasts, setToasts] = useState<MethodsLabToast[]>([]);
  const previousRunStatusRef = useRef<Record<string, string>>({});
  const previousCellCompletedRef = useRef<Record<string, number>>({});
  const previousCellErrorsRef = useRef<Record<string, number>>({});
  const previousDocStatusRef = useRef<Record<string, string>>({});

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

  const refreshRuns = useCallback(async () => {
    setRunsLoading(true);
    try {
      const next = await listMethodsLabRuns();
      setRuns(next);
      if (!selectedRunId && next.length > 0) {
        setSelectedRunId(next[0]?.id ?? null);
      }
    } catch (e: unknown) {
      setRunError(String(e));
    } finally {
      setRunsLoading(false);
    }
  }, [selectedRunId]);

  useEffect(() => {
    void refreshRuns();
  }, [refreshRuns]);

  useEffect(() => {
    getAgentMethods()
      .then(setMethodCatalog)
      .catch(() => setMethodCatalog([]));
  }, []);

  const refreshRunDetail = useCallback(async () => {
    if (!selectedRunId) {
      setRunDetail(null);
      return;
    }
    try {
      const next = await getMethodsLabRun(selectedRunId);
      setRunDetail(next);
      setRunError(null);
    } catch (e: unknown) {
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
      setDocDetail(null);
      return;
    }
    setDocLoading(true);
    getMethodsLabDocResult(runDetail.id, selectedCellId, selectedDocId)
      .then((detail) => {
        setDocDetail(detail);
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
          } else if (detail.status === "failed" || detail.status === "unavailable") {
            pushToast(
              "error",
              `Doc failed: ${filename} (${modelLabel} × ${methodLabel}).`,
            );
          }
        }
        previousDocStatusRef.current[key] = detail.status;
        if (selectedDocId) {
          onSelectDocument(selectedDocId);
        }
      })
      .catch((e: unknown) => setRunError(String(e)))
      .finally(() => setDocLoading(false));
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
              <li key={run.id}>
                <button
                  type="button"
                  className={selectedRunId === run.id ? "active" : ""}
                  onClick={() => setSelectedRunId(run.id)}
                >
                  <div className="prompt-lab-history-name">{run.name}</div>
                  <div className="prompt-lab-history-meta">
                    {run.status} · {run.completed_tasks}/{run.total_tasks}
                  </div>
                </button>
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
