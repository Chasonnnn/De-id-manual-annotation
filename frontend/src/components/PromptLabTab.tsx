import { useCallback, useEffect, useMemo, useState } from "react";
import {
  createPromptLabRun,
  getAgentMethods,
  getPromptLabDocResult,
  getPromptLabRun,
  listPromptLabRuns,
} from "../api/client";
import type {
  AgentMethodOption,
  DocumentSummary,
  PromptLabDocResult,
  PromptLabRunCreateRequest,
  PromptLabRunDetail,
  PromptLabRunSummary,
} from "../types";
import PromptLabCellDetail from "./PromptLabCellDetail";
import PromptLabMatrix from "./PromptLabMatrix";
import PromptLabRunForm from "./PromptLabRunForm";

interface Props {
  documents: DocumentSummary[];
  selectedDocumentId: string | null;
  onSelectDocument: (docId: string) => void;
}

function isActiveStatus(status: string): boolean {
  return status === "queued" || status === "running";
}

export default function PromptLabTab({
  documents,
  selectedDocumentId,
  onSelectDocument,
}: Props) {
  const [runs, setRuns] = useState<PromptLabRunSummary[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<PromptLabRunDetail | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [creatingRun, setCreatingRun] = useState(false);

  const [selectedCellId, setSelectedCellId] = useState<string | null>(null);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [docDetail, setDocDetail] = useState<PromptLabDocResult | null>(null);
  const [docLoading, setDocLoading] = useState(false);
  const [methodCatalog, setMethodCatalog] = useState<AgentMethodOption[]>([]);

  const refreshRuns = useCallback(async () => {
    setRunsLoading(true);
    try {
      const next = await listPromptLabRuns();
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
      const next = await getPromptLabRun(selectedRunId);
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
    getPromptLabDocResult(runDetail.id, selectedCellId, selectedDocId)
      .then((detail) => {
        setDocDetail(detail);
        if (selectedDocId) {
          onSelectDocument(selectedDocId);
        }
      })
      .catch((e: unknown) => setRunError(String(e)))
      .finally(() => setDocLoading(false));
  }, [onSelectDocument, runDetail, selectedCellId, selectedDocId]);

  const handleCreateRun = async (payload: PromptLabRunCreateRequest) => {
    setCreatingRun(true);
    try {
      const created = await createPromptLabRun(payload);
      setSelectedRunId(created.id);
      setRunDetail(created);
      setSelectedCellId(created.matrix.cells[0]?.id ?? null);
      setSelectedDocId(created.doc_ids[0] ?? null);
      setRunError(null);
      const list = await listPromptLabRuns();
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
      <PromptLabRunForm
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
              />
            </>
          ) : (
            <div className="prompt-lab-empty-state">Create or select a Prompt Lab run.</div>
          )}
        </section>
      </div>
    </div>
  );
}
