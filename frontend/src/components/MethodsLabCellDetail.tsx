import AnnotatedText from "./AnnotatedText";
import { computeDiff } from "./DiffOverlay";
import PaneContainer, { useSyncScroll } from "./PaneContainer";
import type {
  MethodsLabDocResult,
  MethodsLabMatrixCellSummary,
  MethodsLabRunDetail,
} from "../types";
import { getPrimaryMetricLabel, getPrimaryMetrics } from "../metricPresentation";
import { getExperimentModelLabelById } from "../modelDisplay";

interface Props {
  run: MethodsLabRunDetail;
  cell: MethodsLabMatrixCellSummary | null;
  onSelectCell: (cellId: string) => void;
  selectedDocId: string | null;
  onSelectDoc: (docId: string) => void;
  detail: MethodsLabDocResult | null;
  loading: boolean;
  canRerunErrorDocs: boolean;
  rerunningErrorDocs: boolean;
  onRerunErrorDocs: () => void | Promise<void>;
}

function fmtPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function isChunkWarning(message: string): boolean {
  const trimmed = message.trim();
  return trimmed.startsWith("Chunked ") || /^Chunk \d+\/\d+:/.test(trimmed);
}

export default function MethodsLabCellDetail({
  run,
  cell,
  onSelectCell,
  selectedDocId,
  onSelectDoc,
  detail,
  loading,
  canRerunErrorDocs,
  rerunningErrorDocs,
  onRerunErrorDocs,
}: Props) {
  const { registerPane, handleScroll } = useSyncScroll();

  if (!cell) {
    return <section className="prompt-lab-detail">Select a matrix cell to inspect details.</section>;
  }

  const handleModelChange = (modelId: string) => {
    const methodId = cell.method_id || run.matrix.methods[0]?.id;
    if (!methodId) return;
    onSelectCell(`${modelId}__${methodId}`);
  };

  const handleMethodChange = (methodId: string) => {
    const modelId = cell.model_id || run.matrix.models[0]?.id;
    if (!modelId) return;
    onSelectCell(`${modelId}__${methodId}`);
  };

  const referenceDiff =
    detail != null
      ? computeDiff(
          detail.hypothesis_spans,
          detail.reference_spans,
          run.runtime.match_mode,
        )
      : [];
  const hypothesisDiff =
    detail != null
      ? computeDiff(
          detail.reference_spans,
          detail.hypothesis_spans,
          run.runtime.match_mode,
        )
      : [];
  const allWarnings = detail?.warnings ?? [];
  const chunkWarnings = allWarnings.filter(isChunkWarning);
  const nonChunkWarnings = allWarnings.filter((message) => !isChunkWarning(message));
  const processedWithChunking = chunkWarnings.length > 0;
  const { primary, exact, usingOverlap } = getPrimaryMetrics(detail?.metrics ?? null, run.runtime.match_mode);
  const rerunErrorDocsDisabled =
    rerunningErrorDocs || cell.error_count === 0 || !canRerunErrorDocs;
  const rerunErrorDocsLabel =
    cell.error_count === 1 ? "Re-run 1 error doc" : `Re-run ${cell.error_count} error docs`;
  const rerunErrorDocsTitle =
    cell.error_count === 0
      ? "No error docs in this cell."
      : !canRerunErrorDocs
        ? "Wait for the run to finish before rerunning error docs."
        : undefined;
  const modelLabel = getExperimentModelLabelById(run.models, cell.model_id, cell.model_label);

  return (
    <section className="prompt-lab-detail">
      <div className="prompt-lab-detail-header">
        <div className="prompt-lab-detail-heading">
          <h3>
            Cell Detail: {modelLabel} × {cell.method_label}
          </h3>
          <div className="prompt-lab-detail-meta">
            {getPrimaryMetricLabel("F1", usingOverlap)} {fmtPct((cell.co_primary_metrics?.overlap?.micro.f1 ?? cell.micro.f1))} ·{" "}
            {getPrimaryMetricLabel("Recall", usingOverlap)} {fmtPct((cell.co_primary_metrics?.overlap?.micro.recall ?? cell.micro.recall))} · Completed{" "}
            {cell.completed_docs}/{cell.total_docs} · Errors {cell.error_count}
          </div>
        </div>
        <div className="prompt-lab-detail-actions">
          <button
            type="button"
            className="prompt-lab-detail-action-btn"
            disabled={rerunErrorDocsDisabled}
            onClick={() => void onRerunErrorDocs()}
            title={rerunErrorDocsTitle}
          >
            {rerunningErrorDocs ? "Re-running..." : rerunErrorDocsLabel}
          </button>
        </div>
      </div>

      <div className="prompt-lab-detail-controls">
        <label htmlFor="methods-lab-model-select">Model</label>
        <select
          id="methods-lab-model-select"
          value={cell.model_id}
          onChange={(e) => handleModelChange(e.target.value)}
        >
          {run.matrix.models.map((model) => (
            <option key={model.id} value={model.id}>
              {getExperimentModelLabelById(run.models, model.id, model.label)}
            </option>
          ))}
        </select>
        <label htmlFor="methods-lab-method-select">Method</label>
        <select
          id="methods-lab-method-select"
          value={cell.method_id}
          onChange={(e) => handleMethodChange(e.target.value)}
        >
          {run.matrix.methods.map((method) => (
            <option key={method.id} value={method.id}>
              {method.label}
            </option>
          ))}
        </select>
        <label htmlFor="methods-lab-doc-select">Document</label>
        <select
          id="methods-lab-doc-select"
          value={selectedDocId ?? ""}
          onChange={(e) => onSelectDoc(e.target.value)}
        >
          {run.doc_ids.map((docId) => (
            <option key={docId} value={docId}>
              {docId}
            </option>
          ))}
        </select>
      </div>

      {loading && <div className="prompt-lab-loading">Loading document-level result...</div>}

      {!loading && detail && (
        <>
          <div className="prompt-lab-detail-summary">
            <span>Status: {detail.status}</span>
            <span>Reference used: {detail.reference_source_used ?? "n/a"}</span>
            {primary && (
              <span>{getPrimaryMetricLabel("Micro F1", usingOverlap)}: {fmtPct(primary.micro.f1)}</span>
            )}
            {primary && (
              <span>{getPrimaryMetricLabel("Micro Recall", usingOverlap)}: {fmtPct(primary.micro.recall)}</span>
            )}
            {usingOverlap && exact && <span>Exact F1: {fmtPct(exact.micro.f1)}</span>}
            {processedWithChunking && <span className="chunk-badge">Processed with chunking</span>}
          </div>

          {detail.error && <div className="prompt-lab-error">{detail.error}</div>}

          {nonChunkWarnings.length > 0 && (
            <div className="prompt-lab-warning">{nonChunkWarnings.join(" | ")}</div>
          )}

          {primary && (
            <div className="metric-cards">
              <div className="metric-card">
                <div className="card-label">{getPrimaryMetricLabel("Micro P", usingOverlap)}</div>
                <div className="card-value">{fmtPct(primary.micro.precision)}</div>
              </div>
              <div className="metric-card">
                <div className="card-label">{getPrimaryMetricLabel("Micro R", usingOverlap)}</div>
                <div className="card-value">{fmtPct(primary.micro.recall)}</div>
              </div>
              <div className="metric-card">
                <div className="card-label">{getPrimaryMetricLabel("Micro F1", usingOverlap)}</div>
                <div className="card-value">{fmtPct(primary.micro.f1)}</div>
              </div>
              <div className="metric-card">
                <div className="card-label">{getPrimaryMetricLabel("Macro F1", usingOverlap)}</div>
                <div className="card-value">{fmtPct(primary.macro.f1)}</div>
              </div>
            </div>
          )}

          {detail.transcript_text && (
            <div className="prompt-lab-detail-pane-container">
              <PaneContainer>
                <div className="pane">
                  <div className="pane-header">
                    Reference ({detail.reference_source_used ?? "n/a"})
                  </div>
                  <div
                    className="pane-body"
                    ref={registerPane(0)}
                    onScroll={(e) => handleScroll(0)((e.target as HTMLDivElement).scrollTop)}
                  >
                    <AnnotatedText
                      text={detail.transcript_text}
                      spans={detail.reference_spans}
                      diffSpans={referenceDiff}
                    />
                  </div>
                </div>
                <div className="pane">
                  <div className="pane-header">Hypothesis</div>
                  <div
                    className="pane-body"
                    ref={registerPane(1)}
                    onScroll={(e) => handleScroll(1)((e.target as HTMLDivElement).scrollTop)}
                  >
                    <AnnotatedText
                      text={detail.transcript_text}
                      spans={detail.hypothesis_spans}
                      diffSpans={hypothesisDiff}
                    />
                  </div>
                </div>
              </PaneContainer>
            </div>
          )}

          <div className="prompt-lab-json-grid">
            <div>
              <h4>Hypothesis Spans</h4>
              <pre>{JSON.stringify(detail.hypothesis_spans, null, 2)}</pre>
            </div>
            <div>
              <h4>Reference Spans</h4>
              <pre>{JSON.stringify(detail.reference_spans, null, 2)}</pre>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
