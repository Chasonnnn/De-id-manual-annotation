import AnnotatedText from "./AnnotatedText";
import { computeDiff } from "./DiffOverlay";
import PaneContainer, { useSyncScroll } from "./PaneContainer";
import type {
  PromptLabDocResult,
  PromptLabMatrixCellSummary,
  PromptLabRunDetail,
} from "../types";

interface Props {
  run: PromptLabRunDetail;
  cell: PromptLabMatrixCellSummary | null;
  onSelectCell: (cellId: string) => void;
  selectedDocId: string | null;
  onSelectDoc: (docId: string) => void;
  detail: PromptLabDocResult | null;
  loading: boolean;
}

function fmtPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export default function PromptLabCellDetail({
  run,
  cell,
  onSelectCell,
  selectedDocId,
  onSelectDoc,
  detail,
  loading,
}: Props) {
  const { registerPane, handleScroll } = useSyncScroll();

  if (!cell) {
    return <section className="prompt-lab-detail">Select a matrix cell to inspect details.</section>;
  }

  const handleModelChange = (modelId: string) => {
    const promptId = cell.prompt_id || run.matrix.prompts[0]?.id;
    if (!promptId) return;
    onSelectCell(`${modelId}__${promptId}`);
  };

  const handlePromptChange = (promptId: string) => {
    const modelId = cell.model_id || run.matrix.models[0]?.id;
    if (!modelId) return;
    onSelectCell(`${modelId}__${promptId}`);
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

  return (
    <section className="prompt-lab-detail">
      <div className="prompt-lab-detail-header">
        <h3>
          Cell Detail: {cell.model_label} × {cell.prompt_label}
        </h3>
        <div className="prompt-lab-detail-meta">
          F1 {fmtPct(cell.micro.f1)} · Completed {cell.completed_docs}/{cell.total_docs} · Errors {cell.error_count}
        </div>
      </div>

      <div className="prompt-lab-detail-controls">
        <label htmlFor="prompt-lab-model-select">Model</label>
        <select
          id="prompt-lab-model-select"
          value={cell.model_id}
          onChange={(e) => handleModelChange(e.target.value)}
        >
          {run.matrix.models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.label}
            </option>
          ))}
        </select>
        <label htmlFor="prompt-lab-prompt-select">Prompt</label>
        <select
          id="prompt-lab-prompt-select"
          value={cell.prompt_id}
          onChange={(e) => handlePromptChange(e.target.value)}
        >
          {run.matrix.prompts.map((prompt) => (
            <option key={prompt.id} value={prompt.id}>
              {prompt.label}
            </option>
          ))}
        </select>
        <label htmlFor="prompt-lab-doc-select">Document</label>
        <select
          id="prompt-lab-doc-select"
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
            {detail.metrics && <span>Micro F1: {fmtPct(detail.metrics.micro.f1)}</span>}
          </div>

          {detail.error && <div className="prompt-lab-error">{detail.error}</div>}

          {(detail.warnings ?? []).length > 0 && (
            <div className="prompt-lab-warning">{detail.warnings.join(" | ")}</div>
          )}

          {detail.metrics && (
            <div className="metric-cards">
              <div className="metric-card">
                <div className="card-label">Micro P</div>
                <div className="card-value">{fmtPct(detail.metrics.micro.precision)}</div>
              </div>
              <div className="metric-card">
                <div className="card-label">Micro R</div>
                <div className="card-value">{fmtPct(detail.metrics.micro.recall)}</div>
              </div>
              <div className="metric-card">
                <div className="card-label">Micro F1</div>
                <div className="card-value">{fmtPct(detail.metrics.micro.f1)}</div>
              </div>
              <div className="metric-card">
                <div className="card-label">Macro F1</div>
                <div className="card-value">{fmtPct(detail.metrics.macro.f1)}</div>
              </div>
            </div>
          )}

          {detail.transcript_text && (
            <div className="prompt-lab-detail-pane-container">
              <PaneContainer>
                <div className="pane">
                  <div className="pane-header">Reference ({detail.reference_source_used ?? "n/a"})</div>
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
