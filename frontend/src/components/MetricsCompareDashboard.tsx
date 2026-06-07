import type {
  MatchMode,
  MetricsCandidate,
  MetricsCandidateSource,
  MetricsCompareResult,
} from "../types";

interface Props {
  candidates: MetricsCandidate[];
  reference: MetricsCandidateSource;
  selectedHypotheses: MetricsCandidateSource[];
  matchMode: MatchMode;
  loading: boolean;
  result: MetricsCompareResult | null;
  onReferenceChange: (source: MetricsCandidateSource) => void;
  onHypothesesChange: (sources: MetricsCandidateSource[]) => void;
  onMatchModeChange: (mode: MatchMode) => void;
  onRefresh: () => void;
  onExportCsv: () => void;
  onOpenDocument: (source: MetricsCandidateSource, docId: string) => void;
}

function fmtPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function fmtNum(value: number): string {
  return value.toFixed(3);
}

function sortCandidatesForDefaults(candidates: MetricsCandidate[]): MetricsCandidate[] {
  return [...candidates].sort((a, b) => {
    const aRank = a.kind === "method_run" || a.kind === "methods_lab_cell" ? 0 : 1;
    const bRank = b.kind === "method_run" || b.kind === "methods_lab_cell" ? 0 : 1;
    if (aRank !== bRank) return aRank - bRank;
    return a.label.localeCompare(b.label);
  });
}

function sortReferenceCandidates(candidates: MetricsCandidate[]): MetricsCandidate[] {
  return [...candidates].sort((a, b) => {
    const rank = (candidate: MetricsCandidate) =>
      candidate.source === "manual" ? 0 : candidate.source === "pre" ? 1 : 2;
    const rankDiff = rank(a) - rank(b);
    return rankDiff === 0 ? a.label.localeCompare(b.label) : rankDiff;
  });
}

export default function MetricsCompareDashboard({
  candidates,
  reference,
  selectedHypotheses,
  matchMode,
  loading,
  result,
  onReferenceChange,
  onHypothesesChange,
  onMatchModeChange,
  onRefresh,
  onExportCsv,
  onOpenDocument,
}: Props) {
  const candidateOptions = sortCandidatesForDefaults(candidates);
  const referenceOptions = sortReferenceCandidates(candidates);
  const hypothesisSet = new Set(selectedHypotheses);
  const ranked = result
    ? [...result.hypotheses].sort((a, b) => b.micro.recall - a.micro.recall || b.micro.f1 - a.micro.f1)
    : [];
  const selectedSource = ranked[0]?.source ?? selectedHypotheses[0] ?? null;
  const selected = selectedSource
    ? result?.hypotheses.find((item) => item.source === selectedSource) ?? null
    : null;

  const toggleHypothesis = (source: MetricsCandidateSource) => {
    if (hypothesisSet.has(source)) {
      onHypothesesChange(selectedHypotheses.filter((item) => item !== source));
      return;
    }
    onHypothesesChange([...selectedHypotheses, source]);
  };

  return (
    <section className="compare-dashboard">
      <div className="compare-dashboard-toolbar">
        <label>
          Reference
          <select
            value={reference}
            onChange={(event) => onReferenceChange(event.target.value as MetricsCandidateSource)}
          >
            {referenceOptions.map((candidate) => (
              <option key={candidate.source} value={candidate.source}>
                {candidate.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Match
          <select value={matchMode} onChange={(event) => onMatchModeChange(event.target.value as MatchMode)}>
            <option value="overlap">Overlap</option>
            <option value="exact">Exact</option>
            <option value="boundary">Trim Space/Punct</option>
            <option value="substring">Substring</option>
          </select>
        </label>
        <button type="button" className="dashboard-tab-refresh" onClick={onRefresh} disabled={loading}>
          {loading ? "Comparing..." : "Refresh Comparison"}
        </button>
        <button type="button" className="dashboard-tab-refresh" onClick={onExportCsv} disabled={!result}>
          Export CSV
        </button>
      </div>

      <div className="compare-dashboard-body">
        <aside className="compare-candidate-picker">
          <h3>Hypotheses</h3>
          {candidateOptions
            .filter((candidate) => candidate.source !== reference)
            .map((candidate) => (
              <label key={candidate.source} className="compare-candidate-option">
                <input
                  type="checkbox"
                  checked={hypothesisSet.has(candidate.source)}
                  onChange={() => toggleHypothesis(candidate.source)}
                />
                <span>
                  <strong>{candidate.label}</strong>
                  <small>
                    {candidate.kind.replace(/_/g, " ")} - {candidate.document_count} docs
                  </small>
                </span>
              </label>
            ))}
          {candidateOptions.length <= 1 && <div className="dashboard-tab-empty">No saved outputs yet.</div>}
        </aside>

        <div className="compare-results">
          {!result && !loading && (
            <div className="dashboard-tab-empty">
              Select saved outputs, then refresh the comparison. Recall is ranked first because missed PHI is the highest-risk failure.
            </div>
          )}
          {loading && <div className="loading">Calculating comparison metrics...</div>}
          {result && (
            <>
              <div className="compare-leaderboard">
                {ranked.map((item) => (
                  <button
                    key={item.source}
                    type="button"
                    className={`compare-leader-card ${item.source === selectedSource ? "active" : ""}`}
                    onClick={() => onHypothesesChange([item.source, ...selectedHypotheses.filter((source) => source !== item.source)])}
                  >
                    <span>{item.label}</span>
                    <strong>{fmtPct(item.micro.recall)}</strong>
                    <small>
                      F1 {fmtPct(item.micro.f1)} - P {fmtPct(item.micro.precision)} - TP {item.micro.tp} / FP{" "}
                      {item.micro.fp} / FN {item.micro.fn}
                    </small>
                    <small>
                      Coverage {item.coverage.compared_documents}/{item.coverage.total_documents}
                      {item.coverage.skipped_documents > 0 ? ` - ${item.coverage.skipped_documents} skipped` : ""}
                    </small>
                  </button>
                ))}
              </div>

              {selected && (
                <>
                  <div className="compare-diagnostics">
                    <span>Exact F1 {fmtPct(selected.exact_micro.f1)}</span>
                    <span>Overlap F1 {fmtPct(selected.overlap_micro.f1)}</span>
                    <span>Gap {fmtPct(selected.exact_overlap_gap_f1)}</span>
                    <span>
                      Missed labels{" "}
                      {Object.entries(selected.missed_label_counts)
                        .map(([label, count]) => `${label}:${count}`)
                        .join(", ") || "none"}
                    </span>
                  </div>
                  <table className="per-label-table dashboard-table compare-doc-table">
                    <thead>
                      <tr>
                        <th>Document</th>
                        <th>Recall</th>
                        <th>F1</th>
                        <th>Precision</th>
                        <th>Exact F1</th>
                        <th>Overlap F1</th>
                        <th>Kappa</th>
                        <th>Matched IoU</th>
                        <th>Ref/Hyp</th>
                        <th>Open</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selected.documents.map((row) => (
                        <tr key={row.id}>
                          <td title={row.filename}>{row.filename}</td>
                          <td>{fmtPct(row.micro.recall)}</td>
                          <td>{fmtPct(row.micro.f1)}</td>
                          <td>{fmtPct(row.micro.precision)}</td>
                          <td>{fmtPct(row.exact_micro.f1)}</td>
                          <td>{fmtPct(row.overlap_micro.f1)}</td>
                          <td>{fmtNum(row.cohens_kappa)}</td>
                          <td>{fmtNum(row.matched_span_mean_iou)}</td>
                          <td>
                            {row.reference_count}/{row.hypothesis_count}
                          </td>
                          <td>
                            <button type="button" onClick={() => onOpenDocument(selected.source, row.id)}>
                              Open
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}
