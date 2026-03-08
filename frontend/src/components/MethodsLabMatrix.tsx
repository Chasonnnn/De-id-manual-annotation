import { useEffect, useMemo, useState } from "react";
import type { MethodsLabMatrixCellSummary, MethodsLabRunDetail } from "../types";

interface Props {
  run: MethodsLabRunDetail;
  selectedCellId: string | null;
  onSelectCell: (cellId: string) => void;
}

const MICRO_METRIC_KEY = "__micro__";

function fmtPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function getMetricScore(
  cell: MethodsLabMatrixCellSummary | undefined,
  metricKey: string,
): number {
  if (!cell) return 0;
  if (metricKey === MICRO_METRIC_KEY) {
    return cell.micro.f1 ?? 0;
  }
  return cell.per_label[metricKey]?.f1 ?? 0;
}

function getHeatColor(score: number): string {
  const clamped = Math.max(0, Math.min(1, score));
  const hue = Math.round(clamped * 120);
  const lightness = 96 - clamped * 26;
  return `hsl(${hue} 62% ${lightness}%)`;
}

export default function MethodsLabMatrix({ run, selectedCellId, onSelectCell }: Props) {
  const [metricKey, setMetricKey] = useState<string>(MICRO_METRIC_KEY);
  const [collapsed, setCollapsed] = useState(false);
  const totalCells = run.matrix.models.length * run.matrix.methods.length;
  const totalRequests = run.total_tasks || run.doc_count * totalCells;

  useEffect(() => {
    const options = [MICRO_METRIC_KEY, ...(run.matrix.available_labels ?? [])];
    if (!options.includes(metricKey)) {
      setMetricKey(MICRO_METRIC_KEY);
    }
  }, [metricKey, run.matrix.available_labels]);

  const cellsById = useMemo(() => {
    const map = new Map<string, MethodsLabMatrixCellSummary>();
    for (const cell of run.matrix.cells) {
      map.set(cell.id, cell);
    }
    return map;
  }, [run.matrix.cells]);

  return (
    <section className={`prompt-lab-matrix ${collapsed ? "collapsed" : ""}`}>
      <div className="prompt-lab-matrix-header">
        <h3>Matrix Results</h3>
        <div className="prompt-lab-matrix-header-actions">
          <div className="prompt-lab-matrix-meta">
            Status: <strong>{run.status}</strong> · Progress: {run.progress.completed_tasks}/
            {run.progress.total_tasks}
          </div>
          <div className="prompt-lab-matrix-meta-secondary">
            Showing {totalCells} cells (model × method), aggregated from {totalRequests} requests
            (docs × methods × models).
          </div>
          <button
            type="button"
            className="prompt-lab-toggle-btn"
            onClick={() => setCollapsed((prev) => !prev)}
            aria-expanded={!collapsed}
          >
            {collapsed ? "Show Matrix" : "Hide Matrix"}
          </button>
        </div>
      </div>
      {!collapsed && (
        <>
          <div className="prompt-lab-matrix-controls">
            <label htmlFor="methods-lab-metric-select">Metric</label>
            <select
              id="methods-lab-metric-select"
              value={metricKey}
              onChange={(e) => setMetricKey(e.target.value)}
            >
              <option value={MICRO_METRIC_KEY}>Micro F1</option>
              {(run.matrix.available_labels ?? []).map((label) => (
                <option key={label} value={label}>
                  {label} F1
                </option>
              ))}
            </select>
          </div>
          <div className="prompt-lab-matrix-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model \ Method</th>
                  {run.matrix.methods.map((method) => (
                    <th key={method.id}>{method.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {run.matrix.models.map((model) => (
                  <tr key={model.id}>
                    <th>{model.label}</th>
                    {run.matrix.methods.map((method) => {
                      const cellId = `${model.id}__${method.id}`;
                      const cell = cellsById.get(cellId);
                      const active = selectedCellId === cellId;
                      const score = getMetricScore(cell, metricKey);
                      return (
                        <td key={cellId}>
                          <button
                            type="button"
                            className={`prompt-lab-matrix-cell ${active ? "active" : ""}`}
                            onClick={() => onSelectCell(cellId)}
                            style={{ background: getHeatColor(score) }}
                          >
                            <div className="prompt-lab-cell-status">{cell?.status ?? "pending"}</div>
                            <div className="prompt-lab-cell-f1">{fmtPct(score)}</div>
                            <div className="prompt-lab-cell-meta">
                              Docs {cell?.completed_docs ?? 0}/{cell?.total_docs ?? run.doc_count}
                            </div>
                            <div className="prompt-lab-cell-meta">
                              Errors {cell?.error_count ?? 0} · Conf{" "}
                              {cell?.mean_confidence != null ? fmtPct(cell.mean_confidence) : "N/A"}
                            </div>
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
