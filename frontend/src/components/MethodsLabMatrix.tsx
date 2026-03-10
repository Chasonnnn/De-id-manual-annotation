import { useEffect, useMemo, useState } from "react";
import type {
  ExperimentDiagnostics,
  MethodsLabMatrixCellSummary,
  MethodsLabRunDetail,
} from "../types";
import { formatMethodBundleLabel } from "../experimentDisplay";
import { getPrimaryMatrixMetrics, getPrimaryMetricLabel } from "../metricPresentation";
import { getExperimentModelLabelById } from "../modelDisplay";

interface Props {
  run: MethodsLabRunDetail;
  experimentDiagnostics?: ExperimentDiagnostics | null;
  selectedCellId: string | null;
  onSelectCell: (cellId: string) => void;
}

const OVERALL_METRIC_KEY = "__micro__";
type MetricMeasure = "f1" | "recall" | "precision";

function fmtPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function getMetricScore(
  cell: MethodsLabMatrixCellSummary | undefined,
  metricKey: string,
  measure: MetricMeasure,
): number {
  if (!cell) return 0;
  const { primaryMicro, primaryPerLabel } = getPrimaryMatrixMetrics(cell);
  if (metricKey === OVERALL_METRIC_KEY) {
    return primaryMicro[measure] ?? 0;
  }
  return primaryPerLabel[metricKey]?.[measure] ?? 0;
}

function getMeasureLabel(measure: MetricMeasure): string {
  switch (measure) {
    case "precision":
      return "Precision";
    case "recall":
      return "Recall";
    default:
      return "F1";
  }
}

function getHeatColor(score: number): string {
  const clamped = Math.max(0, Math.min(1, score));
  const hue = Math.round(clamped * 120);
  const lightness = 96 - clamped * 26;
  return `hsl(${hue} 62% ${lightness}%)`;
}

function formatCatalogStatus(diagnostics: ExperimentDiagnostics | null): string {
  if (!diagnostics) return "status unavailable";
  if (diagnostics.gateway_catalog.reachable) {
    return diagnostics.gateway_catalog.model_count != null
      ? `status reachable · ${diagnostics.gateway_catalog.model_count} models`
      : "status reachable";
  }
  if (diagnostics.gateway_catalog.error) {
    return `status unavailable · ${diagnostics.gateway_catalog.error}`;
  }
  return "status unavailable";
}

export default function MethodsLabMatrix({
  run,
  experimentDiagnostics,
  selectedCellId,
  onSelectCell,
}: Props) {
  const [metricKey, setMetricKey] = useState<string>(OVERALL_METRIC_KEY);
  const [metricMeasure, setMetricMeasure] = useState<MetricMeasure>("f1");
  const [collapsed, setCollapsed] = useState(false);
  const totalCells = run.matrix.models.length * run.matrix.methods.length;
  const totalRequests = run.total_tasks || run.doc_count * totalCells;
  const taskLabel = run.diagnostics.total_tasks === 1 ? "task" : "tasks";
  const taskVerb = run.diagnostics.total_tasks === 1 ? "exists" : "exist";
  const workerLabel = run.diagnostics.effective_worker_count === 1 ? "worker" : "workers";
  const clampMessage = run.diagnostics.clamped_by_task_count
    ? `Only ${run.diagnostics.total_tasks} ${taskLabel} ${taskVerb} for this run, so the backend started ${run.diagnostics.effective_worker_count} ${workerLabel}.`
    : run.diagnostics.clamped_by_server_cap
      ? `The server cap is ${run.diagnostics.max_allowed_concurrency}, so the backend started ${run.diagnostics.effective_worker_count} ${workerLabel}.`
      : null;

  useEffect(() => {
    const options = [OVERALL_METRIC_KEY, ...(run.matrix.available_labels ?? [])];
    if (!options.includes(metricKey)) {
      setMetricKey(OVERALL_METRIC_KEY);
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
          <div className="prompt-lab-matrix-diagnostics" data-testid="methods-lab-matrix-diagnostics">
            <span>Requested {run.diagnostics.requested_concurrency}</span>
            <span>Effective {run.diagnostics.effective_worker_count}</span>
            <span>Tasks {run.diagnostics.total_tasks}</span>
            <span>Cap {run.diagnostics.max_allowed_concurrency}</span>
            <span>Bundle {formatMethodBundleLabel(run.method_bundle)}</span>
            <span>Gateway {run.diagnostics.api_base_host ?? experimentDiagnostics?.api_base_host ?? "n/a"}</span>
            <span>Catalog {formatCatalogStatus(experimentDiagnostics)}</span>
          </div>
          {clampMessage && <div className="prompt-lab-matrix-clamp-message">{clampMessage}</div>}
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
            <label htmlFor="methods-lab-measure-select">Measure</label>
            <select
              id="methods-lab-measure-select"
              value={metricMeasure}
              onChange={(e) => setMetricMeasure(e.target.value as MetricMeasure)}
            >
              <option value="f1">F1</option>
              <option value="recall">Recall</option>
              <option value="precision">Precision</option>
            </select>
            <label htmlFor="methods-lab-metric-select">Label</label>
            <select
              id="methods-lab-metric-select"
              value={metricKey}
              onChange={(e) => setMetricKey(e.target.value)}
            >
              <option value={OVERALL_METRIC_KEY}>Overall</option>
              {(run.matrix.available_labels ?? []).map((label) => (
                <option key={label} value={label}>
                  {label}
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
                    <th>{getExperimentModelLabelById(run.models, model.id, model.label)}</th>
                    {run.matrix.methods.map((method) => {
                      const cellId = `${model.id}__${method.id}`;
                      const cell = cellsById.get(cellId);
                      const active = selectedCellId === cellId;
                      const score = getMetricScore(cell, metricKey, metricMeasure);
                      const { usingOverlap, exactMicro } = getPrimaryMatrixMetrics(cell);
                      const selectedMetricLabel =
                        metricKey === OVERALL_METRIC_KEY
                          ? getPrimaryMetricLabel(`Overall ${getMeasureLabel(metricMeasure)}`, usingOverlap)
                          : getPrimaryMetricLabel(`${metricKey} ${getMeasureLabel(metricMeasure)}`, usingOverlap);
                      return (
                        <td key={cellId}>
                          <button
                            type="button"
                            className={`prompt-lab-matrix-cell ${active ? "active" : ""}`}
                            onClick={() => onSelectCell(cellId)}
                            style={{ background: getHeatColor(score) }}
                          >
                            <div className="prompt-lab-cell-status">{cell?.status ?? "pending"}</div>
                            <div className="prompt-lab-cell-metric-label">{selectedMetricLabel}</div>
                            <div className="prompt-lab-cell-score">{fmtPct(score)}</div>
                            <div className="prompt-lab-cell-meta">
                              Docs {cell?.completed_docs ?? 0}/{cell?.total_docs ?? run.doc_count}
                            </div>
                            <div className="prompt-lab-cell-meta">
                              {usingOverlap
                                ? `Exact F1 ${fmtPct(exactMicro.f1 ?? 0)} · Exact R ${fmtPct(exactMicro.recall ?? 0)}`
                                : `F1 ${fmtPct(cell?.micro.f1 ?? 0)} · R ${fmtPct(cell?.micro.recall ?? 0)}`}
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
