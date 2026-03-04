import React, { useState } from "react";
import type { MetricsResult } from "../types";
import { getLabelColor } from "../types";

interface Props {
  reference: string;
  hypothesis: string;
  matchMode: string;
  metrics: MetricsResult | null;
  loading: boolean;
  onRefresh: () => void;
}

export default function MetricsPanel({
  reference,
  hypothesis,
  matchMode,
  metrics,
  loading,
  onRefresh,
}: Props) {
  const [collapsed, setCollapsed] = useState(false);

  const fmt = (v: number) => (v * 100).toFixed(1) + "%";
  const confusion = metrics?.confusion_matrix;
  const confidence = metrics?.llm_confidence ?? null;
  const confidencePct =
    confidence?.confidence != null ? `${(confidence.confidence * 100).toFixed(1)}%` : "N/A";
  const confidenceBand = confidence?.band.toUpperCase() ?? "N/A";
  const meanLogprob =
    confidence?.mean_logprob != null ? confidence.mean_logprob.toFixed(4) : "N/A";
  const perplexity =
    confidence?.perplexity != null ? confidence.perplexity.toFixed(3) : "N/A";

  return (
    <div className={`metrics-panel ${collapsed ? "collapsed" : ""}`}>
      <div className="metrics-header" onClick={() => setCollapsed(!collapsed)}>
        <span>Metrics {collapsed ? "(click to expand)" : ""}</span>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onRefresh();
            }}
            style={{
              fontSize: 11,
              padding: "2px 8px",
              border: "1px solid #ccc",
              borderRadius: 3,
              background: "#fff",
              cursor: "pointer",
            }}
          >
            Refresh
          </button>
          <span>{collapsed ? "+" : "-"}</span>
        </div>
      </div>

      {!collapsed && (
        <div className="metrics-body">
          {loading && <div className="loading">Computing metrics...</div>}
          {!loading && !metrics && (
            <div className="dashboard-subtitle">
              No per-document metrics yet. Click <strong>Refresh</strong>.
            </div>
          )}
          {metrics && (
            <>
              <div className="metrics-subtitle">
                Comparing <strong>{reference}</strong> vs <strong>{hypothesis}</strong>{" "}
                ({matchMode})
              </div>
              {confidence && (
                <div className={`confidence-summary band-${confidence.band}`}>
                  <strong>LLM Confidence:</strong> {confidencePct} ({confidenceBand}){" "}
                  <span style={{ marginLeft: 8 }}>
                    tokens={confidence.token_count}
                  </span>
                  {!confidence.available && (
                    <span style={{ marginLeft: 8 }}>
                      status={confidence.reason}
                    </span>
                  )}
                  <span style={{ marginLeft: 8 }}>mean_logprob={meanLogprob}</span>
                  <span style={{ marginLeft: 8 }}>perplexity={perplexity}</span>
                </div>
              )}
              <div className="metric-cards">
                <div className="metric-card">
                  <div className="card-label">Micro P</div>
                  <div className="card-value">{fmt(metrics.micro.precision)}</div>
                </div>
                <div className="metric-card">
                  <div className="card-label">Micro R</div>
                  <div className="card-value">{fmt(metrics.micro.recall)}</div>
                </div>
                <div className="metric-card">
                  <div className="card-label">Micro F1</div>
                  <div className="card-value">{fmt(metrics.micro.f1)}</div>
                </div>
                <div className="metric-card">
                  <div className="card-label">Macro P</div>
                  <div className="card-value">{fmt(metrics.macro.precision)}</div>
                </div>
                <div className="metric-card">
                  <div className="card-label">Macro R</div>
                  <div className="card-value">{fmt(metrics.macro.recall)}</div>
                </div>
                <div className="metric-card">
                  <div className="card-label">Macro F1</div>
                  <div className="card-value">{fmt(metrics.macro.f1)}</div>
                </div>
              </div>

              <table className="per-label-table">
                <thead>
                  <tr>
                    <th>Label</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>Support</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(metrics.per_label).map(([label, m]) => (
                    <tr key={label}>
                      <td>
                        <span
                          style={{
                            display: "inline-block",
                            width: 10,
                            height: 10,
                            borderRadius: 2,
                            background: getLabelColor(label),
                            marginRight: 6,
                            verticalAlign: "middle",
                          }}
                        />
                        {label}
                      </td>
                      <td>{fmt(m.precision)}</td>
                      <td>{fmt(m.recall)}</td>
                      <td>{fmt(m.f1)}</td>
                      <td>{m.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {confusion && (
                <div className="confusion-matrix">
                  <h4>Confusion Matrix</h4>
                  <div
                    className="confusion-grid"
                    style={{
                      gridTemplateColumns: `60px repeat(${confusion.labels.length}, 40px)`,
                    }}
                  >
                    <div className="confusion-cell" />
                    {confusion.labels.map((l) => (
                      <div key={l} className="confusion-cell header">
                        {l}
                      </div>
                    ))}
                    {confusion.matrix.map((row, ri) => (
                      <React.Fragment key={ri}>
                        <div
                          key={`rh-${ri}`}
                          className="confusion-cell row-header"
                        >
                          {confusion.labels[ri]}
                        </div>
                        {row.map((val, ci) => {
                          const maxVal = Math.max(
                            ...confusion.matrix.flat(),
                            1,
                          );
                          const intensity = val / maxVal;
                          return (
                            <div
                              key={`${ri}-${ci}`}
                              className="confusion-cell"
                              style={{
                                background: `rgba(74, 108, 247, ${intensity * 0.8})`,
                                color: intensity > 0.5 ? "#fff" : "#333",
                              }}
                            >
                              {val}
                            </div>
                          );
                        })}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              )}

              <div style={{ display: "flex", gap: 24 }}>
                {(metrics.false_positives ?? []).length > 0 && (
                  <div className="fp-fn-section" style={{ flex: 1 }}>
                    <h4>
                      False Positives ({(metrics.false_positives ?? []).length})
                    </h4>
                    <ul className="fp-fn-list">
                      {(metrics.false_positives ?? []).map((s, i) => (
                        <li key={i}>
                          <span
                            className="span-label"
                            style={{
                              background: getLabelColor(s.label),
                            }}
                          >
                            {s.label}
                          </span>
                          &quot;{s.text}&quot; [{s.start}:{s.end}]
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {(metrics.false_negatives ?? []).length > 0 && (
                  <div className="fp-fn-section" style={{ flex: 1 }}>
                    <h4>
                      False Negatives ({(metrics.false_negatives ?? []).length})
                    </h4>
                    <ul className="fp-fn-list">
                      {(metrics.false_negatives ?? []).map((s, i) => (
                        <li key={i}>
                          <span
                            className="span-label"
                            style={{
                              background: getLabelColor(s.label),
                            }}
                          >
                            {s.label}
                          </span>
                          &quot;{s.text}&quot; [{s.start}:{s.end}]
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
