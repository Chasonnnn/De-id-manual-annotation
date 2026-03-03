import { useState } from "react";
import type { DashboardMetricsResult } from "../types";

interface Props {
  dashboard: DashboardMetricsResult | null;
  loading: boolean;
  onRefresh: () => void;
  selectedId: string | null;
  onSelectDocument: (docId: string) => void;
}

export default function DashboardPanel({
  dashboard,
  loading,
  onRefresh,
  selectedId,
  onSelectDocument,
}: Props) {
  const [collapsed, setCollapsed] = useState(false);

  if (!dashboard && !loading) return null;

  const fmtPct = (value: number) => `${(value * 100).toFixed(1)}%`;
  const fmtNum = (value: number) => value.toFixed(3);
  const fmtConfidence = (value: number | null) =>
    value == null ? "N/A" : `${(value * 100).toFixed(1)}%`;

  return (
    <div className={`dashboard-panel ${collapsed ? "collapsed" : ""}`}>
      <div className="dashboard-header" onClick={() => setCollapsed(!collapsed)}>
        <span>Dashboard {collapsed ? "(click to expand)" : ""}</span>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
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
        <div className="dashboard-body">
          {loading && <div className="loading">Computing dashboard metrics...</div>}
          {dashboard && (
            <>
              <div className="dashboard-subtitle">
                Comparing <strong>{dashboard.reference}</strong> vs{" "}
                <strong>{dashboard.hypothesis}</strong> ({dashboard.match_mode})
              </div>

              <div className="metric-cards">
                <div className="metric-card">
                  <div className="card-label">Documents</div>
                  <div className="card-value">{dashboard.documents_compared}</div>
                  <div className="card-sub">
                    of {dashboard.total_documents} uploaded
                  </div>
                </div>
                <div className="metric-card">
                  <div className="card-label">Global Precision</div>
                  <div className="card-value">{fmtPct(dashboard.micro.precision)}</div>
                </div>
                <div className="metric-card">
                  <div className="card-label">Global Recall</div>
                  <div className="card-value">{fmtPct(dashboard.micro.recall)}</div>
                </div>
                <div className="metric-card">
                  <div className="card-label">Global F1</div>
                  <div className="card-value">{fmtPct(dashboard.micro.f1)}</div>
                  <div className="card-sub">
                    TP {dashboard.micro.tp} / FP {dashboard.micro.fp} / FN{" "}
                    {dashboard.micro.fn}
                  </div>
                </div>
                <div className="metric-card">
                  <div className="card-label">LLM Mean Confidence</div>
                  <div className="card-value">
                    {fmtConfidence(dashboard.llm_confidence_summary.mean_confidence)}
                  </div>
                  <div className="card-sub">
                    docs with score: {dashboard.llm_confidence_summary.documents_with_confidence}
                  </div>
                </div>
                <div className="metric-card">
                  <div className="card-label">Confidence Bands</div>
                  <div className="card-sub">
                    H {dashboard.llm_confidence_summary.band_counts.high} / M{" "}
                    {dashboard.llm_confidence_summary.band_counts.medium} / L{" "}
                    {dashboard.llm_confidence_summary.band_counts.low} / NA{" "}
                    {dashboard.llm_confidence_summary.band_counts.na}
                  </div>
                </div>
              </div>

              <table className="per-label-table dashboard-table">
                <thead>
                  <tr>
                    <th>Document</th>
                    <th>Micro P</th>
                    <th>Micro R</th>
                    <th>Micro F1</th>
                    <th>Macro F1</th>
                    <th>Kappa</th>
                    <th>IoU</th>
                    <th>LLM Conf</th>
                    <th>Ref/Hyp</th>
                  </tr>
                </thead>
                <tbody>
                  {dashboard.documents.map((row) => (
                    <tr
                      key={row.id}
                      className={selectedId === row.id ? "selected" : ""}
                      onClick={() => onSelectDocument(row.id)}
                      style={{ cursor: "pointer" }}
                    >
                      <td title={row.filename}>{row.filename}</td>
                      <td>{fmtPct(row.micro.precision)}</td>
                      <td>{fmtPct(row.micro.recall)}</td>
                      <td>{fmtPct(row.micro.f1)}</td>
                      <td>{fmtPct(row.macro.f1)}</td>
                      <td>{fmtNum(row.cohens_kappa)}</td>
                      <td>{fmtNum(row.mean_iou)}</td>
                      <td>
                        {row.llm_confidence?.confidence != null
                          ? fmtPct(row.llm_confidence.confidence)
                          : "N/A"}
                      </td>
                      <td>
                        {row.reference_count}/{row.hypothesis_count}
                      </td>
                    </tr>
                  ))}
                  {dashboard.documents.length === 0 && (
                    <tr>
                      <td colSpan={9} style={{ textAlign: "left" }}>
                        No comparable documents yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </>
          )}
        </div>
      )}
    </div>
  );
}
