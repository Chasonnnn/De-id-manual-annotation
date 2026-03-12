import React, { useState } from "react";
import { getPrimaryMetricLabel, getPrimaryMetrics } from "../metricPresentation";
import type { AnnotationSource, MatchMode, MetricsResult } from "../types";
import { getLabelColor } from "../types";

interface Props {
  reference: AnnotationSource;
  hypothesis: AnnotationSource;
  matchMode: MatchMode;
  sourceOptions: Array<{ value: AnnotationSource; label: string }>;
  metrics: MetricsResult | null;
  loading: boolean;
  onRefresh: () => void;
  onReferenceChange: (ref: AnnotationSource) => void;
  onHypothesisChange: (hyp: AnnotationSource) => void;
  onMatchModeChange: (mode: MatchMode) => void;
}

type SourceOption = Props["sourceOptions"][number];
type PrimaryMetrics = ReturnType<typeof getPrimaryMetrics>["primary"];
type ExactMetrics = ReturnType<typeof getPrimaryMetrics>["exact"];


const SECTION_ROW_STYLE = { display: "flex", gap: 24 } as const;
const LABEL_SWATCH_STYLE = {
  display: "inline-block",
  width: 10,
  height: 10,
  borderRadius: 2,
  marginRight: 6,
  verticalAlign: "middle",
} as const;

function spanItemKey(span: { label: string; start: number; end: number; text: string }): string {
  return `${span.label}:${span.start}:${span.end}:${span.text}`;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatConfidencePercent(value: number | null | undefined): string {
  return value != null ? `${(value * 100).toFixed(1)}%` : "N/A";
}

function formatMetricNumber(value: number | null | undefined, digits: number): string {
  return value != null ? value.toFixed(digits) : "N/A";
}

function MetricsHeader({
  collapsed,
  onToggle,
  onRefresh,
}: {
  collapsed: boolean;
  onToggle: () => void;
  onRefresh: () => void;
}) {
  return (
    <div
      className="metrics-header"
      role="button"
      tabIndex={0}
      onClick={onToggle}
      onKeyDown={(e) => {
        if (e.key !== "Enter" && e.key !== " ") {
          return;
        }
        e.preventDefault();
        onToggle();
      }}
    >
      <span>Metrics {collapsed ? "(click to expand)" : ""}</span>
      <div style={{ display: "flex", gap: 8 }}>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRefresh();
          }}
          className="panel-refresh-btn"
        >
          Refresh
        </button>
        <span>{collapsed ? "+" : "-"}</span>
      </div>
    </div>
  );
}

function MetricsCompareControls({
  reference,
  hypothesis,
  matchMode,
  sourceOptions,
  onReferenceChange,
  onHypothesisChange,
  onMatchModeChange,
}: {
  reference: AnnotationSource;
  hypothesis: AnnotationSource;
  matchMode: MatchMode;
  sourceOptions: SourceOption[];
  onReferenceChange: (ref: AnnotationSource) => void;
  onHypothesisChange: (hyp: AnnotationSource) => void;
  onMatchModeChange: (mode: MatchMode) => void;
}) {
  return (
    <div className="metrics-subtitle">
      <div className="metrics-compare-controls">
        <label>
          Comparing
          <select
            value={reference}
            onChange={(e) => onReferenceChange(e.target.value as AnnotationSource)}
          >
            {sourceOptions.map((option) => (
              <option key={`metrics-ref-${option.value}`} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <span>vs</span>
        <label>
          <select
            value={hypothesis}
            onChange={(e) => onHypothesisChange(e.target.value as AnnotationSource)}
          >
            {sourceOptions.map((option) => (
              <option key={`metrics-hyp-${option.value}`} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="metrics-swap-btn"
          onClick={() => {
            onReferenceChange(hypothesis);
            onHypothesisChange(reference);
          }}
          title="Swap reference and hypothesis"
        >
          Swap
        </button>
        <label>
          Match
          <select
            value={matchMode}
            onChange={(e) => onMatchModeChange(e.target.value as MatchMode)}
          >
            <option value="exact">Exact</option>
            <option value="boundary">Trim Space/Punct</option>
            <option value="overlap">Overlap</option>
          </select>
        </label>
      </div>
    </div>
  );
}

function ConfidenceSummary({
  confidence,
  confidencePct,
  confidenceBand,
  meanLogprob,
  perplexity,
}: {
  confidence: NonNullable<MetricsResult["llm_confidence"]>;
  confidencePct: string;
  confidenceBand: string;
  meanLogprob: string;
  perplexity: string;
}) {
  return (
    <div className={`confidence-summary band-${confidence.band}`}>
      <strong>LLM Confidence:</strong> {confidencePct} ({confidenceBand}){" "}
      <span style={{ marginLeft: 8 }}>tokens={confidence.token_count}</span>
      {!confidence.available && (
        <span style={{ marginLeft: 8 }}>status={confidence.reason}</span>
      )}
      <span style={{ marginLeft: 8 }}>mean_logprob={meanLogprob}</span>
      <span style={{ marginLeft: 8 }}>perplexity={perplexity}</span>
    </div>
  );
}

function ExactDiagnostic({
  exact,
}: {
  exact: NonNullable<ExactMetrics>;
}) {
  return (
    <div className="metrics-diagnostic-note">
      Exact diagnostic: P {formatPercent(exact.micro.precision)} · R{" "}
      {formatPercent(exact.micro.recall)} · F1 {formatPercent(exact.micro.f1)}
    </div>
  );
}

function scoreClass(value: number): string {
  if (value >= 0.8) return "score-good";
  if (value >= 0.5) return "score-fair";
  return "score-poor";
}

function MetricCards({
  primary,
  usingOverlap,
}: {
  primary: PrimaryMetrics;
  usingOverlap: boolean;
}) {
  const microF1 = primary?.micro.f1 ?? 0;
  const macroF1 = primary?.macro.f1 ?? 0;

  return (
    <div className="metric-cards">
      <div className={`metric-card metric-card-hero ${scoreClass(microF1)}`}>
        <div className="card-label">
          {getPrimaryMetricLabel("Micro F1", usingOverlap)}
        </div>
        <div className="card-value">{formatPercent(microF1)}</div>
        <div className="card-sub">
          TP {primary?.micro.tp ?? 0} / FP {primary?.micro.fp ?? 0} / FN {primary?.micro.fn ?? 0}
        </div>
      </div>
      <div className="metric-card">
        <div className="card-label">
          {getPrimaryMetricLabel("Micro P", usingOverlap)}
        </div>
        <div className="card-value">{formatPercent(primary?.micro.precision ?? 0)}</div>
      </div>
      <div className="metric-card">
        <div className="card-label">
          {getPrimaryMetricLabel("Micro R", usingOverlap)}
        </div>
        <div className="card-value">{formatPercent(primary?.micro.recall ?? 0)}</div>
      </div>
      <div className={`metric-card metric-card-hero ${scoreClass(macroF1)}`}>
        <div className="card-label">
          {getPrimaryMetricLabel("Macro F1", usingOverlap)}
        </div>
        <div className="card-value">{formatPercent(macroF1)}</div>
      </div>
      <div className="metric-card metric-card-secondary">
        <div className="card-label">
          {getPrimaryMetricLabel("Macro P", usingOverlap)}
        </div>
        <div className="card-value">{formatPercent(primary?.macro.precision ?? 0)}</div>
      </div>
      <div className="metric-card metric-card-secondary">
        <div className="card-label">
          {getPrimaryMetricLabel("Macro R", usingOverlap)}
        </div>
        <div className="card-value">{formatPercent(primary?.macro.recall ?? 0)}</div>
      </div>
    </div>
  );
}

function PerLabelMetricsTable({
  metrics,
}: {
  metrics: NonNullable<PrimaryMetrics>;
}) {
  return (
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
        {Object.entries(metrics.per_label ?? {}).map(([label, metric]) => (
          <tr key={label}>
            <td>
              <span
                style={{
                  ...LABEL_SWATCH_STYLE,
                  background: getLabelColor(label),
                }}
              />
              {label}
            </td>
            <td>{formatPercent(metric.precision)}</td>
            <td>{formatPercent(metric.recall)}</td>
            <td>{formatPercent(metric.f1)}</td>
            <td>{metric.support}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ConfusionMatrix({
  confusion,
}: {
  confusion: NonNullable<MetricsResult["confusion_matrix"]>;
}) {
  const maxVal = Math.max(...confusion.matrix.flat(), 1);

  return (
    <div className="confusion-matrix">
      <h4>Confusion Matrix</h4>
      <div
        className="confusion-grid"
        style={{
          gridTemplateColumns: `60px repeat(${confusion.labels.length}, 40px)`,
        }}
      >
        <div className="confusion-cell" />
        {confusion.labels.map((label) => (
          <div key={label} className="confusion-cell header">
            {label}
          </div>
        ))}
        {confusion.matrix.map((row, rowIndex) => (
          <React.Fragment key={rowIndex}>
            <div className="confusion-cell row-header">{confusion.labels[rowIndex]}</div>
            {row.map((value, columnIndex) => {
              const intensity = value / maxVal;
              return (
                <div
                  key={`${rowIndex}-${columnIndex}`}
                  className="confusion-cell"
                  style={{
                    background: `color-mix(in srgb, var(--app-primary-light) ${Math.round(intensity * 80)}%, transparent)`,
                    color: intensity > 0.5 ? "#fff" : "var(--app-text)",
                  }}
                >
                  {value}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function MetricsSpanList({
  title,
  spans,
}: {
  title: string;
  spans: MetricsResult["false_positives"] | MetricsResult["false_negatives"];
}) {
  if (!spans || spans.length === 0) {
    return null;
  }

  return (
    <div className="fp-fn-section" style={{ flex: 1 }}>
      <h4>
        {title} ({spans.length})
      </h4>
      <ul className="fp-fn-list">
        {spans.map((span) => (
          <li key={spanItemKey(span)}>
            <span
              className="span-label"
              style={{
                background: getLabelColor(span.label),
              }}
            >
              {span.label}
            </span>
            &quot;{span.text}&quot; [{span.start}:{span.end}]
          </li>
        ))}
      </ul>
    </div>
  );
}

function MetricsContent({
  reference,
  hypothesis,
  matchMode,
  sourceOptions,
  metrics,
  onReferenceChange,
  onHypothesisChange,
  onMatchModeChange,
}: {
  reference: AnnotationSource;
  hypothesis: AnnotationSource;
  matchMode: MatchMode;
  sourceOptions: SourceOption[];
  metrics: NonNullable<MetricsResult>;
  onReferenceChange: (ref: AnnotationSource) => void;
  onHypothesisChange: (hyp: AnnotationSource) => void;
  onMatchModeChange: (mode: MatchMode) => void;
}) {
  const { primary, exact, usingOverlap } = getPrimaryMetrics(metrics, matchMode);
  const confidence = metrics.llm_confidence ?? null;
  const confidencePct = formatConfidencePercent(confidence?.confidence);
  const confidenceBand = confidence?.band.toUpperCase() ?? "N/A";
  const meanLogprob = formatMetricNumber(confidence?.mean_logprob, 4);
  const perplexity = formatMetricNumber(confidence?.perplexity, 3);
  const perLabelMetrics = primary ?? metrics;

  return (
    <>
      <MetricsCompareControls
        reference={reference}
        hypothesis={hypothesis}
        matchMode={matchMode}
        sourceOptions={sourceOptions}
        onReferenceChange={onReferenceChange}
        onHypothesisChange={onHypothesisChange}
        onMatchModeChange={onMatchModeChange}
      />
      {confidence && (
        <ConfidenceSummary
          confidence={confidence}
          confidencePct={confidencePct}
          confidenceBand={confidenceBand}
          meanLogprob={meanLogprob}
          perplexity={perplexity}
        />
      )}
      {usingOverlap && exact && <ExactDiagnostic exact={exact} />}
      <MetricCards primary={primary} usingOverlap={usingOverlap} />
      <PerLabelMetricsTable metrics={perLabelMetrics} />
      {metrics.confusion_matrix && <ConfusionMatrix confusion={metrics.confusion_matrix} />}
      <div style={SECTION_ROW_STYLE}>
        <MetricsSpanList title="False Positives" spans={metrics.false_positives ?? []} />
        <MetricsSpanList title="False Negatives" spans={metrics.false_negatives ?? []} />
      </div>
    </>
  );
}

export default function MetricsPanel({
  reference,
  hypothesis,
  matchMode,
  sourceOptions,
  metrics,
  loading,
  onRefresh,
  onReferenceChange,
  onHypothesisChange,
  onMatchModeChange,
}: Props) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className={`metrics-panel ${collapsed ? "collapsed" : ""}`}>
      <MetricsHeader
        collapsed={collapsed}
        onToggle={() => setCollapsed(!collapsed)}
        onRefresh={onRefresh}
      />

      {!collapsed && (
        <div className="metrics-body">
          {loading && <div className="loading">Computing metrics...</div>}
          {!loading && !metrics && (
            <div className="pane-empty">
              <div className="pane-empty-title">No per-document metrics yet</div>
              <div className="pane-empty-hint">Select a document and click Refresh to compute metrics.</div>
            </div>
          )}
          {metrics && (
            <MetricsContent
              reference={reference}
              hypothesis={hypothesis}
              matchMode={matchMode}
              sourceOptions={sourceOptions}
              metrics={metrics}
              onReferenceChange={onReferenceChange}
              onHypothesisChange={onHypothesisChange}
              onMatchModeChange={onMatchModeChange}
            />
          )}
        </div>
      )}
    </div>
  );
}
