import type {
  DashboardDocumentMetrics,
  DashboardMetricsResult,
  MatchMode,
  MethodsLabMatrixCellSummary,
  MetricsResult,
  PromptLabMatrixCellSummary,
} from "./types";

export const NAME_TOLERANT_METRIC_KEY = "exact_name_affix_tolerant";

type MetricLabelValue = {
  precision: number;
  recall: number;
  f1: number;
  support?: number;
  tp?: number;
  fp?: number;
  fn?: number;
};

type MetricLike = {
  micro: { precision: number; recall: number; f1: number; tp?: number; fp?: number; fn?: number };
  macro: { precision: number; recall: number; f1: number };
  per_label?: Record<string, MetricLabelValue>;
};

export function getPrimaryMetrics(
  metrics: MetricsResult | null | undefined,
  matchMode: MatchMode,
): {
  primary: MetricLike | null;
  exact: MetricLike | null;
  usingNameTolerant: boolean;
} {
  if (!metrics) {
    return { primary: null, exact: null, usingNameTolerant: false };
  }
  const tolerant =
    matchMode === "exact" ? metrics.co_primary_metrics?.[NAME_TOLERANT_METRIC_KEY] ?? null : null;
  return {
    primary: tolerant ?? metrics,
    exact: metrics,
    usingNameTolerant: tolerant != null,
  };
}

export function getPrimaryMatrixMetrics(
  cell: PromptLabMatrixCellSummary | MethodsLabMatrixCellSummary | undefined,
): {
  primaryMicro: PromptLabMatrixCellSummary["micro"];
  primaryPerLabel: PromptLabMatrixCellSummary["per_label"];
  exactMicro: PromptLabMatrixCellSummary["micro"];
  usingNameTolerant: boolean;
} {
  if (!cell) {
    return {
      primaryMicro: { precision: 0, recall: 0, f1: 0, tp: 0, fp: 0, fn: 0 },
      primaryPerLabel: {},
      exactMicro: { precision: 0, recall: 0, f1: 0, tp: 0, fp: 0, fn: 0 },
      usingNameTolerant: false,
    };
  }
  const tolerant = cell.co_primary_metrics?.[NAME_TOLERANT_METRIC_KEY];
  return {
    primaryMicro: tolerant?.micro ?? cell.micro,
    primaryPerLabel: tolerant?.per_label ?? cell.per_label,
    exactMicro: cell.micro,
    usingNameTolerant: tolerant != null,
  };
}

export function getPrimaryDashboardMetrics(
  dashboard: DashboardMetricsResult | null | undefined,
): {
  primaryMicro: DashboardMetricsResult["micro"] | null;
  exactMicro: DashboardMetricsResult["micro"] | null;
  usingNameTolerant: boolean;
} {
  if (!dashboard) {
    return { primaryMicro: null, exactMicro: null, usingNameTolerant: false };
  }
  const tolerant = dashboard.co_primary_metrics?.[NAME_TOLERANT_METRIC_KEY]?.micro ?? null;
  return {
    primaryMicro:
      tolerant != null
        ? {
            precision: tolerant.precision,
            recall: tolerant.recall,
            f1: tolerant.f1,
            tp: tolerant.tp ?? dashboard.micro.tp,
            fp: tolerant.fp ?? dashboard.micro.fp,
            fn: tolerant.fn ?? dashboard.micro.fn,
          }
        : dashboard.micro,
    exactMicro: dashboard.micro,
    usingNameTolerant: tolerant != null,
  };
}

export function getPrimaryDashboardDocumentMetrics(
  row: DashboardDocumentMetrics,
): {
  primaryMicro: DashboardDocumentMetrics["micro"];
  exactMicro: DashboardDocumentMetrics["micro"];
  usingNameTolerant: boolean;
} {
  const tolerant = row.co_primary_metrics?.[NAME_TOLERANT_METRIC_KEY]?.micro ?? null;
  return {
    primaryMicro:
      tolerant != null
        ? {
            precision: tolerant.precision,
            recall: tolerant.recall,
            f1: tolerant.f1,
            tp: tolerant.tp ?? row.micro.tp,
            fp: tolerant.fp ?? row.micro.fp,
            fn: tolerant.fn ?? row.micro.fn,
          }
        : row.micro,
    exactMicro: row.micro,
    usingNameTolerant: tolerant != null,
  };
}

export function getPrimaryMetricLabel(baseLabel: string, usingNameTolerant: boolean): string {
  return usingNameTolerant ? `NAME-Tolerant ${baseLabel}` : baseLabel;
}
