import type {
  AnnotationSource,
  AgentConfig,
  AgentCredentialStatus,
  AgentMethodOption,
  CanonicalDocument,
  CanonicalSpan,
  DashboardMetricsResult,
  LLMConfidenceBand,
  LLMConfidenceMetric,
  DocumentSummary,
  MetricsResult,
  SessionExportBundle,
  SessionImportResult,
  SessionProfile,
} from "../types";

const BASE = "/api";

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text();
    let detail = body;
    try {
      const parsed = JSON.parse(body) as { detail?: unknown };
      if (parsed?.detail !== undefined) {
        detail = String(parsed.detail);
      }
    } catch {
      // Keep plain-text body
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function listDocuments(): Promise<DocumentSummary[]> {
  return request("/documents");
}

export async function getDocument(id: string): Promise<CanonicalDocument> {
  return request(`/documents/${id}`);
}

export async function uploadDocument(file: File): Promise<CanonicalDocument> {
  const form = new FormData();
  form.append("file", file);
  return request("/documents/upload", { method: "POST", body: form });
}

export async function deleteDocument(docId: string): Promise<{ deleted: boolean; doc_id: string }> {
  return request(`/documents/${docId}`, { method: "DELETE" });
}

export async function exportSession(): Promise<SessionExportBundle> {
  return request("/session/export");
}

export async function importSession(file: File): Promise<SessionImportResult> {
  const form = new FormData();
  form.append("file", file);
  return request("/session/import", { method: "POST", body: form });
}

export async function getSessionProfile(): Promise<SessionProfile> {
  return request("/session/profile");
}

export async function updateSessionProfile(
  profile: Partial<SessionProfile>,
): Promise<SessionProfile> {
  return request("/session/profile", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  });
}

export async function updateManualAnnotations(
  docId: string,
  annotations: CanonicalSpan[],
): Promise<CanonicalDocument> {
  return request(`/documents/${docId}/manual-annotations`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(annotations),
  });
}

export async function runAgent(
  docId: string,
  config: AgentConfig,
): Promise<CanonicalDocument> {
  return request(`/documents/${docId}/agent`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

export async function getAgentCredentialStatus(): Promise<AgentCredentialStatus> {
  return request("/agent/credentials/status");
}

export async function getAgentMethods(): Promise<AgentMethodOption[]> {
  const data = await request<{ methods: AgentMethodOption[] }>("/agent/methods");
  return data.methods;
}

export async function getMetrics(
  docId: string,
  reference: AnnotationSource,
  hypothesis: AnnotationSource,
  matchMode: "exact" | "overlap",
): Promise<MetricsResult> {
  const params = new URLSearchParams({
    reference,
    hypothesis,
    match_mode: matchMode,
  });
  const raw = await request<Record<string, unknown>>(
    `/documents/${docId}/metrics?${params}`,
  );
  return normalizeMetrics(raw);
}

export async function getMetricsDashboard(
  reference: AnnotationSource,
  hypothesis: AnnotationSource,
  matchMode: "exact" | "overlap",
): Promise<DashboardMetricsResult> {
  const params = new URLSearchParams({
    reference,
    hypothesis,
    match_mode: matchMode,
  });
  const raw = await request<Record<string, unknown>>(`/metrics/dashboard?${params}`);
  return normalizeDashboardMetrics(raw, reference, hypothesis, matchMode);
}

function normalizeMetrics(raw: Record<string, unknown>): MetricsResult {
  const rawPerLabel = (raw.per_label ?? {}) as Record<
    string,
    { precision: number; recall: number; f1: number; support?: number; tp?: number; fn?: number }
  >;
  const perLabel: MetricsResult["per_label"] = {};
  for (const [label, value] of Object.entries(rawPerLabel)) {
    perLabel[label] = {
      precision: value.precision,
      recall: value.recall,
      f1: value.f1,
      support: typeof value.support === "number" ? value.support : (value.tp ?? 0) + (value.fn ?? 0),
      tp: value.tp,
      fp: (value as { fp?: number }).fp,
      fn: value.fn,
    };
  }

  let confusionMatrix = raw.confusion_matrix as
    | { labels: string[]; matrix: number[][] }
    | undefined;

  if (!confusionMatrix && raw.confusion) {
    const confusion = raw.confusion as {
      labels: string[];
      matrix: Record<string, Record<string, number>>;
    };
    confusionMatrix = {
      labels: confusion.labels,
      matrix: confusion.labels.map((r) =>
        confusion.labels.map((c) => confusion.matrix?.[r]?.[c] ?? 0),
      ),
    };
  }

  return {
    micro: raw.micro as MetricsResult["micro"],
    macro: raw.macro as MetricsResult["macro"],
    per_label: perLabel,
    confusion_matrix: confusionMatrix,
    false_positives: (raw.false_positives as CanonicalSpan[] | undefined) ?? [],
    false_negatives: (raw.false_negatives as CanonicalSpan[] | undefined) ?? [],
    llm_confidence: normalizeLLMConfidence(raw.llm_confidence),
  };
}

function normalizeDashboardMetrics(
  raw: Record<string, unknown>,
  reference: AnnotationSource,
  hypothesis: AnnotationSource,
  matchMode: "exact" | "overlap",
): DashboardMetricsResult {
  const defaultBandCounts: Record<LLMConfidenceBand, number> = {
    high: 0,
    medium: 0,
    low: 0,
    na: 0,
  };

  const rawSummary = isRecord(raw.llm_confidence_summary)
    ? raw.llm_confidence_summary
    : {};
  const rawBandCounts = isRecord(rawSummary.band_counts)
    ? rawSummary.band_counts
    : {};
  const bandCounts: Record<LLMConfidenceBand, number> = {
    high: toNumber(rawBandCounts.high, 0),
    medium: toNumber(rawBandCounts.medium, 0),
    low: toNumber(rawBandCounts.low, 0),
    na: toNumber(rawBandCounts.na, 0),
  };

  const rawDocs = Array.isArray(raw.documents) ? raw.documents : [];
  const documents = rawDocs.map((item) => {
    const row = isRecord(item) ? item : {};
    return {
      id: String(row.id ?? ""),
      filename: String(row.filename ?? ""),
      reference_count: toNumber(row.reference_count, 0),
      hypothesis_count: toNumber(row.hypothesis_count, 0),
      micro: (row.micro ?? {
        precision: 0,
        recall: 0,
        f1: 0,
        tp: 0,
        fp: 0,
        fn: 0,
      }) as DashboardMetricsResult["documents"][number]["micro"],
      macro: (row.macro ?? { precision: 0, recall: 0, f1: 0 }) as DashboardMetricsResult["documents"][number]["macro"],
      cohens_kappa: toNumber(row.cohens_kappa, 0),
      mean_iou: toNumber(row.mean_iou, 0),
      llm_confidence: normalizeLLMConfidence(row.llm_confidence),
    };
  });

  return {
    reference: (raw.reference as AnnotationSource | undefined) ?? reference,
    hypothesis: (raw.hypothesis as AnnotationSource | undefined) ?? hypothesis,
    match_mode: (raw.match_mode as "exact" | "overlap" | undefined) ?? matchMode,
    total_documents: toNumber(raw.total_documents, 0),
    documents_compared: toNumber(raw.documents_compared, documents.length),
    micro: (raw.micro ?? {
      precision: 0,
      recall: 0,
      f1: 0,
      tp: 0,
      fp: 0,
      fn: 0,
    }) as DashboardMetricsResult["micro"],
    avg_document_micro: (raw.avg_document_micro ?? {
      precision: 0,
      recall: 0,
      f1: 0,
    }) as DashboardMetricsResult["avg_document_micro"],
    avg_document_macro: (raw.avg_document_macro ?? {
      precision: 0,
      recall: 0,
      f1: 0,
    }) as DashboardMetricsResult["avg_document_macro"],
    llm_confidence_summary: {
      documents_with_confidence: toNumber(
        rawSummary.documents_with_confidence,
        0,
      ),
      mean_confidence:
        typeof rawSummary.mean_confidence === "number"
          ? rawSummary.mean_confidence
          : null,
      band_counts: { ...defaultBandCounts, ...bandCounts },
    },
    documents,
  };
}

function normalizeLLMConfidence(raw: unknown): LLMConfidenceMetric | null {
  if (!isRecord(raw)) return null;
  const reason = toConfidenceReason(raw.reason);
  const band = toConfidenceBand(raw.band);
  if (reason === null || band === null) return null;

  return {
    available: Boolean(raw.available),
    provider: String(raw.provider ?? ""),
    model: String(raw.model ?? ""),
    reason,
    token_count: toNumber(raw.token_count, 0),
    mean_logprob: typeof raw.mean_logprob === "number" ? raw.mean_logprob : null,
    confidence: typeof raw.confidence === "number" ? raw.confidence : null,
    perplexity: typeof raw.perplexity === "number" ? raw.perplexity : null,
    band,
    high_threshold: toNumber(raw.high_threshold, 0.9),
    medium_threshold: toNumber(raw.medium_threshold, 0.75),
  };
}

function toConfidenceReason(
  value: unknown,
): LLMConfidenceMetric["reason"] | null {
  if (
    value === "ok" ||
    value === "unsupported_provider" ||
    value === "missing_logprobs" ||
    value === "empty_completion"
  ) {
    return value;
  }
  return null;
}

function toConfidenceBand(value: unknown): LLMConfidenceBand | null {
  if (value === "high" || value === "medium" || value === "low" || value === "na") {
    return value;
  }
  return null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object";
}

function toNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}
