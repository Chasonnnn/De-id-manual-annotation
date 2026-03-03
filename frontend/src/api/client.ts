import type {
  AnnotationSource,
  AgentConfig,
  CanonicalDocument,
  CanonicalSpan,
  DocumentSummary,
  MetricsResult,
} from "../types";

const BASE = "/api";

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
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
  };
}
