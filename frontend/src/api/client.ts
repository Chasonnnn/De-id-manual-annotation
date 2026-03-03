import type {
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
  reference: "pre" | "manual" | "agent",
  hypothesis: "pre" | "manual" | "agent",
  matchMode: "exact" | "overlap",
): Promise<MetricsResult> {
  const params = new URLSearchParams({
    reference,
    hypothesis,
    match_mode: matchMode,
  });
  return request(`/documents/${docId}/metrics?${params}`);
}
