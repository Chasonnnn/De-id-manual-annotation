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
  PromptLabDocResult,
  PromptLabRunCreateRequest,
  PromptLabRunDetail,
  PromptLabRunSummary,
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

export async function exportGroundTruth(source: AnnotationSource): Promise<Blob> {
  const params = new URLSearchParams({ source });
  const res = await fetch(`${BASE}/session/export-ground-truth?${params.toString()}`);
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
  return res.blob();
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

export async function createPromptLabRun(
  payload: PromptLabRunCreateRequest,
): Promise<PromptLabRunDetail> {
  const raw = await request<Record<string, unknown>>("/prompt-lab/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return normalizePromptLabRunDetail(raw);
}

export async function listPromptLabRuns(): Promise<PromptLabRunSummary[]> {
  const data = await request<{ runs: Record<string, unknown>[] }>("/prompt-lab/runs");
  return (data.runs ?? []).map(normalizePromptLabRunSummary);
}

export async function getPromptLabRun(runId: string): Promise<PromptLabRunDetail> {
  const raw = await request<Record<string, unknown>>(`/prompt-lab/runs/${runId}`);
  return normalizePromptLabRunDetail(raw);
}

export async function getPromptLabDocResult(
  runId: string,
  cellId: string,
  docId: string,
): Promise<PromptLabDocResult> {
  const raw = await request<Record<string, unknown>>(
    `/prompt-lab/runs/${runId}/cells/${cellId}/documents/${docId}`,
  );
  const metricsRaw = isRecord(raw.metrics) ? normalizeMetrics(raw.metrics) : null;
  return {
    run_id: String(raw.run_id ?? runId),
    cell_id: String(raw.cell_id ?? cellId),
    doc_id: String(raw.doc_id ?? docId),
    status: (raw.status as PromptLabDocResult["status"] | undefined) ?? "pending",
    error: typeof raw.error === "string" ? raw.error : null,
    warnings: Array.isArray(raw.warnings)
      ? raw.warnings.filter((x): x is string => typeof x === "string")
      : [],
    reference_source_used:
      raw.reference_source_used === "manual" || raw.reference_source_used === "pre"
        ? raw.reference_source_used
        : undefined,
    reference_spans: Array.isArray(raw.reference_spans)
      ? (raw.reference_spans as CanonicalSpan[])
      : [],
    hypothesis_spans: Array.isArray(raw.hypothesis_spans)
      ? (raw.hypothesis_spans as CanonicalSpan[])
      : [],
    metrics: metricsRaw,
    llm_confidence: normalizeLLMConfidence(raw.llm_confidence),
    transcript_text: typeof raw.transcript_text === "string" ? raw.transcript_text : null,
    document: isRecord(raw.document)
      ? {
          id: String(raw.document.id ?? docId),
          filename:
            typeof raw.document.filename === "string" ? raw.document.filename : null,
        }
      : { id: docId, filename: null },
    model: isRecord(raw.model) ? normalizePromptLabModel(raw.model, 0) : undefined,
    prompt: isRecord(raw.prompt) ? normalizePromptLabPrompt(raw.prompt, 0) : undefined,
  };
}

export async function getMetrics(
  docId: string,
  reference: AnnotationSource,
  hypothesis: AnnotationSource,
  matchMode: "exact" | "overlap",
  labelProjection: "native" | "coarse_simple" = "native",
): Promise<MetricsResult> {
  const params = new URLSearchParams({
    reference,
    hypothesis,
    match_mode: matchMode,
    label_projection: labelProjection,
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
  labelProjection: "native" | "coarse_simple" = "native",
): Promise<DashboardMetricsResult> {
  const params = new URLSearchParams({
    reference,
    hypothesis,
    match_mode: matchMode,
    label_projection: labelProjection,
  });
  const raw = await request<Record<string, unknown>>(`/metrics/dashboard?${params}`);
  return normalizeDashboardMetrics(
    raw,
    reference,
    hypothesis,
    matchMode,
    labelProjection,
  );
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
    label_projection:
      raw.label_projection === "coarse_simple" ? "coarse_simple" : "native",
    per_label: perLabel,
    confusion_matrix: confusionMatrix,
    false_positives: (raw.false_positives as CanonicalSpan[] | undefined) ?? [],
    false_negatives: (raw.false_negatives as CanonicalSpan[] | undefined) ?? [],
    llm_confidence: normalizeLLMConfidence(raw.llm_confidence),
  };
}

function normalizePromptLabRunSummary(raw: Record<string, unknown>): PromptLabRunSummary {
  return {
    id: String(raw.id ?? ""),
    name: String(raw.name ?? ""),
    status: (raw.status as PromptLabRunSummary["status"] | undefined) ?? "queued",
    created_at: String(raw.created_at ?? ""),
    started_at: typeof raw.started_at === "string" ? raw.started_at : null,
    finished_at: typeof raw.finished_at === "string" ? raw.finished_at : null,
    doc_count: toNumber(raw.doc_count, 0),
    prompt_count: toNumber(raw.prompt_count, 0),
    model_count: toNumber(raw.model_count, 0),
    total_tasks: toNumber(raw.total_tasks, 0),
    completed_tasks: toNumber(raw.completed_tasks, 0),
    failed_tasks: toNumber(raw.failed_tasks, 0),
  };
}

function normalizePromptLabRunDetail(raw: Record<string, unknown>): PromptLabRunDetail {
  const summary = normalizePromptLabRunSummary(raw);
  const docIds = Array.isArray(raw.doc_ids)
    ? raw.doc_ids.map((value) => String(value))
    : [];
  const promptsRaw = Array.isArray(raw.prompts) ? raw.prompts : [];
  const modelsRaw = Array.isArray(raw.models) ? raw.models : [];
  const runtimeRaw = isRecord(raw.runtime) ? raw.runtime : {};
  const matrixRaw = isRecord(raw.matrix) ? raw.matrix : {};
  const matrixModels = Array.isArray(matrixRaw.models)
    ? matrixRaw.models
        .filter(isRecord)
        .map((item) => ({ id: String(item.id ?? ""), label: String(item.label ?? "") }))
    : [];
  const matrixPrompts = Array.isArray(matrixRaw.prompts)
    ? matrixRaw.prompts
        .filter(isRecord)
        .map((item) => ({ id: String(item.id ?? ""), label: String(item.label ?? "") }))
    : [];
  const matrixCellsRaw = Array.isArray(matrixRaw.cells) ? matrixRaw.cells : [];
  const matrixCells = matrixCellsRaw
    .filter(isRecord)
    .map((cell, index) => normalizePromptLabCellSummary(cell, index, summary.doc_count));
  const availableLabels = Array.isArray(matrixRaw.available_labels)
    ? matrixRaw.available_labels.filter((x): x is string => typeof x === "string")
    : [];

  return {
    ...summary,
    doc_ids: docIds,
    prompts: promptsRaw
      .filter(isRecord)
      .map((prompt, index) => normalizePromptLabPrompt(prompt, index)),
    models: modelsRaw
      .filter(isRecord)
      .map((model, index) => normalizePromptLabModel(model, index)),
    runtime: {
      api_base: typeof runtimeRaw.api_base === "string" ? runtimeRaw.api_base : undefined,
      temperature: toNumber(runtimeRaw.temperature, 0),
      match_mode:
        runtimeRaw.match_mode === "overlap"
          ? "overlap"
          : "exact",
      reference_source:
        runtimeRaw.reference_source === "pre"
          ? "pre"
          : "manual",
      fallback_reference_source:
        runtimeRaw.fallback_reference_source === "manual"
          ? "manual"
          : "pre",
      label_profile:
        runtimeRaw.label_profile === "advanced"
          ? "advanced"
          : "simple",
      label_projection:
        runtimeRaw.label_projection === "coarse_simple"
          ? "coarse_simple"
          : "native",
      chunk_mode:
        runtimeRaw.chunk_mode === "off" || runtimeRaw.chunk_mode === "force"
          ? runtimeRaw.chunk_mode
          : "auto",
      chunk_size_chars: toNumber(runtimeRaw.chunk_size_chars, 10000),
    },
    concurrency: toNumber(raw.concurrency, 4),
    warnings: Array.isArray(raw.warnings)
      ? raw.warnings.filter((x): x is string => typeof x === "string")
      : [],
    errors: Array.isArray(raw.errors)
      ? raw.errors.filter((x): x is string => typeof x === "string")
      : [],
    matrix: {
      models: matrixModels,
      prompts: matrixPrompts,
      cells: matrixCells,
      available_labels: availableLabels,
    },
    progress: isRecord(raw.progress)
      ? {
          total_tasks: toNumber(raw.progress.total_tasks, summary.total_tasks),
          completed_tasks: toNumber(raw.progress.completed_tasks, summary.completed_tasks),
          failed_tasks: toNumber(raw.progress.failed_tasks, summary.failed_tasks),
        }
      : {
          total_tasks: summary.total_tasks,
          completed_tasks: summary.completed_tasks,
          failed_tasks: summary.failed_tasks,
        },
  };
}

function normalizePromptLabPrompt(
  raw: Record<string, unknown>,
  index: number,
): PromptLabRunDetail["prompts"][number] {
  const variantType = raw.variant_type === "preset" ? "preset" : "prompt";
  return {
    id: typeof raw.id === "string" ? raw.id : `prompt_${index + 1}`,
    label: String(raw.label ?? `Prompt ${index + 1}`),
    variant_type: variantType,
    preset_method_id:
      typeof raw.preset_method_id === "string" ? raw.preset_method_id : null,
    method_verify_override:
      typeof raw.method_verify_override === "boolean"
        ? raw.method_verify_override
        : null,
    system_prompt:
      typeof raw.system_prompt === "string"
        ? raw.system_prompt
        : variantType === "prompt"
          ? ""
          : undefined,
  };
}

function normalizePromptLabModel(
  raw: Record<string, unknown>,
  index: number,
): PromptLabRunDetail["models"][number] {
  return {
    id: typeof raw.id === "string" ? raw.id : `model_${index + 1}`,
    label: String(raw.label ?? `Model ${index + 1}`),
    model: String(raw.model ?? ""),
    reasoning_effort:
      raw.reasoning_effort === "low" ||
      raw.reasoning_effort === "medium" ||
      raw.reasoning_effort === "high" ||
      raw.reasoning_effort === "xhigh"
        ? raw.reasoning_effort
        : "none",
    anthropic_thinking: Boolean(raw.anthropic_thinking),
    anthropic_thinking_budget_tokens:
      typeof raw.anthropic_thinking_budget_tokens === "number"
        ? raw.anthropic_thinking_budget_tokens
        : null,
  };
}

function normalizePromptLabCellSummary(
  raw: Record<string, unknown>,
  index: number,
  defaultDocCount: number,
): PromptLabRunDetail["matrix"]["cells"][number] {
  const microRaw = isRecord(raw.micro) ? raw.micro : {};
  const perLabelRaw = isRecord(raw.per_label)
    ? (raw.per_label as Record<string, unknown>)
    : {};
  const perLabel: PromptLabRunDetail["matrix"]["cells"][number]["per_label"] = {};
  for (const [label, value] of Object.entries(perLabelRaw)) {
    if (!isRecord(value)) continue;
    perLabel[label] = {
      precision: toNumber(value.precision, 0),
      recall: toNumber(value.recall, 0),
      f1: toNumber(value.f1, 0),
      tp: toNumber(value.tp, 0),
      fp: toNumber(value.fp, 0),
      fn: toNumber(value.fn, 0),
      support: toNumber(value.support, 0),
    };
  }
  return {
    id: String(raw.id ?? `cell_${index + 1}`),
    model_id: String(raw.model_id ?? ""),
    model_label: String(raw.model_label ?? ""),
    prompt_id: String(raw.prompt_id ?? ""),
    prompt_label: String(raw.prompt_label ?? ""),
    status: (raw.status as PromptLabRunDetail["matrix"]["cells"][number]["status"] | undefined) ?? "pending",
    total_docs: toNumber(raw.total_docs, defaultDocCount),
    completed_docs: toNumber(raw.completed_docs, 0),
    failed_docs: toNumber(raw.failed_docs, 0),
    error_count: toNumber(raw.error_count, 0),
    micro: {
      precision: toNumber(microRaw.precision, 0),
      recall: toNumber(microRaw.recall, 0),
      f1: toNumber(microRaw.f1, 0),
      tp: toNumber(microRaw.tp, 0),
      fp: toNumber(microRaw.fp, 0),
      fn: toNumber(microRaw.fn, 0),
    },
    per_label: perLabel,
    mean_confidence: typeof raw.mean_confidence === "number" ? raw.mean_confidence : null,
  };
}

function normalizeDashboardMetrics(
  raw: Record<string, unknown>,
  reference: AnnotationSource,
  hypothesis: AnnotationSource,
  matchMode: "exact" | "overlap",
  labelProjection: "native" | "coarse_simple",
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
    label_projection:
      raw.label_projection === "coarse_simple"
        ? "coarse_simple"
        : labelProjection,
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
