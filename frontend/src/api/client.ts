import {
  DEFAULT_EXPERIMENT_LIMITS,
} from "../types";
import type {
  AnnotationSource,
  AgentConfig,
  AgentCredentialStatus,
  AgentMethodOption,
  AgentRunProgress,
  CanonicalDocument,
  CanonicalSpan,
  DashboardMetricsResult,
  LLMConfidenceBand,
  LLMConfidenceMetric,
  MatchMode,
  DocumentSummary,
  ExperimentLimits,
  MetricsResult,
  MethodsLabDocResult,
  MethodsLabRunCreateRequest,
  MethodsLabRunDetail,
  MethodsLabRunSummary,
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

export async function getAgentProgress(docId: string): Promise<AgentRunProgress> {
  return request(`/documents/${docId}/agent/progress`);
}

export async function getAgentCredentialStatus(): Promise<AgentCredentialStatus> {
  return request("/agent/credentials/status");
}

export async function getAgentMethods(): Promise<AgentMethodOption[]> {
  const data = await request<{ methods: AgentMethodOption[] }>("/agent/methods");
  return data.methods;
}

export async function getExperimentLimits(): Promise<ExperimentLimits> {
  const raw = await request<Record<string, unknown>>("/experiments/limits");
  return normalizeExperimentLimits(raw);
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

export async function deletePromptLabRun(
  runId: string,
): Promise<{ ok: boolean; id: string }> {
  return request(`/prompt-lab/runs/${runId}`, { method: "DELETE" });
}

export async function stopPromptLabRun(
  runId: string,
): Promise<{ ok: boolean; id: string; status: "cancelling" | "cancelled" }> {
  return request(`/prompt-lab/runs/${runId}/cancel`, { method: "POST" });
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

export async function createMethodsLabRun(
  payload: MethodsLabRunCreateRequest,
): Promise<MethodsLabRunDetail> {
  const raw = await request<Record<string, unknown>>("/methods-lab/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return normalizeMethodsLabRunDetail(raw);
}

export async function listMethodsLabRuns(): Promise<MethodsLabRunSummary[]> {
  const data = await request<{ runs: Record<string, unknown>[] }>("/methods-lab/runs");
  return (data.runs ?? []).map(normalizeMethodsLabRunSummary);
}

export async function getMethodsLabRun(runId: string): Promise<MethodsLabRunDetail> {
  const raw = await request<Record<string, unknown>>(`/methods-lab/runs/${runId}`);
  return normalizeMethodsLabRunDetail(raw);
}

export async function deleteMethodsLabRun(
  runId: string,
): Promise<{ ok: boolean; id: string }> {
  return request(`/methods-lab/runs/${runId}`, { method: "DELETE" });
}

export async function stopMethodsLabRun(
  runId: string,
): Promise<{ ok: boolean; id: string; status: "cancelling" | "cancelled" }> {
  return request(`/methods-lab/runs/${runId}/cancel`, { method: "POST" });
}

export async function getMethodsLabDocResult(
  runId: string,
  cellId: string,
  docId: string,
): Promise<MethodsLabDocResult> {
  const raw = await request<Record<string, unknown>>(
    `/methods-lab/runs/${runId}/cells/${cellId}/documents/${docId}`,
  );
  const metricsRaw = isRecord(raw.metrics) ? normalizeMetrics(raw.metrics) : null;
  return {
    run_id: String(raw.run_id ?? runId),
    cell_id: String(raw.cell_id ?? cellId),
    doc_id: String(raw.doc_id ?? docId),
    status: (raw.status as MethodsLabDocResult["status"] | undefined) ?? "pending",
    error: typeof raw.error === "string" ? raw.error : null,
    warnings: Array.isArray(raw.warnings)
      ? raw.warnings.filter((x): x is string => typeof x === "string")
      : [],
    reference_source_used: raw.reference_source_used === "manual" ? "manual" : undefined,
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
    method: isRecord(raw.method) ? normalizeMethodsLabMethod(raw.method, 0) : undefined,
  };
}

export async function getMetrics(
  docId: string,
  reference: AnnotationSource,
  hypothesis: AnnotationSource,
  matchMode: MatchMode,
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
  matchMode: MatchMode,
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
  const normalizeMetricBundle = (
    bundle: Record<string, unknown>,
  ): {
    micro: MetricsResult["micro"];
    macro: MetricsResult["macro"];
    per_label: MetricsResult["per_label"];
    false_positives: CanonicalSpan[];
    false_negatives: CanonicalSpan[];
  } => {
    const rawPerLabel = (bundle.per_label ?? {}) as Record<
      string,
      { precision: number; recall: number; f1: number; support?: number; tp?: number; fn?: number }
    >;
    const perLabel: MetricsResult["per_label"] = {};
    for (const [label, value] of Object.entries(rawPerLabel)) {
      perLabel[label] = {
        precision: value.precision,
        recall: value.recall,
        f1: value.f1,
        support:
          typeof value.support === "number" ? value.support : (value.tp ?? 0) + (value.fn ?? 0),
        tp: value.tp,
        fp: (value as { fp?: number }).fp,
        fn: value.fn,
      };
    }
    return {
      micro: bundle.micro as MetricsResult["micro"],
      macro: bundle.macro as MetricsResult["macro"],
      per_label: perLabel,
      false_positives: (bundle.false_positives as CanonicalSpan[] | undefined) ?? [],
      false_negatives: (bundle.false_negatives as CanonicalSpan[] | undefined) ?? [],
    };
  };

  const normalized = normalizeMetricBundle(raw);

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

  const coPrimaryRaw = isRecord(raw.co_primary_metrics)
    ? (raw.co_primary_metrics as Record<string, unknown>)
    : {};
  const coPrimaryMetrics = Object.fromEntries(
    Object.entries(coPrimaryRaw)
      .filter(([, value]) => isRecord(value))
      .map(([key, value]) => [key, normalizeMetricBundle(value as Record<string, unknown>)]),
  );

  return {
    micro: normalized.micro,
    macro: normalized.macro,
    label_projection:
      raw.label_projection === "coarse_simple" ? "coarse_simple" : "native",
    per_label: normalized.per_label,
    confusion_matrix: confusionMatrix,
    false_positives: normalized.false_positives,
    false_negatives: normalized.false_negatives,
    llm_confidence: normalizeLLMConfidence(raw.llm_confidence),
    co_primary_metrics: Object.keys(coPrimaryMetrics).length > 0 ? coPrimaryMetrics : undefined,
  };
}

function normalizeExperimentLimits(raw: Record<string, unknown>): ExperimentLimits {
  return {
    prompt_lab_default_concurrency: toNumber(
      raw.prompt_lab_default_concurrency,
      DEFAULT_EXPERIMENT_LIMITS.prompt_lab_default_concurrency,
    ),
    prompt_lab_max_concurrency: toNumber(
      raw.prompt_lab_max_concurrency,
      DEFAULT_EXPERIMENT_LIMITS.prompt_lab_max_concurrency,
    ),
    methods_lab_default_concurrency: toNumber(
      raw.methods_lab_default_concurrency,
      DEFAULT_EXPERIMENT_LIMITS.methods_lab_default_concurrency,
    ),
    methods_lab_max_concurrency: toNumber(
      raw.methods_lab_max_concurrency,
      DEFAULT_EXPERIMENT_LIMITS.methods_lab_max_concurrency,
    ),
  };
}

function normalizePromptLabRunSummary(raw: Record<string, unknown>): PromptLabRunSummary {
  return {
    id: String(raw.id ?? ""),
    name: String(raw.name ?? ""),
    status: (raw.status as PromptLabRunSummary["status"] | undefined) ?? "queued",
    cancellable: Boolean(raw.cancellable),
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

function normalizeMethodsLabRunSummary(raw: Record<string, unknown>): MethodsLabRunSummary {
  return {
    id: String(raw.id ?? ""),
    name: String(raw.name ?? ""),
    status: (raw.status as MethodsLabRunSummary["status"] | undefined) ?? "queued",
    cancellable: Boolean(raw.cancellable),
    created_at: String(raw.created_at ?? ""),
    started_at: typeof raw.started_at === "string" ? raw.started_at : null,
    finished_at: typeof raw.finished_at === "string" ? raw.finished_at : null,
    doc_count: toNumber(raw.doc_count, 0),
    method_count: toNumber(raw.method_count, 0),
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
          : runtimeRaw.match_mode === "boundary"
            ? "boundary"
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
          : "off",
      chunk_size_chars: toNumber(runtimeRaw.chunk_size_chars, 10000),
    },
    concurrency: toNumber(raw.concurrency, DEFAULT_EXPERIMENT_LIMITS.prompt_lab_default_concurrency),
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

function normalizeMethodsLabRunDetail(raw: Record<string, unknown>): MethodsLabRunDetail {
  const summary = normalizeMethodsLabRunSummary(raw);
  const docIds = Array.isArray(raw.doc_ids)
    ? raw.doc_ids.map((value) => String(value))
    : [];
  const methodsRaw = Array.isArray(raw.methods) ? raw.methods : [];
  const modelsRaw = Array.isArray(raw.models) ? raw.models : [];
  const runtimeRaw = isRecord(raw.runtime) ? raw.runtime : {};
  const matrixRaw = isRecord(raw.matrix) ? raw.matrix : {};
  const matrixModels = Array.isArray(matrixRaw.models)
    ? matrixRaw.models
        .filter(isRecord)
        .map((item) => ({ id: String(item.id ?? ""), label: String(item.label ?? "") }))
    : [];
  const matrixMethods = Array.isArray(matrixRaw.methods)
    ? matrixRaw.methods
        .filter(isRecord)
        .map((item) => ({ id: String(item.id ?? ""), label: String(item.label ?? "") }))
    : [];
  const matrixCellsRaw = Array.isArray(matrixRaw.cells) ? matrixRaw.cells : [];
  const matrixCells = matrixCellsRaw
    .filter(isRecord)
    .map((cell, index) => normalizeMethodsLabCellSummary(cell, index, summary.doc_count));
  const availableLabels = Array.isArray(matrixRaw.available_labels)
    ? matrixRaw.available_labels.filter((x): x is string => typeof x === "string")
    : [];

  return {
    ...summary,
    doc_ids: docIds,
    methods: methodsRaw
      .filter(isRecord)
      .map((method, index) => normalizeMethodsLabMethod(method, index)),
    models: modelsRaw
      .filter(isRecord)
      .map((model, index) => normalizePromptLabModel(model, index)),
    runtime: {
      api_base: typeof runtimeRaw.api_base === "string" ? runtimeRaw.api_base : undefined,
      temperature: toNumber(runtimeRaw.temperature, 0),
      match_mode:
        runtimeRaw.match_mode === "overlap"
          ? "overlap"
          : runtimeRaw.match_mode === "boundary"
            ? "boundary"
            : "exact",
      label_profile: runtimeRaw.label_profile === "advanced" ? "advanced" : "simple",
      label_projection:
        runtimeRaw.label_projection === "coarse_simple" ? "coarse_simple" : "native",
      chunk_mode:
        runtimeRaw.chunk_mode === "off" || runtimeRaw.chunk_mode === "force"
          ? runtimeRaw.chunk_mode
          : "off",
      chunk_size_chars: toNumber(runtimeRaw.chunk_size_chars, 10000),
      task_timeout_seconds:
        typeof runtimeRaw.task_timeout_seconds === "number"
          ? runtimeRaw.task_timeout_seconds
          : null,
    },
    concurrency: toNumber(raw.concurrency, DEFAULT_EXPERIMENT_LIMITS.methods_lab_default_concurrency),
    warnings: Array.isArray(raw.warnings)
      ? raw.warnings.filter((x): x is string => typeof x === "string")
      : [],
    errors: Array.isArray(raw.errors)
      ? raw.errors.filter((x): x is string => typeof x === "string")
      : [],
    matrix: {
      models: matrixModels,
      methods: matrixMethods,
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

function normalizeMethodsLabMethod(
  raw: Record<string, unknown>,
  index: number,
): MethodsLabRunDetail["methods"][number] {
  return {
    id: typeof raw.id === "string" ? raw.id : `method_${index + 1}`,
    label: String(raw.label ?? `Method ${index + 1}`),
    method_id: String(raw.method_id ?? ""),
    method_verify_override:
      typeof raw.method_verify_override === "boolean"
        ? raw.method_verify_override
        : null,
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
  const coPrimaryMetrics = isRecord(raw.co_primary_metrics)
    ? Object.fromEntries(
        Object.entries(raw.co_primary_metrics)
          .filter(([, value]) => isRecord(value))
          .map(([metricName, value]) => {
            const metricValue = value as Record<string, unknown>;
            const metricPerLabelRaw = isRecord(metricValue.per_label)
              ? (metricValue.per_label as Record<string, unknown>)
              : {};
            const metricPerLabel: PromptLabRunDetail["matrix"]["cells"][number]["per_label"] = {};
            for (const [label, labelValue] of Object.entries(metricPerLabelRaw)) {
              if (!isRecord(labelValue)) continue;
              metricPerLabel[label] = {
                precision: toNumber(labelValue.precision, 0),
                recall: toNumber(labelValue.recall, 0),
                f1: toNumber(labelValue.f1, 0),
                tp: toNumber(labelValue.tp, 0),
                fp: toNumber(labelValue.fp, 0),
                fn: toNumber(labelValue.fn, 0),
                support: toNumber(labelValue.support, 0),
              };
            }
            const metricMicro = isRecord(metricValue.micro) ? metricValue.micro : {};
            const metricMacro = isRecord(metricValue.macro) ? metricValue.macro : {};
            return [
              metricName,
              {
                micro: {
                  precision: toNumber(metricMicro.precision, 0),
                  recall: toNumber(metricMicro.recall, 0),
                  f1: toNumber(metricMicro.f1, 0),
                  tp: toNumber(metricMicro.tp, 0),
                  fp: toNumber(metricMicro.fp, 0),
                  fn: toNumber(metricMicro.fn, 0),
                },
                macro: {
                  precision: toNumber(metricMacro.precision, 0),
                  recall: toNumber(metricMacro.recall, 0),
                  f1: toNumber(metricMacro.f1, 0),
                },
                per_label: metricPerLabel,
              },
            ];
          }),
      )
    : undefined;
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
    co_primary_metrics:
      coPrimaryMetrics && Object.keys(coPrimaryMetrics).length > 0 ? coPrimaryMetrics : undefined,
    mean_confidence: typeof raw.mean_confidence === "number" ? raw.mean_confidence : null,
  };
}

function normalizeMethodsLabCellSummary(
  raw: Record<string, unknown>,
  index: number,
  defaultDocCount: number,
): MethodsLabRunDetail["matrix"]["cells"][number] {
  const microRaw = isRecord(raw.micro) ? raw.micro : {};
  const perLabelRaw = isRecord(raw.per_label)
    ? (raw.per_label as Record<string, unknown>)
    : {};
  const perLabel: MethodsLabRunDetail["matrix"]["cells"][number]["per_label"] = {};
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
  const coPrimaryMetrics = isRecord(raw.co_primary_metrics)
    ? Object.fromEntries(
        Object.entries(raw.co_primary_metrics)
          .filter(([, value]) => isRecord(value))
          .map(([metricName, value]) => {
            const metricValue = value as Record<string, unknown>;
            const metricPerLabelRaw = isRecord(metricValue.per_label)
              ? (metricValue.per_label as Record<string, unknown>)
              : {};
            const metricPerLabel: MethodsLabRunDetail["matrix"]["cells"][number]["per_label"] = {};
            for (const [label, labelValue] of Object.entries(metricPerLabelRaw)) {
              if (!isRecord(labelValue)) continue;
              metricPerLabel[label] = {
                precision: toNumber(labelValue.precision, 0),
                recall: toNumber(labelValue.recall, 0),
                f1: toNumber(labelValue.f1, 0),
                tp: toNumber(labelValue.tp, 0),
                fp: toNumber(labelValue.fp, 0),
                fn: toNumber(labelValue.fn, 0),
                support: toNumber(labelValue.support, 0),
              };
            }
            const metricMicro = isRecord(metricValue.micro) ? metricValue.micro : {};
            const metricMacro = isRecord(metricValue.macro) ? metricValue.macro : {};
            return [
              metricName,
              {
                micro: {
                  precision: toNumber(metricMicro.precision, 0),
                  recall: toNumber(metricMicro.recall, 0),
                  f1: toNumber(metricMicro.f1, 0),
                  tp: toNumber(metricMicro.tp, 0),
                  fp: toNumber(metricMicro.fp, 0),
                  fn: toNumber(metricMicro.fn, 0),
                },
                macro: {
                  precision: toNumber(metricMacro.precision, 0),
                  recall: toNumber(metricMacro.recall, 0),
                  f1: toNumber(metricMacro.f1, 0),
                },
                per_label: metricPerLabel,
              },
            ];
          }),
      )
    : undefined;
  return {
    id: String(raw.id ?? `cell_${index + 1}`),
    model_id: String(raw.model_id ?? ""),
    model_label: String(raw.model_label ?? ""),
    method_id: String(raw.method_id ?? ""),
    method_label: String(raw.method_label ?? ""),
    status:
      (raw.status as MethodsLabRunDetail["matrix"]["cells"][number]["status"] | undefined) ??
      "pending",
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
    co_primary_metrics:
      coPrimaryMetrics && Object.keys(coPrimaryMetrics).length > 0 ? coPrimaryMetrics : undefined,
    mean_confidence: typeof raw.mean_confidence === "number" ? raw.mean_confidence : null,
  };
}

function normalizeDashboardMetrics(
  raw: Record<string, unknown>,
  reference: AnnotationSource,
  hypothesis: AnnotationSource,
  matchMode: MatchMode,
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
    const coPrimaryRaw = isRecord(row.co_primary_metrics)
      ? (row.co_primary_metrics as Record<string, unknown>)
      : {};
    const coPrimaryMetrics = Object.fromEntries(
      Object.entries(coPrimaryRaw)
        .filter(([, value]) => isRecord(value))
        .map(([key, value]) => {
          const payload = value as Record<string, unknown>;
          const micro = isRecord(payload.micro) ? payload.micro : {};
          const macro = isRecord(payload.macro) ? payload.macro : {};
          return [
            key,
            {
              micro: {
                precision: toNumber(micro.precision, 0),
                recall: toNumber(micro.recall, 0),
                f1: toNumber(micro.f1, 0),
                tp: toNumber(micro.tp, 0),
                fp: toNumber(micro.fp, 0),
                fn: toNumber(micro.fn, 0),
              },
              macro: {
                precision: toNumber(macro.precision, 0),
                recall: toNumber(macro.recall, 0),
                f1: toNumber(macro.f1, 0),
              },
            },
          ];
        }),
    );
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
      co_primary_metrics:
        Object.keys(coPrimaryMetrics).length > 0 ? coPrimaryMetrics : undefined,
      cohens_kappa: toNumber(row.cohens_kappa, 0),
      mean_iou: toNumber(row.mean_iou, 0),
      llm_confidence: normalizeLLMConfidence(row.llm_confidence),
    };
  });

  const coPrimaryRaw = isRecord(raw.co_primary_metrics)
    ? (raw.co_primary_metrics as Record<string, unknown>)
    : {};
  const coPrimaryMetrics = Object.fromEntries(
    Object.entries(coPrimaryRaw)
      .filter(([, value]) => isRecord(value))
      .map(([key, value]) => {
        const payload = value as Record<string, unknown>;
        const micro = isRecord(payload.micro) ? payload.micro : {};
        const avgDocumentMicro = isRecord(payload.avg_document_micro) ? payload.avg_document_micro : {};
        const avgDocumentMacro = isRecord(payload.avg_document_macro) ? payload.avg_document_macro : {};
        return [
          key,
          {
            micro: {
              precision: toNumber(micro.precision, 0),
              recall: toNumber(micro.recall, 0),
              f1: toNumber(micro.f1, 0),
              tp: toNumber(micro.tp, 0),
              fp: toNumber(micro.fp, 0),
              fn: toNumber(micro.fn, 0),
            },
            avg_document_micro: {
              precision: toNumber(avgDocumentMicro.precision, 0),
              recall: toNumber(avgDocumentMicro.recall, 0),
              f1: toNumber(avgDocumentMicro.f1, 0),
            },
            avg_document_macro: {
              precision: toNumber(avgDocumentMacro.precision, 0),
              recall: toNumber(avgDocumentMacro.recall, 0),
              f1: toNumber(avgDocumentMacro.f1, 0),
            },
          },
        ];
      }),
  );

  return {
    reference: (raw.reference as AnnotationSource | undefined) ?? reference,
    hypothesis: (raw.hypothesis as AnnotationSource | undefined) ?? hypothesis,
    match_mode:
      raw.match_mode === "overlap"
        ? "overlap"
        : raw.match_mode === "boundary"
          ? "boundary"
          : matchMode,
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
    co_primary_metrics: Object.keys(coPrimaryMetrics).length > 0 ? coPrimaryMetrics : undefined,
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
