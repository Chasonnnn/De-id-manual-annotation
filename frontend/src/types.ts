export type PIILabel = string;
export type LabelProfile = "simple" | "advanced";
export type LabelProjection = "native" | "coarse_simple";

export const PII_LABELS: PIILabel[] = [
  "NAME",
  "LOCATION",
  "SCHOOL",
  "DATE",
  "AGE",
  "PHONE",
  "EMAIL",
  "URL",
  "MISC_ID",
];

export const LABEL_COLORS: Record<string, string> = {
  NAME: "#FFD700",
  LOCATION: "#87CEEB",
  SCHOOL: "#90EE90",
  DATE: "#DDA0DD",
  AGE: "#F0E68C",
  PHONE: "#FA8072",
  EMAIL: "#4682B4",
  URL: "#D2B48C",
  MISC_ID: "#C0C0C0",
};

export function getLabelColor(label: string): string {
  const normalized = label.toUpperCase();
  const known = LABEL_COLORS[normalized];
  if (known) return known;
  let hash = 0;
  for (let i = 0; i < normalized.length; i += 1) {
    hash = normalized.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue} 70% 82%)`;
}

export interface CanonicalSpan {
  start: number;
  end: number;
  label: string;
  text: string;
}

export type LLMConfidenceReason =
  | "ok"
  | "unsupported_provider"
  | "missing_logprobs"
  | "empty_completion";

export type LLMConfidenceBand = "high" | "medium" | "low" | "na";

export interface LLMConfidenceMetric {
  available: boolean;
  provider: string;
  model: string;
  reason: LLMConfidenceReason;
  token_count: number;
  mean_logprob: number | null;
  confidence: number | null;
  perplexity: number | null;
  band: LLMConfidenceBand;
  high_threshold: number;
  medium_threshold: number;
}

export interface AgentChunkDiagnostic {
  chunk_index: number;
  start: number;
  end: number;
  char_count: number;
  span_count: number;
  attempt_count: number;
  retry_used: boolean;
  suspicious_empty: boolean;
  status: "completed" | "failed";
  finish_reason: string | null;
  warnings: string[];
}

export interface AgentRunMetrics {
  llm_confidence: LLMConfidenceMetric | null;
  label_profile?: LabelProfile | null;
  chunk_diagnostics?: AgentChunkDiagnostic[];
}

export interface SavedRunMetadata {
  mode: "manual" | "rule" | "llm" | "method";
  updated_at: string;
  model?: string | null;
  method_id?: string | null;
  label_profile?: LabelProfile | null;
  prompt_snapshot?: Record<string, unknown> | null;
  llm_confidence?: LLMConfidenceMetric | null;
  chunk_diagnostics?: AgentChunkDiagnostic[];
}

export interface UtteranceRow {
  speaker: string;
  text: string;
  global_start: number;
  global_end: number;
}

export interface CanonicalDocument {
  id: string;
  filename: string;
  raw_text: string;
  utterances: UtteranceRow[];
  pre_annotations: CanonicalSpan[];
  label_set: string[];
  manual_annotations: CanonicalSpan[];
  agent_annotations: CanonicalSpan[];
  agent_outputs: {
    rule: CanonicalSpan[];
    llm: CanonicalSpan[];
    llm_runs?: Record<string, CanonicalSpan[]>;
    llm_run_metadata?: Record<string, SavedRunMetadata>;
    methods: Record<string, CanonicalSpan[]>;
    method_run_metadata?: Record<string, SavedRunMetadata>;
  };
  agent_run_warnings: string[];
  agent_run_metrics: AgentRunMetrics;
  status: "pending" | "in_progress" | "reviewed";
}

export interface DocumentSummary {
  id: string;
  filename: string;
  status: "pending" | "in_progress" | "reviewed";
}

export interface MetricsResult {
  micro: { precision: number; recall: number; f1: number };
  macro: { precision: number; recall: number; f1: number };
  label_projection?: LabelProjection;
  per_label: Record<
    string,
    {
      precision: number;
      recall: number;
      f1: number;
      support: number;
      tp?: number;
      fp?: number;
      fn?: number;
    }
  >;
  confusion_matrix?: {
    labels: string[];
    matrix: number[][];
  };
  false_positives?: CanonicalSpan[];
  false_negatives?: CanonicalSpan[];
  llm_confidence?: LLMConfidenceMetric | null;
}

export interface DashboardDocumentMetrics {
  id: string;
  filename: string;
  reference_count: number;
  hypothesis_count: number;
  micro: {
    precision: number;
    recall: number;
    f1: number;
    tp: number;
    fp: number;
    fn: number;
  };
  macro: { precision: number; recall: number; f1: number };
  cohens_kappa: number;
  mean_iou: number;
  llm_confidence?: LLMConfidenceMetric | null;
}

export interface DashboardMetricsResult {
  reference: AnnotationSource;
  hypothesis: AnnotationSource;
  match_mode: MatchMode;
  label_projection?: LabelProjection;
  total_documents: number;
  documents_compared: number;
  micro: {
    precision: number;
    recall: number;
    f1: number;
    tp: number;
    fp: number;
    fn: number;
  };
  avg_document_micro: { precision: number; recall: number; f1: number };
  avg_document_macro: { precision: number; recall: number; f1: number };
  llm_confidence_summary: {
    documents_with_confidence: number;
    mean_confidence: number | null;
    band_counts: Record<LLMConfidenceBand, number>;
  };
  documents: DashboardDocumentMetrics[];
}

export interface SessionExportBundle {
  format: string;
  version: number;
  project?: SessionProfile;
  compatibility?: {
    tool_version?: string;
    import_supported_versions?: number[];
  };
  exported_at: string;
  prompt_lab_runs?: PromptLabRunExport[];
  methods_lab_runs?: MethodsLabRunExport[];
  documents: Array<{
    source: CanonicalDocument;
    manual_annotations: CanonicalSpan[];
    agent_outputs: {
      rule: CanonicalSpan[];
      llm: CanonicalSpan[];
      methods?: Record<string, CanonicalSpan[]>;
    };
    agent_saved_outputs?: {
      llm_runs?: Record<string, CanonicalSpan[]>;
      method_runs?: Record<string, CanonicalSpan[]>;
      llm_run_metadata?: Record<string, SavedRunMetadata>;
      method_run_metadata?: Record<string, SavedRunMetadata>;
    };
    agent_metrics?: {
      llm_confidence?: LLMConfidenceMetric | null;
      label_profile?: LabelProfile | null;
      chunk_diagnostics?: AgentChunkDiagnostic[];
    };
  }>;
  config?: Record<string, unknown>;
}

export interface SessionImportResult {
  bundle_version?: number;
  imported_count: number;
  imported_ids: string[];
  imported_prompt_lab_runs?: number;
  imported_methods_lab_runs?: number;
  skipped_count: number;
  skipped: Array<{ index: number; reason: string }>;
  warnings?: string[];
  total_in_bundle: number;
}

export interface SessionProfile {
  project_name: string;
  author: string;
}

export interface AgentConfig {
  mode: "rule" | "llm" | "method" | "openai";
  system_prompt?: string;
  model?: string;
  temperature?: number;
  api_key?: string;
  api_base?: string;
  reasoning_effort?: "none" | "low" | "medium" | "high" | "xhigh";
  anthropic_thinking?: boolean;
  anthropic_thinking_budget_tokens?: number;
  chunk_mode?: "auto" | "off" | "force";
  chunk_size_chars?: number;
  label_profile?: LabelProfile;
  method_id?: string;
  method_verify?: boolean;
}

export interface AgentMethodPromptTemplate {
  pass_index: number;
  entity_types: string[] | null;
  system_prompt: string;
  source?: "builtin" | "saved";
}

export interface AgentMethodOption {
  id: string;
  label: string;
  description: string;
  requires_presidio: boolean;
  uses_llm: boolean;
  supports_verify_override: boolean;
  default_verify?: boolean;
  prompt_templates?: AgentMethodPromptTemplate[];
  available: boolean;
  unavailable_reason: string | null;
}

export interface AgentCredentialStatus {
  has_api_key: boolean;
  api_key_sources: string[];
  has_api_base: boolean;
  api_base_sources: string[];
}

export interface ExperimentLimits {
  prompt_lab_default_concurrency: number;
  prompt_lab_max_concurrency: number;
  methods_lab_default_concurrency: number;
  methods_lab_max_concurrency: number;
}

export const DEFAULT_EXPERIMENT_LIMITS: ExperimentLimits = {
  prompt_lab_default_concurrency: 4,
  prompt_lab_max_concurrency: 16,
  methods_lab_default_concurrency: 4,
  methods_lab_max_concurrency: 16,
};

export interface AgentRunProgress {
  doc_id: string;
  mode: string | null;
  status: "idle" | "running" | "completed" | "failed";
  completed_chunks: number;
  total_chunks: number;
  progress: number;
  started_at: string | null;
  updated_at: string;
  message?: string | null;
}

export type PromptLabRunStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "completed_with_errors"
  | "failed";

export interface PromptLabPromptInput {
  id?: string;
  label: string;
  variant_type?: "prompt" | "preset";
  preset_method_id?: string | null;
  method_verify_override?: boolean | null;
  system_prompt?: string;
}

export interface PromptLabModelInput {
  id?: string;
  label: string;
  model: string;
  reasoning_effort?: "none" | "low" | "medium" | "high" | "xhigh";
  anthropic_thinking?: boolean;
  anthropic_thinking_budget_tokens?: number | null;
}

export interface PromptLabRuntimeInput {
  api_key?: string;
  api_base?: string;
  temperature: number;
  match_mode: MatchMode;
  reference_source: "manual" | "pre";
  fallback_reference_source: "manual" | "pre";
  label_profile?: LabelProfile;
  label_projection?: LabelProjection;
  chunk_mode?: "auto" | "off" | "force";
  chunk_size_chars?: number;
}

export interface PromptLabRunCreateRequest {
  name?: string;
  doc_ids: string[];
  prompts: PromptLabPromptInput[];
  models: PromptLabModelInput[];
  runtime: PromptLabRuntimeInput;
  concurrency: number;
}

export interface MethodsLabMethodInput {
  id?: string;
  label: string;
  method_id: string;
  method_verify_override?: boolean | null;
}

export interface MethodsLabRuntimeInput {
  api_key?: string;
  api_base?: string;
  temperature: number;
  match_mode: MatchMode;
  label_profile?: LabelProfile;
  label_projection?: LabelProjection;
  chunk_mode?: "auto" | "off" | "force";
  chunk_size_chars?: number;
}

export interface MethodsLabRunCreateRequest {
  name?: string;
  doc_ids: string[];
  methods: MethodsLabMethodInput[];
  models: PromptLabModelInput[];
  runtime: MethodsLabRuntimeInput;
  concurrency: number;
}

export interface PromptLabCellMicro {
  precision: number;
  recall: number;
  f1: number;
  tp: number;
  fp: number;
  fn: number;
}

export interface PromptLabMatrixCellSummary {
  id: string;
  model_id: string;
  model_label: string;
  prompt_id: string;
  prompt_label: string;
  status: PromptLabRunStatus | "pending";
  total_docs: number;
  completed_docs: number;
  failed_docs: number;
  error_count: number;
  micro: PromptLabCellMicro;
  per_label: Record<
    string,
    {
      precision: number;
      recall: number;
      f1: number;
      tp: number;
      fp: number;
      fn: number;
      support: number;
    }
  >;
  mean_confidence: number | null;
}

export interface PromptLabRunSummary {
  id: string;
  name: string;
  status: PromptLabRunStatus;
  cancellable: boolean;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  doc_count: number;
  prompt_count: number;
  model_count: number;
  total_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
}

export interface PromptLabRunDetail extends PromptLabRunSummary {
  doc_ids: string[];
  prompts: PromptLabPromptInput[];
  models: PromptLabModelInput[];
  runtime: Omit<PromptLabRuntimeInput, "api_key"> & { api_base?: string };
  concurrency: number;
  warnings: string[];
  errors: string[];
  matrix: {
    models: Array<{ id: string; label: string }>;
    prompts: Array<{ id: string; label: string }>;
    cells: PromptLabMatrixCellSummary[];
    available_labels: string[];
  };
  progress: {
    total_tasks: number;
    completed_tasks: number;
    failed_tasks: number;
  };
}

export interface PromptLabDocResult {
  run_id: string;
  cell_id: string;
  doc_id: string;
  status: "pending" | "completed" | "failed" | "unavailable" | "cancelled";
  error?: string | null;
  warnings: string[];
  reference_source_used?: "manual" | "pre";
  reference_spans: CanonicalSpan[];
  hypothesis_spans: CanonicalSpan[];
  metrics: MetricsResult | null;
  llm_confidence: LLMConfidenceMetric | null;
  transcript_text: string | null;
  document: { id: string; filename: string | null };
  model?: PromptLabModelInput;
  prompt?: PromptLabPromptInput;
}

export interface PromptLabRunExport {
  id: string;
  [key: string]: unknown;
}

export interface MethodsLabMatrixCellSummary {
  id: string;
  model_id: string;
  model_label: string;
  method_id: string;
  method_label: string;
  status: PromptLabRunStatus | "pending";
  total_docs: number;
  completed_docs: number;
  failed_docs: number;
  error_count: number;
  micro: PromptLabCellMicro;
  per_label: Record<
    string,
    {
      precision: number;
      recall: number;
      f1: number;
      tp: number;
      fp: number;
      fn: number;
      support: number;
    }
  >;
  mean_confidence: number | null;
}

export interface MethodsLabRunSummary {
  id: string;
  name: string;
  status: PromptLabRunStatus;
  cancellable: boolean;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  doc_count: number;
  method_count: number;
  model_count: number;
  total_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
}

export interface MethodsLabRunDetail extends MethodsLabRunSummary {
  doc_ids: string[];
  methods: MethodsLabMethodInput[];
  models: PromptLabModelInput[];
  runtime: Omit<MethodsLabRuntimeInput, "api_key"> & { api_base?: string };
  concurrency: number;
  warnings: string[];
  errors: string[];
  matrix: {
    models: Array<{ id: string; label: string }>;
    methods: Array<{ id: string; label: string }>;
    cells: MethodsLabMatrixCellSummary[];
    available_labels: string[];
  };
  progress: {
    total_tasks: number;
    completed_tasks: number;
    failed_tasks: number;
  };
}

export interface MethodsLabDocResult {
  run_id: string;
  cell_id: string;
  doc_id: string;
  status: "pending" | "completed" | "failed" | "unavailable" | "cancelled";
  error?: string | null;
  warnings: string[];
  reference_source_used?: "manual";
  reference_spans: CanonicalSpan[];
  hypothesis_spans: CanonicalSpan[];
  metrics: MetricsResult | null;
  llm_confidence: LLMConfidenceMetric | null;
  transcript_text: string | null;
  document: { id: string; filename: string | null };
  model?: PromptLabModelInput;
  method?: MethodsLabMethodInput;
}

export interface MethodsLabRunExport {
  id: string;
  [key: string]: unknown;
}

export type PaneType = "raw" | "pre" | "manual" | "agent" | "methods";
export type MatchMode = "exact" | "boundary" | "overlap";
export type AgentView = "combined" | "rule" | "llm";
export type MethodView = string;
export type AnnotationSource =
  | "pre"
  | "manual"
  | "agent"
  | "agent.rule"
  | "agent.llm"
  | `agent.llm_run.${string}`
  | `agent.method.${string}`;
