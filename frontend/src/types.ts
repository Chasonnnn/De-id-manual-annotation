export type PIILabel = string;

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

export interface AgentRunMetrics {
  llm_confidence: LLMConfidenceMetric | null;
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
    methods: Record<string, CanonicalSpan[]>;
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
  documents: Array<{
    source: CanonicalDocument;
    manual_annotations: CanonicalSpan[];
    agent_outputs: {
      rule: CanonicalSpan[];
      llm: CanonicalSpan[];
      methods?: Record<string, CanonicalSpan[]>;
    };
    agent_metrics?: {
      llm_confidence?: LLMConfidenceMetric | null;
    };
  }>;
  config?: Record<string, unknown>;
}

export interface SessionImportResult {
  bundle_version?: number;
  imported_count: number;
  imported_ids: string[];
  skipped_count: number;
  skipped: Array<{ index: number; reason: string }>;
  warnings?: string[];
  total_in_bundle: number;
}

export interface SessionProfile {
  project_name: string;
  author: string;
  notes: string;
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
  method_id?: string;
  method_verify?: boolean;
}

export interface AgentMethodOption {
  id: string;
  label: string;
  description: string;
  requires_presidio: boolean;
  uses_llm: boolean;
  supports_verify_override: boolean;
  available: boolean;
  unavailable_reason: string | null;
}

export interface AgentCredentialStatus {
  has_api_key: boolean;
  api_key_sources: string[];
  has_api_base: boolean;
  api_base_sources: string[];
}

export type PaneType = "raw" | "pre" | "manual" | "agent" | "methods";
export type MatchMode = "exact" | "overlap";
export type AgentView = "combined" | "rule" | "llm";
export type MethodView = string;
export type AnnotationSource =
  | "pre"
  | "manual"
  | "agent"
  | "agent.rule"
  | "agent.llm"
  | `agent.method.${string}`;
