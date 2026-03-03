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
  };
  agent_run_warnings: string[];
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
}

export interface AgentConfig {
  mode: "rule" | "llm" | "openai";
  system_prompt?: string;
  model?: string;
  temperature?: number;
  api_key?: string;
  reasoning_effort?: "none" | "low" | "medium" | "high" | "xhigh";
  anthropic_thinking?: boolean;
  anthropic_thinking_budget_tokens?: number;
}

export type PaneType = "raw" | "pre" | "manual" | "agent";
export type MatchMode = "exact" | "overlap";
export type AgentView = "combined" | "rule" | "llm";
export type AnnotationSource =
  | "pre"
  | "manual"
  | "agent"
  | "agent.rule"
  | "agent.llm";
