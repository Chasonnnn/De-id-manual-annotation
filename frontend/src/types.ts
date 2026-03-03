export type PIILabel =
  | "NAME"
  | "LOCATION"
  | "SCHOOL"
  | "DATE"
  | "AGE"
  | "PHONE"
  | "EMAIL"
  | "URL"
  | "MISC_ID";

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

export const LABEL_COLORS: Record<PIILabel, string> = {
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

export interface CanonicalSpan {
  start: number;
  end: number;
  label: PIILabel;
  text: string;
}

export interface UtteranceRow {
  speaker: string;
  text: string;
  offset: number;
}

export interface CanonicalDocument {
  id: string;
  filename: string;
  raw_text: string;
  utterances: UtteranceRow[];
  pre_annotations: CanonicalSpan[];
  manual_annotations: CanonicalSpan[];
  agent_annotations: CanonicalSpan[];
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
    { precision: number; recall: number; f1: number; support: number }
  >;
  confusion_matrix: {
    labels: string[];
    matrix: number[][];
  };
  false_positives: CanonicalSpan[];
  false_negatives: CanonicalSpan[];
}

export interface AgentConfig {
  mode: "rule" | "openai";
  system_prompt?: string;
  model?: string;
  temperature?: number;
  api_key?: string;
}

export type PaneType = "raw" | "pre" | "manual" | "agent";
export type MatchMode = "exact" | "overlap";
