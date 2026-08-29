export interface CanonicalSpan {
  start: number;
  end: number;
  label: string;
  text: string;
}

const LABEL_COLORS: Record<string, string> = {
  NAME: "#FFD700",
  ADDRESS: "#87CEEB",
  DATE: "#DDA0DD",
  PHONE_NUMBER: "#FA8072",
  FAX_NUMBER: "#FF8C69",
  EMAIL: "#4682B4",
  SSN: "#B0C4DE",
  ACCOUNT_NUMBER: "#C0C0C0",
  DEVICE_IDENTIFIER: "#9ACD32",
  URL: "#D2B48C",
  IP_ADDRESS: "#6A5ACD",
  BIOMETRIC_IDENTIFIER: "#CD5C5C",
  IMAGE: "#778899",
  IDENTIFYING_NUMBER: "#C0C0C0",
  AGE: "#F0E68C",
  SCHOOL: "#90EE90",
  TUTOR_PROVIDER: "#20B2AA",
  CUSTOMIZED_FIELD: "#FFB347",
  OTHER_LOCATIONS_IDENTIFIED: "#7FFFD4",
};

export function getLabelColor(label: string): string {
  const normalized = label.toUpperCase();
  const known = LABEL_COLORS[normalized];
  if (known) return known;
  let hash = 0;
  for (let index = 0; index < normalized.length; index += 1) {
    hash = normalized.charCodeAt(index) + ((hash << 5) - hash);
  }
  return `hsl(${Math.abs(hash) % 360} 70% 82%)`;
}

export type UserRole = "admin" | "annotator";
export type UserState = "pending_activation" | "active" | "deactivated";
export type AssignmentState = "assigned" | "in_progress" | "completed";
export type SaveStatus = "saved" | "saving" | "conflict" | "error";

export interface HostedUser {
  id: string;
  email: string;
  display_name: string;
  role: UserRole;
  state: UserState;
}

export interface SessionSummary {
  id: string;
  external_id: string;
  filename: string;
  assignment_id: string | null;
  assignment_state: AssignmentState | null;
  assignee_id: string | null;
  assignee_name: string | null;
}

export interface WorkspaceResponse {
  sessions: SessionSummary[];
}

export interface HostedAssignment {
  id: string;
  assignee_id: string;
  assignee_name: string;
  state: AssignmentState;
}

export interface HostedDocument {
  id: string;
  external_id: string;
  filename: string;
  raw_text: string;
  label_set: string[];
  reference_annotations: CanonicalSpan[] | null;
  manual_annotations: CanonicalSpan[];
  annotation_revision: number;
  assignment: HostedAssignment | null;
}

export interface SaveAnnotationsRequest {
  spans: CanonicalSpan[];
  expected_revision: number;
  mutation_id: string;
}

export interface SaveAnnotationsResponse {
  revision: number;
  spans: CanonicalSpan[];
}

export interface AdminProgress {
  totals: {
    unassigned: number;
    assigned: number;
    in_progress: number;
    completed: number;
    total: number;
  };
  annotators: Array<{
    user_id: string;
    display_name: string;
    email: string;
    assigned: number;
    in_progress: number;
    completed: number;
  }>;
}
