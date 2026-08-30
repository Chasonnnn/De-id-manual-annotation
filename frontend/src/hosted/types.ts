export interface CanonicalSpan {
  start: number;
  end: number;
  label: string;
  text: string;
}

export const ENTITY_TYPES = [
  "NAME",
  "ADDRESS",
  "DATE",
  "PHONE_NUMBER",
  "FAX_NUMBER",
  "EMAIL",
  "SSN",
  "ACCOUNT_NUMBER",
  "DEVICE_IDENTIFIER",
  "URL",
  "IP_ADDRESS",
  "BIOMETRIC_IDENTIFIER",
  "IMAGE",
  "IDENTIFYING_NUMBER",
  "AGE",
  "SCHOOL",
  "TUTOR_PROVIDER",
  "CUSTOMIZED_FIELD",
  "OTHER_LOCATIONS_IDENTIFIED",
] as const;

const LABEL_COLORS: Record<string, string> = {
  NAME: "#ffe36e",
  ADDRESS: "#a8d8f0",
  DATE: "#e3b7eb",
  PHONE_NUMBER: "#f5a08f",
  FAX_NUMBER: "#f7b29f",
  EMAIL: "#9bbbea",
  SSN: "#c3cce0",
  ACCOUNT_NUMBER: "#d1d5db",
  DEVICE_IDENTIFIER: "#f5c27b",
  URL: "#d9c19f",
  IP_ADDRESS: "#b9a9e8",
  BIOMETRIC_IDENTIFIER: "#e5aaa7",
  IMAGE: "#b8c1cc",
  IDENTIFYING_NUMBER: "#f5c27b",
  AGE: "#efe49b",
  SCHOOL: "#9ee6ad",
  TUTOR_PROVIDER: "#8fd7ce",
  CUSTOMIZED_FIELD: "#f4bd82",
  OTHER_LOCATIONS_IDENTIFIED: "#94e4d6",
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
  folder_id: string | null;
  folder_name: string | null;
  assignment_id: string | null;
  assignment_state: AssignmentState | null;
  manual_annotation_count: number;
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
  folders: SessionFolder[];
}

export interface SessionFolder {
  id: string;
  name: string;
  session_count: number;
  unassigned: number;
  assigned: number;
  in_progress: number;
  completed: number;
}
