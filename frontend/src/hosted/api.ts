import type {
  AdminProgress,
  HostedDocument,
  HostedUser,
  SaveAnnotationsRequest,
  SaveAnnotationsResponse,
  SessionFolder,
  WorkspaceResponse,
} from "./types";

export interface ActivationResponse {
  user: HostedUser;
  activation_url: string;
  activation_expires_at: string;
}

export type IncompleteAssignmentAction =
  | { action: "unassign" }
  | { action: "reassign"; assignee_id: string };

export interface BulkAssignmentPreview {
  plan_digest: string;
  assignments: Array<{ document_id: string; assignee_id: string }>;
}

export interface BulkAssignmentResult {
  plan_digest: string;
  mutation_id: string;
  assignment_ids: string[];
}

export class ApiError extends Error {
  public readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function readCookie(name: string): string | undefined {
  const prefix = `${encodeURIComponent(name)}=`;
  return document.cookie
    .split(";")
    .map((cookie) => cookie.trim())
    .find((cookie) => cookie.startsWith(prefix))
    ?.slice(prefix.length);
}

async function request<T>(path: string, init?: RequestInit, canRefreshCsrf = true): Promise<T> {
  const method = (init?.method ?? "GET").toUpperCase();
  const csrfToken = method === "GET" || method === "HEAD" || path === "/api/auth/login"
    ? undefined
    : readCookie("annotation_csrf");
  const response = await fetch(path, {
    credentials: "same-origin",
    ...init,
    headers: {
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...(csrfToken ? { "X-CSRF-Token": decodeURIComponent(csrfToken) } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: unknown; message?: unknown };
      const detail = body.detail ?? body.message;
      if (typeof detail === "string" && detail.trim()) message = detail;
    } catch {
      // The HTTP status still provides an explicit failure when no JSON body exists.
    }
    const error = new ApiError(response.status, message);
    if (
      canRefreshCsrf
      && method !== "GET"
      && method !== "HEAD"
      && error.status === 403
      && error.message === "CSRF validation failed"
    ) {
      const refreshed = await fetch("/api/auth/csrf", { credentials: "same-origin" });
      if (refreshed.ok) return request(path, init, false);
    }
    throw error;
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export function getCurrentUser(): Promise<HostedUser> {
  return request("/api/auth/me");
}

export function login(email: string, password: string): Promise<HostedUser> {
  return request("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function activate(token: string, password: string): Promise<HostedUser> {
  return request("/api/auth/activate", {
    method: "POST",
    body: JSON.stringify({ token, password }),
  });
}

export function logout(): Promise<void> {
  return request("/api/auth/logout", { method: "POST" });
}

export function getWorkspace(): Promise<WorkspaceResponse> {
  return request("/api/workspace");
}

export function getDocument(documentId: string): Promise<HostedDocument> {
  return request(`/api/documents/${encodeURIComponent(documentId)}`);
}

export function saveAnnotations(
  documentId: string,
  payload: SaveAnnotationsRequest,
): Promise<SaveAnnotationsResponse> {
  return request(`/api/documents/${encodeURIComponent(documentId)}/annotations`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export function completeAssignment(
  assignmentId: string,
): Promise<{ assignment_id: string; state: "completed" }> {
  return request(`/api/assignments/${encodeURIComponent(assignmentId)}/complete`, {
    method: "POST",
  });
}

export function getAdminProgress(): Promise<AdminProgress> {
  return request("/api/admin/progress");
}

export function getAdminFolders(): Promise<SessionFolder[]> {
  return request("/api/admin/folders");
}

export function createAdminFolder(name: string): Promise<SessionFolder> {
  return request("/api/admin/folders", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export function moveSessionsToFolder(
  folderId: string,
  documentIds: string[],
): Promise<SessionFolder> {
  return request(`/api/admin/folders/${encodeURIComponent(folderId)}/sessions`, {
    method: "PUT",
    body: JSON.stringify({ document_ids: documentIds }),
  });
}

export function assignFolder(
  folderId: string,
  assigneeId: string,
): Promise<{ folder_id: string; assignment_ids: string[] }> {
  return request(`/api/admin/folders/${encodeURIComponent(folderId)}/assignment`, {
    method: "PUT",
    body: JSON.stringify({ assignee_id: assigneeId }),
  });
}

export function getAdminUsers(): Promise<HostedUser[]> {
  return request("/api/admin/users");
}

export function createAdminUser(payload: {
  email: string;
  role: "annotator";
}): Promise<ActivationResponse> {
  return request("/api/admin/users", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function resetAdminUserPassword(userId: string): Promise<ActivationResponse> {
  return request(`/api/admin/users/${encodeURIComponent(userId)}/reset-password`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function deactivateAdminUser(
  userId: string,
  incompleteAssignments: IncompleteAssignmentAction,
): Promise<HostedUser> {
  return request(`/api/admin/users/${encodeURIComponent(userId)}/deactivate`, {
    method: "POST",
    body: JSON.stringify({ incomplete_assignments: incompleteAssignments }),
  });
}

export function reactivateAdminUser(userId: string): Promise<HostedUser> {
  return request(`/api/admin/users/${encodeURIComponent(userId)}/reactivate`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function assignSession(payload: {
  document_id: string;
  assignee_id: string;
}): Promise<{ assignment_id: string }> {
  return request(`/api/admin/documents/${encodeURIComponent(payload.document_id)}/assignment`, {
    method: "PUT",
    body: JSON.stringify({ assignee_id: payload.assignee_id }),
  });
}

export function previewBulkAssignment(
  documentIds: string[],
  annotatorIds: string[],
): Promise<BulkAssignmentPreview> {
  return request("/api/admin/assignments/bulk/preview", {
    method: "POST",
    body: JSON.stringify({
      document_ids: documentIds,
      annotator_ids: annotatorIds,
    }),
  });
}

export function applyBulkAssignment(
  documentIds: string[],
  annotatorIds: string[],
  planDigest: string,
  mutationId: string,
): Promise<BulkAssignmentResult> {
  return request("/api/admin/assignments/bulk/apply", {
    method: "POST",
    body: JSON.stringify({
      document_ids: documentIds,
      annotator_ids: annotatorIds,
      plan_digest: planDigest,
      mutation_id: mutationId,
    }),
  });
}
