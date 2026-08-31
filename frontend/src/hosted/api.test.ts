import { afterEach, describe, expect, it, vi } from "vitest";
import {
  activate,
  assignFolder,
  assignSession,
  createAdminFolder,
  createAdminUser,
  deactivateAdminUser,
  login,
  moveSessionsToFolder,
  reactivateAdminUser,
  resetAdminUserPassword,
  saveAnnotations,
} from "./api";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  document.cookie = "annotation_csrf=; Max-Age=0; path=/";
});

describe("hosted API client", () => {
  it("keeps activation tokens out of the request URL and logs", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: "user-1", state: "active" }));
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    vi.stubGlobal("fetch", fetchMock);

    await activate("opaque_token", "correct horse battery staple");

    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/auth/activate");
    expect(String(fetchMock.mock.calls[0]?.[0])).not.toContain("opaque_token");
    expect(fetchMock.mock.calls[0]?.[1]).toEqual(expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        token: "opaque_token",
        password: "correct horse battery staple",
      }),
    }));
    expect(consoleError).not.toHaveBeenCalled();
  });

  it("uses cookie authentication for email/password login", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: "user-1" }));
    vi.stubGlobal("fetch", fetchMock);

    await login("annotator@cornell.edu", "secret");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/auth/login",
      expect.objectContaining({
        method: "POST",
        credentials: "same-origin",
        body: JSON.stringify({ email: "annotator@cornell.edu", password: "secret" }),
      }),
    );
  });

  it("sends the revision and mutation id with the complete span snapshot", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ revision: 4, spans: [] }));
    vi.stubGlobal("fetch", fetchMock);

    await saveAnnotations("doc/1", {
      spans: [],
      expected_revision: 3,
      mutation_id: "mutation-1",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/documents/doc%2F1/annotations",
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({ spans: [], expected_revision: 3, mutation_id: "mutation-1" }),
      }),
    );
  });

  it("adds the CSRF cookie value to authenticated writes", async () => {
    document.cookie = "annotation_csrf=csrf-token-123; path=/";
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ revision: 1, spans: [] }));
    vi.stubGlobal("fetch", fetchMock);

    await saveAnnotations("doc-1", {
      spans: [],
      expected_revision: 0,
      mutation_id: "mutation-csrf",
    });

    expect(fetchMock.mock.calls[0]?.[1]).toEqual(expect.objectContaining({
      headers: expect.objectContaining({ "X-CSRF-Token": "csrf-token-123" }),
    }));
  });

  it("refreshes a rejected CSRF cookie and retries the identical save once", async () => {
    document.cookie = "annotation_csrf=stale-token; path=/";
    const payload = {
      spans: [{ start: 0, end: 5, label: "NAME", text: "Adoni" }],
      expected_revision: 0,
      mutation_id: "mutation-csrf-recovery",
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ detail: "CSRF validation failed" }, 403))
      .mockImplementationOnce(async () => {
        document.cookie = "annotation_csrf=fresh-token; path=/";
        return new Response(null, { status: 204 });
      })
      .mockResolvedValueOnce(jsonResponse({ revision: 1, spans: payload.spans }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(saveAnnotations("doc-1", payload)).resolves.toEqual({
      revision: 1,
      spans: payload.spans,
    });

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/documents/doc-1/annotations");
    expect(fetchMock.mock.calls[1]?.[0]).toBe("/api/auth/csrf");
    expect(fetchMock.mock.calls[2]?.[0]).toBe("/api/documents/doc-1/annotations");
    expect(fetchMock.mock.calls[2]?.[1]).toEqual(expect.objectContaining({
      method: "PUT",
      body: JSON.stringify(payload),
      headers: expect.objectContaining({ "X-CSRF-Token": "fresh-token" }),
    }));
  });

  it("surfaces the original CSRF failure when token refresh is rejected", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ detail: "CSRF validation failed" }, 403))
      .mockResolvedValueOnce(jsonResponse({ detail: "Authentication required" }, 401));
    vi.stubGlobal("fetch", fetchMock);

    await expect(saveAnnotations("doc-1", {
      spans: [],
      expected_revision: 0,
      mutation_id: "mutation-expired-session",
    })).rejects.toEqual(expect.objectContaining({
      status: 403,
      message: "CSRF validation failed",
    }));
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[1]?.[0]).toBe("/api/auth/csrf");
  });

  it("assigns a document through the idempotent document endpoint", async () => {
    const fetchMock = vi.fn().mockImplementation(async () =>
      jsonResponse({ assignment_id: "assignment-1" }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await assignSession({ document_id: "doc-1", assignee_id: "user-2" });

    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/admin/documents/doc-1/assignment");
    expect(fetchMock.mock.calls[0]?.[1]).toEqual(expect.objectContaining({
      method: "PUT",
      body: JSON.stringify({ assignee_id: "user-2" }),
    }));
  });

  it("creates folders, moves sessions, and assigns the folder", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => jsonResponse({ id: "folder-1" }));
    vi.stubGlobal("fetch", fetchMock);

    await createAdminFolder("August intake");
    await moveSessionsToFolder("folder/1", ["doc-2", "doc-1"]);
    await assignFolder("folder/1", "user-1");

    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/admin/folders");
    expect(fetchMock.mock.calls[0]?.[1]).toEqual(expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ name: "August intake" }),
    }));
    expect(fetchMock.mock.calls[1]?.[0]).toBe("/api/admin/folders/folder%2F1/sessions");
    expect(fetchMock.mock.calls[1]?.[1]).toEqual(expect.objectContaining({
      method: "PUT",
      body: JSON.stringify({ document_ids: ["doc-2", "doc-1"] }),
    }));
    expect(fetchMock.mock.calls[2]?.[0]).toBe("/api/admin/folders/folder%2F1/assignment");
    expect(fetchMock.mock.calls[2]?.[1]).toEqual(expect.objectContaining({
      method: "PUT",
      body: JSON.stringify({ assignee_id: "user-1" }),
    }));
  });

  it("preserves explicit HTTP failures for conflict handling", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ detail: "Stale revision" }, 409)));

    await expect(saveAnnotations("doc-1", {
      spans: [],
      expected_revision: 2,
      mutation_id: "mutation-2",
    })).rejects.toEqual(expect.objectContaining({
      status: 409,
      message: "Stale revision",
    }));
  });

  it("creates annotators through the admin endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: "user-1" }));
    vi.stubGlobal("fetch", fetchMock);

    await createAdminUser({
      email: "annotator@cornell.edu",
      role: "annotator",
    });
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/admin/users");
    expect(fetchMock.mock.calls[0]?.[1]).toEqual(expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        email: "annotator@cornell.edu",
        role: "annotator",
      }),
    }));
  });

  it("resets an annotator password through the account endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ user: { id: "user-1" } }));
    vi.stubGlobal("fetch", fetchMock);

    await resetAdminUserPassword("user/1");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/users/user%2F1/reset-password",
      expect.objectContaining({ method: "POST", body: "{}" }),
    );
  });

  it("deactivates an annotator only with the selected unfinished-work action", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: "user-1", state: "deactivated" }));
    vi.stubGlobal("fetch", fetchMock);

    await deactivateAdminUser("user-1", {
      action: "reassign",
      assignee_id: "user-2",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/users/user-1/deactivate",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          incomplete_assignments: { action: "reassign", assignee_id: "user-2" },
        }),
      }),
    );
  });

  it("reactivates an annotator through the account endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: "user-1", state: "active" }));
    vi.stubGlobal("fetch", fetchMock);

    await reactivateAdminUser("user-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/users/user-1/reactivate",
      expect.objectContaining({ method: "POST", body: "{}" }),
    );
  });
});
