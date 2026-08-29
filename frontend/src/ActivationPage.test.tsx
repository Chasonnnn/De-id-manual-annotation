import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createRootPage } from "./rootPage";
import * as api from "./hosted/api";

vi.mock("./hosted/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./hosted/api")>();
  return {
    ...actual,
    activate: vi.fn(),
    getCurrentUser: vi.fn(),
  };
});

describe("account activation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.activate).mockReset();
    vi.mocked(api.getCurrentUser).mockReset();
  });

  afterEach(() => {
    cleanup();
    window.history.replaceState(null, "", "/");
  });

  it.each([
    ["missing", ""],
    ["malformed", "#token=valid_token&extra=value"],
  ])("rejects a %s activation fragment without bootstrapping authentication", async (_name, fragment) => {
    window.history.replaceState(null, "", `/activate${fragment}`);

    render(createRootPage());

    expect(await screen.findByRole("heading", { name: "Activation link unavailable" })).toBeTruthy();
    expect(api.getCurrentUser).not.toHaveBeenCalled();
    expect(window.location.hash).toBe("");
  });

  it("rejects and removes query-string activation material", async () => {
    window.history.replaceState(null, "", "/activate?token=must-not-remain");

    render(createRootPage());

    expect(await screen.findByRole("heading", { name: "Activation link unavailable" })).toBeTruthy();
    expect(api.getCurrentUser).not.toHaveBeenCalled();
    expect(window.location.search).toBe("");
  });

  it("validates password length and confirmation before submitting", async () => {
    window.history.replaceState(null, "", "/activate#token=valid_token");
    render(createRootPage());

    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "short" } });
    fireEvent.change(screen.getByLabelText("Confirm password"), { target: { value: "short" } });
    fireEvent.click(screen.getByRole("button", { name: "Activate account" }));

    expect((await screen.findByRole("alert")).textContent).toContain("at least 12 characters");
    expect(api.activate).not.toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "long-enough-password" } });
    fireEvent.change(screen.getByLabelText("Confirm password"), { target: { value: "different-password" } });
    fireEvent.click(screen.getByRole("button", { name: "Activate account" }));

    expect((await screen.findByRole("alert")).textContent).toContain("must match");
    expect(api.activate).not.toHaveBeenCalled();
  });

  it("activates the account with the captured token", async () => {
    vi.mocked(api.activate).mockResolvedValue({
      id: "user-1",
      email: "annotator@cornell.edu",
      display_name: "Ada Annotator",
      role: "annotator",
      state: "active",
    });
    window.history.replaceState(null, "", "/activate#token=opaque_token");
    render(createRootPage());

    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "long-enough-password" } });
    fireEvent.change(screen.getByLabelText("Confirm password"), { target: { value: "long-enough-password" } });
    fireEvent.click(screen.getByRole("button", { name: "Activate account" }));

    expect(await screen.findByRole("heading", { name: "Account activated" })).toBeTruthy();
    expect(api.activate).toHaveBeenCalledWith("opaque_token", "long-enough-password");
    expect(window.location.hash).toBe("");
  });

  it("shows an explicit activation API failure", async () => {
    vi.mocked(api.activate).mockRejectedValue(new api.ApiError(400, "Activation link expired."));
    window.history.replaceState(null, "", "/activate#token=expired_token");
    render(createRootPage());

    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "long-enough-password" } });
    fireEvent.change(screen.getByLabelText("Confirm password"), { target: { value: "long-enough-password" } });
    fireEvent.click(screen.getByRole("button", { name: "Activate account" }));

    expect((await screen.findByRole("alert")).textContent).toBe("Activation link expired.");
    expect(screen.getByRole("button", { name: "Activate account" })).toBeTruthy();
  });
});
