import type { MethodBundle } from "./types";

export function formatMethodBundleLabel(methodBundle: MethodBundle): string {
  if (methodBundle === "legacy") return "Legacy methods";
  if (methodBundle === "test") return "Test methods";
  return "Audited methods";
}
