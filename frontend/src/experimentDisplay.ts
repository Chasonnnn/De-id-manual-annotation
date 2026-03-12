import type { MethodBundle } from "./types";

export function formatMethodBundleLabel(methodBundle: MethodBundle): string {
  if (methodBundle === "legacy") return "Legacy methods";
  if (methodBundle === "test") return "Test methods";
  if (methodBundle === "v2+post-process") return "V2 + post-process methods";
  return "Audited methods";
}
