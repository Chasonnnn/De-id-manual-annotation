import type { MethodBundle } from "./types";

export function formatMethodBundleLabel(methodBundle: MethodBundle): string {
  if (methodBundle === "legacy") return "Legacy methods";
  if (methodBundle === "test") return "Test methods";
  if (methodBundle === "stable") return "Stable methods";
  if (methodBundle === "v2") return "V2 methods";
  if (methodBundle === "v2+post-process") return "V2 + post-process methods";
  if (methodBundle === "deidentify-v2") return "Colleague demo V2 methods";
  return "Audited methods";
}
