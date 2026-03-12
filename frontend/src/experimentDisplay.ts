import type { MethodBundle } from "./types";

export function formatMethodBundleLabel(methodBundle: MethodBundle): string {
  if (methodBundle === "legacy") return "Legacy methods";
  if (methodBundle === "stable") return "Stable methods";
  if (methodBundle === "audited") return "Audited methods";
  return "V2 methods";
}
