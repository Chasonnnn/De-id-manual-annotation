import type { CanonicalSpan, MatchMode } from "../types";

export interface DiffSpan {
  start: number;
  end: number;
  type: "added" | "removed";
}

function spansOverlap(a: CanonicalSpan, b: CanonicalSpan): boolean {
  return a.start < b.end && b.start < a.end;
}

function iou(a: CanonicalSpan, b: CanonicalSpan): number {
  const start = Math.max(a.start, b.start);
  const end = Math.min(a.end, b.end);
  const inter = Math.max(0, end - start);
  if (inter === 0) return 0;
  const union = (a.end - a.start) + (b.end - b.start) - inter;
  return union > 0 ? inter / union : 0;
}

function isMatch(
  a: CanonicalSpan,
  b: CanonicalSpan,
  matchMode: MatchMode,
): boolean {
  if (a.label !== b.label) return false;
  if (matchMode === "exact") {
    return a.start === b.start && a.end === b.end;
  }
  // overlap mode: align with backend thresholding (IoU >= 0.5)
  if (!spansOverlap(a, b)) return false;
  return iou(a, b) >= 0.5;
}

export function computeDiff(
  reference: CanonicalSpan[],
  hypothesis: CanonicalSpan[],
  matchMode: MatchMode = "exact",
): DiffSpan[] {
  const diffs: DiffSpan[] = [];

  // Spans in hypothesis but not matched in reference => added
  for (const h of hypothesis) {
    const matched = reference.some((r) => isMatch(r, h, matchMode));
    if (!matched) {
      diffs.push({ start: h.start, end: h.end, type: "added" });
    }
  }

  // Spans in reference but not matched in hypothesis => removed
  for (const r of reference) {
    const matched = hypothesis.some((h) => isMatch(r, h, matchMode));
    if (!matched) {
      diffs.push({ start: r.start, end: r.end, type: "removed" });
    }
  }

  return diffs;
}
