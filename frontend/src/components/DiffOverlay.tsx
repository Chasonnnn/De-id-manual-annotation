import type { CanonicalSpan, MatchMode } from "../types";
import { getCodePointLength, sliceByCodePointOffsets } from "../textOffsets";

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

function isBoundaryIgnorable(char: string): boolean {
  return /[\p{P}\s]/u.test(char);
}

function trimBoundaryOffsets(span: CanonicalSpan): { start: number; end: number } {
  const text = span.text ?? "";
  if (!text) {
    return { start: span.start, end: span.end };
  }

  let left = 0;
  const totalCodePoints = getCodePointLength(text);
  let right = totalCodePoints;
  while (
    left < right &&
    isBoundaryIgnorable(sliceByCodePointOffsets(text, left, left + 1))
  ) {
    left += 1;
  }
  while (
    right > left &&
    isBoundaryIgnorable(sliceByCodePointOffsets(text, right - 1, right))
  ) {
    right -= 1;
  }

  const start = span.start + left;
  const end = span.end - (totalCodePoints - right);
  if (start >= end) {
    return { start: span.start, end: span.end };
  }
  return { start, end };
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
  if (matchMode === "boundary") {
    const trimmedA = trimBoundaryOffsets(a);
    const trimmedB = trimBoundaryOffsets(b);
    return trimmedA.start === trimmedB.start && trimmedA.end === trimmedB.end;
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
