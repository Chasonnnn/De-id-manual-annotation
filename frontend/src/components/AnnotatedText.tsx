import type { CSSProperties } from "react";
import type { CanonicalSpan } from "../hosted/types";
import { getLabelColor } from "../hosted/types";

interface Props {
  text: string;
  spans: CanonicalSpan[];
  clickable?: boolean;
  onSpanClick?: (index: number, e: React.MouseEvent | React.KeyboardEvent) => void;
}

interface RenderSegment {
  start: number;
  end: number;
  activeSpanIndices: number[];
}

export default function AnnotatedText({
  text,
  spans,
  clickable = false,
  onSpanClick,
}: Props) {
  const sorted = spans.toSorted((a, b) => a.start - b.start);
  const codePoints = Array.from(text);
  const segments = buildRenderSegments(codePoints.length, sorted);

  return (
    <>
      {segments.map((segment) =>
        renderSegment(
          codePoints,
          segment,
          sorted,
          clickable,
          onSpanClick,
        ),
      )}
    </>
  );
}

function buildRenderSegments(
  totalCodePoints: number,
  spans: CanonicalSpan[],
): RenderSegment[] {
  const boundaries = new Set<number>([0, totalCodePoints]);
  for (const span of spans) {
    boundaries.add(Math.max(0, Math.min(span.start, totalCodePoints)));
    boundaries.add(Math.max(0, Math.min(span.end, totalCodePoints)));
  }
  const sortedBoundaries = Array.from(boundaries).sort((a, b) => a - b);
  const segments: RenderSegment[] = [];

  for (let i = 0; i < sortedBoundaries.length - 1; i += 1) {
    const start = sortedBoundaries[i]!;
    const end = sortedBoundaries[i + 1]!;
    if (start >= end) continue;

    const activeSpanIndices = spans.flatMap((span, index) =>
      span.start < end && span.end > start ? [index] : [],
    );
    segments.push({
      start,
      end,
      activeSpanIndices,
    });
  }

  return segments;
}

function renderSegment(
  codePoints: string[],
  segment: RenderSegment,
  sortedSpans: CanonicalSpan[],
  clickable: boolean,
  onSpanClick: Props["onSpanClick"],
): React.ReactNode {
  const { start, end, activeSpanIndices } = segment;
  const segmentText = codePoints.slice(start, end).join("");

  if (activeSpanIndices.length === 0) {
    return (
      <span
        key={`segment-${start}-${end}`}
        data-offset={start}
        data-offset-end={end}
      >
        {segmentText}
      </span>
    );
  }

  const labels = Array.from(
    new Set(activeSpanIndices.map((index) => sortedSpans[index]!.label)),
  );
  const clickableIndex = pickClickableSpanIndex(activeSpanIndices, sortedSpans);
  const color = getLabelColor(labels[0] ?? "IDENTIFYING_NUMBER");
  const isClickable = clickable && onSpanClick && clickableIndex !== null;
  const handleActivate = (e: React.MouseEvent | React.KeyboardEvent) => {
    if (clickableIndex === null || !onSpanClick) {
      return;
    }
    onSpanClick(clickableIndex, e);
  };

  if (isClickable) {
    return (
      <button
        type="button"
        key={`segment-${start}-${end}`}
        className="ann-span clickable"
        style={{ "--ann-color": color } as CSSProperties}
        data-offset={start}
        data-offset-end={end}
        onClick={handleActivate}
      >
        {segmentText}
        <span
          className="ann-span-label"
          data-annotation-label="true"
          aria-hidden="true"
        >
          {labels.join(" · ")}
        </span>
      </button>
    );
  }

  return (
    <span
      key={`segment-${start}-${end}`}
      className="ann-span"
      style={{ "--ann-color": color } as CSSProperties}
      data-offset={start}
      data-offset-end={end}
    >
      {segmentText}
      <span
        className="ann-span-label"
        data-annotation-label="true"
        aria-hidden="true"
      >
        {labels.join(" · ")}
      </span>
    </span>
  );
}

function pickClickableSpanIndex(
  indices: number[],
  spans: CanonicalSpan[],
): number | null {
  if (indices.length === 0) return null;
  return indices.reduce((best, current) => {
    const bestSpan = spans[best]!;
    const currentSpan = spans[current]!;
    const bestLength = bestSpan.end - bestSpan.start;
    const currentLength = currentSpan.end - currentSpan.start;
    if (currentLength < bestLength) return current;
    if (currentLength > bestLength) return best;
    if (currentSpan.start > bestSpan.start) return current;
    return best;
  });
}
