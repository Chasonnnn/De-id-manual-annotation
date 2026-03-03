import type { CanonicalSpan } from "../types";
import { getLabelColor } from "../types";

export interface DiffSpanInfo {
  start: number;
  end: number;
  type: "added" | "removed";
}

interface Props {
  text: string;
  spans: CanonicalSpan[];
  clickable?: boolean;
  onSpanClick?: (index: number, e: React.MouseEvent) => void;
  diffSpans?: DiffSpanInfo[];
}

export default function AnnotatedText({
  text,
  spans,
  clickable = false,
  onSpanClick,
  diffSpans = [],
}: Props) {
  const sorted = [...spans].sort((a, b) => a.start - b.start);

  const segments: React.ReactNode[] = [];
  let pos = 0;

  for (let i = 0; i < sorted.length; i++) {
    const span = sorted[i]!;
    if (span.start > pos) {
      segments.push(
        renderText(text.slice(pos, span.start), pos, diffSpans, `text-${pos}`),
      );
    }
    const spanText = text.slice(span.start, span.end);
    const color = getLabelColor(span.label);
    const diffClass = getDiffClass(span.start, span.end, diffSpans);
    segments.push(
      <span
        key={`span-${i}`}
        className={`ann-span ${clickable ? "clickable" : ""} ${diffClass}`}
        style={{ background: color }}
        data-offset={span.start}
        data-offset-end={span.end}
        onClick={
          clickable && onSpanClick
            ? (e) => onSpanClick(i, e)
            : undefined
        }
      >
        {spanText}
        <sup className="ann-span-label">{span.label}</sup>
      </span>,
    );
    pos = span.end;
  }

  if (pos < text.length) {
    segments.push(
      renderText(text.slice(pos), pos, diffSpans, `text-${pos}`),
    );
  }

  return <>{segments}</>;
}

/**
 * Render a plain text chunk, splitting it to insert diff highlight spans
 * for "removed" regions that fall within this text range.
 */
function renderText(
  chunk: string,
  offset: number,
  diffSpans: DiffSpanInfo[],
  keyPrefix: string,
): React.ReactNode {
  // Find diff spans that overlap with this text region [offset, offset+chunk.length)
  const chunkEnd = offset + chunk.length;
  const relevant = diffSpans.filter(
    (d) => d.start < chunkEnd && d.end > offset,
  );

  if (relevant.length === 0) {
    // No diff spans overlap -- return a simple span with data-offset
    return (
      <span key={keyPrefix} data-offset={offset} data-offset-end={chunkEnd}>
        {chunk}
      </span>
    );
  }

  // Split the chunk into segments: plain text + diff-highlighted regions
  const parts: React.ReactNode[] = [];
  let pos = offset;

  for (const d of relevant.sort((a, b) => a.start - b.start)) {
    const dStart = Math.max(d.start, offset);
    const dEnd = Math.min(d.end, chunkEnd);

    // Plain text before this diff span
    if (dStart > pos) {
      parts.push(
        <span
          key={`${keyPrefix}-${pos}`}
          data-offset={pos}
          data-offset-end={dStart}
        >
          {chunk.slice(pos - offset, dStart - offset)}
        </span>,
      );
    }

    // Diff-highlighted region
    const diffClass = d.type === "added" ? "diff-added" : "diff-removed";
    parts.push(
      <span
        key={`${keyPrefix}-diff-${dStart}`}
        className={diffClass}
        data-offset={dStart}
        data-offset-end={dEnd}
      >
        {chunk.slice(dStart - offset, dEnd - offset)}
      </span>,
    );
    pos = dEnd;
  }

  // Trailing plain text after last diff span
  if (pos < chunkEnd) {
    parts.push(
      <span
        key={`${keyPrefix}-${pos}`}
        data-offset={pos}
        data-offset-end={chunkEnd}
      >
        {chunk.slice(pos - offset)}
      </span>,
    );
  }

  return parts;
}

function getDiffClass(
  start: number,
  end: number,
  diffSpans: DiffSpanInfo[],
): string {
  for (const d of diffSpans) {
    if (start < d.end && end > d.start) {
      return d.type === "added" ? "diff-added" : "diff-removed";
    }
  }
  return "";
}
