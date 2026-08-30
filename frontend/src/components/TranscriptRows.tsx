import type { CanonicalSpan } from "../hosted/types";
import AnnotatedText from "./AnnotatedText";

interface TranscriptRow {
  start: number;
  end: number;
}

interface Props {
  text: string;
  spans: CanonicalSpan[];
  comparisonSpans?: CanonicalSpan[];
  comparisonMode?: boolean;
  clickable?: boolean;
  onSpanClick?: (index: number, event: React.MouseEvent | React.KeyboardEvent) => void;
}

function transcriptRows(text: string): TranscriptRow[] {
  const codePoints = Array.from(text);
  const rows: TranscriptRow[] = [];
  let start = 0;
  for (let index = 0; index <= codePoints.length; index += 1) {
    if (index !== codePoints.length && codePoints[index] !== "\n") continue;
    rows.push({ start, end: index });
    start = index + 1;
  }
  return rows.length > 0 ? rows : [{ start: 0, end: 0 }];
}

export default function TranscriptRows({
  text,
  spans,
  comparisonSpans = [],
  comparisonMode = false,
  clickable = false,
  onSpanClick,
}: Props) {
  return (
    <div className="transcript-rows">
      {transcriptRows(text).map((row, index) => (
        <div className="transcript-row" data-turn={index + 1} key={`${row.start}-${row.end}`}>
          <span className="turn-number" aria-hidden="true">{index + 1}</span>
          <span className="turn-text">
            {row.start === row.end ? (
              <span data-offset={row.start} data-offset-end={row.end}>{"\u00a0"}</span>
            ) : (
              <AnnotatedText
                text={text}
                spans={spans}
                comparisonSpans={comparisonSpans}
                comparisonMode={comparisonMode}
                clickable={clickable}
                onSpanClick={onSpanClick}
                startOffset={row.start}
                endOffset={row.end}
              />
            )}
          </span>
        </div>
      ))}
    </div>
  );
}
