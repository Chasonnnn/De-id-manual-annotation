import type { AgentChunkDiagnostic } from "../types";

interface Props {
  diagnostics: AgentChunkDiagnostic[];
}

function formatChunkRange(diagnostic: AgentChunkDiagnostic): string {
  return `[${diagnostic.start}, ${diagnostic.end})`;
}

function formatChunkNotes(diagnostic: AgentChunkDiagnostic): string {
  const notes: string[] = [];
  if (diagnostic.suspicious_empty) notes.push("suspicious empty");
  if (diagnostic.finish_reason) notes.push(`finish_reason=${diagnostic.finish_reason}`);
  if (diagnostic.warnings.length > 0) notes.push(diagnostic.warnings.join(" | "));
  return notes.join(" | ");
}

export default function ChunkDiagnosticsPanel({ diagnostics }: Props) {
  if (diagnostics.length === 0) return null;

  const emptyCount = diagnostics.filter((item) => item.span_count === 0).length;
  const retriedCount = diagnostics.filter((item) => item.attempt_count > 1).length;
  const failedCount = diagnostics.filter((item) => item.status === "failed").length;

  return (
    <details className="chunk-diagnostics">
      <summary>
        Chunk Diagnostics: {diagnostics.length} chunk(s) · empty {emptyCount} · retried{" "}
        {retriedCount} · failed {failedCount}
      </summary>
      <div className="chunk-diagnostics-table-wrap">
        <table className="per-label-table chunk-diagnostics-table">
          <thead>
            <tr>
              <th>Chunk</th>
              <th>Range</th>
              <th>Chars</th>
              <th>Spans</th>
              <th>Attempts</th>
              <th>Status</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {diagnostics.map((diagnostic) => (
              <tr key={`chunk-diagnostic-${diagnostic.chunk_index}`}>
                <td>{diagnostic.chunk_index + 1}</td>
                <td>{formatChunkRange(diagnostic)}</td>
                <td>{diagnostic.char_count}</td>
                <td>{diagnostic.span_count}</td>
                <td>{diagnostic.attempt_count}</td>
                <td>{diagnostic.status}</td>
                <td>{formatChunkNotes(diagnostic) || "ok"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
