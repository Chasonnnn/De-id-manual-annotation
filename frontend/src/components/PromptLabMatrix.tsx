import type { PromptLabMatrixCellSummary, PromptLabRunDetail } from "../types";

interface Props {
  run: PromptLabRunDetail;
  selectedCellId: string | null;
  onSelectCell: (cellId: string) => void;
}

function fmtPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export default function PromptLabMatrix({ run, selectedCellId, onSelectCell }: Props) {
  const cellsById = new Map<string, PromptLabMatrixCellSummary>();
  for (const cell of run.matrix.cells) {
    cellsById.set(cell.id, cell);
  }

  return (
    <section className="prompt-lab-matrix">
      <div className="prompt-lab-matrix-header">
        <h3>Matrix Results</h3>
        <div className="prompt-lab-matrix-meta">
          Status: <strong>{run.status}</strong> · Progress: {run.progress.completed_tasks}/
          {run.progress.total_tasks}
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Model \ Prompt</th>
            {run.matrix.prompts.map((prompt) => (
              <th key={prompt.id}>{prompt.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {run.matrix.models.map((model) => (
            <tr key={model.id}>
              <th>{model.label}</th>
              {run.matrix.prompts.map((prompt) => {
                const cellId = `${model.id}__${prompt.id}`;
                const cell = cellsById.get(cellId);
                const active = selectedCellId === cellId;
                return (
                  <td key={cellId}>
                    <button
                      type="button"
                      className={`prompt-lab-matrix-cell ${active ? "active" : ""}`}
                      onClick={() => onSelectCell(cellId)}
                    >
                      <div className="prompt-lab-cell-status">{cell?.status ?? "pending"}</div>
                      <div className="prompt-lab-cell-f1">F1 {fmtPct(cell?.micro.f1 ?? 0)}</div>
                      <div className="prompt-lab-cell-meta">
                        Docs {cell?.completed_docs ?? 0}/{cell?.total_docs ?? run.doc_count}
                      </div>
                      <div className="prompt-lab-cell-meta">
                        Errors {cell?.error_count ?? 0} · Conf {cell?.mean_confidence != null
                          ? fmtPct(cell.mean_confidence)
                          : "N/A"}
                      </div>
                    </button>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
