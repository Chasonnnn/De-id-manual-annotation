import { CODEBOOK } from "../codebook";
import { getLabelColor } from "../hosted/types";

export default function CodebookPanel() {
  return (
    <div className="codebook-panel">
      <p className="codebook-note">Examples are synthetic and contain no session data.</p>
      <dl className="codebook-list">
        {CODEBOOK.map((entry) => (
          <div className="codebook-entry" key={entry.label}>
            <dt>
              <span className="codebook-label" style={{ background: getLabelColor(entry.label) }}>
                {entry.label}
              </span>
            </dt>
            <dd>
              <p>{entry.definition}</p>
              <ul aria-label={`${entry.label} examples`}>
                {entry.examples.map((example) => <li key={example}>{example}</li>)}
              </ul>
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
