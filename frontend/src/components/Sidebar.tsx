import { useCallback, useRef, useState } from "react";
import type { DocumentSummary } from "../types";

interface Props {
  documents: DocumentSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onUpload: (file: File) => void;
  uploading?: boolean;
}

export default function Sidebar({
  documents,
  selectedId,
  onSelect,
  onUpload,
  uploading = false,
}: Props) {
  const [search, setSearch] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const filtered = documents.filter((d) =>
    d.filename.toLowerCase().includes(search.toLowerCase()),
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (uploading) return;
      const file = e.dataTransfer.files[0];
      if (file) onUpload(file);
    },
    [onUpload, uploading],
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (uploading) return;
      const file = e.target.files?.[0];
      if (file) onUpload(file);
      e.target.value = "";
    },
    [onUpload, uploading],
  );

  return (
    <aside className="sidebar">
      <h2>Documents</h2>
      <div className="sidebar-search">
        <input
          type="text"
          placeholder="Search documents..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>
      <ul className="doc-list">
        {filtered.map((doc) => (
          <li
            key={doc.id}
            className={doc.id === selectedId ? "active" : ""}
            onClick={() => onSelect(doc.id)}
          >
            <span className={`status-badge ${doc.status}`} />
            <span className="doc-filename" title={doc.filename}>
              {doc.filename}
            </span>
          </li>
        ))}
        {filtered.length === 0 && (
          <li style={{ color: "#6c7086", cursor: "default" }}>
            No documents found
          </li>
        )}
      </ul>
      <div className="upload-area">
        <div
          className={`drop-zone ${dragOver ? "drag-over" : ""} ${uploading ? "uploading" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            if (!uploading) setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => !uploading && fileRef.current?.click()}
        >
          {uploading ? "Uploading..." : "Drop file here or click to upload"}
          <input
            ref={fileRef}
            type="file"
            accept=".json,.jsonl"
            onChange={handleFileChange}
            disabled={uploading}
          />
        </div>
      </div>
    </aside>
  );
}
