import { useCallback, useRef, useState } from "react";
import type { DocumentSummary, SessionProfile } from "../types";

interface Props {
  documents: DocumentSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onUpload: (file: File) => void;
  onDelete: (id: string) => void;
  onExportSession: () => void;
  onImportSession: (file: File) => void;
  sessionProfile: SessionProfile;
  onSessionProfileChange: (profile: SessionProfile) => void;
  onSaveSessionProfile: () => void;
  uploading?: boolean;
  deletingId?: string | null;
  exporting?: boolean;
  importing?: boolean;
  savingProfile?: boolean;
}

export default function Sidebar({
  documents,
  selectedId,
  onSelect,
  onUpload,
  onDelete,
  onExportSession,
  onImportSession,
  sessionProfile,
  onSessionProfileChange,
  onSaveSessionProfile,
  uploading = false,
  deletingId = null,
  exporting = false,
  importing = false,
  savingProfile = false,
}: Props) {
  const [search, setSearch] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const importRef = useRef<HTMLInputElement>(null);

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
          id="sidebar-search"
          name="sidebar_search"
          aria-label="Search documents"
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
            <span className="doc-row-main">
              <span className={`status-badge ${doc.status}`} />
              <span className="doc-filename" title={doc.filename}>
                {doc.filename}
              </span>
            </span>
            <button
              type="button"
              className="doc-delete-btn"
              title="Delete document"
              disabled={Boolean(deletingId)}
              onClick={(e) => {
                e.stopPropagation();
                onDelete(doc.id);
              }}
            >
              {deletingId === doc.id ? "..." : "Delete"}
            </button>
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
            id="sidebar-upload-file"
            name="upload_file"
            type="file"
            accept=".json,.jsonl"
            onChange={handleFileChange}
            disabled={uploading}
          />
        </div>
        <div className="sidebar-actions">
          <div className="bundle-meta">
            <label htmlFor="bundle-project-name">Project Name</label>
            <input
              id="bundle-project-name"
              type="text"
              value={sessionProfile.project_name}
              onChange={(e) =>
                onSessionProfileChange({
                  ...sessionProfile,
                  project_name: e.target.value,
                })
              }
              placeholder="HIPS Annotation Round 1"
              disabled={savingProfile}
            />
            <label htmlFor="bundle-author">Author</label>
            <input
              id="bundle-author"
              type="text"
              value={sessionProfile.author}
              onChange={(e) =>
                onSessionProfileChange({
                  ...sessionProfile,
                  author: e.target.value,
                })
              }
              placeholder="Your name"
              disabled={savingProfile}
            />
            <label htmlFor="bundle-notes">Notes</label>
            <textarea
              id="bundle-notes"
              value={sessionProfile.notes}
              onChange={(e) =>
                onSessionProfileChange({
                  ...sessionProfile,
                  notes: e.target.value,
                })
              }
              placeholder="Scope, instructions, caveats for collaborators"
              disabled={savingProfile}
            />
            <button
              type="button"
              className="sidebar-action-btn"
              onClick={onSaveSessionProfile}
              disabled={savingProfile || importing || exporting || uploading}
            >
              {savingProfile ? "Saving..." : "Save Bundle Info"}
            </button>
          </div>
          <button
            type="button"
            className="sidebar-action-btn"
            onClick={onExportSession}
            disabled={exporting || importing || uploading || savingProfile}
          >
            {exporting ? "Exporting..." : "Export Session"}
          </button>
          <button
            type="button"
            className="sidebar-action-btn"
            onClick={() => importRef.current?.click()}
            disabled={importing || exporting || uploading || savingProfile}
          >
            {importing ? "Importing..." : "Import Session"}
          </button>
          <input
            ref={importRef}
            id="sidebar-import-file"
            name="import_file"
            type="file"
            accept=".json"
            style={{ display: "none" }}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onImportSession(file);
              e.target.value = "";
            }}
            disabled={importing || exporting || uploading}
          />
        </div>
      </div>
    </aside>
  );
}
