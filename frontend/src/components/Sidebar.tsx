import { useCallback, useEffect, useRef, useState } from "react";
import type { AnnotationSource, DocumentSummary, SessionProfile } from "../types";

interface Props {
  documents: DocumentSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onUpload: (file: File) => void;
  onDelete: (id: string) => void;
  onExportSession: (mode: "full" | "ground_truth", source: AnnotationSource) => void;
  onImportSession: (file: File) => void;
  exportSourceOptions: Array<{ value: AnnotationSource; label: string }>;
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
  exportSourceOptions,
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
  const [exportMode, setExportMode] = useState<"full" | "ground_truth">("full");
  const [exportSource, setExportSource] = useState<AnnotationSource>("manual");

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

  useEffect(() => {
    if (exportSourceOptions.length === 0) return;
    const known = exportSourceOptions.some((option) => option.value === exportSource);
    if (!known) {
      setExportSource(exportSourceOptions[0]?.value ?? "manual");
    }
  }, [exportSource, exportSourceOptions]);

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
            onClick={() => onExportSession(exportMode, exportSource)}
            disabled={exporting || importing || uploading || savingProfile}
          >
            {exporting
              ? "Exporting..."
              : exportMode === "full"
                ? "Export Full Session"
                : "Export Ground Truth JSONs"}
          </button>
          <div className="bundle-meta">
            <label htmlFor="bundle-export-mode">Export Type</label>
            <select
              id="bundle-export-mode"
              value={exportMode}
              onChange={(e) => setExportMode(e.target.value as "full" | "ground_truth")}
              disabled={exporting || importing || uploading || savingProfile}
            >
              <option value="full">Full Session Bundle</option>
              <option value="ground_truth">Ground Truth JSONs (ZIP)</option>
            </select>
            {exportMode === "ground_truth" && (
              <>
                <label htmlFor="bundle-export-source">Ground Truth Source</label>
                <select
                  id="bundle-export-source"
                  value={exportSource}
                  onChange={(e) => setExportSource(e.target.value as AnnotationSource)}
                  disabled={exporting || importing || uploading || savingProfile}
                >
                  {exportSourceOptions.map((option) => (
                    <option key={`export-source-${option.value}`} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </>
            )}
          </div>
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
