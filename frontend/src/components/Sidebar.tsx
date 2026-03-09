import { useCallback, useEffect, useMemo, useReducer, useRef } from "react";
import type {
  AnnotationSource,
  DocumentSummary,
  FolderDetail,
  FolderSummary,
  SessionProfile,
} from "../types";

interface Props {
  documents: DocumentSummary[];
  folders: FolderSummary[];
  folderDetailsById: Record<string, FolderDetail>;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onUpload: (file: File) => void;
  onDelete: (id: string) => void;
  onCreateFolderSample: (folderId: string, count: number) => void;
  onDeleteFolder: (folderId: string) => void;
  onExportSession: (mode: "full" | "ground_truth", source: AnnotationSource) => void;
  onImportSession: (files: File[]) => void;
  exportSourceOptions: Array<{ value: AnnotationSource; label: string }>;
  sessionProfile: SessionProfile;
  onSessionProfileChange: (profile: SessionProfile) => void;
  onSaveSessionProfile: () => void;
  uploading?: boolean;
  deletingId?: string | null;
  folderBusyId?: string | null;
  exporting?: boolean;
  importing?: boolean;
  savingProfile?: boolean;
}

type ExportMode = "full" | "ground_truth";

interface SidebarState {
  search: string;
  dragOver: boolean;
  importDragOver: boolean;
  expandedFolders: Record<string, boolean>;
  exportMode: ExportMode;
  exportSource: AnnotationSource;
}

type SidebarAction =
  | { type: "set_search"; value: string }
  | { type: "set_drag_over"; value: boolean }
  | { type: "set_import_drag_over"; value: boolean }
  | { type: "toggle_folder"; folderId: string }
  | { type: "sync_folders"; folders: FolderSummary[] }
  | { type: "set_export_mode"; value: ExportMode }
  | { type: "set_export_source"; value: AnnotationSource };

function createInitialSidebarState(
  exportSourceOptions: Props["exportSourceOptions"],
): SidebarState {
  return {
    search: "",
    dragOver: false,
    importDragOver: false,
    expandedFolders: {},
    exportMode: "full",
    exportSource: exportSourceOptions[0]?.value ?? "manual",
  };
}

function sidebarReducer(state: SidebarState, action: SidebarAction): SidebarState {
  switch (action.type) {
    case "set_search":
      return state.search === action.value ? state : { ...state, search: action.value };
    case "set_drag_over":
      return state.dragOver === action.value ? state : { ...state, dragOver: action.value };
    case "set_import_drag_over":
      return state.importDragOver === action.value
        ? state
        : { ...state, importDragOver: action.value };
    case "toggle_folder":
      return {
        ...state,
        expandedFolders: {
          ...state.expandedFolders,
          [action.folderId]: !state.expandedFolders[action.folderId],
        },
      };
    case "sync_folders": {
      if (action.folders.length === 0) {
        return Object.keys(state.expandedFolders).length === 0
          ? state
          : { ...state, expandedFolders: {} };
      }
      const nextExpanded = { ...state.expandedFolders };
      let changed = false;
      const activeFolderIds = new Set(action.folders.map((folder) => folder.id));

      for (const folderId of Object.keys(nextExpanded)) {
        if (!activeFolderIds.has(folderId)) {
          delete nextExpanded[folderId];
          changed = true;
        }
      }
      for (const folder of action.folders) {
        if (folder.parent_folder_id === null && nextExpanded[folder.id] === undefined) {
          nextExpanded[folder.id] = true;
          changed = true;
        }
      }
      return changed ? { ...state, expandedFolders: nextExpanded } : state;
    }
    case "set_export_mode":
      return state.exportMode === action.value ? state : { ...state, exportMode: action.value };
    case "set_export_source":
      return state.exportSource === action.value
        ? state
        : { ...state, exportSource: action.value };
    default:
      return state;
  }
}

function folderMatchesSearch(
  folderId: string,
  search: string,
  folderDetailsById: Record<string, FolderDetail>,
): boolean {
  if (!search) return true;
  const detail = folderDetailsById[folderId];
  if (!detail) return false;
  const term = search.toLowerCase();
  if (
    detail.name.toLowerCase().includes(term) ||
    detail.kind.toLowerCase().includes(term) ||
    (detail.source_filename ?? "").toLowerCase().includes(term)
  ) {
    return true;
  }
  if (
    detail.documents.some(
      (doc) =>
        doc.display_name.toLowerCase().includes(term) ||
        doc.filename.toLowerCase().includes(term),
    )
  ) {
    return true;
  }
  return detail.child_folder_ids.some((childId) =>
    folderMatchesSearch(childId, search, folderDetailsById),
  );
}

function folderDepth(folder: FolderSummary, folderById: Map<string, FolderSummary>): number {
  let depth = 0;
  let current = folder;
  while (current.parent_folder_id) {
    const parent = folderById.get(current.parent_folder_id);
    if (!parent) break;
    depth += 1;
    current = parent;
  }
  return depth;
}

function handleDropZoneKeyDown(
  event: React.KeyboardEvent<HTMLDivElement>,
  openPicker: () => void,
  disabled: boolean,
) {
  if (disabled) return;
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openPicker();
  }
}

function SidebarDocumentRow({
  document,
  selectedId,
  deletingId,
  onSelect,
  onDelete,
}: {
  document: DocumentSummary;
  selectedId: string | null;
  deletingId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <li
      className={document.id === selectedId ? "active" : ""}
      role="button"
      tabIndex={0}
      onClick={() => onSelect(document.id)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect(document.id);
        }
      }}
    >
      <span className="doc-row-main">
        <span className={`status-badge ${document.status}`} />
        <span className="doc-filename" title={document.display_name}>
          {document.display_name}
        </span>
      </span>
      <button
        type="button"
        className="doc-delete-btn"
        title="Delete document"
        disabled={Boolean(deletingId)}
        onClick={(e) => {
          e.stopPropagation();
          onDelete(document.id);
        }}
      >
        {deletingId === document.id ? "..." : "Delete"}
      </button>
    </li>
  );
}

function SidebarFolderBranch({
  folder,
  detail,
  depth,
  folderById,
  folderDetailsById,
  expandedFolders,
  folderBusyId,
  selectedId,
  onToggleFolder,
  onSelect,
  onRequestSample,
  onDeleteFolder,
}: {
  folder: FolderSummary;
  detail: FolderDetail;
  depth: number;
  folderById: Map<string, FolderSummary>;
  folderDetailsById: Record<string, FolderDetail>;
  expandedFolders: Record<string, boolean>;
  folderBusyId: string | null;
  selectedId: string | null;
  onToggleFolder: (folderId: string) => void;
  onSelect: (id: string) => void;
  onRequestSample: (folder: FolderSummary) => void;
  onDeleteFolder: (folderId: string) => void;
}) {
  const isExpanded = expandedFolders[folder.id] ?? false;
  const busy = folderBusyId === folder.id;
  const paddingLeft = 16 + depth * 14;

  return (
    <>
      <li
        className={`sidebar-folder-row ${isExpanded ? "expanded" : ""}`}
        style={{ paddingLeft }}
      >
        <button
          type="button"
          className="sidebar-folder-toggle"
          onClick={() => onToggleFolder(folder.id)}
          aria-label={isExpanded ? "Collapse folder" : "Expand folder"}
        >
          {isExpanded ? "▾" : "▸"}
        </button>
        <span className="sidebar-folder-name" title={folder.name}>
          {folder.name}
        </span>
        <span className="sidebar-folder-meta">
          {folder.doc_count} docs
          {folder.kind === "sample" && folder.sample_size ? ` • sample ${folder.sample_size}` : ""}
        </span>
        <div className="sidebar-folder-actions">
          <button
            type="button"
            className="doc-delete-btn"
            onClick={() => onRequestSample(folder)}
            disabled={busy || folder.doc_count === 0}
          >
            {busy ? "..." : "Sample"}
          </button>
          <button
            type="button"
            className="doc-delete-btn"
            onClick={() => onDeleteFolder(folder.id)}
            disabled={busy}
          >
            {busy ? "..." : "Delete"}
          </button>
        </div>
      </li>
      {isExpanded &&
        detail.documents.map((document) => (
          <li
            key={`folder-doc-${document.id}`}
            className={
              document.id === selectedId
                ? "active sidebar-child-doc"
                : "sidebar-child-doc"
            }
            role="button"
            tabIndex={0}
            style={{ paddingLeft: paddingLeft + 26 }}
            onClick={() => onSelect(document.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect(document.id);
              }
            }}
          >
            <span className="doc-row-main">
              <span className={`status-badge ${document.status}`} />
              <span className="doc-filename" title={document.display_name}>
                {document.display_name}
              </span>
            </span>
          </li>
        ))}
      {isExpanded &&
        detail.child_folder_ids.map((childFolderId) => {
          const childFolder = folderById.get(childFolderId);
          const childDetail = folderDetailsById[childFolderId];
          if (!childFolder || !childDetail) return null;
          return (
            <SidebarFolderBranch
              key={`folder-${childFolderId}`}
              folder={childFolder}
              detail={childDetail}
              depth={depth + 1}
              folderById={folderById}
              folderDetailsById={folderDetailsById}
              expandedFolders={expandedFolders}
              folderBusyId={folderBusyId}
              selectedId={selectedId}
              onToggleFolder={onToggleFolder}
              onSelect={onSelect}
              onRequestSample={onRequestSample}
              onDeleteFolder={onDeleteFolder}
            />
          );
        })}
    </>
  );
}

function SidebarDocumentTree({
  filteredDocuments,
  visibleTopLevelFolders,
  folderById,
  folderDetailsById,
  expandedFolders,
  selectedId,
  deletingId,
  folderBusyId,
  onSelect,
  onDelete,
  onToggleFolder,
  onRequestSample,
  onDeleteFolder,
}: {
  filteredDocuments: DocumentSummary[];
  visibleTopLevelFolders: FolderSummary[];
  folderById: Map<string, FolderSummary>;
  folderDetailsById: Record<string, FolderDetail>;
  expandedFolders: Record<string, boolean>;
  selectedId: string | null;
  deletingId: string | null;
  folderBusyId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onToggleFolder: (folderId: string) => void;
  onRequestSample: (folder: FolderSummary) => void;
  onDeleteFolder: (folderId: string) => void;
}) {
  return (
    <ul className="doc-list">
      {filteredDocuments.length > 0 && (
        <li className="sidebar-group-label">Top-Level Documents</li>
      )}
      {filteredDocuments.map((document) => (
        <SidebarDocumentRow
          key={document.id}
          document={document}
          selectedId={selectedId}
          deletingId={deletingId}
          onSelect={onSelect}
          onDelete={onDelete}
        />
      ))}

      {visibleTopLevelFolders.length > 0 && <li className="sidebar-group-label">Folders</li>}
      {visibleTopLevelFolders.map((folder) => {
        const detail = folderDetailsById[folder.id];
        if (!detail) return null;
        return (
          <SidebarFolderBranch
            key={`folder-${folder.id}`}
            folder={folder}
            detail={detail}
            depth={folderDepth(folder, folderById)}
            folderById={folderById}
            folderDetailsById={folderDetailsById}
            expandedFolders={expandedFolders}
            folderBusyId={folderBusyId}
            selectedId={selectedId}
            onToggleFolder={onToggleFolder}
            onSelect={onSelect}
            onRequestSample={onRequestSample}
            onDeleteFolder={onDeleteFolder}
          />
        );
      })}

      {filteredDocuments.length === 0 && visibleTopLevelFolders.length === 0 && (
        <li className="sidebar-empty-row">No documents or folders found</li>
      )}
    </ul>
  );
}

function SidebarDropZone({
  className,
  active,
  disabled,
  label,
  inputRef,
  inputId,
  inputName,
  accept,
  multiple = false,
  onOpenPicker,
  onDragOver,
  onDragLeave,
  onDrop,
  onChange,
}: {
  className: string;
  active: boolean;
  disabled: boolean;
  label: string;
  inputRef: React.RefObject<HTMLInputElement | null>;
  inputId: string;
  inputName: string;
  accept: string;
  multiple?: boolean;
  onOpenPicker: () => void;
  onDragOver: (e: React.DragEvent<HTMLDivElement>) => void;
  onDragLeave: () => void;
  onDrop: (e: React.DragEvent<HTMLDivElement>) => void;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div
      className={`${className} ${active ? "drag-over" : ""} ${disabled ? "uploading" : ""}`}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={() => {
        if (!disabled) onOpenPicker();
      }}
      onKeyDown={(e) => handleDropZoneKeyDown(e, onOpenPicker, disabled)}
      role="button"
      tabIndex={disabled ? -1 : 0}
    >
      {label}
      <input
        ref={inputRef}
        id={inputId}
        name={inputName}
        type="file"
        multiple={multiple}
        accept={accept}
        onChange={onChange}
        disabled={disabled}
      />
    </div>
  );
}

function SidebarActionsPanel({
  sessionProfile,
  onSessionProfileChange,
  onSaveSessionProfile,
  savingProfile,
  uploading,
  exporting,
  importing,
  exportMode,
  exportSource,
  exportSourceOptions,
  onExportModeChange,
  onExportSourceChange,
  onExportSession,
  dragOver,
  importDragOver,
  fileRef,
  importRef,
  onUploadOpenPicker,
  onImportOpenPicker,
  onUploadDragOver,
  onUploadDragLeave,
  onUploadDrop,
  onUploadChange,
  onImportDragOver,
  onImportDragLeave,
  onImportDrop,
  onImportChange,
}: {
  sessionProfile: SessionProfile;
  onSessionProfileChange: (profile: SessionProfile) => void;
  onSaveSessionProfile: () => void;
  savingProfile: boolean;
  uploading: boolean;
  exporting: boolean;
  importing: boolean;
  exportMode: ExportMode;
  exportSource: AnnotationSource;
  exportSourceOptions: Props["exportSourceOptions"];
  onExportModeChange: (value: ExportMode) => void;
  onExportSourceChange: (value: AnnotationSource) => void;
  onExportSession: (mode: ExportMode, source: AnnotationSource) => void;
  dragOver: boolean;
  importDragOver: boolean;
  fileRef: React.RefObject<HTMLInputElement | null>;
  importRef: React.RefObject<HTMLInputElement | null>;
  onUploadOpenPicker: () => void;
  onImportOpenPicker: () => void;
  onUploadDragOver: (e: React.DragEvent<HTMLDivElement>) => void;
  onUploadDragLeave: () => void;
  onUploadDrop: (e: React.DragEvent<HTMLDivElement>) => void;
  onUploadChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onImportDragOver: (e: React.DragEvent<HTMLDivElement>) => void;
  onImportDragLeave: () => void;
  onImportDrop: (e: React.DragEvent<HTMLDivElement>) => void;
  onImportChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className="upload-area">
      <SidebarDropZone
        className="drop-zone"
        active={dragOver}
        disabled={uploading}
        label={uploading ? "Uploading..." : "Drop file here or click to upload"}
        inputRef={fileRef}
        inputId="sidebar-upload-file"
        inputName="upload_file"
        accept=".json,.jsonl"
        onOpenPicker={onUploadOpenPicker}
        onDragOver={onUploadDragOver}
        onDragLeave={onUploadDragLeave}
        onDrop={onUploadDrop}
        onChange={onUploadChange}
      />

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
            onChange={(e) => onExportModeChange(e.target.value as ExportMode)}
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
                onChange={(e) => onExportSourceChange(e.target.value as AnnotationSource)}
                disabled={exporting || importing || uploading || savingProfile}
              >
                {exportSourceOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </>
          )}
        </div>

        <SidebarDropZone
          className="drop-zone import-drop-zone"
          active={importDragOver}
          disabled={importing || exporting || uploading}
          label={
            importing
              ? "Importing..."
              : "Drop session bundle/ground truth here or click to import"
          }
          inputRef={importRef}
          inputId="sidebar-import-file"
          inputName="import_file"
          accept=".json,.zip"
          multiple
          onOpenPicker={onImportOpenPicker}
          onDragOver={onImportDragOver}
          onDragLeave={onImportDragLeave}
          onDrop={onImportDrop}
          onChange={onImportChange}
        />
      </div>
    </div>
  );
}

export default function Sidebar({
  documents,
  folders,
  folderDetailsById,
  selectedId,
  onSelect,
  onUpload,
  onDelete,
  onCreateFolderSample,
  onDeleteFolder,
  onExportSession,
  onImportSession,
  exportSourceOptions,
  sessionProfile,
  onSessionProfileChange,
  onSaveSessionProfile,
  uploading = false,
  deletingId = null,
  folderBusyId = null,
  exporting = false,
  importing = false,
  savingProfile = false,
}: Props) {
  const [state, dispatch] = useReducer(
    sidebarReducer,
    exportSourceOptions,
    createInitialSidebarState,
  );
  const fileRef = useRef<HTMLInputElement>(null);
  const importRef = useRef<HTMLInputElement>(null);

  const folderById = useMemo(
    () => new Map(folders.map((folder) => [folder.id, folder])),
    [folders],
  );

  useEffect(() => {
    dispatch({ type: "sync_folders", folders });
  }, [folders]);

  useEffect(() => {
    if (exportSourceOptions.length === 0) return;
    const known = exportSourceOptions.some((option) => option.value === state.exportSource);
    if (!known) {
      dispatch({
        type: "set_export_source",
        value: exportSourceOptions[0]?.value ?? "manual",
      });
    }
  }, [exportSourceOptions, state.exportSource]);

  const filteredDocuments = useMemo(
    () =>
      documents.filter((doc) => {
        const term = state.search.toLowerCase();
        return (
          term.length === 0 ||
          doc.display_name.toLowerCase().includes(term) ||
          doc.filename.toLowerCase().includes(term)
        );
      }),
    [documents, state.search],
  );

  const visibleTopLevelFolders = useMemo(
    () =>
      folders.filter(
        (folder) =>
          folder.parent_folder_id === null &&
          folderMatchesSearch(folder.id, state.search, folderDetailsById),
      ),
    [folderDetailsById, folders, state.search],
  );

  const openUploadPicker = useCallback(() => {
    fileRef.current?.click();
  }, []);

  const openImportPicker = useCallback(() => {
    importRef.current?.click();
  }, []);

  const handleUploadDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      dispatch({ type: "set_drag_over", value: false });
      if (uploading) return;
      const file = e.dataTransfer.files[0];
      if (file) onUpload(file);
    },
    [onUpload, uploading],
  );

  const handleImportDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      dispatch({ type: "set_import_drag_over", value: false });
      if (importing || exporting || uploading) return;
      const files = Array.from(e.dataTransfer.files ?? []);
      if (files.length > 0) onImportSession(files);
    },
    [exporting, importing, onImportSession, uploading],
  );

  const handleUploadChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (uploading) return;
      const file = e.target.files?.[0];
      if (file) onUpload(file);
      e.target.value = "";
    },
    [onUpload, uploading],
  );

  const handleImportChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files ?? []);
      if (files.length > 0) onImportSession(files);
      e.target.value = "";
    },
    [onImportSession],
  );

  const handleRequestSample = useCallback(
    (folder: FolderSummary) => {
      const suggested = String(Math.min(Math.max(folder.doc_count, 1), 25));
      const rawValue = window.prompt(
        "How many direct transcripts should this saved sample include?",
        suggested,
      );
      if (rawValue === null) return;
      const count = Number.parseInt(rawValue, 10);
      if (!Number.isFinite(count) || count < 1) {
        window.alert("Enter a positive integer sample size.");
        return;
      }
      onCreateFolderSample(folder.id, count);
    },
    [onCreateFolderSample],
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
          placeholder="Search documents and folders..."
          value={state.search}
          onChange={(e) => dispatch({ type: "set_search", value: e.target.value })}
        />
      </div>

      <SidebarDocumentTree
        filteredDocuments={filteredDocuments}
        visibleTopLevelFolders={visibleTopLevelFolders}
        folderById={folderById}
        folderDetailsById={folderDetailsById}
        expandedFolders={state.expandedFolders}
        selectedId={selectedId}
        deletingId={deletingId}
        folderBusyId={folderBusyId}
        onSelect={onSelect}
        onDelete={onDelete}
        onToggleFolder={(folderId) => dispatch({ type: "toggle_folder", folderId })}
        onRequestSample={handleRequestSample}
        onDeleteFolder={onDeleteFolder}
      />

      <SidebarActionsPanel
        sessionProfile={sessionProfile}
        onSessionProfileChange={onSessionProfileChange}
        onSaveSessionProfile={onSaveSessionProfile}
        savingProfile={savingProfile}
        uploading={uploading}
        exporting={exporting}
        importing={importing}
        exportMode={state.exportMode}
        exportSource={state.exportSource}
        exportSourceOptions={exportSourceOptions}
        onExportModeChange={(value) => dispatch({ type: "set_export_mode", value })}
        onExportSourceChange={(value) => dispatch({ type: "set_export_source", value })}
        onExportSession={onExportSession}
        dragOver={state.dragOver}
        importDragOver={state.importDragOver}
        fileRef={fileRef}
        importRef={importRef}
        onUploadOpenPicker={openUploadPicker}
        onImportOpenPicker={openImportPicker}
        onUploadDragOver={(e) => {
          e.preventDefault();
          if (!uploading) {
            dispatch({ type: "set_drag_over", value: true });
          }
        }}
        onUploadDragLeave={() => dispatch({ type: "set_drag_over", value: false })}
        onUploadDrop={handleUploadDrop}
        onUploadChange={handleUploadChange}
        onImportDragOver={(e) => {
          e.preventDefault();
          if (!importing && !exporting && !uploading) {
            dispatch({ type: "set_import_drag_over", value: true });
          }
        }}
        onImportDragLeave={() => dispatch({ type: "set_import_drag_over", value: false })}
        onImportDrop={handleImportDrop}
        onImportChange={handleImportChange}
      />
    </aside>
  );
}
