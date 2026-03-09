import { useCallback, useEffect, useMemo, useReducer, useRef } from "react";
import type {
  AnnotationSource,
  DocumentSummary,
  FolderDetail,
  FolderSummary,
  ImportConflictPolicy,
} from "../types";

interface Props {
  documents: DocumentSummary[];
  folders: FolderSummary[];
  folderDetailsById: Record<string, FolderDetail>;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onIngestFiles: (files: File[], conflictPolicy: ImportConflictPolicy) => void;
  onDelete: (id: string) => void;
  onCreateFolder: (name: string, parentFolderId: string | null) => void;
  onCreateFolderSample: (folderId: string, count: number) => void;
  onDeleteFolder: (folderId: string) => void;
  onPruneFolder: (folderId: string) => void;
  onExportSession: (mode: "full" | "ground_truth", source: AnnotationSource) => void;
  exportSourceOptions: Array<{ value: AnnotationSource; label: string }>;
  ingesting?: boolean;
  deletingId?: string | null;
  folderBusyId?: string | null;
  exporting?: boolean;
}

type ExportMode = "full" | "ground_truth";

interface SidebarState {
  search: string;
  ingestDragOver: boolean;
  expandedFolders: Record<string, boolean>;
  exportMode: ExportMode;
  exportSource: AnnotationSource;
  importConflictPolicy: ImportConflictPolicy;
}

type SidebarAction =
  | { type: "set_search"; value: string }
  | { type: "set_ingest_drag_over"; value: boolean }
  | { type: "toggle_folder"; folderId: string }
  | { type: "sync_folders"; folders: FolderSummary[] }
  | { type: "set_export_mode"; value: ExportMode }
  | { type: "set_export_source"; value: AnnotationSource }
  | { type: "set_import_conflict_policy"; value: ImportConflictPolicy };

function createInitialSidebarState(
  exportSourceOptions: Props["exportSourceOptions"],
): SidebarState {
  return {
    search: "",
    ingestDragOver: false,
    expandedFolders: {},
    exportMode: "full",
    exportSource: exportSourceOptions[0]?.value ?? "manual",
    importConflictPolicy: "replace",
  };
}

function sidebarReducer(state: SidebarState, action: SidebarAction): SidebarState {
  switch (action.type) {
    case "set_search":
      return state.search === action.value ? state : { ...state, search: action.value };
    case "set_ingest_drag_over":
      return state.ingestDragOver === action.value
        ? state
        : { ...state, ingestDragOver: action.value };
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
      return changed ? { ...state, expandedFolders: nextExpanded } : state;
    }
    case "set_export_mode":
      return state.exportMode === action.value ? state : { ...state, exportMode: action.value };
    case "set_export_source":
      return state.exportSource === action.value
        ? state
        : { ...state, exportSource: action.value };
    case "set_import_conflict_policy":
      return state.importConflictPolicy === action.value
        ? state
        : { ...state, importConflictPolicy: action.value };
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
  onCreateFolder,
  onRequestSample,
  onDeleteFolder,
  onPruneFolder,
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
  onCreateFolder: (name: string, parentFolderId: string | null) => void;
  onRequestSample: (folder: FolderSummary) => void;
  onDeleteFolder: (folderId: string) => void;
  onPruneFolder: (folderId: string) => void;
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
            onClick={() => {
              const rawValue = window.prompt("New subfolder name", `${folder.name} subfolder`);
              const name = rawValue?.trim() ?? "";
              if (!name) return;
              onCreateFolder(name, folder.id);
            }}
            disabled={Boolean(folderBusyId)}
          >
            New
          </button>
          <button
            type="button"
            className="doc-delete-btn"
            onClick={() => onRequestSample(folder)}
            disabled={Boolean(folderBusyId) || folder.doc_count === 0}
          >
            {busy ? "..." : "Sample"}
          </button>
          <button
            type="button"
            className="doc-delete-btn"
            onClick={() => onPruneFolder(folder.id)}
            disabled={Boolean(folderBusyId) || folder.doc_count === 0}
          >
            {busy ? "..." : "Prune Empty"}
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
              onCreateFolder={onCreateFolder}
              onRequestSample={onRequestSample}
              onDeleteFolder={onDeleteFolder}
              onPruneFolder={onPruneFolder}
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
  onPruneFolder,
  onCreateFolder,
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
  onPruneFolder: (folderId: string) => void;
  onCreateFolder: (name: string, parentFolderId: string | null) => void;
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
            onCreateFolder={onCreateFolder}
            onRequestSample={onRequestSample}
            onDeleteFolder={onDeleteFolder}
            onPruneFolder={onPruneFolder}
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
  ingesting,
  exporting,
  exportMode,
  exportSource,
  exportSourceOptions,
  importConflictPolicy,
  onImportConflictPolicyChange,
  onExportModeChange,
  onExportSourceChange,
  onExportSession,
  onCreateFolder,
  ingestDragOver,
  ingestRef,
  onIngestOpenPicker,
  onIngestDragOver,
  onIngestDragLeave,
  onIngestDrop,
  onIngestChange,
}: {
  ingesting: boolean;
  exporting: boolean;
  exportMode: ExportMode;
  exportSource: AnnotationSource;
  exportSourceOptions: Props["exportSourceOptions"];
  importConflictPolicy: ImportConflictPolicy;
  onImportConflictPolicyChange: (value: ImportConflictPolicy) => void;
  onExportModeChange: (value: ExportMode) => void;
  onExportSourceChange: (value: AnnotationSource) => void;
  onExportSession: (mode: ExportMode, source: AnnotationSource) => void;
  onCreateFolder: (name: string, parentFolderId: string | null) => void;
  ingestDragOver: boolean;
  ingestRef: React.RefObject<HTMLInputElement | null>;
  onIngestOpenPicker: () => void;
  onIngestDragOver: (e: React.DragEvent<HTMLDivElement>) => void;
  onIngestDragLeave: () => void;
  onIngestDrop: (e: React.DragEvent<HTMLDivElement>) => void;
  onIngestChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className="upload-area">
      <SidebarDropZone
        className="drop-zone"
        active={ingestDragOver}
        disabled={ingesting || exporting}
        label={
          ingesting
            ? "Processing..."
            : "Drop transcript/session bundle/ground truth here or click to add"
        }
        inputRef={ingestRef}
        inputId="sidebar-ingest-file"
        inputName="ingest_file"
        accept=".json,.jsonl,.zip"
        multiple
        onOpenPicker={onIngestOpenPicker}
        onDragOver={onIngestDragOver}
        onDragLeave={onIngestDragLeave}
        onDrop={onIngestDrop}
        onChange={onIngestChange}
      />

      <div className="sidebar-actions">
        <div className="bundle-meta">
          <label htmlFor="bundle-import-conflict">Import Conflicts</label>
          <select
            id="bundle-import-conflict"
            value={importConflictPolicy}
            onChange={(e) => onImportConflictPolicyChange(e.target.value as ImportConflictPolicy)}
            disabled={exporting || ingesting}
          >
            <option value="replace">Replace Current</option>
            <option value="add_new">Add as New</option>
            <option value="keep_current">Keep Current</option>
          </select>
        </div>

        <button
          type="button"
          className="sidebar-action-btn"
          onClick={() => {
            const rawValue = window.prompt("New folder name", "New Folder");
            const name = rawValue?.trim() ?? "";
            if (!name) return;
            onCreateFolder(name, null);
          }}
          disabled={Boolean(ingesting || exporting)}
        >
          New Folder
        </button>

        <button
          type="button"
          className="sidebar-action-btn"
          onClick={() => onExportSession(exportMode, exportSource)}
          disabled={exporting || ingesting}
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
            disabled={exporting || ingesting}
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
                disabled={exporting || ingesting}
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
  onIngestFiles,
  onDelete,
  onCreateFolder,
  onCreateFolderSample,
  onDeleteFolder,
  onPruneFolder,
  onExportSession,
  exportSourceOptions,
  ingesting = false,
  deletingId = null,
  folderBusyId = null,
  exporting = false,
}: Props) {
  const [state, dispatch] = useReducer(
    sidebarReducer,
    exportSourceOptions,
    createInitialSidebarState,
  );
  const ingestRef = useRef<HTMLInputElement>(null);

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

  const openIngestPicker = useCallback(() => {
    ingestRef.current?.click();
  }, []);

  const handleIngestDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      dispatch({ type: "set_ingest_drag_over", value: false });
      if (ingesting || exporting) return;
      const files = Array.from(e.dataTransfer.files ?? []);
      if (files.length > 0) onIngestFiles(files, state.importConflictPolicy);
    },
    [exporting, ingesting, onIngestFiles, state.importConflictPolicy],
  );

  const handleIngestChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (ingesting || exporting) return;
      const files = Array.from(e.target.files ?? []);
      if (files.length > 0) onIngestFiles(files, state.importConflictPolicy);
      e.target.value = "";
    },
    [exporting, ingesting, onIngestFiles, state.importConflictPolicy],
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
        onCreateFolder={onCreateFolder}
        onRequestSample={handleRequestSample}
        onDeleteFolder={onDeleteFolder}
        onPruneFolder={onPruneFolder}
      />

      <SidebarActionsPanel
        ingesting={ingesting}
        exporting={exporting}
        exportMode={state.exportMode}
        exportSource={state.exportSource}
        exportSourceOptions={exportSourceOptions}
        importConflictPolicy={state.importConflictPolicy}
        onImportConflictPolicyChange={(value) =>
          dispatch({ type: "set_import_conflict_policy", value })
        }
        onExportModeChange={(value) => dispatch({ type: "set_export_mode", value })}
        onExportSourceChange={(value) => dispatch({ type: "set_export_source", value })}
        onExportSession={onExportSession}
        onCreateFolder={onCreateFolder}
        ingestDragOver={state.ingestDragOver}
        ingestRef={ingestRef}
        onIngestOpenPicker={openIngestPicker}
        onIngestDragOver={(e) => {
          e.preventDefault();
          if (!ingesting && !exporting) {
            dispatch({ type: "set_ingest_drag_over", value: true });
          }
        }}
        onIngestDragLeave={() => dispatch({ type: "set_ingest_drag_over", value: false })}
        onIngestDrop={handleIngestDrop}
        onIngestChange={handleIngestChange}
      />
    </aside>
  );
}
