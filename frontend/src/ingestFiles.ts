import type { SessionIngestResult } from "./types";

export interface SessionIngestBatchFailure {
  file_name: string;
  message: string;
}

export interface SessionIngestBatchResult {
  file_count: number;
  succeeded_file_count: number;
  failed_file_count: number;
  failed_files: SessionIngestBatchFailure[];
  uploaded_file_count: number;
  imported_file_count: number;
  created_count: number;
  created_ids: string[];
  uploaded_count: number;
  imported_count: number;
  conflict_count?: number;
  replaced_count?: number;
  kept_current_count?: number;
  added_as_new_count?: number;
  imported_prompt_lab_runs?: number;
  imported_methods_lab_runs?: number;
  skipped_count: number;
  skipped: Array<{ index: number; reason: string }>;
  warnings?: string[];
  total_in_bundle: number;
}

function emptyBatchResult(fileCount: number): SessionIngestBatchResult {
  return {
    file_count: fileCount,
    succeeded_file_count: 0,
    failed_file_count: 0,
    failed_files: [],
    uploaded_file_count: 0,
    imported_file_count: 0,
    created_count: 0,
    created_ids: [],
    uploaded_count: 0,
    imported_count: 0,
    conflict_count: 0,
    replaced_count: 0,
    kept_current_count: 0,
    added_as_new_count: 0,
    imported_prompt_lab_runs: 0,
    imported_methods_lab_runs: 0,
    skipped_count: 0,
    skipped: [],
    warnings: [],
    total_in_bundle: 0,
  };
}

function toIngestErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message || error.name;
  }
  return String(error) || "Unknown ingest failure";
}

export async function ingestFiles<FileLike extends { name: string }>(
  files: readonly FileLike[],
  ingestOne: (file: FileLike) => Promise<SessionIngestResult>,
): Promise<SessionIngestBatchResult> {
  const aggregate = emptyBatchResult(files.length);

  for (const file of files) {
    try {
      const result = await ingestOne(file);
      aggregate.succeeded_file_count += 1;
      aggregate.created_count += result.created_count;
      aggregate.created_ids.push(...result.created_ids);
      aggregate.uploaded_count += result.uploaded_count;
      aggregate.imported_count += result.imported_count;
      aggregate.conflict_count = (aggregate.conflict_count ?? 0) + (result.conflict_count ?? 0);
      aggregate.replaced_count = (aggregate.replaced_count ?? 0) + (result.replaced_count ?? 0);
      aggregate.kept_current_count =
        (aggregate.kept_current_count ?? 0) + (result.kept_current_count ?? 0);
      aggregate.added_as_new_count =
        (aggregate.added_as_new_count ?? 0) + (result.added_as_new_count ?? 0);
      if (result.mode === "upload") {
        aggregate.uploaded_file_count += 1;
      } else {
        aggregate.imported_file_count += 1;
      }
      aggregate.imported_prompt_lab_runs =
        (aggregate.imported_prompt_lab_runs ?? 0) + (result.imported_prompt_lab_runs ?? 0);
      aggregate.imported_methods_lab_runs =
        (aggregate.imported_methods_lab_runs ?? 0) + (result.imported_methods_lab_runs ?? 0);
      aggregate.skipped_count += result.skipped_count;
      aggregate.skipped.push(
        ...result.skipped.map((item) => ({
          index: item.index,
          reason: `${file.name}: ${item.reason}`,
        })),
      );
      aggregate.total_in_bundle += result.total_in_bundle;
      aggregate.warnings?.push(...(result.warnings ?? []));
    } catch (error: unknown) {
      aggregate.failed_file_count += 1;
      aggregate.failed_files.push({
        file_name: file.name,
        message: toIngestErrorMessage(error),
      });
    }
  }

  return aggregate;
}
