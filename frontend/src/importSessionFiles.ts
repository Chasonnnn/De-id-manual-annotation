import type { SessionImportResult } from "./types";

export interface SessionImportBatchFailure {
  file_name: string;
  message: string;
}

export interface SessionImportBatchResult extends SessionImportResult {
  file_count: number;
  succeeded_file_count: number;
  failed_file_count: number;
  failed_files: SessionImportBatchFailure[];
}

function emptyBatchResult(fileCount: number): SessionImportBatchResult {
  return {
    file_count: fileCount,
    succeeded_file_count: 0,
    failed_file_count: 0,
    failed_files: [],
    imported_count: 0,
    imported_ids: [],
    imported_prompt_lab_runs: 0,
    imported_methods_lab_runs: 0,
    skipped_count: 0,
    skipped: [],
    warnings: [],
    total_in_bundle: 0,
  };
}

function toImportErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message || error.name;
  }
  return String(error) || "Unknown import failure";
}

export async function importSessionFiles<FileLike extends { name: string }>(
  files: readonly FileLike[],
  importOne: (file: FileLike) => Promise<SessionImportResult>,
): Promise<SessionImportBatchResult> {
  const aggregate = emptyBatchResult(files.length);

  for (const file of files) {
    try {
      const result = await importOne(file);
      aggregate.succeeded_file_count += 1;
      aggregate.imported_count += result.imported_count;
      aggregate.imported_ids.push(...result.imported_ids);
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
        message: toImportErrorMessage(error),
      });
    }
  }

  return aggregate;
}
