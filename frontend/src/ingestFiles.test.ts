import { describe, expect, it } from "vitest";

import { ingestFiles } from "./ingestFiles";

interface TestFile {
  name: string;
}

describe("ingestFiles", () => {
  it("aggregates uploads and structured imports across multiple files", async () => {
    const files: TestFile[] = [
      { name: "raw.json" },
      { name: "bundle.json" },
      { name: "ground-truth.zip" },
    ];

    const result = await ingestFiles(files, async (file) => {
      if (file.name === "raw.json") {
        return {
          mode: "upload",
          created_count: 1,
          created_ids: ["doc-raw"],
          imported_ids: [],
          uploaded_count: 1,
          imported_count: 0,
          imported_prompt_lab_runs: 0,
          imported_methods_lab_runs: 0,
          skipped_count: 0,
          skipped: [],
          warnings: [],
          total_in_bundle: 1,
        };
      }
      if (file.name === "bundle.json") {
        return {
          mode: "import",
          created_count: 2,
          created_ids: ["doc-1", "doc-2"],
          imported_ids: ["doc-1", "doc-2"],
          uploaded_count: 0,
          imported_count: 2,
          conflict_count: 1,
          replaced_count: 1,
          kept_current_count: 0,
          added_as_new_count: 0,
          imported_prompt_lab_runs: 1,
          imported_methods_lab_runs: 0,
          skipped_count: 1,
          skipped: [{ index: 0, reason: "bundle warning" }],
          warnings: ["first warning"],
          total_in_bundle: 3,
        };
      }
      return {
        mode: "import",
        created_count: 1,
        created_ids: ["doc-3"],
        imported_ids: ["doc-3"],
        uploaded_count: 0,
        imported_count: 1,
        conflict_count: 1,
        replaced_count: 0,
        kept_current_count: 0,
        added_as_new_count: 1,
        imported_prompt_lab_runs: 0,
        imported_methods_lab_runs: 2,
        skipped_count: 0,
        skipped: [],
        warnings: ["second warning"],
        total_in_bundle: 1,
      };
    });

    expect(result.file_count).toBe(3);
    expect(result.succeeded_file_count).toBe(3);
    expect(result.failed_file_count).toBe(0);
    expect(result.uploaded_file_count).toBe(1);
    expect(result.imported_file_count).toBe(2);
    expect(result.created_count).toBe(4);
    expect(result.created_ids).toEqual(["doc-raw", "doc-1", "doc-2", "doc-3"]);
    expect(result.uploaded_count).toBe(1);
    expect(result.imported_count).toBe(3);
    expect(result.conflict_count).toBe(2);
    expect(result.replaced_count).toBe(1);
    expect(result.added_as_new_count).toBe(1);
    expect(result.imported_prompt_lab_runs).toBe(1);
    expect(result.imported_methods_lab_runs).toBe(2);
    expect(result.skipped_count).toBe(1);
    expect(result.total_in_bundle).toBe(5);
    expect(result.warnings).toEqual(["first warning", "second warning"]);
  });

  it("records per-file failures and continues ingesting remaining files", async () => {
    const files: TestFile[] = [
      { name: "first.json" },
      { name: "broken.json" },
      { name: "last.zip" },
    ];

    const result = await ingestFiles(files, async (file) => {
      if (file.name === "broken.json") {
        throw new Error("400: unsupported file");
      }
      return {
        mode: file.name.endsWith(".zip") ? "import" : "upload",
        created_count: 1,
        created_ids: [file.name],
        imported_ids: file.name.endsWith(".zip") ? [file.name] : [],
        uploaded_count: file.name.endsWith(".zip") ? 0 : 1,
        imported_count: file.name.endsWith(".zip") ? 1 : 0,
        skipped_count: 0,
        skipped: [],
        total_in_bundle: 1,
      };
    });

    expect(result.file_count).toBe(3);
    expect(result.succeeded_file_count).toBe(2);
    expect(result.failed_file_count).toBe(1);
    expect(result.created_count).toBe(2);
    expect(result.created_ids).toEqual(["first.json", "last.zip"]);
    expect(result.failed_files).toEqual([
      { file_name: "broken.json", message: "400: unsupported file" },
    ]);
  });

  it("returns an empty aggregate when no files are selected", async () => {
    const result = await ingestFiles([], async () => {
      throw new Error("should not run");
    });

    expect(result.file_count).toBe(0);
    expect(result.succeeded_file_count).toBe(0);
    expect(result.failed_file_count).toBe(0);
    expect(result.uploaded_file_count).toBe(0);
    expect(result.imported_file_count).toBe(0);
    expect(result.created_count).toBe(0);
    expect(result.created_ids).toEqual([]);
    expect(result.failed_files).toEqual([]);
    expect(result.warnings).toEqual([]);
  });
});
