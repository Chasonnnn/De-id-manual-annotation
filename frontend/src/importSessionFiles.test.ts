import { describe, expect, it } from "vitest";

import { importSessionFiles } from "./importSessionFiles";

interface TestFile {
  name: string;
}

describe("importSessionFiles", () => {
  it("aggregates successful imports across multiple files", async () => {
    const files: TestFile[] = [{ name: "a.json" }, { name: "b.json" }];

    const result = await importSessionFiles(files, async (file) => {
      if (file.name === "a.json") {
        return {
          imported_count: 2,
          imported_ids: ["doc-1", "doc-2"],
          imported_prompt_lab_runs: 1,
          imported_methods_lab_runs: 0,
          skipped_count: 0,
          skipped: [],
          warnings: ["first warning"],
          total_in_bundle: 2,
        };
      }
      return {
        imported_count: 1,
        imported_ids: ["doc-3"],
        imported_prompt_lab_runs: 0,
        imported_methods_lab_runs: 2,
        skipped_count: 1,
        skipped: [{ index: 0, reason: "bad item" }],
        warnings: ["second warning"],
        total_in_bundle: 2,
      };
    });

    expect(result.file_count).toBe(2);
    expect(result.succeeded_file_count).toBe(2);
    expect(result.failed_file_count).toBe(0);
    expect(result.imported_count).toBe(3);
    expect(result.imported_ids).toEqual(["doc-1", "doc-2", "doc-3"]);
    expect(result.imported_prompt_lab_runs).toBe(1);
    expect(result.imported_methods_lab_runs).toBe(2);
    expect(result.skipped_count).toBe(1);
    expect(result.total_in_bundle).toBe(4);
    expect(result.warnings).toEqual(["first warning", "second warning"]);
  });

  it("records per-file failures and continues importing remaining files", async () => {
    const files: TestFile[] = [
      { name: "first.json" },
      { name: "broken.json" },
      { name: "last.json" },
    ];

    const result = await importSessionFiles(files, async (file) => {
      if (file.name === "broken.json") {
        throw new Error("400: invalid bundle");
      }
      return {
        imported_count: 1,
        imported_ids: [file.name],
        skipped_count: 0,
        skipped: [],
        total_in_bundle: 1,
      };
    });

    expect(result.file_count).toBe(3);
    expect(result.succeeded_file_count).toBe(2);
    expect(result.failed_file_count).toBe(1);
    expect(result.imported_count).toBe(2);
    expect(result.imported_ids).toEqual(["first.json", "last.json"]);
    expect(result.failed_files).toEqual([
      { file_name: "broken.json", message: "400: invalid bundle" },
    ]);
  });

  it("returns an empty aggregate when no files are selected", async () => {
    const result = await importSessionFiles([], async () => {
      throw new Error("should not run");
    });

    expect(result.file_count).toBe(0);
    expect(result.succeeded_file_count).toBe(0);
    expect(result.failed_file_count).toBe(0);
    expect(result.imported_count).toBe(0);
    expect(result.imported_ids).toEqual([]);
    expect(result.failed_files).toEqual([]);
    expect(result.warnings).toEqual([]);
  });
});
