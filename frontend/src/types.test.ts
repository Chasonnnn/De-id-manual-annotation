import { describe, expect, it } from "vitest";

import { DEFAULT_EXPERIMENT_LIMITS, PII_LABELS } from "./types";

describe("PII_LABELS", () => {
  it("uses the unified canonical entity taxonomy", () => {
    expect(PII_LABELS).toEqual([
      "NAME",
      "ADDRESS",
      "DATE",
      "PHONE_NUMBER",
      "FAX_NUMBER",
      "EMAIL",
      "SSN",
      "ACCOUNT_NUMBER",
      "DEVICE_IDENTIFIER",
      "URL",
      "IP_ADDRESS",
      "BIOMETRIC_IDENTIFIER",
      "IMAGE",
      "IDENTIFYING_NUMBER",
      "AGE",
      "SCHOOL",
      "TUTOR_PROVIDER",
    ]);
  });
});

describe("DEFAULT_EXPERIMENT_LIMITS", () => {
  it("defaults both experiment labs to concurrency 32", () => {
    expect(DEFAULT_EXPERIMENT_LIMITS).toMatchObject({
      prompt_lab_default_concurrency: 32,
      prompt_lab_max_concurrency: 32,
      methods_lab_default_concurrency: 32,
      methods_lab_max_concurrency: 32,
    });
  });
});
