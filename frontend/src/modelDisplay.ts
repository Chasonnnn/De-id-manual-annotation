import type { PromptLabModelInput } from "./types";

type ExperimentModelLike = Partial<
  Pick<
    PromptLabModelInput,
    "id" | "label" | "model" | "reasoning_effort" | "anthropic_thinking"
  >
>;

export function formatExperimentModelLabel(
  model: ExperimentModelLike | null | undefined,
  fallback = "model",
): string {
  const baseLabel = model?.model?.trim() || model?.label?.trim() || fallback;
  const qualifiers: string[] = [];

  if (model?.reasoning_effort && model.reasoning_effort !== "none") {
    qualifiers.push(model.reasoning_effort);
  }
  if (model?.anthropic_thinking) {
    qualifiers.push("thinking");
  }

  if (qualifiers.length === 0) {
    return baseLabel;
  }
  return `${baseLabel} (${qualifiers.join(", ")})`;
}

export function getExperimentModelLabelById(
  models: ExperimentModelLike[] | undefined,
  modelId: string | null | undefined,
  fallback = "model",
): string {
  if (!modelId) {
    return fallback;
  }
  const model = models?.find((item) => item.id === modelId);
  return formatExperimentModelLabel(model, fallback);
}
