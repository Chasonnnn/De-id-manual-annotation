export interface ModelPreset {
  id: string;
  label: string;
  group: "OpenAI" | "Anthropic" | "Google Gemini";
  supportsReasoningEffort: boolean;
  supportsAnthropicThinking: boolean;
  defaultReasoningEffort?: "none" | "low" | "medium" | "high" | "xhigh";
}

export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: "openai.gpt-5.3-codex",
    label: "Codex 5.3 (xhigh)",
    group: "OpenAI",
    supportsReasoningEffort: true,
    supportsAnthropicThinking: false,
    defaultReasoningEffort: "xhigh",
  },
  {
    id: "openai.gpt-5.4",
    label: "GPT-5.4 (xhigh)",
    group: "OpenAI",
    supportsReasoningEffort: true,
    supportsAnthropicThinking: false,
    defaultReasoningEffort: "xhigh",
  },
  {
    id: "openai.gpt-5.2-chat",
    label: "ChatGPT 5.2 (xhigh)",
    group: "OpenAI",
    supportsReasoningEffort: true,
    supportsAnthropicThinking: false,
    defaultReasoningEffort: "xhigh",
  },
  {
    id: "anthropic.claude-4.6-opus",
    label: "Claude Opus 4.6 (thinking)",
    group: "Anthropic",
    supportsReasoningEffort: false,
    supportsAnthropicThinking: true,
  },
  {
    id: "google.gemini-3.1-pro-preview",
    label: "Gemini 3.1 Pro Preview",
    group: "Google Gemini",
    supportsReasoningEffort: false,
    supportsAnthropicThinking: false,
  },
  {
    id: "google.gemini-3.1-flash-lite-preview",
    label: "Gemini 3.1 Flash Lite Preview",
    group: "Google Gemini",
    supportsReasoningEffort: false,
    supportsAnthropicThinking: false,
  },
];

const byId = new Map(MODEL_PRESETS.map((preset) => [preset.id, preset]));

export function getModelPreset(modelId: string): ModelPreset | undefined {
  return byId.get(modelId);
}
