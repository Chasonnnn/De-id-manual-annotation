import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import {
  loadConfigFromFile,
  mergeConfig,
  type ConfigEnv,
  type PluginOption,
  type Plugin,
  type UserConfig,
} from "vite";

const CONFIG_ROOT = fileURLToPath(new URL(".", import.meta.url));

export function resolveApiProxyTarget(env: NodeJS.ProcessEnv = process.env): string {
  const rawPort = typeof env.BACKEND_PORT === "string" ? env.BACKEND_PORT.trim() : "";
  const port = rawPort || "8000";
  return `http://localhost:${port}`;
}

export function createBaseViteConfig(): UserConfig {
  return {
    plugins: [react()],
    server: {
      proxy: {
        "/api": {
          target: resolveApiProxyTarget(),
          changeOrigin: true,
        },
      },
    },
    test: {
      environment: "jsdom",
    },
  };
}

function flattenPluginOptions(plugins: PluginOption[] | undefined): Plugin[] {
  if (!plugins) {
    return [];
  }

  const flattened: Plugin[] = [];
  for (const plugin of plugins) {
    if (!plugin) {
      continue;
    }
    if (Array.isArray(plugin)) {
      flattened.push(...flattenPluginOptions(plugin));
      continue;
    }
    flattened.push(plugin);
  }
  return flattened;
}

function dedupePlugins(plugins: PluginOption[] | undefined): Plugin[] | undefined {
  const flattened = flattenPluginOptions(plugins);
  if (flattened.length === 0) {
    return undefined;
  }

  const seenNames = new Set<string>();
  const deduped: Plugin[] = [];
  for (const plugin of flattened) {
    const pluginName = typeof plugin.name === "string" ? plugin.name : "";
    if (pluginName && seenNames.has(pluginName)) {
      continue;
    }
    if (pluginName) {
      seenNames.add(pluginName);
    }
    deduped.push(plugin);
  }
  return deduped;
}

export async function loadMergedViteConfig(
  env: ConfigEnv,
  configRoot: string = CONFIG_ROOT,
): Promise<UserConfig> {
  const baseConfig = createBaseViteConfig();
  const localConfigPath = path.join(configRoot, "vite.config.local.ts");

  if (!existsSync(localConfigPath)) {
    return baseConfig;
  }

  const loadedLocalConfig = await loadConfigFromFile(env, localConfigPath, configRoot, "silent");
  if (loadedLocalConfig === null) {
    throw new Error(`Failed to load local Vite config at ${localConfigPath}`);
  }

  const mergedConfig = mergeConfig(baseConfig, loadedLocalConfig.config);
  return {
    ...mergedConfig,
    plugins: dedupePlugins(mergedConfig.plugins),
  };
}
