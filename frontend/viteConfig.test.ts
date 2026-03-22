// @vitest-environment node

import { mkdtemp, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import type { ConfigEnv } from "vite";
import { afterEach, describe, expect, it } from "vitest";

import { createBaseViteConfig, loadMergedViteConfig, resolveApiProxyTarget } from "./viteConfig";

const DEV_ENV: ConfigEnv = {
  command: "serve",
  mode: "development",
  isSsrBuild: false,
  isPreview: false,
};
const TEST_ROOT = fileURLToPath(new URL(".", import.meta.url));

describe("loadMergedViteConfig", () => {
  const tempDirs: string[] = [];

  afterEach(async () => {
    await Promise.all(tempDirs.map((dir) => rm(dir, { recursive: true, force: true })));
    tempDirs.length = 0;
  });

  it("returns the base config when no local config exists", async () => {
    const dir = await mkdtemp(path.join(TEST_ROOT, ".vite-config-test-"));
    tempDirs.push(dir);

    const config = await loadMergedViteConfig(DEV_ENV, dir);
    const proxy = (config.server as { proxy?: Record<string, { target: string }> } | undefined)?.proxy;

    expect(proxy?.["/api"]?.target).toBe("http://localhost:8000");
    expect(config.test?.environment).toBe(createBaseViteConfig().test?.environment);
  });

  it("merges vite.config.local.ts when present", async () => {
    const dir = await mkdtemp(path.join(TEST_ROOT, ".vite-config-test-"));
    tempDirs.push(dir);
    await writeFile(
      path.join(dir, "vite.config.local.ts"),
      `
export default {
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8001",
        changeOrigin: true,
      },
    },
  },
};
`,
    );

    const config = await loadMergedViteConfig(DEV_ENV, dir);
    const proxy = (config.server as { proxy?: Record<string, { target: string; changeOrigin?: boolean }> } | undefined)?.proxy;

    expect(proxy?.["/api"]?.target).toBe("http://localhost:8001");
    expect(proxy?.["/api"]?.changeOrigin).toBe(true);
    expect(config.test?.environment).toBe("jsdom");
  });

  it("dedupes plugins when local config repeats the React plugin", async () => {
    const dir = await mkdtemp(path.join(TEST_ROOT, ".vite-config-test-"));
    tempDirs.push(dir);
    await writeFile(
      path.join(dir, "vite.config.local.ts"),
      `
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8001",
        changeOrigin: true,
      },
    },
  },
});
`,
    );

    const config = await loadMergedViteConfig(DEV_ENV, dir);
    const pluginNames = (config.plugins ?? [])
      .map((plugin) => ("name" in plugin && typeof plugin.name === "string" ? plugin.name : ""))
      .filter(Boolean);

    expect(pluginNames.filter((name) => name === "vite:react-babel")).toHaveLength(1);
    expect(pluginNames.filter((name) => name === "vite:react-refresh")).toHaveLength(1);
  });
});

describe("resolveApiProxyTarget", () => {
  it("uses port 8000 by default", () => {
    expect(resolveApiProxyTarget({} as NodeJS.ProcessEnv)).toBe("http://localhost:8000");
  });

  it("uses BACKEND_PORT when present", () => {
    expect(resolveApiProxyTarget({ BACKEND_PORT: "8001" } as NodeJS.ProcessEnv)).toBe(
      "http://localhost:8001",
    );
  });
});
