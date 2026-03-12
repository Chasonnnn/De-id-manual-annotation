import { defineConfig } from "vite";

import { loadMergedViteConfig } from "./viteConfig";

export default defineConfig((env) => loadMergedViteConfig(env));
