import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  test: {
    globals: true,
    environment: "jsdom",
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    setupFiles: ["src/testSetup.ts"],
  },
});
