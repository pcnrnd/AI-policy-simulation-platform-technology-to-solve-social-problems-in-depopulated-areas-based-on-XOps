import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 8000,
    open: true
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      output: {
        // 무거운 서드파티를 별도 청크로 분리 — 초기 로드 축소 + 장기 캐싱
        manualChunks: {
          react: ["react", "react-dom"],
          leaflet: ["leaflet"],
          charts: ["chart.js", "react-chartjs-2"]
        }
      }
    }
  }
});
