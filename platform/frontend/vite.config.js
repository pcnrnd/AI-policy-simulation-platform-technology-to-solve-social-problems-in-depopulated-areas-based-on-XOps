import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // 포트는 Vite 기본(5173) 사용
    open: true,
    // dev에서 /api 요청을 xops-service(8000, uvicorn 기본)로 프록시 — CORS 회피
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true }
    }
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
