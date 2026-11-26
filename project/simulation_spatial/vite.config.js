import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 화면의 /api 요청은 로컬 API(api.js)로 넘긴다.
const api = { "/api": `http://127.0.0.1:${process.env.API_PORT || 5182}` };

export default defineConfig({
  plugins: [react()],
  server: { port: 5180, proxy: api },
  preview: { port: 5181, proxy: api }
});
