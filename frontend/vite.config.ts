import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// 宿主 API 端口：38000（本机 5xxx/8xxx 段被其他服务占用，存储栈已占 39530/35432/36379）。
// 前端一律用相对路径打 /api/*，代理只存在于 dev；生产由 Nginx 同源反代，
// 这样「开发时通、上线时 CORS 炸」这条经典分叉在结构上就不成立。
const API_ORIGIN = process.env.KEE_API_ORIGIN ?? "http://127.0.0.1:38000";

export default defineConfig({
  plugins: [react()],
  server: {
    // 351xx/361xx/371xx 段在本机 bind 直接 WSAEACCES（系统保留，netstat 里查不到占用者），
    // 所以 dev/preview 落在实测可绑的 381xx。strictPort 保持开：端口被占要报错，不要悄悄换一个，
    // 否则「文档里写的地址打不开、实际跑在别的端口上」这种状态最难查。
    host: "127.0.0.1",
    port: 38173,
    strictPort: true,
    proxy: {
      "/api": { target: API_ORIGIN, changeOrigin: false },
      "/healthz": { target: API_ORIGIN, changeOrigin: false },
    },
  },
  preview: { host: "127.0.0.1", port: 38174, strictPort: true },
  build: { outDir: "dist", sourcemap: true },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["src/test/setup.ts"],
  },
});
