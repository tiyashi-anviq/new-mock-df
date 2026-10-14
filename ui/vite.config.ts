import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server on 3004. Any call to /api/... is forwarded to the FastAPI service (the /api prefix is removed),
// so the browser only ever talks to one address and there are no cross-origin problems.
// The API is started with `uv run uvicorn app:app --reload`, which listens on port 8000.
// If you start it on another port, set it before npm run dev:   $env:API_PORT=4004
const apiPort = process.env.API_PORT ?? '8000'

export default defineConfig({
  plugins: [react()],
  css: { preprocessorOptions: { scss: { quietDeps: true } } },
  server: {
    port: 3004,
    strictPort: true,
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${apiPort}`,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
