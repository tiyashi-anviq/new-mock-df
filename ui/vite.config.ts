import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiPort = process.env.API_PORT ?? '4004'

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
