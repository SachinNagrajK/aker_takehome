import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Dev only — forward /api/* to the local FastAPI backend.
      // In prod we set VITE_API_BASE to the HF Space URL and skip the proxy.
      // (Image artifacts now load directly from Supabase Storage, so the
      // /doc_store proxy is gone.)
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
