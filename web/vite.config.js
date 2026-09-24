import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /api to the FastAPI backend so the dev server needs zero config.
// Port 8000 is a popular default and often taken (Docker containers love it),
// so LFL_API_PORT points the proxy at a backend started on another port.
const apiPort = process.env.LFL_API_PORT || '8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': `http://localhost:${apiPort}` },
  },
})
