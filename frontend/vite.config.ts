import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/chat': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/session': 'http://127.0.0.1:8000',
      '/analyze-report': 'http://127.0.0.1:8000',
      '/analyze-xray': 'http://127.0.0.1:8000',
      '/feedback': 'http://127.0.0.1:8000',
      '/voice': 'http://127.0.0.1:8000',
      '/speech-to-text': 'http://127.0.0.1:8000',
      '/text-to-speech': 'http://127.0.0.1:8000',
    },
  },
})
