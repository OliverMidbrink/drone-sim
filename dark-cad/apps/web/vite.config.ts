import { defineConfig } from 'vite'
import solid from 'vite-plugin-solid'

export default defineConfig({
  plugins: [solid()],
  base: './', // relative so the same bundle loads from file:// in the desktop shell
  worker: { format: 'es' },
  build: { target: 'es2022', chunkSizeWarningLimit: 4000 },
  server: { port: 5173, proxy: { '/api': process.env.DARKCAD_API ?? 'http://127.0.0.1:8787' } },
  preview: { proxy: { '/api': process.env.DARKCAD_API ?? 'http://127.0.0.1:8787' } },
})
