import { defineConfig } from 'vitest/config'

// Separate from vite.config.ts: the Solid plugin would switch tests to jsdom, and these are pure-logic tests.
export default defineConfig({ test: { environment: 'node', include: ['src/**/*.test.ts'] } })
