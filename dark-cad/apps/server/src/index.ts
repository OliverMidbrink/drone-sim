import { serve } from '@hono/node-server'
import { serveStatic } from '@hono/node-server/serve-static'
import { existsSync } from 'node:fs'
import { mkdir } from 'node:fs/promises'
import { dirname, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createApp } from './app'
import { Silo } from './silo'

const here = dirname(fileURLToPath(import.meta.url))
const dataDir = resolve(process.env.DARKCAD_DATA ?? resolve(here, '../data'))
const port = Number(process.env.PORT ?? 8787)
const webDist = resolve(here, '../../web/dist')

await mkdir(dataDir, { recursive: true })
const app = createApp(new Silo(dataDir))

// Production: the server also serves the built web app.
if (existsSync(webDist)) {
  const rel = relative(process.cwd(), webDist) || '.'
  app.use('/*', serveStatic({ root: rel }))
  app.get('*', serveStatic({ path: `${rel}/index.html` }))
}

serve({ fetch: app.fetch, port, hostname: process.env.HOST ?? '127.0.0.1' }, (i) => {
  console.log(`DarkCAD server on http://${i.address}:${i.port}  ·  data: ${dataDir}${existsSync(webDist) ? '  ·  serving apps/web/dist' : ''}`)
})
