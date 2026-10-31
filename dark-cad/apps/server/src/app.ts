import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { HttpError, Silo } from './silo'

const MAX_ASSET = 200 * 1024 * 1024

/** `If-Match: "3"` / `3` → 3, `*` → '*'. */
function parseIfMatch(h: string | undefined): number | '*' | undefined {
  if (h === undefined) return undefined
  const v = h.trim().replace(/^W\//, '').replace(/^"|"$/g, '')
  if (v === '*') return '*'
  const n = Number(v)
  if (!Number.isInteger(n) || n < 0) throw new HttpError(400, 'bad If-Match header')
  return n
}

async function json<T = any>(req: Request): Promise<T> {
  try { return (await req.json()) as T } catch { throw new HttpError(400, 'body must be JSON') }
}

export function createApp(silo: Silo) {
  const app = new Hono()
  app.use('/api/*', cors({ origin: (o) => o, exposeHeaders: ['ETag'] }))

  app.onError((e, c) => {
    if (e instanceof HttpError) return c.json({ error: e.message, ...(e.body as object | undefined) }, e.status as any)
    console.error(e)
    return c.json({ error: 'internal error' }, 500)
  })

  const api = new Hono()

  // ---- projects
  api.get('/projects', async (c) => c.json(await silo.listProjects()))
  api.post('/projects', async (c) => {
    const b = await json(c.req.raw)
    return c.json(await silo.createProject(String(b.name ?? 'Project'), b.id), 201)
  })
  api.get('/projects/:p', async (c) => c.json(await silo.manifest(c.req.param('p'))))
  api.patch('/projects/:p', async (c) => c.json(await silo.renameProject(c.req.param('p'), String((await json(c.req.raw)).name ?? ''))))

  // ---- tree: folders, rename / move / delete of any node
  api.post('/projects/:p/folders', async (c) => {
    const b = await json(c.req.raw)
    return c.json(await silo.createFolder(c.req.param('p'), String(b.name ?? 'Folder'), b.parent ?? null, b.id), 201)
  })
  api.patch('/projects/:p/nodes/:id', async (c) => {
    const b = await json(c.req.raw)
    return c.json(await silo.renameNode(c.req.param('p'), c.req.param('id'), { name: b.name, parent: b.parent }))
  })
  api.delete('/projects/:p/nodes/:id', async (c) => c.json({ deleted: await silo.deleteNode(c.req.param('p'), c.req.param('id')) }))

  // ---- designs
  api.get('/projects/:p/designs/:d', async (c) => {
    const d = await silo.getDesign(c.req.param('p'), c.req.param('d'))
    c.header('ETag', `"${d.rev}"`)
    return c.json(d)
  })
  api.put('/projects/:p/designs/:d', async (c) => {
    const b = await json(c.req.raw)
    const existed = await silo.getDesign(c.req.param('p'), c.req.param('d')).then(() => true, () => false)
    const d = await silo.putDesign(c.req.param('p'), c.req.param('d'), { doc: b.doc, marker: b.marker, parent: b.parent }, parseIfMatch(c.req.header('If-Match')))
    c.header('ETag', `"${d.rev}"`)
    return c.json({ rev: d.rev, updated: d.updated }, existed ? 200 : 201)
  })

  // ---- versions
  api.get('/projects/:p/designs/:d/versions', async (c) => c.json(await silo.listVersions(c.req.param('p'), c.req.param('d'))))
  api.post('/projects/:p/designs/:d/versions', async (c) => {
    const b = await json(c.req.raw)
    return c.json(await silo.createVersion(c.req.param('p'), c.req.param('d'), String(b.name ?? ''), String(b.note ?? '')), 201)
  })
  api.get('/projects/:p/designs/:d/versions/:v', async (c) => c.json(await silo.getVersion(c.req.param('p'), c.req.param('d'), c.req.param('v'))))
  api.post('/projects/:p/designs/:d/versions/:v/restore', async (c) => {
    const d = await silo.restoreVersion(c.req.param('p'), c.req.param('d'), c.req.param('v'))
    c.header('ETag', `"${d.rev}"`)
    return c.json(d)
  })

  // ---- thumbnails
  api.put('/projects/:p/designs/:d/thumb', async (c) => {
    await silo.putThumb(c.req.param('p'), c.req.param('d'), new Uint8Array(await c.req.arrayBuffer()))
    return c.body(null, 204)
  })
  api.get('/projects/:p/designs/:d/thumb', async (c) => {
    const png = await silo.getThumb(c.req.param('p'), c.req.param('d'))
    return c.body(new Uint8Array(png), 200, { 'Content-Type': 'image/png', 'Cache-Control': 'no-cache' })
  })

  // ---- assets (raw bytes)
  api.put('/projects/:p/assets/:id', async (c) => {
    const len = Number(c.req.header('content-length') ?? 0)
    if (len > MAX_ASSET) throw new HttpError(413, 'file too large')
    const bytes = new Uint8Array(await c.req.arrayBuffer())
    if (bytes.length > MAX_ASSET) throw new HttpError(413, 'file too large')
    const q = c.req.query()
    const node = await silo.putAsset(c.req.param('p'), c.req.param('id'), {
      filename: q.name ?? 'file', mime: c.req.header('content-type') ?? 'application/octet-stream', bytes, parent: q.parent || null,
    })
    return c.json(node, 201)
  })
  api.get('/projects/:p/assets/:id', async (c) => {
    const { node, bytes } = await silo.getAsset(c.req.param('p'), c.req.param('id'))
    return c.body(new Uint8Array(bytes), 200, {
      'Content-Type': node.mime ?? 'application/octet-stream',
      'Content-Disposition': `inline; filename="${encodeURIComponent(node.filename ?? 'file')}"`,
    })
  })

  app.route('/api', api)
  app.get('/api/health', (c) => c.json({ ok: true }))
  return app
}
