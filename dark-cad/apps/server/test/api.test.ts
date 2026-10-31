import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createApp } from '../src/app'
import { Silo } from '../src/silo'

let dir: string
let app: ReturnType<typeof createApp>
beforeEach(async () => { dir = await mkdtemp(join(tmpdir(), 'darkcad-api-')); app = createApp(new Silo(dir)) })
afterEach(async () => { await rm(dir, { recursive: true, force: true }) })

const doc = (name = 'Part') => ({ version: 3, name, units: 'mm', mode: 'part', bodyNames: {}, hidden: [], features: [] })
const req = (method: string, path: string, body?: unknown, headers: Record<string, string> = {}) =>
  app.request(`/api${path}`, { method, headers: { ...(body !== undefined && !(body instanceof Uint8Array) ? { 'content-type': 'application/json' } : {}), ...headers }, body: body === undefined ? undefined : body instanceof Uint8Array ? (body as any) : JSON.stringify(body) })

describe('REST api', () => {
  beforeEach(async () => { expect((await req('POST', '/projects', { name: 'P', id: 'p' })).status).toBe(201) })

  it('creates a design (201) then needs If-Match, answering 409 with the current rev when stale', async () => {
    let r = await req('PUT', '/projects/p/designs/d1', { doc: doc('A') })
    expect(r.status).toBe(201)
    expect(r.headers.get('ETag')).toBe('"1"')
    r = await req('PUT', '/projects/p/designs/d1', { doc: doc('B') })
    expect(r.status).toBe(428)
    r = await req('PUT', '/projects/p/designs/d1', { doc: doc('B') }, { 'If-Match': '"1"' })
    expect(r.status).toBe(200); expect(await r.json()).toMatchObject({ rev: 2 })
    r = await req('PUT', '/projects/p/designs/d1', { doc: doc('C') }, { 'If-Match': '"1"' })
    expect(r.status).toBe(409); expect(await r.json()).toMatchObject({ rev: 2 })
    r = await req('GET', '/projects/p/designs/d1')
    expect(await r.json()).toMatchObject({ rev: 2, doc: { name: 'B' } })
  })

  it('rejects unknown projects, bad ids and bad JSON', async () => {
    expect((await req('GET', '/projects/nope')).status).toBe(404)
    expect((await req('GET', '/projects/p/designs/..%2Fx')).status).toBeGreaterThanOrEqual(400)
    expect((await app.request('/api/projects/p/folders', { method: 'POST', body: 'not json' })).status).toBe(400)
    expect((await req('PUT', '/projects/p/designs/d1', { doc: { x: 1 } })).status).toBe(400)
  })

  it('manages folders and moves a design into one', async () => {
    const f = await (await req('POST', '/projects/p/folders', { name: 'Frames' })).json()
    await req('PUT', '/projects/p/designs/d1', { doc: doc('A') })
    const r = await req('PATCH', '/projects/p/nodes/d1', { parent: f.id, name: 'Arm' })
    expect(await r.json()).toMatchObject({ parent: f.id, name: 'Arm' })
    const m = await (await req('GET', '/projects/p')).json()
    expect(m.nodes.map((n: any) => n.name).sort()).toEqual(['Arm', 'Frames'])
    expect((await req('DELETE', `/projects/p/nodes/${f.id}`)).status).toBe(200)
    expect((await req('GET', '/projects/p/designs/d1')).status).toBe(404)
  })

  it('saves, lists and restores versions', async () => {
    await req('PUT', '/projects/p/designs/d1', { doc: doc('one') })
    const v = await (await req('POST', '/projects/p/designs/d1/versions', { name: 'Milestone', note: 'n' })).json()
    await req('PUT', '/projects/p/designs/d1', { doc: doc('two') }, { 'If-Match': '1' })
    expect((await (await req('GET', '/projects/p/designs/d1/versions')).json()).some((x: any) => x.name === 'Milestone')).toBe(true)
    const r = await req('POST', `/projects/p/designs/d1/versions/${v.id}/restore`)
    expect(await r.json()).toMatchObject({ rev: 3, doc: { name: 'one' } })
  })

  it('round-trips an asset byte for byte, and a thumbnail', async () => {
    const bytes = new Uint8Array(Array.from({ length: 1000 }, (_, i) => i % 256))
    let r = await req('PUT', '/projects/p/assets/a1?name=part.step', bytes, { 'content-type': 'application/step' })
    expect(r.status).toBe(201)
    r = await req('GET', '/projects/p/assets/a1')
    expect(r.headers.get('content-type')).toBe('application/step')
    expect([...new Uint8Array(await r.arrayBuffer())]).toEqual([...bytes])
    await req('PUT', '/projects/p/designs/d1', { doc: doc() })
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 9])
    expect((await req('PUT', '/projects/p/designs/d1/thumb', png, { 'content-type': 'image/png' })).status).toBe(204)
    r = await req('GET', '/projects/p/designs/d1/thumb')
    expect(r.headers.get('content-type')).toBe('image/png')
    expect([...new Uint8Array(await r.arrayBuffer())]).toEqual([...png])
  })
})
