import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { mkdtemp, rm, readdir, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Silo, HttpError, atomicWrite, safeFilename } from '../src/silo'

let dir: string
let clock = 1_000_000
let silo: Silo
beforeEach(async () => { dir = await mkdtemp(join(tmpdir(), 'darkcad-')); clock = 1_000_000; silo = new Silo(dir, () => clock) })
afterEach(async () => { await rm(dir, { recursive: true, force: true }) })

const doc = (name = 'Part', n = 0) => ({ version: 3, name, units: 'mm', mode: 'part', bodyNames: {}, hidden: [], features: Array.from({ length: n }, (_, i) => ({ id: `f${i}` })) })
const fails = async (p: Promise<unknown>, status: number) => { try { await p } catch (e) { expect(e).toBeInstanceOf(HttpError); expect((e as HttpError).status).toBe(status); return } throw new Error('expected failure') }

describe('atomic writes', () => {
  it('replaces a file and leaves no temp files behind', async () => {
    const f = join(dir, 'x.json')
    await atomicWrite(f, 'one'); await atomicWrite(f, 'two')
    expect(await readFile(f, 'utf8')).toBe('two')
    expect((await readdir(dir)).filter((n) => n.endsWith('.tmp'))).toEqual([])
  })
  it('a failed rename keeps the old content and cleans up', async () => {
    const f = join(dir, 'x.json')
    await atomicWrite(f, 'keep')
    await fails(atomicWrite(join(dir, 'missing', 'x.json'), 'nope') as Promise<unknown>, 0).catch(() => undefined)
    expect(await readFile(f, 'utf8')).toBe('keep')
  })
  it('sanitises file names', () => {
    expect(safeFilename('../../etc/passwd')).toBe('passwd')
    expect(safeFilename('C:\\x\\part.step')).toBe('part.step')
    expect(safeFilename('..hidden')).toBe('_hidden')
    expect(safeFilename('')).toBe('file')
  })
})

describe('projects and the tree', () => {
  it('creates a silo directory and lists projects', async () => {
    const m = await silo.createProject('Drone', 'p1')
    expect(m.nodes).toEqual([])
    expect((await readdir(join(dir, 'silos', 'p1'))).sort()).toEqual(['assets', 'designs', 'manifest.json'])
    expect((await silo.listProjects()).map((p) => p.name)).toEqual(['Drone'])
    await fails(silo.createProject('again', 'p1'), 409)
    await fails(silo.createProject('x', '../evil'), 400)
  })

  it('folders nest, rename, move and delete with their contents', async () => {
    await silo.createProject('P', 'p')
    const a = await silo.createFolder('p', 'A'), b = await silo.createFolder('p', 'B', a.id)
    await silo.putDesign('p', 'd1', { doc: doc('Wing'), parent: b.id })
    await silo.putAsset('p', 'as1', { filename: 'f.dxf', mime: 'image/vnd.dxf', bytes: new Uint8Array([1, 2, 3]), parent: b.id })
    await fails(silo.renameNode('p', a.id, { parent: b.id }), 400)
    await silo.renameNode('p', b.id, { name: 'Bee', parent: null })
    expect((await silo.manifest('p')).nodes.find((n) => n.id === b.id)).toMatchObject({ name: 'Bee', parent: null })
    const gone = await silo.deleteNode('p', b.id)
    expect(gone.sort()).toEqual([b.id, 'as1', 'd1'].sort())
    expect((await silo.manifest('p')).nodes.map((n) => n.id)).toEqual([a.id])
    await fails(silo.getDesign('p', 'd1'), 404)
    expect(await readdir(join(dir, 'silos', 'p', 'assets'))).toEqual([])
  })

  it('keeps names unique within a folder', async () => {
    await silo.createProject('P', 'p')
    await silo.createFolder('p', 'Parts'); const second = await silo.createFolder('p', 'parts')
    expect(second.name).toBe('parts 2')
  })

  it('a design must live in a folder, not in a design', async () => {
    await silo.createProject('P', 'p')
    await silo.putDesign('p', 'd1', { doc: doc() })
    await fails(silo.putDesign('p', 'd2', { doc: doc(), parent: 'd1' }), 404)
  })

  it('serialises concurrent manifest edits without losing any', async () => {
    await silo.createProject('P', 'p')
    await Promise.all(Array.from({ length: 12 }, (_, i) => silo.createFolder('p', `F${i}`)))
    expect((await silo.manifest('p')).nodes).toHaveLength(12)
  })
})

describe('designs and revisions', () => {
  beforeEach(async () => { await silo.createProject('P', 'p') })

  it('creates with rev 1 and registers the design in the tree', async () => {
    const d = await silo.putDesign('p', 'd1', { doc: doc('Wing'), marker: 0 })
    expect(d.rev).toBe(1)
    expect((await silo.manifest('p')).nodes[0]).toMatchObject({ id: 'd1', kind: 'design', name: 'Wing', mode: 'part', rev: 1 })
    expect(await silo.getDesign('p', 'd1')).toMatchObject({ rev: 1, marker: 0, doc: { name: 'Wing' } })
  })

  it('requires If-Match to update, and rejects a stale revision with the current one', async () => {
    await silo.putDesign('p', 'd1', { doc: doc() })
    await fails(silo.putDesign('p', 'd1', { doc: doc('x') }), 428)
    const v2 = await silo.putDesign('p', 'd1', { doc: doc('x') }, 1)
    expect(v2.rev).toBe(2)
    try { await silo.putDesign('p', 'd1', { doc: doc('stale') }, 1) } catch (e) {
      expect((e as HttpError).status).toBe(409); expect((e as HttpError).body).toMatchObject({ rev: 2 })
    }
    expect((await silo.getDesign('p', 'd1')).doc.name).toBe('x')
    expect((await silo.putDesign('p', 'd1', { doc: doc('forced') }, '*')).rev).toBe(3)
  })

  it('renames the tree node when the document is renamed', async () => {
    await silo.putDesign('p', 'd1', { doc: doc('Old') })
    await silo.putDesign('p', 'd1', { doc: doc('New') }, 1)
    expect((await silo.manifest('p')).nodes[0].name).toBe('New')
  })

  it('rejects things that are not documents', async () => {
    await fails(silo.putDesign('p', 'd1', { doc: { hello: 1 } }), 400)
    await fails(silo.putDesign('p', '../x', { doc: doc() }), 400)
  })

  it('never exposes a half-written document to a reader', async () => {
    await silo.putDesign('p', 'd1', { doc: doc('a', 200) })
    const writes = Array.from({ length: 20 }, (_, i) => silo.putDesign('p', 'd1', { doc: doc(`v${i}`, 200) }, '*'))
    const reads = Array.from({ length: 40 }, () => silo.getDesign('p', 'd1'))
    const got = await Promise.all([...reads, ...writes])
    expect(got.length).toBe(60)
  })
})

describe('versions', () => {
  beforeEach(async () => { await silo.createProject('P', 'p'); await silo.putDesign('p', 'd1', { doc: doc('v1', 1), marker: 1 }) })

  it('saves named versions, lists newest first and reads them back', async () => {
    const a = await silo.createVersion('p', 'd1', 'First draft', 'note')
    clock += 5000
    await silo.putDesign('p', 'd1', { doc: doc('v2', 2), marker: 2 }, 1)
    clock += 5000
    const b = await silo.createVersion('p', 'd1', 'Second')
    const list = (await silo.listVersions('p', 'd1')).filter((v) => v.kind === 'manual')
    expect(list.map((v) => v.name)).toEqual(['Second', 'First draft'])
    expect(list[1]).toMatchObject({ note: 'note', kind: 'manual', rev: 1 })
    expect((await silo.getVersion('p', 'd1', a.id)).doc.name).toBe('v1')
    expect((await silo.getVersion('p', 'd1', b.id)).doc.features).toHaveLength(2)
    await fails(silo.getVersion('p', 'd1', 'nope'), 404)
  })

  it('restores a version as a new revision and keeps the version', async () => {
    const a = await silo.createVersion('p', 'd1', 'keep')
    await silo.putDesign('p', 'd1', { doc: doc('changed', 3), marker: 3 }, 1)
    const d = await silo.restoreVersion('p', 'd1', a.id)
    expect(d).toMatchObject({ rev: 3, marker: 1, doc: { name: 'v1' } })
    expect((await silo.listVersions('p', 'd1')).filter((v) => v.kind === 'manual')).toHaveLength(1)
  })

  it('takes an automatic checkpoint at most every 10 minutes while editing, and prunes to 20', async () => {
    let rev = 1
    const save = async () => { rev = (await silo.putDesign('p', 'd1', { doc: doc(`r${rev}`) }, rev)).rev }
    await save()                      // rev 2: first checkpoint
    expect((await silo.listVersions('p', 'd1')).filter((v) => v.kind === 'auto')).toHaveLength(1)
    clock += 60_000; await save()     // a minute later: none
    expect((await silo.listVersions('p', 'd1')).filter((v) => v.kind === 'auto')).toHaveLength(1)
    for (let i = 0; i < 30; i++) { clock += 11 * 60_000; await save() }
    const autos = (await silo.listVersions('p', 'd1')).filter((v) => v.kind === 'auto')
    expect(autos).toHaveLength(20)
    expect(autos[0].created).toBeGreaterThan(autos[19].created)
  })

  it('does not prune manual versions', async () => {
    for (let i = 0; i < 25; i++) { clock += 1000; await silo.createVersion('p', 'd1', `m${i}`) }
    expect((await silo.listVersions('p', 'd1')).filter((v) => v.kind === 'manual')).toHaveLength(25)
  })
})

describe('assets and thumbnails', () => {
  beforeEach(async () => { await silo.createProject('P', 'p') })

  it('round-trips bytes and metadata', async () => {
    const bytes = new Uint8Array([0, 1, 2, 250, 251, 255])
    const n = await silo.putAsset('p', 'a1', { filename: '../bracket.step', mime: 'application/step', bytes })
    expect(n).toMatchObject({ kind: 'asset', name: 'bracket.step', filename: 'bracket.step', size: 6 })
    const got = await silo.getAsset('p', 'a1')
    expect([...got.bytes]).toEqual([...bytes])
    expect(await readdir(join(dir, 'silos', 'p', 'assets', 'a1'))).toEqual(['bracket.step'])
  })

  it('replacing an asset keeps one node and one file', async () => {
    await silo.putAsset('p', 'a1', { filename: 'a.png', mime: 'image/png', bytes: new Uint8Array([1]) })
    await silo.putAsset('p', 'a1', { filename: 'b.png', mime: 'image/png', bytes: new Uint8Array([2, 3]) })
    expect((await silo.manifest('p')).nodes).toHaveLength(1)
    expect(await readdir(join(dir, 'silos', 'p', 'assets', 'a1'))).toEqual(['b.png'])
  })

  it('stores PNG thumbnails only', async () => {
    await silo.putDesign('p', 'd1', { doc: doc() })
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2])
    await silo.putThumb('p', 'd1', png)
    expect([...(await silo.getThumb('p', 'd1'))]).toEqual([...png])
    await fails(silo.putThumb('p', 'd1', new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9])), 400)
    await fails(silo.getThumb('p', 'nope'), 404)
  })
})

void writeFile
