import { randomBytes } from 'node:crypto'
import { mkdir, readFile, readdir, rename, rm, stat, writeFile, open } from 'node:fs/promises'
import { join, basename } from 'node:path'

/**
 * Storage for DarkCAD projects. Every project is a "silo": one self-contained directory.
 *
 *   silos/<projectId>/manifest.json                          the folder / design / asset tree
 *   silos/<projectId>/designs/<designId>/doc.json            current document (+ rev, marker)
 *   silos/<projectId>/designs/<designId>/versions/<id>.json  named versions and automatic checkpoints
 *   silos/<projectId>/designs/<designId>/thumb.png
 *   silos/<projectId>/assets/<assetId>/<filename>
 *
 * Writes are atomic (temp file in the same directory, then rename), and every document carries a `rev` that must
 * match for an update to be accepted, so a stale tab can never overwrite newer data.
 */

export type NodeKind = 'folder' | 'design' | 'asset'
export interface TreeNode {
  id: string
  kind: NodeKind
  name: string
  parent: string | null
  /** designs only */
  mode?: 'part' | 'assembly' | 'hybrid'
  /** designs only: revision of the stored document, so clients can tell whether their copy is current */
  rev?: number
  /** assets only */
  mime?: string
  size?: number
  filename?: string
  updated: number
}
export interface Manifest { id: string; name: string; created: number; updated: number; rev: number; nodes: TreeNode[] }
export interface StoredDesign { rev: number; updated: number; marker: number; doc: any }
export interface VersionMeta { id: string; name: string; note: string; kind: 'manual' | 'auto'; created: number; rev: number }
export interface StoredVersion extends VersionMeta { marker: number; doc: any }

export class HttpError extends Error {
  constructor(public status: number, message: string, public body?: unknown) { super(message) }
}

const ID = /^[A-Za-z0-9_-]{1,64}$/
export function checkId(id: string, what = 'id'): string {
  if (!ID.test(id)) throw new HttpError(400, `invalid ${what}`)
  return id
}
export const newId = (prefix: string) => `${prefix}${Date.now().toString(36)}${randomBytes(3).toString('hex')}`

/** Keep only the final path segment and drop anything that is not safe in a file name. */
export function safeFilename(name: string): string {
  const base = basename(name.replace(/\\/g, '/')).replace(/[\u0000-\u001f<>:"|?*]/g, '_').replace(/^\.+/, '_').trim()
  return (base || 'file').slice(0, 200)
}

/** Write a file so readers see either the old content or the new content, never a partial file. */
export async function atomicWrite(path: string, data: string | Uint8Array) {
  const tmp = `${path}.${randomBytes(4).toString('hex')}.tmp`
  const fh = await open(tmp, 'w')
  try { await fh.writeFile(data); await fh.sync() } finally { await fh.close() }
  try { await rename(tmp, path) } catch (e) { await rm(tmp, { force: true }); throw e }
}

const exists = (p: string) => stat(p).then(() => true, () => false)
async function readJson<T>(p: string): Promise<T | undefined> {
  try { return JSON.parse(await readFile(p, 'utf8')) as T } catch (e: any) { if (e?.code === 'ENOENT') return undefined; throw e }
}

const AUTO_EVERY_MS = 10 * 60_000
const AUTO_KEEP = 20

export class Silo {
  constructor(readonly root: string, private now: () => number = Date.now) {}

  private silos = () => join(this.root, 'silos')
  private dir = (p: string) => join(this.silos(), checkId(p, 'project id'))
  private designDir = (p: string, d: string) => join(this.dir(p), 'designs', checkId(d, 'design id'))

  // One writer at a time per project: manifest edits and revision checks are read-modify-write.
  private locks = new Map<string, Promise<unknown>>()
  async locked<T>(key: string, fn: () => Promise<T>): Promise<T> {
    const prev = this.locks.get(key) ?? Promise.resolve()
    const run = prev.then(fn, fn)
    const tail = run.catch(() => undefined)
    this.locks.set(key, tail)
    try { return await run } finally { if (this.locks.get(key) === tail) this.locks.delete(key) }
  }

  // ---------------------------------------------------------------- projects
  async listProjects(): Promise<Pick<Manifest, 'id' | 'name' | 'created' | 'updated'>[]> {
    let ids: string[] = []
    try { ids = await readdir(this.silos()) } catch (e: any) { if (e?.code !== 'ENOENT') throw e }
    const out = []
    for (const id of ids) {
      const m = await readJson<Manifest>(join(this.silos(), id, 'manifest.json'))
      if (m) out.push({ id: m.id, name: m.name, created: m.created, updated: m.updated })
    }
    return out.sort((a, b) => a.created - b.created)
  }

  async createProject(name: string, id = newId('p')): Promise<Manifest> {
    checkId(id, 'project id')
    return this.locked(id, async () => {
      if (await exists(join(this.dir(id), 'manifest.json'))) throw new HttpError(409, 'project already exists')
      const t = this.now()
      const m: Manifest = { id, name: name.trim() || 'Project', created: t, updated: t, rev: 1, nodes: [] }
      await mkdir(join(this.dir(id), 'designs'), { recursive: true })
      await mkdir(join(this.dir(id), 'assets'), { recursive: true })
      await atomicWrite(join(this.dir(id), 'manifest.json'), JSON.stringify(m, null, 2))
      return m
    })
  }

  async manifest(p: string): Promise<Manifest> {
    const m = await readJson<Manifest>(join(this.dir(p), 'manifest.json'))
    if (!m) throw new HttpError(404, 'project not found')
    return m
  }

  private async edit<T>(p: string, fn: (m: Manifest) => T | Promise<T>): Promise<T> {
    return this.locked(p, async () => {
      const m = await this.manifest(p)
      const out = await fn(m)
      m.rev++; m.updated = this.now()
      await atomicWrite(join(this.dir(p), 'manifest.json'), JSON.stringify(m, null, 2))
      return out
    })
  }

  async renameProject(p: string, name: string) { return this.edit(p, (m) => { m.name = name.trim() || m.name; return m }) }

  // ---------------------------------------------------------------- tree
  private node(m: Manifest, id: string, kind?: NodeKind): TreeNode {
    const n = m.nodes.find((x) => x.id === id)
    if (!n || (kind && n.kind !== kind)) throw new HttpError(404, `${kind ?? 'node'} not found`)
    return n
  }
  private checkParent(m: Manifest, parent: string | null | undefined) {
    if (parent == null) return null
    this.node(m, parent, 'folder')
    return parent
  }
  /** Names are unique within a folder so the browser never shows two identical rows. */
  private unique(m: Manifest, name: string, parent: string | null, skip?: string) {
    const taken = new Set(m.nodes.filter((n) => n.parent === parent && n.id !== skip).map((n) => n.name.toLowerCase()))
    if (!taken.has(name.toLowerCase())) return name
    const dot = name.lastIndexOf('.'), stem = dot > 0 ? name.slice(0, dot) : name, ext = dot > 0 ? name.slice(dot) : ''
    for (let i = 2; ; i++) if (!taken.has(`${stem} ${i}${ext}`.toLowerCase())) return `${stem} ${i}${ext}`
  }

  async createFolder(p: string, name: string, parent: string | null = null, id = newId('f')): Promise<TreeNode> {
    checkId(id)
    return this.edit(p, (m) => {
      if (m.nodes.some((n) => n.id === id)) throw new HttpError(409, 'id already in use')
      const par = this.checkParent(m, parent)
      const n: TreeNode = { id, kind: 'folder', name: this.unique(m, name.trim() || 'Folder', par), parent: par, updated: this.now() }
      m.nodes.push(n)
      return n
    })
  }

  async renameNode(p: string, id: string, patch: { name?: string; parent?: string | null }): Promise<TreeNode> {
    return this.edit(p, (m) => {
      const n = this.node(m, id)
      let parent = n.parent
      if (patch.parent !== undefined) {
        parent = this.checkParent(m, patch.parent)
        // A folder cannot move into itself or one of its own descendants.
        for (let c: string | null = parent; c; c = m.nodes.find((x) => x.id === c)?.parent ?? null) if (c === id) throw new HttpError(400, 'cannot move a folder into itself')
      }
      const name = (patch.name ?? n.name).trim() || n.name
      n.name = this.unique(m, name, parent, id); n.parent = parent; n.updated = this.now()
      return n
    })
  }

  async deleteNode(p: string, id: string): Promise<string[]> {
    return this.edit(p, async (m) => {
      this.node(m, id)
      const gone = new Set<string>([id])
      for (let grew = true; grew;) { grew = false; for (const n of m.nodes) if (n.parent && gone.has(n.parent) && !gone.has(n.id)) { gone.add(n.id); grew = true } }
      for (const n of m.nodes.filter((x) => gone.has(x.id))) {
        if (n.kind === 'design') await rm(this.designDir(p, n.id), { recursive: true, force: true })
        if (n.kind === 'asset') await rm(join(this.dir(p), 'assets', n.id), { recursive: true, force: true })
      }
      m.nodes = m.nodes.filter((n) => !gone.has(n.id))
      return [...gone]
    })
  }

  // ---------------------------------------------------------------- designs
  async getDesign(p: string, d: string): Promise<StoredDesign> {
    const s = await readJson<StoredDesign>(join(this.designDir(p, d), 'doc.json'))
    if (!s) throw new HttpError(404, 'design not found')
    return s
  }

  /**
   * Create or update a design. `ifMatch` is the revision the client last saw: `undefined` is only valid for a new
   * design, `'*'` forces the write. A wrong revision is a 409 carrying the current one.
   */
  async putDesign(p: string, d: string, input: { doc: any; marker?: number; parent?: string | null }, ifMatch?: number | '*'): Promise<StoredDesign> {
    validateDoc(input.doc)
    return this.locked(p, async () => {
      const m = await this.manifest(p)
      const dir = this.designDir(p, d)
      const cur = await readJson<StoredDesign>(join(dir, 'doc.json'))
      const t = this.now()
      if (cur) {
        if (ifMatch === undefined) throw new HttpError(428, 'If-Match is required to update a design')
        if (ifMatch !== '*' && ifMatch !== cur.rev) throw new HttpError(409, 'the design changed since you loaded it', { rev: cur.rev, updated: cur.updated })
      } else if (ifMatch !== undefined && ifMatch !== '*' && ifMatch !== 0) throw new HttpError(409, 'the design no longer exists', { rev: 0 })
      const next: StoredDesign = { rev: (cur?.rev ?? 0) + 1, updated: t, marker: Math.max(0, Math.round(input.marker ?? input.doc.features.length)), doc: input.doc }
      await mkdir(join(dir, 'versions'), { recursive: true })
      await atomicWrite(join(dir, 'doc.json'), JSON.stringify(next))
      // keep the tree in step with the document
      let n = m.nodes.find((x) => x.id === d)
      if (n && n.kind !== 'design') throw new HttpError(409, 'id belongs to another kind of item')
      const name = String(input.doc.name || 'Untitled')
      if (!n) {
        const par = this.checkParent(m, input.parent)
        n = { id: d, kind: 'design', name: this.unique(m, name, par), parent: par, mode: input.doc.mode, rev: next.rev, updated: t }
        m.nodes.push(n)
      } else {
        if (n.name !== name) n.name = this.unique(m, name, n.parent, d)
        n.mode = input.doc.mode; n.rev = next.rev; n.updated = t
      }
      m.rev++; m.updated = t
      await atomicWrite(join(this.dir(p), 'manifest.json'), JSON.stringify(m, null, 2))
      await this.autoCheckpoint(p, d, next)
      return next
    })
  }

  // ---------------------------------------------------------------- versions
  private versionsDir = (p: string, d: string) => join(this.designDir(p, d), 'versions')

  async listVersions(p: string, d: string): Promise<VersionMeta[]> {
    let files: string[] = []
    try { files = await readdir(this.versionsDir(p, d)) } catch (e: any) { if (e?.code !== 'ENOENT') throw e }
    const out: VersionMeta[] = []
    for (const f of files.filter((x) => x.endsWith('.json'))) {
      const v = await readJson<StoredVersion>(join(this.versionsDir(p, d), f))
      if (v) out.push({ id: v.id, name: v.name, note: v.note, kind: v.kind, created: v.created, rev: v.rev })
    }
    return out.sort((a, b) => b.created - a.created || (a.id < b.id ? 1 : -1))
  }

  private async writeVersion(p: string, d: string, name: string, note: string, kind: 'manual' | 'auto', cur: StoredDesign): Promise<VersionMeta> {
    const created = this.now()
    const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40) || 'version'
    const id = `${String(created).padStart(13, '0')}-${slug}`
    const v: StoredVersion = { id, name, note, kind, created, rev: cur.rev, marker: cur.marker, doc: cur.doc }
    await mkdir(this.versionsDir(p, d), { recursive: true })
    await atomicWrite(join(this.versionsDir(p, d), `${id}.json`), JSON.stringify(v))
    return { id, name, note, kind, created, rev: cur.rev }
  }

  async createVersion(p: string, d: string, name: string, note = ''): Promise<VersionMeta> {
    return this.locked(p, async () => this.writeVersion(p, d, name.trim() || 'Version', note, 'manual', await this.getDesign(p, d)))
  }

  /** At most one "Autosave" checkpoint every 10 minutes, and only the newest 20 are kept. Runs inside the project lock. */
  private async autoCheckpoint(p: string, d: string, cur: StoredDesign) {
    const autos = (await this.listVersions(p, d)).filter((v) => v.kind === 'auto')
    // The very first save of a design has no history worth keeping; after that, one checkpoint per interval.
    if (cur.rev > 1 && (!autos.length || this.now() - autos[0].created >= AUTO_EVERY_MS)) await this.writeVersion(p, d, 'Autosave', '', 'auto', cur)
    const all = (await this.listVersions(p, d)).filter((v) => v.kind === 'auto')
    for (const old of all.slice(AUTO_KEEP)) await rm(join(this.versionsDir(p, d), `${old.id}.json`), { force: true })
  }

  async getVersion(p: string, d: string, v: string): Promise<StoredVersion> {
    const s = await readJson<StoredVersion>(join(this.versionsDir(p, d), `${checkId(v, 'version id')}.json`))
    if (!s) throw new HttpError(404, 'version not found')
    return s
  }

  /** Make a saved version the current document (a new revision; the version itself stays). */
  async restoreVersion(p: string, d: string, v: string): Promise<StoredDesign> {
    const ver = await this.getVersion(p, d, v)
    const cur = await this.getDesign(p, d)
    return this.putDesign(p, d, { doc: ver.doc, marker: ver.marker }, cur.rev)
  }

  // ---------------------------------------------------------------- thumbnails
  async putThumb(p: string, d: string, png: Uint8Array) {
    await this.manifest(p)
    if (!(png.length > 8 && png[0] === 0x89 && png[1] === 0x50 && png[2] === 0x4e && png[3] === 0x47)) throw new HttpError(400, 'thumbnail must be a PNG')
    const dir = this.designDir(p, d)
    if (!(await exists(join(dir, 'doc.json')))) throw new HttpError(404, 'design not found')
    await atomicWrite(join(dir, 'thumb.png'), png)
  }
  async getThumb(p: string, d: string): Promise<Buffer> {
    try { return await readFile(join(this.designDir(p, d), 'thumb.png')) } catch (e: any) { if (e?.code === 'ENOENT') throw new HttpError(404, 'no thumbnail'); throw e }
  }

  // ---------------------------------------------------------------- assets
  async putAsset(p: string, id: string, input: { filename: string; mime: string; bytes: Uint8Array; parent?: string | null }): Promise<TreeNode> {
    checkId(id, 'asset id')
    const filename = safeFilename(input.filename)
    return this.edit(p, async (m) => {
      const par = this.checkParent(m, input.parent)
      const existing = m.nodes.find((x) => x.id === id)
      if (existing && existing.kind !== 'asset') throw new HttpError(409, 'id belongs to another kind of item')
      const dir = join(this.dir(p), 'assets', id)
      await rm(dir, { recursive: true, force: true })
      await mkdir(dir, { recursive: true })
      await atomicWrite(join(dir, filename), input.bytes)
      const t = this.now()
      if (existing) Object.assign(existing, { mime: input.mime, size: input.bytes.length, filename, updated: t })
      const n = existing ?? { id, kind: 'asset' as const, name: this.unique(m, filename, par), parent: par, mime: input.mime, size: input.bytes.length, filename, updated: t }
      if (!existing) m.nodes.push(n)
      return n
    })
  }
  async getAsset(p: string, id: string): Promise<{ node: TreeNode; bytes: Buffer }> {
    const m = await this.manifest(p)
    const node = this.node(m, checkId(id, 'asset id'), 'asset')
    return { node, bytes: await readFile(join(this.dir(p), 'assets', id, node.filename!)) }
  }
}

function validateDoc(doc: any) {
  if (!doc || typeof doc !== 'object' || typeof doc.version !== 'number' || !Array.isArray(doc.features)) throw new HttpError(400, 'body.doc is not a DarkCAD document')
}
