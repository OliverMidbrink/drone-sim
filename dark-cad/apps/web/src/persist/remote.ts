import { ConflictError, HttpError, type Manifest, type Payload, type ProjectInfo, type RemotePort, type StoredDesign, type TreeNode, type VersionMeta } from './types'

const PROJECT_KEY = 'darkcad:project'

/** REST client for apps/server. All paths are relative to the project chosen by `connect()`. */
export class RemotePersistence implements RemotePort {
  project = ''
  private assets = new Map<string, Uint8Array>()

  constructor(private base = '/api', private fetcher: typeof fetch = (...a) => fetch(...a)) {}

  private async req(method: string, path: string, init: { body?: BodyInit; json?: unknown; headers?: Record<string, string> } = {}): Promise<Response> {
    const headers: Record<string, string> = { ...init.headers }
    let body = init.body
    if (init.json !== undefined) { body = JSON.stringify(init.json); headers['Content-Type'] = 'application/json' }
    const r = await this.fetcher(this.base + path, { method, headers, body })
    if (!r.ok) {
      let info: any = {}
      try { info = await r.json() } catch { /* not JSON */ }
      if (r.status === 409 && typeof info.rev === 'number') throw new ConflictError(info.rev)
      throw new HttpError(r.status, info.error ?? `${method} ${path}: ${r.status}`)
    }
    return r
  }
  private p(path = '') { return `/projects/${encodeURIComponent(this.project)}${path}` }

  /** Pick (or create) the project this browser works in. */
  async connect(): Promise<Manifest> {
    const list: ProjectInfo[] = await (await this.req('GET', '/projects')).json()
    let want: string | undefined
    try { want = localStorage.getItem(PROJECT_KEY) ?? undefined } catch { /* no storage */ }
    let proj = list.find((x) => x.id === want) ?? list[0]
    if (!proj) proj = await (await this.req('POST', '/projects', { json: { name: 'My Drive' } })).json()
    this.project = proj!.id
    try { localStorage.setItem(PROJECT_KEY, this.project) } catch { /* no storage */ }
    return this.manifest()
  }

  async manifest(): Promise<Manifest> { return (await this.req('GET', this.p())).json() }
  async getDesign(id: string): Promise<StoredDesign | undefined> {
    try { return await (await this.req('GET', this.p(`/designs/${encodeURIComponent(id)}`))).json() } catch (e) { if (e instanceof HttpError && e.status === 404) return undefined; throw e }
  }
  async putDesign(id: string, payload: Payload, ifMatch: number | '*' | undefined) {
    const headers: Record<string, string> = {}
    if (ifMatch !== undefined) headers['If-Match'] = `"${ifMatch}"`
    try {
      return await (await this.req('PUT', this.p(`/designs/${encodeURIComponent(id)}`), { json: payload, headers })).json() as { rev: number; updated: number }
    } catch (e) {
      // We never saw this design's revision but the server has it: same as a conflict.
      if (e instanceof HttpError && e.status === 428) { const cur = await this.getDesign(id); if (cur) throw new ConflictError(cur.rev) }
      throw e
    }
  }
  async createFolder(name: string, parent: string | null): Promise<TreeNode> { return (await this.req('POST', this.p('/folders'), { json: { name, parent } })).json() }
  async patchNode(id: string, patch: { name?: string; parent?: string | null }): Promise<TreeNode> { return (await this.req('PATCH', this.p(`/nodes/${encodeURIComponent(id)}`), { json: patch })).json() }
  async deleteNode(id: string) { await this.req('DELETE', this.p(`/nodes/${encodeURIComponent(id)}`)) }

  async listVersions(design: string): Promise<VersionMeta[]> { return (await this.req('GET', this.p(`/designs/${encodeURIComponent(design)}/versions`))).json() }
  async createVersion(design: string, name: string, note = ''): Promise<VersionMeta> { return (await this.req('POST', this.p(`/designs/${encodeURIComponent(design)}/versions`), { json: { name, note } })).json() }
  async getVersion(design: string, version: string): Promise<{ doc: any; marker: number } & VersionMeta> { return (await this.req('GET', this.p(`/designs/${encodeURIComponent(design)}/versions/${encodeURIComponent(version)}`))).json() }

  async putThumb(design: string, png: Blob) { await this.req('PUT', this.p(`/designs/${encodeURIComponent(design)}/thumb`), { body: png, headers: { 'Content-Type': 'image/png' } }) }
  thumbUrl(design: string, v?: number) { return `${this.base}${this.p(`/designs/${encodeURIComponent(design)}/thumb`)}${v ? `?v=${v}` : ''}` }
  assetUrl(id: string) { return `${this.base}${this.p(`/assets/${encodeURIComponent(id)}`)}` }

  async putAsset(id: string, file: { name: string; type: string; bytes: Uint8Array }, parent: string | null): Promise<TreeNode> {
    const q = new URLSearchParams({ name: file.name }); if (parent) q.set('parent', parent)
    const node: TreeNode = await (await this.req('PUT', this.p(`/assets/${encodeURIComponent(id)}?${q}`), { body: file.bytes as BodyInit, headers: { 'Content-Type': file.type || 'application/octet-stream' } })).json()
    this.assets.set(id, file.bytes)
    return node
  }
  /** Bytes of an uploaded file; cached for the session. */
  async assetBytes(id: string): Promise<Uint8Array> {
    let b = this.assets.get(id)
    if (!b) { b = new Uint8Array(await (await this.req('GET', this.p(`/assets/${encodeURIComponent(id)}`))).arrayBuffer()); this.assets.set(id, b) }
    return b
  }
}
