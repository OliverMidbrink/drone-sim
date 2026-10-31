import { Store, emptyDocument, parse, type DesignMode, type Document, type DocStore, type Feature } from '@darkcad/core'
import { LocalPersistence, type DesignMeta, type Index } from './persist/local'
import { RemotePersistence } from './persist/remote'
import { Syncer } from './persist/sync'
import type { Manifest, Payload, SaveState, TreeNode } from './persist/types'

export type { DesignMeta }

/**
 * Several designs open as tabs, plus a library of every saved design. Each design has its own Store (and
 * therefore its own undo history). The workspace itself implements the document-store surface by delegating
 * to the active design, so UI code keeps calling `store.add(...)` without caring which tab is current.
 *
 * Persistence is two-layered: every change goes to localStorage at once (the offline cache), and, when a server is
 * attached with `connect()`, is queued for a debounced, revision-checked upload.
 */
export class Workspace implements DocStore {
  private stores = new Map<string, Store>()
  /** Read-only copies of designs that are not open (components of another design, remote designs). */
  private spare = new Map<string, Document>()
  private index: Index
  private listeners = new Set<() => void>()
  private detach?: () => void
  private dirty = new Set<string>()
  private timer?: ReturnType<typeof setTimeout>
  readonly local: LocalPersistence
  remote?: RemotePersistence
  syncer?: Syncer
  private reconnect?: ReturnType<typeof setTimeout>
  private lastPull = 0
  /** The project tree on the server, as of the last pull. */
  manifest?: Manifest

  constructor(storage?: Storage | LocalPersistence) {
    this.local = storage instanceof LocalPersistence ? storage : new LocalPersistence(storage)
    this.index = this.loadIndex()
    this.attach()
    if (typeof window !== 'undefined') window.addEventListener('pagehide', () => this.flush())
  }

  // ---------------------------------------------------------------- persistence
  private loadIndex(): Index {
    const raw = this.local.readIndex()
    if (raw && Array.isArray(raw.library)) {
      for (const m of raw.library) if (typeof m.name !== 'string' || !m.name.trim()) m.name = 'Untitled'
      const known = new Set(raw.library.map((m) => m.id))
      raw.open = (raw.open ?? []).filter((id) => known.has(id))
      if (!raw.open.length) raw.open = raw.library.length ? [raw.library[0].id] : []
      if (raw.open.length) { if (!raw.open.includes(raw.active)) raw.active = raw.open[0]; return raw }
    }
    const id = this.newId()
    const doc = this.local.readLegacy() ?? emptyDocument('Untitled')
    this.stores.set(id, new Store(doc))
    const idx: Index = { open: [id], active: id, library: [{ id, name: doc.name, updated: Date.now(), features: doc.features.length, mode: doc.mode, marker: doc.features.length }] }
    this.index = idx
    this.dirty.add(id)
    this.flush()
    return idx
  }

  private newId() { return `d${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}` }
  private metaOf(id: string) { return this.index.library.find((m) => m.id === id) }

  private storeFor(id: string): Store {
    let s = this.stores.get(id)
    if (!s) {
      const m = this.metaOf(id)
      const doc = this.spare.get(id) ?? this.local.readDoc(id) ?? emptyDocument(m?.name ?? 'Untitled')
      this.spare.delete(id)
      s = new Store(doc, m?.marker)
      this.stores.set(id, s)
    }
    return s
  }

  private touch(id: string) {
    const s = this.stores.get(id)
    const m = this.metaOf(id)
    if (!s || !m) return
    m.name = s.doc.name; m.features = s.doc.features.length; m.mode = s.doc.mode; m.marker = s.marker; m.updated = Date.now()
    if (this.syncer) { m.unsynced = true; this.syncer.markDirty(id) }
    this.dirty.add(id)
    clearTimeout(this.timer)
    this.timer = setTimeout(() => this.flush(), 250)
  }

  flush() {
    clearTimeout(this.timer)
    try {
      for (const id of this.dirty) { const s = this.stores.get(id); if (s) this.local.writeDoc(id, s.doc) }
      this.local.writeIndex(this.index)
    } catch { /* quota or private mode: designs stay in memory for this session */ }
    this.dirty.clear()
  }

  // ---------------------------------------------------------------- server
  /** Attach the server: reconcile the library with the project tree, then keep saving in the background. */
  async connect(remote: RemotePersistence, opts: { debounceMs?: number; backoffMs?: number[] } = {}) {
    this.remote = remote
    this.syncer = new Syncer(remote, (id) => this.payload(id), opts)
    this.syncer.subscribe(() => this.emit())
    this.syncer.onSynced = (id, rev) => {
      const m = this.metaOf(id)
      if (m) { m.rev = rev; if (!this.syncer!.isPending(id)) m.unsynced = false; this.local.writeIndex(this.index) }
      this.emit()
    }
    // Edits made before the server answered are still queued by `touch` through the syncer set above.
    await this.pull()
  }

  get saveState(): SaveState { return this.syncer?.state ?? 'local' }
  get conflicts(): { id: string; name: string }[] { return [...(this.syncer?.conflicts.keys() ?? [])].map((id) => ({ id, name: this.metaOf(id)?.name ?? id })) }

  private payload(id: string): Payload | undefined {
    const s = this.stores.get(id)
    const m = this.metaOf(id)
    if (!m) return undefined
    const doc = s?.doc ?? this.spare.get(id) ?? this.local.readDoc(id)
    return doc ? { doc, marker: s?.marker ?? m.marker ?? doc.features.length, parent: m.parent ?? null } : undefined
  }

  /** Download the project tree and bring the library in line with it. Safe to call repeatedly. */
  async pull(): Promise<void> {
    const r = this.remote, sy = this.syncer
    if (!r || !sy) return
    let m: Manifest
    try { m = await r.connect() } catch {
      sy.setOffline(true)
      clearTimeout(this.reconnect)
      this.reconnect = setTimeout(() => { void this.pull() }, 8000)
      return
    }
    this.lastPull = Date.now()
    this.manifest = m
    const designs = new Map(m.nodes.filter((n) => n.kind === 'design').map((n) => [n.id, n]))
    let changed = false
    for (const meta of this.index.library) if (meta.rev !== undefined) sy.setRev(meta.id, meta.rev)
    for (const n of designs.values()) {
      const meta = this.metaOf(n.id)
      if (!meta) {
        this.index.library.push({ id: n.id, name: n.name, updated: n.updated, features: -1, mode: n.mode, parent: n.parent, rev: n.rev, marker: undefined })
        changed = true
        continue
      }
      meta.parent = n.parent
      if (n.rev !== undefined && meta.rev !== n.rev) {
        // The server has a different revision than we last saw.
        if (meta.unsynced) { sy.markDirty(n.id) /* the upload will be refused (409) and parked as a conflict */ }
        else { await this.adopt(n.id); changed = true }
      } else if (meta.unsynced) sy.markDirty(n.id)
      else if (n.rev !== undefined && !this.local.readDoc(n.id)) { await this.adopt(n.id); changed = true } // browser storage lost the document: the server has it
    }
    for (const meta of [...this.index.library]) {
      if (designs.has(meta.id)) continue
      if (meta.rev !== undefined && !meta.unsynced) { this.removeLocal(meta.id); changed = true } // deleted elsewhere
      else if (meta.unsynced || meta.features > 0 || meta.rev !== undefined) { sy.setRev(meta.id, undefined); meta.rev = undefined; sy.markDirty(meta.id) }
      // an untouched, never-saved design stays local-only until its first edit
    }
    sy.setOffline(false)
    if (changed) { this.flush(); this.emit() }
    void sy.flush()
    clearTimeout(this.reconnect)
  }

  /** Re-read the project tree (folders, assets, designs) from the server. */
  async refreshManifest() {
    if (!this.remote?.project) return
    try { this.manifest = await this.remote.manifest(); this.emit() } catch { this.syncer?.setOffline(true) }
  }

  /** Rename any design in the library (not just the active one). */
  async renameDesign(id: string, name: string) {
    if (!(await this.ensureDoc(id))) return
    this.storeFor(id).rename(name)
    this.touch(id)
    this.flush(); this.emit()
  }

  /** Refresh from the server when the page regains focus (another tab or device may have saved). */
  maybePull() { if (this.remote && Date.now() - this.lastPull > 5000) void this.pull() }

  /** Replace the local copy of a design with the server's. */
  private async adopt(id: string) {
    const d = await this.remote!.getDesign(id)
    const meta = this.metaOf(id)
    if (!d || !meta) return
    const doc = parse(JSON.stringify(d.doc))
    this.local.writeDoc(id, doc)
    meta.rev = d.rev; meta.unsynced = false; meta.name = doc.name; meta.features = doc.features.length; meta.mode = doc.mode; meta.marker = d.marker; meta.updated = d.updated
    this.syncer?.setRev(id, d.rev)
    const s = this.stores.get(id)
    if (s) { s.replace(doc, d.marker); this.dirty.delete(id) }
    this.spare.delete(id)
  }

  /** Choose which copy wins after a conflict. */
  async resolveConflict(id: string, choice: 'mine' | 'theirs') {
    if (!this.syncer) return
    if (choice === 'theirs') {
      this.syncer.resolve(id, 'theirs')
      await this.adopt(id)
      this.emit()
    } else this.syncer.resolve(id, 'mine')
  }

  /** Make sure a design's document is available locally (downloads it if it only exists on the server). */
  async ensureDoc(id: string): Promise<Document | undefined> {
    const have = this.docOf(id)
    if (have) return have
    if (!this.remote || !this.metaOf(id)) return undefined
    try { await this.adopt(id) } catch { return undefined }
    return this.docOf(id)
  }

  /** The document of any design in the library: the live one when it is open. */
  docOf(id: string): Document | undefined {
    const s = this.stores.get(id)
    if (s) return s.doc
    const hit = this.spare.get(id)
    if (hit) return hit
    if (!this.metaOf(id)) return undefined
    const doc = this.local.readDoc(id)
    if (doc) this.spare.set(id, doc)
    return doc
  }

  // ---------------------------------------------------------------- tabs and library
  get activeId() { return this.index.active }
  private get current(): Store { return this.storeFor(this.index.active) }

  private meta(id: string): DesignMeta {
    const m = this.metaOf(id)!
    const s = this.stores.get(id)
    return s ? { ...m, name: s.doc.name, features: s.doc.features.length, mode: s.doc.mode } : { ...m }
  }
  tabs(): DesignMeta[] { return this.index.open.map((id) => this.meta(id)) }
  library(): DesignMeta[] { return this.index.library.map((m) => this.meta(m.id)).sort((a, b) => b.updated - a.updated) }
  design(id: string): DesignMeta | undefined { return this.metaOf(id) ? this.meta(id) : undefined }

  uniqueName(base = 'Untitled'): string {
    const names = new Set(this.library().map((m) => m.name))
    if (!names.has(base)) return base
    for (let n = 2; ; n++) if (!names.has(`${base} ${n}`)) return `${base} ${n}`
  }

  /** New design in a new tab (optionally from a document, e.g. a file the user opened). */
  create(doc?: Document, opts: { parent?: string | null; mode?: DesignMode; open?: boolean } = {}): string {
    const id = this.newId()
    const d = doc ?? emptyDocument(this.uniqueName('Untitled'), opts.mode ?? 'part')
    this.stores.set(id, new Store(d))
    this.index.library.push({ id, name: d.name, updated: Date.now(), features: d.features.length, mode: d.mode, parent: opts.parent ?? null, marker: d.features.length })
    this.dirty.add(id)
    if (opts.open === false) { this.flush(); this.touch(id); this.emit(); return id }
    this.index.open.push(id)
    this.touch(id)
    this.activate(id)
    return id
  }

  /** Open a library design in a tab (or just switch to it if it already is one). Downloads it first if needed. */
  async open(id: string): Promise<void> {
    if (!this.metaOf(id)) return
    if (!this.docOf(id)) { if (!(await this.ensureDoc(id))) return }
    if (!this.index.open.includes(id)) this.index.open.push(id)
    this.activate(id)
  }

  activate(id: string) {
    if (!this.index.open.includes(id)) return
    this.index.active = id
    this.attach()
    this.dirty.add(id)
    this.flush()
    this.emit()
  }

  /** Close a tab; the design stays in the library. Never leaves the workspace without a tab. */
  close(id: string) {
    const i = this.index.open.indexOf(id)
    if (i < 0) return
    this.flush()
    this.index.open.splice(i, 1)
    if (!this.index.open.length) { this.create(); return }
    if (this.index.active === id) this.activate(this.index.open[Math.min(i, this.index.open.length - 1)])
    else { this.flush(); this.emit() }
  }

  private removeLocal(id: string) {
    this.index.library = this.index.library.filter((m) => m.id !== id)
    this.stores.delete(id); this.spare.delete(id)
    this.dirty.delete(id)
    this.local.removeDoc(id)
    this.syncer?.drop(id)
    const wasActive = this.index.active === id
    this.index.open = this.index.open.filter((x) => x !== id)
    if (!this.index.open.length) { this.create(); return }
    if (wasActive) this.activate(this.index.open[0])
  }

  /** Delete a design for good (locally and on the server). */
  deleteDesign(id: string) {
    const synced = this.metaOf(id)?.rev !== undefined
    this.removeLocal(id)
    if (synced && this.remote) void this.remote.deleteNode(id).catch(() => { /* offline: it comes back on the next pull */ })
    this.flush(); this.emit()
  }

  /** Move a design into another folder of the project tree. */
  async moveDesign(id: string, parent: string | null) {
    const m = this.metaOf(id)
    if (!m) return
    m.parent = parent
    if (this.remote && m.rev !== undefined) await this.remote.patchNode(id, { parent })
    this.flush(); this.emit()
  }

  private attach() {
    this.detach?.()
    const id = this.index.active
    this.detach = this.current.subscribe(() => { this.touch(id); this.emit() })
  }

  // ---------------------------------------------------------------- DocStore (delegates to the active design)
  subscribe(fn: () => void) { this.listeners.add(fn); return () => { this.listeners.delete(fn) } }
  private emit() { this.listeners.forEach((f) => f()) }
  get doc(): Document { return this.current.doc }
  set doc(d: Document) { this.current.doc = d }
  get marker() { return this.current.marker }
  get atEnd() { return this.current.atEnd }
  nextId(prefix: string) { return this.current.nextId(prefix) }
  add(f: Feature, at?: number) { this.current.add(f, at) }
  update<T extends Feature>(id: string, patch: Partial<T>, key?: string) { this.current.update<T>(id, patch, key) }
  remove(id: string) { this.current.remove(id) }
  move(id: string, to: number) { this.current.move(id, to) }
  suppress(id: string, on = true) { this.current.suppress(id, on) }
  rename(name: string) { this.current.rename(name) }
  setMode(mode: DesignMode) { this.current.setMode(mode) }
  setBodyName(root: string, name: string) { this.current.setBodyName(root, name) }
  setBodyHidden(root: string, hidden: boolean) { this.current.setBodyHidden(root, hidden) }
  setMarker(n: number) { this.current.setMarker(n) }
  stepMarker(d: number) { this.current.stepMarker(d) }
  get canUndo() { return this.current.canUndo }
  get canRedo() { return this.current.canRedo }
  undo() { this.current.undo() }
  redo() { this.current.redo() }
  replace(doc: Document, marker?: number) { this.current.replace(doc, marker) }
  load(doc: Document, marker?: number) { this.current.load(doc, marker) }
}

export type { TreeNode }
