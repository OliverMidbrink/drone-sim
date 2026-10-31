import { Store, emptyDocument, parse, serialize, type Document, type DocStore, type Feature } from '@darkcad/core'

export interface DesignMeta { id: string; name: string; updated: number; features: number }
interface Index { open: string[]; active: string; library: DesignMeta[] }

const INDEX_KEY = 'darkcad:workspace:v1'
const LEGACY_KEY = 'darkcad:autosave'
const docKey = (id: string) => `darkcad:design:${id}`

function safeStorage(): Storage | undefined {
  try { return globalThis.localStorage } catch { return undefined }
}

/**
 * Several designs open as tabs, plus a library of every saved design. Each design has its own Store (and
 * therefore its own undo history). The workspace itself implements the document-store surface by delegating
 * to the active design, so UI code keeps calling `store.add(...)` without caring which tab is current.
 */
export class Workspace implements DocStore {
  private stores = new Map<string, Store>()
  private index: Index
  private listeners = new Set<() => void>()
  private detach?: () => void
  private dirty = new Set<string>()
  private timer?: ReturnType<typeof setTimeout>

  constructor(private storage: Storage | undefined = safeStorage()) {
    this.index = this.load()
    this.attach()
    if (typeof window !== 'undefined') window.addEventListener('pagehide', () => this.flush())
  }

  // ---------------------------------------------------------------- persistence
  private load(): Index {
    try {
      const raw = this.storage?.getItem(INDEX_KEY)
      if (raw) {
        const idx = JSON.parse(raw) as Index
        const known = new Set(idx.library.map((m) => m.id))
        idx.open = idx.open.filter((id) => known.has(id))
        if (!idx.open.length) idx.open = idx.library.length ? [idx.library[0].id] : []
        if (idx.open.length) { if (!idx.open.includes(idx.active)) idx.active = idx.open[0]; return idx }
      }
    } catch { /* corrupt index: fall through to a fresh workspace */ }
    const id = this.newId()
    let doc = emptyDocument('Untitled')
    try { const legacy = this.storage?.getItem(LEGACY_KEY); if (legacy) doc = parse(legacy) } catch { /* ignore unreadable legacy autosave */ }
    this.stores.set(id, new Store(doc))
    const idx: Index = { open: [id], active: id, library: [{ id, name: doc.name, updated: Date.now(), features: doc.features.length }] }
    this.index = idx
    this.dirty.add(id)
    this.flush()
    return idx
  }

  private newId() { return `d${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}` }

  private storeFor(id: string): Store {
    let s = this.stores.get(id)
    if (!s) {
      let doc: Document | undefined
      try { const raw = this.storage?.getItem(docKey(id)); if (raw) doc = parse(raw) } catch { /* unreadable design: open it empty rather than crash */ }
      s = new Store(doc ?? emptyDocument(this.index.library.find((m) => m.id === id)?.name ?? 'Untitled'))
      this.stores.set(id, s)
    }
    return s
  }

  private touch(id: string) {
    const s = this.stores.get(id)
    const m = this.index.library.find((x) => x.id === id)
    if (!s || !m) return
    m.name = s.doc.name; m.features = s.doc.features.length; m.updated = Date.now()
    this.dirty.add(id)
    clearTimeout(this.timer)
    this.timer = setTimeout(() => this.flush(), 250)
  }

  flush() {
    clearTimeout(this.timer)
    try {
      for (const id of this.dirty) { const s = this.stores.get(id); if (s) this.storage?.setItem(docKey(id), serialize(s.doc)) }
      this.storage?.setItem(INDEX_KEY, JSON.stringify(this.index))
    } catch { /* quota or private mode: designs stay in memory for this session */ }
    this.dirty.clear()
  }

  // ---------------------------------------------------------------- tabs and library
  get activeId() { return this.index.active }
  private get current(): Store { return this.storeFor(this.index.active) }

  private meta(id: string): DesignMeta {
    const m = this.index.library.find((x) => x.id === id)!
    const s = this.stores.get(id)
    return s ? { ...m, name: s.doc.name, features: s.doc.features.length } : { ...m }
  }
  tabs(): DesignMeta[] { return this.index.open.map((id) => this.meta(id)) }
  library(): DesignMeta[] { return this.index.library.map((m) => this.meta(m.id)).sort((a, b) => b.updated - a.updated) }

  uniqueName(base = 'Untitled'): string {
    const names = new Set(this.library().map((m) => m.name))
    if (!names.has(base)) return base
    for (let n = 2; ; n++) if (!names.has(`${base} ${n}`)) return `${base} ${n}`
  }

  /** New design in a new tab (optionally from a document, e.g. a file the user opened). */
  create(doc?: Document): string {
    const id = this.newId()
    const d = doc ?? emptyDocument(this.uniqueName('Untitled'))
    this.stores.set(id, new Store(d))
    this.index.library.push({ id, name: d.name, updated: Date.now(), features: d.features.length })
    this.index.open.push(id)
    this.dirty.add(id)
    this.activate(id)
    return id
  }

  /** Open a library design in a tab (or just switch to it if it already is one). */
  open(id: string) {
    if (!this.index.library.some((m) => m.id === id)) return
    if (!this.index.open.includes(id)) this.index.open.push(id)
    this.activate(id)
  }

  activate(id: string) {
    if (!this.index.open.includes(id)) return
    this.index.active = id
    this.switchAttached()
    this.dirty.add(id)
    this.flush()
    this.emit()
  }

  /** Close a tab; the design stays in the library. Never leaves the workspace without a tab. */
  close(id: string) {
    const i = this.index.open.indexOf(id)
    if (i < 0) return
    this.touch(id); this.flush()
    this.index.open.splice(i, 1)
    if (!this.index.open.length) { this.create(); return }
    if (this.index.active === id) this.activate(this.index.open[Math.min(i, this.index.open.length - 1)])
    else { this.flush(); this.emit() }
  }

  /** Delete a design for good. */
  deleteDesign(id: string) {
    this.index.library = this.index.library.filter((m) => m.id !== id)
    this.stores.delete(id)
    this.dirty.delete(id)
    try { this.storage?.removeItem(docKey(id)) } catch { /* nothing to clean up */ }
    const wasActive = this.index.active === id
    this.index.open = this.index.open.filter((x) => x !== id)
    if (!this.index.open.length) { this.create(); return }
    if (wasActive) this.activate(this.index.open[0])
    else { this.flush(); this.emit() }
  }

  private attach() {
    this.detach?.()
    const id = this.index.active
    this.detach = this.current.subscribe(() => { this.touch(id); this.emit() })
  }
  private switchAttached() { this.attach() }

  // ---------------------------------------------------------------- DocStore (delegates to the active design)
  subscribe(fn: () => void) { this.listeners.add(fn); return () => { this.listeners.delete(fn) } }
  private emit() { this.listeners.forEach((f) => f()) }
  get doc(): Document { return this.current.doc }
  set doc(d: Document) { this.current.doc = d }
  nextId(prefix: string) { return this.current.nextId(prefix) }
  add(f: Feature, at?: number) { this.current.add(f, at) }
  update<T extends Feature>(id: string, patch: Partial<T>, key?: string) { this.current.update<T>(id, patch, key) }
  remove(id: string) { this.current.remove(id) }
  move(id: string, to: number) { this.current.move(id, to) }
  rename(name: string) { this.current.rename(name) }
  get canUndo() { return this.current.canUndo }
  get canRedo() { return this.current.canRedo }
  undo() { this.current.undo() }
  redo() { this.current.redo() }
  replace(doc: Document) { this.current.replace(doc) }
}
