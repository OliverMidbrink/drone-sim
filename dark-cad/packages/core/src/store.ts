import { emptyDocument, type DesignMode, type Document, type Feature } from './model'

interface Snapshot { doc: Document; marker: number }

/**
 * Document + linear undo/redo history. Every edit replaces the document immutably.
 *
 * `marker` is the rollback bar of the timeline: features before it are active, features at or after it are rolled
 * back. It is view state, not part of the document, so moving it is not an undo step; undo and redo restore it
 * together with the document. New features are inserted at the marker (like Fusion), and a marker at the very end
 * stays at the end as features are added or removed.
 */
export class Store {
  doc: Document
  marker: number
  private past: Snapshot[] = []
  private future: Snapshot[] = []
  private listeners = new Set<() => void>()
  private counter = 0

  constructor(doc: Document = emptyDocument(), marker?: number) {
    this.doc = doc
    this.counter = doc.features.length
    this.marker = this.clamp(marker ?? doc.features.length, doc)
  }

  private clamp(m: number, doc = this.doc) { return Math.max(0, Math.min(Math.round(m), doc.features.length)) }
  get atEnd() { return this.marker >= this.doc.features.length }

  subscribe(fn: () => void) { this.listeners.add(fn); return () => { this.listeners.delete(fn) } }
  private emit() { this.listeners.forEach((f) => f()) }

  nextId(prefix: string): string {
    let id: string
    do id = `${prefix}${++this.counter}`; while (this.doc.features.some((f) => f.id === id))
    return id
  }

  /** `coalesceKey` merges consecutive edits with the same key into one undo step (e.g. a drag). */
  private lastKey?: string
  commit(next: Document, coalesceKey?: string, marker: number = this.marker) {
    if (!(coalesceKey && coalesceKey === this.lastKey)) this.past.push({ doc: this.doc, marker: this.marker })
    this.lastKey = coalesceKey
    this.doc = next
    this.marker = this.clamp(marker, next)
    this.future = []
    this.emit()
  }

  /** Inserts at `at` (default: the marker). */
  add(feature: Feature, at?: number) {
    const end = this.atEnd
    const i = Math.max(0, Math.min(at ?? this.marker, this.doc.features.length))
    const fs = [...this.doc.features]
    fs.splice(i, 0, feature)
    this.commit({ ...this.doc, features: fs }, undefined, end || i <= this.marker ? this.marker + 1 : this.marker)
  }

  update<T extends Feature>(id: string, patch: Partial<T>, coalesceKey?: string) {
    this.commit({ ...this.doc, features: this.doc.features.map((f) => (f.id === id ? ({ ...f, ...patch } as Feature) : f)) }, coalesceKey)
  }

  remove(id: string) {
    const i = this.doc.features.findIndex((f) => f.id === id)
    if (i < 0) return
    const end = this.atEnd
    const hidden = this.doc.hidden.filter((h) => h !== id)
    const { [id]: _gone, ...bodyNames } = this.doc.bodyNames
    this.commit({ ...this.doc, hidden, bodyNames, features: this.doc.features.filter((f) => f.id !== id) }, undefined, end || i < this.marker ? this.marker - 1 : this.marker)
  }

  /** Move a feature to index `to` (its position after it has been taken out of the list). */
  move(id: string, to: number) {
    const fs = [...this.doc.features]
    const i = fs.findIndex((f) => f.id === id)
    if (i < 0) return
    const end = this.atEnd
    const [f] = fs.splice(i, 1)
    const j = Math.max(0, Math.min(to, fs.length))
    fs.splice(j, 0, f)
    if (j === i) return
    this.commit({ ...this.doc, features: fs }, undefined, end ? fs.length : this.marker)
  }

  suppress(id: string, on = true) { this.update(id, { suppressed: on || undefined } as Partial<Feature>) }

  rename(name: string) { this.commit({ ...this.doc, name }) }
  setMode(mode: DesignMode) { if (mode !== this.doc.mode) this.commit({ ...this.doc, mode }) }
  /** Names a body (keyed by the feature that created it). An empty name removes the custom name. */
  setBodyName(root: string, name: string) {
    const bodyNames = { ...this.doc.bodyNames }
    if (name.trim()) bodyNames[root] = name.trim(); else delete bodyNames[root]
    this.commit({ ...this.doc, bodyNames })
  }
  setBodyHidden(root: string, hidden: boolean) {
    const rest = this.doc.hidden.filter((h) => h !== root)
    this.commit({ ...this.doc, hidden: hidden ? [...rest, root] : rest })
  }

  // ---- timeline marker (not an undo step) ----
  setMarker(n: number) {
    const m = this.clamp(n)
    if (m === this.marker) return
    this.marker = m
    this.lastKey = undefined
    this.emit()
  }
  stepMarker(delta: number) { this.setMarker(this.marker + delta) }

  get canUndo() { return this.past.length > 0 }
  get canRedo() { return this.future.length > 0 }
  undo() { const p = this.past.pop(); if (!p) return; this.future.push({ doc: this.doc, marker: this.marker }); this.restore(p) }
  redo() { const n = this.future.pop(); if (!n) return; this.past.push({ doc: this.doc, marker: this.marker }); this.restore(n) }
  private restore(s: Snapshot) { this.doc = s.doc; this.marker = this.clamp(s.marker); this.lastKey = undefined; this.emit() }

  /** Replace the document and forget the history. */
  replace(doc: Document, marker?: number) {
    this.past = []; this.future = []; this.lastKey = undefined
    this.doc = doc
    this.counter = doc.features.length
    this.marker = this.clamp(marker ?? doc.features.length, doc)
    this.emit()
  }

  /** Replace the document as an undoable edit (e.g. restoring a saved version). */
  load(doc: Document, marker?: number) { this.commit(doc, undefined, marker ?? doc.features.length) }
}

/** What the UI needs from a document store; implemented by `Store` and by the multi-design workspace. */
export type DocStore = Pick<Store, 'doc' | 'marker' | 'atEnd' | 'subscribe' | 'nextId' | 'add' | 'update' | 'remove' | 'move' | 'suppress' | 'rename' | 'setMode' | 'setBodyName' | 'setBodyHidden' | 'setMarker' | 'stepMarker' | 'canUndo' | 'canRedo' | 'undo' | 'redo' | 'replace' | 'load'>
