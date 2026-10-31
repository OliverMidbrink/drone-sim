import { emptyDocument, type Document, type Feature } from './model'

/** Document + linear undo/redo history. Every edit replaces the document immutably. */
export class Store {
  doc: Document
  private past: Document[] = []
  private future: Document[] = []
  private listeners = new Set<() => void>()
  private counter = 0

  constructor(doc: Document = emptyDocument()) {
    this.doc = doc
    this.counter = doc.features.length
  }

  subscribe(fn: () => void) { this.listeners.add(fn); return () => { this.listeners.delete(fn) } }
  private emit() { this.listeners.forEach((f) => f()) }

  nextId(prefix: string): string {
    let id: string
    do id = `${prefix}${++this.counter}`; while (this.doc.features.some((f) => f.id === id))
    return id
  }

  /** `coalesceKey` merges consecutive edits with the same key into one undo step (e.g. a drag). */
  private lastKey?: string
  commit(next: Document, coalesceKey?: string) {
    if (!(coalesceKey && coalesceKey === this.lastKey)) this.past.push(this.doc)
    this.lastKey = coalesceKey
    this.doc = next
    this.future = []
    this.emit()
  }

  add(feature: Feature, at?: number) {
    const fs = [...this.doc.features]
    fs.splice(at ?? fs.length, 0, feature)
    this.commit({ ...this.doc, features: fs })
  }

  update<T extends Feature>(id: string, patch: Partial<T>, coalesceKey?: string) {
    this.commit({ ...this.doc, features: this.doc.features.map((f) => (f.id === id ? ({ ...f, ...patch } as Feature) : f)) }, coalesceKey)
  }

  remove(id: string) {
    this.commit({ ...this.doc, features: this.doc.features.filter((f) => f.id !== id) })
  }

  move(id: string, to: number) {
    const fs = [...this.doc.features]
    const i = fs.findIndex((f) => f.id === id)
    if (i < 0) return
    const [f] = fs.splice(i, 1)
    fs.splice(Math.max(0, Math.min(to, fs.length)), 0, f)
    this.commit({ ...this.doc, features: fs })
  }

  rename(name: string) { this.commit({ ...this.doc, name }) }

  get canUndo() { return this.past.length > 0 }
  get canRedo() { return this.future.length > 0 }
  undo() { const p = this.past.pop(); if (!p) return; this.future.push(this.doc); this.doc = p; this.lastKey = undefined; this.emit() }
  redo() { const n = this.future.pop(); if (!n) return; this.past.push(this.doc); this.doc = n; this.lastKey = undefined; this.emit() }

  replace(doc: Document) { this.past = []; this.future = []; this.lastKey = undefined; this.doc = doc; this.counter = doc.features.length; this.emit() }
}

/** What the UI needs from a document store; implemented by `Store` and by the multi-design workspace. */
export type DocStore = Pick<Store, 'doc' | 'subscribe' | 'nextId' | 'add' | 'update' | 'remove' | 'move' | 'rename' | 'canUndo' | 'canRedo' | 'undo' | 'redo' | 'replace'>
