import { describe, it, expect } from 'vitest'
import { Workspace } from './workspace'
import type { SketchFeature } from '@darkcad/core'

class MemoryStorage implements Storage {
  private m = new Map<string, string>()
  get length() { return this.m.size }
  clear() { this.m.clear() }
  getItem(k: string) { return this.m.get(k) ?? null }
  key(i: number) { return [...this.m.keys()][i] ?? null }
  removeItem(k: string) { this.m.delete(k) }
  setItem(k: string, v: string) { this.m.set(k, v) }
}

const sketch = (id: string): SketchFeature => ({ type: 'sketch', id, name: id, plane: 'XY', offset: 0, entities: [], constraints: [] })

describe('workspace', () => {
  it('starts with one tab named Untitled', () => {
    const w = new Workspace(new MemoryStorage())
    expect(w.tabs().map((t) => t.name)).toEqual(['Untitled'])
    expect(w.doc.features).toEqual([])
  })

  it('new designs get unique names and become active', () => {
    const w = new Workspace(new MemoryStorage())
    const first = w.activeId
    w.create(); w.create()
    expect(w.tabs().map((t) => t.name)).toEqual(['Untitled', 'Untitled 2', 'Untitled 3'])
    expect(w.activeId).not.toBe(first)
  })

  it('each design keeps its own features and its own undo history', () => {
    const w = new Workspace(new MemoryStorage())
    const a = w.activeId
    w.add(sketch('a1'))
    const b = w.create()
    expect(w.doc.features.length).toBe(0)
    expect(w.canUndo).toBe(false)
    w.add(sketch('b1')); w.add(sketch('b2'))
    w.activate(a)
    expect(w.doc.features.map((f) => f.id)).toEqual(['a1'])
    w.undo()
    expect(w.doc.features.length).toBe(0)
    w.activate(b)
    expect(w.doc.features.map((f) => f.id)).toEqual(['b1', 'b2'])
  })

  it('notifies subscribers on edits and on tab changes', () => {
    const w = new Workspace(new MemoryStorage())
    let n = 0
    w.subscribe(() => n++)
    w.add(sketch('s'))
    w.create()
    expect(n).toBeGreaterThanOrEqual(2)
  })

  it('persists designs and the tab layout across a "reload"', () => {
    const storage = new MemoryStorage()
    const w = new Workspace(storage)
    w.add(sketch('s1'))
    w.rename('Pump adapter')
    const second = w.create()
    w.add(sketch('t1')); w.add(sketch('t2'))
    w.flush()

    const again = new Workspace(storage)
    expect(again.tabs().map((t) => t.name)).toEqual(['Pump adapter', 'Untitled'])
    expect(again.activeId).toBe(second)
    expect(again.doc.features.map((f) => f.id)).toEqual(['t1', 't2'])
    again.activate(again.tabs()[0].id)
    expect(again.doc.features.map((f) => f.id)).toEqual(['s1'])
  })

  it('closing a tab keeps the design in the library and it can be reopened', () => {
    const w = new Workspace(new MemoryStorage())
    w.add(sketch('keep'))
    const first = w.activeId
    w.create()
    w.close(first)
    expect(w.tabs().length).toBe(1)
    expect(w.library().map((m) => m.id)).toContain(first)
    w.open(first)
    expect(w.activeId).toBe(first)
    expect(w.doc.features.map((f) => f.id)).toEqual(['keep'])
  })

  it('closing the last tab opens a fresh design instead of leaving none', () => {
    const w = new Workspace(new MemoryStorage())
    w.close(w.activeId)
    expect(w.tabs().length).toBe(1)
    expect(w.doc.features).toEqual([])
  })

  it('deleting a design removes it for good (and from storage)', () => {
    const storage = new MemoryStorage()
    const w = new Workspace(storage)
    const doomed = w.activeId
    w.add(sketch('x'))
    w.create()
    w.flush()
    w.deleteDesign(doomed)
    expect(w.library().some((m) => m.id === doomed)).toBe(false)
    expect(storage.getItem(`darkcad:design:${doomed}`)).toBeNull()
    expect(new Workspace(storage).library().some((m) => m.id === doomed)).toBe(false)
  })

  it('adopts a legacy single-document autosave as the first design', () => {
    const storage = new MemoryStorage()
    const seed = new Workspace(new MemoryStorage())
    seed.add(sketch('old')); seed.rename('Legacy')
    storage.setItem('darkcad:autosave', JSON.stringify(seed.doc))
    const w = new Workspace(storage)
    expect(w.tabs().map((t) => t.name)).toEqual(['Legacy'])
    expect(w.doc.features.map((f) => f.id)).toEqual(['old'])
  })

  it('survives a corrupt index', () => {
    const storage = new MemoryStorage()
    storage.setItem('darkcad:workspace:v1', '{not json')
    expect(new Workspace(storage).tabs().length).toBe(1)
  })
})
