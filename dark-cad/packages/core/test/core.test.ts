import { describe, it, expect } from 'vitest'
import { Store, emptyDocument, serialize, parse, type SketchFeature } from '../src'

const sketch = (id: string): SketchFeature => ({ type: 'sketch', id, name: id, plane: 'XY', offset: 0, entities: [{ id: 'e1', kind: 'rect', x: 0, y: 0, w: 10, h: 5 }] })

describe('store', () => {
  it('adds, undoes and redoes', () => {
    const s = new Store()
    s.add(sketch('s1'))
    s.add(sketch('s2'))
    expect(s.doc.features.length).toBe(2)
    s.undo(); expect(s.doc.features.length).toBe(1)
    s.redo(); expect(s.doc.features.length).toBe(2)
    s.undo(); s.undo(); s.undo()
    expect(s.doc.features.length).toBe(0)
    expect(s.canUndo).toBe(false)
  })

  it('coalesces drag edits into one undo step', () => {
    const s = new Store()
    s.add(sketch('s1'))
    for (let i = 1; i <= 20; i++) s.update<SketchFeature>('s1', { offset: i }, 'drag:s1')
    expect(s.doc.features[0]).toMatchObject({ offset: 20 })
    s.undo()
    expect(s.doc.features[0]).toMatchObject({ offset: 0 })
  })

  it('a new edit clears redo', () => {
    const s = new Store()
    s.add(sketch('s1')); s.undo(); s.add(sketch('s2'))
    expect(s.canRedo).toBe(false)
  })

  it('moves and removes features, notifying subscribers', () => {
    const s = new Store()
    let n = 0
    s.subscribe(() => n++)
    s.add(sketch('a')); s.add(sketch('b')); s.add(sketch('c'))
    s.move('c', 0)
    expect(s.doc.features.map((f) => f.id)).toEqual(['c', 'a', 'b'])
    s.remove('a')
    expect(s.doc.features.map((f) => f.id)).toEqual(['c', 'b'])
    expect(n).toBe(5)
  })

  it('generates unique ids', () => {
    const s = new Store()
    s.add(sketch('s1'))
    expect(s.nextId('s')).not.toBe('s1')
  })
})

describe('serialisation', () => {
  it('round-trips and is canonical', () => {
    const s = new Store(); s.add(sketch('s1'))
    const text = serialize(s.doc)
    expect(parse(text)).toEqual(s.doc)
    expect(serialize(parse(text))).toBe(text)
  })

  it('is independent of key insertion order and rounds float noise', () => {
    const a = { version: 1, name: 'x', units: 'mm', features: [{ b: 0.1 + 0.2, a: 1 }] } as any
    const b = { features: [{ a: 1, b: 0.3 }], units: 'mm', name: 'x', version: 1 } as any
    expect(serialize(a)).toBe(serialize(b))
  })

  it('rejects newer or invalid documents', () => {
    expect(() => parse('{"version":99,"features":[]}')).toThrow(/version 99/)
    expect(() => parse('{}')).toThrow(/not a DarkCAD/)
    expect(() => parse('{"version":1}')).toThrow(/feature list/)
  })

  it('empty document parses', () => { expect(parse(serialize(emptyDocument('t'))).name).toBe('t') })
})
