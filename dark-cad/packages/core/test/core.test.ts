import { describe, it, expect } from 'vitest'
import { Store, emptyDocument, serialize, parse, type SketchFeature, type RevolveFeature } from '../src'

const sketch = (id: string): SketchFeature => ({ type: 'sketch', id, name: id, plane: 'XY', offset: 0, entities: [{ id: 'e1', kind: 'point', x: 0, y: 0 }], constraints: [] })

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

  it('a document without a title gets "Untitled"', () => {
    expect(parse('{"version":1,"units":"mm","features":[]}').name).toBe('Untitled')
    expect(parse('{"version":1,"name":"  ","units":"mm","features":[]}').name).toBe('Untitled')
  })
  it('empty document parses', () => { expect(parse(serialize(emptyDocument('t'))).name).toBe('t') })
})

describe('v1 → v2 migration', () => {
  const v1 = JSON.stringify({
    version: 1, name: 'old', units: 'mm',
    features: [{ type: 'sketch', id: 's1', name: 'S', plane: 'XY', offset: 0, entities: [
      { id: 'a', kind: 'rect', x: 1, y: 2, w: 10, h: 5 }, { id: 'b', kind: 'circle', x: 3, y: 4, r: 2 }, { id: 'c', kind: 'polygon', points: [[0, 0], [4, 0], [0, 3]] }] }],
  })
  it('turns rect / circle / polygon into points, lines and constraints', () => {
    const d = parse(v1)
    expect(d.version).toBe(3)
    const s = d.features[0] as SketchFeature
    expect(s.entities.filter((e) => e.kind === 'point').length).toBe(4 + 1 + 3)
    expect(s.entities.filter((e) => e.kind === 'line').length).toBe(4 + 3)
    expect(s.entities.filter((e) => e.kind === 'circle')).toHaveLength(1)
    expect(s.constraints.map((c) => c.type)).toEqual(['horizontal', 'vertical', 'horizontal', 'vertical'])
    const ids = s.entities.map((e) => e.id)
    expect(new Set(ids).size).toBe(ids.length)
    const pts = new Map(s.entities.filter((e) => e.kind === 'point').map((e: any) => [e.id, e]))
    for (const e of s.entities) if (e.kind === 'line') { expect(pts.has(e.a)).toBe(true); expect(pts.has(e.b)).toBe(true) }
  })
  it('keeps the rectangle geometry', () => {
    const s = parse(v1).features[0] as SketchFeature
    const lines = s.entities.filter((e) => e.kind === 'line').slice(0, 4) as any[]
    const pt = (id: string) => s.entities.find((e) => e.id === id) as any
    expect(lines.map((l) => [pt(l.a).x, pt(l.a).y])).toEqual([[1, 2], [11, 2], [11, 7], [1, 7]])
  })
  it('v2 documents without constraints get an empty list', () => {
    const d = parse('{"version":2,"name":"x","units":"mm","features":[{"type":"sketch","id":"s","name":"s","plane":"XY","offset":0}]}')
    expect((d.features[0] as SketchFeature).constraints).toEqual([])
  })
})

describe('revolve feature', () => {
  it('survives a serialize / parse round trip and fills defaults for hand-made files', () => {
    const rev: RevolveFeature = { type: 'revolve', id: 'r1', name: 'Revolve 1', profiles: [{ sketch: 's1', pt: [1, 2] }], axis: { kind: 'sketch', sketch: 's1', id: 'axis-y' }, extent: 'angle', angle: 90, mode: 'add' }
    const doc = { ...emptyDocument('t'), features: [sketch('s1'), rev] }
    expect(parse(serialize(doc)).features[1]).toEqual(rev)
    const bare = parse('{"version":2,"name":"x","units":"mm","features":[{"type":"revolve","id":"r","name":"r"}]}').features[0] as RevolveFeature
    expect(bare.profiles).toEqual([]); expect(bare.extent).toBe('full'); expect(bare.angle).toBe(360); expect(bare.mode).toBe('new')
  })
})
