import { describe, it, expect } from 'vitest'
import { Store, emptyDocument, parse, serialize, resolveFrames, sketchFrame, collectDeps, wouldLoop, type Feature, type SketchFeature, type PlaneFeature, type Document } from '../src'

const sk = (id: string, extra: Partial<SketchFeature> = {}): SketchFeature => ({ type: 'sketch', id, name: id, plane: 'XY', offset: 0, entities: [], constraints: [], ...extra })
const ex = (id: string, sketch: string): Feature => ({ type: 'extrude', id, name: id, sketch, distance: 5, mode: 'new' })
const ids = (s: Store) => s.doc.features.map((f) => f.id)

describe('timeline marker', () => {
  it('stays at the end while features are appended', () => {
    const s = new Store()
    s.add(sk('a')); s.add(sk('b'))
    expect(s.marker).toBe(2)
    expect(s.atEnd).toBe(true)
  })

  it('moving the marker is not an undo step', () => {
    const s = new Store()
    s.add(sk('a')); s.add(sk('b'))
    s.setMarker(1)
    expect(s.marker).toBe(1)
    s.undo()
    expect(ids(s)).toEqual(['a'])
    expect(s.marker).toBe(1)
  })

  it('inserts at the marker and advances it', () => {
    const s = new Store()
    s.add(sk('a')); s.add(sk('c'))
    s.setMarker(1)
    s.add(sk('b'))
    expect(ids(s)).toEqual(['a', 'b', 'c'])
    expect(s.marker).toBe(2)
    expect(s.atEnd).toBe(false)
  })

  it('undo restores the marker that was current before the edit', () => {
    const s = new Store()
    s.add(sk('a')); s.add(sk('c'))
    s.setMarker(1)
    s.add(sk('b'))
    s.undo()
    expect(ids(s)).toEqual(['a', 'c'])
    expect(s.marker).toBe(1)
    s.redo()
    expect(s.marker).toBe(2)
  })

  it('removing before the marker shifts it back', () => {
    const s = new Store()
    s.add(sk('a')); s.add(sk('b')); s.add(sk('c'))
    s.setMarker(2)
    s.remove('a')
    expect(s.marker).toBe(1)
    s.setMarker(0); s.remove('b')
    expect(s.marker).toBe(0)
  })

  it('suppress toggles and is undoable', () => {
    const s = new Store()
    s.add(sk('a'))
    s.suppress('a')
    expect(s.doc.features[0].suppressed).toBe(true)
    s.suppress('a', false)
    expect(s.doc.features[0].suppressed).toBeUndefined()
    s.undo()
    expect(s.doc.features[0].suppressed).toBe(true)
  })

  it('clamps the marker and steps by one', () => {
    const s = new Store()
    s.add(sk('a')); s.add(sk('b'))
    s.setMarker(99); expect(s.marker).toBe(2)
    s.stepMarker(-1); expect(s.marker).toBe(1)
    s.setMarker(-5); expect(s.marker).toBe(0)
  })

  it('body names and visibility are removed with their feature and undoable', () => {
    const s = new Store()
    s.add(sk('a')); s.add(ex('e1', 'a'))
    s.setBodyName('e1', ' Bracket '); s.setBodyHidden('e1', true)
    expect(s.doc.bodyNames).toEqual({ e1: 'Bracket' })
    expect(s.doc.hidden).toEqual(['e1'])
    s.remove('e1')
    expect(s.doc.bodyNames).toEqual({}); expect(s.doc.hidden).toEqual([])
    s.undo()
    expect(s.doc.bodyNames).toEqual({ e1: 'Bracket' })
  })

  it('load() is an undoable replacement', () => {
    const s = new Store()
    s.add(sk('a'))
    const other: Document = { ...emptyDocument('v1'), features: [sk('x'), sk('y')] }
    s.load(other)
    expect(ids(s)).toEqual(['x', 'y'])
    s.undo()
    expect(ids(s)).toEqual(['a'])
  })
})

describe('construction planes', () => {
  const plane = (id: string, def: PlaneFeature['def']): PlaneFeature => ({ type: 'plane', id, name: id, def })

  it('offsets a world plane along its normal', () => {
    const f = resolveFrames([plane('p', { kind: 'offset', base: { kind: 'world', plane: 'XY' }, distance: 7 })])
    expect(f.frames.get('p')!.origin).toEqual([0, 0, 7])
  })

  it('chains planes and puts sketches on them', () => {
    const fs: Feature[] = [
      plane('p1', { kind: 'offset', base: { kind: 'world', plane: 'XY' }, distance: 5 }),
      plane('p2', { kind: 'offset', base: { kind: 'plane', id: 'p1' }, distance: 3 }),
      sk('s', { planeRef: 'p2' }),
    ]
    expect(sketchFrame(fs, fs[2] as SketchFeature).origin).toEqual([0, 0, 8])
    expect(resolveFrames(fs).frames.get('s')!.dir).toEqual([0, 0, 1])
  })

  it('rotates a plane about a world axis', () => {
    const f = resolveFrames([plane('p', { kind: 'angle', base: { kind: 'world', plane: 'XY' }, axis: 'X', angle: 90 })]).frames.get('p')!
    expect(f.dir[0]).toBeCloseTo(0); expect(f.dir[1]).toBeCloseTo(-1); expect(f.dir[2]).toBeCloseTo(0)
    expect(f.xdir).toEqual([1, 0, 0])
  })

  it('finds the midplane of two parallel planes and rejects crossing ones', () => {
    const mid = resolveFrames([plane('m', { kind: 'mid', a: { kind: 'world', plane: 'XY' }, b: { kind: 'world', plane: 'XY' } })])
    expect(mid.frames.get('m')!.origin).toEqual([0, 0, 0])
    const fs: Feature[] = [
      plane('a', { kind: 'offset', base: { kind: 'world', plane: 'XY' }, distance: 10 }),
      plane('m', { kind: 'mid', a: { kind: 'world', plane: 'XY' }, b: { kind: 'plane', id: 'a' } }),
      plane('bad', { kind: 'mid', a: { kind: 'world', plane: 'XY' }, b: { kind: 'world', plane: 'XZ' } }),
    ]
    const r = resolveFrames(fs)
    expect(r.frames.get('m')!.origin).toEqual([0, 0, 5])
    expect(r.errors.get('bad')).toMatch(/parallel/)
  })

  it('reports planes used before they exist, and everything that depends on them', () => {
    const fs: Feature[] = [sk('s', { planeRef: 'p' }), plane('p', { kind: 'offset', base: { kind: 'world', plane: 'XY' }, distance: 1 })]
    const r = resolveFrames(fs)
    expect(r.errors.get('s')).toMatch(/not found/)
    expect(r.frames.has('p')).toBe(true)
  })

  it('suppressed planes disappear', () => {
    const r = resolveFrames([{ ...plane('p', { kind: 'offset', base: { kind: 'world', plane: 'XY' }, distance: 1 }), suppressed: true }, sk('s', { planeRef: 'p' })])
    expect(r.errors.get('s')).toMatch(/not found/)
  })
})

describe('document v3', () => {
  it('migrates v2 documents', () => {
    const d = parse('{"version":2,"name":"x","units":"mm","features":[]}')
    expect(d).toMatchObject({ version: 3, mode: 'part', bodyNames: {}, hidden: [] })
  })
  it('round-trips mode, names and hidden bodies', () => {
    const d: Document = { ...emptyDocument('t', 'hybrid'), bodyNames: { e1: 'A' }, hidden: ['e1'] }
    expect(parse(serialize(d))).toEqual(d)
  })
})

describe('component dependencies', () => {
  const comp = (id: string, ref: string): Feature => ({ type: 'component', id, name: id, ref, placement: { pos: [0, 0, 0], rot: [0, 0, 0] } })
  const docs: Record<string, Document> = {
    root: { ...emptyDocument('root', 'assembly'), features: [comp('c1', 'A'), comp('c2', 'B'), comp('c3', 'gone')] },
    A: { ...emptyDocument('A'), features: [comp('x', 'B'), { type: 'import-step', id: 'imp', name: 'imp', asset: 'asset1' }] },
    B: { ...emptyDocument('B'), features: [] },
  }
  const get = (id: string) => docs[id]

  it('collects transitive designs, missing ones and assets', () => {
    const d = collectDeps('root', docs.root, get)
    expect(Object.keys(d.docs).sort()).toEqual(['A', 'B'])
    expect(d.missing).toEqual(['gone'])
    expect(d.assets).toEqual(['asset1'])
  })

  it('does not follow loops', () => {
    const loop = { ...docs, B: { ...docs.B, features: [comp('y', 'root')] } }
    const d = collectDeps('root', loop.root, (id) => loop[id])
    expect(Object.keys(d.docs).sort()).toEqual(['A', 'B'])
  })

  it('detects would-be loops', () => {
    expect(wouldLoop('root', 'root', get)).toBe(true)
    expect(wouldLoop('B', 'A', get)).toBe(true) // A already contains B
    expect(wouldLoop('A', 'B', get)).toBe(false)
  })
})
