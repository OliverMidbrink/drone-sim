import { describe, it, expect, beforeAll } from 'vitest'
import { loadOCC } from '@darkcad/kernel/node'
import { volume, tightBounds } from '@darkcad/kernel/shape'
import { writeSTEP } from '@darkcad/kernel/step'
import { make } from '@darkcad/kernel/occ'
import { emptyDocument, type Document, type Feature } from '@darkcad/core'
import { Draft, addRect, addCircle } from '@darkcad/sketch'
import { rebuild } from '../src/rebuild'

let oc: any
beforeAll(async () => { oc = await loadOCC() })
const doc = (features: Feature[], extra: Partial<Document> = {}): Document => ({ ...emptyDocument('t'), ...extra, features })
const sk = (id: string, build: (d: Draft) => void, extra: Record<string, unknown> = {}): Feature => {
  const d = new Draft(); build(d)
  return { type: 'sketch', id, name: id, plane: 'XY', offset: 0, entities: d.entities, constraints: d.constraints, ...extra } as Feature
}
const rect = (id: string, w: number, h: number, extra: Record<string, unknown> = {}) => sk(id, (d) => addRect(d, -w / 2, -h / 2, w, h), extra)
const ex = (id: string, sketch: string, distance: number, mode: 'new' | 'add' | 'cut' = 'new', extra: Record<string, unknown> = {}): Feature => ({ type: 'extrude', id, name: id, sketch, distance, mode, ...extra } as Feature)

describe('rolling back', () => {
  const full = doc([rect('s1', 10, 10), ex('e1', 's1', 5), rect('s2', 4, 4), ex('e2', 's2', 10, 'add')])

  it('building only the first features leaves later ones out', () => {
    const r = rebuild(oc, { ...full, features: full.features.slice(0, 2) })
    expect(r.visible).toEqual(['e1'])
    expect(volume(oc, r.bodies.get('e1'))).toBeCloseTo(500, 3)
  })
})

describe('suppressing and blocking', () => {
  it('a suppressed feature is skipped and the features that need it say so', () => {
    const r = rebuild(oc, doc([rect('s1', 10, 10, { suppressed: true }), ex('e1', 's1', 5)]))
    expect(r.bodies.size).toBe(0)
    expect(r.errors.get('e1')).toMatch(/suppressed/)
  })

  it('a feature after a failed one says it is blocked by it', () => {
    const r = rebuild(oc, doc([rect('s1', 10, 10), ex('e1', 's1', 5, 'cut'), rect('s2', 4, 4), ex('e2', 's2', 5, 'add')]))
    expect(r.errors.get('e1')).toMatch(/nothing to cut from/)
    expect(r.errors.get('e2')).toBe('Blocked by "e1": nothing to cut from')
  })

  it('names the upstream feature for a sketch that failed', () => {
    const r = rebuild(oc, doc([{ type: 'plane', id: 'p', name: 'Plane A', def: { kind: 'offset', base: { kind: 'plane', id: 'nope' }, distance: 1 } }, rect('s1', 10, 10, { planeRef: 'p' }), ex('e1', 's1', 5)]))
    expect(r.errors.get('p')).toMatch(/not found/)
    expect(r.errors.get('s1')).toMatch(/not found|Blocked/)
    expect(r.errors.get('e1')).toMatch(/^Blocked by "s1"/)
  })

  it('reordering that puts a body feature before its sketch breaks only what depends on it', () => {
    const r = rebuild(oc, doc([ex('e1', 's1', 5), rect('s1', 10, 10)]))
    expect(r.errors.get('e1')).toMatch(/not found/)
  })
})

describe('body lineage', () => {
  it('Add and Cut keep the body of the feature that created it', () => {
    const r = rebuild(oc, doc([
      rect('s1', 20, 20), ex('e1', 's1', 5),
      sk('s2', (d) => addCircle(d, 0, 0, 3), { offset: 5 }), ex('e2', 's2', 4, 'add'),
      sk('s3', (d) => addCircle(d, 0, 0, 1), { offset: 9 }), ex('e3', 's3', -9, 'cut'),
    ]))
    expect(r.errors.size).toBe(0)
    expect(r.visible).toEqual(['e3'])
    expect([...r.roots]).toEqual([['e1', 'e1'], ['e2', 'e1'], ['e3', 'e1']])
  })

  it('separate bodies have separate roots', () => {
    const r = rebuild(oc, doc([rect('s1', 5, 5), ex('e1', 's1', 5), rect('s2', 5, 5, { offset: 20 }), ex('e2', 's2', 5)]))
    expect(r.roots.get('e1')).toBe('e1')
    expect(r.roots.get('e2')).toBe('e2')
  })
})

describe('construction planes', () => {
  it('extrudes a sketch placed on an offset plane', () => {
    const r = rebuild(oc, doc([
      { type: 'plane', id: 'p', name: 'p', def: { kind: 'offset', base: { kind: 'world', plane: 'XY' }, distance: 7 } },
      rect('s1', 10, 10, { planeRef: 'p' }), ex('e1', 's1', 3),
    ]))
    expect(r.errors.size).toBe(0)
    const b = tightBounds(oc, r.bodies.get('e1'))
    expect(b.min[2]).toBeCloseTo(7, 1); expect(b.max[2]).toBeCloseTo(10, 1)
  })

  it('an angled plane tilts the extrusion', () => {
    const r = rebuild(oc, doc([
      { type: 'plane', id: 'p', name: 'p', def: { kind: 'angle', base: { kind: 'world', plane: 'XY' }, axis: 'X', angle: 90 } },
      rect('s1', 10, 10, { planeRef: 'p' }), ex('e1', 's1', 4),
    ]))
    expect(r.errors.size).toBe(0)
    const b = tightBounds(oc, r.bodies.get('e1'))
    expect(b.max[2] - b.min[2]).toBeCloseTo(10, 1) // the sketch's y now runs along Z
    expect(Math.abs(b.max[1] - b.min[1])).toBeCloseTo(4, 1)
  })
})

describe('components and STEP import', () => {
  const part = (name: string): Document => doc([rect('s1', 10, 10), ex('e1', 's1', 5)], { name })
  const comp = (id: string, ref: string, pos: [number, number, number] = [0, 0, 0], rot: [number, number, number] = [0, 0, 0]): Feature => ({ type: 'component', id, name: id, ref, placement: { pos, rot } })

  it('places an instance of another design', () => {
    const r = rebuild(oc, doc([comp('c1', 'A', [100, 0, 0])], { mode: 'assembly' }), { deps: { A: part('A') } })
    expect(r.errors.size).toBe(0)
    const b = tightBounds(oc, r.bodies.get('c1'))
    expect(b.min[0]).toBeCloseTo(95, 1); expect(b.max[0]).toBeCloseTo(105, 1)
    expect(volume(oc, r.bodies.get('c1'))).toBeCloseTo(500, 2)
  })

  it('rotates before moving', () => {
    const r = rebuild(oc, doc([comp('c1', 'A', [0, 0, 0], [0, 90, 0])]), { deps: { A: part('A') } })
    const b = tightBounds(oc, r.bodies.get('c1'))
    expect(b.max[0] - b.min[0]).toBeCloseTo(5, 1) // the 5 mm height now lies along X
    expect(b.max[2] - b.min[2]).toBeCloseTo(10, 1)
  })

  it('two instances of one component are separate bodies', () => {
    const r = rebuild(oc, doc([comp('c1', 'A'), comp('c2', 'A', [50, 0, 0])]), { deps: { A: part('A') } })
    expect(r.visible).toEqual(['c1', 'c2'])
  })

  it('reports a missing design, an empty one, and a loop', () => {
    const empty = doc([], { name: 'Empty' })
    const selfish = doc([comp('x', 'root')], { name: 'Selfish' })
    const r = rebuild(oc, doc([comp('c1', 'gone'), comp('c2', 'E'), comp('c3', 'S')]), { designId: 'root', deps: { E: empty, S: selfish } })
    expect(r.errors.get('c1')).toMatch(/missing/)
    expect(r.errors.get('c2')).toMatch(/no solid bodies/)
    expect(r.errors.get('c3')).toMatch(/no solid bodies/) // S's own component fails with the loop, leaving S empty
  })

  it('flags a component that contains itself', () => {
    const a = doc([comp('x', 'A')], { name: 'A' })
    const r = rebuild(oc, a, { designId: 'A', deps: { A: a } })
    expect(r.errors.get('x')).toMatch(/loop/)
  })

  it('imports a STEP asset as a body', () => {
    const solid = rebuild(oc, part('A')).bodies.get('e1')
    const bytes = writeSTEP(oc, solid)
    const r = rebuild(oc, doc([{ type: 'import-step', id: 'i1', name: 'i1', asset: 'a1' }]), { assets: { a1: bytes } })
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('i1'))).toBeCloseTo(500, 1)
    const missing = rebuild(oc, doc([{ type: 'import-step', id: 'i1', name: 'i1', asset: 'nope' }]))
    expect(missing.errors.get('i1')).toMatch(/not available/)
  })

  it('a STEP part can be placed as a component too', () => {
    const bytes = writeSTEP(oc, rebuild(oc, part('A')).bodies.get('e1'))
    const step = doc([{ type: 'import-step', id: 'i1', name: 'i1', asset: 'a1' }], { name: 'Imported' })
    const r = rebuild(oc, doc([comp('c1', 'S', [0, 0, 20])]), { deps: { S: step }, assets: { a1: bytes } })
    expect(r.errors.size).toBe(0)
    expect(tightBounds(oc, r.bodies.get('c1')).min[2]).toBeCloseTo(20, 1)
  })
})
void make
