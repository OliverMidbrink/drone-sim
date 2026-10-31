import { describe, it, expect, beforeAll } from 'vitest'
import { loadOCC } from '@darkcad/kernel/node'
import { volume, faces } from '@darkcad/kernel/shape'
import { emptyDocument, type Document, type Feature } from '@darkcad/core'
import { rebuild } from '../src/rebuild'
import { fingerprintFace } from '../src/defeature'
import { zRange } from '@darkcad/kernel/shape'

let oc: any
beforeAll(async () => { oc = await loadOCC() })
const doc = (...features: Feature[]): Document => ({ ...emptyDocument('t'), features })
const rect = (id: string, w: number, h: number, offset = 0): Feature => ({ type: 'sketch', id, name: id, plane: 'XY', offset, entities: [{ id: id + 'r', kind: 'rect', x: -w / 2, y: -h / 2, w, h }] })
const circ = (id: string, r: number, offset = 0): Feature => ({ type: 'sketch', id, name: id, plane: 'XY', offset, entities: [{ id: id + 'c', kind: 'circle', x: 0, y: 0, r }] })

describe('rebuild', () => {
  it('extrudes a rectangle', () => {
    const r = rebuild(oc, doc(rect('s1', 20, 10), { type: 'extrude', id: 'e1', name: 'e', sketch: 's1', distance: 5, mode: 'new' }))
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('e1'))).toBeCloseTo(1000, 3)
  })

  it('sketch with an inner circle extrudes as a tube-like hole', () => {
    const s: Feature = { type: 'sketch', id: 's1', name: 's', plane: 'XY', offset: 0, entities: [
      { id: 'a', kind: 'rect', x: -10, y: -10, w: 20, h: 20 }, { id: 'b', kind: 'circle', x: 0, y: 0, r: 3 }] }
    const r = rebuild(oc, doc(s, { type: 'extrude', id: 'e1', name: 'e', sketch: 's1', distance: 4, mode: 'new' }))
    expect(volume(oc, r.bodies.get('e1'))).toBeCloseTo((400 - Math.PI * 9) * 4, 2)
  })

  it('extrude add and cut modify the previous body', () => {
    const r = rebuild(oc, doc(
      rect('s1', 20, 20), { type: 'extrude', id: 'e1', name: 'plate', sketch: 's1', distance: 5, mode: 'new' },
      circ('s2', 5, 5), { type: 'extrude', id: 'e2', name: 'boss', sketch: 's2', distance: 8, mode: 'add' },
      circ('s3', 2, 13), { type: 'extrude', id: 'e3', name: 'hole', sketch: 's3', distance: -13, mode: 'cut' },
    ))
    expect(r.errors.size).toBe(0)
    expect(r.visible).toEqual(['e3'])
    expect(volume(oc, r.bodies.get('e3'))).toBeCloseTo(2000 + Math.PI * 25 * 8 - Math.PI * 4 * 13, 2)
  })

  it('lofts a square to a circle', () => {
    const r = rebuild(oc, doc(rect('s1', 20, 20), circ('s2', 6, 15), { type: 'loft', id: 'l1', name: 'l', sketches: ['s1', 's2'], mode: 'new' }))
    expect(r.errors.size).toBe(0)
    const v = volume(oc, r.bodies.get('l1'))
    expect(v).toBeGreaterThan(Math.PI * 36 * 15)
    expect(v).toBeLessThan(400 * 15)
  })

  it('boolean combines two bodies and consumes both', () => {
    const r = rebuild(oc, doc(
      rect('s1', 20, 20), { type: 'extrude', id: 'a', name: 'a', sketch: 's1', distance: 10, mode: 'new' },
      circ('s2', 6, 0), { type: 'extrude', id: 'b', name: 'b', sketch: 's2', distance: 20, mode: 'new' },
      { type: 'boolean', id: 'c', name: 'c', op: 'cut', target: 'a', tool: 'b' }))
    expect(r.visible).toEqual(['c'])
    expect(volume(oc, r.bodies.get('c'))).toBeCloseTo(4000 - Math.PI * 36 * 10, 2)
  })

  it('thread feature: female G1/4 hole in a block', () => {
    const r = rebuild(oc, doc(
      rect('s1', 30, 30), { type: 'extrude', id: 'blk', name: 'block', sketch: 's1', distance: 20, mode: 'new' },
      { type: 'thread', id: 't1', name: 'port', target: 'blk', designation: 'G1/4', kind: 'female', length: 12, origin: [0, 0, 20], dir: [0, 0, -1] }))
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('t1'))).toBeLessThan(18000 - Math.PI * (11.8 / 2) ** 2 * 12)
  })

  it('delete-feature finds the face by fingerprint and removes the boss', () => {
    const base = doc(
      rect('s1', 40, 40), { type: 'extrude', id: 'e1', name: 'plate', sketch: 's1', distance: 5, mode: 'new' },
      circ('s2', 5, 5), { type: 'extrude', id: 'e2', name: 'boss', sketch: 's2', distance: 8, mode: 'add' })
    const built = rebuild(oc, base)
    const shape = built.bodies.get('e2')
    const side = faces(oc, shape).find((f) => zRange(oc, f)[1] > 12.9 && zRange(oc, f)[0] < 5.1 && zRange(oc, f)[0] > 4.9)!
    const fp = fingerprintFace(oc, side)
    const r = rebuild(oc, { ...base, features: [...base.features, { type: 'delete-feature', id: 'd1', name: 'del', target: 'e2', face: { surface: fp.surface, centroid: fp.centroid, area: fp.area } }] })
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('d1'))).toBeCloseTo(8000, 3)
  })

  it('reports per-feature errors and keeps going', () => {
    const r = rebuild(oc, doc(
      { type: 'extrude', id: 'bad', name: 'bad', sketch: 'missing', distance: 5, mode: 'new' },
      rect('s1', 10, 10), { type: 'extrude', id: 'ok', name: 'ok', sketch: 's1', distance: 5, mode: 'new' }))
    expect(r.errors.get('bad')).toMatch(/not found/)
    expect(r.bodies.has('ok')).toBe(true)
  })

  it('delete-feature with a stale reference fails cleanly', () => {
    const base = doc(rect('s1', 40, 40), { type: 'extrude', id: 'e1', name: 'p', sketch: 's1', distance: 5, mode: 'new' })
    const r = rebuild(oc, { ...base, features: [...base.features, { type: 'delete-feature', id: 'd', name: 'd', target: 'e1', face: { surface: 'Cylinder', centroid: [100, 100, 100], area: 5 } }] })
    expect(r.errors.get('d')).toMatch(/reference lost/)
  })
})
