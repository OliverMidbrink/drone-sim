import { describe, it, expect, beforeAll } from 'vitest'
import { loadOCC } from '@darkcad/kernel/node'
import { volume, isValid } from '@darkcad/kernel/shape'
import { emptyDocument, type AxisRef, type Document, type Feature, type RevolveFeature } from '@darkcad/core'
import { Draft, addRect, addArc3, profiles, interiorPoint } from '@darkcad/sketch'
import { rebuild } from '../src/rebuild'

let oc: any
beforeAll(async () => { oc = await loadOCC() })
const sk = (id: string, build: (d: Draft) => void, plane: 'XY' | 'XZ' | 'YZ' = 'XY', offset = 0): Feature => { const d = new Draft(); build(d); return { type: 'sketch', id, name: id, plane, offset, entities: d.entities, constraints: d.constraints } }
const doc = (...features: Feature[]): Document => ({ ...emptyDocument('t'), features })
const regionPts = (s: Feature) => { const x = s as any; return profiles(x.entities, x.constraints).regions.map((r) => interiorPoint(r)) }
const rev = (s: Feature, axis: AxisRef | undefined, extra: Partial<RevolveFeature> = {}, which?: number[]): RevolveFeature => {
  const pts = regionPts(s)
  return { type: 'revolve', id: 'r', name: 'r', profiles: pts.filter((_, i) => !which || which.includes(i)).map((pt) => ({ sketch: s.id, pt })), axis, extent: 'full', angle: 360, mode: 'new', ...extra }
}
const AY: AxisRef = { kind: 'sketch', sketch: 's', id: 'axis-y' }
const washer = sk('s', (d) => addRect(d, 5, 0, 5, 4))

describe('revolve', () => {
  it('a full revolve of an offset rectangle is a washer (Pappus)', () => {
    const r = rebuild(oc, doc(washer, rev(washer, AY)))
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('r'))).toBeCloseTo(Math.PI * (100 - 25) * 4, 2)
    expect(isValid(oc, r.bodies.get('r'))).toBe(true)
  })

  it('angle and symmetric sweeps take the matching fraction', () => {
    const quarter = rebuild(oc, doc(washer, rev(washer, AY, { extent: 'angle', angle: 90 })))
    expect(quarter.errors.size).toBe(0)
    expect(volume(oc, quarter.bodies.get('r'))).toBeCloseTo((Math.PI * 75 * 4) / 4, 2)
    const sym = rebuild(oc, doc(washer, rev(washer, AY, { extent: 'symmetric', angle: 120 })))
    expect(volume(oc, sym.bodies.get('r'))).toBeCloseTo((Math.PI * 75 * 4) / 3, 2)
  })

  it('uses a world axis and a drawn construction line', () => {
    const xz = sk('s', (d) => addRect(d, 5, 0, 5, 4), 'XZ')
    const r = rebuild(oc, doc(xz, rev(xz, { kind: 'world', axis: 'Z' })))
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('r'))).toBeCloseTo(Math.PI * 75 * 4, 2)
    const withLine = sk('s', (d) => { addRect(d, 5, 0, 5, 4); d.line(d.point(2, -5), d.point(2, 9), true) })
    const line = (withLine as any).entities.find((e: any) => e.kind === 'line' && e.construction)
    const r2 = rebuild(oc, doc(withLine, rev(withLine, { kind: 'sketch', sketch: 's', id: line.id })))
    expect(r2.errors.size).toBe(0)
    expect(volume(oc, r2.bodies.get('r'))).toBeCloseTo(Math.PI * (64 - 9) * 4, 2) // radii 3..8
  })

  it('revolving only some of several touching profiles, and all of them together', () => {
    const two = sk('s', (d) => { addRect(d, 5, 0, 5, 4); addRect(d, 5, 4, 5, 3) })
    const both = rebuild(oc, doc(two, rev(two, AY)))
    expect(both.errors.size).toBe(0)
    expect(volume(oc, both.bodies.get('r'))).toBeCloseTo(Math.PI * 75 * 7, 2)
    const pts = regionPts(two)
    const bigIdx = pts.findIndex((p) => p[1] < 4)
    const one = rebuild(oc, doc(two, rev(two, AY, {}, [bigIdx])))
    expect(volume(oc, one.bodies.get('r'))).toBeCloseTo(Math.PI * 75 * 4, 2)
  })

  it('an arc profile revolves into a sphere', () => {
    const half = sk('s', (d) => { const a = d.get(addArc3(d, [0, -10], [10, 0], [0, 10])!) as any; d.line(a.a, a.b) })
    const r = rebuild(oc, doc(half, rev(half, AY)))
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('r'))).toBeCloseTo((4 / 3) * Math.PI * 1000, 1)
  })

  it('a profile that crosses the axis is reported', () => {
    const cross = sk('s', (d) => addRect(d, -2, 0, 7, 4))
    const r = rebuild(oc, doc(cross, rev(cross, AY)))
    expect(r.errors.get('r')).toMatch(/crosses the axis/)
  })

  it('an unfinished revolve is skipped without an error', () => {
    const r = rebuild(oc, doc(washer, rev(washer, undefined)))
    expect(r.errors.size).toBe(0)
    expect(r.bodies.has('r')).toBe(false)
    const r2 = rebuild(oc, doc(washer, rev(washer, AY, { profiles: [] })))
    expect(r2.errors.size).toBe(0)
  })

  it('cut removes the revolved ring from an existing body', () => {
    const base = sk('b', (d) => addRect(d, -12, -12, 24, 24))
    const cutter = sk('s', (d) => addRect(d, 5, 0, 5, 4), 'XZ')
    const r = rebuild(oc, doc(base, { type: 'extrude', id: 'e', name: 'e', sketch: 'b', distance: 4, mode: 'new' }, cutter, rev(cutter, { kind: 'world', axis: 'Z' }, { mode: 'cut' })))
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('r'))).toBeCloseTo(24 * 24 * 4 - Math.PI * 75 * 4, 2)
  })

  it('a profile whose sketch changed so the region is gone says so', () => {
    const r = rebuild(oc, doc(washer, rev(washer, AY, { profiles: [{ sketch: 's', pt: [-50, -50] }] })))
    expect(r.errors.get('r')).toMatch(/no longer exists/)
  })
})
