import { describe, it, expect, beforeAll } from 'vitest'
import { loadOCC } from '@darkcad/kernel/node'
import { volume, isValid, countFaces } from '@darkcad/kernel/shape'
import { emptyDocument, type Document, type Feature } from '@darkcad/core'
import { Draft, addRect, addCircle, addSlot, addArc3, addPolygon, addPolyline } from '@darkcad/sketch'
import { rebuild } from '../src/rebuild'
import { sketchPreview, sketchProfiles } from '../src/sketch'

let oc: any
beforeAll(async () => { oc = await loadOCC() })
const sk = (id: string, build: (d: Draft) => void, plane: 'XY' | 'XZ' | 'YZ' = 'XY', offset = 0): Feature => { const d = new Draft(); build(d); return { type: 'sketch', id, name: id, plane, offset, entities: d.entities, constraints: d.constraints } }
const doc = (...features: Feature[]): Document => ({ ...emptyDocument('t'), features })
const ext = (id: string, sketch: string, distance: number): Feature => ({ type: 'extrude', id, name: id, sketch, distance, mode: 'new' })

describe('sketches with arcs build real kernel geometry', () => {
  it('a slot extrudes to its exact volume', () => {
    const r = rebuild(oc, doc(sk('s', (d) => addSlot(d, [0, 0], [20, 0], 8)), ext('e', 's', 5)))
    expect(r.errors.size).toBe(0)
    const body = r.bodies.get('e')
    expect(volume(oc, body)).toBeCloseTo((20 * 8 + Math.PI * 16) * 5, 3)
    expect(isValid(oc, body)).toBe(true)
    expect(countFaces(oc, body)).toBe(2 + 4) // top, bottom, two sides, two round ends
  })

  it('a D shape (arc and chord) and an angled slot', () => {
    const d1 = sk('s', (d) => { const a = d.get(addArc3(d, [10, 0], [0, 10], [-10, 0])!) as any; d.line(a.a, a.b) })
    const r = rebuild(oc, doc(d1, ext('e', 's', 2)))
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('e'))).toBeCloseTo(Math.PI * 50 * 2, 3)
    const r2 = rebuild(oc, doc(sk('s', (d) => addSlot(d, [0, 0], [10, 10], 6)), ext('e', 's', 1)))
    expect(volume(oc, r2.bodies.get('e'))).toBeCloseTo((Math.hypot(10, 10) * 6 + Math.PI * 9) * 1, 3)
  })

  it('a hole in a slot, and holes in a polygon', () => {
    const r = rebuild(oc, doc(sk('s', (d) => { addSlot(d, [0, 0], [20, 0], 12); addCircle(d, 0, 0, 2); addCircle(d, 20, 0, 2) }), ext('e', 's', 3)))
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('e'))).toBeCloseTo((20 * 12 + Math.PI * 36 - 2 * Math.PI * 4) * 3, 3)
    const r2 = rebuild(oc, doc(sk('s', (d) => { addPolygon(d, [0, 0], [10, 0], 6); addCircle(d, 0, 0, 3) }), ext('e', 's', 2)))
    expect(volume(oc, r2.bodies.get('e'))).toBeCloseTo(((3 * Math.sqrt(3)) / 2 * 100 - Math.PI * 9) * 2, 3)
  })

  it('construction geometry is ignored by extrude', () => {
    const r = rebuild(oc, doc(sk('s', (d) => { addRect(d, 0, 0, 10, 10); addRect(d, -5, -5, 30, 30, true); addCircle(d, 5, 5, 1, true) }), ext('e', 's', 2)))
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('e'))).toBeCloseTo(200, 3)
  })

  it('a rectangle split by a line extrudes as one solid of the full volume', () => {
    const r = rebuild(oc, doc(sk('s', (d) => {
      const { points, lines } = addRect(d, 0, 0, 10, 10)
      const m1 = d.point(5, 0), m2 = d.point(5, 10)
      ;(d.get(lines[0]) as any).b = m1; d.line(m1, points[1]); (d.get(lines[2]) as any).b = m2; d.line(m2, points[3]); d.line(m1, m2)
    }), ext('e', 's', 2)))
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('e'))).toBeCloseTo(200, 3)
  })

  it('an open chain cannot be extruded and says why', () => {
    const r = rebuild(oc, doc(sk('s', (d) => addPolyline(d, [[0, 0], [5, 0], [5, 5]], false)), ext('e', 's', 2)))
    expect(r.errors.get('e')).toMatch(/no closed profile/)
  })

  it('lofts a slot to a circle', () => {
    const r = rebuild(oc, doc(sk('a', (d) => addSlot(d, [-6, 0], [6, 0], 8)), sk('b', (d) => addCircle(d, 0, 0, 5), 'XY', 12), { type: 'loft', id: 'l', name: 'l', sketches: ['a', 'b'], mode: 'new' }))
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('l'))).toBeGreaterThan(Math.PI * 25 * 12 * 0.9)
  })

  it('previews expose curves, construction flags and shaded regions', () => {
    const s = sk('s', (d) => { addSlot(d, [0, 0], [10, 0], 4); addRect(d, 20, 20, 5, 5, true) }) as any
    const p = sketchPreview(s)
    expect(p.regions).toHaveLength(1)
    expect(p.curves.some((c) => c.construction)).toBe(true)
    expect(sketchProfiles(s).regions[0].outer.area).toBeCloseTo(10 * 4 + Math.PI * 4, 6)
  })

  it('rectangles that touch along partial edges extrude as one solid with the summed volume', () => {
    const r = rebuild(oc, doc(sk('s', (d) => { addRect(d, 0, 0, 10, 10); addRect(d, 10, 6, 5, 4); addRect(d, 10, 0, 5, 4) }), ext('e', 's', 2)))
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('e'))).toBeCloseTo(140 * 2, 3)
    expect(isValid(oc, r.bodies.get('e'))).toBe(true)
  })

  it('overlapping rectangles extrude as their union', () => {
    const r = rebuild(oc, doc(sk('s', (d) => { addRect(d, 0, 0, 10, 10); addRect(d, 5, 5, 10, 10) }), ext('e', 's', 1)))
    expect(r.errors.size).toBe(0)
    expect(volume(oc, r.bodies.get('e'))).toBeCloseTo(175, 3)
  })
})
