import { describe, it, expect } from 'vitest'
import { Draft, addRect, addCircle, buildGeo, snap, pickEntity, boxSelect, type SnapOptions } from '../src'

const geoOf = (d: Draft) => buildGeo(d.entities)
const opts = (o: Partial<SnapOptions> = {}): SnapOptions => ({ pxPerMm: 10, object: true, grid: true, ...o })

describe('snapping', () => {
  it('grid step 1 mm at 20 px/mm, 0.1 mm at 200 px/mm, 10 mm at 2 px/mm', () => {
    const g = geoOf(new Draft())
    expect(snap([3.4, 2.6], g, opts({ pxPerMm: 20 })).pt).toEqual([3, 3])
    expect(snap([3.04, 2.97], g, opts({ pxPerMm: 200 })).pt).toEqual([3, 3])
    expect(snap([3.41, 2.62], g, opts({ pxPerMm: 200 })).pt[0]).toBeCloseTo(3.4, 9)
    expect(snap([13, 26], g, opts({ pxPerMm: 2 })).pt).toEqual([10, 30])
  })

  it('sliding along an axis keeps the grid', () => {
    const g = geoOf(new Draft())
    const r = snap([5.0000003, 0.02], g, opts({ pxPerMm: 20 }))
    expect(r.kind).toBe('on-curve')
    expect(r.pt).toEqual([5, 0])
  })

  it('no snapping at all when both are off', () => {
    const g = geoOf(new Draft())
    expect(snap([1.234, 5.678], g, opts({ object: false, grid: false })).pt).toEqual([1.234, 5.678])
  })

  it('endpoint beats on-curve and grid', () => {
    const d = new Draft(); const { points } = addRect(d, 0, 0, 10, 10)
    const r = snap([10.2, 0.1], geoOf(d), opts())
    expect(r.kind).toBe('endpoint')
    expect(r.ref).toBe(points[1])
    expect(r.pt).toEqual([10, 0])
  })

  it('midpoint, center and quadrant', () => {
    const d = new Draft(); addRect(d, 0, 0, 10, 10)
    expect(snap([5.1, 0.1], geoOf(d), opts()).kind).toBe('midpoint')
    const d2 = new Draft(); addCircle(d2, 20, 20, 5)
    const g = geoOf(d2)
    expect(snap([20.1, 19.9], g, opts()).kind).toBe('center')
    const q = snap([25.1, 20.05], g, opts())
    expect(q.kind).toBe('quadrant')
    expect(q.pt[0]).toBeCloseTo(25, 9)
  })

  it('intersection of two curves', () => {
    const d = new Draft()
    const a = d.point(0, 0), b = d.point(10, 10), c = d.point(0, 10), e = d.point(10, 0)
    d.line(a, b); d.line(c, e)
    const r = snap([5.1, 4.9], geoOf(d), opts())
    expect(r.kind).toBe('intersection')
    expect(r.pt[0]).toBeCloseTo(5, 9); expect(r.pt[1]).toBeCloseTo(5, 9)
  })

  it('perpendicular and tangent need a previous point', () => {
    const d = new Draft(); const a = d.point(0, 0), b = d.point(20, 0); d.line(a, b)
    const g = geoOf(d)
    const r = snap([7.2, 0.8], g, opts({ from: [7, 6], grid: false }))
    expect(r.kind).toBe('perpendicular')
    expect(r.pt).toEqual([7, 0])
    const d2 = new Draft(); addCircle(d2, 0, 0, 5)
    const t = snap([2.6, 4.4], geoOf(d2), opts({ from: [10, 0], grid: false, pxPerMm: 5 }))
    expect(t.kind).toBe('tangent')
    expect(Math.hypot(...t.pt)).toBeCloseTo(5, 9)
    expect(((t.pt[0] - 10) * t.pt[0] + t.pt[1] * t.pt[1])).toBeCloseTo(0, 6) // radius ⟂ tangent line
  })

  it('aligns with an existing point horizontally and vertically, with guides', () => {
    const d = new Draft(); d.point(10, 10); d.point(-5, 30)
    const r = snap([-4.9, 10.05], geoOf(d), opts({ grid: false }))
    expect(r.kind).toBe('inference')
    expect(r.pt).toEqual([-5, 10])
    expect(r.guides.length).toBe(2)
  })

  it('locks the angle in steps and keeps the grid length', () => {
    const g = geoOf(new Draft())
    const r = snap([10, 3], g, opts({ from: [0, 0], angleStep: Math.PI / 12, object: false, pxPerMm: 20 }))
    expect(r.kind).toBe('angle')
    expect(Math.atan2(r.pt[1], r.pt[0]) * 180 / Math.PI).toBeCloseTo(15, 6)
    expect(Math.hypot(...r.pt)).toBeCloseTo(Math.round(Math.hypot(10, 3)), 9)
  })

  it('excludes the geometry being drawn', () => {
    const d = new Draft(); const p = d.point(5, 5)
    const r = snap([5.1, 5.1], geoOf(d), opts({ exclude: new Set([p]), grid: false }))
    expect(r.kind).not.toBe('endpoint')
  })
})

describe('picking', () => {
  it('prefers points over curves and ignores far clicks', () => {
    const d = new Draft(); const { points, lines } = addRect(d, 0, 0, 10, 10)
    const g = geoOf(d)
    expect(pickEntity(g, [0.1, 0.1], 0.5)).toBe(points[0])
    expect(pickEntity(g, [5, 0.1], 0.5)).toBe(lines[0])
    expect(pickEntity(g, [5, 5], 0.5)).toBeUndefined()
  })
  it('window select needs full containment, crossing select only a touch', () => {
    const d = new Draft(); const { lines } = addRect(d, 0, 0, 10, 10)
    const rect = { x0: -1, y0: -1, x1: 5, y1: 11 }
    expect(boxSelect(d.entities, rect, false).filter((id) => lines.includes(id))).toEqual([lines[3]])
    const cross = boxSelect(d.entities, rect, true)
    expect(cross).toEqual(expect.arrayContaining([lines[0], lines[2], lines[3]]))
    expect(cross).not.toContain(lines[1])
  })
})
