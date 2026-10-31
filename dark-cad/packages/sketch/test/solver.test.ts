import { describe, it, expect } from 'vitest'
import { Draft, addRect, addCircle, solve, buildGeo, dist, type SketchModel } from '../src'

const run = (m: SketchModel, opts = {}) => solve(m.entities, m.constraints, opts)
const P = (r: ReturnType<typeof run>, id: string) => { const e = r.entities.find((x) => x.id === id) as any; return [e.x, e.y] as [number, number] }

describe('solver', () => {
  it('a bare rectangle keeps its shape and has 4 degrees of freedom', () => {
    const d = new Draft(); addRect(d, 0, 0, 10, 5)
    const r = run(d.model())
    expect(r.ok).toBe(true)
    expect(r.dof).toBe(4)
  })

  it('width and height dimensions plus a corner pin fully constrain a rectangle', () => {
    const d = new Draft(); const { points, lines } = addRect(d, 1, 1, 10, 5)
    d.constrain('distance', [lines[0]], { value: 20 })
    d.constrain('distance', [lines[1]], { value: 8 })
    d.constrain('hdist', ['origin', points[0]], { value: 3, sign: 1 })
    d.constrain('vdist', ['origin', points[0]], { value: 4, sign: 1 })
    const r = run(d.model())
    expect(r.ok).toBe(true)
    expect(r.dof).toBe(0)
    expect(r.free.size).toBe(0)
    expect(P(r, points[0])[0]).toBeCloseTo(3, 6)
    expect(P(r, points[0])[1]).toBeCloseTo(4, 6)
    expect(P(r, points[2])[0]).toBeCloseTo(23, 6)
    expect(P(r, points[2])[1]).toBeCloseTo(12, 6)
  })

  it('an under-constrained entity is reported free, a pinned one is not', () => {
    const d = new Draft(); const a = d.point(0, 0), b = d.point(10, 0), c = d.point(20, 0)
    d.constrain('fix', [a]); d.constrain('distance', [a, b], { value: 10 }); d.constrain('horizontal', [a, b])
    const r = run(d.model())
    expect(r.free.has(a)).toBe(false)
    expect(r.free.has(b)).toBe(false)
    expect(r.free.has(c)).toBe(true)
    expect(r.dof).toBe(2)
  })

  it('horizontal / vertical / parallel / perpendicular', () => {
    const d = new Draft()
    const p = [d.point(0, 0), d.point(10, 1), d.point(0, 5), d.point(8, 9)]
    const l1 = d.line(p[0], p[1]), l2 = d.line(p[2], p[3])
    d.constrain('horizontal', [l1]); d.constrain('perpendicular', [l1, l2]); d.constrain('fix', [p[0]]); d.constrain('fix', [p[2]])
    const r = run(d.model())
    expect(r.ok).toBe(true)
    expect(P(r, p[1])[1]).toBeCloseTo(0, 6)
    expect(P(r, p[3])[0]).toBeCloseTo(0, 6) // l2 vertical
    const d2 = new Draft()
    const q = [d2.point(0, 0), d2.point(10, 0), d2.point(0, 5), d2.point(9, 4)]
    const a = d2.line(q[0], q[1]), b = d2.line(q[2], q[3])
    d2.constrain('parallel', [a, b]); d2.constrain('fix', [q[0]]); d2.constrain('fix', [q[1]]); d2.constrain('fix', [q[2]])
    const r2 = run(d2.model())
    expect(P(r2, q[3])[1]).toBeCloseTo(5, 6)
  })

  it('equal lines, midpoint and symmetric', () => {
    const d = new Draft()
    const p = [d.point(0, 0), d.point(10, 0), d.point(0, 3), d.point(7, 3)]
    const a = d.line(p[0], p[1]), b = d.line(p[2], p[3])
    d.constrain('fix', [p[0]]); d.constrain('fix', [p[1]]); d.constrain('fix', [p[2]]); d.constrain('horizontal', [b]); d.constrain('equal', [a, b])
    const r = run(d.model())
    expect(P(r, p[3])[0]).toBeCloseTo(10, 6)
    const d2 = new Draft()
    const m = d2.point(1, 1), s = d2.point(0, 0), e = d2.point(10, 4), l = d2.line(s, e)
    d2.constrain('fix', [s]); d2.constrain('fix', [e]); d2.constrain('midpoint', [m, l])
    expect(P(run(d2.model()), m)).toEqual([5, 2].map((v) => expect.closeTo(v, 6)))
    const d3 = new Draft()
    const a1 = d3.point(3, 2), b1 = d3.point(-1, 5), ax = 'axis-y'
    d3.constrain('fix', [a1]); d3.constrain('symmetric', [a1, b1, ax])
    const r3 = run(d3.model())
    expect(P(r3, b1)[0]).toBeCloseTo(-3, 6)
    expect(P(r3, b1)[1]).toBeCloseTo(2, 6)
  })

  it('radius, diameter and concentric on circles', () => {
    const d = new Draft()
    const a = addCircle(d, 0, 0, 5), b = addCircle(d, 3, 3, 2)
    d.constrain('diameter', [a.circle], { value: 20 }); d.constrain('radius', [b.circle], { value: 4 }); d.constrain('concentric', [a.circle, b.circle])
    d.constrain('fix', [a.center])
    const r = run(d.model())
    expect(r.ok).toBe(true)
    expect((r.entities.find((e) => e.id === a.circle) as any).r).toBeCloseTo(10, 6)
    expect((r.entities.find((e) => e.id === b.circle) as any).r).toBeCloseTo(4, 6)
    expect(P(r, b.center)[0]).toBeCloseTo(0, 6)
    expect(r.dof).toBe(0)
  })

  it('tangent line to a circle, and two externally tangent circles', () => {
    const d = new Draft()
    const c = addCircle(d, 0, 0, 5)
    const p = d.point(-10, 8), q = d.point(10, 8), l = d.line(p, q)
    d.constrain('fix', [c.center]); d.constrain('radius', [c.circle], { value: 5 }); d.constrain('horizontal', [l]); d.constrain('tangent', [l, c.circle])
    const r = run(d.model())
    expect(r.ok).toBe(true)
    expect(Math.abs(P(r, p)[1])).toBeCloseTo(5, 6)
    const d2 = new Draft()
    const a = addCircle(d2, 0, 0, 3), b = addCircle(d2, 10, 0, 2)
    d2.constrain('fix', [a.center]); d2.constrain('radius', [a.circle], { value: 3 }); d2.constrain('radius', [b.circle], { value: 2 })
    d2.constrain('horizontal', [a.center, b.center]); d2.constrain('tangent', [a.circle, b.circle], { sign: 1 })
    const r2 = run(d2.model())
    expect(P(r2, b.center)[0]).toBeCloseTo(5, 6)
  })

  it('point on a circle, and an arc keeps its end points on one radius', () => {
    const d = new Draft()
    const c = d.point(0, 0), a = d.point(5, 0), b = d.point(0, 6), arc = d.arc(c, a, b)
    d.constrain('fix', [c]); d.constrain('fix', [a])
    const r = run(d.model())
    expect(r.ok).toBe(true)
    expect(Math.hypot(...P(r, b))).toBeCloseTo(5, 6)
    expect(buildGeo(r.entities).byId.get(arc)).toBeTruthy()
  })

  it('angle dimension between two lines', () => {
    const d = new Draft()
    const o = d.point(0, 0), a = d.point(10, 0), b = d.point(10, 10)
    const l1 = d.line(o, a), l2 = d.line(o, b)
    d.constrain('fix', [o]); d.constrain('fix', [a]); d.constrain('angle', [l1, l2], { value: 30, sign: 1 }); d.constrain('distance', [l2], { value: 10 })
    const r = run(d.model())
    expect(r.ok).toBe(true)
    expect(P(r, b)[0]).toBeCloseTo(10 * Math.cos(Math.PI / 6), 6)
    expect(P(r, b)[1]).toBeCloseTo(10 * Math.sin(Math.PI / 6), 6)
  })

  it('detects conflicting constraints and keeps the geometry', () => {
    const d = new Draft()
    const a = d.point(0, 0), b = d.point(10, 0), l = d.line(a, b)
    d.constrain('fix', [a]); d.constrain('horizontal', [l]); d.constrain('distance', [l], { value: 10 })
    const bad = d.constrain('distance', [l], { value: 25 })
    const r = run(d.model())
    expect(r.ok).toBe(false)
    expect(r.conflicts).toEqual([bad])
    expect(r.entities).toBe(d.model().entities)
  })

  it('flags redundant constraints without failing', () => {
    const d = new Draft()
    const a = d.point(0, 0), b = d.point(10, 0), l = d.line(a, b)
    d.constrain('fix', [a]); d.constrain('horizontal', [l])
    const dup = d.constrain('horizontal', [a, b])
    const r = run(d.model())
    expect(r.ok).toBe(true)
    expect(r.redundant).toEqual([dup])
  })

  it('dragging a point with soft pins moves only what the constraints allow', () => {
    const d = new Draft(); const { points } = addRect(d, 0, 0, 10, 5)
    // drag the top-right corner: the rectangle stays rectangular, other corners follow
    const r = run(d.model(), { pins: [{ id: points[2], x: 14, y: 9 }], quick: true })
    expect(r.ok).toBe(true)
    const [x, y] = P(r, points[2])
    expect(x).toBeCloseTo(14, 1); expect(y).toBeCloseTo(9, 1)
    expect(P(r, points[1])[0]).toBeCloseTo(P(r, points[2])[0], 5) // vertical edge preserved
    expect(P(r, points[3])[1]).toBeCloseTo(P(r, points[2])[1], 5) // horizontal edge preserved
  })

  it('drag keeps a dimensioned length while moving the pinned end around', () => {
    const d = new Draft()
    const a = d.point(0, 0), b = d.point(10, 0), l = d.line(a, b)
    d.constrain('fix', [a]); d.constrain('distance', [l], { value: 10 })
    const r = run(d.model(), { pins: [{ id: b, x: 0, y: 20 }], quick: true })
    expect(dist(P(r, a), P(r, b))).toBeCloseTo(10, 6)
    expect(P(r, b)[1]).toBeGreaterThan(8)
  })

  it('scales to a few hundred entities quickly', () => {
    const d = new Draft()
    for (let i = 0; i < 120; i++) { const { lines } = addRect(d, i * 12, 0, 10, 5); d.constrain('distance', [lines[0]], { value: 10 }) }
    const t = performance.now()
    const r = run(d.model())
    expect(r.ok).toBe(true)
    expect(performance.now() - t).toBeLessThan(5000)
  })
})
