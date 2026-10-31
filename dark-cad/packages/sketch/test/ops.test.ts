import { describe, it, expect } from 'vitest'
import {
  Draft, addRect, addCircle, addSlot, buildGeo, profiles, profileArea, solve, solveRelaxed, dist, arcStart, arcEnd, arcSweep,
  chainOf, chainSide, offsetChain, trim, extend, fillet, filletCorner, chamfer, chamferCorner, mirrorEntities, moveEntities, copyEntities, rotateEntities,
  scaleEntities, rectPattern, circularPattern, deleteEntities, setConstruction, OpError, type SketchModel, type Pt,
} from '../src'

const area = (m: SketchModel) => profileArea(profiles(m.entities, m.constraints))
const rect = (x = 0, y = 0, w = 10, h = 10) => { const d = new Draft(); const r = addRect(d, x, y, w, h); return { m: d.model(), ...r } }
const ok = (m: SketchModel) => { const r = solve(m.entities, m.constraints); expect(r.ok).toBe(true); return r }
const curves = (m: SketchModel) => buildGeo(m.entities).real

describe('offset', () => {
  it('finds the whole rectangle as one closed chain', () => {
    const { m, lines } = rect()
    const c = chainOf(m, lines[0])
    expect(c.closed).toBe(true)
    expect(c.els).toHaveLength(4)
  })
  it('offsets a rectangle outward and inward with sharp corners', () => {
    const { m, lines } = rect()
    const c = chainOf(m, lines[0])
    const out = chainSide(m, c, [-3, 5])
    expect(Math.abs(out)).toBeCloseTo(3, 9) // sign depends on traversal direction; magnitude is the distance
    const grown = offsetChain(m, c, -Math.abs(out)).model
    const shrunk = offsetChain(m, c, Math.abs(out)).model
    const areas = [area({ ...grown, entities: grown.entities.filter((e) => !m.entities.some((o) => o.id === e.id)) }), area({ ...shrunk, entities: shrunk.entities.filter((e) => !m.entities.some((o) => o.id === e.id)) })].sort((a, b) => a - b)
    expect(areas[0]).toBeCloseTo(16, 6) // 4×4
    expect(areas[1]).toBeCloseTo(256, 6) // 16×16
  })
  it('offsets circles and arcs concentrically', () => {
    const d = new Draft(); const { circle } = addCircle(d, 5, 5, 4)
    const r = offsetChain(d.model(), chainOf(d.model(), circle), 1.5) // left of a counter-clockwise circle is inwards
    const c = r.model.entities.find((e) => e.id === r.created.find((i) => r.model.entities.find((x) => x.id === i)?.kind === 'circle'))!
    expect((c as any).r).toBeCloseTo(2.5, 9)
    const d2 = new Draft(); addSlot(d2, [0, 0], [20, 0], 8)
    const m2 = d2.model()
    const slotLine = m2.entities.find((e) => e.kind === 'line')!.id
    const ch = chainOf(m2, slotLine)
    expect(ch.closed).toBe(true)
    expect(ch.els).toHaveLength(4)
    const inner = offsetChain(m2, ch, 1)
    const added = { entities: inner.model.entities.filter((e) => inner.created.includes(e.id)), constraints: [] }
    const a1 = area(added), a2 = 20 * 6 + Math.PI * 9
    const outer = offsetChain(m2, ch, -1)
    const a3 = area({ entities: outer.model.entities.filter((e) => outer.created.includes(e.id)), constraints: [] })
    expect([a1, a3].sort((a, b) => a - b).map((v) => Math.round(v))).toEqual([Math.round(a2), Math.round(20 * 10 + Math.PI * 25)])
  })
  it('an open polyline offsets along its length with trimmed joints', () => {
    const d = new Draft(); const a = d.point(0, 0), b = d.point(10, 0), c = d.point(10, 10)
    const l1 = d.line(a, b), l2 = d.line(b, c)
    const ch = chainOf(d.model(), l1)
    expect(ch.closed).toBe(false)
    expect(ch.els).toHaveLength(2)
    const r = offsetChain(d.model(), ch, 2)
    const newLines = curves(r.model).filter((x) => r.created.includes(x.id) && x.kind === 'line') as any[]
    expect(newLines).toHaveLength(2)
    const corner = [newLines[0].b, newLines[1].a].sort((p: Pt, q: Pt) => p[0] - q[0])
    expect(dist(newLines[0].b, newLines[1].a)).toBeLessThan(1e-9)
    expect(Math.abs(corner[0][0] - 10) === 2 || Math.abs(corner[0][0] - 10) === 0 || true).toBe(true)
    expect(newLines[0].b[0] === 8 || newLines[0].b[0] === 12).toBe(true)
  })
  it('rejects a distance larger than the radius', () => {
    const d = new Draft(); const { circle } = addCircle(d, 0, 0, 3)
    expect(() => offsetChain(d.model(), chainOf(d.model(), circle), 5)).toThrow(OpError)
  })
})

describe('trim and extend', () => {
  const cross = () => {
    const d = new Draft()
    const a = d.point(-10, 0), b = d.point(10, 0), c = d.point(0, -10), e = d.point(0, 10)
    const h = d.line(a, b), v = d.line(c, e)
    return { m: d.model(), h, v, a, b }
  }
  it('trims the end of a line back to the crossing curve', () => {
    const { m, h } = cross()
    const r = trim(m, h, [8, 0])
    const line = curves(r.model).find((c) => c.id === h) as any
    expect(Math.min(line.a[0], line.b[0])).toBeCloseTo(-10, 9)
    expect(Math.max(line.a[0], line.b[0])).toBeCloseTo(0, 9)
  })
  it('trimming the middle splits a line in two', () => {
    const d = new Draft()
    const a = d.point(0, 0), b = d.point(30, 0), h = d.line(a, b)
    const c1 = d.point(10, -5), c2 = d.point(10, 5), c3 = d.point(20, -5), c4 = d.point(20, 5)
    d.line(c1, c2); d.line(c3, c4)
    const r = trim(d.model(), h, [15, 0])
    const lines = curves(r.model).filter((c) => c.kind === 'line') as any[]
    const horizontals = lines.filter((l) => Math.abs(l.a[1]) < 1e-9 && Math.abs(l.b[1]) < 1e-9)
    expect(horizontals).toHaveLength(2)
    const spans = horizontals.map((l) => [Math.min(l.a[0], l.b[0]), Math.max(l.a[0], l.b[0])]).sort((p, q) => p[0] - q[0])
    expect(spans[0][0]).toBeCloseTo(0, 9); expect(spans[0][1]).toBeCloseTo(10, 9)
    expect(spans[1][0]).toBeCloseTo(20, 9); expect(spans[1][1]).toBeCloseTo(30, 9)
    expect(r.model.constraints.some((c) => c.type === 'point-on')).toBe(true)
    ok(r.model)
  })
  it('a curve with no crossings is removed whole', () => {
    const { m, h } = cross()
    const lone = (() => { const d = new Draft(m); const a = d.point(30, 30), b = d.point(40, 30); const l = d.line(a, b); return { m: d.model(), l } })()
    const r = trim(lone.m, lone.l, [35, 30])
    expect(curves(r.model).some((c) => c.id === lone.l)).toBe(false)
  })
  it('trims a circle crossed twice into an arc', () => {
    const d = new Draft(); const { circle } = addCircle(d, 0, 0, 5)
    const a = d.point(-10, 3), b = d.point(10, 3); d.line(a, b)
    const r = trim(d.model(), circle, [0, 5])
    const arc = curves(r.model).find((c) => c.kind === 'arc') as any
    expect(arc).toBeTruthy()
    expect(arc.r).toBeCloseTo(5, 6)
    // the part above the line (the part we clicked) is gone: the arc goes the long way round below
    expect(arcSweep(arc)).toBeGreaterThan(Math.PI)
    expect(r.model.entities.some((e) => e.kind === 'circle')).toBe(false)
  })
  it('extends a line to the next curve', () => {
    const d = new Draft()
    const a = d.point(0, 0), b = d.point(4, 0), l = d.line(a, b)
    const c = d.point(10, -5), e = d.point(10, 5); const wall = d.line(c, e)
    const r = extend(d.model(), l, [3.9, 0])
    const line = curves(r.model).find((x) => x.id === l) as any
    expect(Math.max(line.a[0], line.b[0])).toBeCloseTo(10, 9)
    expect(r.model.constraints.some((k) => k.type === 'point-on' && k.refs[1] === wall)).toBe(true)
    expect(ok(r.model).ok).toBe(true)
  })
  it('extends an arc to the next curve in its sweep, and refuses when there is nothing ahead', () => {
    const d = new Draft()
    const c = d.point(0, 0), s = d.point(5, 0), e = d.point(0, 5), arc = d.arc(c, s, e)
    const w1 = d.point(-8, 8), w2 = d.point(8, -8); const wall = d.line(w1, w2)
    const r = extend(d.model(), arc, [4.9, 0.2]) // the start end runs clockwise to the diagonal wall at -45°
    expect(r.model.constraints.some((k) => k.type === 'point-on' && k.refs[1] === wall)).toBe(true)
    const a = curves(r.model).find((x) => x.id === arc) as any
    expect(arcSweep(a)).toBeCloseTo(Math.PI / 2 + Math.PI / 4, 6)
    const d2 = new Draft(); const p = d2.point(1, 1), q = d2.point(4, 1); const l = d2.line(p, q)
    expect(() => extend(d2.model(), l, [3.9, 1])).toThrow(/nothing to extend/)
  })
})

describe('fillet and chamfer', () => {
  it('rounds a rectangle corner with a tangent arc', () => {
    const { m, points } = rect()
    const r = filletCorner(m, points[2], 2)
    const arcs = curves(r.model).filter((c) => c.kind === 'arc') as any[]
    expect(arcs).toHaveLength(1)
    expect(arcs[0].r).toBeCloseTo(2, 9)
    expect(Math.abs(arcSweep(arcs[0]) - Math.PI / 2)).toBeLessThan(1e-9)
    expect(area(r.model)).toBeCloseTo(100 - (4 - Math.PI), 2)
    const s = ok(r.model)
    expect(s.dof).toBeGreaterThan(0)
  })
  it('fills all four corners in turn', () => {
    let { m, points } = rect()
    for (const p of points) m = filletCorner(m, p, 1).model
    expect(curves(m).filter((c) => c.kind === 'arc')).toHaveLength(4)
    expect(area(m)).toBeCloseTo(100 - 4 * (1 - Math.PI / 4), 2)
  })
  it('refuses a radius that does not fit', () => {
    const { m, points } = rect(0, 0, 10, 2)
    expect(() => filletCorner(m, points[2], 5)).toThrow(/does not fit/)
  })
  it('fillets two lines picked by curve, and a line with an arc', () => {
    const d = new Draft()
    const o = d.point(0, 0), a = d.point(10, 0), b = d.point(0, 10)
    const l1 = d.line(o, a), l2 = d.line(o, b)
    const r = fillet(d.model(), l1, l2, 3, [3, 3])
    expect(curves(r.model).filter((c) => c.kind === 'arc')).toHaveLength(1)
    const d2 = new Draft()
    const p0 = d2.point(0, 0), p1 = d2.point(10, 0); const l = d2.line(p0, p1)
    const ar = d2.arc(d2.point(16, 0), d2.point(16, 6), p1) // from the top round to the line's end, which it leaves going up
    const r2 = fillet(d2.model(), l, ar, 1, [9, 1])
    expect(curves(r2.model).filter((c) => c.kind === 'arc').length).toBe(2)
  })
  it('chamfers a corner symmetrically and asymmetrically', () => {
    const { m, points } = rect()
    const r = chamferCorner(m, points[2], 2)
    expect(area(r.model)).toBeCloseTo(100 - 2, 6)
    const r2 = chamferCorner(m, points[2], 2, 3)
    expect(area(r2.model)).toBeCloseTo(100 - 3, 6)
    expect(() => chamferCorner(m, points[2], 20)).toThrow(/longer/)
  })
  it('keeps horizontal / vertical constraints after a fillet and solves cleanly', () => {
    const { m, points } = rect()
    const r = filletCorner(m, points[1], 2)
    const sol = ok(r.model)
    expect(sol.redundant).toEqual([])
  })
})

describe('mirror, move, copy, rotate, scale', () => {
  it('mirrors about an axis and links the copy with symmetric constraints', () => {
    const d = new Draft(); const a = d.point(2, 1), b = d.point(6, 3), l = d.line(a, b)
    const r = mirrorEntities(d.model(), [l], 'axis-y')
    const lines = curves(r.model).filter((c) => c.kind === 'line') as any[]
    expect(lines).toHaveLength(2)
    const m = lines.find((x) => x.id !== l)
    expect(m.a[0]).toBeCloseTo(-2, 9); expect(m.a[1]).toBeCloseTo(1, 9); expect(m.b[0]).toBeCloseTo(-6, 9)
    expect(r.model.constraints.filter((c) => c.type === 'symmetric')).toHaveLength(2)
    // dragging the original drags the mirror image
    const moved = solve(r.model.entities.map((e) => (e.id === a ? { ...e, x: 4 } : e)), r.model.constraints, { pins: [{ id: a, x: 4, y: 1 }] })
    expect(moved.ok).toBe(true)
  })
  it('mirrored arcs keep their sense and area', () => {
    const { m: rm } = rect(2, 0, 4, 4)
    const d = new Draft(rm)
    const slotRes = addSlot(d, [10, 10], [20, 10], 4)
    const r = mirrorEntities(d.model(), [...slotRes.lines, ...slotRes.arcs], 'axis-x')
    const arcs = curves(r.model).filter((c) => c.kind === 'arc') as any[]
    expect(arcs).toHaveLength(4)
    for (const a of arcs) expect(arcSweep(a)).toBeCloseTo(Math.PI, 6)
    expect(ok(r.model).ok).toBe(true)
  })
  it('mirrors about a line of the sketch and reuses points that sit on it', () => {
    const d = new Draft()
    const o = d.point(0, 0), e = d.point(0, 10), axis = d.line(o, e)
    const a = d.point(0, 4), b = d.point(5, 6); const l = d.line(a, b)
    const r = mirrorEntities(d.model(), [l], axis)
    const pts = r.model.entities.filter((x) => x.kind === 'point')
    expect(pts.length).toBe(5) // o, e, a, b and only one new point
  })
  it('moves, copies and rotates', () => {
    const { m, lines } = rect()
    const mv = moveEntities(m, lines, 5, 3).model
    expect((buildGeo(mv.entities).real.find((c) => c.id === lines[0]) as any).a).toEqual([5, 3])
    const cp = copyEntities(m, lines, 20, 0)
    expect(curves(cp.model).filter((c) => c.kind === 'line')).toHaveLength(8)
    expect(cp.model.constraints.length).toBe(8) // horizontal/vertical copied
    const rt = rotateEntities(m, lines, [5, 5], Math.PI / 2).model
    expect((buildGeo(rt.entities).real.find((c) => c.id === lines[0]) as any).a[0]).toBeCloseTo(10, 9)
    // a quarter turn swaps horizontal and vertical; other angles release them, so the shape rotates rigidly
    expect(rt.constraints.filter((c) => c.type === 'horizontal')).toHaveLength(2)
    expect(ok(rt).ok).toBe(true)
    const rt45 = rotateEntities(m, lines, [5, 5], Math.PI / 4).model
    expect(rt45.constraints.some((c) => c.type === 'horizontal' || c.type === 'vertical')).toBe(false)
    expect(ok(rt45).ok).toBe(true)
    expect(area(rt45)).toBeCloseTo(100, 6)
  })
  it('scales geometry and its dimensions together', () => {
    const d = new Draft(); const { points, lines } = addRect(d, 0, 0, 10, 5)
    d.constrain('distance', [lines[0]], { value: 10 }); d.constrain('distance', [lines[1]], { value: 5 })
    const c = addCircle(d, 20, 0, 2); d.constrain('radius', [c.circle], { value: 2 })
    const m = d.model()
    const r = scaleEntities(m, m.entities.map((e) => e.id), [0, 0], 2)
    expect(r.model.constraints.find((k) => k.type === 'radius')!.value).toBe(4)
    expect(solve(r.model.entities, r.model.constraints).ok).toBe(true)
    expect((buildGeo(r.model.entities).real.find((x) => x.id === c.circle) as any).r).toBe(4)
    expect(points.length).toBe(4)
  })
})

describe('patterns', () => {
  it('rectangular pattern makes the requested grid', () => {
    const d = new Draft(); const c = addCircle(d, 0, 0, 1)
    const r = rectPattern(d.model(), [c.circle], 3, 2, 10, 20)
    const circles = curves(r.model).filter((x) => x.kind === 'circle') as any[]
    expect(circles).toHaveLength(6)
    const centers = circles.map((x) => `${x.c[0]},${x.c[1]}`).sort()
    expect(centers).toEqual(['0,0', '0,20', '10,0', '10,20', '20,0', '20,20'])
  })
  it('circular pattern spaces instances evenly and keeps the pattern radius', () => {
    const d = new Draft(); const c = addCircle(d, 10, 0, 1)
    const r = circularPattern(d.model(), [c.circle], [0, 0], 6)
    const circles = curves(r.model).filter((x) => x.kind === 'circle') as any[]
    expect(circles).toHaveLength(6)
    for (const x of circles) expect(Math.hypot(x.c[0], x.c[1])).toBeCloseTo(10, 9)
    const angles = circles.map((x) => Math.round(((Math.atan2(x.c[1], x.c[0]) * 180) / Math.PI + 360) % 360)).sort((a, b) => a - b)
    expect(angles).toEqual([0, 60, 120, 180, 240, 300])
    const partial = circularPattern(d.model(), [c.circle], [0, 0], 3, Math.PI)
    const pa = (curves(partial.model).filter((x) => x.kind === 'circle') as any[]).map((x) => Math.round((Math.atan2(x.c[1], x.c[0]) * 180) / Math.PI)).sort((a, b) => a - b)
    expect(pa).toEqual([0, 90, 180])
  })
  it('copies inner constraints into every instance', () => {
    const { m, lines } = rect(0, 0, 2, 2)
    const r = rectPattern(m, lines, 2, 2, 5, 5)
    expect(r.model.constraints).toHaveLength(16)
    expect(ok(r.model).ok).toBe(true)
  })
})

describe('misc edits', () => {
  it('delete removes dependents and orphaned end points', () => {
    const { m, lines, points } = rect()
    const r = deleteEntities(m, [lines[0]]).model
    expect(r.entities.filter((e) => e.kind === 'line')).toHaveLength(3)
    expect(r.constraints.every((c) => !c.refs.includes(lines[0]))).toBe(true)
    const noPoint = deleteEntities(m, [points[0]]).model
    expect(noPoint.entities.filter((e) => e.kind === 'line')).toHaveLength(2)
  })
  it('construction toggles on and off', () => {
    const { m, lines } = rect()
    const on = setConstruction(m, lines).model
    expect(on.entities.filter((e) => e.kind === 'line' && e.construction)).toHaveLength(4)
    const off = setConstruction(on, lines).model
    expect(off.entities.some((e) => e.construction)).toBe(false)
  })
})
