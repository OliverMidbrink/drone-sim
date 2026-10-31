import { describe, it, expect } from 'vitest'
import { Draft, addRect, addCircle, addSlot, addPolygon, addPolyline, addArc3, profiles, profileArea, solve } from '../src'

const area = (d: Draft) => profileArea(profiles(d.entities, d.constraints))

describe('profiles', () => {
  it('a rectangle is one region', () => {
    const d = new Draft(); addRect(d, 0, 0, 10, 5)
    const p = profiles(d.entities, d.constraints)
    expect(p.regions).toHaveLength(1)
    expect(p.regions[0].outer.area).toBeCloseTo(50, 9)
    expect(p.open).toEqual([])
  })

  it('a circle inside a rectangle is a hole; a rectangle inside that is solid again', () => {
    const d = new Draft(); addRect(d, -20, -20, 40, 40); addCircle(d, 0, 0, 10); addRect(d, -3, -3, 6, 6)
    const p = profiles(d.entities, d.constraints)
    expect(p.regions).toHaveLength(2)
    expect(p.regions[0].holes).toHaveLength(1)
    expect(profileArea(p)).toBeCloseTo(1600 - Math.PI * 100 + 36, 1)
  })

  it('a line across a rectangle splits it into two regions', () => {
    const d = new Draft(); const { points } = addRect(d, 0, 0, 10, 10)
    const mid1 = d.point(5, 0), mid2 = d.point(5, 10)
    const bottom = d.entities.filter((e) => e.kind === 'line')
    // split top and bottom edges at x=5 by re-wiring them through midpoints
    const lines = bottom as any[]
    lines[0].b = mid1; d.line(mid1, points[1])
    lines[2].b = mid2; d.line(mid2, points[3])
    d.line(mid1, mid2)
    const p = profiles(d.entities, d.constraints)
    expect(p.regions).toHaveLength(2)
    expect(p.regions.map((r) => r.outer.area).sort()).toEqual([50, 50])
  })

  it('a slot has the area of a rectangle plus a circle', () => {
    const d = new Draft(); addSlot(d, [0, 0], [20, 0], 8)
    expect(area(d)).toBeCloseTo(20 * 8 + Math.PI * 16, 2)
    const r = solve(d.entities, d.constraints)
    expect(r.ok).toBe(true)
    expect(r.dof).toBe(5)
  })

  it('a regular hexagon is constrained like a regular polygon (4 degrees of freedom)', () => {
    const d = new Draft(); addPolygon(d, [0, 0], [10, 0], 6)
    expect(area(d)).toBeCloseTo((3 * Math.sqrt(3) / 2) * 100, 6)
    const r = solve(d.entities, d.constraints)
    expect(r.ok).toBe(true)
    expect(r.dof).toBe(4)
    const c = addPolygon(new Draft(), [0, 0], [8, 0], 5, 'circumscribed')
    expect(c.points).toHaveLength(5)
    const d2 = new Draft(); addPolygon(d2, [0, 0], [8, 0], 4, 'circumscribed')
    expect(area(d2)).toBeCloseTo(256, 6) // square with apothem 8: side 16
  })

  it('ignores construction geometry and reports open chains', () => {
    const d = new Draft(); addRect(d, 0, 0, 10, 10, true)
    expect(profiles(d.entities, d.constraints).regions).toHaveLength(0)
    const d2 = new Draft(); const { lines } = addPolyline(d2, [[0, 0], [5, 0], [5, 5]], false)
    const p = profiles(d2.entities, d2.constraints)
    expect(p.regions).toHaveLength(0)
    expect(p.open.sort()).toEqual([...lines].sort())
  })

  it('a dangling spur on a rectangle does not break the loop', () => {
    const d = new Draft(); const { points } = addRect(d, 0, 0, 10, 10)
    const tip = d.point(15, 15); const spur = d.line(points[2], tip)
    const p = profiles(d.entities, d.constraints)
    expect(p.regions).toHaveLength(1)
    expect(p.open).toEqual([spur])
  })

  it('a half-disc from an arc and a chord', () => {
    const d = new Draft()
    const a = addArc3(d, [10, 0], [0, 10], [-10, 0])!
    const arc = d.get(a) as any
    d.line(arc.a, arc.b)
    expect(area(d)).toBeCloseTo(Math.PI * 50, 2)
  })

  it('separate points joined by a coincident constraint or sitting on each other still close the loop', () => {
    const d = new Draft()
    const p = [d.point(0, 0), d.point(10, 0), d.point(10, 10), d.point(0, 10)]
    const q = [d.point(10, 0), d.point(10, 10), d.point(0, 10), d.point(0, 0)]
    for (let i = 0; i < 4; i++) d.line(p[i], q[i])
    expect(area(d)).toBeCloseTo(100, 9)
    const d2 = new Draft()
    const a = d2.point(0, 0), b = d2.point(10, 0), c = d2.point(5, 8), c2 = d2.point(5, 8.001)
    d2.line(a, b); d2.line(b, c); d2.line(c2, a)
    expect(profiles(d2.entities, d2.constraints).regions).toHaveLength(0)
    d2.constrain('coincident', [c, c2])
    expect(profiles(d2.entities, d2.constraints).regions).toHaveLength(1)
  })

  it('is orientation independent', () => {
    const d = new Draft(); const p = [d.point(0, 0), d.point(0, 4), d.point(3, 4), d.point(3, 0)]
    p.forEach((a, i) => d.line(a, p[(i + 1) % 4]))
    const r = profiles(d.entities, d.constraints).regions[0]
    expect(r.outer.area).toBeCloseTo(12, 9)
    expect(r.outer.segs).toHaveLength(4)
  })

  describe('shared and touching edges', () => {
    const regionsOf = (d: Draft) => profiles(d.entities, d.constraints)
    const areas = (d: Draft) => regionsOf(d).regions.map((r) => Math.round(r.outer.area * 1e6) / 1e6).sort((a, b) => a - b)

    it('three rectangles touching along partial edges (T-junctions) are three filled regions', () => {
      const d = new Draft(); addRect(d, 0, 0, 10, 10); addRect(d, 10, 6, 5, 4); addRect(d, 10, 0, 5, 4)
      expect(areas(d)).toEqual([20, 20, 100])
      expect(profileArea(regionsOf(d))).toBeCloseTo(140, 9)
      expect(regionsOf(d).open).toEqual([])
    })

    it('rectangles sharing a whole edge with their own corner points', () => {
      const d = new Draft(); addRect(d, 0, 0, 10, 10); addRect(d, 10, 0, 6, 10); addRect(d, 16, 0, 4, 10)
      expect(areas(d)).toEqual([40, 60, 100])
    })

    it('a T-junction from a smaller rect that sits on the middle of a long edge', () => {
      const d = new Draft(); addRect(d, 0, 0, 20, 10); addRect(d, 5, 10, 5, 5)
      expect(areas(d)).toEqual([25, 200])
    })

    it('overlapping rectangles make three regions', () => {
      const d = new Draft(); addRect(d, 0, 0, 10, 10); addRect(d, 5, 5, 10, 10)
      expect(areas(d)).toEqual([25, 75, 75])
      expect(profileArea(regionsOf(d))).toBeCloseTo(175, 9)
    })

    it('a circle crossing a rectangle is split into regions that add up', () => {
      const d = new Draft(); addRect(d, 0, 0, 10, 10); addCircle(d, 10, 5, 3)
      const p = regionsOf(d)
      expect(p.regions).toHaveLength(3)
      expect(profileArea(p)).toBeCloseTo(100 + (Math.PI * 9) / 2, 6) // the circle is centred on the edge: half of it sticks out
    })

    it('a circle that touches nothing stays a loop and keeps nesting', () => {
      const d = new Draft(); addRect(d, -20, -20, 40, 40); addCircle(d, 0, 0, 10)
      const p = regionsOf(d)
      expect(p.regions).toHaveLength(1)
      expect(p.regions[0].holes).toHaveLength(1)
    })

    it('duplicate lines drawn on top of each other do not break the regions', () => {
      const d = new Draft(); addRect(d, 0, 0, 10, 10); addRect(d, 0, 0, 10, 10)
      expect(areas(d)).toEqual([100])
    })

    it('open curves are still reported as open', () => {
      const d = new Draft(); addRect(d, 0, 0, 10, 10)
      d.line(d.point(10, 5), d.point(20, 5))
      const p = regionsOf(d)
      expect(p.regions).toHaveLength(1)
      expect(p.open).toHaveLength(1)
    })
  })
})
