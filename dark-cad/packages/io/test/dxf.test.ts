import { describe, it, expect } from 'vitest'
import { sketchToDxf, dxfToSketch } from '../src/dxf'
import { Draft, addRect, addSlot, addCircle, buildGeo, profiles, profileArea, arcSweep, solve, type SketchModel } from '@darkcad/sketch'

const area = (m: SketchModel) => profileArea(profiles(m.entities, m.constraints))
const lw = (rows: (string | number)[]) => ['0', 'SECTION', '2', 'ENTITIES', ...rows.map(String), '0', 'ENDSEC', '0', 'EOF'].join('\n')

describe('dxf', () => {
  it('round-trips a rectangle with a circular hole', () => {
    const d = new Draft(); addRect(d, -10, -5, 20, 10); addCircle(d, 1, 1, 3) // clear of the edges: a tangent circle would split the rectangle into two regions
    const back = dxfToSketch(sketchToDxf(d.entities))
    const g = buildGeo(back.entities).real
    expect(g.filter((c) => c.kind === 'line')).toHaveLength(4)
    expect(g.find((c) => c.kind === 'circle')).toMatchObject({ c: [1, 1], r: 3 })
    expect(back.entities.filter((e) => e.kind === 'point').length).toBe(5) // four shared corners + the circle centre
    expect(area(back)).toBeCloseTo(200 - Math.PI * 9, 6)
  })

  it('keeps arcs as arcs through a slot round-trip', () => {
    const d = new Draft(); addSlot(d, [0, 0], [20, 0], 8)
    const text = sketchToDxf(d.entities)
    expect(text).toContain('ARC')
    const back = dxfToSketch(text)
    const g = buildGeo(back.entities).real
    const arcs = g.filter((c) => c.kind === 'arc') as any[]
    expect(arcs).toHaveLength(2)
    for (const a of arcs) { expect(a.r).toBeCloseTo(4, 6); expect(arcSweep(a)).toBeCloseTo(Math.PI, 6) }
    expect(g.filter((c) => c.kind === 'line')).toHaveLength(2)
    expect(area(back)).toBeCloseTo(20 * 8 + Math.PI * 16, 4)
    expect(solve(back.entities, back.constraints).ok).toBe(true)
  })

  it('exports construction geometry on its own layer and restores it', () => {
    const d = new Draft(); const a = d.point(0, 0), b = d.point(5, 0); d.line(a, b, true)
    const text = sketchToDxf(d.entities)
    expect(text).toContain('CONSTRUCTION')
    const back = dxfToSketch(text)
    const l = buildGeo(back.entities).curves.find((c) => c.kind === 'line' && !c.virtual)!
    expect(l.construction).toBe(true)
  })

  it('merges coincident LINE ends regardless of order and direction', () => {
    const line = (a: number[], b: number[]) => ['0', 'LINE', '8', '0', '10', a[0], '20', a[1], '11', b[0], '21', b[1]]
    const back = dxfToSketch(lw([...line([4, 0], [0, 0]), ...line([0, 3], [4, 0]), ...line([0, 0], [0, 3])]))
    expect(back.entities.filter((e) => e.kind === 'point')).toHaveLength(3)
    expect(back.entities.filter((e) => e.kind === 'line')).toHaveLength(3)
    expect(area(back)).toBeCloseTo(6, 9)
  })

  it('reads a closed LWPOLYLINE', () => {
    const back = dxfToSketch(lw(['0', 'LWPOLYLINE', '70', '1', '10', '0', '20', '0', '10', '5', '20', '0', '10', '5', '20', '5']))
    expect(back.entities.filter((e) => e.kind === 'line')).toHaveLength(3)
    expect(area(back)).toBeCloseTo(12.5, 9)
  })

  it('turns LWPOLYLINE bulges into arcs (semicircle, both directions)', () => {
    // A D shape: line from (0,0) to (10,0), then a ccw semicircle bulge=1 from (10,0) back to (10,10)... closed by a line.
    const back = dxfToSketch(lw(['0', 'LWPOLYLINE', '70', '1', '10', '0', '20', '0', '10', '10', '20', '0', '42', '1', '10', '10', '20', '10']))
    const arcs = buildGeo(back.entities).real.filter((c) => c.kind === 'arc') as any[]
    expect(arcs).toHaveLength(1)
    expect(arcs[0].r).toBeCloseTo(5, 9)
    expect(arcs[0].c[0]).toBeCloseTo(10, 9); expect(arcs[0].c[1]).toBeCloseTo(5, 9)
    expect(arcSweep(arcs[0])).toBeCloseTo(Math.PI, 9)
    expect(area(back)).toBeCloseTo(50 + (Math.PI * 25) / 2, 6) // the triangle (0,0)-(10,0)-(10,10) plus the bulge
    // Negative bulge bows the other way (to the left of the chord) but shares the same centre for a semicircle.
    const neg = dxfToSketch(lw(['0', 'LWPOLYLINE', '70', '1', '10', '0', '20', '0', '10', '10', '20', '0', '42', '-1', '10', '10', '20', '10']))
    const narcs = buildGeo(neg.entities).real.filter((c) => c.kind === 'arc') as any[]
    expect(narcs[0].c[0]).toBeCloseTo(10, 9); expect(narcs[0].c[1]).toBeCloseTo(5, 9)
    expect(arcSweep(narcs[0])).toBeCloseTo(Math.PI, 9)
    // clockwise p→q is counter-clockwise q→p: the arc starts at (10,10) and ends at (10,0)
    expect(narcs[0].a0).toBeCloseTo(Math.PI / 2, 9)
  })

  it('reads old-style POLYLINE with bulge vertices', () => {
    const v = (x: number, y: number, b = 0) => ['0', 'VERTEX', '10', x, '20', y, ...(b ? ['42', b] : [])]
    const back = dxfToSketch(lw(['0', 'POLYLINE', '70', '1', ...v(0, 0), ...v(4, 0, 1), ...v(4, 4), ...v(0, 4), '0', 'SEQEND']))
    expect(buildGeo(back.entities).real.filter((c) => c.kind === 'arc')).toHaveLength(1)
    expect(buildGeo(back.entities).real.filter((c) => c.kind === 'line')).toHaveLength(3)
  })

  it('a full-turn ARC is a circle, and unknown entities are ignored', () => {
    const back = dxfToSketch(lw(['0', 'ARC', '10', '0', '20', '0', '40', '5', '50', '0', '51', '360', '0', 'SPLINE', '10', '1', '20', '1']))
    expect(buildGeo(back.entities).real.map((c) => c.kind)).toEqual(['circle'])
  })

  it('reads a partial ARC with the right end points', () => {
    const back = dxfToSketch(lw(['0', 'ARC', '10', '1', '20', '2', '40', '3', '50', '90', '51', '180']))
    const [a] = buildGeo(back.entities).real as any[]
    expect(a.kind).toBe('arc')
    expect(a.c).toEqual([1, 2])
    expect(a.r).toBeCloseTo(3, 9)
    expect(arcSweep(a)).toBeCloseTo(Math.PI / 2, 9)
  })
})
