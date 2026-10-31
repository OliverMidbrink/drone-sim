import { describe, it, expect } from 'vitest'
import { sketchToDxf, dxfToSketch } from '../src/dxf'
import type { SketchEntity } from '@darkcad/core'

describe('dxf', () => {
  it('round-trips a rectangle with a circular hole', () => {
    const src: SketchEntity[] = [{ id: 'a', kind: 'rect', x: -10, y: -5, w: 20, h: 10 }, { id: 'b', kind: 'circle', x: 1, y: 2, r: 3 }]
    const back = dxfToSketch(sketchToDxf(src))
    expect(back.length).toBe(2)
    const rect = back.find((e) => e.kind === 'rect')!
    expect(rect).toMatchObject({ x: -10, y: -5, w: 20, h: 10 })
    expect(back.find((e) => e.kind === 'circle')).toMatchObject({ x: 1, y: 2, r: 3 })
  })

  it('chains out-of-order, reversed LINEs into a polygon', () => {
    const line = (a: number[], b: number[]) => ['0', 'LINE', '10', a[0], '20', a[1], '11', b[0], '21', b[1]].join('\n')
    const text = ['0', 'SECTION', '2', 'ENTITIES', line([4, 0], [0, 0]), line([0, 3], [4, 0]), line([0, 0], [0, 3]), '0', 'ENDSEC', '0', 'EOF'].join('\n')
    const [poly] = dxfToSketch(text)
    expect(poly.kind).toBe('polygon')
    expect((poly as any).points.length).toBe(3)
  })

  it('reads closed LWPOLYLINE and ignores open chains', () => {
    const lw = ['0', 'LWPOLYLINE', '70', '1', '10', '0', '20', '0', '10', '5', '20', '0', '10', '5', '20', '5'].join('\n')
    const open = ['0', 'LINE', '10', '0', '20', '0', '11', '1', '21', '1'].join('\n')
    const text = ['0', 'SECTION', '2', 'ENTITIES', lw, open, '0', 'ENDSEC', '0', 'EOF'].join('\n')
    const r = dxfToSketch(text)
    expect(r.length).toBe(1)
    expect(r[0].kind).toBe('polygon')
  })

  it('an ARC of a full circle becomes a closed polygon', () => {
    const text = ['0', 'SECTION', '2', 'ENTITIES', '0', 'ARC', '10', '0', '20', '0', '40', '5', '50', '0', '51', '360', '0', 'ENDSEC', '0', 'EOF'].join('\n')
    const [p] = dxfToSketch(text)
    expect(p.kind).toBe('polygon')
  })
})
