import { describe, it, expect } from 'vitest'
import { standards, threadDims, findSize, listDesignations, tapDrill } from '../src/index'

describe('thread tables', () => {
  it('every size is well formed and designations are unique', () => {
    const names = listDesignations()
    expect(new Set(names).size).toBe(names.length)
    for (const s of standards) for (const z of s.sizes) {
      expect(z.pitch).toBeGreaterThan(0)
      expect(z.major).toBeGreaterThan(z.pitch)
      if (z.tpi) expect(z.pitch).toBeCloseTo(25.4 / z.tpi, 3)
    }
  })

  it('metric minor diameters follow ISO 68-1 (d3 = d − 1.22687·P)', () => {
    const m8 = threadDims('M8', 'male')
    expect(m8.rMinor * 2).toBeCloseTo(8 - 1.22687 * 1.25, 3)
    expect(threadDims('M8', 'female').rMinor * 2).toBeCloseTo(8 - 1.08253 * 1.25, 3)
  })

  it('metric tap drills match the common chart', () => {
    const want: Record<string, number> = { M3: 2.5, M4: 3.3, M5: 4.2, M6: 5, M8: 6.8, M10: 8.5, M12: 10.2 }
    for (const [d, t] of Object.entries(want)) expect(tapDrill(findSize(d)!.size)).toBeCloseTo(t, 6)
  })

  it('G 1/4 matches ISO 228-1 basic dimensions', () => {
    const g = threadDims('G 1/4', 'male')
    expect(g.rMajor * 2).toBeCloseTo(13.157, 3)
    expect(g.rMinor * 2).toBeCloseTo(11.445, 2)
    expect(g.pitch).toBeCloseTo(1.337, 3)
    expect(g.angleDeg).toBe(55)
  })

  it('male cylinder resizes to the major diameter, female bore to the tap drill', () => {
    expect(threadDims('G1/4', 'male').rCylinder * 2).toBeCloseTo(13.157, 3)
    expect(threadDims('G1/4', 'female').rCylinder * 2).toBeCloseTo(11.8, 3)
    expect(threadDims('M12x1.5', 'female').rCylinder * 2).toBeCloseTo(10.5, 3)
  })

  it('BSPT carries the 1:16 taper', () => {
    expect(threadDims('R1/4', 'male').taper).toBeCloseTo(1 / 16, 6)
    expect(threadDims('G1/4', 'male').taper).toBe(0)
  })

  it('rejects unknown designations', () => {
    expect(() => threadDims('M99', 'male')).toThrow(/unknown thread/)
  })
})
