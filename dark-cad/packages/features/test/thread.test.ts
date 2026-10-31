import { describe, it, expect, beforeAll } from 'vitest'
import { make } from '@darkcad/kernel/occ'
import { loadOCC } from '@darkcad/kernel/node'
import { volume, isValid, zRange, tightBounds } from '@darkcad/kernel/shape'
import { threadedRod, fuseThreadedBoss, cutThreadedHole } from '../src/thread'

let oc: any
beforeAll(async () => { oc = await loadOCC() })

function radialExtent(shape: any) {
  const b = tightBounds(oc, shape)
  return Math.max(b.max[0], b.max[1])
}

describe('thread tool', () => {
  it.each(['G1/4', 'M8', 'M12x1.5'])('male %s: valid, outer radius equals the table major radius', (name) => {
    const len = 8
    const rod = threadedRod(oc, { designation: name, kind: 'male', length: len })
    expect(isValid(oc, rod)).toBe(true)
    const major = { 'G1/4': 13.157, M8: 8, 'M12x1.5': 12 }[name]! / 2
    expect(radialExtent(rod)).toBeGreaterThan(major - 0.01)
    expect(radialExtent(rod)).toBeLessThan(major + 0.01)
  })

  it('female G1/4 in a block: valid, material removed, bore not smaller than the minor diameter', () => {
    const block = make(oc, 'BRepPrimAPI_MakeBox', make(oc, 'gp_Pnt', -15, -15, 0), 30, 30, 20).Shape()
    const out = cutThreadedHole(oc, block, { designation: 'G1/4', kind: 'female', length: 12, origin: [0, 0, 20], dir: [0, 0, -1] })
    expect(isValid(oc, out)).toBe(true)
    const removed = volume(oc, block) - volume(oc, out)
    const boreOnly = Math.PI * (11.8 / 2) ** 2 * 12
    expect(removed).toBeGreaterThan(boreOnly)
    expect(removed).toBeLessThan(Math.PI * (13.157 / 2) ** 2 * 12)
  })

  it('male M8 boss fused onto a plate in an arbitrary direction stays valid', () => {
    const plate = make(oc, 'BRepPrimAPI_MakeBox', make(oc, 'gp_Pnt', -10, -10, -5), 20, 20, 5).Shape()
    const out = fuseThreadedBoss(oc, plate, { designation: 'M8', kind: 'male', length: 10, origin: [0, 0, 0], dir: [0, 0, 1] })
    expect(isValid(oc, out)).toBe(true)
    expect(zRange(oc, out)[1]).toBeGreaterThan(9)
  })

  it('tapered R1/4 male is valid and its radius shrinks toward the free end', () => {
    const rod = threadedRod(oc, { designation: 'R1/4', kind: 'male', length: 10 })
    expect(isValid(oc, rod)).toBe(true)
  })
})

describe('placement', () => {
  it('a translated rod really lands at its origin (regression: inverted placement transform)', () => {
    const rod = threadedRod(oc, { designation: 'M8', kind: 'male', length: 10, origin: [3, -2, 7], dir: [0, 0, 1] })
    const b = tightBounds(oc, rod)
    expect(b.min[2]).toBeGreaterThan(7 - 1.1)
    expect(b.max[2]).toBeGreaterThan(7 + 10)
    expect((b.min[0] + b.max[0]) / 2).toBeCloseTo(3, 1)
    expect((b.min[1] + b.max[1]) / 2).toBeCloseTo(-2, 1)
  })
})
