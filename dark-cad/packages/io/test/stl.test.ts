import { it, expect } from 'vitest'
import { meshesToStl } from '../src/stl'

it('writes a valid binary STL with correct counts and a unit normal', () => {
  const m = { positions: [0, 0, 0, 1, 0, 0, 0, 1, 0], indices: [0, 1, 2] }
  const out = meshesToStl([m, m])
  expect(out.length).toBe(84 + 2 * 50)
  const dv = new DataView(out.buffer)
  expect(dv.getUint32(80, true)).toBe(2)
  expect(dv.getFloat32(84 + 8, true)).toBeCloseTo(1, 6) // nz of first triangle
})
