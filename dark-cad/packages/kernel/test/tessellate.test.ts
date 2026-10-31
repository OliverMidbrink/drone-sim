import { it, expect } from 'vitest'
import { make } from '../src/occ'
import { loadOCC } from '../src/node'
import { tessellate } from '../src/tessellate'

it('tessellates a box: 6 face ranges, 12 triangles, outward normals, 12 edges', async () => {
  const oc = await loadOCC()
  const box = make(oc, 'BRepPrimAPI_MakeBox', 2, 3, 4).Shape()
  const m = tessellate(oc, box)
  expect(m.faceRanges.length).toBe(6)
  expect(m.indices.length / 3).toBe(12)
  expect(m.edges.length / 6).toBe(12)
  // Volume from the mesh (divergence theorem) must be +24 if winding is outward.
  let v = 0
  const p = m.positions, ix = m.indices
  for (let i = 0; i < ix.length; i += 3) {
    const a = ix[i] * 3, b = ix[i + 1] * 3, c = ix[i + 2] * 3
    v += (p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) - p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c]) + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c])) / 6
  }
  expect(v).toBeCloseTo(24, 4)
})

it('tessellates a cylinder with curved faces and positive volume', async () => {
  const oc = await loadOCC()
  const cyl = make(oc, 'BRepPrimAPI_MakeCylinder', 5, 10).Shape()
  const m = tessellate(oc, cyl, 0.01)
  let v = 0
  const p = m.positions, ix = m.indices
  for (let i = 0; i < ix.length; i += 3) {
    const a = ix[i] * 3, b = ix[i + 1] * 3, c = ix[i + 2] * 3
    v += (p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) - p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c]) + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c])) / 6
  }
  expect(v).toBeGreaterThan(Math.PI * 25 * 10 * 0.99)
  expect(v).toBeLessThan(Math.PI * 25 * 10 * 1.001)
})
