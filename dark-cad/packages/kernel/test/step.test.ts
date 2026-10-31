import { it, expect } from 'vitest'
import { make } from '../src/occ'
import { loadOCC } from '../src/node'
import { writeSTEP, readSTEP } from '../src/step'
import { faces, volume, isValid } from '../src/shape'

it('STEP round trip preserves volume, face count and validity', async () => {
  const oc = await loadOCC()
  const box = make(oc, 'BRepPrimAPI_MakeBox', 20, 10, 5).Shape()
  const cyl = make(oc, 'BRepPrimAPI_MakeCylinder', 3, 20).Shape()
  const src = make(oc, 'BRepAlgoAPI_Cut', box, cyl).Shape()
  const bytes = writeSTEP(oc, src)
  expect(new TextDecoder().decode(bytes.slice(0, 20))).toContain('ISO-10303-21')
  const back = readSTEP(oc, bytes)
  expect(isValid(oc, back)).toBe(true)
  expect(volume(oc, back)).toBeCloseTo(volume(oc, src), 3)
  expect(faces(oc, back).length).toBe(faces(oc, src).length)
})
