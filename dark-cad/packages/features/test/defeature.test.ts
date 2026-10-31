import { describe, it, expect, beforeAll } from 'vitest'
import { make, call } from '@darkcad/kernel/occ'
import { loadOCC } from '@darkcad/kernel/node'
import { faces, volume, isValid, zRange } from '@darkcad/kernel/shape'
import { deleteFeature, selectProtrusion } from '../src/defeature'

let oc: any
beforeAll(async () => { oc = await loadOCC() })

function plateWithBoss(filletRadius = 0) {
  const plate = make(oc, 'BRepPrimAPI_MakeBox', make(oc, 'gp_Pnt', -20, -20, 0), 40, 40, 5).Shape()
  const boss = make(oc, 'BRepPrimAPI_MakeCylinder', make(oc, 'gp_Ax2', make(oc, 'gp_Pnt', 0, 0, 5), make(oc, 'gp_Dir', 0, 0, 1)), 5, 8).Shape()
  let s = call(make(oc, 'BRepAlgoAPI_Fuse', plate, boss), 'Shape')
  if (filletRadius) {
    const mk = make(oc, 'BRepFilletAPI_MakeFillet', s, oc.ChFi3d_FilletShape.ChFi3d_Rational)
    const ex = make(oc, 'TopExp_Explorer', s, oc.TopAbs_ShapeEnum.TopAbs_EDGE, oc.TopAbs_ShapeEnum.TopAbs_SHAPE)
    for (; ex.More(); ex.Next()) {
      const e = call(oc.TopoDS, 'Edge', ex.Current())
      const bb = make(oc, 'Bnd_Box'); call(oc.BRepBndLib, 'Add', e, bb, false)
      const [a, b] = zRange(oc, e), w = bb.CornerMax().X() - bb.CornerMin().X()
      if (Math.abs(a - 5) < 1e-6 && Math.abs(b - 5) < 1e-6 && w > 9 && w < 11) { call(mk, 'Add', filletRadius, e); break }
    }
    s = mk.Shape()
  }
  return s
}

const sideOfBoss = (s: any) => faces(oc, s).findIndex((f) => zRange(oc, f)[1] > 12.9 && zRange(oc, f)[0] < 7 && zRange(oc, f)[0] > 4.9)

describe('feature delete', () => {
  it('selects the whole boss (side + top) but not the plate, from the side face', () => {
    const s = plateWithBoss()
    const picked = selectProtrusion(oc, s, sideOfBoss(s))
    expect(picked.length).toBe(2)
  })

  it('selects boss + fillet blend from any boss face', () => {
    const s = plateWithBoss(1.5)
    expect(selectProtrusion(oc, s, sideOfBoss(s)).length).toBe(3)
  })

  it('deleting a boss restores the flat plate', () => {
    const s = plateWithBoss()
    const r = deleteFeature(oc, s, sideOfBoss(s))
    expect(isValid(oc, r.shape)).toBe(true)
    expect(volume(oc, r.shape)).toBeCloseTo(8000, 3)
    expect(faces(oc, r.shape).length).toBe(6)
  })

  it('deleting a filleted boss restores the flat plate', () => {
    const s = plateWithBoss(1.5)
    const r = deleteFeature(oc, s, sideOfBoss(s))
    expect(volume(oc, r.shape)).toBeCloseTo(8000, 3)
  })

  it('refuses to treat the base face as a feature', () => {
    const s = plateWithBoss()
    const top = faces(oc, s).findIndex((f) => Math.abs(zRange(oc, f)[0] - 5) < 1e-3 && Math.abs(zRange(oc, f)[1] - 5) < 1e-3)
    expect(top).toBeGreaterThanOrEqual(0)
    expect(() => selectProtrusion(oc, s, top, top)).toThrow(/base/)
  })
})
