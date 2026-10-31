import { describe, it, expect } from 'vitest'
import { loadOCC } from '../src/node'

describe('helix on a cylinder', () => {
  it('rises exactly one pitch per turn, independent of radius', async () => {
    const oc = await loadOCC()
    for (const r of [3, 6.5785]) {
      const pitch = 1.337, turns = 5
      const surf = new oc.Geom_CylindricalSurface_1(new oc.gp_Ax3_1(), r)
      const dir = new oc.gp_Dir2d_4(2 * Math.PI, pitch)
      const line = new oc.Geom2d_Line_3(new oc.gp_Pnt2d_3(0, 0), dir)
      // gp_Dir2d is normalised, so the line parameter is arclength in (u, v) space.
      const tEnd = turns * Math.hypot(2 * Math.PI, pitch)
      const edge = new oc.BRepBuilderAPI_MakeEdge_31(
        new oc.Handle_Geom2d_Curve_2(line), new oc.Handle_Geom_Surface_2(surf), 0, tEnd).Edge()
      oc.BRepLib.BuildCurves3d_2(edge)
      const ad = new oc.BRepAdaptor_Curve_2(edge)
      const p = ad.Value(ad.LastParameter())
      expect(p.Z()).toBeCloseTo(turns * pitch, 4)
      expect(Math.hypot(p.X(), p.Y())).toBeCloseTo(r, 4)
    }
  })
})
