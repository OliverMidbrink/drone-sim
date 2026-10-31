import { describe, it, expect, beforeAll } from 'vitest'
import { make, call } from '../src/occ'
import { loadOCC } from '../src/node'
import { threadRidge } from '../src/thread'
import { faces, zRange, volume, isValid } from '../src/shape'

let oc: any
beforeAll(async () => { oc = await loadOCC() })

const G14 = { pitch: 1.337, major: 13.157, minor: 11.445 }

describe('M0 spike: threads', () => {
  it('20-turn G 1/4 male thread: valid and timed', () => {
    const len = 20 * G14.pitch
    const ridge = threadRidge(oc, { pitch: G14.pitch, rMajor: G14.major / 2, rMinor: G14.minor / 2, angleDeg: 55, length: len })
    const core = make(oc, 'BRepPrimAPI_MakeCylinder', G14.minor / 2, len + 2 * G14.pitch).Shape()
    const t0 = performance.now()
    const out = make(oc, 'BRepAlgoAPI_Fuse', core, ridge).Shape()
    const ms = performance.now() - t0
    console.log('M0 male 20 turns fuse ms', Math.round(ms), 'valid', isValid(oc, out))
    expect(isValid(oc, out)).toBe(true)
    expect(ms).toBeLessThan(20000)
  })

  it('G 1/4 female thread: groove cut from a bored block is valid', () => {
    const len = 8 * G14.pitch
    const groove = threadRidge(oc, { pitch: G14.pitch, rMajor: G14.major / 2, rMinor: G14.minor / 2, angleDeg: 55, length: len })
    const block = make(oc, 'BRepPrimAPI_MakeCylinder', 10, len + 2 * G14.pitch).Shape()
    const bore = make(oc, 'BRepPrimAPI_MakeCylinder', G14.minor / 2, len + 2 * G14.pitch).Shape()
    const sleeve = make(oc, 'BRepAlgoAPI_Cut', block, bore).Shape()
    const t0 = performance.now()
    const out = make(oc, 'BRepAlgoAPI_Cut', sleeve, groove).Shape()
    console.log('M0 female 8 turns cut ms', Math.round(performance.now() - t0), 'valid', isValid(oc, out))
    expect(isValid(oc, out)).toBe(true)
    expect(volume(oc, out)).toBeLessThan(volume(oc, sleeve))
  })

  it('BSPT taper 1:16: helix advances one pitch per turn along the axis; radius follows the cone', () => {
    const pitch = G14.pitch, turns = 6, taper = 1 / 16, r0 = G14.major / 2 - 0.5
    const alpha = Math.atan(taper / 2)
    const surf = make(oc, 'Geom_ConicalSurface', make(oc, 'gp_Ax3'), alpha, r0)
    const slant = pitch / Math.cos(alpha)
    const line = make(oc, 'Geom2d_Line', make(oc, 'gp_Pnt2d', 0, 0), make(oc, 'gp_Dir2d', 2 * Math.PI, slant))
    const edge = make(oc, 'BRepBuilderAPI_MakeEdge', new oc.Handle_Geom2d_Curve_2(line), new oc.Handle_Geom_Surface_2(surf), 0, turns * Math.hypot(2 * Math.PI, slant)).Edge()
    oc.BRepLib.BuildCurves3d_2(edge)
    const ad = make(oc, 'BRepAdaptor_Curve', edge)
    const p = ad.Value(ad.LastParameter())
    expect(p.Z()).toBeCloseTo(turns * pitch, 3)
    expect(Math.hypot(p.X(), p.Y())).toBeCloseTo(r0 + p.Z() * (taper / 2), 3)
  })

  it('BSPT tapered thread ridge builds a valid solid', () => {
    const len = 6 * G14.pitch
    const ridge = threadRidge(oc, { pitch: G14.pitch, rMajor: G14.major / 2, rMinor: G14.minor / 2, angleDeg: 55, length: len, taper: 1 / 16 })
    expect(isValid(oc, ridge)).toBe(true)
  })
})

const failed = (df: any) => (typeof df.HasErrors === 'function' ? df.HasErrors() : !df.IsDone())

describe('M0 spike: defeature', () => {
  function plateWithBoss(fillet: boolean) {
    const plate = make(oc, 'BRepPrimAPI_MakeBox', make(oc, 'gp_Pnt', -20, -20, 0), 40, 40, 5).Shape()
    const boss = make(oc, 'BRepPrimAPI_MakeCylinder', make(oc, 'gp_Ax2', make(oc, 'gp_Pnt', 0, 0, 5), make(oc, 'gp_Dir', 0, 0, 1)), 5, 8).Shape()
    let s = make(oc, 'BRepAlgoAPI_Fuse', plate, boss).Shape()
    if (fillet) {
      const mk = make(oc, 'BRepFilletAPI_MakeFillet', s, oc.ChFi3d_FilletShape.ChFi3d_Rational)
      const ex = make(oc, 'TopExp_Explorer', s, oc.TopAbs_ShapeEnum.TopAbs_EDGE, oc.TopAbs_ShapeEnum.TopAbs_SHAPE)
      for (; ex.More(); ex.Next()) {
        const e = call(oc.TopoDS, 'Edge', ex.Current())
        const [a, b] = zRange(oc, e)
        const bb = make(oc, 'Bnd_Box'); call(oc.BRepBndLib, 'Add', e, bb, false)
        const w = bb.CornerMax().X() - bb.CornerMin().X()
        if (Math.abs(a - 5) < 1e-6 && Math.abs(b - 5) < 1e-6 && w > 9 && w < 11) { call(mk, 'Add', 1.5, e); break }
      }
      s = mk.Shape()
    }
    return s
  }
  function defeature(shape: any) {
    const df = make(oc, 'BRepAlgoAPI_Defeaturing')
    call(df, 'SetShape', shape)
    for (const f of faces(oc, shape)) if (zRange(oc, f)[1] > 5 + 1e-6) call(df, 'AddFaceToRemove', f)
    call(df, 'Build')
    return df
  }
  it('removes a boss from a flat plate, healing the top face', () => {
    const df = defeature(plateWithBoss(false))
    expect(failed(df)).toBe(false)
    const out = df.Shape()
    expect(isValid(oc, out)).toBe(true)
    expect(volume(oc, out)).toBeCloseTo(40 * 40 * 5, 3)
    expect(faces(oc, out).length).toBe(6)
  })
  it('removes a boss that has a fillet blend at its base', () => {
    const df = defeature(plateWithBoss(true))
    expect(failed(df)).toBe(false)
    const out = df.Shape()
    expect(isValid(oc, out)).toBe(true)
    expect(volume(oc, out)).toBeCloseTo(40 * 40 * 5, 3)
    expect(faces(oc, out).length).toBe(6)
  })
})
