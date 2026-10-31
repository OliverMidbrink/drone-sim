import { call, make } from './occ'

export function faces(oc: any, shape: any): any[] {
  const out: any[] = []
  const exp = make(oc, 'TopExp_Explorer', shape, oc.TopAbs_ShapeEnum.TopAbs_FACE, oc.TopAbs_ShapeEnum.TopAbs_SHAPE)
  for (; exp.More(); exp.Next()) out.push(call(oc.TopoDS, 'Face', exp.Current()))
  return out
}


export function zRange(oc: any, shape: any): [number, number] {
  const b = make(oc, 'Bnd_Box')
  call(oc.BRepBndLib, 'Add', shape, b, false)
  return [b.CornerMin().Z(), b.CornerMax().Z()]
}

export function volume(oc: any, shape: any): number {
  const g = make(oc, 'GProp_GProps')
  call(oc.BRepGProp, 'VolumeProperties', shape, g, false, false, false)
  return g.Mass()
}

export function isValid(oc: any, shape: any): boolean {
  const a = make(oc, 'BRepCheck_Analyzer', shape, true)
  return call(a, 'IsValid')
}

export function countFaces(oc: any, shape: any): number { return faces(oc, shape).length }

/** Tight bounds: mesh first, then box the triangulation (the geometric box is loose on B-splines). */
export function tightBounds(oc: any, shape: any, deflection = 0.005) {
  make(oc, 'BRepMesh_IncrementalMesh', shape, deflection, false, 0.2, false)
  const b = make(oc, 'Bnd_Box')
  call(oc.BRepBndLib, 'Add', shape, b, true)
  const lo = b.CornerMin(), hi = b.CornerMax()
  return { min: [lo.X(), lo.Y(), lo.Z()] as const, max: [hi.X(), hi.Y(), hi.Z()] as const }
}

/** Move a shape built in the local frame (Z axis, X to the right) onto a frame at `origin` with normal `dir` and optional x direction. */
export function placeFrame(oc: any, shape: any, origin: [number, number, number], dir: [number, number, number], xdir?: [number, number, number]) {
  const to = xdir
    ? make(oc, 'gp_Ax3', make(oc, 'gp_Pnt', ...origin), make(oc, 'gp_Dir', ...dir), make(oc, 'gp_Dir', ...xdir))
    : make(oc, 'gp_Ax3', make(oc, 'gp_Pnt', ...origin), make(oc, 'gp_Dir', ...dir))
  const trsf = make(oc, 'gp_Trsf')
  call(trsf, 'SetTransformation', to, make(oc, 'gp_Ax3')) // coordinates relative to `to` → global
  return call(make(oc, 'BRepBuilderAPI_Transform', shape, trsf, true), 'Shape')
}
