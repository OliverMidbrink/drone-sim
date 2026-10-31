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
