import { call, make } from '@darkcad/kernel/occ'
import { faces } from '@darkcad/kernel/shape'

function edgesOf(oc: any, face: any): any[] {
  const out: any[] = []
  const ex = make(oc, 'TopExp_Explorer', face, oc.TopAbs_ShapeEnum.TopAbs_EDGE, oc.TopAbs_ShapeEnum.TopAbs_SHAPE)
  for (; ex.More(); ex.Next()) out.push(call(oc.TopoDS, 'Edge', ex.Current()))
  return out
}

function area(oc: any, face: any): number {
  const g = make(oc, 'GProp_GProps')
  call(oc.BRepGProp, 'SurfaceProperties', face, g, false, false)
  return g.Mass()
}

function isPlane(oc: any, face: any): boolean {
  return make(oc, 'BRepAdaptor_Surface', face, true).GetType() === oc.GeomAbs_SurfaceType.GeomAbs_Plane
}

/**
 * Faces making up the protrusion the seed face belongs to. Grows across shared edges (so blends, tops and
 * side walls all join in) and never enters a planar face much larger than the seed ("base-like"), or an
 * explicitly given base face. Returns indices into `faces(oc, shape)`.
 */
export function selectProtrusion(oc: any, shape: any, seedIndex: number, baseIndex?: number, baseRatio = 4): number[] {
  const fs = faces(oc, shape)
  const edges = fs.map((f) => edgesOf(oc, f))
  const adjacent = (i: number, j: number) => edges[i].some((e) => edges[j].some((g) => e.IsSame(g)))
  const seedArea = area(oc, fs[seedIndex])
  const baseLike = fs.map((f, i) => i === baseIndex || (baseIndex === undefined && i !== seedIndex && isPlane(oc, f) && area(oc, f) > seedArea * baseRatio))
  if (baseLike[seedIndex]) throw new Error('seed face is part of the base, not a protrusion')
  const seen = new Set([seedIndex])
  const queue = [seedIndex]
  while (queue.length) {
    const i = queue.shift()!
    for (let j = 0; j < fs.length; j++) if (!seen.has(j) && !baseLike[j] && adjacent(i, j)) { seen.add(j); queue.push(j) }
  }
  return [...seen]
}

export interface DefeatureResult { shape: any; removed: number }

/** Delete a protrusion: selects it from one picked face, removes it and heals the gap with the neighbouring faces. */
export function deleteFeature(oc: any, shape: any, seedIndex: number, baseIndex?: number): DefeatureResult {
  const fs = faces(oc, shape)
  const picked = selectProtrusion(oc, shape, seedIndex, baseIndex)
  if (picked.length === fs.length) throw new Error('selection covers the whole solid')
  const df = make(oc, 'BRepAlgoAPI_Defeaturing')
  call(df, 'SetShape', shape)
  for (const i of picked) call(df, 'AddFaceToRemove', fs[i])
  call(df, 'Build')
  const failed = typeof df.HasErrors === 'function' ? df.HasErrors() : !df.IsDone()
  if (failed) throw new Error('defeaturing failed for this selection')
  return { shape: df.Shape(), removed: picked.length }
}
