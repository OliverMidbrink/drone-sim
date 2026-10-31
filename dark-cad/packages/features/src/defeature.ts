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

export interface FaceFingerprint { surface: string; centroid: [number, number, number]; area: number }

export function fingerprintFace(oc: any, face: any): FaceFingerprint {
  const g = make(oc, 'GProp_GProps')
  call(oc.BRepGProp, 'SurfaceProperties', face, g, false, false)
  const c = g.CentreOfMass()
  const t = make(oc, 'BRepAdaptor_Surface', face, true).GetType()
  const surface = Object.entries(oc.GeomAbs_SurfaceType).find(([, v]) => v === t)?.[0].replace('GeomAbs_', '') ?? 'Other'
  return { surface, centroid: [c.X(), c.Y(), c.Z()], area: g.Mass() }
}

/** Re-find a face after upstream edits: same surface type, nearest centroid, similar area. Throws if nothing is close. */
export function resolveFace(oc: any, shape: any, ref: FaceFingerprint, tol = 0.5): number {
  let best = -1, bestScore = Infinity
  faces(oc, shape).forEach((f, i) => {
    const fp = fingerprintFace(oc, f)
    if (fp.surface !== ref.surface) return
    const d = Math.hypot(fp.centroid[0] - ref.centroid[0], fp.centroid[1] - ref.centroid[1], fp.centroid[2] - ref.centroid[2])
    const da = Math.abs(fp.area - ref.area) / Math.max(ref.area, 1e-9)
    if (d > tol || da > 0.05) return
    const score = d + da
    if (score < bestScore) { bestScore = score; best = i }
  })
  if (best < 0) throw new Error('reference lost: the picked face no longer exists')
  return best
}
