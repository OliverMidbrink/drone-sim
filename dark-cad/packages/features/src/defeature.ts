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

function planeOf(oc: any, face: any): { o: number[]; n: number[] } | undefined {
  const sa = make(oc, 'BRepAdaptor_Surface', face, true)
  if (sa.GetType() !== oc.GeomAbs_SurfaceType.GeomAbs_Plane) return
  const pl = sa.Plane()
  const p = pl.Location(), d = pl.Axis().Direction()
  return { o: [p.X(), p.Y(), p.Z()], n: [d.X(), d.Y(), d.Z()] }
}

function edgePoints(oc: any, edge: any): number[][] {
  const c = make(oc, 'BRepAdaptor_Curve', edge)
  const a = c.FirstParameter(), b = c.LastParameter()
  return [0, 0.25, 0.5, 0.75, 1].map((t) => { const q = c.Value(a + (b - a) * t); return [q.X(), q.Y(), q.Z()] })
}

/**
 * Faces making up the feature (boss, rib, pocket, hole…) the seed face belongs to.
 *
 * A feature is a connected set of faces whose boundary with the rest of the solid lies entirely in one plane
 * (the base). For every planar face as base candidate we grow from the seed without crossing that plane's
 * faces, keep the candidates whose boundary really is coplanar, and take the smallest. Blends, tops and walls
 * therefore all join in, and the same rule fills pockets as well as removing bosses.
 * `baseIndex` pins the base face explicitly. Returns indices into `faces(oc, shape)`.
 */
export function selectProtrusion(oc: any, shape: any, seedIndex: number, baseIndex?: number): number[] {
  const fs = faces(oc, shape)
  if (seedIndex < 0 || seedIndex >= fs.length) throw new Error('no face selected')
  if (baseIndex === seedIndex) throw new Error('the picked face is the base face, not part of a feature')
  const edges = fs.map((f) => edgesOf(oc, f))
  const n = fs.length
  const adj: boolean[][] = Array.from({ length: n }, () => new Array(n).fill(false))
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) adj[i][j] = adj[j][i] = edges[i].some((e) => edges[j].some((g) => e.IsSame(g)))
  const areas = fs.map((f) => area(oc, f))
  const total = areas.reduce((a, b) => a + b, 0)
  const planes = fs.map((f) => planeOf(oc, f))
  const scale = Math.sqrt(total) || 1
  const tol = 1e-5 * scale
  const dist = (pl: { o: number[]; n: number[] }, q: number[]) => Math.abs((q[0] - pl.o[0]) * pl.n[0] + (q[1] - pl.o[1]) * pl.n[1] + (q[2] - pl.o[2]) * pl.n[2])

  const tryBase = (b: number): number[] | undefined => {
    const pl = planes[b]
    if (!pl || b === seedIndex) return
    // Faces lying in the base plane (e.g. other islands of the same flat) are barriers too.
    const barrier = fs.map((_, i) => i === b || (!!planes[i] && dist(pl, planes[i]!.o) < tol && Math.abs(Math.abs(planes[i]!.n[0] * pl.n[0] + planes[i]!.n[1] * pl.n[1] + planes[i]!.n[2] * pl.n[2]) - 1) < 1e-6))
    if (barrier[seedIndex]) return
    const seen = new Set([seedIndex]), queue = [seedIndex]
    while (queue.length) {
      const i = queue.shift()!
      for (let j = 0; j < n; j++) if (!seen.has(j) && !barrier[j] && adj[i][j]) { seen.add(j); queue.push(j) }
    }
    // Boundary edges (shared with a face outside the set) must all lie in the base plane.
    for (const i of seen) for (const e of edges[i]) {
      const outside = [...Array(n).keys()].some((j) => !seen.has(j) && edges[j].some((g) => e.IsSame(g)))
      if (outside && !edgePoints(oc, e).every((q) => dist(pl, q) < tol)) return
    }
    return [...seen]
  }

  const candidates = baseIndex !== undefined ? [baseIndex] : fs.map((_, i) => i)
  let best: number[] | undefined, bestArea = Infinity
  for (const b of candidates) {
    const set = tryBase(b)
    if (!set) continue
    const a = set.reduce((s, i) => s + areas[i], 0)
    if (baseIndex === undefined && a > 0.5 * total) continue // a "feature" is the smaller part of the solid
    if (a < bestArea) { best = set; bestArea = a }
  }
  if (!best) throw new Error('could not find a feature at that face (no flat base around it)')
  return best.sort((a, b) => a - b)
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
