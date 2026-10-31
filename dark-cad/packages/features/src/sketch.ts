import { call, make } from '@darkcad/kernel/occ'
import { placeFrame } from '@darkcad/kernel/shape'
import { worldFrame, type AxisRef, type Frame, type Plane, type ProfileRef, type SketchFeature, type Vec3 } from '@darkcad/core'
import { buildGeo, profiles, regionAt, tessellate, type Loop, type Pt, type Profiles, type Region, type Seg } from '@darkcad/sketch'

export function planeFrame(plane: Plane, offset: number): Frame { return worldFrame(plane, offset) }

/** The frame a sketch lies on: the resolved one (construction planes) when present, else its world plane. */
export const sketchFrameOf = (sk: SketchFeature): Frame => sk.frame ?? worldFrame(sk.plane, sk.offset)

export type Pt2 = Pt
/** Drawable form of a sketch: polylines in plane coordinates, plus shaded regions (holes cut out). */
export interface SketchPreview {
  curves: { pts: Pt2[]; closed: boolean; construction: boolean }[]
  regions: { outer: Pt2[]; holes: Pt2[][] }[]
  /** Straight lines (drawn and construction), for picking a revolve axis. */
  lines: { id: string; a: Pt2; b: Pt2; construction: boolean }[]
}

/** Closed profiles of a sketch, derived from its curves (construction geometry is ignored). */
export const sketchProfiles = (sk: SketchFeature): Profiles => profiles(sk.entities, sk.constraints ?? [])

/** Sampled geometry of a sketch for display (3D viewport). Pure: no kernel needed. */
export function sketchPreview(sk: SketchFeature): SketchPreview {
  const geo = buildGeo(sk.entities)
  const curves = geo.real.map((c) => ({ pts: tessellate(c, 0.01), closed: c.kind === 'circle', construction: c.construction }))
  const regions = sketchProfiles(sk).regions.map((r) => ({ outer: r.outer.poly, holes: r.holes.map((h) => h.poly) }))
  const lines = geo.real.filter((c) => c.kind === 'line').map((c: any) => ({ id: c.id as string, a: c.a as Pt2, b: c.b as Pt2, construction: c.construction as boolean }))
  return { curves, regions, lines }
}

function edgeOf(oc: any, s: Seg) {
  const pnt = (p: Pt) => make(oc, 'gp_Pnt', p[0], p[1], 0)
  const circ = (c: Pt, r: number) => make(oc, 'gp_Circ', make(oc, 'gp_Ax2', pnt(c), make(oc, 'gp_Dir', 0, 0, 1)), r)
  if (s.kind === 'line') return make(oc, 'BRepBuilderAPI_MakeEdge', pnt(s.from), pnt(s.to)).Edge()
  if (s.kind === 'circle') return make(oc, 'BRepBuilderAPI_MakeEdge', circ(s.c, s.r)).Edge()
  // The kernel's arc runs counter-clockwise from a0 to a1.
  const ang = (p: Pt) => Math.atan2(p[1] - s.c[1], p[0] - s.c[0])
  const [from, to] = s.ccw ? [s.from, s.to] : [s.to, s.from]
  const a0 = ang(from)
  let a1 = ang(to)
  if (a1 <= a0 + 1e-12) a1 += 2 * Math.PI
  return make(oc, 'BRepBuilderAPI_MakeEdge', circ(s.c, s.r), a0, a1).Edge()
}

function wireOf(oc: any, loop: Loop) {
  const mw = make(oc, 'BRepBuilderAPI_MakeWire')
  for (const s of loop.segs) mw.Add_1(edgeOf(oc, s))
  if (!mw.IsDone()) throw new Error('sketch profile could not be closed into a wire')
  return mw.Wire()
}

const faceOf = (oc: any, loop: Loop) => make(oc, 'BRepBuilderAPI_MakeFace', wireOf(oc, loop), true).Face()

/** Planar face(s) of a sketch (holes cut out), placed on the sketch plane. */
export function sketchFace(oc: any, sk: SketchFeature, only?: Region[]) {
  const regions = only ?? sketchProfiles(sk).regions
  if (!regions.length) throw new Error(sk.entities.some((e) => e.kind !== 'point' && !e.construction) ? `sketch "${sk.name}" has no closed profile` : `sketch "${sk.name}" is empty`)
  let acc: any
  for (const r of regions) {
    let f = faceOf(oc, r.outer)
    for (const h of r.holes) f = call(make(oc, 'BRepAlgoAPI_Cut', f, faceOf(oc, h)), 'Shape')
    acc = acc ? call(make(oc, 'BRepAlgoAPI_Fuse', acc, f), 'Shape') : f
  }
  if (regions.length > 1) acc = unify(oc, acc)
  const fr = sketchFrameOf(sk)
  return placeFrame(oc, acc, fr.origin, fr.dir, fr.xdir)
}

/** Merge faces that lie in one plane (the seams left where regions touch). */
function unify(oc: any, shape: any) {
  try {
    const u = make(oc, 'ShapeUpgrade_UnifySameDomain', shape, true, true, true)
    u.Build()
    return u.Shape()
  } catch { return shape }
}

/** Outer wire (largest profile) of a sketch, placed on its plane, for lofting. */
export function sketchOuterWire(oc: any, sk: SketchFeature) {
  const { regions } = sketchProfiles(sk)
  if (!regions.length) throw new Error(sk.entities.length ? `sketch "${sk.name}" has no closed profile` : `sketch "${sk.name}" is empty`)
  const fr = sketchFrameOf(sk)
  const placed = placeFrame(oc, wireOf(oc, regions[0].outer), fr.origin, fr.dir, fr.xdir)
  return call(oc.TopoDS, 'Wire', placed)
}

/** The regions of each sketch that the picked profile references point into (duplicates removed). */
export function resolveProfiles(sketches: Map<string, SketchFeature>, refs: ProfileRef[]): Map<string, Region[]> {
  const out = new Map<string, Region[]>()
  const cache = new Map<string, Profiles>()
  for (const ref of refs) {
    const sk = sketches.get(ref.sketch)
    if (!sk) throw new Error(`sketch "${ref.sketch}" not found`)
    const p = cache.get(sk.id) ?? sketchProfiles(sk)
    cache.set(sk.id, p)
    const r = regionAt(p, ref.pt)
    if (!r) throw new Error(`a picked profile of "${sk.name}" no longer exists`)
    const list = out.get(sk.id) ?? []
    if (!list.includes(r)) list.push(r)
    out.set(sk.id, list)
  }
  return out
}

export interface WorldAxis { origin: Vec3; dir: Vec3 }

/** World-space end points of a revolve axis reference; sketch axes and world axes are drawn `half` mm either side of the origin. */
export function axisSegment(sketches: Map<string, SketchFeature>, ref: AxisRef, half = 100): [Vec3, Vec3] {
  if (ref.kind === 'world') {
    const d: Vec3 = ref.axis === 'X' ? [1, 0, 0] : ref.axis === 'Y' ? [0, 1, 0] : [0, 0, 1]
    return [[-d[0] * half, -d[1] * half, -d[2] * half], [d[0] * half, d[1] * half, d[2] * half]]
  }
  const sk = sketches.get(ref.sketch)
  if (!sk) throw new Error(`sketch "${ref.sketch}" not found`)
  let a: Pt, b: Pt
  if (ref.id === 'axis-x') { a = [-half, 0]; b = [half, 0] }
  else if (ref.id === 'axis-y') { a = [0, -half]; b = [0, half] }
  else {
    const c = buildGeo(sk.entities).byId.get(ref.id)
    if (!c || c.kind !== 'line') throw new Error(`axis line is missing from "${sk.name}"`)
    a = c.a; b = c.b
  }
  const f = sketchFrameOf(sk)
  const x = f.xdir, n = f.dir, y: Vec3 = [n[1] * x[2] - n[2] * x[1], n[2] * x[0] - n[0] * x[2], n[0] * x[1] - n[1] * x[0]]
  const w = (p: Pt): Vec3 => [f.origin[0] + x[0] * p[0] + y[0] * p[1], f.origin[1] + x[1] * p[0] + y[1] * p[1], f.origin[2] + x[2] * p[0] + y[2] * p[1]]
  return [w(a), w(b)]
}

/** World-space line of a revolve axis reference. */
export function axisWorld(sketches: Map<string, SketchFeature>, ref: AxisRef): WorldAxis {
  const [p0, p1] = axisSegment(sketches, ref, 1)
  const d: Vec3 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
  const l = Math.hypot(...d)
  if (l < 1e-9) throw new Error('the axis line has no length')
  const o: Vec3 = ref.kind === 'world' ? [0, 0, 0] : p0
  return { origin: o, dir: [d[0] / l, d[1] / l, d[2] / l] }
}

/** Does a profile of the sketch lie on both sides of the axis (only checked when the axis is in the sketch's plane)? */
export function crossesAxis(sk: SketchFeature, regions: Region[], axis: WorldAxis): boolean {
  const f = sketchFrameOf(sk)
  const n = f.dir
  const along = axis.dir[0] * n[0] + axis.dir[1] * n[1] + axis.dir[2] * n[2]
  const off = (axis.origin[0] - f.origin[0]) * n[0] + (axis.origin[1] - f.origin[1]) * n[1] + (axis.origin[2] - f.origin[2]) * n[2]
  if (Math.abs(along) > 1e-6 || Math.abs(off) > 1e-6) return false
  const x = f.xdir, y: Vec3 = [n[1] * x[2] - n[2] * x[1], n[2] * x[0] - n[0] * x[2], n[0] * x[1] - n[1] * x[0]]
  const loc = (v: Vec3): Pt => [v[0] * x[0] + v[1] * x[1] + v[2] * x[2], v[0] * y[0] + v[1] * y[1] + v[2] * y[2]]
  const o = loc([axis.origin[0] - f.origin[0], axis.origin[1] - f.origin[1], axis.origin[2] - f.origin[2]]), d = loc(axis.dir)
  let lo = Infinity, hi = -Infinity
  for (const r of regions) for (const q of r.outer.poly) {
    const s = d[0] * (q[1] - o[1]) - d[1] * (q[0] - o[0])
    lo = Math.min(lo, s); hi = Math.max(hi, s)
  }
  return lo < -1e-6 && hi > 1e-6
}
