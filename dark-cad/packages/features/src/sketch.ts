import { call, make } from '@darkcad/kernel/occ'
import { placeFrame } from '@darkcad/kernel/shape'
import type { Plane, SketchEntity, SketchFeature, Vec3 } from '@darkcad/core'

export function planeFrame(plane: Plane, offset: number): { origin: Vec3; dir: Vec3; xdir: Vec3 } {
  switch (plane) {
    case 'XY': return { origin: [0, 0, offset], dir: [0, 0, 1], xdir: [1, 0, 0] }
    case 'XZ': return { origin: [0, -offset, 0], dir: [0, -1, 0], xdir: [1, 0, 0] } // local y → +Z
    case 'YZ': return { origin: [offset, 0, 0], dir: [1, 0, 0], xdir: [0, 1, 0] } // local y → +Z
  }
}

/** Polygon approximation of a profile, used only for containment tests (hole detection). */
export function outline(e: SketchEntity): [number, number][] {
  switch (e.kind) {
    case 'rect': return [[e.x, e.y], [e.x + e.w, e.y], [e.x + e.w, e.y + e.h], [e.x, e.y + e.h]]
    case 'circle': return Array.from({ length: 48 }, (_, i) => [e.x + e.r * Math.cos((i / 48) * 2 * Math.PI), e.y + e.r * Math.sin((i / 48) * 2 * Math.PI)] as [number, number])
    case 'polygon': return e.points
  }
}

const polyArea = (p: [number, number][]) => Math.abs(p.reduce((s, [x, y], i) => s + x * p[(i + 1) % p.length][1] - p[(i + 1) % p.length][0] * y, 0) / 2)
function inside(pt: [number, number], poly: [number, number][]): boolean {
  let c = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j]
    if (yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) c = !c
  }
  return c
}

/** Nesting depth of every profile: even = solid region, odd = hole. */
export function nesting(entities: SketchEntity[]): number[] {
  const polys = entities.map(outline)
  return polys.map((p, i) => polys.filter((q, j) => j !== i && polyArea(q) > polyArea(p) && inside(p[0], q)).length)
}

function wireOf(oc: any, e: SketchEntity) {
  if (e.kind === 'circle') {
    const c = make(oc, 'gp_Circ', make(oc, 'gp_Ax2', make(oc, 'gp_Pnt', e.x, e.y, 0), make(oc, 'gp_Dir', 0, 0, 1)), e.r)
    return make(oc, 'BRepBuilderAPI_MakeWire', make(oc, 'BRepBuilderAPI_MakeEdge', c).Edge()).Wire()
  }
  const poly = make(oc, 'BRepBuilderAPI_MakePolygon')
  for (const [x, y] of outline(e)) poly.Add_1(make(oc, 'gp_Pnt', x, y, 0))
  poly.Close()
  return poly.Wire()
}

function faceOf(oc: any, e: SketchEntity) {
  return make(oc, 'BRepBuilderAPI_MakeFace', wireOf(oc, e), true).Face()
}

/** Planar face(s) of a sketch (holes cut out), placed on the sketch plane. */
export function sketchFace(oc: any, sk: SketchFeature) {
  if (!sk.entities.length) throw new Error(`sketch "${sk.name}" is empty`)
  const depth = nesting(sk.entities)
  const order = sk.entities.map((_, i) => i).sort((a, b) => polyArea(outline(sk.entities[b])) - polyArea(outline(sk.entities[a])))
  let acc: any
  for (const i of order) {
    const f = faceOf(oc, sk.entities[i])
    if (!acc) acc = f
    else acc = call(make(oc, depth[i] % 2 ? 'BRepAlgoAPI_Cut' : 'BRepAlgoAPI_Fuse', acc, f), 'Shape')
  }
  const fr = planeFrame(sk.plane, sk.offset)
  return placeFrame(oc, acc, fr.origin, fr.dir, fr.xdir)
}

/** Outer wire (largest profile) of a sketch, placed on its plane, for lofting. */
export function sketchOuterWire(oc: any, sk: SketchFeature) {
  if (!sk.entities.length) throw new Error(`sketch "${sk.name}" is empty`)
  const biggest = sk.entities.reduce((a, b) => (polyArea(outline(b)) > polyArea(outline(a)) ? b : a))
  const fr = planeFrame(sk.plane, sk.offset)
  const placed = placeFrame(oc, wireOf(oc, biggest), fr.origin, fr.dir, fr.xdir)
  return call(oc.TopoDS, 'Wire', placed)
}
