import { call, make } from '@darkcad/kernel/occ'
import { placeFrame } from '@darkcad/kernel/shape'
import { threadRidge } from '@darkcad/kernel/thread'
import { findSize, threadDims, type Kind } from '@darkcad/standards'

export interface ThreadOp {
  designation: string
  kind: Kind
  /** Thread length along the axis. */
  length: number
  /** Axis origin (the free end of the thread) and direction (pointing into the material for a hole / out of the base for a boss). */
  origin?: [number, number, number]
  dir?: [number, number, number]
  /** Radius of the plain cylinder the thread is applied to. If given, that cylinder is first grown or shrunk to fit the thread. */
  existingRadius?: number
}

/** Extra length either side so booleans never see coincident end faces. */
const RUNOUT = 1

const place = (oc: any, shape: any, origin: [number, number, number], dir: [number, number, number]) => placeFrame(oc, shape, origin, dir)

/** Local-frame geometry: axis = +Z from z = 0 (free end). Taper makes radius grow with z, as on R threads. */
function localParts(oc: any, op: ThreadOp) {
  const d = threadDims(op.designation, op.kind)
  const gauge = findSize(op.designation)?.size.gaugeLength ?? 0
  const half = d.taper / 2
  const shift = d.taper ? -gauge * half : 0 // basic diameter applies at the gauge plane, not at the free end
  const rootR0 = d.rMinor + shift
  const majorR0 = d.rMajor + shift
  const ridge = threadRidge(oc, {
    pitch: d.pitch, rMajor: majorR0, rMinor: rootR0, angleDeg: d.angleDeg, length: op.length, taper: d.taper,
  })
  return { d, ridge, rootR0, majorR0, half }
}

function solidCone(oc: any, r0: number, half: number, from: number, to: number) {
  const r = (z: number) => r0 + z * half
  if (half === 0) {
    const c = make(oc, 'BRepPrimAPI_MakeCylinder', r0, to - from).Shape()
    return from === 0 ? c : call(make(oc, 'BRepBuilderAPI_Transform', c, (() => { const t = make(oc, 'gp_Trsf'); t.SetTranslation_1(make(oc, 'gp_Vec', 0, 0, from)); return t })(), true), 'Shape')
  }
  const cone = make(oc, 'BRepPrimAPI_MakeCone', r(from), r(to), to - from).Shape()
  if (from === 0) return cone
  const t = make(oc, 'gp_Trsf'); t.SetTranslation_1(make(oc, 'gp_Vec', 0, 0, from))
  return call(make(oc, 'BRepBuilderAPI_Transform', cone, t, true), 'Shape')
}

/** Stand-alone male threaded rod, built at the origin along +Z. The cylinder is sized from the table. */
export function threadedRod(oc: any, op: ThreadOp) {
  const { ridge, rootR0, half } = localParts(oc, { ...op, kind: 'male' })
  const core = solidCone(oc, rootR0, half, -RUNOUT, op.length + RUNOUT)
  const rod = call(make(oc, 'BRepAlgoAPI_Fuse', core, ridge), 'Shape')
  return place(oc, rod, op.origin ?? [0, 0, 0], op.dir ?? [0, 0, 1])
}

/** Male thread on a boss. An existing boss of a different radius is first cut back so the result is exactly the table size. */
export function fuseThreadedBoss(oc: any, body: any, op: ThreadOp) {
  const rod = threadedRod(oc, { ...op, kind: 'male' })
  let base = body
  if (op.existingRadius) {
    const d = threadDims(op.designation, 'male')
    const r = Math.max(op.existingRadius, d.rMajor) + 0.01
    const sleeve = placeFrame(oc, make(oc, 'BRepPrimAPI_MakeCylinder', r, op.length + RUNOUT).Shape(), op.origin ?? [0, 0, 0], op.dir ?? [0, 0, 1])
    // Pull the tool back along -dir so it starts RUNOUT before the free end.
    base = call(make(oc, 'BRepAlgoAPI_Cut', body, shiftAlong(oc, sleeve, op, -RUNOUT)), 'Shape')
  }
  return call(make(oc, 'BRepAlgoAPI_Fuse', base, rod), 'Shape')
}

function shiftAlong(oc: any, shape: any, op: ThreadOp, dist: number) {
  const dir = op.dir ?? [0, 0, 1]
  const n = Math.hypot(...dir)
  const t = make(oc, 'gp_Trsf')
  t.SetTranslation_1(make(oc, 'gp_Vec', (dir[0] / n) * dist, (dir[1] / n) * dist, (dir[2] / n) * dist))
  return call(make(oc, 'BRepBuilderAPI_Transform', shape, t, true), 'Shape')
}

/** Female thread: cut a tap-drill bore, then the helical groove. An existing hole that is too large is first filled in. */
export function cutThreadedHole(oc: any, body: any, op: ThreadOp) {
  const { d, ridge } = localParts(oc, { ...op, kind: 'female' })
  const gauge = findSize(op.designation)?.size.gaugeLength ?? 0
  const borePlain = d.taper === 0
  const boreR0 = borePlain ? d.rCylinder : d.rMinor - (gauge * d.taper) / 2
  const bore = solidCone(oc, boreR0, borePlain ? 0 : d.taper / 2, -RUNOUT, op.length)
  const tool = place(oc, call(make(oc, 'BRepAlgoAPI_Fuse', bore, ridge), 'Shape'), op.origin ?? [0, 0, 0], op.dir ?? [0, 0, 1])
  let base = body
  if (op.existingRadius && op.existingRadius > d.rCylinder) {
    const plug = placeFrame(oc, make(oc, 'BRepPrimAPI_MakeCylinder', op.existingRadius + 0.01, op.length).Shape(), op.origin ?? [0, 0, 0], op.dir ?? [0, 0, 1])
    base = call(make(oc, 'BRepAlgoAPI_Fuse', body, plug), 'Shape')
  }
  return call(make(oc, 'BRepAlgoAPI_Cut', base, tool), 'Shape')
}
