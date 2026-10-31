import { call, make } from './occ'

export interface ThreadSpec {
  pitch: number
  /** radius of the thread crest/major (external) */
  rMajor: number
  /** radius of the root/minor */
  rMinor: number
  /** flank angle full included, degrees (60 ISO, 55 Whitworth) */
  angleDeg: number
  length: number
  /** diameter taper ratio (BSPT 1/16); radius grows by taper/2 per unit length. Default 0 (parallel). */
  taper?: number
}

/** Helical thread ridge solid (to fuse onto a minor-diameter core for a male thread). */
export function threadRidge(oc: any, s: ThreadSpec) {
  const turns = s.length / s.pitch
  const alpha = Math.atan((s.taper ?? 0) / 2) // cone semi-angle
  const slantPitch = s.pitch / Math.cos(alpha) // v runs along the slant; pitch is measured along the axis
  const surf = alpha === 0
    ? make(oc, 'Geom_CylindricalSurface', make(oc, 'gp_Ax3'), s.rMinor)
    : make(oc, 'Geom_ConicalSurface', make(oc, 'gp_Ax3'), alpha, s.rMinor)
  const line = make(oc, 'Geom2d_Line', make(oc, 'gp_Pnt2d', 0, 0), make(oc, 'gp_Dir2d', 2 * Math.PI, slantPitch))
  const tEnd = turns * Math.hypot(2 * Math.PI, slantPitch)
  const edge = make(oc, 'BRepBuilderAPI_MakeEdge', new oc.Handle_Geom2d_Curve_2(line), new oc.Handle_Geom_Surface_2(surf), 0, tEnd).Edge()
  oc.BRepLib.BuildCurves3d_2(edge)
  const spine = make(oc, 'BRepBuilderAPI_MakeWire', edge).Wire()

  // Trapezoid profile in the XZ plane at the helix start (angle 0 → +X), overlapping the core slightly.
  const half = Math.tan((s.angleDeg / 2) * Math.PI / 180)
  const h = s.rMajor - s.rMinor
  const overlap = 0.05 * s.pitch
  const wBase = Math.min(s.pitch * 0.95, 2 * (h + overlap) * half + 0.1 * s.pitch)
  const wTop = Math.max(0.05 * s.pitch, wBase - 2 * (h + overlap) * half)
  const pts: [number, number][] = [
    [s.rMinor - overlap, -wBase / 2], [s.rMajor, -wTop / 2], [s.rMajor, wTop / 2], [s.rMinor - overlap, wBase / 2],
  ]
  const poly = make(oc, 'BRepBuilderAPI_MakePolygon')
  for (const [x, z] of pts) poly.Add_1(make(oc, 'gp_Pnt', x, 0, z))
  poly.Close()
  const profile = poly.Wire()

  const pipe = make(oc, 'BRepOffsetAPI_MakePipeShell', spine)
  call(pipe, 'SetMode', make(oc, 'gp_Dir', 0, 0, 1)) // binormal mode: profile keeps orientation to the axis
  call(pipe, 'Add', profile, false, false)
  call(pipe, 'Build')
  pipe.MakeSolid()
  return pipe.Shape()
}
