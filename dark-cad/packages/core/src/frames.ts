import type { Feature, Frame, Plane, PlaneBase, PlaneDef, SketchFeature, Vec3 } from './model'

type V = Vec3
const dot = (a: V, b: V) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a: V, b: V): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const add = (a: V, b: V, k = 1): V => [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k]
const unit = (a: V): V => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l] }

/** One of the three world planes, `offset` mm along its normal. */
export function worldFrame(plane: Plane, offset = 0): Frame {
  switch (plane) {
    case 'XY': return { origin: [0, 0, offset], dir: [0, 0, 1], xdir: [1, 0, 0] }
    case 'XZ': return { origin: [0, -offset, 0], dir: [0, -1, 0], xdir: [1, 0, 0] } // local y → +Z
    case 'YZ': return { origin: [offset, 0, 0], dir: [1, 0, 0], xdir: [0, 1, 0] } // local y → +Z
  }
}

/** The plane's local y axis. */
export const frameY = (f: Frame): V => cross(f.dir, f.xdir)

/** Rotate `v` by `deg` degrees about a unit axis (Rodrigues). */
function rotate(v: V, axis: V, deg: number): V {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a)
  const k = cross(axis, v), d = dot(axis, v) * (1 - c)
  return [v[0] * c + k[0] * s + axis[0] * d, v[1] * c + k[1] * s + axis[1] * d, v[2] * c + k[2] * s + axis[2] * d]
}

type Lookup = (id: string) => Feature | undefined

function baseFrame(base: PlaneBase, find: Lookup, depth: number): Frame {
  if (depth > 32) throw new Error('construction planes refer to each other in a loop')
  if (base.kind === 'world') return worldFrame(base.plane, 0)
  const f = find(base.id)
  if (!f) throw new Error(`plane "${base.id}" not found`)
  if (base.kind === 'plane') {
    if (f.type !== 'plane') throw new Error(`"${f.name}" is not a construction plane`)
    return defFrame(f.def, find, depth + 1)
  }
  if (f.type !== 'sketch') throw new Error(`"${f.name}" is not a sketch`)
  return frameOfSketch(f, find, depth + 1)
}

/** Frame of a construction plane definition. */
export function defFrame(def: PlaneDef, find: Lookup, depth = 0): Frame {
  switch (def.kind) {
    case 'offset': {
      const b = baseFrame(def.base, find, depth)
      return { origin: add(b.origin, b.dir, def.distance), dir: b.dir, xdir: b.xdir }
    }
    case 'angle': {
      const b = baseFrame(def.base, find, depth)
      const axis: V = def.axis === 'X' ? [1, 0, 0] : def.axis === 'Y' ? [0, 1, 0] : [0, 0, 1]
      return { origin: rotate(b.origin, axis, def.angle), dir: unit(rotate(b.dir, axis, def.angle)), xdir: unit(rotate(b.xdir, axis, def.angle)) }
    }
    case 'mid': {
      const a = baseFrame(def.a, find, depth), b = baseFrame(def.b, find, depth)
      if (Math.abs(Math.abs(dot(a.dir, b.dir)) - 1) > 1e-6) throw new Error('the two planes of a midplane must be parallel')
      const gap = dot([b.origin[0] - a.origin[0], b.origin[1] - a.origin[1], b.origin[2] - a.origin[2]], a.dir)
      return { origin: add(a.origin, a.dir, gap / 2), dir: a.dir, xdir: a.xdir }
    }
  }
}

function frameOfSketch(sk: SketchFeature, find: Lookup, depth = 0): Frame {
  if (!sk.planeRef) return worldFrame(sk.plane, sk.offset)
  const p = find(sk.planeRef)
  if (!p) throw new Error(`construction plane "${sk.planeRef}" not found`)
  if (p.type !== 'plane') throw new Error(`"${p.name}" is not a construction plane`)
  return defFrame(p.def, find, depth + 1)
}

/** Frame of a plane feature, looking referenced features up in `features`. */
export function planeFeatureFrame(features: readonly Feature[], plane: Extract<Feature, { type: 'plane' }>): Frame {
  return defFrame(plane.def, (id) => features.find((f) => f.id === id))
}

/** Frame a sketch lies on. Throws when its construction plane is missing or broken. */
export function sketchFrame(features: readonly Feature[], sk: SketchFeature): Frame {
  return frameOfSketch(sk, (id) => features.find((f) => f.id === id))
}

export interface ResolvedFrames {
  /** Frame per sketch and per construction plane id. */
  frames: Map<string, Frame>
  /** Why a sketch or plane has no frame (e.g. its base plane is missing). */
  errors: Map<string, string>
}

/**
 * Resolves every sketch and construction plane in order. A feature can only use planes and sketches before it,
 * so reordering the timeline can break a plane (and everything on it), which is then reported here.
 */
export function resolveFrames(features: readonly Feature[]): ResolvedFrames {
  const frames = new Map<string, Frame>(), errors = new Map<string, string>()
  const seen: Feature[] = []
  for (const f of features) {
    if (f.suppressed) continue
    if (f.type !== 'sketch' && f.type !== 'plane') continue
    try {
      const find = (id: string) => seen.find((x) => x.id === id)
      frames.set(f.id, f.type === 'plane' ? defFrame(f.def, find) : frameOfSketch(f, find))
      seen.push(f)
    } catch (e: any) { errors.set(f.id, String(e?.message ?? e)) }
  }
  return { frames, errors }
}
