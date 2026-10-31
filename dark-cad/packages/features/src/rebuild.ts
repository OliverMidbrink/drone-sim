import { call, make } from '@darkcad/kernel/occ'
import { isValid } from '@darkcad/kernel/shape'
import { readSTEP } from '@darkcad/kernel/step'
import { resolveFrames, type Document, type Feature, type Placement, type SketchFeature } from '@darkcad/core'
import { sketchFace, sketchOuterWire, sketchFrameOf, resolveProfiles, axisWorld, crossesAxis } from './sketch'
import { cutThreadedHole, fuseThreadedBoss } from './thread'
import { deleteFeature, resolveFace } from './defeature'

export interface BuildResult {
  /** Result shape per body-producing feature id. */
  bodies: Map<string, any>
  /** Bodies not consumed by a later feature, in feature order. */
  visible: string[]
  /** Why a feature failed. Features that failed only because something before them did say "Blocked by …". */
  errors: Map<string, string>
  /** For every body-producing feature, the feature that first created the body (stable across Add / Cut / Thread …). */
  roots: Map<string, string>
}

export interface BuildOptions {
  /** Designs used as components, by design id (see `collectDeps`). */
  deps?: Record<string, Document>
  /** STEP files by asset id. */
  assets?: Record<string, Uint8Array> | Map<string, Uint8Array>
  /** Id of the design being built; used to catch components that contain themselves. */
  designId?: string
  /** Designs currently being built (outermost first). */
  stack?: string[]
  /** Built designs by id, shared by every instance of a component within one build. */
  cache?: Map<string, BuildResult>
}

/** gp_Trsf for a component placement: rotate about X, then Y, then Z, then move. */
export function placementTrsf(oc: any, p: Placement) {
  const t = make(oc, 'gp_Trsf')
  const rot = (axis: [number, number, number], deg: number) => {
    if (!deg) return
    const r = make(oc, 'gp_Trsf')
    call(r, 'SetRotation', make(oc, 'gp_Ax1', make(oc, 'gp_Pnt', 0, 0, 0), make(oc, 'gp_Dir', ...axis)), (deg * Math.PI) / 180)
    call(t, 'Multiply', r)
  }
  const move = make(oc, 'gp_Trsf')
  call(move, 'SetTranslation', make(oc, 'gp_Vec', ...p.pos))
  // t = move · Rz · Ry · Rx, so a point is rotated about X first.
  call(t, 'Multiply', move)
  rot([0, 0, 1], p.rot[2]); rot([0, 1, 0], p.rot[1]); rot([1, 0, 0], p.rot[0])
  return t
}

function compound(oc: any, shapes: any[]) {
  const comp = make(oc, 'TopoDS_Compound')
  const builder = make(oc, 'BRep_Builder')
  builder.MakeCompound(comp)
  for (const s of shapes) builder.Add(comp, s)
  return comp
}

/** Thrown when a feature cannot be built because a feature before it failed or is suppressed. */
class Blocked extends Error {}

export function rebuild(oc: any, doc: Document, opts: BuildOptions = {}): BuildResult {
  const cache = opts.cache ?? new Map<string, BuildResult>()
  const bodies = new Map<string, any>()
  const roots = new Map<string, string>()
  const consumed = new Set<string>()
  const errors = new Map<string, string>()
  const suppressed = new Map<string, string>()
  const sketches = new Map<string, SketchFeature>()
  const names = new Map(doc.features.map((f) => [f.id, f.name]))
  const nameOf = (id: string) => `"${names.get(id) ?? id}"`
  const { frames, errors: frameErrors } = resolveFrames(doc.features)
  const lastVisible = () => [...bodies.keys()].filter((k) => !consumed.has(k)).pop()
  /** Why something that should exist does not: it failed, is suppressed, or simply is not there. */
  const missing = (id: string, what: string): never => {
    if (suppressed.has(id)) throw new Blocked(`${what} ${nameOf(id)} is suppressed`)
    if (errors.has(id)) throw new Blocked(`Blocked by ${nameOf(id)}: ${errors.get(id)!.replace(/^Blocked by "[^"]*": /, '')}`)
    throw new Error(`${what} "${id}" not found`)
  }
  const body = (id: string) => {
    if (!bodies.has(id)) missing(id, 'feature')
    return bodies.get(id)
  }
  const sk = (id: string) => {
    const s = sketches.get(id)
    if (!s) missing(id, 'sketch')
    return s!
  }
  let parent: string | undefined
  const combine = (f: { id: string; mode: 'new' | 'add' | 'cut' }, solid: any) => {
    if (f.mode === 'new') return solid
    const base = lastVisible()
    if (!base) {
      const failed = [...errors.keys()].reverse().find((id) => bodiesProducing.has(id))
      if (failed) throw new Blocked(`Blocked by ${nameOf(failed)}: ${errors.get(failed)!.replace(/^Blocked by "[^"]*": /, '')}`)
      throw new Error(`nothing to ${f.mode === 'add' ? 'add to' : 'cut from'}`)
    }
    consumed.add(base)
    parent = base
    return call(make(oc, f.mode === 'add' ? 'BRepAlgoAPI_Fuse' : 'BRepAlgoAPI_Cut', bodies.get(base), solid), 'Shape')
  }
  const bodiesProducing = new Set(doc.features.filter((f) => ['extrude', 'revolve', 'loft', 'boolean', 'thread', 'delete-feature', 'component', 'import-step'].includes(f.type)).map((f) => f.id))
  const consume = (id: string) => { consumed.add(id); parent = id }

  for (const f of doc.features as Feature[]) {
    parent = undefined
    if (f.suppressed) { suppressed.set(f.id, f.name); continue }
    try {
      let out: any
      let trusted = false
      switch (f.type) {
        case 'sketch': {
          const fe = frameErrors.get(f.id)
          if (fe) throw new Error(fe)
          sketches.set(f.id, { ...f, frame: frames.get(f.id) })
          continue
        }
        case 'plane': {
          const fe = frameErrors.get(f.id)
          if (fe) throw new Error(fe)
          continue
        }
        case 'extrude': {
          if (!Number.isFinite(f.distance) || Math.abs(f.distance) < 1e-6) throw new Error('extrude distance must not be zero')
          const s = sk(f.sketch)
          const n = sketchFrameOf(s).dir
          const prism = make(oc, 'BRepPrimAPI_MakePrism', sketchFace(oc, s), make(oc, 'gp_Vec', n[0] * f.distance, n[1] * f.distance, n[2] * f.distance), false, true)
          out = combine(f, prism.Shape())
          break
        }
        case 'revolve': {
          // Not complete yet (the user is still picking): nothing to build and nothing to complain about.
          if (!f.profiles.length || !f.axis) continue
          const axis = axisWorld(sketches, f.axis)
          const picked = resolveProfiles(sketches, f.profiles)
          const ax1 = make(oc, 'gp_Ax1', make(oc, 'gp_Pnt', ...axis.origin), make(oc, 'gp_Dir', ...axis.dir))
          const total = f.extent === 'full' ? 2 * Math.PI : Math.max(1e-3, Math.min(359.999, f.angle)) * Math.PI / 180
          let acc: any
          for (const [id, regions] of picked) {
            const s = sk(id)
            if (crossesAxis(s, regions, axis)) throw new Error('profile crosses the axis')
            let face = sketchFace(oc, s, regions)
            if (f.extent === 'symmetric') {
              const rot = make(oc, 'gp_Trsf')
              call(rot, 'SetRotation', ax1, -total / 2)
              face = call(make(oc, 'BRepBuilderAPI_Transform', face, rot, true), 'Shape')
            }
            const solid = make(oc, 'BRepPrimAPI_MakeRevol', face, ax1, total, false).Shape()
            acc = acc ? call(make(oc, 'BRepAlgoAPI_Fuse', acc, solid), 'Shape') : solid
          }
          out = combine(f, acc)
          break
        }
        case 'loft': {
          const gen = make(oc, 'BRepOffsetAPI_ThruSections', true, false, 1e-6)
          call(gen, 'AddWire', sketchOuterWire(oc, sk(f.sketches[0])))
          call(gen, 'AddWire', sketchOuterWire(oc, sk(f.sketches[1])))
          call(gen, 'Build')
          out = combine(f, gen.Shape())
          break
        }
        case 'boolean': {
          const cls = { union: 'BRepAlgoAPI_Fuse', cut: 'BRepAlgoAPI_Cut', common: 'BRepAlgoAPI_Common' }[f.op]
          out = call(make(oc, cls, body(f.target), body(f.tool)), 'Shape')
          consume(f.target); consumed.add(f.tool)
          break
        }
        case 'thread': {
          const op = { designation: f.designation, kind: f.kind, length: f.length, origin: f.origin, dir: f.dir, existingRadius: f.existingRadius }
          out = f.kind === 'male' ? fuseThreadedBoss(oc, body(f.target), op) : cutThreadedHole(oc, body(f.target), op)
          consume(f.target)
          break
        }
        case 'delete-feature': {
          const shape = body(f.target)
          out = deleteFeature(oc, shape, resolveFace(oc, shape, f.face)).shape
          consume(f.target)
          break
        }
        case 'import-step': {
          const bytes = opts.assets instanceof Map ? opts.assets.get(f.asset) : opts.assets?.[f.asset]
          if (!bytes) throw new Error(`the STEP file "${f.asset}" is not available`)
          out = readSTEP(oc, bytes)
          trusted = true
          break
        }
        case 'component': {
          const stack = [...(opts.stack ?? []), ...(opts.designId ? [opts.designId] : [])]
          if (stack.includes(f.ref)) throw new Error(`component loop: "${opts.deps?.[f.ref]?.name ?? f.ref}" contains itself`)
          const def = opts.deps?.[f.ref]
          if (!def) throw new Error(`the design "${f.ref}" this component refers to is missing`)
          let sub = cache.get(f.ref)
          if (!sub) { sub = rebuild(oc, def, { ...opts, designId: f.ref, stack, cache }); cache.set(f.ref, sub) }
          if (!sub.visible.length) throw new Error(`"${def.name}" has no solid bodies to place${sub.errors.size ? ` (${sub.errors.size} of its features failed)` : ''}`)
          const comp = compound(oc, sub.visible.map((id) => sub.bodies.get(id)))
          out = call(make(oc, 'BRepBuilderAPI_Transform', comp, placementTrsf(oc, f.placement), true), 'Shape')
          trusted = true
          break
        }
      }
      if (!trusted && !isValid(oc, out)) throw new Error('kernel produced an invalid solid')
      bodies.set(f.id, out)
      roots.set(f.id, parent && roots.has(parent) ? roots.get(parent)! : f.id)
    } catch (e: any) {
      errors.set(f.id, String(e?.message ?? e))
    }
  }
  return { bodies, visible: [...bodies.keys()].filter((k) => !consumed.has(k)), errors, roots }
}
