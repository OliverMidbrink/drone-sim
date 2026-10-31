import { call, make } from '@darkcad/kernel/occ'
import { isValid } from '@darkcad/kernel/shape'
import type { Document, Feature } from '@darkcad/core'
import { sketchFace, sketchOuterWire, planeFrame } from './sketch'
import { cutThreadedHole, fuseThreadedBoss } from './thread'
import { deleteFeature, resolveFace } from './defeature'

export interface BuildResult {
  /** Result shape per body-producing feature id. */
  bodies: Map<string, any>
  /** Bodies not consumed by a later feature, in feature order. */
  visible: string[]
  errors: Map<string, string>
}

export function rebuild(oc: any, doc: Document): BuildResult {
  const bodies = new Map<string, any>()
  const consumed = new Set<string>()
  const errors = new Map<string, string>()
  const sketches = new Map<string, Extract<Feature, { type: 'sketch' }>>()
  const lastVisible = () => [...bodies.keys()].filter((k) => !consumed.has(k)).pop()
  const body = (id: string) => {
    if (!bodies.has(id)) throw new Error(`depends on "${id}", which has no result`)
    return bodies.get(id)
  }
  const sk = (id: string) => {
    const s = sketches.get(id)
    if (!s) throw new Error(`sketch "${id}" not found`)
    return s
  }
  const combine = (f: { id: string; mode: 'new' | 'add' | 'cut' }, solid: any) => {
    if (f.mode === 'new') return solid
    const base = lastVisible()
    if (!base) throw new Error(`nothing to ${f.mode === 'add' ? 'add to' : 'cut from'}`)
    consumed.add(base)
    return call(make(oc, f.mode === 'add' ? 'BRepAlgoAPI_Fuse' : 'BRepAlgoAPI_Cut', bodies.get(base), solid), 'Shape')
  }

  for (const f of doc.features) {
    try {
      let out: any
      switch (f.type) {
        case 'sketch': sketches.set(f.id, f); continue
        case 'extrude': {
          const s = sk(f.sketch)
          const n = planeFrame(s.plane, s.offset).dir
          const prism = make(oc, 'BRepPrimAPI_MakePrism', sketchFace(oc, s), make(oc, 'gp_Vec', n[0] * f.distance, n[1] * f.distance, n[2] * f.distance), false, true)
          out = combine(f, prism.Shape())
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
          consumed.add(f.target); consumed.add(f.tool)
          break
        }
        case 'thread': {
          const op = { designation: f.designation, kind: f.kind, length: f.length, origin: f.origin, dir: f.dir, existingRadius: f.existingRadius }
          out = f.kind === 'male' ? fuseThreadedBoss(oc, body(f.target), op) : cutThreadedHole(oc, body(f.target), op)
          consumed.add(f.target)
          break
        }
        case 'delete-feature': {
          const shape = body(f.target)
          out = deleteFeature(oc, shape, resolveFace(oc, shape, f.face)).shape
          consumed.add(f.target)
          break
        }
      }
      if (!isValid(oc, out)) throw new Error('kernel produced an invalid solid')
      bodies.set(f.id, out)
    } catch (e: any) {
      errors.set(f.id, String(e?.message ?? e))
    }
  }
  return { bodies, visible: [...bodies.keys()].filter((k) => !consumed.has(k)), errors }
}
