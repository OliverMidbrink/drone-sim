import { describe, it, expect } from 'vitest'
import { make } from '../src/occ'
import { loadOCC } from '../src/node'
import { threadRidge } from '../src/thread'

function props(oc: any, shape: any) {
  const g = make(oc, 'GProp_GProps')
  oc.BRepGProp.VolumeProperties_1(shape, g, false, false, false)
  return { volume: g.Mass() }
}
function valid(oc: any, shape: any) { return make(oc, 'BRepCheck_Analyzer', shape, true).IsValid_2 ? make(oc, 'BRepCheck_Analyzer', shape, true).IsValid_2() : make(oc, 'BRepCheck_Analyzer', shape, true).IsValid() }

describe('G 1/4 male thread', () => {
  it('builds a valid fused solid', async () => {
    const oc = await loadOCC()
    const P = 1.337, major = 13.157, minor = 11.445, len = 5 * P
    const ridge = threadRidge(oc, { pitch: P, rMajor: major / 2, rMinor: minor / 2, angleDeg: 55, length: len })
    console.log('ridge valid', valid(oc, ridge), 'vol', props(oc, ridge).volume)
    const core = make(oc, 'BRepPrimAPI_MakeCylinder', minor / 2, len + 2 * P).Shape()
    const t0 = performance.now()
    const fuse = make(oc, 'BRepAlgoAPI_Fuse', core, ridge)
    const ms = performance.now() - t0
    const out = fuse.Shape()
    console.log('fuse ms', Math.round(ms), 'valid', valid(oc, out), 'vol core', props(oc, core).volume, 'vol out', props(oc, out).volume)
    expect(valid(oc, out)).toBe(true)
    expect(props(oc, out).volume).toBeGreaterThan(props(oc, core).volume)
  })
})
