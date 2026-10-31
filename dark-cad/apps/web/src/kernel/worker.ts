/// <reference lib="webworker" />
import ocFactory from 'opencascade.js/dist/opencascade.wasm.js'
import wasmUrl from 'opencascade.js/dist/opencascade.wasm.wasm?url'
import { call, faces, make, tessellate, writeSTEP } from '@darkcad/kernel'
import { fingerprintFace, rebuild, type BuildResult } from '@darkcad/features'
import type { FaceInfo, Request, Response } from './protocol'

let oc: any
let last: BuildResult | undefined
let seq = 0
const send = (r: Response, transfer: Transferable[] = []) => (self as any).postMessage(r, transfer)

const ready = (async () => {
  oc = await new (ocFactory as any)({ locateFile: (p: string) => (p.endsWith('.wasm') ? wasmUrl : p) })
  send({ type: 'ready' })
})()

self.onmessage = async (ev: MessageEvent<Request>) => {
  await ready
  const msg = ev.data
  try {
    if (msg.type === 'build') {
      const t0 = performance.now()
      last = rebuild(oc, msg.doc)
      const bodies = last.visible.map((id) => ({ id, mesh: tessellate(oc, last!.bodies.get(id)) }))
      const transfer = bodies.flatMap(({ mesh: m }) => [m.positions.buffer, m.normals.buffer, m.indices.buffer, m.edges.buffer] as Transferable[])
      send({ type: 'built', seq: ++seq, bodies, errors: [...last.errors], ms: performance.now() - t0 }, transfer)
    } else if (msg.type === 'step') {
      if (!last || !last.visible.length) throw new Error('nothing to export')
      const comp = make(oc, 'TopoDS_Compound')
      const builder = make(oc, 'BRep_Builder')
      builder.MakeCompound(comp)
      for (const id of last.visible) builder.Add(comp, last.bodies.get(id))
      const bytes = writeSTEP(oc, comp)
      send({ type: 'step', bytes }, [bytes.buffer])
    } else if (msg.type === 'faceInfo') {
      const shape = last?.bodies.get(msg.body)
      if (!shape) throw new Error('body not found')
      const face = faces(oc, shape)[msg.face]
      const fp = fingerprintFace(oc, face)
      const info: FaceInfo = { ...fp }
      if (fp.surface === 'Cylinder') {
        const cyl = make(oc, 'BRepAdaptor_Surface', face, true).Cylinder()
        const ax = cyl.Axis(), p = ax.Location(), d = ax.Direction()
        info.cylinder = { origin: [p.X(), p.Y(), p.Z()], dir: [d.X(), d.Y(), d.Z()], radius: cyl.Radius() }
      }
      send({ type: 'faceInfo', info })
    }
  } catch (e: any) {
    send({ type: 'error', message: String(e?.message ?? e) })
  }
}
void call
