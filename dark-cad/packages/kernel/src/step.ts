import { call, make } from './occ'

export function writeSTEP(oc: any, shape: any): Uint8Array {
  const w = make(oc, 'STEPControl_Writer')
  call(w, 'Transfer', shape, oc.STEPControl_StepModelType.STEPControl_AsIs, true)
  const path = '/o.stp' // opencascade.js 1.1.1 mangles longer CString paths
  call(w, 'Write', path)
  const bytes = oc.FS.readFile(path) as Uint8Array
  oc.FS.unlink(path)
  return bytes
}

export function readSTEP(oc: any, bytes: Uint8Array): any {
  const path = '/i.stp'
  oc.FS.writeFile(path, bytes)
  const r = make(oc, 'STEPControl_Reader')
  const status = call(r, 'ReadFile', path)
  oc.FS.unlink(path)
  if (status !== oc.IFSelect_ReturnStatus.IFSelect_RetDone) throw new Error('STEP read failed')
  call(r, 'TransferRoots')
  return r.OneShape()
}
