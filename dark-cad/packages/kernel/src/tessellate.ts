import { call, make } from './occ'
import { faces } from './shape'

export interface Mesh {
  positions: Float32Array
  normals: Float32Array
  indices: Uint32Array
  /** One entry per B-rep face, in `faces(oc, shape)` order: triangle range for picking. */
  faceRanges: { start: number; count: number }[]
  /** Line segments (pairs of xyz) for B-rep edges. */
  edges: Float32Array
}

export function tessellate(oc: any, shape: any, deflection = 0.05, angular = 0.4): Mesh {
  make(oc, 'BRepMesh_IncrementalMesh', shape, deflection, false, angular, false)
  const pos: number[] = []
  const idx: number[] = []
  const ranges: { start: number; count: number }[] = []
  const fs = faces(oc, shape)
  for (const f of fs) {
    const loc = make(oc, 'TopLoc_Location')
    const h = call(oc.BRep_Tool, 'Triangulation', f, loc)
    const start = idx.length / 3
    if (h.IsNull()) { ranges.push({ start, count: 0 }); continue }
    const tri = h.get()
    const trsf = loc.Transformation()
    const base = pos.length / 3
    const n = tri.NbNodes()
    for (let i = 1; i <= n; i++) {
      const p = tri.Node(i).Transformed(trsf)
      pos.push(p.X(), p.Y(), p.Z())
    }
    const reversed = f.Orientation_1() === oc.TopAbs_Orientation.TopAbs_REVERSED
    const m = tri.NbTriangles()
    for (let i = 1; i <= m; i++) {
      const t = tri.Triangle(i)
      const a = t.Value(1) - 1 + base, b = t.Value(2) - 1 + base, c = t.Value(3) - 1 + base
      if (reversed) idx.push(a, c, b); else idx.push(a, b, c)
    }
    ranges.push({ start, count: m })
  }
  // Smooth normals within each face (faces don't share vertices, so edges stay crisp).
  const nrm = new Float32Array(pos.length)
  for (let i = 0; i < idx.length; i += 3) {
    const [a, b, c] = [idx[i] * 3, idx[i + 1] * 3, idx[i + 2] * 3]
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2]
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2]
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx
    for (const k of [a, b, c]) { nrm[k] += nx; nrm[k + 1] += ny; nrm[k + 2] += nz }
  }
  for (let i = 0; i < nrm.length; i += 3) {
    const l = Math.hypot(nrm[i], nrm[i + 1], nrm[i + 2]) || 1
    nrm[i] /= l; nrm[i + 1] /= l; nrm[i + 2] /= l
  }
  return {
    positions: new Float32Array(pos), normals: nrm, indices: new Uint32Array(idx), faceRanges: ranges,
    edges: edgeLines(oc, shape),
  }
}

function edgeLines(oc: any, shape: any): Float32Array {
  const out: number[] = []
  const seen: any[] = []
  const ex = make(oc, 'TopExp_Explorer', shape, oc.TopAbs_ShapeEnum.TopAbs_EDGE, oc.TopAbs_ShapeEnum.TopAbs_SHAPE)
  for (; ex.More(); ex.Next()) {
    const e = call(oc.TopoDS, 'Edge', ex.Current())
    if (seen.some((s) => s.IsSame(e))) continue
    seen.push(e)
    const c = make(oc, 'BRepAdaptor_Curve', e)
    const t0 = c.FirstParameter(), t1 = c.LastParameter()
    const isLine = c.GetType() === oc.GeomAbs_CurveType.GeomAbs_Line
    const n = isLine ? 1 : 48
    let prev = c.Value(t0)
    for (let i = 1; i <= n; i++) {
      const p = c.Value(t0 + ((t1 - t0) * i) / n)
      out.push(prev.X(), prev.Y(), prev.Z(), p.X(), p.Y(), p.Z())
      prev = p
    }
  }
  return new Float32Array(out)
}
