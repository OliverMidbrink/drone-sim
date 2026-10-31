export interface TriMesh { positions: ArrayLike<number>; indices: ArrayLike<number> }

/** Binary STL from one or more triangle meshes (units: mm, as modelled). */
export function meshesToStl(meshes: TriMesh[], header = 'DarkCAD'): Uint8Array {
  const tris = meshes.reduce((n, m) => n + m.indices.length / 3, 0)
  const buf = new ArrayBuffer(84 + tris * 50)
  const dv = new DataView(buf)
  for (let i = 0; i < Math.min(80, header.length); i++) dv.setUint8(i, header.charCodeAt(i))
  dv.setUint32(80, tris, true)
  let o = 84
  for (const m of meshes) {
    const p = m.positions, ix = m.indices
    for (let t = 0; t < ix.length; t += 3) {
      const a = ix[t] * 3, b = ix[t + 1] * 3, c = ix[t + 2] * 3
      const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2]
      const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2]
      let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx
      const l = Math.hypot(nx, ny, nz) || 1
      nx /= l; ny /= l; nz /= l
      for (const v of [nx, ny, nz, p[a], p[a + 1], p[a + 2], p[b], p[b + 1], p[b + 2], p[c], p[c + 1], p[c + 2]]) { dv.setFloat32(o, v, true); o += 4 }
      dv.setUint16(o, 0, true); o += 2
    }
  }
  return new Uint8Array(buf)
}
