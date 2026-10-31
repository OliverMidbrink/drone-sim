import type { SketchEntity } from '@darkcad/core'
import { outline } from '@darkcad/features'

/** ASCII DXF R12: widest compatibility. Circles stay circles; rectangles/polygons become closed LINE loops. */
export function sketchToDxf(entities: SketchEntity[]): string {
  const out: string[] = ['0', 'SECTION', '2', 'HEADER', '9', '$INSUNITS', '70', '4', '0', 'ENDSEC', '0', 'SECTION', '2', 'ENTITIES']
  const n = (v: number) => String(Math.round(v * 1e6) / 1e6)
  for (const e of entities) {
    if (e.kind === 'circle') out.push('0', 'CIRCLE', '8', '0', '10', n(e.x), '20', n(e.y), '30', '0', '40', n(e.r))
    else {
      const p = outline(e)
      p.forEach(([x, y], i) => {
        const [x2, y2] = p[(i + 1) % p.length]
        out.push('0', 'LINE', '8', '0', '10', n(x), '20', n(y), '30', '0', '11', n(x2), '21', n(y2), '31', '0')
      })
    }
  }
  out.push('0', 'ENDSEC', '0', 'EOF')
  return out.join('\n') + '\n'
}

function pairs(text: string): [number, string][] {
  const lines = text.split(/\r?\n/)
  const out: [number, string][] = []
  for (let i = 0; i + 1 < lines.length; i += 2) out.push([parseInt(lines[i].trim(), 10), lines[i + 1].trim()])
  return out
}

interface Seg { a: [number, number]; b: [number, number] }

/** Reads LINE, CIRCLE, ARC (as chords), LWPOLYLINE (bulges ignored). Loose LINEs are chained into closed polygons by endpoint. */
export function dxfToSketch(text: string, tol = 1e-4): SketchEntity[] {
  const entities: SketchEntity[] = []
  const segs: Seg[] = []
  let id = 0
  const nid = () => `d${++id}`
  const ps = pairs(text)
  let inEntities = false
  for (let i = 0; i < ps.length; i++) {
    const [code, val] = ps[i]
    if (code === 2 && val === 'ENTITIES') inEntities = true
    if (code === 0 && val === 'ENDSEC') inEntities = false
    if (!inEntities || code !== 0) continue
    const grab: Record<number, number[]> = {}
    let j = i + 1
    for (; j < ps.length && ps[j][0] !== 0; j++) (grab[ps[j][0]] ??= []).push(parseFloat(ps[j][1]))
    const g = (c: number, k = 0) => grab[c]?.[k] ?? 0
    if (val === 'LINE') segs.push({ a: [g(10), g(20)], b: [g(11), g(21)] })
    else if (val === 'CIRCLE') entities.push({ id: nid(), kind: 'circle', x: g(10), y: g(20), r: g(40) })
    else if (val === 'ARC') {
      const a0 = (g(50) * Math.PI) / 180
      let a1 = (g(51) * Math.PI) / 180
      if (a1 <= a0) a1 += 2 * Math.PI
      const k = Math.max(4, Math.ceil(((a1 - a0) / (2 * Math.PI)) * 32))
      let prev: [number, number] = [g(10) + g(40) * Math.cos(a0), g(20) + g(40) * Math.sin(a0)]
      for (let s = 1; s <= k; s++) {
        const t = a0 + ((a1 - a0) * s) / k
        const p: [number, number] = [g(10) + g(40) * Math.cos(t), g(20) + g(40) * Math.sin(t)]
        segs.push({ a: prev, b: p }); prev = p
      }
    } else if (val === 'LWPOLYLINE') {
      const xs = grab[10] ?? [], ys = grab[20] ?? []
      const pts = xs.map((x, k) => [x, ys[k]] as [number, number])
      if (pts.length >= 3 && ((g(70) & 1) === 1 || near(pts[0], pts[pts.length - 1], tol))) entities.push({ id: nid(), kind: 'polygon', points: near(pts[0], pts[pts.length - 1], tol) ? pts.slice(0, -1) : pts })
      else pts.slice(0, -1).forEach((p, k) => segs.push({ a: p, b: pts[k + 1] }))
    }
    i = j - 1
  }
  // Chain loose segments into loops.
  const used = new Array(segs.length).fill(false)
  for (let s = 0; s < segs.length; s++) {
    if (used[s]) continue
    used[s] = true
    const loop: [number, number][] = [segs[s].a, segs[s].b]
    let grew = true
    while (grew && !near(loop[0], loop[loop.length - 1], tol)) {
      grew = false
      for (let k = 0; k < segs.length; k++) {
        if (used[k]) continue
        const end = loop[loop.length - 1]
        if (near(segs[k].a, end, tol)) { loop.push(segs[k].b); used[k] = grew = true; break }
        if (near(segs[k].b, end, tol)) { loop.push(segs[k].a); used[k] = grew = true; break }
      }
    }
    if (near(loop[0], loop[loop.length - 1], tol) && loop.length >= 4) {
      const pts = loop.slice(0, -1)
      entities.push(asRect(pts, tol) ?? { id: nid(), kind: 'polygon', points: pts })
    }
  }
  return entities.map((e, k) => ({ ...e, id: `e${k + 1}` }))
}

const near = (a: [number, number], b: [number, number], tol: number) => Math.hypot(a[0] - b[0], a[1] - b[1]) <= tol

function asRect(pts: [number, number][], tol: number): SketchEntity | undefined {
  if (pts.length !== 4) return
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1])
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys)
  const corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]] as [number, number][]
  if (corners.every((c) => pts.some((p) => near(p, c, tol)))) return { id: 'r', kind: 'rect', x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
}
