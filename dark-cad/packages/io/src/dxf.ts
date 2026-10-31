import type { SketchEntity } from '@darkcad/core'
import { buildGeo, Draft, TAU, type Pt, type SketchModel } from '@darkcad/sketch'

const CONSTRUCTION_LAYER = 'CONSTRUCTION'

/**
 * ASCII DXF R12: widest compatibility. Lines, arcs and circles are written natively (arcs stay arcs);
 * construction geometry goes on its own layer so it round-trips. Free points are not exported.
 */
export function sketchToDxf(entities: SketchEntity[]): string {
  const n = (v: number) => String(Math.round(v * 1e6) / 1e6 || 0)
  const layer = (c: boolean) => ['8', c ? CONSTRUCTION_LAYER : '0']
  const out: string[] = [
    '0', 'SECTION', '2', 'HEADER', '9', '$INSUNITS', '70', '4', '0', 'ENDSEC',
    '0', 'SECTION', '2', 'TABLES',
    '0', 'TABLE', '2', 'LTYPE', '70', '1', '0', 'LTYPE', '2', 'CONTINUOUS', '70', '0', '3', 'Solid line', '72', '65', '73', '0', '40', '0.0', '0', 'ENDTAB',
    '0', 'TABLE', '2', 'LAYER', '70', '2',
    '0', 'LAYER', '2', '0', '70', '0', '62', '7', '6', 'CONTINUOUS',
    '0', 'LAYER', '2', CONSTRUCTION_LAYER, '70', '0', '62', '8', '6', 'CONTINUOUS',
    '0', 'ENDTAB', '0', 'ENDSEC',
    '0', 'SECTION', '2', 'ENTITIES',
  ]
  for (const c of buildGeo(entities).real) {
    if (c.kind === 'line') out.push('0', 'LINE', ...layer(c.construction), '10', n(c.a[0]), '20', n(c.a[1]), '30', '0', '11', n(c.b[0]), '21', n(c.b[1]), '31', '0')
    else if (c.kind === 'circle') out.push('0', 'CIRCLE', ...layer(c.construction), '10', n(c.c[0]), '20', n(c.c[1]), '30', '0', '40', n(c.r))
    else out.push('0', 'ARC', ...layer(c.construction), '10', n(c.c[0]), '20', n(c.c[1]), '30', '0', '40', n(c.r), '50', n(deg(c.a0)), '51', n(deg(c.a1)))
  }
  out.push('0', 'ENDSEC', '0', 'EOF')
  return out.join('\n') + '\n'
}

const deg = (r: number) => (((r * 180) / Math.PI) % 360 + 360) % 360

function pairs(text: string): [number, string][] {
  const lines = text.split(/\r?\n/)
  const out: [number, string][] = []
  for (let i = 0; i + 1 < lines.length; i += 2) out.push([parseInt(lines[i].trim(), 10), lines[i + 1].trim()])
  return out
}

type Raw =
  | { kind: 'line'; a: Pt; b: Pt; construction: boolean }
  | { kind: 'circle'; c: Pt; r: number; construction: boolean }
  /** Counter-clockwise from `a` to `b` about `c`. */
  | { kind: 'arc'; c: Pt; a: Pt; b: Pt; construction: boolean }

/** The arc a polyline bulge describes, as counter-clockwise (centre, start, end). */
function bulgeArc(p: Pt, q: Pt, bulge: number): Raw & { kind: 'arc' } {
  const theta = 4 * Math.atan(bulge) // signed sweep, + = counter-clockwise
  const chord = Math.hypot(q[0] - p[0], q[1] - p[1])
  const r = chord / (2 * Math.sin(Math.abs(theta) / 2))
  // Centre sits on the chord's perpendicular bisector, h from the midpoint towards the left of p→q for a ccw arc under π.
  const h = r * Math.cos(theta / 2) * (bulge > 0 ? 1 : -1)
  const ux = (q[0] - p[0]) / chord, uy = (q[1] - p[1]) / chord
  const c: Pt = [(p[0] + q[0]) / 2 - uy * h, (p[1] + q[1]) / 2 + ux * h]
  return bulge > 0 ? { kind: 'arc', c, a: p, b: q, construction: false } : { kind: 'arc', c, a: q, b: p, construction: false }
}

/**
 * Reads LINE, CIRCLE, ARC, LWPOLYLINE and POLYLINE (bulges become arcs). Curves that meet at a point share it.
 * Layer `CONSTRUCTION` becomes construction geometry. Other entity types are skipped.
 */
export function dxfToSketch(text: string, tol = 1e-4): SketchModel {
  const raws: Raw[] = []
  const ps = pairs(text)
  let inEntities = false
  const pushPoly = (pts: Pt[], bulges: number[], closed: boolean, construction: boolean) => {
    const count = closed ? pts.length : pts.length - 1
    for (let k = 0; k < count; k++) {
      const p = pts[k], q = pts[(k + 1) % pts.length], b = bulges[k] ?? 0
      if (Math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-12) continue
      if (Math.abs(b) > 1e-9) raws.push({ ...bulgeArc(p, q, b), construction })
      else raws.push({ kind: 'line', a: p, b: q, construction })
    }
  }
  const groups = (i: number) => {
    const grab: Record<number, string[]> = {}
    let j = i + 1
    for (; j < ps.length && ps[j][0] !== 0; j++) (grab[ps[j][0]] ??= []).push(ps[j][1])
    return { grab, end: j }
  }
  for (let i = 0; i < ps.length; i++) {
    const [code, val] = ps[i]
    if (code === 2 && val === 'ENTITIES') inEntities = true
    if (code === 0 && val === 'ENDSEC') inEntities = false
    if (!inEntities || code !== 0) continue
    const { grab, end } = groups(i)
    const g = (c: number, k = 0) => parseFloat(grab[c]?.[k] ?? '0') || 0
    const construction = (grab[8]?.[0] ?? '').toUpperCase() === CONSTRUCTION_LAYER
    if (val === 'LINE') raws.push({ kind: 'line', a: [g(10), g(20)], b: [g(11), g(21)], construction })
    else if (val === 'CIRCLE') raws.push({ kind: 'circle', c: [g(10), g(20)], r: g(40), construction })
    else if (val === 'ARC') {
      const a0 = (g(50) * Math.PI) / 180
      let a1 = (g(51) * Math.PI) / 180
      while (a1 <= a0) a1 += TAU
      const c: Pt = [g(10), g(20)], r = g(40)
      if (a1 - a0 >= TAU - 1e-9) raws.push({ kind: 'circle', c, r, construction })
      else raws.push({ kind: 'arc', c, a: [c[0] + r * Math.cos(a0), c[1] + r * Math.sin(a0)], b: [c[0] + r * Math.cos(a1), c[1] + r * Math.sin(a1)], construction })
    } else if (val === 'LWPOLYLINE') {
      // Group 10 starts each vertex; 42 (bulge) follows its vertex, so walk the pairs in order.
      const pts: Pt[] = [], bulges: number[] = []
      for (let j = i + 1; j < end; j++) {
        const [c, v] = ps[j]
        if (c === 10) { pts.push([parseFloat(v), 0]); bulges.push(0) }
        else if (c === 20 && pts.length) pts[pts.length - 1][1] = parseFloat(v)
        else if (c === 42 && pts.length) bulges[pts.length - 1] = parseFloat(v)
      }
      pushPoly(pts, bulges, (g(70) & 1) === 1, construction)
    } else if (val === 'POLYLINE') {
      // Old-style: VERTEX entities follow until SEQEND.
      const pts: Pt[] = [], bulges: number[] = []
      let j = end
      while (j < ps.length && !(ps[j][0] === 0 && ps[j][1] === 'SEQEND')) {
        if (ps[j][0] === 0 && ps[j][1] === 'VERTEX') {
          const v = groups(j)
          const flags = parseFloat(v.grab[70]?.[0] ?? '0') || 0
          if ((flags & 16) === 0 && (flags & 128) === 0) { pts.push([parseFloat(v.grab[10]?.[0] ?? '0'), parseFloat(v.grab[20]?.[0] ?? '0')]); bulges.push(parseFloat(v.grab[42]?.[0] ?? '0') || 0) }
          j = v.end
        } else j++
      }
      pushPoly(pts, bulges, (g(70) & 1) === 1, construction)
      i = j
      continue
    }
    i = end - 1
  }
  return build(raws, tol)
}

/** Turn raw curves into entities, merging end points that coincide (within `tol`) into one shared point. */
function build(raws: Raw[], tol: number): SketchModel {
  const d = new Draft()
  const cells = new Map<string, { id: string; p: Pt }[]>()
  const key = (ix: number, iy: number) => ix + ',' + iy
  const pointAt = (p: Pt, construction: boolean): string => {
    const ix = Math.round(p[0] / tol), iy = Math.round(p[1] / tol)
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      for (const h of cells.get(key(ix + dx, iy + dy)) ?? []) if (Math.hypot(h.p[0] - p[0], h.p[1] - p[1]) <= tol) return h.id
    }
    const id = d.point(p[0], p[1], construction)
    const k = key(ix, iy)
    cells.set(k, [...(cells.get(k) ?? []), { id, p }])
    return id
  }
  for (const r of raws) {
    if (r.kind === 'line') {
      const a = pointAt(r.a, r.construction), b = pointAt(r.b, r.construction)
      if (a !== b) d.line(a, b, r.construction)
    } else if (r.kind === 'circle') {
      if (r.r > 0) d.circle(d.point(r.c[0], r.c[1], r.construction), r.r, r.construction)
    } else {
      const a = pointAt(r.a, r.construction), b = pointAt(r.b, r.construction)
      if (a !== b) d.arc(d.point(r.c[0], r.c[1], r.construction), a, b, r.construction)
    }
  }
  // A construction point shared with real geometry is a real point.
  const real = new Set<string>()
  for (const e of d.entities) if (e.kind !== 'point' && !e.construction) for (const ref of e.kind === 'line' ? [e.a, e.b] : e.kind === 'arc' ? [e.a, e.b] : []) real.add(ref)
  for (const e of d.entities) if (e.kind === 'point' && real.has(e.id)) delete e.construction
  return d.model()
}
