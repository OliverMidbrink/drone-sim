import { buildGeo, curveBounds, distTo, tessellate, type Curve, type Geo, type Pt } from './geom'
import type { SketchEntity } from '@darkcad/core'

/** Entity under a point: points win over curves; the nearest within `tol` mm. Virtual entities are only offered on request. */
export function pickEntity(geo: Geo, p: Pt, tol: number, opts: { virtual?: boolean; only?: Array<'point' | 'line' | 'arc' | 'circle'>; exclude?: Set<string> } = {}): string | undefined {
  const ok = (kind: string, id: string) => (!opts.only || opts.only.includes(kind as any)) && !opts.exclude?.has(id)
  let bestPt: { id: string; d: number } | undefined
  for (const [id, q] of geo.points) {
    if (id.startsWith('$') || (id === 'origin' && !opts.virtual) || !ok('point', id)) continue
    const d = Math.hypot(p[0] - q[0], p[1] - q[1])
    if (d <= tol && (!bestPt || d < bestPt.d)) bestPt = { id, d }
  }
  if (bestPt) return bestPt.id
  let bestC: { id: string; d: number } | undefined
  for (const c of opts.virtual ? geo.curves : geo.real) {
    if (!ok(c.kind, c.id)) continue
    const d = distTo(c, p)
    // Real curves win ties against the plane axes.
    const score = d + (c.virtual ? 1e-3 : 0)
    if (d <= tol && (!bestC || score < bestC.d)) bestC = { id: c.id, d: score }
  }
  return bestC?.id
}

export interface Rect { x0: number; y0: number; x1: number; y1: number }

const inRect = (p: Pt, r: Rect) => p[0] >= r.x0 && p[0] <= r.x1 && p[1] >= r.y0 && p[1] <= r.y1

function segHitsRect(a: Pt, b: Pt, r: Rect): boolean {
  if (inRect(a, r) || inRect(b, r)) return true
  const edges: [Pt, Pt][] = [[[r.x0, r.y0], [r.x1, r.y0]], [[r.x1, r.y0], [r.x1, r.y1]], [[r.x1, r.y1], [r.x0, r.y1]], [[r.x0, r.y1], [r.x0, r.y0]]]
  const cr = (u: Pt, v: Pt, w: Pt) => (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
  return edges.some(([c, d]) => cr(a, b, c) * cr(a, b, d) <= 0 && cr(c, d, a) * cr(c, d, b) <= 0)
}

function curveInRect(c: Curve, r: Rect, crossing: boolean): boolean {
  if (!crossing) { const b = curveBounds(c); return b.x0 >= r.x0 && b.x1 <= r.x1 && b.y0 >= r.y0 && b.y1 <= r.y1 }
  const pts = tessellate(c, 0.05)
  const n = c.kind === 'circle' ? pts.length : pts.length - 1
  for (let i = 0; i < n; i++) if (segHitsRect(pts[i], pts[(i + 1) % pts.length], r)) return true
  return false
}

/** Window select (everything fully inside) or crossing select (anything touched). Virtual entities are never selected. */
export function boxSelect(entities: SketchEntity[], rect: Rect, crossing: boolean): string[] {
  const geo = buildGeo(entities)
  const out: string[] = []
  for (const e of entities) {
    if (e.kind === 'point') { if (inRect([e.x, e.y], rect)) out.push(e.id) }
    else { const c = geo.byId.get(e.id); if (c && curveInRect(c, rect, crossing)) out.push(e.id) }
  }
  return out
}
