import { arcEnd, arcStart, dist, intersect, inSweep, angleOf, dot, polar, sub, len, nearest, midpoint, type ArcC, type CircleC, type Curve, type Geo, type Pt } from './geom'
import { gridLevels, snapToGrid } from './grid'

export type SnapKind = 'endpoint' | 'center' | 'midpoint' | 'intersection' | 'quadrant' | 'origin' | 'perpendicular' | 'tangent' | 'on-curve' | 'inference' | 'angle' | 'grid' | 'none'

export interface Guide { from: Pt; to: Pt }

export interface SnapResult {
  pt: Pt
  kind: SnapKind
  /** Entity the snap landed on: a point id for endpoint/center/origin, a curve id for on-curve/midpoint/quadrant/perpendicular/tangent. */
  ref?: string
  /** For intersections: both curve ids. */
  refs?: string[]
  label: string
  guides: Guide[]
}

export interface SnapOptions {
  /** Zoom in pixels per mm: converts the pixel pick radius to mm and chooses the grid step. */
  pxPerMm: number
  object: boolean
  grid: boolean
  /** Pick radius in pixels. */
  radiusPx?: number
  /** Previous point of the thing being drawn: enables perpendicular / tangent snaps and horizontal/vertical alignment. */
  from?: Pt
  /** Lock the direction from `from` to multiples of this angle (radians), e.g. with Shift held. */
  angleStep?: number
  /** Entities to ignore (the geometry being drawn or dragged). */
  exclude?: Set<string>
}

const LABEL: Record<SnapKind, string> = {
  endpoint: 'Endpoint', center: 'Center', midpoint: 'Midpoint', intersection: 'Intersection', quadrant: 'Quadrant', origin: 'Origin',
  perpendicular: 'Perpendicular', tangent: 'Tangent', 'on-curve': 'On curve', inference: 'Aligned', angle: 'Angle', grid: 'Grid', none: '',
}
const PRIORITY: Record<string, number> = { endpoint: 0, origin: 0, center: 1, intersection: 2, midpoint: 3, quadrant: 4, perpendicular: 5, tangent: 5, 'on-curve': 6 }

interface Cand { pt: Pt; kind: SnapKind; ref?: string; refs?: string[] }

/** Where does a cursor at `raw` land? Object snaps first, then alignment guides, then the adaptive grid. */
export function snap(raw: Pt, geo: Geo, o: SnapOptions): SnapResult {
  const r = (o.radiusPx ?? 10) / Math.max(o.pxPerMm, 1e-9)
  const ex = o.exclude ?? new Set<string>()
  const guides: Guide[] = []
  const result = (pt: Pt, kind: SnapKind, extra: Partial<SnapResult> = {}): SnapResult => ({ pt, kind, label: LABEL[kind], guides, ...extra })

  if (o.object) {
    const best = bestCandidate(raw, geo, o, r, ex)
    if (best) return result(best.pt, best.kind, { ref: best.ref, refs: best.refs })
  }

  let pt: Pt = [...raw]
  let kind: SnapKind = 'none'
  const { minor } = gridLevels(o.pxPerMm)

  if (o.from && o.angleStep) {
    const v = sub(raw, o.from), l = len(v)
    if (l > 1e-9) {
      const a = Math.round(angleOf(v) / o.angleStep) * o.angleStep
      const ll = o.grid ? Math.max(minor, Math.round(l / minor) * minor) : l
      pt = polar(o.from, ll, a)
      kind = 'angle'
      guides.push({ from: o.from, to: pt })
    }
  }

  if (o.object && kind === 'none') {
    // Alignment with existing points (and the origin): snap one or both coordinates, draw a dashed guide.
    let ix: Pt | undefined, iy: Pt | undefined
    const anchors: Pt[] = [...[...geo.points.entries()].filter(([id]) => !ex.has(id) && !id.startsWith('$')).map(([, p]) => p)]
    if (o.from) anchors.push(o.from)
    for (const a of anchors) {
      if (Math.abs(raw[0] - a[0]) < r && (!ix || Math.abs(raw[0] - a[0]) < Math.abs(raw[0] - ix[0]))) ix = a
      if (Math.abs(raw[1] - a[1]) < r && (!iy || Math.abs(raw[1] - a[1]) < Math.abs(raw[1] - iy[1]))) iy = a
    }
    if (ix) pt[0] = ix[0]
    if (iy) pt[1] = iy[1]
    if (ix || iy) kind = 'inference'
    if (ix) guides.push({ from: ix, to: [pt[0], pt[1]] })
    if (iy) guides.push({ from: iy, to: [pt[0], pt[1]] })
    if (kind === 'inference' && o.grid) {
      const g = snapToGrid(raw, minor)
      if (!ix) pt[0] = g[0]
      if (!iy) pt[1] = g[1]
    }
  }

  if (kind === 'none' && o.grid) { pt = snapToGrid(raw, minor); kind = 'grid' }
  return result(pt, kind)
}

function bestCandidate(raw: Pt, geo: Geo, o: SnapOptions, r: number, ex: Set<string>): Cand | undefined {
  let best: { c: Cand; score: number } | undefined
  const offer = (c: Cand) => {
    const d = dist(raw, c.pt)
    if (d > r) return
    const score = (PRIORITY[c.kind] ?? 9) * 1e3 + d / r
    if (!best || score < best.score) best = { c, score }
  }
  const centers = new Set<string>()
  for (const cu of geo.real) if (cu.kind !== 'line') centers.add(cu.kind === 'arc' ? cu.ic : cu.ic)
  for (const [id, p] of geo.points) {
    if (ex.has(id) || id.startsWith('$')) continue
    offer({ pt: p, kind: id === 'origin' ? 'origin' : centers.has(id) ? 'center' : 'endpoint', ref: id })
  }
  const curves = geo.curves.filter((c) => !ex.has(c.id))
  for (const c of curves) {
    if (c.virtual) continue
    if (c.kind === 'line' || c.kind === 'arc') offer({ pt: midpoint(c), kind: 'midpoint', ref: c.id })
    if (c.kind !== 'line') for (const q of quadrants(c)) offer({ pt: q, kind: 'quadrant', ref: c.id })
  }
  for (let i = 0; i < curves.length; i++) for (let j = i + 1; j < curves.length; j++) {
    const a = curves[i], b = curves[j]
    if (a.virtual && b.virtual) { offer({ pt: [0, 0], kind: 'origin', ref: 'origin' }); continue }
    for (const q of intersect(a, b)) offer({ pt: q, kind: 'intersection', refs: [a.id, b.id] })
  }
  if (o.from) {
    for (const c of curves) {
      if (c.virtual) continue
      for (const q of perpendicularFoot(c, o.from)) offer({ pt: q, kind: 'perpendicular', ref: c.id })
      if (c.kind !== 'line') for (const q of tangentPoints(c, o.from)) offer({ pt: q, kind: 'tangent', ref: c.id })
    }
  }
  for (const c of curves) {
    const n = nearest(c, raw)
    let pt = n.pt
    // Along a horizontal or vertical line (the plane axes included) keep the grid: slide to the nearest grid line crossing.
    if (o.grid && c.kind === 'line') {
      const { minor } = gridLevels(o.pxPerMm)
      if (Math.abs(c.a[1] - c.b[1]) < 1e-9) pt = [Math.round(pt[0] / minor) * minor, pt[1]]
      else if (Math.abs(c.a[0] - c.b[0]) < 1e-9) pt = [pt[0], Math.round(pt[1] / minor) * minor]
    }
    offer({ pt, kind: 'on-curve', ref: c.id })
  }
  return best?.c
}

function quadrants(c: ArcC | CircleC): Pt[] {
  const out: Pt[] = []
  for (let k = 0; k < 4; k++) {
    const ang = (k * Math.PI) / 2
    if (c.kind === 'circle' || inSweep(c, ang)) out.push(polar(c.c, c.r, ang))
  }
  return c.kind === 'arc' ? out.filter((q) => dist(q, arcStart(c)) > 1e-6 && dist(q, arcEnd(c)) > 1e-6) : out
}

/** Foot of the perpendicular from p onto the curve (within its extent). */
export function perpendicularFoot(c: Curve, p: Pt): Pt[] {
  if (c.kind === 'line') {
    const ab = sub(c.b, c.a), l2 = dot(ab, ab)
    if (l2 < 1e-12) return []
    const t = dot(sub(p, c.a), ab) / l2
    return t > 0 && t < 1 ? [[c.a[0] + ab[0] * t, c.a[1] + ab[1] * t]] : []
  }
  const v = sub(p, c.c)
  if (len(v) < 1e-9) return []
  const base = angleOf(v)
  return [base, base + Math.PI].filter((a) => c.kind === 'circle' || inSweep(c, a)).map((a) => polar(c.c, c.r, a))
}

/** Points on a circle/arc where a line from p touches it tangentially. */
export function tangentPoints(c: ArcC | CircleC, p: Pt): Pt[] {
  const d = dist(p, c.c)
  if (d <= c.r + 1e-9) return []
  const base = angleOf(sub(p, c.c)), off = Math.acos(c.r / d)
  return [base + off, base - off].filter((a) => c.kind === 'circle' || inSweep(c, a)).map((a) => polar(c.c, c.r, a))
}
