import type { SketchEntity } from '@darkcad/core'

export type Pt = [number, number]
export const TAU = Math.PI * 2
export const EPS = 1e-9

export const sub = (a: Pt, b: Pt): Pt => [a[0] - b[0], a[1] - b[1]]
export const add = (a: Pt, b: Pt): Pt => [a[0] + b[0], a[1] + b[1]]
export const scale = (a: Pt, k: number): Pt => [a[0] * k, a[1] * k]
export const dot = (a: Pt, b: Pt) => a[0] * b[0] + a[1] * b[1]
export const cross = (a: Pt, b: Pt) => a[0] * b[1] - a[1] * b[0]
export const len = (a: Pt) => Math.hypot(a[0], a[1])
export const dist = (a: Pt, b: Pt) => Math.hypot(a[0] - b[0], a[1] - b[1])
export const mid = (a: Pt, b: Pt): Pt => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
export const norm = (a: Pt): Pt => { const l = len(a); return l < EPS ? [1, 0] : [a[0] / l, a[1] / l] }
export const rot90 = (a: Pt): Pt => [-a[1], a[0]]
export const lerp = (a: Pt, b: Pt, t: number): Pt => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
export const angleOf = (a: Pt) => Math.atan2(a[1], a[0])
export const polar = (c: Pt, r: number, ang: number): Pt => [c[0] + r * Math.cos(ang), c[1] + r * Math.sin(ang)]
export const near = (a: Pt, b: Pt, tol = 1e-6) => dist(a, b) <= tol
/** Angle wrapped into [0, 2π). */
export const wrap = (a: number) => ((a % TAU) + TAU) % TAU
/** Angle wrapped into (-π, π]. */
export const wrapPi = (a: number) => { const w = wrap(a + Math.PI) - Math.PI; return w === -Math.PI ? Math.PI : w }

export interface LineC { kind: 'line'; id: string; a: Pt; b: Pt; ia: string; ib: string; construction: boolean; virtual?: boolean }
/** Counter-clockwise from angle a0 to a1 (a1 > a0, sweep at most 2π). */
export interface ArcC { kind: 'arc'; id: string; c: Pt; r: number; a0: number; a1: number; ic: string; ia: string; ib: string; construction: boolean; virtual?: boolean }
export interface CircleC { kind: 'circle'; id: string; c: Pt; r: number; ic: string; construction: boolean; virtual?: boolean }
export type Curve = LineC | ArcC | CircleC

export const VIRTUAL_POINTS = ['origin', '$xe', '$ye'] as const
export const VIRTUAL_IDS = ['origin', 'axis-x', 'axis-y']
export const isVirtual = (id: string) => VIRTUAL_IDS.includes(id) || id.startsWith('$')

/** Entities resolved to coordinates. Virtual entities (origin and the two plane axes) are always present. */
export interface Geo {
  points: Map<string, Pt>
  curves: Curve[]
  byId: Map<string, Curve>
  /** Real (non-virtual) curves only. */
  real: Curve[]
  point(id: string): Pt
}

export function arcOf(id: string, c: Pt, a: Pt, b: Pt, ic: string, ia: string, ib: string, construction = false): ArcC {
  const a0 = angleOf(sub(a, c))
  let a1 = angleOf(sub(b, c))
  if (a1 <= a0 + 1e-12) a1 += TAU
  return { kind: 'arc', id, c, r: dist(a, c), a0, a1, ic, ia, ib, construction }
}

export function buildGeo(entities: SketchEntity[]): Geo {
  const points = new Map<string, Pt>([['origin', [0, 0]], ['$xe', [1, 0]], ['$ye', [0, 1]]])
  for (const e of entities) if (e.kind === 'point') points.set(e.id, [e.x, e.y])
  const pt = (id: string): Pt => points.get(id) ?? [0, 0]
  const BIG = 1e5
  const virtual: Curve[] = [
    { kind: 'line', id: 'axis-x', a: [-BIG, 0], b: [BIG, 0], ia: 'origin', ib: '$xe', construction: true, virtual: true },
    { kind: 'line', id: 'axis-y', a: [0, -BIG], b: [0, BIG], ia: 'origin', ib: '$ye', construction: true, virtual: true },
  ]
  const real: Curve[] = []
  for (const e of entities) {
    if (e.kind === 'line') real.push({ kind: 'line', id: e.id, a: pt(e.a), b: pt(e.b), ia: e.a, ib: e.b, construction: !!e.construction })
    else if (e.kind === 'arc') real.push(arcOf(e.id, pt(e.c), pt(e.a), pt(e.b), e.c, e.a, e.b, !!e.construction))
    else if (e.kind === 'circle') real.push({ kind: 'circle', id: e.id, c: pt(e.c), r: e.r, ic: e.c, construction: !!e.construction })
  }
  const curves = [...virtual, ...real]
  return { points, curves, real, byId: new Map(curves.map((c) => [c.id, c])), point: pt }
}

export const arcStart = (c: ArcC) => polar(c.c, c.r, c.a0)
export const arcEnd = (c: ArcC) => polar(c.c, c.r, c.a1)
export const arcMid = (c: ArcC) => polar(c.c, c.r, (c.a0 + c.a1) / 2)
export const arcSweep = (c: ArcC) => c.a1 - c.a0
/** Is the direction angle inside the arc's sweep? (tolerance in radians) */
export function inSweep(c: ArcC, ang: number, tol = 1e-9): boolean {
  const d = wrap(ang - c.a0)
  return d <= arcSweep(c) + tol || d >= TAU - tol
}

export function endpoints(c: Curve): Pt[] {
  if (c.kind === 'line') return [c.a, c.b]
  if (c.kind === 'arc') return [arcStart(c), arcEnd(c)]
  return []
}
export function midpoint(c: Curve): Pt {
  if (c.kind === 'line') return mid(c.a, c.b)
  if (c.kind === 'arc') return arcMid(c)
  return polar(c.c, c.r, 0)
}

/** Nearest point on the curve, its distance, and a parameter (line: 0..1, arc: 0..1 along the sweep, circle: angle). */
export function nearest(c: Curve, p: Pt): { pt: Pt; d: number; t: number } {
  if (c.kind === 'line') {
    const ab = sub(c.b, c.a), l2 = dot(ab, ab)
    const t = l2 < EPS ? 0 : Math.max(0, Math.min(1, dot(sub(p, c.a), ab) / l2))
    const q = lerp(c.a, c.b, t)
    return { pt: q, d: dist(p, q), t }
  }
  const v = sub(p, c.c)
  const ang = len(v) < EPS ? 0 : angleOf(v)
  if (c.kind === 'circle') { const q = polar(c.c, c.r, ang); return { pt: q, d: dist(p, q), t: ang } }
  if (inSweep(c, ang)) { const q = polar(c.c, c.r, ang); return { pt: q, d: dist(p, q), t: wrap(ang - c.a0) / arcSweep(c) } }
  const s = arcStart(c), e = arcEnd(c)
  return dist(p, s) <= dist(p, e) ? { pt: s, d: dist(p, s), t: 0 } : { pt: e, d: dist(p, e), t: 1 }
}

/** Infinite-extent nearest point (full line, full circle). */
export function nearestInfinite(c: Curve, p: Pt): { pt: Pt; d: number } {
  if (c.kind === 'line') {
    const ab = sub(c.b, c.a), l2 = dot(ab, ab)
    const t = l2 < EPS ? 0 : dot(sub(p, c.a), ab) / l2
    const q = lerp(c.a, c.b, t)
    return { pt: q, d: dist(p, q) }
  }
  const v = sub(p, c.c)
  const q = polar(c.c, c.r, len(v) < EPS ? 0 : angleOf(v))
  return { pt: q, d: dist(p, q) }
}

/** Distance from p to the curve. */
export const distTo = (c: Curve, p: Pt) => nearest(c, p).d

/** Is q (assumed on the supporting line/circle) within the curve's extent? */
export function onExtent(c: Curve, q: Pt, tol = 1e-7): boolean { return within(c, q, tol) }
function within(c: Curve, q: Pt, tol: number): boolean {
  if (c.kind === 'line') {
    const ab = sub(c.b, c.a), l = len(ab)
    if (l < EPS) return false
    const t = dot(sub(q, c.a), ab) / (l * l)
    return t >= -tol / l && t <= 1 + tol / l
  }
  if (c.kind === 'circle') return true
  return inSweep(c, angleOf(sub(q, c.c)), tol / Math.max(c.r, EPS))
}

/**
 * Intersection points of two curves. By default only points on both curves' actual extent count; with
 * `extend` the lines and arcs are treated as unbounded lines and full circles.
 */
export function intersect(a: Curve, b: Curve, extend = false, tol = 1e-7): Pt[] {
  const raw = intersectInfinite(a, b)
  return extend ? raw : raw.filter((q) => within(a, q, tol) && within(b, q, tol))
}

function intersectInfinite(a: Curve, b: Curve): Pt[] {
  if (a.kind === 'line' && b.kind === 'line') {
    const r = sub(a.b, a.a), s = sub(b.b, b.a), den = cross(r, s)
    if (Math.abs(den) < 1e-12 * Math.max(1, len(r) * len(s))) return []
    const t = cross(sub(b.a, a.a), s) / den
    return [[a.a[0] + t * r[0], a.a[1] + t * r[1]]]
  }
  if (a.kind === 'line') return lineCircle(a, b as ArcC | CircleC)
  if (b.kind === 'line') return lineCircle(b, a as ArcC | CircleC)
  return circleCircle(a as ArcC | CircleC, b as ArcC | CircleC)
}

function lineCircle(l: LineC, c: ArcC | CircleC): Pt[] {
  const d = norm(sub(l.b, l.a))
  const f = sub(l.a, c.c)
  const b = dot(f, d), cc = dot(f, f) - c.r * c.r
  const disc = b * b - cc
  if (disc < -1e-9 * Math.max(1, c.r * c.r)) return []
  const sq = Math.sqrt(Math.max(0, disc))
  const ts = sq < 1e-9 ? [-b] : [-b - sq, -b + sq]
  return ts.map((t) => [l.a[0] + d[0] * t, l.a[1] + d[1] * t] as Pt)
}

function circleCircle(a: ArcC | CircleC, b: ArcC | CircleC): Pt[] {
  const d = dist(a.c, b.c)
  if (d < EPS) return []
  if (d > a.r + b.r + 1e-9 || d < Math.abs(a.r - b.r) - 1e-9) return []
  const x = (d * d + a.r * a.r - b.r * b.r) / (2 * d)
  const h2 = a.r * a.r - x * x
  const u = norm(sub(b.c, a.c))
  const base: Pt = [a.c[0] + u[0] * x, a.c[1] + u[1] * x]
  if (h2 < 1e-12) return [base]
  const h = Math.sqrt(h2), v = rot90(u)
  return [[base[0] + v[0] * h, base[1] + v[1] * h], [base[0] - v[0] * h, base[1] - v[1] * h]]
}

/** Circle through three points. */
export function circle3(p1: Pt, p2: Pt, p3: Pt): { c: Pt; r: number } | undefined {
  const d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
  if (Math.abs(d) < 1e-12) return
  const s1 = dot(p1, p1), s2 = dot(p2, p2), s3 = dot(p3, p3)
  const c: Pt = [(s1 * (p2[1] - p3[1]) + s2 * (p3[1] - p1[1]) + s3 * (p1[1] - p2[1])) / d, (s1 * (p3[0] - p2[0]) + s2 * (p1[0] - p3[0]) + s3 * (p2[0] - p1[0])) / d]
  return { c, r: dist(c, p1) }
}

/** Direction of travel leaving the curve at one of its ends (`end` 0 = start, 1 = end), pointing away from the curve. */
export function leaveDir(c: Curve, end: 0 | 1): Pt {
  if (c.kind === 'line') return norm(end === 0 ? sub(c.b, c.a) : sub(c.a, c.b))
  if (c.kind === 'arc') {
    const ang = end === 0 ? c.a0 : c.a1
    const t: Pt = [-Math.sin(ang), Math.cos(ang)] // CCW tangent
    return end === 0 ? t : [-t[0], -t[1]]
  }
  return [1, 0]
}

/** Signed curvature of the path leaving the curve at `end` (positive = turning left). */
export function leaveCurvature(c: Curve, end: 0 | 1): number {
  if (c.kind === 'line') return 0
  const k = 1 / Math.max(c.r, EPS)
  return end === 0 ? k : -k
}

/** Points along the curve (open polyline for lines/arcs, closed ring without repeated end for circles). */
export function tessellate(c: Curve, tol = 0.01): Pt[] {
  if (c.kind === 'line') return [c.a, c.b]
  const sweep = c.kind === 'circle' ? TAU : arcSweep(c)
  const a0 = c.kind === 'circle' ? 0 : c.a0
  const step = c.r < tol ? Math.PI / 2 : 2 * Math.acos(Math.max(0, 1 - tol / c.r))
  const n = Math.max(c.kind === 'circle' ? 24 : 4, Math.min(720, Math.ceil(sweep / Math.max(step, 1e-3))))
  const out: Pt[] = []
  const count = c.kind === 'circle' ? n : n + 1
  for (let i = 0; i < count; i++) out.push(polar(c.c, c.r, a0 + (sweep * i) / n))
  if (c.kind === 'arc') { out[0] = arcStart(c); out[n] = arcEnd(c) }
  return out
}

export function polyArea(p: Pt[]): number {
  let s = 0
  for (let i = 0; i < p.length; i++) { const q = p[(i + 1) % p.length]; s += p[i][0] * q[1] - q[0] * p[i][1] }
  return s / 2
}

export function pointInPoly(p: Pt, poly: Pt[]): boolean {
  let c = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j]
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) c = !c
  }
  return c
}

export function bounds(pts: Pt[]) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
  for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y) }
  return { x0, y0, x1, y1 }
}

/** Bounding box of the whole curve. */
export function curveBounds(c: Curve) {
  if (c.kind === 'line') return bounds([c.a, c.b])
  if (c.kind === 'circle') return { x0: c.c[0] - c.r, y0: c.c[1] - c.r, x1: c.c[0] + c.r, y1: c.c[1] + c.r }
  return bounds(tessellate(c, c.r / 50))
}
