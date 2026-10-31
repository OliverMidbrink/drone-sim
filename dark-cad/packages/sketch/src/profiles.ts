import type { SketchConstraint, SketchEntity } from '@darkcad/core'
import { arcEnd, arcMid, arcStart, buildGeo, distTo, endpoints, intersect, leaveCurvature, leaveDir, midpoint, nearest, pointInPoly, polyArea, tessellate, angleOf, dist, lerp, polar, wrap, TAU, type ArcC, type CircleC, type Curve, type Geo, type Pt } from './geom'

/** One piece of a closed boundary, in traversal order. */
export type Seg =
  | { kind: 'line'; id: string; from: Pt; to: Pt }
  | { kind: 'arc'; id: string; c: Pt; r: number; from: Pt; to: Pt; mid: Pt; ccw: boolean }
  | { kind: 'circle'; id: string; c: Pt; r: number }

/** A closed loop of curves, always counter-clockwise. */
export interface Loop { segs: Seg[]; poly: Pt[]; area: number }
/** A solid region of the sketch: an outer loop with holes cut out of it. */
export interface Region { outer: Loop; holes: Loop[] }
export interface Profiles {
  regions: Region[]
  /** Non-construction curves that are not part of any closed loop (kept and drawn, but cannot be extruded). */
  open: string[]
}

const TOL_POLY = 0.005

const reverseSeg = (s: Seg): Seg => s.kind === 'line' ? { ...s, from: s.to, to: s.from } : s.kind === 'arc' ? { ...s, from: s.to, to: s.from, ccw: !s.ccw } : s

function segPoints(s: Seg): Pt[] {
  if (s.kind === 'line') return [s.from]
  if (s.kind === 'circle') return tessellate({ kind: 'circle', id: s.id, c: s.c, r: s.r, ic: '', construction: false }, TOL_POLY)
  // arc: sample along the travelled direction (excluding the end point, which starts the next segment)
  const a0 = Math.atan2(s.from[1] - s.c[1], s.from[0] - s.c[0]), a1 = Math.atan2(s.to[1] - s.c[1], s.to[0] - s.c[0])
  let sweep = s.ccw ? a1 - a0 : a0 - a1
  sweep = ((sweep % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)
  if (sweep < 1e-9) sweep = 2 * Math.PI
  const step = s.r < TOL_POLY ? Math.PI / 2 : 2 * Math.acos(Math.max(0, 1 - TOL_POLY / s.r))
  const n = Math.max(4, Math.min(720, Math.ceil(sweep / Math.max(step, 1e-3))))
  const out: Pt[] = []
  for (let i = 0; i < n; i++) {
    const a = a0 + (s.ccw ? 1 : -1) * (sweep * i) / n
    out.push([s.c[0] + s.r * Math.cos(a), s.c[1] + s.r * Math.sin(a)])
  }
  out[0] = s.from
  return out
}

/** Exact signed area enclosed by a closed chain of segments (positive when counter-clockwise). */
export function segsArea(segs: Seg[]): number {
  let a = 0
  for (const s of segs) {
    if (s.kind === 'circle') a += Math.PI * s.r * s.r
    else {
      a += (s.from[0] * s.to[1] - s.to[0] * s.from[1]) / 2
      if (s.kind === 'arc') {
        const a0 = Math.atan2(s.from[1] - s.c[1], s.from[0] - s.c[0]), a1 = Math.atan2(s.to[1] - s.c[1], s.to[0] - s.c[0])
        let th = s.ccw ? a1 - a0 : a0 - a1
        th = ((th % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)
        if (th < 1e-12) th = 2 * Math.PI
        th = s.ccw ? th : -th
        // area between the chord and the arc: ½ r² (θ − sin θ), signed along the travel direction
        a += 0.5 * s.r * s.r * (th - Math.sin(th))
      }
    }
  }
  return a
}

function makeLoop(segs: Seg[]): Loop {
  const poly = segs.flatMap(segPoints)
  const signed = segsArea(segs)
  if (signed < 0) {
    const rs = segs.map(reverseSeg).reverse()
    const rp = rs.flatMap(segPoints)
    return { segs: rs, poly: rp, area: -signed }
  }
  return { segs, poly, area: signed }
}

interface Edge { curve: Curve; n0: number; n1: number; src: string; also?: string[] }
interface Half { e: number; dir: 0 | 1; from: number; to: number; angle: number; curv: number }

const segOf = (e: Edge, dir: 0 | 1): Seg => {
  const c = e.curve
  if (c.kind === 'line') return dir === 0 ? { kind: 'line', id: c.id, from: c.a, to: c.b } : { kind: 'line', id: c.id, from: c.b, to: c.a }
  if (c.kind === 'arc') {
    const s = arcStart(c), t = arcEnd(c)
    return { kind: 'arc', id: c.id, c: c.c, r: c.r, from: dir === 0 ? s : t, to: dir === 0 ? t : s, mid: arcMid(c), ccw: dir === 0 }
  }
  return { kind: 'circle', id: c.id, c: c.c, r: c.r }
}

/** Groups point ids that are the same point: joined by a coincident constraint or lying on top of each other. */
export function nodeIndex(ids: string[], pos: (id: string) => Pt, constraints: SketchConstraint[]): Map<string, number> {
  const parent = new Map<string, string>(ids.map((i) => [i, i]))
  const find = (a: string): string => { let r = a; while (parent.get(r) !== r) r = parent.get(r)!; parent.set(a, r); return r }
  const union = (a: string, b: string) => { if (parent.has(a) && parent.has(b)) parent.set(find(a), find(b)) }
  for (const c of constraints) if (c.type === 'coincident') union(c.refs[0], c.refs[1])
  const cell = 1e-5
  const buckets = new Map<string, string[]>()
  for (const id of ids) {
    const [x, y] = pos(id)
    const kx = Math.round(x / cell), ky = Math.round(y / cell)
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (const o of buckets.get(`${kx + dx},${ky + dy}`) ?? []) if (dist(pos(o), [x, y]) < 1e-6) union(id, o)
    const k = `${kx},${ky}`
    buckets.set(k, [...(buckets.get(k) ?? []), id])
  }
  const index = new Map<string, number>(), roots = new Map<string, number>()
  for (const id of ids) {
    const r = find(id)
    if (!roots.has(r)) roots.set(r, roots.size)
    index.set(id, roots.get(r)!)
  }
  return index
}


const TOL_SPLIT = 1e-6

/**
 * Turns the drawn curves into a planar arrangement: every curve is split wherever another curve crosses it, touches it
 * (T-junctions) or lies along it (shared and partially shared edges), and duplicate pieces are dropped. Pieces remember
 * the curve they came from (`src`). Circles that touch nothing are left out (the caller handles them as closed loops).
 */
function planarize(curves: Curve[], geo: Geo, constraints: SketchConstraint[]): { edges: Edge[] } {
  const cuts = curves.map(() => [] as Pt[])
  curves.forEach((a, i) => curves.forEach((b, j) => {
    if (i === j) return
    for (const p of intersect(a, b, false, TOL_SPLIT)) cuts[i].push(p)
    for (const p of endpoints(b)) if (distTo(a, p) < TOL_SPLIT) cuts[i].push(p)
  }))

  interface Piece { curve: Curve; src: string; ia: string; ib: string }
  const pieces: Piece[] = []
  const pos = new Map<string, Pt>()
  let syn = 0
  const synth = (p: Pt) => { const id = `$s${syn++}`; pos.set(id, p); return id }
  const uniq = (xs: number[], tol: number) => { const out: number[] = []; for (const x of [...xs].sort((u, v) => u - v)) if (!out.length || x - out[out.length - 1] > tol) out.push(x); return out }

  curves.forEach((c, i) => {
    if (c.kind === 'line') {
      const l = dist(c.a, c.b)
      const ts = uniq(cuts[i].map((p) => nearest(c, p).t).filter((t) => t * l > TOL_SPLIT && (1 - t) * l > TOL_SPLIT), TOL_SPLIT / l)
      const bs = [0, ...ts, 1]
      const ids = bs.map((t, k) => (k === 0 ? c.ia : k === bs.length - 1 ? c.ib : synth(lerp(c.a, c.b, t))))
      for (let k = 0; k + 1 < bs.length; k++) pieces.push({ curve: { ...c, a: k === 0 ? c.a : lerp(c.a, c.b, bs[k]), b: k + 2 === bs.length ? c.b : lerp(c.a, c.b, bs[k + 1]) }, src: c.id, ia: ids[k], ib: ids[k + 1] })
    } else if (c.kind === 'arc') {
      const sw = c.a1 - c.a0, len = sw * c.r
      const ts = uniq(cuts[i].map((p) => nearest(c, p).t).filter((t) => t * len > TOL_SPLIT && (1 - t) * len > TOL_SPLIT), TOL_SPLIT / len)
      const bs = [0, ...ts, 1]
      const ids = bs.map((t, k) => (k === 0 ? c.ia : k === bs.length - 1 ? c.ib : synth(polar(c.c, c.r, c.a0 + t * sw))))
      for (let k = 0; k + 1 < bs.length; k++) pieces.push({ curve: { ...c, a0: c.a0 + bs[k] * sw, a1: c.a0 + bs[k + 1] * sw }, src: c.id, ia: ids[k], ib: ids[k + 1] })
    } else {
      let as = uniq(cuts[i].map((p) => wrap(Math.atan2(p[1] - c.c[1], p[0] - c.c[0]))), TOL_SPLIT / c.r)
      if (as.length > 1 && as[0] + TAU - as[as.length - 1] <= TOL_SPLIT / c.r) as = as.slice(0, -1)
      if (!as.length) return // untouched circle
      if (as.length === 1) as = [as[0], wrap(as[0] + Math.PI)].sort((u, v) => u - v)
      const ids = as.map((a) => synth(polar(c.c, c.r, a)))
      as.forEach((a, k) => {
        const next = k + 1 < as.length ? as[k + 1] : as[0] + TAU
        const arc: ArcC = { kind: 'arc', id: c.id, c: c.c, r: c.r, a0: a, a1: next, ic: c.ic, ia: '', ib: '', construction: c.construction }
        pieces.push({ curve: arc, src: c.id, ia: ids[k], ib: ids[(k + 1) % as.length] })
      })
    }
  })

  const ids = [...new Set(pieces.flatMap((p) => [p.ia, p.ib]))]
  const at = (id: string) => pos.get(id) ?? geo.point(id)
  const nodes = nodeIndex(ids, at, constraints)
  const edges: Edge[] = []
  const byNodes = new Map<string, Edge[]>()
  for (const p of pieces) {
    const n0 = nodes.get(p.ia)!, n1 = nodes.get(p.ib)!
    if (n0 === n1) continue
    const key = n0 < n1 ? `${n0}:${n1}` : `${n1}:${n0}`
    const m = midpoint(p.curve)
    const list = byNodes.get(key) ?? []
    const twin = list.find((e) => dist(midpoint(e.curve), m) < TOL_SPLIT * 10)
    if (twin) { if (twin.src !== p.src) (twin.also ??= []).push(p.src); continue } // the same edge drawn twice
    const e: Edge = { curve: p.curve, n0, n1, src: p.src }
    list.push(e); byNodes.set(key, list); edges.push(e)
  }
  return { edges }
}

interface Comp { faces: Loop[]; outer?: Loop; curveIds: Set<string> }

/**
 * Closed profiles of a sketch. Curves that meet at shared (or coincident) points form a planar graph; its bounded faces
 * are the regions. A component lying inside a region (circle in a rectangle) becomes a hole, one level deeper a solid again.
 */
export function profiles(entities: SketchEntity[], constraints: SketchConstraint[] = []): Profiles {
  const geo = buildGeo(entities)
  const usable = geo.real.filter((c) => !c.construction && (c.kind !== 'line' || dist(c.a, c.b) > 1e-9))
  const comps: Comp[] = []

  const plan = planarize(usable.filter((c) => c.kind !== 'circle' || c.r > 1e-9), geo, constraints)
  const edges = plan.edges
  const split = new Set(edges.flatMap((e) => [e.src, ...(e.also ?? [])]))
  // Circles that touch nothing are closed on their own; the ones that cross or touch other curves became arcs above.
  for (const c of usable) if (c.kind === 'circle' && c.r > 1e-9 && !split.has(c.id)) {
    const loop = makeLoop([segOf({ curve: c, n0: 0, n1: 0, src: c.id }, 0)])
    comps.push({ faces: [loop], outer: loop, curveIds: new Set([c.id]) })
  }

  // Prune dangling edges (open chains) until every node has at least two edges.
  const alive = edges.map(() => true)
  const degree = new Map<number, number>()
  const bump = (n: number, d: number) => degree.set(n, (degree.get(n) ?? 0) + d)
  edges.forEach((e) => { bump(e.n0, 1); bump(e.n1, 1) })
  for (let changed = true; changed;) {
    changed = false
    edges.forEach((e, i) => {
      if (alive[i] && ((degree.get(e.n0) ?? 0) < 2 || (degree.get(e.n1) ?? 0) < 2)) { alive[i] = false; bump(e.n0, -1); bump(e.n1, -1); changed = true }
    })
  }
  const live = edges.map((e, i) => i).filter((i) => alive[i])
  const used = new Set<string>(comps.flatMap((c) => [...c.curveIds]))

  if (live.length) {
    // Half-edges, sorted counter-clockwise around every node.
    const halves: Half[] = []
    for (const i of live) {
      const e = edges[i]
      halves.push({ e: i, dir: 0, from: e.n0, to: e.n1, angle: angleOf(leaveDir(e.curve, 0)), curv: leaveCurvature(e.curve, 0) })
      halves.push({ e: i, dir: 1, from: e.n1, to: e.n0, angle: angleOf(leaveDir(e.curve, 1)), curv: leaveCurvature(e.curve, 1) })
    }
    const around = new Map<number, Half[]>()
    for (const h of halves) around.set(h.from, [...(around.get(h.from) ?? []), h])
    for (const list of around.values()) list.sort((a, b) => (Math.abs(a.angle - b.angle) > 1e-9 ? a.angle - b.angle : a.curv - b.curv))
    const key = (h: Half) => h.e * 2 + h.dir
    const twin = (h: Half) => halves.find((t) => t.e === h.e && t.dir !== h.dir)!
    const next = (h: Half): Half => {
      const t = twin(h), list = around.get(h.to)!
      const i = list.indexOf(list.find((x) => key(x) === key(t))!)
      return list[(i - 1 + list.length) % list.length]
    }
    const visited = new Set<number>()
    // connected components over nodes
    const compOf = new Map<number, number>()
    const adj = new Map<number, number[]>()
    for (const i of live) { const e = edges[i]; adj.set(e.n0, [...(adj.get(e.n0) ?? []), e.n1]); adj.set(e.n1, [...(adj.get(e.n1) ?? []), e.n0]) }
    let nc = 0
    for (const start of adj.keys()) {
      if (compOf.has(start)) continue
      const stack = [start]
      compOf.set(start, nc)
      while (stack.length) { const n = stack.pop()!; for (const m of adj.get(n) ?? []) if (!compOf.has(m)) { compOf.set(m, nc); stack.push(m) } }
      nc++
    }
    const local: Comp[] = Array.from({ length: nc }, () => ({ faces: [], curveIds: new Set<string>() }))
    for (const h0 of halves) {
      if (visited.has(key(h0))) continue
      const cyc: Half[] = []
      for (let h = h0; !visited.has(key(h)); h = next(h)) { visited.add(key(h)); cyc.push(h) }
      const segs = cyc.map((h) => segOf(edges[h.e], h.dir))
      const poly = segs.flatMap(segPoints)
      const signed = segsArea(segs)
      const comp = local[compOf.get(h0.from)!]
      for (const h of cyc) for (const id of [edges[h.e].src, ...(edges[h.e].also ?? [])]) { comp.curveIds.add(id); used.add(id) }
      if (Math.abs(signed) < 1e-12) continue
      if (signed > 0) comp.faces.push({ segs, poly, area: signed })
      else if (!comp.outer || -signed > comp.outer.area) comp.outer = makeLoop(segs)
    }
    comps.push(...local.filter((c) => c.faces.length || c.outer))
  }

  // Nesting: depth = number of other components whose outer boundary encloses this one.
  const outerOf = (c: Comp) => c.outer ?? c.faces.reduce((a, b) => (b.area > a.area ? b : a))
  const depth = comps.map((c, i) => comps.filter((o, j) => j !== i && outerOf(o).area > outerOf(c).area && pointInPoly(outerOf(c).poly[0], outerOf(o).poly)).length)

  const regions: Region[] = []
  comps.forEach((c, i) => {
    if (depth[i] % 2) return
    for (const f of c.faces) regions.push({ outer: f, holes: [] })
  })
  comps.forEach((c, i) => {
    if (!(depth[i] % 2)) return
    const hole = makeLoop(outerOf(c).segs)
    let best: Region | undefined
    comps.forEach((p, j) => {
      if (depth[j] !== depth[i] - 1) return
      for (const f of p.faces) {
        const r = regions.find((rg) => rg.outer === f)
        if (r && pointInPoly(hole.poly[0], f.poly) && (!best || f.area < best.outer.area)) best = r
      }
    })
    best?.holes.push(hole)
  })
  regions.sort((a, b) => b.outer.area - a.outer.area)
  const open = usable.filter((c) => !used.has(c.id)).map((c) => c.id)
  return { regions, open }
}

/** Outer boundary of the largest region (for lofting), or undefined when the sketch has no closed profile. */
export function outerLoop(p: Profiles): Loop | undefined { return p.regions[0]?.outer }

/** Sum of region areas minus holes (mm²). */
export const profileArea = (p: Profiles) => p.regions.reduce((s, r) => s + r.outer.area - r.holes.reduce((h, l) => h + l.area, 0), 0)

/** Is the point inside the region (inside its outer loop and outside every hole)? */
export const regionContains = (r: Region, p: Pt) => pointInPoly(p, r.outer.poly) && !r.holes.some((h) => pointInPoly(p, h.poly))

/** The innermost region containing the point, if any. */
export function regionAt(p: Profiles, pt: Pt): Region | undefined {
  let best: Region | undefined
  for (const r of p.regions) if (regionContains(r, pt) && (!best || r.outer.area < best.outer.area)) best = r
  return best
}

/** A point strictly inside the region (away from its boundary), used to refer to the region later. */
export function interiorPoint(r: Region): Pt {
  // try the centroid of the outer loop, then scan for a point clear of the boundary
  const poly = r.outer.poly
  const b = poly.reduce((a, q) => ({ x0: Math.min(a.x0, q[0]), y0: Math.min(a.y0, q[1]), x1: Math.max(a.x1, q[0]), y1: Math.max(a.y1, q[1]) }), { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity })
  let best: Pt = poly[0], bestD = -1
  const n = 24
  for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) {
    const q: Pt = [b.x0 + ((b.x1 - b.x0) * (i + 0.5)) / (n + 1), b.y0 + ((b.y1 - b.y0) * (j + 0.5)) / (n + 1)]
    if (!regionContains(r, q)) continue
    let d = Infinity
    for (const loop of [r.outer, ...r.holes]) for (let k = 0; k < loop.poly.length; k++) d = Math.min(d, segDist(q, loop.poly[k], loop.poly[(k + 1) % loop.poly.length]))
    if (d > bestD) { bestD = d; best = q }
  }
  return best
}

function segDist(p: Pt, a: Pt, b: Pt): number {
  const abx = b[0] - a[0], aby = b[1] - a[1], l2 = abx * abx + aby * aby
  const t = l2 < 1e-18 ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / l2))
  return Math.hypot(p[0] - (a[0] + t * abx), p[1] - (a[1] + t * aby))
}
