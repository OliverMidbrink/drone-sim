import type { SketchConstraint, SketchEntity } from '@darkcad/core'
import {
  add, angleOf, arcEnd, arcStart, buildGeo, dist, dot, cross, intersect, leaveDir, len, nearest, nearestInfinite, norm, onExtent, polar, rot90, scale, sub, TAU, wrap,
  type ArcC, type Curve, type Geo, type LineC, type Pt, arcOf,
} from './geom'
import { Draft, refsOf, type SketchModel } from './model'
import { nodeIndex } from './profiles'

export class OpError extends Error {}
export interface OpResult { model: SketchModel; /** Entities created by the operation (for selecting the result). */ created: string[] }

const finish = (d: Draft): OpResult => {
  const ids = new Set(d.entities.map((e) => e.id))
  return { model: d.model(), created: [...d.created].filter((id) => ids.has(id)) }
}
const fail = (msg: string): never => { throw new OpError(msg) }

// ------------------------------------------------------------------------------------------------ helpers
const rotateAbout = (p: Pt, c: Pt, a: number): Pt => { const v = sub(p, c), co = Math.cos(a), si = Math.sin(a); return [c[0] + v[0] * co - v[1] * si, c[1] + v[0] * si + v[1] * co] }
const reflect = (p: Pt, a: Pt, b: Pt): Pt => {
  const u = norm(sub(b, a)), v = sub(p, a), t = dot(v, u)
  const foot: Pt = [a[0] + u[0] * t, a[1] + u[1] * t]
  return [2 * foot[0] - p[0], 2 * foot[1] - p[1]]
}

/** The selected entities plus the points they are built on. */
function closure(d: Draft, ids: Iterable<string>) {
  const points = new Set<string>(), curves: string[] = []
  for (const id of ids) {
    const e = d.get(id)
    if (!e) continue
    if (e.kind === 'point') points.add(id)
    else { curves.push(id); for (const r of refsOf(e)) points.add(r) }
  }
  return { points, curves }
}

const setPoint = (d: Draft, id: string, p: Pt) => { const e = d.get(id); if (e?.kind === 'point') { e.x = p[0]; e.y = p[1] } }
const isPointUsed = (d: Draft, id: string, except?: string) => d.entities.some((e) => e.id !== except && e.id !== id && refsOf(e).includes(id)) || d.constraints.some((c) => c.refs.includes(id))
/** Remove points nothing refers to any more. */
function prune(d: Draft, ids: string[]) { for (const id of ids) if (d.get(id)?.kind === 'point' && !isPointUsed(d, id)) d.remove([id]) }

const LENGTH_TYPES = new Set(['distance', 'equal', 'hdist', 'vdist', 'midpoint'])
/** Editing the length of a curve invalidates constraints that depend on it. */
function dropLengthConstraints(d: Draft, ids: string[]) { d.constraints = d.constraints.filter((c) => !(LENGTH_TYPES.has(c.type) && c.refs.some((r) => ids.includes(r)))) }

// ------------------------------------------------------------------------------------------------ simple edits
export function deleteEntities(model: SketchModel, ids: string[]): OpResult { const d = new Draft(model); d.remove(ids); return finish(d) }

export function setConstruction(model: SketchModel, ids: string[], on?: boolean): OpResult {
  const d = new Draft(model)
  const targets = d.entities.filter((e) => ids.includes(e.id))
  const value = on ?? !targets.every((e) => e.construction)
  for (const e of targets) { if (value) e.construction = true; else delete e.construction }
  return finish(d)
}

/** Move the selection by (dx, dy). Constraints that no longer hold are dropped by the caller's relaxed solve. */
export function moveEntities(model: SketchModel, ids: string[], dx: number, dy: number): OpResult {
  return transformInPlace(model, ids, (p) => [p[0] + dx, p[1] + dy], 1)
}
export function rotateEntities(model: SketchModel, ids: string[], center: Pt, angle: number): OpResult {
  const r = transformInPlace(model, ids, (p) => rotateAbout(p, center, angle), 1)
  // Horizontal / vertical constraints survive only quarter-turns (swapping), otherwise they are released.
  const { points, curves } = closure(new Draft(model), ids)
  const moved = new Set([...points, ...curves])
  const quarter = angle / (Math.PI / 2)
  const isQuarter = Math.abs(quarter - Math.round(quarter)) < 1e-9
  const odd = isQuarter && Math.abs(Math.round(quarter)) % 2 === 1
  const swap: Record<string, SketchConstraint['type']> = { horizontal: 'vertical', vertical: 'horizontal' }
  const constraints: SketchConstraint[] = []
  const identity = Math.abs(angle / TAU - Math.round(angle / TAU)) < 1e-9
  for (const c of r.model.constraints) {
    if ((c.type === 'hdist' || c.type === 'vdist') && !identity && c.refs.every((x) => moved.has(x) || x === 'origin' || x.startsWith('axis'))) continue
    if (swap[c.type] && c.refs.length && c.refs.every((x) => moved.has(x))) {
      if (!isQuarter) continue
      constraints.push(odd ? { ...c, type: swap[c.type] } : c)
    } else constraints.push(c)
  }
  return { ...r, model: { ...r.model, constraints } }
}
export function scaleEntities(model: SketchModel, ids: string[], base: Pt, factor: number): OpResult {
  if (!(factor > 0)) fail('scale factor must be positive')
  return transformInPlace(model, ids, (p) => [base[0] + (p[0] - base[0]) * factor, base[1] + (p[1] - base[1]) * factor], factor)
}

function transformInPlace(model: SketchModel, ids: string[], xf: (p: Pt) => Pt, k: number): OpResult {
  const d = new Draft(model)
  const { points, curves } = closure(d, ids)
  if (!points.size && !curves.length) fail('nothing selected')
  for (const id of points) setPoint(d, id, xf(d.pt(id)))
  for (const id of curves) { const e = d.get(id); if (e?.kind === 'circle') e.r *= k }
  const moved = new Set([...points, ...curves])
  d.constraints = d.constraints.map((c) => {
    if (c.type === 'fix' && points.has(c.refs[0])) return { ...c, at: xf(c.at ?? d.pt(c.refs[0])) }
    if (k !== 1 && c.value !== undefined && ['distance', 'hdist', 'vdist', 'radius', 'diameter'].includes(c.type) && c.refs.every((r) => moved.has(r))) return { ...c, value: c.value * k }
    return c
  })
  return finish(d)
}

// ------------------------------------------------------------------------------------------------ copy / mirror / patterns
interface CopyOpts { flip?: boolean; rscale?: number; constraints?: boolean; reuse?: (pointId: string) => string | undefined }

/** Duplicate entities (and the points they are built on) through a point transform. Returns old id → new id. */
export function copyInto(d: Draft, ids: string[], xf: (p: Pt) => Pt, o: CopyOpts = {}): Map<string, string> {
  const { points, curves } = closure(d, ids)
  const map = new Map<string, string>()
  for (const id of points) {
    const re = o.reuse?.(id)
    const e = d.get(id)
    map.set(id, re ?? d.point(...xf(d.pt(id)), e?.kind === 'point' && !!e.construction))
  }
  for (const id of curves) {
    const e = d.get(id)!
    const m = (r: string) => map.get(r)!
    if (e.kind === 'line') map.set(id, d.line(m(e.a), m(e.b), !!e.construction))
    else if (e.kind === 'arc') map.set(id, o.flip ? d.arc(m(e.c), m(e.b), m(e.a), !!e.construction) : d.arc(m(e.c), m(e.a), m(e.b), !!e.construction))
    else if (e.kind === 'circle') map.set(id, d.circle(m(e.c), e.r * (o.rscale ?? 1), !!e.construction))
  }
  if (o.constraints && !o.flip) {
    const k = o.rscale ?? 1
    for (const c of [...d.constraints]) {
      if (c.type === 'fix' || !c.refs.every((r) => map.has(r))) continue
      if (['hdist', 'vdist'].includes(c.type) && c.refs.some((r) => r === 'origin')) continue
      d.constrain(c.type, c.refs.map((r) => map.get(r)!), { ...(c.value !== undefined ? { value: ['distance', 'hdist', 'vdist', 'radius', 'diameter'].includes(c.type) ? c.value * k : c.value } : {}), ...(c.sign ? { sign: c.sign } : {}) })
    }
  }
  return map
}

export function copyEntities(model: SketchModel, ids: string[], dx: number, dy: number): OpResult {
  const d = new Draft(model)
  if (!closure(d, ids).points.size) fail('nothing selected')
  copyInto(d, ids, (p) => [p[0] + dx, p[1] + dy], { constraints: true })
  return finish(d)
}

/** Mirror about a line (a line entity, or 'axis-x' / 'axis-y'). The copies stay linked by symmetric constraints. */
export function mirrorEntities(model: SketchModel, ids: string[], axis: string): OpResult {
  const d = new Draft(model)
  const geo = buildGeo(d.entities)
  const ax = geo.byId.get(axis)
  if (!ax || ax.kind !== 'line') return fail('pick a line to mirror about')
  const sel = ids.filter((i) => i !== axis)
  const { points } = closure(d, sel)
  if (!points.size) fail('nothing selected')
  const mirror = (p: Pt) => reflect(p, ax.a, ax.b)
  const onAxis = (p: Pt) => Math.abs(cross(norm(sub(ax.b, ax.a)), sub(p, ax.a))) < 1e-6
  const onAxisPoint = new Set([...points].filter((id) => onAxis(d.pt(id)) && id !== ax.ia && id !== ax.ib))
  for (const p of [ax.ia, ax.ib]) if (points.has(p)) onAxisPoint.add(p)
  const map = copyInto(d, sel, mirror, { flip: true, reuse: (id) => (onAxisPoint.has(id) ? id : undefined) })
  for (const id of points) if (!onAxisPoint.has(id)) d.constrain('symmetric', [id, map.get(id)!, axis])
  for (const id of sel) { const e = d.get(id); if (e?.kind === 'circle' || e?.kind === 'arc') d.constrain('equal', [id, map.get(id)!]) }
  return finish(d)
}

export function rectPattern(model: SketchModel, ids: string[], nx: number, ny: number, dx: number, dy: number): OpResult {
  if (nx < 1 || ny < 1 || nx * ny < 2) fail('a pattern needs at least two instances')
  if (nx * ny > 400) fail('pattern is too large (max 400 instances)')
  const d = new Draft(model)
  if (!closure(d, ids).points.size) fail('nothing selected')
  const base = ids.filter((id) => d.get(id))
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
    if (!i && !j) continue
    copyInto(d, base, (p) => [p[0] + i * dx, p[1] + j * dy], { constraints: true })
  }
  return finish(d)
}

/** `count` instances spread over `total` radians about `center` (a full turn spaces them evenly without repeating the first). */
export function circularPattern(model: SketchModel, ids: string[], center: Pt, count: number, total = TAU): OpResult {
  if (count < 2) fail('a pattern needs at least two instances')
  if (count > 400) fail('pattern is too large (max 400 instances)')
  const d = new Draft(model)
  if (!closure(d, ids).points.size) fail('nothing selected')
  const base = ids.filter((id) => d.get(id))
  const step = Math.abs(total - TAU) < 1e-9 ? total / count : total / (count - 1)
  for (let k = 1; k < count; k++) copyInto(d, base, (p) => rotateAbout(p, center, step * k), { constraints: true })
  return finish(d)
}

// ------------------------------------------------------------------------------------------------ offset
export interface Chain { els: { id: string; fwd: boolean }[]; closed: boolean }

/** Connected curves through degree-2 joints, starting from one curve. A circle is a chain of its own. */
export function chainOf(model: SketchModel, id: string): Chain {
  const d = new Draft(model)
  const geo = buildGeo(d.entities)
  const start = geo.byId.get(id)
  if (!start || start.virtual) return fail('pick a curve')
  if (start.kind === 'circle') return { els: [{ id, fwd: true }], closed: true }
  const cs = geo.real.filter((c): c is LineC | ArcC => c.kind !== 'circle' && c.construction === start.construction)
  const ptIds = [...new Set(cs.flatMap((c) => [c.ia, c.ib]))]
  const node = nodeIndex(ptIds, (p) => geo.point(p), d.constraints)
  const ends = (c: LineC | ArcC) => [node.get(c.ia)!, node.get(c.ib)!] as const
  const at = new Map<number, { c: LineC | ArcC; end: 0 | 1 }[]>()
  for (const c of cs) { const [a, b] = ends(c); at.set(a, [...(at.get(a) ?? []), { c, end: 0 }]); at.set(b, [...(at.get(b) ?? []), { c, end: 1 }]) }
  const walk = (c0: LineC | ArcC, fromEnd: 0 | 1) => {
    // fromEnd: which end of c0 we leave through
    const out: { id: string; fwd: boolean }[] = []
    let cur = c0, exit: 0 | 1 = fromEnd, closed = false
    for (let guard = 0; guard < cs.length + 2; guard++) {
      const n = ends(cur)[exit]
      const here = at.get(n) ?? []
      if (here.length !== 2) break
      const other = here.find((h) => !(h.c === cur && h.end === exit))!
      if (other.c === c0 && other.end === (1 - fromEnd)) { closed = true; break }
      if (other.c === c0) break
      out.push({ id: other.c.id, fwd: other.end === 0 })
      cur = other.c; exit = (1 - other.end) as 0 | 1
    }
    return { out, closed }
  }
  const f = walk(start, 1)
  if (f.closed) return { els: [{ id, fwd: true }, ...f.out], closed: true }
  const b = walk(start, 0)
  const back = b.out.map((e) => ({ id: e.id, fwd: !e.fwd })).reverse()
  return { els: [...back, { id, fwd: true }, ...f.out], closed: false }
}

interface El { c: Curve; fwd: boolean; s: Pt; e: Pt }
const travel = (c: Curve, fwd: boolean): [Pt, Pt] => c.kind === 'line' ? (fwd ? [c.a, c.b] : [c.b, c.a]) : c.kind === 'arc' ? (fwd ? [arcStart(c), arcEnd(c)] : [arcEnd(c), arcStart(c)]) : [polar(c.c, c.r, 0), polar(c.c, c.r, 0)]

/** Direction of travel at a point of the element. */
function tangentAt(el: El, p: Pt): Pt {
  const c = el.c
  if (c.kind === 'line') return norm(sub(el.e, el.s))
  const t = rot90(norm(sub(p, c.c)))
  return el.fwd ? t : [-t[0], -t[1]]
}
const leftNormal = (el: El, p: Pt): Pt => { const t = tangentAt(el, p); return [-t[1], t[0]] }

/** Signed distance from `p` to the chain: positive on the left of the direction of travel. */
export function chainSide(model: SketchModel, chain: Chain, p: Pt): number {
  const geo = buildGeo(model.entities)
  let best: { d: number; q: Pt; el: El } | undefined
  for (const ce of chain.els) {
    const c = geo.byId.get(ce.id)!
    const [s, e] = travel(c, ce.fwd)
    const el: El = { c, fwd: ce.fwd, s, e }
    const n = nearest(c, p)
    if (!best || n.d < best.d) best = { d: n.d, q: n.pt, el }
  }
  if (!best) return 0
  const t = tangentAt(best.el, best.q)
  const side = cross(t, sub(p, best.q))
  return side >= 0 ? best.d : -best.d
}

function offsetCurve(el: El, d: number): Curve {
  const c = el.c
  if (c.kind === 'line') { const n = leftNormal(el, c.a); return { ...c, a: add(c.a, scale(n, d)), b: add(c.b, scale(n, d)) } }
  const r = c.kind === 'circle' || el.fwd ? c.r - d : c.r + d // counter-clockwise travel keeps the left side towards the centre
  if (r <= 1e-9) return fail('offset distance is larger than a radius in the chain')
  return { ...c, r }
}

/** Offset a chain of connected curves by a signed distance (positive = left of the direction of travel). */
export function offsetChain(model: SketchModel, chain: Chain, dist_: number): OpResult {
  if (Math.abs(dist_) < 1e-9) fail('offset distance is zero')
  const d = new Draft(model)
  const geo = buildGeo(d.entities)
  if (chain.els.length === 1 && geo.byId.get(chain.els[0].id)?.kind === 'circle') {
    const c = geo.byId.get(chain.els[0].id) as Extract<Curve, { kind: 'circle' }>
    const r = c.r - dist_ // travelling counter-clockwise, left is inwards
    if (r <= 1e-9) fail('offset distance is larger than the radius')
    d.circle(d.point(c.c[0], c.c[1]), r, c.construction)
    return finish(d)
  }
  const els: El[] = chain.els.map((ce) => { const c = geo.byId.get(ce.id)!; const [s, e] = travel(c, ce.fwd); return { c, fwd: ce.fwd, s, e } })
  const offs = els.map((el) => offsetCurve(el, dist_))
  const n = els.length
  const joint = (i: number): Pt => {
    // joint between element i-1 and element i (closed chains wrap)
    if (i === 0 && !chain.closed) return add(els[0].s, scale(leftNormal(els[0], els[0].s), dist_))
    if (i === n && !chain.closed) return add(els[n - 1].e, scale(leftNormal(els[n - 1], els[n - 1].e), dist_))
    const a = (i - 1 + n) % n, b = i % n
    const P = els[a].e
    const smooth = cross(tangentAt(els[a], P), tangentAt(els[b], P)) ** 2 < 1e-10 && dot(tangentAt(els[a], P), tangentAt(els[b], P)) > 0
    if (!smooth) {
      const cands = intersect(offs[a], offs[b], true)
      if (cands.length) return cands.reduce((m, q) => (dist(q, P) < dist(m, P) ? q : m))
    }
    return add(P, scale(leftNormal(els[a], P), dist_))
  }
  const count = chain.closed ? n : n + 1
  const V = Array.from({ length: count }, (_, i) => d.point(...joint(i)))
  const centers = new Map<string, string>()
  els.forEach((el, i) => {
    const from = V[i], to = V[(i + 1) % count]
    if (el.c.kind === 'line') d.line(from, to, el.c.construction)
    else if (el.c.kind === 'arc') {
      const ic = (el.c as ArcC).ic
      if (!centers.has(ic)) centers.set(ic, d.point(el.c.c[0], el.c.c[1]))
      if (el.fwd) d.arc(centers.get(ic)!, from, to, el.c.construction); else d.arc(centers.get(ic)!, to, from, el.c.construction)
    }
  })
  return finish(d)
}

// ------------------------------------------------------------------------------------------------ trim / extend
const unitParam = (c: Curve, q: Pt) => nearest(c, q).t

/** Existing point at a location, or a new one held on `other` by a point-on constraint. */
function pointAt(d: Draft, pos: Pt, other: string | undefined, skip: string[]): string {
  const hit = d.entities.find((e) => e.kind === 'point' && !skip.includes(e.id) && dist([e.x, e.y], pos) < 1e-6)
  if (hit) return hit.id
  const id = d.point(pos[0], pos[1])
  if (other) d.constrain('point-on', [id, other])
  return id
}

function setEnd(e: SketchEntity, end: 0 | 1, id: string) {
  if (e.kind === 'line') { if (end === 0) e.a = id; else e.b = id }
  else if (e.kind === 'arc') { if (end === 0) e.a = id; else e.b = id }
}
const getEnd = (e: SketchEntity, end: 0 | 1) => (e.kind === 'line' || e.kind === 'arc' ? (end === 0 ? e.a : e.b) : '')

/** Remove the part of a curve under the cursor, between the nearest intersections with other curves. */
export function trim(model: SketchModel, curveId: string, at: Pt): OpResult {
  const d = new Draft(model)
  const geo = buildGeo(d.entities)
  const cur = geo.byId.get(curveId)
  if (!cur || cur.virtual) return fail('pick a curve to trim')
  const e = d.get(curveId)!
  const cuts: { t: number; pt: Pt; other: string }[] = []
  for (const o of geo.real) {
    if (o.id === curveId) continue
    for (const q of intersect(cur, o)) {
      const t = cur.kind === 'circle' ? wrap(angleOf(sub(q, cur.c))) : unitParam(cur, q)
      if (!cuts.some((k) => dist(k.pt, q) < 1e-6)) cuts.push({ t, pt: q, other: o.id })
    }
  }
  cuts.sort((a, b) => a.t - b.t)

  if (cur.kind === 'circle') {
    if (cuts.length < 2) { d.remove([curveId]); return finish(d) }
    const tp = wrap(angleOf(sub(at, cur.c)))
    let hiI = cuts.findIndex((k) => k.t > tp)
    if (hiI < 0) hiI = 0
    const loI = (hiI - 1 + cuts.length) % cuts.length
    const keepStart = pointAt(d, cuts[hiI].pt, cuts[hiI].other, []), keepEnd = pointAt(d, cuts[loI].pt, cuts[loI].other, [])
    const centerId = (e as Extract<SketchEntity, { kind: 'circle' }>).c
    const arc = d.arc(centerId, keepStart, keepEnd, !!e.construction)
    d.constraints = d.constraints.map((c) => ({ ...c, refs: c.refs.map((r) => (r === curveId ? arc : r)) }))
    d.entities = d.entities.filter((x) => x.id !== curveId)
    return finish(d)
  }

  const inner = cuts.filter((k) => k.t > 1e-7 && k.t < 1 - 1e-7)
  const tp = unitParam(cur, at)
  const lo = [...inner].reverse().find((k) => k.t <= tp)
  const hi = inner.find((k) => k.t >= tp)
  if (!lo && !hi) {
    // A line that only touches others at its own ends cannot be cut up: remove it whole.
    const ps = refsOf(e)
    d.remove([curveId]); prune(d, ps)
    return finish(d)
  }
  dropLengthConstraints(d, [curveId])
  const own = refsOf(e)
  const oldA = getEnd(e, 0), oldB = getEnd(e, 1)
  const cp = (k: { pt: Pt; other: string }) => pointAt(d, k.pt, k.other, own)
  if (!lo) { setEnd(e, 0, cp(hi!)); prune(d, [oldA]) }
  else if (!hi) { setEnd(e, 1, cp(lo)); prune(d, [oldB]) }
  else {
    const loP = cp(lo), hiP = cp(hi)
    if (e.kind === 'line') {
      setEnd(e, 1, loP)
      const nl = d.line(hiP, oldB, !!e.construction)
      for (const c of d.constraints.filter((c) => (c.type === 'horizontal' || c.type === 'vertical') && c.refs.length === 1 && c.refs[0] === curveId)) d.constrain(c.type, [nl])
    } else if (e.kind === 'arc') {
      const center = e.c
      setEnd(e, 1, loP)
      const na = d.arc(center, hiP, oldB, !!e.construction)
      d.constrain('equal', [curveId, na])
    }
  }
  return finish(d)
}

/** Lengthen the end of a line or arc nearest the click up to the next curve in its path. */
export function extend(model: SketchModel, curveId: string, at: Pt): OpResult {
  const d = new Draft(model)
  const geo = buildGeo(d.entities)
  const cur = geo.byId.get(curveId)
  if (!cur || cur.virtual || cur.kind === 'circle') return fail('pick a line or arc to extend')
  const e = d.get(curveId)!
  const end: 0 | 1 = unitParam(cur, at) < 0.5 ? 0 : 1
  const endP = end === 0 ? travel(cur, false)[1] : travel(cur, true)[1]
  const outward = (q: Pt): number | undefined => {
    if (cur.kind === 'line') { const u = scale(leaveDir(cur, end), -1); const s = dot(sub(q, endP), u); return s > 1e-7 ? s : undefined }
    const a = (cur as ArcC)
    const ang = angleOf(sub(q, a.c))
    const delta = end === 1 ? wrap(ang - a.a1) : wrap(a.a0 - ang)
    return delta > 1e-7 && delta < TAU - 1e-7 ? delta : undefined
  }
  let best: { s: number; pt: Pt; other: string } | undefined
  for (const o of geo.curves) {
    if (o.id === curveId) continue
    for (const q of intersect(cur, o, true)) {
      if (!onExtent(o, q)) continue
      const s = outward(q)
      if (s !== undefined && (!best || s < best.s)) best = { s, pt: q, other: o.id }
    }
  }
  if (!best) return fail('nothing to extend to')
  dropLengthConstraints(d, [curveId])
  const old = getEnd(e, end)
  const target = best.other
  if (!isPointUsed(d, old, curveId) && d.constraints.every((c) => !(c.refs.includes(old)))) {
    setPoint(d, old, best.pt)
    d.constrain('point-on', [old, target])
  } else {
    const np = pointAt(d, best.pt, target, [])
    setEnd(e, end, np)
    prune(d, [old])
  }
  return finish(d)
}

// ------------------------------------------------------------------------------------------------ fillet / chamfer
interface Corner { a: LineC | ArcC; b: LineC | ArcC; endA: 0 | 1; endB: 0 | 1; p: Pt }

function findCorner(d: Draft, geo: Geo, idA: string, idB: string): Corner {
  const a = geo.byId.get(idA), b = geo.byId.get(idB)
  if (!a || !b || a.kind === 'circle' || b.kind === 'circle' || a.virtual || b.virtual || idA === idB) return fail('pick two lines or arcs that meet at a corner')
  const ea = [travel(a, false)[1], travel(a, true)[1]], eb = [travel(b, false)[1], travel(b, true)[1]]
  const coin = d.constraints.filter((c) => c.type === 'coincident')
  for (const i of [0, 1] as const) for (const j of [0, 1] as const) {
    const ia = i === 0 ? a.ia : a.ib, ib = j === 0 ? b.ia : b.ib
    if (dist(ea[i], eb[j]) < 1e-6 || ia === ib || coin.some((c) => (c.refs[0] === ia && c.refs[1] === ib) || (c.refs[0] === ib && c.refs[1] === ia))) return { a, b, endA: i, endB: j, p: ea[i] }
  }
  return fail('the two curves do not meet at a corner')
}

/** The two curves meeting at a point (for picking a corner by its vertex). */
export function curvesAtPoint(model: SketchModel, pointId: string): string[] {
  const d = new Draft(model)
  const geo = buildGeo(d.entities)
  const pos = geo.point(pointId)
  const coin = d.constraints.filter((c) => c.type === 'coincident')
  const same = (p: string) => p === pointId || dist(geo.point(p), pos) < 1e-6 || coin.some((c) => c.refs.includes(p) && c.refs.includes(pointId))
  return geo.real.filter((c) => c.kind !== 'circle' && !c.construction && (same((c as LineC).ia) || same((c as LineC).ib))).map((c) => c.id)
}

const dirLeaving = (c: LineC | ArcC, end: 0 | 1) => scale(leaveDir(c, end), 1)

function offsetFor(c: LineC | ArcC, side: 1 | -1, r: number): Curve | undefined {
  if (c.kind === 'line') { const t = norm(sub(c.b, c.a)), n: Pt = [-t[1], t[0]]; return { ...c, a: add(c.a, scale(n, side * r)), b: add(c.b, scale(n, side * r)) } }
  const rr = c.r + side * r
  return rr > 1e-9 ? ({ ...c, r: rr } as ArcC) : undefined
}

/** Round the corner where two curves meet with a tangent arc. `near` picks which of the possible arcs (cursor position). */
export function fillet(model: SketchModel, idA: string, idB: string, radius: number, near: Pt): OpResult {
  if (!(radius > 0)) fail('fillet radius must be positive')
  const d = new Draft(model)
  const geo = buildGeo(d.entities)
  const k = findCorner(d, geo, idA, idB)
  const cands: { c: Pt; t1: Pt; t2: Pt }[] = []
  for (const sa of [1, -1] as const) for (const sb of [1, -1] as const) {
    const oa = offsetFor(k.a, sa, radius), ob = offsetFor(k.b, sb, radius)
    if (!oa || !ob) continue
    for (const c of intersect(oa, ob, true)) {
      const t1 = nearestInfinite(k.a, c).pt, t2 = nearestInfinite(k.b, c).pt
      const ok = (cu: LineC | ArcC, t: Pt, end: 0 | 1) => onExtent(cu, t) && dist(t, end === 0 ? travel(cu, false)[1] : travel(cu, true)[1]) > 1e-6 && Math.abs(dist(t, c) - radius) < 1e-6 * Math.max(1, radius)
      // the tangent points must lie on the curves, not beyond their far ends
      if (ok(k.a, t1, k.endA) && ok(k.b, t2, k.endB) && dist(t1, t2) > 1e-9) cands.push({ c, t1, t2 })
    }
  }
  if (!cands.length) return fail('the fillet radius does not fit these curves')
  const best = cands.reduce((m, q) => (dist(q.c, near) < dist(m.c, near) ? q : m))
  const ea = d.get(k.a.id)!, eb = d.get(k.b.id)!
  const oldA = getEnd(ea, k.endA), oldB = getEnd(eb, k.endB)
  dropLengthConstraints(d, [k.a.id, k.b.id])
  const t1 = d.point(best.t1[0], best.t1[1]), t2 = d.point(best.t2[0], best.t2[1]), c = d.point(best.c[0], best.c[1])
  setEnd(ea, k.endA, t1); setEnd(eb, k.endB, t2)
  const ccw = cross(sub(best.t1, best.c), sub(best.t2, best.c)) > 0
  const arc = ccw ? d.arc(c, t1, t2) : d.arc(c, t2, t1)
  d.constrain('tangent', [k.a.id, arc]); d.constrain('tangent', [k.b.id, arc])
  prune(d, [oldA, oldB])
  return finish(d)
}

/** Fillet the corner at a vertex shared by exactly two curves. */
export function filletCorner(model: SketchModel, pointId: string, radius: number): OpResult {
  const ids = curvesAtPoint(model, pointId)
  if (ids.length !== 2) return fail('a corner needs exactly two curves meeting at the point')
  const d = new Draft(model)
  const geo = buildGeo(d.entities)
  const k = findCorner(d, geo, ids[0], ids[1])
  const u = norm(add(dirLeaving(k.a, k.endA), dirLeaving(k.b, k.endB)))
  return fillet(model, ids[0], ids[1], radius, add(k.p, scale(u, radius * 2)))
}

/** Cut the corner between two lines with a straight edge, `d1` along the first and `d2` along the second. */
export function chamfer(model: SketchModel, idA: string, idB: string, d1: number, d2 = d1): OpResult {
  if (!(d1 > 0 && d2 > 0)) fail('chamfer distance must be positive')
  const d = new Draft(model)
  const geo = buildGeo(d.entities)
  const k = findCorner(d, geo, idA, idB)
  if (k.a.kind !== 'line' || k.b.kind !== 'line') return fail('chamfer works on two straight lines')
  const la = len(sub(k.a.b, k.a.a)), lb = len(sub(k.b.b, k.b.a))
  if (d1 >= la - 1e-9 || d2 >= lb - 1e-9) return fail('chamfer distance is longer than the line')
  const t1 = add(k.p, scale(dirLeaving(k.a, k.endA), d1)), t2 = add(k.p, scale(dirLeaving(k.b, k.endB), d2))
  const ea = d.get(k.a.id)!, eb = d.get(k.b.id)!
  const oldA = getEnd(ea, k.endA), oldB = getEnd(eb, k.endB)
  dropLengthConstraints(d, [k.a.id, k.b.id])
  const p1 = d.point(t1[0], t1[1]), p2 = d.point(t2[0], t2[1])
  setEnd(ea, k.endA, p1); setEnd(eb, k.endB, p2)
  d.line(p1, p2)
  prune(d, [oldA, oldB])
  return finish(d)
}

export function chamferCorner(model: SketchModel, pointId: string, d1: number, d2 = d1): OpResult {
  const ids = curvesAtPoint(model, pointId)
  if (ids.length !== 2) return fail('a corner needs exactly two curves meeting at the point')
  return chamfer(model, ids[0], ids[1], d1, d2)
}
