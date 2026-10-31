import type { SketchConstraint, SketchEntity } from '@darkcad/core'
import { wrapPi } from './geom'

export interface Pin { id: string; x: number; y: number }
export interface SolveOptions {
  /** Soft targets for points (dragging). Constraints always win over pins. */
  pins?: Pin[]
  maxIter?: number
  /** Skip the (more expensive) degrees-of-freedom analysis, e.g. while dragging. */
  quick?: boolean
}

export interface SolveResult {
  /** Entities with solved coordinates; the input entities unchanged when `ok` is false. */
  entities: SketchEntity[]
  ok: boolean
  /** Remaining degrees of freedom (0 = fully constrained). -1 when not analysed. */
  dof: number
  /** Constraints that cannot be satisfied together with the ones before them. */
  conflicts: string[]
  /** Constraints that add nothing (already implied by others). */
  redundant: string[]
  /** Entities (points, curves) that can still move. Only meaningful when dof >= 0. */
  free: Set<string>
  /** Largest remaining constraint error. */
  residual: number
}

const TOL = 1e-9
const OK = 1e-6

interface Row { vars: number[]; f: (v: Float64Array) => number[]; n: number; id: string; internal?: boolean }

/** Ends of a line entity, including the virtual plane axes. */
function lineEnds(id: string, ents: Map<string, SketchEntity>): [string, string] | undefined {
  if (id === 'axis-x') return ['origin', '$xe']
  if (id === 'axis-y') return ['origin', '$ye']
  const e = ents.get(id)
  return e?.kind === 'line' ? [e.a, e.b] : undefined
}

export function solve(entities: SketchEntity[], constraints: SketchConstraint[], opts: SolveOptions = {}): SolveResult {
  const sys = new System(entities, constraints)
  const rows: Row[] = []
  for (const e of entities) if (e.kind === 'arc') rows.push(sys.arcRadius(e))
  const internalCount = rows.length
  const used: SketchConstraint[] = []
  for (const c of constraints) { const r = sys.compile(c); if (r) { rows.push(r); used.push(c) } }

  const start = sys.state.slice()
  const pinRows = opts.pins?.length ? sys.pinRows(opts.pins) : []

  const attempt = (rs: Row[], from: Float64Array, withPins: boolean) => {
    const v = from.slice()
    if (withPins && pinRows.length) { lm(v, [...rs, ...pinRows], opts.maxIter ?? 60); }
    const err = lm(v, rs, opts.maxIter ?? 100)
    return { v, err }
  }

  let { v, err } = attempt(rows, start, true)
  let conflicts: string[] = []
  if (err > OK) {
    // Find the first constraint that cannot be added to the ones before it.
    for (let k = internalCount; k < rows.length; k++) {
      const sub = rows.slice(0, k + 1)
      if (lm(start.slice(), sub, 200) > OK) { conflicts = [rows[k].id]; break }
    }
    return { entities, ok: false, dof: -1, conflicts, redundant: [], free: new Set(), residual: err }
  }

  const out = sys.write(v)
  if (opts.quick) return { entities: out, ok: true, dof: -1, conflicts: [], redundant: [], free: new Set(), residual: err }
  const a = analyse(v, rows, internalCount, sys)
  return { entities: out, ok: true, dof: a.dof, conflicts: [], redundant: a.redundant, free: a.free, residual: err }
}

/** Solve from the stored geometry and report degrees of freedom, without needing to move anything. */
export const analyze = (entities: SketchEntity[], constraints: SketchConstraint[]) => solve(entities, constraints)

class System {
  state: Float64Array
  /** point id → index of x (y follows); circle id → index of r. */
  idx = new Map<string, number>()
  ents = new Map<string, SketchEntity>()
  constX = new Map<string, [number, number]>([['origin', [0, 0]], ['$xe', [1, 0]], ['$ye', [0, 1]]])
  order: string[] = []

  /** Point id → representative of its coincident group (points joined by a constraint count as one). */
  private rep = new Map<string, string>()
  private find(a: string): string { let r = a; while (this.rep.has(r) && this.rep.get(r) !== r) r = this.rep.get(r)!; return r }

  constructor(private entities: SketchEntity[], constraints: SketchConstraint[] = []) {
    for (const c of constraints) if (c.type === 'coincident' && c.refs.length === 2) this.rep.set(this.find(c.refs[0]), this.find(c.refs[1]))
    const s: number[] = []
    for (const e of entities) {
      this.ents.set(e.id, e)
      if (e.kind === 'point') { this.idx.set(e.id, s.length); s.push(e.x, e.y) }
      else if (e.kind === 'circle') { this.idx.set(e.id, s.length); s.push(e.r) }
    }
    this.state = Float64Array.from(s)
  }

  write(v: Float64Array): SketchEntity[] {
    return this.entities.map((e) => {
      const i = this.idx.get(e.id)
      if (i === undefined) return e
      if (e.kind === 'point') return { ...e, x: v[i], y: v[i + 1] }
      if (e.kind === 'circle') return { ...e, r: Math.abs(v[i]) }
      return e
    })
  }

  /** End points of a line or arc entity. */
  private ends(id: string): string[] {
    const e = this.ents.get(id)
    return e?.kind === 'line' ? [e.a, e.b] : e?.kind === 'arc' ? [e.a, e.b] : []
  }
  /** A pair of end points (one from each curve) that are the same point, if the two curves touch at an end. */
  shared(a: string, b: string): [string, string] | undefined {
    for (const p of this.ends(a)) for (const q of this.ends(b)) if (p === q || this.find(p) === this.find(q)) return [p, q]
  }

  has(id: string) { return this.constX.has(id) || this.ents.get(id)?.kind === 'point' }
  px(v: Float64Array, id: string) { const k = this.constX.get(id); return k ? k[0] : v[this.idx.get(id)!] }
  py(v: Float64Array, id: string) { const k = this.constX.get(id); return k ? k[1] : v[this.idx.get(id)! + 1] }

  varsOf(ids: string[]): number[] {
    const out = new Set<number>()
    const addPoint = (id: string) => { const i = this.idx.get(id); if (i !== undefined && this.ents.get(id)?.kind === 'point') { out.add(i); out.add(i + 1) } }
    for (const id of ids) {
      if (id === 'axis-x' || id === 'axis-y') continue
      const e = this.ents.get(id)
      if (!e) { addPoint(id); continue }
      if (e.kind === 'point') addPoint(id)
      else if (e.kind === 'line') { addPoint(e.a); addPoint(e.b) }
      else if (e.kind === 'arc') { addPoint(e.c); addPoint(e.a); addPoint(e.b) }
      else { addPoint(e.c); out.add(this.idx.get(e.id)!) }
    }
    return [...out]
  }

  radius(v: Float64Array, id: string): number {
    const e = this.ents.get(id)
    if (e?.kind === 'circle') return Math.abs(v[this.idx.get(id)!])
    if (e?.kind === 'arc') return Math.hypot(this.px(v, e.a) - this.px(v, e.c), this.py(v, e.a) - this.py(v, e.c))
    return 0
  }
  center(id: string): string | undefined { const e = this.ents.get(id); return e?.kind === 'circle' || e?.kind === 'arc' ? e.c : undefined }

  arcRadius(e: Extract<SketchEntity, { kind: 'arc' }>): Row {
    return { id: e.id, internal: true, n: 1, vars: this.varsOf([e.id]), f: (v) => [
      Math.hypot(this.px(v, e.b) - this.px(v, e.c), this.py(v, e.b) - this.py(v, e.c)) - Math.hypot(this.px(v, e.a) - this.px(v, e.c), this.py(v, e.a) - this.py(v, e.c))] }
  }

  pinRows(pins: Pin[]): Row[] {
    const W = 0.02
    return pins.filter((p) => this.ents.get(p.id)?.kind === 'point').map((p) => ({
      id: 'pin:' + p.id, n: 2, vars: this.varsOf([p.id]), internal: true,
      f: (v) => [W * (this.px(v, p.id) - p.x), W * (this.py(v, p.id) - p.y)],
    }))
  }

  /** Unit direction and start of a line. */
  private line(v: Float64Array, id: string) {
    const [a, b] = lineEnds(id, this.ents)!
    const ax = this.px(v, a), ay = this.py(v, a), dx = this.px(v, b) - ax, dy = this.py(v, b) - ay
    const l = Math.hypot(dx, dy) || 1e-12
    return { ax, ay, dx, dy, l, ux: dx / l, uy: dy / l }
  }

  compile(c: SketchConstraint): Row | undefined {
    const r = c.refs
    const kind = (id: string) => (id === 'axis-x' || id === 'axis-y' ? 'line' : this.ents.get(id)?.kind ?? (this.has(id) ? 'point' : undefined))
    const kinds = r.map(kind)
    if (kinds.some((k) => !k)) return
    const row = (n: number, f: (v: Float64Array) => number[]): Row => ({ id: c.id, n, vars: this.varsOf(r), f })
    const val = c.value ?? 0
    const isLine = (i: number) => kinds[i] === 'line'
    const isRound = (i: number) => kinds[i] === 'circle' || kinds[i] === 'arc'
    const isPoint = (i: number) => kinds[i] === 'point'
    const sgn = c.sign ?? 1
    switch (c.type) {
      case 'coincident':
        if (!(isPoint(0) && isPoint(1))) return
        return row(2, (v) => [this.px(v, r[0]) - this.px(v, r[1]), this.py(v, r[0]) - this.py(v, r[1])])
      case 'point-on': {
        if (!isPoint(0)) return
        if (isLine(1)) return row(1, (v) => { const L = this.line(v, r[1]); return [L.ux * (this.py(v, r[0]) - L.ay) - L.uy * (this.px(v, r[0]) - L.ax)] })
        if (isRound(1)) { const ce = this.center(r[1])!; return row(1, (v) => [Math.hypot(this.px(v, r[0]) - this.px(v, ce), this.py(v, r[0]) - this.py(v, ce)) - this.radius(v, r[1])]) }
        return
      }
      case 'horizontal': case 'vertical': {
        const hor = c.type === 'horizontal'
        if (r.length === 1 && isLine(0)) { const [a, b] = lineEnds(r[0], this.ents)!; return row(1, (v) => [hor ? this.py(v, a) - this.py(v, b) : this.px(v, a) - this.px(v, b)]) }
        if (r.length === 2 && isPoint(0) && isPoint(1)) return row(1, (v) => [hor ? this.py(v, r[0]) - this.py(v, r[1]) : this.px(v, r[0]) - this.px(v, r[1])])
        return
      }
      case 'parallel': case 'perpendicular': {
        if (!(isLine(0) && isLine(1))) return
        const par = c.type === 'parallel'
        return row(1, (v) => { const A = this.line(v, r[0]), B = this.line(v, r[1]); return [par ? A.ux * B.uy - A.uy * B.ux : A.ux * B.ux + A.uy * B.uy] })
      }
      case 'equal': {
        if (isLine(0) && isLine(1)) return row(1, (v) => [this.line(v, r[0]).l - this.line(v, r[1]).l])
        if (isRound(0) && isRound(1)) return row(1, (v) => [this.radius(v, r[0]) - this.radius(v, r[1])])
        return
      }
      case 'tangent': {
        const lineI = isLine(0) ? 0 : isLine(1) ? 1 : -1
        if (lineI >= 0) {
          const ci = 1 - lineI
          if (!isRound(ci)) return
          const ce = this.center(r[ci])!
          // Touching at an end point: the radius there is perpendicular to the line. (A distance condition would be degenerate:
          // a line through a point of the circle is never farther than r, so tangency is an extremum with no gradient.)
          const sh = this.shared(r[lineI], r[ci])
          if (sh) return row(1, (v) => { const L = this.line(v, r[lineI]); return [L.ux * (this.px(v, ce) - this.px(v, sh[0])) + L.uy * (this.py(v, ce) - this.py(v, sh[0]))] })
          return row(1, (v) => { const L = this.line(v, r[lineI]); return [Math.abs(L.ux * (this.py(v, ce) - L.ay) - L.uy * (this.px(v, ce) - L.ax)) - this.radius(v, r[ci])] })
        }
        if (!(isRound(0) && isRound(1))) return
        const c0 = this.center(r[0])!, c1 = this.center(r[1])!
        const sh2 = this.shared(r[0], r[1])
        if (sh2) return row(1, (v) => {
          const ax = this.px(v, c0) - this.px(v, sh2[0]), ay = this.py(v, c0) - this.py(v, sh2[0]), bx = this.px(v, c1) - this.px(v, sh2[1]), by = this.py(v, c1) - this.py(v, sh2[1])
          return [(ax * by - ay * bx) / ((Math.hypot(ax, ay) * Math.hypot(bx, by)) || 1e-12)]
        })
        return row(1, (v) => {
          const d = Math.hypot(this.px(v, c0) - this.px(v, c1), this.py(v, c0) - this.py(v, c1)), r0 = this.radius(v, r[0]), r1 = this.radius(v, r[1])
          return [d - (sgn >= 0 ? r0 + r1 : Math.abs(r0 - r1))]
        })
      }
      case 'concentric': {
        if (!(isRound(0) && isRound(1))) return
        const c0 = this.center(r[0])!, c1 = this.center(r[1])!
        return row(2, (v) => [this.px(v, c0) - this.px(v, c1), this.py(v, c0) - this.py(v, c1)])
      }
      case 'midpoint': {
        if (!(isPoint(0) && isLine(1))) return
        const [a, b] = lineEnds(r[1], this.ents)!
        return row(2, (v) => [this.px(v, r[0]) - (this.px(v, a) + this.px(v, b)) / 2, this.py(v, r[0]) - (this.py(v, a) + this.py(v, b)) / 2])
      }
      case 'symmetric': {
        if (!(isPoint(0) && isPoint(1) && isLine(2))) return
        return row(2, (v) => {
          const L = this.line(v, r[2])
          const mx = (this.px(v, r[0]) + this.px(v, r[1])) / 2, my = (this.py(v, r[0]) + this.py(v, r[1])) / 2
          return [L.ux * (my - L.ay) - L.uy * (mx - L.ax), L.ux * (this.px(v, r[1]) - this.px(v, r[0])) + L.uy * (this.py(v, r[1]) - this.py(v, r[0]))]
        })
      }
      case 'fix': {
        if (!isPoint(0) || this.constX.has(r[0])) return
        const i = this.idx.get(r[0])!
        const at = c.at ?? [this.state[i], this.state[i + 1]]
        return row(2, (v) => [this.px(v, r[0]) - at[0], this.py(v, r[0]) - at[1]])
      }
      case 'distance': {
        if (r.length === 1 && isLine(0)) return row(1, (v) => [this.line(v, r[0]).l - val])
        if (r.length === 2 && isPoint(0) && isPoint(1)) return row(1, (v) => [Math.hypot(this.px(v, r[0]) - this.px(v, r[1]), this.py(v, r[0]) - this.py(v, r[1])) - val])
        const pi = isPoint(0) ? 0 : isPoint(1) ? 1 : -1, li = 1 - pi
        if (r.length === 2 && pi >= 0 && isLine(li)) return row(1, (v) => { const L = this.line(v, r[li]); return [Math.abs(L.ux * (this.py(v, r[pi]) - L.ay) - L.uy * (this.px(v, r[pi]) - L.ax)) - val] })
        return
      }
      case 'hdist': case 'vdist': {
        if (!(isPoint(0) && isPoint(1))) return
        const h = c.type === 'hdist'
        return row(1, (v) => [sgn * (h ? this.px(v, r[1]) - this.px(v, r[0]) : this.py(v, r[1]) - this.py(v, r[0])) - val])
      }
      case 'radius': case 'diameter': {
        if (!isRound(0)) return
        const k = c.type === 'radius' ? 1 : 2
        return row(1, (v) => [k * this.radius(v, r[0]) - val])
      }
      case 'angle': {
        if (!(isLine(0) && isLine(1))) return
        return row(1, (v) => { const A = this.line(v, r[0]), B = this.line(v, r[1]); return [wrapPi(Math.atan2(A.ux * B.uy - A.uy * B.ux, A.ux * B.ux + A.uy * B.uy) - (sgn * val * Math.PI) / 180)] })
      }
    }
  }
}

function residuals(rows: Row[], v: Float64Array): Float64Array {
  const out: number[] = []
  for (const r of rows) out.push(...r.f(v))
  return Float64Array.from(out)
}

const maxAbs = (a: ArrayLike<number>) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m }

/** Sparse Jacobian by central differences, one block per row-group. Returns entries as [row, col, value]. */
function jacobian(rows: Row[], v: Float64Array) {
  const entries: [number, number, number][] = []
  let base = 0
  for (const r of rows) {
    for (const col of r.vars) {
      const x0 = v[col], h = 1e-6 * Math.max(1, Math.abs(x0))
      v[col] = x0 + h; const fp = r.f(v)
      v[col] = x0 - h; const fm = r.f(v)
      v[col] = x0
      for (let k = 0; k < r.n; k++) { const d = (fp[k] - fm[k]) / (2 * h); if (d !== 0) entries.push([base + k, col, d]) }
    }
    base += r.n
  }
  // Consumers group consecutive entries by row, and multi-equation rows are produced column by column.
  entries.sort((p, q) => p[0] - q[0])
  return { entries, m: base }
}

/** Levenberg-Marquardt on `v` in place. Returns the largest remaining residual among the non-pin rows. */
function lm(v: Float64Array, rows: Row[], maxIter: number): number {
  const { cols, pos } = compact(rows)
  const n = cols.length
  if (!rows.length || !n) return 0
  const hard = rows.filter((r) => !r.id.startsWith('pin:'))
  let r = residuals(rows, v)
  let cost = dotSelf(r)
  let lambda = 1e-3
  for (let it = 0; it < maxIter; it++) {
    if (maxAbs(r) < TOL) break
    const { entries } = jacobian(rows, v)
    // Normal equations from the sparse entries (grouped by row).
    const N = new Float64Array(n * n), g = new Float64Array(n)
    let i = 0
    while (i < entries.length) {
      let j = i
      while (j < entries.length && entries[j][0] === entries[i][0]) j++
      for (let a = i; a < j; a++) {
        const ca = pos.get(entries[a][1])!
        g[ca] += entries[a][2] * r[entries[a][0]]
        for (let b = i; b < j; b++) N[ca * n + pos.get(entries[b][1])!] += entries[a][2] * entries[b][2]
      }
      i = j
    }
    let accepted = false
    for (let tries = 0; tries < 30; tries++) {
      const delta = cholSolve(N, g, n, lambda)
      if (delta) {
        const t = v.slice()
        for (let k = 0; k < n; k++) t[cols[k]] -= delta[k]
        const rt = residuals(rows, t)
        const ct = dotSelf(rt)
        if (ct < cost) { v.set(t); r = rt; cost = ct; lambda = Math.max(lambda / 4, 1e-14); accepted = true; break }
      }
      lambda *= 5
      if (lambda > 1e14) break
    }
    if (!accepted) break
  }
  const rh = residuals(hard, v)
  return maxAbs(rh)
}

/** Only variables that some row touches take part in the linear algebra. */
function compact(rows: Row[]) {
  const set = new Set<number>()
  for (const r of rows) for (const c of r.vars) set.add(c)
  const cols = [...set].sort((a, b) => a - b)
  return { cols, pos: new Map(cols.map((c, i) => [c, i])) }
}

const dotSelf = (a: Float64Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return s }

/** Solve (N + λ·diag) x = b by Cholesky; undefined when not positive definite. Only variables that appear get damped. */
function cholSolve(N: Float64Array, b: Float64Array, n: number, lambda: number): Float64Array | undefined {
  const L = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = N[i * n + j]
      if (i === j) s += lambda * Math.max(N[i * n + i], 1e-6) + 1e-14
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k]
      if (i === j) { if (s <= 0) return; L[i * n + i] = Math.sqrt(s) } else L[i * n + j] = s / L[j * n + j]
    }
  }
  const y = new Float64Array(n), x = new Float64Array(n)
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i] }
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i] }
  return x
}

/** DOF, per-variable freedom and redundant constraints from the row space of the Jacobian at the solution. */
function analyse(v: Float64Array, rows: Row[], internalCount: number, sys: System) {
  const total = v.length
  const { cols, pos } = compact(rows)
  const n = cols.length
  const basis: Float64Array[] = []
  const redundant: string[] = []
  const { entries, m } = jacobian(rows, v)
  const dense: Float64Array[] = Array.from({ length: m }, () => new Float64Array(n))
  for (const [r, c, d] of entries) dense[r][pos.get(c)!] = d
  let base = 0
  for (let ri = 0; ri < rows.length; ri++) {
    const row = rows[ri]
    let dependent = false
    for (let k = 0; k < row.n; k++) {
      const vec = dense[base + k]
      const nrm = Math.sqrt(dotSelf(vec))
      if (nrm < 1e-12) { dependent = true; continue }
      for (let i = 0; i < n; i++) vec[i] /= nrm
      for (let pass = 0; pass < 2; pass++) for (const b of basis) { let d = 0; for (let i = 0; i < n; i++) d += b[i] * vec[i]; if (d !== 0) for (let i = 0; i < n; i++) vec[i] -= d * b[i] }
      const rem = Math.sqrt(dotSelf(vec))
      if (rem < 1e-6) { dependent = true; continue }
      for (let i = 0; i < n; i++) vec[i] /= rem
      basis.push(vec)
    }
    if (dependent && ri >= internalCount) redundant.push(row.id)
    base += row.n
  }
  const dof = total - basis.length
  const freeVar = new Float64Array(total).fill(1)
  for (const b of basis) for (let i = 0; i < n; i++) freeVar[cols[i]] -= b[i] * b[i]
  const isFree = (i: number | undefined) => i !== undefined && freeVar[i] > 1e-6
  const free = new Set<string>()
  const pointFree = (id: string) => { const i = sys.idx.get(id); return sys.ents.get(id)?.kind === 'point' && (isFree(i) || isFree(i! + 1)) }
  for (const e of sys.ents.values()) {
    if (e.kind === 'point' && pointFree(e.id)) free.add(e.id)
    else if (e.kind === 'line' && (pointFree(e.a) || pointFree(e.b))) free.add(e.id)
    else if (e.kind === 'arc' && (pointFree(e.a) || pointFree(e.b) || pointFree(e.c))) free.add(e.id)
    else if (e.kind === 'circle' && (pointFree(e.c) || isFree(sys.idx.get(e.id)))) free.add(e.id)
  }
  return { dof, redundant, free }
}

/** Solve; when constraints conflict, drop the offending ones one by one (a move or scale can invalidate old dimensions). */
export function solveRelaxed(entities: SketchEntity[], constraints: SketchConstraint[], opts: SolveOptions = {}) {
  let cs = constraints
  const dropped: string[] = []
  for (let i = 0; i < 16; i++) {
    const r = solve(entities, cs, opts)
    if (r.ok || !r.conflicts.length) return { result: r, constraints: cs, dropped }
    dropped.push(...r.conflicts)
    cs = cs.filter((c) => !r.conflicts.includes(c.id))
  }
  return { result: solve(entities, cs, opts), constraints: cs, dropped }
}
