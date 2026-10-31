import type { ConstraintType, SketchConstraint, SketchEntity } from '@darkcad/core'
import { circle3, isVirtual, type Pt } from './geom'

type DistributiveOmit<T, K extends keyof any> = T extends unknown ? Omit<T, K> : never

export interface SketchModel { entities: SketchEntity[]; constraints: SketchConstraint[] }

export const emptyModel = (): SketchModel => ({ entities: [], constraints: [] })

/** Ids a curve entity depends on. */
export function refsOf(e: SketchEntity): string[] {
  switch (e.kind) {
    case 'point': return []
    case 'line': return [e.a, e.b]
    case 'arc': return [e.c, e.a, e.b]
    case 'circle': return [e.c]
  }
}

/** Mutable working copy of a sketch, with id allocation and the common edits. Call `model()` to get the result. */
export class Draft {
  entities: SketchEntity[]
  constraints: SketchConstraint[]
  private ec = 0
  private cc = 0
  /** Ids created through this draft (tools use it to draw a live preview of just the new geometry). */
  created = new Set<string>()

  constructor(base: SketchModel = emptyModel()) {
    this.entities = base.entities.map((e) => ({ ...e }))
    this.constraints = base.constraints.map((c) => ({ ...c, refs: [...c.refs] }))
    for (const e of this.entities) this.ec = Math.max(this.ec, numSuffix(e.id))
    for (const c of this.constraints) this.cc = Math.max(this.cc, numSuffix(c.id))
  }

  model(): SketchModel { return { entities: this.entities, constraints: this.constraints } }
  get(id: string) { return this.entities.find((e) => e.id === id) }
  pt(id: string): Pt {
    if (id === 'origin') return [0, 0]
    const e = this.get(id)
    return e?.kind === 'point' ? [e.x, e.y] : [0, 0]
  }

  private add(e: DistributiveOmit<SketchEntity, 'id'>): string {
    const id = `e${++this.ec}`
    this.entities.push({ ...e, id } as SketchEntity)
    this.created.add(id)
    return id
  }
  point(x: number, y: number, construction = false): string { return this.add({ kind: 'point', x, y, ...(construction ? { construction } : {}) }) }
  line(a: string, b: string, construction = false): string { return this.add({ kind: 'line', a, b, ...(construction ? { construction } : {}) }) }
  arc(c: string, a: string, b: string, construction = false): string { return this.add({ kind: 'arc', c, a, b, ...(construction ? { construction } : {}) }) }
  circle(c: string, r: number, construction = false): string { return this.add({ kind: 'circle', c, r, ...(construction ? { construction } : {}) }) }

  constrain(type: ConstraintType, refs: string[], extra: Partial<Pick<SketchConstraint, 'value' | 'sign' | 'at'>> = {}): string {
    const id = `c${++this.cc}`
    this.constraints.push({ id, type, refs, ...extra })
    this.created.add(id)
    return id
  }

  /** Remove entities and everything that depended on them; endpoints left unused by removed curves go too. */
  remove(ids: Iterable<string>) {
    const gone = new Set(ids)
    const dropPoints = new Set<string>()
    let changed = true
    while (changed) {
      changed = false
      for (const e of this.entities) {
        if (gone.has(e.id)) continue
        if (refsOf(e).some((r) => gone.has(r))) { gone.add(e.id); changed = true }
      }
    }
    for (const e of this.entities) if (gone.has(e.id) && e.kind !== 'point') for (const r of refsOf(e)) dropPoints.add(r)
    const keep = this.entities.filter((e) => !gone.has(e.id))
    const used = new Set<string>()
    for (const e of keep) for (const r of refsOf(e)) used.add(r)
    for (const c of this.constraints) if (!c.refs.some((r) => gone.has(r))) for (const r of c.refs) used.add(r)
    for (const p of dropPoints) if (!used.has(p) && !gone.has(p) && this.get(p)?.kind === 'point') gone.add(p)
    this.entities = this.entities.filter((e) => !gone.has(e.id))
    this.constraints = this.constraints.filter((c) => !c.refs.some((r) => gone.has(r)))
  }

  /** Point ids used by a curve (or the point itself). */
  pointsOf(id: string): string[] {
    const e = this.get(id)
    if (!e) return isVirtual(id) ? [id] : []
    return e.kind === 'point' ? [id] : refsOf(e)
  }
}

const numSuffix = (id: string) => { const m = /(\d+)$/.exec(id); return m ? parseInt(m[1], 10) : 0 }

/** Replace one point by another everywhere (merging coincident points). */
export function mergePoints(d: Draft, keep: string, drop: string) {
  if (keep === drop) return
  for (const e of d.entities) {
    if (e.kind === 'line') { if (e.a === drop) e.a = keep; if (e.b === drop) e.b = keep }
    else if (e.kind === 'arc') { if (e.a === drop) e.a = keep; if (e.b === drop) e.b = keep; if (e.c === drop) e.c = keep }
    else if (e.kind === 'circle' && e.c === drop) e.c = keep
  }
  for (const c of d.constraints) c.refs = c.refs.map((r) => (r === drop ? keep : r))
  d.entities = d.entities.filter((e) => e.id !== drop)
  // constraints that collapsed onto a single point (e.g. coincident keep,keep) are meaningless
  d.constraints = d.constraints.filter((c) => !(c.type === 'coincident' && c.refs[0] === c.refs[1]))
}

// ----------------------------------------------------------------------------------------------- shape builders
/** Rectangle from two opposite corners: 4 points, 4 lines, horizontal/vertical constraints. */
export function addRect(d: Draft, x: number, y: number, w: number, h: number, construction = false) {
  const p = [d.point(x, y), d.point(x + w, y), d.point(x + w, y + h), d.point(x, y + h)]
  const l = p.map((a, i) => d.line(a, p[(i + 1) % 4], construction))
  d.constrain('horizontal', [l[0]]); d.constrain('vertical', [l[1]]); d.constrain('horizontal', [l[2]]); d.constrain('vertical', [l[3]])
  return { points: p, lines: l }
}

export function addCircle(d: Draft, x: number, y: number, r: number, construction = false) {
  const c = d.point(x, y)
  return { center: c, circle: d.circle(c, r, construction) }
}

export function addPolyline(d: Draft, pts: Pt[], closed = true) {
  const p = pts.map(([x, y]) => d.point(x, y))
  const n = closed ? p.length : p.length - 1
  const l = Array.from({ length: n }, (_, i) => d.line(p[i], p[(i + 1) % p.length]))
  return { points: p, lines: l }
}

/** Arc through three points (start, a point on the arc, end), stored counter-clockwise. */
export function addArc3(d: Draft, a: Pt, through: Pt, b: Pt, ends?: { a?: string; b?: string; c?: string }): string | undefined {
  const cc = circle3(a, through, b)
  if (!cc) return
  const ccw = (through[0] - a[0]) * (b[1] - a[1]) - (through[1] - a[1]) * (b[0] - a[0]) > 0 // a→through→b runs counter-clockwise
  const pa = ends?.a ?? d.point(a[0], a[1]), pb = ends?.b ?? d.point(b[0], b[1])
  const c = ends?.c ?? d.point(cc.c[0], cc.c[1])
  return ccw ? d.arc(c, pa, pb) : d.arc(c, pb, pa)
}

/** Slot between two centres with the given width: two lines and two end arcs, tangent, equal radii (5 degrees of freedom). */
export function addSlot(d: Draft, c1: Pt, c2: Pt, width: number, ends?: { c1?: string; c2?: string }) {
  const dx = c2[0] - c1[0], dy = c2[1] - c1[1], L = Math.hypot(dx, dy) || 1
  const nx = -dy / L, ny = dx / L, r = width / 2
  const p1 = ends?.c1 ?? d.point(c1[0], c1[1]), p2 = ends?.c2 ?? d.point(c2[0], c2[1])
  const a1 = d.point(c1[0] + nx * r, c1[1] + ny * r), a2 = d.point(c2[0] + nx * r, c2[1] + ny * r)
  const b1 = d.point(c1[0] - nx * r, c1[1] - ny * r), b2 = d.point(c2[0] - nx * r, c2[1] - ny * r)
  const top = d.line(a1, a2), bottom = d.line(b1, b2)
  const arc1 = d.arc(p1, a1, b1), arc2 = d.arc(p2, b2, a2)
  d.constrain('tangent', [top, arc1]); d.constrain('tangent', [top, arc2]); d.constrain('tangent', [bottom, arc1]); d.constrain('tangent', [bottom, arc2])
  d.constrain('equal', [arc1, arc2])
  return { centers: [p1, p2], lines: [top, bottom], arcs: [arc1, arc2], points: [a1, a2, b1, b2] }
}

/** Regular polygon on a construction circle. `p` is a corner (inscribed) or the middle of a side (circumscribed). */
export function addPolygon(d: Draft, c: Pt, p: Pt, n: number, mode: 'inscribed' | 'circumscribed' = 'inscribed', centerId?: string) {
  const v = [p[0] - c[0], p[1] - c[1]]
  const base = Math.atan2(v[1], v[0]), rad = Math.hypot(v[0], v[1])
  const R = mode === 'inscribed' ? rad : rad / Math.cos(Math.PI / n)
  const a0 = mode === 'inscribed' ? base : base - Math.PI / n
  const cp = centerId ?? d.point(c[0], c[1])
  const ring = d.circle(cp, R, true)
  const pts = Array.from({ length: n }, (_, i) => d.point(c[0] + R * Math.cos(a0 + (2 * Math.PI * i) / n), c[1] + R * Math.sin(a0 + (2 * Math.PI * i) / n)))
  const lines = pts.map((a, i) => d.line(a, pts[(i + 1) % n]))
  for (const q of pts) d.constrain('point-on', [q, ring])
  for (let i = 1; i < n; i++) d.constrain('equal', [lines[0], lines[i]])
  return { center: cp, ring, points: pts, lines }
}
