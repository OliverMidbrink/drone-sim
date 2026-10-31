import type { ConstraintType, SketchConstraint, SketchEntity } from '@darkcad/core'
import {
  Draft, OpError, addArc3, addPolygon, addRect, addSlot, angleOf, boxSelect, buildGeo, chainOf, chainSide, chamfer, chamferCorner, circle3, circularPattern, constraintArity,
  copyEntities, curvesAtPoint, deleteEntities, dist, extend, fillet, filletCorner, mirrorEntities, moveEntities, offsetChain, pickEntity, planConstraint, polar, rectPattern,
  rotateEntities, scaleEntities, setConstruction, snap, solve, solveRelaxed, sub, trim, len, rot90, norm, dot, cross, type Chain, type Geo, type PickedEntity, type Pt,
  type SketchModel, type SnapResult, type SolveResult,
} from '@darkcad/sketch'

export type ToolId =
  | 'select' | 'line' | 'rect' | 'circle' | 'arc' | 'slot' | 'polygon' | 'point'
  | 'offset' | 'trim' | 'extend' | 'fillet' | 'chamfer' | 'mirror' | 'move' | 'rotate' | 'scale' | 'rpattern' | 'cpattern'
  | 'dimension' | 'constrain'

/** Tools that first need a selection (made with the select behaviour), then act on it. */
export const SELECTION_TOOLS = new Set<ToolId>(['mirror', 'move', 'rotate', 'scale', 'rpattern', 'cpattern'])

export interface ToolOptions {
  rect: '2pt' | 'center' | '3pt'
  circle: 'center' | '2pt' | '3pt'
  arc: '3pt' | 'center'
  polySides: number
  polyMode: 'inscribed' | 'circumscribed'
  /** Whether the circle tool's field asks for a diameter or a radius. */
  circleDim: 'diameter' | 'radius'
  filletR: number
  chamferD: number
  copy: boolean
  dim: 'aligned' | 'horizontal' | 'vertical'
  dimRound: 'auto' | 'radius' | 'diameter'
  constraint: ConstraintType
  nx: number; ny: number; dx: number; dy: number
  count: number; total: number
}

export const defaultOptions = (): ToolOptions => ({
  rect: '2pt', circle: 'center', arc: '3pt', polySides: 6, polyMode: 'inscribed', circleDim: 'diameter', filletR: 2, chamferD: 1, copy: false,
  dim: 'aligned', dimRound: 'auto', constraint: 'coincident', nx: 3, ny: 2, dx: 10, dy: 10, count: 6, total: 360,
})

/** Everything the session needs from the outside world. */
export interface Host {
  model(): SketchModel
  commit(model: SketchModel, coalesceKey?: string): void
  /** Pixels per mm on the sketch plane. */
  scale(): number
  /** Pointer (client coordinates) → plane coordinates. */
  toLocal(clientX: number, clientY: number): Pt
  toScreen(p: Pt): [number, number]
  /** Top-left of the drawing surface in client coordinates. */
  canvasOrigin(): { left: number; top: number }
  /** Something visible changed: redraw. */
  notify(): void
  status(msg: string): void
  selectionChanged(ids: string[]): void
}

export interface Click { pt: Pt; snap?: SnapResult; pid?: string }
/** One typed dimension field: `live` is the value measured from the pointer (shown greyed), `text` what was typed (bold). */
export interface Field { key: string; label: string; text: string; unit: string; live: string }
interface Built { model: SketchModel; created: string[]; relax?: boolean; select?: string[]; keepTool?: boolean; readout?: string }

export interface ValueInput { label: string; unit: string; text: string; at: Pt; submit: (v: number) => void; touched?: boolean }

export interface Modifiers { shift: boolean; ctrl: boolean; alt: boolean }

export interface Analysis { ok: boolean; dof: number; conflicts: Set<string>; redundant: Set<string>; free: Set<string> }

const DEG = Math.PI / 180
const HV_TOL = 1.5 * DEG
const rnd = (v: number, n = 6) => Math.round(v * 10 ** n) / 10 ** n

export class Session {
  tool: ToolId = 'select'
  opts: ToolOptions = defaultOptions()
  /** Snapping to the grid and to objects. Alt suppresses it temporarily. */
  snapOn = true
  showConstraints = true

  sel = new Set<string>()
  hover: string | undefined
  /** Hover target for the selection-style tools (a point id or a curve id). */
  cursor: Pt = [0, 0]
  snapResult: SnapResult | undefined
  clicks: Click[] = []
  /** Last point of the current line chain. */
  chain: { pid: string; pt: Pt; start: string } | undefined
  preview: Built | undefined
  box: { a: Pt; b: Pt } | undefined
  fields: Field[] = []
  activeField = 0
  valueInput: ValueInput | undefined
  /** Phase of selection-driven tools. */
  phase: 'pick' | 'act' = 'pick'
  /** Picks of the dimension and constraint tools. */
  picks: string[] = []
  highlight: string[] = []
  focusConstraint: string | undefined
  /** Screen rectangles of drawn dimension labels, filled in by the renderer (for double-click editing). */
  dimBoxes: { id: string; x: number; y: number; w: number; h: number }[] = []
  chainPreview: Chain | undefined
  message = ''
  /** Which of two ambiguous things a trim/extend/fillet pointer is over, for drawing. */
  hint = ''

  private drag: { ids: string[]; start: Pt; origin: Map<string, Pt>; moved: boolean; px: Pt } | undefined
  private boxStart: { px: Pt; shift: boolean } | undefined
  private cacheEntities: SketchEntity[] | undefined
  private cacheGeo: Geo | undefined
  private cacheAnalysis: { e: SketchEntity[]; c: SketchConstraint[]; a: Analysis } | undefined

  constructor(public host: Host) {}

  // ------------------------------------------------------------------------------------------------ basics
  get model() { return this.host.model() }
  get geo(): Geo {
    const e = this.model.entities
    if (this.cacheEntities !== e || !this.cacheGeo) { this.cacheEntities = e; this.cacheGeo = buildGeo(e) }
    return this.cacheGeo
  }
  private tol(px = 7) { return px / Math.max(this.host.scale(), 1e-9) }
  private say(msg: string) { this.message = msg; this.host.status(msg) }
  private bump() { this.host.notify() }

  analysis(): Analysis {
    const m = this.model
    const c = this.cacheAnalysis
    if (c && c.e === m.entities && c.c === m.constraints) return c.a
    const r: SolveResult = solve(m.entities, m.constraints)
    const a: Analysis = { ok: r.ok, dof: r.dof, conflicts: new Set(r.conflicts), redundant: new Set(r.redundant), free: r.free }
    this.cacheAnalysis = { e: m.entities, c: m.constraints, a }
    return a
  }

  setSelection(ids: Iterable<string>, silent = false) {
    this.sel = new Set(ids)
    if (!silent) this.host.selectionChanged([...this.sel])
    this.bump()
  }

  setTool(t: ToolId, constraint?: ConstraintType) {
    this.resetStage()
    this.tool = t
    if (constraint) this.opts.constraint = constraint
    this.phase = SELECTION_TOOLS.has(t) && this.sel.size ? 'act' : 'pick'
    if (t === 'constrain') { this.picks = [...this.sel]; this.tryConstraint() }
    if (t !== 'select' && !SELECTION_TOOLS.has(t) && t !== 'constrain' && t !== 'dimension') this.setSelection([])
    this.say(this.hintText())
    this.bump()
  }

  resetStage() {
    this.clicks = []; this.chain = undefined; this.preview = undefined; this.fields = []; this.activeField = 0
    this.slotLenTyped = false; this.arcRadiusTyped = false
    this.valueInput = undefined; this.picks = []; this.highlight = []; this.chainPreview = undefined; this.box = undefined; this.drag = undefined; this.boxStart = undefined
  }

  hintText(): string {
    const t = this.tool, o = this.opts
    if (SELECTION_TOOLS.has(t) && this.phase === 'pick') return 'Select the geometry (click or drag a box), then press Enter'
    switch (t) {
      case 'select': return 'Click to select · drag to move · drag a box to select · Shift adds'
      case 'line': return this.clicks.length || this.chain ? 'Click the next point · type a length, Tab for the angle · Enter or Esc ends the chain' : 'Click the start point'
      case 'rect': return o.rect === '3pt' ? 'Click two corners of one side, then the width' : o.rect === 'center' ? 'Click the centre, then a corner' : 'Click one corner, then the opposite corner'
      case 'circle': return o.circle === 'center' ? 'Click the centre, then set the radius' : o.circle === '2pt' ? 'Click both ends of a diameter' : 'Click three points on the circle'
      case 'arc': return o.arc === '3pt' ? 'Click the start, a point on the arc, then the end' : 'Click the centre, the start, then the end'
      case 'slot': return 'Click the two centres, then set the width'
      case 'polygon': return `Click the centre, then a ${o.polyMode === 'inscribed' ? 'corner' : 'side midpoint'} (${o.polySides} sides)`
      case 'point': return 'Click to place a point'
      case 'offset': return this.chainPreview ? 'Move to set the distance, click or type a value and press Enter' : 'Click a curve to offset its whole chain'
      case 'trim': return 'Click the part of a curve to remove'
      case 'extend': return 'Click near the end of a line or arc to extend it to the next curve'
      case 'fillet': return 'Click a corner point, or two curves that meet'
      case 'chamfer': return 'Click a corner point, or two curves that meet'
      case 'mirror': return 'Click the mirror line (a sketch line or an axis)'
      case 'move': return this.clicks.length ? 'Click the destination · type dx, Tab, dy' : 'Click a base point'
      case 'rotate': return this.clicks.length === 0 ? 'Click the centre of rotation' : this.clicks.length === 1 ? 'Click a start direction, or type an angle' : 'Click the new direction'
      case 'scale': return this.clicks.length === 0 ? 'Click the base point' : this.clicks.length === 1 ? 'Click a reference, or type a factor' : 'Click the new size'
      case 'rpattern': return 'Set the counts and spacing, then press Apply'
      case 'cpattern': return this.clicks.length ? 'Set the count and angle, then press Apply' : 'Click the centre of the pattern'
      case 'dimension': return 'Click geometry to dimension: a line, a circle, two points, or two lines for an angle'
      case 'constrain': return `${this.opts.constraint}: pick ${Math.max(1, constraintArity(this.opts.constraint) - this.picks.length)} more`
    }
  }

  // ------------------------------------------------------------------------------------------------ committing
  /** Solve and store. Returns false when the result was rejected. */
  private apply(m: SketchModel, o: { relax?: boolean; key?: string; strict?: boolean; created?: string[] } = {}): boolean {
    const baseline = this.analysis()
    let model = m
    if (baseline.ok) {
      if (o.relax) {
        const r = solveRelaxed(m.entities, m.constraints, { quick: true })
        if (!r.result.ok) { this.say('That edit cannot be solved with the existing constraints'); return false }
        model = { entities: r.result.entities, constraints: r.constraints }
        if (r.dropped.length) this.say(`Released ${r.dropped.length} constraint${r.dropped.length > 1 ? 's' : ''} that no longer hold`)
      } else {
        const s = solve(m.entities, m.constraints, { quick: true })
        if (s.ok) model = { entities: s.entities, constraints: m.constraints }
        else if (o.strict) { this.say('That conflicts with the existing constraints'); return false }
        else {
          // Keep the new geometry, drop the constraints this edit tried to add.
          const added = new Set(o.created ?? [])
          const keep = m.constraints.filter((c) => !added.has(c.id))
          const again = solve(m.entities, keep, { quick: true })
          model = { entities: again.ok ? again.entities : m.entities, constraints: keep }
          this.say('Some automatic constraints conflicted and were skipped')
        }
      }
    }
    this.host.commit(model, o.key)
    return true
  }

  private commitBuilt(b: Built, key?: string): boolean {
    const ok = this.apply(b.model, { relax: b.relax, key, created: b.created })
    if (ok) this.setSelection(b.select ?? [])
    return ok
  }

  // ------------------------------------------------------------------------------------------------ snapping
  private snapAt(raw: Pt, mods: Modifiers, from?: Pt, exclude?: Set<string>): SnapResult {
    const on = this.snapOn && !mods.alt
    return snap(raw, this.geo, { pxPerMm: this.host.scale(), object: on, grid: on, from, angleStep: mods.shift && from ? 15 * DEG : undefined, exclude })
  }

  private needsFrom(): Pt | undefined {
    const t = this.tool
    if (t === 'line') return this.chain?.pt ?? this.clicks[0]?.pt
    if ((t === 'move' || t === 'scale' || t === 'rotate') && this.clicks.length) return this.clicks[0].pt
    return undefined
  }

  private snappingTool() {
    const t = this.tool
    return t === 'line' || t === 'rect' || t === 'circle' || t === 'arc' || t === 'slot' || t === 'polygon' || t === 'point' || t === 'cpattern'
      || (SELECTION_TOOLS.has(t) && this.phase === 'act' && t !== 'mirror' && t !== 'rpattern') || t === 'dimension'
  }

  // ------------------------------------------------------------------------------------------------ typed input
  private num(key: string): number | undefined {
    const f = this.fields.find((x) => x.key === key)
    if (!f || !f.text.length) return undefined
    const v = parseFloat(f.text)
    return Number.isFinite(v) ? v : undefined
  }
  private setFields(spec: [string, string, string][]) {
    if (this.fields.length === spec.length && this.fields.every((f, i) => f.key === spec[i][0] && f.label === spec[i][1])) return
    this.fields = spec.map(([key, label, unit]) => ({ key, label, unit, text: '', live: '' }))
    this.activeField = 0
  }
  private updateFields() {
    const t = this.tool, n = this.clicks.length, o = this.opts
    const MM = 'mm', DG = '°'
    let spec: [string, string, string][] = []
    if (t === 'line' && (this.chain || n >= 1)) spec = [['len', 'Length', MM], ['ang', 'Angle', DG]]
    else if (t === 'rect' && o.rect !== '3pt' && n >= 1) spec = [['w', 'Width', MM], ['h', 'Height', MM]]
    else if (t === 'rect' && o.rect === '3pt' && n >= 2) spec = [['w', 'Width', MM]]
    else if (t === 'circle' && o.circle !== '3pt' && n >= 1) spec = o.circleDim === 'diameter' ? [['d', 'Diameter', MM]] : [['r', 'Radius', MM]]
    else if (t === 'polygon' && n >= 1) spec = [['od', 'Outer Ø', MM], ['id', 'Inner Ø', MM], ['n', 'Sides', ''], ['rot', 'Rotation', DG]]
    else if (t === 'slot' && n === 1) spec = [['len', 'Length', MM], ['ang', 'Angle', DG]]
    else if (t === 'slot' && n >= 2) spec = [['w', 'Width', MM]]
    else if (t === 'arc' && o.arc === 'center' && n === 1) spec = [['r', 'Radius', MM]]
    else if (t === 'arc' && o.arc === 'center' && n >= 2) spec = [['sweep', 'Sweep', DG]]
    else if (t === 'offset' && this.chainPreview) spec = [['d', 'Distance', MM]]
    else if (t === 'move' && this.phase === 'act' && n >= 1) spec = [['dx', 'dx', MM], ['dy', 'dy', MM]]
    else if (t === 'rotate' && this.phase === 'act' && n >= 1) spec = [['a', 'Angle', DG]]
    else if (t === 'scale' && this.phase === 'act' && n >= 1) spec = [['k', 'Factor', '']]
    if (!spec.length) { this.fields = []; return }
    this.setFields(spec)
  }

  /** Values the shape being drawn would have for the pointer alone (nothing typed), shown greyed in the fields. */
  private liveValues(q: Pt): Record<string, number> {
    const t = this.tool, o = this.opts, cs = this.clicks
    const out: Record<string, number> = {}
    const wrap = (a: number) => ((a / DEG) % 360 + 360) % 360
    if (t === 'line') {
      const from = this.chain?.pt ?? cs[0]?.pt
      if (from) { const v = sub(q, from); out.len = len(v); out.ang = wrap(angleOf(v)) }
    } else if (t === 'rect') {
      if (o.rect === '3pt') {
        if (cs.length >= 2) out.w = Math.abs(dot(sub(q, cs[0].pt), rot90(norm(sub(cs[1].pt, cs[0].pt)))))
      } else if (cs.length >= 1) {
        const k = o.rect === 'center' ? 2 : 1
        out.w = Math.abs(q[0] - cs[0].pt[0]) * k; out.h = Math.abs(q[1] - cs[0].pt[1]) * k
      }
    } else if (t === 'circle') {
      if (o.circle !== '3pt' && cs.length >= 1) { const R = o.circle === 'center' ? dist(cs[0].pt, q) : dist(cs[0].pt, q) / 2; out.r = R; out.d = 2 * R }
    } else if (t === 'polygon') {
      if (cs.length >= 1) {
        const g = this.polyGeom(cs[0].pt, q)
        out.od = 2 * g.rOut; out.id = 2 * g.rOut * Math.cos(Math.PI / g.n); out.n = g.n; out.rot = wrap(g.ang)
      }
    } else if (t === 'slot') {
      if (cs.length === 1) { const v = sub(q, cs[0].pt); out.len = len(v); out.ang = wrap(angleOf(v)) }
      else if (cs.length >= 2) out.w = 2 * Math.abs(cross(norm(sub(cs[1].pt, cs[0].pt)), sub(q, cs[0].pt)))
    } else if (t === 'arc' && o.arc === 'center') {
      if (cs.length === 1) out.r = dist(cs[0].pt, q)
      else if (cs.length >= 2) {
        const c = cs[0].pt
        let sw = angleOf(sub(q, c)) - angleOf(sub(cs[1].pt, c))
        sw = ((sw % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)
        out.sweep = sw / DEG
      }
    }
    return out
  }
  private applyLive(q: Pt) {
    if (!this.fields.length) return
    const v = this.liveValues(q)
    for (const f of this.fields) f.live = v[f.key] !== undefined ? String(rnd(v[f.key], f.unit === '°' ? 1 : 3)) : ''
  }

  /** Polygon placement from the pointer and any typed outer diameter, inner diameter (across flats), sides or rotation. */
  private polyGeom(c0: Pt, q: Pt, ignoreTyped = false): { rOut: number; ang: number; n: number } {
    const o = this.opts
    const typedN = ignoreTyped ? undefined : this.num('n')
    const n = typedN !== undefined ? Math.max(3, Math.min(64, Math.round(typedN))) : o.polySides
    const k = Math.cos(Math.PI / n)
    const v = sub(q, c0), rad = len(v), a = angleOf(v)
    const inscribed = o.polyMode === 'inscribed'
    let rOut = inscribed ? rad : rad / k
    let ang = inscribed ? a : a - Math.PI / n
    if (!ignoreTyped) {
      const OD = this.num('od'), ID = this.num('id'), ROT = this.num('rot')
      if (OD !== undefined) rOut = OD / 2
      else if (ID !== undefined) rOut = ID / 2 / k
      if (ROT !== undefined) ang = ROT * DEG
    }
    return { rOut, ang, n }
  }

  // ------------------------------------------------------------------------------------------------ building geometry
  private reusable(c: Click): string | undefined {
    if (c.pid) return c.pid
    const s = c.snap
    if (s && (s.kind === 'endpoint' || s.kind === 'center' || s.kind === 'origin') && s.ref && !s.ref.startsWith('$')) return s.ref
  }

  /** Constraints that tie a freshly made point to whatever its click snapped onto. */
  private attach(d: Draft, id: string, c: Click) {
    const s = c.snap
    if (!s || c.pid) return
    const ref = s.ref
    if ((s.kind === 'endpoint' || s.kind === 'center' || s.kind === 'origin') && ref) d.constrain('coincident', [id, ref])
    else if (s.kind === 'on-curve' || s.kind === 'quadrant' || s.kind === 'tangent' || s.kind === 'perpendicular') { if (ref) d.constrain('point-on', [id, ref]) }
    else if (s.kind === 'midpoint' && ref) d.constrain(d.get(ref)?.kind === 'line' || ref.startsWith('axis') ? 'midpoint' : 'point-on', [id, ref])
    else if (s.kind === 'intersection' && s.refs) for (const r of s.refs) d.constrain('point-on', [id, r])
  }

  /** The point a click stands for: an existing one it snapped to, or a new one tied to the snapped geometry. */
  private pointFor(d: Draft, c: Click): string {
    const re = this.reusable(c)
    if (re) return re
    const id = d.point(c.pt[0], c.pt[1])
    this.attach(d, id, c)
    return id
  }

  /** Driving width/height dimensions for the typed sides of a rectangle (points are addRect's: BL, BR, TR, TL). */
  private dimRect(d: Draft, p: string[], w?: number, h?: number) {
    if (w !== undefined) d.constrain('hdist', [p[0], p[1]], { value: w })
    if (h !== undefined) d.constrain('vdist', [p[0], p[3]], { value: h })
  }
  /** Radius for the circle tools: the typed one (diameter or radius, per the option), else `fallback`. */
  private circleR(fallback: number): number {
    const dia = this.opts.circleDim === 'diameter'
    const v = this.num(dia ? 'd' : 'r')
    if (v === undefined) return fallback
    return dia ? v / 2 : v
  }
  private dimCircle(d: Draft, id: string, R: number) {
    if (this.num(this.opts.circleDim === 'diameter' ? 'd' : 'r') === undefined) return
    if (this.opts.circleDim === 'diameter') d.constrain('diameter', [id], { value: 2 * R })
    else d.constrain('radius', [id], { value: R })
  }
  /** Set while a typed radius / length from an earlier stage of the same shape is waiting to become a dimension. */
  private arcRadiusTyped = false
  private slotLenTyped = false

  private build(final: boolean, at: Click): Built | undefined {
    const t = this.tool, o = this.opts, cs = this.clicks
    const d = new Draft(this.model)
    const done = (select?: string[], extra: Partial<Built> = {}): Built => ({ model: d.model(), created: [...d.created], select, ...extra })
    const q = at.pt

    switch (t) {
      case 'line': {
        const from: Click | undefined = this.chain ? { pt: this.chain.pt, pid: this.chain.pid } : cs[0]
        if (!from) return
        let to: Click = at
        const L = this.num('len'), A = this.num('ang')
        if (L !== undefined || A !== undefined) {
          const v = sub(q, from.pt)
          const l = L ?? len(v), a = A !== undefined ? A * DEG : angleOf(v)
          to = { pt: polar(from.pt, l, a) }
        }
        let b = to.pt
        const tv = sub(b, from.pt)
        const hasSnap = to.snap && ['endpoint', 'center', 'origin', 'intersection', 'midpoint', 'quadrant'].includes(to.snap.kind)
        let hv: 'h' | 'v' | undefined
        if (len(tv) > 1e-9) {
          const ang = Math.abs(Math.atan2(tv[1], tv[0]))
          const nearH = Math.min(ang, Math.PI - ang) < (hasSnap || this.num('ang') !== undefined ? 1e-9 : HV_TOL)
          const nearV = Math.abs(ang - Math.PI / 2) < (hasSnap || this.num('ang') !== undefined ? 1e-9 : HV_TOL)
          if (nearH) { hv = 'h'; if (!hasSnap) b = [b[0], from.pt[1]] }
          else if (nearV) { hv = 'v'; if (!hasSnap) b = [from.pt[0], b[1]] }
        }
        if (len(sub(b, from.pt)) < 1e-9) return
        const a = this.reusable(from) ?? (() => { const id = d.point(from.pt[0], from.pt[1]); this.attach(d, id, from); return id })()
        const endClick: Click = { ...to, pt: b }
        const bId = this.reusable(endClick) ?? (() => { const id = d.point(b[0], b[1]); this.attach(d, id, endClick); return id })()
        if (a === bId) return
        const l = d.line(a, bId)
        if (final && L !== undefined) d.constrain('distance', [l], { value: len(sub(b, from.pt)) })
        if (hv === 'h') d.constrain('horizontal', [l]); else if (hv === 'v') d.constrain('vertical', [l])
        const s = to.snap
        if (s?.ref && !hv) {
          const cu = this.geo.byId.get(s.ref)
          if (s.kind === 'perpendicular' && cu?.kind === 'line') d.constrain('perpendicular', [l, s.ref])
          if (s.kind === 'tangent' && cu && cu.kind !== 'line') d.constrain('tangent', [l, s.ref])
        }
        return done([], { readout: `${rnd(len(sub(b, from.pt)), 3)}  ∠ ${rnd(((angleOf(sub(b, from.pt)) / DEG) + 360) % 360, 1)}°`, select: [] })
      }
      case 'rect': {
        if (o.rect === '3pt') {
          if (cs.length < 2) return
          const [p1, p2] = [cs[0], cs[1]]
          const u = norm(sub(p2.pt, p1.pt)), n = rot90(u)
          const Wt = this.num('w')
          const signed = Wt !== undefined ? Wt * (dot(sub(q, p1.pt), n) < 0 ? -1 : 1) : dot(sub(q, p1.pt), n)
          if (Math.abs(signed) < 1e-9 || len(sub(p2.pt, p1.pt)) < 1e-9) return
          const corners: Pt[] = [p1.pt, p2.pt, [p2.pt[0] + n[0] * signed, p2.pt[1] + n[1] * signed], [p1.pt[0] + n[0] * signed, p1.pt[1] + n[1] * signed]]
          const ids = corners.map((c, i) => (i < 2 ? this.pointFor(d, cs[i]) : d.point(c[0], c[1])))
          if (new Set(ids).size < 4) return
          const ls = ids.map((id, i) => d.line(id, ids[(i + 1) % 4]))
          d.constrain('perpendicular', [ls[0], ls[1]]); d.constrain('perpendicular', [ls[1], ls[2]]); d.constrain('perpendicular', [ls[2], ls[3]])
          if (final && Wt !== undefined) d.constrain('distance', [ls[1]], { value: Math.abs(signed) })
          return done([], { readout: `${rnd(len(sub(p2.pt, p1.pt)), 3)} × ${rnd(Math.abs(signed), 3)}` })
        }
        if (cs.length < 1) return
        const c0 = cs[0].pt
        let w = q[0] - c0[0], h = q[1] - c0[1]
        const W = this.num('w'), H = this.num('h')
        const sx = w < 0 ? -1 : 1, sy = h < 0 ? -1 : 1
        if (o.rect === 'center') {
          const hw = W !== undefined ? W / 2 : Math.abs(w), hh = H !== undefined ? H / 2 : Math.abs(h)
          if (hw < 1e-9 || hh < 1e-9) return
          const r = addRect(d, c0[0] - hw, c0[1] - hh, hw * 2, hh * 2)
          const diag = d.line(r.points[0], r.points[2], true)
          const existing = this.reusable(cs[0])
          const c = existing ?? d.point(c0[0], c0[1], true)
          if (!existing) this.attach(d, c, cs[0])
          d.constrain('midpoint', [c, diag])
          if (final) this.dimRect(d, r.points, W !== undefined ? hw * 2 : undefined, H !== undefined ? hh * 2 : undefined)
          return done([], { readout: `${rnd(hw * 2, 3)} × ${rnd(hh * 2, 3)}` })
        }
        if (W !== undefined) w = W * sx
        if (H !== undefined) h = H * sy
        if (Math.abs(w) < 1e-9 || Math.abs(h) < 1e-9) return
        const r = addRect(d, Math.min(c0[0], c0[0] + w), Math.min(c0[1], c0[1] + h), Math.abs(w), Math.abs(h))
        // addRect's corner 0 is bottom-left, then anticlockwise; find which corner is the clicked one
        const startIdx = sx > 0 ? (sy > 0 ? 0 : 3) : sy > 0 ? 1 : 2
        const endIdx = (startIdx + 2) % 4
        this.attach(d, r.points[startIdx], cs[0])
        this.attach(d, r.points[endIdx], { ...at, pt: [c0[0] + w, c0[1] + h] })
        if (final) this.dimRect(d, r.points, W !== undefined ? Math.abs(w) : undefined, H !== undefined ? Math.abs(h) : undefined)
        return done([], { readout: `${rnd(Math.abs(w), 3)} × ${rnd(Math.abs(h), 3)}` })
      }
      case 'circle': {
        if (o.circle === 'center') {
          if (cs.length < 1) return
          const R = this.circleR(dist(cs[0].pt, q))
          if (R < 1e-9) return
          const c = this.pointFor(d, cs[0])
          const ci = d.circle(c, R)
          if (final) this.dimCircle(d, ci, R)
          return done([], { readout: `R ${rnd(R, 3)}  Ø ${rnd(R * 2, 3)}` })
        }
        if (o.circle === '2pt') {
          if (cs.length < 1) return
          const typedR = this.circleR(NaN)
          let e = q
          if (Number.isFinite(typedR)) { const dir = dist(cs[0].pt, q) > 1e-9 ? norm(sub(q, cs[0].pt)) : [1, 0] as Pt; e = polar(cs[0].pt, typedR * 2, angleOf(dir)) }
          const m: Pt = [(cs[0].pt[0] + e[0]) / 2, (cs[0].pt[1] + e[1]) / 2], R = dist(cs[0].pt, e) / 2
          if (R < 1e-9) return
          const ci = d.circle(d.point(m[0], m[1]), R)
          if (final && Number.isFinite(typedR)) this.dimCircle(d, ci, R)
          return done([], { readout: `Ø ${rnd(R * 2, 3)}` })
        }
        if (cs.length < 2) return
        const cc = circle3(cs[0].pt, cs[1].pt, q)
        if (!cc) return
        d.circle(d.point(cc.c[0], cc.c[1]), cc.r)
        return done([], { readout: `R ${rnd(cc.r, 3)}` })
      }
      case 'arc': {
        if (o.arc === '3pt') {
          if (cs.length < 2) return
          const a = this.pointFor(d, cs[0]), b = this.pointFor(d, at)
          const id = addArc3(d, cs[0].pt, cs[1].pt, q, { a, b })
          if (!id) return
          return done([], {})
        }
        if (cs.length < 2) return
        const c = cs[0].pt, r = dist(c, cs[1].pt)
        if (r < 1e-9) return
        const SW = this.num('sweep')
        const a0 = angleOf(sub(cs[1].pt, c))
        const end = polar(c, r, SW !== undefined ? a0 + SW * DEG : angleOf(sub(q, c)))
        if (dist(end, cs[1].pt) < 1e-9) return
        const cp = this.pointFor(d, cs[0]), sp = this.pointFor(d, cs[1])
        const ep = d.point(end[0], end[1])
        if (SW === undefined && at.snap && (at.snap.kind === 'endpoint' || at.snap.kind === 'on-curve') && at.snap.ref) d.constrain(at.snap.kind === 'endpoint' ? 'coincident' : 'point-on', [ep, at.snap.ref])
        const ai = SW !== undefined && SW < 0 ? d.arc(cp, ep, sp) : d.arc(cp, sp, ep)
        if (final && this.arcRadiusTyped) d.constrain('radius', [ai], { value: r })
        return done([], { readout: `R ${rnd(r, 3)}` })
      }
      case 'slot': {
        if (cs.length < 2) return
        const W = this.num('w') ?? 2 * Math.abs(cross(norm(sub(cs[1].pt, cs[0].pt)), sub(q, cs[0].pt)))
        if (W < 1e-9 || dist(cs[0].pt, cs[1].pt) < 1e-9) return
        const c1 = this.pointFor(d, cs[0]), c2 = this.pointFor(d, cs[1])
        const sl = addSlot(d, cs[0].pt, cs[1].pt, W, { c1, c2 })
        if (final && this.num('w') !== undefined) d.constrain('diameter', [sl.arcs[0]], { value: W })
        if (final && this.slotLenTyped) d.constrain('distance', [c1, c2], { value: dist(cs[0].pt, cs[1].pt) })
        return done([], { readout: `${rnd(dist(cs[0].pt, cs[1].pt), 3)} · w ${rnd(W, 3)}` })
      }
      case 'polygon': {
        if (cs.length < 1) return
        const g = this.polyGeom(cs[0].pt, q)
        if (g.rOut < 1e-9) return
        const p = polar(cs[0].pt, g.rOut, g.ang)
        const cid = this.pointFor(d, cs[0])
        const pg = addPolygon(d, cs[0].pt, p, g.n, 'inscribed', cid)
        if (final && (this.num('od') !== undefined || this.num('id') !== undefined)) d.constrain('diameter', [pg.ring], { value: 2 * g.rOut })
        if (final) this.opts = { ...this.opts, polySides: g.n }
        return done([], { readout: `${g.n} sides · Ø ${rnd(2 * g.rOut, 3)} · across flats ${rnd(2 * g.rOut * Math.cos(Math.PI / g.n), 3)}` })
      }
      case 'point': {
        const id = d.point(q[0], q[1])
        this.attach(d, id, at)
        return done([])
      }
      case 'cpattern': {
        if (this.phase !== 'act' || cs.length < 1) return
        const r = circularPattern(this.model, [...this.sel], cs[0].pt, Math.max(2, Math.round(o.count)), (o.total * DEG))
        return { ...r, select: r.created.filter((i) => r.model.entities.some((e) => e.id === i)), relax: true }
      }
      default: return
    }
  }

  /** Build for tools that operate on existing geometry. Throws OpError for impossible requests. */
  private buildModify(final: boolean, raw: Pt, at: Click): Built | undefined {
    const t = this.tool, o = this.opts, m = this.model
    const geo = this.geo
    const tol = this.tol(8)
    const wrap = (r: { model: SketchModel; created: string[] }, extra: Partial<Built> = {}): Built => ({ ...r, select: r.created.filter((i) => r.model.entities.some((e) => e.id === i && e.kind !== 'point')), ...extra })
    switch (t) {
      case 'offset': {
        if (!this.chainPreview) return
        const typed = this.num('d')
        const dd = typed !== undefined ? typed * (chainSide(m, this.chainPreview, raw) < 0 ? -1 : 1) : chainSide(m, this.chainPreview, raw)
        if (Math.abs(dd) < 1e-9) return
        const r = offsetChain(m, this.chainPreview, dd)
        return wrap(r, { readout: `offset ${rnd(Math.abs(dd), 3)}` })
      }
      case 'trim': case 'extend': {
        const id = pickEntity(geo, raw, tol, { only: ['line', 'arc', 'circle'] })
        if (!id) return
        return wrap(t === 'trim' ? trim(m, id, raw) : extend(m, id, raw))
      }
      case 'fillet': case 'chamfer': {
        const isF = t === 'fillet'
        const pid = pickEntity(geo, raw, tol, { only: ['point'] })
        if (pid && this.picks.length === 0) {
          const cu = curvesAtPoint(m, pid)
          if (cu.length === 2) return wrap(isF ? filletCorner(m, pid, o.filletR) : chamferCorner(m, pid, o.chamferD))
          return
        }
        const cid = pickEntity(geo, raw, tol, { only: ['line', 'arc'] })
        if (this.picks.length === 1 && cid && cid !== this.picks[0]) return wrap(isF ? fillet(m, this.picks[0], cid, o.filletR, raw) : chamfer(m, this.picks[0], cid, o.chamferD))
        return
      }
      case 'mirror': {
        const id = pickEntity(geo, raw, tol, { only: ['line'], virtual: true, exclude: this.sel })
        if (!id) return
        return wrap(mirrorEntities(m, [...this.sel], id), { relax: true })
      }
      case 'move': {
        if (this.clicks.length < 1) return
        const base = this.clicks[0].pt
        const dx = this.num('dx') ?? q0(at)[0] - base[0], dy = this.num('dy') ?? q0(at)[1] - base[1]
        if (Math.hypot(dx, dy) < 1e-9) return
        const r = o.copy ? copyEntities(m, [...this.sel], dx, dy) : moveEntities(m, [...this.sel], dx, dy)
        return wrap(r, { relax: true, readout: `Δ ${rnd(dx, 3)}, ${rnd(dy, 3)}`, select: o.copy ? undefined : [...this.sel] })
      }
      case 'rotate': {
        if (this.clicks.length < 1) return
        const c = this.clicks[0].pt
        const typed = this.num('a')
        let ang: number
        if (typed !== undefined) ang = typed * DEG
        else {
          if (this.clicks.length < 2) return
          ang = angleOf(sub(at.pt, c)) - angleOf(sub(this.clicks[1].pt, c))
        }
        if (Math.abs(ang) < 1e-9) return
        return wrap(rotateEntities(m, [...this.sel], c, ang), { relax: true, readout: `${rnd(ang / DEG, 2)}°`, select: [...this.sel] })
      }
      case 'scale': {
        if (this.clicks.length < 1) return
        const b = this.clicks[0].pt
        let k = this.num('k')
        if (k === undefined) {
          if (this.clicks.length < 2) return
          const d0 = dist(b, this.clicks[1].pt)
          if (d0 < 1e-9) return
          k = dist(b, at.pt) / d0
        }
        if (!(k > 1e-6) || Math.abs(k - 1) < 1e-9) return
        return wrap(scaleEntities(m, [...this.sel], b, k), { relax: true, readout: `× ${rnd(k, 4)}`, select: [...this.sel] })
      }
      case 'rpattern': {
        if (this.phase !== 'act' || !this.sel.size) return
        return wrap(rectPattern(m, [...this.sel], Math.max(1, Math.round(o.nx)), Math.max(1, Math.round(o.ny)), o.dx, o.dy), { relax: true })
      }
      default: return
    }
  }

  /** Recompute the live preview for the current pointer position. */
  private refresh(raw: Pt, mods: Modifiers) {
    const t = this.tool
    this.preview = undefined
    this.hint = ''
    this.updateFields()
    this.highlight = this.tool === 'offset' && this.chainPreview ? this.chainPreview.els.map((e) => e.id) : []
    try {
      if (t === 'select' || t === 'dimension' || t === 'constrain') { this.hover = pickEntity(this.geo, raw, this.tol(), { virtual: t !== 'select' }); return }
      if (SELECTION_TOOLS.has(t) && this.phase === 'pick') { this.hover = pickEntity(this.geo, raw, this.tol()); return }
      const exclude = new Set<string>()
      const sn = this.snappingTool() ? this.snapAt(raw, mods, this.needsFrom(), exclude) : undefined
      this.snapResult = sn
      const at: Click = { pt: sn?.pt ?? raw, snap: sn }
      this.hover = undefined
      if (['offset', 'trim', 'extend', 'fillet', 'chamfer', 'mirror', 'rpattern'].includes(t)) {
        this.hover = pickEntity(this.geo, raw, this.tol(8), { only: t === 'mirror' ? ['line'] : t === 'fillet' || t === 'chamfer' ? ['line', 'arc', 'point'] : undefined, virtual: t === 'mirror' })
        this.preview = this.buildModify(false, raw, at)
        if (t === 'fillet' || t === 'chamfer') this.hint = this.hover ? '' : 'Pick a corner'
        return
      }
      if (['move', 'rotate', 'scale', 'cpattern'].includes(t)) {
        if (t === 'cpattern') this.preview = this.build(false, at)
        else this.preview = this.buildModify(false, raw, at)
        return
      }
      this.applyLive(at.pt)
      this.preview = this.build(false, at)
    } catch (e) {
      if (e instanceof OpError) this.hint = e.message
      else throw e
    }
  }

  // ------------------------------------------------------------------------------------------------ pointer
  pointerMove(clientX: number, clientY: number, mods: Modifiers, buttons: number) {
    const raw = this.host.toLocal(clientX, clientY)
    this.cursor = raw
    if (this.boxStart) {
      this.box = { a: this.box?.a ?? raw, b: raw }
    }
    if (this.drag && (buttons & 1)) { this.dragTo(raw, clientX, clientY, mods); return }
    this.refresh(raw, mods)
    this.bump()
  }

  pointerDown(clientX: number, clientY: number, mods: Modifiers, detail: number) {
    const raw = this.host.toLocal(clientX, clientY)
    this.cursor = raw
    const t = this.tool
    if (detail >= 2 && this.editDimAt(clientX, clientY)) return
    if (t === 'select' || (SELECTION_TOOLS.has(t) && this.phase === 'pick')) return this.selectDown(raw, clientX, clientY, mods, t === 'select')
    if (t === 'dimension') return this.dimensionClick(raw, clientX, clientY)
    if (t === 'constrain') return this.constrainClick(raw, mods)
    this.refresh(raw, mods)
    const sn = this.snappingTool() ? this.snapResult ?? this.snapAt(raw, mods, this.needsFrom()) : undefined
    const at: Click = { pt: sn?.pt ?? raw, snap: sn }
    try { this.toolClick(raw, at, mods, detail) } catch (e) { if (e instanceof OpError) this.say(e.message); else throw e }
    this.fields = []
    this.refresh(raw, mods)
    this.say(this.hintText())
    this.bump()
  }

  pointerUp(clientX: number, clientY: number, mods: Modifiers) {
    const raw = this.host.toLocal(clientX, clientY)
    if (this.boxStart) {
      const bs = this.boxStart
      const moved = this.box && Math.hypot(this.host.toScreen(this.box.b)[0] - bs.px[0], this.host.toScreen(this.box.b)[1] - bs.px[1]) > 4
      if (this.box && moved) {
        const a = this.box.a, b = this.box.b
        const rect = { x0: Math.min(a[0], b[0]), y0: Math.min(a[1], b[1]), x1: Math.max(a[0], b[0]), y1: Math.max(a[1], b[1]) }
        const ids = boxSelect(this.model.entities, rect, b[0] < a[0])
        this.setSelection(bs.shift ? [...this.sel, ...ids] : ids)
      } else if (!bs.shift) this.setSelection([])
      this.box = undefined; this.boxStart = undefined
    }
    if (this.drag) {
      const moved = this.drag.moved
      this.drag = undefined
      if (moved) this.say(this.statusText())
    }
    this.cursor = raw
    this.bump()
  }

  // ---- select behaviour
  private selectDown(raw: Pt, cx: number, cy: number, mods: Modifiers, allowDrag: boolean) {
    const id = pickEntity(this.geo, raw, this.tol())
    const add = mods.shift || mods.ctrl
    if (id) {
      if (add) { const s = new Set(this.sel); if (s.has(id)) s.delete(id); else s.add(id); this.setSelection(s) }
      else if (!this.sel.has(id)) this.setSelection([id])
      if (allowDrag && this.sel.has(id)) {
        const ids = [...this.sel]
        const origin = new Map<string, Pt>()
        const d = new Draft(this.model)
        for (const s of ids) for (const p of d.pointsOf(s)) if (p !== 'origin' && !p.startsWith('$')) origin.set(p, d.pt(p))
        if (origin.size) this.drag = { ids, start: raw, origin, moved: false, px: [cx, cy] }
      }
    } else {
      this.boxStart = { px: this.host.toScreen(raw), shift: add }
      this.box = { a: raw, b: raw }
      if (!add) this.setSelection([])
    }
    this.bump()
  }

  private dragTo(raw: Pt, cx: number, cy: number, mods: Modifiers) {
    const dr = this.drag!
    if (!dr.moved && Math.hypot(cx - dr.px[0], cy - dr.px[1]) < 3) return
    dr.moved = true
    const dx = raw[0] - dr.start[0], dy = raw[1] - dr.start[1]
    const m = this.model
    const pins = [...dr.origin].map(([id, p]) => ({ id, x: p[0] + dx, y: p[1] + dy }))
    const r = solve(m.entities, m.constraints, { pins, quick: true })
    if (!r.ok) { this.say('Blocked by constraints'); this.bump(); return }
    this.host.commit({ entities: r.entities, constraints: m.constraints }, `drag:${dr.ids.join(',')}`)
    void mods
    this.bump()
  }

  // ---- tool clicks
  private toolClick(raw: Pt, at: Click, mods: Modifiers, detail: number) {
    const t = this.tool, o = this.opts
    const need = (): number => {
      switch (t) {
        case 'rect': return o.rect === '3pt' ? 3 : 2
        case 'circle': return o.circle === '3pt' ? 3 : 2
        case 'arc': return 3
        case 'slot': return 3
        case 'polygon': return 2
        case 'line': return 2
        case 'move': return 2
        case 'rotate': return 3
        case 'scale': return 3
        case 'point': return 1
        default: return 1
      }
    }
    switch (t) {
      case 'line': {
        if (detail >= 2 && (this.chain || this.clicks.length)) { this.chain = undefined; this.clicks = []; return }
        if (!this.chain && !this.clicks.length) { this.clicks = [at]; return }
        const b = this.build(true, at)
        if (!b) return
        const ok = this.commitBuilt(b)
        if (ok) {
          // continue from the end of the line just made
          const l = [...b.created].reverse().find((i) => b.model.entities.find((e) => e.id === i)?.kind === 'line')
          const line = b.model.entities.find((e) => e.id === l)
          const solved = this.model.entities.find((e) => e.id === (line?.kind === 'line' ? line.b : ''))
          const endPt: Pt = solved?.kind === 'point' ? [solved.x, solved.y] : at.pt
          const startId = this.chain?.start ?? (line?.kind === 'line' ? line.a : '')
          const endId = line?.kind === 'line' ? line.b : ''
          this.clicks = []
          this.chain = endId === startId ? undefined : { pid: endId, pt: endPt, start: startId }
        }
        return
      }
      case 'rect': case 'circle': case 'arc': case 'slot': case 'polygon': {
        if (this.clicks.length + 1 < need()) { this.slotLenTyped = false; this.arcRadiusTyped = false; this.clicks = [...this.clicks, at]; return }
        const b = this.build(true, at)
        if (b && this.commitBuilt(b)) { this.clicks = []; this.slotLenTyped = false; this.arcRadiusTyped = false }
        return
      }
      case 'point': {
        const b = this.build(true, at)
        if (b) this.commitBuilt(b)
        return
      }
      case 'offset': {
        if (!this.chainPreview) {
          const id = pickEntity(this.geo, raw, this.tol(8), { only: ['line', 'arc', 'circle'] })
          if (id) { this.chainPreview = chainOf(this.model, id); this.setSelection(this.chainPreview.els.map((e) => e.id), true) }
          return
        }
        const b = this.buildModify(true, raw, at)
        if (b && this.commitBuilt(b)) { this.chainPreview = undefined; this.highlight = [] }
        return
      }
      case 'trim': case 'extend': case 'mirror': {
        const b = this.buildModify(true, raw, at)
        if (b) this.commitBuilt(b)
        if (t === 'mirror' && b) { this.phase = 'pick' }
        return
      }
      case 'fillet': case 'chamfer': {
        const pid = pickEntity(this.geo, raw, this.tol(8), { only: ['point'] })
        if (pid) { const b = this.buildModify(true, raw, at); if (b) this.commitBuilt(b); this.picks = []; return }
        const cid = pickEntity(this.geo, raw, this.tol(8), { only: ['line', 'arc'] })
        if (!cid) { this.picks = []; return }
        if (!this.picks.length) { this.picks = [cid]; this.highlight = [cid]; return }
        const b = this.buildModify(true, raw, at)
        if (b) this.commitBuilt(b)
        this.picks = []; this.highlight = []
        return
      }
      case 'move': case 'rotate': case 'scale': {
        if (this.phase !== 'act') return
        if (this.clicks.length + 1 < need()) { this.clicks = [...this.clicks, at]; return }
        const b = this.buildModify(true, raw, at)
        if (b && this.commitBuilt(b)) { this.clicks = []; if (t !== 'move' || !o.copy) this.phase = 'pick'; else this.phase = 'act' }
        return
      }
      case 'cpattern': {
        if (this.phase === 'act' && this.clicks.length === 0) this.clicks = [at]
        else if (this.phase === 'act') this.clicks = [at]
        return
      }
      default: return
    }
  }

  // ------------------------------------------------------------------------------------------------ keyboard
  /** Returns true when the key was consumed. */
  keyDown(e: { key: string; ctrlKey: boolean; metaKey: boolean; shiftKey: boolean; altKey: boolean }, preventDefault: () => void): boolean {
    const key = e.key
    const mod = e.ctrlKey || e.metaKey
    if (this.valueInput) return this.valueKey(key, preventDefault)
    if (mod) return false
    // typed numbers
    if (this.fields.length) {
      const f = this.fields[this.activeField]
      const refreshNow = () => { this.refresh(this.cursor, { shift: false, ctrl: false, alt: false }); this.bump() }
      if (/^[0-9.\-]$/.test(key) || key === ',') {
        const ch = key === ',' ? '.' : key
        if (ch === '-' && f.text.length) return true
        if (ch === '.' && f.text.includes('.')) return true
        f.text += ch
        // inner and outer diameter of a polygon describe the same size: the last one typed wins
        if (f.key === 'od' || f.key === 'id') for (const o of this.fields) if ((o.key === 'od' || o.key === 'id') && o !== f) o.text = ''
        preventDefault(); refreshNow(); return true
      }
      if (key === 'Backspace' && f.text.length) { f.text = f.text.slice(0, -1); preventDefault(); refreshNow(); return true }
      if (key === 'Tab') {
        const nF = this.fields.length
        this.activeField = (this.activeField + (e.shiftKey ? nF - 1 : 1)) % nF
        preventDefault(); this.bump(); return true
      }
      if (key === 'Enter' && this.fields.some((x) => x.text.length)) { preventDefault(); this.commitTyped(); return true }
    }
    if (key === 'Escape') { this.escape(); return true }
    if (key === 'Enter') { this.enter(); return true }
    if (key === 'Delete' || key === 'Backspace') { this.deleteSelection(); return true }
    const k = key.toLowerCase()
    const map: Record<string, ToolId> = { v: 'select', l: 'line', r: 'rect', c: 'circle', a: 'arc', u: 'slot', p: 'polygon', n: 'point', o: 'offset', t: 'trim', e: 'extend', f: 'fillet', h: 'chamfer', i: 'mirror', m: 'move', q: 'rotate', k: 'scale', d: 'dimension' }
    if (map[k]) { this.setTool(map[k]); return true }
    if (k === 'x') { this.toggleConstruction(); return true }
    if (k === 's') { this.snapOn = !this.snapOn; this.say(this.snapOn ? 'Snapping on' : 'Snapping off'); this.bump(); return true }
    return false
  }

  private escape() {
    if (this.fields.some((f) => f.text.length)) { for (const f of this.fields) f.text = ''; this.bump(); return }
    if (this.clicks.length || this.chain || this.picks.length || this.chainPreview || this.phase === 'act' && SELECTION_TOOLS.has(this.tool)) {
      this.resetStage()
      if (SELECTION_TOOLS.has(this.tool)) this.phase = 'pick'
      this.say(this.hintText()); this.bump(); return
    }
    if (this.tool !== 'select') { this.setTool('select'); return }
    if (this.sel.size) { this.setSelection([]); return }
    this.requestClose?.()
  }
  requestClose: (() => void) | undefined

  private enter() {
    const t = this.tool
    if (t === 'line' && (this.chain || this.clicks.length)) { this.chain = undefined; this.clicks = []; this.preview = undefined; this.say(this.hintText()); this.bump(); return }
    if (SELECTION_TOOLS.has(t) && this.phase === 'pick') {
      if (!this.sel.size) return this.say('Select something first')
      this.phase = 'act'; this.say(this.hintText()); this.bump(); return
    }
    if (t === 'rpattern' || t === 'cpattern') this.applyPattern()
  }

  /** Enter on an early stage of the slot / centre-arc tools: typed length, angle or radius place the next click. */
  private advanceTyped(at: Click): boolean {
    const t = this.tool, cs = this.clicks
    if (t === 'slot' && cs.length === 1) {
      const L = this.num('len'), A = this.num('ang')
      if (L === undefined && A === undefined) return false
      const v = sub(at.pt, cs[0].pt)
      const l = L ?? len(v)
      if (l < 1e-9) return false
      this.clicks = [...cs, { pt: polar(cs[0].pt, l, A !== undefined ? A * DEG : angleOf(v)) }]
      this.slotLenTyped = L !== undefined
      return true
    }
    if (t === 'arc' && this.opts.arc === 'center' && cs.length === 1) {
      const R = this.num('r')
      if (R === undefined || R < 1e-9) return false
      this.clicks = [...cs, { pt: polar(cs[0].pt, R, angleOf(sub(at.pt, cs[0].pt))) }]
      this.arcRadiusTyped = true
      return true
    }
    return false
  }

  /** Make a field the active one (clicking it, Tab). */
  focusField(i: number) { if (i >= 0 && i < this.fields.length) { this.activeField = i; this.bump() } }

  /** Enter with typed numbers: commit the current build using them. */
  private commitTyped() {
    const raw = this.cursor
    const sn = this.snappingTool() ? this.snapResult : undefined
    const at: Click = { pt: sn?.pt ?? raw, snap: sn }
    const t = this.tool
    try {
      if (this.advanceTyped(at)) {
        this.fields = []
        this.refresh(raw, { shift: false, ctrl: false, alt: false })
        this.say(this.hintText()); this.bump()
        return
      }
      let b: Built | undefined
      if (t === 'offset' || t === 'move' || t === 'rotate' || t === 'scale') b = this.buildModify(true, raw, at)
      else b = this.build(true, at)
      if (!b) return this.say('Move the pointer to set a direction, then type')
      if (this.commitBuilt(b)) {
        if (t === 'line') {
          const line = b.model.entities.filter((e) => e.kind === 'line').pop()
          if (line?.kind === 'line') {
            const end = this.model.entities.find((x) => x.id === line.b)
            const startId = this.chain?.start ?? line.a
            this.chain = end?.kind === 'point' && line.b !== startId ? { pid: line.b, pt: [end.x, end.y], start: startId } : undefined
            this.clicks = []
          }
        } else if (t === 'offset') { this.chainPreview = undefined }
        else if (t === 'move' || t === 'rotate' || t === 'scale') { this.clicks = []; this.phase = 'pick' }
        else this.clicks = []
      }
    } catch (e) { if (e instanceof OpError) this.say(e.message); else throw e }
    this.fields = []
    this.refresh(raw, { shift: false, ctrl: false, alt: false })
    this.bump()
  }

  applyPattern() {
    try {
      const t = this.tool
      if (t !== 'rpattern' && t !== 'cpattern') return
      if (!this.sel.size) return this.say('Select geometry to pattern first')
      if (t === 'cpattern' && !this.clicks.length) return this.say('Click the centre of the pattern first')
      const b = t === 'cpattern' ? this.build(true, this.clicks[0]) : this.buildModify(true, this.cursor, { pt: this.cursor })
      if (b && this.commitBuilt(b)) { this.clicks = []; this.phase = 'pick'; this.say('Pattern created') }
    } catch (e) { if (e instanceof OpError) this.say(e.message); else throw e }
    this.bump()
  }

  // ------------------------------------------------------------------------------------------------ utility commands
  deleteSelection() {
    if (!this.sel.size) return
    const r = deleteEntities(this.model, [...this.sel])
    this.apply(r.model, { relax: false })
    this.setSelection([])
  }

  toggleConstruction() {
    if (!this.sel.size) return this.say('Select geometry to toggle between normal and construction')
    const r = setConstruction(this.model, [...this.sel])
    this.apply(r.model)
    this.bump()
  }

  deleteConstraint(id: string) {
    const m = this.model
    this.apply({ entities: m.entities, constraints: m.constraints.filter((c) => c.id !== id) }, { strict: false })
  }

  setConstraintValue(id: string, value: number) {
    const m = this.model
    const c = m.constraints.find((x) => x.id === id)
    if (!c || !(value > 0 || c.type === 'hdist' || c.type === 'vdist')) return
    const next = { entities: m.entities, constraints: m.constraints.map((x) => (x.id === id ? { ...x, value } : x)) }
    const r = solve(next.entities, next.constraints, { quick: true })
    if (!r.ok) { this.say('That value conflicts with the other constraints'); this.bump(); return }
    this.host.commit({ entities: r.entities, constraints: next.constraints }, `dim:${id}`)
    this.say(this.statusText())
  }

  // ------------------------------------------------------------------------------------------------ constraints
  private kindOf(id: string): PickedEntity['kind'] | undefined {
    if (this.geo.points.has(id)) return 'point'
    return this.geo.byId.get(id)?.kind
  }

  private constrainClick(raw: Pt, mods: Modifiers) {
    const id = pickEntity(this.geo, raw, this.tol(8), { virtual: true })
    if (!id) { if (!mods.shift) { this.picks = []; this.setSelection([]) } this.bump(); return }
    this.picks = [...this.picks, id]
    this.setSelection(this.picks, true)
    this.tryConstraint()
    this.bump()
  }

  /** Add the constraint if the picks complete it. */
  tryConstraint(): boolean {
    const type = this.opts.constraint
    const picked = this.picks.map((id) => ({ id, kind: this.kindOf(id) })).filter((p): p is PickedEntity => !!p.kind)
    const plan = planConstraint(type, picked, (id) => new Draft(this.model).pointsOf(id))
    if (plan.status === 'more') { this.say(this.hintText()); return false }
    if (plan.status === 'invalid') { this.say(plan.reason); this.picks = []; this.setSelection([], true); return false }
    const d = new Draft(this.model)
    const extra: Partial<Pick<SketchConstraint, 'sign' | 'value'>> = {}
    if (plan.type === 'tangent' && plan.refs.every((r) => this.geo.byId.get(r)?.kind !== 'line')) {
      const [a, b] = plan.refs.map((r) => this.geo.byId.get(r)!) as [any, any]
      const dd = dist(a.c, b.c)
      extra.sign = Math.abs(dd - (a.r + b.r)) <= Math.abs(dd - Math.abs(a.r - b.r)) ? 1 : -1
    }
    if (plan.type === 'fix') for (const ref of plan.refs) d.constrain('fix', [ref], { at: d.pt(ref) })
    else d.constrain(plan.type, plan.refs, extra)
    const ok = this.apply(d.model(), { strict: true })
    if (ok) this.say(`Added ${plan.type}${this.analysis().dof === 0 ? ' · fully constrained' : ''}`)
    this.picks = []
    this.setSelection([], true)
    this.host.selectionChanged([])
    return ok
  }

  // ------------------------------------------------------------------------------------------------ dimensions
  private dimPlan(picks: string[]): { type: ConstraintType; refs: string[]; value: number; sign?: 1 | -1; label: string } | undefined {
    const geo = this.geo
    const kinds = picks.map((p) => this.kindOf(p))
    const pt = (id: string) => geo.point(id)
    if (picks.length === 1) {
      const c = geo.byId.get(picks[0])
      if (kinds[0] === 'line' && c?.kind === 'line' && !c.virtual) return { type: 'distance', refs: [picks[0]], value: dist(c.a, c.b), label: 'Length' }
      if (c && (c.kind === 'circle' || c.kind === 'arc')) {
        const useDia = this.opts.dimRound === 'diameter' || (this.opts.dimRound === 'auto' && c.kind === 'circle')
        return useDia ? { type: 'diameter', refs: picks, value: c.r * 2, label: 'Diameter' } : { type: 'radius', refs: picks, value: c.r, label: 'Radius' }
      }
      return
    }
    if (picks.length !== 2) return
    if (kinds[0] === 'point' && kinds[1] === 'point') {
      const a = pt(picks[0]), b = pt(picks[1])
      if (this.opts.dim === 'horizontal') return { type: 'hdist', refs: picks, value: Math.abs(b[0] - a[0]), sign: b[0] - a[0] < 0 ? -1 : 1, label: 'Horizontal' }
      if (this.opts.dim === 'vertical') return { type: 'vdist', refs: picks, value: Math.abs(b[1] - a[1]), sign: b[1] - a[1] < 0 ? -1 : 1, label: 'Vertical' }
      return { type: 'distance', refs: picks, value: dist(a, b), label: 'Distance' }
    }
    const pi = kinds[0] === 'point' ? 0 : kinds[1] === 'point' ? 1 : -1
    if (pi >= 0 && (kinds[1 - pi] === 'line')) {
      const l = geo.byId.get(picks[1 - pi])
      if (l?.kind !== 'line') return
      const p = pt(picks[pi])
      const n = norm(sub(l.b, l.a))
      return { type: 'distance', refs: [picks[pi], picks[1 - pi]], value: Math.abs(cross(n, sub(p, l.a))), label: 'Distance' }
    }
    if (kinds[0] === 'line' && kinds[1] === 'line') {
      const A = geo.byId.get(picks[0]), B = geo.byId.get(picks[1])
      if (A?.kind !== 'line' || B?.kind !== 'line') return
      const ang = Math.atan2(cross(norm(sub(A.b, A.a)), norm(sub(B.b, B.a))), dot(norm(sub(A.b, A.a)), norm(sub(B.b, B.a))))
      return { type: 'angle', refs: picks, value: Math.abs(ang) / DEG, sign: ang < 0 ? -1 : 1, label: 'Angle' }
    }
  }

  private dimensionClick(raw: Pt, cx: number, cy: number) {
    const id = pickEntity(this.geo, raw, this.tol(8), { virtual: true })
    if (!id) { this.picks = []; this.valueInput = undefined; this.highlight = []; this.bump(); return }
    let picks = [...this.picks, id]
    if (picks.length > 2 || (picks.length === 2 && !this.dimPlan(picks))) picks = [id]
    this.picks = picks
    this.highlight = picks
    const plan = this.dimPlan(picks)
    this.valueInput = undefined
    if (plan) {
      const at = this.host.toScreen(raw)
      const isAngle = plan.type === 'angle'
      this.valueInput = {
        label: plan.label, unit: isAngle ? '°' : 'mm', text: String(rnd(plan.value, isAngle ? 3 : 4)), at: [at[0] + 14, at[1] + 14],
        submit: (v) => {
          const d = new Draft(this.model)
          if (!(v > 0) && !(v === 0 && (plan.type === 'hdist' || plan.type === 'vdist' || plan.type === 'distance'))) return this.say('Enter a positive value')
          d.constrain(plan.type, plan.refs, { value: v, ...(plan.sign ? { sign: plan.sign } : {}) })
          if (this.apply(d.model(), { strict: true })) this.say(this.statusText())
          this.picks = []; this.highlight = []
        },
      }
    }
    void cx; void cy
    this.say(this.hintText()); this.bump()
  }

  private editDimAt(cx: number, cy: number): boolean {
    const r = this.host.canvasOrigin()
    const x = cx - r.left, y = cy - r.top
    const hit = this.dimBoxes.find((b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h)
    if (!hit) return false
    const c = this.model.constraints.find((k) => k.id === hit.id)
    if (!c) return false
    this.editDimension(c.id, [x, y])
    return true
  }

  editDimension(id: string, at?: [number, number]) {
    const c = this.model.constraints.find((k) => k.id === id)
    if (!c || c.value === undefined) return
    const isAngle = c.type === 'angle'
    this.valueInput = {
      label: c.type, unit: isAngle ? '°' : 'mm', text: String(rnd(c.value, 4)), at: (at ?? [200, 120]) as Pt,
      submit: (v) => this.setConstraintValue(id, v),
    }
    this.bump()
  }

  private valueKey(key: string, preventDefault: () => void): boolean {
    const vi = this.valueInput!
    preventDefault()
    if (key === 'Escape') { this.valueInput = undefined; this.picks = []; this.highlight = []; this.bump(); return true }
    if (key === 'Enter') {
      const v = parseFloat(vi.text)
      this.valueInput = undefined
      if (Number.isFinite(v)) vi.submit(v)
      this.bump()
      return true
    }
    // The first keystroke replaces the prefilled value.
    if (key === 'Backspace') { vi.text = vi.touched ? vi.text.slice(0, -1) : ''; vi.touched = true }
    else if (/^[0-9.\-]$/.test(key) || key === ',') {
      if (!vi.touched) vi.text = ''
      vi.touched = true
      vi.text += key === ',' ? '.' : key
    }
    this.bump()
    return true
  }

  /** Change tool options from the toolbar and refresh the live preview. */
  setOption(patch: Partial<ToolOptions>) {
    Object.assign(this.opts, patch)
    const changedShape = 'rect' in patch || 'circle' in patch || 'arc' in patch
    if (changedShape) { const t = this.tool; this.resetStage(); this.tool = t }
    this.refresh(this.cursor, { shift: false, ctrl: false, alt: false })
    this.say(this.hintText())
    this.bump()
  }

  statusText(): string {
    const a = this.analysis()
    if (!a.ok) return 'Over-constrained: conflicting constraints'
    return a.dof === 0 ? 'Fully constrained' : `${a.dof} degree${a.dof === 1 ? '' : 's'} of freedom`
  }
}

const q0 = (c: Click) => c.pt
