import type { SketchConstraint, SketchEntity } from '@darkcad/core'
import {
  angleOf, buildGeo, cross, dist, formatMm, gridLevels, norm, polar, profiles, sub, tessellate, type Curve, type Geo, type Pt, type Profiles, type SnapResult,
} from '@darkcad/sketch'
import type { Session } from './session'

export interface RenderEnv {
  ctx: CanvasRenderingContext2D
  w: number
  h: number
  /** Plane coordinates → canvas pixels. */
  P: (p: Pt) => [number, number]
  /** Visible plane rectangle. */
  bounds: { u0: number; u1: number; v0: number; v1: number }
  scale: number
  col: (name: string) => string
  axisLabels: [string, string]
  /** Other sketches in the design, as plane polylines already mapped to this plane's canvas. */
  others: { pts: [number, number][]; closed: boolean }[]
}

let profileCache: { e: SketchEntity[]; c: SketchConstraint[]; p: Profiles } | undefined
const profilesOf = (e: SketchEntity[], c: SketchConstraint[]): Profiles => {
  if (profileCache && profileCache.e === e && profileCache.c === c) return profileCache.p
  const p = profiles(e, c)
  profileCache = { e, c, p }
  return p
}

const GLYPH: Partial<Record<SketchConstraint['type'], string>> = {
  horizontal: 'H', vertical: 'V', parallel: '∥', perpendicular: '⟂', equal: '=', tangent: 'T', concentric: '◎', midpoint: 'M', symmetric: '⇋', fix: 'F', 'point-on': '•',
}

export function renderSketch(env: RenderEnv, s: Session) {
  const { ctx, w, h, P, col } = env
  ctx.clearRect(0, 0, w, h)
  const model = s.model
  const geo = s.geo
  const an = s.analysis()
  drawGrid(env)
  drawOthers(env)

  // regions (solid fill with holes, even-odd)
  const pr = profilesOf(model.entities, model.constraints)
  ctx.fillStyle = col('--accent-fill')
  for (const r of pr.regions) {
    ctx.beginPath()
    for (const loop of [r.outer, ...r.holes]) {
      loop.poly.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]) })
      ctx.closePath()
    }
    ctx.fill('evenodd')
  }

  // curves
  const conflictIds = new Set<string>()
  for (const c of model.constraints) if (an.conflicts.has(c.id)) for (const r of c.refs) conflictIds.add(r)
  const hi = new Set(s.highlight)
  const focus = s.focusConstraint ? new Set(model.constraints.find((c) => c.id === s.focusConstraint)?.refs ?? []) : undefined
  const open = new Set(pr.open)
  for (const c of geo.real) {
    const selected = s.sel.has(c.id), hovered = s.hover === c.id || hi.has(c.id) || focus?.has(c.id)
    const constrained = an.ok && an.dof >= 0 && !an.free.has(c.id)
    ctx.lineWidth = selected ? 2.6 : hovered ? 2.4 : 1.7
    ctx.strokeStyle = conflictIds.has(c.id) ? col('--danger') : selected ? col('--warn') : hovered ? '#ffffff' : c.construction ? '#9a9a9a' : constrained ? col('--ok') : open.has(c.id) ? '#ff9ec7' : col('--accent')
    ctx.setLineDash(c.construction ? [6, 4] : [])
    strokeCurve(env, c)
  }
  ctx.setLineDash([])

  // points
  for (const e of model.entities) {
    if (e.kind !== 'point') continue
    const q = P([e.x, e.y])
    const selected = s.sel.has(e.id), hovered = s.hover === e.id || hi.has(e.id) || focus?.has(e.id)
    const fixed = an.ok && an.dof >= 0 && !an.free.has(e.id)
    ctx.fillStyle = selected ? col('--warn') : hovered ? '#ffffff' : fixed ? col('--ok') : e.construction ? '#9a9a9a' : col('--accent')
    ctx.beginPath(); ctx.arc(q[0], q[1], selected || hovered ? 4.2 : 3, 0, Math.PI * 2); ctx.fill()
  }

  if (s.showConstraints) { drawConstraints(env, s, geo, an.conflicts, an.redundant) }
  drawDimensions(env, s, geo, an.conflicts)

  // live preview of the tool in progress
  if (s.preview) {
    const g = buildGeo(s.preview.model.entities)
    const created = new Set(s.preview.created)
    ctx.strokeStyle = col('--text'); ctx.lineWidth = 1.5; ctx.setLineDash([5, 4])
    for (const c of g.real) if (created.has(c.id)) { ctx.strokeStyle = c.construction ? '#9a9a9a' : col('--text'); strokeCurve(env, c) }
    ctx.setLineDash([])
    ctx.fillStyle = col('--text')
    for (const e of s.preview.model.entities) if (e.kind === 'point' && created.has(e.id)) { const q = P([e.x, e.y]); ctx.beginPath(); ctx.arc(q[0], q[1], 2.5, 0, Math.PI * 2); ctx.fill() }
  }
  drawLead(env, s)

  // pending clicks
  ctx.fillStyle = col('--warn')
  for (const c of s.clicks) { const q = P(c.pt); ctx.beginPath(); ctx.arc(q[0], q[1], 3.5, 0, Math.PI * 2); ctx.fill() }
  if (s.chain) { const q = P(s.chain.pt); ctx.beginPath(); ctx.arc(q[0], q[1], 3.5, 0, Math.PI * 2); ctx.fill() }

  // selection box
  if (s.box) {
    const a = P(s.box.a), b = P(s.box.b)
    const crossing = s.box.b[0] < s.box.a[0]
    ctx.lineWidth = 1; ctx.setLineDash(crossing ? [5, 3] : [])
    ctx.strokeStyle = crossing ? col('--ok') : col('--accent'); ctx.fillStyle = crossing ? 'rgba(70,167,88,0.10)' : 'rgba(50,145,255,0.10)'
    ctx.beginPath(); ctx.rect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1])); ctx.fill(); ctx.stroke(); ctx.setLineDash([])
  }

  if (s.snapResult && s.snapResult.kind !== 'none') drawSnap(env, s.snapResult)

  // coordinates and grid step
  const [ax, ay] = env.axisLabels
  const { minor } = gridLevels(env.scale)
  ctx.fillStyle = col('--text-dim'); ctx.font = '11px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
  ctx.fillText(`${ax} ${formatMm(s.cursor[0], minor)}   ${ay} ${formatMm(s.cursor[1], minor)} mm   ·   grid ${minor} mm`, 10, h - 10)

}

function strokeCurve(env: RenderEnv, c: Curve) {
  const { ctx, P } = env
  const pts = tessellate(c, 0.25 / Math.max(env.scale, 1e-6) + 0.002)
  ctx.beginPath()
  pts.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]) })
  if (c.kind === 'circle') ctx.closePath()
  ctx.stroke()
}

function drawGrid(env: RenderEnv) {
  const { ctx, P, bounds, col, scale } = env
  const { minor, major } = gridLevels(scale)
  const { u0, u1, v0, v1 } = bounds
  const lines = (step: number, color: string, width: number) => {
    const nu = (u1 - u0) / step, nv = (v1 - v0) / step
    if (nu > 400 || nv > 400) return
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath()
    for (let u = Math.floor(u0 / step) * step; u <= u1; u += step) { const a = P([u, v0]), b = P([u, v1]); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]) }
    for (let v = Math.floor(v0 / step) * step; v <= v1; v += step) { const a = P([u0, v]), b = P([u1, v]); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]) }
    ctx.stroke()
  }
  lines(minor, col('--sketch-grid'), 1)
  lines(major, 'rgba(255,255,255,0.13)', 1)
  // plane axes
  ctx.lineWidth = 1.2
  ctx.strokeStyle = col('--axis-x'); ctx.beginPath(); { const a = P([u0, 0]), b = P([u1, 0]); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]) } ctx.stroke()
  ctx.strokeStyle = col('--axis-y'); ctx.beginPath(); { const a = P([0, v0]), b = P([0, v1]); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]) } ctx.stroke()
  ctx.fillStyle = col('--text-dim'); ctx.font = '11px system-ui'
  { const e = P([u1, 0]), n = P([0, v1]); ctx.fillText(env.axisLabels[0], e[0] - 14, e[1] - 6); ctx.fillText(env.axisLabels[1], n[0] + 6, n[1] + 14) }
  const o = P([0, 0])
  ctx.fillStyle = col('--text'); ctx.beginPath(); ctx.arc(o[0], o[1], 3, 0, Math.PI * 2); ctx.fill()
}

function drawOthers(env: RenderEnv) {
  const { ctx, col } = env
  ctx.strokeStyle = col('--text-faint'); ctx.lineWidth = 1.2
  for (const o of env.others) {
    ctx.beginPath()
    o.pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
    if (o.closed) ctx.closePath()
    ctx.stroke()
  }
}

// ---------------------------------------------------------------------------------------------------- constraints
function anchorOf(geo: Geo, id: string): { p: Pt; dir?: Pt } | undefined {
  const pt = geo.points.get(id)
  if (pt) return { p: pt }
  const c = geo.byId.get(id)
  if (!c || c.virtual) return
  if (c.kind === 'line') return { p: [(c.a[0] + c.b[0]) / 2, (c.a[1] + c.b[1]) / 2], dir: norm(sub(c.b, c.a)) }
  const mid = c.kind === 'arc' ? (c.a0 + c.a1) / 2 : Math.PI / 4
  return { p: polar(c.c, c.r, mid), dir: [Math.cos(mid), Math.sin(mid)] }
}

function badge(env: RenderEnv, text: string, x: number, y: number, color: string) {
  const { ctx } = env
  ctx.font = '600 10px system-ui'
  const w = Math.max(14, ctx.measureText(text).width + 6)
  ctx.fillStyle = 'rgba(10,10,10,0.82)'; ctx.strokeStyle = color; ctx.lineWidth = 1
  ctx.beginPath(); ctx.roundRect(x - w / 2, y - 7, w, 14, 3); ctx.fill(); ctx.stroke()
  ctx.fillStyle = color; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
  ctx.fillText(text, x, y + 0.5)
  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
}

function drawConstraints(env: RenderEnv, s: Session, geo: Geo, conflicts: Set<string>, redundant: Set<string>) {
  const { ctx, P, col } = env
  const used = new Map<string, number>()
  for (const c of s.model.constraints) {
    const g = GLYPH[c.type]
    const focused = s.focusConstraint === c.id
    const color = conflicts.has(c.id) ? col('--danger') : redundant.has(c.id) ? col('--warn') : focused ? '#ffffff' : col('--text-dim')
    if (c.type === 'coincident') {
      for (const r of c.refs) { const p = geo.points.get(r); if (p) { const q = P(p); ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(q[0], q[1], 6, 0, Math.PI * 2); ctx.stroke() } }
      continue
    }
    if (!g) continue
    const target = c.refs.find((r) => geo.byId.get(r) && !geo.byId.get(r)!.virtual) ?? c.refs[0]
    const a = anchorOf(geo, target)
    if (!a) continue
    const q = P(a.p)
    const n = used.get(target) ?? 0
    used.set(target, n + 1)
    let ox = 0, oy = -16 - n * 15
    if (a.dir) {
      const d = P([a.p[0] + a.dir[0], a.p[1] + a.dir[1]])
      const sx = d[0] - q[0], sy = d[1] - q[1], l = Math.hypot(sx, sy) || 1
      ox = (-sy / l) * (14 + n * 15); oy = (sx / l) * (14 + n * 15)
    }
    badge(env, g, q[0] + ox, q[1] + oy, color)
  }
}

// ---------------------------------------------------------------------------------------------------- dimensions
function arrow(ctx: CanvasRenderingContext2D, from: [number, number], to: [number, number]) {
  const d = Math.atan2(to[1] - from[1], to[0] - from[0])
  for (const a of [d + 2.75, d - 2.75]) { ctx.moveTo(to[0], to[1]); ctx.lineTo(to[0] + Math.cos(a) * 7, to[1] + Math.sin(a) * 7) }
}

function dimText(env: RenderEnv, s: Session, id: string, text: string, x: number, y: number, color: string) {
  const { ctx } = env
  ctx.font = '12px system-ui'
  const w = ctx.measureText(text).width + 8
  ctx.fillStyle = 'rgba(10,10,10,0.85)'
  ctx.fillRect(x - w / 2, y - 9, w, 18)
  ctx.fillStyle = color; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
  ctx.fillText(text, x, y + 0.5)
  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
  s.dimBoxes.push({ id, x: x - w / 2, y: y - 9, w, h: 18 })
}

function drawDimensions(env: RenderEnv, s: Session, geo: Geo, conflicts: Set<string>) {
  const { ctx, P, col } = env
  s.dimBoxes = []
  const OFF = 22
  for (const c of s.model.constraints) {
    if (c.value === undefined) continue
    const color = conflicts.has(c.id) ? col('--danger') : s.focusConstraint === c.id ? '#ffffff' : '#8ec5ff'
    ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.beginPath()
    const label = (v: number, dec = 3) => String(Math.round(v * 10 ** dec) / 10 ** dec)
    let text = '', tx = 0, ty = 0, ok = true
    const pt = (id: string) => geo.points.get(id)
    const seg = (a: [number, number], b: [number, number]) => { ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]) }
    const offsetSeg = (A: [number, number], B: [number, number], off: number) => {
      const dx = B[0] - A[0], dy = B[1] - A[1], l = Math.hypot(dx, dy) || 1
      const nx = (-dy / l) * off, ny = (dx / l) * off
      const A2: [number, number] = [A[0] + nx, A[1] + ny], B2: [number, number] = [B[0] + nx, B[1] + ny]
      seg(A, A2); seg(B, B2); seg(A2, B2); arrow(ctx, B2, A2); arrow(ctx, A2, B2)
      tx = (A2[0] + B2[0]) / 2; ty = (A2[1] + B2[1]) / 2
    }
    if (c.type === 'distance' && c.refs.length === 1) {
      const l = geo.byId.get(c.refs[0])
      if (l?.kind !== 'line') ok = false
      else { offsetSeg(P(l.a), P(l.b), -OFF); text = label(c.value) }
    } else if (c.type === 'distance' && c.refs.length === 2) {
      const a = pt(c.refs[0]), b = pt(c.refs[1])
      if (a && b) { offsetSeg(P(a), P(b), -OFF); text = label(c.value) }
      else {
        const p = pt(c.refs[0]) ?? pt(c.refs[1]), l = geo.byId.get(c.refs[0]) ?? geo.byId.get(c.refs[1])
        if (p && l?.kind === 'line') {
          const n = norm(sub(l.b, l.a)), t = (p[0] - l.a[0]) * n[0] + (p[1] - l.a[1]) * n[1]
          const f: Pt = [l.a[0] + n[0] * t, l.a[1] + n[1] * t]
          const A = P(p), B = P(f)
          seg(A, B); arrow(ctx, B, A); arrow(ctx, A, B)
          tx = (A[0] + B[0]) / 2; ty = (A[1] + B[1]) / 2; text = label(c.value)
        } else ok = false
      }
    } else if (c.type === 'hdist' || c.type === 'vdist') {
      const a = pt(c.refs[0]), b = pt(c.refs[1])
      if (!a || !b) ok = false
      else {
        const A = P(a), B = P(b)
        if (c.type === 'hdist') {
          const y = Math.max(A[1], B[1]) + OFF
          seg(A, [A[0], y]); seg(B, [B[0], y]); seg([A[0], y], [B[0], y])
          arrow(ctx, [A[0], y], [B[0], y]); arrow(ctx, [B[0], y], [A[0], y]); tx = (A[0] + B[0]) / 2; ty = y
        } else {
          const x = Math.max(A[0], B[0]) + OFF
          seg(A, [x, A[1]]); seg(B, [x, B[1]]); seg([x, A[1]], [x, B[1]])
          arrow(ctx, [x, A[1]], [x, B[1]]); arrow(ctx, [x, B[1]], [x, A[1]]); tx = x; ty = (A[1] + B[1]) / 2
        }
        text = label(c.value)
      }
    } else if (c.type === 'radius' || c.type === 'diameter') {
      const cu = geo.byId.get(c.refs[0])
      if (!cu || cu.kind === 'line') ok = false
      else {
        const ang = cu.kind === 'arc' ? (cu.a0 + cu.a1) / 2 : Math.PI / 4
        const edge = polar(cu.c, cu.r, ang), E = P(edge), C = P(cu.c)
        if (c.type === 'diameter') { const E2 = P(polar(cu.c, cu.r, ang + Math.PI)); seg(E2, E); arrow(ctx, E2, E); arrow(ctx, E, E2) }
        else { seg(C, E); arrow(ctx, C, E) }
        const dx = E[0] - C[0], dy = E[1] - C[1], l = Math.hypot(dx, dy) || 1
        tx = E[0] + (dx / l) * 24; ty = E[1] + (dy / l) * 24
        seg(E, [E[0] + (dx / l) * 10, E[1] + (dy / l) * 10])
        text = `${c.type === 'radius' ? 'R' : 'Ø'}${label(c.value)}`
      }
    } else if (c.type === 'angle') {
      const A = geo.byId.get(c.refs[0]), B = geo.byId.get(c.refs[1])
      if (A?.kind !== 'line' || B?.kind !== 'line') ok = false
      else {
        const da = sub(A.b, A.a), db = sub(B.b, B.a)
        const det = cross(da, db)
        if (Math.abs(det) < 1e-12) ok = false
        else {
          const t = cross(sub(B.a, A.a), db) / det
          const X: Pt = [A.a[0] + da[0] * t, A.a[1] + da[1] * t]
          const ua = norm(da), ub = norm(db)
          const a0 = angleOf(ua), a1 = angleOf(ub)
          let sweep = a1 - a0
          while (sweep > Math.PI) sweep -= 2 * Math.PI
          while (sweep < -Math.PI) sweep += 2 * Math.PI
          const R = 38 / Math.max(env.scale, 1e-9)
          const n = 24
          for (let i = 0; i <= n; i++) { const q = P(polar(X, R, a0 + (sweep * i) / n)); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]) }
          const m = P(polar(X, R + 16 / Math.max(env.scale, 1e-9), a0 + sweep / 2))
          tx = m[0]; ty = m[1]; text = `${label(c.value, 2)}°`
        }
      }
    } else continue
    if (!ok) { ctx.stroke(); continue }
    ctx.stroke()
    dimText(env, s, c.id, text, tx, ty, color)
  }
}

// ---------------------------------------------------------------------------------------------------- snap marker
function drawSnap(env: RenderEnv, r: SnapResult) {
  const { ctx, P, col } = env
  ctx.setLineDash([4, 4]); ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1
  for (const g of r.guides) {
    // extend the guide a little past the snapped point so it reads as an alignment line
    const a = P(g.from), b = P(g.to), dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy) || 1
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0] + (dx / l) * 14, b[1] + (dy / l) * 14); ctx.stroke()
  }
  ctx.setLineDash([])
  const [x, y] = P(r.pt)
  ctx.strokeStyle = col('--warn'); ctx.fillStyle = col('--warn'); ctx.lineWidth = 1.6
  ctx.beginPath()
  switch (r.kind) {
    case 'endpoint': ctx.rect(x - 5, y - 5, 10, 10); break
    case 'origin': ctx.arc(x, y, 6, 0, Math.PI * 2); break
    case 'center': ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.moveTo(x - 2, y); ctx.lineTo(x + 2, y); break
    case 'midpoint': ctx.moveTo(x, y - 6); ctx.lineTo(x + 6, y + 5); ctx.lineTo(x - 6, y + 5); ctx.closePath(); break
    case 'intersection': ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6); ctx.moveTo(x + 6, y - 6); ctx.lineTo(x - 6, y + 6); break
    case 'quadrant': ctx.moveTo(x, y - 6); ctx.lineTo(x + 6, y); ctx.lineTo(x, y + 6); ctx.lineTo(x - 6, y); ctx.closePath(); break
    case 'perpendicular': ctx.moveTo(x - 6, y + 6); ctx.lineTo(x - 6, y - 4); ctx.lineTo(x + 6, y - 4); ctx.moveTo(x - 6, y + 6); ctx.lineTo(x + 6, y + 6); break
    case 'tangent': ctx.arc(x, y - 3, 4, 0, Math.PI * 2); ctx.moveTo(x - 7, y + 4); ctx.lineTo(x + 7, y + 4); break
    case 'on-curve': ctx.arc(x, y, 4, 0, Math.PI * 2); break
    default: ctx.moveTo(x - 4, y); ctx.lineTo(x + 4, y); ctx.moveTo(x, y - 4); ctx.lineTo(x, y + 4)
  }
  ctx.stroke()
  if (r.label && r.kind !== 'grid') {
    ctx.font = '11px system-ui'
    const w = ctx.measureText(r.label).width + 8
    ctx.fillStyle = 'rgba(10,10,10,0.85)'; ctx.fillRect(x + 10, y - 22, w, 16)
    ctx.fillStyle = col('--warn'); ctx.fillText(r.label, x + 14, y - 10)
  }
}

/** Distance readout next to the pointer while something is being drawn. */
function drawLead(env: RenderEnv, s: Session) {
  const { ctx, P, col } = env
  const text = s.preview?.readout
  if (!text) return
  const [x, y] = P(s.snapResult?.pt ?? s.cursor)
  ctx.font = '11px ui-monospace, monospace'
  const w = ctx.measureText(text).width + 10
  ctx.fillStyle = 'rgba(10,10,10,0.88)'; ctx.strokeStyle = col('--border-strong'); ctx.lineWidth = 1
  ctx.beginPath(); ctx.roundRect(x + 14, y + 12, w, 18, 4); ctx.fill(); ctx.stroke()
  ctx.fillStyle = col('--text'); ctx.fillText(text, x + 19, y + 25)
}

export const _dist = dist
