import { createEffect, createSignal, onCleanup, onMount } from 'solid-js'
import type { DocStore, SketchEntity, SketchFeature } from '@darkcad/core'
import { nesting, outline, planeFrame } from '@darkcad/features'
import type { Viewport } from './viewport'
import { Icon } from './icons'

export type SketchTool = 'select' | 'rect' | 'circle' | 'polygon'
type Pt = [number, number]

const AXES: Record<string, [string, string]> = { XY: ['X', 'Y'], XZ: ['X', 'Z'], YZ: ['Y', 'Z'] }

function inside(p: Pt, poly: Pt[]): boolean {
  let c = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j]
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) c = !c
  }
  return c
}

function hit(e: SketchEntity, p: Pt, tol: number): boolean {
  if (e.kind === 'circle') return Math.hypot(p[0] - e.x, p[1] - e.y) <= e.r + tol
  return inside(p, outline(e))
}

function moved(e: SketchEntity, dx: number, dy: number): SketchEntity {
  if (e.kind === 'polygon') return { ...e, points: e.points.map(([x, y]) => [x + dx, y + dy] as Pt) }
  return { ...e, x: e.x + dx, y: e.y + dy }
}

/**
 * Sketch editor drawn as a transparent overlay on the 3D viewport. It uses the viewport's own camera
 * (orthographic, looking at the sketch plane), so the model and other sketches stay visible behind it
 * and everything is in the same global coordinates.
 */
export function Sketcher(props: {
  store: DocStore; sketch: SketchFeature; others: SketchFeature[]; viewport: Viewport
  selected: string | undefined; onSelect: (id?: string) => void; onClose: () => void
  tool: SketchTool; setTool: (t: SketchTool) => void
}) {
  let canvas!: HTMLCanvasElement
  const tool = () => props.tool
  const [snap, setSnap] = createSignal(1)
  const [draft, setDraft] = createSignal<SketchEntity | undefined>()
  const [poly, setPoly] = createSignal<Pt[]>([])
  createEffect(() => { tool(); setPoly([]) }) // switching tools abandons a half-drawn polygon
  const [cursor, setCursor] = createSignal<Pt>([0, 0])
  const [tick, setTick] = createSignal(0)
  let start: Pt | undefined
  let dragging: { last: Pt } | undefined
  let panning: { x: number; y: number } | undefined

  const frame = () => planeFrame(props.sketch.plane, props.sketch.offset)
  const entities = () => props.sketch.entities
  const setEntities = (next: SketchEntity[], key?: string) => props.store.update<SketchFeature>(props.sketch.id, { entities: next }, key)
  const nextId = () => { let n = entities().length + 1; while (entities().some((e) => e.id === `e${n}`)) n++; return `e${n}` }
  const P = (f: ReturnType<typeof frame>, u: number, v: number) => props.viewport.planeToScreen(f, u, v)

  /** Pixels per mm on the sketch plane. */
  const scale = () => { const f = frame(), a = P(f, 0, 0), b = P(f, 1, 0); return Math.hypot(b[0] - a[0], b[1] - a[1]) || 1 }

  const toLocal = (ev: { clientX: number; clientY: number }, snapIt = true): Pt => {
    const p = props.viewport.screenToPlane(frame(), ev.clientX, ev.clientY) ?? [0, 0]
    return snapIt && snap() > 0 ? [Math.round(p[0] / snap()) * snap(), Math.round(p[1] / snap()) * snap()] : p
  }

  function draw() {
    const ctx = canvas.getContext('2d')!
    const dpr = devicePixelRatio, w = canvas.clientWidth, h = canvas.clientHeight
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr) }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    const css = getComputedStyle(canvas)
    const col = (n: string) => css.getPropertyValue(n).trim()
    const f = frame(), sc = scale()
    const r = canvas.getBoundingClientRect()
    const corner = (x: number, y: number) => props.viewport.screenToPlane(f, r.left + x, r.top + y) ?? [0, 0]
    const cs = [corner(0, 0), corner(w, 0), corner(w, h), corner(0, h)]
    const u0 = Math.min(...cs.map((c) => c[0])), u1 = Math.max(...cs.map((c) => c[0]))
    const v0 = Math.min(...cs.map((c) => c[1])), v1 = Math.max(...cs.map((c) => c[1]))

    // grid
    let step = 1
    while (step * sc < 12) step *= 5
    while (step * sc > 90) step /= 5
    ctx.lineWidth = 1
    ctx.strokeStyle = col('--sketch-grid')
    ctx.beginPath()
    for (let u = Math.floor(u0 / step) * step; u <= u1; u += step) { const a = P(f, u, v0), b = P(f, u, v1); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]) }
    for (let v = Math.floor(v0 / step) * step; v <= v1; v += step) { const a = P(f, u0, v), b = P(f, u1, v); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]) }
    ctx.stroke()
    // plane axes through the plane origin
    const [ax, ay] = AXES[props.sketch.plane]
    const o = P(f, 0, 0)
    ctx.strokeStyle = col('--axis-x'); ctx.beginPath(); { const a = P(f, u0, 0), b = P(f, u1, 0); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]) } ctx.stroke()
    ctx.strokeStyle = col('--axis-y'); ctx.beginPath(); { const a = P(f, 0, v0), b = P(f, 0, v1); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]) } ctx.stroke()
    ctx.fillStyle = col('--text-dim'); ctx.font = '11px system-ui'
    { const e = P(f, u1, 0), n = P(f, 0, v1); ctx.fillText(ax, e[0] - 14, e[1] - 6); ctx.fillText(ay, n[0] + 6, n[1] + 14) }
    ctx.beginPath(); ctx.arc(o[0], o[1], 3, 0, Math.PI * 2); ctx.fillStyle = col('--text'); ctx.fill()

    // other sketches in the design, dimmed (same global coordinates)
    ctx.strokeStyle = col('--text-faint'); ctx.lineWidth = 1.2
    for (const s of props.others) {
      const sf = planeFrame(s.plane, s.offset)
      for (const e of s.entities) {
        ctx.beginPath()
        outline(e).forEach(([u, v], i) => { const p = P(sf, u, v); i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]) })
        ctx.closePath(); ctx.stroke()
      }
    }

    const path = (e: SketchEntity, ctx2 = ctx) => {
      if (e.kind === 'circle') {
        ctx2.moveTo(P(f, e.x + e.r, e.y)[0], P(f, e.x + e.r, e.y)[1])
        for (let i = 1; i <= 64; i++) { const a = (i / 64) * Math.PI * 2, p = P(f, e.x + e.r * Math.cos(a), e.y + e.r * Math.sin(a)); ctx2.lineTo(p[0], p[1]) }
      } else {
        outline(e).forEach(([u, v], i) => { const p = P(f, u, v); i ? ctx2.lineTo(p[0], p[1]) : ctx2.moveTo(p[0], p[1]) })
        ctx2.closePath()
      }
    }
    const list = entities()
    const depth = nesting(list)
    // Fill each solid region with its direct holes cut out (even-odd), so the model behind shows through holes.
    list.forEach((e, i) => {
      if (depth[i] % 2) return
      ctx.beginPath()
      path(e)
      list.forEach((hole, j) => { if (depth[j] === depth[i] + 1 && inside(outline(hole)[0], outline(e))) { ctx.moveTo(0, 0); path(hole) } })
      ctx.fillStyle = col('--accent-fill')
      ctx.fill('evenodd')
    })
    list.forEach((e) => {
      ctx.beginPath(); path(e)
      const sel = e.id === props.selected
      ctx.lineWidth = sel ? 2.5 : 1.6
      ctx.strokeStyle = sel ? col('--warn') : col('--accent')
      ctx.stroke()
    })
    const d = draft()
    if (d) { ctx.beginPath(); path(d); ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5; ctx.strokeStyle = col('--text'); ctx.stroke(); ctx.setLineDash([]) }
    const pp = poly()
    if (pp.length) {
      ctx.beginPath()
      pp.forEach(([u, v], i) => { const p = P(f, u, v); i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]) })
      const c = P(f, cursor()[0], cursor()[1]); ctx.lineTo(c[0], c[1])
      ctx.setLineDash([5, 4]); ctx.strokeStyle = col('--text'); ctx.stroke(); ctx.setLineDash([])
    }
    const [cx, cy] = cursor()
    ctx.fillStyle = col('--text-dim')
    ctx.fillText(`${ax} ${cx.toFixed(2)}   ${ay} ${cy.toFixed(2)} mm`, 10, h - 10)
  }

  createEffect(() => { void [props.sketch, props.others, props.selected, draft(), poly(), cursor(), tick(), tool()]; draw() })
  onMount(() => {
    const off = props.viewport.subscribe(() => setTick((t) => t + 1))
    const ro = new ResizeObserver(() => setTick((t) => t + 1))
    ro.observe(canvas)
    onCleanup(() => { off(); ro.disconnect() })
  })

  const closePoly = () => {
    const pts = poly()
    if (pts.length >= 3) { const id = nextId(); setEntities([...entities(), { id, kind: 'polygon', points: pts }]); props.onSelect(id) }
    setPoly([])
  }

  function onDown(ev: PointerEvent) {
    canvas.setPointerCapture(ev.pointerId)
    if (ev.button === 1 || ev.button === 2) { panning = { x: ev.clientX, y: ev.clientY }; return }
    const p = toLocal(ev, !ev.altKey)
    const t = tool()
    if (t === 'polygon') {
      const pts = poly()
      if (pts.length >= 3 && Math.hypot(p[0] - pts[0][0], p[1] - pts[0][1]) < 8 / scale()) closePoly()
      else setPoly([...pts, p])
    } else if (t === 'rect' || t === 'circle') start = p
    else {
      const raw = toLocal(ev, false), tol = 4 / scale()
      const found = [...entities()].reverse().find((e) => hit(e, raw, tol))
      props.onSelect(found?.id)
      if (found) dragging = { last: p }
    }
  }

  function onMove(ev: PointerEvent) {
    if (panning) {
      props.viewport.pan(ev.clientX - panning.x, ev.clientY - panning.y)
      panning = { x: ev.clientX, y: ev.clientY }
      return
    }
    const p = toLocal(ev, !ev.altKey)
    setCursor(p)
    if (start && tool() === 'rect') setDraft({ id: 'draft', kind: 'rect', x: Math.min(start[0], p[0]), y: Math.min(start[1], p[1]), w: Math.abs(p[0] - start[0]), h: Math.abs(p[1] - start[1]) })
    if (start && tool() === 'circle') setDraft({ id: 'draft', kind: 'circle', x: start[0], y: start[1], r: Math.hypot(p[0] - start[0], p[1] - start[1]) })
    if (dragging && props.selected) {
      const dx = p[0] - dragging.last[0], dy = p[1] - dragging.last[1]
      if (dx || dy) {
        dragging.last = p
        setEntities(entities().map((e) => (e.id === props.selected ? moved(e, dx, dy) : e)), `drag:${props.sketch.id}`)
      }
    }
  }

  function onUp() {
    panning = undefined
    dragging = undefined
    const d = draft()
    if (d && start) {
      const ok = d.kind === 'rect' ? d.w > 1e-6 && d.h > 1e-6 : d.kind === 'circle' ? d.r > 1e-6 : false
      if (ok) { const id = nextId(); setEntities([...entities(), { ...d, id }]); props.onSelect(id) }
    }
    start = undefined
    setDraft(undefined)
  }

  function onKey(ev: KeyboardEvent) {
    const tag = (ev.target as HTMLElement)?.tagName
    if (tag === 'INPUT' || tag === 'SELECT') return
    if (ev.key === 'Escape') { if (poly().length) setPoly([]); else if (draft()) { start = undefined; setDraft(undefined) } else props.onClose() }
    else if (ev.key === 'Enter') closePoly()
    else if ((ev.key === 'Delete' || ev.key === 'Backspace') && props.selected) { setEntities(entities().filter((e) => e.id !== props.selected)); props.onSelect(undefined) }
    else if (ev.key === 'v') props.setTool('select')
    else if (ev.key === 'r') props.setTool('rect')
    else if (ev.key === 'c') props.setTool('circle')
    else if (ev.key === 'p') props.setTool('polygon')
  }
  onMount(() => { window.addEventListener('keydown', onKey); onCleanup(() => window.removeEventListener('keydown', onKey)) })

  const Pill = (p: { t: SketchTool; label: string; key: string; icon: 'select' | 'rect' | 'circle' | 'polygon' }) => (
    <button class="pill-tool" classList={{ active: tool() === p.t }} title={`${p.label} (${p.key})`} onClick={() => props.setTool(p.t)}>
      <Icon name={p.icon} size={16} /><span>{p.label}</span>
    </button>
  )

  return (
    <div class="sketch-layer">
      <canvas class="sketch-overlay" ref={canvas} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}
        onWheel={(e) => { e.preventDefault(); props.viewport.zoomAt(Math.exp(e.deltaY * 0.0015), e.clientX, e.clientY) }}
        onContextMenu={(e) => e.preventDefault()} onDblClick={() => tool() === 'polygon' && closePoly()} />
      <div class="sketch-pill">
        <span class="sketch-title">Editing <b>{props.sketch.name}</b></span>
        <span class="sep" />
        <Pill t="select" label="Select" key="V" icon="select" />
        <Pill t="rect" label="Rectangle" key="R" icon="rect" />
        <Pill t="circle" label="Circle" key="C" icon="circle" />
        <Pill t="polygon" label="Polygon" key="P" icon="polygon" />
        <span class="sep" />
        <label class="inline">Snap
          <select value={String(snap())} onChange={(e) => setSnap(parseFloat(e.currentTarget.value))}>
            <option value="0">off</option><option value="0.1">0.1</option><option value="0.5">0.5</option><option value="1">1</option><option value="5">5</option>
          </select>
        </label>
        <button class="primary" onClick={props.onClose}>Finish sketch</button>
      </div>
      <div class="sketch-hint">{tool() === 'polygon' ? 'Click points · Enter or click the first point to close' : 'Drag to draw · wheel zoom · right-drag pan · Alt = no snap'}</div>
    </div>
  )
}
