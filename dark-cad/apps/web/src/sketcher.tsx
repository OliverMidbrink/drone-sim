import { createEffect, createSignal, onCleanup, onMount } from 'solid-js'
import type { SketchEntity, SketchFeature, Store } from '@darkcad/core'
import { nesting, outline } from '@darkcad/features'

export type SketchTool = 'select' | 'rect' | 'circle' | 'polygon'
type Pt = [number, number]

const AXES: Record<string, [string, string]> = { XY: ['X', 'Y'], XZ: ['X', 'Z'], YZ: ['Y', 'Z'] }

function hit(e: SketchEntity, p: Pt, tol: number): boolean {
  if (e.kind === 'circle') return Math.hypot(p[0] - e.x, p[1] - e.y) <= e.r + tol
  const poly = outline(e)
  let c = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j]
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) c = !c
  }
  return c
}

function moved(e: SketchEntity, dx: number, dy: number): SketchEntity {
  if (e.kind === 'polygon') return { ...e, points: e.points.map(([x, y]) => [x + dx, y + dy] as Pt) }
  return { ...e, x: e.x + dx, y: e.y + dy }
}

export function Sketcher(props: { store: Store; sketch: SketchFeature; selected: string | undefined; onSelect: (id?: string) => void; onClose: () => void; tool: SketchTool; setTool: (t: SketchTool) => void }) {
  let canvas!: HTMLCanvasElement
  const tool = () => props.tool
  const setTool = props.setTool
  const [snap, setSnap] = createSignal(1)
  const [draft, setDraft] = createSignal<SketchEntity | undefined>()
  const [poly, setPoly] = createSignal<Pt[]>([])
  createEffect(() => { tool(); setPoly([]) }) // switching tools abandons a half-drawn polygon
  const [cursor, setCursor] = createSignal<Pt>([0, 0])
  const [view, setView] = createSignal({ cx: 0, cy: 0, scale: 8 })
  let start: Pt | undefined
  let dragging: { last: Pt } | undefined
  let panning: { x: number; y: number } | undefined

  const toWorld = (ev: { clientX: number; clientY: number }, snapIt = true): Pt => {
    const r = canvas.getBoundingClientRect(), v = view()
    let x = (ev.clientX - r.left - r.width / 2) / v.scale + v.cx
    let y = -(ev.clientY - r.top - r.height / 2) / v.scale + v.cy
    if (snapIt && snap() > 0) { x = Math.round(x / snap()) * snap(); y = Math.round(y / snap()) * snap() }
    return [x, y]
  }

  const entities = () => props.sketch.entities
  const setEntities = (next: SketchEntity[], key?: string) => props.store.update<SketchFeature>(props.sketch.id, { entities: next }, key)
  const nextId = () => { let n = entities().length + 1; while (entities().some((e) => e.id === `e${n}`)) n++; return `e${n}` }

  function draw() {
    const ctx = canvas.getContext('2d')!
    const dpr = devicePixelRatio, w = canvas.clientWidth, h = canvas.clientHeight
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) { canvas.width = w * dpr; canvas.height = h * dpr }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const css = getComputedStyle(canvas)
    const col = (n: string) => css.getPropertyValue(n).trim()
    ctx.fillStyle = col('--bg'); ctx.fillRect(0, 0, w, h)
    const v = view()
    const sx = (x: number) => w / 2 + (x - v.cx) * v.scale
    const sy = (y: number) => h / 2 - (y - v.cy) * v.scale
    // grid
    let step = 1
    while (step * v.scale < 12) step *= 5
    while (step * v.scale > 90) step /= 5
    const x0 = Math.floor((v.cx - w / 2 / v.scale) / step) * step, x1 = v.cx + w / 2 / v.scale
    const y0 = Math.floor((v.cy - h / 2 / v.scale) / step) * step, y1 = v.cy + h / 2 / v.scale
    ctx.lineWidth = 1
    ctx.strokeStyle = col('--grid')
    ctx.beginPath()
    for (let x = x0; x <= x1; x += step) { ctx.moveTo(sx(x), 0); ctx.lineTo(sx(x), h) }
    for (let y = y0; y <= y1; y += step) { ctx.moveTo(0, sy(y)); ctx.lineTo(w, sy(y)) }
    ctx.stroke()
    const [ax, ay] = AXES[props.sketch.plane]
    ctx.strokeStyle = col('--axis-x'); ctx.beginPath(); ctx.moveTo(0, sy(0)); ctx.lineTo(w, sy(0)); ctx.stroke()
    ctx.strokeStyle = col('--axis-y'); ctx.beginPath(); ctx.moveTo(sx(0), 0); ctx.lineTo(sx(0), h); ctx.stroke()
    ctx.fillStyle = col('--text-dim'); ctx.font = '11px system-ui'
    ctx.fillText(ax, w - 16, sy(0) - 6); ctx.fillText(ay, sx(0) + 6, 14)

    const path = (e: SketchEntity) => {
      ctx.beginPath()
      if (e.kind === 'circle') ctx.arc(sx(e.x), sy(e.y), e.r * v.scale, 0, Math.PI * 2)
      else outline(e).forEach(([x, y], i) => (i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))), ctx.closePath())
    }
    const list = entities()
    const depth = nesting(list)
    const order = list.map((_, i) => i).sort((a, b) => depth[a] - depth[b])
    for (const i of order) {
      path(list[i])
      ctx.fillStyle = depth[i] % 2 ? col('--bg') : col('--accent-fill')
      ctx.fill()
    }
    list.forEach((e) => {
      path(e)
      const sel = e.id === props.selected
      ctx.lineWidth = sel ? 2.5 : 1.5
      ctx.strokeStyle = sel ? col('--warn') : col('--accent')
      ctx.stroke()
    })
    const d = draft()
    if (d) { path(d); ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5; ctx.strokeStyle = col('--text'); ctx.stroke(); ctx.setLineDash([]) }
    const pp = poly()
    if (pp.length) {
      ctx.beginPath()
      pp.forEach(([x, y], i) => (i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))))
      ctx.lineTo(sx(cursor()[0]), sy(cursor()[1]))
      ctx.setLineDash([5, 4]); ctx.strokeStyle = col('--text'); ctx.stroke(); ctx.setLineDash([])
    }
    // cursor read-out
    const [cx, cy] = cursor()
    ctx.fillStyle = col('--text-dim')
    ctx.fillText(`${ax} ${cx.toFixed(2)}   ${ay} ${cy.toFixed(2)} mm`, 10, h - 10)
  }

  createEffect(() => { void [props.sketch, props.selected, draft(), poly(), cursor(), view(), tool()]; draw() })
  onMount(() => {
    const ro = new ResizeObserver(draw)
    ro.observe(canvas)
    onCleanup(() => ro.disconnect())
  })

  const closePoly = () => {
    const pts = poly()
    if (pts.length >= 3) { const id = nextId(); setEntities([...entities(), { id, kind: 'polygon', points: pts }]); props.onSelect(id) }
    setPoly([])
  }

  function onDown(ev: PointerEvent) {
    canvas.setPointerCapture(ev.pointerId)
    if (ev.button === 1 || ev.button === 2) { panning = { x: ev.clientX, y: ev.clientY }; return }
    const p = toWorld(ev, !ev.altKey)
    const t = tool()
    if (t === 'polygon') {
      const pts = poly()
      if (pts.length >= 3 && Math.hypot(p[0] - pts[0][0], p[1] - pts[0][1]) < 8 / view().scale) closePoly()
      else setPoly([...pts, p])
    } else if (t === 'rect' || t === 'circle') start = p
    else {
      const tol = 4 / view().scale
      const list = entities()
      const smallestFirst = [...list].reverse().find((e) => hit(e, toWorld(ev, false), tol))
      props.onSelect(smallestFirst?.id)
      if (smallestFirst) dragging = { last: p }
    }
  }

  function onMove(ev: PointerEvent) {
    if (panning) {
      const v = view()
      setView({ ...v, cx: v.cx - (ev.clientX - panning.x) / v.scale, cy: v.cy + (ev.clientY - panning.y) / v.scale })
      panning = { x: ev.clientX, y: ev.clientY }
      return
    }
    const p = toWorld(ev, !ev.altKey)
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

  function onWheel(ev: WheelEvent) {
    ev.preventDefault()
    const before = toWorld(ev, false)
    const v = view()
    const scale = Math.min(400, Math.max(0.5, v.scale * Math.exp(-ev.deltaY * 0.0015)))
    setView({ ...v, scale })
    const after = toWorld(ev, false)
    setView({ ...view(), cx: view().cx + before[0] - after[0], cy: view().cy + before[1] - after[1] })
  }

  function onKey(ev: KeyboardEvent) {
    if ((ev.target as HTMLElement)?.tagName === 'INPUT' || (ev.target as HTMLElement)?.tagName === 'SELECT') return
    if (ev.key === 'Escape') { if (poly().length) setPoly([]); else if (draft()) { start = undefined; setDraft(undefined) } else props.onClose() }
    else if (ev.key === 'Enter') closePoly()
    else if ((ev.key === 'Delete' || ev.key === 'Backspace') && props.selected) { setEntities(entities().filter((e) => e.id !== props.selected)); props.onSelect(undefined) }
    else if (ev.key === 'v') setTool('select')
    else if (ev.key === 'r') setTool('rect')
    else if (ev.key === 'c') setTool('circle')
    else if (ev.key === 'p') setTool('polygon')
  }
  onMount(() => { window.addEventListener('keydown', onKey); onCleanup(() => window.removeEventListener('keydown', onKey)) })

  return (
    <div class="sketcher">
      <div class="sketch-bar">
        <span class="sketch-title">Editing <b>{props.sketch.name}</b></span>
        <label class="inline">Snap
          <select value={String(snap())} onChange={(e) => setSnap(parseFloat(e.currentTarget.value))}>
            <option value="0">off</option><option value="0.1">0.1</option><option value="0.5">0.5</option><option value="1">1</option><option value="5">5</option>
          </select>
        </label>
        <span class="hint">{tool() === 'polygon' ? 'click points · Enter or click the first point to close' : 'drag to draw · wheel zoom · right-drag pan · Alt = no snap'}</span>
        <button class="primary" onClick={props.onClose}>Finish sketch</button>
      </div>
      <canvas ref={canvas} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onWheel={onWheel} onContextMenu={(e) => e.preventDefault()} onDblClick={() => tool() === 'polygon' && closePoly()} />
    </div>
  )
}
