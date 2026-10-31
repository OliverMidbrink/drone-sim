import { createEffect, createMemo, createSignal, For, onCleanup, onMount, Show } from 'solid-js'
import type { ConstraintType, DocStore, Frame, SketchFeature } from '@darkcad/core'
import { buildGeo, profiles, tessellate, type Pt } from '@darkcad/sketch'
import type { Viewport } from '../viewport'
import { Icon, type IconName } from '../icons'
import { renderSketch } from './render'
import { Session, type Host, type ToolId, type ToolOptions } from './session'

const AXES: Record<string, [string, string]> = { XY: ['X', 'Y'], XZ: ['X', 'Z'], YZ: ['Y', 'Z'] }

interface Btn { tool: ToolId; label: string; icon: IconName; key: string }
const DRAW: Btn[] = [
  { tool: 'line', label: 'Line', icon: 'line', key: 'L' },
  { tool: 'rect', label: 'Rectangle', icon: 'rect', key: 'R' },
  { tool: 'circle', label: 'Circle', icon: 'circle', key: 'C' },
  { tool: 'arc', label: 'Arc', icon: 'arc', key: 'A' },
  { tool: 'slot', label: 'Slot', icon: 'slot', key: 'U' },
  { tool: 'polygon', label: 'Polygon', icon: 'polygon', key: 'P' },
  { tool: 'point', label: 'Point', icon: 'point', key: 'N' },
]
const MODIFY: Btn[] = [
  { tool: 'offset', label: 'Offset', icon: 'offset', key: 'O' },
  { tool: 'trim', label: 'Trim', icon: 'trim', key: 'T' },
  { tool: 'extend', label: 'Extend', icon: 'extend', key: 'E' },
  { tool: 'fillet', label: 'Fillet', icon: 'fillet', key: 'F' },
  { tool: 'chamfer', label: 'Chamfer', icon: 'chamfer', key: 'H' },
  { tool: 'mirror', label: 'Mirror', icon: 'mirror', key: 'I' },
  { tool: 'move', label: 'Move/Copy', icon: 'move', key: 'M' },
  { tool: 'rotate', label: 'Rotate', icon: 'rotateCw', key: 'Q' },
  { tool: 'scale', label: 'Scale', icon: 'scale', key: 'K' },
  { tool: 'rpattern', label: 'Rect. pattern', icon: 'rpattern', key: '' },
  { tool: 'cpattern', label: 'Circ. pattern', icon: 'cpattern', key: '' },
]
const CHIPS: [ConstraintType, string, string][] = [
  ['coincident', '◦', 'Coincident: make two points one'],
  ['point-on', '•', 'Point on curve'],
  ['horizontal', 'H', 'Horizontal'],
  ['vertical', 'V', 'Vertical'],
  ['parallel', '∥', 'Parallel'],
  ['perpendicular', '⟂', 'Perpendicular'],
  ['equal', '=', 'Equal length or radius'],
  ['tangent', 'T', 'Tangent'],
  ['concentric', '◎', 'Concentric'],
  ['midpoint', 'M', 'Midpoint'],
  ['symmetric', '⇋', 'Symmetric about a line'],
  ['fix', 'F', 'Fix in place'],
]

const Seg = <T extends string>(p: { value: T; options: [T, string][]; onChange: (v: T) => void }) => (
  <span class="seg">
    <For each={p.options}>{([v, label]) => <button classList={{ on: p.value === v }} onClick={() => p.onChange(v)}>{label}</button>}</For>
  </span>
)

const NumIn = (p: { label: string; value: number; step?: number; min?: number; onInput: (v: number) => void }) => (
  <label class="opt">{p.label}
    <input type="number" step={p.step ?? 1} min={p.min} value={p.value} onInput={(e) => { const v = parseFloat(e.currentTarget.value); if (Number.isFinite(v)) p.onInput(v) }} />
  </label>
)

/**
 * Sketch editor drawn as a transparent overlay on the 3D viewport. It uses the viewport's own camera
 * (orthographic, looking at the sketch plane), so the model and other sketches stay visible behind it.
 * All interaction logic lives in `Session`; this component wires events, the canvas and the toolbar.
 */
export function Sketcher(props: {
  store: DocStore; sketch: SketchFeature; frame: Frame; frameOf: (id: string) => Frame | undefined; others: SketchFeature[]; viewport: Viewport
  selected: string[]; onSelect: (ids: string[]) => void; onClose: () => void
}) {
  let canvas!: HTMLCanvasElement
  const [tick, setTick] = createSignal(0)
  const [listOpen, setListOpen] = createSignal(true)
  const bump = () => setTick((t) => t + 1)
  const frame = () => props.frame
  const P = (f: ReturnType<typeof frame>, u: number, v: number) => props.viewport.planeToScreen(f, u, v)
  const scale = () => { const f = frame(), a = P(f, 0, 0), b = P(f, 1, 0); return Math.hypot(b[0] - a[0], b[1] - a[1]) || 1 }

  const host: Host = {
    model: () => ({ entities: props.sketch.entities, constraints: props.sketch.constraints }),
    commit: (m, key) => props.store.update<SketchFeature>(props.sketch.id, { entities: m.entities, constraints: m.constraints }, key),
    scale,
    toLocal: (cx, cy) => props.viewport.screenToPlane(frame(), cx, cy) ?? [0, 0],
    toScreen: (p) => P(frame(), p[0], p[1]),
    canvasOrigin: () => { const r = canvas.getBoundingClientRect(); return { left: r.left, top: r.top } },
    notify: bump,
    status: () => bump(),
    selectionChanged: (ids) => props.onSelect(ids),
  }
  const session = new Session(host)
  session.requestClose = () => props.onClose()
  const S = () => { tick(); return session }

  // Selection made in the design panels flows into the session.
  createEffect(() => {
    const ids = props.selected
    if (ids.length === session.sel.size && ids.every((i) => session.sel.has(i))) return
    session.setSelection(ids, true)
  })
  // The document changed under us (undo, redo, panel edits): drop dead references and redraw.
  createEffect(() => {
    const live = new Set(props.sketch.entities.map((e) => e.id))
    const stale = [...session.sel].filter((i) => !live.has(i))
    if (stale.length) session.setSelection([...session.sel].filter((i) => live.has(i)), true)
    props.sketch.constraints
    bump()
  })

  const openCurves = createMemo(() => profiles(props.sketch.entities, props.sketch.constraints).open.length)

  function draw() {
    const ctx = canvas.getContext('2d')!
    const dpr = devicePixelRatio, w = canvas.clientWidth, h = canvas.clientHeight
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr) }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const css = getComputedStyle(canvas)
    const col = (n: string) => css.getPropertyValue(n).trim()
    const f = frame()
    const r = canvas.getBoundingClientRect()
    const corner = (x: number, y: number) => props.viewport.screenToPlane(f, r.left + x, r.top + y) ?? [0, 0]
    const cs = [corner(0, 0), corner(w, 0), corner(w, h), corner(0, h)]
    const bounds = { u0: Math.min(...cs.map((c) => c[0])), u1: Math.max(...cs.map((c) => c[0])), v0: Math.min(...cs.map((c) => c[1])), v1: Math.max(...cs.map((c) => c[1])) }
    // Other sketches in the design, dimmed (same global coordinates).
    const others = props.others.flatMap((s) => {
      const sf = props.frameOf(s.id)
      if (!sf) return []
      return buildGeo(s.entities).real.map((c) => ({
        pts: tessellate(c, 0.3 / scale() + 0.002).map(([u, v]) => P(sf, u, v)),
        closed: c.kind === 'circle',
      }))
    })
    renderSketch({ ctx, w, h, P: (p: Pt) => P(f, p[0], p[1]), bounds, scale: scale(), col, axisLabels: AXES[props.sketch.planeRef ? 'XY' : props.sketch.plane], others }, session)
  }

  createEffect(() => { void [tick(), props.sketch, props.others, props.selected]; draw() })
  onMount(() => {
    const off = props.viewport.subscribe(bump)
    const ro = new ResizeObserver(bump)
    ro.observe(canvas)
    onCleanup(() => { off(); ro.disconnect() })
  })

  // ---- pointer
  let panning: { x: number; y: number } | undefined
  const mods = (e: { shiftKey: boolean; ctrlKey: boolean; metaKey: boolean; altKey: boolean }) => ({ shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey, alt: e.altKey })
  function onDown(ev: PointerEvent) {
    canvas.setPointerCapture(ev.pointerId)
    if (ev.button === 1 || ev.button === 2) { panning = { x: ev.clientX, y: ev.clientY }; return }
    if (ev.button !== 0) return
    session.pointerDown(ev.clientX, ev.clientY, mods(ev), ev.detail)
  }
  function onMove(ev: PointerEvent) {
    if (panning) {
      props.viewport.pan(ev.clientX - panning.x, ev.clientY - panning.y)
      panning = { x: ev.clientX, y: ev.clientY }
      return
    }
    session.pointerMove(ev.clientX, ev.clientY, mods(ev), ev.buttons)
  }
  function onUp(ev: PointerEvent) {
    if (panning) { panning = undefined; return }
    session.pointerUp(ev.clientX, ev.clientY, mods(ev))
  }

  // ---- keyboard
  function onKey(ev: KeyboardEvent) {
    const tag = (ev.target as HTMLElement)?.tagName
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
    const mod = ev.ctrlKey || ev.metaKey
    if (mod && ev.key.toLowerCase() === 'z') { ev.preventDefault(); ev.shiftKey ? props.store.redo() : props.store.undo(); return }
    if (mod && ev.key.toLowerCase() === 'y') { ev.preventDefault(); props.store.redo(); return }
    if (session.keyDown(ev, () => ev.preventDefault())) ev.preventDefault()
  }
  onMount(() => { window.addEventListener('keydown', onKey); onCleanup(() => window.removeEventListener('keydown', onKey)) })

  // ---- toolbar
  const ToolBtn = (b: Btn) => (
    <button class="tool sk" classList={{ active: S().tool === b.tool }} title={b.key ? `${b.label} (${b.key})` : b.label} onClick={() => session.setTool(b.tool)}>
      <Icon name={b.icon} size={18} /><span>{b.label}</span>
    </button>
  )
  const opt = <K extends keyof ToolOptions>(k: K) => S().opts[k]
  const set = (patch: Partial<ToolOptions>) => session.setOption(patch)

  const Options = () => (
    <>
      <Show when={S().tool === 'rect'}><Seg value={opt('rect')} options={[['2pt', '2-point'], ['center', 'Center'], ['3pt', '3-point']]} onChange={(v) => set({ rect: v })} /></Show>
      <Show when={S().tool === 'circle'}>
        <Seg value={opt('circle')} options={[['center', 'Center'], ['2pt', '2-point'], ['3pt', '3-point']]} onChange={(v) => set({ circle: v })} />
        <Seg value={opt('circleDim')} options={[['diameter', 'Diameter'], ['radius', 'Radius']]} onChange={(v) => set({ circleDim: v })} />
      </Show>
      <Show when={S().tool === 'arc'}><Seg value={opt('arc')} options={[['3pt', '3-point'], ['center', 'Center'] ]} onChange={(v) => set({ arc: v })} /></Show>
      <Show when={S().tool === 'polygon'}>
        <NumIn label="Sides" value={opt('polySides')} min={3} onInput={(v) => set({ polySides: Math.max(3, Math.min(64, Math.round(v))) })} />
        <Seg value={opt('polyMode')} options={[['inscribed', 'Inscribed'], ['circumscribed', 'Circumscribed']]} onChange={(v) => set({ polyMode: v })} />
      </Show>
      <Show when={S().tool === 'fillet'}><NumIn label="Radius" value={opt('filletR')} step={0.5} min={0.001} onInput={(v) => set({ filletR: v })} /></Show>
      <Show when={S().tool === 'chamfer'}><NumIn label="Distance" value={opt('chamferD')} step={0.5} min={0.001} onInput={(v) => set({ chamferD: v })} /></Show>
      <Show when={S().tool === 'move'}>
        <label class="opt"><input type="checkbox" checked={opt('copy')} onChange={(e) => set({ copy: e.currentTarget.checked })} /> Copy</label>
      </Show>
      <Show when={S().tool === 'rpattern'}>
        <NumIn label="Columns" value={opt('nx')} min={1} onInput={(v) => set({ nx: v })} />
        <NumIn label="Rows" value={opt('ny')} min={1} onInput={(v) => set({ ny: v })} />
        <NumIn label="Spacing X" value={opt('dx')} step={1} onInput={(v) => set({ dx: v })} />
        <NumIn label="Spacing Y" value={opt('dy')} step={1} onInput={(v) => set({ dy: v })} />
        <button class="primary" onClick={() => session.applyPattern()}>Apply</button>
      </Show>
      <Show when={S().tool === 'cpattern'}>
        <NumIn label="Count" value={opt('count')} min={2} onInput={(v) => set({ count: v })} />
        <NumIn label="Angle°" value={opt('total')} step={15} onInput={(v) => set({ total: v })} />
        <button class="primary" onClick={() => session.applyPattern()}>Apply</button>
      </Show>
      <Show when={S().tool === 'dimension'}>
        <Seg value={opt('dim')} options={[['aligned', 'Aligned'], ['horizontal', 'Horizontal'], ['vertical', 'Vertical']]} onChange={(v) => set({ dim: v })} />
        <Seg value={opt('dimRound')} options={[['auto', 'Auto'], ['radius', 'Radius'], ['diameter', 'Diameter']]} onChange={(v) => set({ dimRound: v })} />
      </Show>
    </>
  )
  const hasOptions = () => ['rect', 'circle', 'arc', 'polygon', 'fillet', 'chamfer', 'move', 'rpattern', 'cpattern', 'dimension'].includes(S().tool)

  const status = () => {
    const a = S().analysis()
    const text = S().statusText()
    return { text, cls: !a.ok ? 'bad' : a.dof === 0 ? 'full' : 'free' }
  }
  const at = () => host.toScreen(S().snapResult?.pt ?? S().cursor)

  const glyph = (t: ConstraintType) => CHIPS.find((c) => c[0] === t)?.[1] ?? ''
  const unitOf = (t: ConstraintType) => (t === 'angle' ? '°' : 'mm')

  return (
    <div class="sketch-layer">
      <canvas class="sketch-overlay" ref={canvas} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}
        onWheel={(e) => { e.preventDefault(); props.viewport.zoomAt(Math.exp(e.deltaY * 0.0015), e.clientX, e.clientY) }}
        onContextMenu={(e) => e.preventDefault()} />

      <div class="sketch-top">
        <div class="sketch-bar">
          <section class="sk-group"><div class="sk-tools">
            <button class="tool sk" classList={{ active: S().tool === 'select' }} title="Select (V)" onClick={() => session.setTool('select')}><Icon name="select" size={18} /><span>Select</span></button>
            <For each={DRAW}>{ToolBtn}</For>
          </div><h4>Draw</h4></section>
          <section class="sk-group"><div class="sk-tools"><For each={MODIFY}>{ToolBtn}</For></div><h4>Modify</h4></section>
          <section class="sk-group">
            <div class="sk-tools">
              <button class="tool sk" classList={{ active: S().tool === 'dimension' }} title="Dimension (D)" onClick={() => session.setTool('dimension')}><Icon name="dimension" size={18} /><span>Dimension</span></button>
              <div class="chips">
                <For each={CHIPS}>{([t, g, title]) => (
                  <button class="chip" classList={{ active: S().tool === 'constrain' && S().opts.constraint === t }} title={title} onClick={() => session.setTool('constrain', t)}>{g}</button>
                )}</For>
              </div>
            </div><h4>Constrain</h4>
          </section>
          <section class="sk-group"><div class="sk-tools">
            <button class="tool sk" title="Toggle construction (X)" onClick={() => session.toggleConstruction()}><Icon name="construction" size={18} /><span>Construction</span></button>
            <button class="tool sk" title="Delete selection (Del)" onClick={() => session.deleteSelection()}><Icon name="trash" size={18} /><span>Delete</span></button>
            <button class="tool sk" classList={{ active: S().snapOn }} title="Snapping (S, hold Alt to bypass)" onClick={() => { session.snapOn = !session.snapOn; bump() }}><Icon name="magnet" size={18} /><span>Snap</span></button>
            <button class="tool sk" classList={{ active: S().showConstraints }} title="Show constraint glyphs" onClick={() => { session.showConstraints = !session.showConstraints; bump() }}><Icon name="eye" size={18} /><span>Glyphs</span></button>
          </div><h4>Utility</h4></section>
          <section class="sk-group fin">
            <div class="sk-tools col">
              <span class="sk-status" classList={{ [status().cls]: true }} title="Solver state of this sketch"><Icon name={status().cls === 'full' ? 'lock' : 'constrain'} size={13} /> {status().text}</span>
              <button class="primary" onClick={props.onClose}><Icon name="check" size={13} /> Finish sketch</button>
            </div>
          </section>
        </div>
        <Show when={hasOptions()}><div class="sketch-options"><b>{S().tool}</b><Options /></div></Show>
        <div class="sketch-hint">
          {S().message || S().hintText()}
          <Show when={openCurves() > 0}><span class="warn"> · {openCurves()} open curve(s) are not part of a closed profile and cannot be extruded</span></Show>
        </div>
      </div>

      <Show when={S().fields.length}>
        <div class="sk-fields" style={{ left: `${at()[0] + 16}px`, top: `${at()[1] + 40}px` }}>
          <For each={S().fields}>{(f, i) => (
            <div class="sk-field" classList={{ active: S().activeField === i(), typed: f.text.length > 0 }}
              onPointerDown={(e) => { e.preventDefault(); e.stopPropagation(); session.focusField(i()) }}>
              <span>{f.label}</span>
              <b classList={{ ghost: !f.text.length }}>{f.text.length ? f.text : f.live || '·'}{S().activeField === i() ? <i class="caret" /> : null}</b>
              <em>{f.unit}</em>
            </div>
          )}</For>
          <small>Tab / Shift+Tab switch · Enter apply</small>
        </div>
      </Show>
      <Show when={S().valueInput}>
        <div class="sk-value" style={{ left: `${S().valueInput!.at[0]}px`, top: `${S().valueInput!.at[1]}px` }}>
          <span>{S().valueInput?.label}</span>
          <b>{S().valueInput?.text}<i class="caret" /></b>
          <span>{S().valueInput?.unit}</span>
          <small>Enter</small>
        </div>
      </Show>

      <div class="float sk-list" classList={{ collapsed: !listOpen() }}>
        <div class="float-head" onClick={() => setListOpen(!listOpen())} style="cursor:pointer">
          <span>Constraints · {props.sketch.constraints.length}</span><Icon name={listOpen() ? 'chevDown' : 'chevUp'} size={14} />
        </div>
        <Show when={listOpen()}>
          <div class="sk-rows">
            <For each={props.sketch.constraints}>{(c) => (
              <div class="sk-row" classList={{ bad: S().analysis().conflicts.has(c.id), dup: S().analysis().redundant.has(c.id) }}
                onMouseEnter={() => { session.focusConstraint = c.id; bump() }} onMouseLeave={() => { session.focusConstraint = undefined; bump() }}>
                <span class="g">{glyph(c.type) || (c.value !== undefined ? '⟷' : '')}</span>
                <span class="n">{c.type}</span>
                <Show when={c.value !== undefined}>
                  <input type="number" step="0.1" value={Math.round(c.value! * 1e4) / 1e4} onChange={(e) => { const v = parseFloat(e.currentTarget.value); if (Number.isFinite(v)) session.setConstraintValue(c.id, v) }} />
                  <span class="u">{unitOf(c.type)}</span>
                </Show>
                <button class="x" title="Delete constraint" onClick={() => session.deleteConstraint(c.id)}>×</button>
              </div>
            )}</For>
            <Show when={!props.sketch.constraints.length}><p class="dim">No constraints yet. Snapping to points adds coincident constraints; dimension to drive sizes.</p></Show>
          </div>
        </Show>
      </div>
    </div>
  )
}
