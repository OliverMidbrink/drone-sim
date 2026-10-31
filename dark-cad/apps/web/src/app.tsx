import { For, Match, Show, Switch, batch, createMemo, createSignal, onCleanup, onMount } from 'solid-js'
import { Store, parse, serialize, type Feature, type SketchFeature, type ThreadFeature } from '@darkcad/core'
import { listDesignations } from '@darkcad/standards'
import { dxfToSketch, meshesToStl, sketchToDxf } from '@darkcad/io'
import { KernelClient } from './kernel/client'
import type { FaceInfo } from './kernel/protocol'
import { Viewport, type BodyMesh, type Pick } from './viewport'
import { Sketcher } from './sketcher'
import { platform } from './platform'

type Tool = 'select' | 'thread' | 'delete-feature'
const AUTOSAVE = 'darkcad:autosave'
const ICON: Record<Feature['type'], string> = { sketch: '✎', extrude: '⬒', loft: '◭', boolean: '⊕', thread: '⌇', 'delete-feature': '⌫' }
const BODY_TYPES = new Set<Feature['type']>(['extrude', 'loft', 'boolean', 'thread', 'delete-feature'])

function loadInitial(): Store {
  try {
    const raw = localStorage.getItem(AUTOSAVE)
    if (raw) return new Store(parse(raw))
  } catch { /* corrupt or unavailable autosave: start fresh */ }
  return new Store()
}

const v3 = (v: { x: number; y: number; z: number }): [number, number, number] => [v.x, v.y, v.z]

export function App() {
  const store = loadInitial()
  const kernel = new KernelClient()
  let host!: HTMLDivElement
  let viewport: Viewport | undefined

  const [doc, setDoc] = createSignal(store.doc)
  const [selected, setSelected] = createSignal<string[]>([])
  const [tool, setTool] = createSignal<Tool>('select')
  const [designation, setDesignation] = createSignal('G1/4')
  const [bodies, setBodies] = createSignal<BodyMesh[]>([])
  const [errors, setErrors] = createSignal(new Map<string, string>())
  const [status, setStatus] = createSignal('Loading geometry kernel…')
  const [building, setBuilding] = createSignal(false)
  const [editing, setEditing] = createSignal<string | undefined>()
  const [entitySel, setEntitySel] = createSignal<string | undefined>()
  const [faceSel, setFaceSel] = createSignal<{ body: string; face: number }[]>([])
  const [faceInfo, setFaceInfo] = createSignal<FaceInfo | undefined>()
  let firstBuild = true

  const features = () => doc().features
  // The store is not a signal; reading doc() makes these re-evaluate on every commit/undo/redo.
  const canUndo = () => (doc(), store.canUndo)
  const canRedo = () => (doc(), store.canRedo)
  const byId = (id?: string) => features().find((f) => f.id === id)
  const primary = createMemo(() => byId(selected()[selected().length - 1]))
  const sketchBeingEdited = () => { const f = byId(editing()); return f?.type === 'sketch' ? f : undefined }

  // ---- rebuild pipeline: debounce edits, drop stale results ----
  let timer: ReturnType<typeof setTimeout> | undefined
  let ticket = 0
  const schedule = () => {
    clearTimeout(timer)
    timer = setTimeout(async () => {
      const mine = ++ticket
      setBuilding(true)
      try {
        const out = await kernel.build(store.doc)
        if (mine !== ticket) return
        setBodies(out.bodies)
        setErrors(out.errors)
        viewport?.setBodies(out.bodies, firstBuild && out.bodies.length > 0)
        if (out.bodies.length) firstBuild = false
        setStatus(out.errors.size ? `${out.errors.size} feature(s) failed` : `Rebuilt in ${out.ms.toFixed(0)} ms`)
      } catch (e: any) {
        setStatus(`Kernel error: ${e.message}`)
      } finally {
        if (mine === ticket) setBuilding(false)
      }
    }, 80)
  }

  const unsub = store.subscribe(() => {
    setDoc(store.doc)
    try { localStorage.setItem(AUTOSAVE, serialize(store.doc)) } catch { /* private mode / quota: autosave is best effort */ }
    schedule()
  })

  onMount(() => {
    try {
      viewport = new Viewport(host)
    } catch (e: any) {
      setStatus(`3D view unavailable (${String(e.message).split(':').pop()?.trim()}): WebGL is required. Check graphics drivers / hardware acceleration.`)
      kernel.ready.then(schedule)
      return
    }
    // Debug/test hook: lets e2e tests find faces and read results without guessing pixel positions.
    ;(window as any).__darkcad = { viewport, kernel, store, bodies }
    kernel.ready.then(() => { setStatus('Ready'); schedule() })
    const el = viewport.renderer.domElement
    let down: { x: number; y: number } | undefined
    el.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY } })
    el.addEventListener('pointerup', (e) => { if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 4 && e.button === 0) onPick(viewport!.pick(e.clientX, e.clientY)) })
    let raf = 0
    el.addEventListener('pointermove', (e) => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => viewport?.setHover(tool() === 'select' ? undefined : viewport!.pick(e.clientX, e.clientY))) })
    window.addEventListener('keydown', onKey)
    onCleanup(() => { window.removeEventListener('keydown', onKey); viewport?.dispose(); unsub() })
  })

  // ---- helpers to create features ----
  const nameFor = (label: string, type: Feature['type']) => `${label} ${features().filter((f) => f.type === type).length + 1}`
  const select = (...ids: string[]) => setSelected(ids)
  const visibleIds = () => bodies().map((b) => b.id)

  function addSketch() {
    const id = store.nextId('sk')
    store.add({ type: 'sketch', id, name: nameFor('Sketch', 'sketch'), plane: 'XY', offset: 0, entities: [] })
    select(id); setEditing(id); setEntitySel(undefined)
  }

  function addExtrude() {
    const sk = selected().map(byId).find((f) => f?.type === 'sketch')
    if (!sk) return setStatus('Select a sketch in the tree first')
    const id = store.nextId('ex')
    store.add({ type: 'extrude', id, name: nameFor('Extrude', 'extrude'), sketch: sk.id, distance: 10, mode: visibleIds().length ? 'add' : 'new' })
    select(id)
  }

  function addLoft() {
    const sks = selected().map(byId).filter((f): f is SketchFeature => f?.type === 'sketch')
    if (sks.length !== 2) return setStatus('Select exactly two sketches (Ctrl-click) to loft between')
    const id = store.nextId('lf')
    store.add({ type: 'loft', id, name: nameFor('Loft', 'loft'), sketches: [sks[0].id, sks[1].id], mode: visibleIds().length ? 'add' : 'new' })
    select(id)
  }

  function addBoolean() {
    const ids = selected().filter((i) => visibleIds().includes(i))
    if (ids.length !== 2) return setStatus('Select two solid bodies (Ctrl-click) to combine; the first is the target')
    const id = store.nextId('bo')
    store.add({ type: 'boolean', id, name: nameFor('Combine', 'boolean'), op: 'union', target: ids[0], tool: ids[1] })
    select(id)
  }

  async function onPick(p?: Pick) {
    const t = tool()
    if (!p) { viewport?.setSelectedFaces([]); setFaceSel([]); setFaceInfo(undefined); return }
    const face = [{ body: p.body, face: p.face }]
    setFaceSel(face); viewport?.setSelectedFaces(face)
    let info: FaceInfo
    try { info = await kernel.faceInfo(p.body, p.face) } catch (e: any) { return setStatus(e.message) }
    setFaceInfo(info)
    if (t === 'delete-feature') {
      const id = store.nextId('df')
      store.add({ type: 'delete-feature', id, name: nameFor('Delete feature', 'delete-feature'), target: p.body, face: { surface: info.surface, centroid: info.centroid, area: info.area } })
      select(id); setTool('select'); viewport?.setSelectedFaces([]); setFaceSel([])
    } else if (t === 'thread') {
      if (!info.cylinder || !viewport) return setStatus(`That face is ${info.surface.toLowerCase()}; pick a cylindrical face (boss or hole wall)`)
      const c = info.cylinder
      const d = Math.hypot(...c.dir)
      const dir = c.dir.map((x) => x / d) as [number, number, number]
      const along = (q: [number, number, number]) => (q[0] - c.origin[0]) * dir[0] + (q[1] - c.origin[1]) * dir[1] + (q[2] - c.origin[2]) * dir[2]
      const ts = viewport.faceVertices(p.body, p.face).map((q) => along(v3(q)))
      const tMin = Math.min(...ts), tMax = Math.max(...ts)
      const centre = along(v3(viewport.bodyCenter(p.body)))
      const freeEnd = Math.abs(tMin - centre) > Math.abs(tMax - centre) ? tMin : tMax // far from the body's bulk
      const into = freeEnd === tMax ? -1 : 1
      const origin = c.origin.map((o, i) => o + dir[i] * freeEnd) as [number, number, number]
      // Outward-facing wall = boss (male); inward-facing wall = hole (female).
      const radial = [p.point.x - c.origin[0], p.point.y - c.origin[1], p.point.z - c.origin[2]]
      const k = radial[0] * dir[0] + radial[1] * dir[1] + radial[2] * dir[2]
      const perp = radial.map((r, i) => r - k * dir[i])
      const outward = p.normal.x * perp[0] + p.normal.y * perp[1] + p.normal.z * perp[2] > 0
      const id = store.nextId('th')
      store.add({
        type: 'thread', id, name: `${designation()} ${outward ? 'male' : 'female'}`, target: p.body, designation: designation(),
        kind: outward ? 'male' : 'female', length: Math.round((tMax - tMin) * 1000) / 1000, origin, dir: dir.map((x) => x * into) as [number, number, number], existingRadius: c.radius,
      })
      select(id); setTool('select'); viewport.setSelectedFaces([]); setFaceSel([])
    } else {
      setStatus(`${info.surface} face · area ${info.area.toFixed(2)} mm²${info.cylinder ? ` · Ø${(info.cylinder.radius * 2).toFixed(3)} mm` : ''}`)
    }
  }

  // ---- files ----
  const baseName = () => (doc().name || 'part').replace(/[^\w.-]+/g, '_')
  async function save() { await platform.save(`${baseName()}.darkcad`, serialize(store.doc), 'application/json') }
  async function openDoc() {
    const f = await platform.open('.darkcad,.json')
    if (!f) return
    try { store.replace(parse(f.text)); setSelected([]); setEditing(undefined); firstBuild = true } catch (e: any) { setStatus(`Cannot open ${f.name}: ${e.message}`) }
  }
  async function exportStep() {
    try { await platform.save(`${baseName()}.step`, await kernel.step(), 'application/step') } catch (e: any) { setStatus(e.message) }
  }
  async function exportStl() {
    if (!bodies().length) return setStatus('Nothing to export')
    await platform.save(`${baseName()}.stl`, meshesToStl(bodies().map((b) => b.mesh), 'DarkCAD'), 'model/stl')
  }
  async function exportDxf() {
    const sk = (primary()?.type === 'sketch' ? primary() : features().find((f) => f.type === 'sketch')) as SketchFeature | undefined
    if (!sk) return setStatus('Select a sketch to export')
    await platform.save(`${sk.name.replace(/\W+/g, '_')}.dxf`, sketchToDxf(sk.entities), 'image/vnd.dxf')
  }
  async function importDxf() {
    const f = await platform.open('.dxf')
    if (!f) return
    const entities = dxfToSketch(f.text)
    if (!entities.length) return setStatus('No closed profiles found in that DXF')
    const id = store.nextId('sk')
    store.add({ type: 'sketch', id, name: f.name.replace(/\.dxf$/i, ''), plane: 'XY', offset: 0, entities })
    select(id); setStatus(`Imported ${entities.length} profile(s)`)
  }

  function onKey(e: KeyboardEvent) {
    const tag = (e.target as HTMLElement)?.tagName
    if (tag === 'INPUT' || tag === 'SELECT' || editing()) return
    const mod = e.ctrlKey || e.metaKey
    if (mod && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? store.redo() : store.undo() }
    else if (mod && e.key.toLowerCase() === 'y') { e.preventDefault(); store.redo() }
    else if (mod && e.key.toLowerCase() === 's') { e.preventDefault(); save() }
    else if (e.key === 'Escape') { setTool('select'); viewport?.setSelectedFaces([]); setFaceSel([]) }
    else if (e.key === 'Delete' && primary()) { store.remove(primary()!.id); setSelected([]) }
    else if (e.key === 'f') viewport?.fit()
  }

  const Btn = (p: { label: string; onClick: () => void; title?: string; active?: boolean; disabled?: boolean }) => (
    <button class={p.active ? 'active' : ''} title={p.title} disabled={p.disabled} onClick={p.onClick}>{p.label}</button>
  )

  const toolHint = () => tool() === 'thread' ? `Thread ${designation()}: click a cylindrical face (boss or hole wall)` : tool() === 'delete-feature' ? 'Feature delete: click any face of the protrusion to remove' : ''

  return (
    <div class="app">
      <header class="toolbar">
        <span class="brand">Dark<b>CAD</b></span>
        <div class="group">
          <Btn label="New" onClick={() => { if (confirm('Discard the current document?')) { store.replace({ ...store.doc, features: [], name: 'Untitled' }); setSelected([]); setEditing(undefined) } }} />
          <Btn label="Open" onClick={openDoc} />
          <Btn label="Save" title="Ctrl+S" onClick={save} />
        </div>
        <div class="group">
          <Btn label="Undo" title="Ctrl+Z" disabled={!canUndo()} onClick={() => store.undo()} />
          <Btn label="Redo" title="Ctrl+Shift+Z" disabled={!canRedo()} onClick={() => store.redo()} />
        </div>
        <div class="group">
          <Btn label="✎ Sketch" onClick={addSketch} />
          <Btn label="⬒ Extrude" onClick={addExtrude} />
          <Btn label="◭ Loft" onClick={addLoft} />
          <Btn label="⊕ Combine" onClick={addBoolean} />
        </div>
        <div class="group">
          <select title="Thread standard and size" value={designation()} onChange={(e) => setDesignation(e.currentTarget.value)}>
            <For each={listDesignations()}>{(d) => <option value={d}>{d}</option>}</For>
          </select>
          <Btn label="⌇ Thread" active={tool() === 'thread'} onClick={() => setTool(tool() === 'thread' ? 'select' : 'thread')} />
          <Btn label="⌫ Delete feature" active={tool() === 'delete-feature'} onClick={() => setTool(tool() === 'delete-feature' ? 'select' : 'delete-feature')} />
        </div>
        <div class="group">
          <Btn label="Import DXF" onClick={importDxf} />
          <Btn label="DXF" title="Export selected sketch" onClick={exportDxf} />
          <Btn label="STEP" onClick={exportStep} />
          <Btn label="STL" onClick={exportStl} />
          <Btn label="Fit" title="F" onClick={() => viewport?.fit()} />
        </div>
      </header>

      <aside class="tree">
        <h3>Features</h3>
        <input class="docname" value={doc().name} onChange={(e) => store.rename(e.currentTarget.value)} />
        <ul>
          <For each={features()}>{(f) => (
            <li classList={{ sel: selected().includes(f.id), err: errors().has(f.id), body: visibleIds().includes(f.id) }}
              title={errors().get(f.id)}
              onClick={(e) => setSelected(e.ctrlKey || e.metaKey ? (selected().includes(f.id) ? selected().filter((i) => i !== f.id) : [...selected(), f.id]) : [f.id])}
              onDblClick={() => { if (f.type === 'sketch') { setEditing(f.id); setEntitySel(undefined) } }}>
              <span class="icon">{ICON[f.type]}</span>
              <span class="label">{f.name}</span>
              <Show when={errors().has(f.id)}><span class="badge" title={errors().get(f.id)}>!</span></Show>
              <Show when={visibleIds().includes(f.id)}><span class="dot" title="visible solid" /></Show>
            </li>
          )}</For>
        </ul>
        <Show when={!features().length}><p class="empty">Start with <b>✎ Sketch</b>, draw a shape, then <b>⬒ Extrude</b>.</p></Show>
      </aside>

      <main class="stage">
        <div class="viewport" ref={host} />
        <Show when={toolHint()}><div class="tool-hint">{toolHint()} · Esc cancels</div></Show>
        <Show when={sketchBeingEdited()}>{(sk) => (
          <Sketcher store={store} sketch={sk()} selected={entitySel()} onSelect={setEntitySel} onClose={() => setEditing(undefined)} />
        )}</Show>
        <Show when={building()}><div class="busy">building…</div></Show>
      </main>

      <aside class="props">
        <h3>Properties</h3>
        <Show when={primary()} fallback={<p class="empty">Select a feature.</p>}>
          {(f) => <Properties f={f()} store={store} features={features()} entitySel={entitySel()} setEntitySel={setEntitySel} error={errors().get(f().id)} edit={() => setEditing(f().id)} />}
        </Show>
        <Show when={faceInfo() && faceSel().length}>
          <h3>Face</h3>
          <dl class="kv">
            <dt>Surface</dt><dd>{faceInfo()!.surface}</dd>
            <dt>Area</dt><dd>{faceInfo()!.area.toFixed(2)} mm²</dd>
            <Show when={faceInfo()!.cylinder}><dt>Diameter</dt><dd>{(faceInfo()!.cylinder!.radius * 2).toFixed(3)} mm</dd></Show>
          </dl>
        </Show>
      </aside>

      <footer class="status"><span>{status()}</span><span class="dim">{bodies().length} solid(s) · {features().length} feature(s)</span></footer>
    </div>
  )
}

function Num(p: { label: string; value: number; onInput: (v: number) => void; step?: number }) {
  return (
    <label class="field"><span>{p.label}</span>
      <input type="number" step={p.step ?? 1} value={Math.round(p.value * 1e6) / 1e6} onInput={(e) => { const v = parseFloat(e.currentTarget.value); if (Number.isFinite(v)) p.onInput(v) }} />
    </label>
  )
}

function Properties(p: { f: Feature; store: Store; features: Feature[]; entitySel?: string; setEntitySel: (id?: string) => void; error?: string; edit: () => void }) {
  const up = (patch: Record<string, unknown>) => p.store.update(p.f.id, patch as any, `prop:${p.f.id}:${Object.keys(patch).join(',')}`)
  const sketches = () => p.features.filter((f) => f.type === 'sketch')
  const bodies = () => p.features.filter((f) => BODY_TYPES.has(f.type) && f.id !== p.f.id)
  const Pick = (q: { label: string; value: string; options: Feature[]; onChange: (v: string) => void }) => (
    <label class="field"><span>{q.label}</span>
      <select value={q.value} onChange={(e) => q.onChange(e.currentTarget.value)}>
        <For each={q.options}>{(o) => <option value={o.id} selected={o.id === q.value}>{o.name}</option>}</For>
      </select>
    </label>
  )
  const Choice = (q: { label: string; value: string; options: string[]; onChange: (v: string) => void }) => (
    <label class="field"><span>{q.label}</span>
      <select value={q.value} onChange={(e) => q.onChange(e.currentTarget.value)}>
        <For each={q.options}>{(o) => <option value={o} selected={o === q.value}>{o}</option>}</For>
      </select>
    </label>
  )
  return (
    <div class="form">
      <label class="field"><span>Name</span><input value={p.f.name} onChange={(e) => up({ name: e.currentTarget.value })} /></label>
      <Show when={p.error}><div class="error">{p.error}</div></Show>
      <Switch>
        <Match when={p.f.type === 'sketch' && (p.f as SketchFeature)}>{(s) => (<>
          <Choice label="Plane" value={s().plane} options={['XY', 'XZ', 'YZ']} onChange={(v) => up({ plane: v })} />
          <Num label="Offset (mm)" value={s().offset} onInput={(v) => up({ offset: v })} />
          <button onClick={p.edit}>Edit sketch</button>
          <h4>Shapes</h4>
          <For each={s().entities}>{(e) => (
            <div class="entity" classList={{ sel: p.entitySel === e.id }} onClick={() => p.setEntitySel(e.id)}>
              <div class="row"><b>{e.kind}</b><button class="x" title="Delete shape" onClick={(ev) => { ev.stopPropagation(); p.store.update(s().id, { entities: s().entities.filter((x) => x.id !== e.id) }) }}>×</button></div>
              <Show when={e.kind === 'rect' && e}>{(r) => (<>
                <Num label="x" value={r().x} step={0.5} onInput={(v) => p.store.update(s().id, { entities: s().entities.map((x) => (x.id === e.id ? { ...x, x: v } : x)) as any }, `ent:${e.id}:x`)} />
                <Num label="y" value={r().y} step={0.5} onInput={(v) => p.store.update(s().id, { entities: s().entities.map((x) => (x.id === e.id ? { ...x, y: v } : x)) as any }, `ent:${e.id}:y`)} />
                <Num label="width" value={r().w} step={0.5} onInput={(v) => p.store.update(s().id, { entities: s().entities.map((x) => (x.id === e.id ? { ...x, w: Math.max(0.001, v) } : x)) as any }, `ent:${e.id}:w`)} />
                <Num label="height" value={r().h} step={0.5} onInput={(v) => p.store.update(s().id, { entities: s().entities.map((x) => (x.id === e.id ? { ...x, h: Math.max(0.001, v) } : x)) as any }, `ent:${e.id}:h`)} />
              </>)}</Show>
              <Show when={e.kind === 'circle' && e}>{(c) => (<>
                <Num label="center x" value={c().x} step={0.5} onInput={(v) => p.store.update(s().id, { entities: s().entities.map((x) => (x.id === e.id ? { ...x, x: v } : x)) as any }, `ent:${e.id}:x`)} />
                <Num label="center y" value={c().y} step={0.5} onInput={(v) => p.store.update(s().id, { entities: s().entities.map((x) => (x.id === e.id ? { ...x, y: v } : x)) as any }, `ent:${e.id}:y`)} />
                <Num label="diameter" value={c().r * 2} step={0.5} onInput={(v) => p.store.update(s().id, { entities: s().entities.map((x) => (x.id === e.id ? { ...x, r: Math.max(0.0005, v / 2) } : x)) as any }, `ent:${e.id}:r`)} />
              </>)}</Show>
              <Show when={e.kind === 'polygon'}><span class="dim">{(e as any).points.length} points: drag in the sketch to move</span></Show>
            </div>
          )}</For>
        </>)}</Match>
        <Match when={p.f.type === 'extrude' && p.f}>{(e) => { const x = e() as Extract<Feature, { type: 'extrude' }>; return (<>
          <Pick label="Sketch" value={x.sketch} options={sketches()} onChange={(v) => up({ sketch: v })} />
          <Num label="Distance (mm)" value={x.distance} step={0.5} onInput={(v) => up({ distance: v })} />
          <Choice label="Mode" value={x.mode} options={['new', 'add', 'cut']} onChange={(v) => up({ mode: v })} />
        </>) }}</Match>
        <Match when={p.f.type === 'loft' && p.f}>{(e) => { const x = e() as Extract<Feature, { type: 'loft' }>; return (<>
          <Pick label="From" value={x.sketches[0]} options={sketches()} onChange={(v) => up({ sketches: [v, x.sketches[1]] })} />
          <Pick label="To" value={x.sketches[1]} options={sketches()} onChange={(v) => up({ sketches: [x.sketches[0], v] })} />
          <Choice label="Mode" value={x.mode} options={['new', 'add', 'cut']} onChange={(v) => up({ mode: v })} />
        </>) }}</Match>
        <Match when={p.f.type === 'boolean' && p.f}>{(e) => { const x = e() as Extract<Feature, { type: 'boolean' }>; return (<>
          <Choice label="Operation" value={x.op} options={['union', 'cut', 'common']} onChange={(v) => up({ op: v })} />
          <Pick label="Target" value={x.target} options={bodies()} onChange={(v) => up({ target: v })} />
          <Pick label="Tool" value={x.tool} options={bodies()} onChange={(v) => up({ tool: v })} />
        </>) }}</Match>
        <Match when={p.f.type === 'thread' && p.f}>{(e) => { const x = e() as ThreadFeature; return (<>
          <Choice label="Thread" value={x.designation} options={listDesignations()} onChange={(v) => up({ designation: v })} />
          <Choice label="Type" value={x.kind} options={['male', 'female']} onChange={(v) => up({ kind: v })} />
          <Num label="Length (mm)" value={x.length} step={0.5} onInput={(v) => up({ length: Math.max(0.1, v) })} />
          <button onClick={() => up({ origin: x.origin.map((o, i) => o + x.dir[i] * x.length), dir: x.dir.map((d) => -d) })}>Flip direction</button>
          <p class="dim">The cylinder is resized to the table size (male → major Ø, female → tap-drill Ø) before the thread is cut.</p>
        </>) }}</Match>
        <Match when={p.f.type === 'delete-feature'}><p class="dim">Removes the protrusion containing the picked face and heals the surface underneath.</p></Match>
      </Switch>
      <button class="danger" onClick={() => p.store.remove(p.f.id)}>Delete feature</button>
    </div>
  )
}
void batch
