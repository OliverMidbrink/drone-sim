import { For, Match, Show, Switch, createEffect, createMemo, createSignal, onCleanup, onMount } from 'solid-js'
import { parse, serialize, type DocStore, type Feature, type SketchFeature, type ThreadFeature } from '@darkcad/core'
import { Workspace } from './workspace'
import { findSize, listDesignations, standards, threadDims } from '@darkcad/standards'
import { planeFrame } from '@darkcad/features'
import { dxfToSketch, meshesToStl, sketchToDxf } from '@darkcad/io'
import { KernelClient } from './kernel/client'
import type { FaceInfo } from './kernel/protocol'
import { Viewport, type BodyMesh, type Pick } from './viewport'
import { Sketcher, type SketchTool } from './sketcher'
import { ViewCube } from './viewcube'
import { Icon, type IconName } from './icons'
import { platform } from './platform'

type Tool = 'select' | 'thread' | 'delete-feature'
const ICON: Record<Feature['type'], IconName> = { sketch: 'sketch', extrude: 'extrude', loft: 'loft', boolean: 'combine', thread: 'thread', 'delete-feature': 'deleteFeature' }
const BODY_TYPES = new Set<Feature['type']>(['extrude', 'loft', 'boolean', 'thread', 'delete-feature'])

const v3 = (v: { x: number; y: number; z: number }): [number, number, number] => [v.x, v.y, v.z]

export function App() {
  const workspace = new Workspace()
  const store: DocStore = workspace
  const kernel = new KernelClient()
  let host!: HTMLDivElement
  let cubeHost!: HTMLDivElement
  let viewport: Viewport | undefined

  const [doc, setDoc] = createSignal(workspace.doc)
  const [selected, setSelected] = createSignal<string[]>([])
  const [tool, setTool] = createSignal<Tool>('select')
  const [designation, setDesignation] = createSignal('G1/4')
  const [bodies, setBodies] = createSignal<BodyMesh[]>([])
  const [errors, setErrors] = createSignal(new Map<string, string>())
  const [status, setStatus] = createSignal('Loading geometry kernel…')
  const [building, setBuilding] = createSignal(false)
  const [editing, setEditing] = createSignal<string | undefined>()
  const [sketchTool, setSketchTool] = createSignal<SketchTool>('rect')
  const [treeOpen, setTreeOpen] = createSignal(true)
  const [propsOpen, setPropsOpen] = createSignal(true)
  const [renamingRoot, setRenamingRoot] = createSignal(false)
  const [planePicking, setPlanePicking] = createSignal(false)
  const [ortho, setOrthoSig] = createSignal(false)
  const [threadKind, setThreadKind] = createSignal<'auto' | 'male' | 'female'>('auto')
  const [threadLen, setThreadLen] = createSignal('')
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

  // Fires on every edit and every tab/design change. The workspace persists designs itself.
  const [wsTick, setWsTick] = createSignal(0)
  const unsub = workspace.subscribe(() => {
    setDoc(workspace.doc)
    setWsTick((t) => t + 1)
    schedule()
  })
  const tabs = () => (wsTick(), workspace.tabs())
  const activeId = () => (wsTick(), workspace.activeId)
  const library = () => (wsTick(), workspace.library())
  const [galleryOpen, setGalleryOpen] = createSignal(false)
  const [renamingTab, setRenamingTab] = createSignal<string | undefined>()

  onMount(() => {
    try {
      viewport = new Viewport(host)
    } catch (e: any) {
      setStatus(`3D view unavailable (${String(e.message).split(':').pop()?.trim()}): WebGL is required. Check graphics drivers / hardware acceleration.`)
      kernel.ready.then(schedule)
      return
    }
    // Debug/test hook: lets e2e tests find faces and read results without guessing pixel positions.
    const cube = new ViewCube(cubeHost, viewport)
    const offCam = viewport.subscribe(() => setOrthoSig(viewport!.isOrtho))
    ;(window as any).__darkcad = { viewport, kernel, store, bodies }
    kernel.ready.then(() => { setStatus('Ready'); schedule() })
    const el = viewport.renderer.domElement
    let down: { x: number; y: number } | undefined
    el.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY } })
    el.addEventListener('pointerup', (e) => { if (!viewport!.isPlanePicking && !viewport!.isLocked && down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 4 && e.button === 0) onPick(viewport!.pick(e.clientX, e.clientY)) })
    let raf = 0
    el.addEventListener('pointermove', (e) => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => viewport?.setHover(tool() === 'select' ? undefined : viewport!.pick(e.clientX, e.clientY))) })
    window.addEventListener('keydown', onKey)
    onCleanup(() => { window.removeEventListener('keydown', onKey); offCam(); cube.dispose(); viewport?.dispose(); unsub() })
  })

  // ---- helpers to create features ----
  const nameFor = (label: string, type: Feature['type']) => `${label} ${features().filter((f) => f.type === type).length + 1}`
  const select = (...ids: string[]) => setSelected(ids)
  const visibleIds = () => bodies().map((b) => b.id)

  /** Enter sketch mode: orthographic view straight onto the plane, model stays visible. */
  function beginEdit(id: string) {
    const sk = byId(id)
    if (sk?.type !== 'sketch') return
    viewport?.stopPlanePick(); setPlanePicking(false)
    setTool('select'); setEditing(id); setEntitySel(undefined); select(id); setPropsOpen(true)
    viewport?.enterSketch(planeFrame(sk.plane, sk.offset))
  }
  function endEdit() {
    if (!editing()) return
    setEditing(undefined)
    viewport?.exitSketch()
  }
  // Editing the plane/offset in the panel re-aims the camera; entity edits must not.
  const sketchPlaneKey = createMemo(() => { const s = sketchBeingEdited(); return s ? `${s.plane}:${s.offset}` : '' })
  createEffect(() => { const k = sketchPlaneKey(); const s = sketchBeingEdited(); if (k && s && viewport?.isLocked) viewport.alignTo(planeFrame(s.plane, s.offset)) })
  createEffect(() => { if (editing() && !sketchBeingEdited()) endEdit() }) // the sketch was deleted or undone away

  function addSketch() {
    if (editing()) return setStatus('Finish the current sketch first')
    if (planePicking()) { viewport?.stopPlanePick(); setPlanePicking(false); return setStatus('Ready') }
    setTool('select'); setPlanePicking(true)
    setStatus('Select a plane to sketch on')
    viewport?.startPlanePick((plane) => {
      setPlanePicking(false); setStatus('Ready')
      const id = store.nextId('sk')
      store.add({ type: 'sketch', id, name: nameFor('Sketch', 'sketch'), plane, offset: 0, entities: [] })
      beginEdit(id)
    })
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
      const tk = threadKind()
      const kind: 'male' | 'female' = tk === 'auto' ? (outward ? 'male' : 'female') : tk
      const customLen = parseFloat(threadLen()) > 0 ? parseFloat(threadLen()) : undefined
      const id = store.nextId('th')
      store.add({
        type: 'thread', id, name: `${designation()} ${kind}`, target: p.body, designation: designation(),
        kind, length: customLen ?? Math.round((tMax - tMin) * 1000) / 1000, origin, dir: dir.map((x) => x * into) as [number, number, number], existingRadius: c.radius,
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
    try { resetUi(); workspace.create(parse(f.text)) } catch (e: any) { setStatus(`Cannot open ${f.name}: ${e.message}`) }
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
    if (e.key === 'Escape' && galleryOpen()) { setGalleryOpen(false); return }
    if (e.key === 'Escape' && viewport?.isPlanePicking) { viewport.stopPlanePick(); setPlanePicking(false); setStatus('Ready'); return }
    if (tag === 'INPUT' || tag === 'SELECT' || editing()) return
    const mod = e.ctrlKey || e.metaKey
    if (mod && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? store.redo() : store.undo() }
    else if (mod && e.key.toLowerCase() === 'y') { e.preventDefault(); store.redo() }
    else if (mod && e.key.toLowerCase() === 's') { e.preventDefault(); save() }
    else if (e.key === 'Escape') { setTool('select'); viewport?.setSelectedFaces([]); setFaceSel([]) }
    else if (e.key === 'Delete' && primary()) { store.remove(primary()!.id); setSelected([]) }
    else if (e.key === 'f') viewport?.fit()
  }

  // Ribbon button: icon above label (Fusion-style).
  const Tool = (p: { icon: IconName; label: string; onClick: () => void; title?: string; active?: boolean; disabled?: boolean }) => (
    <button class="tool" classList={{ active: p.active }} title={p.title ?? p.label} disabled={p.disabled} onClick={p.onClick}>
      <Icon name={p.icon} size={20} /><span>{p.label}</span>
    </button>
  )
  const Group = (p: { title: string; children: any }) => (
    <section class="ribbon-group"><div class="ribbon-tools">{p.children}</div><h4>{p.title}</h4></section>
  )

  const toolHint = () => planePicking() ? 'Select a plane to sketch on' : ''
  /** Leave sketch / tool modes and clear selection before another design becomes active. */
  function resetUi() {
    viewport?.stopPlanePick(); setPlanePicking(false)
    endEdit(); setTool('select'); setSelected([]); setFaceSel([]); setFaceInfo(undefined); viewport?.setSelectedFaces([])
    firstBuild = true
  }
  const newTab = () => { resetUi(); workspace.create(); setGalleryOpen(false) }
  const switchTab = (id: string) => { if (id === workspace.activeId) return; resetUi(); workspace.activate(id) }
  const closeTab = (id: string) => { if (id === workspace.activeId) resetUi(); workspace.close(id) }
  const openDesign = (id: string) => { resetUi(); workspace.open(id); setGalleryOpen(false) }
  const deleteDesign = (id: string, name: string) => { if (confirm(`Delete "${name}"? This cannot be undone.`)) { if (id === workspace.activeId) resetUi(); workspace.deleteDesign(id) } }
  const ago = (t: number) => { const m = Math.round((Date.now() - t) / 60000); return m < 1 ? 'just now' : m < 60 ? `${m} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} d ago` }

  const toggleTool = (t: Tool) => { if (editing()) return setStatus('Finish the sketch first'); viewport?.stopPlanePick(); setPlanePicking(false); setTool(tool() === t ? 'select' : t); viewport?.setSelectedFaces([]); setFaceSel([]) }
  const sizeInfo = () => { const hit = findSize(designation()); if (!hit) return; const m = threadDims(hit.size.designation, 'male'), f = threadDims(hit.size.designation, 'female'); return { pitch: hit.size.pitch, major: m.rMajor * 2, tap: f.rCylinder * 2, taper: hit.std.taper, std: hit.std.name } }

  const Seg = (p: { value: string; options: [string, string][]; onChange: (v: string) => void }) => (
    <div class="seg"><For each={p.options}>{([v, label]) => <button classList={{ on: p.value === v }} onClick={() => p.onChange(v)}>{label}</button>}</For></div>
  )

  return (
    <div class="app">
      <header class="topbar">
        <span class="brand">DarkCAD</span>
        <div class="designs-wrap">
          <button class="icon-btn" classList={{ on: galleryOpen() }} title="All designs" onClick={() => setGalleryOpen(!galleryOpen())}><Icon name="grid" /></button>
          <Show when={galleryOpen()}>
            <div class="gallery-backdrop" onClick={() => setGalleryOpen(false)} />
            <div class="gallery float">
              <div class="float-head"><span>Designs</span><button class="small-btn" onClick={newTab}><Icon name="plus" size={14} /> New design</button></div>
              <ul>
                <For each={library()}>{(d) => (
                  <li classList={{ current: d.id === activeId() }} onClick={() => openDesign(d.id)}>
                    <span class="gicon"><Icon name="cube" size={16} /></span>
                    <span class="gtext"><b>{d.name}</b><small>{d.features} feature{d.features === 1 ? '' : 's'} · {ago(d.updated)}</small></span>
                    <Show when={tabs().some((t) => t.id === d.id)}><span class="gopen">open</span></Show>
                    <button class="icon-btn small" title="Delete design" onClick={(e) => { e.stopPropagation(); deleteDesign(d.id, d.name) }}><Icon name="deleteFeature" size={14} /></button>
                  </li>
                )}</For>
              </ul>
            </div>
          </Show>
        </div>
        <div class="tabs" role="tablist">
          <For each={tabs()}>{(t) => (
            <div class="tab" role="tab" classList={{ active: t.id === activeId() }} aria-selected={t.id === activeId()} onClick={() => switchTab(t.id)} onDblClick={() => { switchTab(t.id); setRenamingTab(t.id) }}>
              <Show when={renamingTab() !== t.id} fallback={
                <input class="tab-input" value={t.name} ref={(el) => queueMicrotask(() => el.select())} onClick={(e) => e.stopPropagation()}
                  onBlur={(e) => { workspace.rename(e.currentTarget.value.trim() || 'Untitled'); setRenamingTab(undefined) }}
                  onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); if (e.key === 'Escape') setRenamingTab(undefined) }} />
              }><span class="tab-name" title={t.name}>{t.name}</span></Show>
              <button class="tab-x" title="Close tab" onClick={(e) => { e.stopPropagation(); closeTab(t.id) }}><Icon name="close" size={12} /></button>
            </div>
          )}</For>
          <button class="icon-btn small" title="New design" onClick={newTab}><Icon name="plus" size={16} /></button>
        </div>
        <span class="spacer" />
        <button class="icon-btn" title="Open a .darkcad file…" onClick={openDoc}><Icon name="open" /></button>
        <button class="icon-btn" title="Save as a .darkcad file (Ctrl+S)" onClick={save}><Icon name="save" /></button>
        <span class="bar-sep" />
        <button class="icon-btn" title="Undo (Ctrl+Z)" disabled={!canUndo()} onClick={() => store.undo()}><Icon name="undo" /></button>
        <button class="icon-btn" title="Redo (Ctrl+Shift+Z)" disabled={!canRedo()} onClick={() => store.redo()}><Icon name="redo" /></button>
        <button class="icon-btn" title="Fit view (F)" onClick={() => viewport?.fit()}><Icon name="fit" /></button>
      </header>

      <nav class="ribbon">
        <Group title="2D">
          <Tool icon="sketch" label="Create Sketch" title="Create a sketch: pick a plane" active={planePicking() || !!editing()} onClick={addSketch} />
        </Group>
        <Group title="3D">
          <Tool icon="extrude" label="Extrude" onClick={addExtrude} />
          <Tool icon="loft" label="Loft" title="Loft between two sketches (Ctrl-click to select)" onClick={addLoft} />
          <Tool icon="combine" label="Combine" title="Union / cut / common of two solids (Ctrl-click to select)" onClick={addBoolean} />
        </Group>
        <Group title="Modify">
          <Tool icon="thread" label="Thread" title="Thread a cylinder: options open on the right" active={tool() === 'thread'} onClick={() => toggleTool('thread')} />
          <Tool icon="deleteFeature" label="Delete feature" title="Remove a boss, pocket or hole: pick any of its faces" active={tool() === 'delete-feature'} onClick={() => toggleTool('delete-feature')} />
        </Group>
        <Group title="Import / Export">
          <Tool icon="import" label="Import" title="Import a DXF as a new sketch" onClick={importDxf} />
          <span class="sep" />
          <Tool icon="export" label="STEP" title="Export all solids as STEP" onClick={exportStep} />
          <Tool icon="export" label="STL" title="Export all solids as STL" onClick={exportStl} />
          <Tool icon="export" label="DXF" title="Export the selected sketch as DXF" onClick={exportDxf} />
        </Group>
      </nav>

      <main class="stage">
        <div class="viewport" ref={host} />
        <Show when={sketchBeingEdited() && viewport}>{(_) => (
          <Sketcher store={store} viewport={viewport!} sketch={sketchBeingEdited()!} others={features().filter((f): f is SketchFeature => f.type === 'sketch' && f.id !== editing())}
            selected={entitySel()} onSelect={setEntitySel} onClose={endEdit} tool={sketchTool()} setTool={setSketchTool} />
        )}</Show>
        <Show when={toolHint()}><div class="tool-hint">{toolHint()} <kbd>Esc</kbd></div></Show>
        <Show when={building()}><div class="busy">building…</div></Show>

        <div class="float tree" classList={{ collapsed: !treeOpen() }}>
          <div class="tree-root" onClick={() => setTreeOpen(!treeOpen())}>
            <span class="chev" classList={{ open: treeOpen() }}><Icon name="chevron" size={14} /></span>
            <Icon name="folder" size={15} />
            <Show when={!renamingRoot()} fallback={
              <input class="root-name" value={doc().name} ref={(el) => queueMicrotask(() => el.select())} onClick={(e) => e.stopPropagation()}
                onBlur={(e) => { store.rename(e.currentTarget.value.trim() || 'Untitled design'); setRenamingRoot(false) }}
                onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); if (e.key === 'Escape') setRenamingRoot(false) }} />
            }>
              <span class="root-label" title="Double-click to rename" onDblClick={(e) => { e.stopPropagation(); setRenamingRoot(true) }}>{doc().name}</span>
            </Show>
            <span class="count">{features().length}</span>
          </div>
          <Show when={treeOpen()}>
            <ul>
              <For each={features()}>{(f) => (
                <li classList={{ sel: selected().includes(f.id), err: errors().has(f.id), body: visibleIds().includes(f.id) }}
                  title={errors().get(f.id)}
                  onClick={(e) => { setSelected(e.ctrlKey || e.metaKey ? (selected().includes(f.id) ? selected().filter((i) => i !== f.id) : [...selected(), f.id]) : [f.id]); setPropsOpen(true) }}
                  onDblClick={() => { if (f.type === 'sketch' && !editing()) beginEdit(f.id) }}>
                  <span class="icon"><Icon name={ICON[f.type]} size={15} /></span>
                  <span class="label">{f.name}</span>
                  <Show when={errors().has(f.id)}><span class="badge" title={errors().get(f.id)}>!</span></Show>
                  <Show when={visibleIds().includes(f.id)}><span class="dot" title="visible solid" /></Show>
                </li>
              )}</For>
            </ul>
            <Show when={!features().length}><p class="empty">Empty design. <b>Create Sketch</b>, then <b>Extrude</b>.</p></Show>
          </Show>
        </div>

        <div class="viewcube">
          <button class="vc roll-l" title="Roll view 90° counter-clockwise" onClick={() => viewport?.roll(1)} disabled={!!editing()}><Icon name="rotateCcw" size={14} /></button>
          <button class="vc up" title="Look from higher" onClick={() => viewport?.stepView(0, 1)} disabled={!!editing()}><Icon name="chevUp" size={14} /></button>
          <button class="vc roll-r" title="Roll view 90° clockwise" onClick={() => viewport?.roll(-1)} disabled={!!editing()}><Icon name="rotateCw" size={14} /></button>
          <button class="vc left" title="Orbit left" onClick={() => viewport?.stepView(-1, 0)} disabled={!!editing()}><Icon name="chevLeft" size={14} /></button>
          <div class="vc-canvas" ref={cubeHost} />
          <button class="vc right" title="Orbit right" onClick={() => viewport?.stepView(1, 0)} disabled={!!editing()}><Icon name="chevRight" size={14} /></button>
          <button class="vc home" title="Home view" onClick={() => viewport?.home()} disabled={!!editing()}><Icon name="home" size={14} /></button>
          <button class="vc down" title="Look from lower" onClick={() => viewport?.stepView(0, -1)} disabled={!!editing()}><Icon name="chevDown" size={14} /></button>
          <button class="vc proj" title="Switch between perspective and orthographic" onClick={() => viewport?.setOrtho(!ortho())} disabled={!!editing()}>{ortho() ? 'Ortho' : 'Persp'}</button>
        </div>

        <div class="right-stack">
          <Show when={tool() === 'thread'}>
            <div class="float panel">
              <div class="float-head"><span>Thread</span><button class="icon-btn small" title="Close" onClick={() => setTool('select')}><Icon name="close" size={14} /></button></div>
              <p class="dim lead">Click a cylindrical face: a boss becomes a male thread, a hole wall a female thread. The cylinder is resized to the screw size first.</p>
              <div class="form">
                <label class="field"><span>Size</span>
                  <select title="Thread standard and size" value={designation()} onChange={(e) => setDesignation(e.currentTarget.value)}>
                    <For each={standards}>{(st) => (
                      <optgroup label={st.name}><For each={st.sizes}>{(z) => <option value={z.designation} selected={z.designation === designation()}>{z.designation}</option>}</For></optgroup>
                    )}</For>
                  </select>
                </label>
                <label class="field"><span>Type</span>
                  <Seg value={threadKind()} options={[['auto', 'Auto'], ['male', 'Male'], ['female', 'Female']]} onChange={(v) => setThreadKind(v as 'auto' | 'male' | 'female')} />
                </label>
                <label class="field"><span>Length (mm)</span>
                  <input type="number" min="0" step="0.5" placeholder="auto: full face" value={threadLen()} onInput={(e) => setThreadLen(e.currentTarget.value)} />
                </label>
                <Show when={sizeInfo()}>{(i) => (
                  <dl class="kv">
                    <dt>Pitch</dt><dd>{i().pitch} mm</dd>
                    <dt>Major Ø</dt><dd>{i().major.toFixed(3)} mm</dd>
                    <dt>Tap drill Ø</dt><dd>{i().tap.toFixed(2)} mm</dd>
                    <Show when={i().taper}><dt>Taper</dt><dd>1:{Math.round(1 / i().taper!)}</dd></Show>
                  </dl>
                )}</Show>
              </div>
            </div>
          </Show>
          <Show when={tool() === 'delete-feature'}>
            <div class="float panel">
              <div class="float-head"><span>Delete feature</span><button class="icon-btn small" title="Close" onClick={() => setTool('select')}><Icon name="close" size={14} /></button></div>
              <p class="dim lead">Click any face of a boss, rib, pocket or hole. The whole feature is removed and the surface underneath is healed.</p>
            </div>
          </Show>
          <Show when={primary() && propsOpen()}>
            <div class="float panel props">
              <div class="float-head"><span>Properties</span><button class="icon-btn small" title="Close" onClick={() => setPropsOpen(false)}><Icon name="close" size={14} /></button></div>
              <Properties f={primary()!} store={store} features={features()} entitySel={entitySel()} setEntitySel={setEntitySel} error={errors().get(primary()!.id)} edit={() => beginEdit(primary()!.id)} />
              <Show when={faceInfo() && faceSel().length}>
                <h4 class="sub">Picked face</h4>
                <dl class="kv">
                  <dt>Surface</dt><dd>{faceInfo()!.surface}</dd>
                  <dt>Area</dt><dd>{faceInfo()!.area.toFixed(2)} mm²</dd>
                  <Show when={faceInfo()!.cylinder}><dt>Diameter</dt><dd>{(faceInfo()!.cylinder!.radius * 2).toFixed(3)} mm</dd></Show>
                </dl>
              </Show>
            </div>
          </Show>
        </div>
      </main>

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

function Properties(p: { f: Feature; store: DocStore; features: Feature[]; entitySel?: string; setEntitySel: (id?: string) => void; error?: string; edit: () => void }) {
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
