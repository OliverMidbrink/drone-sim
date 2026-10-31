import { For, Match, Show, Switch, createEffect, createMemo, createSignal, on, onCleanup, onMount } from 'solid-js'
import { collectDeps, emptyDocument, parse, worldFrame, resolveFrames, serialize, wouldLoop, type AxisRef, type ComponentFeature, type DesignMode, type DocStore, type Document, type Feature, type Frame, type Plane, type PlaneBase, type PlaneDef, type PlaneFeature, type RevolveFeature, type SketchFeature, type ThreadFeature } from '@darkcad/core'
import { Workspace } from './workspace'
import { findSize, listDesignations, standards, threadDims } from '@darkcad/standards'
import { axisSegment, sketchPreview } from '@darkcad/features'
import { dxfToSketch, meshesToStl, sketchToDxf } from '@darkcad/io'
import { KernelClient } from './kernel/client'
import type { FaceInfo } from './kernel/protocol'
import { Viewport, type AxisHL, type BodyMesh, type Pick, type ProfileHL, type SketchView } from './viewport'
import { Sketcher } from './sketch/Sketcher'
import { buildGeo, deleteEntities, gridLevels, interiorPoint, profiles, regionAt, solve } from '@darkcad/sketch'
import { ViewCube } from './viewcube'
import { Icon, type IconName } from './icons'
import { platform } from './platform'
import { Timeline } from './timeline/Timeline'
import { ICON } from './timeline/info'
import { Browser, NO_ORIGIN, type BodyRow, type OriginVis } from './tree/Browser'
import { DataPanel, type Item } from './data/DataPanel'
import { Versions } from './data/Versions'
import { RemotePersistence } from './persist/remote'

type Tool = 'select' | 'thread' | 'delete-feature'
const BODY_TYPES = new Set<Feature['type']>(['extrude', 'revolve', 'loft', 'boolean', 'thread', 'delete-feature', 'component', 'import-step'])
const SIDEBAR_KEY = 'darkcad:sidebar'
/** 'XY' | 'XZ' | 'YZ' | 'plane:<id>' (what the plane picker returns) as a base for a construction plane. */
const baseOf = (key: string): PlaneBase => key.startsWith('plane:') ? { kind: 'plane', id: key.slice(6) } : { kind: 'world', plane: key as Plane }
const encBase = (b: PlaneBase) => b.kind === 'world' ? `w:${b.plane}` : b.kind === 'plane' ? `p:${b.id}` : `s:${b.id}`
const decBase = (v: string): PlaneBase => v[0] === 'w' ? { kind: 'world', plane: v.slice(2) as Plane } : v[0] === 'p' ? { kind: 'plane', id: v.slice(2) } : { kind: 'sketch', id: v.slice(2) }

/** Which box of a Revolve is waiting for a click in the viewport. */
export interface RevolveCmd { id: string; active: 'profile' | 'axis' | 'none'; isNew: boolean }
const planeFrameOf = (pl: Plane): Frame => worldFrame(pl, 0)
const v3 = (v: { x: number; y: number; z: number }): [number, number, number] => [v.x, v.y, v.z]

export function App() {
  const workspace = new Workspace()
  const store: DocStore = workspace
  const kernel = new KernelClient()
  let host!: HTMLDivElement
  let cubeHost!: HTMLDivElement
  let viewport: Viewport | undefined

  // equals:false: a rollback-marker move re-emits the same document object.
  const [wsTick, setWsTick] = createSignal(0)
  const [doc, setDoc] = createSignal(workspace.doc, { equals: false })
  const [selected, setSelected] = createSignal<string[]>([])
  const [tool, setTool] = createSignal<Tool>('select')
  const [designation, setDesignation] = createSignal('G1/4')
  const [bodies, setBodies] = createSignal<BodyMesh[]>([])
  const [origin, setOrigin] = createSignal<OriginVis>(NO_ORIGIN)
  const [sideOpen, setSideOpen] = createSignal(typeof localStorage !== 'undefined' && localStorage.getItem(SIDEBAR_KEY) === '1')
  const [cwd, setCwd] = createSignal<string | null>(null)
  const [versionsOpen, setVersionsOpen] = createSignal(false)
  const [insertOpen, setInsertOpen] = createSignal(false)
  const [insertAt, setInsertAt] = createSignal({ x: 0, y: 0 })
  const [conflictOpen, setConflictOpen] = createSignal(false)
  const [gizmoMode, setGizmoMode] = createSignal<'translate' | 'rotate'>('translate')
  createEffect(() => { try { localStorage.setItem(SIDEBAR_KEY, sideOpen() ? '1' : '0') } catch { /* private mode */ } })
  const [errors, setErrors] = createSignal(new Map<string, string>())
  const [status, setStatus] = createSignal('Loading geometry kernel…')
  const [building, setBuilding] = createSignal(false)
  const [editing, setEditing] = createSignal<string | undefined>()
  const [propsOpen, setPropsOpen] = createSignal(true)
  const [planePicking, setPlanePicking] = createSignal(false)
  const [ortho, setOrthoSig] = createSignal(false)
  const [threadKind, setThreadKind] = createSignal<'auto' | 'male' | 'female'>('auto')
  const [threadLen, setThreadLen] = createSignal('')
  const [entitySel, setEntitySel] = createSignal<string[]>([])
  const [faceSel, setFaceSel] = createSignal<{ body: string; face: number }[]>([])
  const [faceInfo, setFaceInfo] = createSignal<FaceInfo | undefined>()
  const [revolveCmd, setRevolveCmd] = createSignal<RevolveCmd | undefined>()
  const [hoverPick, setHoverPick] = createSignal<{ profile?: ProfileHL; axis?: AxisHL }>({})
  let firstBuild = true

  const features = () => doc().features
  const mode = () => doc().mode
  const marker = () => (wsTick(), doc(), workspace.marker)
  /** What is built and shown: everything before the rollback marker. */
  const active = createMemo(() => features().slice(0, marker()))
  const frameInfo = createMemo(() => resolveFrames(active()))
  const frameOf = (id: string): Frame | undefined => frameInfo().frames.get(id)
  // The store is not a signal; reading doc() makes these re-evaluate on every commit/undo/redo.
  const canUndo = () => (doc(), store.canUndo)
  const canRedo = () => (doc(), store.canRedo)
  const byId = (id?: string) => features().find((f) => f.id === id)
  const primary = createMemo(() => byId(selected()[selected().length - 1]))
  const sketchBeingEdited = () => { const f = byId(editing()); return f?.type === 'sketch' ? f : undefined }

  // Finished sketches are drawn in the 3D view; the one being edited is drawn by the sketch overlay instead.
  const previews = new WeakMap<SketchFeature, SketchView['preview']>()
  const refreshSketches = () => {
    const list: SketchView[] = active()
      .filter((f): f is SketchFeature => f.type === 'sketch' && !f.suppressed && f.visible !== false && f.id !== editing() && !!frameOf(f.id))
      .map((f) => {
        let preview = previews.get(f)
        if (!preview) { preview = sketchPreview(f); previews.set(f, preview) }
        return { id: f.id, frame: frameOf(f.id)!, preview }
      })
    viewport?.setSketches(list, selected())
  }
  createEffect(refreshSketches)

  // Construction planes the user left visible, plus the origin items switched on in the tree.
  const refreshPlanes = () => {
    const views = active()
      .filter((f): f is PlaneFeature => f.type === 'plane' && !f.suppressed && f.visible !== false && !!frameOf(f.id))
      .map((f) => ({ id: f.id, frame: frameOf(f.id)!, label: f.name, selected: selected().includes(f.id) }))
    const o = origin()
    for (const pl of o.planes) views.push({ id: `origin:${pl}`, frame: planeFrameOf(pl), label: pl, selected: false })
    bodies() // re-fit the plane size when the model changes
    viewport?.setPlaneViews(views, { axes: o.axes, point: o.point })
  }
  createEffect(refreshPlanes)

  // Picked revolve profiles and axis (orange) plus whatever is under the pointer (white).
  const profileCache = new WeakMap<SketchFeature, ReturnType<typeof profiles>>()
  const profilesOf = (sk: SketchFeature) => { let p = profileCache.get(sk); if (!p) { p = profiles(sk.entities, sk.constraints); profileCache.set(sk, p) } return p }
  const sketchMap = () => new Map(active().filter((f): f is SketchFeature => f.type === 'sketch').map((f) => [f.id, f]))
  const revolveSel = createMemo(() => { const f = primary(); return f?.type === 'revolve' ? f : undefined })
  const refreshPick = () => {
    const profs: ProfileHL[] = [], axes: AxisHL[] = []
    const rev = revolveSel()
    if (rev) {
      for (const ref of rev.profiles) {
        const sk = byId(ref.sketch)
        if (sk?.type !== 'sketch') continue
        const fr = frameOf(sk.id)
        if (!fr) continue
        const r = regionAt(profilesOf(sk), ref.pt)
        if (r) profs.push({ frame: fr, outer: r.outer.poly, holes: r.holes.map((h) => h.poly), kind: 'selected' })
      }
      if (rev.axis) { try { const [a, b] = axisSegment(sketchMap(), rev.axis, 150); axes.push({ a, b, kind: 'selected' }) } catch { /* axis gone: the error shows in the tree */ } }
    }
    const h = hoverPick()
    if (revolveCmd()) { if (h.profile) profs.push(h.profile); if (h.axis) axes.push(h.axis) }
    viewport?.setPickHighlights(profs, axes)
  }
  createEffect(refreshPick)

  // ---- rebuild pipeline: debounce edits, drop stale results ----
  let timer: ReturnType<typeof setTimeout> | undefined
  let ticket = 0
  const [built, setBuilt] = createSignal<BodyMesh[]>([])
  const rootType = (root?: string) => features().find((f) => f.id === root)?.type
  /** Bodies that are drawn: hidden bodies and hidden components stay out of the view (and out of STL export). */
  const shown = () => {
    const d = doc(), hidden = new Set(d.hidden)
    return built().filter((b) => {
      const r = b.root ?? b.id
      if (hidden.has(r)) return false
      const f = d.features.find((x) => x.id === r)
      return !(f && (f.type === 'component' || f.type === 'plane') && (f as ComponentFeature).visible === false)
    })
  }
  const pushBodies = (fit = false) => { const v = shown(); setBodies(v); viewport?.setBodies(v, fit) }
  createEffect(on(() => `${doc().hidden.join()}|${features().map((f) => (f as ComponentFeature).visible === false ? f.id : '').join()}`, () => pushBodies(), { defer: true }))

  const schedule = () => {
    clearTimeout(timer)
    timer = setTimeout(async () => {
      const mine = ++ticket
      setBuilding(true)
      try {
        const id = workspace.activeId
        const sliced: Document = { ...store.doc, features: store.doc.features.slice(0, store.marker) }
        // Components pull in other designs; load whatever is missing (it may only exist on the server).
        let deps = collectDeps(id, sliced, (x) => workspace.docOf(x))
        for (let i = 0; i < 8 && deps.missing.length; i++) {
          await Promise.all(deps.missing.map((x) => workspace.ensureDoc(x)))
          deps = collectDeps(id, sliced, (x) => workspace.docOf(x))
        }
        const assets: Record<string, Uint8Array> = {}
        await Promise.all(deps.assets.map(async (a) => { try { assets[a] = await workspace.remote!.assetBytes(a) } catch { /* the import reports the missing asset */ } }))
        const out = await kernel.build(sliced, { designId: id, deps: deps.docs, assets })
        if (mine !== ticket) return
        const errs = new Map(out.errors)
        for (const [fid, msg] of frameInfo().errors) if (!errs.has(fid)) errs.set(fid, msg)
        setBuilt(out.bodies)
        setErrors(errs)
        pushBodies(firstBuild && out.bodies.length > 0)
        if (out.bodies.length) firstBuild = false
        setStatus(errs.size ? `${errs.size} feature(s) failed` : `Rebuilt in ${out.ms.toFixed(0)} ms`)
        queueThumb(id)
      } catch (e: any) {
        setStatus(`Kernel error: ${e.message}`)
      } finally {
        if (mine === ticket) setBuilding(false)
      }
    }, 80)
  }

  // ---- design thumbnails for the data panel: captured shortly after a build settles ----
  const lastThumb = new Map<string, string>()
  let thumbTimer: ReturnType<typeof setTimeout> | undefined
  function queueThumb(id: string, tries = 0) {
    clearTimeout(thumbTimer)
    if (!workspace.remote?.project) return
    thumbTimer = setTimeout(async () => {
      if (id !== workspace.activeId || !viewport || editing()) return
      if (!shown().length) return
      if (workspace.saveState !== 'saved') { if (tries < 6) queueThumb(id, tries + 1); return } // the server needs to know the design first
      const url = viewport.snapshot(320, 240)
      if (!url || lastThumb.get(id) === url) return
      try { await workspace.remote!.putThumb(id, await (await fetch(url)).blob()); lastThumb.set(id, url) } catch { /* thumbnails are cosmetic */ }
    }, 2500)
  }

  // Fires on every edit and every tab/design change. The workspace persists designs itself.
  // Only edits to the feature list, the marker or another design need a rebuild; names, visibility and the like do not.
  let lastKey: { features?: Feature[]; marker?: number; id?: string } = {}
  const unsub = workspace.subscribe(() => {
    setDoc(workspace.doc)
    setWsTick((t) => t + 1)
    const k = { features: workspace.doc.features, marker: workspace.marker, id: workspace.activeId }
    if (k.features !== lastKey.features || k.marker !== lastKey.marker || k.id !== lastKey.id) { lastKey = k; schedule() }
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
    ;(window as any).__darkcad = { viewport, kernel, store, workspace, bodies, gridLevels }
    kernel.ready.then(() => { setStatus('Ready'); schedule() })
    refreshSketches(); refreshPick(); refreshPlanes()
    // Server storage: only where there is a server to talk to (the Electron build keeps files on disk itself).
    if (typeof location !== 'undefined' && /^https?:$/.test(location.protocol)) {
      void workspace.connect(new RemotePersistence()).then(() => setWsTick((t) => t + 1))
      const refocus = () => { if (document.visibilityState === 'visible') void workspace.maybePull() }
      document.addEventListener('visibilitychange', refocus)
      onCleanup(() => document.removeEventListener('visibilitychange', refocus))
    }
    const el = viewport.renderer.domElement
    let down: { x: number; y: number } | undefined
    el.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY } })
    el.addEventListener('pointerup', (e) => { if (!viewport!.isPlanePicking && !viewport!.isLocked && down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 4 && e.button === 0) { if (revolveCmd()) onRevolvePick(e.clientX, e.clientY); else onPick(viewport!.pick(e.clientX, e.clientY)) } })
    let raf = 0
    el.addEventListener('pointermove', (e) => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => {
        if (revolveCmd()) return onRevolveHover(e.clientX, e.clientY)
        viewport?.setHover(tool() === 'select' ? undefined : viewport!.pick(e.clientX, e.clientY))
      })
    })
    el.addEventListener('dragover', (e) => { if (e.dataTransfer?.types.includes('application/x-darkcad-design')) e.preventDefault() })
    el.addEventListener('drop', (e) => {
      const id = e.dataTransfer?.getData('application/x-darkcad-design')
      if (!id) return
      e.preventDefault()
      void insertComponent(id, viewport!.groundPoint(e.clientX, e.clientY))
    })
    window.addEventListener('keydown', onKey)
    onCleanup(() => { window.removeEventListener('keydown', onKey); offCam(); cube.dispose(); viewport?.dispose(); unsub() })
  })

  // ---- helpers to create features ----
  const nameFor = (label: string, type: Feature['type']) => `${label} ${features().filter((f) => f.type === type).length + 1}`
  const select = (...ids: string[]) => setSelected(ids)
  const visibleIds = () => bodies().map((b) => b.id)
  /** Solid bodies of the design itself (components do not count: an Extrude does not merge into them). */
  const partCount = () => shown().filter((b) => rootType(b.root ?? b.id) !== 'component').length
  const designName = (id: string) => workspace.library().find((d) => d.id === id)?.name ?? 'missing design'

  /** Enter sketch mode: orthographic view straight onto the plane, model stays visible. */
  function beginEdit(id: string) {
    const sk = byId(id)
    if (sk?.type !== 'sketch') return
    if (features().indexOf(sk) >= marker()) store.setMarker(features().indexOf(sk) + 1) // a rolled-back sketch comes back to be edited
    const fr = frameOf(id)
    if (!fr) return setStatus(`Cannot edit "${sk.name}": ${errors().get(id) ?? frameInfo().errors.get(id) ?? 'its plane is missing'}`)
    viewport?.stopPlanePick(); setPlanePicking(false)
    setTool('select'); setEditing(id); setEntitySel([]); select(id); setPropsOpen(true)
    viewport?.enterSketch(fr)
  }
  function endEdit() {
    if (!editing()) return
    setEditing(undefined)
    viewport?.exitSketch()
  }
  // Editing the plane/offset in the panel re-aims the camera; entity edits must not.
  const sketchPlaneKey = createMemo(() => { const s = sketchBeingEdited(); const f = s && frameOf(s.id); return f ? JSON.stringify(f) : '' })
  createEffect(on(sketchPlaneKey, (k) => { const s = sketchBeingEdited(); const f = s && frameOf(s.id); if (k && f && viewport?.isLocked) viewport.alignTo(f) }, { defer: true }))
  createEffect(() => { if (editing() && !sketchBeingEdited()) endEdit() }) // the sketch was deleted or undone away

  function addSketch() {
    if (editing()) return setStatus('Finish the current sketch first')
    if (planePicking()) { viewport?.stopPlanePick(); setPlanePicking(false); return setStatus('Ready') }
    setTool('select'); setPlanePicking(true)
    setStatus('Select a plane to sketch on')
    viewport?.startPlanePick((key) => {
      setPlanePicking(false); setStatus('Ready')
      const id = store.nextId('sk')
      const ref = key.startsWith('plane:') ? key.slice(6) : undefined
      store.add({ type: 'sketch', id, name: nameFor('Sketch', 'sketch'), plane: ref ? 'XY' : (key as Plane), offset: 0, planeRef: ref, entities: [], constraints: [] })
      beginEdit(id)
    }, planePickList())
  }
  /** Construction planes that can be picked (only those before the rollback marker have a frame). */
  const planePickList = () => active().filter((f): f is PlaneFeature => f.type === 'plane' && !!frameOf(f.id)).map((f) => ({ id: f.id, label: f.name, frame: frameOf(f.id)! }))

  /** Construction plane: pick one base plane (or two for a midplane) in the view, then adjust the numbers in the panel. */
  function addPlane(kind: PlaneDef['kind']) {
    if (editing()) return setStatus('Finish the current sketch first')
    if (planePicking()) { viewport?.stopPlanePick(); setPlanePicking(false); return setStatus('Ready') }
    setTool('select'); setPlanePicking(true)
    const finish = (def: PlaneDef) => {
      setPlanePicking(false); setStatus('Ready')
      const id = store.nextId('pl')
      store.add({ type: 'plane', id, name: nameFor('Plane', 'plane'), def })
      select(id); setPropsOpen(true)
    }
    const pickOne = (msg: string, cb: (key: string) => void) => { setStatus(msg); viewport?.startPlanePick(cb, planePickList()) }
    if (kind === 'mid') pickOne('Midplane: select the first plane', (a) => pickOne('Midplane: select the second (parallel) plane', (b) => finish({ kind: 'mid', a: baseOf(a), b: baseOf(b) })))
    else pickOne(kind === 'offset' ? 'Offset plane: select the plane to offset' : 'Plane at angle: select the plane to rotate', (key) =>
      finish(kind === 'offset' ? { kind: 'offset', base: baseOf(key), distance: 10 } : { kind: 'angle', base: baseOf(key), axis: 'X', angle: 45 }))
  }

  // ---- components ----
  async function insertComponent(ref: string, at?: [number, number, number]) {
    if (editing()) return setStatus('Finish the sketch first')
    const id = workspace.activeId
    if (!(await workspace.ensureDoc(ref))) return setStatus('That design is not available')
    if (wouldLoop(id, ref, (x) => workspace.docOf(x))) return setStatus(`Cannot insert "${designName(ref)}": it already contains this design`)
    if (mode() === 'part') store.setMode('hybrid') // inserting a component turns a part into a hybrid design
    const n = features().filter((f) => f.type === 'component' && f.ref === ref).length + 1
    const fid = store.nextId('co')
    store.add({ type: 'component', id: fid, name: `${designName(ref)}:${n}`, ref, placement: { pos: at ?? [0, 0, 0], rot: [0, 0, 0] }, grounded: !features().some((f) => f.type === 'component') })
    select(fid); setPropsOpen(true); setInsertOpen(false)
    setStatus(`Inserted ${designName(ref)}`)
  }
  const insertable = () => workspace.library().filter((d) => d.id !== workspace.activeId && !wouldLoop(workspace.activeId, d.id, (x) => workspace.docOf(x)))
  const compSel = createMemo(() => { const f = primary(); return f?.type === 'component' ? f : undefined })
  const toggleGround = () => { const c = compSel(); if (c) store.update<ComponentFeature>(c.id, { grounded: !c.grounded }) }
  // The move / rotate gizmo follows the selected, ungrounded component.
  createEffect(() => {
    const c = compSel(); doc()
    const i = c ? features().indexOf(c) : -1
    const g = c
    if (!viewport) return
    if (!g || g.grounded || g.suppressed || i >= marker() || editing() || g.visible === false) { viewport.setGizmo(undefined); return }
    viewport.setGizmo({ pos: g.placement.pos, rot: g.placement.rot, mode: gizmoMode() },
      (pos, rot) => store.update<ComponentFeature>(g.id, { placement: { pos, rot } }, `gizmo:${g.id}`))
  })

  function setDesignMode(m: DesignMode) {
    if (m === mode()) return
    if (m === 'part' && features().some((f) => f.type === 'component')) return setStatus('Remove the components first: a part cannot contain components')
    store.setMode(m)
  }

  function addExtrude() {
    const sk = selected().map(byId).find((f) => f?.type === 'sketch')
    if (!sk) return setStatus('Select a sketch in the tree first')
    const id = store.nextId('ex')
    store.add({ type: 'extrude', id, name: nameFor('Extrude', 'extrude'), sketch: sk.id, distance: 10, mode: partCount() ? 'add' : 'new' })
    select(id)
  }

  function addLoft() {
    const sks = selected().map(byId).filter((f): f is SketchFeature => f?.type === 'sketch')
    if (sks.length !== 2) return setStatus('Select exactly two sketches (Ctrl-click) to loft between')
    const id = store.nextId('lf')
    store.add({ type: 'loft', id, name: nameFor('Loft', 'loft'), sketches: [sks[0].id, sks[1].id], mode: partCount() ? 'add' : 'new' })
    select(id)
  }

  // ---- revolve: click Profile / Axis boxes, then pick in the viewport ----
  const revolveOf = (id?: string) => { const f = byId(id); return f?.type === 'revolve' ? f : undefined }
  function addRevolve() {
    if (editing()) return setStatus('Finish the sketch first')
    viewport?.stopPlanePick(); setPlanePicking(false); setTool('select')
    const id = store.nextId('rv')
    // A selected sketch with exactly one closed region is picked right away.
    const pre = selected().map(byId).find((f) => f?.type === 'sketch') as SketchFeature | undefined
    const preRegions = pre ? profilesOf(pre).regions : []
    const profs = pre && preRegions.length === 1 ? [{ sketch: pre.id, pt: interiorPoint(preRegions[0]) }] : []
    store.add({ type: 'revolve', id, name: nameFor('Revolve', 'revolve'), profiles: profs, extent: 'full', angle: 360, mode: partCount() ? 'add' : 'new' })
    select(id); setPropsOpen(true)
    setRevolveCmd({ id, active: profs.length ? 'axis' : 'profile', isNew: true })
    setStatus(profs.length ? 'Revolve: click the axis (a sketch line, a sketch axis or a world axis)' : 'Revolve: click one or more sketch profiles, then the axis')
  }
  function setRevolveActive(id: string, active: RevolveCmd['active']) {
    select(id); setPropsOpen(true)
    setRevolveCmd({ id, active, isNew: revolveCmd()?.id === id ? revolveCmd()!.isNew : false })
    setStatus(active === 'profile' ? 'Revolve: click sketch profiles (click again to deselect)' : 'Revolve: click the axis')
  }
  function endRevolve(cancel: boolean) {
    const cmd = revolveCmd()
    setRevolveCmd(undefined); setHoverPick({})
    if (cmd && cancel && cmd.isNew) { store.remove(cmd.id); setSelected([]) }
    setStatus('Ready')
  }
  function onRevolveHover(x: number, y: number) {
    const cmd = revolveCmd()
    if (!cmd || !viewport) return
    const next: { profile?: ProfileHL; axis?: AxisHL } = {}
    if (cmd.active === 'profile') {
      const axisFirst = (revolveOf(cmd.id)?.profiles.length ?? 0) > 0 ? viewport.pickAxis(x, y, 5) : undefined
      if (axisFirst && axisFirst.kind !== 'line') next.axis = axisFirst.hl
      else next.profile = viewport.pickProfile(x, y)?.hl
    } else if (cmd.active === 'axis') next.axis = viewport.pickAxis(x, y)?.hl
    setHoverPick(next); refreshPick()
  }
  function onRevolvePick(x: number, y: number) {
    const cmd = revolveCmd(), f = revolveOf(cmd?.id)
    if (!cmd || !f || !viewport || cmd.active === 'none') return
    const pickAxis = (tol?: number) => {
      const hit = viewport!.pickAxis(x, y, tol)
      if (!hit) return false
      store.update<RevolveFeature>(f.id, { axis: hit.ref }); setRevolveCmd({ ...cmd, active: 'none' }); setHoverPick({}); setStatus('Revolve: axis set. Adjust the options, then OK')
      return true
    }
    if (cmd.active === 'axis') { pickAxis(); return }
    // Profile box: click regions to toggle them. Once something is picked, a click right on a construction line or axis sets the axis.
    if (f.profiles.length) { const hit = viewport.pickAxis(x, y, 5); if (hit && hit.kind !== 'line') { pickAxis(5); return } }
    const hit = viewport.pickProfile(x, y)
    const sk = byId(hit?.sketch)
    if (!hit || sk?.type !== 'sketch') return
    const p = profilesOf(sk), region = regionAt(p, hit.pt)
    if (!region) return
    const same = f.profiles.findIndex((r) => r.sketch === sk.id && regionAt(p, r.pt) === region)
    const next = same >= 0 ? f.profiles.filter((_, i) => i !== same) : [...f.profiles, { sketch: sk.id, pt: interiorPoint(region) }]
    store.update<RevolveFeature>(f.id, { profiles: next })
    setStatus(`Revolve: ${next.length} profile(s) selected. Click more, or click the Axis box / a construction line`)
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
    insertDxfText(f.name, f.text)
  }
  function insertDxfText(name: string, text: string) {
    const f = { name, text }
    const model = dxfToSketch(f.text)
    const curves = model.entities.filter((e) => e.kind !== 'point').length
    if (!curves) return setStatus('No lines, arcs or circles found in that DXF')
    if (mode() === 'assembly') store.setMode('hybrid')
    const id = store.nextId('sk')
    store.add({ type: 'sketch', id, name: f.name.replace(/\.dxf$/i, ''), plane: 'XY', offset: 0, entities: model.entities, constraints: model.constraints })
    select(id); setStatus(`Imported ${curves} curve(s)`)
  }

  /** A STEP file becomes its own design (first feature: import-step), inserted as a component when we are in a design. */
  async function importStepFile(file: File) {
    const r = workspace.remote
    if (!r?.project) { setStatus('STEP import needs the server (it stores the file)'); return }
    setStatus(`Importing ${file.name}…`)
    try {
      const assetId = `a${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`
      await r.putAsset(assetId, { name: file.name, type: file.type || 'application/step', bytes: new Uint8Array(await file.arrayBuffer()) }, cwd())
      const name = workspace.uniqueName(file.name.replace(/\.(step|stp)$/i, ''))
      const d: Document = { ...emptyDocument(name, 'part'), features: [{ type: 'import-step', id: 'is1', name, asset: assetId }] }
      const inAssembly = mode() !== 'part' && !editing()
      const newId = workspace.create(d, { parent: cwd(), open: false })
      await workspace.refreshManifest()
      if (inAssembly) await insertComponent(newId) // assemblies and hybrids get the part placed straight away
      else { resetUi(); await workspace.open(newId) }
      setStatus(`Imported ${file.name}`)
    } catch (e: any) { setStatus(`STEP import failed: ${e.message}`) }
  }
  async function insertDxfAsset(item: Item) {
    try { insertDxfText(item.name.replace(/\.dxf$/i, ''), new TextDecoder().decode(await workspace.remote!.assetBytes(item.id))) } catch (e: any) { setStatus(`Cannot read ${item.name}: ${e.message}`) }
  }

  function onKey(e: KeyboardEvent) {
    const tag = (e.target as HTMLElement)?.tagName
    if (e.key === 'Escape' && galleryOpen()) { setGalleryOpen(false); return }
    if (e.key === 'Escape' && (versionsOpen() || insertOpen() || groupMenu())) { setVersionsOpen(false); setInsertOpen(false); setGroupMenu(undefined); return }
    if (e.key === 'Escape' && viewport?.isPlanePicking) { viewport.stopPlanePick(); setPlanePicking(false); setStatus('Ready'); return }
    if (e.key === 'Escape' && revolveCmd()) { endRevolve(true); return }
    if (tag === 'INPUT' || tag === 'SELECT' || editing()) return
    const mod = e.ctrlKey || e.metaKey
    if (mod && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? store.redo() : store.undo() }
    else if (mod && e.key.toLowerCase() === 'y') { e.preventDefault(); store.redo() }
    else if (mod && e.key.toLowerCase() === 's') { e.preventDefault(); save() }
    else if (e.key === 'Escape') { setTool('select'); viewport?.setSelectedFaces([]); setFaceSel([]) }
    else if (e.key === 'Delete' && primary()) { removeFeature(primary()!.id) }
    else if (e.key === 'f') viewport?.fit()
  }

  function removeFeature(id: string) {
    if (revolveCmd()?.id === id) { setRevolveCmd(undefined); setHoverPick({}) }
    store.remove(id); setSelected(selected().filter((x) => x !== id))
  }
  /** Timeline double-click: sketches open the sketcher, everything else opens its properties. */
  function editFeature(id: string) {
    const f = byId(id)
    if (!f) return
    const i = features().indexOf(f)
    if (i >= marker()) store.setMarker(i + 1)
    select(id); setPropsOpen(true)
    if (f.type === 'sketch' && !editing()) beginEdit(id)
  }
  function selectRow(id: string, additive: boolean) {
    setSelected(additive ? (selected().includes(id) ? selected().filter((x) => x !== id) : [...selected(), id]) : [id])
    setPropsOpen(true)
  }
  const bodyRows = createMemo<BodyRow[]>(() => {
    const d = doc(), hidden = new Set(d.hidden)
    return built().filter((b) => rootType(b.root ?? b.id) !== 'component').map((b, i) => { const root = b.root ?? b.id; return { root, id: b.id, name: d.bodyNames[root] || `Body ${i + 1}`, hidden: hidden.has(root) } })
  })
  const saveChip = () => { wsTick(); return workspace.saveState }
  const CHIP: Record<string, string> = { saved: 'Saved', saving: 'Saving…', offline: 'Offline', conflict: 'Conflict', local: 'Local only' }

  // Ribbon button: icon above label (Fusion-style).
  const Tool = (p: { icon: IconName; label: string; onClick: (e: MouseEvent & { currentTarget: HTMLButtonElement }) => void; title?: string; active?: boolean; disabled?: boolean }) => (
    <button class="tool" classList={{ active: p.active }} title={p.title ?? p.label} disabled={p.disabled} onClick={(e) => p.onClick(e)}>
      <Icon name={p.icon} size={20} /><span>{p.label}</span>
    </button>
  )
  interface RTool { icon: IconName; label: string; title?: string; onClick: (e: MouseEvent & { currentTarget: HTMLButtonElement }) => void; active?: boolean; disabled?: boolean; gap?: boolean }
  interface RGroup { title: string; tools: RTool[] }
  const ribbonGroups = createMemo<RGroup[]>(() => {
    const m = mode(), out: RGroup[] = []
    if (m !== 'assembly') out.push(
      { title: '2D', tools: [
        { icon: 'sketch', label: 'Create Sketch', title: 'Create a sketch: pick a plane', get active() { return planePicking() || !!editing() }, onClick: () => addSketch() },
      ] },
      { title: '3D', tools: [
        { icon: 'extrude', label: 'Extrude', onClick: () => addExtrude() },
        { icon: 'revolve', label: 'Revolve', title: 'Revolve profiles around an axis: pick profiles, then the axis', get active() { return !!revolveCmd() }, onClick: () => (revolveCmd() ? endRevolve(true) : addRevolve()) },
        { icon: 'loft', label: 'Loft', title: 'Loft between two sketches (Ctrl-click to select)', onClick: () => addLoft() },
        { icon: 'combine', label: 'Combine', title: 'Union / cut / common of two solids (Ctrl-click to select)', onClick: () => addBoolean() },
      ] },
      { title: 'Construct', tools: [
        { icon: 'plane', label: 'Offset Plane', title: 'A plane parallel to another, a distance away', onClick: () => addPlane('offset') },
        { icon: 'plane', label: 'Plane at Angle', title: 'A plane rotated about a world axis', onClick: () => addPlane('angle') },
        { icon: 'plane', label: 'Midplane', title: 'A plane half way between two parallel planes', onClick: () => addPlane('mid') },
      ] },
      { title: 'Modify', tools: [
        { icon: 'thread', label: 'Thread', title: 'Thread a cylinder: options open on the right', get active() { return tool() === 'thread' }, onClick: () => toggleTool('thread') },
        { icon: 'deleteFeature', label: 'Delete feature', title: 'Remove a boss, pocket or hole: pick any of its faces', get active() { return tool() === 'delete-feature' }, onClick: () => toggleTool('delete-feature') },
      ] },
    )
    if (m !== 'part') out.push({ title: 'Assemble', tools: [
      { icon: 'component', label: 'Insert', title: 'Insert a design as a component (or drag it from the data panel)', get active() { return insertOpen() },
        onClick: (e) => { const r = e?.currentTarget?.getBoundingClientRect(); setInsertAt({ x: r?.left ?? 120, y: r?.bottom ?? 120 }); setInsertOpen(!insertOpen()) } },
      { icon: 'move', label: 'Move', title: 'Move the selected component with the gizmo', get active() { return gizmoMode() === 'translate' }, get disabled() { return !compSel() }, onClick: () => setGizmoMode('translate') },
      { icon: 'rotateCw', label: 'Rotate', title: 'Rotate the selected component with the gizmo', get active() { return gizmoMode() === 'rotate' }, get disabled() { return !compSel() }, onClick: () => setGizmoMode('rotate') },
      { icon: 'pin', get label() { return compSel()?.grounded ? 'Unground' : 'Ground' }, title: 'Fix the selected component in space', get active() { return !!compSel()?.grounded }, get disabled() { return !compSel() }, onClick: () => toggleGround() },
    ] })
    out.push({ title: 'Import / Export', tools: [
      ...(m !== 'assembly' ? [{ icon: 'import' as IconName, label: 'Import', title: 'Import a DXF as a new sketch', gap: true, onClick: () => importDxf() }] : []),
      { icon: 'export', label: 'STEP', title: 'Export all solids as STEP', onClick: () => exportStep() },
      { icon: 'export', label: 'STL', title: 'Export all solids as STL', onClick: () => exportStl() },
      ...(m !== 'assembly' ? [{ icon: 'export' as IconName, label: 'DXF', title: 'Export the selected sketch as DXF', onClick: () => exportDxf() }] : []),
    ] })
    return out
  })

  // The ribbon never scrolls: when it gets too narrow, groups show fewer tools on the bar
  // (the largest group shrinks first). The header dropdown of every group always lists all of them.
  const [navW, setNavW] = createSignal(4000)
  let navEl!: HTMLElement
  onMount(() => {
    const ro = new ResizeObserver(() => setNavW(navEl.clientWidth))
    ro.observe(navEl); setNavW(navEl.clientWidth)
    onCleanup(() => ro.disconnect())
  })
  const toolW = (t: RTool) => Math.max(54, Math.round(t.label.length * 6.6) + 18) + 2
  const barCount = createMemo<number[]>(() => {
    const gs = ribbonGroups(), avail = navW() - 20
    const n = gs.map((g) => g.tools.length)
    const width = (i: number) => 21 + Math.max(gs[i].tools.slice(0, n[i]).reduce((a, t) => a + toolW(t), 0), gs[i].title.length * 8 + (gs[i].tools.length > 1 ? 16 : 0))
    const total = () => gs.reduce((s, _, i) => s + width(i), 0)
    while (total() > avail) {
      let pick = -1
      for (let i = 0; i < n.length; i++) if (gs[i].tools.length > 1 && n[i] > 1 && (pick < 0 || n[i] >= n[pick])) pick = i
      if (pick < 0) for (let i = n.length - 1; i >= 0; i--) if (gs[i].tools.length > 1 && n[i] === 1) { pick = i; break }
      if (pick < 0) break
      n[pick]--
    }
    return n
  })
  const [groupMenu, setGroupMenu] = createSignal<{ title: string; x: number; y: number } | undefined>()

  const toolHint = () => planePicking() ? 'Select a plane to sketch on' : ''
  /** Leave sketch / tool modes and clear selection before another design becomes active. */
  function resetUi() {
    viewport?.stopPlanePick(); setPlanePicking(false)
    setRevolveCmd(undefined); setHoverPick({})
    endEdit(); setTool('select'); setSelected([]); setFaceSel([]); setFaceInfo(undefined); viewport?.setSelectedFaces([])
    setInsertOpen(false); setOrigin(NO_ORIGIN); setBuilt([]); setBodies([])
    firstBuild = true
  }
  const newTab = (m: DesignMode = 'part') => { resetUi(); workspace.create(undefined, { parent: cwd(), mode: m }); setGalleryOpen(false) }
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
        <button class="icon-btn" classList={{ on: sideOpen() }} title="Project data panel" aria-label="Data panel" aria-pressed={sideOpen()} onClick={() => setSideOpen(!sideOpen())}><Icon name="menu" /></button>
        <div class="designs-wrap">
          <button class="icon-btn" classList={{ on: galleryOpen() }} title="All designs" onClick={() => setGalleryOpen(!galleryOpen())}><Icon name="grid" /></button>
          <Show when={galleryOpen()}>
            <div class="gallery-backdrop" onClick={() => setGalleryOpen(false)} />
            <div class="gallery float">
              <div class="float-head"><span>Designs</span><button class="small-btn" onClick={() => newTab()}><Icon name="plus" size={14} /> New design</button></div>
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
          <button class="icon-btn small" title="New design" onClick={() => newTab()}><Icon name="plus" size={16} /></button>
        </div>
        <span class="spacer" />
        <div class="chip-wrap">
          <button class="save-chip" classList={{ [saveChip()]: true }} data-state={saveChip()} title={saveChip() === 'conflict' ? 'The design changed on the server: click to resolve' : saveChip() === 'offline' ? 'Server unreachable: changes are kept in this browser and retried' : saveChip() === 'local' ? 'No server: designs are stored in this browser only' : 'Storage status'}
            onClick={() => { if (saveChip() === 'conflict') setConflictOpen(!conflictOpen()) }}>
            <Icon name={saveChip() === 'conflict' ? 'warning' : 'cloud'} size={14} /> {CHIP[saveChip()]}
          </button>
          <Show when={conflictOpen() && saveChip() === 'conflict'}>
            <div class="gallery-backdrop" onClick={() => setConflictOpen(false)} />
            <div class="float conflict">
              <div class="float-head"><span>Changed elsewhere</span></div>
              <For each={(wsTick(), workspace.conflicts)}>{(c) => (
                <div class="conflict-row"><b>{c.name}</b>
                  <p class="dim">Another tab or device saved a newer version of this design.</p>
                  <div class="btn-row">
                    <button onClick={() => { void workspace.resolveConflict(c.id, 'mine'); setConflictOpen(false) }}>Keep mine</button>
                    <button onClick={() => { if (c.id === workspace.activeId) resetUi(); void workspace.resolveConflict(c.id, 'theirs'); setConflictOpen(false) }}>Load server copy</button>
                  </div>
                </div>
              )}</For>
            </div>
          </Show>
        </div>
        <div class="chip-wrap">
          <button class="icon-btn" title="Versions: save, browse and restore milestones" aria-label="Versions" onClick={() => setVersionsOpen(!versionsOpen())}><Icon name="history" /></button>
          <Show when={versionsOpen()}>
            <Versions workspace={workspace} designId={workspace.activeId} onClose={() => setVersionsOpen(false)} status={setStatus}
              onRestore={(d, m) => { resetUi(); store.load(d, m) }} onOpenAsNew={(d, name) => { resetUi(); workspace.create({ ...d, name: workspace.uniqueName(name) }) }} />
          </Show>
        </div>
        <span class="bar-sep" />
        <button class="icon-btn" title="Open a .darkcad file…" onClick={openDoc}><Icon name="open" /></button>
        <button class="icon-btn" title="Save as a .darkcad file (Ctrl+S)" onClick={save}><Icon name="save" /></button>
        <span class="bar-sep" />
        <button class="icon-btn" title="Undo (Ctrl+Z)" disabled={!canUndo()} onClick={() => store.undo()}><Icon name="undo" /></button>
        <button class="icon-btn" title="Redo (Ctrl+Shift+Z)" disabled={!canRedo()} onClick={() => store.redo()}><Icon name="redo" /></button>
        <button class="icon-btn" title="Fit view (F)" onClick={() => viewport?.fit()}><Icon name="fit" /></button>
      </header>

      <nav class="ribbon" ref={navEl}>
        <For each={ribbonGroups()}>{(g, gi) => (
          <section class="ribbon-group" data-group={g.title}>
            <div class="ribbon-tools">
              <For each={g.tools.slice(0, barCount()[gi()] ?? g.tools.length)}>{(t) => (<>
                <Tool icon={t.icon} label={t.label} title={t.title} active={t.active} disabled={t.disabled} onClick={t.onClick} />
                <Show when={t.gap}><span class="sep" /></Show>
              </>)}</For>
            </div>
            <Show when={g.tools.length > 1} fallback={<h4>{g.title}</h4>}>
              <h4>
                <button class="grp-btn" aria-haspopup="menu" aria-expanded={groupMenu()?.title === g.title} aria-label={`${g.title}: all tools`} title="Show all tools in this group"
                  onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); setGroupMenu(groupMenu()?.title === g.title ? undefined : { title: g.title, x: r.left, y: r.bottom }) }}>
                  {g.title}<Icon name="chevDown" size={11} />
                </button>
              </h4>
            </Show>
          </section>
        )}</For>
      </nav>

      <div class="workarea" classList={{ 'side-open': sideOpen() }}>
        <Show when={sideOpen()}>
          <aside class="sidebar" aria-label="Project data">
            <DataPanel workspace={workspace} tick={wsTick()} activeId={activeId()} openIds={tabs().map((t) => t.id)} cwd={cwd()} setCwd={setCwd}
              onOpen={(id) => { resetUi(); void workspace.open(id) }} onInsert={(id) => void insertComponent(id)} onDelete={deleteDesign}
              onNewDesign={(m) => newTab(m)} onImportStep={importStepFile} onInsertDxf={insertDxfAsset} status={setStatus} />
          </aside>
        </Show>
        <div class="center">
      <main class="stage">
        <div class="viewport" ref={host} />
        <Show when={sketchBeingEdited() && viewport && frameOf(editing()!)}>{(_) => (
          <Sketcher store={store} viewport={viewport!} sketch={sketchBeingEdited()!} frame={frameOf(editing()!)!} frameOf={frameOf}
            others={active().filter((f): f is SketchFeature => f.type === 'sketch' && f.id !== editing())}
            selected={entitySel()} onSelect={setEntitySel} onClose={endEdit} />
        )}</Show>
        <Show when={toolHint()}><div class="tool-hint">{toolHint()} <kbd>Esc</kbd></div></Show>
        <Show when={building()}><div class="busy">building…</div></Show>

        <Browser designId={activeId()} doc={doc()} marker={marker()} store={store} errors={errors()} bodies={bodyRows()} selected={selected()} editing={editing()}
          origin={origin()} setOrigin={setOrigin} setMode={setDesignMode} onSelect={selectRow} onEditSketch={(id) => beginEdit(id)} />

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
              <Properties f={primary()!} store={store} features={features()} designName={designName} gizmo={{ mode: gizmoMode(), set: setGizmoMode }} revolve={{ cmd: revolveCmd(), activate: setRevolveActive, end: endRevolve }} entitySel={entitySel()} setEntitySel={setEntitySel} error={errors().get(primary()!.id)} edit={() => beginEdit(primary()!.id)} />
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
      <Timeline store={store} features={features()} marker={marker()} selected={selected()} errors={errors()} onSelect={selectRow} onEdit={editFeature} onDelete={removeFeature} />
        </div>
      </div>

      <Show when={groupMenu()}>{(gm) => {
        const g = () => ribbonGroups().find((x) => x.title === gm().title)
        return (<Show when={g()}>{(grp) => (<>
          <div class="ctx-backdrop" onPointerDown={() => setGroupMenu(undefined)} />
          <div class="ctx group-menu" role="menu" aria-label={`${grp().title} tools`} style={{ left: `${Math.min(gm().x, window.innerWidth - 220)}px`, top: `${gm().y + 2}px` }} onPointerDown={(e) => e.stopPropagation()}>
            <For each={grp().tools}>{(t) => (
              <button role="menuitem" data-tool={t.label} classList={{ on: t.active }} disabled={t.disabled} title={t.title ?? t.label}
                onClick={(e) => { t.onClick(e); setGroupMenu(undefined) }}>
                <Icon name={t.icon} size={16} /><span>{t.label}</span>
                <Show when={t.active}><Icon name="check" size={13} /></Show>
              </button>
            )}</For>
          </div>
        </>)}</Show>)
      }}</Show>

      <Show when={insertOpen()}>
        <div class="ctx-backdrop" onPointerDown={() => setInsertOpen(false)} />
        <div class="ctx insert-menu" style={{ left: `${insertAt().x}px`, top: `${insertAt().y + 4}px` }} onPointerDown={(e) => e.stopPropagation()}>
          <For each={insertable()}>{(d) => <button data-design={d.id} onClick={() => void insertComponent(d.id)}>{d.name}</button>}</For>
          <Show when={!insertable().length}><span class="dim pad">No other designs to insert yet</span></Show>
        </div>
      </Show>

      <footer class="status"><span>{status()}</span><span class="dim">{bodies().length} solid(s) · {features().length} feature(s)</span></footer>
    </div>
  )
}

function Num(p: { label: string; value: number; onInput: (v: number) => void; step?: number; disabled?: boolean }) {
  return (
    <label class="field"><span>{p.label}</span>
      <input type="number" step={p.step ?? 1} disabled={p.disabled} value={Math.round(p.value * 1e6) / 1e6} onInput={(e) => { const v = parseFloat(e.currentTarget.value); if (Number.isFinite(v)) p.onInput(v) }} />
    </label>
  )
}

/** A Fusion-style selection box: click it to make it the target of the next viewport click. */
function SelBox(p: { label: string; text: string; active: boolean; empty: boolean; onActivate: () => void; onClear?: () => void }) {
  return (
    <div class="selbox"><span>{p.label}</span>
      <button class="box" classList={{ active: p.active, empty: p.empty }} onClick={p.onActivate} title="Click, then pick in the viewport">
        <span>{p.text}</span>
        <Show when={p.onClear}><span class="clear" title="Clear" onClick={(e) => { e.stopPropagation(); p.onClear!() }}>×</span></Show>
      </button>
    </div>
  )
}
const Seg2 = (p: { value: string; options: [string, string][]; onChange: (v: string) => void }) => (
  <div class="seg"><For each={p.options}>{([v, label]) => <button classList={{ on: p.value === v }} onClick={() => p.onChange(v)}>{label}</button>}</For></div>
)

const fmt = (v: number) => String(Math.round(v * 1e3) / 1e3)

/** Curves and dimensions of a sketch: select, delete, and edit driving dimensions (the solver moves the geometry). */
function SketchInfo(p: { s: SketchFeature; store: DocStore; entitySel: string[]; setEntitySel: (ids: string[]) => void }) {
  const geo = createMemo(() => buildGeo(p.s.entities))
  const curves = () => geo().real.filter((c) => !c.virtual)
  const dims = () => p.s.constraints.filter((c) => c.value !== undefined)
  const solved = createMemo(() => solve(p.s.entities, p.s.constraints))
  const open = createMemo(() => profiles(p.s.entities, p.s.constraints).open.length)
  const describe = (c: ReturnType<typeof curves>[number]) => {
    if (c.kind === 'line') return `length ${fmt(Math.hypot(c.b[0] - c.a[0], c.b[1] - c.a[1]))} mm`
    if (c.kind === 'circle') return `Ø ${fmt(c.r * 2)} mm`
    return `R ${fmt(c.r)} mm`
  }
  const remove = (id: string) => {
    const r = deleteEntities({ entities: p.s.entities, constraints: p.s.constraints }, [id])
    const next = solve(r.model.entities, r.model.constraints, { quick: true })
    p.store.update<SketchFeature>(p.s.id, { entities: next.ok ? next.entities : r.model.entities, constraints: r.model.constraints })
    p.setEntitySel([])
  }
  const setDim = (id: string, v: number) => {
    const cs = p.s.constraints.map((c) => (c.id === id ? { ...c, value: v } : c))
    const r = solve(p.s.entities, cs, { quick: true })
    if (r.ok) p.store.update<SketchFeature>(p.s.id, { entities: r.entities, constraints: cs }, `dim:${id}`)
  }
  return (<>
    <p class="dim lead">{!solved().ok ? 'Over-constrained: conflicting constraints' : solved().dof === 0 ? 'Fully constrained' : `${solved().dof} degree${solved().dof === 1 ? '' : 's'} of freedom`}
      {open() ? ` · ${open()} open curve(s) cannot be extruded` : ''}</p>
    <Show when={dims().length}>
      <h4>Dimensions</h4>
      <For each={dims()}>{(c) => <Num label={`${c.type}${c.type === 'angle' ? ' (°)' : ''}`} value={c.value!} step={0.5} onInput={(v) => setDim(c.id, v)} />}</For>
    </Show>
    <h4>Curves</h4>
    <For each={curves()}>{(c) => (
      <div class="entity" classList={{ sel: p.entitySel.includes(c.id) }} onClick={() => p.setEntitySel([c.id])}>
        <div class="row"><span><b>{c.kind}</b>{c.construction ? ' · construction' : ''} <span class="dim">{describe(c)}</span></span>
          <button class="x" title="Delete" onClick={(ev) => { ev.stopPropagation(); remove(c.id) }}>×</button></div>
      </div>
    )}</For>
    <Show when={!curves().length}><p class="dim">Nothing drawn yet.</p></Show>
  </>)
}

function Properties(p: { f: Feature; store: DocStore; features: Feature[]; designName: (id: string) => string; gizmo: { mode: 'translate' | 'rotate'; set: (m: 'translate' | 'rotate') => void }; revolve: { cmd?: RevolveCmd; activate: (id: string, a: RevolveCmd['active']) => void; end: (cancel: boolean) => void }; entitySel: string[]; setEntitySel: (ids: string[]) => void; error?: string; edit: () => void }) {
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
  /** Planes and sketches that come before this feature (a feature can only use earlier ones). */
  const earlier = () => p.features.slice(0, Math.max(0, p.features.findIndex((f) => f.id === p.f.id)))
  const BaseSel = (q: { label: string; value: PlaneBase; onChange: (b: PlaneBase) => void; sketches?: boolean }) => (
    <label class="field"><span>{q.label}</span>
      <select value={encBase(q.value)} onChange={(e) => q.onChange(decBase(e.currentTarget.value))}>
        <For each={['XY', 'XZ', 'YZ'] as Plane[]}>{(pl) => <option value={`w:${pl}`} selected={encBase(q.value) === `w:${pl}`}>{pl} plane</option>}</For>
        <For each={earlier().filter((f) => f.type === 'plane')}>{(f) => <option value={`p:${f.id}`} selected={encBase(q.value) === `p:${f.id}`}>{f.name}</option>}</For>
        <Show when={q.sketches !== false}><For each={earlier().filter((f) => f.type === 'sketch')}>{(f) => <option value={`s:${f.id}`} selected={encBase(q.value) === `s:${f.id}`}>{f.name} (sketch plane)</option>}</For></Show>
      </select>
    </label>
  )
  const triple = (v: [number, number, number], i: number, x: number): [number, number, number] => { const n = [...v] as [number, number, number]; n[i] = x; return n }
  return (
    <div class="form">
      <label class="field"><span>Name</span><input value={p.f.name} onChange={(e) => up({ name: e.currentTarget.value })} /></label>
      <Show when={p.error}><div class="error">{p.error}</div></Show>
      <Switch>
        <Match when={p.f.type === 'sketch' && (p.f as SketchFeature)}>{(s) => (<>
          <label class="field"><span>Plane</span>
            <select value={s().planeRef ? `plane:${s().planeRef}` : s().plane} onChange={(e) => { const v = e.currentTarget.value; v.startsWith('plane:') ? up({ planeRef: v.slice(6) }) : up({ plane: v, planeRef: undefined }) }}>
              <For each={['XY', 'XZ', 'YZ']}>{(pl) => <option value={pl} selected={!s().planeRef && s().plane === pl}>{pl}</option>}</For>
              <For each={earlier().filter((f) => f.type === 'plane')}>{(f) => <option value={`plane:${f.id}`} selected={s().planeRef === f.id}>{f.name}</option>}</For>
            </select>
          </label>
          <Show when={!s().planeRef}><Num label="Offset (mm)" value={s().offset} onInput={(v) => up({ offset: v })} /></Show>
          <button onClick={p.edit}>Edit sketch</button>
          <SketchInfo s={s()} store={p.store} entitySel={p.entitySel} setEntitySel={p.setEntitySel} />
        </>)}</Match>
        <Match when={p.f.type === 'extrude' && p.f}>{(e) => { const x = e() as Extract<Feature, { type: 'extrude' }>; return (<>
          <Pick label="Sketch" value={x.sketch} options={sketches()} onChange={(v) => up({ sketch: v })} />
          <Num label="Distance (mm)" value={x.distance} step={0.5} onInput={(v) => up({ distance: v })} />
          <Choice label="Mode" value={x.mode} options={['new', 'add', 'cut']} onChange={(v) => up({ mode: v })} />
        </>) }}</Match>
        <Match when={p.f.type === 'revolve' && p.f}>{(e) => { const x = () => e() as RevolveFeature; const on = () => p.revolve.cmd?.id === x().id ? p.revolve.cmd.active : undefined
          const skMap = () => new Map(sketches().map((s) => [s.id, s as SketchFeature]))
          const axisName = (a?: AxisRef) => !a ? '' : a.kind === 'world' ? `${a.axis} axis (world)` : `${skMap().get(a.sketch)?.name ?? a.sketch}: ${a.id === 'axis-x' ? 'X axis' : a.id === 'axis-y' ? 'Y axis' : 'line'}`
          return (<>
          <SelBox label="Profile" active={on() === 'profile'} empty={!x().profiles.length} text={x().profiles.length ? `${x().profiles.length} profile${x().profiles.length === 1 ? '' : 's'} selected` : 'Select profile(s)'}
            onActivate={() => p.revolve.activate(x().id, 'profile')} onClear={x().profiles.length ? () => up({ profiles: [] }) : undefined} />
          <SelBox label="Axis" active={on() === 'axis'} empty={!x().axis} text={x().axis ? axisName(x().axis) : 'Select axis'}
            onActivate={() => p.revolve.activate(x().id, 'axis')} onClear={x().axis ? () => up({ axis: undefined }) : undefined} />
          <label class="field"><span>Type</span><Seg2 value={x().extent} options={[['full', 'Full'], ['angle', 'Angle'], ['symmetric', 'Symmetric']]} onChange={(v) => up({ extent: v })} /></label>
          <Show when={x().extent !== 'full'}><Num label={x().extent === 'symmetric' ? 'Total angle (°)' : 'Angle (°)'} value={x().angle} step={5} onInput={(v) => up({ angle: Math.max(0.01, Math.min(359.99, v)) })} /></Show>
          <Choice label="Operation" value={x().mode} options={['new', 'add', 'cut']} onChange={(v) => up({ mode: v })} />
          <Show when={on() !== undefined || p.revolve.cmd?.id === x().id}>
            <div class="btn-row"><button class="primary" onClick={() => p.revolve.end(false)}>OK</button><button onClick={() => p.revolve.end(true)}>Cancel</button></div>
          </Show>
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
        <Match when={p.f.type === 'plane' && p.f}>{(e) => { const x = () => e() as PlaneFeature; const d = () => x().def
          const base = (): PlaneBase => d().kind === 'mid' ? (d() as Extract<PlaneDef, { kind: 'mid' }>).a : (d() as Extract<PlaneDef, { kind: 'offset' }>).base
          const setKind = (k: string) => {
            if (k === d().kind) return
            up({ def: k === 'offset' ? { kind: 'offset', base: base(), distance: 10 } : k === 'angle' ? { kind: 'angle', base: base(), axis: 'X', angle: 45 } : { kind: 'mid', a: base(), b: base() } })
          }
          return (<>
          <label class="field"><span>Type</span><Seg2 value={d().kind} options={[['offset', 'Offset'], ['angle', 'Angle'], ['mid', 'Midplane']]} onChange={setKind} /></label>
          <Show when={d().kind === 'offset'}>
            <BaseSel label="Plane" value={(d() as Extract<PlaneDef, { kind: 'offset' }>).base} onChange={(b) => up({ def: { ...d(), base: b } })} />
            <Num label="Distance (mm)" value={(d() as Extract<PlaneDef, { kind: 'offset' }>).distance} step={0.5} onInput={(v) => up({ def: { ...d(), distance: v } })} />
          </Show>
          <Show when={d().kind === 'angle'}>
            <BaseSel label="Plane" value={(d() as Extract<PlaneDef, { kind: 'angle' }>).base} onChange={(b) => up({ def: { ...d(), base: b } })} />
            <Choice label="About axis" value={(d() as Extract<PlaneDef, { kind: 'angle' }>).axis} options={['X', 'Y', 'Z']} onChange={(v) => up({ def: { ...d(), axis: v } })} />
            <Num label="Angle (°)" value={(d() as Extract<PlaneDef, { kind: 'angle' }>).angle} step={5} onInput={(v) => up({ def: { ...d(), angle: v } })} />
          </Show>
          <Show when={d().kind === 'mid'}>
            <BaseSel label="First plane" value={(d() as Extract<PlaneDef, { kind: 'mid' }>).a} onChange={(b) => up({ def: { ...d(), a: b } })} />
            <BaseSel label="Second plane" value={(d() as Extract<PlaneDef, { kind: 'mid' }>).b} onChange={(b) => up({ def: { ...d(), b: b } })} />
            <p class="dim">The two planes must be parallel.</p>
          </Show>
        </>) }}</Match>
        <Match when={p.f.type === 'component' && p.f}>{(e) => { const x = () => e() as ComponentFeature; const g = () => !!x().grounded
          const pos = () => x().placement.pos, rot = () => x().placement.rot
          const setPos = (i: number, v: number) => up({ placement: { ...x().placement, pos: triple(pos(), i, v) } })
          const setRot = (i: number, v: number) => up({ placement: { ...x().placement, rot: triple(rot(), i, v) } })
          return (<>
          <label class="field"><span>Design</span><input readOnly value={p.designName(x().ref)} /></label>
          <Show when={g()}><p class="dim">Grounded: fixed in space. Unground it to move it.</p></Show>
          <h4>Position (mm)</h4>
          <Num label="X" value={pos()[0]} step={1} disabled={g()} onInput={(v) => setPos(0, v)} />
          <Num label="Y" value={pos()[1]} step={1} disabled={g()} onInput={(v) => setPos(1, v)} />
          <Num label="Z" value={pos()[2]} step={1} disabled={g()} onInput={(v) => setPos(2, v)} />
          <h4>Rotation (°)</h4>
          <Num label="About X" value={rot()[0]} step={5} disabled={g()} onInput={(v) => setRot(0, v)} />
          <Num label="About Y" value={rot()[1]} step={5} disabled={g()} onInput={(v) => setRot(1, v)} />
          <Num label="About Z" value={rot()[2]} step={5} disabled={g()} onInput={(v) => setRot(2, v)} />
          <label class="field"><span>Gizmo</span><Seg2 value={p.gizmo.mode} options={[['translate', 'Move'], ['rotate', 'Rotate']]} onChange={(v) => p.gizmo.set(v as 'translate' | 'rotate')} /></label>
          <button class="ground-btn" onClick={() => up({ grounded: !g() })}>{g() ? 'Unground' : 'Ground'}</button>
        </>) }}</Match>
        <Match when={p.f.type === 'import-step'}><p class="dim">A solid read from an uploaded STEP file. Insert this design into an assembly to place it.</p></Match>
        <Match when={p.f.type === 'delete-feature'}><p class="dim">Removes the protrusion containing the picked face and heals the surface underneath.</p></Match>
      </Switch>
      <button onClick={() => p.store.suppress(p.f.id, !p.f.suppressed)}>{p.f.suppressed ? 'Unsuppress' : 'Suppress'}</button>
      <button class="danger" onClick={() => p.store.remove(p.f.id)}>Delete feature</button>
    </div>
  )
}
