import { For, Show, createEffect, createMemo, createSignal, on, onCleanup, type JSX } from 'solid-js'
import type { ComponentFeature, DesignMode, DocStore, Document, Feature, Plane, PlaneFeature, SketchFeature } from '@darkcad/core'
import { Icon, type IconName } from '../icons'
import { health } from '../timeline/info'

export interface BodyRow { root: string; id: string; name: string; hidden: boolean }
export interface OriginVis { planes: Plane[]; axes: ('X' | 'Y' | 'Z')[]; point: boolean }
export const NO_ORIGIN: OriginVis = { planes: [], axes: [], point: false }

export interface BrowserProps {
  designId: string
  doc: Document
  marker: number
  store: DocStore
  errors: Map<string, string>
  bodies: BodyRow[]
  selected: string[]
  editing?: string
  origin: OriginVis
  setOrigin: (v: OriginVis) => void
  setMode: (m: DesignMode) => void
  onSelect: (id: string, additive: boolean) => void
  onEditSketch: (id: string) => void
}

const MODES: [DesignMode, string, string][] = [
  ['part', 'Part', 'Sketches and solid modelling'],
  ['assembly', 'Assembly', 'Place components'],
  ['hybrid', 'Hybrid', 'Modelling and components'],
]
const PLANE_NAMES: Record<Plane, string> = { XY: 'XY (Top)', XZ: 'XZ (Front)', YZ: 'YZ (Right)' }
type Folder = 'root' | 'origin' | 'sketches' | 'construction' | 'bodies' | 'components'
const DEFAULT_OPEN: Record<Folder, boolean> = { root: true, origin: false, sketches: true, construction: true, bodies: true, components: true }
const openKey = (id: string) => `darkcad:tree:${id}`

function loadOpen(id: string): Record<Folder, boolean> {
  try { return { ...DEFAULT_OPEN, ...JSON.parse(localStorage.getItem(openKey(id)) ?? '{}') } } catch { return { ...DEFAULT_OPEN } }
}

/** Fusion-style browser: design > Origin, Sketches, Construction, Bodies, Components. */
export function Browser(p: BrowserProps) {
  const [open, setOpen] = createSignal(loadOpen(p.designId))
  // Expand / collapse state is remembered per design.
  createEffect(on(() => p.designId, (id) => setOpen(loadOpen(id)), { defer: true }))
  const toggle = (f: Folder) => {
    const next = { ...open(), [f]: !open()[f] }
    setOpen(next)
    try { localStorage.setItem(openKey(p.designId), JSON.stringify(next)) } catch { /* private mode */ }
  }

  const [typeOpen, setTypeOpen] = createSignal(false)
  createEffect(on(() => p.designId, () => setTypeOpen(false), { defer: true }))

  // ---- renaming: double-click a name, or select a row and press F2 ----
  const [renaming, setRenaming] = createSignal<string | undefined>()
  const [lastRow, setLastRow] = createSignal<string | undefined>()
  const commit = (key: string, raw: string) => {
    setRenaming(undefined)
    const name = raw.trim()
    if (key === 'root') p.store.rename(name || 'Untitled')
    else if (key.startsWith('body:')) p.store.setBodyName(key.slice(5), name) // empty = back to the default name
    else if (key.startsWith('f:') && name) p.store.update(key.slice(2), { name } as Partial<Feature>)
  }
  const onKey = (e: KeyboardEvent) => {
    const tag = (e.target as HTMLElement)?.tagName
    if (e.key !== 'F2' || tag === 'INPUT' || tag === 'SELECT' || p.editing) return
    const k = lastRow()
    if (k && k !== 'origin' && p.selected.length) { e.preventDefault(); setRenaming(k) }
    else if (p.selected.length) { e.preventDefault(); setRenaming(`f:${p.selected[p.selected.length - 1]}`) }
  }
  window.addEventListener('keydown', onKey)
  onCleanup(() => window.removeEventListener('keydown', onKey))

  const features = () => p.doc.features
  const idx = (id: string) => features().findIndex((f) => f.id === id)
  const ofType = <T extends Feature['type']>(t: T) => features().filter((f): f is Extract<Feature, { type: T }> => f.type === t)
  const sketches = createMemo(() => ofType('sketch'))
  const planes = createMemo(() => ofType('plane'))
  const comps = createMemo(() => ofType('component'))
  const showComponents = () => p.doc.mode !== 'part' || comps().length > 0

  // ---- row ----
  const Name = (q: { k: string; text: string }) => (
    <Show when={renaming() === q.k} fallback={<span class="label" title="Double-click to rename" onDblClick={(e) => { e.stopPropagation(); if (q.k !== 'origin') setRenaming(q.k) }}>{q.text}</span>}>
      <input class="row-name" value={q.text} ref={(el) => queueMicrotask(() => { el.focus(); el.select() })} onClick={(e) => e.stopPropagation()} onDblClick={(e) => e.stopPropagation()}
        onBlur={(e) => { if (renaming() === q.k) commit(q.k, e.currentTarget.value) }}
        onKeyDown={(e) => { e.stopPropagation(); if (e.key === 'Enter') e.currentTarget.blur(); if (e.key === 'Escape') setRenaming(undefined) }} />
    </Show>
  )
  const Eye = (q: { on: boolean; title: string; toggle: () => void }) => (
    <button class="icon-btn small eye" classList={{ off: !q.on }} title={`${q.on ? 'Hide' : 'Show'} ${q.title}`} onClick={(e) => { e.stopPropagation(); q.toggle() }}>
      <Icon name={q.on ? 'eye' : 'eyeOff'} size={14} />
    </button>
  )

  const FolderRow = (q: { id: Folder; label: string; icon: IconName; count?: number; depth?: number; eye?: JSX.Element; empty?: string; children?: JSX.Element }) => (<>
    <li class="folder" data-folder={q.id} style={{ 'padding-left': `${8 + (q.depth ?? 0) * 14}px` }} onClick={() => toggle(q.id)}>
      <span class="chev" classList={{ open: open()[q.id] }}><Icon name="chevron" size={13} /></span>
      <span class="icon"><Icon name={q.icon} size={15} /></span>
      <span class="label">{q.label}</span>
      {q.eye}
      <Show when={q.count !== undefined}><span class="count">{q.count}</span></Show>
    </li>
    <Show when={open()[q.id]}>
      {q.children}
      <Show when={q.empty && q.count === 0}><li class="none" style={{ 'padding-left': `${22 + ((q.depth ?? 0) + 1) * 14}px` }}>{q.empty}</li></Show>
    </Show>
  </>)

  const featureRow = (f: Feature, icon: IconName, depth: number, eye?: JSX.Element, extra?: JSX.Element) => {
    const key = `f:${f.id}`
    const h = () => health(f, p.errors.get(f.id))
    return (
      <li data-kind={f.type} data-id={f.id} style={{ 'padding-left': `${8 + depth * 14}px` }}
        classList={{ sel: p.selected.includes(f.id), err: h() === 'error', blocked: h() === 'blocked', off: h() === 'suppressed', rolled: idx(f.id) >= p.marker }}
        title={p.errors.get(f.id)}
        onClick={(e) => { setLastRow(key); p.onSelect(f.id, e.ctrlKey || e.metaKey) }}
        onDblClick={() => { if (f.type === 'sketch' && !p.editing) p.onEditSketch(f.id) }}>
        <span class="icon"><Icon name={icon} size={15} /></span>
        <Name k={key} text={f.name} />
        <Show when={h() === 'error' || h() === 'blocked'}><span class="badge" classList={{ warn: h() === 'blocked' }} title={p.errors.get(f.id)}>!</span></Show>
        {extra}
        {eye}
      </li>
    )
  }

  const originToggle = (kind: 'plane' | 'axis' | 'point', v?: string) => {
    const o = p.origin
    if (kind === 'plane') p.setOrigin({ ...o, planes: o.planes.includes(v as Plane) ? o.planes.filter((x) => x !== v) : [...o.planes, v as Plane] })
    else if (kind === 'axis') p.setOrigin({ ...o, axes: o.axes.includes(v as 'X') ? o.axes.filter((x) => x !== v) : [...o.axes, v as 'X'] })
    else p.setOrigin({ ...o, point: !o.point })
  }
  const originAll = () => p.origin.planes.length === 3 && p.origin.axes.length === 3 && p.origin.point
  const originRow = (kind: 'plane' | 'axis' | 'point', v: string, label: string, icon: IconName, on: boolean) => (
    <li class="origin-item" data-kind={`origin-${kind}`} data-id={v} style={{ 'padding-left': `${8 + 2 * 14}px` }} onClick={() => originToggle(kind, v)}>
      <span class="icon"><Icon name={icon} size={15} /></span>
      <span class="label">{label}</span>
      <Eye on={on} title={label} toggle={() => originToggle(kind, v)} />
    </li>
  )

  return (
    <div class="float tree" classList={{ editing: !!p.editing }}>
      <ul class="rows">
        <li class="tree-root" data-folder="root" classList={{ sel: lastRow() === 'root' && p.selected.length === 0 }} onClick={() => { toggle('root'); setLastRow('root') }}>
          <span class="chev" classList={{ open: open().root }}><Icon name="chevron" size={14} /></span>
          <Icon name="folder" size={15} />
          <Name k="root" text={p.doc.name} />
          <span class="count">{features().length}</span>
        </li>
        <Show when={open().root}>
          <li class="type-row" data-kind="design-type" data-mode={p.doc.mode} style={{ 'padding-left': `${8 + 14}px` }} title="Design type">
            <span class="icon"><Icon name={p.doc.mode === 'part' ? 'cube' : p.doc.mode === 'assembly' ? 'component' : 'grid'} size={15} /></span>
            <span class="label">{MODES.find((m) => m[0] === p.doc.mode)?.[1]}</span>
            <button class="icon-btn small eye type-edit" title="Change the design type" aria-haspopup="menu" aria-expanded={typeOpen()}
              onClick={(e) => { e.stopPropagation(); setTypeOpen(!typeOpen()) }}><Icon name="edit" size={13} /></button>
          </li>
          <Show when={typeOpen()}>
            <For each={MODES}>{([m, label, hint]) => (
              <li class="type-opt" data-mode={m} classList={{ sel: p.doc.mode === m }} style={{ 'padding-left': `${8 + 2 * 14}px` }} title={hint}
                onClick={(e) => { e.stopPropagation(); setTypeOpen(false); p.setMode(m) }}>
                <span class="label">{label}</span>
                <Show when={p.doc.mode === m}><Icon name="check" size={13} /></Show>
              </li>
            )}</For>
          </Show>
          <FolderRow id="origin" label="Origin" icon="axis" depth={1}
            eye={<Eye on={originAll()} title="the origin" toggle={() => p.setOrigin(originAll() ? NO_ORIGIN : { planes: ['XY', 'XZ', 'YZ'], axes: ['X', 'Y', 'Z'], point: true })} />}>
            <For each={['XY', 'XZ', 'YZ'] as Plane[]}>{(pl) => originRow('plane', pl, PLANE_NAMES[pl], 'plane', p.origin.planes.includes(pl))}</For>
            <For each={['X', 'Y', 'Z'] as const}>{(a) => originRow('axis', a, `${a} axis`, 'axis', p.origin.axes.includes(a))}</For>
            {originRow('point', 'origin', 'Origin point', 'point', p.origin.point)}
          </FolderRow>

          <FolderRow id="sketches" label="Sketches" icon="sketch" depth={1} count={sketches().length} empty="No sketches yet">
            <For each={sketches()}>{(s) => featureRow(s, 'sketch', 2,
              <Eye on={(s as SketchFeature).visible !== false} title="sketch" toggle={() => p.store.update<SketchFeature>(s.id, { visible: (s as SketchFeature).visible === false })} />)}</For>
          </FolderRow>

          <FolderRow id="construction" label="Construction" icon="plane" depth={1} count={planes().length} empty="No construction planes">
            <For each={planes()}>{(pl) => featureRow(pl, 'plane', 2,
              <Eye on={(pl as PlaneFeature).visible !== false} title="plane" toggle={() => p.store.update<PlaneFeature>(pl.id, { visible: (pl as PlaneFeature).visible === false })} />)}</For>
          </FolderRow>

          <FolderRow id="bodies" label="Bodies" icon="cube" depth={1} count={p.bodies.length} empty="No bodies yet">
            <For each={p.bodies}>{(b) => {
              const key = `body:${b.root}`
              return (
                <li data-kind="body" data-id={b.root} style={{ 'padding-left': `${8 + 2 * 14}px` }}
                  classList={{ sel: p.selected.includes(b.id), err: p.errors.has(b.root), off: b.hidden, rolled: idx(b.root) >= p.marker }}
                  onClick={(e) => { setLastRow(key); p.onSelect(b.id, e.ctrlKey || e.metaKey) }}>
                  <span class="icon"><Icon name="cube" size={15} /></span>
                  <Name k={key} text={b.name} />
                  <Eye on={!b.hidden} title="body" toggle={() => p.store.setBodyHidden(b.root, !b.hidden)} />
                </li>
              )
            }}</For>
          </FolderRow>

          <Show when={showComponents()}>
            <FolderRow id="components" label="Components" icon="component" depth={1} count={comps().length} empty="Insert a design from the data panel">
              <For each={comps()}>{(c) => featureRow(c, 'component', 2,
                <Eye on={(c as ComponentFeature).visible !== false} title="component" toggle={() => p.store.update<ComponentFeature>(c.id, { visible: (c as ComponentFeature).visible === false })} />,
                <Show when={(c as ComponentFeature).grounded}><span class="pin" title="Grounded"><Icon name="pin" size={12} /></span></Show>)}</For>
            </FolderRow>
          </Show>
        </Show>
      </ul>
    </div>
  )
}
