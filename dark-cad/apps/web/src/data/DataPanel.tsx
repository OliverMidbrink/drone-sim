import { For, Show, createMemo, createSignal } from 'solid-js'
import type { DesignMode } from '@darkcad/core'
import { Icon, type IconName } from '../icons'
import type { Workspace } from '../workspace'
import type { TreeNode } from '../persist/types'

export interface Item {
  id: string
  kind: 'folder' | 'design' | 'asset'
  name: string
  parent: string | null
  mode?: DesignMode
  mime?: string
  filename?: string
  updated: number
  /** Only in this browser so far (not on the server). */
  local?: boolean
}

export interface DataPanelProps {
  workspace: Workspace
  /** Bumped by the app whenever the workspace changes. */
  tick: number
  activeId: string
  openIds: string[]
  /** Folder shown in the list (null = project root). Owned by the app so that "New design" lands here. */
  cwd: string | null
  setCwd: (id: string | null) => void
  onOpen: (id: string) => void
  onInsert: (id: string) => void
  onDelete: (id: string, name: string) => void
  onNewDesign: (mode: DesignMode) => void
  onImportStep: (file: File) => Promise<void>
  onInsertDxf: (asset: Item) => Promise<void>
  status: (msg: string) => void
}

const ago = (t: number) => { const m = Math.round((Date.now() - t) / 60000); return m < 1 ? 'just now' : m < 60 ? `${m} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} d ago` }
const ext = (name: string) => (name.match(/\.([a-z0-9]+)$/i)?.[1] ?? '').toLowerCase()
/** Alphabetical (case-insensitive, "Part 2" before "Part 10"). Never depends on edit time, so rows stay put while you work. */
const byName = (a: Item, b: Item) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base', numeric: true }) || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)
const MODE_LABEL: Record<DesignMode, string> = { part: 'Part', assembly: 'Assembly', hybrid: 'Hybrid' }
const iconFor = (i: Item): IconName => i.kind === 'folder' ? 'folder' : i.kind === 'design' ? (i.mode === 'assembly' ? 'component' : 'cube') : /^image\//.test(i.mime ?? '') ? 'image' : ext(i.filename ?? i.name) === 'pdf' ? 'pdf' : 'newFile'

/** Project sidebar: folders, designs and uploaded files of the current project. */
export function DataPanel(p: DataPanelProps) {
  const ws = p.workspace
  const remote = () => ws.remote
  const online = () => !!remote()?.project && ws.saveState !== 'offline'
  const [menu, setMenu] = createSignal<{ item: Item; x: number; y: number } | undefined>()
  const [newMenu, setNewMenu] = createSignal(false)
  const [renaming, setRenaming] = createSignal<string | undefined>()
  const [over, setOver] = createSignal<string | undefined>() // folder id (or 'root') under a drag
  const [fileOver, setFileOver] = createSignal(false)
  const [busy, setBusy] = createSignal('')
  let fileInput!: HTMLInputElement
  let stepInput!: HTMLInputElement

  let seen = new Map<string, Item>()
  const items = createMemo<Item[]>(() => {
    void p.tick
    const lib = new Map(ws.library().map((d) => [d.id, d]))
    const out: Item[] = []
    for (const n of ws.manifest?.nodes ?? []) {
      if (n.kind === 'design') {
        const d = lib.get(n.id)
        out.push({ ...n, name: d?.name ?? n.name, updated: d?.updated ?? n.updated, mode: d?.mode ?? n.mode, parent: d?.parent ?? n.parent })
        lib.delete(n.id)
      } else out.push({ ...n })
    }
    for (const d of lib.values()) out.push({ id: d.id, kind: 'design', name: d.name, parent: d.parent ?? null, mode: d.mode, updated: d.updated, local: true })
    const folders = new Set(out.filter((i) => i.kind === 'folder').map((i) => i.id))
    for (const i of out) if (i.parent && !folders.has(i.parent)) i.parent = null
    // Keep the same object for an unchanged item, so its row is not rebuilt on every refresh
    // (a rebuilt row would swallow the second click of a double-click).
    const next = new Map<string, Item>()
    const stable = out.map((i) => {
      const old = seen.get(i.id)
      const keep = old && (Object.keys(i) as (keyof Item)[]).every((k) => old[k] === i[k]) && Object.keys(old).length === Object.keys(i).length ? old : i
      next.set(i.id, keep)
      return keep
    })
    seen = next
    return stable
  })
  const folders = () => items().filter((i) => i.kind === 'folder').sort(byName)
  const crumbs = createMemo(() => {
    const chain: Item[] = []
    let cur = p.cwd
    while (cur) { const f = items().find((i) => i.id === cur); if (!f) break; chain.unshift(f); cur = f.parent }
    return chain
  })
  const listing = createMemo(() => items().filter((i) => i.parent === p.cwd).sort(byName))
  const descendants = (id: string): Set<string> => { const s = new Set([id]); let grew = true; while (grew) { grew = false; for (const i of items()) if (i.parent && s.has(i.parent) && !s.has(i.id)) { s.add(i.id); grew = true } } return s }

  const guard = async (what: string, fn: () => Promise<unknown>) => {
    try { setBusy(what); await fn(); await ws.refreshManifest() } catch (e: any) { p.status(`${what} failed: ${e.message}`) } finally { setBusy('') }
  }

  const newFolder = () => {
    if (!remote()?.project) return p.status('Folders need the server')
    const name = prompt('Folder name', 'New folder')?.trim()
    if (name) void guard('Create folder', () => remote()!.createFolder(name, p.cwd))
  }

  const rename = async (i: Item, raw: string) => {
    setRenaming(undefined)
    const name = raw.trim()
    if (!name || name === i.name) return
    if (i.kind === 'design') await ws.renameDesign(i.id, name)
    else await guard('Rename', () => remote()!.patchNode(i.id, { name }))
  }

  const move = async (i: Item, parent: string | null) => {
    setMenu(undefined)
    if (i.parent === parent || (parent && descendants(i.id).has(parent))) return
    if (i.kind === 'design') await guard('Move', () => ws.moveDesign(i.id, parent))
    else await guard('Move', () => remote()!.patchNode(i.id, { parent }))
  }

  const remove = (i: Item) => {
    setMenu(undefined)
    if (i.kind === 'design') return p.onDelete(i.id, i.name)
    if (!confirm(`Delete ${i.kind} "${i.name}"${i.kind === 'folder' ? ' and everything in it' : ''}? This cannot be undone.`)) return
    void guard('Delete', async () => {
      const gone = i.kind === 'folder' ? descendants(i.id) : new Set([i.id])
      for (const d of items()) if (d.kind === 'design' && gone.has(d.id) && ws.design(d.id)) ws.deleteDesign(d.id)
      await remote()!.deleteNode(i.id)
    })
  }

  async function upload(files: FileList | File[]) {
    for (const f of Array.from(files)) {
      const e = ext(f.name)
      if (e === 'step' || e === 'stp') { await p.onImportStep(f); continue }
      if (!remote()?.project) { p.status('Uploads need the server'); return }
      await guard(`Upload ${f.name}`, async () => remote()!.putAsset(`a${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, { name: f.name, type: f.type, bytes: new Uint8Array(await f.arrayBuffer()) }, p.cwd))
    }
  }

  const dragStart = (e: DragEvent, i: Item) => {
    e.dataTransfer?.setData('application/x-darkcad-node', i.id)
    if (i.kind === 'design') e.dataTransfer?.setData('application/x-darkcad-design', i.id)
    e.dataTransfer!.effectAllowed = 'copyMove'
  }
  const dropOn = (e: DragEvent, folder: string | null) => {
    e.preventDefault(); e.stopPropagation(); setOver(undefined); setFileOver(false)
    if (e.dataTransfer?.files?.length) { void upload(e.dataTransfer.files); return }
    const id = e.dataTransfer?.getData('application/x-darkcad-node')
    const it = items().find((x) => x.id === id)
    if (it) void move(it, folder)
  }
  const overOn = (e: DragEvent, key: string) => { if (e.dataTransfer?.types.includes('Files') || e.dataTransfer?.types.includes('application/x-darkcad-node')) { e.preventDefault(); e.stopPropagation(); setOver(key) } }

  // A folder opens on click, but a click on its name waits a moment so a double-click can rename it instead.
  let navTimer: ReturnType<typeof setTimeout> | undefined
  const openFolder = (i: Item, e: MouseEvent) => {
    clearTimeout(navTimer)
    if ((e.target as HTMLElement).tagName === 'B') navTimer = setTimeout(() => p.setCwd(i.id), 250)
    else p.setCwd(i.id)
  }
  const closeMenu = () => setMenu(undefined)
  const act = (i: Item) => { if (i.kind === 'folder') p.setCwd(i.id); else if (i.kind === 'design') p.onOpen(i.id); else if (ext(i.filename ?? i.name) === 'dxf') void p.onInsertDxf(i); else window.open(remote()!.assetUrl(i.id), '_blank') }

  return (
    <div class="dp" classList={{ 'file-over': fileOver() }}
      onDragOver={(e) => { if (e.dataTransfer?.types.includes('Files')) { e.preventDefault(); setFileOver(true) } }}
      onDragLeave={(e) => { if (e.currentTarget === e.target) setFileOver(false) }}
      onDrop={(e) => dropOn(e, p.cwd)}>
      <div class="dp-head">
        <span class="dp-title">Data</span>
        <span class="dp-actions">
          <button class="icon-btn small" title="New folder" disabled={!online()} onClick={newFolder}><Icon name="folderPlus" size={15} /></button>
          <span class="dp-new">
            <button class="small-btn" title="New design" onClick={() => setNewMenu(!newMenu())}><Icon name="plus" size={13} /> New</button>
            <Show when={newMenu()}>
              <div class="ctx-backdrop" onPointerDown={() => setNewMenu(false)} />
              <div class="ctx dp-menu" onPointerDown={(e) => e.stopPropagation()}>
                <For each={['part', 'assembly', 'hybrid'] as DesignMode[]}>{(m) => <button data-mode={m} onClick={() => { setNewMenu(false); p.onNewDesign(m) }}>{MODE_LABEL[m]} design</button>}</For>
              </div>
            </Show>
          </span>
          <button class="icon-btn small" title="Upload files (STEP, DXF, PNG, PDF…)" disabled={!online()} onClick={() => fileInput.click()}><Icon name="import" size={15} /></button>
          <button class="small-btn" title="Import a STEP file as a new design" onClick={() => stepInput.click()}>STEP</button>
        </span>
        <input ref={fileInput} type="file" multiple hidden data-testid="upload" onChange={(e) => { void upload(e.currentTarget.files ?? []); e.currentTarget.value = '' }} />
        <input ref={stepInput} type="file" accept=".step,.stp" hidden data-testid="upload-step" onChange={(e) => { const f = e.currentTarget.files?.[0]; if (f) void p.onImportStep(f); e.currentTarget.value = '' }} />
      </div>

      <nav class="dp-crumbs" aria-label="Folder">
        <button classList={{ drop: over() === 'root' }} onClick={() => p.setCwd(null)} onDragOver={(e) => overOn(e, 'root')} onDragLeave={() => setOver(undefined)} onDrop={(e) => dropOn(e, null)}><Icon name="cloud" size={13} /> My Drive</button>
        <For each={crumbs()}>{(c) => (<>
          <span class="sepch"><Icon name="chevron" size={11} /></span>
          <button classList={{ drop: over() === c.id }} onClick={() => p.setCwd(c.id)} onDragOver={(e) => overOn(e, c.id)} onDragLeave={() => setOver(undefined)} onDrop={(e) => dropOn(e, c.id)}>{c.name}</button>
        </>)}</For>
      </nav>

      <ul class="dp-list">
        <For each={listing()}>{(i) => (
          <li class="dp-row" data-kind={i.kind} data-id={i.id} data-name={i.name} draggable={true}
            classList={{ current: i.id === p.activeId, drop: over() === i.id }}
            onDragStart={(e) => dragStart(e, i)}
            onDragOver={(e) => { if (i.kind === 'folder') overOn(e, i.id) }} onDragLeave={() => setOver(undefined)} onDrop={(e) => { if (i.kind === 'folder') dropOn(e, i.id); else { e.stopPropagation(); setFileOver(false) } }}
            onClick={(e) => { if (i.kind === 'folder') openFolder(i, e); else if (i.kind === 'design') p.onOpen(i.id) }}
            onDblClick={() => act(i)}
            onContextMenu={(e) => { e.preventDefault(); setMenu({ item: i, x: e.clientX, y: e.clientY }) }}>
            <span class="dp-thumb">
              <Show when={i.kind === 'design' && remote()?.project && !i.local} fallback={
                <Show when={i.kind === 'asset' && /^image\//.test(i.mime ?? '')} fallback={<Icon name={iconFor(i)} size={18} />}>
                  <img src={remote()!.assetUrl(i.id)} alt="" loading="lazy" />
                </Show>}>
                <img src={remote()!.thumbUrl(i.id, i.updated)} alt="" loading="lazy" onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; (e.currentTarget.nextElementSibling as HTMLElement | null)?.classList.remove('hide') }} />
                <span class="hide ph"><Icon name={iconFor(i)} size={18} /></span>
              </Show>
            </span>
            <span class="dp-text">
              <Show when={renaming() === i.id} fallback={<b title="Double-click to rename" onDblClick={(e) => { e.stopPropagation(); clearTimeout(navTimer); setRenaming(i.id) }}>{i.name}</b>}>
                <input class="row-name" value={i.name} ref={(el) => queueMicrotask(() => { el.focus(); el.select() })} onClick={(e) => e.stopPropagation()} onDblClick={(e) => e.stopPropagation()}
                  onBlur={(e) => { if (renaming() === i.id) void rename(i, e.currentTarget.value) }}
                  onKeyDown={(e) => { e.stopPropagation(); if (e.key === 'Enter') e.currentTarget.blur(); if (e.key === 'Escape') setRenaming(undefined) }} />
              </Show>
              <small>
                <Show when={i.kind === 'design'}><span class="tag">{MODE_LABEL[i.mode ?? 'part']}</span> </Show>
                <Show when={i.kind === 'asset'}><span class="tag">{(ext(i.filename ?? i.name) || 'file').toUpperCase()}</span> </Show>
                {i.kind === 'folder' ? `${items().filter((x) => x.parent === i.id).length} item(s)` : ago(i.updated)}
                <Show when={i.local}> · not saved to server</Show>
              </small>
            </span>
            <Show when={i.kind === 'design' && p.openIds.includes(i.id)}><span class="gopen">open</span></Show>
            <button class="icon-btn small more" title="More" onClick={(e) => { e.stopPropagation(); const r = e.currentTarget.getBoundingClientRect(); setMenu({ item: i, x: r.left, y: r.bottom }) }}><Icon name="more" size={15} /></button>
          </li>
        )}</For>
        <Show when={!listing().length}><li class="none">{p.cwd ? 'This folder is empty.' : 'Nothing here yet. Create a design, or drop files to upload.'}</li></Show>
      </ul>
      <Show when={busy()}><div class="dp-busy">{busy()}…</div></Show>
      <Show when={fileOver()}><div class="dp-drop">Drop files to upload</div></Show>

      <Show when={menu()} keyed>
        {(snap) => { const m = () => snap; const i = () => snap.item
          return (
            <>
              <div class="ctx-backdrop" onPointerDown={closeMenu} />
              <div class="ctx dp-ctx" style={{ left: `${Math.min(m().x, window.innerWidth - 200)}px`, top: `${Math.min(m().y, window.innerHeight - 320)}px` }}>
                <Show when={i().kind === 'design'}>
                  <button data-act="open" onClick={() => { closeMenu(); p.onOpen(i().id) }}>Open</button>
                  <button data-act="insert" onClick={() => { closeMenu(); p.onInsert(i().id) }}>Insert as component</button>
                </Show>
                <Show when={i().kind === 'folder'}><button data-act="open" onClick={() => { closeMenu(); p.setCwd(i().id) }}>Open folder</button></Show>
                <Show when={i().kind === 'asset' && ext(i().filename ?? i().name) === 'dxf'}><button data-act="dxf" onClick={() => { closeMenu(); void p.onInsertDxf(i()) }}>Insert as sketch</button></Show>
                <Show when={i().kind === 'asset'}><button data-act="download" onClick={() => { closeMenu(); window.open(remote()!.assetUrl(i().id), '_blank') }}>Open / download</button></Show>
                <button data-act="rename" onClick={() => { closeMenu(); setRenaming(i().id) }}>Rename</button>
                <Show when={remote()?.project}>
                  <hr />
                  <span class="ctx-label">Move to</span>
                  <Show when={i().parent !== null}><button data-act="move-root" onClick={() => void move(i(), null)}>Project root</button></Show>
                  <For each={folders().filter((f) => f.id !== i().parent && !descendants(i().id).has(f.id))}>{(f) => <button data-act="move" onClick={() => void move(i(), f.id)}>{f.name}</button>}</For>
                </Show>
                <hr />
                <button class="danger-item" data-act="delete" onClick={() => remove(i())}>Delete</button>
              </div>
            </>
          )
        }}
      </Show>
    </div>
  )
}

export type { TreeNode }
