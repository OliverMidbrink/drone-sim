import { For, Show, createSignal, onCleanup } from 'solid-js'
import type { DocStore, Feature } from '@darkcad/core'
import { Icon } from '../icons'
import { ICON, TYPE_LABEL, dropIndex, health, nearestGap, summary } from './info'

export interface TimelineProps {
  store: DocStore
  features: Feature[]
  marker: number
  selected: string[]
  errors: Map<string, string>
  /** Select (ctrl = add to selection). */
  onSelect: (id: string, additive: boolean) => void
  /** Open the feature for editing (sketch mode for sketches, properties otherwise). */
  onEdit: (id: string) => void
  onDelete: (id: string) => void
}

/** Fusion-style history bar: one chip per feature, a draggable rollback marker, playback buttons and a context menu. */
export function Timeline(p: TimelineProps) {
  let strip!: HTMLDivElement
  const gaps: HTMLElement[] = []
  const [dragging, setDragging] = createSignal(false)
  const [hint, setHint] = createSignal<{ id: string; x: number; y: number } | undefined>()
  const [menu, setMenu] = createSignal<{ id: string; x: number; y: number } | undefined>()
  const [renaming, setRenaming] = createSignal<{ id: string; x: number; y: number } | undefined>()
  const [dropGap, setDropGap] = createSignal<number | undefined>()
  const n = () => p.features.length
  const byId = (id: string) => p.features.find((f) => f.id === id)

  // ---- marker dragging: the bar snaps to the gap nearest the pointer and rolls the model there while you drag ----
  function startDrag(e: PointerEvent) {
    e.preventDefault()
    setDragging(true)
    const move = (ev: PointerEvent) => {
      const centres = gaps.slice(0, n() + 1).map((g) => { const r = g.getBoundingClientRect(); return r.left + r.width / 2 })
      p.store.setMarker(nearestGap(centres, ev.clientX))
      // keep the marker in view while dragging towards the edge
      const r = strip.getBoundingClientRect()
      if (ev.clientX > r.right - 30) strip.scrollLeft += 14
      else if (ev.clientX < r.left + 30) strip.scrollLeft -= 14
    }
    // Window listeners: the marker element is re-created in each gap it passes, so it cannot hold pointer capture.
    const up = () => { setDragging(false); window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); window.removeEventListener('pointercancel', up) }
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up); window.addEventListener('pointercancel', up)
  }

  // ---- reordering by dragging a chip into a gap ----
  let dragId: string | undefined
  const gapFromEvent = (e: DragEvent, i: number) => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    return e.clientX < r.left + r.width / 2 ? i : i + 1
  }
  const drop = (gap: number) => {
    const id = dragId; dragId = undefined; setDropGap(undefined)
    if (!id) return
    const from = p.features.findIndex((f) => f.id === id)
    if (from >= 0 && dropIndex(from, gap) !== from) p.store.move(id, dropIndex(from, gap))
  }

  // ---- keyboard: arrows move the marker while the bar has focus ----
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'ArrowLeft') p.store.stepMarker(-1)
    else if (e.key === 'ArrowRight') p.store.stepMarker(1)
    else if (e.key === 'Home') p.store.setMarker(0)
    else if (e.key === 'End') p.store.setMarker(n())
    else return
    e.preventDefault()
  }

  const closeMenu = () => setMenu(undefined)
  window.addEventListener('pointerdown', closeMenu)
  onCleanup(() => window.removeEventListener('pointerdown', closeMenu))

  const Gap = (g: { i: number }) => (
    <div class="tl-gap" classList={{ drop: dropGap() === g.i }} data-gap={g.i} ref={(el) => (gaps[g.i] = el)}
      onClick={() => p.store.setMarker(g.i)}
      onDragOver={(e) => { if (dragId) { e.preventDefault(); setDropGap(g.i) } }} onDragLeave={() => setDropGap(undefined)} onDrop={(e) => { e.preventDefault(); drop(g.i) }}>
      <Show when={p.marker === g.i}>
        <div class="tl-marker" classList={{ drag: dragging() }} role="slider" tabindex="0" aria-label="Rollback marker" aria-valuemin={0} aria-valuemax={n()} aria-valuenow={p.marker}
          title="Drag to roll the model back or forward" onPointerDown={startDrag} onKeyDown={onKey} onClick={(e) => e.stopPropagation()}>
          <span class="tl-marker-grip" />
        </div>
      </Show>
    </div>
  )

  return (
    <div class="timeline" data-marker={p.marker}>
      <div class="tl-controls">
        <button class="icon-btn small" title="Roll back to the start (Home)" disabled={p.marker === 0} onClick={() => p.store.setMarker(0)}><Icon name="skipBack" size={14} /></button>
        <button class="icon-btn small" title="Step back (Left)" disabled={p.marker === 0} onClick={() => p.store.stepMarker(-1)}><Icon name="stepBack" size={14} /></button>
        <button class="icon-btn small" title="Step forward (Right)" disabled={p.marker >= n()} onClick={() => p.store.stepMarker(1)}><Icon name="stepFwd" size={14} /></button>
        <button class="icon-btn small" title="Roll forward to the end (End)" disabled={p.marker >= n()} onClick={() => p.store.setMarker(n())}><Icon name="skipFwd" size={14} /></button>
      </div>
      <div class="tl-strip" ref={strip} onScroll={() => setHint(undefined)}>
        <Gap i={0} />
        <For each={p.features}>{(f, i) => {
          const h = () => health(f, p.errors.get(f.id))
          return (<>
            <div class="tl-item" classList={{ sel: p.selected.includes(f.id), rolled: i() >= p.marker, err: h() === 'error', blocked: h() === 'blocked', off: h() === 'suppressed' }}
              role="button" tabindex="-1" data-id={f.id} data-health={h()} data-type={f.type} draggable={true}
              onClick={(e) => p.onSelect(f.id, e.ctrlKey || e.metaKey)} onDblClick={() => p.onEdit(f.id)}
              onContextMenu={(e) => { e.preventDefault(); p.onSelect(f.id, false); setMenu({ id: f.id, x: e.clientX, y: e.clientY }) }}
              onMouseEnter={(e) => { const r = e.currentTarget.getBoundingClientRect(); setHint({ id: f.id, x: r.left + r.width / 2, y: r.top }) }} onMouseLeave={() => setHint(undefined)}
              onDragStart={(e) => { dragId = f.id; e.dataTransfer?.setData('text/plain', f.id); setHint(undefined) }} onDragEnd={() => { dragId = undefined; setDropGap(undefined) }}
              onDragOver={(e) => { if (dragId && dragId !== f.id) { e.preventDefault(); setDropGap(gapFromEvent(e, i())) } }} onDrop={(e) => { e.preventDefault(); drop(gapFromEvent(e, i())) }}>
              <Icon name={ICON[f.type]} size={17} />
              <Show when={h() === 'error' || h() === 'blocked'}><span class="tl-badge" title={p.errors.get(f.id)}><Icon name="warning" size={11} /></span></Show>
              <Show when={h() === 'suppressed'}><span class="tl-slash" /></Show>
            </div>
            <Gap i={i() + 1} />
          </>)
        }}</For>
        <Show when={!n()}><span class="tl-empty">Operations you perform appear here, in order. Drag the marker to roll back.</span></Show>
      </div>
      <div class="tl-count">{p.marker}/{n()}</div>

      <Show when={hint() && byId(hint()!.id) && !menu() && !renaming()}>
        {(_) => { const f = () => byId(hint()!.id)!; const err = () => p.errors.get(f().id); const h = () => health(f(), err())
          return (
            <div class="tl-tip" classList={{ bad: h() === 'error' || h() === 'blocked' }} style={{ left: `${hint()!.x}px`, top: `${hint()!.y - 8}px` }}>
              <b>{f().name}</b> <small>{TYPE_LABEL[f().type]}</small>
              <div class="dim">{summary(f())}</div>
              <Show when={f().suppressed}><div class="note">Suppressed: skipped by the rebuild</div></Show>
              <Show when={err()}><div class="err"><Icon name="warning" size={12} /> {err()}</div></Show>
              <Show when={byId(f().id) && p.features.indexOf(f()) >= p.marker}><div class="note">Rolled back</div></Show>
            </div>
          )
        }}
      </Show>

      <Show when={menu()}>
        <div class="ctx" style={{ left: `${Math.min(menu()!.x, window.innerWidth - 180)}px`, top: `${Math.min(menu()!.y, window.innerHeight - 190)}px` }} onPointerDown={(e) => e.stopPropagation()}>
          {(() => { const f = byId(menu()!.id); if (!f) return null; const idx = p.features.indexOf(f); return (<>
            <button onClick={() => { closeMenu(); p.onEdit(f.id) }}>Edit</button>
            <button onClick={() => { const r = menu()!; closeMenu(); setRenaming({ id: f.id, x: r.x, y: r.y }) }}>Rename</button>
            <button onClick={() => { closeMenu(); p.store.suppress(f.id, !f.suppressed) }}>{f.suppressed ? 'Unsuppress' : 'Suppress'}</button>
            <button onClick={() => { closeMenu(); p.store.setMarker(idx + 1) }}>Roll back to here</button>
            <button class="danger-item" onClick={() => { closeMenu(); p.onDelete(f.id) }}>Delete</button>
          </>) })()}
        </div>
      </Show>

      <Show when={renaming()}>
        <input class="tl-rename" style={{ left: `${renaming()!.x}px`, top: `${renaming()!.y}px` }} value={byId(renaming()!.id)?.name ?? ''}
          ref={(el) => queueMicrotask(() => { el.focus(); el.select() })}
          onBlur={(e) => { const id = renaming()!.id; const v = e.currentTarget.value.trim(); setRenaming(undefined); if (v && byId(id)) p.store.update(id, { name: v }) }}
          onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); if (e.key === 'Escape') setRenaming(undefined) }} />
      </Show>
    </div>
  )
}
