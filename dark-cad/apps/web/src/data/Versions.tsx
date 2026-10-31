import { For, Show, createResource, createSignal } from 'solid-js'
import { parse, type Document } from '@darkcad/core'
import { Icon } from '../icons'
import type { Workspace } from '../workspace'
import type { VersionMeta } from '../persist/types'

const when = (t: number) => new Date(t).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })

/** Named milestones of the open design, kept on the server next to the design. Restoring is an ordinary (undoable) edit. */
export function Versions(p: { workspace: Workspace; designId: string; onRestore: (doc: Document, marker: number) => void; onOpenAsNew: (doc: Document, name: string) => void; onClose: () => void; status: (m: string) => void }) {
  const ws = p.workspace
  const [name, setName] = createSignal('')
  const [note, setNote] = createSignal('')
  const [err, setErr] = createSignal('')
  const [busy, setBusy] = createSignal(false)
  const [list, { refetch }] = createResource<VersionMeta[]>(async () => {
    if (!ws.remote?.project) return []
    try { await ws.syncer?.flush(); setErr(''); return await ws.remote.listVersions(p.designId) } catch (e: any) { setErr(e.status === 404 ? 'This design is not saved to the server yet. Make an edit and try again in a moment.' : `Cannot load versions: ${e.message}`); return [] }
  })

  const run = async (fn: () => Promise<void>) => { setBusy(true); setErr(''); try { await fn() } catch (e: any) { setErr(e.message) } finally { setBusy(false) } }

  const save = () => run(async () => {
    await ws.syncer?.flush() // the server snapshots what it has, so send the latest first
    const v = await ws.remote!.createVersion(p.designId, name().trim() || `Version ${(list()?.filter((x) => x.kind === 'manual').length ?? 0) + 1}`, note().trim())
    setName(''); setNote(''); p.status(`Saved version "${v.name}"`)
    await refetch()
  })
  const fetchDoc = async (v: VersionMeta) => { const r = await ws.remote!.getVersion(p.designId, v.id); return { doc: parse(JSON.stringify(r.doc)), marker: r.marker } }
  const restore = (v: VersionMeta) => run(async () => { const { doc, marker } = await fetchDoc(v); p.onRestore(doc, marker); p.status(`Restored "${v.name}" (Ctrl+Z undoes it)`); p.onClose() })
  const openNew = (v: VersionMeta) => run(async () => { const { doc } = await fetchDoc(v); p.onOpenAsNew(doc, `${doc.name} (${v.name})`); p.onClose() })

  return (
    <>
      <div class="gallery-backdrop" onClick={p.onClose} />
      <div class="float versions" role="dialog" aria-label="Versions">
        <div class="float-head"><span>Versions</span><button class="icon-btn small" title="Close" onClick={p.onClose}><Icon name="close" size={14} /></button></div>
        <Show when={ws.remote?.project} fallback={<p class="dim">Versions are stored on the server. It is not reachable right now.</p>}>
          <div class="form">
            <input class="v-name" placeholder="Version name" value={name()} onInput={(e) => setName(e.currentTarget.value)} onKeyDown={(e) => { if (e.key === 'Enter') void save() }} />
            <input class="v-note" placeholder="Note (optional)" value={note()} onInput={(e) => setNote(e.currentTarget.value)} />
            <button class="primary v-save" disabled={busy()} onClick={() => void save()}><Icon name="tag" size={14} /> Save version</button>
          </div>
          <Show when={err()}><div class="error v-err">{err()}</div></Show>
          <ul class="v-list">
            <For each={list() ?? []}>{(v) => (
              <li data-kind={v.kind} data-name={v.name}>
                <span class="v-main"><b>{v.name}</b><small>{v.kind === 'auto' ? 'automatic' : 'saved'} · {when(v.created)}{v.note ? ` · ${v.note}` : ''}</small></span>
                <button class="small-btn" disabled={busy()} data-act="restore" onClick={() => void restore(v)}>Restore</button>
                <button class="small-btn" disabled={busy()} data-act="open-new" title="Open this version as a new design" onClick={() => void openNew(v)}>Open as new</button>
              </li>
            )}</For>
            <Show when={!list.loading && !(list() ?? []).length && !err()}><li class="none">No versions yet.</li></Show>
          </ul>
        </Show>
      </div>
    </>
  )
}
