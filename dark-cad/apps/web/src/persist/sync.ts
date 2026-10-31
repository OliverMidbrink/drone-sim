import { ConflictError, type Payload, type RemotePort, type SaveState } from './types'

export interface SyncOptions {
  /** Quiet time after the last edit before it is sent. */
  debounceMs?: number
  /** Retry delays after a network failure; the last one repeats. */
  backoffMs?: number[]
}

/**
 * Debounced, serial upload queue with retry. Edits mark a design dirty; after a quiet period the latest state is PUT
 * with the revision the client last saw. Network failures keep the design queued and retry with backoff (state
 * "offline"); a 409 parks that design as a conflict until the user picks a side.
 */
export class Syncer {
  state: SaveState = 'saved'
  /** Designs the server changed behind our back (id → server revision). */
  readonly conflicts = new Map<string, number>()
  onSynced?: (id: string, rev: number) => void

  private revs = new Map<string, number>()
  private pending = new Set<string>()
  private version = new Map<string, number>()
  private forced = new Set<string>()
  private running?: Promise<void>
  private timer?: ReturnType<typeof setTimeout>
  private retry?: ReturnType<typeof setTimeout>
  private attempt = 0
  private listeners = new Set<() => void>()
  private debounce: number
  private backoff: number[]

  constructor(private port: RemotePort, private source: (id: string) => Payload | undefined, opts: SyncOptions = {}) {
    this.debounce = opts.debounceMs ?? 1000
    this.backoff = opts.backoffMs ?? [1000, 2000, 5000, 10000, 30000]
  }

  subscribe(fn: () => void) { this.listeners.add(fn); return () => { this.listeners.delete(fn) } }
  private set(s: SaveState) { if (s !== this.state) { this.state = s; this.listeners.forEach((f) => f()) } }

  rev(id: string) { return this.revs.get(id) }
  setRev(id: string, rev: number | undefined) { if (rev === undefined) this.revs.delete(id); else this.revs.set(id, rev) }
  get pendingIds() { return [...this.pending] }
  isPending(id: string) { return this.pending.has(id) }

  /** The design changed locally. */
  markDirty(id: string) {
    this.pending.add(id)
    this.version.set(id, (this.version.get(id) ?? 0) + 1)
    if (this.state !== 'offline' && this.state !== 'conflict') this.set('saving')
    clearTimeout(this.timer)
    this.timer = setTimeout(() => { void this.pump() }, this.debounce)
  }

  /** Forget a design (deleted). */
  drop(id: string) { this.pending.delete(id); this.conflicts.delete(id); this.revs.delete(id); this.forced.delete(id); this.settle() }

  /** Send everything now and wait for it. */
  async flush() { clearTimeout(this.timer); clearTimeout(this.retry); await this.pump() }

  /** Server said the design moved on: stop retrying it until `resolve` is called. */
  flagConflict(id: string, serverRev: number) { this.conflicts.set(id, serverRev); this.settle() }

  /** 'mine' overwrites the server copy; 'theirs' discards our pending changes (the caller loads the server copy). */
  resolve(id: string, choice: 'mine' | 'theirs') {
    const serverRev = this.conflicts.get(id)
    this.conflicts.delete(id)
    if (choice === 'mine') { this.forced.add(id); this.pending.add(id); this.version.set(id, (this.version.get(id) ?? 0) + 1); void this.pump() }
    else { this.pending.delete(id); if (serverRev !== undefined) this.revs.set(id, serverRev); this.settle() }
  }

  /** The caller could not reach the server at all (e.g. while connecting). */
  setOffline(on: boolean) { if (on) this.set('offline'); else if (this.state === 'offline') this.settle() }

  /** Network came back (or the user asked): try again right away. */
  retryNow() { clearTimeout(this.retry); void this.pump() }

  private settle() {
    this.set(this.conflicts.size ? 'conflict' : this.pending.size ? (this.state === 'offline' ? 'offline' : 'saving') : 'saved')
  }

  private pump(): Promise<void> {
    if (this.running) return this.running
    this.running = this.run().finally(() => { this.running = undefined })
    return this.running
  }

  private async run() {
    while (true) {
      const id = [...this.pending].find((i) => !this.conflicts.has(i))
      if (!id) break
      const payload = this.source(id)
      if (!payload) { this.pending.delete(id); continue } // design is gone
      const v = this.version.get(id) ?? 0
      const force = this.forced.has(id)
      this.set(this.state === 'offline' ? 'offline' : 'saving')
      try {
        const r = await this.port.putDesign(id, payload, force ? '*' : this.revs.get(id))
        this.forced.delete(id)
        this.revs.set(id, r.rev)
        if ((this.version.get(id) ?? 0) === v) this.pending.delete(id)
        this.attempt = 0
        this.onSynced?.(id, r.rev)
      } catch (e) {
        if (e instanceof ConflictError) { this.conflicts.set(id, e.rev); continue }
        const delay = this.backoff[Math.min(this.attempt++, this.backoff.length - 1)]
        this.set('offline')
        clearTimeout(this.retry)
        this.retry = setTimeout(() => { void this.pump() }, delay)
        return
      }
    }
    this.settle()
    if (this.state !== 'offline') this.set(this.conflicts.size ? 'conflict' : this.pending.size ? 'saving' : 'saved')
  }
}
