import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { emptyDocument } from '@darkcad/core'
import { Syncer } from './sync'
import { ConflictError, type Payload, type RemotePort } from './types'

class FakePort implements RemotePort {
  calls: { id: string; ifMatch: number | '*' | undefined; payload: Payload }[] = []
  rev = new Map<string, number>()
  failNext = 0
  conflictNext = 0
  async putDesign(id: string, payload: Payload, ifMatch: number | '*' | undefined) {
    this.calls.push({ id, ifMatch, payload })
    if (this.failNext > 0) { this.failNext--; throw new TypeError('network down') }
    if (this.conflictNext > 0) { this.conflictNext--; throw new ConflictError(9) }
    const rev = (this.rev.get(id) ?? 0) + 1
    this.rev.set(id, rev)
    return { rev, updated: Date.now() }
  }
}

const doc = (name: string) => emptyDocument(name)

describe('Syncer', () => {
  let port: FakePort, docs: Map<string, Payload>, s: Syncer
  beforeEach(() => {
    vi.useFakeTimers()
    port = new FakePort(); docs = new Map()
    docs.set('a', { doc: doc('A'), marker: 0 }); docs.set('b', { doc: doc('B'), marker: 0 })
    s = new Syncer(port, (id) => docs.get(id), { debounceMs: 100, backoffMs: [1000, 2000] })
  })
  afterEach(() => { vi.useRealTimers() })

  it('debounces edits into a single upload of the latest state', async () => {
    s.markDirty('a'); s.markDirty('a')
    docs.set('a', { doc: doc('A2'), marker: 0 })
    s.markDirty('a')
    expect(s.state).toBe('saving')
    await vi.advanceTimersByTimeAsync(99)
    expect(port.calls).toHaveLength(0)
    await vi.advanceTimersByTimeAsync(10)
    expect(port.calls).toHaveLength(1)
    expect(port.calls[0].payload.doc.name).toBe('A2')
    expect(s.state).toBe('saved')
  })

  it('sends the revision it last saw with every later upload', async () => {
    s.markDirty('a'); await vi.advanceTimersByTimeAsync(150)
    s.markDirty('a'); await vi.advanceTimersByTimeAsync(150)
    expect(port.calls.map((c) => c.ifMatch)).toEqual([undefined, 1])
    expect(s.rev('a')).toBe(2)
  })

  it('goes offline on a network error, retries with backoff and recovers', async () => {
    port.failNext = 2
    s.markDirty('a')
    await vi.advanceTimersByTimeAsync(150)
    expect(s.state).toBe('offline')
    await vi.advanceTimersByTimeAsync(1000)
    expect(s.state).toBe('offline') // second failure
    expect(port.calls).toHaveLength(2)
    await vi.advanceTimersByTimeAsync(2000)
    expect(port.calls).toHaveLength(3)
    expect(s.state).toBe('saved')
  })

  it('keeps edits made while offline and sends the newest state when back', async () => {
    port.failNext = 1
    s.markDirty('a'); await vi.advanceTimersByTimeAsync(150)
    docs.set('a', { doc: doc('later'), marker: 0 }); s.markDirty('a')
    await vi.advanceTimersByTimeAsync(5000)
    expect(port.calls.at(-1)!.payload.doc.name).toBe('later')
    expect(s.state).toBe('saved')
  })

  it('an edit during an upload is not lost', async () => {
    let release!: () => void
    const slow: RemotePort = { putDesign: (id, p, m) => new Promise((res) => { release = () => res({ rev: 1, updated: 0 }); void m; void id; void p }) }
    const s2 = new Syncer(slow, (id) => docs.get(id), { debounceMs: 10 })
    s2.markDirty('a'); await vi.advanceTimersByTimeAsync(20)
    s2.markDirty('a') // while the first is in flight
    release(); await vi.advanceTimersByTimeAsync(0)
    expect(s2.isPending('a')).toBe(true)
  })

  it('a 409 parks the design as a conflict; the others still save', async () => {
    port.conflictNext = 1
    s.markDirty('a'); s.markDirty('b')
    await vi.advanceTimersByTimeAsync(150)
    expect(s.state).toBe('conflict')
    expect([...s.conflicts.keys()]).toEqual(['a'])
    expect(port.calls.map((c) => c.id)).toEqual(['a', 'b'])
    expect(s.rev('b')).toBe(1)
  })

  it('"keep mine" overwrites with If-Match: *, "use theirs" drops the pending change', async () => {
    port.conflictNext = 2
    s.markDirty('a'); s.markDirty('b'); await vi.advanceTimersByTimeAsync(150)
    expect(s.conflicts.size).toBe(2)
    s.resolve('a', 'mine'); await vi.advanceTimersByTimeAsync(0)
    expect(port.calls.at(-1)).toMatchObject({ id: 'a', ifMatch: '*' })
    s.resolve('b', 'theirs')
    expect(s.isPending('b')).toBe(false)
    expect(s.rev('b')).toBe(9)
    expect(s.state).toBe('saved')
  })

  it('skips designs that were deleted meanwhile', async () => {
    s.markDirty('a'); docs.delete('a')
    await vi.advanceTimersByTimeAsync(150)
    expect(port.calls).toHaveLength(0)
    expect(s.state).toBe('saved')
  })

  it('flush() sends right away', async () => {
    s.markDirty('a'); await s.flush()
    expect(port.calls).toHaveLength(1)
    expect(s.state).toBe('saved')
  })
})
