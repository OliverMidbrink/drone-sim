import type { Document } from '@darkcad/core'
import type { Mesh } from '@darkcad/kernel'
import type { FaceInfo, Request, Response } from './protocol'

export interface BuildOutput { bodies: { id: string; root: string; mesh: Mesh }[]; errors: Map<string, string>; ms: number }

/** Typed wrapper around the kernel worker. One request in flight per kind; later builds supersede earlier ones. */
export class KernelClient {
  private worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
  private waiters: ((r: Response) => void)[] = []
  readonly ready: Promise<void>

  constructor() {
    this.ready = new Promise((res) => {
      const off = this.on((r) => { if (r.type === 'ready') { off(); res() } })
    })
    this.worker.onmessage = (ev: MessageEvent<Response>) => this.waiters.slice().forEach((w) => w(ev.data))
  }

  private on(fn: (r: Response) => void) {
    this.waiters.push(fn)
    return () => { this.waiters = this.waiters.filter((w) => w !== fn) }
  }

  private request<T extends Response['type']>(req: Request, want: T): Promise<Extract<Response, { type: T }>> {
    return new Promise((resolve, reject) => {
      const off = this.on((r) => {
        if (r.type === 'error') { off(); reject(new Error(r.message)) }
        else if (r.type === want) { off(); resolve(r as any) }
      })
      this.worker.postMessage(req)
    })
  }

  private chain: Promise<unknown> = Promise.resolve()
  /** The worker is single-threaded and replies in order; serialise calls so replies match requests. */
  private queued<T>(fn: () => Promise<T>): Promise<T> {
    const p = this.chain.then(fn, fn)
    this.chain = p.catch(() => undefined)
    return p
  }

  private sentAssets = new Set<string>()
  /** `assets` are STEP files by id; each is sent to the worker once and cached there. */
  async build(doc: Document, extra: { designId?: string; deps?: Record<string, Document>; assets?: Record<string, Uint8Array> } = {}): Promise<BuildOutput> {
    await this.ready
    const fresh: Record<string, Uint8Array> = {}
    for (const [id, bytes] of Object.entries(extra.assets ?? {})) if (!this.sentAssets.has(id)) fresh[id] = bytes
    const r = await this.queued(() => this.request({ type: 'build', doc, designId: extra.designId, deps: extra.deps ?? {}, assets: fresh }, 'built'))
    for (const id of Object.keys(fresh)) this.sentAssets.add(id)
    return { bodies: r.bodies, errors: new Map(r.errors), ms: r.ms }
  }

  async step(): Promise<Uint8Array> {
    await this.ready
    return (await this.queued(() => this.request({ type: 'step' }, 'step'))).bytes
  }

  async faceInfo(body: string, face: number): Promise<FaceInfo> {
    await this.ready
    return (await this.queued(() => this.request({ type: 'faceInfo', body, face }, 'faceInfo'))).info
  }
}
