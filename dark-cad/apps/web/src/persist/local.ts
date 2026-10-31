import { parse, serialize, type Document } from '@darkcad/core'

export interface DesignMeta {
  id: string
  name: string
  updated: number
  /** -1 while the design has not been downloaded yet. */
  features: number
  mode?: 'part' | 'assembly' | 'hybrid'
  /** Folder in the project tree on the server (null = project root). */
  parent?: string | null
  /** Last revision known to be on the server. */
  rev?: number
  /** Local changes the server does not have yet. Survives reloads. */
  unsynced?: boolean
  /** Rollback marker (see Store). */
  marker?: number
}
export interface Index { open: string[]; active: string; library: DesignMeta[] }

const INDEX_KEY = 'darkcad:workspace:v1'
const LEGACY_KEY = 'darkcad:autosave'
const docKey = (id: string) => `darkcad:design:${id}`

function safeStorage(): Storage | undefined {
  try { return globalThis.localStorage } catch { return undefined }
}

/** Offline cache and fallback: everything the workspace knows, in localStorage (synchronous). */
export class LocalPersistence {
  constructor(readonly storage: Storage | undefined = safeStorage()) {}

  readIndex(): Index | undefined {
    try {
      const raw = this.storage?.getItem(INDEX_KEY)
      return raw ? (JSON.parse(raw) as Index) : undefined
    } catch { return undefined }
  }
  writeIndex(idx: Index) { this.storage?.setItem(INDEX_KEY, JSON.stringify(idx)) }
  readDoc(id: string): Document | undefined {
    try { const raw = this.storage?.getItem(docKey(id)); return raw ? parse(raw) : undefined } catch { return undefined }
  }
  writeDoc(id: string, doc: Document) { this.storage?.setItem(docKey(id), serialize(doc)) }
  removeDoc(id: string) { try { this.storage?.removeItem(docKey(id)) } catch { /* nothing to clean up */ } }
  readLegacy(): Document | undefined {
    try { const raw = this.storage?.getItem(LEGACY_KEY); return raw ? parse(raw) : undefined } catch { return undefined }
  }
}
