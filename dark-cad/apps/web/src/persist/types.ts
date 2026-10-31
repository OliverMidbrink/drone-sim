import type { Document, DesignMode } from '@darkcad/core'

/** Shape of the server's manifest nodes (apps/server). */
export interface TreeNode {
  id: string
  kind: 'folder' | 'design' | 'asset'
  name: string
  parent: string | null
  mode?: DesignMode
  mime?: string
  filename?: string
  rev?: number
  updated: number
}
export interface Manifest { id: string; name: string; created: number; updated: number; rev: number; nodes: TreeNode[] }
export interface ProjectInfo { id: string; name: string; updated: number }
export interface StoredDesign { rev: number; updated: number; marker: number; doc: Document }
export interface VersionMeta { id: string; name: string; note: string; kind: 'manual' | 'auto'; created: number; rev: number }

/** What gets written for a design. */
export interface Payload { doc: Document; marker: number; parent?: string | null }

/** Saved = server has the latest. Saving = changes waiting or in flight. Offline = server unreachable (retrying). */
export type SaveState = 'saved' | 'saving' | 'offline' | 'conflict' | 'local'

export class ConflictError extends Error {
  constructor(readonly rev: number) { super('the design changed on the server') }
}
export class HttpError extends Error {
  constructor(readonly status: number, message: string) { super(message) }
}

/** The part of the remote store the sync queue needs; implemented by `RemotePersistence` and by test fakes. */
export interface RemotePort {
  putDesign(id: string, payload: Payload, ifMatch: number | '*' | undefined): Promise<{ rev: number; updated: number }>
}
