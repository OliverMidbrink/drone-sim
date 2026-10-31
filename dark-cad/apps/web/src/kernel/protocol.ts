import type { Document } from '@darkcad/core'
import type { Mesh } from '@darkcad/kernel'

export interface FaceInfo {
  surface: string
  centroid: [number, number, number]
  area: number
  cylinder?: { origin: [number, number, number]; dir: [number, number, number]; radius: number }
}

export type Request =
  | { type: 'build'; doc: Document; designId?: string; deps: Record<string, Document>; /** STEP files the worker has not seen yet. */ assets: Record<string, Uint8Array> }
  | { type: 'step' }
  | { type: 'faceInfo'; body: string; face: number }

export type Response =
  | { type: 'ready' }
  | { type: 'built'; seq: number; bodies: { id: string; root: string; mesh: Mesh }[]; errors: [string, string][]; ms: number }
  | { type: 'step'; bytes: Uint8Array }
  | { type: 'faceInfo'; info: FaceInfo }
  | { type: 'error'; message: string }
