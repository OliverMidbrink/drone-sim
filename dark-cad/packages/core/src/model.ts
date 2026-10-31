/** Pure, serialisable document model. No kernel, no DOM. */

export type Plane = 'XY' | 'XZ' | 'YZ'
export type Vec3 = [number, number, number]

export type SketchEntity =
  | { id: string; kind: 'rect'; x: number; y: number; w: number; h: number }
  | { id: string; kind: 'circle'; x: number; y: number; r: number }
  | { id: string; kind: 'polygon'; points: [number, number][] }

export interface SketchFeature {
  type: 'sketch'
  id: string
  name: string
  plane: Plane
  /** Offset of the sketch plane along its normal. */
  offset: number
  entities: SketchEntity[]
}

export interface ExtrudeFeature {
  type: 'extrude'
  id: string
  name: string
  sketch: string
  distance: number
  /** new body, add to the previous body, or cut from it. */
  mode: 'new' | 'add' | 'cut'
}

export interface LoftFeature {
  type: 'loft'
  id: string
  name: string
  sketches: [string, string]
  mode: 'new' | 'add' | 'cut'
}

export interface BooleanFeature {
  type: 'boolean'
  id: string
  name: string
  op: 'union' | 'cut' | 'common'
  /** Feature ids whose resulting bodies are combined (tool is consumed). */
  target: string
  tool: string
}

export interface ThreadFeature {
  type: 'thread'
  id: string
  name: string
  /** Body (feature id) the thread is added to. */
  target: string
  designation: string
  kind: 'male' | 'female'
  length: number
  origin: Vec3
  dir: Vec3
}

export interface DeleteFeatureFeature {
  type: 'delete-feature'
  id: string
  name: string
  target: string
  /** Face the user picked. Resolved at rebuild time by a geometric fingerprint, not by index. */
  face: FaceRef
}

export interface FaceRef {
  /** Fingerprint used to re-find the face after upstream edits. */
  surface: string
  centroid: Vec3
  area: number
}

export type Feature =
  | SketchFeature | ExtrudeFeature | LoftFeature | BooleanFeature | ThreadFeature | DeleteFeatureFeature

export interface Document {
  version: 1
  name: string
  units: 'mm'
  features: Feature[]
}

export const FORMAT_VERSION = 1

export function emptyDocument(name = 'Untitled'): Document {
  return { version: FORMAT_VERSION, name, units: 'mm', features: [] }
}
