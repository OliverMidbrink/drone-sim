/** Pure, serialisable document model. No kernel, no DOM. */

export type Plane = 'XY' | 'XZ' | 'YZ'
export type Vec3 = [number, number, number]

/**
 * A sketch is a small constraint graph. Points are entities of their own; lines and arcs refer to them by id, so
 * connected curves really share their end points. Profiles (closed loops) are derived from the curves, never stored.
 * Reserved ids (always present, fixed): `origin`, `axis-x`, `axis-y`.
 */
export type SketchEntity =
  | { id: string; kind: 'point'; x: number; y: number; construction?: boolean }
  | { id: string; kind: 'line'; a: string; b: string; construction?: boolean }
  /** Counter-clockwise arc about point `c`, from point `a` to point `b`. The radius is |c-a|; |c-b| is kept equal to it. */
  | { id: string; kind: 'arc'; c: string; a: string; b: string; construction?: boolean }
  | { id: string; kind: 'circle'; c: string; r: number; construction?: boolean }

export type ConstraintType =
  | 'coincident' // [point, point]
  | 'point-on' // [point, line|arc|circle]
  | 'horizontal' | 'vertical' // [line] or [point, point]
  | 'parallel' | 'perpendicular' // [line, line]
  | 'equal' // [line, line] or [circle|arc, circle|arc]
  | 'tangent' // [line, circle|arc] or [circle|arc, circle|arc]; `sign` +1 external / -1 internal for two circles
  | 'concentric' // [circle|arc, circle|arc]
  | 'midpoint' // [point, line]
  | 'symmetric' // [point, point, line]
  | 'fix' // [point] pinned at `at`
  // driving dimensions (value in mm, or degrees for 'angle')
  | 'distance' // [line] length, [point, point] or [point, line]
  | 'hdist' | 'vdist' // [point, point], signed by `sign`
  | 'radius' | 'diameter' // [circle|arc]
  | 'angle' // [line, line], signed by `sign`

export interface SketchConstraint {
  id: string
  type: ConstraintType
  refs: string[]
  value?: number
  sign?: 1 | -1
  at?: [number, number]
}

export const DIMENSION_TYPES: ConstraintType[] = ['distance', 'hdist', 'vdist', 'radius', 'diameter', 'angle']

/** Fields every feature shares. */
export interface FeatureBase {
  /** Suppressed features are skipped by the rebuild (like Fusion's "Suppress"); features that need them show a warning. */
  suppressed?: boolean
}

/** Position of a plane in space: origin, normal and x direction (y = dir × xdir). */
export interface Frame { origin: Vec3; dir: Vec3; xdir: Vec3 }

export interface SketchFeature extends FeatureBase {
  type: 'sketch'
  id: string
  name: string
  plane: Plane
  /** Offset of the sketch plane along its normal. */
  offset: number
  /** A construction plane (feature id) to sketch on. When set it replaces `plane` and `offset`. */
  planeRef?: string
  /** Transient: the resolved frame, filled in by `resolveFrames`. Never stored in a document. */
  frame?: Frame
  entities: SketchEntity[]
  constraints: SketchConstraint[]
  /** Hidden sketches are not drawn in the 3D view. Absent means visible. */
  visible?: boolean
}

export interface ExtrudeFeature extends FeatureBase {
  type: 'extrude'
  id: string
  name: string
  sketch: string
  distance: number
  /** new body, add to the previous body, or cut from it. */
  mode: 'new' | 'add' | 'cut'
}

export interface LoftFeature extends FeatureBase {
  type: 'loft'
  id: string
  name: string
  sketches: [string, string]
  mode: 'new' | 'add' | 'cut'
}

/** A closed region of a sketch, identified by a point inside it (sketch coordinates), so it survives sketch edits. */
export interface ProfileRef {
  sketch: string
  pt: [number, number]
}

/** A revolve axis: a line of a sketch (a drawn or construction line, or that sketch's X / Y axis), or a world axis. */
export type AxisRef =
  | { kind: 'sketch'; sketch: string; id: string }
  | { kind: 'world'; axis: 'X' | 'Y' | 'Z' }

export interface RevolveFeature extends FeatureBase {
  type: 'revolve'
  id: string
  name: string
  /** Picked profiles (regions). Empty or a missing axis means the feature is not complete yet. */
  profiles: ProfileRef[]
  axis?: AxisRef
  extent: 'full' | 'angle' | 'symmetric'
  /** Degrees; used by 'angle' and 'symmetric' (total sweep). */
  angle: number
  mode: 'new' | 'add' | 'cut'
}

export interface BooleanFeature extends FeatureBase {
  type: 'boolean'
  id: string
  name: string
  op: 'union' | 'cut' | 'common'
  /** Feature ids whose resulting bodies are combined (tool is consumed). */
  target: string
  tool: string
}

export interface ThreadFeature extends FeatureBase {
  type: 'thread'
  id: string
  name: string
  /** Body (feature id) the thread is added to. */
  target: string
  designation: string
  kind: 'male' | 'female'
  length: number
  /** Free/open end of the thread, and the direction into the material. */
  origin: Vec3
  dir: Vec3
  /** Radius of the cylinder the thread was applied to; it is resized to fit the screw. */
  existingRadius?: number
}

export interface DeleteFeatureFeature extends FeatureBase {
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

/** What a construction plane is measured from: a world plane, another construction plane or a sketch's plane. */
export type PlaneBase =
  | { kind: 'world'; plane: Plane }
  | { kind: 'plane'; id: string }
  | { kind: 'sketch'; id: string }

export type PlaneDef =
  /** Parallel to `base`, `distance` mm along its normal. */
  | { kind: 'offset'; base: PlaneBase; distance: number }
  /** `base` rotated by `angle` degrees about a world axis through the origin. */
  | { kind: 'angle'; base: PlaneBase; axis: 'X' | 'Y' | 'Z'; angle: number }
  /** Half way between two parallel planes. */
  | { kind: 'mid'; a: PlaneBase; b: PlaneBase }

export interface PlaneFeature extends FeatureBase {
  type: 'plane'
  id: string
  name: string
  def: PlaneDef
  /** Hidden planes are not drawn in the 3D view. Absent means visible. */
  visible?: boolean
}

export interface Placement {
  /** Millimetres. */
  pos: Vec3
  /** Degrees, applied about X, then Y, then Z. */
  rot: Vec3
}

/** An instance of another design (a component) placed in this one. */
export interface ComponentFeature extends FeatureBase {
  type: 'component'
  id: string
  name: string
  /** Id of the design that defines the component. */
  ref: string
  placement: Placement
  /** Grounded components are fixed in space; the first one is grounded automatically. */
  grounded?: boolean
  visible?: boolean
}

/** A solid read from an imported STEP file (an uploaded asset). */
export interface ImportStepFeature extends FeatureBase {
  type: 'import-step'
  id: string
  name: string
  asset: string
}

export type Feature =
  | SketchFeature | ExtrudeFeature | LoftFeature | RevolveFeature | BooleanFeature | ThreadFeature | DeleteFeatureFeature
  | PlaneFeature | ComponentFeature | ImportStepFeature

/** Part: sketches, bodies, operations. Assembly: placed components. Hybrid: both. */
export type DesignMode = 'part' | 'assembly' | 'hybrid'

export interface Document {
  version: 3
  name: string
  units: 'mm'
  mode: DesignMode
  /** User names of bodies, keyed by the feature that created the body (it survives later Add / Cut features). */
  bodyNames: Record<string, string>
  /** Bodies hidden in the 3D view (root feature ids). */
  hidden: string[]
  features: Feature[]
}

export const FORMAT_VERSION = 3

export function emptyDocument(name = 'Untitled', mode: DesignMode = 'part'): Document {
  return { version: FORMAT_VERSION, name, units: 'mm', mode, bodyNames: {}, hidden: [], features: [] }
}

/** Feature types that produce a body in the rebuild. */
export const BODY_FEATURES: readonly Feature['type'][] = ['extrude', 'revolve', 'loft', 'boolean', 'thread', 'delete-feature', 'component', 'import-step']
