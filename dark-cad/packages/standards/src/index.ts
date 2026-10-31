import bspp from '../data/bspp.json'
import bspt from '../data/bspt.json'
import isoMetric from '../data/iso-metric.json'

export interface StandardSize {
  designation: string
  pitch: number
  major: number
  tpi?: number
  tapDrill?: number
  gaugeLength?: number
}

export interface ThreadStandard {
  id: string
  name: string
  family: string
  profile: string
  angleDeg: number
  taper?: number
  source: string
  verified: boolean
  sizes: StandardSize[]
}

/** To add a standard: drop a JSON file in data/ and add it here. No other code changes. */
export const standards: ThreadStandard[] = [isoMetric, bspp, bspt]

export type Kind = 'male' | 'female'

export interface ThreadDims {
  designation: string
  standard: string
  kind: Kind
  pitch: number
  /** crest/major radius of the thread cut */
  rMajor: number
  /** root radius for a male thread; bore (crest) radius for a female thread */
  rMinor: number
  /** radius the plain cylinder is resized to before the thread is cut */
  rCylinder: number
  angleDeg: number
  taper: number
  verified: boolean
}

export function findSize(designation: string): { std: ThreadStandard; size: StandardSize } | undefined {
  const key = designation.replace(/\s+/g, '').toLowerCase()
  for (const std of standards) {
    const size = std.sizes.find((s) => s.designation.toLowerCase() === key)
    if (size) return { std, size }
  }
}

/** Thread depth h of the basic profile, as a multiple of the pitch. */
export function depthFactor(profile: string): number {
  switch (profile) {
    case 'iso-60': return 0.613435 // external: (17/24)·H with H = 0.866025·P → d − d3 = 1.22687·P
    case 'whitworth-55': return 0.640327
    default: throw new Error(`unknown thread profile ${profile}`)
  }
}

/** Tap drill diameter: the tabulated value, falling back to d − P for tables that omit it. */
export function tapDrill(size: StandardSize): number {
  return size.tapDrill ?? size.major - size.pitch
}

export function threadDims(designation: string, kind: Kind): ThreadDims {
  const hit = findSize(designation)
  if (!hit) throw new Error(`unknown thread ${designation}`)
  const { std, size } = hit
  const h = depthFactor(std.profile) * size.pitch
  const rMajor = size.major / 2
  const external = kind === 'male'
  // Female: the bore (thread crest) is the minor diameter D1; the groove is cut out to the major diameter.
  const d1 = std.profile === 'iso-60' ? size.major - 1.08253 * size.pitch : size.major - 2 * h
  return {
    designation: size.designation,
    standard: std.id,
    kind,
    pitch: size.pitch,
    rMajor,
    rMinor: external ? rMajor - h : d1 / 2,
    rCylinder: external ? rMajor : tapDrill(size) / 2,
    angleDeg: std.angleDeg,
    taper: std.taper ?? 0,
    verified: std.verified,
  }
}

export function listDesignations(): string[] {
  return standards.flatMap((s) => s.sizes.map((z) => z.designation))
}
