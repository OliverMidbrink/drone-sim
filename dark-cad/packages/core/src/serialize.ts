import { FORMAT_VERSION, type Document } from './model'
import { migrateSketchV1 } from './migrate'

/** Canonical JSON: stable key order, 2-space indent, trailing newline: diff-friendly for git. */
export function serialize(doc: Document): string {
  const sort = (v: any): any =>
    Array.isArray(v) ? v.map(sort)
      : v && typeof v === 'object' ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sort(v[k])]))
        : typeof v === 'number' ? Math.round(v * 1e9) / 1e9 : v
  return JSON.stringify(sort(doc), null, 2) + '\n'
}

type Migration = (raw: any) => any
/** migrations[n] upgrades version n → n+1. Add one entry whenever FORMAT_VERSION changes. */
const migrations: Record<number, Migration> = {
  2: (raw) => ({ ...raw, mode: 'part', bodyNames: {}, hidden: [] }),
  1: (raw) => ({ ...raw, features: !Array.isArray(raw.features) ? raw.features : raw.features.map((f: any) => (f?.type === 'sketch' ? { ...f, ...migrateSketchV1(f) } : f)) }),
}

export function parse(text: string): Document {
  let raw = JSON.parse(text)
  if (!raw || typeof raw !== 'object' || typeof raw.version !== 'number') throw new Error('not a DarkCAD document')
  if (raw.version > FORMAT_VERSION) throw new Error(`document is version ${raw.version}; this build reads up to ${FORMAT_VERSION}`)
  while (raw.version < FORMAT_VERSION) {
    const m = migrations[raw.version]
    if (!m) throw new Error(`no migration from version ${raw.version}`)
    raw = { ...m(raw), version: raw.version + 1 }
  }
  if (!Array.isArray(raw.features)) throw new Error('document has no feature list')
  raw.features = raw.features.map((f: any) => (
    f?.type === 'sketch' ? { ...f, entities: f.entities ?? [], constraints: f.constraints ?? [] }
    : f?.type === 'revolve' ? { ...f, profiles: Array.isArray(f.profiles) ? f.profiles : [], extent: f.extent ?? 'full', angle: f.angle ?? 360, mode: f.mode ?? 'new' }
    : f))
  // Older autosaves and hand-made files can lack a title; every design needs one to show in its tab.
  if (!['part', 'assembly', 'hybrid'].includes(raw.mode)) raw.mode = 'part'
  if (!raw.bodyNames || typeof raw.bodyNames !== 'object') raw.bodyNames = {}
  if (!Array.isArray(raw.hidden)) raw.hidden = []
  if (typeof raw.name !== 'string' || !raw.name.trim()) raw.name = 'Untitled'
  return raw as Document
}
