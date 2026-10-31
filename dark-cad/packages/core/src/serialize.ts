import { FORMAT_VERSION, type Document } from './model'

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
const migrations: Record<number, Migration> = {}

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
  return raw as Document
}
