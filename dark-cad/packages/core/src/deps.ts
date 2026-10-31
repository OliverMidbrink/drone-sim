import type { Document } from './model'

export interface Deps {
  /** Every design reachable through component references (not including the root). */
  docs: Record<string, Document>
  /** Referenced designs that could not be loaded. */
  missing: string[]
  /** Asset ids used by STEP imports anywhere in the tree (root included). */
  assets: string[]
}

/**
 * Collects the designs a design uses as components, transitively. Loops are not followed: the rebuild reports them
 * on the component that closes the loop.
 */
export function collectDeps(rootId: string, root: Document, getDoc: (id: string) => Document | undefined): Deps {
  const docs: Record<string, Document> = {}, missing = new Set<string>(), assets = new Set<string>()
  const visit = (d: Document) => {
    for (const f of d.features) {
      if (f.suppressed) continue
      if (f.type === 'import-step') assets.add(f.asset)
      if (f.type !== 'component' || f.ref === rootId || f.ref in docs || missing.has(f.ref)) continue
      const sub = getDoc(f.ref)
      if (!sub) { missing.add(f.ref); continue }
      docs[f.ref] = sub
      visit(sub)
    }
  }
  visit(root)
  return { docs, missing: [...missing], assets: [...assets] }
}

/** Would placing `ref` inside `host` make the designs refer to each other in a circle? */
export function wouldLoop(hostId: string, ref: string, getDoc: (id: string) => Document | undefined): boolean {
  if (hostId === ref) return true
  const seen = new Set<string>()
  const reaches = (id: string): boolean => {
    if (id === hostId) return true
    if (seen.has(id)) return false
    seen.add(id)
    const d = getDoc(id)
    return !!d && d.features.some((f) => f.type === 'component' && reaches(f.ref))
  }
  return reaches(ref)
}
