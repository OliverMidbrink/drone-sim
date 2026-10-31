import type { Feature } from '@darkcad/core'
import type { IconName } from '../icons'

export const ICON: Record<Feature['type'], IconName> = {
  sketch: 'sketch', extrude: 'extrude', loft: 'loft', boolean: 'combine', thread: 'thread', 'delete-feature': 'deleteFeature', revolve: 'revolve',
  plane: 'plane', component: 'component', 'import-step': 'stepImport',
}

export const TYPE_LABEL: Record<Feature['type'], string> = {
  sketch: 'Sketch', extrude: 'Extrude', loft: 'Loft', boolean: 'Combine', thread: 'Thread', 'delete-feature': 'Delete feature', revolve: 'Revolve',
  plane: 'Construction plane', component: 'Component', 'import-step': 'STEP import',
}

const n = (v: number) => String(Math.round(v * 1000) / 1000)

/** One line describing what a feature does, for tooltips. */
export function summary(f: Feature): string {
  switch (f.type) {
    case 'sketch': return `${f.planeRef ? 'on a construction plane' : `on ${f.plane}${f.offset ? ` + ${n(f.offset)} mm` : ''}`} · ${f.entities.filter((e) => e.kind !== 'point').length} curve(s)`
    case 'extrude': return `${n(f.distance)} mm · ${f.mode}`
    case 'revolve': return `${f.extent === 'full' ? 'full' : `${n(f.angle)}°`} · ${f.mode}`
    case 'loft': return `between two sketches · ${f.mode}`
    case 'boolean': return f.op
    case 'thread': return `${f.designation} ${f.kind} · ${n(f.length)} mm`
    case 'delete-feature': return 'removes a boss, pocket or hole'
    case 'plane': return f.def.kind === 'offset' ? `offset ${n(f.def.distance)} mm` : f.def.kind === 'angle' ? `${n(f.def.angle)}° about ${f.def.axis}` : 'midplane'
    case 'component': return `at ${f.placement.pos.map(n).join(', ')}${f.grounded ? ' · grounded' : ''}`
    case 'import-step': return 'solid from a STEP file'
  }
}

export type Health = 'ok' | 'error' | 'blocked' | 'suppressed'

/** A feature that failed only because something before it did (or is switched off) is "blocked", not broken itself. */
export function health(f: Feature, err: string | undefined): Health {
  if (f.suppressed) return 'suppressed'
  if (!err) return 'ok'
  return /^Blocked by |is suppressed$/.test(err) ? 'blocked' : 'error'
}

/** Index of the gap (0..n) whose centre x is closest to `x`. */
export function nearestGap(centres: number[], x: number): number {
  let best = 0, d = Infinity
  centres.forEach((c, i) => { const k = Math.abs(c - x); if (k < d) { d = k; best = i } })
  return best
}

/** Where a feature ends up when dropped into gap `gap` of a list in which it currently sits at `from`. */
export const dropIndex = (from: number, gap: number) => (gap > from ? gap - 1 : gap)
