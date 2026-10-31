import type { ConstraintType } from '@darkcad/core'

export type PickKind = 'point' | 'line' | 'arc' | 'circle'
export interface PickedEntity { id: string; kind: PickKind }

/** What a constraint needs, as alternatives (each a list of entity classes, in the order the solver wants them). */
const PATTERNS: Partial<Record<ConstraintType, string[][]>> = {
  coincident: [['point', 'point']],
  'point-on': [['point', 'curve']],
  horizontal: [['line'], ['point', 'point']],
  vertical: [['line'], ['point', 'point']],
  parallel: [['line', 'line']],
  perpendicular: [['line', 'line']],
  equal: [['line', 'line'], ['round', 'round']],
  tangent: [['line', 'round'], ['round', 'round']],
  concentric: [['round', 'round']],
  midpoint: [['point', 'line']],
  symmetric: [['point', 'point', 'line']],
}

const isClass = (k: PickKind, c: string) => (c === 'curve' ? k !== 'point' : c === 'round' ? k === 'arc' || k === 'circle' : c === k)

function permutations<T>(xs: T[]): T[][] {
  if (xs.length <= 1) return [xs]
  return xs.flatMap((x, i) => permutations([...xs.slice(0, i), ...xs.slice(i + 1)]).map((p) => [x, ...p]))
}

export type Plan =
  | { status: 'ready'; type: ConstraintType; refs: string[] }
  /** Compatible so far, pick more. */
  | { status: 'more' }
  | { status: 'invalid'; reason: string }

/**
 * Decide how a constraint applies to a set of picked entities: which order the solver expects the references in,
 * whether more picks are needed, or whether the combination makes no sense.
 * `pointsOf` lists the points a curve is built from (used by Fix).
 */
export function planConstraint(type: ConstraintType, picks: PickedEntity[], pointsOf: (id: string) => string[] = () => []): Plan {
  if (!picks.length) return { status: 'more' }
  if (type === 'fix') {
    const ids = [...new Set(picks.flatMap((p) => (p.kind === 'point' ? [p.id] : pointsOf(p.id))))].filter((id) => id !== 'origin' && !id.startsWith('$'))
    return ids.length ? { status: 'ready', type, refs: ids } : { status: 'invalid', reason: 'The origin and the axes are already fixed' }
  }
  if (type === 'coincident' && picks.length === 2 && picks.some((p) => p.kind === 'point') && picks.some((p) => p.kind !== 'point')) return planConstraint('point-on', picks, pointsOf)
  const patterns = PATTERNS[type]
  if (!patterns) return { status: 'invalid', reason: `${type} is not a geometric constraint` }
  let partial = false
  for (const pat of patterns) {
    if (picks.length > pat.length) continue
    for (const perm of permutations(picks)) {
      if (!perm.every((p, i) => isClass(p.kind, pat[i]))) continue
      if (perm.length === pat.length) return { status: 'ready', type, refs: perm.map((p) => p.id) }
      partial = true
    }
    // a partial match must also work for some arrangement of the remaining slots: try any subset of slots
    if (picks.length < pat.length && !partial) {
      const slots = pat.map((_, i) => i)
      const fits = (chosen: number[], rest: PickedEntity[]): boolean => !rest.length || chosen.some((s, ci) => isClass(rest[0].kind, pat[s]) && fits(chosen.filter((_, k) => k !== ci), rest.slice(1)))
      if (fits(slots, picks)) partial = true
    }
  }
  if (partial) return { status: 'more' }
  const same = picks.length === 1 ? picks[0].kind : picks.map((p) => p.kind).join(' + ')
  return { status: 'invalid', reason: `${type} does not apply to ${same}` }
}

/** Arity hint for the UI (how many picks the shortest pattern wants). */
export const constraintArity = (type: ConstraintType): number => (type === 'fix' ? 1 : Math.min(...(PATTERNS[type]?.map((p) => p.length) ?? [1])))
