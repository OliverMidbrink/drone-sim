import type { SketchConstraint, SketchEntity } from './model'

/** v1 sketches held closed shapes (rect / circle / polygon). v2 holds points, lines, arcs, circles and constraints. */
export function migrateSketchV1(raw: any): { entities: SketchEntity[]; constraints: SketchConstraint[] } {
  const entities: SketchEntity[] = []
  const constraints: SketchConstraint[] = []
  let n = 0
  const id = () => `e${++n}`
  const cid = () => `c${constraints.length + 1}`
  const pt = (x: number, y: number) => { const p = id(); entities.push({ id: p, kind: 'point', x, y }); return p }
  const loop = (ps: string[]) => ps.map((a, i) => { const l = id(); entities.push({ id: l, kind: 'line', a, b: ps[(i + 1) % ps.length] }); return l })
  for (const e of Array.isArray(raw?.entities) ? raw.entities : []) {
    if (e.kind === 'rect') {
      const ls = loop([pt(e.x, e.y), pt(e.x + e.w, e.y), pt(e.x + e.w, e.y + e.h), pt(e.x, e.y + e.h)])
      ;(['horizontal', 'vertical', 'horizontal', 'vertical'] as const).forEach((type, i) => constraints.push({ id: cid(), type, refs: [ls[i]] }))
    } else if (e.kind === 'circle') {
      entities.push({ id: id(), kind: 'circle', c: pt(e.x, e.y), r: e.r })
    } else if (e.kind === 'polygon') {
      loop((e.points as [number, number][]).map(([x, y]) => pt(x, y)))
    }
  }
  return { entities, constraints }
}
