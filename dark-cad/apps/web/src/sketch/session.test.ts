import { describe, expect, it } from 'vitest'
import { emptyModel, profiles, type Pt, type SketchModel } from '@darkcad/sketch'
import { Session, type Host } from './session'

/** A headless host: 10 px per mm, screen = plane scaled (no flip needed for logic). */
function setup() {
  let model: SketchModel = emptyModel()
  const host: Host = {
    model: () => model,
    commit: (m) => { model = m },
    scale: () => 10,
    toLocal: (x, y) => [x / 10, y / 10],
    toScreen: (p) => [p[0] * 10, p[1] * 10],
    canvasOrigin: () => ({ left: 0, top: 0 }),
    notify: () => {},
    status: () => {},
    selectionChanged: () => {},
  }
  const s = new Session(host)
  const none = { shift: false, ctrl: false, alt: false }
  const click = (p: Pt, mods = none) => { s.pointerMove(p[0] * 10, p[1] * 10, mods, 0); s.pointerDown(p[0] * 10, p[1] * 10, mods, 1); s.pointerUp(p[0] * 10, p[1] * 10, mods) }
  const hover = (p: Pt) => s.pointerMove(p[0] * 10, p[1] * 10, none, 0)
  const key = (k: string, shiftKey = false) => s.keyDown({ key: k, ctrlKey: false, metaKey: false, shiftKey, altKey: false }, () => {})
  const type = (txt: string) => { for (const c of txt) key(c) }
  return { s, click, hover, key, type, get model() { return model }, set model(m: SketchModel) { model = m } }
}

const kinds = (m: SketchModel, k: string) => m.entities.filter((e) => e.kind === k).length

describe('sketch session', () => {
  it('draws a rectangle with horizontal and vertical constraints', () => {
    const t = setup()
    t.s.setTool('rect')
    t.click([-10, -5]); t.click([10, 5])
    expect(kinds(t.model, 'line')).toBe(4)
    expect(profiles(t.model.entities, t.model.constraints).regions.length).toBe(1)
    expect(t.s.analysis().dof).toBe(4)
  })

  it('chains lines and closes a triangle onto the first point', () => {
    const t = setup()
    t.s.setTool('line')
    t.click([0, 0]); t.click([20, 0]); t.click([10, 15]); t.click([0, 0])
    expect(kinds(t.model, 'line')).toBe(3)
    expect(profiles(t.model.entities, t.model.constraints).regions.length).toBe(1)
  })

  it('typed length and angle place a line exactly', () => {
    const t = setup()
    t.s.setTool('line')
    t.click([0, 0])
    t.hover([10, 3])
    for (const c of '25') t.key(c)
    t.key('Tab')
    for (const c of '90') t.key(c)
    t.key('Enter')
    const pts = t.model.entities.filter((e) => e.kind === 'point') as { x: number; y: number }[]
    const end = pts[pts.length - 1]
    expect(end.x).toBeCloseTo(0, 6)
    expect(end.y).toBeCloseTo(25, 6)
  })

  it('a slot snapped to the origin and an axis is fully constrained by two dimensions', () => {
    const t = setup()
    t.s.setTool('slot')
    t.click([0, 0]); t.click([30, 0]); t.click([30, 6])
    expect(kinds(t.model, 'arc')).toBe(2)
    // centre on the origin (pinned) and the other centre on the X axis: length and width are left
    expect(t.s.analysis().dof).toBe(2)
    t.s.setTool('dimension')
    t.click([15, 6])
    expect(t.s.valueInput?.label).toBe('Length')
    for (const c of '40') t.key(c)
    t.key('Enter')
    t.click([-6, 0])
    expect(t.s.valueInput?.label).toBe('Radius')
    t.key('4')
    t.key('Enter')
    expect(t.s.analysis().dof).toBe(0)
    expect(t.s.statusText()).toBe('Fully constrained')
  })

  it('fix and horizontal constraints reduce the degrees of freedom of a free slot', () => {
    const t = setup()
    t.s.setTool('slot')
    t.click([10, 10]); t.click([40, 14]); t.click([40, 20])
    expect(t.s.analysis().dof).toBe(5)
    t.s.setTool('constrain', 'fix'); t.click([10, 10])
    t.s.setTool('constrain', 'horizontal'); t.click([10, 10]); t.click([40, 14])
    expect(t.s.analysis().dof).toBe(2)
  })

  it('offsets a rectangle outwards and keeps both loops', () => {
    const t = setup()
    t.s.setTool('rect')
    t.click([0, 0]); t.click([20, 10])
    t.s.setTool('offset')
    t.click([10, 0]) // bottom edge selects the chain
    t.hover([10, -3])
    for (const c of '2') t.key(c)
    t.key('Enter')
    expect(kinds(t.model, 'line')).toBeGreaterThanOrEqual(8)
    expect(profiles(t.model.entities, t.model.constraints).regions.length).toBeGreaterThanOrEqual(1)
  })

  it('delete removes dependent geometry', () => {
    const t = setup()
    t.s.setTool('circle')
    t.click([0, 0]); t.click([6, 0]) // grid step at 10 px/mm is 2 mm
    expect(kinds(t.model, 'circle')).toBe(1)
    t.s.setTool('select')
    t.click([6, 0])
    t.key('Delete')
    expect(kinds(t.model, 'circle')).toBe(0)
  })

  describe('typed shape fields', () => {
    const circles = (m: SketchModel) => m.entities.filter((e) => e.kind === 'circle') as { r: number; construction?: boolean }[]

    it('shows the live pointer value greyed until something is typed', () => {
      const t = setup()
      t.s.setTool('rect')
      t.click([0, 0]); t.hover([14, 6])
      const [w, h] = t.s.fields
      expect([w.label, h.label]).toEqual(['Width', 'Height'])
      expect([w.text, w.live, h.live]).toEqual(['', '14', '6'])
      t.type('9')
      expect(t.s.fields[0].text).toBe('9')
      expect(t.s.fields[1].live).toBe('6')
    })

    it('a circle takes a diameter and makes a driving diameter dimension', () => {
      const t = setup()
      t.s.setTool('circle')
      t.click([0, 0]); t.hover([4, 0])
      expect(t.s.fields[0].label).toBe('Diameter')
      expect(t.s.fields[0].live).toBe('8')
      t.type('30'); t.key('Enter')
      const c = circles(t.model)[0]
      expect(c.r).toBeCloseTo(15, 6)
      const dim = t.model.constraints.find((k) => k.type === 'diameter')
      expect(dim?.value).toBeCloseTo(30, 6)
    })

    it('a circle can be sized by radius instead', () => {
      const t = setup()
      t.s.opts.circleDim = 'radius'
      t.s.setTool('circle')
      t.click([0, 0]); t.hover([4, 0])
      expect(t.s.fields[0].label).toBe('Radius')
      t.type('7'); t.key('Enter')
      expect(circles(t.model)[0].r).toBeCloseTo(7, 6)
      expect(t.model.constraints.some((k) => k.type === 'radius' && k.value === 7)).toBe(true)
    })

    it('a polygon is sized by outer diameter (corners) or inner diameter (across flats)', () => {
      const n = 6, k = Math.cos(Math.PI / n)
      const outer = setup()
      outer.s.setTool('polygon')
      outer.click([0, 0]); outer.hover([5, 1])
      expect(outer.s.fields.map((f) => f.label)).toEqual(['Outer Ø', 'Inner Ø', 'Sides', 'Rotation'])
      outer.type('20'); outer.key('Enter')
      expect(circles(outer.model).find((c) => c.construction)!.r).toBeCloseTo(10, 6)

      const inner = setup()
      inner.s.setTool('polygon')
      inner.click([0, 0]); inner.hover([5, 1])
      inner.key('Tab'); inner.type('20')
      // typing the inner diameter updates the outer one live
      expect(Number(inner.s.fields[0].live)).toBeCloseTo(20 / k, 2)
      inner.key('Enter')
      expect(circles(inner.model).find((c) => c.construction)!.r).toBeCloseTo(10 / k, 6)
      expect(inner.model.constraints.some((c) => c.type === 'diameter')).toBe(true)
    })

    it('a polygon takes sides and rotation', () => {
      const t = setup()
      t.s.setTool('polygon')
      t.click([0, 0]); t.hover([5, 1])
      t.type('10'); t.key('Tab'); t.key('Tab'); t.type('8'); t.key('Tab'); t.type('0'); t.key('Enter')
      expect(t.model.entities.filter((e) => e.kind === 'line').length).toBe(8)
      const pts = t.model.entities.filter((e) => e.kind === 'point') as { x: number; y: number }[]
      expect(pts.some((p) => Math.abs(p.x - 5) < 1e-6 && Math.abs(p.y) < 1e-6)).toBe(true) // a corner at rotation 0
    })

    it('Tab moves between width and height and Shift+Tab goes back', () => {
      const t = setup()
      t.s.setTool('rect')
      t.click([0, 0]); t.hover([14, 6])
      t.type('20'); t.key('Tab'); t.type('8')
      expect(t.s.activeField).toBe(1)
      t.key('Tab', true)
      expect(t.s.activeField).toBe(0)
      t.key('Enter')
      const pts = t.model.entities.filter((e) => e.kind === 'point') as { x: number; y: number }[]
      expect(Math.max(...pts.map((p) => p.x))).toBeCloseTo(20, 6)
      expect(Math.max(...pts.map((p) => p.y))).toBeCloseTo(8, 6)
    })

    it('typed width and height become dimensions: a centred rectangle on a fixed centre is fully constrained', () => {
      const t = setup()
      t.s.opts.rect = 'center'
      t.s.setTool('rect')
      t.click([0, 0]); t.hover([5, 3])
      t.type('20'); t.key('Tab'); t.type('10'); t.key('Enter')
      expect(t.model.constraints.filter((c) => c.type === 'hdist' || c.type === 'vdist').map((c) => c.value).sort()).toEqual([10, 20])
      expect(t.s.analysis().dof).toBe(0)
      expect(t.s.statusText()).toBe('Fully constrained')
    })

    it('a rectangle without typed values gets no dimensions', () => {
      const t = setup()
      t.s.setTool('rect')
      t.click([0, 0]); t.click([10, 6])
      expect(t.model.constraints.some((c) => c.type === 'hdist' || c.type === 'vdist')).toBe(false)
    })

    it('slot: typed length and angle place the second centre, then a typed width finishes it', () => {
      const t = setup()
      t.s.setTool('slot')
      t.click([0, 0]); t.hover([10, 2])
      expect(t.s.fields.map((f) => f.label)).toEqual(['Length', 'Angle'])
      t.type('40'); t.key('Tab'); t.type('0'); t.key('Enter')
      expect(t.s.clicks.length).toBe(2)
      expect(t.s.fields.map((f) => f.label)).toEqual(['Width'])
      t.hover([20, 3]); t.type('10'); t.key('Enter')
      expect(kinds(t.model, 'arc')).toBe(2)
      expect(t.model.constraints.some((c) => c.type === 'distance' && c.value === 40)).toBe(true)
      expect(t.model.constraints.some((c) => c.type === 'diameter' && c.value === 10)).toBe(true)
      const xs = (t.model.entities.filter((e) => e.kind === 'point') as { x: number }[]).map((p) => p.x)
      expect(Math.max(...xs)).toBeCloseTo(40, 6)
    })

    it('centre arc: typed radius then sweep', () => {
      const t = setup()
      t.s.opts.arc = 'center'
      t.s.setTool('arc')
      t.click([0, 0]); t.hover([6, 0])
      expect(t.s.fields[0].label).toBe('Radius')
      t.type('10'); t.key('Enter')
      expect(t.s.clicks.length).toBe(2)
      t.hover([0, 10]); t.type('90'); t.key('Enter')
      const arc = t.model.entities.find((e) => e.kind === 'arc')!
      expect(arc.kind).toBe('arc')
      const pts = new Map((t.model.entities.filter((e) => e.kind === 'point') as { id: string; x: number; y: number }[]).map((p) => [p.id, p]))
      const end = pts.get((arc as { b: string }).b)!
      expect(end.x).toBeCloseTo(0, 5)
      expect(end.y).toBeCloseTo(10, 5)
      expect(t.model.constraints.some((c) => c.type === 'radius' && c.value === 10)).toBe(true)
    })
  })
})
