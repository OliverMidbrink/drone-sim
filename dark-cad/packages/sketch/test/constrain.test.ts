import { describe, it, expect } from 'vitest'
import { planConstraint, type PickedEntity } from '../src'

const P = (id: string): PickedEntity => ({ id, kind: 'point' })
const L = (id: string): PickedEntity => ({ id, kind: 'line' })
const A = (id: string): PickedEntity => ({ id, kind: 'arc' })
const C = (id: string): PickedEntity => ({ id, kind: 'circle' })

describe('planConstraint', () => {
  it('orders references the way the solver wants them', () => {
    expect(planConstraint('midpoint', [L('l'), P('p')])).toEqual({ status: 'ready', type: 'midpoint', refs: ['p', 'l'] })
    expect(planConstraint('symmetric', [L('l'), P('a'), P('b')])).toEqual({ status: 'ready', type: 'symmetric', refs: ['a', 'b', 'l'] })
    expect(planConstraint('tangent', [C('c'), L('l')])).toEqual({ status: 'ready', type: 'tangent', refs: ['l', 'c'] })
    expect(planConstraint('point-on', [A('a'), P('p')])).toEqual({ status: 'ready', type: 'point-on', refs: ['p', 'a'] })
  })
  it('asks for more picks while the selection can still grow into a match', () => {
    expect(planConstraint('parallel', [L('a')]).status).toBe('more')
    expect(planConstraint('symmetric', [P('a'), P('b')]).status).toBe('more')
    expect(planConstraint('symmetric', [L('l'), P('a')]).status).toBe('more')
    expect(planConstraint('coincident', [P('a')]).status).toBe('more')
  })
  it('horizontal and vertical take one line or two points', () => {
    expect(planConstraint('horizontal', [L('a')])).toEqual({ status: 'ready', type: 'horizontal', refs: ['a'] })
    expect(planConstraint('vertical', [P('a'), P('b')])).toEqual({ status: 'ready', type: 'vertical', refs: ['a', 'b'] })
    expect(planConstraint('horizontal', [P('a')]).status).toBe('more')
  })
  it('rejects nonsense combinations', () => {
    expect(planConstraint('parallel', [L('a'), C('c')]).status).toBe('invalid')
    expect(planConstraint('concentric', [L('a')]).status).toBe('invalid')
    expect(planConstraint('equal', [L('a'), C('c')]).status).toBe('invalid')
    expect(planConstraint('tangent', [L('a'), L('b')]).status).toBe('invalid')
  })
  it('equal works for lines and for round curves, mixed round kinds included', () => {
    expect(planConstraint('equal', [L('a'), L('b')]).status).toBe('ready')
    expect(planConstraint('equal', [A('a'), C('c')]).status).toBe('ready')
  })
  it('coincident on a point and a curve becomes point-on', () => {
    expect(planConstraint('coincident', [P('p'), L('l')])).toEqual({ status: 'ready', type: 'point-on', refs: ['p', 'l'] })
  })
  it('fix pins every point of the picked curves and skips the origin', () => {
    const r = planConstraint('fix', [L('l'), P('origin')], (id) => (id === 'l' ? ['a', 'b'] : []))
    expect(r).toEqual({ status: 'ready', type: 'fix', refs: ['a', 'b'] })
    expect(planConstraint('fix', [P('origin')]).status).toBe('invalid')
  })
})
