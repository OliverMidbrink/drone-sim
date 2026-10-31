import { describe, it, expect } from 'vitest'
import { gridLevels, snapToGrid, formatMm } from '../src/grid'

describe('adaptive grid', () => {
  it('follows the 1-2-5 sequence as you zoom', () => {
    const steps = [0.05, 0.2, 1, 3, 8, 20, 60, 150, 600].map((px) => gridLevels(px).minor)
    // pixels per mm → mm between lines: 14px minimum
    expect(steps).toEqual([500, 100, 20, 5, 2, 1, 0.5, 0.1, 0.05])
  })
  it('keeps lines at least minPx apart and never much wider than 2.5x that', () => {
    for (let px = 0.01; px < 5000; px *= 1.37) {
      const { minor } = gridLevels(px, 14)
      expect(minor * px).toBeGreaterThanOrEqual(14 - 1e-6)
      expect(minor * px).toBeLessThan(14 * 2.5 + 1e-6)
    }
  })
  it('major lines are powers of ten and land on minor lines', () => {
    for (const px of [0.3, 1, 4, 9, 33, 120, 800]) {
      const { minor, major } = gridLevels(px)
      expect(Math.log10(major) % 1).toBeCloseTo(0, 9)
      expect(major).toBeGreaterThan(minor)
      expect(Math.round(major / minor) * minor).toBeCloseTo(major, 9)
    }
  })
  it('snaps to the step without float noise', () => {
    expect(snapToGrid([1.234, -2.71], 0.5)).toEqual([1, -2.5])
    expect(snapToGrid([0.31, 0.29], 0.1)).toEqual([0.3, 0.3])
    expect(snapToGrid([3.3, 4.4], 0)).toEqual([3.3, 4.4])
  })
  it('formats with digits matching the grid', () => {
    expect(formatMm(12.3456, 1)).toBe('12.3')
    expect(formatMm(12.3456, 0.01)).toBe('12.346')
    expect(formatMm(-0.0000001, 0.1)).toBe('0.00')
  })
})
