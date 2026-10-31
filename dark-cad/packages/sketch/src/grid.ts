import type { Pt } from './geom'

const MANTISSAS = [1, 2, 5]

/**
 * Adaptive grid: the smallest 1-2-5 step (…0.1, 0.2, 0.5, 1, 2, 5, 10…) whose lines are at least `minPx` apart on screen.
 * `pxPerMm` is the zoom. The major line is the next power of ten, so a major line always falls on a minor one.
 */
export function gridLevels(pxPerMm: number, minPx = 14): { minor: number; major: number } {
  if (!(pxPerMm > 0)) return { minor: 1, major: 10 }
  let k = Math.floor(Math.log10(minPx / pxPerMm)) - 1
  for (let guard = 0; guard < 40; guard++, k++) {
    for (const m of MANTISSAS) {
      const step = m * 10 ** k
      if (step * pxPerMm >= minPx - 1e-9) return { minor: clean(step), major: clean(10 ** (Math.floor(Math.log10(step * 1.0000001)) + 1)) }
    }
  }
  return { minor: 1, major: 10 }
}

/** Strip float noise from a grid value (0.30000000000000004 → 0.3). */
export const clean = (v: number) => Math.round(v * 1e9) / 1e9

export const snapToGrid = (p: Pt, step: number): Pt => (step > 0 ? [clean(Math.round(p[0] / step) * step), clean(Math.round(p[1] / step) * step)] : p)

/** Format a distance with just enough digits for the current grid step. */
export function formatMm(v: number, step = 0.01): string {
  const digits = Math.max(0, Math.min(6, Math.ceil(-Math.log10(step)) + 1))
  return (Math.abs(v) < 5 * 10 ** -(digits + 1) ? 0 : v).toFixed(digits)
}
