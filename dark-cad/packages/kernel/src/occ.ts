import { createRequire } from 'node:module'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

// Node loader for the prebuilt opencascade.js WASM (M0 spike only; the web/worker
// build and the trimmed custom build replace this behind KernelPort).
let cached: Promise<any> | undefined
export function loadOCC(): Promise<any> {
  cached ??= (async () => {
    const require = createRequire(import.meta.url)
    const dir = join(dirname(require.resolve('opencascade.js/package.json')), 'dist')
    // The emscripten glue is ESM-flavoured but expects CJS globals under Node.
    Object.assign(globalThis, { require, __dirname: dir })
    const mod = require(join(dir, 'opencascade.wasm.js')).default
    return new mod({ wasmBinary: readFileSync(join(dir, 'opencascade.wasm.wasm')), locateFile: (p: string) => join(dir, p) })
  })()
  return cached
}

/** opencascade.js names overloads Foo_1..Foo_N; construct the first one that accepts these args. */
export function make(oc: any, base: string, ...args: any[]): any {
  const names = [base, ...Array.from({ length: 80 }, (_, i) => `${base}_${i + 1}`)].filter((n) => oc[n])
  for (const n of names) {
    try { return new oc[n](...args) } catch (e: any) {
      if (!/invalid number of parameters|Cannot pass|expected|no accessible constructor/i.test(String(e?.message ?? e))) throw e
    }
  }
  throw new Error(`no overload of ${base} accepts ${args.length} args`)
}

/** Call the first overload obj.base / base_1..N that accepts these args. */
export function call(obj: any, base: string, ...args: any[]): any {
  const names = [base, ...Array.from({ length: 40 }, (_, i) => `${base}_${i + 1}`)].filter((n) => typeof obj[n] === 'function')
  for (const n of names) {
    try { return obj[n](...args) } catch (e: any) {
      if (!/invalid number of parameters|Cannot pass|expected|Cannot convert|is not a function/i.test(String(e?.message ?? e))) throw e
    }
  }
  throw new Error(`no overload of ${base} accepts ${args.length} args`)
}
