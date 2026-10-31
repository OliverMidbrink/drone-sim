import { createRequire } from 'node:module'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

// Node loader for the prebuilt opencascade.js WASM (tests and headless tools). The browser build is
// loaded by the worker instead; everything else only sees the `oc` handle.
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
