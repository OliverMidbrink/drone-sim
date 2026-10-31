/** opencascade.js names overloads Foo_1..Foo_N; construct the first one that accepts these args. */
export function make(oc: any, base: string, ...args: any[]): any {
  const names = [base, ...Array.from({ length: 80 }, (_, i) => `${base}_${i + 1}`)].filter((n) => oc[n])
  for (const n of names) {
    try { return new oc[n](...args) } catch (e: any) {
      if (!/invalid number of parameters|Cannot pass|Cannot convert|expected|no accessible constructor/i.test(String(e?.message ?? e))) throw e
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
