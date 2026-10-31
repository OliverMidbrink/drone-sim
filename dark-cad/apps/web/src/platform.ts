/** Thin file I/O port. The web build uses the browser; the desktop shell can replace it via window.darkcadDesktop. */
export interface Platform {
  save(name: string, data: string | Uint8Array, mime: string): Promise<void>
  open(accept: string): Promise<{ name: string; text: string; bytes: Uint8Array } | undefined>
}

const web: Platform = {
  async save(name, data, mime) {
    const blob = new Blob([data as BlobPart], { type: mime })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = name
    a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 10_000)
  },
  open(accept) {
    return new Promise((resolve) => {
      const input = document.createElement('input')
      input.type = 'file'
      input.accept = accept
      input.onchange = async () => {
        const f = input.files?.[0]
        if (!f) return resolve(undefined)
        const bytes = new Uint8Array(await f.arrayBuffer())
        resolve({ name: f.name, bytes, text: new TextDecoder().decode(bytes) })
      }
      input.oncancel = () => resolve(undefined)
      input.click()
    })
  },
}

export const platform: Platform = (globalThis as any).darkcadDesktop ?? web
