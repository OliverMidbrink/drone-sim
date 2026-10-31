// Electron shell: loads the exact same Vite bundle as the browser build, plus native file dialogs.
const { app, BrowserWindow, dialog, ipcMain, net, protocol } = require('electron')
const { readFile, writeFile } = require('node:fs/promises')
const path = require('node:path')
const { pathToFileURL } = require('node:url')

const DIST = path.join(__dirname, '..', 'web', 'dist')
const smoke = process.argv.includes('--smoke')
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png' }

// Serve over app:// (not file://): module workers and streaming WASM compile need a real, secure origin.
protocol.registerSchemesAsPrivileged([{ scheme: 'app', privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true } }])

function registerFiles() {
  ipcMain.handle('save', async (_e, { name, data }) => {
    const ext = path.extname(name).slice(1)
    const r = await dialog.showSaveDialog({ defaultPath: name, filters: ext ? [{ name: ext.toUpperCase(), extensions: [ext] }] : [] })
    if (r.canceled || !r.filePath) return false
    await writeFile(r.filePath, typeof data === 'string' ? data : Buffer.from(data))
    return true
  })
  ipcMain.handle('open', async (_e, accept) => {
    const exts = String(accept).split(',').map((s) => s.trim().replace(/^\./, '')).filter(Boolean)
    const r = await dialog.showOpenDialog({ properties: ['openFile'], filters: [{ name: 'Supported', extensions: exts }] })
    if (r.canceled || !r.filePaths[0]) return undefined
    const bytes = await readFile(r.filePaths[0])
    return { name: path.basename(r.filePaths[0]), bytes: new Uint8Array(bytes), text: bytes.toString('utf8') }
  })
}

async function createWindow() {
  const win = new BrowserWindow({
    width: 1500, height: 920, show: !smoke, backgroundColor: '#14161b', title: 'DarkCAD',
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, sandbox: false },
  })
  win.setMenuBarVisibility(false)
  if (smoke) win.webContents.on('console-message', (_e, level, message) => { if (level >= 2) console.log('RENDERER', message.slice(0, 300)) })
  await win.loadURL('app://darkcad/index.html')
  if (smoke) {
    const t0 = Date.now()
    const poll = setInterval(async () => {
      const text = await win.webContents.executeJavaScript(`document.querySelector('.status span')?.textContent ?? ''`)
      const desktop = await win.webContents.executeJavaScript(`typeof window.darkcadDesktop?.save === 'function'`)
      if (text === 'Ready' || text.startsWith('Rebuilt')) { clearInterval(poll); console.log(`DESKTOP_SMOKE_OK status="${text}" nativeFileDialogs=${desktop}`); app.quit() }
      else if (Date.now() - t0 > 90_000) { clearInterval(poll); console.log(`DESKTOP_SMOKE_TIMEOUT status="${text}"`); app.exit(1) }
    }, 500)
  }
}

app.whenReady().then(() => {
  protocol.handle('app', async (req) => {
    const url = new URL(req.url)
    let rel = decodeURIComponent(url.pathname)
    if (rel === '/' || rel === '') rel = '/index.html'
    const file = path.normalize(path.join(DIST, rel))
    if (!file.startsWith(DIST)) return new Response('forbidden', { status: 403 })
    const res = await net.fetch(pathToFileURL(file).toString())
    const headers = new Headers(res.headers)
    headers.set('content-type', MIME[path.extname(file)] ?? 'application/octet-stream')
    return new Response(res.body, { status: res.status, headers })
  })
  registerFiles()
  createWindow()
})
app.on('window-all-closed', () => app.quit())
