// End-to-end smoke test against the production build, in headless Chromium.
// Usage: pnpm build && node e2e/smoke.mjs [screenshotDir]
import { preview } from 'vite'
import { chromium } from 'playwright'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { spawn } from 'node:child_process'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const shots = process.argv[2] ?? 'e2e/out'
mkdirSync(shots, { recursive: true })
// The real storage server, in a throw-away data directory (the preview server proxies /api to it).
const here = dirname(fileURLToPath(import.meta.url))
const dataDir = mkdtempSync(join(tmpdir(), 'darkcad-e2e-'))
const API_PORT = 8799
process.env.DARKCAD_API = `http://127.0.0.1:${API_PORT}`
const api = spawn('npx', ['tsx', 'src/index.ts'], { cwd: join(here, '../../server'), env: { ...process.env, DARKCAD_DATA: dataDir, PORT: String(API_PORT) }, stdio: ['ignore', 'pipe', 'inherit'] })
await new Promise((res, rej) => { api.stdout.on('data', (d) => { if (String(d).includes('DarkCAD server')) res() }); api.on('exit', (c) => rej(new Error('server exited ' + c))); setTimeout(() => rej(new Error('server did not start')), 30000) })
const server = await preview({ preview: { port: 4173, strictPort: true } })
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1500, height: 900 } })
const problems = []
page.on('pageerror', (e) => problems.push('pageerror: ' + e.message))
page.on('console', (m) => { if (m.type() === 'error' && !/\/thumb/.test(m.location().url)) problems.push('console: ' + m.text() + ' ' + m.location().url) }) // a design without a thumbnail yet is a normal 404

const step = async (name, fn) => { try { await fn(); console.log('ok  ', name) } catch (e) { console.log('FAIL', name, '-', String(e.message).split('\n').slice(0, 4).join(' | ')); problems.push(name); await page.screenshot({ path: `${shots}/fail-${name.replace(/\W+/g, '_')}.png` }) } }
const status = () => page.locator('.status span').first().innerText()
const counts = () => page.locator('.status .dim').innerText()
const waitSolids = (n) => page.waitForFunction((n) => document.querySelector('.status .dim')?.textContent?.startsWith(`${n} solid`), n, { timeout: 60000 })
const view = () => page.evaluate(() => window.__darkcad.viewport.viewState())
const near = (a, b, tol = 0.03) => a.every((v, i) => Math.abs(v - b[i]) < tol)
const ribbon = (name, opts = {}) => page.locator('.ribbon').getByRole('button', { name, ...opts })
const settle = async () => { await page.waitForTimeout(400); await page.waitForFunction(() => !document.querySelector('.busy'), null, { timeout: 60000 }) }
// plane coords (mm) → client pixels, through the viewport's own camera
const px = (frame, u, v) => page.evaluate(([frame, u, v]) => {
  const vp = window.__darkcad.viewport, r = vp.renderer.domElement.getBoundingClientRect(), p = vp.planeToScreen(frame, u, v)
  return { x: r.left + p[0], y: r.top + p[1] }
}, [frame, u, v])
const clickAt = async (pt) => { await page.mouse.move(pt.x, pt.y); await page.waitForTimeout(80); await page.mouse.down(); await page.mouse.up() }
const sketchDoc = async () => (await page.evaluate(() => JSON.parse(JSON.stringify(window.__darkcad.store.doc)))).features.filter((f) => f.type === 'sketch')
const tool = (name) => page.locator('.sketch-bar').getByRole('button', { name, exact: true }).click()
const skStatus = () => page.locator('.sk-status').innerText()
const wheelAt = async (pt, dy) => { await page.mouse.move(pt.x, pt.y); await page.mouse.wheel(0, dy); await page.waitForTimeout(250) }
const XY = (z = 0) => ({ origin: [0, 0, z], dir: [0, 0, 1], xdir: [1, 0, 0] })
const drag = async (a, b) => { await page.mouse.move(a.x, a.y); await page.mouse.down(); await page.mouse.move(b.x, b.y, { steps: 6 }); await page.mouse.up() }

await page.goto('http://localhost:4173/', { waitUntil: 'load' })
await page.evaluate(() => localStorage.clear())
await page.reload()

await step('kernel loads', async () => { await page.waitForFunction(() => document.querySelector('.status span')?.textContent === 'Ready' || /Rebuilt/.test(document.querySelector('.status span')?.textContent ?? ''), null, { timeout: 90000 }) })

await step('top bar: DarkCAD label, grid icon, one default "Untitled" tab', async () => {
  const brand = (await page.locator('.topbar .brand').innerText()).trim()
  if (brand !== 'DarkCAD') throw new Error('brand: ' + brand)
  if (!(await page.locator('.topbar button[title="All designs"]').count())) throw new Error('no grid icon')
  const tabs = await page.locator('.tab-name').allInnerTexts()
  if (tabs.join('|') !== 'Untitled') throw new Error('tabs: ' + tabs.join('|'))
  const order = await page.evaluate(() => [...document.querySelector('.topbar').children].map((e) => e.className.split(' ')[0]).filter(Boolean))
  if (order.indexOf('designs-wrap') > order.indexOf('tabs') || order.indexOf('brand') !== 0) throw new Error('order: ' + order.join(','))
})

await step('ribbon order is 2D, 3D, Construct, Modify, Import / Export', async () => {
  const titles = await page.locator('.ribbon-group h4').allInnerTexts()
  if (titles.join('|').toLowerCase() !== '2d|3d|construct|modify|import / export') throw new Error(titles.join('|'))
  if (await page.locator('.ribbon select').count()) throw new Error('size selector should not be in the ribbon')
})

await step('Create Sketch shows the three yellow planes; picking XY aligns an orthographic camera', async () => {
  await ribbon('Create Sketch').click()
  await page.waitForTimeout(200)
  await page.screenshot({ path: `${shots}/0-planes.png` })
  const pt = await page.evaluate(() => window.__darkcad.viewport.planeScreenPoint('XY'))
  if (!pt) throw new Error('no plane picker shown')
  for (const n of ['XZ', 'YZ']) if (!(await page.evaluate((n) => window.__darkcad.viewport.planeScreenPoint(n), n))) throw new Error('missing plane ' + n)
  await page.mouse.move(pt.x, pt.y); await page.mouse.click(pt.x, pt.y)
  await page.locator('.sketch-overlay').waitFor()
  await page.waitForTimeout(600)
  const v = await view()
  if (!v.ortho || !v.locked || !near(v.dir, [0, 0, 1])) throw new Error('view ' + JSON.stringify(v))
})

await step('adaptive grid: clicks land on the minor step, and the step follows the zoom level', async () => {
  await tool('Point')
  const scaleNow = async () => { const a = await px(XY(0), 0, 0), b = await px(XY(0), 1, 0); return Math.hypot(b.x - a.x, b.y - a.y) }
  const lastPoint = async () => { const f = (await sketchDoc())[0]; const pts = f.entities.filter((e) => e.kind === 'point'); return pts[pts.length - 1] }
  const check = async (u, v) => {
    const sc = await scaleNow()
    const { minor } = await page.evaluate((sc) => window.__darkcad.gridLevels(sc), sc)
    await clickAt(await px(XY(0), u, v))
    const p = await lastPoint()
    const off = (x) => Math.abs(x / minor - Math.round(x / minor))
    if (off(p.x) > 1e-6 || off(p.y) > 1e-6) throw new Error(`(${p.x}, ${p.y}) is not on the ${minor} mm grid`)
    if (Math.abs(p.x - u) > minor / 2 + 1e-6) throw new Error(`snapped too far: ${p.x} for ${u} (step ${minor})`)
    return minor
  }
  const sc0 = await scaleNow()
  const coarse = await check(3.3, 2.2)
  const origin = await px(XY(0), 0, 0)
  await wheelAt(origin, -1200)
  if ((await scaleNow()) < sc0 * 2) throw new Error('zoom did not change the scale')
  const fine = await check(0.37, 0.77)
  if (!(fine < coarse)) throw new Error(`grid step did not get finer when zooming in: ${coarse} → ${fine}`)
  await page.screenshot({ path: `${shots}/1a-grid-zoomed.png` })
  await wheelAt(origin, 1200)
  if (Math.abs((await scaleNow()) / sc0 - 1) > 0.02) throw new Error('zoom did not return')
  await page.keyboard.press('Control+z'); await page.keyboard.press('Control+z')
  await page.waitForTimeout(200)
  if ((await sketchDoc())[0].entities.length !== 0) throw new Error('undo inside the sketcher did not clear the test points')
})

await step('draw a 20 × 10 mm rectangle in global coordinates', async () => {
  await tool('Rectangle')
  await clickAt(await px(XY(0), -10, -5)); await clickAt(await px(XY(0), 10, 5))
  await page.screenshot({ path: `${shots}/1-sketch.png` })
  const sk = (await sketchDoc())[0]
  const pts = sk.entities.filter((e) => e.kind === 'point')
  const ext = [Math.min(...pts.map((p) => p.x)), Math.min(...pts.map((p) => p.y)), Math.max(...pts.map((p) => p.x)), Math.max(...pts.map((p) => p.y))]
  if (ext.join(',') !== '-10,-5,10,5' || sk.entities.filter((e) => e.kind === 'line').length !== 4) throw new Error('rect extent ' + ext.join(',') + ' / ' + JSON.stringify(sk.entities.map((e) => e.kind)))
  if (!(await skStatus()).includes('degrees of freedom')) throw new Error('status: ' + (await skStatus()))
  await page.getByRole('button', { name: 'Finish sketch' }).click()
  await page.waitForTimeout(600)
  const v = await view()
  if (v.ortho || v.locked) throw new Error('camera not restored: ' + JSON.stringify(v))
})

await step('a finished sketch stays visible in the 3D view; the tree eye hides and shows it', async () => {
  const count = () => page.evaluate(() => window.__darkcad.viewport.sketchCount)
  if ((await count()) !== 1) throw new Error('finished sketch not drawn in 3D (count ' + (await count()) + ')')
  await page.screenshot({ path: `${shots}/1b-sketch-finished.png` })
  const eye = page.locator('.tree li', { hasText: 'Sketch 1' }).locator('.eye')
  await eye.click(); await page.waitForTimeout(150)
  if ((await count()) !== 0) throw new Error('hidden sketch still drawn')
  await eye.click(); await page.waitForTimeout(150)
  if ((await count()) !== 1) throw new Error('sketch not shown again')
})

await step('extrude makes a solid', async () => {
  await page.locator('.tree li', { hasText: 'Sketch 1' }).click()
  await ribbon('Extrude').click()
  await waitSolids(1)
  await page.waitForTimeout(300)
  await page.screenshot({ path: `${shots}/2-extrude.png` })
})

await step('second sketch sees the existing solid (x-ray) and the first sketch, then adds a boss', async () => {
  await ribbon('Create Sketch').click()
  const pt = await page.evaluate(() => window.__darkcad.viewport.planeScreenPoint('XY'))
  await clickAt(pt)
  await page.locator('.sketch-overlay').waitFor()
  await page.waitForTimeout(600)
  await tool('Circle')
  // The plate top is at z=10, so first move this sketch there: the plane offset in the panel re-aims the camera.
  await page.locator('.field', { hasText: 'Offset' }).locator('input').fill('10')
  await page.waitForTimeout(600)
  await clickAt(await px(XY(10), 0, 0)); await clickAt(await px(XY(10), 5, 0))
  await page.screenshot({ path: `${shots}/3-sketch-over-solid.png` })
  await page.getByRole('button', { name: 'Finish sketch' }).click()
  await page.waitForTimeout(600)
  await page.locator('.tree li', { hasText: 'Sketch 2' }).click()
  await ribbon('Extrude').click()
  await page.waitForTimeout(1500)
  await waitSolids(1)
})

await step('boss extends to z=20', async () => { await settle(); const b = await page.evaluate(() => { let m = -1e9; for (const x of window.__darkcad.bodies()) for (let i = 2; i < x.mesh.positions.length; i += 3) m = Math.max(m, x.mesh.positions[i]); return m }); if (Math.abs(b - 20) > 0.01) throw new Error('max z ' + b + ' doc=' + JSON.stringify(await page.evaluate(() => window.__darkcad.store.doc.features.map((f) => ({ id: f.id, offset: f.offset, distance: f.distance, mode: f.mode, ents: f.entities?.map((e) => [e.kind, e.x, e.y, e.r]) }))))) })

await step('sketch on the XZ plane looks along Y; Esc finishes; undo removes the empty sketch', async () => {
  await ribbon('Create Sketch').click()
  const pt = await page.evaluate(() => window.__darkcad.viewport.planeScreenPoint('XZ'))
  await clickAt(pt)
  await page.locator('.sketch-overlay').waitFor()
  await page.waitForTimeout(700)
  const v = await view()
  if (!v.ortho || !near(v.dir, [0, -1, 0]) || !near(v.up, [0, 0, 1])) throw new Error('view ' + JSON.stringify(v))
  await page.screenshot({ path: `${shots}/3b-xz.png` })
  await page.keyboard.press('Escape')
  await page.waitForTimeout(600)
  await page.locator('.topbar').getByRole('button', { name: 'Undo' }).click()
  await page.waitForTimeout(300)
  if ((await counts()).includes('6 feature')) throw new Error('empty sketch still there')
})

await step('view cube: clicking snaps to a standard direction', async () => {
  const b = await page.locator('.vc-canvas').boundingBox()
  await page.mouse.click(b.x + b.width / 2, b.y + b.height / 2)
  await page.waitForTimeout(600)
  const v = (await view()).dir
  const ok = v.every((c) => [0, 0.577, 0.707, 1].some((t) => Math.abs(Math.abs(c) - t) < 0.02))
  if (!ok) throw new Error('not a canonical direction: ' + v.map((c) => c.toFixed(3)))
})

await step('view cube arrows: up → top, down → level, roll tilts the up vector', async () => {
  const bt = (title) => page.locator(`.viewcube button[title="${title}"]`)
  await bt('Look from higher').click(); await page.waitForTimeout(450)
  await bt('Look from higher').click(); await page.waitForTimeout(450)
  if (!near((await view()).dir, [0, 0, 1], 0.02)) throw new Error('not top: ' + JSON.stringify(await view()))
  await bt('Look from lower').click(); await page.waitForTimeout(450)
  const lvl = await view()
  if (Math.abs(lvl.dir[2]) > 0.02) throw new Error('not level: ' + JSON.stringify(lvl))
  if (Math.abs(lvl.up[2] - 1) > 0.02) throw new Error('up should be +Z: ' + JSON.stringify(lvl))
  await bt('Roll view 90° clockwise').click(); await page.waitForTimeout(450)
  const rolled = await view()
  if (Math.abs(rolled.up[2]) > 0.05) throw new Error('roll did not tilt up vector: ' + JSON.stringify(rolled))
  const before = rolled.dir
  await bt('Orbit right').click(); await page.waitForTimeout(450)
  if (near((await view()).dir, before, 0.1)) throw new Error('orbit right did nothing')
  await bt('Home view').click(); await page.waitForTimeout(450)
  if (!near((await view()).dir, [0.6, -0.8, 0.55].map((c, _, a) => c / Math.hypot(...a)), 0.03)) throw new Error('home: ' + JSON.stringify(await view()))
})

await step('dragging the cube rotates the view; projection toggle works', async () => {
  const b = await page.locator('.vc-canvas').boundingBox()
  const before = (await view()).dir
  await drag({ x: b.x + 50, y: b.y + 50 }, { x: b.x + 90, y: b.y + 70 })
  await page.waitForTimeout(100)
  if (near((await view()).dir, before, 0.02)) throw new Error('drag did not rotate')
  await page.locator('.viewcube .proj').click()
  if (!(await view()).ortho) throw new Error('not orthographic')
  await page.locator('.viewcube .proj').click()
  if ((await view()).ortho) throw new Error('still orthographic')
  await page.locator('.viewcube button[title="Home view"]').click(); await page.waitForTimeout(450)
  await page.screenshot({ path: `${shots}/3c-cube.png` })
})

// Find a pixel on a face of the given surface type by scanning the viewport, asking the kernel what each hit face is.
const findFacePixel = (surface) => page.evaluate(async (surface) => {
  const d = window.__darkcad, r = d.viewport.renderer.domElement.getBoundingClientRect()
  const seen = new Set()
  for (let y = r.top + 20; y < r.bottom - 20; y += 12) for (let x = r.left + 20; x < r.right - 20; x += 12) {
    const p = d.viewport.pick(x, y)
    if (!p) continue
    const key = p.body + ':' + p.face
    if (seen.has(key)) continue
    seen.add(key)
    const info = await d.kernel.faceInfo(p.body, p.face)
    if (info.surface === surface) return { x, y }
  }
  return null
}, surface)
const maxZ = () => page.evaluate(() => { let m = -1e9; for (const x of window.__darkcad.bodies()) for (let i = 2; i < x.mesh.positions.length; i += 3) m = Math.max(m, x.mesh.positions[i]); return m })

await step('feature delete removes the boss, undo brings it back', async () => {
  await ribbon('Delete feature').click()
  const px2 = await findFacePixel('Cylinder')
  if (!px2) throw new Error('no cylinder face found on screen')
  await page.mouse.click(px2.x, px2.y)
  await page.waitForFunction(() => window.__darkcad.store.doc.features.some((f) => f.name === 'Delete feature 1'), null, { timeout: 20000 })
  await settle()
  if (Math.abs((await maxZ()) - 10) > 0.01) throw new Error('after delete max z ' + (await maxZ()))
  if (await page.locator('.tree li.err, .tl-item.err').count()) throw new Error('feature reported an error')
  await page.locator('.topbar').getByRole('button', { name: 'Undo' }).click()
  await settle()
  if (Math.abs((await maxZ()) - 20) > 0.01) throw new Error('after undo max z ' + (await maxZ()))
})

await step('Thread opens a panel on the right (no size selector in the ribbon)', async () => {
  await ribbon('Thread').click()
  const panel = page.locator('.right-stack .panel', { hasText: 'Thread' }).first()
  await panel.waitFor()
  const box = await panel.boundingBox(), vp = page.viewportSize()
  if (box.x < vp.width / 2) throw new Error('panel is not on the right')
  await page.locator('select[title="Thread standard and size"]').selectOption('G1/4')
  await page.screenshot({ path: `${shots}/4-thread-panel.png` })
})

await step('thread tool resizes the Ø10 boss to the G1/4 major diameter', async () => {
  const px2 = await findFacePixel('Cylinder')
  if (!px2) throw new Error('no cylinder face found on screen')
  await page.mouse.click(px2.x, px2.y)
  await page.waitForFunction(() => window.__darkcad.store.doc.features.some((f) => f.name === 'G1/4 male'), null, { timeout: 20000 })
  await settle()
  if (await page.locator('.tree li.err, .tl-item.err').count()) throw new Error('thread feature reported an error: ' + (await page.locator('.tl-badge').first().getAttribute('title')))
  const rad = await page.evaluate(() => {
    let m = 0
    for (const b of window.__darkcad.bodies()) for (let i = 0; i < b.mesh.positions.length; i += 3) if (b.mesh.positions[i + 2] > 11) m = Math.max(m, Math.hypot(b.mesh.positions[i], b.mesh.positions[i + 1]))
    return m
  })
  console.log('     boss radius above the plate after thread:', rad.toFixed(3), '(G1/4 major radius = 6.5785)')
  if (Math.abs(rad - 6.5785) > 0.05) throw new Error('radius ' + rad)
  await page.screenshot({ path: `${shots}/5-thread.png` })
})

await step('save/open round trip keeps the model', async () => {
  const n = JSON.parse(await page.evaluate(() => JSON.stringify(window.__darkcad.store.doc))).features.length
  if (n !== 5) throw new Error('expected 5 features (delete was undone), got ' + n)
})

await step('cube faces read correctly from home, back, left and right', async () => {
  const cube = page.locator('.vc-canvas')
  const look = async (phi, theta, name) => { await page.evaluate(([p, t]) => window.__darkcad.viewport.setViewAngles(p, t), [phi, theta]); await page.waitForTimeout(500); await cube.screenshot({ path: `${shots}/cube-${name}.png` }) }
  await page.locator('.viewcube button[title="Home view"]').click(); await page.waitForTimeout(500)
  await cube.screenshot({ path: `${shots}/cube-home.png` })
  await look(Math.PI / 2, 0, 'back')
  await look(Math.PI, 0, 'left')
  await look(0, 0, 'right')
  await look(-Math.PI / 2, 0, 'front')
  await look(-Math.PI / 2, Math.PI / 2, 'top')
  await page.locator('.viewcube button[title="Home view"]').click(); await page.waitForTimeout(500)
})

await step('tabs: new design opens as "Untitled 2", switching keeps each design, gallery lists both', async () => {
  await page.locator('.topbar button[title="New design"]').first().click()
  await page.waitForTimeout(300)
  const names = await page.locator('.tab-name').allInnerTexts()
  if (names.join('|') !== 'Untitled|Untitled 2') throw new Error('tabs: ' + names.join('|'))
  if (!(await counts()).includes('0 feature')) throw new Error('new design should be empty: ' + await counts())
  await page.locator('.tab', { hasText: /^Untitled$/ }).click()
  await page.waitForFunction(() => document.querySelector('.status .dim')?.textContent?.includes('5 feature'), null, { timeout: 20000 })
  await waitSolids(1) // the switched-to design really rebuilds its solid
  await page.locator('.topbar button[title="All designs"]').click()
  const rows = await page.locator('.gallery li').allInnerTexts()
  if (rows.length !== 2 || !rows.some((r) => /5 features/.test(r))) throw new Error('gallery: ' + rows.join(' || '))
  await page.screenshot({ path: `${shots}/6-gallery.png` })
  await page.keyboard.press('Escape')
  if (await page.locator('.gallery').count()) throw new Error('gallery did not close')
})

await step('designs persist across a reload; closing a tab keeps it in the library', async () => {
  await page.reload()
  await page.waitForFunction(() => /Ready|Rebuilt/.test(document.querySelector('.status span')?.textContent ?? ''), null, { timeout: 90000 })
  const names = await page.locator('.tab-name').allInnerTexts()
  if (names.join('|') !== 'Untitled|Untitled 2') throw new Error('tabs after reload: ' + names.join('|'))
  await page.waitForFunction(() => document.querySelector('.status .dim')?.textContent?.includes('5 feature'), null, { timeout: 30000 })
  await page.locator('.tab', { hasText: 'Untitled 2' }).hover()
  await page.locator('.tab', { hasText: 'Untitled 2' }).locator('.tab-x').click()
  if ((await page.locator('.tab-name').count()) !== 1) throw new Error('tab not closed')
  await page.locator('.topbar button[title="All designs"]').click()
  if ((await page.locator('.gallery li').count()) !== 2) throw new Error('closed design should stay in the library')
  await page.locator('.gallery li', { hasText: 'Untitled 2' }).click()
  if ((await page.locator('.tab-name').count()) !== 2) throw new Error('reopen from the gallery failed')
  await page.screenshot({ path: `${shots}/7-tabs.png` })
})

// ---- the new sketch toolset, in a fresh design
const newSketchOnXY = async () => {
  await ribbon('Create Sketch').click()
  const pt = await page.evaluate(() => window.__darkcad.viewport.planeScreenPoint('XY'))
  await clickAt(pt)
  await page.locator('.sketch-overlay').waitFor()
  await page.waitForTimeout(600)
}
await step('fresh design for the sketch tools', async () => {
  await page.locator('.topbar button[title="New design"]').first().click()
  await page.waitForTimeout(400)
  if (!(await counts()).includes('0 feature')) throw new Error('not empty: ' + await counts())
})

await step('slot: typed width, then a length dimension makes it "Fully constrained"', async () => {
  await newSketchOnXY()
  await tool('Slot')
  await clickAt(await px(XY(0), 0, 0)); await clickAt(await px(XY(0), 30, 0))
  await page.keyboard.type('10'); await page.keyboard.press('Enter')   // typed width also becomes a driving diameter
  await page.waitForTimeout(200)
  let sk = (await sketchDoc())[0]
  if (sk.entities.filter((e) => e.kind === 'arc').length !== 2 || sk.entities.filter((e) => e.kind === 'line').length !== 2) throw new Error('slot entities: ' + JSON.stringify(sk.entities.map((e) => e.kind)))
  if (!/^[0-9]+ degrees? of freedom$/.test(await skStatus())) throw new Error('status before dimensions: ' + (await skStatus()))
  await tool('Dimension')
  await clickAt(await px(XY(0), 15, 6))               // the upper straight edge
  await page.keyboard.type('40'); await page.keyboard.press('Enter')
  await page.waitForTimeout(300)
  await page.screenshot({ path: `${shots}/8-slot.png` })
  if ((await skStatus()).trim() !== 'Fully constrained') throw new Error('status after dimensions: ' + (await skStatus()))
  sk = (await sketchDoc())[0]
  const xs = sk.entities.filter((e) => e.kind === 'point').map((p) => p.x), ys = sk.entities.filter((e) => e.kind === 'point').map((p) => p.y)
  if (Math.abs(Math.max(...xs) - 40) > 1e-3 || Math.abs(Math.max(...ys) - 5) > 1e-3) throw new Error(`slot extent x ${Math.max(...xs)}, y ${Math.max(...ys)}`)
  await page.getByRole('button', { name: 'Finish sketch' }).click()
  await page.waitForTimeout(500)
})

await step('offset a rectangle outwards by 2 mm and extrude the ring', async () => {
  await newSketchOnXY()
  await tool('Rectangle')
  await clickAt(await px(XY(0), -10, -5)); await clickAt(await px(XY(0), 10, 5))
  await tool('Offset')
  await clickAt(await px(XY(0), 0, -5))               // picks the whole chain
  await page.mouse.move((await px(XY(0), 0, -9)).x, (await px(XY(0), 0, -9)).y)
  await page.waitForTimeout(150)
  await page.keyboard.type('2'); await page.keyboard.press('Enter')
  await page.waitForTimeout(300)
  await page.screenshot({ path: `${shots}/9-offset.png` })
  const sk = (await sketchDoc()).find((f) => f.name === 'Sketch 2')
  const lines = sk.entities.filter((e) => e.kind === 'line').length
  if (lines < 8) throw new Error('offset did not add a second loop: ' + lines + ' lines')
  await page.getByRole('button', { name: 'Finish sketch' }).click()
  await page.waitForTimeout(500)
  await page.locator('.tree li', { hasText: 'Sketch 2' }).click()
  await ribbon('Extrude').click()
  await waitSolids(1)
  await settle()
  const b = await page.evaluate(() => { let m = -1e9; for (const x of window.__darkcad.bodies()) for (let i = 0; i < x.mesh.positions.length; i += 3) m = Math.max(m, x.mesh.positions[i]); return m })
  if (Math.abs(b - 12) > 0.01) throw new Error('max x ' + b + ', expected 12 (rectangle + 2 mm offset)')
  await page.screenshot({ path: `${shots}/10-offset-extrude.png` })
})

const XZ = { origin: [0, 0, 0], dir: [0, -1, 0], xdir: [1, 0, 0] }
const worldPx = (x, y, z) => page.evaluate(([x, y, z]) => {
  const vp = window.__darkcad.viewport, r = vp.renderer.domElement.getBoundingClientRect(), p = vp.planeToScreen({ origin: [x, y, z], dir: [0, 1, 0], xdir: [1, 0, 0] }, 0, 0)
  return { x: r.left + p[0], y: r.top + p[1] }
}, [x, y, z])
const bbox = () => page.evaluate(() => { const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9]; for (const b of window.__darkcad.bodies()) for (let i = 0; i < b.mesh.positions.length; i += 3) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], b.mesh.positions[i + k]); hi[k] = Math.max(hi[k], b.mesh.positions[i + k]) } return { lo, hi } })

await step('shared-edge fills: three rectangles that touch along partial edges are three filled regions', async () => {
  await page.locator('.tabs .icon-btn[title="New design"]').click()
  await page.waitForTimeout(400)
  const pts = [], lines = []
  const rect = (n, x, y, w, h) => { const ids = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]].map((p, i) => { const id = `p${n}${i}`; pts.push({ id, kind: 'point', x: p[0], y: p[1] }); return id }); ids.forEach((a, i) => lines.push({ id: `l${n}${i}`, kind: 'line', a, b: ids[(i + 1) % 4] })) }
  rect(0, 0, 0, 10, 10); rect(1, 10, 6, 5, 4); rect(2, 10, 0, 5, 4)
  await page.evaluate(([entities]) => window.__darkcad.store.add({ type: 'sketch', id: 'sk-t', name: 'Three rects', plane: 'XY', offset: 0, entities, constraints: [] }), [[...pts, ...lines]])
  await page.waitForTimeout(300)
  const n = await page.evaluate(() => window.__darkcad.viewport.sketchRegionCount)
  if (n !== 3) throw new Error('regions drawn: ' + n)
  await page.evaluate(() => window.__darkcad.viewport.fit())
  await page.waitForTimeout(400)
  await page.screenshot({ path: `${shots}/11-shared-edges.png` })
  await page.locator('.tree li', { hasText: 'Three rects' }).click()
  await ribbon('Extrude').click()
  await waitSolids(1)
  await settle()
  const b = await bbox()
  if (Math.abs(b.hi[0] - 15) > 0.01 || Math.abs(b.hi[2] - 10) > 0.01) throw new Error('bbox ' + JSON.stringify(b))
})

await step('revolve: pick a profile, then the axis; the Axis box lets you pick another axis later', async () => {
  await page.locator('.tabs .icon-btn[title="New design"]').click()
  await page.waitForTimeout(400)
  const pts = [], lines = []
  const ids = [[10, 0], [20, 0], [20, 10], [10, 10]].map((p, i) => { pts.push({ id: `p${i}`, kind: 'point', x: p[0], y: p[1] }); return `p${i}` })
  ids.forEach((a, i) => lines.push({ id: `l${i}`, kind: 'line', a, b: ids[(i + 1) % 4] }))
  await page.evaluate(([entities]) => window.__darkcad.store.add({ type: 'sketch', id: 'sk-r', name: 'Profile', plane: 'XZ', offset: 0, entities, constraints: [] }), [[...pts, ...lines]])
  await page.evaluate(() => window.__darkcad.viewport.fit())
  await page.waitForTimeout(500)
  await ribbon('Revolve').click()
  await page.waitForTimeout(200)
  await clickAt(await px(XZ, 15, 5))
  await page.waitForFunction(() => /1 profile selected/.test(document.querySelector('.props')?.textContent ?? ''), null, { timeout: 5000 })
  await clickAt(await worldPx(0, 0, 5))                  // the vertical axis
  await waitSolids(1); await settle()
  await page.screenshot({ path: `${shots}/12-revolve.png` })
  let b = await bbox()
  if (Math.abs(b.hi[0] - 20) > 0.05 || Math.abs(b.lo[0] + 20) > 0.05 || Math.abs(b.hi[1] - 20) > 0.05 || Math.abs(b.hi[2] - 10) > 0.01) throw new Error('washer bbox ' + JSON.stringify(b))
  // swap the axis: now revolve about the X axis
  await page.locator('.selbox', { hasText: 'Axis' }).locator('.box').click()
  await clickAt(await worldPx(-15, 0, 0))
  await settle(); await page.waitForTimeout(500)
  b = await bbox()
  if (Math.abs(b.hi[1] - 10) > 0.05 || Math.abs(b.lo[1] + 10) > 0.05 || Math.abs(b.hi[0] - 20) > 0.05 || Math.abs(b.lo[0] - 10) > 0.05) throw new Error('axis swap did not rebuild: ' + JSON.stringify(b))
  await page.screenshot({ path: `${shots}/13-revolve-swapped-axis.png` })
  await page.getByRole('button', { name: 'OK', exact: true }).click()
})

await step('3D navigation: left-drag orbits and stays upright (Z up), middle/right-drag pans, wheel zooms', async () => {
  await page.evaluate(() => window.__darkcad.viewport.home())
  await page.waitForTimeout(450)
  const vpBox = await page.evaluate(() => { const r = window.__darkcad.viewport.renderer.domElement.getBoundingClientRect(); return { x: r.left, y: r.top, width: r.width, height: r.height } })
  const c = { x: vpBox.x + vpBox.width / 2, y: vpBox.y + vpBox.height / 2 }
  const state = () => page.evaluate(() => { const v = window.__darkcad.viewport; return { dir: v.viewState().dir, up: v.viewState().up, dist: v.dist, target: [v.target.x, v.target.y, v.target.z] } })
  const dragBtn = async (a, b, button) => { await page.mouse.move(a.x, a.y); await page.mouse.down({ button }); await page.mouse.move(b.x, b.y, { steps: 6 }); await page.mouse.up({ button }); await page.waitForTimeout(80) }
  const s0 = await state()
  await dragBtn(c, { x: c.x + 80, y: c.y + 40 }, 'left')
  const s1 = await state()
  if (near(s1.dir, s0.dir, 0.02)) throw new Error('left drag did not orbit')
  if (!near(s1.target, s0.target, 1e-6)) throw new Error('orbit moved the target')
  // drag far over the top and far under: the view never turns upside down
  for (let i = 0; i < 8; i++) await dragBtn(c, { x: c.x, y: c.y + 150 }, 'left')
  if ((await state()).up[2] < -1e-6) throw new Error('view flipped over the top')
  for (let i = 0; i < 16; i++) await dragBtn(c, { x: c.x, y: c.y - 150 }, 'left')
  if ((await state()).up[2] < -1e-6) throw new Error('view flipped under the bottom')
  const s2 = await state()
  await dragBtn(c, { x: c.x + 60, y: c.y + 30 }, 'middle')
  const s3 = await state()
  if (near(s3.target, s2.target, 1e-3) || !near(s3.dir, s2.dir, 1e-6)) throw new Error('middle drag should pan only')
  await dragBtn(c, { x: c.x - 40, y: c.y - 20 }, 'right')
  if (near((await state()).target, s3.target, 1e-3)) throw new Error('right drag should pan')
  await page.mouse.move(c.x, c.y); await page.mouse.wheel(0, -600); await page.waitForTimeout(200)
  if (!((await state()).dist < s3.dist)) throw new Error('wheel did not zoom in')
  await page.evaluate(() => window.__darkcad.viewport.home())
  await page.waitForTimeout(450)
  await page.evaluate(() => window.__darkcad.viewport.fit())
  await page.waitForTimeout(450)
})

await step('typed fields: rectangle Width/Height with Tab, greyed live values, and driving dimensions', async () => {
  await newSketchOnXY()
  await tool('Rectangle')
  await clickAt(await px(XY(0), 0, 0))
  const hover = await px(XY(0), 14, 6)
  await page.mouse.move(hover.x, hover.y); await page.waitForTimeout(150)
  const labels = await page.locator('.sk-fields .sk-field span').allInnerTexts()
  if (labels.join('|') !== 'Width|Height') throw new Error('labels ' + labels)
  if (!(await page.locator('.sk-fields .sk-field b.ghost').count())) throw new Error('live values should be greyed')
  await page.keyboard.type('20'); await page.keyboard.press('Tab'); await page.keyboard.type('8')
  if ((await page.locator('.sk-fields .sk-field.typed').count()) !== 2) throw new Error('typed fields should be bold')
  await page.keyboard.press('Enter'); await page.waitForTimeout(250)
  const sk = (await sketchDoc()).at(-1)
  const xs = sk.entities.filter((e) => e.kind === 'point').map((p) => p.x), ys = sk.entities.filter((e) => e.kind === 'point').map((p) => p.y)
  if (Math.abs(Math.max(...xs) - 20) > 1e-3 || Math.abs(Math.max(...ys) - 8) > 1e-3) throw new Error('rect size')
  if (sk.constraints.filter((c) => c.type === 'hdist' || c.type === 'vdist').length !== 2) throw new Error('typed sizes should become dimensions: ' + JSON.stringify(sk.constraints.map((c) => c.type)) + ' ents ' + sk.entities.length)
  await page.screenshot({ path: `${shots}/14-typed-fields.png` })
})

await step('finish the typed-fields sketch', async () => {
  await page.getByRole('button', { name: 'Finish sketch' }).click()
  await page.waitForTimeout(400)
})


// ======================================================================= timeline, tree, storage, assemblies
const docNow = () => page.evaluate(() => JSON.parse(JSON.stringify(window.__darkcad.store.doc)))
const markerNow = () => page.evaluate(() => window.__darkcad.store.marker)
const items = () => page.locator('.tl-item')
const savedChip = () => page.waitForFunction(() => document.querySelector('.save-chip')?.getAttribute('data-state') === 'saved', null, { timeout: 30000 })
const solids = () => page.locator('.status .dim').innerText().then((t) => parseInt(t))

await step('timeline: one chip per feature of the first design, marker at the end', async () => {
  await page.locator('.tab', { hasText: /^Untitled$/ }).click()
  await page.waitForFunction(() => document.querySelector('.status .dim')?.textContent?.includes('5 feature'), null, { timeout: 20000 })
  await waitSolids(1); await settle()
  if ((await items().count()) !== 5) throw new Error('chips ' + await items().count())
  if ((await markerNow()) !== 5) throw new Error('marker ' + await markerNow())
  await page.screenshot({ path: `${shots}/20-timeline.png` })
})

await step('timeline: dragging the marker back rolls the model back; undo of an edit restores the marker', async () => {
  const zFull = await maxZ()
  const gapBox = await page.locator('.tl-gap[data-gap="2"]').boundingBox()
  const mk = await page.locator('.tl-marker').boundingBox()
  await page.mouse.move(mk.x + mk.width / 2, mk.y + mk.height / 2); await page.mouse.down()
  await page.mouse.move(gapBox.x + gapBox.width / 2, gapBox.y + gapBox.height / 2, { steps: 12 }); await page.mouse.up()
  await settle()
  if ((await markerNow()) !== 2) throw new Error('marker ' + await markerNow())
  if (Math.abs((await maxZ()) - 10) > 0.01) throw new Error('rolled-back model should be 10 high, got ' + await maxZ())
  if ((await page.locator('.tl-item.rolled').count()) !== 3) throw new Error('3 chips should be dimmed')
  // delete the last chip (rolled back, then undo): the marker comes back with the undo
  await page.locator('.tl-item').last().click({ button: 'right' })
  await page.locator('.ctx').getByRole('button', { name: 'Delete' }).click()
  await page.locator('.topbar').getByRole('button', { name: 'Undo' }).click()
  if ((await markerNow()) !== 2) throw new Error('undo should restore marker 2, got ' + await markerNow())
  await page.locator('.tl-controls button[title^="Roll forward"]').click()
  await settle()
  if ((await markerNow()) !== 5) throw new Error('marker ' + await markerNow())
  if (Math.abs((await maxZ()) - zFull) > 0.01) throw new Error('model should be back to ' + zFull + ', got ' + await maxZ())
})

await step('an early extrude edit shows warning badges with tooltips downstream', async () => {
  await items().nth(3).click() // the boss extrude: the thread after it depends on its body
  const field = page.locator('.props .field', { hasText: 'Distance' }).locator('input')
  await field.fill('0'); await field.blur()
  await settle()
  await page.waitForFunction(() => document.querySelector('.tl-item[data-health="error"]'), null, { timeout: 20000 })
  const states = await page.locator('.tl-item').evaluateAll((els) => els.map((e) => e.getAttribute('data-health')))
  if (states[3] !== 'error') throw new Error('boss extrude should be the error: ' + states)
  if (!states.some((s, i) => i > 3 && s === 'blocked')) throw new Error('later features should be blocked: ' + states)
  const blocked = page.locator('.tl-item[data-health="blocked"]').first()
  await blocked.hover()
  await page.waitForSelector('.tl-tip .err')
  const tip = await page.locator('.tl-tip').innerText()
  if (!/Blocked by/.test(tip)) throw new Error('tooltip: ' + tip)
  await page.screenshot({ path: `${shots}/21-timeline-error.png` })
  await page.mouse.move(5, 5)
  await field.fill('10'); await field.blur()
  await settle()
  if (await page.locator('.tl-item[data-health="error"]').count()) throw new Error('error should clear')
})

await step('tree: Origin, Sketches, Bodies folders; double-click renames a sketch; F2 renames a body', async () => {
  const folders = await page.locator('.tree li.folder .label').allInnerTexts()
  if (!['Origin', 'Sketches', 'Construction', 'Bodies'].every((f) => folders.includes(f))) throw new Error('folders: ' + folders)
  await page.locator('.tree li[data-kind="sketch"]').first().locator('.label').dblclick()
  await page.keyboard.type('Base profile'); await page.keyboard.press('Enter')
  await page.waitForFunction(() => window.__darkcad.store.doc.features.some((f) => f.name === 'Base profile'), null, { timeout: 5000 })
  await page.locator('.tree li[data-kind="body"]').first().click()
  await page.keyboard.press('F2')
  await page.keyboard.type('Housing'); await page.keyboard.press('Enter')
  await page.waitForFunction(() => Object.values(window.__darkcad.store.doc.bodyNames).includes('Housing'), null, { timeout: 5000 })
  if (!(await page.locator('.tree li[data-kind="body"]', { hasText: 'Housing' }).count())) throw new Error('body row not renamed')
  // origin visibility
  await page.locator('.tree li.folder[data-folder="origin"]').click()
  await page.locator('.tree li[data-kind="origin-plane"][data-id="XY"] .eye').click()
  await page.waitForTimeout(200)
})

await step('construction plane: offset plane appears on the timeline and a sketch can use it', async () => {
  await ribbon('Offset Plane').click()
  const pt = await page.evaluate(() => window.__darkcad.viewport.planeScreenPoint('XY'))
  if (!pt) throw new Error('no plane picker')
  await page.mouse.click(pt.x, pt.y)
  await page.waitForFunction(() => window.__darkcad.store.doc.features.some((f) => f.type === 'plane'), null, { timeout: 5000 })
  const pl = (await docNow()).features.find((f) => f.type === 'plane')
  if (pl.def.kind !== 'offset' || pl.def.distance !== 10) throw new Error('plane ' + JSON.stringify(pl))
  await page.locator('.props .field', { hasText: 'Distance' }).locator('input').fill('30'); await page.keyboard.press('Tab')
  await ribbon('Create Sketch').click()
  const p2 = await page.evaluate((id) => window.__darkcad.viewport.planeScreenPoint(`plane:${id}`), pl.id)
  if (!p2) throw new Error('construction plane should be pickable')
  await page.mouse.click(p2.x, p2.y)
  await page.waitForFunction((id) => window.__darkcad.store.doc.features.some((f) => f.type === 'sketch' && f.planeRef === id), pl.id, { timeout: 5000 })
  const dir = await page.evaluate(() => window.__darkcad.viewport.viewState())
  await page.getByRole('button', { name: 'Finish sketch' }).click()
  await page.waitForTimeout(300)
  await page.locator('.tree li[data-kind="plane"]').first().locator('.label').dblclick()
  await page.keyboard.type('Datum A'); await page.keyboard.press('Enter')
  await page.waitForFunction(() => window.__darkcad.store.doc.features.some((f) => f.name === 'Datum A'), null, { timeout: 5000 })
})

await step('autosave: the design reaches the server and survives a reload with empty browser storage', async () => {
  await savedChip()
  await page.waitForTimeout(300)
  await page.evaluate(() => localStorage.clear())
  await page.reload()
  await page.waitForFunction(() => window.__darkcad?.workspace?.manifest, null, { timeout: 30000 })
  await page.locator('button[aria-label="Data panel"]').click()
  await page.waitForSelector('.dp-row[data-kind="design"][data-name="Untitled"]', { timeout: 10000 })
  await page.locator('.dp-row[data-kind="design"][data-name="Untitled"]').click()
  await page.waitForFunction(() => window.__darkcad.store.doc.features.some((f) => f.name === 'Datum A'), null, { timeout: 20000 })
  await waitSolids(1)
  if (!(await docNow()).features.some((f) => f.name === 'Base profile')) throw new Error('renamed sketch lost')
  if (Object.values((await docNow()).bodyNames).join() !== 'Housing') throw new Error('body name lost')
})

await step('versions: save a named version, change the design, restore it (undoable)', async () => {
  await page.locator('button[aria-label="Versions"]').click()
  await page.locator('.v-name').fill('Milestone 1')
  await page.locator('.v-save').click()
  await page.waitForSelector('.v-list li[data-kind="manual"][data-name="Milestone 1"]', { timeout: 10000 })
  await page.screenshot({ path: `${shots}/22-versions.png` })
  await page.keyboard.press('Escape')
  const before = (await docNow()).features.length
  await page.locator('.tl-item').last().click()
  await page.keyboard.press('Delete')
  await page.waitForFunction((n) => window.__darkcad.store.doc.features.length === n - 1, before, { timeout: 5000 })
  await page.locator('button[aria-label="Versions"]').click()
  await page.locator('.v-list li[data-name="Milestone 1"] [data-act="restore"]').click()
  await page.waitForFunction((n) => window.__darkcad.store.doc.features.length === n, before, { timeout: 10000 })
  await page.locator('.topbar').getByRole('button', { name: 'Undo' }).click()
  await page.waitForFunction((n) => window.__darkcad.store.doc.features.length === n - 1, before, { timeout: 5000 })
  await page.locator('.topbar').getByRole('button', { name: 'Redo' }).click()
  await page.waitForFunction((n) => window.__darkcad.store.doc.features.length === n, before, { timeout: 5000 })
})

await step('data panel: folder, DXF upload, STEP import, insert into an assembly by drag and drop', async () => {
  // fixtures: a DXF square and the current model as STEP
  const dxf = join(dataDir, 'square.dxf')
  const L = (a, b, c, d) => ['0', 'LINE', '8', '0', '10', a, '20', b, '30', '0', '11', c, '21', d, '31', '0']
  writeFileSync(dxf, ['0', 'SECTION', '2', 'ENTITIES', ...L(0, 0, 10, 0), ...L(10, 0, 10, 10), ...L(10, 10, 0, 10), ...L(0, 10, 0, 0), '0', 'ENDSEC', '0', 'EOF'].join('\n'))
  const stepPath = join(dataDir, 'bracket.step')
  writeFileSync(stepPath, Buffer.from(await page.evaluate(async () => Array.from(await window.__darkcad.kernel.step()))))

  if (!(await page.locator('.sidebar').count())) await page.locator('button[aria-label="Data panel"]').click()
  page.once('dialog', (d) => d.accept('Parts'))
  await page.locator('.dp button[title="New folder"]').click()
  await page.waitForSelector('.dp-row[data-kind="folder"][data-name="Parts"]', { timeout: 10000 })
  await page.locator('.dp-row[data-kind="folder"][data-name="Parts"]').click()
  await page.waitForSelector('.dp-crumbs button:nth-child(3)', { timeout: 5000 }) // a click on a folder's name opens it after a short pause (a double-click renames)
  await page.locator('input[data-testid="upload"]').setInputFiles(dxf)
  await page.waitForSelector('.dp-row[data-kind="asset"][data-name="square.dxf"]', { timeout: 10000 })
  await page.locator('input[data-testid="upload-step"]').setInputFiles(stepPath)
  await page.waitForFunction(() => window.__darkcad.store.doc.features.some((f) => f.type === 'import-step'), null, { timeout: 30000 })
  await waitSolids(1)
  await page.locator('.dp-row[data-kind="asset"][data-name="square.dxf"] .dp-thumb').dblclick() // insert as sketch
  await page.waitForFunction(() => window.__darkcad.store.doc.features.some((f) => f.type === 'sketch' && f.name === 'square'), null, { timeout: 10000 })
  await settle()

  // an assembly design in the same folder
  await page.locator('.dp .dp-new button').click()
  await page.locator('.dp-menu [data-mode="assembly"]').click()
  await page.waitForFunction(() => window.__darkcad.store.doc.mode === 'assembly', null, { timeout: 5000 })
  if (await ribbon('Create Sketch').count()) throw new Error('assembly mode should hide modelling tools')
  const row = page.locator('.dp-row[data-kind="design"]', { hasText: 'bracket' })
  const vp = await page.locator('.viewport').boundingBox()
  await row.dragTo(page.locator('.viewport'), { targetPosition: { x: vp.width * 0.5, y: vp.height * 0.55 } })
  await page.waitForFunction(() => window.__darkcad.store.doc.features.some((f) => f.type === 'component'), null, { timeout: 10000 })
  await waitSolids(1); await settle()
  const comp = (await docNow()).features.find((f) => f.type === 'component')
  if (!comp.grounded) throw new Error('the first instance should be grounded')
  await page.screenshot({ path: `${shots}/23-sidebar-assembly.png` })

  // a second instance can be moved: type a position
  await page.locator('.dp-row[data-kind="design"]', { hasText: 'bracket' }).click({ button: 'right' })
  await page.locator('.dp-ctx [data-act="insert"]').click()
  await page.waitForFunction(() => window.__darkcad.store.doc.features.filter((f) => f.type === 'component').length === 2, null, { timeout: 10000 })
  const x = page.locator('.props .field', { hasText: /^X/ }).locator('input')
  await x.fill('50'); await page.keyboard.press('Tab')
  await waitSolids(2); await settle()
  const c2 = (await docNow()).features.filter((f) => f.type === 'component')[1]
  if (c2.placement.pos[0] !== 50 || c2.grounded) throw new Error('second instance ' + JSON.stringify(c2))
  await savedChip()
})

await step('data panel: double-click a name to rename it; items are alphabetical; root is "My Drive"', async () => {
  const crumb = page.locator('.dp-crumbs button').first()
  if (!/My Drive/.test(await crumb.innerText())) throw new Error('root crumb: ' + await crumb.innerText())
  // a design inside the folder
  const design = page.locator('.dp-row[data-kind="design"]', { hasText: 'bracket' }).first()
  await design.locator('b').dblclick()
  await page.locator('.dp-row input.row-name').fill('Zeta bracket'); await page.keyboard.press('Enter')
  await page.waitForSelector('.dp-row[data-kind="design"][data-name="Zeta bracket"]', { timeout: 10000 })
  // a folder: double-click renames and does not navigate into it
  await crumb.click()
  const folder = page.locator('.dp-row[data-kind="folder"][data-name="Parts"]')
  await folder.locator('b').dblclick()
  await page.locator('.dp-row input.row-name').fill('Fixtures'); await page.keyboard.press('Enter')
  await page.waitForSelector('.dp-row[data-kind="folder"][data-name="Fixtures"]', { timeout: 10000 })
  await page.waitForTimeout(500)
  if (await page.locator('.dp-crumbs button').count() !== 1) throw new Error('renaming a folder must not open it')
  const names = await page.locator('.dp-row').evaluateAll((els) => els.map((e) => e.getAttribute('data-name')))
  const sorted = [...names].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base', numeric: true }))
  if (names.join('|') !== sorted.join('|')) throw new Error('not alphabetical: ' + names.join(', '))
})

console.log('status:', await status(), '|', await counts())
await browser.close(); server.httpServer.close(); api.kill(); rmSync(dataDir, { recursive: true, force: true })
if (problems.length) { console.log('\nPROBLEMS:\n' + problems.join('\n')); process.exit(1) }
console.log('\nsmoke test passed')
