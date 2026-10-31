// End-to-end smoke test against the production build, in headless Chromium.
// Usage: pnpm build && node e2e/smoke.mjs [screenshotDir]
import { preview } from 'vite'
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'

const shots = process.argv[2] ?? 'e2e/out'
mkdirSync(shots, { recursive: true })
const server = await preview({ preview: { port: 4173, strictPort: true } })
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: 1500, height: 900 } })
const problems = []
page.on('pageerror', (e) => problems.push('pageerror: ' + e.message))
page.on('console', (m) => { if (m.type() === 'error') problems.push('console: ' + m.text()) })

const step = async (name, fn) => { try { await fn(); console.log('ok  ', name) } catch (e) { console.log('FAIL', name, '-', String(e.message).split('\n')[0]); problems.push(name); await page.screenshot({ path: `${shots}/fail-${name.replace(/\W+/g, '_')}.png` }) } }
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

await step('ribbon order is 2D, 3D, Modify, Import / Export', async () => {
  const titles = await page.locator('.ribbon-group h4').allInnerTexts()
  if (titles.join('|').toLowerCase() !== '2d|3d|modify|import / export') throw new Error(titles.join('|'))
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

await step('draw a 20 × 10 mm rectangle in global coordinates', async () => {
  await drag(await px(XY(0), -10, -5), await px(XY(0), 10, 5))
  await page.screenshot({ path: `${shots}/1-sketch.png` })
  const vals = await page.locator('.entity').first().locator('input').evaluateAll((els) => els.map((e) => e.value))
  if (vals.join(',') !== '-10,-5,20,10') throw new Error('rect x,y,w,h = ' + vals.join(','))
  await page.getByRole('button', { name: 'Finish sketch' }).click()
  await page.waitForTimeout(600)
  const v = await view()
  if (v.ortho || v.locked) throw new Error('camera not restored: ' + JSON.stringify(v))
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
  await page.getByRole('button', { name: 'Circle' }).click()
  // The plate top is at z=10, so first move this sketch there: the plane offset in the panel re-aims the camera.
  await page.locator('.field', { hasText: 'Offset' }).locator('input').fill('10')
  await page.waitForTimeout(600)
  await drag(await px(XY(10), 0, 0), await px(XY(10), 5, 0))
  await page.screenshot({ path: `${shots}/3-sketch-over-solid.png` })
  await page.getByRole('button', { name: 'Finish sketch' }).click()
  await page.waitForTimeout(600)
  await page.locator('.tree li', { hasText: 'Sketch 2' }).click()
  await ribbon('Extrude').click()
  await page.waitForTimeout(1500)
  await waitSolids(1)
})

await step('boss extends to z=20', async () => { await settle(); const b = await page.evaluate(() => { let m = -1e9; for (const x of window.__darkcad.bodies()) for (let i = 2; i < x.mesh.positions.length; i += 3) m = Math.max(m, x.mesh.positions[i]); return m }); if (Math.abs(b - 20) > 0.01) throw new Error('max z ' + b + ' doc=' + JSON.stringify(await page.evaluate(() => window.__darkcad.store.doc.features.map((f) => ({ id: f.id, offset: f.offset, distance: f.distance, mode: f.mode, ents: f.entities?.map((e) => [e.kind, e.x, e.y, e.r, e.w, e.h]) }))))) })

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
  await page.waitForFunction(() => document.querySelector('.tree')?.textContent?.includes('Delete feature 1'), null, { timeout: 20000 })
  await settle()
  if (Math.abs((await maxZ()) - 10) > 0.01) throw new Error('after delete max z ' + (await maxZ()))
  if (await page.locator('.tree li.err').count()) throw new Error('feature reported an error')
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
  await page.waitForFunction(() => document.querySelector('.tree')?.textContent?.includes('G1/4 male'), null, { timeout: 20000 })
  await settle()
  if (await page.locator('.tree li.err').count()) throw new Error('thread feature reported an error: ' + await page.locator('.tree li.err').getAttribute('title'))
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

console.log('status:', await status(), '|', await counts())
await browser.close(); server.httpServer.close()
if (problems.length) { console.log('\nPROBLEMS:\n' + problems.join('\n')); process.exit(1) }
console.log('\nsmoke test passed')
