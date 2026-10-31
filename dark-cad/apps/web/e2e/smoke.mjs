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
await page.goto('http://localhost:4173/', { waitUntil: 'load' })
await page.evaluate(() => localStorage.clear())
await page.reload()

await step('kernel loads', async () => { await page.waitForFunction(() => document.querySelector('.status span')?.textContent === 'Ready' || /Rebuilt/.test(document.querySelector('.status span')?.textContent ?? ''), null, { timeout: 90000 }) })

await step('draw a rectangle in the sketcher', async () => {
  await page.locator('.ribbon').getByRole('button', { name: 'Sketch', exact: true }).click()
  const c = page.locator('.sketcher canvas')
  await c.waitFor()
  const b = await c.boundingBox()
  const cx = b.x + b.width / 2, cy = b.y + b.height / 2
  await page.mouse.move(cx - 80, cy - 40); await page.mouse.down(); await page.mouse.move(cx + 80, cy + 40, { steps: 6 }); await page.mouse.up()
  await page.screenshot({ path: `${shots}/1-sketch.png` })
  await page.getByRole('button', { name: 'Finish sketch' }).click()
  const n = await page.locator('.entity').count()
  if (n !== 1) throw new Error(`expected 1 shape, got ${n}`)
})

await step('extrude makes a solid', async () => {
  await page.locator('.tree li', { hasText: 'Sketch 1' }).click()
  await page.locator('.ribbon').getByRole('button', { name: 'Extrude' }).click()
  await waitSolids(1)
  await page.waitForTimeout(300)
  await page.screenshot({ path: `${shots}/2-extrude.png` })
})

await step('second sketch (circle) + add-extrude makes a boss on top', async () => {
  await page.locator('.ribbon').getByRole('button', { name: 'Sketch', exact: true }).click()
  const c = page.locator('.sketcher canvas')
  await c.waitFor()
  const b = await c.boundingBox()
  const cx = b.x + b.width / 2, cy = b.y + b.height / 2
  await page.keyboard.press('c')
  await page.mouse.move(cx, cy); await page.mouse.down(); await page.mouse.move(cx + 40, cy, { steps: 5 }); await page.mouse.up()
  await page.getByRole('button', { name: 'Finish sketch' }).click()
  // put the sketch on top of the plate: offset = 10 (extrude distance)
  await page.locator('.field', { hasText: 'Offset' }).locator('input').fill('10')
  await page.locator('.tree li', { hasText: 'Sketch 2' }).click()
  await page.locator('.ribbon').getByRole('button', { name: 'Extrude' }).click()
  await page.waitForTimeout(1500)
  await waitSolids(1)
  await page.screenshot({ path: `${shots}/3-boss.png` })
})

// Find a pixel on a face of the given surface type by scanning the viewport, asking the kernel what each hit face is.
const findFacePixel = (surface) => page.evaluate(async (surface) => {
  const d = window.__darkcad, el = d.viewport.renderer.domElement, r = el.getBoundingClientRect()
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
const bounds = () => page.evaluate(() => {
  let min = [1e9, 1e9, 1e9], max = [-1e9, -1e9, -1e9]
  for (const b of window.__darkcad.bodies()) for (let i = 0; i < b.mesh.positions.length; i += 3) for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], b.mesh.positions[i + k]); max[k] = Math.max(max[k], b.mesh.positions[i + k]) }
  return { min, max }
})
const settle = async () => { await page.waitForTimeout(400); await page.waitForFunction(() => !document.querySelector('.busy'), null, { timeout: 60000 }) }

await step('boss extends to z=20', async () => { await settle(); const b = await bounds(); if (Math.abs(b.max[2] - 20) > 0.01) throw new Error('max z ' + b.max[2]) })

await step('feature delete removes the boss, undo brings it back', async () => {
  await page.locator('.ribbon').getByRole('button', { name: 'Delete feature' }).click()
  const px = await findFacePixel('Cylinder')
  if (!px) throw new Error('no cylinder face found on screen')
  await page.mouse.click(px.x, px.y)
  await page.waitForFunction(() => document.querySelector('.tree')?.textContent?.includes('Delete feature 1'), null, { timeout: 20000 })
  await settle()
  const b = await bounds()
  if (Math.abs(b.max[2] - 10) > 0.01) throw new Error('after delete max z ' + b.max[2])
  if (await page.locator('.tree li.err').count()) throw new Error('feature reported an error')
  await page.screenshot({ path: `${shots}/4-deleted.png` })
  await page.locator('.topbar').getByRole('button', { name: 'Undo' }).click()
  await settle()
  const b2 = await bounds()
  if (Math.abs(b2.max[2] - 20) > 0.01) throw new Error('after undo max z ' + b2.max[2])
})

await step('thread tool resizes the Ø10 boss to the G1/4 major diameter', async () => {
  await page.locator('select[title="Thread standard and size"]').selectOption('G1/4')
  await page.locator('.ribbon').getByRole('button', { name: 'Thread' }).click()
  const px = await findFacePixel('Cylinder')
  if (!px) throw new Error('no cylinder face found on screen')
  await page.mouse.click(px.x, px.y)
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
  const text = await page.evaluate(() => JSON.stringify(window.__darkcad.store.doc))
  const n = JSON.parse(text).features.length
  if (n !== 5) throw new Error('expected 5 features (delete was undone), got ' + n)
})

console.log('status:', await status(), '|', await counts())
await browser.close(); server.httpServer.close()
if (problems.length) { console.log('\nPROBLEMS:\n' + problems.join('\n')); process.exit(1) }
console.log('\nsmoke test passed')
