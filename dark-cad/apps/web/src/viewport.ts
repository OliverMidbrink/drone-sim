import * as THREE from 'three'
import { planeFrame } from '@darkcad/features'
import type { Plane } from '@darkcad/core'
import type { Mesh } from '@darkcad/kernel'

export interface BodyMesh { id: string; mesh: Mesh }
export interface Pick { body: string; face: number; point: THREE.Vector3; normal: THREE.Vector3 }
export type Frame = ReturnType<typeof planeFrame>

const BODY_COLORS = [0xb9c3d3, 0x8da2c4, 0xc4b79a, 0x93b8a0, 0xb893a0]
const FOV = 40
const Z = new THREE.Vector3(0, 0, 1)
const PLANE_LABEL: Record<Plane, string> = { XY: 'Top (XY)', XZ: 'Front (XZ)', YZ: 'Right (YZ)' }
const YELLOW = 0xffd60a

type Cam = THREE.PerspectiveCamera | THREE.OrthographicCamera

/** A view orientation: camera quaternion looking at `target` from `dist` away. Z is world-up. */
export class Viewport {
  readonly renderer: THREE.WebGLRenderer
  readonly scene = new THREE.Scene()
  private persp = new THREE.PerspectiveCamera(FOV, 1, 0.1, 20000)
  private ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, -20000, 20000)
  camera: Cam = this.persp
  target = new THREE.Vector3()
  orient = new THREE.Quaternion()
  dist = 160
  private orthoOn = false
  private locked = false
  private anim?: { t0: number; ms: number; fromQ: THREE.Quaternion; toQ: THREE.Quaternion; fromT: THREE.Vector3; toT: THREE.Vector3; fromD: number; toD: number; done?: () => void }
  private listeners = new Set<() => void>()
  private group = new THREE.Group()
  private overlay = new THREE.Group()
  private ground = new THREE.Group()
  private meshes: { id: string; obj: THREE.Mesh; mesh: Mesh }[] = []
  private ray = new THREE.Raycaster()
  private raf = 0
  private hover?: { body: string; face: number }
  private selected: { body: string; face: number }[] = []
  private xray = false
  private disposed = false
  private saved?: { ortho: boolean; orient: THREE.Quaternion; target: THREE.Vector3; dist: number }
  // plane picking (sketch creation)
  private planes?: { group: THREE.Group; items: { name: Plane; mesh: THREE.Mesh; line: THREE.LineLoop }[]; cb: (p: Plane) => void; size: number }

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    this.renderer.setClearColor(0x0a0a0a)
    host.appendChild(this.renderer.domElement)
    this.scene.add(new THREE.HemisphereLight(0xdfe7ff, 0x20242c, 1.1))
    const key = new THREE.DirectionalLight(0xffffff, 1.6)
    key.position.set(80, -60, 120)
    this.scene.add(key)
    const grid = new THREE.GridHelper(200, 20, 0x2e2e2e, 0x1a1a1a)
    grid.rotation.x = Math.PI / 2
    this.ground.add(grid, new THREE.AxesHelper(20))
    this.scene.add(this.ground, this.group, this.overlay)
    this.orient = this.orientFor(new THREE.Vector3(0.6, -0.8, 0.55), Z)
    this.wireMouse()
    new ResizeObserver(() => this.resize()).observe(host)
    this.resize()
    const loop = (now: number) => {
      if (this.disposed) return
      this.stepAnim(now)
      this.renderer.render(this.scene, this.camera)
      this.raf = requestAnimationFrame(loop)
    }
    this.raf = requestAnimationFrame(loop)
  }

  dispose() { this.disposed = true; cancelAnimationFrame(this.raf); this.renderer.dispose(); this.renderer.domElement.remove() }

  // ---------------------------------------------------------------- camera
  subscribe(fn: () => void) { this.listeners.add(fn); return () => { this.listeners.delete(fn) } }
  private emit() { this.listeners.forEach((f) => f()) }

  get isOrtho() { return this.orthoOn }
  get isLocked() { return this.locked }
  get isPlanePicking() { return !!this.planes }

  /** Direction from the target towards the camera. */
  viewDir(): THREE.Vector3 { return new THREE.Vector3(0, 0, 1).applyQuaternion(this.orient) }
  viewState() {
    const d = this.viewDir(), u = new THREE.Vector3(0, 1, 0).applyQuaternion(this.orient)
    return { dir: [d.x, d.y, d.z] as [number, number, number], up: [u.x, u.y, u.z] as [number, number, number], ortho: this.orthoOn, locked: this.locked, dist: this.dist }
  }

  private orientFor(dir: THREE.Vector3, up: THREE.Vector3): THREE.Quaternion {
    const m = new THREE.Matrix4().lookAt(dir.clone().normalize().multiplyScalar(10), new THREE.Vector3(), up)
    return new THREE.Quaternion().setFromRotationMatrix(m)
  }

  private applyCamera() {
    const back = new THREE.Vector3(0, 0, 1).applyQuaternion(this.orient)
    const cam = this.camera
    cam.position.copy(this.target).addScaledVector(back, this.dist)
    cam.quaternion.copy(this.orient)
    if (cam instanceof THREE.OrthographicCamera) {
      // same visible height as the perspective camera at this distance, so switching doesn't jump
      const h = 2 * this.dist * Math.tan((FOV * Math.PI) / 360)
      const w = h * (this.host.clientWidth / Math.max(1, this.host.clientHeight))
      cam.left = -w / 2; cam.right = w / 2; cam.top = h / 2; cam.bottom = -h / 2
      cam.updateProjectionMatrix()
    }
    cam.updateMatrixWorld(true)
    this.emit()
  }

  private resize() {
    const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1
    this.renderer.setSize(w, h)
    this.persp.aspect = w / h
    this.persp.updateProjectionMatrix()
    this.applyCamera()
  }

  setOrtho(on: boolean) {
    if (on === this.orthoOn) return
    this.orthoOn = on
    this.camera = on ? this.ortho : this.persp
    this.applyCamera()
  }

  private animateTo(q: THREE.Quaternion, t: THREE.Vector3, d: number, ms = 280, done?: () => void) {
    this.anim = { t0: performance.now(), ms, fromQ: this.orient.clone(), toQ: q, fromT: this.target.clone(), toT: t.clone(), fromD: this.dist, toD: d, done }
  }

  private stepAnim(now: number) {
    const a = this.anim
    if (!a) return
    const k = Math.min(1, (now - a.t0) / a.ms)
    const e = 1 - Math.pow(1 - k, 3)
    this.orient.slerpQuaternions(a.fromQ, a.toQ, e)
    this.target.lerpVectors(a.fromT, a.toT, e)
    this.dist = a.fromD + (a.toD - a.fromD) * e
    this.applyCamera()
    if (k >= 1) { this.anim = undefined; a.done?.() }
  }

  /** Turntable orbit by a pointer drag (pixels). */
  orbit(dx: number, dy: number) {
    if (this.locked) return
    this.anim = undefined
    const k = 0.007
    const yaw = new THREE.Quaternion().setFromAxisAngle(Z, -dx * k)
    const q1 = yaw.multiply(this.orient.clone())
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(q1)
    const q2 = new THREE.Quaternion().setFromAxisAngle(right, -dy * k).multiply(q1)
    const fz = (q: THREE.Quaternion) => Math.abs(new THREE.Vector3(0, 0, 1).applyQuaternion(q).z)
    // don't flip over the poles, but always allow moving away from one
    this.orient.copy(fz(q2) < 0.9995 || fz(q2) <= fz(q1) ? q2 : q1).normalize()
    this.applyCamera()
  }

  pan(dx: number, dy: number) {
    this.anim = undefined
    const h = this.camera instanceof THREE.OrthographicCamera ? this.camera.top - this.camera.bottom : 2 * this.dist * Math.tan((FOV * Math.PI) / 360)
    const s = h / Math.max(1, this.host.clientHeight)
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.orient)
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.orient)
    this.target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s)
    this.applyCamera()
  }

  /** Zoom towards the pointer. */
  zoomAt(factor: number, clientX: number, clientY: number) {
    this.anim = undefined
    const before = this.pointOnTargetPlane(clientX, clientY)
    this.dist = Math.min(8000, Math.max(0.5, this.dist * factor))
    this.applyCamera()
    const after = this.pointOnTargetPlane(clientX, clientY)
    if (before && after) { this.target.add(before.sub(after)); this.applyCamera() }
  }

  private ndc(clientX: number, clientY: number) {
    const r = this.renderer.domElement.getBoundingClientRect()
    return new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1)
  }

  private pointOnTargetPlane(clientX: number, clientY: number): THREE.Vector3 | undefined {
    this.ray.setFromCamera(this.ndc(clientX, clientY), this.camera)
    const n = new THREE.Vector3(0, 0, 1).applyQuaternion(this.orient)
    return this.ray.ray.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(n, this.target), new THREE.Vector3()) ?? undefined
  }

  // spherical view angles about world Z: phi = azimuth of the camera, theta = elevation
  private angles() { const d = this.viewDir(); return { phi: Math.atan2(d.y, d.x), theta: Math.asin(Math.max(-1, Math.min(1, d.z))) } }

  setViewAngles(phi: number, theta: number) {
    if (this.locked) return
    const d = new THREE.Vector3(Math.cos(theta) * Math.cos(phi), Math.cos(theta) * Math.sin(phi), Math.sin(theta))
    const pole = Math.abs(Math.sin(theta)) > 0.99999
    const up = pole ? new THREE.Vector3(Math.cos(phi), Math.sin(phi), 0).multiplyScalar(-Math.sign(Math.sin(theta))) : Z
    this.animateTo(this.orientFor(d, up), this.target, this.dist)
  }

  /** View from a direction (a cube face, edge or corner). */
  viewFromDirection(dir: THREE.Vector3) {
    const d = dir.clone().normalize()
    const horizontal = Math.hypot(d.x, d.y) > 1e-6
    this.setViewAngles(horizontal ? Math.atan2(d.y, d.x) : this.angles().phi, Math.asin(d.z))
  }

  /** Step the view by 90°, snapping to the 45° grid first. `yaw` and `pitch` are -1, 0 or 1. */
  stepView(yaw: number, pitch: number) {
    const a = this.angles(), snap = (v: number) => Math.round(v / (Math.PI / 4)) * (Math.PI / 4)
    const theta = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, snap(a.theta) + pitch * (Math.PI / 2)))
    this.setViewAngles(snap(a.phi) + yaw * (Math.PI / 2), theta)
  }

  /** Roll the view about the viewing axis by ±90°. */
  roll(sign: number) {
    if (this.locked) return
    const q = this.orient.clone().multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), (sign * Math.PI) / 2))
    this.animateTo(q, this.target, this.dist)
  }

  home() { if (!this.locked) this.animateTo(this.orientFor(new THREE.Vector3(0.6, -0.8, 0.55), Z), this.target, this.dist) }

  fit() {
    const box = new THREE.Box3().setFromObject(this.group)
    if (box.isEmpty() || this.locked) return
    const c = box.getCenter(new THREE.Vector3()), r = box.getSize(new THREE.Vector3()).length() / 2 || 10
    this.animateTo(this.orient.clone(), c, r * 3.2)
  }

  // ---------------------------------------------------------------- sketch mode
  /** Look straight at a sketch plane in orthographic projection. Bodies stay visible (x-ray), the ground grid hides. */
  enterSketch(f: Frame) {
    if (!this.saved) this.saved = { ortho: this.orthoOn, orient: this.orient.clone(), target: this.target.clone(), dist: this.dist }
    this.locked = false
    this.setOrtho(true)
    this.alignTo(f)
    this.locked = true
    this.ground.visible = false
    this.setXray(true)
  }

  alignTo(f: Frame) {
    const n = new THREE.Vector3(...f.dir), o = new THREE.Vector3(...f.origin), x = new THREE.Vector3(...f.xdir)
    const y = n.clone().cross(x)
    const t = this.target.clone().addScaledVector(n, -this.target.clone().sub(o).dot(n)) // target projected onto the plane
    this.animateTo(this.orientFor(n, y), t, this.dist, 320)
  }

  exitSketch() {
    this.ground.visible = true
    this.setXray(false)
    this.locked = false
    const s = this.saved
    this.saved = undefined
    if (!s) return
    this.animateTo(s.orient, s.target, s.dist, 320, () => this.setOrtho(s.ortho))
  }

  /** Plane (u, v in mm) → pixel inside the viewport. Works for any frame. */
  planeToScreen(f: Frame, u: number, v: number): [number, number] {
    const x = new THREE.Vector3(...f.xdir), n = new THREE.Vector3(...f.dir), y = n.clone().cross(x)
    const p = new THREE.Vector3(...f.origin).addScaledVector(x, u).addScaledVector(y, v).project(this.camera)
    return [((p.x + 1) / 2) * this.host.clientWidth, ((1 - p.y) / 2) * this.host.clientHeight]
  }

  /** Pointer (client coords) → plane coordinates (u, v in mm). */
  screenToPlane(f: Frame, clientX: number, clientY: number): [number, number] | undefined {
    this.ray.setFromCamera(this.ndc(clientX, clientY), this.camera)
    const n = new THREE.Vector3(...f.dir), o = new THREE.Vector3(...f.origin), x = new THREE.Vector3(...f.xdir)
    const hit = this.ray.ray.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(n, o), new THREE.Vector3())
    if (!hit) return
    const d = hit.sub(o)
    return [d.dot(x), d.dot(n.clone().cross(x))]
  }

  // ---------------------------------------------------------------- bodies
  setBodies(bodies: BodyMesh[], fit = false) {
    this.group.clear()
    this.meshes = []
    bodies.forEach(({ id, mesh }, i) => {
      const g = new THREE.BufferGeometry()
      g.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3))
      g.setAttribute('normal', new THREE.BufferAttribute(mesh.normals, 3))
      g.setIndex(new THREE.BufferAttribute(mesh.indices, 1))
      const obj = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: BODY_COLORS[i % BODY_COLORS.length], roughness: 0.55, metalness: 0.1, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }))
      const eg = new THREE.BufferGeometry()
      eg.setAttribute('position', new THREE.BufferAttribute(mesh.edges, 3))
      obj.add(new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: 0x0a0a0a })))
      this.group.add(obj)
      this.meshes.push({ id, obj, mesh })
    })
    this.setXray(this.xray)
    this.refreshOverlay()
    if (fit) { const box = new THREE.Box3().setFromObject(this.group); if (!box.isEmpty()) { this.target.copy(box.getCenter(new THREE.Vector3())); this.dist = (box.getSize(new THREE.Vector3()).length() / 2 || 10) * 3.4; this.applyCamera() } }
  }

  /** Translucent bodies, used while sketching so the model stays visible behind the sketch. */
  setXray(on: boolean) {
    this.xray = on
    for (const m of this.meshes) {
      const mat = m.obj.material as THREE.MeshStandardMaterial
      mat.transparent = on; mat.opacity = on ? 0.32 : 1; mat.depthWrite = !on; mat.needsUpdate = true
    }
  }

  pick(clientX: number, clientY: number): Pick | undefined {
    this.ray.setFromCamera(this.ndc(clientX, clientY), this.camera)
    const hit = this.ray.intersectObjects(this.meshes.map((m) => m.obj), false)[0]
    if (!hit || hit.faceIndex == null) return
    const m = this.meshes.find((x) => x.obj === hit.object)!
    const face = m.mesh.faceRanges.findIndex((fr) => hit.faceIndex! >= fr.start && hit.faceIndex! < fr.start + fr.count)
    if (face < 0) return
    return { body: m.id, face, point: hit.point, normal: hit.face!.normal.clone() }
  }

  setHover(p?: { body: string; face: number }) {
    if (this.hover?.body === p?.body && this.hover?.face === p?.face) return
    this.hover = p
    this.refreshOverlay()
  }

  setSelectedFaces(f: { body: string; face: number }[]) { this.selected = f; this.refreshOverlay() }

  /** Vertices of one face, from the mesh (used by tools that need the face's extent). */
  faceVertices(body: string, face: number): THREE.Vector3[] {
    const m = this.meshes.find((x) => x.id === body)?.mesh
    if (!m) return []
    const fr = m.faceRanges[face]
    const out: THREE.Vector3[] = []
    for (let t = fr.start; t < fr.start + fr.count; t++) for (let k = 0; k < 3; k++) {
      const i = m.indices[t * 3 + k] * 3
      out.push(new THREE.Vector3(m.positions[i], m.positions[i + 1], m.positions[i + 2]))
    }
    return out
  }

  bodyCenter(body: string): THREE.Vector3 {
    const m = this.meshes.find((x) => x.id === body)
    return m ? new THREE.Box3().setFromObject(m.obj).getCenter(new THREE.Vector3()) : new THREE.Vector3()
  }

  private refreshOverlay() {
    this.overlay.clear()
    const add = (f: { body: string; face: number }, color: number, opacity: number) => {
      const m = this.meshes.find((x) => x.id === f.body)
      const fr = m?.mesh.faceRanges[f.face]
      if (!m || !fr || !fr.count) return
      const g = new THREE.BufferGeometry()
      g.setAttribute('position', new THREE.BufferAttribute(m.mesh.positions, 3))
      g.setAttribute('normal', new THREE.BufferAttribute(m.mesh.normals, 3))
      g.setIndex(new THREE.BufferAttribute(m.mesh.indices.subarray(fr.start * 3, (fr.start + fr.count) * 3), 1))
      this.overlay.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthTest: true, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 })))
    }
    this.selected.forEach((f) => add(f, 0x3291ff, 0.7))
    if (this.hover) add(this.hover, 0xffffff, 0.22)
  }

  // ---------------------------------------------------------------- plane picking (new sketch)
  /** Show the three origin planes in yellow; calls back with the one the user clicks. */
  startPlanePick(cb: (p: Plane) => void) {
    this.stopPlanePick()
    const box = new THREE.Box3().setFromObject(this.group)
    // Sized from what is on screen (about 40% of the view height), but never smaller than the model needs.
    const visibleH = 2 * this.dist * Math.tan((FOV * Math.PI) / 360)
    const diag = box.isEmpty() ? 0 : box.getSize(new THREE.Vector3()).length()
    const size = Math.min(400, Math.max(visibleH * 0.4, diag * 0.6, 10))
    const group = new THREE.Group()
    const items: { name: Plane; mesh: THREE.Mesh; line: THREE.LineLoop }[] = []
    for (const name of ['XY', 'XZ', 'YZ'] as Plane[]) {
      const f = planeFrame(name, 0)
      const x = new THREE.Vector3(...f.xdir), n = new THREE.Vector3(...f.dir), y = n.clone().cross(x)
      const rot = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, n))
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshBasicMaterial({ color: YELLOW, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false }))
      mesh.quaternion.copy(rot)
      mesh.position.copy(x.clone().multiplyScalar(size / 2)).addScaledVector(y, size / 2) // quadrant away from the others
      const pts = [[0, 0], [size, 0], [size, size], [0, size]].map(([u, v]) => x.clone().multiplyScalar(u).addScaledVector(y, v))
      const line = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: YELLOW, transparent: true, opacity: 0.85 }))
      const label = this.labelSprite(PLANE_LABEL[name])
      label.position.copy(x.clone().multiplyScalar(size * 0.5)).addScaledVector(y, size * 0.5)
      label.scale.set(size * 0.34, size * 0.085, 1)
      group.add(mesh, line, label)
      items.push({ name, mesh, line })
    }
    this.planes = { group, items, cb, size }
    this.scene.add(group)
    group.updateMatrixWorld(true) // hit-tests must work before the first rendered frame
  }

  stopPlanePick() {
    if (!this.planes) return
    this.scene.remove(this.planes.group)
    this.planes = undefined
  }

  /** Client-pixel position on a pickable plane square that is not covered by another plane (for hints and tests). */
  planeScreenPoint(name: Plane): { x: number; y: number } | undefined {
    if (!this.planes) return
    const f = planeFrame(name, 0)
    const x = new THREE.Vector3(...f.xdir), n = new THREE.Vector3(...f.dir), y = n.clone().cross(x)
    const r = this.renderer.domElement.getBoundingClientRect(), size = this.planes.size
    let fallback: { x: number; y: number } | undefined
    for (const k of [0.5, 0.75, 0.3, 0.9, 0.15]) for (const m of [0.5, 0.75, 0.3, 0.9, 0.15]) {
      const p = x.clone().multiplyScalar(k * size).addScaledVector(y, m * size).project(this.camera)
      const pt = { x: r.left + ((p.x + 1) / 2) * r.width, y: r.top + ((1 - p.y) / 2) * r.height }
      fallback ??= pt
      // must hit this plane and not sit under a floating panel
      const safe = [[0, 0], [4, 0], [-4, 0], [0, 4], [0, -4]].every(([dx, dy]) => this.planeAt(pt.x + dx, pt.y + dy) === name && document.elementFromPoint(pt.x + dx, pt.y + dy) === this.renderer.domElement)
      if (safe) return pt
    }
    return fallback
  }

  private labelSprite(text: string): THREE.Sprite {
    const c = document.createElement('canvas')
    c.width = 256; c.height = 64
    const g = c.getContext('2d')!
    g.font = '600 30px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'
    g.fillStyle = '#ffd60a'; g.fillText(text, 128, 34)
    return new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthTest: false }))
  }

  private planeAt(clientX: number, clientY: number): Plane | undefined {
    if (!this.planes) return
    this.ray.setFromCamera(this.ndc(clientX, clientY), this.camera)
    const hit = this.ray.intersectObjects(this.planes.items.map((i) => i.mesh), false)[0]
    return hit && this.planes.items.find((i) => i.mesh === hit.object)?.name
  }

  // ---------------------------------------------------------------- mouse
  private wireMouse() {
    const el = this.renderer.domElement
    let drag: { x: number; y: number; mode: 'orbit' | 'pan'; moved: boolean } | undefined
    el.addEventListener('contextmenu', (e) => e.preventDefault())
    el.addEventListener('pointerdown', (e) => {
      el.setPointerCapture(e.pointerId)
      drag = { x: e.clientX, y: e.clientY, mode: e.button === 1 || e.button === 2 || e.shiftKey ? 'pan' : 'orbit', moved: false }
    })
    el.addEventListener('pointermove', (e) => {
      if (this.planes && !drag) {
        const hover = this.planeAt(e.clientX, e.clientY)
        for (const it of this.planes.items) {
          const on = it.name === hover
          ;(it.mesh.material as THREE.MeshBasicMaterial).opacity = on ? 0.5 : 0.16
          ;(it.line.material as THREE.LineBasicMaterial).opacity = on ? 1 : 0.85
        }
        el.style.cursor = hover ? 'pointer' : ''
      }
      if (!drag) return
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y
      if (!drag.moved && Math.hypot(dx, dy) < 3) return
      drag.moved = true
      drag.x = e.clientX; drag.y = e.clientY
      if (drag.mode === 'pan') this.pan(dx, dy); else this.orbit(dx, dy)
    })
    // Registered before the app's own pointerup handler, so a plane click can swallow the event.
    el.addEventListener('pointerup', (e) => {
      const d = drag
      drag = undefined
      if (d && !d.moved && e.button === 0 && this.planes) {
        const hit = this.planeAt(e.clientX, e.clientY)
        if (hit) { const cb = this.planes.cb; this.stopPlanePick(); el.style.cursor = ''; e.stopImmediatePropagation(); cb(hit) }
      }
    })
    el.addEventListener('wheel', (e) => { e.preventDefault(); this.zoomAt(Math.exp(e.deltaY * 0.0015), e.clientX, e.clientY) }, { passive: false })
  }
}
