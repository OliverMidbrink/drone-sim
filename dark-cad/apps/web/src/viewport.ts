import * as THREE from 'three'
import { planeFrame, type SketchPreview } from '@darkcad/features'
import type { AxisRef, Plane, Vec3 } from '@darkcad/core'
import { pointInPoly } from '@darkcad/sketch'
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js'
import type { Mesh } from '@darkcad/kernel'
import { turntableOrbit } from './nav'

export interface BodyMesh { id: string; mesh: Mesh; /** Feature that first created the body. */ root?: string }
export interface SketchView { id: string; frame: Frame; preview: SketchPreview }
export interface Pick { body: string; face: number; point: THREE.Vector3; normal: THREE.Vector3 }
export type Frame = ReturnType<typeof planeFrame>
/** A shaded sketch region drawn as a pick highlight. */
export interface ProfileHL { frame: Frame; outer: [number, number][]; holes: [number, number][][]; kind: 'selected' | 'hover' }
export interface AxisHL { a: [number, number, number]; b: [number, number, number]; kind: 'selected' | 'hover' }
export interface ProfileHit { sketch: string; pt: [number, number]; hl: ProfileHL }
export interface AxisHit { ref: AxisRef; hl: AxisHL; kind: 'line' | 'construction' | 'sketch-axis' | 'world' }

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
  private sketchGroup = new THREE.Group()
  private pickGroup = new THREE.Group()
  private sketchViews: SketchView[] = []
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
  private planes?: { group: THREE.Group; items: { key: string; frame: Frame; mesh: THREE.Mesh; line: THREE.LineLoop }[]; cb: (key: string) => void; size: number }
  private planeViews = new THREE.Group()
  private gizmo?: { ctl: TransformControls; obj: THREE.Object3D; helper: THREE.Object3D; dragging: boolean; onChange?: (pos: Vec3, rot: Vec3) => void }

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
    this.scene.add(this.ground, this.group, this.overlay, this.sketchGroup, this.pickGroup, this.planeViews)
    this.orient = this.orientFor(new THREE.Vector3(0.6, -0.8, 0.55), Z)
    this.wireMouse()
    new ResizeObserver(() => this.resize()).observe(host)
    this.resize()
    const loop = (now: number) => {
      if (this.disposed) return
      this.stepAnim(now)
      if (this.gizmo) this.gizmo.ctl.camera = this.camera
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
    for (const o of this.pickGroup.children) { const m = (o as any).material; if (m?.resolution) m.resolution.set(w, h) }
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

  /** Turntable orbit by a pointer drag (pixels): yaw about world Z, so the view stays upright. */
  orbit(dx: number, dy: number) {
    if (this.locked) return
    this.anim = undefined
    this.orient.copy(turntableOrbit(this.orient, dx, dy))
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

  /**
   * Finished sketches, drawn as lines (and a faint fill) in global coordinates so they stay visible next to the solids.
   * They ignore depth, so a sketch buried inside a body can still be seen. The sketch being edited is drawn by the overlay instead.
   */
  setSketches(sketches: SketchView[], selected: string[] = []) {
    this.sketchViews = sketches
    for (const c of [...this.sketchGroup.children]) {
      this.sketchGroup.remove(c)
      c.traverse((o: any) => { o.geometry?.dispose(); o.material?.dispose() })
    }
    for (const sk of sketches) {
      const on = selected.includes(sk.id)
      const f = sk.frame
      const x = new THREE.Vector3(...f.xdir), n = new THREE.Vector3(...f.dir), y = n.clone().cross(x), o = new THREE.Vector3(...f.origin)
      const world = ([u, v]: [number, number]) => o.clone().addScaledVector(x, u).addScaledVector(y, v)
      const g = new THREE.Group()
      g.name = sk.id
      g.renderOrder = 10
      const verts: number[] = []
      for (const c of sk.preview.curves) {
        const pts = c.pts.map(world)
        const m = c.closed ? pts.length : pts.length - 1
        for (let i = 0; i < m; i++) { const a = pts[i], b = pts[(i + 1) % pts.length]; verts.push(a.x, a.y, a.z, b.x, b.y, b.z) }
      }
      if (verts.length) {
        const lg = new THREE.BufferGeometry()
        lg.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3))
        const lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: on ? 0xf5a623 : 0x5aa9ff, transparent: true, depthTest: false }))
        lines.renderOrder = 11
        g.add(lines)
      }
      for (const r of sk.preview.regions) {
        const shape = new THREE.Shape(r.outer.map(([u, v]) => new THREE.Vector2(u, v)))
        for (const h of r.holes) shape.holes.push(new THREE.Path(h.map(([u, v]) => new THREE.Vector2(u, v))))
        const mesh = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({ color: on ? 0xf5a623 : 0x5aa9ff, transparent: true, opacity: on ? 0.22 : 0.12, side: THREE.DoubleSide, depthTest: false, depthWrite: false }))
        mesh.matrixAutoUpdate = false
        mesh.matrix.makeBasis(x, y, n).setPosition(o)
        mesh.renderOrder = 10
        g.add(mesh)
      }
      this.sketchGroup.add(g)
    }
  }

  /**
   * A small PNG of the model from an isometric view, for the project browser. Renders one frame with only the bodies
   * visible, from a throw-away camera, so the user's view is untouched. Returns undefined when there is nothing to show.
   */
  snapshot(width = 320, height = 240): string | undefined {
    const box = new THREE.Box3().setFromObject(this.group)
    if (box.isEmpty() || this.disposed) return
    const hidden = [this.ground, this.sketchGroup, this.pickGroup, this.planeViews, this.overlay].map((o) => [o, o.visible] as const)
    hidden.forEach(([o]) => { o.visible = false })
    if (this.planes) this.planes.group.visible = false
    try {
      const c = box.getCenter(new THREE.Vector3()), r = (box.getSize(new THREE.Vector3()).length() / 2) || 10
      const w = this.renderer.domElement.width, h = this.renderer.domElement.height
      const cam = new THREE.PerspectiveCamera(FOV, w / h, 0.1, 20000)
      // fit the bounding sphere into the *narrower* field of view of the crop
      const fov = (FOV * Math.PI) / 360
      const need = r / Math.sin(Math.min(fov, Math.atan(Math.tan(fov) * (w / h))))
      cam.position.copy(c).addScaledVector(new THREE.Vector3(0.6, -0.8, 0.55).normalize(), need * 1.08)
      cam.up.set(0, 0, 1); cam.lookAt(c); cam.updateMatrixWorld(true)
      this.renderer.render(this.scene, cam)
      const out = document.createElement('canvas')
      out.width = width; out.height = height
      const g = out.getContext('2d')!
      g.fillStyle = '#111'; g.fillRect(0, 0, width, height)
      // centre-crop the frame to the thumbnail's aspect ratio
      const k = Math.min(w / width, h / height)
      const sw = width * k, sh = height * k
      g.drawImage(this.renderer.domElement, (w - sw) / 2, (h - sh) / 2, sw, sh, 0, 0, width, height)
      return out.toDataURL('image/png')
    } finally {
      hidden.forEach(([o, v]) => { o.visible = v })
      if (this.planes) this.planes.group.visible = true
    }
  }

  /** Number of finished sketches currently drawn (debug/test hook). */
  get sketchCount() { return this.sketchGroup.children.length }
  /** Shaded regions across the visible sketches (a test hook for the fill logic). */
  get sketchRegionCount() { return this.sketchViews.reduce((n, s) => n + s.preview.regions.length, 0) }

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


  // ---------------------------------------------------------------- picking sketch profiles and axes (revolve)
  /** Draw the picked profiles and axes (selected: orange, hover: white). */
  setPickHighlights(profiles: ProfileHL[], axes: AxisHL[]) {
    for (const c of [...this.pickGroup.children]) { this.pickGroup.remove(c); c.traverse((o: any) => { o.geometry?.dispose(); o.material?.dispose() }) }
    for (const hl of profiles) {
      const f = hl.frame
      const x = new THREE.Vector3(...f.xdir), n = new THREE.Vector3(...f.dir), y = n.clone().cross(x), o = new THREE.Vector3(...f.origin)
      const shape = new THREE.Shape(hl.outer.map(([u, v]) => new THREE.Vector2(u, v)))
      for (const h of hl.holes) shape.holes.push(new THREE.Path(h.map(([u, v]) => new THREE.Vector2(u, v))))
      const mesh = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({ color: hl.kind === 'selected' ? 0xf5a623 : 0xffffff, transparent: true, opacity: hl.kind === 'selected' ? 0.55 : 0.28, side: THREE.DoubleSide, depthTest: false, depthWrite: false }))
      mesh.matrixAutoUpdate = false
      mesh.matrix.makeBasis(x, y, n).setPosition(o)
      mesh.renderOrder = 20
      this.pickGroup.add(mesh)
    }
    const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1
    for (const hl of axes) {
      const g = new LineSegmentsGeometry()
      g.setPositions([...hl.a, ...hl.b])
      const m = new LineMaterial({ color: hl.kind === 'selected' ? 0xf5a623 : 0xffffff, linewidth: hl.kind === 'selected' ? 4 : 3, depthTest: false, transparent: true, opacity: 0.95 })
      m.resolution.set(w, h)
      const line = new LineSegments2(g, m)
      line.renderOrder = 21
      this.pickGroup.add(line)
    }
  }

  /** The sketch profile (closed region) under the pointer, among the visible sketches. Smaller regions win over larger ones. */
  pickProfile(clientX: number, clientY: number): ProfileHit | undefined {
    this.ray.setFromCamera(this.ndc(clientX, clientY), this.camera)
    let best: { t: number; area: number; hit: ProfileHit } | undefined
    for (const sk of this.sketchViews) {
      const f = sk.frame
      const n = new THREE.Vector3(...f.dir), o = new THREE.Vector3(...f.origin), x = new THREE.Vector3(...f.xdir), y = n.clone().cross(x)
      const p = this.ray.ray.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(n, o), new THREE.Vector3())
      if (!p) continue
      const t = p.distanceTo(this.ray.ray.origin), d = p.sub(o), pt: [number, number] = [d.dot(x), d.dot(y)]
      for (const r of sk.preview.regions) {
        if (!pointInPoly(pt, r.outer) || r.holes.some((h) => pointInPoly(pt, h))) continue
        let area = 0
        for (let i = 0; i < r.outer.length; i++) { const a = r.outer[i], b = r.outer[(i + 1) % r.outer.length]; area += a[0] * b[1] - b[0] * a[1] }
        area = Math.abs(area) / 2
        if (!best || t < best.t - 1e-6 || (Math.abs(t - best.t) <= 1e-6 && area < best.area)) best = { t, area, hit: { sketch: sk.id, pt, hl: { frame: f, outer: r.outer, holes: r.holes, kind: 'hover' } } }
      }
    }
    return best?.hit
  }

  /** Client position of a world point, clipped against the camera's near plane. Returns undefined when fully behind it. */
  private clientSegment(a: THREE.Vector3, b: THREE.Vector3): [{ x: number; y: number }, { x: number; y: number }] | undefined {
    const inv = this.camera.matrixWorldInverse
    let va = a.clone().applyMatrix4(inv), vb = b.clone().applyMatrix4(inv)
    if (this.camera instanceof THREE.PerspectiveCamera) {
      const near = -this.camera.near * 2
      if (va.z > near && vb.z > near) return
      if (va.z > near) va = va.clone().lerp(vb, (near - va.z) / (vb.z - va.z))
      else if (vb.z > near) vb = vb.clone().lerp(va, (near - vb.z) / (va.z - vb.z))
    }
    const r = this.renderer.domElement.getBoundingClientRect()
    const px = (v: THREE.Vector3) => { const p = v.clone().applyMatrix4(this.camera.matrixWorld).project(this.camera); return { x: r.left + ((p.x + 1) / 2) * r.width, y: r.top + ((1 - p.y) / 2) * r.height } }
    return [px(va), px(vb)]
  }

  /** The straight line (drawn, construction, sketch axis or world axis) nearest to the pointer, within `tol` pixels. */
  pickAxis(clientX: number, clientY: number, tol = 8): AxisHit | undefined {
    type Cand = { ref: AxisRef; a: THREE.Vector3; b: THREE.Vector3; kind: AxisHit['kind'] }
    const cands: Cand[] = []
    const HALF = 400
    for (const sk of this.sketchViews) {
      const f = sk.frame
      const x = new THREE.Vector3(...f.xdir), n = new THREE.Vector3(...f.dir), y = n.clone().cross(x), o = new THREE.Vector3(...f.origin)
      const w = ([u, v]: [number, number]) => o.clone().addScaledVector(x, u).addScaledVector(y, v)
      for (const l of sk.preview.lines) cands.push({ ref: { kind: 'sketch', sketch: sk.id, id: l.id }, a: w(l.a), b: w(l.b), kind: l.construction ? 'construction' : 'line' })
      cands.push({ ref: { kind: 'sketch', sketch: sk.id, id: 'axis-x' }, a: w([-HALF, 0]), b: w([HALF, 0]), kind: 'sketch-axis' })
      cands.push({ ref: { kind: 'sketch', sketch: sk.id, id: 'axis-y' }, a: w([0, -HALF]), b: w([0, HALF]), kind: 'sketch-axis' })
    }
    cands.push({ ref: { kind: 'world', axis: 'X' }, a: new THREE.Vector3(-HALF, 0, 0), b: new THREE.Vector3(HALF, 0, 0), kind: 'world' })
    cands.push({ ref: { kind: 'world', axis: 'Y' }, a: new THREE.Vector3(0, -HALF, 0), b: new THREE.Vector3(0, HALF, 0), kind: 'world' })
    cands.push({ ref: { kind: 'world', axis: 'Z' }, a: new THREE.Vector3(0, 0, -HALF), b: new THREE.Vector3(0, 0, HALF), kind: 'world' })
    const rank = { construction: 0, line: 0, 'sketch-axis': 1, world: 2 } as const
    let best: { d: number; c: Cand } | undefined
    for (const c of cands) {
      const s = this.clientSegment(c.a, c.b)
      if (!s) continue
      const [p, q] = s
      const dx = q.x - p.x, dy = q.y - p.y, l2 = dx * dx + dy * dy
      const t = l2 < 1e-9 ? 0 : Math.max(0, Math.min(1, ((clientX - p.x) * dx + (clientY - p.y) * dy) / l2))
      const d = Math.hypot(clientX - (p.x + t * dx), clientY - (p.y + t * dy))
      if (d > tol) continue
      // closer wins; within 2 px, the more specific kind (drawn line over sketch axis over world axis) wins
      if (!best || d < best.d - 2 || (Math.abs(d - best.d) <= 2 && rank[c.kind] < rank[best.c.kind])) best = { d, c }
    }
    if (!best) return
    const c = best.c
    return { ref: c.ref, kind: c.kind, hl: { a: [c.a.x, c.a.y, c.a.z], b: [c.b.x, c.b.y, c.b.z], kind: 'hover' } }
  }

  // ---------------------------------------------------------------- plane picking (new sketch)
  /**
   * Show the three origin planes (and any construction planes) in yellow; calls back with the key of the one the
   * user clicks: 'XY' | 'XZ' | 'YZ', or `plane:<id>` for a construction plane.
   */
  startPlanePick(cb: (key: string) => void, extra: { id: string; label: string; frame: Frame }[] = []) {
    this.stopPlanePick()
    const box = new THREE.Box3().setFromObject(this.group)
    // Sized from what is on screen (about 40% of the view height), but never smaller than the model needs.
    const visibleH = 2 * this.dist * Math.tan((FOV * Math.PI) / 360)
    const diag = box.isEmpty() ? 0 : box.getSize(new THREE.Vector3()).length()
    const size = Math.min(400, Math.max(visibleH * 0.4, diag * 0.6, 10))
    const group = new THREE.Group()
    const items: { key: string; frame: Frame; mesh: THREE.Mesh; line: THREE.LineLoop }[] = []
    const entries = [...(['XY', 'XZ', 'YZ'] as Plane[]).map((name) => ({ key: name as string, label: PLANE_LABEL[name], frame: planeFrame(name, 0) })), ...extra.map((e) => ({ key: `plane:${e.id}`, label: e.label, frame: e.frame }))]
    for (const { key, label: text, frame: f } of entries) {
      const x = new THREE.Vector3(...f.xdir), n = new THREE.Vector3(...f.dir), y = n.clone().cross(x), o = new THREE.Vector3(...f.origin)
      const rot = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, n))
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshBasicMaterial({ color: YELLOW, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false }))
      mesh.quaternion.copy(rot)
      mesh.position.copy(o).addScaledVector(x, size / 2).addScaledVector(y, size / 2) // quadrant away from the others
      const pts = [[0, 0], [size, 0], [size, size], [0, size]].map(([u, v]) => o.clone().addScaledVector(x, u).addScaledVector(y, v))
      const line = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: YELLOW, transparent: true, opacity: 0.85 }))
      const label = this.labelSprite(text)
      label.position.copy(o).addScaledVector(x, size * 0.5).addScaledVector(y, size * 0.5)
      label.scale.set(size * 0.34, size * 0.085, 1)
      group.add(mesh, line, label)
      items.push({ key, frame: f, mesh, line })
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
  planeScreenPoint(key: string): { x: number; y: number } | undefined {
    const item = this.planes?.items.find((i) => i.key === key)
    if (!this.planes || !item) return
    const f = item.frame
    const x = new THREE.Vector3(...f.xdir), n = new THREE.Vector3(...f.dir), y = n.clone().cross(x), o = new THREE.Vector3(...f.origin)
    const r = this.renderer.domElement.getBoundingClientRect(), size = this.planes.size
    let fallback: { x: number; y: number } | undefined
    for (const k of [0.5, 0.75, 0.3, 0.9, 0.15]) for (const m of [0.5, 0.75, 0.3, 0.9, 0.15]) {
      const p = o.clone().addScaledVector(x, k * size).addScaledVector(y, m * size).project(this.camera)
      const pt = { x: r.left + ((p.x + 1) / 2) * r.width, y: r.top + ((1 - p.y) / 2) * r.height }
      fallback ??= pt
      // must hit this plane and not sit under a floating panel
      const safe = [[0, 0], [4, 0], [-4, 0], [0, 4], [0, -4]].every(([dx, dy]) => this.planeAt(pt.x + dx, pt.y + dy) === key && document.elementFromPoint(pt.x + dx, pt.y + dy) === this.renderer.domElement)
      if (safe) return pt
    }
    return fallback
  }

  /** Planes that stay drawn in the model (construction planes and the origin planes the user switched on). */
  setPlaneViews(views: { id: string; frame: Frame; label: string; selected?: boolean }[], origin: { axes: ('X' | 'Y' | 'Z')[]; point: boolean } = { axes: [], point: false }) {
    for (const c of [...this.planeViews.children]) { this.planeViews.remove(c); c.traverse((o: any) => { o.geometry?.dispose(); o.material?.map?.dispose(); o.material?.dispose() }) }
    const box = new THREE.Box3().setFromObject(this.group)
    const size = Math.min(300, Math.max(box.isEmpty() ? 0 : box.getSize(new THREE.Vector3()).length() * 0.9, 40))
    for (const v of views) {
      const f = v.frame
      const x = new THREE.Vector3(...f.xdir), n = new THREE.Vector3(...f.dir), y = n.clone().cross(x), o = new THREE.Vector3(...f.origin)
      const color = v.selected ? 0xf5a623 : 0xffd60a
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: v.selected ? 0.22 : 0.1, side: THREE.DoubleSide, depthWrite: false }))
      mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, n))
      mesh.position.copy(o).addScaledVector(x, size / 2).addScaledVector(y, size / 2)
      const pts = [[0, 0], [size, 0], [size, size], [0, size]].map(([u, w]) => o.clone().addScaledVector(x, u).addScaledVector(y, w))
      const line = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.6 }))
      const label = this.labelSprite(v.label)
      label.position.copy(o).addScaledVector(x, size * 0.78).addScaledVector(y, size * 0.93)
      label.scale.set(size * 0.3, size * 0.075, 1)
      this.planeViews.add(mesh, line, label)
    }
    const len = Math.max(size * 0.8, 30)
    const dirs = { X: [1, 0, 0], Y: [0, 1, 0], Z: [0, 0, 1] } as const
    const colors = { X: 0xff5d5d, Y: 0x46a758, Z: 0x3291ff }
    for (const a of origin.axes) {
      const d = dirs[a]
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-d[0] * len * 0.25, -d[1] * len * 0.25, -d[2] * len * 0.25), new THREE.Vector3(d[0] * len, d[1] * len, d[2] * len)]), new THREE.LineBasicMaterial({ color: colors[a], depthTest: false, transparent: true }))
      line.renderOrder = 12
      const tag = this.labelSprite(a)
      ;(tag.material as THREE.SpriteMaterial).color.set(colors[a])
      tag.position.set(d[0] * len * 1.06, d[1] * len * 1.06, d[2] * len * 1.06)
      tag.scale.set(len * 0.2, len * 0.05, 1)
      this.planeViews.add(line, tag)
    }
    if (origin.point) {
      const dot = new THREE.Mesh(new THREE.SphereGeometry(Math.max(len * 0.025, 0.8), 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }))
      dot.renderOrder = 12
      this.planeViews.add(dot)
    }
  }

  /** The point of the ground plane (z = 0) under a client position, e.g. where a dragged design is dropped. */
  groundPoint(clientX: number, clientY: number): Vec3 | undefined {
    this.ray.setFromCamera(this.ndc(clientX, clientY), this.camera)
    const hit = this.ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), new THREE.Vector3())
    return hit ? [Math.round(hit.x * 100) / 100, Math.round(hit.y * 100) / 100, 0] : undefined
  }

  // ---------------------------------------------------------------- move / rotate gizmo
  /** True while the pointer is over (or dragging) a gizmo handle; clicks there must not pick faces. */
  get gizmoHot() { return !!this.gizmo && (this.gizmo.dragging || !!this.gizmo.ctl.axis) }

  /**
   * Shows a move or rotate gizmo at a component's placement (position mm, rotation degrees about X, then Y, then Z).
   * `onChange` is called continuously while the user drags. Pass `undefined` to remove it.
   */
  setGizmo(target: { pos: Vec3; rot: Vec3; mode: 'translate' | 'rotate' } | undefined, onChange?: (pos: Vec3, rot: Vec3) => void) {
    if (!target) {
      if (!this.gizmo) return
      this.gizmo.ctl.detach(); this.scene.remove(this.gizmo.helper, this.gizmo.obj); this.gizmo.ctl.dispose()
      this.gizmo = undefined
      return
    }
    if (!this.gizmo) {
      const obj = new THREE.Object3D()
      const ctl = new TransformControls(this.camera, this.renderer.domElement)
      const helper = ctl.getHelper()
      const g = { ctl, obj, helper, dragging: false, onChange } as NonNullable<Viewport['gizmo']>
      ctl.addEventListener('dragging-changed', (e: any) => { g.dragging = !!e.value })
      ctl.addEventListener('objectChange', () => {
        const eu = new THREE.Euler().setFromQuaternion(obj.quaternion, 'ZYX'), r = (v: number) => Math.round(((v * 180) / Math.PI) * 1000) / 1000
        g.onChange?.([Math.round(obj.position.x * 1000) / 1000, Math.round(obj.position.y * 1000) / 1000, Math.round(obj.position.z * 1000) / 1000], [r(eu.x) + 0, r(eu.y) + 0, r(eu.z) + 0])
      })
      ctl.setSize(0.85)
      ctl.attach(obj)
      this.scene.add(obj, helper)
      this.gizmo = g
    }
    const g = this.gizmo
    g.onChange = onChange
    g.ctl.setMode(target.mode)
    if (!g.dragging) {
      g.obj.position.set(...target.pos)
      g.obj.rotation.set((target.rot[0] * Math.PI) / 180, (target.rot[1] * Math.PI) / 180, (target.rot[2] * Math.PI) / 180, 'ZYX')
      g.obj.updateMatrixWorld(true)
    }
  }

  private labelSprite(text: string): THREE.Sprite {
    const c = document.createElement('canvas')
    c.width = 256; c.height = 64
    const g = c.getContext('2d')!
    g.font = '600 30px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'
    g.fillStyle = '#ffd60a'; g.fillText(text, 128, 34)
    return new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthTest: false }))
  }

  private planeAt(clientX: number, clientY: number): string | undefined {
    if (!this.planes) return
    this.ray.setFromCamera(this.ndc(clientX, clientY), this.camera)
    const hit = this.ray.intersectObjects(this.planes.items.map((i) => i.mesh), false)[0]
    return hit && this.planes.items.find((i) => i.mesh === hit.object)?.key
  }

  // ---------------------------------------------------------------- mouse
  private wireMouse() {
    const el = this.renderer.domElement
    let drag: { x: number; y: number; mode: 'orbit' | 'pan'; moved: boolean } | undefined
    el.addEventListener('contextmenu', (e) => e.preventDefault())
    el.addEventListener('pointerdown', (e) => {
      if (this.gizmo?.ctl.axis) return // the move/rotate handle owns this drag
      el.setPointerCapture(e.pointerId)
      drag = { x: e.clientX, y: e.clientY, mode: e.button === 1 || e.button === 2 || e.shiftKey ? 'pan' : 'orbit', moved: false }
    })
    el.addEventListener('pointermove', (e) => {
      if (this.planes && !drag) {
        const hover = this.planeAt(e.clientX, e.clientY)
        for (const it of this.planes.items) {
          const on = it.key === hover
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
