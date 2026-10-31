import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { Mesh } from '@darkcad/kernel'

export interface BodyMesh { id: string; mesh: Mesh }
export interface Pick { body: string; face: number; point: THREE.Vector3; normal: THREE.Vector3 }

const BODY_COLORS = [0xb9c3d3, 0x8da2c4, 0xc4b79a, 0x93b8a0, 0xb893a0]

export class Viewport {
  readonly renderer: THREE.WebGLRenderer
  readonly scene = new THREE.Scene()
  readonly camera = new THREE.PerspectiveCamera(40, 1, 0.1, 5000)
  readonly controls: OrbitControls
  private group = new THREE.Group()
  private overlay = new THREE.Group()
  private meshes: { id: string; obj: THREE.Mesh; mesh: Mesh }[] = []
  private ray = new THREE.Raycaster()
  private raf = 0
  private hover?: { body: string; face: number }
  private selected: { body: string; face: number }[] = []
  private disposed = false

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    this.renderer.setClearColor(0x0a0a0a)
    host.appendChild(this.renderer.domElement)
    this.camera.up.set(0, 0, 1) // CAD convention: Z up
    this.camera.position.set(60, -80, 50)
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.scene.add(new THREE.HemisphereLight(0xdfe7ff, 0x20242c, 1.1))
    const key = new THREE.DirectionalLight(0xffffff, 1.6)
    key.position.set(80, -60, 120)
    this.scene.add(key)
    const grid = new THREE.GridHelper(200, 20, 0x2e2e2e, 0x1a1a1a)
    grid.rotation.x = Math.PI / 2
    this.scene.add(grid, new THREE.AxesHelper(20), this.group, this.overlay)
    new ResizeObserver(() => this.resize()).observe(host)
    this.resize()
    const loop = () => { if (this.disposed) return; this.controls.update(); this.renderer.render(this.scene, this.camera); this.raf = requestAnimationFrame(loop) }
    loop()
  }

  dispose() { this.disposed = true; cancelAnimationFrame(this.raf); this.renderer.dispose(); this.renderer.domElement.remove() }

  private resize() {
    const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1
    this.renderer.setSize(w, h)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
  }

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
    this.refreshOverlay()
    if (fit) this.fit()
  }

  fit() {
    const box = new THREE.Box3().setFromObject(this.group)
    if (box.isEmpty()) return
    const c = box.getCenter(new THREE.Vector3()), r = box.getSize(new THREE.Vector3()).length() / 2 || 10
    const dir = new THREE.Vector3(0.6, -0.8, 0.55).normalize()
    this.camera.position.copy(c).addScaledVector(dir, r * 3.4)
    this.controls.target.copy(c)
  }

  pick(clientX: number, clientY: number): Pick | undefined {
    const r = this.renderer.domElement.getBoundingClientRect()
    this.ray.setFromCamera(new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1), this.camera)
    const hits = this.ray.intersectObjects(this.meshes.map((m) => m.obj), false)
    const hit = hits[0]
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
}
