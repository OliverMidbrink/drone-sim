import * as THREE from 'three'
import type { Viewport } from './viewport'

const SIZE = 112
const BAND = 0.62 // |coordinate| beyond this on a face counts as "near that edge"

function faceTexture(text: string): THREE.CanvasTexture {
  const c = document.createElement('canvas')
  c.width = c.height = 128
  const g = c.getContext('2d')!
  g.fillStyle = '#161616'; g.fillRect(0, 0, 128, 128)
  g.strokeStyle = '#2e2e2e'; g.lineWidth = 3; g.strokeRect(1.5, 1.5, 125, 125)
  g.fillStyle = '#e6e6e6'; g.font = '600 19px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'
  g.fillText(text, 64, 66)
  const t = new THREE.CanvasTexture(c)
  t.anisotropy = 4
  return t
}

/**
 * Navigation cube. Front is -Y (Z up). Hover shows the face / edge / corner under the pointer;
 * click snaps the main view to look from there; drag rotates the main view.
 */
export class ViewCube {
  private renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50)
  private highlight: THREE.Mesh
  private ray = new THREE.Raycaster()
  private cube: THREE.Group
  private off: () => void

  constructor(host: HTMLElement, private vp: Viewport) {
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    this.renderer.setSize(SIZE, SIZE)
    host.appendChild(this.renderer.domElement)
    // Built face by face (not BoxGeometry) so every label has a known orientation. For each face: the outward
    // normal, and the "right" and "up" directions an observer outside the cube sees (Z is up; Front is -Y).
    const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z)
    const faces: { label: string; n: THREE.Vector3; right: THREE.Vector3; up: THREE.Vector3 }[] = [
      { label: 'FRONT', n: V(0, -1, 0), right: V(1, 0, 0), up: V(0, 0, 1) },
      { label: 'BACK', n: V(0, 1, 0), right: V(-1, 0, 0), up: V(0, 0, 1) },
      { label: 'RIGHT', n: V(1, 0, 0), right: V(0, 1, 0), up: V(0, 0, 1) },
      { label: 'LEFT', n: V(-1, 0, 0), right: V(0, -1, 0), up: V(0, 0, 1) },
      { label: 'TOP', n: V(0, 0, 1), right: V(1, 0, 0), up: V(0, 1, 0) },
      { label: 'BOTTOM', n: V(0, 0, -1), right: V(1, 0, 0), up: V(0, -1, 0) },
    ]
    this.cube = new THREE.Group()
    for (const f of faces) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: faceTexture(f.label) }))
      m.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(f.right, f.up, f.n)) // plane's +X/+Y/+Z → right/up/outward
      m.position.copy(f.n)
      this.cube.add(m)
    }
    this.scene.add(this.cube)
    this.scene.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)), new THREE.LineBasicMaterial({ color: 0x4a4a4a })))
    this.highlight = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0x3291ff, transparent: true, opacity: 0.8 }))
    this.highlight.visible = false
    this.scene.add(this.highlight)

    const el = this.renderer.domElement
    let down: { x: number; y: number; moved: boolean } | undefined
    el.addEventListener('pointerdown', (e) => { el.setPointerCapture(e.pointerId); down = { x: e.clientX, y: e.clientY, moved: false } })
    el.addEventListener('pointermove', (e) => {
      if (down) {
        const dx = e.clientX - down.x, dy = e.clientY - down.y
        if (!down.moved && Math.hypot(dx, dy) < 4) return
        down.moved = true; down.x = e.clientX; down.y = e.clientY
        this.setHover(undefined)
        this.vp.orbit(dx, dy)
      } else this.setHover(this.regionAt(e))
    })
    el.addEventListener('pointerup', (e) => {
      const d = down
      down = undefined
      if (d && !d.moved && !this.vp.isLocked) { const r = this.regionAt(e); if (r) this.vp.viewFromDirection(new THREE.Vector3(...r)) }
    })
    el.addEventListener('pointerleave', () => { if (!down) this.setHover(undefined) })
    this.off = vp.subscribe(() => this.render())
    this.render()
  }

  dispose() { this.off(); this.renderer.dispose(); this.renderer.domElement.remove() }

  /** Which face / edge / corner is under the pointer, as a sign vector like [1, 0, -1] (or undefined). */
  private regionAt(e: PointerEvent): [number, number, number] | undefined {
    const r = this.renderer.domElement.getBoundingClientRect()
    this.ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), this.camera)
    const hit = this.ray.intersectObjects(this.cube.children, false)[0]
    if (!hit) return
    const p = hit.point
    return [p.x, p.y, p.z].map((v) => (Math.abs(v) > BAND ? Math.sign(v) : 0)) as [number, number, number]
  }

  private setHover(s?: [number, number, number]) {
    if (!s) this.highlight.visible = false
    else {
      const size = s.map((v) => (v === 0 ? 1.2 : 0.4 + 0.04))
      this.highlight.scale.set(size[0], size[1], size[2])
      this.highlight.position.set(s[0] * 0.8, s[1] * 0.8, s[2] * 0.8)
      this.highlight.visible = true
    }
    this.renderer.domElement.style.cursor = s && !this.vp.isLocked ? 'pointer' : ''
    this.render()
  }

  private render() {
    const back = new THREE.Vector3(0, 0, 1).applyQuaternion(this.vp.orient)
    this.camera.position.copy(back).multiplyScalar(7.5)
    this.camera.quaternion.copy(this.vp.orient)
    this.renderer.render(this.scene, this.camera)
  }
}
