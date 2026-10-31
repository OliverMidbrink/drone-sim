import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { turntableOrbit } from './nav'

const start = () => new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(new THREE.Vector3(60, -80, 55), new THREE.Vector3(), new THREE.Vector3(0, 0, 1)))
const up = (q: THREE.Quaternion) => new THREE.Vector3(0, 1, 0).applyQuaternion(q)

describe('turntable orbit', () => {
  it('keeps the view upright however far you drag', () => {
    let q = start()
    for (let i = 0; i < 100; i++) q = turntableOrbit(q, 7, 30)
    for (let i = 0; i < 100; i++) q = turntableOrbit(q, -11, -40)
    expect(up(q).z).toBeGreaterThanOrEqual(0)
    expect(Math.abs(new THREE.Vector3(0, 0, 1).applyQuaternion(q).z)).toBeLessThan(1)
  })

  it('keeps the camera on the sphere around the target and a unit quaternion', () => {
    let q = start()
    const r0 = new THREE.Vector3(0, 0, 100).applyQuaternion(q).length()
    for (const [dx, dy] of [[30, 10], [-12, 55], [200, -40]]) q = turntableOrbit(q, dx, dy)
    expect(new THREE.Vector3(0, 0, 100).applyQuaternion(q).length()).toBeCloseTo(r0, 6)
    expect(q.length()).toBeCloseTo(1, 9)
  })

  it('a huge vertical drag stops at the pole instead of flipping', () => {
    const q = turntableOrbit(start(), 0, 5000)
    expect(up(q).z).toBeGreaterThan(0)
    expect(new THREE.Vector3(0, 0, 1).applyQuaternion(q).z).toBeGreaterThan(0.99)
    const back = turntableOrbit(q, 0, -5000)
    expect(new THREE.Vector3(0, 0, 1).applyQuaternion(back).z).toBeLessThan(-0.99)
    expect(up(back).z).toBeGreaterThan(0)
  })

  it('a horizontal drag only turns about Z', () => {
    const q0 = start(), q1 = turntableOrbit(q0, 80, 0)
    expect(new THREE.Vector3(0, 0, 1).applyQuaternion(q1).z).toBeCloseTo(new THREE.Vector3(0, 0, 1).applyQuaternion(q0).z, 9)
  })
})
