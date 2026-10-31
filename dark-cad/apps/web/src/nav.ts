import * as THREE from 'three'

const Z = new THREE.Vector3(0, 0, 1)
export const ORBIT_RATE = 0.007

const LIMIT = Math.PI / 2 - 0.001

/**
 * Turntable orbit: the camera moves on a sphere around the target by azimuth (about world Z) and elevation,
 * with Z always up. The elevation is clamped just short of the poles, so the view can never flip.
 * Returns the new camera orientation.
 */
export function turntableOrbit(orient: THREE.Quaternion, dx: number, dy: number, rate = ORBIT_RATE): THREE.Quaternion {
  const back = new THREE.Vector3(0, 0, 1).applyQuaternion(orient)
  let phi: number
  if (Math.hypot(back.x, back.y) > 1e-4) phi = Math.atan2(back.y, back.x)
  else {
    // looking straight along Z: the azimuth is carried by the up vector (it points away from the camera's side)
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(orient), s = Math.sign(back.z) || 1
    phi = Math.atan2(-s * up.y, -s * up.x)
  }
  const theta = Math.max(-LIMIT, Math.min(LIMIT, Math.asin(Math.max(-1, Math.min(1, back.z))) + dy * rate))
  phi -= dx * rate
  const dir = new THREE.Vector3(Math.cos(theta) * Math.cos(phi), Math.cos(theta) * Math.sin(phi), Math.sin(theta))
  const m = new THREE.Matrix4().lookAt(dir.multiplyScalar(10), new THREE.Vector3(), Z)
  return new THREE.Quaternion().setFromRotationMatrix(m).normalize()
}
