"""The SLAM side of the drone: one worker process per camera, and the link the simulator talks to it through.

The worker is a sealed box. It is handed exactly what a real onboard computer would have:

    * the camera pictures (the fisheye frame, through shared memory),
    * the gyro samples since the previous picture, and the gyro bias the flight computer estimated,
    * optionally the flight computer's inertial position and metric scale (accelerometer, never a map or the truth).

It returns a camera pose in its own frame (the camera at its first picture is the origin, and its
unit of length is whatever the first baseline was: a camera alone cannot know metres) plus its map.
Nothing about the simulated world, the drone's true position or orientation, or the terrain ever
crosses this boundary; `SlamLink.submit` is the only way in and its arguments are pictures and gyro.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

SLOTS = 6  # pictures in flight per camera
MAP_EVERY = 0.5  # s between map snapshots sent back


@dataclass
class FixPolicy:
    """How a camera's SLAM treats its anchor fixes (loop-closure-like snaps of the pose). The defaults are the raw behaviour: apply every fix whole."""

    gain: float = 1.0  # share of the way the pose moves toward each fix (1 = snap, 0.25 = a quarter per fix: the fix noise is averaged out over several fixes)
    guard: bool = False  # refuse a fix further from the current pose than the inertial data makes plausible (see ImuAidedSlam._fix_budget)
    prior_gate: bool = False  # only let anchors take part whose picture position, under the current pose, is where the inertial uncertainty allows
    blame: bool = False  # find the anchors that pull a suspicious fix away from the inertial pose; ban them after `strikes` offences
    strikes: int = 2
    sigma_m: float = 0.4  # metres: the noise floor of a fix (frozen anchors carry their triangulation errors)
    drift_rel: float = 0.15  # metres of plausible pose error per metre flown since the last accepted fix
    drift_acc: float = 0.05  # m/s^2: inertial acceleration error, so 0.5*a*T^2 after T seconds without a fix
    z_max: float = 3.0  # a fix further than z_max budgets from the current pose is refused
    z_blame: float = 1.5  # ... and one further than z_blame budgets is searched for culprits
    side_check: bool = True  # rig SLAM: refit each camera's side of a joint fix alone; if they disagree, use the side with more inliers
    side_gap_m: float = 0.3  # ... disagreement allowed in position (metres, once the scale is calibrated)
    side_gap_deg: float = 3.0  # ... and in rotation


@dataclass
class MapPolicy:
    """How a camera's SLAM builds and uses its anchor map (see ImuAidedSlam's MapPolicy section). Modelled on ORB-SLAM's map point
    management: create points only with enough parallax, fuse duplicates, match by projection, cull points that keep failing."""

    anchor_min_views: int = 3  # steps that must have triangulated a landmark before it may become an anchor
    anchor_min_angle_deg: float = 3.0  # ... and the widest triangulation angle it must have had
    fuse: bool = True  # link a new anchor candidate to an existing anchor that projects next to it instead of duplicating it
    fuse_px: float = 8.0
    fuse_hamming: int = 50
    fuse_depth_tolerance: float = 0.35
    guided: bool = True  # fixes match anchors near where the current pose projects them, not against the whole map
    guided_px: float = 30.0
    guided_ratio: float = 0.9  # ratio test among the few features near the prediction (looser than the whole-map one)
    global_fallback: bool = True  # ... and against the whole map when that finds too few
    cull: bool = True  # ban anchors that are matched but end up outliers of the fix much more often than inliers
    cull_strikes: int = 3
    rigid_carry: bool = True  # once the metric scale is calibrated a fix moves the live map without rescaling it


@dataclass
class CameraConfig:
    name: str
    fisheye_size: int
    slam_size: int
    slam_fov: float
    cv_camera_to_body: np.ndarray  # extrinsics: calibration, not truth
    lens_fov: float = 160.0  # field of view of the fisheye picture's image circle
    fixes: bool = True  # snap the pose to old anchors (loop-closure-like) now and then
    anchor_window: float = 0.0  # s after the first picture during which new anchors are made; 0 = always
    anchor_refresh: bool = False  # True: anchors follow the live landmarks they sit on (that feeds pose errors back into the map; see raw_slam)
    scale_hint: bool = True  # hold the step length to the inertial displacement once the flight computer's scale is calibrated
    fix_policy: FixPolicy = field(default_factory=FixPolicy)
    lever: np.ndarray = field(default_factory=lambda: np.zeros(3))
    map_policy: MapPolicy | None = None  # None: the raw anchor map (every landmark anchored at once, global matching)  # lens position in the body frame, metres (the rig SLAM needs it; the per-camera one does not)


@dataclass
class SlamResult:
    name: str
    t: float  # time of the picture
    changed: bool  # a new pose (a step and/or an anchor fix)
    rotation: np.ndarray  # camera (OpenCV axes) -> SLAM frame
    position: np.ndarray  # camera centre in the SLAM frame, SLAM units
    stats: dict
    telemetry: dict = field(default_factory=dict)  # numbers from inside the SLAM, for diagnostics


def _expm(v: np.ndarray) -> np.ndarray:
    from imu_slam import rotation_vector_to_matrix

    return rotation_vector_to_matrix(v)


def worker_main(config: CameraConfig, shm_name: str, inbox, outbox) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: F401  (flight/, slam/, viewer/ and 3D-SLAM/ on the path)

    import cv2

    from imu_slam import ImuAidedSlam
    from lens import pinhole_matrix, rectify_maps
    from raw_session import RawMapStore

    cv2.setNumThreads(2)
    if not config.fixes:
        import raw_slam

        raw_slam.LOCALIZE_EVERY = 10**9
    maps = rectify_maps(config.fisheye_size, config.slam_size, config.slam_fov, config.lens_fov)
    intrinsic = pinhole_matrix(config.slam_size, config.slam_fov)
    shm = shared_memory.SharedMemory(name=shm_name)
    frames = np.ndarray((SLOTS, config.fisheye_size, config.fisheye_size, 3), np.uint8, buffer=shm.buf)
    rbc = config.cv_camera_to_body

    def fresh() -> dict:
        return dict(slam=None, body=np.eye(3), last_t=None, map=RawMapStore(), map_at=0.0, spent=0.0, count=0)

    state = fresh()
    while True:
        message = inbox.get()
        if message[0] == "stop":
            break
        if message[0] == "restart":
            state = fresh()
            continue
        _, slot, t, gyro, bias, hint = message
        picture = cv2.remap(frames[slot], *maps, cv2.INTER_LINEAR)
        began = time.perf_counter()
        if state["slam"] is None:
            state["slam"] = ImuAidedSlam(picture, intrinsic)
            state["slam"].policy = config.fix_policy
            state["slam"].map_policy = config.map_policy
            state["slam"].set_motion_hint(hint, keyframe=True)
            state["last_t"] = t
            changed = True
        else:
            for row in gyro:  # integrate the gyro: body now -> body at the first picture
                dt = row[0] - state["last_t"]
                if dt > 0.0:
                    state["body"] = state["body"] @ _expm((row[1:4] - bias) * dt)
                    state["last_t"] = row[0]
            camera_rotation = rbc.T @ state["body"] @ rbc
            state["slam"].use_hint = config.scale_hint
            state["slam"].policy = config.fix_policy
            try:
                changed = state["slam"].process(picture, camera_rotation, hint)
            except Exception:
                import traceback

                crash = Path(__file__).resolve().parents[1] / "out" / f"slam_crash_{config.name}_{int(time.time())}.txt"
                crash.parent.mkdir(exist_ok=True)
                crash.write_text(traceback.format_exc())
                raise
        slam = state["slam"]
        slam.refresh_anchors = config.anchor_refresh
        if config.anchor_window > 0.0 and state.setdefault("t0", t) + config.anchor_window < t:
            slam.make_anchors = False
        if changed and state["slam"].commits:
            _update_map(state["map"], slam, picture)
        state["spent"] += time.perf_counter() - began
        state["count"] += 1
        stats = dict(
            tracks=len(slam.track_points), steps=slam.commits, fixes=slam.fixes, restarts=slam.restarts, landmarks=len(slam.landmarks),
            skipped=slam.skipped_no_scale, failed=slam.failed_fits, refused=slam.refused_fixes, banned=len(slam.banned), corrected=slam.rotation_corrections, ms=1000.0 * state["spent"] / state["count"],
        )
        outbox.put(("result", SlamResult(config.name, t, bool(changed), slam.world_rotation.copy(), slam.world_position.copy(), stats, dict(slam.telemetry)), slot))
        now = time.perf_counter()
        if now - state["map_at"] > MAP_EVERY and state["map"].count:
            state["map_at"] = now
            store = state["map"]
            outbox.put(("map", config.name, store.positions[: store.count].astype(np.float32), (store.colors[: store.count] * 255).astype(np.uint8)))
    shm.close()


def rig_worker_main(configs: list, shm_names: dict, inbox, outbox) -> None:
    """One process for the whole rig: both cameras' pictures go into the same RigSlam."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import paths  # noqa: F401

    import cv2

    from lens import pinhole_matrix, rectify_maps
    from raw_session import RawMapStore
    from rig_slam import ID_BASE, RigCamera, RigSlam

    cv2.setNumThreads(2)
    first = configs[0]
    if not first.fixes:
        import raw_slam

        raw_slam.LOCALIZE_EVERY = 10**9
    maps = rectify_maps(first.fisheye_size, first.slam_size, first.slam_fov, first.lens_fov)
    intrinsic = pinhole_matrix(first.slam_size, first.slam_fov)
    shms = {c.name: shared_memory.SharedMemory(name=shm_names[c.name]) for c in configs}
    frames = {c.name: np.ndarray((SLOTS, c.fisheye_size, c.fisheye_size, 3), np.uint8, buffer=shms[c.name].buf) for c in configs}
    specs = [RigCamera(c.name, c.cv_camera_to_body, np.asarray(c.lever, dtype=np.float64)) for c in configs]

    def fresh() -> dict:
        return dict(slam=None, body=np.eye(3), last_t=None, map=RawMapStore(), map_at=0.0, spent=0.0, count=0)

    state = fresh()
    while True:
        message = inbox.get()
        if message[0] == "stop":
            break
        if message[0] == "restart":
            state = fresh()
            continue
        _, slot, t, gyro, bias, hint, names = message
        pictures = {name: cv2.remap(frames[name][slot], *maps, cv2.INTER_LINEAR) for name in names}
        began = time.perf_counter()
        if state["slam"] is None:
            blank = np.zeros((first.slam_size, first.slam_size, 3), np.uint8)
            state["slam"] = RigSlam(specs, intrinsic, {c.name: pictures.get(c.name, blank) for c in configs})
            state["slam"].policy = first.fix_policy
            state["slam"].map_policy = first.map_policy
            state["slam"].set_hint(hint)
            for cam in state["slam"].cams:
                cam.key_hint = None if hint is None else np.asarray(hint[0], dtype=np.float64).copy()
            state["last_t"] = t
            changed = True
        else:
            for row in gyro:
                dt = row[0] - state["last_t"]
                if dt > 0.0:
                    state["body"] = state["body"] @ _expm((row[1:4] - bias) * dt)
                    state["last_t"] = row[0]
            state["slam"].use_hint = configs[0].scale_hint
            try:
                changed = state["slam"].process(pictures, state["body"], hint)
            except Exception:
                import traceback

                crash = Path(__file__).resolve().parents[1] / "out" / f"slam_crash_rig_{int(time.time())}.txt"
                crash.parent.mkdir(exist_ok=True)
                crash.write_text(traceback.format_exc())
                raise
        slam = state["slam"]
        slam.make_anchors = not (configs[0].anchor_window > 0.0 and state.setdefault("t0", t) + configs[0].anchor_window < t)
        if changed and slam.commits:
            _update_rig_map(state["map"], slam, pictures, ID_BASE)
        state["spent"] += time.perf_counter() - began
        state["count"] += 1
        ms = 1000.0 * state["spent"] / state["count"]
        per_camera = {
            c.spec.name: dict(tracks=len(c.track_points), steps=slam.commits, fixes=slam.fixes, restarts=slam.restarts, landmarks=sum(1 for g in slam.landmarks if g // ID_BASE == c.index), skipped=0, failed=0, corrected=0.0, ms=ms)
            for c in slam.cams
        }
        stats = dict(
            tracks=sum(len(c.track_points) for c in slam.cams), steps=slam.commits, fixes=slam.fixes, restarts=slam.restarts, landmarks=len(slam.landmarks),
            skipped=slam.skipped_no_scale, failed=slam.failed_fits, corrected=slam.rotation_corrections, ms=ms, refused=slam.refused_fixes, banned=len(slam.banned), cams=per_camera,
        )
        outbox.put(("result", SlamResult("rig", t, bool(changed), slam.rotation.copy(), slam.position.copy(), stats, dict(slam.telemetry)), slot))
        now = time.perf_counter()
        if now - state["map_at"] > MAP_EVERY and state["map"].count:
            state["map_at"] = now
            store = state["map"]
            outbox.put(("map", "rig", store.positions[: store.count].astype(np.float32), (store.colors[: store.count] * 255).astype(np.uint8)))
    for shm in shms.values():
        shm.close()


def _update_rig_map(store, slam, pictures: dict, id_base: int) -> None:
    new = [track for track in slam.landmarks if track not in store.row]
    colors: dict[int, np.ndarray] = {}
    if new:
        for cam in slam.cams:
            picture = pictures.get(cam.spec.name)
            if picture is None:
                continue
            pixel = dict(zip(cam.track_ids.tolist(), cam.track_points))
            height, width = picture.shape[:2]
            for track in new:
                if track in pixel:
                    x, y = np.round(pixel[track]).astype(int)
                    colors[track] = picture[min(max(y, 0), height - 1), min(max(x, 0), width - 1)][::-1].astype(np.float64) / 255.0
    store.update(slam.landmarks, colors)


def _update_map(store, slam, picture) -> None:
    new = [track for track in slam.landmarks if track not in store.row]
    colors: dict[int, np.ndarray] = {}
    if new:
        pixel = dict(zip(slam.track_ids.tolist(), slam.track_points))
        height, width = picture.shape[:2]
        for track in new:
            if track in pixel:
                x, y = np.round(pixel[track]).astype(int)
                colors[track] = picture[min(max(y, 0), height - 1), min(max(x, 0), width - 1)][::-1].astype(np.float64) / 255.0
    store.update(slam.landmarks, colors)


class SlamLink:
    """Owns the workers. `submit` sends a picture and the gyro since the last one; `poll` collects what came back."""

    def __init__(self, cameras: list[CameraConfig]) -> None:
        context = mp.get_context("spawn")  # a clean process: it must not inherit the simulator's world
        self.configs = {c.name: c for c in cameras}
        self.shm: dict[str, shared_memory.SharedMemory] = {}
        self.frames: dict[str, np.ndarray] = {}
        self.inboxes, self.processes = {}, {}
        self.outbox = context.Queue()
        self.next_slot = {c.name: 0 for c in cameras}
        self.pending: dict[str, list[float]] = {c.name: [] for c in cameras}  # times of pictures in flight
        self.results: list[SlamResult] = []
        self.maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.stats: dict[str, dict] = {}
        self.last_result: dict[str, SlamResult] = {}
        for c in cameras:
            size = SLOTS * c.fisheye_size * c.fisheye_size * 3
            self.shm[c.name] = shared_memory.SharedMemory(create=True, size=size)
            self.frames[c.name] = np.ndarray((SLOTS, c.fisheye_size, c.fisheye_size, 3), np.uint8, buffer=self.shm[c.name].buf)
            self.inboxes[c.name] = context.Queue()
            self.processes[c.name] = context.Process(target=worker_main, args=(c, self.shm[c.name].name, self.inboxes[c.name], self.outbox), daemon=True)
            self.processes[c.name].start()
        self.started: dict[str, float] = {}

    def restart(self) -> None:
        self.wait_idle()
        for name, inbox in self.inboxes.items():
            inbox.put(("restart",))
        self.started.clear()
        self.maps.clear()
        self.stats.clear()
        self.last_result.clear()

    def submit(self, name: str, t: float, picture: np.ndarray, gyro: np.ndarray, bias: np.ndarray, hint=None) -> None:
        """One picture (BGR fisheye), the gyro rows [t, gx, gy, gz] since the previous picture, the flight computer's gyro bias, and
        optionally its inertial state: (body position in metres, SLAM units per metre for this camera, whether that scale is calibrated)."""
        while len(self.pending[name]) >= SLOTS - 1:  # all slots busy: the SLAM is behind, wait for it
            self._collect(block=True)
        slot = self.next_slot[name]
        self.next_slot[name] = (slot + 1) % SLOTS
        self.frames[name][slot] = picture
        self.pending[name].append(t)
        self.started.setdefault(name, t)
        self.inboxes[name].put(("frame", slot, t, gyro, bias, hint))

    def _collect(self, block: bool = False) -> None:
        import queue

        waiting = block
        while True:
            try:
                message = self.outbox.get(timeout=0.5) if waiting else self.outbox.get_nowait()
            except queue.Empty:
                if waiting:
                    dead = [name for name, p in self.processes.items() if not p.is_alive() and self.pending.get(name)]
                    if dead:
                        raise RuntimeError(f"the SLAM worker died: {', '.join(dead)}")
                    continue
                return
            waiting = False
            if message[0] == "result":
                result: SlamResult = message[1]
                self.pending[result.name].remove(result.t)
                self.results.append(result)
                self.stats[result.name] = result.stats
                for camera, stats in result.stats.get("cams", {}).items():
                    self.stats[camera] = stats
                self.last_result[result.name] = result
            elif message[0] == "map":
                self.maps[message[1]] = (message[2], message[3])

    def poll(self) -> list[SlamResult]:
        """Everything that finished since the last call, oldest first."""
        self._collect()
        out, self.results = self.results, []
        return out

    def wait_until(self, t: float) -> None:
        """Block until every picture taken at or before `t` has been processed (keeps the SLAM's latency bounded)."""
        while any(p and p[0] <= t for p in self.pending.values()):
            self._collect(block=True)

    def wait_idle(self) -> None:
        while any(self.pending.values()):
            self._collect(block=True)

    def close(self) -> None:
        for inbox in self.inboxes.values():
            inbox.put(("stop",))
        for process in self.processes.values():
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
        for shm in self.shm.values():
            shm.close()
            shm.unlink()


class RigLink(SlamLink):
    """The same link, with one worker for both cameras (see rig_slam.py). Results come back under the name "rig"; the per-camera numbers are in `stats` too."""

    def __init__(self, cameras: list[CameraConfig]) -> None:
        context = mp.get_context("spawn")
        self.configs = {c.name: c for c in cameras}
        self.shm, self.frames = {}, {}
        for c in cameras:
            self.shm[c.name] = shared_memory.SharedMemory(create=True, size=SLOTS * c.fisheye_size * c.fisheye_size * 3)
            self.frames[c.name] = np.ndarray((SLOTS, c.fisheye_size, c.fisheye_size, 3), np.uint8, buffer=self.shm[c.name].buf)
        self.outbox = context.Queue()
        self.inboxes = {"rig": context.Queue()}
        self.next_slot = 0
        self.pending = {"rig": []}
        self.results, self.maps, self.stats, self.last_result = [], {}, {}, {}
        self.started = {}
        self.processes = {"rig": context.Process(target=rig_worker_main, args=(list(cameras), {n: s.name for n, s in self.shm.items()}, self.inboxes["rig"], self.outbox), daemon=True)}
        self.processes["rig"].start()

    def submit_rig(self, t: float, pictures: dict, gyro: np.ndarray, bias: np.ndarray, hint=None) -> None:
        """The pictures (BGR fisheye, by camera name) taken at time t, the gyro rows since the previous submission, the gyro bias, and the inertial hint."""
        while len(self.pending["rig"]) >= SLOTS - 1:
            self._collect(block=True)
        slot = self.next_slot
        self.next_slot = (slot + 1) % SLOTS
        for name, picture in pictures.items():
            self.frames[name][slot] = picture
        self.pending["rig"].append(t)
        self.started.setdefault("rig", t)
        self.inboxes["rig"].put(("frame", slot, t, gyro, bias, hint, list(pictures)))
