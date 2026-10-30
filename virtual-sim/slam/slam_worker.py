"""The SLAM side of the drone: one worker process per camera, and the link the simulator talks to it through.

The worker is a sealed box. It is handed exactly what a real onboard computer would have:

    * the camera pictures (the fisheye frame, through shared memory),
    * the gyro samples since the previous picture, and the gyro bias the flight computer estimated.

It returns a camera pose in its own frame (the camera at its first picture is the origin, and its
unit of length is whatever the first baseline was: a camera alone cannot know metres) plus its map.
Nothing about the simulated world, the drone's true position or orientation, or the terrain ever
crosses this boundary; `SlamLink.submit` is the only way in and its arguments are pictures and gyro.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

SLOTS = 6  # pictures in flight per camera
MAP_EVERY = 0.5  # s between map snapshots sent back


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


@dataclass
class SlamResult:
    name: str
    t: float  # time of the picture
    changed: bool  # a new pose (a step and/or an anchor fix)
    rotation: np.ndarray  # camera (OpenCV axes) -> SLAM frame
    position: np.ndarray  # camera centre in the SLAM frame, SLAM units
    stats: dict


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
        _, slot, t, gyro, bias = message
        picture = cv2.remap(frames[slot], *maps, cv2.INTER_LINEAR)
        began = time.perf_counter()
        if state["slam"] is None:
            state["slam"] = ImuAidedSlam(picture, intrinsic)
            state["last_t"] = t
            changed = True
        else:
            for row in gyro:  # integrate the gyro: body now -> body at the first picture
                dt = row[0] - state["last_t"]
                if dt > 0.0:
                    state["body"] = state["body"] @ _expm((row[1:4] - bias) * dt)
                    state["last_t"] = row[0]
            camera_rotation = rbc.T @ state["body"] @ rbc
            changed = state["slam"].process(picture, camera_rotation)
        slam = state["slam"]
        if config.anchor_window > 0.0 and state.setdefault("t0", t) + config.anchor_window < t:
            slam.make_anchors = False
        if changed and state["slam"].commits:
            _update_map(state["map"], slam, picture)
        state["spent"] += time.perf_counter() - began
        state["count"] += 1
        stats = dict(
            tracks=len(slam.track_points), steps=slam.commits, fixes=slam.fixes, restarts=slam.restarts, landmarks=len(slam.landmarks),
            skipped=slam.skipped_no_scale, failed=slam.failed_fits, corrected=slam.rotation_corrections, ms=1000.0 * state["spent"] / state["count"],
        )
        outbox.put(("result", SlamResult(config.name, t, bool(changed), slam.world_rotation.copy(), slam.world_position.copy(), stats), slot))
        now = time.perf_counter()
        if now - state["map_at"] > MAP_EVERY and state["map"].count:
            state["map_at"] = now
            store = state["map"]
            outbox.put(("map", config.name, store.positions[: store.count].astype(np.float32), (store.colors[: store.count] * 255).astype(np.uint8)))
    shm.close()


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

    def submit(self, name: str, t: float, picture: np.ndarray, gyro: np.ndarray, bias: np.ndarray) -> None:
        """One picture (BGR fisheye), the gyro rows [t, gx, gy, gz] since the previous picture, the flight computer's gyro bias."""
        while len(self.pending[name]) >= SLOTS - 1:  # all slots busy: the SLAM is behind, wait for it
            self._collect(block=True)
        slot = self.next_slot[name]
        self.next_slot[name] = (slot + 1) % SLOTS
        self.frames[name][slot] = picture
        self.pending[name].append(t)
        self.started.setdefault(name, t)
        self.inboxes[name].put(("frame", slot, t, gyro, bias))

    def _collect(self, block: bool = False) -> None:
        import queue

        waiting = block
        while True:
            try:
                message = self.outbox.get(timeout=0.5) if waiting else self.outbox.get_nowait()
            except queue.Empty:
                if waiting:
                    if not any(p.is_alive() for p in self.processes.values()):
                        raise RuntimeError("the SLAM worker died")
                    continue
                return
            waiting = False
            if message[0] == "result":
                result: SlamResult = message[1]
                self.pending[result.name].remove(result.t)
                self.results.append(result)
                self.stats[result.name] = result.stats
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
