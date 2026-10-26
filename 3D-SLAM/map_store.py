"""Tiered landmark storage: live (in view), RAM archive, optional disk cache."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _empty_points() -> np.ndarray:
    return np.empty((0, 3), dtype=np.float64)


def _empty_colors() -> np.ndarray:
    return np.empty((0, 3), dtype=np.float64)


def visible_from_camera(
    points_world: np.ndarray,
    world_rotation: np.ndarray,
    world_position: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    *,
    margin_ratio: float = 0.2,
) -> np.ndarray:
    if len(points_world) == 0:
        return np.zeros(0, dtype=bool)

    relative = points_world - world_position
    camera = relative @ world_rotation

    depth = camera[:, 2]
    in_front = depth > 0.15

    safe_z = np.maximum(depth, 1e-6)
    u = intrinsic[0, 0] * camera[:, 0] / safe_z + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera[:, 1] / safe_z + intrinsic[1, 2]

    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    in_frame = (
        (u >= -margin_x)
        & (u <= width + margin_x)
        & (v >= -margin_y)
        & (v <= height + margin_y)
    )
    return in_front & in_frame


def visible_from_orbit(
    points_world: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    fov_y_deg: float = 45.0,
    aspect: float = 1.0,
    *,
    margin: float = 0.25,
) -> np.ndarray:
    if len(points_world) == 0:
        return np.zeros(0, dtype=bool)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    norm = np.linalg.norm(right)
    if norm < 1e-8:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
    else:
        right /= norm
    up = np.cross(right, forward)

    vectors = points_world - eye
    depth = vectors @ forward
    in_front = depth > 0.05

    half_h = np.tan(np.radians(fov_y_deg) / 2.0) * (1.0 + margin)
    half_w = half_h * aspect

    lateral = vectors @ right
    vertical = vectors @ up
    safe_depth = np.maximum(depth, 1e-6)
    in_frustum = (np.abs(lateral / safe_depth) <= half_w) & (
        np.abs(vertical / safe_depth) <= half_h
    )
    return in_front & in_frustum


class MapStore:
    # Total observation weight a landmark can accumulate; beyond this, new
    # observations keep a small constant influence so the map can still adapt.
    WEIGHT_CAP = 400.0

    def __init__(
        self,
        cache_dir: Path | None,
        ram_archive_limit: int = 1_000_000,
        max_live_points: int = 50_000,
    ) -> None:
        self.cache_dir = cache_dir
        self.ram_archive_limit = ram_archive_limit
        self.max_live_points = max_live_points

        self.live_positions = _empty_points()
        self.live_colors = _empty_colors()
        self.live_ids = np.empty(0, dtype=np.int64)
        self.live_weights = np.empty(0, dtype=np.float64)
        self.live_steps = np.empty(0, dtype=np.int32)
        self.live_born = np.empty(0, dtype=np.float64)
        self.archive_positions = _empty_points()
        self.archive_colors = _empty_colors()
        self.archive_ids = np.empty(0, dtype=np.int64)
        self.archive_weights = np.empty(0, dtype=np.float64)
        self.archive_steps = np.empty(0, dtype=np.int32)
        self.archive_born = np.empty(0, dtype=np.float64)
        self.rejected_ids = np.empty(0, dtype=np.int64)  # pruned tracks stay out of the map
        self.protected_ids = np.empty(0, dtype=np.int64)  # mesh corners: exempt from thinning/pruning
        self.disk_point_count = 0
        self._chunk_id = 0
        self._disk_positions = _empty_points()
        self._disk_colors = _empty_colors()
        self._disk_chunks_loaded = 0

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return (
            len(self.live_positions)
            + len(self.archive_positions)
            + self.disk_point_count
        )

    @property
    def live_count(self) -> int:
        return len(self.live_positions)

    @property
    def archive_count(self) -> int:
        return len(self.archive_positions) + self.disk_point_count

    def lookup(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Landmark positions for track ids: (found mask, positions of found ids)."""
        found = np.zeros(len(ids), dtype=bool)
        if len(self.live_ids) == 0 or len(ids) == 0:
            return found, _empty_points()
        order = np.argsort(self.live_ids)
        sorted_ids = self.live_ids[order]
        pos = np.minimum(np.searchsorted(sorted_ids, ids), len(sorted_ids) - 1)
        found = sorted_ids[pos] == ids
        return found, self.live_positions[order[pos[found]]]

    def lookup_all(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Live landmarks for track ids: (found mask, positions, weights) of the found ids."""
        found = np.zeros(len(ids), dtype=bool)
        if len(self.live_ids) == 0 or len(ids) == 0:
            return found, _empty_points(), np.empty(0)
        order = np.argsort(self.live_ids)
        sorted_ids = self.live_ids[order]
        pos = np.minimum(np.searchsorted(sorted_ids, ids), len(sorted_ids) - 1)
        found = sorted_ids[pos] == ids
        rows = order[pos[found]]
        return found, self.live_positions[rows], self.live_weights[rows]

    def positions_for(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position and colour of landmarks in RAM (live or archived): (found, positions, colours)."""
        n = len(ids)
        found = np.zeros(n, dtype=bool)
        positions = np.zeros((n, 3), dtype=np.float64)
        colors = np.zeros((n, 3), dtype=np.float64)
        all_ids = np.concatenate([self.live_ids, self.archive_ids])
        if len(all_ids) == 0 or n == 0:
            return found, positions, colors
        order = np.argsort(all_ids)
        sorted_ids = all_ids[order]
        slot = np.minimum(np.searchsorted(sorted_ids, ids), len(sorted_ids) - 1)
        found = sorted_ids[slot] == ids
        rows = order[slot[found]]
        positions[found] = np.vstack([self.live_positions, self.archive_positions])[rows]
        colors[found] = np.vstack([self.live_colors, self.archive_colors])[rows]
        return found, positions, colors

    def lookup_weights(self, ids: np.ndarray) -> np.ndarray:
        """Accumulated observation weight per id (0 where the id is not live)."""
        out = np.zeros(len(ids), dtype=np.float64)
        if len(self.live_ids) == 0 or len(ids) == 0:
            return out
        order = np.argsort(self.live_ids)
        sorted_ids = self.live_ids[order]
        pos = np.minimum(np.searchsorted(sorted_ids, ids), len(sorted_ids) - 1)
        found = sorted_ids[pos] == ids
        out[found] = self.live_weights[order[pos[found]]]
        return out

    def add(
        self,
        world_points: np.ndarray,
        colors: np.ndarray,
        ids: np.ndarray,
        weights: np.ndarray | None = None,
        step: int = 0,
        now: float = 0.0,
    ) -> None:
        """Insert observations; refine landmarks that share a track id.

        Each observation carries a weight (e.g. from its triangulation angle):
        well-conditioned observations pull a landmark harder than noisy ones.
        """
        if len(world_points) == 0:
            return
        if weights is None:
            weights = np.ones(len(ids), dtype=np.float64)
        if len(self.rejected_ids):
            allowed = ~np.isin(ids, self.rejected_ids)
            if not allowed.all():
                world_points, colors, ids, weights = (
                    world_points[allowed], colors[allowed], ids[allowed], weights[allowed],
                )
                if len(ids) == 0:
                    return

        is_new = np.ones(len(ids), dtype=bool)
        if len(self.live_ids) > 0:
            order = np.argsort(self.live_ids)
            sorted_ids = self.live_ids[order]
            pos = np.minimum(np.searchsorted(sorted_ids, ids), len(sorted_ids) - 1)
            matched = sorted_ids[pos] == ids
            if matched.any():
                rows = order[pos[matched]]
                w_new = weights[matched]
                w_old = self.live_weights[rows]
                alpha = (w_new / (w_old + w_new))[:, None]
                self.live_positions[rows] += (world_points[matched] - self.live_positions[rows]) * alpha
                self.live_colors[rows] += (colors[matched] - self.live_colors[rows]) * alpha
                self.live_weights[rows] = np.minimum(w_old + w_new, self.WEIGHT_CAP)
                self.live_steps[rows] = step
                is_new = ~matched

        if is_new.any():
            self.live_positions = np.vstack([self.live_positions, world_points[is_new]])
            self.live_colors = np.vstack([self.live_colors, colors[is_new]])
            self.live_ids = np.concatenate([self.live_ids, ids[is_new].astype(np.int64)])
            self.live_weights = np.concatenate([self.live_weights, weights[is_new]])
            self.live_steps = np.concatenate(
                [self.live_steps, np.full(int(is_new.sum()), step, dtype=np.int32)]
            )
            self.live_born = np.concatenate(
                [self.live_born, np.full(int(is_new.sum()), now, dtype=np.float64)]
            )

        if len(self.live_positions) > self.max_live_points:
            overflow = len(self.live_positions) - self.max_live_points
            self._move_to_archive(
                self.live_positions[:overflow],
                self.live_colors[:overflow],
                self.live_ids[:overflow],
                self.live_weights[:overflow],
                self.live_steps[:overflow],
                self.live_born[:overflow],
            )
            self._keep_live(slice(overflow, None))

    def _keep_live(self, selector) -> None:
        self.live_positions = self.live_positions[selector]
        self.live_colors = self.live_colors[selector]
        self.live_ids = self.live_ids[selector]
        self.live_weights = self.live_weights[selector]
        self.live_steps = self.live_steps[selector]
        self.live_born = self.live_born[selector]

    def archive_out_of_walker_view(
        self,
        world_rotation: np.ndarray,
        world_position: np.ndarray,
        intrinsic: np.ndarray,
        width: int,
        height: int,
    ) -> int:
        if len(self.live_positions) == 0:
            return 0

        visible = visible_from_camera(
            self.live_positions,
            world_rotation,
            world_position,
            intrinsic,
            width,
            height,
        )
        if visible.all():
            return 0

        gone = ~visible
        count = int(gone.sum())
        self._move_to_archive(
            self.live_positions[gone],
            self.live_colors[gone],
            self.live_ids[gone],
            self.live_weights[gone],
            self.live_steps[gone],
            self.live_born[gone],
        )
        self._keep_live(visible)
        return count

    def reactivate_in_view(
        self,
        world_rotation: np.ndarray,
        world_position: np.ndarray,
        intrinsic: np.ndarray,
        width: int,
        height: int,
    ) -> int:
        """Bring archived landmarks that are in view again back into the live set.

        This is what lets a returning camera meet the *same* landmarks (with their
        ids and accumulated weight) instead of piling new points on top of them.
        """
        if len(self.archive_positions) == 0:
            return 0
        visible = visible_from_camera(
            self.archive_positions,
            world_rotation,
            world_position,
            intrinsic,
            width,
            height,
        )
        count = int(visible.sum())
        if count == 0:
            return 0
        self.live_positions = np.vstack([self.live_positions, self.archive_positions[visible]])
        self.live_colors = np.vstack([self.live_colors, self.archive_colors[visible]])
        self.live_ids = np.concatenate([self.live_ids, self.archive_ids[visible]])
        self.live_weights = np.concatenate([self.live_weights, self.archive_weights[visible]])
        self.live_steps = np.concatenate([self.live_steps, self.archive_steps[visible]])
        self.live_born = np.concatenate([self.live_born, self.archive_born[visible]])
        keep = ~visible
        self.archive_positions = self.archive_positions[keep]
        self.archive_colors = self.archive_colors[keep]
        self.archive_ids = self.archive_ids[keep]
        self.archive_weights = self.archive_weights[keep]
        self.archive_steps = self.archive_steps[keep]
        self.archive_born = self.archive_born[keep]
        return count

    def find_existing(
        self,
        world_points: np.ndarray,
        pixels: np.ndarray,
        world_rotation: np.ndarray,
        world_position: np.ndarray,
        intrinsic: np.ndarray,
        width: int,
        height: int,
        exclude_ids: np.ndarray,
        *,
        pixel_radius: int = 2,
        tolerance: float = 0.06,
        min_weight: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match fresh observations to landmarks that already exist in the map.

        Live landmarks are projected into the current frame; an observation is
        the same landmark if one projects within a couple of pixels *and* sits at
        a consistent 3D position. Returns (matched mask, landmark ids of matches).
        """
        n = len(world_points)
        matched = np.zeros(n, dtype=bool)
        out_ids = np.zeros(n, dtype=np.int64)
        if n == 0 or len(self.live_positions) == 0:
            return matched, out_ids

        camera = (self.live_positions - world_position) @ world_rotation
        z = camera[:, 2]
        usable = (z > 0.15) & (self.live_weights >= min_weight)
        if len(exclude_ids):
            usable &= ~np.isin(self.live_ids, exclude_ids)
        safe_z = np.maximum(z, 1e-6)
        u = np.round(intrinsic[0, 0] * camera[:, 0] / safe_z + intrinsic[0, 2]).astype(int)
        v = np.round(intrinsic[1, 1] * camera[:, 1] / safe_z + intrinsic[1, 2]).astype(int)
        usable &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
        rows = np.flatnonzero(usable)
        if len(rows) == 0:
            return matched, out_ids

        rows = rows[np.argsort(-z[rows])]  # nearest written last, so it wins a pixel
        grid = np.full((height, width), -1, dtype=np.int64)
        grid[v[rows], u[rows]] = rows

        qx = np.round(pixels[:, 0]).astype(int)
        qy = np.round(pixels[:, 1]).astype(int)
        best_err = np.full(n, np.inf)
        best_row = np.full(n, -1, dtype=np.int64)
        for dy in range(-pixel_radius, pixel_radius + 1):
            for dx in range(-pixel_radius, pixel_radius + 1):
                cand = grid[np.clip(qy + dy, 0, height - 1), np.clip(qx + dx, 0, width - 1)]
                has = cand >= 0
                err = np.full(n, np.inf)
                err[has] = np.linalg.norm(
                    self.live_positions[cand[has]] - world_points[has], axis=1
                ) / np.maximum(z[cand[has]], 1e-6)
                better = err < best_err
                best_err[better] = err[better]
                best_row[better] = cand[better]

        ok = best_err < tolerance
        if not ok.any():
            return matched, out_ids
        order = np.argsort(best_err)
        order = order[ok[order]]
        _, first = np.unique(best_row[order], return_index=True)  # one query per landmark
        chosen = order[first]
        matched[chosen] = True
        out_ids[chosen] = self.live_ids[best_row[chosen]]
        return matched, out_ids

    def warp(self, correction, start_step: int, end_step: int) -> None:
        """Loop-closure correction: landmarks seen during (start, end] move with it."""
        span = max(end_step - start_step, 1)
        for positions, steps in (
            (self.live_positions, self.live_steps),
            (self.archive_positions, self.archive_steps),
        ):
            if len(positions) == 0:
                continue
            fraction = np.clip((steps - start_step) / span, 0.0, 1.0)
            moving = fraction > 0.0
            if moving.any():
                positions[moving] = correction.apply(positions[moving], fraction[moving])

    def prune_isolated(
        self,
        now: float,
        trajectory: np.ndarray,
        min_age: float,
        isolation_factor: float,
        check_window: float = 8.0,
        exempt_ids: np.ndarray | None = None,
    ) -> int:
        """Delete landmarks that never gained company.

        A landmark older than `min_age` (seconds) is removed when its nearest
        neighbour in the whole map is farther than `isolation_factor` times its
        own distance from the camera path (depth-relative, since far points are
        naturally sparser). Removed tracks are remembered so they are not re-added.
        Each landmark is only judged while its age is within `check_window`
        seconds past `min_age`, so a call only queries the points that just came
        of age (neighbours are still searched in the whole map).
        """
        if min_age <= 0 or isolation_factor <= 0 or len(trajectory) == 0:
            return 0
        self._sync_disk_cache_for_render()
        blocks = [self.live_positions, self.archive_positions]
        if len(self._disk_positions):
            blocks.append(self._disk_positions)
        points = np.vstack(blocks)
        n_live, n_arch = len(self.live_positions), len(self.archive_positions)
        if n_live + n_arch < 50:
            return 0

        cloud = points.astype(np.float32)

        born = np.concatenate([self.live_born, self.archive_born])
        age = now - born
        due = np.flatnonzero((age >= min_age) & (age < min_age + check_window))
        if len(due) == 0:
            return 0
        queries = cloud[due]

        index = cv2.flann_Index(cloud, dict(algorithm=1, trees=4))
        _, d2 = index.knnSearch(queries, 2, params={})
        nearest = np.sqrt(d2[:, 1])  # column 0 is the point itself

        poses = np.ascontiguousarray(trajectory[:: max(1, len(trajectory) // 400)], dtype=np.float32)
        if len(poses) >= 2:
            path_index = cv2.flann_Index(poses, dict(algorithm=1, trees=1))
            _, pd2 = path_index.knnSearch(queries, 1, params={})
            path_dist = np.sqrt(pd2[:, 0])
        else:
            path_dist = np.linalg.norm(queries - poses[0], axis=1)

        doomed = np.zeros(n_live + n_arch, dtype=bool)
        doomed[due] = nearest > isolation_factor * np.maximum(path_dist, 1e-6)
        if exempt_ids is not None and len(exempt_ids) and doomed.any():
            ids = np.concatenate([self.live_ids, self.archive_ids])
            doomed &= ~np.isin(ids, exempt_ids)
        return self.remove_rows(doomed)

    def remove_rows(self, doomed: np.ndarray) -> int:
        """Delete landmarks by row (live rows first, then RAM-archive rows).

        Their track ids are remembered so those tracks are never re-added.
        """
        n_live = len(self.live_positions)
        ids = np.concatenate([self.live_ids, self.archive_ids])
        if len(self.protected_ids):
            # Corners of mesh faces are the surface's resolution: never thin them away.
            doomed = doomed & ~np.isin(ids, self.protected_ids)
        count = int(doomed.sum())
        if count == 0:
            return 0
        self.rejected_ids = np.union1d(self.rejected_ids, ids[doomed])
        self._keep_live(~doomed[:n_live])
        keep_a = ~doomed[n_live:]
        self.archive_positions = self.archive_positions[keep_a]
        self.archive_colors = self.archive_colors[keep_a]
        self.archive_ids = self.archive_ids[keep_a]
        self.archive_weights = self.archive_weights[keep_a]
        self.archive_steps = self.archive_steps[keep_a]
        self.archive_born = self.archive_born[keep_a]
        return count

    def _move_to_archive(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        ids: np.ndarray,
        weights: np.ndarray,
        steps: np.ndarray,
        born: np.ndarray,
    ) -> None:
        if len(positions) == 0:
            return
        self.archive_positions = np.vstack([self.archive_positions, positions])
        self.archive_colors = np.vstack([self.archive_colors, colors])
        self.archive_ids = np.concatenate([self.archive_ids, ids])
        self.archive_weights = np.concatenate([self.archive_weights, weights])
        self.archive_steps = np.concatenate([self.archive_steps, steps])
        self.archive_born = np.concatenate([self.archive_born, born])

        if len(self.archive_positions) > self.ram_archive_limit:
            self.flush_archive_to_disk()

    def flush_archive_to_disk(self) -> None:
        if len(self.archive_positions) == 0:
            return
        if self.cache_dir is None:
            # No disk cache: drop oldest half of RAM archive to bound memory.
            keep = len(self.archive_positions) // 2
            self.archive_positions = self.archive_positions[keep:]
            self.archive_colors = self.archive_colors[keep:]
            self.archive_ids = self.archive_ids[keep:]
            self.archive_weights = self.archive_weights[keep:]
            self.archive_steps = self.archive_steps[keep:]
            self.archive_born = self.archive_born[keep:]
            return

        path = self.cache_dir / f"chunk_{self._chunk_id:05d}.npz"
        np.savez_compressed(
            path,
            positions=self.archive_positions.astype(np.float32),
            colors=self.archive_colors.astype(np.float32),
        )
        self.disk_point_count += len(self.archive_positions)
        self._chunk_id += 1
        self.archive_positions = _empty_points()
        self.archive_colors = _empty_colors()
        self.archive_ids = np.empty(0, dtype=np.int64)
        self.archive_weights = np.empty(0, dtype=np.float64)
        self.archive_steps = np.empty(0, dtype=np.int32)
        self.archive_born = np.empty(0, dtype=np.float64)
        self._disk_chunks_loaded = -1  # invalidate GPU-side disk cache

    def _sync_disk_cache_for_render(self) -> None:
        if self.cache_dir is None or self._chunk_id == 0:
            return
        if self._disk_chunks_loaded == self._chunk_id:
            return

        positions_list: list[np.ndarray] = []
        colors_list: list[np.ndarray] = []
        for chunk_index in range(self._chunk_id):
            path = self.cache_dir / f"chunk_{chunk_index:05d}.npz"
            if not path.is_file():
                continue
            data = np.load(path)
            positions_list.append(data["positions"].astype(np.float64))
            colors_list.append(data["colors"].astype(np.float64))

        if positions_list:
            self._disk_positions = np.vstack(positions_list)
            self._disk_colors = np.vstack(colors_list)
        else:
            self._disk_positions = _empty_points()
            self._disk_colors = _empty_colors()
        self._disk_chunks_loaded = self._chunk_id

    def render_subset(
        self,
        eye: np.ndarray,
        target: np.ndarray,
        aspect: float,
        *,
        max_points: int = 150_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Points to upload to the GPU for the current orbit view."""
        blocks: list[np.ndarray] = []
        color_blocks: list[np.ndarray] = []

        if len(self.live_positions) > 0:
            live_mask = visible_from_orbit(
                self.live_positions, eye, target, aspect=aspect
            )
            if live_mask.any():
                blocks.append(self.live_positions[live_mask])
                color_blocks.append(self.live_colors[live_mask])

        if len(self.archive_positions) > 0:
            archive_mask = visible_from_orbit(
                self.archive_positions, eye, target, aspect=aspect
            )
            if archive_mask.any():
                blocks.append(self.archive_positions[archive_mask])
                color_blocks.append(self.archive_colors[archive_mask])

        self._sync_disk_cache_for_render()
        if len(self._disk_positions) > 0:
            disk_mask = visible_from_orbit(
                self._disk_positions, eye, target, aspect=aspect
            )
            if disk_mask.any():
                blocks.append(self._disk_positions[disk_mask])
                color_blocks.append(self._disk_colors[disk_mask])

        if not blocks:
            return _empty_points(), _empty_colors()

        positions = np.vstack(blocks)
        colors = np.vstack(color_blocks)

        if len(positions) > max_points:
            step = max(1, len(positions) // max_points)
            positions = positions[::step]
            colors = colors[::step]

        return positions, colors

    def finalize(self) -> None:
        self.flush_archive_to_disk()
