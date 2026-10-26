"""Coloured surface mesh grown from the tracked landmarks.

Every pose update the tracked points are triangulated in the image (Delaunay) and
each triangle is lifted to 3D using its three landmarks as corners. A triangle is
only believed if it passes geometric checks (no depth jump across it, not a long
sliver, not edge-on) *and* a photo-consistency check: the pixels inside it must
look the same in the previous keyframe as in the current frame when warped by the
motion of its three corners. A triangle spanning an occlusion edge (a trunk and
the background behind it) fails that test because its interior does not move like
its corners. Triangles that keep passing from different views earn votes and are
shown; those that keep failing are discarded.

Faces store landmark ids, not positions, so map corrections (loop closure,
landmark refinement) move the mesh with the map.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MIN_VOTES = 3  # distinct views a triangle must pass in before it is shown
MIN_VERTEX_WEIGHT = 4.0  # landmark triangulation quality needed to be a corner
MAX_EDGE_PX_REL = 0.05  # longest image edge, as a fraction of the image width
MIN_ANGLE_DEG = 10.0  # thinnest allowed corner in the image
MAX_DEPTH_JUMP = 0.30  # (far - near) / near across a triangle
MAX_DEPTH_JUMP_FLAT = 0.15  # same, for texture-poor triangles the photo test cannot judge
MAX_EDGE_3D_REL = 0.12  # longest 3D edge, as a fraction of the triangle's depth
MIN_QUALITY_3D = 0.30  # 4*sqrt(3)*area / sum(edge^2): 1 = equilateral, ~0 = needle/spike
MIN_FACING = 0.12  # |normal . view ray|: reject triangles seen edge-on
MIN_NCC = 0.5  # photo-consistency (zero-mean normalised cross-correlation)
MIN_TEXTURE_STD = 3.0  # grey levels; below this a triangle is "flat"
PURGE_SECONDS = 5.0

_HASH = (
    np.uint64(0x9E3779B97F4A7C15),
    np.uint64(0xC2B2AE3D27D4EB4F),
    np.uint64(0x165667B19E3779F9),
)

# Barycentric sample points inside a triangle for the photo test.
_BARY = np.array(
    [
        [1 / 3, 1 / 3, 1 / 3],
        [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6],
        [0.1, 0.45, 0.45], [0.45, 0.1, 0.45], [0.45, 0.45, 0.1],
        [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
    ],
    dtype=np.float32,
)


def _face_keys(sorted_ids: np.ndarray) -> np.ndarray:
    ids = sorted_ids.astype(np.uint64)
    return (ids[:, 0] * _HASH[0]) ^ (ids[:, 1] * _HASH[1]) ^ (ids[:, 2] * _HASH[2])


def _pixel_key(xy: np.ndarray) -> np.ndarray:
    q = np.round(xy * 8.0).astype(np.int64)
    return q[..., 0] * (1 << 20) + q[..., 1]


def _sample(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear grey values at float pixel positions of shape (..., 2)."""
    flat = np.ascontiguousarray(xy.reshape(-1, 2), dtype=np.float32)
    out = cv2.remap(
        image,
        flat[:, 0].reshape(-1, 1),
        flat[:, 1].reshape(-1, 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return out.reshape(xy.shape[:-1]).astype(np.float32)


class MeshBuilder:
    def __init__(self, store, min_votes: int = MIN_VOTES) -> None:
        self.store = store  # landmark owner; accepted faces' corners are protected in it
        self.min_votes = min_votes
        self.keys = np.empty(0, dtype=np.uint64)  # sorted face hashes
        self.tris = np.empty((0, 3), dtype=np.int64)  # sorted landmark ids per face
        self.votes = np.empty(0, dtype=np.int16)
        self.fails = np.empty(0, dtype=np.int16)
        self.version = 0
        self._last_purge = 0.0

    def __len__(self) -> int:
        return int(self.accepted().sum())

    def accepted(self) -> np.ndarray:
        return (self.votes >= self.min_votes) & (self.fails <= self.votes)

    def vertex_ids(self) -> np.ndarray:
        return np.unique(self.tris[self.accepted()])

    # ------------------------------------------------------------------ update
    def update(
        self,
        *,
        rotation: np.ndarray,
        position: np.ndarray,
        ids: np.ndarray,
        pixels: np.ndarray,
        points: np.ndarray,
        weights: np.ndarray,
        gray: np.ndarray,
        key_gray: np.ndarray | None,
        key_ids: np.ndarray,
        key_pts: np.ndarray,
    ) -> int:
        """Triangulate this frame's tracked landmarks; vote on the triangles. Returns #passing."""
        height, width = gray.shape[:2]
        good = weights >= MIN_VERTEX_WEIGHT
        if good.sum() < 12:
            return 0
        ids, pixels, points = ids[good], pixels[good].astype(np.float32), points[good]

        subdiv = cv2.Subdiv2D((0, 0, width, height))
        subdiv.insert(np.ascontiguousarray(pixels))
        raw = subdiv.getTriangleList()
        if len(raw) == 0:
            return 0
        corners = raw.reshape(-1, 3, 2)

        # Delaunay vertex coordinates -> point indices.
        point_keys = _pixel_key(pixels)
        order = np.argsort(point_keys)
        sorted_keys = point_keys[order]
        slot = np.minimum(np.searchsorted(sorted_keys, _pixel_key(corners)), len(sorted_keys) - 1)
        real = (sorted_keys[slot] == _pixel_key(corners)).all(axis=1)
        tri = order[slot[real]]
        if len(tri) == 0:
            return 0
        corner_ids = ids[tri]  # two tracks can share a landmark after association
        distinct = (
            (corner_ids[:, 0] != corner_ids[:, 1])
            & (corner_ids[:, 1] != corner_ids[:, 2])
            & (corner_ids[:, 0] != corner_ids[:, 2])
        )
        tri = tri[distinct]
        if len(tri) == 0:
            return 0
        px = pixels[tri]  # (T, 3, 2)

        ok = self._geometry_ok(px, points[tri], rotation, position, width)
        photo, flat = self._photo_ok(
            tri, px, ids, gray, key_gray, key_ids, key_pts, ok
        )
        ok &= photo
        # Texture-poor triangles cannot be judged by the photo test: demand a smaller depth jump.
        if flat.any():
            z = ((points[tri] - position) @ rotation)[:, :, 2]
            jump = (z.max(axis=1) - z.min(axis=1)) / np.maximum(z.min(axis=1), 1e-6)
            ok &= ~(flat & (jump > MAX_DEPTH_JUMP_FLAT))

        self._vote(np.sort(ids[tri], axis=1), ok)
        return int(ok.sum())

    def _geometry_ok(
        self,
        px: np.ndarray,
        pts: np.ndarray,
        rotation: np.ndarray,
        position: np.ndarray,
        width: int,
    ) -> np.ndarray:
        edges_px = np.stack([px[:, 1] - px[:, 0], px[:, 2] - px[:, 1], px[:, 0] - px[:, 2]], axis=1)
        length_px = np.linalg.norm(edges_px, axis=2)
        ok = length_px.max(axis=1) <= MAX_EDGE_PX_REL * width

        # Smallest corner angle in the image.
        a, b, c = length_px[:, 1], length_px[:, 2], length_px[:, 0]
        cos_a = (b**2 + c**2 - a**2) / np.maximum(2 * b * c, 1e-9)
        cos_b = (a**2 + c**2 - b**2) / np.maximum(2 * a * c, 1e-9)
        cos_c = (a**2 + b**2 - c**2) / np.maximum(2 * a * b, 1e-9)
        ok &= np.maximum.reduce([cos_a, cos_b, cos_c]) <= np.cos(np.radians(MIN_ANGLE_DEG))

        cam = (pts - position) @ rotation  # (T, 3, 3) in camera axes
        z = cam[:, :, 2]
        ok &= z.min(axis=1) > 0.1
        ok &= (z.max(axis=1) - z.min(axis=1)) / np.maximum(z.min(axis=1), 1e-6) <= MAX_DEPTH_JUMP

        e3 = np.stack([pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 1], pts[:, 0] - pts[:, 2]], axis=1)
        ok &= np.linalg.norm(e3, axis=2).max(axis=1) <= MAX_EDGE_3D_REL * z.mean(axis=1)

        normal = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
        area2 = np.linalg.norm(normal, axis=1)  # twice the area
        quality = 2.0 * np.sqrt(3.0) * area2 / np.maximum((np.linalg.norm(e3, axis=2) ** 2).sum(axis=1), 1e-12)
        ok &= quality >= MIN_QUALITY_3D
        normal /= np.maximum(area2[:, None], 1e-12)
        ray = pts.mean(axis=1) - position
        ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12)
        ok &= np.abs(np.einsum("ij,ij->i", normal, ray)) >= MIN_FACING
        return ok

    def _photo_ok(
        self,
        tri: np.ndarray,
        px: np.ndarray,
        ids: np.ndarray,
        gray: np.ndarray,
        key_gray: np.ndarray | None,
        key_ids: np.ndarray,
        key_pts: np.ndarray,
        candidates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interior pixels must match between the keyframe and now. Returns (pass, flat)."""
        passed = np.zeros(len(tri), dtype=bool)
        flat = np.zeros(len(tri), dtype=bool)
        if key_gray is None or len(key_ids) == 0:
            return passed, flat

        tri_ids = ids[tri]
        slot = np.minimum(np.searchsorted(key_ids, tri_ids), len(key_ids) - 1)
        in_key = (key_ids[slot] == tri_ids).all(axis=1) & candidates
        rows = np.flatnonzero(in_key)
        if len(rows) == 0:
            return passed, flat

        now_xy = px[rows]  # (R, 3, 2)
        then_xy = key_pts[slot[rows]].astype(np.float32)
        now_samples = np.einsum("sv,rvd->rsd", _BARY, now_xy)
        then_samples = np.einsum("sv,rvd->rsd", _BARY, then_xy)
        a = _sample(gray, now_samples)
        b = _sample(key_gray, then_samples)

        a0 = a - a.mean(axis=1, keepdims=True)
        b0 = b - b.mean(axis=1, keepdims=True)
        std = np.minimum(a.std(axis=1), b.std(axis=1))
        ncc = (a0 * b0).sum(axis=1) / np.maximum(
            np.linalg.norm(a0, axis=1) * np.linalg.norm(b0, axis=1), 1e-6
        )
        textured = std >= MIN_TEXTURE_STD
        passed[rows] = np.where(textured, ncc >= MIN_NCC, True)
        flat[rows] = ~textured
        return passed, flat

    def _vote(self, tri_ids: np.ndarray, ok: np.ndarray) -> None:
        keys = _face_keys(tri_ids)
        # A triangle can only vote once per update.
        keys, first = np.unique(keys, return_index=True)
        tri_ids, ok = tri_ids[first], ok[first]

        if len(self.keys):
            slot = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
            known = self.keys[slot] == keys
        else:
            slot = np.zeros(len(keys), dtype=np.int64)
            known = np.zeros(len(keys), dtype=bool)
        if known.any():
            rows = slot[known]
            passed = ok[known]
            np.add.at(self.votes, rows[passed], 1)
            np.add.at(self.fails, rows[~passed], 1)
            # Faces that just earned their last vote: shield their corners right away.
            earned = rows[passed][self.votes[rows[passed]] == self.min_votes]
            if len(earned):
                self.store.protected_ids = np.union1d(self.store.protected_ids, self.tris[earned].ravel())
        new = ~known & ok
        if new.any():
            new_keys, new_tris = keys[new], tri_ids[new]
            at = np.searchsorted(self.keys, new_keys)
            self.keys = np.insert(self.keys, at, new_keys)
            self.tris = np.insert(self.tris, at, new_tris, axis=0)
            self.votes = np.insert(self.votes, at, 1)
            self.fails = np.insert(self.fails, at, 0)
        self.version += 1

    # ------------------------------------------------------------------ upkeep
    def purge(self, now: float, force: bool = False) -> int:
        """Drop faces whose corners left the map; refresh which landmarks are protected."""
        if (not force and now - self._last_purge < PURGE_SECONDS) or len(self.keys) == 0:
            return 0
        self._last_purge = now
        store = self.store
        found, _, _ = store.positions_for(self.tris.ravel())
        alive = found.reshape(-1, 3).all(axis=1)
        # Long-failing faces are not worth keeping either.
        alive &= self.fails <= np.maximum(self.votes, 2) + 4
        removed = int((~alive).sum())
        if removed:
            self.keys, self.tris = self.keys[alive], self.tris[alive]
            self.votes, self.fails = self.votes[alive], self.fails[alive]
            self.version += 1
        store.protected_ids = self.vertex_ids()
        return removed

    # ------------------------------------------------------------------ output
    def mesh_arrays(
        self, max_faces: int = 400_000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flat triangle soup: positions (F,3,3), colours (F,3,3), face normals (F,3)."""
        idx = np.flatnonzero(self.accepted())
        if len(idx) > max_faces:
            idx = idx[np.argsort(-self.votes[idx])[:max_faces]]
        if len(idx) == 0:
            empty = np.empty((0, 3, 3))
            return empty, empty, np.empty((0, 3))
        found, pos, col = self.store.positions_for(self.tris[idx].ravel())
        alive = found.reshape(-1, 3).all(axis=1)
        pos = pos.reshape(-1, 3, 3)[alive]
        col = col.reshape(-1, 3, 3)[alive]
        normal = np.cross(pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        return pos, col, normal

    def indexed_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shared-vertex mesh for export: vertices (V,3), colours (V,3) in 0..1, faces (F,3)."""
        idx = np.flatnonzero(self.accepted())
        if len(idx) == 0:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
        tris = self.tris[idx]
        vertex_ids = np.unique(tris)
        found, pos, col = self.store.positions_for(vertex_ids)
        keep_vertex = found
        remap = np.cumsum(keep_vertex) - 1
        slot = np.searchsorted(vertex_ids, tris)
        face_ok = keep_vertex[slot].all(axis=1)
        faces = remap[slot[face_ok]]
        return pos[keep_vertex], np.clip(col[keep_vertex], 0.0, 1.0), faces


def write_ply(path: Path, vertices: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> None:
    vertex_dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    face_dtype = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    v = np.empty(len(vertices), dtype=vertex_dtype)
    v["x"], v["y"], v["z"] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    rgb = np.round(colors * 255.0).astype(np.uint8)
    v["r"], v["g"], v["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    f = np.empty(len(faces), dtype=face_dtype)
    f["n"] = 3
    f["a"], f["b"], f["c"] = faces[:, 0], faces[:, 1], faces[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(v.tobytes())
        handle.write(f.tobytes())


def write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Binary STL (geometry only; STL cannot carry colour)."""
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    dtype = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    data = np.zeros(len(faces), dtype=dtype)
    data["normal"], data["v"] = normal, tri
    with open(path, "wb") as handle:
        handle.write(b"drone-sim SLAM mesh".ljust(80, b"\0"))
        handle.write(np.uint32(len(faces)).tobytes())
        handle.write(data.tobytes())
