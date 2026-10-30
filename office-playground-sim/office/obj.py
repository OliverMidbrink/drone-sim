"""Wavefront .obj / .mtl reader: every `o` object of a file as triangle meshes, one per material.

Written for Blender's exports: y up, one `o` block per asset with its pivot at the local origin,
quads and n-gons (fan triangulated), `v/vt/vn` face corners. Texture paths in the .mtl are
usually absolute paths from the author's machine, so only their file names are used and they are
looked up in the folders given to `load_obj`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ObjMesh:
    positions: np.ndarray  # (N, 3) float32
    normals: np.ndarray  # (N, 3) float32
    uvs: np.ndarray  # (N, 2) float32, glTF convention (v = 0 at the top of the image)
    indices: np.ndarray  # (M,) uint32, triangles
    material: str


def read_mtl(path: Path) -> dict[str, dict[str, str]]:
    """material name -> {keyword: rest of the line} for the keywords we care about (map_Kd, Kd)."""
    materials: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for raw in path.read_text().splitlines():
        parts = raw.strip().split(None, 1)
        if len(parts) < 2:
            continue
        if parts[0] == "newmtl":
            current = materials.setdefault(parts[1].strip(), {})
        elif current is not None and parts[0] in ("map_Kd", "Kd"):
            current[parts[0]] = parts[1].strip()
    return materials


def read_obj(path: Path) -> tuple[dict[str, list[ObjMesh]], dict[str, dict[str, str]]]:
    """({object name: meshes (one per material)}, materials of the .mtl the file points to)."""
    positions: list[list[float]] = []
    normals: list[list[float]] = []
    uvs: list[list[float]] = []
    materials: dict[str, dict[str, str]] = {}
    objects: dict[str, dict[str, list[tuple[int, int, int]]]] = {}  # name -> material -> face corners as (v, vt, vn) triples, three per triangle
    current: dict[str, list[tuple[int, int, int]]] | None = None
    material = ""
    for raw in path.read_text().splitlines():
        if not raw or raw[0] == "#":
            continue
        head, _, rest = raw.partition(" ")
        if head == "v":
            positions.append([float(x) for x in rest.split()[:3]])
        elif head == "vn":
            normals.append([float(x) for x in rest.split()[:3]])
        elif head == "vt":
            uvs.append([float(x) for x in rest.split()[:2]])
        elif head == "o":
            current = objects.setdefault(rest.strip(), {})
        elif head == "mtllib":
            mtl = path.parent / rest.strip()
            if mtl.is_file():
                materials = read_mtl(mtl)
        elif head == "usemtl":
            material = rest.strip()
        elif head == "f":
            if current is None:
                current = objects.setdefault(path.stem, {})
            corners = []
            for token in rest.split():
                fields = token.split("/")
                v = int(fields[0]) - 1
                t = int(fields[1]) - 1 if len(fields) > 1 and fields[1] else -1
                n = int(fields[2]) - 1 if len(fields) > 2 and fields[2] else -1
                corners.append((v, t, n))
            triangles = current.setdefault(material, [])
            for i in range(1, len(corners) - 1):
                triangles.extend((corners[0], corners[i], corners[i + 1]))

    all_positions = np.array(positions, np.float32).reshape(-1, 3)
    all_normals = np.array(normals, np.float32).reshape(-1, 3)
    all_uvs = np.array(uvs, np.float32).reshape(-1, 2)
    meshes: dict[str, list[ObjMesh]] = {}
    for name, by_material in objects.items():
        meshes[name] = []
        for material_name, corners in by_material.items():
            triples = np.array(corners, np.int64).reshape(-1, 3)
            unique, inverse = np.unique(triples, axis=0, return_inverse=True)
            vertex = all_positions[unique[:, 0]]
            uv = all_uvs[unique[:, 1]] if len(all_uvs) and (unique[:, 1] >= 0).all() else np.zeros((len(unique), 2), np.float32)
            normal = all_normals[unique[:, 2]] if len(all_normals) and (unique[:, 2] >= 0).all() else _flat_normals(vertex, inverse.reshape(-1))
            uv = uv.copy()
            uv[:, 1] = 1.0 - uv[:, 1]  # OBJ puts v = 0 at the bottom of the image
            meshes[name].append(ObjMesh(vertex, normal, uv, inverse.reshape(-1).astype(np.uint32), material_name))
    return meshes, materials


def _flat_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Smooth vertex normals from the triangles, for files that ship none."""
    triangles = indices.reshape(-1, 3)
    face = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]], vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face)
    return (normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)).astype(np.float32)
