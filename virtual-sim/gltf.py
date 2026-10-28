"""Minimal glTF 2.0 reader (.gltf + .bin files): triangle meshes baked into world space, with base-colour textures.

Handles what the Godot forest demo's Sketchfab exports use: node matrices or TRS, interleaved
buffers, the metallic-roughness and the specular-glossiness extension's base colour, and the alpha
modes (MASK and BLEND are both treated as a cut-out).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np

COMPONENT_TYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
COMPONENT_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


@dataclass
class Primitive:
    positions: np.ndarray  # (N, 3) float32, world space of the glTF (y up)
    normals: np.ndarray  # (N, 3) float32
    uvs: np.ndarray  # (N, 2) float32, glTF convention (v = 0 at the top of the image)
    indices: np.ndarray  # (M,) uint32
    texture: np.ndarray | None  # (H, W, 4) uint8 RGBA, or None
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # base colour factor
    cutout: bool = False  # the texture's alpha cuts the shape out (leaves, needles)
    double_sided: bool = False


@dataclass
class Model:
    primitives: list[Primitive] = field(default_factory=list)


def _read_buffer(base: Path, buffer: dict) -> bytes:
    uri = buffer.get("uri", "")
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    return (base / unquote(uri)).read_bytes()


def _read_accessor(document: dict, buffers: list[bytes], index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    dtype = np.dtype(COMPONENT_TYPES[accessor["componentType"]])
    width = COMPONENT_COUNTS[accessor["type"]]
    count = accessor["count"]
    if "bufferView" not in accessor:
        return np.zeros((count, width), dtype)
    view = document["bufferViews"][accessor["bufferView"]]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride") or dtype.itemsize * width
    return np.array(np.ndarray((count, width), dtype, buffers[view["buffer"]], offset, (stride, dtype.itemsize)))


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # stored column-major
    matrix = np.eye(4)
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix[:3, :3] = rotation * np.array(node.get("scale", (1.0, 1.0, 1.0)))
    matrix[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return matrix


def _mesh_instances(document: dict) -> list[tuple[int, np.ndarray]]:
    """(mesh index, matrix to world space) for every mesh in the default scene."""
    found: list[tuple[int, np.ndarray]] = []

    def walk(index: int, parent: np.ndarray) -> None:
        node = document["nodes"][index]
        matrix = parent @ _node_matrix(node)
        if "mesh" in node:
            found.append((node["mesh"], matrix))
        for child in node.get("children", []):
            walk(child, matrix)

    scene = document["scenes"][document.get("scene", 0)]
    for root in scene["nodes"]:
        walk(root, np.eye(4))
    return found


def _material(document: dict, index: int | None) -> tuple[int | None, tuple, bool, bool]:
    """(texture index, colour factor, cutout, double sided)"""
    material = document["materials"][index] if index is not None else {}
    metallic = material.get("pbrMetallicRoughness", {})
    glossy = material.get("extensions", {}).get("KHR_materials_pbrSpecularGlossiness", {})
    texture = (metallic.get("baseColorTexture") or glossy.get("diffuseTexture") or {}).get("index")
    factor = metallic.get("baseColorFactor") or glossy.get("diffuseFactor") or (1.0, 1.0, 1.0, 1.0)
    return texture, tuple(factor), material.get("alphaMode", "OPAQUE") != "OPAQUE", bool(material.get("doubleSided", False))


def _load_texture(path: Path, max_size: int) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[2] == 4 else cv2.COLOR_BGR2RGBA)
    if max(rgba.shape[:2]) > max_size:
        factor = max_size / max(rgba.shape[:2])
        rgba = cv2.resize(rgba, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgba)


def load_gltf(path: Path, max_texture: int = 1024) -> Model:
    """All meshes of the file's default scene with every node transform applied."""
    document = json.loads(path.read_text())
    buffers = [_read_buffer(path.parent, buffer) for buffer in document["buffers"]]
    textures: dict[int, np.ndarray | None] = {}
    model = Model()
    for mesh_index, matrix in _mesh_instances(document):
        normal_matrix = np.linalg.inv(matrix[:3, :3])
        for primitive in document["meshes"][mesh_index]["primitives"]:
            if primitive.get("mode", 4) != 4:
                continue  # triangles only
            attributes = primitive["attributes"]
            positions = _read_accessor(document, buffers, attributes["POSITION"]).astype(np.float64)
            positions = positions @ matrix[:3, :3].T + matrix[:3, 3]
            if "NORMAL" in attributes:
                normals = _read_accessor(document, buffers, attributes["NORMAL"]).astype(np.float64) @ normal_matrix
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            else:
                normals = np.tile([0.0, 1.0, 0.0], (len(positions), 1))
            uvs = _read_accessor(document, buffers, attributes["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in attributes else np.zeros((len(positions), 2), np.float32)
            if "indices" in primitive:
                indices = _read_accessor(document, buffers, primitive["indices"]).reshape(-1).astype(np.uint32)
            else:
                indices = np.arange(len(positions), dtype=np.uint32)
            texture_index, color, cutout, double_sided = _material(document, primitive.get("material"))
            texture = None
            if texture_index is not None:
                if texture_index not in textures:
                    source = document["images"][document["textures"][texture_index]["source"]]
                    textures[texture_index] = _load_texture(path.parent / unquote(source["uri"]), max_texture) if "uri" in source else None
                texture = textures[texture_index]
            model.primitives.append(
                Primitive(positions.astype(np.float32), normals.astype(np.float32), uvs, indices, texture, color, cutout, double_sided)
            )
    return model
