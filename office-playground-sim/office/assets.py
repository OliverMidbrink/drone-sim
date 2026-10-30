"""The two office asset packs: unzip them on first use and load their models.

    assets/Office_Asset_Pack.zip             131 textured, modular models (walls, doors, windows, floor and
                                             ceiling tiles, furniture, IT gear, stationery): the main pack
    assets/LowPolyOfficeTechPack_v1.0.zip    28 flat-coloured low poly props: used for what the main pack lacks
                                             (bookshelves, plants, laptops, bins, a photocopier ...)

The zips stay where they are. Their contents are unpacked into office-playground-sim/office/assets/ (which is
git-ignored: the low poly pack's licence forbids redistributing its files). `AssetLibrary.get`
returns a model by name: the main pack's by its object name without the `SM_` prefix and `_001`
suffix ("Table_Type_1"), the low poly pack's as "lp/<file name>" ("lp/bookshelf").

Both packs are authored y up. The main pack's models face +x (a wall's visible face, a monitor's
screen), the low poly pack's face +z. Both are turned to this sim's convention in `layout.Placer.put`.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gltf import Primitive, _load_texture, load_gltf

from .obj import read_obj

OFFICE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OFFICE_DIR.parent.parent
ZIP_DIR = PROJECT_ROOT / "assets"
UNPACK_DIR = OFFICE_DIR / "assets"

# pack folder -> (zip name, folder inside the zip, members worth unpacking (prefix match))
PACKS = {
    "Office_Asset_Pack": ("Office_Asset_Pack.zip", "Office_Asset_Pack", ("OBJ/", "Textures_2k/", "ReadME")),
    "LowPolyOfficeTechPack": ("LowPolyOfficeTechPack_v1.0.zip", "LowPolyOfficeTechPack_v1.0", ("GLB/", "LICENSE", "README")),
}


def ensure_unpacked(zip_dir: Path = ZIP_DIR) -> Path:
    """Unzip any pack that is not unpacked yet and return the folder they live in. Fails with a clear message if a zip is missing."""
    for folder, (zip_name, inner, wanted) in PACKS.items():
        target = UNPACK_DIR / folder
        if (target / ".complete").is_file():
            continue
        archive = zip_dir / zip_name
        if not archive.is_file():
            raise FileNotFoundError(f"office asset pack not found: {archive} (put {zip_name} into {zip_dir})")
        print(f"Unpacking {zip_name} ...")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                relative = member[len(inner) + 1 :] if member.startswith(inner + "/") else member
                if not relative or member.endswith("/") or not relative.startswith(wanted):
                    continue
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(zf.read(member))
        (target / ".complete").write_text("ok\n")
    return UNPACK_DIR


@dataclass
class Asset:
    name: str
    primitives: list[Primitive]  # in the model's own frame, as authored
    lo: np.ndarray  # bounding box corners of that frame
    hi: np.ndarray


def _asset(name: str, primitives: list[Primitive]) -> Asset:
    points = np.concatenate([p.positions for p in primitives])
    return Asset(name, primitives, points.min(axis=0), points.max(axis=0))


class AssetLibrary:
    """Loads the packs once and hands out models by name. Textures are shared between all models of a pack."""

    def __init__(self, texture_size: int = 2048, zip_dir: Path = ZIP_DIR) -> None:
        self.root = ensure_unpacked(zip_dir)
        self.texture_size = texture_size
        self._assets: dict[str, Asset] = {}
        self._textures: dict[Path, np.ndarray | None] = {}
        self._load_office_pack()

    def _texture(self, path: Path) -> np.ndarray | None:
        if path not in self._textures:
            self._textures[path] = _load_texture(path, self.texture_size)
        return self._textures[path]

    def _load_office_pack(self) -> None:
        folder = self.root / "Office_Asset_Pack"
        meshes, materials = read_obj(folder / "OBJ" / "Office_Asset_Pack.obj")
        for object_name, parts in meshes.items():
            key = re.sub(r"_001$", "", re.sub(r"^SM_", "", object_name))
            primitives = []
            for mesh in parts:
                texture_ref = materials.get(mesh.material, {}).get("map_Kd")
                texture = self._texture(folder / "Textures_2k" / re.split(r"[\\/]", texture_ref)[-1]) if texture_ref else None
                primitives.append(Primitive(mesh.positions, mesh.normals, mesh.uvs, mesh.indices, texture, (1.0, 1.0, 1.0, 1.0), False, True))
            if primitives:
                self._assets[key] = _asset(key, primitives)

    def names(self) -> list[str]:
        folder = self.root / "LowPolyOfficeTechPack" / "GLB"
        return sorted(self._assets) + sorted(f"lp/{p.stem}" for p in folder.glob("*.glb"))

    def get(self, name: str) -> Asset:
        if name not in self._assets:
            if not name.startswith("lp/"):
                raise KeyError(f"no office asset called {name!r}")
            path = self.root / "LowPolyOfficeTechPack" / "GLB" / f"{name[3:]}.glb"
            if not path.is_file():
                raise KeyError(f"no office asset called {name!r}")
            model = load_gltf(path)
            for primitive in model.primitives:
                primitive.double_sided = True
            self._assets[name] = _asset(name, model.primitives)
        return self._assets[name]
