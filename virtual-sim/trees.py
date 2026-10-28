"""The Godot forest demo's trees: loading the models and scattering them over a terrain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gltf import Model, load_gltf
from terrain import Terrain, fbm, smoothstep

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Species:
    folder: str  # under <godot demo>/objects/
    height: tuple[float, float]  # metres, min and max
    weight: float  # relative share among the trees
    draw_distance: float  # metres; the heavy models are only drawn near the camera


SPECIES = [
    Species("pine-tree-4", (9.0, 18.0), 3.0, 1400.0),
    Species("pine-tree-3", (10.0, 20.0), 3.0, 900.0),
    Species("pine-tree-2", (8.0, 16.0), 3.0, 1200.0),
    Species("pine-tree-1", (12.0, 24.0), 2.0, 450.0),
    Species("birch-tree-1", (8.0, 14.0), 1.5, 200.0),
]
TREELINE = (150.0, 250.0)  # trees thin out between these altitudes
SPAWN_CLEARING = 12.0  # metres around the origin kept free of trees
SINK = 0.15  # metres the trunk base is sunk into the ground


@dataclass
class TreeModel:
    species: Species
    model: Model  # normalised: 1 m tall, trunk base at the origin
    radius: float  # bounding sphere around (0, 0.5, 0), for culling


def find_godot_objects(godot_dir: Path | None) -> Path | None:
    if godot_dir is not None:
        return godot_dir / "objects" if (godot_dir / "objects").is_dir() else godot_dir
    found = sorted(ROOT.glob("godot-forest-demo-*/objects"))
    return found[0] if found else None


def _normalise(model: Model) -> float:
    """Scale to 1 m tall, stand on y = 0, trunk on the axis. Returns the bounding radius around (0, 0.5, 0)."""
    everything = np.concatenate([p.positions for p in model.primitives])
    low, high = float(everything[:, 1].min()), float(everything[:, 1].max())
    height = high - low
    base = everything[everything[:, 1] < low + 0.03 * height]
    centre_xz = base[:, [0, 2]].mean(axis=0)
    radius = 0.0
    for primitive in model.primitives:
        p = primitive.positions
        p[:, 0] -= centre_xz[0]
        p[:, 2] -= centre_xz[1]
        p[:, 1] -= low
        p /= height
        radius = max(radius, float(np.linalg.norm(p - np.array([0.0, 0.5, 0.0]), axis=1).max()))
    return radius


def load_models(godot_dir: Path | None = None) -> list[TreeModel]:
    objects = find_godot_objects(godot_dir)
    if objects is None:
        print("No godot-forest-demo folder found: no trees (use --godot-dir)")
        return []
    models = []
    for species in SPECIES:
        path = objects / species.folder / "scene.gltf"
        if not path.is_file():
            print(f"Tree model missing, skipped: {path}")
            continue
        model = load_gltf(path)
        models.append(TreeModel(species, model, _normalise(model)))
    return models


def place_trees(terrain: Terrain, models: list[TreeModel], seed: int, count: int) -> list[np.ndarray]:
    """Per model an (N, 6) float32 array of instances: x, y, z, yaw, height, brightness.

    Trees stand on a jittered grid (so they never overlap), in patches of forest (low-frequency
    noise) and sparsely in between, and thin out on steep slopes and towards the treeline.
    """
    if not models or count <= 0:
        return [np.empty((0, 6), np.float32) for _ in models]
    rng = np.random.default_rng(seed + 7919)
    cell = terrain.size / np.sqrt(count * 4.0)  # about 4 candidates per wanted tree
    steps = int((terrain.size - 2.0 * cell) / cell)
    grid = (np.arange(steps) + 0.5) * cell - terrain.half + cell
    gx, gz = np.meshgrid(grid, grid)
    x = gx.ravel() + rng.uniform(-0.45, 0.45, gx.size) * cell
    z = gz.ravel() + rng.uniform(-0.45, 0.45, gz.size) * cell

    forest = fbm(rng, 256, 5, 4)
    index = np.clip(((np.stack([x, z]) / terrain.size + 0.5) * 255).round().astype(int), 0, 255)
    patch = forest[index[1], index[0]]
    chance = 0.04 + 0.96 * smoothstep(0.4, 0.6, patch)
    ground = terrain.height_at(x, z)
    chance *= 1.0 - smoothstep(TREELINE[0], TREELINE[1], ground)
    chance *= smoothstep(0.78, 0.9, terrain.flatness_at(x, z))  # not on cliffs
    keep = (rng.random(x.size) < chance) & (np.hypot(x, z) > SPAWN_CLEARING)
    chosen = np.flatnonzero(keep)
    if len(chosen) > count:
        chosen = rng.choice(chosen, count, replace=False)
    x, z, ground = x[chosen], z[chosen], ground[chosen]

    weights = np.array([m.species.weight for m in models])
    species = rng.choice(len(models), len(chosen), p=weights / weights.sum())
    low = np.array([m.species.height[0] for m in models])[species]
    high = np.array([m.species.height[1] for m in models])[species]
    instances = np.stack(
        [x, ground - SINK, z, rng.uniform(0.0, 2.0 * np.pi, len(chosen)), rng.uniform(low, high), rng.uniform(0.82, 1.12, len(chosen))],
        axis=1,
    ).astype(np.float32)
    return [instances[species == i] for i in range(len(models))]
