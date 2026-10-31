# DarkCAD

A dark-themed parametric CAD that runs in the browser and, from the same bundle, as an offline desktop app (Electron).
Geometry comes from OpenCascade (OCCT) compiled to WebAssembly, running in a Web Worker.

First use case: pump pipe adapters, so threads (BSP and metric, male and female) and a feature-delete tool are first-class.

## Run

```sh
nvm use            # Node 22 (see .nvmrc)
pnpm install
pnpm dev           # http://localhost:5173
pnpm test          # kernel, features, standards, core, io, workspace
pnpm e2e           # production build driven in headless Chromium
pnpm desktop       # build + open the Electron app
```

If Electron starts as plain Node, `ELECTRON_RUN_AS_NODE` is set in your environment (VS Code's terminal does this): unset it.

## Using it

- **Tabs:** the top bar is `DarkCAD`, a grid icon, then one tab per open design (`Untitled` by default). `+` opens a new design,
  double-click a tab to rename it, `×` closes it. The grid icon lists every saved design to reopen or delete.
  Designs autosave in the browser; each has its own undo history. The folder/save icons on the right open and save `.darkcad` files.
- **Create Sketch:** the three origin planes light up in yellow, click one. The camera swings to look straight at that plane in
  orthographic projection, with the model (translucent) and your other sketches still visible, in the same global coordinates.
  Draw rectangles, circles, polygons (`V R C P`); a shape inside another becomes a hole. Edit exact numbers in the panel on the right.
  `Finish sketch` (or `Esc`) returns to your previous view.
- **3D:** select a sketch, **Extrude** (new / add / cut). **Loft** between two sketches (Ctrl-click). **Combine** two bodies.
- **Modify → Thread:** options open on the right (size, male/female/auto, length). Click a cylindrical face: a boss becomes a male
  thread, a hole wall a female one. The cylinder is resized to the screw size first (male → major Ø, female → tap-drill Ø).
- **Modify → Delete feature:** click any face of a boss, rib, pocket or hole; the whole feature is removed and the surface healed.
- **Import / Export:** DXF import (new sketch); STEP, STL and DXF export.
- **View cube** (top right): click a face, edge or corner to look from there; drag it to orbit; the arrows step the view 90°
  (the two curved ones roll it); the house resets; `Persp / Ortho` switches projection. Front is -Y, Right is +X, Z is up.
- Undo / redo (`Ctrl+Z`, `Ctrl+Shift+Z`), `F` fits the view, `Esc` cancels a tool.

## Layout

```
apps/web        Vite + SolidJS + three.js; kernel worker; 2D sketcher
apps/desktop    Electron shell: serves apps/web/dist over app://, native open/save dialogs
packages/core       document model, undo store, canonical JSON + migrations (no kernel, no DOM)
packages/kernel     thin OCCT helpers (overload resolver, shapes, tessellation, STEP, helix thread sweep)
packages/features   rebuild engine, sketch→face, extrude/loft/boolean, Thread tool, Feature delete
packages/standards  thread tables as data (one JSON per standard) + dimension derivation
packages/io         DXF import/export, STL export
```

## Adding a thread standard

Drop `packages/standards/data/<name>.json` (family, profile, angle, optional `taper`, sizes with `pitch`, `major`, `tapDrill`)
and add it to the `standards` array in `packages/standards/src/index.ts`. A new profile shape needs an entry in `depthFactor`.
`pnpm --filter @darkcad/standards test` validates the tables.

## Known limits (honest list)

- Thread tables are from memory of ISO 228-1 / 7-1 / 68-1 and flagged `"verified": false`. Check against the standards before machining.
  Tolerance classes and print compensation are not modelled yet; BSPT gauge-plane handling is simplified.
- Thread profile is a sharp trapezoid with a hard end: no root/crest rounding, no run-out chamfer yet.
- The kernel is the prebuilt `opencascade.js` (66 MB wasm, ~14 MB gzipped). A trimmed custom OCCT build is planned.
- Persistent naming is geometric (centroid, area, surface type). Reordering features can lose a face reference, which is reported as an error.
- Sketches have no constraint solver yet (numeric edits only); no arcs/splines/trim/fillet.
- No fillet/chamfer/revolve features yet; no assemblies.
