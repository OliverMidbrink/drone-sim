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
  `Finish sketch` (or `Esc` on an idle tool) returns to your previous view. Finished sketches stay visible in 3D; the eye in the
  design tree hides or shows them.
- **Sketch tools** (Fusion-style toolbar; shortcut in brackets):
  - *Draw:* Line `L` (chained), Rectangle `R` (2-point, center, 3-point), Circle `C` (center, 2-point, 3-point), Arc `A` (3-point, center),
    Slot `U`, Polygon `P` (inscribed or circumscribed, n sides), Point `N`.
  - *Modify:* Offset `O` (whole tangent-continuous chains, outer corners trimmed), Trim `T`, Extend `E`, Fillet `F`, Chamfer `H`,
    Mirror `I` (about a line or an axis), Move/Copy `M`, Rotate `Q`, Scale `K`, Rectangular and Circular pattern (set the numbers, press Apply).
    Mirror, Move, Rotate, Scale and the patterns act on the current selection: select first (click, or drag a box: left-to-right
    encloses, right-to-left crosses), then pick the tool.
  - *Constrain:* Dimension `D` (a line, a circle or arc, two points, a point and a line, or two lines for an angle; type the value and press Enter;
    double-click a dimension to edit it), plus coincident, point-on-curve, horizontal, vertical, parallel, perpendicular, equal, tangent,
    concentric, midpoint, symmetric and fix. The bottom-left list shows every constraint (edit values, delete, hover to highlight).
    Dragging a point or curve with the Select tool moves it while the solver keeps the constraints.
  - *Utility:* Construction `X` (dashed, ignored by Extrude), Delete `Del`, Snap `S`, constraint glyphs on or off.
  - *Snapping:* an adaptive grid (1-2-5 steps picked from the zoom, shown in the status line), then endpoint, origin, center, intersection,
    midpoint, quadrant, perpendicular, tangent and on-curve, plus dashed alignment guides to existing points. `Shift` snaps angles to
    15°, `Alt` bypasses snapping. Snapping to a point adds a coincident constraint, on a curve a point-on-curve constraint, and near-horizontal
    or vertical lines get a constraint.
  - *Typed input (dimension fields):* after the first click a field box follows the cursor. Every field shows the live value from the mouse
    greyed; type to replace it (bold). `Tab` / `Shift+Tab` (or a click) move between fields, `Enter` places the shape.
    Line: Length, Angle. Rectangle: Width, Height. Circle: Diameter (or Radius, toggle in the options bar). Polygon: Outer Ø (corners),
    Inner Ø (across flats), Sides, Rotation; typing one diameter updates the other live. Slot: Length, Angle (then `Enter`), then Width.
    Centre arc: Radius (then `Enter`), then Sweep. Typed lengths, widths, diameters and radii also create the matching driving
    dimensions, so the shape is dimensioned as it is drawn.
  - *Profiles:* shapes that touch or overlap are split along their shared edges, so three rectangles side by side (even with T-junctions)
    are three separate filled regions; curves that cross split the regions they cross.
  - *Status:* the toolbar shows "Fully constrained" or "N degrees of freedom"; fully constrained geometry is green, conflicting
    constraints are red. Open chains (not closed into a profile) are drawn in pink and are not extruded. `Ctrl+Z` works inside the sketcher.
- **3D:** select a sketch, **Extrude** (new / add / cut). **Loft** between two sketches (Ctrl-click). **Combine** two bodies.
  **Revolve** (like Fusion): click Revolve, click one or more shaded sketch profiles (tangent or not), then click the axis (a sketch line,
  construction line, a sketch axis or a world X/Y/Z line). In the Revolve panel (and later in Properties) click the Profile or Axis box
  to pick again. Type: Full, Angle or Symmetric; Operation: New, Add or Cut. A profile that crosses the axis is reported as an error.
- **Modify → Thread:** options open on the right (size, male/female/auto, length). Click a cylindrical face: a boss becomes a male
  thread, a hole wall a female one. The cylinder is resized to the screw size first (male → major Ø, female → tap-drill Ø).
- **Modify → Delete feature:** click any face of a boss, rib, pocket or hole; the whole feature is removed and the surface healed.
- **Import / Export:** DXF import (new sketch; lines, arcs, circles, polyline bulges) and export (arcs stay arcs); STEP and STL export.
- **Navigation:** left-drag orbits (turntable about the Z axis, so the view is always upright and never flips over the poles);
  right-drag, middle-drag or `Shift` + drag pans; the wheel zooms towards the cursor. In a sketch the camera is locked onto the
  plane: left-click draws, middle or right drag pans, the wheel zooms.
- **View cube** (top right): click a face, edge or corner to look from there; drag it to orbit; the arrows step the view 90°
  (the two curved ones roll it); the house resets; `Persp / Ortho` switches projection. Front is -Y, Right is +X, Z is up.
- Undo / redo (`Ctrl+Z`, `Ctrl+Shift+Z`), `F` fits the view, `Esc` cancels a tool.

## Layout

```
apps/web        Vite + SolidJS + three.js; kernel worker; 2D sketcher (src/sketch: session, renderer, toolbar)
apps/desktop    Electron shell: serves apps/web/dist over app://, native open/save dialogs
packages/core       document model, undo store, canonical JSON + migrations (no kernel, no DOM)
packages/kernel     thin OCCT helpers (overload resolver, shapes, tessellation, STEP, helix thread sweep)
packages/features   rebuild engine, sketch→face, extrude/loft/boolean, Thread tool, Feature delete
packages/sketch     pure sketch engine: geometry, adaptive grid, snapping, constraint solver, profiles, modify ops (no DOM, no kernel)
packages/standards  thread tables as data (one JSON per standard) + dimension derivation
packages/io         DXF import/export (arcs, bulges, construction layer), STL export
```

## Adding a thread standard

Drop `packages/standards/data/<name>.json` (family, profile, angle, optional `taper`, sizes with `pitch`, `major`, `tapDrill`)
and add it to the `standards` array in `packages/standards/src/index.ts`. A new profile shape needs an entry in `depthFactor`.
`pnpm --filter @darkcad/standards test` validates the tables.

## Known limits (honest list)

- Sketcher: no splines or ellipses. Open chains are kept and shown but cannot be extruded.
  Old (v1) designs are migrated on open: a rectangle becomes four constrained lines, a polygon becomes lines, a circle stays a circle.

- Thread tables are from memory of ISO 228-1 / 7-1 / 68-1 and flagged `"verified": false`. Check against the standards before machining.
  Tolerance classes and print compensation are not modelled yet; BSPT gauge-plane handling is simplified.
- Thread profile is a sharp trapezoid with a hard end: no root/crest rounding, no run-out chamfer yet.
- The kernel is the prebuilt `opencascade.js` (66 MB wasm, ~14 MB gzipped). A trimmed custom OCCT build is planned.
- Persistent naming is geometric (centroid, area, surface type). Reordering features can lose a face reference, which is reported as an error.
- Sketches have no constraint solver yet (numeric edits only); no arcs/splines/trim/fillet.
- No fillet/chamfer features yet; Revolve cannot use a body edge as its axis or a "to object" extent; no assemblies.
