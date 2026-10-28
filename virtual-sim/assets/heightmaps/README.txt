Drop your own terrain here as  terrain.png  and the simulator uses it instead of generating one.

  - greyscale PNG, 8 or 16 bit (16 bit avoids terracing); black = 0 m, white = --heightmap-max metres
  - stretched over the whole --size x --size metre area (default 2000 m, white = 600 m)
  - trees are scattered on it automatically

Any other path works too:  python virtual-sim/simulator.py --heightmap my_dem.png --heightmap-max 900

To get a starting file, export a generated world:
  python virtual-sim/simulator.py --seed 3 --export-heightmap virtual-sim/assets/heightmaps/terrain.png
