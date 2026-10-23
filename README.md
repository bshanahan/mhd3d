# o-t-3d

Three-dimensional magnetohydrodynamics experiments using [BOUT++](https://bout-dev.readthedocs.io/). The `mhd3d` model evolves density, pressure, covariant velocity, and contravariant magnetic field. The repository includes inputs for a periodic 3D Orszag–Tang vortex and a stationary, expanding flux-tube experiment.

## Contents

- `mhd3d.cxx` — MHD model.
- `data/BOUT.inp` — periodic Orszag–Tang setup.
- `build/flare/BOUT.inp` — expanding flux-tube setup.
- `data/otdiag.py` and `build/boutmhd.py` — simulation diagnostics.
- `build/flare/plot_flared_by.py` and `build/flare/switchback_analysis.py` — flux-tube visualization and analysis.

The analysis scripts are tracked at their existing paths under `build/`; generated files in that directory are ignored.

## Build

Install BOUT++ with its CMake package configuration, then run:

```sh
cmake -S . -B build -DBOUT_BUILD_DIR=/path/to/BOUT-build
cmake --build build -j
```

`BOUT_BUILD_DIR` defaults to `../BOUT-dev/build0` relative to this repository. The model uses the BOUT++ API variant with metric accessor functions; see `COORD_METRIC_IS_FUNCTION` near the top of `mhd3d.cxx` if your BOUT++ release uses metric data members.

## Run

The supplied Orszag–Tang input is in `data/BOUT.inp`:

```sh
mpirun -np 8 ./build/mhd3d -d data
```

For the expanding flux tube, copy its input into a separate output directory:

```sh
mkdir -p runs/flare
cp build/flare/BOUT.inp runs/flare/BOUT.inp
mpirun -np 8 ./build/mhd3d -d runs/flare
```

Choose the MPI process count for your local BOUT++ setup. Simulation outputs can be large and are excluded from Git.

## Analyze

The Python scripts use NumPy, Matplotlib, and `boutdata`. The interactive flux-tube plot also uses Plotly and scikit-image.

```sh
python data/otdiag.py -p data -o data/plots
python build/boutmhd.py -p runs/flare -o runs/flare/plots --y-bndry
python build/flare/plot_flared_by.py runs/flare --out runs/flare/flared_By.html
```

`build/flare/switchback_analysis.py` provides both importable diagnostics and a command-line interface for flux-tube dumps. See the script docstrings for options and assumptions.
