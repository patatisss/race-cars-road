# race-cars-road

Python experiments for generating closed polar roads with clustered angles, correlated radii, and different interpolators.

Complete the setup below first, then start with **`experiment.ipynb`**. Double-click **`open-notebook.cmd`** to open it
in JupyterLab using the project environment. Edit the parameter cell and run the
cells again. Restart the kernel if you change imported functions substantially;
ordinary edits are reloaded automatically.

- `road.py`: sampling, optional smoothing, interpolation, and geometry.
- `experiment.py`: editable defaults, comparison plots, and saving runs.
- `experiment.ipynb`: a small workspace for trying ideas without rewriting the runner.
- `test_road.py`: checks for closure, interpolation, reproducibility, and local variation.

## What to change

| Setting | Meaning |
| --- | --- |
| `seed` | Reproduce a particular draw with the same parameters and code. |
| `count` | Number of angular control points, including the point at zero. |
| `centers_deg` | Angular mode locations; edit the list to move, add, or remove clusters. |
| `width_deg` | Approximate width of each circular mode in degrees. |
| `uniform` | Fraction of the angular distribution assigned to a uniform background. |
| `radius_range` | Interval from which the starting radius is sampled, in metres. |
| `variation` | Strength of the random walk in log radius; 0 gives a circle. This is not a maximum percentage deviation. |
| `quiet` | Minimum local noise standard deviation relative to the modes. |
| `smooth_samples` | Gaussian smoothing width in control-point indices; 0 disables smoothing. |

The angular mixture uses von Mises distributions, the circular counterpart of
normal distributions. Radii follow a correlated walk with more local variance
near the same modes. Subtracting a variance-weighted endpoint trend produces a
closed bridge. Working in log radius keeps the sampled radii positive.

Smoothing is a separate experiment. Its width is measured in **sample indices**,
so it does not have a constant physical or angular width on a clustered grid.
It changes the samples supplied to *all* interpolators. Both raw and used
samples are plotted and saved. Set it to zero to inspect the raw generator.

Interpolation acts on **radius versus angle**, not on x and y separately.
Linear radius interpolation has corners in the centerline; its between-join
curvature need not be zero. PCHIP has continuous first derivatives and may have
curvature jumps. The cubic uses periodic boundary conditions. PCHIP uses cyclic
neighbors to compute matching seam slopes. Join values are omitted from the
linear/PCHIP curvature plots rather than treating them as smooth.

These are candidate centerlines for experiments. A cubic can overshoot even
positive control radii; the plot labels that case. Road width, boundary
intersections, vehicle feasibility, elevation, and simulator export are separate
steps. The CSV files here contain experimental geometry, not the simulator's
track format. They use radians and metres, and are sampled densely *within each
angular interval*, not uniformly in road distance.

## Setup and run from PowerShell

From this repository folder, using Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe experiment.py
.\.venv\Scripts\python.exe experiment.py --seed 28
.\.venv\Scripts\python.exe -m unittest discover -s . -p "test_*.py"
```

Each run creates a new directory under `outputs/roads/` with parameters, package
versions, raw and used samples, all three interpolated curves, and a comparison
image. Generated outputs and the environment are ignored by Git; the code,
notebook, and dependency list can be versioned. Output directory names use UTC.

To replay a run, use `--config` with its `parameters.json`. Changing the seed or
sample count draws a new realization. Interpolators in one run always use the
same samples. Keep the sample arrays fixed when isolating interpolation effects.

Only NumPy, SciPy, and Matplotlib are required for the scripts. JupyterLab is
included for the notebook workflow. The exact tested top-level versions are
listed in `requirements.txt`; transitive dependencies are not locked.

## Mathematical implementation references

- [SciPy periodic cubic spline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html)
- [SciPy PCHIP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)
- [SciPy circular Gaussian filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html)
