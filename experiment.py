"""Edit SETTINGS, run this file, or call compare(settings) from the notebook."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from road import TAU, angular_density, evaluate, fit_radius, sample_angles, sample_radii, smooth_radii

SETTINGS = dict(seed=27, count=64, centers_deg=[30, 120, 210, 300], width_deg=12,
                uniform=0.15, radius_range=[80, 120], variation=0.25, quiet=0.1, smooth_samples=1.5)
METHODS = ("linear", "pchip", "cubic")


def compare(settings):
    angle_seed, radius_seed = np.random.SeedSequence(settings["seed"]).spawn(2)
    angle_rng, radius_rng = np.random.default_rng(angle_seed), np.random.default_rng(radius_seed)
    modes, width = settings["centers_deg"], settings["width_deg"]
    angles = sample_angles(settings["count"], modes, width, angle_rng, settings["uniform"])
    start_radius = radius_rng.uniform(*settings["radius_range"])
    raw = sample_radii(angles, modes, width, radius_rng, start_radius, settings["variation"], settings["quiet"])
    used = smooth_radii(raw, settings["smooth_samples"])
    closed = np.r_[angles, TAU]
    query = (closed[:-1, None] + np.diff(closed)[:, None] * np.linspace(0, 1, 32, endpoint=False)).ravel()
    query = np.r_[query, TAU]
    curves = {method: evaluate(fit_radius(angles, used, method), query) for method in METHODS}
    for method in ("linear", "pchip"):
        curves[method]["curvature"][::32] = np.nan

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    density_ax, road_ax, radius_ax, curvature_ax = axes.flat
    grid = np.linspace(0, TAU, 2049)
    density_ax.plot(np.rad2deg(grid), angular_density(grid, modes, width, settings["uniform"]) * np.pi / 180, color="0.25")
    density_ax.plot(np.rad2deg(angles), np.zeros_like(angles), "|", color="0.45", clip_on=False)
    density_ax.set(title="Angular sampling", xlabel="Polar angle (°)", ylabel="Density (1/°)")
    road_ax.scatter(raw * np.cos(angles), raw * np.sin(angles), s=12, color="0.7", label="Raw samples", zorder=3)
    road_ax.scatter(used * np.cos(angles), used * np.sin(angles), s=10, color="0.15", label="Used samples", zorder=4)
    radius_ax.scatter(np.rad2deg(angles), raw, s=12, color="0.7")
    radius_ax.scatter(np.rad2deg(angles), used, s=10, color="0.15", zorder=4)
    for index, (method, values) in enumerate(curves.items()):
        color, style = f"C{index}", "--" if method == "linear" else "-"
        label = method + (" (radius ≤ 0)" if np.any(values["radius"] <= 0) else "")
        road_ax.plot(values["x"], values["y"], style, color=color, label=label)
        radius_ax.plot(np.rad2deg(query), values["radius"], style, color=color)
        curvature_ax.plot(np.rad2deg(query), values["curvature"], style, color=color)
    road_ax.set(title="Candidate centerlines", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    radius_ax.set(title="Radius and optional smoothing", xlabel="Polar angle (°)", ylabel="Radius (m)")
    curvature_ax.set(title="Curvature between joins", xlabel="Polar angle (°)", ylabel="Curvature (1/m)")
    figure.legend(*road_ax.get_legend_handles_labels(), loc="outside lower center", ncol=5)
    figure.suptitle(f"Seed {settings['seed']} · {settings['count']} samples · smoothing {settings['smooth_samples']:g} samples")
    return figure, np.column_stack([angles, raw, used]), curves


def save_run(settings, figure, samples, curves, directory=None):
    if directory is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = Path(__file__).resolve().parent / "outputs" / "roads" / stamp
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "parameters.json").write_text(json.dumps(settings, indent=2) + "\n")
    packages = {name: version(name) for name in ("numpy", "scipy", "matplotlib")}
    (directory / "versions.json").write_text(json.dumps(packages, indent=2) + "\n")
    np.savetxt(directory / "samples.csv", samples, delimiter=",", header="angle_rad,raw_radius_m,used_radius_m", comments="")
    for method, values in curves.items():
        np.savetxt(directory / f"{method}.csv", np.column_stack(list(values.values())), delimiter=",",
                   header="angle_rad,radius_m,x_m,y_m,curvature_per_m", comments="")
    figure.savefig(directory / "comparison.png", dpi=160)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Replay a saved parameters.json")
    parser.add_argument("--seed", type=int, help="Override the random seed")
    parser.add_argument("--out", type=Path, help="New output directory")
    args = parser.parse_args()
    settings = SETTINGS | (json.loads(args.config.read_text()) if args.config else {})
    if args.seed is not None:
        settings["seed"] = args.seed
    plt.switch_backend("Agg")
    figure, samples, curves = compare(settings)
    print(save_run(settings, figure, samples, curves, args.out))
    plt.close(figure)
