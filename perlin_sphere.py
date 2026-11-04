"""Generate closed Perlin-perturbed sphere routes with constrained 2D projections."""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SETTINGS = dict(
    seed=174390035, base_seed=19666008, sphere_radius=100.0, wandering=0.65,
    noise_cells=12, min_radius_3d=5.0, tight_radius=10.0, min_radius_2d=5.0,
    region_fraction=0.8, visit_fraction=0.95, total_turns=(6, 10),
    top_turns=(2, 4), bottom_turns=(2, 4), max_attempts=50000,
    screening_samples=801, verification_samples=12801,
)


def periodic_perlin(angle, seed, cells=12):
    rng = np.random.default_rng(seed)
    gradients = rng.uniform(-1, 1, size=(cells, 3))
    coordinate = angle * cells / (2 * np.pi)
    index = np.floor(coordinate).astype(int)
    fraction = (coordinate - index)[:, None]
    left, right = gradients[index % cells], gradients[(index + 1) % cells]
    fade = fraction**3 * (fraction * (6 * fraction - 15) + 10)
    fade_first = 30 * fraction**2 * (fraction - 1)**2
    fade_second = 60 * fraction * (2 * fraction**2 - 3 * fraction + 1)
    difference = right * (fraction - 1) - left * fraction
    value = left * fraction + difference * fade
    first = left + (right - left) * fade + difference * fade_first
    second = 2 * (right - left) * fade_first + difference * fade_second
    rate = cells / (2 * np.pi)
    return value, first * rate, second * rate**2


@lru_cache(maxsize=4)
def base_route(wandering, count, base_seed):
    rng = np.random.default_rng(base_seed)
    angle = np.linspace(0, 2 * np.pi, count)
    route = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(count)))
    first = np.column_stack((-np.sin(angle), np.cos(angle), np.zeros(count)))
    second = -route.copy()
    for frequency in range(1, 13):
        coefficients = rng.normal(size=(2, 3)) / frequency**2
        route += wandering * (
            np.cos(frequency * angle)[:, None] * coefficients[0]
            + np.sin(frequency * angle)[:, None] * coefficients[1]
        )
        first += wandering * frequency * (
            -np.sin(frequency * angle)[:, None] * coefficients[0]
            + np.cos(frequency * angle)[:, None] * coefficients[1]
        )
        second -= wandering * frequency**2 * (
            np.cos(frequency * angle)[:, None] * coefficients[0]
            + np.sin(frequency * angle)[:, None] * coefficients[1]
        )
    left = -route[0] / np.linalg.norm(route[0])
    tangent = first[0] - np.dot(first[0], left) * left
    tangent /= np.linalg.norm(tangent)
    roll = np.random.default_rng(base_seed).uniform(-np.pi, np.pi)
    north = -np.sin(roll) * tangent + np.cos(roll) * np.cross(left, tangent)
    top_offset = angle[:, None] - 1.75
    lift = 0.7 * np.exp(4 * (np.cos(top_offset) - 1))
    route += lift * north
    first += -4 * np.sin(top_offset) * lift * north
    second += (16 * np.sin(top_offset)**2 - 4 * np.cos(top_offset)) * lift * north
    offset = angle[:, None] - np.array([1.75, 5])
    bump = np.array([0.2, 0.6]) * np.exp(4 * (np.cos(offset) - 1))
    weight = 0.2 + bump.sum(axis=1, keepdims=True)
    weight_first = (-4 * np.sin(offset) * bump).sum(axis=1, keepdims=True)
    weight_second = ((16 * np.sin(offset)**2 - 4 * np.cos(offset)) * bump).sum(axis=1, keepdims=True)
    return angle, route, first, second, weight, weight_first, weight_second


def surface_route(seed, settings, count):
    angle, base, base_first, base_second, weight, weight_first, weight_second = base_route(
        settings["wandering"], count, settings["base_seed"])
    noise, noise_first, noise_second = periodic_perlin(angle, seed, settings["noise_cells"])
    route = base + weight * noise
    first = base_first + weight * noise_first + weight_first * noise
    second = base_second + weight * noise_second + 2 * weight_first * noise_first + weight_second * noise
    norm = np.linalg.norm(route, axis=1, keepdims=True)
    logarithmic_slope = np.sum(route * first, axis=1, keepdims=True) / norm**2
    slope_derivative = (np.sum(first**2 + route * second, axis=1, keepdims=True)
                        / norm**2 - 2 * logarithmic_slope**2)
    radius = settings["sphere_radius"]
    acceleration = radius / norm * (second - 2 * logarithmic_slope * first
                                   + (logarithmic_slope**2 - slope_derivative) * route)
    velocity = radius / norm * (first - logarithmic_slope * route)
    route = radius * route / norm
    route[-1] = route[0]
    left = -route[0] / np.linalg.norm(route[0])
    tangent = velocity[0] / np.linalg.norm(velocity[0])
    rotation = np.column_stack((left, tangent, np.cross(left, tangent)))
    roll = np.random.default_rng(settings["base_seed"]).uniform(-np.pi, np.pi)
    rotation = rotation @ np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)],
                                   [0, np.sin(roll), np.cos(roll)]])
    return route @ rotation, velocity @ rotation, acceleration @ rotation


def has_crossing(points):
    starts, ends = points[:-1], points[1:]
    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    for index in range(len(starts) - 2):
        stop = len(starts) - (index == 0)
        other_start, other_end = starts[index + 2:stop], ends[index + 2:stop]
        direction = ends[index] - starts[index]
        side_a = cross(direction, other_start - starts[index])
        side_b = cross(direction, other_end - starts[index])
        side_c = cross(other_end - other_start, starts[index] - other_start)
        side_d = cross(other_end - other_start, ends[index] - other_start)
        overlap = (np.maximum(np.minimum(starts[index], ends[index]),
                              np.minimum(other_start, other_end))
                   <= np.minimum(np.maximum(starts[index], ends[index]),
                                 np.maximum(other_start, other_end))).all(axis=1)
        if np.any(overlap & (side_a * side_b <= 0) & (side_c * side_d <= 0)):
            return True
    return False


def tight_turn_peaks(curvature, tight_radius=10):
    values = curvature[:-1]
    tight = values >= 1 / tight_radius
    if tight.all():
        return np.array([np.argmax(values)])
    starts = np.flatnonzero(tight & ~np.roll(tight, 1))
    peaks = []
    for start in starts:
        indices = (start + np.arange(len(values))) % len(values)
        end = np.flatnonzero(~tight[indices])[0]
        section = indices[:end]
        peaks.append(section[np.argmax(values[section])])
    return np.asarray(peaks, dtype=int)


def check_route(seed, settings, count):
    points, velocity, acceleration = surface_route(seed, settings, count)
    speed_3d = np.linalg.norm(velocity, axis=1)
    speed_2d = np.linalg.norm(velocity[:, :2], axis=1)
    if not np.isfinite(points).all() or speed_3d.min() < 1e-8 or speed_2d.min() < 1:
        return None
    curvature_3d = np.linalg.norm(np.cross(velocity, acceleration), axis=1) / speed_3d**3
    curvature_2d = (velocity[:, 0] * acceleration[:, 1]
                    - velocity[:, 1] * acceleration[:, 0]) / speed_2d**3
    if (curvature_3d.max() > 0.98 / settings["min_radius_3d"]
            or np.abs(curvature_2d).max() > 0.98 / settings["min_radius_2d"]):
        return None
    visit = settings["sphere_radius"] * settings["visit_fraction"]
    if points[:, 2].max() < visit or points[:, 2].min() > -visit:
        return None
    peaks = tight_turn_peaks(curvature_3d, settings["tight_radius"])
    cap = settings["sphere_radius"] * settings["region_fraction"]
    counts = dict(total_turns=len(peaks), top_turns=int(np.sum(points[peaks, 2] >= cap)),
                  bottom_turns=int(np.sum(points[peaks, 2] <= -cap)))
    if any(not settings[key][0] <= value <= settings[key][1] for key, value in counts.items()):
        return None
    if has_crossing(points[:, :2]):
        return None
    return dict(points=points, curvature_3d=curvature_3d, curvature_2d=curvature_2d,
                peaks=peaks, seed=seed, **counts)


def generate(settings=None):
    settings = SETTINGS | (settings or {})
    for key in ("sphere_radius", "min_radius_3d", "tight_radius", "min_radius_2d"):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 < settings["region_fraction"] <= settings["visit_fraction"] < 1:
        raise ValueError("Require 0 < region_fraction <= visit_fraction < 1")
    if settings["min_radius_3d"] >= settings["tight_radius"]:
        raise ValueError("min_radius_3d must be below tight_radius")
    for key in ("total_turns", "top_turns", "bottom_turns"):
        bounds = settings[key]
        if len(bounds) != 2 or any(not isinstance(n, int) or n < 0 for n in bounds) or bounds[0] > bounds[1]:
            raise ValueError(f"{key} must contain ordered nonnegative integer bounds")
    for key in ("seed", "base_seed", "noise_cells", "max_attempts", "screening_samples", "verification_samples"):
        if not isinstance(settings[key], int) or settings[key] < (0 if "seed" in key else 1):
            raise ValueError(f"Invalid {key}")
    if not np.isfinite(settings["wandering"]) or settings["wandering"] < 0:
        raise ValueError("wandering must be finite and nonnegative")
    if not settings["verification_samples"] >= settings["screening_samples"] >= max(65, 4 * settings["noise_cells"] + 1):
        raise ValueError("Use verification_samples >= screening_samples >= max(65, 4*noise_cells+1)")
    for attempt in range(settings["max_attempts"]):
        seed = settings["seed"] + 1000 * attempt
        if check_route(seed, settings, settings["screening_samples"]) is None:
            continue
        result = check_route(seed, settings, settings["verification_samples"])
        if result is not None:
            return result | dict(settings=settings, attempts=attempt + 1)
    raise RuntimeError("No route met all constraints within max_attempts. Try another seed or increase max_attempts.")


def plot(result):
    points, peaks, settings = result["points"], result["peaks"], result["settings"]
    radius = settings["sphere_radius"]
    cap = radius * settings["region_fraction"]
    cap_peaks = peaks[np.abs(points[peaks, 2]) >= cap]
    highlights = points.copy()
    highlights[result["curvature_3d"] < 1 / settings["tight_radius"]] = np.nan
    figure = plt.figure(figsize=(11, 5), layout="constrained")
    surface = figure.add_subplot(1, 2, 1, projection="3d")
    projection = figure.add_subplot(1, 2, 2)
    longitude, latitude = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 13))
    surface.plot_wireframe(radius * np.sin(latitude) * np.cos(longitude),
                           radius * np.sin(latitude) * np.sin(longitude),
                           radius * np.cos(latitude), color="0.8", linewidth=0.4, alpha=0.5)
    pole_indices = [np.argmax(points[:, 2]), np.argmin(points[:, 2])]
    for axis, dimensions in ((surface, 3), (projection, 2)):
        axis.plot(*points[:, :dimensions].T, color="tab:blue", linewidth=2)
        axis.plot(*highlights[:, :dimensions].T, color="tab:orange", linewidth=4)
        axis.scatter(*points[cap_peaks, :dimensions].T, color="tab:purple", s=45)
        axis.scatter(*points[0, :dimensions], color="tab:red", s=25)
        axis.scatter(*points[pole_indices, :dimensions].T, color="tab:orange", marker="*", s=70)
        axis.set(xlim=(-1.1 * radius, 1.1 * radius), ylim=(-1.1 * radius, 1.1 * radius),
                 xlabel="x (m)", ylabel="y (m)")
    surface.set(zlim=(-1.1 * radius, 1.1 * radius), zlabel="z (m)", title="Sphere route")
    surface.set_box_aspect((1, 1, 1))
    surface.view_init(elev=24, azim=-58)
    projection.set(aspect="equal", title=f"Projection · min. radius {1 / np.abs(result['curvature_2d']).max():.2f} m")
    projection.grid(alpha=0.2)
    figure.suptitle(f"{result['total_turns']} tight turns · {result['top_turns']} near top · {result['bottom_turns']} near bottom\n"
                   "Orange: tight bends · purple: pole-region bend peaks · stars: pole visits · red: start/finish")
    return figure


def save_run(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    figure = plot(result)
    figure.savefig(directory / "comparison.png", dpi=180)
    plt.close(figure)
    points = result["points"]
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))]
    data = np.column_stack((distance, points, result["curvature_3d"], result["curvature_2d"]))
    np.savetxt(directory / "route.csv", data, delimiter=",",
               header="s_projection_m,x_m,y_m,z_m,curvature_3d_1pm,curvature_2d_1pm", comments="")
    (directory / "parameters.json").write_text(json.dumps(result["settings"], indent=2) + "\n")
    summary = {key: result[key] for key in ("seed", "attempts", "total_turns", "top_turns", "bottom_turns")}
    summary |= dict(min_radius_3d=float(1 / result["curvature_3d"].max()),
                    min_radius_2d=float(1 / np.abs(result["curvature_2d"]).max()),
                    top_height=float(points[:, 2].max()), bottom_height=float(points[:, 2].min()))
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON settings overrides or saved parameters.json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, help="New output directory")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text()) if args.config else {}
    if args.seed is not None:
        settings["seed"] = args.seed
    result = generate(settings)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.out or Path(__file__).resolve().parent / "outputs" / "perlin-sphere" / stamp
    print(save_run(result, output))
    print(f"Seed {result['seed']} · {result['total_turns']} turns · "
          f"{result['top_turns']} top · {result['bottom_turns']} bottom · {result['attempts']} attempts")
