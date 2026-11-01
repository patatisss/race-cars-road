"""Small building blocks for closed polar-road experiments. Angles are radians."""

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, PPoly, make_interp_spline
from scipy.ndimage import gaussian_filter1d
from scipy.special import i0e

TAU = 2 * np.pi


def angular_density(angle, centers_deg, width_deg, uniform=0.15):
    if not len(centers_deg) or width_deg <= 0 or not 0 <= uniform <= 1:
        raise ValueError("Use at least one mode, positive width, and uniform in [0, 1].")
    concentration = 1 / np.deg2rad(width_deg) ** 2
    distance = np.asarray(angle)[..., None] - np.deg2rad(centers_deg)
    peaks = np.exp(concentration * (np.cos(distance) - 1)).mean(axis=-1)
    return uniform / TAU + (1 - uniform) * peaks / (TAU * i0e(concentration))


def sample_angles(count, centers_deg, width_deg, rng, uniform=0.15):
    angular_density(0, centers_deg, width_deg, uniform)
    if count < 4:
        raise ValueError("Use at least four angular samples.")
    centers = rng.choice(np.deg2rad(centers_deg), count - 1)
    clustered = rng.vonmises(centers, 1 / np.deg2rad(width_deg) ** 2) % TAU
    angles = np.where(rng.random(count - 1) < uniform, rng.uniform(0, TAU, count - 1), clustered)
    return np.sort(np.r_[0.0, angles])


def sample_radii(angles, centers_deg, width_deg, rng, start_radius=100, variation=0.2, quiet=0.1):
    if start_radius <= 0 or variation < 0 or not 0 <= quiet <= 1:
        raise ValueError("Use positive radius, nonnegative variation, and quiet in [0, 1].")
    grid = np.linspace(0, TAU, 4097)
    activity = angular_density(angles, centers_deg, width_deg, uniform=0)
    activity /= angular_density(grid, centers_deg, width_deg, uniform=0).max()
    variance = quiet**2 + (1 - quiet**2) * np.clip(activity, 0, 1)
    weights = (variance + np.roll(variance, -1)) / 2
    clock = np.r_[0.0, np.cumsum(weights)]
    walk = np.r_[0.0, np.cumsum(rng.normal(size=len(angles)) * np.sqrt(weights))]
    bridge = walk - clock / clock[-1] * walk[-1]
    return start_radius * np.exp(variation * np.sqrt(6 / clock[-1]) * bridge[:-1])


def smooth_radii(radii, samples=0):
    if samples < 0:
        raise ValueError("Smoothing width must be nonnegative.")
    log_radius = np.log(radii / radii[0])
    smoothed = gaussian_filter1d(log_radius, samples, mode="wrap") if samples else log_radius
    return radii[0] * np.exp(smoothed - smoothed[0])


def fit_radius(angles, radii, method="cubic"):
    angles, radii = np.asarray(angles, dtype=float), np.asarray(radii, dtype=float)
    if (angles.ndim != 1 or radii.shape != angles.shape or len(angles) < 4
            or not np.isfinite(angles).all() or not np.isfinite(radii).all()
            or angles[0] != 0 or angles[-1] >= TAU or np.any(np.diff(angles) <= 0)
            or np.any(radii <= 0)):
        raise ValueError("Need positive radii and distinct increasing angles in [0, 2π), starting at zero.")
    closed_angles, closed_radii = np.r_[angles, TAU], np.r_[radii, radii[0]]
    if method == "linear":
        return PPoly.from_spline(make_interp_spline(closed_angles, closed_radii, k=1))
    if method == "cubic":
        return CubicSpline(closed_angles, closed_radii, bc_type="periodic")
    if method == "pchip":
        neighbors = np.r_[angles[-2:] - TAU, angles, angles[:2] + TAU]
        return PchipInterpolator(neighbors, np.r_[radii[-2:], radii, radii[:2]])
    raise ValueError(f"Unknown interpolator: {method}")


def evaluate(curve, angles):
    radius, first, second = (curve(angles, order) for order in range(3))
    denominator = (radius**2 + first**2) ** 1.5
    curvature = np.divide(radius**2 + 2 * first**2 - radius * second, denominator,
                          out=np.full_like(radius, np.nan), where=(radius > 0) & (denominator > 0))
    return dict(angle=angles, radius=radius, x=radius * np.cos(angles),
                y=radius * np.sin(angles), curvature=curvature)
