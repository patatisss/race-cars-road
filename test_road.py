import unittest

import numpy as np
from numpy.testing import assert_allclose

from road import TAU, angular_density, evaluate, fit_radius, sample_angles, sample_radii, smooth_radii


class RoadTests(unittest.TestCase):
    def test_circular_density_and_repeatable_samples(self):
        grid = np.linspace(0, TAU, 10001)
        density = angular_density(grid, [0, 120, 240], 12)
        self.assertAlmostEqual(np.trapezoid(density, grid), 1, places=10)
        self.assertAlmostEqual(density[0], density[-1], places=12)
        first = sample_angles(128, [0, 120, 240], 12, np.random.default_rng(3))
        second = sample_angles(128, [0, 120, 240], 12, np.random.default_rng(3))
        assert_allclose(first, second)
        self.assertTrue(np.all(np.diff(first) > 0))
        clustered = angular_density(first, [0, 120, 240], 12) > np.median(density)
        self.assertGreater(np.mean(clustered), 0.7)

    def test_periodic_interpolation_and_circle_curvature(self):
        angles = np.linspace(0, TAU, 32, endpoint=False)
        radii = 100 + 10 * np.sin(3 * angles)
        for method, orders in [("linear", 1), ("pchip", 2), ("cubic", 3)]:
            curve = fit_radius(angles, radii, method)
            assert_allclose(curve(angles), radii, atol=1e-10)
            for order in range(orders):
                assert_allclose(curve(0, order), curve(TAU, order), atol=1e-8)
            circle = fit_radius(angles, np.full_like(angles, 100), method)
            assert_allclose(evaluate(circle, np.linspace(0, TAU, 100))["curvature"], 0.01)

    def test_clustered_radial_variation_and_smoothing(self):
        angles = np.linspace(0, TAU, 256, endpoint=False)
        activity = angular_density(angles, [0, 120, 240], 12, uniform=0)
        activity /= activity.max()
        local = (activity + np.roll(activity, -1)) / 2
        near, far = [], []
        for seed in range(80):
            radii = sample_radii(angles, [0, 120, 240], 12, np.random.default_rng(seed), variation=0.3)
            increments = np.log(np.roll(radii, -1) / radii)
            near.extend(increments[local > 0.7])
            far.extend(increments[local < 0.05])
            self.assertEqual(radii[0], 100)
            self.assertTrue(np.all(radii > 0))
        self.assertGreater(np.var(near), 5 * np.var(far))
        smoothed = smooth_radii(radii, 2)
        self.assertEqual(smoothed[0], radii[0])
        self.assertLess(np.sum(np.diff(np.r_[smoothed, smoothed[0]])**2),
                        np.sum(np.diff(np.r_[radii, radii[0]])**2))
        assert_allclose(smooth_radii(radii, 0), radii)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            sample_angles(2, [0], 12, np.random.default_rng())
        with self.assertRaises(ValueError):
            angular_density(0, [], 12)
        with self.assertRaises(ValueError):
            fit_radius([0, 1, 1, 2], [100] * 4)
        with self.assertRaises(ValueError):
            fit_radius([0, 1, 2, 3], [100, -1, 100, 100])


if __name__ == "__main__":
    unittest.main()
