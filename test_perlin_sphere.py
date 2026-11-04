import unittest

import numpy as np

from perlin_sphere import generate, has_crossing, periodic_perlin, tight_turn_peaks


class PerlinSphereTests(unittest.TestCase):
    def test_periodic_noise_derivatives(self):
        angle = np.linspace(0, 2 * np.pi, 24001)
        value, first, second = periodic_perlin(angle, 35)
        for array in (value, first, second):
            np.testing.assert_allclose(array[0], array[-1], atol=1e-12)
        step = angle[1] - angle[0]
        fraction = (angle * 12 / (2 * np.pi)) % 1
        inside_cells = (fraction > 0.01) & (fraction < 0.99)
        np.testing.assert_allclose(np.gradient(value, step, axis=0)[inside_cells],
                                   first[inside_cells], atol=1e-4)
        np.testing.assert_allclose(np.gradient(first, step, axis=0)[inside_cells],
                                   second[inside_cells], atol=1e-3)

    def test_turn_across_lap_seam_is_counted_once(self):
        curvature = np.array([0.14, 0.12, 0.02, 0.13, 0.02, 0.11, 0.14])
        np.testing.assert_array_equal(tight_turn_peaks(curvature), [3, 0])
        self.assertEqual(len(tight_turn_peaks(np.full(10, 0.15))), 1)
        self.assertEqual(len(tight_turn_peaks(np.full(10, 0.02))), 0)

    def test_crossing_and_contact(self):
        self.assertTrue(has_crossing(np.array([[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]])))
        self.assertTrue(has_crossing(np.array([[0, 0], [2, 0], [1, 1], [1, 0], [0, 0]])))
        self.assertFalse(has_crossing(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])))

    def test_invalid_settings(self):
        for settings in ({"min_radius_2d": -1}, {"total_turns": (10, 6)},
                         {"verification_samples": 40}, {"max_attempts": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                generate(settings)


if __name__ == "__main__":
    unittest.main()
