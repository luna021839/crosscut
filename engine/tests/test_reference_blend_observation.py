from __future__ import annotations

import unittest

import numpy as np

from crosscut.reference_blend_observation import measure_aligned_reference_blend


class ReferenceBlendObservationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(43)
        self.left = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
        self.right = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
        self.support = np.ones((24, 32), dtype=bool)

    def test_known_blends_and_endpoints_are_measured_not_authorized(self):
        for alpha in (0.0, .2, .5, .8, 1.0):
            with self.subTest(alpha=alpha):
                target = np.rint((1-alpha)*self.left + alpha*self.right).astype(np.uint8)
                result = measure_aligned_reference_blend(target, self.left, self.right, self.support)
                self.assertAlmostEqual(result["incoming_alpha"], alpha, delta=.001)
                self.assertLess(result["mixture_mse"], 1e-5)
                self.assertFalse(result["transition_kind_authorized"])
                self.assertFalse(result["source_identity_inferred"])

    def test_invalid_regions_are_excluded_and_inputs_are_not_mutated(self):
        target = np.rint(.6*self.left + .4*self.right).astype(np.uint8)
        self.support[:, :16] = False
        target[:, :16] = 255
        saved = target.copy()
        result = measure_aligned_reference_blend(target, self.left, self.right, self.support)
        self.assertAlmostEqual(result["incoming_alpha"], .4, delta=.001)
        self.assertEqual(result["support_fraction"], .5)
        np.testing.assert_array_equal(target, saved)

    def test_identical_sources_and_empty_support_do_not_invent_alpha(self):
        result = measure_aligned_reference_blend(self.left, self.left, self.left, self.support)
        self.assertEqual(result["status"], "indistinguishable_sources")
        self.assertNotIn("incoming_alpha", result)
        result = measure_aligned_reference_blend(self.left, self.left, self.right, ~self.support)
        self.assertEqual(result["status"], "no_shared_support")

    def test_outside_convex_range_keeps_unbounded_diagnostic(self):
        left = np.full_like(self.left, 100)
        right = np.full_like(self.right, 150)
        result = measure_aligned_reference_blend(np.full_like(left, 220), left, right, self.support)
        self.assertEqual(result["incoming_alpha"], 1.0)
        self.assertGreater(result["unbounded_incoming_alpha"], 1.0)
        self.assertGreater(result["mixture_mse"], 0.0)

    def test_invalid_canvases_and_masks_fail(self):
        for target, mask in ((self.left.astype(float), self.support),
                             (self.left[:-1], self.support),
                             (self.left, self.support.astype(np.uint8))):
            with self.subTest(shape=target.shape), self.assertRaises(ValueError):
                measure_aligned_reference_blend(target, self.left, self.right, mask)
