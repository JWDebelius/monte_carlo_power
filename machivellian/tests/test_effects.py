from unittest import TestCase, main

import numpy as np
import numpy.testing as npt

from machivellian.effects import (z_effect,
                                  z_power,
                                  cv_z_effect,
                                  _check_shapes,
                                  )


class PowerSimulation(TestCase):

    def setUp(self):
        np.random.seed(5)
        self.counts = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
        self.power = np.array([0.04698995, 0.10223453, 0.15022543, 0.19666182,
                               0.24228406, 0.28709498, 0.33092360, 0.37356885,
                               0.41484527, 0.45459791])
        self.alpha = 0.05 / 2
        self.effect = 0.5

    def test_z_effect(self):
        known_effects = np.array([0.12754385,  0.17842640,  0.18489939,
                                  0.18700873,  0.18797725,  0.18851606,
                                  0.18885525,  0.18908777,  0.18925724,
                                  0.18938648])
        test_effects = z_effect(self.counts, self.power, self.alpha)
        self.assertEqual(known_effects.shape, test_effects.shape)
        npt.assert_almost_equal(known_effects, test_effects, 5)

    def test_z_power(self):
        known_power = np.array([0.19991357,  0.49063676,  0.70541390,
                                0.84087872,  0.91836203,  0.95977970,
                                0.98082831,  0.99110988,  0.99597313,
                                0.99821261])
        test_power = z_power(self.counts, self.effect, self.alpha)
        npt.assert_almost_equal(known_power, test_power)

    def test_z_power_array(self):
        known_power = np.array([0.19991357,  0.49063676,  0.70541390,
                                0.84087872,  0.91836203,  0.95977970,
                                0.98082831,  0.99110988,  0.99597313,
                                0.99821261])
        test_power = z_power(self.counts,
                             np.array([0.5, 0.5, np.nan]),
                             self.alpha)
        npt.assert_almost_equal(known_power, test_power)

    def test_cv_z_effect(self):
        known_summary = np.array([[15.,   0.10223453,   0.10909324],
                                  [25.,   0.15022543,   0.15306127],
                                  [35.,   0.19666182,   0.19673070],
                                  [45.,   0.24228406,   0.24009355],
                                  [55.,   0.28709498,   0.28293113],
                                  [65.,   0.33092360,   0.32499586],
                                  [75.,   0.37356885,   0.36605964],
                                  [85.,   0.41484527,   0.40592785],
                                  [95.,   0.45459791,   0.44444222]])
        known_effect = {'effect': 0.18704606,
                        'effect_std': 0.00332908,
                        'effect_n': 9,
                        'train_r2': 0.99699031,
                        'train_rmse': 1.524549345004565e-09,
                        }
        effect, summary = cv_z_effect(self.counts, self.power, self.alpha)
        npt.assert_almost_equal(known_summary, summary, 5)
        self.assertEqual(known_effect.keys(), effect.keys())
        for k, v in effect.items():
            npt.assert_almost_equal(v, known_effect[k], 5)

    def test_check_shapes_2d(self):
        counts, power = _check_shapes(self.counts,
                                      np.vstack([self.power, self.power]))
        self.assertEqual(counts.shape, power.shape)
        self.assertEqual(len(power.shape), 1)

    def test_check_shapes_1d(self):
        counts, power = _check_shapes(self.counts, self.power)
        self.assertEqual(counts.shape, power.shape)

    def test_check_shapes_too_many_counts(self):
        with self.assertRaises(ValueError):
            _check_shapes(np.atleast_2d(self.counts), self.power)

    def test_check_shapes_1d_error(self):
        with self.assertRaises(ValueError):
            _check_shapes(self.counts[:3], self.power)

    def test_check_shapes_other_error(self):
        with self.assertRaises(ValueError):
            _check_shapes(self.counts[:3], np.vstack([self.power, self.power]))

if __name__ == '__main__':
    main()
