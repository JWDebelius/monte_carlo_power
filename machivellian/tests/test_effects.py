from unittest import TestCase, main

import numpy as np
import numpy.testing as npt

from machivellian.effects import (z_effect,
                                  z_power,
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
        known_d = 0.18868749
        known_sd = 1.80722488e-07
        known_dn = 10
        d, sd, dn = z_effect(self.counts, self.power, self.alpha)
        npt.assert_almost_equal(known_d, d, 5)
        npt.assert_almost_equal(np.log10(known_sd), np.log10(sd), 3)
        self.assertEqual(known_dn, dn)

    def test_z_effect_lims(self):
        known_d = 0.18883863
        known_sd = 3.41713978e-08
        known_dn = 8
        d, sd, dn = z_effect(self.counts, self.power, self.alpha, size_lim=20)
        npt.assert_almost_equal(known_d, d, 5)
        npt.assert_almost_equal(np.log10(known_sd), np.log10(sd), 3)
        self.assertEqual(known_dn, dn)

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
