from unittest import TestCase, main

import numpy as np
import numpy.testing as npt

from emp_power.effects import (f_effect,
                               t_effect,
                               z_effect,
                               f_power,
                               t_power,
                               z_power,
                               )


class PowerSimulation(TestCase):

    def setUp(self):
        self.counts = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
        self.power = np.array([0.04698995, 0.10223453, 0.15022543, 0.19666182,
                               0.24228406, 0.28709498, 0.33092360, 0.37356885,
                               0.41484527, 0.45459791])
        self.alpha = 0.05
        self.effect = 0.5

    def test_f_effect(self):
        known_effects = np.array([[np.nan,      0.18543664,  0.19105543,
                                   0.19188928,  0.19194147,  0.19182950,
                                   0.19168930,  0.19155769,  0.19144256,
                                   0.19134509]])
        test_effects = f_effect(self.counts, self.power, self.alpha)

        self.assertTrue(np.isnan(test_effects[0, 0]))
        self.assertEqual(known_effects.shape, test_effects.shape)
        npt.assert_almost_equal(known_effects[0, 1:], test_effects[0, 1:])

    def test_t_effect(self):
        known_effects = np.array([[np.nan,      0.25190748,  0.26429554,
                                   0.26728415,  0.26832483,  0.26877072,
                                   0.26898195,  0.26908683,  0.26914712,
                                   0.26918235]])
        test_effects = t_effect(self.counts, self.power, self.alpha)
        self.assertTrue(np.isnan(test_effects[0, 0]))
        self.assertEqual(known_effects.shape, test_effects.shape)
        npt.assert_almost_equal(known_effects[0, 1:], test_effects[0, 1:])

    def test_t_effect_ratio(self):
        known_effects = np.array([[np.nan,      0.21553381,  0.22731081,
                                   0.23036026,  0.23151793,  0.23206086,
                                   0.23235050,  0.23252443,  0.23263891,
                                   0.23271512]])
        test_effects = t_effect(self.counts, self.power, self.alpha, ratio=2)
        self.assertTrue(np.isnan(test_effects[0, 0]))
        self.assertEqual(known_effects.shape, test_effects.shape)
        npt.assert_almost_equal(known_effects[0, 1:], test_effects[0, 1:])

    def test_z_effect(self):
        known_effects = np.array([[0.12754385,  0.17842640,  0.18489939,
                                   0.18700873,  0.18797725,  0.18851606,
                                   0.18885525,  0.18908777,  0.18925724,
                                   0.18938648]])
        test_effects = z_effect(self.counts, self.power, self.alpha)
        self.assertEqual(known_effects.shape, test_effects.shape)
        npt.assert_almost_equal(known_effects, test_effects)

    def test_f_power(self):
        known_power = np.array([0.12656744,  0.43392591,  0.66810794,
                                0.81885655,  0.90630465,  0.95353589,
                                0.97773116,  0.98962623,  0.99528244,
                                0.99789883])
        test_power = f_power(self.counts, self.effect, self.alpha)
        npt.assert_almost_equal(known_power, test_power)

    def test_t_power(self):
        known_power = np.array([0.10768599,  0.26244303,  0.41010033,
                                0.54068791,  0.65018550,  0.73848657,
                                0.80758442,  0.86036751,  0.89989407,
                                0.92900109])
        test_power = t_power(self.counts, self.effect, self.alpha)
        npt.assert_almost_equal(known_power, test_power)

    def test_t_power_ratio(self):
        known_power = np.array([0.13533966,  0.33962944,  0.52172130,
                                0.66738897,  0.77602251,  0.85309549,
                                0.90574798,  0.94065587,  0.96323879,
                                0.97755102])
        test_power = t_power(self.counts, self.effect, self.alpha, ratio=2)
        npt.assert_almost_equal(known_power, test_power)

    def test_z_power(self):
        known_power = np.array([0.19991357,  0.49063676,  0.70541390,
                                0.84087872,  0.91836203,  0.95977970,
                                0.98082831,  0.99110988,  0.99597313,
                                0.99821261])
        test_power = z_power(self.counts, self.effect, self.alpha)
        npt.assert_almost_equal(known_power, test_power)


if __name__ == '__main__':
    main()
