from unittest import TestCase, main

import numpy as np
import numpy.testing as npt

from absloute_power.utils import (extrapolate_f,
                                  linear_confidence,
                                  z_effect,
                                  z_power,
                                  cohen_d_one_sample)


class UtilTest(TestCase):
    def setUp(self):
        self.trad_power = np.array([0.09408713, 0.17053126, 0.24337034,
                                    0.31399518, 0.38179451, 0.44605756,
                                    0.50624597, 0.56202401, 0.61323563,
                                    0.65987019, 0.70202859])
        self.extr_power = np.array([0.10042458, 0.17905605, 0.23205983,
                                    0.31053907, 0.36133113, 0.43416739,
                                    0.48006051, 0.54443074, 0.58421407,
                                    0.63910410, 0.67253182])
        self.counts = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        self.effect = 0.46320073720216759

    def test_extrapolate_f(self):
        cnts = np.array([10, 20, 30, 40])
        pwr = np.array([[0.19, 0.41, 0.48, 0.64],
                        [0.17, 0.34, 0.43, 0.63],
                        [0.14, 0.33, 0.52, 0.65]])
        counts = np.arange(25, 101, 25)
        known = np.array([0.418684,  0.718118,  0.878858,  0.952119])
        test = extrapolate_f(counts, pwr, cnts, 0.05)
        npt.assert_array_almost_equal(known, test, 6)

    def test_linear_confidence(self):
        x = np.array([4.0, 2.5, 3.2, 5.8, 7.4, 4.4, 8.3, 8.5])
        y = np.array([2.1, 4.0, 1.5, 6.3, 5.0, 5.8, 8.1, 7.1])
        knowx = np.array([1.9, 2.5, 3.1, 3.7, 4.3, 4.9, 5.5, 6.1, 6.7, 7.3,
                          7.9, 8.5,  9.1])
        knowy = np.array([2.16729457, 2.63570239, 3.10411022, 3.57251804,
                          4.04092586, 4.50933368, 4.97774150, 5.44614933,
                          5.91455715, 6.38296497, 6.85137279,  7.31978061,
                          7.78818844])
        knowc = np.array([0.34275587, 0.30215816, 0.26431028, 0.23057039,
                          0.20299726, 0.18437863, 0.17755415, 0.18384199,
                          0.20202150, 0.22928128, 0.26281082,  0.30051881,
                          0.34102198])
        testx, testy, testc = linear_confidence(x, y)
        npt.assert_array_almost_equal(knowx, testx)
        npt.assert_array_almost_equal(knowy, testy)
        npt.assert_array_almost_equal(knowc, testc)

    def test_z_effect(self):
        test = z_effect(self.counts, self.trad_power)
        npt.assert_almost_equal(self.effect, test.mean(), 5)

    def test_z_power(self):
        test = z_power(self.counts, self.effect)
        npt.assert_array_almost_equal(test, self.extr_power)

    def test_cohen_d_one_sample(self):
        test = cohen_d_one_sample(self.counts, 0)
        known = 1.8973665961010275
        self.assertEqual(test, known)


if __name__ == '__main__':
    main()
