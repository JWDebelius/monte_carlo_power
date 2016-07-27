from unittest import TestCase, main

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt

from emp_power.simulate import (simulate_ttest_1,
                                simulate_ttest_ind,
                                simulate_anova,
                                simulate_correlation,
                                simulate_permanova,
                                simulate_mantel,
                                _convert_to_mirror,
                                _check_param,
                                _simulate_gauss_vec,
                                _vec_size,
                                )

class PowerSimulation(TestCase):

    def setUp(self):
        np.random.seed(5)
        self.length = 3
        self.mu_lim = [0, 2]
        self.sigma_lim = [1, 3]
        self.count_lim = [10, 11]
        self.dm = np.array([[0, 1, 2],
                            [1, 0, 3],
                            [2, 3, 0]])

    def test_ttest_1_simulate(self):
        known_mu = 1
        known_simga = 1
        known_n = 10
        known_dist = np.array([0.7060402, 1.37159057, 0.97669625, 1.84177767,
                               2.36758552, 1.57470769, -0.88576241, 1.17092537,
                               0.59690669, 2.63762849])

        [mu, sigma, n], [dist] = simulate_ttest_1(self.mu_lim,
                                                  self.sigma_lim,
                                                  self.count_lim)

        self.assertEqual(mu, known_mu)
        self.assertEqual(sigma, known_simga)
        self.assertEqual(n, known_n)
        npt.assert_almost_equal(known_dist, dist, 5)

    def test_ttest_ind_simulate(self):
        kparams = [1, 0, 2, 2, 10]
        ksamples = [np.array([5.86154237, 0.49581574, 1.21921968, 4.16496223,
                             -0.81846481, -0.18327332, 1.37520645, 0.34026008,
                             -1.38552922, 0.59024698]),
                    np.array([-0.71765789, 1.20694321, -3.32957706,
                              -1.40035808, 2.30278202, 3.71466201, -3.02235912,
                              1.28969502, -1.96121577, -1.71370631])]
        params, samples = simulate_ttest_ind(self.mu_lim, self.sigma_lim,
                                             self.count_lim)
        self.assertEqual(kparams, params)
        for ks, s in zip(*(ksamples, samples)):
            npt.assert_almost_equal(ks, s, 5)

    def test_anova_simulate(self):
        kmu = np.array([1, 0, 1])
        ksigma = 2
        kn = 10
        ksamples = [np.array([5.86154237,  0.49581574,  1.21921968,
                              4.16496223, -0.81846481, -0.18327332,
                              1.37520645,  0.34026008, -1.38552922,
                              0.59024698]),
                    np.array([-0.71765789,  1.20694321, -3.32957706,
                              -1.40035808,  2.30278202, 3.71466201,
                              -3.02235912,  1.28969502, -1.96121577,
                              -1.71370631]),
                    np.array([-0.74375837,  0.15498414,  2.99287965,
                              2.42484254,  1.11828849, 0.27337824,
                              1.00657769,  0.78813912,  2.58610664,
                              -0.26314326])]
        [mu, sigma, n], samples = simulate_anova(self.mu_lim, self.sigma_lim,
                                                 self.count_lim, 3)

        npt.assert_array_equal(kmu, mu)
        self.assertEqual(ksigma, sigma)
        self.assertEqual(kn, n)
        for ks, s in zip(*(ksamples, samples)):
            npt.assert_almost_equal(ks, s, 5)

    def test_simulate_correlation(self):
        known_sigma = 2
        known_n = 10
        known_m = 1
        known_b = 1
        known_x = np.array([06.626557, -2.725262,  9.588900, -8.203579,
                            -2.065268, -2.917239, -0.267240,  9.816419,
                            06.165634,  2.989178])
        known_y = np.array([06.51051, -0.34040,  9.77181, -6.64841,
                            -1.66502, -2.38306,  0.86463, 12.92136,
                            10.66871,  3.87037])
        [sigma, n, m, b], [x, y] = simulate_correlation(
            slope_lim=self.sigma_lim,
            sigma_lim=self.sigma_lim,
            count_lim=self.count_lim,
            intercept_lim=self.mu_lim,
            x_lim=[-10, 10],
            )
        self.assertEqual(sigma, known_sigma)
        self.assertEqual(n, known_n)
        self.assertEqual(m, known_m)
        self.assertEqual(b, known_b)

        npt.assert_almost_equal(known_x, x, 5)
        npt.assert_almost_equal(known_y, y, 5)

    def test_simulate_permanova(self):
        known_grouping = pd.Series([0, 0, 1, 1], index=dm_ids, name='groups')

        params, [dm, grouping] = simulate_permanova(num_samples=4,
                                                    num0=2,
                                                    wdist=0.2,
                                                    wspread=0.1,
                                                    bdist=0.5,
                                                    bspread=0.1
                                                    )
        npt.assert_almost_equal(permanova_dm, dm.data.astype(float), 5)
        self.assertEqual(dm_ids, dm.ids)
        pdt.assert_series_equal(known_grouping, grouping)

    def test_simulate_permanova_no_num0(self):
        known_grouping = pd.Series([0, 0, 1, 1], index=dm_ids, name='groups')

        params, [dm, grouping] = simulate_permanova(num_samples=4,
                                                    num0=None,
                                                    wdist=0.2,
                                                    wspread=0.1,
                                                    bdist=0.5,
                                                    bspread=0.1
                                                    )
        self.assertEqual(dm_ids, dm.ids)
        self.assertEqual((4, 4), dm.shape)
        pdt.assert_series_equal(known_grouping, grouping)

    def test_simulate_mantel(self):
        params, [x, y] = simulate_mantel(slope_lim=self.sigma_lim,
                                         sigma_lim=self.sigma_lim,
                                         count_lim=[4, 5],
                                         intercept_lim=self.mu_lim,
                                         x_lim=[-10, 10]
                                         )
        npt.assert_almost_equal(mantel_x, x.data)
        self.assertEqual(dm_ids, x.ids)
        npt.assert_almost_equal(mantel_y, y.data)
        self.assertEqual(dm_ids, x.ids)

    def test_convert_to_mirror(self):
        vec = np.arange(0, ((self.length) * (self.length - 1))/2) + 1
        test_dm = _convert_to_mirror(self.length, vec)
        npt.assert_array_equal(test_dm, self.dm)

    def test_check_param_list(self):
        param = [0, 1]
        new_param = _check_param(param, 'param')
        self.assertTrue(isinstance(new_param, float))
        self.assertTrue(0 < new_param < 1)

    def test_check_param_float(self):
        param = 0.5
        new_param = _check_param(param, 'param')
        self.assertEqual(param, new_param)

    def test_check_param_error(self):
        param = 'param'
        with self.assertRaises(ValueError):
            _check_param(param, 'param')

    def test_simulate_gauss_vec_less(self):
        vec = _simulate_gauss_vec(0, 1, 3)
        self.assertEqual(vec.shape, (3, ))
        self.assertFalse((vec < 0).any())
        self.assertFalse((vec > 1).any())

    def test_vec_size(self):
        test_shape = _vec_size(5)
        self.assertEqual(test_shape, 10)

permanova_dm = np.array([[0.00000000, 0.24412275, 0.74307712, 0.47479079],
                         [0.24412275, 0.00000000, 0.51096098, 0.65824811],
                         [0.74307712, 0.51096098, 0.00000000, 0.16691298],
                         [0.47479079, 0.65824811, 0.16691298, 0.00000000]])

mantel_x = np.array([[00.0000000,   9.3518188,   2.9623433,  14.8301359],
                     [09.3518188,   0.0000000,  12.3141620,   5.4783172],
                     [02.9623433,  12.3141620,   0.0000000,  17.7924792],
                     [14.8301359,   5.4783172,  17.7924792,   0.0000000]])

mantel_y = np.array([[00.0000000,   8.3847378,   6.8125418,  11.5334174],
                     [08.3847378,   0.0000000,  15.1972795,   3.1486796],
                     [06.8125418,  15.1972795,   0.0000000,  18.3459592],
                     [11.5334174,   3.1486796,  18.3459592,   0.0000000]])
dm_ids = ('s.1', 's.2', 's.3', 's.4')


if __name__ == '__main__':
    main()
