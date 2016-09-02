from unittest import TestCase, main

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt

from machivellian.simulate import (simulate_ttest_1,
                                   simulate_ttest_ind,
                                   simulate_anova,
                                   simulate_correlation,
                                   simulate_permanova,
                                   simulate_mantel,
                                   simulate_discrete,
                                   simulate_lognormal,
                                   simulate_uniform,
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

    def test_simulate_lognormal(self):
        (k_m1, k_m2) = (0.44398634217947897, 1.7414646123547528)
        (k_s1, k_s2) = (1.4134383106788528, 2.8372218158758429)
        k_n = 10
        k_v1 = np.array([48.41121309,   1.09162746,   1.82013940,  14.59568269,
                         00.43121108,   0.67553144,   2.03226702,   0.97798153,
                         00.28883120,   1.16696462])
        k_v2 = np.array([2.06140398e+00,   3.16154969e+01,   5.06959262e-02,
                         7.82627345e-01,   1.49637590e+02,   1.10890095e+03,
                         7.83877984e-02,   3.55535510e+01,   3.53194115e-01,
                         5.01767782e-01])

        [(m1, m2), (s1, s2), n], [v1, v2] = simulate_lognormal(self.mu_lim,
                                                               self.sigma_lim,
                                                               10)

        npt.assert_almost_equal(k_m1, m1, 7)
        npt.assert_almost_equal(k_m2, m2, 7)
        npt.assert_almost_equal(k_s1, s1, 7)
        npt.assert_almost_equal(k_s2, s2, 7)
        self.assertEqual(k_n, n)
        npt.assert_almost_equal(k_v1, v1, 5)
        npt.assert_almost_equal(k_v2, v2, 5)

    def test_simulate_unifrom(self):
        # Sets up known values
        known_r = 0.44398634217947897
        known_d = 2.741464612354753
        known_n = 10
        known_v1 = np.array([0.09178048,  0.40785070,  0.21684790,  0.27160592,
                             0.34005263,  0.23017051,  0.13177537,  0.08334566,
                             0.03584802,  0.32785741])
        known_v2 = np.array([2.93739988,  2.81175203,  3.13214464,  2.86315526,
                             2.92537930,  2.87292006,  3.02063786,  2.99890468,
                             3.00782498,  2.85948467])
        # Simulates the data
        [r, d, n], [v1, v2] = simulate_uniform(self.mu_lim,
                                               self.sigma_lim,
                                               10)
        # Tests the results
        npt.assert_almost_equal(known_r, r, 7)
        npt.assert_almost_equal(known_d, d, 7)
        self.assertEqual(known_n, n)
        npt.assert_almost_equal(known_v1, v1, 5)
        npt.assert_almost_equal(known_v2, v2, 5)

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

    def test_simulate_discrete(self):
        p_lim = 0.5
        size_lim = 5
        num_groups = 2
        known = pd.DataFrame(
            np.array([[0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).T.astype(float),
            columns=['outcome', 'group', 'dummy'],
            index=['s.%i' % i for i in range(10)]
            )
        [tp_values, tsize, tnum_groups], test = \
            simulate_discrete(p_lim, size_lim, num_groups)
        pdt.assert_frame_equal(known, test)
        self.assertEqual(tp_values, [p_lim] * 2)
        self.assertEqual(tsize, size_lim)
        self.assertEqual(tnum_groups, num_groups)

    def test_convert_to_mirror(self):
        vec = np.arange(0, ((self.length) * (self.length - 1))/2) + 1
        test_dm = _convert_to_mirror(self.length, vec)
        npt.assert_array_equal(test_dm, self.dm)

    def test_check_param_list(self):
        param = [0, 1]
        new_param = _check_param(param, 'param')
        self.assertTrue(isinstance(new_param, np.ndarray))
        self.assertTrue(0 < new_param < 1)

    def test_check_param_float(self):
        param = 0.5
        new_param = _check_param(param, 'param')
        self.assertEqual(param, new_param)

    def test_check_param_error(self):
        param = 'param'
        with self.assertRaises(TypeError):
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
