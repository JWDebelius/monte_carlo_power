from unittest import TestCase, main

import numpy as np
import numpy.testing as npt
# import pandas as pd
# import pandas.util.testing as pdt
import skbio

from emp_power.utils.simulate import (ttest_1_simulate,
                                      ttest_ind_simulate,
                                      simulate_distance_matrix,
                                      convert_to_mirror,
                                      _check_param,
                                      _simulate_gauss_vec,
                                      _vec_size,
                                      # build_groups,
                                      # generate_vector,
                                      # generate_diff_vector,
                                      # simulate_table,
                                      )


class PowerSimulation(TestCase):

    def setUp(self):
        np.random.seed(5)
        self.length = 3
        self.dm = np.array([[0, 1, 2],
                            [1, 0, 3],
                            [2, 3, 0]])
        self.mu_lim = [0, 2]
        self.sigma_lim = [1, 3]
        self.count_lim = [10, 11]

    def test_ttest_1_simulate(self):
        known_mu = 1
        known_simga = 1
        known_n = 10
        known_dist = np.array([0.7060402, 1.37159057, 0.97669625, 1.84177767,
                               2.36758552, 1.57470769, -0.88576241, 1.17092537,
                               0.59690669, 2.63762849])

        [mu, sigma, n], [dist] = ttest_1_simulate(self.mu_lim,
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
        params, samples = ttest_ind_simulate(self.mu_lim, self.sigma_lim,
                                             self.count_lim)
        self.assertEqual(kparams, params)
        for ks, s in zip(*(ksamples, samples)):
            npt.assert_almost_equal(ks, s, 5)

    def test_simulate_distance_matrix(self):
        num_samples = 4
        num0 = 2
        wdist = 0.2
        wspread = 0.1
        bdist = 0.5
        bspread = 0.1
        known_ids = np.array(['s.%i' % (i + 1) for i in np.arange(4)])

        dm, grouping = simulate_distance_matrix(num_samples=num_samples,
                                                num0=num0,
                                                wdist=wdist,
                                                wspread=wspread,
                                                bdist=bdist,
                                                bspread=bspread
                                                )

        npt.assert_array_equal(known_ids, dm.ids)
        self.assertTrue(isinstance(dm, skbio.DistanceMatrix))
        self.assertTrue(dm.shape, (4, 4))
        npt.assert_array_equal(known_ids, grouping.index.values)
        npt.assert_array_equal(np.array([0, 0, 1, 1]),
                               grouping.values)

    def test_convert_to_mirror(self):
        vec = np.arange(0, ((self.length) * (self.length - 1))/2) + 1
        test_dm = convert_to_mirror(self.length, vec)
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

    # def test_build_groups(self):
    #     known_groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    #     test_groups = build_groups(4, 2)
    #     npt.assert_array_equal(known_groups, test_groups)

    # def test_generate_vector(self):
    #     known_params = {'offset': 0.1,
    #                     'num_groups': 4,
    #                     'obs_per_group': 2}
    #     known_vec = np.array([0.121993, 0.770732, 0.106719,
    #                           0.818611, 0.388411, 0.511744,
    #                           0.665908, 0.418418])
    #     test_vec, test_params = generate_vector(4, 2, 0.1)

    #     npt.assert_almost_equal(known_vec, test_vec, 5)
    #     self.assertEqual(known_params.keys(), test_params.keys())
    #     self.assertEqual(known_params['offset'],
    #                      test_params['offset'])
    #     self.assertEqual(known_params['num_groups'],
    #                      test_params['num_groups'])
    #     self.assertEqual(known_params['obs_per_group'],
    #                      test_params['obs_per_group'])

    # def test_generate_diff_vec(self):
    #     known_vec = np.array([3.18891871, 4.04766352, 14.83709634,
    #                           15.67954637, 11.12877624, 10.13964337,
    #                           6.64064811, 5.55860975])
    #     known_params = {'p-value': 0.00023550431122179829,
    #                     'mus': np.array([5, 9, 9, 6]),
    #                     'scale': 0.75,
    #                     'obs_per_group': 2,
    #                     'offset': 0.1,
    #                     'sigma': 3,
    #                     'num_groups': 4}
    #     test_vec, test_params = generate_diff_vector(4, 2, mu_lim=[5, 10],
    #                                                  sigma_lim=[2, 4])
    #     npt.assert_almost_equal(known_vec, test_vec, 5)
    #     self.assertEqual(sorted(known_params.keys()),
    #                      sorted(test_params.keys()))
    #     self.assertEqual(known_params['offset'],
    #                      test_params['offset'])
    #     self.assertEqual(known_params['num_groups'],
    #                      test_params['num_groups'])
    #     self.assertEqual(known_params['obs_per_group'],
    #                      test_params['obs_per_group'])
    #     self.assertEqual(known_params['scale'], test_params['scale'])
    #     self.assertEqual(known_params['p-value'], test_params['p-value'])
    #     self.assertEqual(known_params['sigma'], test_params['sigma'])
    #     npt.assert_array_equal(known_params['mus'], test_params['mus'])

    # def test_simulate_table(self):
    #     known_closed = pd.DataFrame(
    #         data=np.array([[0.000601, 0.999399],
    #                        [0.000243, 0.999757],
    #                        [0.000190, 0.999810],
    #                        [0.000153, 0.999847]]),
    #         index=['s.1', 's.2', 's.3', 's.4'],
    #         columns=['f.1', 'f.2']
    #         )
    #     known_groups = pd.Series([0.0, 0.0, 1.0, 1.0],
    #                              index=['s.1', 's.2', 's.3', 's.4'],
    #                              name='grouping')
    #     known_params = [{'scale': 0.75, 'mus': np.array([2, 0]),
    #                      'p-value': np.nan, 'obs_per_group': 2,
    #                      'offset': 0.1, 'sigma': 5, 'num_groups': 2},
    #                     {'obs_per_group': 2, 'num_groups': 2, 'offset': 0.1}]
    #     closed, groups, params = simulate_table(2, 2,
    #                                             num_features=2,
    #                                             num_sig=1)
    #     # self.assertTrue(closed.equals(known_closed))
    #     pdt.assert_index_equal(known_closed.index, closed.index)
    #     pdt.assert_index_equal(known_closed.columns, closed.columns)
    #     npt.assert_almost_equal(known_closed.values, closed.values, 5)
    #     pdt.assert_series_equal(known_groups, groups)
    #     self.assertEqual(len(known_params), len(params))

if __name__ == '__main__':
    main()
