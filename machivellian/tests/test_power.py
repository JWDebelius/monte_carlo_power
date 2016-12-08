# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main

import numpy as np
import numpy.testing as npt
from scipy.stats import kruskal

from machivellian.power import (subsample_power,
                                confidence_bound,
                                _calculate_power,
                                _compare_distributions,
                                _check_subsample_power_inputs,
                                )


class PowerAnalysisTest(TestCase):

    def setUp(self):

        def f(x):
            """returns the p value of a kruskal wallis test"""
            return kruskal(*x)[1]

        self.f = f
        self.num_p = 1

        # Sets the random seed
        np.random.seed(5)
        # Sets up the distributions of data for use
        self.s1 = np.arange(0, 10, 1)
        # Sets up two distributions which will never be equal by a rank-sum
        # test.
        self.samps = [np.ones((10))/10., np.ones((10))]
        self.pop = [np.arange(0, 10, 0.1), np.arange(0, 20, 0.2)]
        self.counts = np.array([5, 15, 25, 35, 45])
        # Sets up a vector of alpha values
        self.alpha = np.power(10, np.array([-1, -1.301, -2, -3])).round(3)

    def test_subsample_power_defaults(self):
        test_p = subsample_power(self.f, self.pop, self.counts,
                                 num_iter=10, num_runs=5)
        self.assertEqual(test_p.shape, (5, 5))

    def test_subsample_power_matches(self):
        test_p = subsample_power(self.f,
                                 samples=self.pop,
                                 counts=self.counts,
                                 num_iter=10,
                                 num_runs=5,
                                 draw_mode="matched")
        self.assertEqual(test_p.shape, (5, 5))

    def test_subsample_power_multi_p(self):
        test_p = subsample_power(lambda x: np.array([0.5, 0.5]),
                                 samples=self.pop,
                                 counts=self.counts,
                                 num_iter=10,
                                 num_runs=5)
        self.assertEqual(test_p.shape, (5, 5, 2))

    def test_subsample_power_kwargs(self):
        def test(x, b=True):
            if b:
                return self.f(x)
            else:
                return np.array([self.f(x)] * 2)

        test_p_bt = subsample_power(test,
                                    samples=self.pop,
                                    counts=self.counts,
                                    num_iter=10,
                                    num_runs=5,
                                    test_kwargs={'b': True})
        test_p_bf = subsample_power(test,
                                    samples=self.pop,
                                    counts=self.counts,
                                    num_iter=10,
                                    num_runs=5,
                                    test_kwargs={'b': False})
        self.assertEqual(test_p_bt.shape, (5, 5))
        self.assertEqual(test_p_bf.shape, (5, 5, 2))

    def test_confidence_bound_default(self):
        # Sets the know confidence bound
        known = 2.2830070
        test = confidence_bound(self.s1)
        npt.assert_almost_equal(test, known, 3)

    def test_confidence_bound_df(self):
        known = 2.15109
        test = confidence_bound(self.s1, df=15)
        npt.assert_almost_equal(known, test, 3)

    def test_confidence_bound_alpha(self):
        known = 3.2797886
        test = confidence_bound(self.s1, alpha=0.01)
        npt.assert_almost_equal(known, test, 3)

    def test_confidence_bound_nan(self):
        # Sets the value to test
        samples = np.array([[4, 3.2, 3.05],
                            [2, 2.8, 2.95],
                            [5, 2.9, 3.07],
                            [1, 3.1, 2.93],
                            [3, np.nan, 3.00]])
        # Sets the know value
        known = np.array([2.2284, 0.2573, 0.08573])
        # Tests the function
        test = confidence_bound(samples, axis=0)
        npt.assert_almost_equal(known, test, 3)

    def test_confidence_bound_axis_none(self):
        # Sets the value to test
        samples = np.array([[4, 3.2, 3.05],
                            [2, 2.8, 2.95],
                            [5, 2.9, 3.07],
                            [1, 3.1, 2.93],
                            [3, np.nan, 3.00]])
        # Sest the known value
        known = 0.52852
        # Tests the output
        test = confidence_bound(samples, axis=None)
        npt.assert_almost_equal(known, test, 3)

    def test_calculate_power_numeric(self):
        # Sets up the values to test
        crit = 0.025
        # Sets the known value
        known = 0.5
        # Calculates the test value
        test = _calculate_power(p_values=self.alpha,
                                alpha=crit,
                                numeric=True)
        # Checks the test value
        npt.assert_almost_equal(known, test)

    def test_calculate_power_reject(self):
        crit = 0.025
        reject = self.alpha < crit
        known = 0.5
        test = _calculate_power(p_values=reject, alpha=crit, numeric=False)
        npt.assert_almost_equal(known, test)

    def test_calculate_power_n(self):
        crit = 0.025
        known = np.array([0.5, 0.5])
        alpha = np.vstack((self.alpha, self.alpha))
        test = _calculate_power(alpha, crit)
        npt.assert_almost_equal(known, test)

    def test_compare_distributions_all_mode(self):
        known = np.ones((100))*0.0026998
        test = _compare_distributions(self.f, self.samps, 1, num_iter=100)
        npt.assert_allclose(known, test, 5)

    def test_compare_distributions_matched_mode(self):
        # Sets the known value
        known_mean = 0.162195
        known_std = 0.121887
        known_shape = (100,)
        # Tests the sample value
        test = _compare_distributions(self.f, self.pop, self.num_p,
                                      mode='matched', num_iter=100,
                                      bootstrap=False)
        npt.assert_allclose(known_mean, test.mean(), rtol=0.1, atol=0.02)
        npt.assert_allclose(known_std, test.std(), rtol=0.1, atol=0.02)
        self.assertEqual(known_shape, test.shape)

    def test_compare_distributions_multiple_returns(self):
        known = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

        def f(x):
            return np.array([1, 2, 3])

        test = _compare_distributions(f, self.pop, 3, mode='matched',
                                      num_iter=3, bootstrap=False)
        npt.assert_array_equal(known, test)

    def test_compare_distributions_bootstrap_more(self):
        known = np.array([-76.10736642, -82.08492357, -74.45798197,
                          -72.0498448, -82.54530595])
        test = _compare_distributions(self.f, self.pop, self.num_p,
                                      counts=1000,
                                      num_iter=5)
        npt.assert_almost_equal(known, np.log10(test), 5)

    def test_check_subsample_power_inputs_draw_mode_error(self):
        with self.assertRaises(ValueError):
            _check_subsample_power_inputs(test=self.f,
                                          samples=[np.ones((2)), np.ones((5))],
                                          counts=self.counts,
                                          draw_mode="Alice Price Healy")

    def test_check_subsample_power_inputs_matched_mode(self):
        with self.assertRaises(ValueError):
            _check_subsample_power_inputs(test=self.f,
                                          samples=[np.ones((2)), np.ones((5))],
                                          counts=self.counts,
                                          draw_mode="matched")

    def test_check_subsample_power_inputs_low_counts(self):
        with self.assertRaises(ValueError):
            _check_subsample_power_inputs(test=self.f,
                                          samples=self.samps,
                                          counts=np.arange(-5, 0)
                                          )

    def test_check_subsample_power_inputs_bootstrap_counts(self):
        with self.assertRaises(ValueError):
            _check_subsample_power_inputs(test=self.f,
                                          samples=[np.ones((3)), np.ones((5))],
                                          counts=self.counts,
                                          bootstrap=False)

    def test_check_subsample_power_inputs_ratio(self):
        with self.assertRaises(ValueError):
            _check_subsample_power_inputs(test=self.f,
                                          samples=self.samps,
                                          counts=self.counts,
                                          ratio=np.array([1, 2, 3]))

    def test_check_subsample_power_inputs_test(self):
        # Defines a test function
        def test(x):
            return 'Hello World!'
        with self.assertRaises(TypeError):
            _check_subsample_power_inputs(test=test,
                                          samples=self.samps,
                                          counts=self.counts)

    def test_check_subsample_power_inputs_bootstrap_error(self):
        with self.assertRaises(ValueError):
            _check_subsample_power_inputs(test=self.f,
                                          samples=self.samps,
                                          counts=np.arange(10, 1000, 10),
                                          bootstrap=False)

    def test_check_sample_power_inputs(self):
        # Defines the know returns
        known_num_p = 1
        known_ratio = np.ones((2))
        # Runs the code for the returns
        test_ratio, test_num_p = \
            _check_subsample_power_inputs(test=self.f,
                                          samples=self.samps,
                                          counts=self.counts,
                                          )
        # Checks the returns are sane
        self.assertEqual(known_num_p, test_num_p)
        npt.assert_array_equal(known_ratio, test_ratio)

if __name__ == '__main__':
    main()
