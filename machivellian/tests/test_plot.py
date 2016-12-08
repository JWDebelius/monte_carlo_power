from unittest import TestCase, main

import numpy as np
import numpy.testing as npt

from machivellian.plot import (_set_ticks,
                               _get_symetrical,
                               _summarize_t,
                               _get_colors,
                               _summarize_trace,
                               plot_power_curve,
                               _get_effect_interval
                               )


class PlotTest(TestCase):

    def setUp(self):
        self.counts = np.arange(5, 50, 10)
        self.l_ = np.array([0.5, 0.4, 0.6])
        self.k_mean = np.array([0.299159,  0.614718,  0.803765,  0.905440,
                                0.956298])

    def test_plot_power_curve_error(self):
        with self.assertRaises(ValueError):
            plot_power_curve(1, np.ones(5))

    def test_set_ticks(self):
        known = np.arange(0, 1.1, 0.25)
        test = _set_ticks([0, 1], 4)
        npt.assert_array_equal(known, test)

    def test_get_symetrical(self):
        known = [-5, 5]
        self.assertEqual(known, _get_symetrical([-2, 5]))
        self.assertEqual(known, _get_symetrical(known))

    def test_summarize_t(self):
        alpha = 0.05
        noncentrality = 1
        df = 4

        known_x = np.array(
            [-7.5, -7.4, -7.3, -7.2, -7.1, -7.0, -6.9, -6.8, -6.7, -6.6, -6.5,
             -6.4, -6.3, -6.2, -6.1, -6.0, -5.9, -5.8, -5.7, -5.6, -5.5, -5.4,
             -5.3, -5.2, -5.1, -5.0, -4.9, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3,
             -4.2, -4.1, -4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2,
             -3.1, -3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1,
             -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0,
             -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.0,  0.1,
             00.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0,  1.1,  1.2,
             01.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.0,  2.1,  2.2,  2.3,
             02.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.0,  3.1,  3.2,  3.3,  3.4,
             03.5,  3.6,  3.7,  3.8,  3.9,  4.0,  4.1,  4.2,  4.3,  4.4,  4.5,
             04.6,  4.7,  4.8,  4.9,  5.0,  5.1,  5.2,  5.3,  5.4,  5.5,  5.6,
             05.7,  5.8,  5.9,  6.0,  6.1,  6.2,  6.3,  6.4,  6.5,  6.6,  6.7,
             06.8,  6.9,  7.0,  7.1,  7.2,  7.3,  7.4,  7.5]
             )
        known_y1 = np.array(
         [0.000,  0.000,  0.000,  0.001,  0.001,  0.001,  0.001,  0.001,
          0.001,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001,
          0.001,  0.001,  0.001,  0.002,  0.002,  0.002,  0.002,  0.002,
          0.002,  0.003,  0.003,  0.003,  0.003,  0.004,  0.004,  0.005,
          0.005,  0.006,  0.006,  0.007,  0.007,  0.008,  0.009,  0.010,
          0.011,  0.013,  0.014,  0.016,  0.018,  0.020,  0.022,  0.025,
          0.028,  0.032,  0.036,  0.040,  0.046,  0.052,  0.059,  0.066,
          0.075,  0.085,  0.096,  0.109,  0.123,  0.138,  0.155,  0.174,
          0.194,  0.215,  0.236,  0.259,  0.281,  0.302,  0.322,  0.340,
          0.355,  0.366,  0.373,  0.375,  0.373,  0.366,  0.355,  0.340,
          0.322,  0.302,  0.281,  0.259,  0.236,  0.215,  0.194,  0.174,
          0.155,  0.138,  0.123,  0.109,  0.096,  0.085,  0.075,  0.066,
          0.059,  0.052,  0.046,  0.040,  0.036,  0.032,  0.028,  0.025,
          0.022,  0.020,  0.018,  0.016,  0.014,  0.013,  0.011,  0.010,
          0.009,  0.008,  0.007,  0.007,  0.006,  0.006,  0.005,  0.005,
          0.004,  0.004,  0.003,  0.003,  0.003,  0.003,  0.002,  0.002,
          0.002,  0.002,  0.002,  0.002,  0.001,  0.001,  0.001,  0.001,
          0.001,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001,
          0.001,  0.001,  0.001,  0.001,  0.000,  0.000,  0.000])
        known_y2 = np.array(
         [0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,
          0.000,  0.000,  0.000,  0.000,  0.000,  0.001,  0.001,  0.001,
          0.001,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001,
          0.001,  0.001,  0.001,  0.001,  0.001,  0.002,  0.002,  0.002,
          0.002,  0.002,  0.002,  0.003,  0.003,  0.003,  0.003,  0.004,
          0.004,  0.005,  0.005,  0.006,  0.006,  0.007,  0.007,  0.008,
          0.009,  0.010,  0.011,  0.013,  0.014,  0.016,  0.018,  0.020,
          0.022,  0.025,  0.028,  0.032,  0.036,  0.040,  0.046,  0.052,
          0.059,  0.066,  0.075,  0.085,  0.096,  0.109,  0.123,  0.138,
          0.155,  0.174,  0.194,  0.215,  0.236,  0.259,  0.281,  0.302,
          0.322,  0.340,  0.355,  0.366,  0.373,  0.375,  0.373,  0.366,
          0.355,  0.340,  0.322,  0.302,  0.281,  0.259,  0.236,  0.215,
          0.194,  0.174,  0.155,  0.138,  0.123,  0.109,  0.096,  0.085,
          0.075,  0.066,  0.059,  0.052,  0.046,  0.040,  0.036,  0.032,
          0.028,  0.025,  0.022,  0.020,  0.018,  0.016,  0.014,  0.013,
          0.011,  0.010,  0.009,  0.008,  0.007,  0.007,  0.006,  0.006,
          0.005,  0.005,  0.004,  0.004,  0.003,  0.003,  0.003,  0.003,
          0.002,  0.002,  0.002,  0.002,  0.002,  0.002,  0.001,  0.001,
          0.001,  0.001,  0.001,  0.001,  0.001,  0.001,  0.001])
        known_crit = 2.7764451051977987

        x, y1, y2, crit = _summarize_t(noncentrality, df, alpha)

        npt.assert_almost_equal(x, known_x, 3)
        npt.assert_almost_equal(y1, known_y1, 3)
        npt.assert_almost_equal(y2, known_y2, 3)
        npt.assert_almost_equal(crit, known_crit, 3)

    def test_get_colors(self):
        known_color1 = [0.15, 0.15, 0.15]
        known_color2 = np.array([0.76863,  0.30588,  0.32157])

        color1, color2 = _get_colors()

        npt.assert_array_equal(color1, known_color1)
        npt.assert_array_equal(color2, known_color2)

    def test_summarize_trace_trace1d(self):
        trace = np.ones(5)

        known_mean = np.ones(5)

        mean, lo, hi = _summarize_trace(trace)

        npt.assert_almost_equal(known_mean, mean, 5)
        self.assertTrue(np.isnan(lo).all())
        self.assertTrue(np.isnan(hi).all())

    def test_summarize_trace_trace_std(self):
        trace = (np.atleast_2d(np.arange(1, 4)) * np.ones((5, 1))).T

        known_mean = np.array([2.,  2.,  2.,  2.,  2.])
        known_lo = np.array([1.1835,  1.1835,  1.1835,  1.1835,  1.1835])
        known_hi = np.array([2.8165,  2.8165,  2.8165,  2.8165,  2.8165])

        mean, lo, hi = _summarize_trace(trace)

        npt.assert_almost_equal(known_mean, mean, 5)
        npt.assert_almost_equal(known_lo, lo, 5)
        npt.assert_almost_equal(known_hi, hi, 5)

    def test_summarize_trace_trace_ci(self):
        trace = (np.atleast_2d(np.arange(1, 4)) * np.ones((5, 1))).T

        known_mean = np.array([2.,  2.,  2.,  2.,  2.])
        known_lo = np.array([-1.04243, -1.04243, -1.04243, -1.04243, -1.04243])
        known_hi = np.array([5.04243,  5.04243,  5.04243,  5.04243,  5.04243])

        mean, lo, hi = _summarize_trace(trace, ci_alpha=0.05)

        npt.assert_almost_equal(known_mean, mean, 5)
        npt.assert_almost_equal(known_lo, lo, 5)
        npt.assert_almost_equal(known_hi, hi, 5)

    def test_get_effect_interval_single(self):
        mean, low, hi = _get_effect_interval(self.counts, self.l_[0])

        npt.assert_almost_equal(mean, self.k_mean, 5)
        self.assertTrue(np.isnan(low).all())
        self.assertTrue(np.isnan(hi).all())

    def test_get_effect_interval_no_ci(self):
        known_lo = np.array([0.239040,  0.490191,  0.672526,  0.796770,
                             0.877286])
        known_hi = np.array([0.365331,  0.728362,  0.896776,  0.963771,
                             0.987995])
        mean, lo, hi = _get_effect_interval(self.counts, self.l_)
        npt.assert_almost_equal(mean, self.k_mean, 5)
        npt.assert_almost_equal(lo, known_lo, 5)
        npt.assert_almost_equal(hi, known_hi, 5)

    def test_get_effect_interval_ci(self):
        known_lo = np.array([0.113691,  0.187622,  0.252683,  0.313220,
                             0.370066])
        known_hi = np.array([0.560994,  0.929215,  0.991258,  0.999074,
                             0.999912])

        mean, lo, hi = _get_effect_interval(self.counts, self.l_,
                                            ci_alpha=0.05)
        npt.assert_almost_equal(mean, self.k_mean, 5)
        npt.assert_almost_equal(lo, known_lo, 5)
        npt.assert_almost_equal(hi, known_hi, 5)


if __name__ == '__main__':
    main()
