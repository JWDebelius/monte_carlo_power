
from __future__ import division

from unittest import TestCase, main

import numpy as np
import numpy.testing as npt

from emp_power.traditional import (calc_ttest_1,
                                   calc_ttest_ind,
                                   calc_pearson,
                                   calc_anova,
                                   _get_vitals
                                   )


class TraditionalPowerTest(TestCase):

    def setUp(self):
        self.counts = np.arange(5, 55, 5)
        self.x1 = np.array([8.510,  11.956,  12.614,   8.037,   7.032,  -4.054,
                            6.833,  10.379,   6.468,   5.120,   2.965,   0.703,
                            18.982,  8.226,   1.846,  11.849,   5.241,  11.916,
                            8.660,   6.600,   8.736,   6.211,   7.627,   3.138,
                            8.611])
        self.x2 = np.array([6.813,  -4.885,   4.714,   2.638,  -9.236,   8.857,
                            14.112, -5.521,   5.070,   8.812,   4.362,   1.025,
                            -6.977, -2.642,  -3.629,  -1.490,  -6.695,  -0.288,
                            0.565,   7.183,   0.829,  10.003,   4.842,   2.254,
                            13.568])
        self.x3 = np.array([4.860,  -0.100,  -0.422,  -2.604,  -4.160,   0.074,
                            5.109,   4.882,   3.517,  11.881,  -7.552,  -0.674,
                            2.048,   1.017,   3.978,  -3.119,  10.277,  -2.903,
                            4.618,  -4.941,   7.597,  -6.340,  -0.906,   3.008,
                            2.513])
        self.y1 = np.array([31.632, 27.830,  42.293,  20.808,  35.741,   6.197,
                            29.549, 31.029,   3.497,   6.446,  -3.054, -10.016,
                            25.008,  7.998,  -3.806,  23.126,   5.606,  27.181,
                            18.440, 16.105,  17.236,  16.886,  29.777,   9.859,
                            20.154])
        self.b1 = np.array([])

    def test_get_vitals(self):
        known_vitals = (7.3682399999999992, 4.455545203720864)
        self.assertEqual(_get_vitals(self.x1),
                         known_vitals)

    def test_contingency(self):
        pass

    def test_calc_ttest_1(self):
        known = np.array([0.388928, 0.787640, 0.941328, 0.985874, 0.996909,
                          0.999371, 0.999879, 1.000000, 1.000000, 1.000000])
        test = calc_ttest_1(self.x1, 3, self.counts)
        for k, t in zip(*(known, test)):
            npt.assert_approx_equal(k, t, 4)

    def test_calc_ttest_ind(self):
        known = np.array([0.263753, 0.524261, 0.714399, 0.837774, 0.91173,
                          0.953603, 0.976299, 0.988181, 0.994227, 0.997231])
        test = calc_ttest_ind(self.x1, self.x2, self.counts)
        for k, t in zip(*(known, test)):
            npt.assert_approx_equal(k, t, 4)

    def test_calc_anova(self):
        x1 = np.array([4.66090788,   8.32387767, 15.26801578,  7.28272918,
                       9.14823571,  13.90081104,  6.94812081,  0.87343727,
                       6.24484506,  12.06734819])
        x2 = np.array([05.96077091, 10.15055585, 13.87804038,  9.89452164,
                       02.92755353, 14.68067911,  9.22386107, 11.04836723,
                       13.29938893, 10.57197312])
        x3 = np.array([18.52338822, 19.55156514, 17.69189464, 20.97648344,
                       11.76546829, 18.60122748, 11.30508469, 13.79754774,
                       15.80490159, 12.69230230])
        known = np.array([0.801018, 0.8887184, 0.9402648, 0.9690108,
                          0.9843836])
        test = calc_anova(x1, x2, x3, counts=np.arange(5, 10))
        for k, t in zip(*(known, test)):
            npt.assert_approx_equal(k, t, 4)

    def test_calc_anova_counts_error(self):
        with self.assertRaises(ValueError):
            calc_anova(self.x1, self.x2, self.x3)

    def test_calc_pearson(self):
        known = np.array([0.308253, 0.733990, 0.915513, 0.976167, 0.993823,
                          0.998499, 0.999653, 0.999923, 1.000000, 1.000000])
        test = calc_pearson(self.x1, self.y1, self.counts)
        for k, t in zip(*(known, test)):
            npt.assert_approx_equal(k, t, 4)

if __name__ == '__main__':
    main()
