from __future__ import division

from unittest import TestCase, main

import numpy as np
import numpy.testing as npt
import pandas as pd
import skbio
import scipy

from absloute_power.distance import (simulate_distance_matrix,
                                     coeffient_of_determination,
                                     cohens_f2,
                                     omega2,
                                     _check_param,
                                     convert_to_mirror,
                                     _simulate_gauss_vec,
                                     _vec_size,
                                     _permanova_ssw,
                                     _permanova_sst,
                                     )


class DistanceTest(TestCase):

    def setUp(self):
        np.random.seed(5)
        self.length = 3
        self.dm = np.array([[0, 1, 2],
                            [1, 0, 3],
                            [2, 3, 0]])

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

    def test_permanova_ssw(self):
        groups = pd.Series([0, 0, 0],
                           index=['s.1', 's.2', 's.3']
                           )
        dm = skbio.DistanceMatrix(self.dm, ['s.1', 's.2', 's.3'])
        ssw = _permanova_ssw(dm, groups)
        self.assertEqual(14./3, ssw)

    def test_permanova_sst(self):
        groups = pd.Series([0, 0, 0],
                           index=['s.1', 's.2', 's.3']
                           )
        dm = skbio.DistanceMatrix(self.dm, ['s.1', 's.2', 's.3'])
        sst = _permanova_sst(dm, groups)
        self.assertEqual(14./3, sst)

    def test_coeffient_of_determination(self):
        groups = pd.Series([0, 0, 0],
                           index=['s.1', 's.2', 's.3']
                           )
        dm = skbio.DistanceMatrix(self.dm, ['s.1', 's.2', 's.3'])
        coeff = coeffient_of_determination(dm, groups)
        self.assertEqual(1, coeff)

    def test_cohens_f2(self):
        groups = pd.Series([0, 0, 1, 1],
                           index=['s.1', 's.2', 's.3', 's.4'],
                            )
        dm = skbio.DistanceMatrix(np.array([[0.0, 0.2, 0.4, 0.4],
                                            [0.2, 0.0, 0.4, 0.4],
                                            [0.4, 0.4, 0.0, 0.2],
                                            [0.4, 0.4, 0.2, 0.0],
                                            ]),
                                  ['s.1', 's.2', 's.3', 's.4']
                                  )
        f2 = cohens_f2(dm, groups)
        self.assertEqual(f2, 0.125)

    def test_omega2(self):
        groups = pd.Series([0, 0, 1, 1],
                           index=['s.1', 's.2', 's.3', 's.4'],
                           )
        dm = skbio.DistanceMatrix(np.array([[0.0, 0.2, 0.4, 0.4],
                                            [0.2, 0.0, 0.4, 0.4],
                                            [0.4, 0.4, 0.0, 0.2],
                                            [0.4, 0.4, 0.2, 0.0],
                                            ]),
                                  ['s.1', 's.2', 's.3', 's.4']
                                  )
        w2 = omega2(dm, groups)
        npt.assert_almost_equal(w2, 0.821428, 5)

if __name__ == '__main__':
    main()
