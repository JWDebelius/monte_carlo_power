from unittest import TestCase, main

import numpy as np
import numpy.testing as npt
import pandas as pd
import skbio
from machivellian.bootstrap import (_bootstrap_dm,
                                    bootstrap_permanova,
                                    bootstrap_mantel,
                                    )


class TestPermutationBootstrap(TestCase):
    def setUp(self):
        np.random.seed(5)
        array = np.array([[0, 1, 2, 3],
                          [1, 0, 1, 2],
                          [2, 1, 0, 1],
                          [3, 2, 1, 0]])

        self.ids = ['s.1', 's.2', 's.3', 's.4']
        self.dm1 = skbio.DistanceMatrix(array, ids=self.ids)
        self.dm2 = skbio.DistanceMatrix(array * 2, ids=self.ids)
        self.groups = pd.DataFrame(data=[[0, 0, 1, 1], ['bar']*4],
                                   columns=self.ids,
                                   index=['group', 'foo']).T
        self.permanova_res = {'method name': 'PERMANOVA',
                              'number of groups': 2,
                              'number of permutations': 5,
                              'p-value': 0.5,
                              'sample size': 4,
                              'test statistic': 8.0,
                              'test statistic name': 'pseudo-F'}

    def test_bootstrap_dm_unique(self):
        test_dm = _bootstrap_dm(self.ids, self.dm1)
        npt.assert_array_equal(test_dm.data, self.dm1.data)
        self.assertEqual(test_dm.ids, (0, 1, 2, 3))

    def test_bootstrap_dm_nonunique_new_new_anme(self):
        ids = ['s.1', 's.1', 's.2']
        known_dm = np.array([[0., 0., 1.],
                             [0., 0., 1.],
                             [1., 1., 0.]])
        test_dm = _bootstrap_dm(ids, self.dm1, ['1', '1.1', '2'])
        npt.assert_array_equal(known_dm, test_dm.data)
        self.assertEqual(('1', '1.1', '2'), test_dm.ids)

    def test_boostrap_permanova_df(self):
        test_result = bootstrap_permanova(self.ids,
                                          self.dm1,
                                          self.groups['group'],
                                          permutations=5).to_dict()
        self.assertEqual(self.permanova_res.keys(), test_result.keys())
        for k, v in self.permanova_res.items():
            self.assertEqual(v, test_result[k])

    def test_boostrap_mantel(self):
        known = (1.0, 1./3, 4)
        test = bootstrap_mantel(self.ids, self.dm1, self.dm2,
                                permutations=5)
        self.assertTrue(
            np.array([known[i] == test[i] for i in range(3)]).all()
            )

if __name__ == '__main__':
    main()
