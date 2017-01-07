from unittest import TestCase, main

import numpy as np
import numpy.testing as npt
import pandas as pd

from machivellian.beta import (expand_otu_ids,
                               collapse_otu_ids,
                               subsample_features,
                               bootstrap_permanova,
                               )


class PowerBetaTest(TestCase):
    def setUp(self):
        np.random.seed(5)
        self.feat_ids = np.array(['otu.%i' % i for i in np.arange(1, 6)],
                                 dtype=('object'))
        self.obs_ids = np.array(['obs.1', 'obs.2'])
        self.counts = np.arange(0, 5)
        self.table = np.array([[0, 1, 2, 3, 4],
                               [1, 2, 3, 4, 5]])
        self.id_list = np.array(['otu.2', 'otu.3', 'otu.3', 'otu.4', 'otu.4',
                                 'otu.4', 'otu.5', 'otu.5', 'otu.5', 'otu.5'],
                                dtype='object')

    def test_expand_otu_ids(self):
        id_list = expand_otu_ids(self.feat_ids, self.counts)
        npt.assert_array_equal(self.id_list, id_list)

    def test_collapse_otu_ids_default(self):
        ids, counts = collapse_otu_ids(self.id_list)
        npt.assert_array_equal(ids, self.feat_ids[1:])
        npt.assert_array_equal(counts, self.counts[1:])

    def test_collapse_otu_ids_order(self):
        ids, counts = collapse_otu_ids(self.id_list, order=self.feat_ids)
        npt.assert_array_equal(ids, self.feat_ids)
        npt.assert_array_equal(counts, self.counts)

    def test_subsample_features_no_ids_no_replace(self):
        k_sub = np.array([[0, 0, 1, 2, 2],
                          [0, 1, 0, 3, 1]])
        t_sub = subsample_features(self.table, depth=5, bootstrap=False)
        npt.assert_array_equal(k_sub, t_sub)

    def test_subsample_features(self):
        k_sub = np.array([[0, 1, 0, 1, 3],
                          [0, 0, 1, 2, 2]])
        t_sub = subsample_features(self.table, depth=5,
                                   feature_ids=self.feat_ids)
        npt.assert_array_equal(k_sub, t_sub)

    def test_boostrap_permanova_defaults(self):
        known_dm = np.array([[0.0,  0.4,  0.6,  0.5],
                             [0.4,  0.0,  0.8,  0.8],
                             [0.6,  0.8,  0.0,  0.5],
                             [0.5,  0.8,  0.5,  0.0]])
        ids = np.array(['o.1', 'o.2', 'o.3', 'o.4'])
        known_res = {'method name': 'PERMANOVA',
                     'test statistic name': 'pseudo-F',
                     'sample size': 4,
                     'number of groups': 2,
                     'test statistic': 3.60976,
                     'p-value': 0.6,
                     'number of permutations': 9,
                     }

        table = pd.DataFrame(np.array([[0, 1, 2, 3, 4],
                                       [1, 2, 3, 4, 0],
                                       [2, 3, 4, 0, 1],
                                       [3, 4, 0, 1, 2]]),
                             index=ids)
        grouping = pd.Series([0, 0, 1, 1], index=ids)
        test_res, test_dm = bootstrap_permanova(ids, table * 2,
                                                grouping=grouping,
                                                depth=10, permutations=9)
        npt.assert_array_equal(known_dm, test_dm.data)
        self.assertEqual(set(test_res.keys()), set(known_res.keys()))
        for k, v in known_res.items():
            if k in {'test statistic', 'p-value'}:
                npt.assert_almost_equal(v, test_res[k], 5)
            else:
                self.assertEqual(v, test_res[k])

if __name__ == '__main__':
    main()
