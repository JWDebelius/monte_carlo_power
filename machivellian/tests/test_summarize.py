from unittest import TestCase, main

import copy
from functools import partial

import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from machivellian.summarize import (summarize_power,
                                    calc_z_effect,
                                    calc_z_power,
                                    _build_summary_frame,
                                    # _calculate_effect_size,
                                    # modify_effect_size,
                                    )


class SummarizeTest(TestCase):

    def setUp(self):
        self.power_summary = {
            'counts': np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95]),
            'emperical': np.array([[0.04698995, 0.10223453, 0.15022543,
                                    0.19666182, 0.24228406, 0.28709498,
                                    0.33092360, 0.37356885, 0.41484527,
                                    0.45459791]]),
            'traditional': np.array([0.04698995, 0.10223453, 0.15022543,
                                     0.19666182, 0.24228406, 0.28709498,
                                     0.33092360, 0.37356885, 0.41484527,
                                     0.45459791]),
            'num_obs': 100,
            'original_p': 0.001,
            'alpha': 0.05,
            'statistic': 12,
            'alpha_adj': 0.5,
            }
        self.early_summary = pd.DataFrame(
            data=np.array([[5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
                           [0.04698995, 0.10223453, 0.15022543, 0.19666182,
                            0.24228406, 0.28709498, 0.33092360, 0.37356885,
                            0.41484527, 0.45459791],
                           [0.04698995, 0.10223453, 0.15022543, 0.19666182,
                            0.24228406, 0.28709498, 0.33092360, 0.37356885,
                            0.41484527, 0.45459791],
                           np.arange(10),
                           ]),
            index=['counts', 'empirical', 'traditional', 'sim_position']
            ).T
        self.early_summary['counts'] = \
            self.early_summary['counts'].apply(lambda x: int(x))
        self.early_summary['sim_position'] = \
            self.early_summary['sim_position'].apply(lambda x: int(x))
        self.early_summary = self.early_summary[['counts', 'empirical',
                                                 'sim_position',
                                                 'traditional']]

        self.df = pd.DataFrame(
            data=np.array([[35, 0.025, 0.19666182, 0.5, 1, 0],
                           [45, 0.025, 0.24228406, 0.5, 1, 1]]),
            columns=['counts', 'alpha', 'empirical', 'z_effect',
                     'sim_num', 'sim_pos'],
            index=['A', 'B']
            )
        self.df['test'] = 'test'
        self.df['color'] = 'k'

    def test_summary_power(self):
        test = summarize_power(self.power_summary,
                               sim_num=0,
                               test='test',
                               colors={i: 'k' % i
                                       for i in np.arange(5, 100, 10)})
        columns = pd.Index(['counts', 'empirical', 'sim_position',
                            'traditional', 'test', 'ori_alpha', 'alpha',
                            'alpha_adj_factor', 'sim_num', 'p_all',
                            'statistic', 'colors'])
        index = pd.Index(['test.0.%i' % i for i in np.arange(0, 10)],
                         name='index')
        pdt.assert_index_equal(test.columns, columns)
        pdt.assert_index_equal(test.index, index)
        self.assertEqual(test['test'].unique(), 'test')
        self.assertEqual(test['alpha'].unique(), 0.025)
        self.assertEqual(test['alpha_adj_factor'].unique(), 0.5)
        self.assertEqual(test['ori_alpha'].unique(), 0.05)
        self.assertEqual(test['sim_num'].unique(), 0)
        self.assertEqual(test['colors'].unique(), 'k')
        self.assertEqual(test['statistic'].unique(), 12)

    # def test_modify_effect_sizes(self):
    #     drop_index = ['B']
    #     mod = modify_effect_size(self.df, drop_index, ['z'])
    #     self.assertTrue(pd.isnull(mod.loc['B', 'z_effect']))

    def test_calc_z_effect(self):
        known = pd.Series([0.18700873,  0.18797725], index=['A', 'B'])
        test = self.df.apply(calc_z_effect, axis=1)
        pdt.assert_series_equal(known, test)

    def test_calc_z_power(self):
        known = pd.Series([0.84087872,  0.91836203], index=['A', 'B'])
        test = self.df.apply(partial(calc_z_power, col2='z_effect'), axis=1)
        pdt.assert_series_equal(known, test)

    def test_build_summary_frame(self):
        test = _build_summary_frame(self.power_summary)
        pdt.assert_frame_equal(self.early_summary, test)

    def test_build_summary_frame_no_trad(self):
        self.power_summary['traditional'] = None
        known = pd.Series(np.ones(10,) * np.nan, name='traditional')
        test = _build_summary_frame(self.power_summary)
        pdt.assert_series_equal(known, test['traditional'])

    # def test_calculate_effect_size(self):
    #     known = copy.deepcopy(self.df)
    #     known['z_effect'] = [0.18700873,  0.18797725]
    #     _calculate_effect_size(self.df, ['z'])
    #     pdt.assert_frame_equal(known, self.df)

    # def test_calculate_power(self):
    #     known = copy.deepcopy(self.df)
    #     known['z_power'] = [0.84087872,  0.91836203]
    #     known['z_mean'] = 0.5

if __name__ == '__main__':
    main()
