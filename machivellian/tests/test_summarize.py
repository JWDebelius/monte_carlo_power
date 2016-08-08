from unittest import TestCase, main

import copy
from functools import partial

import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from machivellian.summarize import (summarize_power,
                                    calc_f_effect,
                                    calc_t_effect,
                                    calc_z_effect,
                                    calc_f_power,
                                    calc_t_power,
                                    calc_z_power,
                                    _build_summary_frame,
                                    _calculate_effect_size,
                                    modify_effect_size,
                                    _get_dists,
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
            index=['counts', 'emperical', 'traditional', 'sim_position']
            ).T
        self.early_summary['counts'] = \
            self.early_summary['counts'].apply(lambda x: int(x))
        self.early_summary['sim_position'] = \
            self.early_summary['sim_position'].apply(lambda x: int(x))
        self.early_summary = self.early_summary[['counts', 'emperical',
                                                 'sim_position',
                                                 'traditional']]

        self.df = pd.DataFrame(
            data=np.array([[35, 0.05, 0.19666182, 0.5, 1, 0],
                           [45, 0.05, 0.24228406, 0.5, 1, 1]]),
            columns=['counts', 'alpha', 'emperical', 'z_effect',
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
                                       for i in np.arange(5, 100, 10)},
                               dists=['z'])
        columns = pd.Index(['counts', 'emperical', 'sim_position',
                            'traditional', 'test', 'alpha',
                            'sim_num', 'colors', 'z_effect',
                            'z_mean', 'z_power'])
        index = pd.Index(['test.0.%i' % i for i in np.arange(0, 10)],
                         name='index')
        pdt.assert_index_equal(test.columns, columns)
        pdt.assert_index_equal(test.index, index)
        self.assertEqual(test['test'].unique(), 'test')
        self.assertEqual(test['alpha'].unique(), 0.05)
        self.assertEqual(test['sim_num'].unique(), 0)
        self.assertEqual(test['colors'].unique(), 'k')

    def test_modify_effect_sizes(self):
        drop_index = ['B']
        mod = modify_effect_size(self.df, drop_index, ['z'])
        self.assertTrue(pd.isnull(mod.loc['B', 'z_effect']))

    def test_calc_f_effect(self):
        known = pd.Series([0.191889, 0.191941], index=['A', 'B'])
        test = self.df.apply(calc_f_effect, axis=1)
        pdt.assert_series_equal(known, test)

    def test_calc_t_effect(self):
        known = pd.Series([0.26728415,  0.26832483], index=['A', 'B'])
        test = self.df.apply(calc_t_effect, axis=1)
        pdt.assert_series_equal(known, test)

    def test_calc_z_effect(self):
        known = pd.Series([0.18700873,  0.18797725], index=['A', 'B'])
        test = self.df.apply(calc_z_effect, axis=1)
        pdt.assert_series_equal(known, test)

    def test_calc_f_power(self):
        known = pd.Series([0.81885655,  0.90630465], index=['A', 'B'])
        test = self.df.apply(partial(calc_f_power, col2='z_effect'), axis=1)
        pdt.assert_series_equal(known, test)

    def test_calc_t_power(self):
        known = pd.Series([0.54068791,  0.65018550], index=['A', 'B'])
        test = self.df.apply(partial(calc_t_power, col2='z_effect'), axis=1)
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

    def test_calculate_effect_size(self):
        known = copy.deepcopy(self.df)
        known['z_effect'] = [0.18700873,  0.18797725]
        _calculate_effect_size(self.df, ['z'])
        pdt.assert_frame_equal(known, self.df)

    def test_calculate_power(self):
        known = copy.deepcopy(self.df)
        known['z_power'] = [0.84087872,  0.91836203]
        known['z_mean'] = 0.5

    def test_get_dists(self):
        known = ['f', 'f2', 't', 'z']
        test = _get_dists(None)
        self.assertEqual(known, test)

if __name__ == '__main__':
    main()
