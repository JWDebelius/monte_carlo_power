# Experimental Validation

The goal of this notebook is to apply the power method validated in notebooks 1 - X to real data. We've chosen to perform a meta analysis of the effect size associated with a lean Body Mass Index () to an obese BMI ().

We've collected data from the following studies...

All studies were processed through Qiita ([www.qiita.ucsd.edu](www.qiita.ucsd.edu)). All samples were picked closed reference using SortMeRNA [[*]()] against the August 2013 released of Greengenes [*]().

```python
>>> import os
>>> from functools import partial
...
>>> import biom
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> import pandas as pd
>>> import seaborn as sn
>>> import scipy
>>> import skbio
...
>>> import machivellian.plot as plot
>>> import machivellian.summarize as summarize
>>> from machivellian.bootstrap import bootstrap_permanova
>>> from machivellian.power import subsample_power, confidence_bound
>>> from machivellian.effects import z_power, z_effect
...
>>> from machivellian.effects import z_power
```

We

```python
>>> bmi_map = pd.read_csv('./data/otu_table_and_mapping_ibd/clean_ibd_map.txt',
...                       sep='\t', dtype=str)
>>> bmi_map.set_index('#SampleID', inplace=True)
```

```python
>>> bmi_dm = skbio.DistanceMatrix.read('data/otu_table_and_mapping_ibd/1k/unweighted_unifrac_dm.txt')
```

```python
>>> map_ids = set(bmi_map.index)
>>> dm_ids = set(bmi_dm.ids)
```

```python
>>> rep_ids = map_ids.intersection(dm_ids)
```

```python
...
>>> bmi_map = bmi_map.loc[rep_ids]
>>> bmi_dm = bmi_dm.filter(rep_ids)
```

```python
>>> bmi_map.columns
Index(['IBD', 'IBD_TYPE', 'STUDY_ID', 'IBD_STUDY'], dtype='object')
```

```python
>>> bmi_map.groupby(['STUDY_ID', 'IBD']).count()['IBD_TYPE']
STUDY_ID  IBD
1070      CD      23
          HC      35
          UC      15
1458      CD      18
          HC      18
          UC      38
1460      HC      15
1629      CD     217
          HC      54
          UC     214
Name: IBD_TYPE, dtype: int64
```

```python
>>> summaries = {}
```

```python
>>> for study_name, study_map in bmi_map.groupby('STUDY_ID'):
...     if study_name == '1460':
...         continue
...     print(study_name)
...     study_dm = bmi_dm.filter(study_map.index)
...     study_groups = [study_map.groupby('IBD').groups[group] for group in ['UC', 'HC']]
...     min_group_size = min([len(group) for group in study_groups])
...     study_counts = np.arange(10, min_group_size*1.1, 3)
...
...     def study_test(ids):
...         """A quick wraper for bootstrap_peramnaova"""
...         ids = np.hstack(ids)
...         res = bootstrap_permanova(ids, dm=study_dm,
...                                   grouping=study_map['IBD'],
...                                   permutations=100
...                                   )
...         return res['p-value']
...
...     study_pwr = subsample_power(test=study_test,
...                                 samples=study_groups,
...                                 counts=study_counts,
...                                 num_iter=100,
...                                 num_runs=3,
...                                 alpha=0.01,
...                                 )
...
>>> #     study_eff = z_effect(study_counts, study_pwr, alpha=0.02)
...
...     summaries[study_name] = {'power': study_pwr,
...                              'counts': study_counts,
...                              'alpha': 0.01,
...                              }
1070
1458
1629
```

```python
>>> % matplotlib inline
>>> plt.plot(summaries['1070']['counts'],
...          summaries['1070']['power'].T, 'o')
>>> plt.plot(np.linspace(5, 50, 100),
...          z_power(np.linspace(5, 50, 100), 0.30358342298881896))
[<matplotlib.lines.Line2D at 0x10cf7e0b8>]
```

```python
>>> z_effect(summaries['1458']['counts'],
...          summaries['1458']['power'])
(0.74895855567005343, 0.00011390955600248221, 12)
```

```python
>>> % matplotlib inline
>>> plt.plot(summaries['1458']['counts'],
...          summaries['1458']['power'].T, 'o')
>>> plt.plot(np.linspace(5, 50, 100),
...          z_power(np.linspace(5, 50, 100), 0.74895855567005343))
[<matplotlib.lines.Line2D at 0x10ceabd68>]
```

```python
>>> z_effect(summaries['1629']['counts'],
...          summaries['1629']['power'])
(0.67128665563312873, 2.6273517899491746e-05, 51)
```

```python
>>> % matplotlib inline
>>> plt.plot(summaries['1629']['counts'],
...          summaries['1629']['power'].T, 'o')
>>> plt.plot(np.linspace(5, 50, 100),
...          z_power(np.linspace(5, 50, 100), 0.67128665563312873))
[<matplotlib.lines.Line2D at 0x10d3acd30>]
```

```python
>>> for study_name, study_summary in summaries.items():
...     print('%s: %1.2f +/- %1.2f' % (study_name,
...           np.nanmean(study_summary['study_eff']),
...           confidence_bound(study_summary['study_eff'])))
```

```python

```

```python
>>> for study_name, study_map in bmi_map.groupby('STUDY_ID'):
...     if study_name == '1460':
...         continue
...     print(study_name)
...     study_dm = bmi_dm.filter(study_map.index)
...     study_groups = [study_map.groupby('IBD').groups[group] for group in ['CD', 'HC']]
...     min_group_size = min([len(group) for group in study_groups])
...     study_counts = np.arange(10, min_group_size*1.1, 3)
...
...     def study_test(ids):
...         """A quick wraper for bootstrap_peramnaova"""
...         ids = np.hstack(ids)
...         res = bootstrap_permanova(ids, dm=study_dm, grouping=study_map['IBD'])
...         return res['p-value']
...
...     study_pwr = subsample_power(test=study_test,
...                                 samples=study_groups,
...                                 counts=study_counts,
...                                 num_iter=100,
...                                 num_runs=5,
...                                 alpha=0.05
...                                 )
...
...     study_eff = z_effect(study_counts, study_pwr, alpha=0.02)
...
...     summaries[study_name] = {'study_pwr': study_pwr,
...                              'study_eff': study_eff.reshape(study_pwr.shape),
...                              'study_count': study_counts,
...                              }
1070
1458
1629
```

```python
>>> for study_name, study_summary in summaries.items():
...     print('%s: %1.2f +/- %1.2f' % (study_name,
...           np.nanmean(study_summary['study_eff']),
...           confidence_bound(study_summary['study_eff'])))
1629: nan +/- nan
1070: 0.90 +/- 0.05
1458: 0.76 +/- 0.05
/Users/jdebelius/miniconda2/envs/power_play3/lib/python3.5/site-packages/numpy/lib/nanfunctions.py:703: RuntimeWarning: Mean of empty slice
  warnings.warn("Mean of empty slice", RuntimeWarning)
/Users/jdebelius/miniconda2/envs/power_play3/lib/python3.5/site-packages/numpy/lib/nanfunctions.py:1202: RuntimeWarning: Degrees of freedom <= 0 for slice.
  warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning)
/Users/jdebelius/Repositories/monte_carlo_power/machivellian/power.py:218: RuntimeWarning: invalid value encountered in sqrt
  bound = np.nanstd(vec, axis=axis, ddof=1) / np.sqrt(num_counts - 1) * \
```

```python
>>> summaries
{'1070': {'study_count': array([ 10.,  13.,  16.,  19.,  22.,  25.]),
  'study_eff': array([[ 0.87280491,  0.92504591,         nan,         nan,         nan,
                  nan],
         [ 0.90446522,         nan,         nan,         nan,         nan,
                  nan],
         [ 0.8337631 ,  1.00082407,         nan,         nan,         nan,
                  nan],
         [ 0.87280491,  0.9593042 ,         nan,         nan,         nan,
                  nan],
         [ 0.81528243,  0.941466  ,         nan,         nan,         nan,
                  nan]]),
  'study_pwr': array([[ 0.76,  0.9 ,  0.98,  1.  ,  1.  ,  1.  ],
         [ 0.79,  0.96,  0.97,  1.  ,  1.  ,  1.  ],
         [ 0.72,  0.94,  1.  ,  1.  ,  1.  ,  1.  ],
         [ 0.76,  0.92,  0.99,  1.  ,  1.  ,  1.  ],
         [ 0.7 ,  0.91,  0.98,  1.  ,  1.  ,  1.  ]])},
 '1458': {'study_count': array([ 10.,  13.,  16.,  19.]),
  'study_eff': array([[ 0.66531239,  0.72308877,  0.84862599,         nan],
         [ 0.64152495,  0.73957121,  0.80718393,         nan],
         [ 0.70522684,  0.81309209,  0.88238498,         nan],
         [ 0.69719309,  0.7071316 ,  0.83382512,         nan],
         [ 0.64945243,  0.81309209,  0.86470512,         nan]]),
  'study_pwr': array([[ 0.52,  0.71,  0.91,  0.98],
         [ 0.49,  0.73,  0.88,  0.98],
         [ 0.57,  0.81,  0.93,  0.99],
         [ 0.56,  0.69,  0.9 ,  1.  ],
         [ 0.5 ,  0.81,  0.92,  0.99]])},
 '1629': {'study_count': array([ 10.,  13.,  16.,  19.,  22.,  25.,  28.,  31.,  34.,  37.,  40.,
          43.,  46.,  49.,  52.,  55.,  58.]),
  'study_eff': array([[ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,
           nan,  nan,  nan,  nan,  nan,  nan],
         [ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,
           nan,  nan,  nan,  nan,  nan,  nan],
         [ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,
           nan,  nan,  nan,  nan,  nan,  nan],
         [ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,
           nan,  nan,  nan,  nan,  nan,  nan],
         [ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,
           nan,  nan,  nan,  nan,  nan,  nan]]),
  'study_pwr': array([[ 1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,
           1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
         [ 1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,
           1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
         [ 0.99,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,
           1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
         [ 0.99,  0.99,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,
           1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
         [ 0.98,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,
           1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ]])}}
```

```python

```
