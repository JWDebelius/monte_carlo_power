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
>>> bmi_map = pd.read_csv('./data/merged_otu_table_and_mapping_bmi/merged_bmi_mapping_final2.txt',
...                       sep='\t', dtype=str)
>>> bmi_map.set_index('#SampleID', inplace=True)
```

```python
>>> bmi_map = bmi_map.loc[bmi_map['COUNTRY'] == 'GAZ:United States of America']
>>> bmi_map = bmi_map.loc[bmi_map['AGE'].apply(lambda x: float(x) > 20)]
```

```python
>>> bmi_dm = skbio.DistanceMatrix.read('data/merged_otu_table_and_mapping_bmi/1k/unweighted_unifrac_dm.txt')
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
>>> summaries = {}
```

```python
>>> alpha = 0.01
>>> counts = np.arange(5, 50, 5)
```

```python
>>> for study_name, study_map in bmi_map.groupby('original_study'):
...     if study_name == 'COMBO_Wu':
...         continue
...     print(study_name)
...     study_dm = bmi_dm.filter(study_map.index)
...     study_groups = [study_map.groupby('bmi_group_coded').groups[group]
...                     for group in ['Normal', 'Obese']]
...     min_group_size = min([len(group) for group in study_groups])
...     study_counts = counts[counts <= min_group_size]
...
...     def study_test(ids):
...         """A quick wraper for bootstrap_peramnaova"""
...         ids = np.hstack(ids)
...         res = bootstrap_permanova(ids, dm=study_dm,
...                                   grouping=study_map['bmi_group_coded'],
...                                   permutations=100)
...         return res['p-value']
...
...     study_pwr = subsample_power(test=study_test,
...                                 samples=study_groups,
...                                 counts=study_counts,
...                                 num_iter=100,
...                                 num_runs=3,
...                                 alpha=alpha,
...                                 )
...
...     summaries[study_name] = {'power': study_pwr,
...                              'counts': study_counts,
...                              'alpha': alpha,
...                              }
HMP
Turnbaugh_mz_dz_twins
Yatsunenko_GG
amish_Fraser
```

```python
>>> summaries
{'HMP': {'alpha': 0.01,
  'counts': array([ 5, 10, 15, 20]),
  'power': array([[ 0.01,  0.04,  0.11,  0.22],
         [ 0.01,  0.04,  0.15,  0.18],
         [ 0.01,  0.01,  0.14,  0.25]])},
 'Turnbaugh_mz_dz_twins': {'alpha': 0.01,
  'counts': array([ 5, 10, 15, 20, 25, 30]),
  'power': array([[ 0.02,  0.23,  0.44,  0.69,  0.93,  0.96],
         [ 0.02,  0.16,  0.43,  0.72,  0.89,  0.99],
         [ 0.02,  0.13,  0.39,  0.65,  0.85,  0.9 ]])},
 'Yatsunenko_GG': {'alpha': 0.01,
  'counts': array([ 5, 10, 15, 20, 25, 30, 35]),
  'power': array([[ 0.  ,  0.03,  0.14,  0.28,  0.53,  0.66,  0.8 ],
         [ 0.  ,  0.06,  0.2 ,  0.21,  0.46,  0.62,  0.83],
         [ 0.02,  0.07,  0.29,  0.45,  0.49,  0.68,  0.81]])},
 'amish_Fraser': {'alpha': 0.01,
  'counts': array([ 5, 10, 15, 20, 25, 30, 35, 40, 45]),
  'power': array([[ 0.02,  0.04,  0.07,  0.09,  0.24,  0.28,  0.42,  0.66,  0.64],
         [ 0.02,  0.06,  0.05,  0.12,  0.24,  0.35,  0.44,  0.62,  0.66],
         [ 0.01,  0.05,  0.13,  0.19,  0.21,  0.33,  0.41,  0.51,  0.78]])}}
```

```python

```

```python
>>> % matplotlib inline
>>> plt.plot(summaries['Turnbaugh_mz_dz_twins']['counts'],
...          summaries['Turnbaugh_mz_dz_twins']['power'].T, 'o')
...
>>> eff = z_effect(summaries['Turnbaugh_mz_dz_twins']['counts'],
...          summaries['Turnbaugh_mz_dz_twins']['power'], alpha=0.1)
...
>>> plt.plot(summaries['Turnbaugh_mz_dz_twins']['counts'],
...          z_power(summaries['Turnbaugh_mz_dz_twins']['counts'],
...                  eff[0]))
[<matplotlib.lines.Line2D at 0x10db634a8>]
```

```python
>>> from scipy.optimize import curve_fit
```

```python
>>> from machivellian.effects import z_power
```

```python
>>> curve_fit(z_power())
```

```python
>>> z_effect(summaries['HMP']['counts'][1:],
...          summaries['HMP']['power'][:, 1:])
(0.1322348036612524, 0.00025041906494875681, 9)
```

```python
>>> \
array([[ 0.22,  0.45,  0.64,  0.84],
       [ 0.21,  0.48,  0.66,  0.81],
       [ 0.18,  0.49,  0.66,  0.83]])
```

```python
>>> summaries['Yatsunenko_GG']['study_eff']
array([[ 0.23450755,  0.27578986,  0.3104443 ,  0.32870689,  0.35223206,
         0.38565073,  0.4085863 ,  0.4382363 ,  0.44973182],
       [ 0.23450755,  0.28656457,  0.31157971,  0.3334177 ,  0.33357403,
         0.38736182,  0.38007182,  0.41252595,  0.43671876],
       [ 0.24665236,  0.29840778,  0.30240247,  0.32396509,  0.36157012,
         0.38994236,  0.42026227,  0.41989613,  0.43402234]])
```

```python
>>> z_effect(summaries['Yatsunenko_GG']['study_count'],
...          summaries['Yatsunenko_GG']['study_pwr']).reshape(summaries['Yatsunenko_GG']['study_pwr'].shape)
array([[ 0.12893125,  0.1843581 ,  0.22866525,  0.25405317,  0.28311615,
         0.32099871,  0.34763179,  0.38040978,  0.3945964 ],
       [ 0.12893125,  0.1951328 ,  0.22980066,  0.25876397,  0.26445812,
         0.3227098 ,  0.31911731,  0.35469942,  0.38158334],
       [ 0.14107605,  0.20697602,  0.22062341,  0.24931137,  0.29245421,
         0.32529034,  0.35930776,  0.3620696 ,  0.37888691]])
```

```python
>>> plt.plot(summaries['Yatsunenko_GG']['study_count'],
...          summaries['Yatsunenko_GG']['study_pwr'].T, 'o')
>>> plt.plot(summaries['Yatsunenko_GG']['study_count'],
...          z_power(summaries['Yatsunenko_GG']['study_count'],
...                  np.nanmean(summaries['Yatsunenko_GG']['study_eff'])
...                  )
...          )
[<matplotlib.lines.Line2D at 0x10bbfa668>]
```

```python
>>> from scipy.optimize import curve_fit
```

```python
>>> def func(x, a, b, c):
...     return a * np.square(x - b) + c
```

```python
>>> help(z_power)
Help on function z_power in module machivellian.effects:

z_power(counts, effect, alpha=0.05)
    Estimates power for a z distribution from an effect size

    This is based on the equations in [1]_, the equation assumes a positive
    magnitude to the effect size and a one-tailed test.

    Parameters
    ----------
    counts : array
        The number of observations for each power depth
    effect : float
        A standard measure of the difference between the underlying populations
    alpha : float
        The critial value used to calculate the power

    Returns
    -------
    ndarray
        The statistical power at the depth specified by `counts`

    References
    ----------
    .. [1] Lui, X.S. (2014) *Statistical power analysis for the social and
    behavioral sciences: basic and advanced techniques.* New York: Routledge.
    378 pg.
```

```python
>>> from scipy.stats import norm as z
>>> def power_fit_test(counts, effect):
...     power = z.cdf(effect * np.sqrt(counts) - z.pdf(0.95))
...     return power
```

```python
>>> sim_y = (summaries['Yatsunenko_GG']['study_count'], 0.25)
>>> sim_y
(array([ 15.,  20.,  25.,  30.,  35.,  40.,  45.,  50.,  55.]), 0.25)
```

```python
>>> summaries['Yatsunenko_GG']['study_count']
array([ 15.,  20.,  25.,  30.,  35.,  40.,  45.,  50.,  55.])
```

```python
>>> popt, pcov = curve_fit(func, xdata=summaries['Yatsunenko_GG']['study_count'],
...                        ydata=sim_y,
...                        bounds=[0, 1.5])
```

```python
>>> xdata = np.linspace(-4, 10, 50)
>>> y = func(xdata, 2.5, 1.3, 0.5)
>>> ydata = y + np.random.normal(size=xdata.shape) * 50
```

```python
>>> plt.plot(xdata, ydata, 'o')
>>> plt.plot(xdata, func(xdata, *popt))
[<matplotlib.lines.Line2D at 0x10e6c35f8>]
```

```python
>>> popt, pcov = curve_fit(func, xdata, ydata, bounds=[0, [3., 2., 1.]])
```

```python
>>> popt
array([  2.11262121e+00,   4.58576404e-01,   7.13322783e-13])
```

```python
>>> pcov
array([[  1.60270283e-32,   1.08983792e-32,  -2.26141372e-31],
       [  1.08983792e-32,   1.61191349e-32,  -7.97561174e-32],
       [ -2.26141372e-31,  -7.97561174e-32,   7.52102563e-30]])
```

```python

```
