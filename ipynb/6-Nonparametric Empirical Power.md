# Parametric and Empirical Power Comparison

In the last two notebooks, we simulated data and calculated empirical power. Here, we'll compare the parametric power calculations, the empirical power values and the predicted power. This notebook will generate Figure S1.

```python
>>> import os
>>> import pickle
>>> import warnings
...
>>> from functools import partial
...
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> import pandas as pd
>>> import seaborn as sn
>>> import scipy
...
>>> import machivellian.plot as plot
>>> import machivellian.summarize as summarize
...
>>> from machivellian.effects import z_power
...
>>> % matplotlib inline
>>> sn.set_style('ticks')
```

# 2. Parameters

We've performed 100 simulations, which are stored in the simulation directory.

```python
>>> num_rounds = 100
```

```python
>>> sim_location = './simulations'
>>> if not os.path.exists(sim_location):
...     raise ValueError('The power simulations do not exist.'
...                      'Please go back to notebooks 2 and 3 and'
...                      'calculate power.'
...                      )
>>> return_dir = os.path.join(sim_location, 'extrapolation')
>>> if not os.path.exists(return_dir):
...     os.makedirs(return_dir)
```

Power in previous notebooks was calculated with between 5 and no more htan 100 observations, with 10 observation steps.

```python
>>> counts = np.arange(5, 100, 10)
```

We'll use the spectral colormap, scaled by the count depth.

```python
>>> colormap = 'Spectral'
...
>>> colors = {count: list(sn.color_palette(colormap, n_colors=len(counts))[i])
...           for (i, count) in enumerate(counts)}
```

# 3. Loading Power Calculations

We'll compare the behavior of distribution-based power, emperical power and the power calculated from curve fitting on the parametric tests. We can compare the behavior of emperical power, and power fit to emperical values for all tests.

```python
>>> # tests = ['lognormal', 'uniform', 'permanova', 'mantel']
... tests = ['permanova']
```

Finally, we'll create a set of parameters for each of the distributions being evaluated. This includes the clean name, which provides a prettier title in plots, the number of groups in the tested (2 for everything except the ANOVAs), and the location of both the input emperical power calculations and output summary tables.

```python
>>> distributions = {'lognormal': {'clean_name': 'Log Normal Distribution',
...                                'input_dir': './simulations/power/lognormal/',
...                                'return_fp': './simulations/extrapolation/lognormal.txt',
...                                },
...                  'uniform': {'clean_name': 'Uniform Distribution',
...                              'input_dir': './simulations/power/lognormal/',
...                              'return_fp': './simulations/extrapolation/uniform.txt'
...                              },
...                  'permanova': {'clean_name': 'PERMANOVA',
...                                'input_dir': './simulations/power/permanova/power',
...                                'return_fp': './simulations/extrapolation/permanova.txt'
...                                },
...                  'mantel': {'clean_name': 'Mantel',
...                             'input_dir': './simulations/power/mantel/',
...                             'return_fp': './simulations/extrapolation/mantel.txt'
...                             },
...                  }
```

We'll start by loading the poewr data for all the tests we've preformed.

```python
>>> all_powers = []
>>> titles = []
>>> for test_name in tests:
...     titles.append(distributions[test_name]['clean_name'])
...     power_dir = distributions[test_name]['input_dir']
...     return_fp = distributions[test_name]['return_fp']
...
...     if not os.path.exists(power_dir):
...         raise ValueError('%s does not exist' % power_dir)
...
...     summaries = []
...
...     for i in np.arange(1, 5):
...         # Loads through the power simulation for the round
...         power_fp = os.path.join(power_dir, 'simulation_%i.p' % i)
...
...         with open(power_fp, 'rb') as f_:
...             sim_p = pickle.load(f_)
...         sim = {'counts': sim_p[0], 'empirical': sim_p[1], 'alpha': 0.05, 'original_p': None, 'alpha_adj': 1}
...         sim['alpha_adj'] = 1
...         sim['statistic'] = np.nan
...         summaries.append(
...             summarize.summarize_power(power_summary=sim,
...                                       sim_num=i,
...                                       test=test_name,
...                                       colors=colors)
...             )
...     summaries = pd.concat(summaries)
...     summaries.to_csv(return_fp, sep='\t')
...
...     all_powers.append(summaries)
>>> all_powers = pd.concat(all_powers)
>>> all_powers['sim_id'] = all_powers['test'] + '.' + all_powers['sim_num'].astype(str)
```

Next, we'll estimate the effect size, and use that to predict the power as we did with the parametric data in notebook 4. We'll use the same boundary conditions that we imposed before.

```python
>>> sn.distplot(all_powers['empirical'], kde=False)
```

```python
>>> all_powers
                counts  empirical  sim_position  traditional       test  \
index
permanova.1.0      5.0       0.58             0          NaN  permanova
permanova.1.1     15.0       1.00             1          NaN  permanova
permanova.1.2     25.0       1.00             2          NaN  permanova
permanova.1.3     35.0       1.00             3          NaN  permanova
permanova.1.4     45.0       1.00             4          NaN  permanova
permanova.1.5     55.0       1.00             5          NaN  permanova
permanova.1.6     65.0       1.00             6          NaN  permanova
permanova.1.7     75.0       1.00             7          NaN  permanova
permanova.1.8     85.0       1.00             8          NaN  permanova
permanova.1.10     5.0       0.55            10          NaN  permanova
permanova.1.11    15.0       1.00            11          NaN  permanova
permanova.1.12    25.0       1.00            12          NaN  permanova
permanova.1.13    35.0       1.00            13          NaN  permanova
permanova.1.14    45.0       1.00            14          NaN  permanova
permanova.1.15    55.0       1.00            15          NaN  permanova
permanova.1.16    65.0       1.00            16          NaN  permanova
permanova.1.17    75.0       1.00            17          NaN  permanova
permanova.1.18    85.0       1.00            18          NaN  permanova
permanova.1.20     5.0       0.51            20          NaN  permanova
permanova.1.21    15.0       1.00            21          NaN  permanova
permanova.1.22    25.0       1.00            22          NaN  permanova
permanova.1.23    35.0       1.00            23          NaN  permanova
permanova.1.24    45.0       1.00            24          NaN  permanova
permanova.1.25    55.0       1.00            25          NaN  permanova
permanova.1.26    65.0       1.00            26          NaN  permanova
permanova.1.27    75.0       1.00            27          NaN  permanova
permanova.1.28    85.0       1.00            28          NaN  permanova
permanova.2.0      5.0       0.13             0          NaN  permanova
permanova.2.1     15.0       0.80             1          NaN  permanova
permanova.2.2     25.0       0.99             2          NaN  permanova
...                ...        ...           ...          ...        ...
permanova.3.22    25.0       1.00            22          NaN  permanova
permanova.3.23    35.0       1.00            23          NaN  permanova
permanova.3.24    45.0       1.00            24          NaN  permanova
permanova.3.25    55.0       1.00            25          NaN  permanova
permanova.3.26    65.0       1.00            26          NaN  permanova
permanova.3.27    75.0       1.00            27          NaN  permanova
permanova.4.0      5.0       0.81             0          NaN  permanova
permanova.4.1     15.0       1.00             1          NaN  permanova
permanova.4.2     25.0       1.00             2          NaN  permanova
permanova.4.3     35.0       1.00             3          NaN  permanova
permanova.4.4     45.0       1.00             4          NaN  permanova
permanova.4.5     55.0       1.00             5          NaN  permanova
permanova.4.6     65.0       1.00             6          NaN  permanova
permanova.4.7     75.0       1.00             7          NaN  permanova
permanova.4.10     5.0       0.77            10          NaN  permanova
permanova.4.11    15.0       1.00            11          NaN  permanova
permanova.4.12    25.0       1.00            12          NaN  permanova
permanova.4.13    35.0       1.00            13          NaN  permanova
permanova.4.14    45.0       1.00            14          NaN  permanova
permanova.4.15    55.0       1.00            15          NaN  permanova
permanova.4.16    65.0       1.00            16          NaN  permanova
permanova.4.17    75.0       1.00            17          NaN  permanova
permanova.4.20     5.0       0.80            20          NaN  permanova
permanova.4.21    15.0       1.00            21          NaN  permanova
permanova.4.22    25.0       1.00            22          NaN  permanova
permanova.4.23    35.0       1.00            23          NaN  permanova
permanova.4.24    45.0       1.00            24          NaN  permanova
permanova.4.25    55.0       1.00            25          NaN  permanova
permanova.4.26    65.0       1.00            26          NaN  permanova
permanova.4.27    75.0       1.00            27          NaN  permanova

                ori_alpha  alpha  alpha_adj  sim_num p_all  statistic  \
index
permanova.1.0        0.05   0.05          1        1  None        NaN
permanova.1.1        0.05   0.05          1        1  None        NaN
permanova.1.2        0.05   0.05          1        1  None        NaN
permanova.1.3        0.05   0.05          1        1  None        NaN
permanova.1.4        0.05   0.05          1        1  None        NaN
permanova.1.5        0.05   0.05          1        1  None        NaN
permanova.1.6        0.05   0.05          1        1  None        NaN
permanova.1.7        0.05   0.05          1        1  None        NaN
permanova.1.8        0.05   0.05          1        1  None        NaN
permanova.1.10       0.05   0.05          1        1  None        NaN
permanova.1.11       0.05   0.05          1        1  None        NaN
permanova.1.12       0.05   0.05          1        1  None        NaN
permanova.1.13       0.05   0.05          1        1  None        NaN
permanova.1.14       0.05   0.05          1        1  None        NaN
permanova.1.15       0.05   0.05          1        1  None        NaN
permanova.1.16       0.05   0.05          1        1  None        NaN
permanova.1.17       0.05   0.05          1        1  None        NaN
permanova.1.18       0.05   0.05          1        1  None        NaN
permanova.1.20       0.05   0.05          1        1  None        NaN
permanova.1.21       0.05   0.05          1        1  None        NaN
permanova.1.22       0.05   0.05          1        1  None        NaN
permanova.1.23       0.05   0.05          1        1  None        NaN
permanova.1.24       0.05   0.05          1        1  None        NaN
permanova.1.25       0.05   0.05          1        1  None        NaN
permanova.1.26       0.05   0.05          1        1  None        NaN
permanova.1.27       0.05   0.05          1        1  None        NaN
permanova.1.28       0.05   0.05          1        1  None        NaN
permanova.2.0        0.05   0.05          1        2  None        NaN
permanova.2.1        0.05   0.05          1        2  None        NaN
permanova.2.2        0.05   0.05          1        2  None        NaN
...                   ...    ...        ...      ...   ...        ...
permanova.3.22       0.05   0.05          1        3  None        NaN
permanova.3.23       0.05   0.05          1        3  None        NaN
permanova.3.24       0.05   0.05          1        3  None        NaN
permanova.3.25       0.05   0.05          1        3  None        NaN
permanova.3.26       0.05   0.05          1        3  None        NaN
permanova.3.27       0.05   0.05          1        3  None        NaN
permanova.4.0        0.05   0.05          1        4  None        NaN
permanova.4.1        0.05   0.05          1        4  None        NaN
permanova.4.2        0.05   0.05          1        4  None        NaN
permanova.4.3        0.05   0.05          1        4  None        NaN
permanova.4.4        0.05   0.05          1        4  None        NaN
permanova.4.5        0.05   0.05          1        4  None        NaN
permanova.4.6        0.05   0.05          1        4  None        NaN
permanova.4.7        0.05   0.05          1        4  None        NaN
permanova.4.10       0.05   0.05          1        4  None        NaN
permanova.4.11       0.05   0.05          1        4  None        NaN
permanova.4.12       0.05   0.05          1        4  None        NaN
permanova.4.13       0.05   0.05          1        4  None        NaN
permanova.4.14       0.05   0.05          1        4  None        NaN
permanova.4.15       0.05   0.05          1        4  None        NaN
permanova.4.16       0.05   0.05          1        4  None        NaN
permanova.4.17       0.05   0.05          1        4  None        NaN
permanova.4.20       0.05   0.05          1        4  None        NaN
permanova.4.21       0.05   0.05          1        4  None        NaN
permanova.4.22       0.05   0.05          1        4  None        NaN
permanova.4.23       0.05   0.05          1        4  None        NaN
permanova.4.24       0.05   0.05          1        4  None        NaN
permanova.4.25       0.05   0.05          1        4  None        NaN
permanova.4.26       0.05   0.05          1        4  None        NaN
permanova.4.27       0.05   0.05          1        4  None        NaN

                                                          colors       sim_id
index
permanova.1.0   [0.814148415537, 0.219684737031, 0.304805855541]  permanova.1
permanova.1.1   [0.933025763315, 0.391311037774, 0.271972331931]  permanova.1
permanova.1.2    [0.981776240994, 0.607381790876, 0.34579008993]  permanova.1
permanova.1.3   [0.994694348644, 0.809227231671, 0.486966571387]  permanova.1
permanova.1.4   [0.998231449548, 0.945174935986, 0.657054999295]  permanova.1
permanova.1.5   [0.955786238698, 0.982314495479, 0.680046155172]  permanova.1
permanova.1.6   [0.820299895371, 0.927566324963, 0.612687450998]  permanova.1
permanova.1.7    [0.59100347582, 0.835524807958, 0.644290678641]  permanova.1
permanova.1.8   [0.360015384122, 0.716186099193, 0.665513284066]  permanova.1
permanova.1.10  [0.814148415537, 0.219684737031, 0.304805855541]  permanova.1
permanova.1.11  [0.933025763315, 0.391311037774, 0.271972331931]  permanova.1
permanova.1.12   [0.981776240994, 0.607381790876, 0.34579008993]  permanova.1
permanova.1.13  [0.994694348644, 0.809227231671, 0.486966571387]  permanova.1
permanova.1.14  [0.998231449548, 0.945174935986, 0.657054999295]  permanova.1
permanova.1.15  [0.955786238698, 0.982314495479, 0.680046155172]  permanova.1
permanova.1.16  [0.820299895371, 0.927566324963, 0.612687450998]  permanova.1
permanova.1.17   [0.59100347582, 0.835524807958, 0.644290678641]  permanova.1
permanova.1.18  [0.360015384122, 0.716186099193, 0.665513284066]  permanova.1
permanova.1.20  [0.814148415537, 0.219684737031, 0.304805855541]  permanova.1
permanova.1.21  [0.933025763315, 0.391311037774, 0.271972331931]  permanova.1
permanova.1.22   [0.981776240994, 0.607381790876, 0.34579008993]  permanova.1
permanova.1.23  [0.994694348644, 0.809227231671, 0.486966571387]  permanova.1
permanova.1.24  [0.998231449548, 0.945174935986, 0.657054999295]  permanova.1
permanova.1.25  [0.955786238698, 0.982314495479, 0.680046155172]  permanova.1
permanova.1.26  [0.820299895371, 0.927566324963, 0.612687450998]  permanova.1
permanova.1.27   [0.59100347582, 0.835524807958, 0.644290678641]  permanova.1
permanova.1.28  [0.360015384122, 0.716186099193, 0.665513284066]  permanova.1
permanova.2.0   [0.814148415537, 0.219684737031, 0.304805855541]  permanova.2
permanova.2.1   [0.933025763315, 0.391311037774, 0.271972331931]  permanova.2
permanova.2.2    [0.981776240994, 0.607381790876, 0.34579008993]  permanova.2
...                                                          ...          ...
permanova.3.22   [0.981776240994, 0.607381790876, 0.34579008993]  permanova.3
permanova.3.23  [0.994694348644, 0.809227231671, 0.486966571387]  permanova.3
permanova.3.24  [0.998231449548, 0.945174935986, 0.657054999295]  permanova.3
permanova.3.25  [0.955786238698, 0.982314495479, 0.680046155172]  permanova.3
permanova.3.26  [0.820299895371, 0.927566324963, 0.612687450998]  permanova.3
permanova.3.27   [0.59100347582, 0.835524807958, 0.644290678641]  permanova.3
permanova.4.0   [0.814148415537, 0.219684737031, 0.304805855541]  permanova.4
permanova.4.1   [0.933025763315, 0.391311037774, 0.271972331931]  permanova.4
permanova.4.2    [0.981776240994, 0.607381790876, 0.34579008993]  permanova.4
permanova.4.3   [0.994694348644, 0.809227231671, 0.486966571387]  permanova.4
permanova.4.4   [0.998231449548, 0.945174935986, 0.657054999295]  permanova.4
permanova.4.5   [0.955786238698, 0.982314495479, 0.680046155172]  permanova.4
permanova.4.6   [0.820299895371, 0.927566324963, 0.612687450998]  permanova.4
permanova.4.7    [0.59100347582, 0.835524807958, 0.644290678641]  permanova.4
permanova.4.10  [0.814148415537, 0.219684737031, 0.304805855541]  permanova.4
permanova.4.11  [0.933025763315, 0.391311037774, 0.271972331931]  permanova.4
permanova.4.12   [0.981776240994, 0.607381790876, 0.34579008993]  permanova.4
permanova.4.13  [0.994694348644, 0.809227231671, 0.486966571387]  permanova.4
permanova.4.14  [0.998231449548, 0.945174935986, 0.657054999295]  permanova.4
permanova.4.15  [0.955786238698, 0.982314495479, 0.680046155172]  permanova.4
permanova.4.16  [0.820299895371, 0.927566324963, 0.612687450998]  permanova.4
permanova.4.17   [0.59100347582, 0.835524807958, 0.644290678641]  permanova.4
permanova.4.20  [0.814148415537, 0.219684737031, 0.304805855541]  permanova.4
permanova.4.21  [0.933025763315, 0.391311037774, 0.271972331931]  permanova.4
permanova.4.22   [0.981776240994, 0.607381790876, 0.34579008993]  permanova.4
permanova.4.23  [0.994694348644, 0.809227231671, 0.486966571387]  permanova.4
permanova.4.24  [0.998231449548, 0.945174935986, 0.657054999295]  permanova.4
permanova.4.25  [0.955786238698, 0.982314495479, 0.680046155172]  permanova.4
permanova.4.26  [0.820299895371, 0.927566324963, 0.612687450998]  permanova.4
permanova.4.27   [0.59100347582, 0.835524807958, 0.644290678641]  permanova.4

[102 rows x 13 columns]
```

```python
>>> all_powers['z_effect'] = all_powers.apply(summarize.calc_z_effect, axis=1)
...
>>> def clean_effect(x):
...     """Cleans up the effect size calculation"""
...     if ((10 < x['counts']) & (0.1 < x['empirical']) & (x['empirical'] < 0.95)):
...         return x['z_effect']
...     else:
...         return np.nan
...
>>> all_powers['z_clean'] = all_powers.apply(clean_effect, axis='columns')
```

We'll calculate the grouped effects for the tests, along with a confidence interval.

```python
>>> effects = pd.concat([
...         all_powers.groupby('sim_id').first()[['test', 'statistic', 'alpha_adj', 'sim_num']],
...         all_powers.groupby('sim_id').count()[['z_clean']].rename(columns={'z_clean': 'param_count'}),
...         all_powers.groupby('sim_id').mean()[['z_clean']].rename(columns={'z_clean': 'param_mean'}),
...         all_powers.groupby('sim_id').std()[['z_clean']].rename(columns={'z_clean': 'param_std'}),
...     ], axis=1)
```

We'll also exclude any values which were predicted with fewer than 3 observations, or where the calibration parameter is less than 0.2.

```python
>>> effects.loc[effects['param_count'] <= 3, ['param_mean', 'param_std']] = np.nan
>>> effects.loc[effects['param_mean'] < 0.2] = np.nan
```

We'll now use the average effect for a test to predict the new power.

```python
>>> def predict_power_from_sim(x, effects):
...     """Predicts the statisical power based on the mean effect"""
...     param = effects.loc[x['sim_id'], 'param_mean']
...     return z_power(x['counts'], param, x['alpha'])
```

```python
>>> all_powers['predicted'] = all_powers.apply(partial(predict_power_from_sim, effects=effects), axis=1)
```

Finally, let's evaluate the comparison.

```python
>>> pe_fig = plot.summarize_regression(all_powers.loc[all_powers['counts'] > 10],
...                                    test_names=tests,
...                                    titles=titles,
...                                    x='empirical',
...                                    y='predicted',
...                                    gradient='colors',
...                                    alpha=0.1,
...                                    ylim=[-0.2, 0.2],
...                                    ylabel='Predicted'
...                                    )
>>> # pe_fig.axes[7].set_xlabel('Empirical Power')
... plot.add_labels(pe_fig.axes)
```

We have therefore demonstrate the performance of the calibration parameter on both parametric and nonparametric data.

We will now apply this method to a meta analysis performed on real studies.
