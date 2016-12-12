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
>>> tests = ['lognormal', 'uniform', 'permanova', 'mantel']
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
...                                'input_dir': './simulations/power/permanova/',
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
...     for i in range(5):
...         # Loads through the power simulation for the round
...         power_fp = os.path.join(power_dir, 'simulation_%i.p' % i)
...
...         with open(power_fp, 'rb') as f_:
...             sim = pickle.load(f_)
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
