# Curve fitting parameter
In this notebook, we'll use the empirical power we calculated in [*Notebook*] to power we predict using [equation]. This will be used to generate Figure XXX.

Let's start by importing the packages we'll need for this analysis.

```python
>>> import os
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
>>> from machivellian.effects import z_power
...
>>> % matplotlib inline
>>> sn.set_style('ticks')
```

In the previous notebook, we summarized the power into a long-form dataframe. Let's import that dataframe and begin working.

```python
>>> all_powers = pd.read_csv('./simulations/parametric_power_summary.txt', sep='\t')
>>> all_powers.set_index('index', inplace=True)
```

We'll look at all parametric tests.

```python
>>> tests = ['ttest_1', 'ttest_ind', 'anova_3', 'anova_8', 'correlation']
>>> titles = ['One Sample\nT Test', 'Independent Sample\n T Test', 'One way ANOVA\n(3 groups)',
...           'One way ANOVA\n(8 groups)', 'Correlation']
```

The colors were read in as strings, so we need to convert them to the lists which will be used to display the actual color values.

```python
>>> def clean_up_colors(color_str):
...     color_str = color_str.replace('[', '').replace(']', '')
...     colors = [float(c) for c in color_str.split(', ')]
...     return colors
```

```python
>>> all_powers['colors'] = all_powers['colors'].apply(clean_up_colors)
```

Now, let's calculate the curve fitting parameter. The `summarize` module wraps the `effects.z_effect` function to make it easier to apply to the long form dataframe we're using with these simulations.

```python
>>> all_powers['z_effect'] = all_powers.apply(summarize.calc_z_effect, axis=1)
```

We can also establish some boundary conditions. Power is based on a cumulative distribution function (CDF). For CDFs, as $x \rightarrow -\infty$, $\textrm{CDF}(x) \rightarrow 0$ and and $x \rightarrow \infty$, $\textrm{CDF}(x) \rightarrow 1$. This means that we cannot estimate effect sizes well as power approaches 0 or 1. We also find that it's hard to get an accurate effect size estimate when B($n, \alpha$) < 0.1 or B($n, \alpha$) > 0.95. We also saw a high degree of variance and poor performance of the empirical estimate for small resample sizes ($n$ = 5), so we'll add the boundary data $n \geq 10$.

```python
>>> def clean_effect(x):
...     """Cleans up the effect size calculation"""
...     if ((10 < x['counts']) & (0.1 < x['empirical']) & (x['empirical'] < 0.95)):
...         return x['z_effect']
...     else:
...         return np.nan
>>> all_powers['z_clean'] = all_powers.apply(clean_effect, axis='columns')
```

For each test, we'll take the average of the cleaned effect size, and calculate the mean, variance, and number of points which contribute to the result.

```python
>>> effects = pd.concat([
...         all_powers.groupby('sim_id').first()[['test', 'statistic', 'alpha_adj', 'sim_num']],
...         all_powers.groupby('sim_id').count()[['z_clean']].rename(columns={'z_clean': 'param_count'}),
...         all_powers.groupby('sim_id').mean()[['z_clean']].rename(columns={'z_clean': 'param_mean'}),
...         all_powers.groupby('sim_id').std()[['z_clean']].rename(columns={'z_clean': 'param_std'}),
...     ], axis=1)
```

We'll remove any effect calculated from less than 3 datapoints, since the variability of these is expected to be very high.

```python
>>> effects.loc[effects['param_count'] <= 3, ['param_mean', 'param_std']] = np.nan
>>> effects.loc[effects['param_mean'] < 0.1] = np.nan
```

```python
>>> effects['param_ci'] = (effects['param_std'] / np.sqrt(effects['param_count']) *
...                         scipy.stats.t.ppf(1-0.025, df=effects['param_count'] - 1))
/Users/jdebelius/miniconda2/envs/power_play3/lib/python3.5/site-packages/scipy/stats/_distn_infrastructure.py:868: RuntimeWarning: invalid value encountered in greater
  cond = logical_and(cond, (asarray(arg) > 0))
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

Let's evaluate the over-all fit for each simulation by the test.

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
>>> pe_fig.axes[7].set_xlabel('Empirical Power')
>>> plot.add_labels(pe_fig.axes)
```

Let's also compare the predicted fit against the distribution-based power.

```python
>>> tp_fig = plot.summarize_regression(all_powers.loc[(all_powers['counts'] > 10) &
...                                                   (all_powers['sim_position'] < 10)],
...                                    test_names=tests,
...                                    titles=titles,
...                                    x='traditional',
...                                    y='predicted',
...                                    gradient='colors',
...                                    alpha=0.25,
...                                    ylim=[-0.2, 0.2],
...                                    ylabel='Predicted'
...                                    )
>>> tp_fig.axes[7].set_xlabel('Distribution Power')
>>> plot.add_labels(tp_fig.axes)
```

Let's save the calibration parameter calculated for each experiment.

```python
>>> effects.to_csv('./simulations/parametric_effect_summary.txt', sep='\t', index_label='sim_id')
```

We've now demonstrated that power can be accurately predicted for parametric distributions. Next, we'll look at the power values under an alternate sample size and critical value.
