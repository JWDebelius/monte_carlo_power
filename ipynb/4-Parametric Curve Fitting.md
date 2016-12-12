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
>>> from machivellian.effects import z_power, z_effect
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

Now, let's calculate the curve fitting parameter. We'll use a nonlinear curve fit ...

[curve fitting equation...]

For each test, we'll take the average of the cleaned effect size, and calculate the mean, variance, and number of points which contribute to the result.

```python
>>> effects = {}
>>> for sim_id, sim in all_powers.groupby('sim_id'):
...     d, sd, nd = z_effect(sim['counts'], sim['empirical'], sim['alpha'].mean(),
...                          size_lim=10)
...     effects[sim_id] = {'statistic': sim['statistic'].unique()[0],
...                        'param': d,
...                        'param_se': sd,
...                        'params_n': nd,
...                        'test': sim['test'].unique()[0],
...                        'alpha_adj': sim['alpha_adj'].unique()[0],
...                        'sim_num': sim['sim_num'].unique()[0],
...                        'colors': sn.color_palette()[0]
...                        }
...
>>> effects = pd.DataFrame.from_dict(effects, orient='index')
```

We'll now use the average effect for a test to predict the new power.

```python
>>> def predict_power_from_sim(x, effects):
...     """Predicts the statisical power based on the mean effect"""
...     param = effects.loc[x['sim_id'], 'param']
...     return z_power(x['counts'], param, x['alpha'])
```

```python
>>> all_powers['predicted'] = all_powers.apply(partial(predict_power_from_sim, effects=effects),
...                                            axis=1)
```

Let's also compare the predicted fit against the distribution-based power.

```python
>>> tp_fig = plot.summarize_regression(all_powers.loc[(all_powers['counts'] > 10)],
...                                    test_names=tests,
...                                    titles=titles,
...                                    x='traditional',
...                                    y='predicted',
...                                    gradient='colors',
...                                    alpha=0.25,
...                                    ylim=[-0.2, 0.2],
...                                    ylabel='Predicted'
...                                    )
>>> tp_fig.axes[2].set_xlabel('Distribution Power')
>>> plot.add_labels(tp_fig.axes)
```

Let's save the calibration parameter calculated for each experiment.

```python
>>> effects.to_csv('./simulations/parametric_effect_summary.txt', sep='\t', index_label='sim_id')
```

We've now demonstrated that power can be accurately predicted for parametric distributions. Next, we'll look at the power values under an alternate sample size and critical value.
