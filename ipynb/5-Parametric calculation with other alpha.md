In the last notebook, we calculated a calibration parameter, $\delta$ for each monte carlo simulation and demonstrated that the $\bar{\delta}$ for an experiment could accurate estimate the power calculated for empirical values and the power based on the distribution.

Here, we'll evaluate the ability of the calibration parameter to estimate power with a different critical value. We'll compare this value to the distribution-based for the same sample size and effect.

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
>>> import machivellian.traditional as trad
>>> from machivellian.effects import z_power
...
>>> % matplotlib inline
>>> sn.set_style('ticks')
```

We'll calculate power for alternate alpha value: $\alpha$ = 0.01.

```python
>>> alpha = 0.01
```

We'll also use a different set of sample sizes than we used in the original study.

```python
>>> counts = np.arange(10, 151, 10)
```

We'll also look at all the tests.

```python
>>> tests = ['ttest_1', 'ttest_ind', 'anova_3', 'anova_8', 'correlation']
>>> titles = ['One Sample\nT Test', 'Independent Sample\n T Test', 'One way ANOVA\n(3 groups)',
...           'One way ANOVA\n(8 groups)', 'Correlation']
```

We'll load the per-study effect sizes from the data we generated in the previous notebook.

```python
>>> effects = pd.read_csv('./simulations/parametric_effect_summary.txt', sep='\t')
>>> effects.set_index('sim_id', inplace=True)
```

We'll also go back to the test simulations, and load the values we simulated using the alternate alpha value. These are stored in the files as `traditional_prime`.

```python
>>> power_pattern = './simulations/power/%s/simulation_%i.p'
...
>>> summaries = []
>>> for id_, row_ in effects.dropna().iterrows():
...     with open(power_pattern % (row_['test'], row_['sim_num']), 'rb') as f_:
...         sim = pickle.load(f_)
...     traditional = sim['traditional_prime']
...     effect = row_['param']
...     predicted = z_power(counts, effect, alpha=alpha * row_['alpha_adj'])
...     summary = pd.DataFrame(np.vstack([counts, traditional, predicted]),
...                            index=['counts', 'traditional', 'predicted']).T
...     summary['test'] = row_['test']
...     summary['sim_num'] = row_['sim_num']
...     summary['sim_id'] = id_
...     summary['colors'] = summary['test'].apply(lambda x: sn.color_palette()[0])
...
...     summaries.append(summary)
>>> power_prime = pd.concat(summaries)
```

```python
>>> tpp_fig = plot.summarize_regression(power_prime,
...                                     test_names=tests,
...                                     titles=titles,
...                                     x='traditional',
...                                     y='predicted',
...                                     gradient='colors',
...                                     alpha=0.25,
...                                     ylim=[-0.2, 0.2],
...                                     ylabel='Predicted'
...                                    )
>>> tpp_fig.axes[2].set_xlabel('Distribution Power')
>>> # plot.add_labels(pe_fig.axes)
... tpp_fig.savefig('/Users/jdebelius/Desktop/alt_p.pdf')
```

Although the performance here is somewhat worse than with the orginal critical value and prediction slightly underestimates the distribution-based power, especially for larger sample sizes, we still find a strong relationship between the distribution-based power and the predicted power.
