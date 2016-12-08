# Parametric and Empirical Power Comparison

In the last two notebooks, we simulated data and calculated empirical power. Here, we'll compare the parametric power calculations, the empirical power values and the predicted power. This notebook will generate Figure S1.

```python
>>> import os
>>> import pickle
>>> import warnings
...
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> import pandas as pd
>>> import seaborn as sn
...
>>> import machivellian.plot as plot
>>> import machivellian.summarize as summarize
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
>>> tests = ['ttest_1', 'ttest_ind', 'anova_3', 'anova_8', 'correlation']
```

Finally, we'll create a set of parameters for each of the distributions being evaluated. This includes the clean name, which provides a prettier title in plots, the number of groups in the tested (2 for everything except the ANOVAs), and the location of both the input emperical power calculations and output summary tables.

```python
>>> distributions = {'ttest_1': {'clean_name': 'One Sample\nT Test',
...                                'input_dir': './simulations/power/ttest_1/',
...                                'return_fp': './simulations/extrapolation/ttest_1.txt'
...                                },
...                  'ttest_ind': {'clean_name': 'Independent Sample\n T Test',
...                                'input_dir': './simulations/power/ttest_ind',
...                                'return_fp': './simulations/extrapolation/ttest_ind.txt'
...                                },
...                  'anova_3': {'clean_name': 'One way ANOVA\n(3 groups)',
...                              'input_dir': './simulations/power/anova_3',
...                              'return_fp': './simulations/extrapolation/anova_3.txt'
...                              },
...                  'anova_8': {'clean_name': 'One way ANOVA\n(8 groups)',
...                              'input_dir': './simulations/power/anova_8',
...                              'return_fp': './simulations/extrapolation/anova_8.txt'
...                              },
...                  'correlation': {'clean_name': 'Correlation',
...                                  'input_dir': './simulations/power/correlation',
...                                  'return_fp': './simulations/extrapolation/correlation.txt'
...                                  },
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
...     for i in range(num_rounds):
...         # Loads through the power simulation for the round
...         power_fp = os.path.join(power_dir, 'simulation_%i.p' % i)
...
...         with open(power_fp, 'rb') as f_:
...             sim = pickle.load(f_)
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

# 4. Power Calculation

We're going to plot a comparison between the emperical and traditional power calculations, to see if there's a strong relationship. We'll plot the distribution-based power on the x axis and the emperical power on the y axis.

```python
>>> te_fig = plot.summarize_regression(all_powers,
...                                    test_names=tests,
...                                    titles=titles,
...                                    x='traditional',
...                                    y='empirical',
...                                    gradient='colors',
...                                    alpha=0.1,
...                                    ylim=[-0.2, 0.2],
...                                    ylabel='Empirical'
...                                    )
>>> te_fig.axes[7].set_xlabel('Distribution Power')
>>> plot.add_labels(te_fig.axes)
```

We see a high degree of correlation between the distribution-based power and the empirical power, although there is poor performance in the correlation with 5 observations per group.

Let's save the comparison dataframe, which we'll use to calculate the empirical power values in the [next notebook]().

```python
>>> all_powers.to_csv('./simulations/parametric_power_summary.txt',
...                   sep='\t')
```

We've demonstrated the ability of the empirical distribution to recapitulate the empirical distribution. Next, we'll loke at the the performance of the power prediction.
