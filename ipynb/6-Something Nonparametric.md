# Nonparametric Monte Carlo Simulations

We explored the ability of the method to work with nonparametric data using cross validation. We simulated 4 types of data appropriate for testing with nonparametric tests, and estimated empirical power. Each empirical power value is assumed to be independent for sample sizes greater than 10 observations per group, since ...

```python
>>> import os
>>> import pickle
...
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> import pandas as pd
>>> import seaborn as sn
>>> import scipy
>>> import sklearn.model_selection as slm
...
>>> import machivellian.plot as plot
>>> import machivellian.summarize as summarize
...
>>> from machivellian.effects import z_power, z_effect
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

# 3. Power Calculations

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

..

```python
>>> all_powers = []
>>> titles = []
```

```python
>>> test_name = 'permanova'
...
>>> titles.append(distributions[test_name]['clean_name'])
>>> power_dir = distributions[test_name]['input_dir']
>>> return_fp = distributions[test_name]['return_fp']
```

```python
>>> if not os.path.exists(power_dir):
...     raise ValueError('%s does not exist' % power_dir)
```

```python
>>> summaries = []
```

```python
>>> i = 3
```

```python
>>> # Loads through the power simulation for the round
... power_fp = os.path.join(power_dir, 'simulation_%i.p' % i)
```

```python
>>> with open(power_fp, 'rb') as f_:
...     sim = pickle.load(f_)
```

```python
>>> sim
{'alpha': 0.05,
 'alpha_adj': 1.0,
 'counts': array([ 5, 15, 25, 35, 45, 55, 65]),
 'empirical': array([[ 0.54,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
        [ 0.58,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
        [ 0.57,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
        [ 0.59,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
        [ 0.57,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ]]),
 'original_p': 0.01,
 'original_size': 78,
 'permutations': 99,
 'statistic': None,
 'traditional': None,
 'traditional_prime': None}
```

```python
>>> sim['alpha_adj'] = 1
>>> sim['statistic'] = np.nan
```

```python
>>> eff, se, ne = z_effect(sim['counts'], sim['empirical'])
```

```python
>>> plt.plot(sim['counts'], sim['empirical'].T, 'o')
>>> plt.plot(np.linspace(0, 60, 100),
...          z_power(np.linspace(0, 60, 100), eff))
[<matplotlib.lines.Line2D at 0x1114defd0>]
```

```python

```
