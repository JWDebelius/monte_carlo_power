We're going to use this notebook to simulate data sets we'll use moving forward. Our analyses will examine two major types of tests: parametric and permutative for categorical and continuous data. Our parametric tests will include a one sample T test, independent sample T test, one way Analysis of Variance applied to three groups and eight groups, and a linear correlation. Permutative tests will include a PERMANOVA [<a href="#1">1</a>] and Mantel [<a href="#2">2</a>] test to look for relationships in distance matrices. The parameters for each simulation will be selected at random from a set of limits, as described below.

Subsequent notebooks will then calculate power for the data sets using Monte Carlo Simulation and by fitting the simulated curves.

```python
>>> import os
>>> import pickle
>>> import warnings
>>> warnings.filterwarnings('ignore')
...
>>> import matplotlib.cm as cm
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> import pandas as pd
>>> import scipy
>>> import skbio
>>> import seaborn as sn
...
>>> import machivellian.beta as beta
>>> import machivellian.simulate as sim
...
>>> % matplotlib inline
>>> sn.set_style('ticks')
```

# Setting up the simulations

We'll set a random seed so the simulation results are consistent.

```python
>>> np.random.seed(25)
```

The simulations will be saved in a directory under the current on called simulations. You can change this directory, but if you choose to do so, you'll need to update the directory in subsequent notebooks.

```python
>>> sim_location = './simulations/data/'
>>> if not os.path.exists(sim_location):
...     os.makedirs(sim_location)
```

We'll perform 100 simulations for each data type.

```python
>>> num_rounds = 100
```

And, we'll track the information for each of the simulations we plan to build and then build all thes simulations at once.

```python
>>> distributions = {}
```

## Helper Functions

We'll define a helper function which will allow us to easily retrieve functions associated with the simulation.

```python
>>> def retrieve_test(simulation_type):
...     """The simulation function, test, and simulation parameters"""
...     simulation = distributions[simulation_type]['function']
...     kwargs = distributions[simulation_type]['kwargs']
...
...     return simulation, kwargs
```

We will also design a helper function to simplify the appearance of distribution axes.

```python
>>> def clean_up_axis(ax):
...     """Formats the axis"""
...     ax.set_yticks([])
...     ax.set_yticklabels('')
...     ax.set_xticklabels('')
...
...     sn.despine(left=True, right=True, top=True, offset=10)
```

## One Sample T test

We'll start by simulating data for a one sample t test. This test checks an observation is drawn from a sample. We are testing the alternatively hypotheses,

$\begin{matrix}
\textbf{H}_{0} & x = \bar{x} \\
\textbf{H}_{1} & x \neq \bar{x}\\
\end{matrix} \tag{1}$

It is assumed the sample is asymptotically normal.

```python
>>> _, data = sim.simulate_ttest_1(mu_lim=[5, 6],
...                             sigma_lim=[4, 6],
...                             count_lim=[100, 101])
```

```python
>>> ax = plt.subplot(3, 1, 1)
>>> sn.distplot(data, ax=ax, bins=np.arange(-10, 31, 2.5), hist_kws={'alpha': 0.2}, color='k')
>>> ylim = ax.get_ylim()
>>> ax.plot([0, 0], [0, 0.125], 'k-')
>>> ax.set_ylim(ylim)
>>> clean_up_axis(ax)
```

We'll simulate the data using a random normal distribution. The means for the distribution will be between 5 and 10, the standard deviations between 5 and 8, and the sample size between 60 and 100 observations.

```python
>>> distributions['ttest_1'] = {'function': sim.simulate_ttest_1,
...                             'kwargs': {'mu_lim': [2, 10],
...                                        'sigma_lim': [5, 8],
...                                        'count_lim': [60, 100]}
...                             }
```

## Two Sample Independent T test

We'll also simulate data for a two sample T test.

The case II t test is a test for two independent samples, to determine if the samples are drawn from two different normal distributions.

$\begin{matrix}
\textbf{H}_{0} & \bar{x}_{1} = \bar{x}_{2} \\
\textbf{H}_{1} & \bar{x}_{1} \neq \bar{x}_{2}\\
\end{matrix} \tag{2}$

```python
>>> _, data = sim.simulate_ttest_ind(mu_lim=[0, 6], sigma_lim=[4, 6], count_lim=[100, 101])
```

```python
>>> ax1 = plt.subplot(2, 1, 1)
>>> ax2 = plt.subplot(2, 1, 2, sharey=ax1, sharex=ax1)
...
>>> sn.distplot(data[0], ax=ax1, bins=np.arange(-10, 31, 2.5), hist_kws={'alpha': 0.2}, color='k')
>>> sn.distplot(data[1], ax=ax2, bins=np.arange(-10, 31, 2.5), hist_kws={'alpha': 0.2}, color='k')
...
>>> ax1.set_ylim([0, 0.125])
>>> ax1.set_yticks([-1])
>>> ax1.set_xticklabels('')
>>> clean_up_axis(ax)
...
>>> ax.set_position((0.125, 0.25, 0.75, 0.4))
```

We'll set up simulations with means between 0 and 10, with standard deviations between 5 and 15, and sample sizes between 60 and 100. The simulated distributions will have the same number of observations per sample, although this will vary between simulations.

```python
>>> distributions['ttest_ind'] = {'function': sim.simulate_ttest_ind,
...                               'kwargs': {'mu_lim': [0, 10],
...                                          'sigma_lim': [5, 15],
...                                          'count_lim': [60, 100]}
...                               }
```

## One way ANOVA

A one-way Analysis of Variance (ANOVA) is typically  used to compare a set of multiple distributions ($n \geq 3$), to determine if one or more means are different.

$\begin{matrix}
\textbf{H}_{0} & \bar{x}_{1} = \bar{x}_{2} = ... \bar{x}_{k} & \\
\textbf{H}_{1} & \bar{x}_{i} \neq \bar{x}_{j} &\textrm{ }\exists\textrm{ }i,j\textrm{ } \epsilon\textrm{ }\left [1,\textrm{ } k\right ],\textrm{ }i\textrm{ }\neq\textrm{ }j
\end{matrix} \tag{3}$

We'll once again simulate normal distributions, to compare the data.

```python
>>> _, data = sim.simulate_anova(mu_lim=[0, 6], sigma_lim=[4, 6], count_lim=[100, 101], num_pops=3)
```

```python
>>> ax1 = plt.subplot(3, 1, 1)
>>> ax2 = plt.subplot(3, 1, 2, sharey=ax1, sharex=ax1)
>>> ax3 = plt.subplot(3, 1, 3, sharex=ax1, sharey=ax1)
>>> sn.distplot(data[0], ax=ax1, color='k', hist_kws={'alpha': 0.2})
>>> sn.distplot(data[1], ax=ax2, color='k', hist_kws={'alpha': 0.2})
>>> sn.distplot(data[2], ax=ax3, color='k', hist_kws={'alpha': 0.2})
...
>>> clean_up_axis(ax1)
>>> clean_up_axis(ax2)
>>> clean_up_axis(ax3)
```

We'll simulate two sets of distributions for ANOVA: a 3 sample and an 8 sample ANOVA. In both cases, we'll have means between 0 and 10, standard deviations between 5 and 15, and we'll once again have between 60 and 100 observations per sample, although the samples will be the same size.

```python
>>> distributions['anova_3'] = {'function': sim.simulate_anova,
...                             'kwargs': {'mu_lim': [0, 10],
...                                        'sigma_lim': [5, 15],
...                                        'count_lim': [60, 100],
...                                        'num_pops': 3}
...                             }
>>> distributions['anova_8'] = {'function': sim.simulate_anova,
...                             'kwargs': {'mu_lim': [0, 10],
...                                        'sigma_lim': [5, 15],
...                                        'count_lim': [60, 100],
...                                        'num_pops': 8}
...                             }
>>> distributions['anova_20'] = {'function': sim.simulate_anova,
...                              'kwargs': {'mu_lim': [0, 10],
...                                         'sigma_lim': [5, 15],
...                                         'count_lim': [60, 100],
...                                         'num_pops': 20}
...                              }
```

# Continous Distributions

Many biological phenomena are characterized by continuous, rather than discrete variables. Therefore, we're going to explore the performance of the method with continuous data.

## Univariate Correlation

We'll start by looking at simple, linear correlation. These equations most often follow the format of 
$\begin{align}
y = mx + b + \epsilon
\end{align}\tag{5}$

Here, we'll focus on the relationship between the variables, and check that there is a relationship between the

$\begin{align}
\textbf{H}_{0}: m = 0\\
\textbf{H}_{1}: m \neq 0\\
\end{align}\tag{6}$

```python
>>> [s, n, m, b], [x, y] = sim.simulate_correlation([1, 5], [-2, 2], [25, 50], [100, 101], [-20, 20])
>>> ax = plt.axes()
>>> ax.plot(x, y, 'ko', alpha=0.5)
>>> ax.plot(np.arange(-20, 21, 1), m * np.arange(-20, 21,) + b, 'r-')
>>> ax.set_xlim
>>> sn.despine(offset=10)
```

We'll simulate distance matrices with slopes between 1 and 5, intercepts between -2 and 2, and standard deviations between 25 and 50. The wide range of standard deviations help insure appropriate effect sizes.

```python
>>> distributions['correlation'] = {'function': sim.simulate_correlation,
...                                 'kwargs': {'slope_lim': [1, 5],
...                                            'intercept_lim': [-2, 2],
...                                            'sigma_lim': [25, 50],
...                                            'count_lim': [60, 100],
...                                            'x_lim': [-20, 20]}
...                                  }
```

# Simulates the data

We'll next simulate the data, which will be used in subsequent notebooks.

```python
>>> for test_name in distributions.keys():
...     # Gets the simulation function, test, and arguments
...     simulation, kwargs = retrieve_test(test_name)
...     if not os.path.exists(os.path.join(sim_location, test_name)):
...         os.makedirs(os.path.join(sim_location, test_name))
...     distributions[test_name]['p-values'] = []
...     # Permforms the simulations
...     for i in range(num_rounds):
...         file = os.path.join(sim_location, '%s/simulation_%i.p' % (test_name, i))
...         params, samples = simulation(**kwargs)
...         with open(file, 'wb') as f_:
...             pickle.dump({'samples': samples, 'params': params}, f_)
```

We've now simulated data following a variety of distributions. We'll perform power calculations using all the simulations in the next notebook: [2-Empirical Power Calculations](2-Empirical%20Power%20Calculations.md).

We'll then evaluate the parametric data starting in [](), and the nonparametric in []().

# Works Cited


<ol><li id="1">Anderson, M.A. (2001). A new method for non-parametric multivariate analysis of variance. <em>Austral Ecology</em>. <strong>26</strong>: 32 - 46.
</li><li id="2">Mantel, M. (1967). The detection of disease clustering and a generalized regression approach. <em>Cancer Reserach</em>. <stromg>27</strong>: 209-220.
</li><li id="3">Clarke, K.R. (1993). Non-parametric multivariate analyses of changes in community structure. <em>Austral Ecology</em>. <strong>18</strong>: 117-143.
</li><li id="4">Kurtz, Z.D.; Muller, C.L.; Miraldi, E.R.; Littman, D.R.; Blaser, M.J.; Bonneau, R.A. (2015). "<a href="https://www.ncbi.nlm.nih.gov/pubmed/25950956">Sparse and compositional robust interference of microbial ecological networks.</a>" <em>PLoS Computational Biology</em>. **11**:1004226.
</li></ol>
