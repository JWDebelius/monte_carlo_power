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

## Rank sum test

Data from lognormal and uniform distributions are not aspymototically normal, and therefore cannot easily be tested using a normal distribution. Therefore, a rank-sum distribution may be a more appropriate test. We use a Mann Whitney U test, which ranks the observations from each group to determine if there is a difference in the ranks.

### Log Normal Distribution

We will simulate log normal data, which has a long right tail, using means between 0 and 10, standard deviations between 5 and 15, and sample sizes between 60 and 100.

```python
>>> _, data = sim.simulate_lognormal([0, 3], [0.25, 2], 100)
```

```python
>>> ax1 = plt.subplot(3, 1, 1)
>>> ax2 = plt.subplot(3, 1, 2, sharex=ax1, sharey=ax1)
...
>>> sn.distplot(data[0], ax=ax1, bins=np.arange(-5, 10, 2.5), color='k')
>>> sn.distplot(data[1], ax=ax2, bins=np.arange(-5, 10, 2.5), color='k')
...
>>> ax1.set_xlim([-25, 150])
...
>>> clean_up_axis(ax1)
...
>>> distributions['lognormal'] = {'function': sim.simulate_lognormal,
...                               'kwargs': {'mu_lim': [0, 10],
...                                          'sigma_lim': [5, 15],
...                                          'count_lim': [60, 100]}
...                               }
```

## Rank sum Uniform Distribution

We also chose to compare uniform distributions using a rank-sum test.

```python
>>> _, data = sim.simulate_uniform([5, 20], [3, 10], [60, 100])
```

```python
>>> ax1 = plt.subplot(2, 1, 1)
>>> ax2 = plt.subplot(2, 1, 2, sharey=ax1, sharex=ax1)
...
>>> sn.distplot(data[0], ax=ax1, bins=np.arange(-10, 31, 2.5), hist_kws={'alpha': 0.2}, color='k')
>>> sn.distplot(data[1], ax=ax2, bins=np.arange(-10, 31, 2.5), hist_kws={'alpha': 0.2}, color='k')
>>> clean_up_axis(ax1)
...
>>> distributions['uniform'] = {'function': sim.simulate_uniform,
...                               'kwargs': {'range_lim': [5, 25],
...                                          'delta_lim': [3, 10],
...                                          'counts_lim': [60, 100]}
...                               }
```

## PERMANOVA on distance matrix

Distance matrices are a common data type in microbiome research. However, observations in distance matrices are not independent, and therefore, violate assumptions of a parametric test. While multiple tests exist for distance matrices, include PERMANOVA [<a href="#1">1</a>] and ANOSIM [<a href="#3">3</a>]. We've chosen to focus on PERMANOVA, here. With these tests, we apply a hypothesis test

$\begin{align}
\textbf{H}_{0}\textrm{ }d_{i} = d_{j}\textrm{ }\forall\textrm{ }i,\textrm{ }j;\textrm{ }i\textrm{ }\neq\textrm{ }j\\
\textbf{H}_{1}\textrm{ }d_{i}\textrm{ }\neq d_{j}\textrm{ }\exists\textrm{ }i,\textrm{ }j;\textrm{ }i\textrm{ }\neq\textrm{ }j
\end{align}\tag{4}$

To do this, we'll simulate a feature x observation table using a zero-inflated negative binomial model [[4](#4)]. We'll then calculate the distance between the samples using that model and a common microbial distance metric.

```python
>>> _, (feat_table, grouping) = sim.simulate_feature_table(n_lim=[1, 50],
...                                                        p_lim=[0.01, 0.6],
...                                                        psi_lim=[0.25, 0.75],
...                                                        num_observations=[99, 101],
...                                                        percent_different=0.05)
```

We can visualize the simulations by rarifying the feature table to 5000 counts, calculating the bray-curtis distance between the samples, and then performing a principle coordinates analysis.

```python
>>> rare_table = beta.subsample_features(feat_table.values, 5000, bootstrap=False)
>>> dm = skbio.DistanceMatrix.from_iterable(rare_table,
...                                         metric=scipy.spatial.distance.braycurtis,
...                                         keys=feat_table.index)
>>> pc = skbio.stats.ordination.pcoa(dm)
```

```python
>>> def color_scatter(x):
...     """Colors the scatter plot"""
...     if x == 0:
...         return sn.color_palette()[0]
...     else:
...         return sn.color_palette()[2]
...
>>> ord_ = pc.samples[['PC1', 'PC2']]
>>> ord_['color'] = grouping.apply(color_scatter)
...
...
>>> blues = cm.Blues_r
>>> blues.set_under('k')
```

```python
>>> fig = plt.figure()
>>> ax1 = fig.add_subplot(1, 3, 1)
>>> # ax1.set_aspect()
... ax2 = fig.add_subplot(1, 3, 2)
>>> # ax2.set_aspect('equal')
... ax3 = fig.add_subplot(1, 3, 3)
>>> # ax1_p = fig.add_subplot(6, 3, 1)
...
... sn.heatmap(rare_table.T, vmin=1, cmap=blues, ax=ax1)
>>> ax1.set_yticks([-1])
>>> ax1.set_xticks([-1])
...
>>> sn.heatmap(dm.data, ax=ax2)
...
>>> ax3.scatter(ord_['PC1'], ord_['PC2'], color=ord_['color'], alpha=0.5)
>>> ax3.set_aspect('equal')
```

```python
>>> blues = cm.Blues
>>> blues.set_under([0.5, 0.5, 0.5])
...
>>> ax = sn.heatmap(feat_table.T, vmin=1, cmap=blues)
>>> ax.set_xticks([-1])
>>> ax.set_yticks([-1])
[<matplotlib.axis.YTick at 0x111165978>]
```

```python
>>> fig, [ax1, ax2] = plt.subplots(1, 2)
>>> ax1.set_aspect('equal')
>>> sn.heatmap(dm.data, ax=ax1)
>>> ax1.set_yticks([])
>>> yt = ax1.set_xticks([])
...
>>> ax2.scatter(ord_['PC1'], ord_['PC2'], color=ord_['color'], alpha=0.5)
>>> ax2.set_aspect('equal')
```

```python
>>> skbio.stats.distance.permanova(dm, grouping, permutations=99)
method name               PERMANOVA
test statistic name        pseudo-F
sample size                     198
number of groups                  2
test statistic              6.69025
p-value                        0.01
number of permutations           99
Name: PERMANOVA results, dtype: object
```

We'll build our simulations containing between 120 and 200 observations (60 to 100 observations per group), with the size of the second group selected using a binomial distribution with probability 0.5. The number of groups in the second dataset is selected with a binomial distribution. THe within group distances will be between 0.3 and 0.6, the variance between 0.5 and 0.8, while the between distances will be between 0.45 and 0.65 with the variance in 0.5 and 0.8.

```python
>>> distributions['permanova'] = {'function': sim.simulate_feature_table,
...                               'kwargs': {'n_lim': [1, 50],
...                                          'p_lim': [0.01, 0.6],
...                                          'psi_lim': [0.3, 0.98],
...                                          'num_observations': [60, 100],
...                                          'percent_different': [0.0005, 0.0125]}
...                               }
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

## Distance Correlation

With distance matrices, once again, the assumption of independence is not appropriate. The Mantel test [<a href="#2">2</a>] is a common tests for continuous matrices. Statistical power methods have not been developed for many of these techniques.

We're again testing the hypothesis that there is a relationship between the two distance matrices, which we model as a slope. So, again, we're looking at the relationship between the two matrices.

```python
>>> p, [x, y] = sim.simulate_mantel([1, 5], [-2, 2], [2, 10], [100, 101], x_lim=[-10, 10])
...
>>> x_prime = pd.DataFrame(x.data, index=x.ids, columns=x.ids)
>>> x_order = x_prime.mean().sort_values().index
```

```python
>>> fig = plt.figure()
>>> ax1 = fig.add_subplot(1, 3, 1)
>>> ax2 = fig.add_subplot(1, 3, 2, sharex=ax1, sharey=ax1)
>>> ax3 = fig.add_subplot(1, 3, 3)
>>> fig.set_size_inches(9.5, 3)
>>> sn.heatmap(x.filter(x_order).data, ax=ax1, cmap='Greys', vmax=20)
>>> sn.heatmap(y.filter(x_order).data, ax=ax2, cmap='Reds', vmax=80)
>>> ax1.axis('equal')
>>> ax1.set_xticks([])
>>> ax1.set_yticks([])
>>> ax2.axis('equal')
>>> ax2.set_xticks([])
>>> ax2.set_yticks([])
>>> ax2.set_position((0.36, 0.125, 0.19, 0.775))
>>> fig.axes[4].set_position((0.56, 0.125, 0.3, 0.775))
>>> ax3.plot(x.condensed_form(), y.condensed_form(), 'ko', alpha=0.05)
>>> ax3.set_aspect(20/70)
>>> m, b, _, _, _ = scipy.stats.linregress(x.condensed_form(), y.condensed_form())
>>> ax3.plot(np.linspace(0, 20, 100), np.linspace(0, 20, 100) * m + b, 'r-')
>>> sn.despine(ax=ax3)
```

We'll simulate the data using similar parameters as we used previously. Slope between 1 and 5, intercept between -2 and 2, standard deviation between 5 and 40, and between 60 and 100 obsservations.

```python
>>> distributions['mantel'] = {'function': sim.simulate_mantel,
...                            'kwargs': {'slope_lim': [1, 5],
...                                       'intercept_lim': [-2, 2],
...                                       'sigma_lim': [5, 40],
...                                       'count_lim': [60, 100],
...                                       'x_lim': [-10, 10],
...                                       }
...                            }
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
