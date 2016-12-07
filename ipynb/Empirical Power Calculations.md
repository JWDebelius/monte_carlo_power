# Empirical Power Calculations

In the [last notebook](1-Build%20Simulations.md), we generated simulated data for five types of parametric tests: a one sample t test, an independent sample t test, a one-way ANOVA with three groups, a one-way ANOVA with eight groups, and a simple linear correlation, as well as simulating distance matrices for group-wise comparison and distance correlation.

We will calculate the the distribution-based power for parametric tests using equations described in *Statistical Power Analysis for the Social and Behavioral Sciences : Basic and Advanced Techniques* [<a href="#1">1</a>].

The notebook will take the form of building functions which will facilitate the power calculations, and then applying them. The orignal 
We ran this notebook using supercomputers, including the Knight Lab supercomputer and UC San Diego Jupyterhub. We recommend this notebook be used for review and not be run on a local computer. The estimated run time for serial processing is at least 6 hours, however may take longer depending on the speed of your system.

You can download the precalculated files from [].

This simulation will take three steps.

1. [**Set analysis parameters.**]()<br>
2. [**Build a database describing each test.**]()<br>For each test, we'll perform the following steps. 
    * Define the statistical test and test statistic
    * Effect size calculation for parametric tests. <br>*Note: This step is only applied to parametric tests*
    * Simulation extraction.
    * Update the database for testing.
3. [**Perform power calculations in parallel.**]()<br>This speeds up the behavior on a cluster-based system.

We will explore the following tests:

* [One Sample T Test]()
* [Independent Sample T test]()
* [One way ANOVA]()
* [Linear Correlation]()

```python
>>> import copy
>>> import os
>>> import pickle
>>> import warnings
...
>>> from functools import partial
>>> from multiprocessing import Pool
...
>>> import numpy as np
>>> import scipy.stats
>>> import skbio
...
>>> from machivellian.power import subsample_power
>>> from machivellian.bootstrap import bootstrap_mantel, bootstrap_permanova
>>> import machivellian.traditional as trad
...
>>> %matplotlib inline
```

```python
>>> warnings.filterwarnings('ignore')
```

# Simulation Parameters
In notebook 1, we simulated 100 distributions for each test.

```python
>>> num_rounds = 100
```

We assume that all the data associated with this study will be in a sub directory called `simulations`. The data is expected to be in a data directory; the power results will be saved in the `power` directory.

```python
>>> base_directory = './simulations/'
...
>>> sim_location = os.path.join(base_directory, 'data')
>>> if not os.path.exists(sim_location):
...     raise ValueError('The simulation directory does not exist.'
...                      'Go back and simulate some data!')
...
>>> power_location = os.path.join(base_directory, 'power')
>>> if not os.path.exists(power_location):
...     os.makedirs(power_location)
```

Power calculations are expensive, so users may prefer not to perform them if power has already been calculated. This can be specified with the `overwrite` parameter, which will prevent new power calculations if a power calculation has already been performed.

```python
>>> overwrite = False
```

This notebook is intended to be run in parallel, thus we'll set the number of threads. By default, we'll use 1. However, on a system with more threads, a larger number of processing steps can be performed to limit runtime.

```python
>>> num_cpus = 1
```

Power calculations will be performed using bootstrapping. We will calculate power with subsample sizes start with 5, up to 100 observations, counting by 10s.

```python
>>> counts = np.arange(5, 100, 10)
```

Each power calculation will be calculated using 100 tests, and we'll repeat the calculations 5 times per depth.

```python
>>> num_iter = 100
>>> num_runs = 5
```

We'll test using a critical value of 0.05, whcih is a commonly used value in biology.

```python
>>> alpha = 0.05
```

We will evaluate five parameteric distributions and two distance-based calculations.

```python
>>> distributions = {}
```

# Parametric Distributions
## One Sample T test

A case I t-test checks if an observation is drawn from a sample. We are testing the alternatively hypotheses,

$\begin{matrix}
\textbf{H}_{0} & x = \bar{x} \\
\textbf{H}_{1} & x \neq \bar{x}\\
\end{matrix} \tag{2.1}$

where $\bar{x}$ is the mean of the population, $x$ is the value being compared to the sample, $s$ is the standard devation of the sample, and there are $n$ observations in the sample.

### Test Statistic

The test statistic for the case I t test is given as
$t = \frac{(\bar{x} - x)\sqrt{n}}{s} \tag{2.2}$
The probability distribution follows a T distribution with $n-1$ degrees of freedom, where $n$ is the number of observations in the sample.


For the emperical test, we will use the `scipy.stats.ttest_1samp` function, which returns a p value.

```python
>>> def emp_ttest_1(sample, x0=0):
...     return scipy.stats.ttest_1samp(sample[0], x0)[1]
```

### Effect Size

The non centrality paramter for the statistic, $\lambda$ is given by
$\begin{align*}
\lambda &= \frac{t}{\sqrt{n}}\\
&=\frac{1}{\sqrt{n}}\left(\frac{(\bar{x} - x)\sqrt{n}}{s}\right )\\
&=\frac{(\bar{x} - x)}{s}
\end{align*}\tag{2.4}$

We will encorperate this in the power calculation. To allow for testing, the power calculation has been moved into a library.

### Simulation extraction

We need to extract the samples for the simulation, so we have appropriate samples and test keywords for the test. We're comparing out simulated values to a mean of 0, so we'll set this as a test argument.

```python
>>> def extract_ttest_1_samples(sim):
...     """Extracts the sample and test kwargs for 1 sample t test"""
...     samples = sim['samples']
...     test_kwargs = {'x0': 0}
...
...     return samples, test_kwargs
```

### Update distribution tracking

We're going to have to specify a test alpha for our power calculations. By default, scipy uses a two-tail t-test 
The quality of fit later will depending on using the correct alpha value. However, the effect size calculation is based on a one-tail test. So, we'll

```python
>>> distributions['ttest_1'] = {
...     'extraction': extract_ttest_1_samples,
...     'test': emp_ttest_1,
...     'statistic': partial(trad.effect_ttest_1, x0=0),
...     'traditional': partial(trad.calc_ttest_1, x0=0),
...     'power_kwargs': {},
...     'test_alpha':
...     'permutations': None
...     }
```

## 3.2 Independent T test

The case II t test is a test for two independent samples, to determine if the samples are drawn from different distributions.

$\begin{matrix}
\textbf{H}_{0} & \bar{x}_{1} = \bar{x}_{2} \\
\textbf{H}_{1} & \bar{x}_{1} \neq \bar{x}_{2}\\
\end{matrix} \tag{3.1}$

### 3.2.1 Test Statistic

There are several ways to calculate this t statistic, but we will operate on the assumption that the two populations have different variances, giving the most extensibe calculation of the test statistic. So,

$\begin{align*}
t &= \frac{\bar{x}_{1} - \bar{x}_{2}}{\sqrt{\frac{s_{1}^{2}}{n_{1}} + \frac{s_{2}^{2}}{n_{2}}}}\\
&= \frac{\bar{x}_{1} - \bar{x}_{2}}{\sqrt{\frac{n_{2}s_{1}^{2} + n_{1}s_{2}^{2}}{n_{1}n_{2}}}}
\end{align*}\tag{3.2}$

The t statistic follows a T distribution with $df$ degrees of freedom, where $df$ is given as
$df = \frac{(s_{1}^{2}/n_{1} + s_{2}^{2}/n_{2})^{2}}{(s_{1}^{2}/n_{1})^2/(n_{1}-1) + s_{2}^{2}/n_{2})^2/(n_{2}-1)} \tag{3.3}$

For the sake of simplicity, we'll assume that $n_{1} = n_{2}$, which allows us to redefine equation (2.1) as
$\begin{align*}
t &= \frac{(\bar{x}_{1} - \bar{x}_{2})}{\sqrt{\frac{s_{1}^{2}}{n} + \frac{s_{2}^{2}}{n}}}\\
&= \frac{\sqrt{n}(\bar{x}_{1} - \bar{x}_{2})}{\sqrt{s_{1}^{2} + s_{2}^{2}}}
\end{align*}\tag{3.4}$
which means the test statitic is now drawn from a t distribution with df degrees of freedom, where
df is defined as
$\begin{align*}
df &= \left (n-1 \right ) \left (\frac{\left (s_{1}^{2} + s_{2}^{2}  \right )^{2}}{\left (s_{1}^{2} \right)^{2} + \left (s_{2}^{2}  \right )^{2}} \right )\\
\end{align*}\tag{3.5}$

For the emperical test, we can use the `scipy.stats.ttest_ind` function, which will return a p value.

For the sake of simplicity, we'll assume that $n_{1} = n_{2}$, which allows us to redefine equation (2.1) as
$\begin{align*}
t &= \frac{(\bar{x}_{1} - \bar{x}_{2})}{\sqrt{\frac{s_{1}^{2}}{n} + \frac{s_{2}^{2}}{n}}}\\
&= \frac{\sqrt{n}(\bar{x}_{1} - \bar{x}_{2})}{\sqrt{s_{1}^{2} + s_{2}^{2}}}
\end{align*}\tag{3.4}$
which means the test statitic is now drawn from a t distribution with df degrees of freedom, where
df is defined as
$\begin{align*}
df &= \left (n-1 \right ) \left (\frac{\left (s_{1}^{2} + s_{2}^{2}  \right )^{2}}{\left (s_{1}^{2} \right)^{2} + \left (s_{2}^{2}  \right )^{2}} \right )\\
\end{align*}\tag{3.5}$

For the emperical test, we can use the `scipy.stats.ttest_ind` function, which will return a p value.

```python
>>> def emp_ttest_ind(samples):
...     sample1, sample2 = samples
...     return scipy.stats.ttest_ind(sample1, sample2, equal_var=False)[1]
```

### 3.2.2 Noncentrality Parameter

The effect size, non-centrality parameter, for an independent sample t test where samples are the same size is once again related to the test statistic as
$\begin{align*}
\lambda &= \frac{t}{\sqrt{n}}\\
&= \left (\frac{\sqrt{n} \left (\bar{x}_{1} - \bar{x}_{2} \right )}{\sqrt{s_{1}^{2} + s_{2}^{2}}} \right ) \left (\frac{1}{\sqrt{n}} \right )\\
&= \left (\frac{\bar{x}_{1}^{2} - \bar{x}_{2}^{2}}{\sqrt{s_{1}^{2} + s_{2}^{2}}} \right )
\end{align*}\tag{3.8}$

### Simulation extraction

```python
>>> def extract_ttest_ind_samples(sim):
...     """Extracts the sample and test kwargs for 1 sample t test"""
...     samples = sim['samples']
...     test_kwargs = {}
...
...     return samples, test_kwargs
```

```python
>>> distributions['ttest_ind'] = {
...     'extraction': extract_ttest_ind_samples,
...     'test': emp_ttest_ind,
...     'statistic': trad.effect_ttest_ind,
...     'traditional': trad.calc_ttest_ind,
...     'power_kwargs': {},
...     'permutations': None
...     }
```

## 3.3 One way Analysis of Variance

Assume there exist a set of samples, $\{S_{1}, S_{2}, ..., S_{k} \}$ where there are a total of $N$ observations distributed over the $k$ groups. The $i$th sample, $S_{i}$ contains $n_{i}$ observations, and has a mean of $\bar{x}_{.i}$ and a standard deviation, $s_{i}$ where

$\begin{align*}
s_{i} = \sqrt{\frac{\sum_{j=1}^{n}{\left (x_{ij} - \bar{x}_{.i} \right)^{2}}}{n_{i}-1}}
\end{align*}\tag{4.1}$

A one-way Analysis of Variance (ANOVA) tests that at least one sample mean in a set of three or more are not equal. Assume that

$\begin{matrix}
\textbf{H}_{0} & \bar{x}_{1} = \bar{x}_{2} = ... \bar{x}_{k} & \\
\textbf{H}_{1} & \bar{x}_{i} \neq \bar{x}_{j} & \exists i,j \epsilon [1, k], i \neq j
\end{matrix} \tag{4.2}$

### 3.3.1 Test Statistic

Assume there exist a set of samples, $\{S_{1}, S_{2}, ..., S_{k} \}$ where there are a total of $N$ observations distributed over the $k$ groups. The $i$th sample, $S_{i}$ contains $n_{i}$ observations, and has a mean of $\bar{x}_{.i}$ and a standard deviation, $s_{i}$ where

$\begin{align*}
s_{i} = \sqrt{\frac{\sum_{j=1}^{n}{\left (x_{ij} - \bar{x}_{.i} \right)^{2}}}{n_{i}-1}}
\end{align*}\tag{4.1}$

A one-way Analysis of Variance (ANOVA) tests that at least one sample mean in a set of three or more are not equal. Assume that

$\begin{matrix}
\textbf{H}_{0} & \bar{x}_{1} = \bar{x}_{2} = ... \bar{x}_{k} & \\
\textbf{H}_{1} & \bar{x}_{i} \neq \bar{x}_{j} & \exists i,j \epsilon [1, k], i \neq j
\end{matrix} \tag{4.2}$

The test statistic for ANOVA is given by
$\begin{align*}
F &= \frac{\frac{\textrm{SS}_{\textrm{between}}}{\textrm{DF}_{\textrm{between}}}}{\frac{\textrm{SS}_{\textrm{within}}}{\textrm{DF}_{\textrm{within}}}}
\end{align*}\tag{4.3}$
and test statistic is drawn from an $F$ distribution with $k - 1$ and $N - k$ degrees of freedom [[3](#Zar)].

For the emperical test, we can use the `scipy.stats.f_oneway` function, which will return a p value.

```python
>>> def emp_anova(samples):
...     return scipy.stats.f_oneway(*samples)[1]
```

### Noncentrality Parameter

Under the alternatively hypothesis, the non-centrality $F'$ is given by

$\begin{align*}
F' = \left(\frac{\textrm{SS}_{\textrm{between}}}{\textrm{SS}_{\textrm{within}}} \right) \left (\frac{\textrm{DF}_{\textrm{within}}}{\textrm{DF}_{\textrm{between}}}{} \right )
\end{align*}\tag{4.9}$

For a given pair of hypotheses, the noncentrality parameter is defined according to equation (2.4), where the grand mean can be substituted for the the test mean. The overall effect size is therefore given as
$\begin{align*}
\lambda &= \sum_{i=1}^{k}{\lambda_{i}^{2}}\\
&= \sum_{i=1}^{k}\left (\frac{\bar{x}_{i} - \bar{x}_{..}}{s_{i}} \right )^{2} 
\end{align*} \tag{4.10}$

### Simulation Extraction

```python
>>> def extract_anova_samples(sim):
...     """Extracts the sample and test kwargs for an ANOVA"""
...     samples = sim['samples']
...     test_kwargs = {}
...
...     return samples, test_kwargs
```

```python
>>> distributions['anova_3'] = {
...     'extraction': extract_anova_samples,
...     'test': emp_anova,
...     'statistic': trad.effect_anova,
...     'traditional': trad.calc_anova,
...     'power_kwargs': {},
...     'permutations': {},
...     }
...
>>> distributions['anova_8'] = {
...     'extraction': extract_anova_samples,
...     'test': emp_anova,
...     'statistic': trad.effect_anova,
...     'traditional': trad.calc_anova,
...     'power_kwargs': {},
...     'permutations': {},
...     }
```

## 3.4 Pearson's R

Pearson's correlation coeffecient looks for a linear one-to-one relationship between two vectors, $x$ and $y$, both of size $n$. Closely related vectors have a correlation coeffecient with an absloute value of 1, unrelated data have a correlation coeffecient of 0.

### 3.4.1 Test Statistic

The correlation coeffecient between the two vectors is given by
$\begin{align*}
r = \frac{\sum{xy}}{\sqrt{\sum{x^{2}}\sum{y^{2}}}}
\end{align*}\tag{5.1}$

We can test the hypotheses,
$\begin{matrix}
\textbf{H}_{0} & r = 0 \\
\textbf{H}_{1} & x \neq 0\\
\end{matrix} \tag{5.2}$
with a test statistic drawn from the $t$ distribution with $n - 2$ degrees of freedom. The statistic is calculated as
$\begin{align*}
t = \frac{r\sqrt{n-2}}{\sqrt{1 - r^{2}}}
\end{align*}\tag{5.3}$

Scipy's `scipy.stats.pearsonr` can calculate the correlation coeffecient *and* a p value for the coeffecient.

```python
>>> def emp_pearson(samples):
...     return scipy.stats.pearsonr(*samples)[1]
```

```python
>>> def stat_pearson(sample1, sample2):
...     return scipy.stats.pearsonr(sample1, sample2)[0]
```

### 3.4.2 Noncentrality Parameter

The noncentrality parameter for pearson's correlation coeffecient is given by
$\begin{align}
\lambda = \frac{r}{\sqrt{1 - r^{2}}}
\end{align}\tag{5.4}$

### Simulation

```python
>>> def extract_linear_samples(sim):
...     """Extracts the sample and test kwargs for an linear correlation"""
...     samples = sim['samples']
...     test_kwargs = {}
...
...     return samples, test_kwargs
```

```python
>>> distributions['correlation'] = {
...     'extraction': extract_linear_samples,
...     'test': emp_pearson,
...     'statistic': stat_pearson,
...     'traditional':trad.calc_pearson,
...     'power_kwargs': {'draw_mode': 'matched'},
...     'permutations': None
...     }
```

# Nonparametric Distributions

## Rank Sum Tests

A rank sum test

...

```python
>>> def emp_rank_sum(ids):
...     return scipy.stats.kruskal(*ids)[1]
```

### Log Normal Data Extraction

We use log-normal data as a model for highly skewed data.

```python
>>> def extract_lognormal_samples(sim):
...     samples = sim['samples']
...     test_kwargs = {}
...     return samples, test_kwargs
```

```python
>>> distributions['lognormal'] = {
...     'extraction': extract_lognormal_samples,
...     'test': emp_rank_sum,
...     'traditional': None,
...     'power_kwargs': {},
...     'permutations': None,
...     }
```

### Uniform Data Extraction

We also investigate uniform data.

```python
>>> def extract_unifrom_samples(sim):
...     samples = sim['samples']
...     test_kwargs = {}
...     return samples, test_kwargs
```

```python
>>> distributions['uniform'] = {
...     'extraction': extract_unifrom_samples,
...     'test': emp_rank_sum,
...     'traditional': None,
...     'power_kwargs': {},
...     'permutations': None,
...     }
```

## Permutative Tests

Something about permutaiton tests

### Permanova

```python
>>> def emp_permanova(id_, dm, groups, permutations=99):
...     return bootstrap_permanova(id_, dm, groups,
...                                permutations=permutations)['p-value']
```

```python
>>> def extract_permanova_samples(sim):
...     dm, groups = sim['samples']
...     samples = [groups.loc[groups == i].index for i in [0, 1]]
...     test_kwargs = {'dm': dm,
...                    'groups': groups}
...     return samples, test_kwargs
```

```python
>>> distributions['permanova'] = {
...     'extraction': extract_permanova_samples,
...     'test': emp_permanova,
...     'traditional': None,
...     'power_kwargs': {},
...     'permutations': 99
...     }
```

### Mantel

```python
>>> def emp_mantel(ids, dm1, dm2, permutations=99):
...     return bootstrap_mantel(ids, dm1, dm2, permutations=permutations)[1]
```

```python
>>> def extract_mantel_samples(sim):
...     x, y = sim['samples']
...     samples = [np.array(x.ids)]
...     test_kwargs = {'dm1': x,
...                    'dm2': y}
...     return samples, test_kwargs
```

```python
>>> distributions['mantel'] = {
...     'extraction': extract_mantel_samples,
...     'test': emp_mantel,
...     'traditional': None,
...     'power_kwargs': {'draw_mode': 'matched'},
...     'permutations': 99
...     }
```

# Power Calculation

```python
>>> def calculate_power(sim, save_fp, extraction, test, statistic,
...                     traditional, power_kwargs, permutations):
...     """Extracts the information"""
...     samples, test_kwargs = extraction(sim)
...
...     max_count = min([len(sample) for sample in samples])
...     counts = np.arange(5, max_count, 10)
...
...     if traditional is not None:
...         trad_power = traditional(*samples, counts=counts)
...     else:
...         trad_power = None
...
...
...     emp_power = subsample_power(test=test,
...                                 samples=samples,
...                                 counts=counts,
...                                 num_iter=num_iter,
...                                 num_runs=num_runs,
...                                 alpha=alpha,
...                                 bootstrap=True,
...                                 test_kwargs=test_kwargs,
...                                 **power_kwargs
...                                 )
...
...     power_summary = {'emperical': emp_power,
...                      'traditional': trad_power,
...                      'original_p': test(samples, **test_kwargs),
...                      'statistic': statistic(*samples, **test_kwargs),
...                      'permutations': permutations,
...                      'alpha': alpha,
...                      'counts': counts,
...                      'original_size': len(samples[0]),
...                      }
...     with open(power_fp, 'wb') as f_:
...         pickle.dump(power_summary, f_)
...     return power_fp
```

```python
>>> %%time
... for test_name, test_summary in distributions.items():
...     print(test_name)
...     sim_dir = os.path.join(sim_location, test_name)
...     power_dir = os.path.join(power_location, test_name)
...
...     if not os.path.exists(power_dir):
...         os.makedirs(power_dir)
...
...     for i in range(num_rounds):
...         sim_fp = os.path.join(sim_dir, 'simulation_%i.p' % i)
...         power_fp = os.path.join(power_dir,  'simulation_%i.p' % i)
...         if os.path.exists(power_fp):
...             continue
...         with open(sim_fp, 'rb') as f_:
...             sim = pickle.load(f_)
...         test_summary['sim'] = sim
...         test_summary['save_fp'] = power_fp
...         calculate_power(**test_summary)
anova_3
ttest_ind
correlation
ttest_1
anova_8
CPU times: user 5min 56s, sys: 1.83 s, total: 5min 57s
Wall time: 5min 59s
```
