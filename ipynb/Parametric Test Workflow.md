# Parametric Test Workflow

This notebook is intended to illustrate the steps used to calculate adn validate power for a parametric distribution. We will work with a single simulated data set, rather than a full complement of 100 simulations; although we will indicate the corresponding notebook in which the full complement is explored. The notebook will also be used to build figure 1 for the manuscript.

This notebook will focus on a one-sample T test, compared against 0.

```python
>>> from functools import partial
>>> import os
>>> import pickle
...
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> import pandas as pd
>>> import scipy
>>> import seaborn as sn
...
>>> from machivellian.effects import z_effect, z_power
>>> from machivellian.plot import plot_alternate_t, add_noncentrality, plot_power_curve
>>> from machivellian.simulate import simulate_ttest_1
>>> from machivellian.traditional import effect_ttest_1, calc_ttest_1
>>> from machivellian.power import subsample_power, confidence_bound
```

We'll set up the notebook to have a consistent display style and show the plots inline.

```python
>>> % matplotlib inline
>>> sn.set_style('ticks')
```

We will also use a seed, so the data generated is the same every time the notebook is run.

```python
>>> np.random.seed(5)
```

We'll perform 100 simulations, and use a critical value of 0.05 for our two-tailed test.

```python
>>> alpha=0.05
>>> counts = np.arange(5, 100, 10)
>>> num_iter = 100
>>> num_runs = 5
```

# Data Simulation

We'll simulate data for a one-sample T test, where we compare the alterate hypotheses,

$\begin{matrix}
\textbf{H}_{0} & \mu_{1} = 0 \\
\textbf{H}_{1} & \mu_{1} \neq 0\\
\end{matrix}\tag{1}$

For this example, we'll pre-select a mean and variance, although in the full simulation ([Building Simulations Notebook](Build%20Simulations.md)), we vary the mean and variance. Our underlying population will be a normal distribution with $\mu = 2$ and $\sigma = 5$. We'll draw 100 observations.

```python
>>> mu = 1
>>> sigma = 5
>>> n = 100
...
>>> compare = 0
```

```python
>>> sample = np.random.normal(loc=mu, scale=sigma, size=n)
```

```python
>>> scipy.stats.ttest_1samp(sample, compare)
Ttest_1sampResult(statistic=3.1159238139591463, pvalue=0.002399605156919199)
```

Let's look at our data, and hypotheses.

```python
>>> fig = plt.figure()
>>> fig.set_size_inches(10, 5)
>>> ax1 = fig.add_subplot(2, 4, 1)
>>> ax2 = fig.add_subplot(2, 4, 5, sharey=ax1)
...
>>> x = np.arange(-15, 15.1, 0.25)
>>> y = scipy.stats.norm.pdf(x, loc=mu, scale=sigma)
>>> sn.distplot(sample, kde=False, norm_hist=True, ax=ax2)
>>> ax1.plot(x, y, color=sn.color_palette()[0])
...
>>> ax1.set_yticks([-1])
>>> sn.despine(ax=ax1, left=True, offset=10)
>>> sn.despine(ax=ax2, left=True, offset=10)
...
...
>>> yax = ax1.get_ylim()[1]
>>> ax1.plot([0, 0], [0, yax*1.1], 'r-')
>>> ax2.plot([0, 0], [0, yax*1.], 'r-')
>>> ax1.set_ylim([0, yax])
...
>>> ax1.set_xticks(ax2.get_xticks())
>>> ax1.set_xticklabels('')
...
>>> ax1.text(-14, 0.1, '(A)', size=15)
>>> ax2.text(-14, 0.1, '(B)', size=15)
```

```python
>>> sample_mean = np.mean(sample)
>>> sample_std = np.std(sample)
>>> print('%1.2f +/- %1.2f' % (sample_mean, sample_std))
1.46 +/- 4.65
```

Now, let's look at our hypothesis testing proceedure. We're going to test our hypothesis using a one-sample t test. The statistic for this test is calculated as [[1](#1)]
$\begin{matrix}
t = \sqrt{n}\frac{(\bar{x} - 0)}{s}\\
\end{matrix}\tag{2}$

where the test statistic, $t$ is drawn from a t distribution with $n - 1$ degrees of freedom. For the sake of simplicity, we'll use the scipy implemention of a one-sample t test: `scipy.stats.ttest_1`, and wrap it accept a list with the sample and to return a p-value. It is worth noting that this is a two-tailed test, and we will later fit the data with a one-tailed test.

```python
>>> def test(samples):
...     samples = samples[0]
...     return scipy.stats.ttest_1samp(samples, popmean=compare)[1]
```

```python
>>> test([sample])
0.002399605156919199
```

# Hypothetical Power

Let's plan to calculate power for sample sizes between 5 and 95 observations per group.

```python
>>> counts = np.arange(5, 100, 10)
```

Hypothesis testing provides a framework for testing statistical power. Power is defined as the probability of finding a significant difference between the two samples, assuming the samples are different (which is to say, assuming the alternate distribution is True). To measure this, we describe the alternate hypothesis using the noncentrality parameter.

The non-centrality parameter, for an independent sample t test where samples are the same size is once again related to the test statistic as [[2](#2)]

$\begin{align*}
\lambda(n) &= \sqrt{n}\frac{(\bar{x} - 0)}{s}
\end{align*}\tag{3}$

This can also be expressed as an sample constant developed by Jacob Cohen [[3](#3)], multiplied by the sample size.

$\begin{align}
d = \frac{(\bar{x} - 0)}{s}
\end{align}\tag{4}$
$\begin{align}
\lambda(n) = d\sqrt{n}
\end{align}\tag{5}$

We've implemented the effect size calculation in the `effect_ttest_1` function.

```python
>>> d = effect_ttest_1(sample, compare)
>>> print(d)
0.31316212625
```

We'll calculate the power over a series of sample sizes. In these simulations, we'll look at power between 5 and 95 observations, counting my 10. To create smooth curves, we'll use more interpolation for plotting smooth curves.

```python
>>> emp_counts = np.arange(5, 100, 10)
>>> dist_counts = np.arange(5, 100, 1)
```

We can look at distribution-based powe using an implementation (`machiavellian.traditional.calc_ttest_1`) which accepts the samples and power, and returns the counts.

```python
>>> dist_pwr = calc_ttest_1(sample, x0=compare, counts=dist_counts, alpha=alpha)
```

Let's look at the theoretical power curve.

```python
>>> ax3 = fig.add_subplot(242)
>>> ax3.set_aspect(aspect=100)
...
>>> plot_power_curve(ax3, counts=dist_counts, power_trace=dist_pwr)
>>> ax3.set_xticks(np.arange(0, 101, 25))
>>> ax3.set_xlabel('Observations per group', size=12)
>>> ax3.set_ylabel('Theoretical Power', size=12)
...
>>> ax3.text(5, 0.9, '(C)', size=15)
...
>>> fig
```

Next, we'll compute an empirical power curve.

```python
>>> emp_power = subsample_power(test,
...                             counts=emp_counts,
...                             samples=[sample],
...                             num_iter=100,
...                             num_runs=5,
...                             alpha=alpha
...                             )
```

We'll plot the empirical curve, as well.

```python
>>> ax4 = fig.add_subplot(243)
>>> ax4.set_aspect(aspect=100)
...
>>> plot_power_curve(ax4, counts=emp_counts, power_scatter=emp_power)
...
>>> ax4.set_xlabel('Observations per group', size=12)
>>> ax4.set_ylabel('Empirical Power', size=12)
>>> ax4.set_xticks(np.arange(0, 101, 25))
...
>>> ax4.text(5, 0.9, '(D)', size=15)
...
>>> fig
```

Now, let's fit the empirical curve to estimate the curve fitting parameter.

In other simulations, we determined that power cannot be estimated effectively for values less than 0.1, greater than 0.95, or for sample sizes less than 10. So, we'll exclude these from the estimate we just made.

```python
>>> l_ = z_effect(emp_counts, emp_power, alpha=alpha/2)
[ 5 15 25 35 45 55 65 75 85 95  5 15 25 35 45 55 65 75 85 95  5 15 25 35 45
 55 65 75 85 95  5 15 25 35 45 55 65 75 85 95  5 15 25 35 45 55 65 75 85 95] [ 0.03  0.21  0.4   0.39  0.55  0.65  0.72  0.77  0.82  0.83  0.04  0.18
  0.36  0.46  0.63  0.66  0.69  0.77  0.83  0.91  0.09  0.23  0.33  0.46
  0.52  0.65  0.71  0.73  0.78  0.86  0.05  0.16  0.31  0.44  0.52  0.66
  0.71  0.8   0.85  0.86  0.09  0.21  0.4   0.47  0.51  0.59  0.79  0.74
  0.82  0.88]
```

```python
>>> print('Calibration parameter: %1.2f +/- %1.2f' % (np.nanmean(l_), confidence_bound(l_)))
Calibration parameter: 0.31 +/- 0.01
```

Let's us this calibration parameter to estimate the power curve. We'll calculate error based on the confidence interval for our effect size parameter. So, we'll calculate the lower limit for the power based on the lower limit for hte effect size, and the upper limit for the power based on the upper limit for the effect size.

```python
>>> from machivellian.plot import plot_interval_power, _get_effect_interval
```

```python
>>> ax5 = fig.add_subplot(244)
>>> ax5.set_aspect(aspect=100)
...
>>> plot_interval_power(ax5, counts=dist_counts, effects=l_, power_alpha=alpha/2)
...
>>> ax5.set_xlabel('Observations per group', size=12)
>>> ax5.set_ylabel('Predicted Power', size=12)
>>> ax5.set_xticks(np.arange(0, 101, 25))
...
>>> ax5.text(5, 0.9, '(E)', size=15)
...
>>> fig
```

We can evaluate the quality of the fit by comparing the values we used for the prediction with the predicted values.

```python
>>> emp_pred = z_power(emp_counts, np.nanmean(l_), alpha=alpha/2)
...
>>> ax6 = fig.add_subplot(246)
...
>>> _ = sn.residplot(np.hstack([emp_pred] * num_runs), np.hstack(emp_power), ax=ax6)
>>> ax6.set_ylim([-0.15, 0.15])
>>> sn.despine(ax=ax6, top=True, right=True, bottom=True)
>>> fig
```

```python
>>> pred_power = z_power(counts[1:], np.nanmean(l_), alpha/2)
```

```python
>>> emp_power
array([[ 0.04,  0.25,  0.28,  0.49,  0.49,  0.66,  0.7 ,  0.77,  0.89,
         0.82],
       [ 0.06,  0.15,  0.31,  0.46,  0.51,  0.69,  0.75,  0.83,  0.81,
         0.81],
       [ 0.04,  0.17,  0.37,  0.4 ,  0.56,  0.64,  0.72,  0.78,  0.85,
         0.86],
       [ 0.06,  0.23,  0.26,  0.38,  0.55,  0.69,  0.65,  0.78,  0.73,
         0.93],
       [ 0.04,  0.21,  0.28,  0.39,  0.48,  0.62,  0.68,  0.73,  0.8 ,  0.9 ]])
```

```python
>>> pred_power
array([ 0.10065149,  0.21813579,  0.33188774,  0.43831015,  0.53441928,
        0.61881059,  0.69126521,  0.75233609,  0.80302387,  0.84454399])
```

```python
>>> ax = plt.axes()
>>> sn.regplot(np.hstack([pred_power] * 5), np.hstack(emp_power[:, 1:]), ax=ax)
>>> # ax.plot([0, 1], [0, 1], 'k-')
... ax.set_xlim([0, 1])
>>> ax.set_ylim([0, 1])
(0, 1)
```

```python
>>> sn.residplot(np.hstack([pred_power] * 5), np.hstack(emp_power[:, 1:]))
```

```python
>>> def check_power_fit(counts, emp_power, alpha, lower_lim=0.1, upper_lim=0.95,
...                     size_lim=0):
...     """check the empirical power"""
...     l_ = z_effect(counts, emp_power, alpha, lower_lim, upper_lim, size_lim)
...     pred_power = z_power(counts, np.nanmean(l_), alpha)
```

```python
>>> counts = emp_counts
```

```python
>>> l_ = z_effect(counts, emp_power, alpha/2, size_lim=10)
>>> np.nanmean(l_)
```

```python
>>> l_
```

```python
>>> sn.distplot(l_[np.isnan(l_) == False])
```

```python
>>> pred_power = z_power(counts, np.nanmean(l_), alpha/2)
>>> pred_power
```

```python
>>> scipy.stats.linregress()
```
