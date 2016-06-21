r"""
Empirical Power Estimation (:mod:`skbio.stats.power`)
=====================================================

.. currentmodule:: skbio.stats.power

The purpose of this module is to provide empirical, post-hoc power estimation
of normally and non-normally distributed data. It also provides support to
subsample data to facilitate this analysis.

The underlying principle is based on subsampling and Monte Carlo simulation.
Assume that there is some set of populations, :math:`K_{1}, K_{2}, ... K_{n}`
which have some property, :math:`\mu` such that :math:`\mu_{1} \neq \mu_{2}
\neq ... \neq \mu_{n}`. For each of the populations, a sample, :math:`S` can be
drawn, with a parameter, :math:`x` where :math:`x \approx \mu` and for the
samples, we can use a test, :math:`f`, to show that :math:`x_{1} \neq x_{2}
\neq ... \neq x_{n}`.

Since we know that :math:`\mu_{1} \neq \mu_{2} \neq ... \neq \mu_{n}`,
we know we should reject the null hypothesis. If we fail to reject the null
hypothesis, we have committed a Type II error and our result is a false
negative. We can estimate the frequency of Type II errors at various sampling
depths by repeatedly subsampling the populations and observing how often we
see a false negative. If we repeat this several times for each subsampling
depth, and vary the depths we use, we can start to approximate a relationship
between the number of samples we use and the rate of false negatives, also
called the statistical power of the test.

To generate complete power curves from data which appears underpowered, the
`statsmodels.stats.power` package can be used to solve for an effect size. The
effect size can be used to extrapolate a power curve for the data.

Most functions in this module accept a statistical test function which takes a
list of samples and returns a p value. The test is then evaluated over a series
of subsamples.

Sampling may be handled in two ways. For any set of samples, we may simply
choose to draw :math:`n` observations at random for each sample. Alternatively,
if metadata is available, samples can be matched based on a set of control
categories so that paired samples are drawn at random from the set of available
matches.

"""

# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
import scipy.stats


def confidence_bound(vec, alpha=0.05, df=None, axis=None):
    r"""Calculates a confidence bound assuming a normal distribution

    Parameters
    ----------
    vec : array_like
        The array of values to use in the bound calculation.
    alpha : float, optional
        The critical value, used for the confidence bound calculation.
    df : float, optional
        The degrees of freedom associated with the
        distribution. If None is given, df is assumed to be the number of
        elements in specified axis.
    axis : positive int, optional
        The axis over which to take the deviation. When axis
        is None, a single value will be calculated for the whole matrix.

    Returns
    -------
    bound : float
        The confidence bound around the mean. The confidence interval is
        [mean - bound, mean + bound].

    """

    # Determines the number of non-nan counts
    vec = np.asarray(vec)
    vec_shape = vec.shape
    if axis is None and len(vec_shape) == 1:
        num_counts = vec_shape[0] - np.isnan(vec).sum()
    elif axis is None:
        num_counts = vec_shape[0] * vec_shape[1] - np.isnan(vec).sum()
    else:
        num_counts = vec_shape[axis] - np.isnan(vec).sum() / \
            (vec_shape[0] * vec_shape[1])

    # Gets the df if not supplied
    if df is None:
        df = num_counts - 1

    # Calculates the bound
    # In the conversion from scipy.stats.nanstd -> np.nanstd `ddof=1` had to be
    # added to match the scipy default of `bias=False`.
    bound = np.nanstd(vec, axis=axis, ddof=1) / np.sqrt(num_counts - 1) * \
        scipy.stats.t.ppf(1 - alpha / 2, df)

    return bound


def z_power(counts, eff, alpha=0.05):
    """Estimates power for a z distribution from an effect size

    This is based on the equations in
        Lui, X.S. (2014) *Statistical power analysis for the social and
        behavioral sciences: basic and advanced techniques.* New York:
        Routledge. 378 pg.
    The equation assumes a positive magnitude to the effect size and a
    two-tailed test.

    Parameters
    ----------
    counts : array
        The number of observations for each power depth
    effect : float
        A standard measure of the difference between the underlying populations
     alpha : float
        The critial value used to calculate the power

    Returns
    power : array
        The statistical power at the depth specified by `counts`

    """
    z = scipy.stats.norm
    power = ((z.cdf(eff * np.sqrt(counts/2) - z.ppf(1 - alpha/2)) +
             (z.cdf(z.ppf(alpha/2) - eff * np.sqrt(counts/2)))))
    return power


def subsample_power(test, samples, counts, draw_mode='ind', numeric=True,
                    alpha=0.05, ratio=None, bootstrap=True, num_iter=500,
                    num_runs=10):
    """Subsamples data to iteratively calculate power

    Parameters
    ----------
    test : function
        The statistical test which accepts a list of arrays of values
        (sample ids or numeric values) and returns a p value or one-dimensional
        array of p values when `numeric == True`; or a boolean value
        indicating the null hypothesis should be rejected, or a
        one-dimensional array of boolean values indicating the null
        hypothesis should be rejected when `numeric == False`.
    samples : array_like
        `samples` can be a list of lists or a list of arrays where each
        sublist or row in the array corresponds to a sampled group.
    counts : array-like
        The depths at which to sample the data. If `bootstrap == False`, the
        largest count depth times the group ratio cannot be greater than the
        number of observations in each group.
    draw_mode : {"ind", "matched"}, optional
        "matched" samples should be used when observations in
        samples have corresponding observations in other groups. For instance,
        this may be useful when working with regression data where
        :math:`x_{1}, x_{2}, ..., x_{n}` maps to
        :math:`y_{1}, y_{2}, ..., y_{n}`. Sample vectors must be the same
        length in "matched" mode.
    numeric : bool, optional
        Indicates whether `test` returns a numeric p-value or array of numeric
        p values (`numeric=True`), or a boolean (`numeric=False`).
    alpha : float, optional
        The critical value used to calculate the power.
    ratio : 1-D array, optional
        The fraction of the sample counts which should be
        assigned to each group. If this is a 1-D array, it must be the same
        length as `samples`. If no value is supplied (`ratio` is None),
        then an equal number of observations will be drawn for each sample. In
        `matched` mode, this will be set to one. If `bootstrap == False`, then
        the product of the `ratio` and a sampling depth specified by `counts`
        cannot be greater than the number of observations in the respective
        sample.
    bootstrap : bool, optional
        Indicates whether subsampling should be performed with replacement
        (`bootstrap == True`) or without.
    num_iter : positive int, optional
        The number of p-values to generate for each point
        on the curve.
    num_runs : positive int, optional
        The number of times to calculate each curve.

    Returns
    -------
    power : array
        The power calculated for each subsample at each count. The array has
        `num_runs` rows, a length with the same number of elements as
        `sample_counts` and a depth equal to the number of p values returned by
        `test`. If `test` returns a float, the returned array will be
        two-dimensional instead of three.

    Raises
    ------
    ValueError
        If the `mode` is "matched", an error will occur if the arrays in
        `samples` are not the same length.
    ValueError
        There is a ValueError if there are fewer samples than the minimum
        count.
    ValueError
        If the `counts_interval` is greater than the difference between the
        sample start and the max value, the function raises a ValueError.
    ValueError
        There are not an equal number of groups in `samples` and in `ratios`.
    TypeError
        `test` does not return a float or a 1-dimensional numpy array.
    ValueError
        When `replace` is true, and `counts` and `ratio` will draw more
        observations than exist in a sample.
    """

    # Checks the inputs
    ratio, num_p = _check_subsample_power_inputs(test=test,
                                                 samples=samples,
                                                 draw_mode=draw_mode,
                                                 ratio=ratio,
                                                 bootstrap=bootstrap,
                                                 counts=counts)

    # Prealocates the power array
    power = np.zeros((num_runs, len(counts), num_p))

    for id2, c in enumerate(counts):
        count = np.round(c * ratio, 0).astype(int)
        for id1 in range(num_runs):
            ps = _compare_distributions(test=test,
                                        samples=samples,
                                        num_p=num_p,
                                        counts=count,
                                        num_iter=num_iter,
                                        bootstrap=bootstrap,
                                        mode=draw_mode)
            power[id1, id2, :] = _calculate_power(ps,
                                                  numeric=numeric,
                                                  alpha=alpha)

    power = power.squeeze()

    return power


def _compare_distributions(test, samples, num_p, counts=5, mode="ind",
                           bootstrap=True, num_iter=100):
    r"""Compares two distribution arrays iteratively
    """

    # Prealocates the pvalue matrix
    p_values = np.zeros((num_p, num_iter))

    # Determines the number of samples per group
    num_groups = len(samples)
    samp_lens = [len(sample) for sample in samples]

    if isinstance(counts, int):
        counts = np.array([counts] * num_groups)

    for idx in range(num_iter):
        if mode == "matched":
            pos = np.random.choice(np.arange(0, samp_lens[0]), counts[0],
                                   replace=bootstrap)
            subs = [sample[pos] for sample in samples]
        else:
            subs = [np.random.choice(np.array(pop), counts[i],
                                     replace=bootstrap)
                    for i, pop in enumerate(samples)]

        p_values[:, idx] = test(subs)

    if num_p == 1:
        p_values = p_values.squeeze()

    return p_values


def _calculate_power(p_values, alpha=0.05, numeric=True):
    r"""Calculates statistical power empirically for p-values

    Parameters
    ----------
    p_values : 1-D array
        A 1-D numpy array with the test results.
    alpha : float
        The critical value for the power calculation.
    numeric : Boolean
        Indicates whether a numeric p value should be used

    Returns
    -------
    power : float
        The empirical power, or the fraction of observed p values below the
        critical value.

    """

    if numeric:
        reject = np.atleast_2d(p_values < alpha)
    else:
        reject = np.atleast_2d(p_values)

    w = (reject).sum(axis=1)/reject.shape[1]

    return w


def _check_subsample_power_inputs(test, samples, counts, draw_mode='ind',
                                  ratio=None, bootstrap=True):
    """Makes sure that everything is sane before power calculations

    Parameters
    ----------
    test : function
        The statistical test which accepts a list of arrays of values
        (sample ids or numeric values) and returns a p value or one-dimensional
        array of p values.
    samples : array-like
        `samples` can be a list of lists or a list of arrays where each
        sublist or row in the array corresponds to a sampled group.
    counts : 1-D array
        The number of samples to use for each power depth calculation. If
        `replace` is False, than `counts` and `ratio` must be scaled so that
        no more samples are drawn than exist in a sample.
    draw_mode : {"ind", "matched"}, optional
        "matched" samples should be used when observations in
        samples have corresponding observations in other groups. For instance,
        this may be useful when working with regression data where
        :math:`x_{1}, x_{2}, ..., x_{n}` maps to
        :math:`y_{1}, y_{2}, ..., y_{n}`. Sample vectors must be the same
        length in "matched" mode.
        If there is no reciprocal relationship between samples, then
        "ind" mode should be used.
    ratio : 1-D array, optional
        The fraction of the sample counts which should be
        assigned to each group. If this is a 1-D array, it must be the same
        length as `samples`. If no value is supplied (`ratio` is None),
        then an equal number of observations will be drawn for each sample. In
        `matched` mode, this will be set to one.
    bootstrap : Bool
        Whether samples should be bootstrapped or subsampled without
        replacement. When `bootstrap == False`, `counts` and `ratio` must
        be scaled so that no more observations are drawn than exist in a
        sample.

    Returns
    -------
    ratio : 1-D array
        The fraction of the sample counts which should be assigned to each
        group.
    num_p : positive integer
        The number of p values returned by `test`.

    Raises
    ------
    ValueError
        If the `mode` is "matched", an error will occur if the arrays in
        `samples` are not the same length.
    ValueError
        There is a ValueError if there are fewer samples than the minimum
        count.
    ValueError
        If the `counts_interval` is greater than the difference between the
        sample start and the max value, the function raises a ValueError.
    ValueError
        There are not an equal number of groups in `samples` and in `ratios`.
    TypeError
        `test` does not return a float or a 1-dimensional numpy array.
    ValueError
        When `replace` is true, and `counts` and `ratio` will draw more
        observations than exist in a sample.

    """

    # Checks the sample drawing model
    if draw_mode not in {'ind', 'matched'}:
        raise ValueError('mode must be "matched" or "ind".')

    # Determines the minimum number of ids in a category
    id_counts = np.array([len(id_) for id_ in samples])
    num_groups = len(samples)

    # Checks the ratio argument
    if ratio is None or draw_mode == 'matched':
        ratio = np.ones((num_groups))
    else:
        ratio = np.asarray(ratio)
    if not ratio.shape == (num_groups,):
        raise ValueError('There must be a ratio for each group.')

    ratio_counts = np.array([id_counts[i] / ratio[i]
                             for i in range(num_groups)])
    largest = ratio_counts.min()

    # Determines the number of p values returned by the test
    p_return = test(samples)
    if isinstance(p_return, float):
        num_p = 1
    elif isinstance(p_return, np.ndarray) and len(p_return.shape) == 1:
        num_p = p_return.shape[0]
    else:
        raise TypeError('test must return a float or one-dimensional array.')

    # Checks the subsample size
    counts = np.asarray(counts)
    if counts.min() < 2:
        raise ValueError('you cannot test less than 2 samples per group')

    elif not bootstrap and counts.max() > largest:
        raise ValueError('Sampling depth is too high. Please use replacement '
                         'or pick fewer observations.')

    return ratio, num_p
