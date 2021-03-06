#-----------------------------------------------------------------------------
# Copyright (c) 2016, Machiavellian Project.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import scipy
import skbio


def simulate_ttest_1(mu_lim, sigma_lim, count_lim=100):
    """Simulates data for a one sample t test compared to 0.

    Parameters
    ----------
    mu_lim : list, float
        The limits for selecting a mean
    sigma_lim : list, float
        The limits for selecting a standard deviation
    count_lim : list, float, optional
        the number of observations which should be drawn for the sample

    Returns
    -------
    list
        The values for `[mu, sigma, n]` for the sample.
    list
        The simulated normal distribution.

    Raises
    ------
    TypeError
        If any parameter is not a float or a list.
    """

    # Gets the distribution parameters
    mu = _check_param(mu_lim, 'mu lim', np.random.randint)
    sigma = _check_param(sigma_lim, 'sigma lim', np.random.randint)
    n = int(_check_param(count_lim, 'count lim', np.random.randint))

    # Draws a sample that fits the parameters
    return [mu, sigma, n], [mu + np.random.randn(n) * sigma]


def simulate_ttest_ind(mu_lim, sigma_lim, count_lim=100):
    """Simulates data for an independent sample t test

    Parameters
    ----------
    mu_lim : list, float
        The limits for selecting a mean
    sigma_lim : list, float
        The limits for selecting a standard deivation
    count_lim : list, float, optional
        the number of observations which should be drawn for the sample

    Returns
    -------
    list
        The values for `[mu1, mu2, sigma1, sigma2, n]` for the sample.
    list
        The simulated normal distributions

    Raises
    ------
    TypeError
        If any parameter is not a float or a list.
    """
    # Gets the distribution paramters
    mu1 = _check_param(mu_lim, 'mu lim', np.random.randint)
    mu2 = _check_param(mu_lim, 'mu lim', np.random.randint)
    sigma1 = _check_param(sigma_lim, 'sigma lim', np.random.randint)
    sigma2 = _check_param(sigma_lim, 'sigma lim', np.random.randint)
    n = int(_check_param(count_lim, 'count lim', np.random.randint))

    # Returns a pair of distributions
    samples = [mu1 + np.random.randn(n) * sigma1,
               mu2 + np.random.randn(n) * sigma2]

    return [mu1, mu2, sigma1, sigma2, n], samples


def simulate_anova(mu_lim, sigma_lim, count_lim, num_pops):
    """Simulates data for a one way ANOVA

    Parameters
    ----------
    mu_lim : list, float
        The limits for selecting a mean
    sigma_lim : list, float
        The limits for selecting a standard deivation
    count_lim : list, float
        the number of observations which should be drawn for the sample
    num_pops: float
        The number of populations to use in the ANOVA simulation

    Returns
    -------
    list
        The list of parameters for the simulations in the forms of
        [list of means per group, sigma, sample size].
    list
        The simulated normal distributions

    Raises
    ------
    TypeError
        If the `mu_lim`, `sigma_lim`, or `count_lim` are not a float or list.
    """

    # Defines the distribtuion parameters
    mus = _check_param(mu_lim, 'mu lim', np.random.randint, num_pops)
    sigma = _check_param(sigma_lim, 'sigma lim', np.random.randint)
    n = int(_check_param(count_lim, 'count lim', np.random.randint))

    # Draws samples which fit the population
    samples = [mu + np.random.randn(n)*sigma for mu in mus]

    return [mus, sigma, n], samples


def simulate_discrete(p_lim, count_lim, num_groups=2):
    """Simulates discrete counts for a chi-square test

    Parameters
    ----------
    p_lim : list, float
        The limits for simulated binomial probabilities
    count_lim : list, float
        the number of observations which should be drawn for the sample
    num_groups : int, optional
        The number of groups to compare.

    Returns
    -------
    DataFrame
        A dataframe with the group designation, outcome of the test, and a
        dummy column for the counts.
    list
        The parameters with the randomly selected p values, number of
        samples per group, and the number of groups
    """
    # Gets the parameters
    p_values = [_check_param(p_lim, 'binomial p') for i in range(num_groups)]
    size = int(_check_param(count_lim, 'group size'))
    summaries = []
    for i, p in enumerate(p_values):
        index = ['s.%i' % i for i in np.arange(0, size) + size * i]
        dichomous = np.vstack([np.random.binomial(1, p, size),
                               np.ones(size) * i,
                               np.ones(size)])
        summaries.append(pd.DataFrame(dichomous.T,
                                      index=index,
                                      columns=['outcome', 'group', 'dummy']))
    return [p_values, size, num_groups], pd.concat(summaries)


def simulate_lognormal(mu_lim, sigma_lim, count_lim):
    """Simulates log normal data of a specified size

    Parameters
    ----------
    mu_lim : list, float
        The limits for selecting a mean for the log normal distributions
    sigma_lim : list, float
        The limits for selecting a standard deivation for the log normal
        distributions
    count_lim : list, float
        the number of observations which should be drawn for the sample

    Returns
    -------
    list
        The values for the means, standard deviations and sample size
    list
        The sample vectors for the log noraml distributions

    Raises
    ------
    TypeError
        When the limits are not a list, integer or float

    """
    # Estimates the parameters
    [x1, x2] = _check_param(mu_lim, 'mu lim', np.random.uniform, 2)
    [s1, s2] = _check_param(sigma_lim, 'sigma lim', np.random.uniform, 2)
    n = int(_check_param(count_lim, 'count lim', np.random.randint))

    v1 = np.random.lognormal(x1, s1, n)
    v2 = np.random.lognormal(x2, s2, n)

    return [(x1, x2), (s1, s2), n], [v1, v2]


def simulate_uniform(range_lim, delta_lim, counts_lim):
    """Simulates uniform data of a specified size

    Parameters
    ----------
    range_lim : list, int
        The upper limit of the uniform distribution
    delta_lim: list, int
        The offset between the two distributions
    count_lim : list, float
        the number of observations which should be drawn for the sample

    Returns
    -------
    list
        A list with the upper limit, offset and sample size for the samples
    list
        The two vectors with the uniform distributions

    Raises
    ------
    TypeError
        When the limits are not a list, integer or float

    """

    r_ = _check_param(range_lim, 'range_lim', np.random.uniform)
    d_ = _check_param(delta_lim, 'delta_lim', np.random.uniform)
    n_ = _check_param(counts_lim, 'counts_lim', np.random.randint)

    v1 = np.random.uniform(0, r_, n_)
    v2 = np.random.uniform(0, r_, n_) + d_

    return [r_, d_, n_], [v1, v2]


def simulate_feature_table(n_lim, p_lim, psi_lim, num_observations=100,
                           num_features=500, percent_different=0.05,
                           threshhold=6000):
    """
    Simulates an feature  table using a zero inflated negative binomial model

    ... More about hte model...

    Parameters
    ----------
    n_lim: list of ints
        The limits for the number of trials in the negative binomial simulation
    p_lim: list of floats
        The limits for the p-values for hte negative binomial simulation
        model
    psi_lim: list of floats
        Limits for the probability of a zero in a model
    num_observations: int, list of ints
        The number of observations (samples) in the feature table. Can be an
        intger or a range.
    num_features: int, list of ints
        The number of features in the feature table
    percent_different: float, list of floats
        The percentage of the features where the probabilites are drawn from
        different distributions.
    threshhold: int, optional
        The minimum number of counts for a sample to be included in the
        final table

    Returns
    -------
    list
        The number of observations, number of features, percent of features
        different, negative binomial parameters, negative binomaial probabilies
        for group 1, negative binomail probabilites for group 2, and
        probability of a 0 for the table.
    DataFrame
        An observation x feature table containing integer counts
    Series
        Identifies the group which samples belong to

    Also See
    --------

    References
    ----------
    ..[1] Kutz, Z.D. et al. (2015) "Sparse and Compositionally Robust Inference
    of Microbial Ecological Networks." PLoS Compuational Biology. 11: e10004226
    http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004226

    """

    # Checks the table size
    num_obs = _check_param(num_observations, 'num_observations',
                           random=np.random.randint)
    num_feat = _check_param(num_features, 'num_features',
                            random=np.random.randint)
    perc_diff = _check_param(percent_different, 'perc_diff')

    # Calculates the number of features which are different between
    # the two conditions
    num_same = int(num_feat * (1 - perc_diff))
    num_diff = num_feat - num_same

    # Gets simulation parameters for features which are the same
    n_same = _check_param(n_lim, 'n_lim', random=np.random.randint,
                          size=num_same)
    p_same = _check_param(p_lim, 'p_lim', size=num_same)
    psi_same = _check_param(psi_lim, 'psi_lim', size=num_same)

    # Gets simulation parameters for features which are different
    n_diff = _check_param(n_lim, 'n_lim', random=np.random.randint,
                          size=num_diff)
    p_g1 = _check_param(p_lim, 'p_lim', size=num_diff)
    p_g2 = _check_param(p_lim, 'p_lim', size=num_diff)
    psi_diff = _check_param(psi_lim, 'psi_lim', size=num_diff)

    # Generates the feature table
    table = pd.DataFrame(
        np.concatenate([
            np.vstack([zero_inflated_nb(n, p, psi, size=(2 * num_obs))
                       for (n, p, psi) in zip(*(n_same, p_same, psi_same))]),
            np.vstack([np.hstack([zero_inflated_nb(n, p1, psi, size=num_obs),
                                  zero_inflated_nb(n, p2, psi, size=num_obs)])
                       for (n, p1, p2, psi)
                       in zip(*(n_diff, p_g1, p_g2, psi_diff))])
            ]),
        index=['f.%i' % i for i in range(num_feat)],
        columns=['o.%i' % i for i in range(2 * num_obs)]).T

    # Generates the grouping object
    grouping = pd.Series(np.hstack([np.zeros(num_obs), np.ones(num_obs)]),
                         index=['o.%i' % i for i in range(2 * num_obs)])

    # Returns summary and objects
    summary = [num_obs, num_feat, perc_diff, np.hstack([n_same, n_diff]),
               np.hstack([p_same, p_g1]), np.hstack([p_same, p_g2]),
               np.hstack([psi_same, psi_diff])]

    # Drops missing samples
    drop = (table.sum(1) > threshhold)
    table = table.loc[drop]
    grouping = grouping.loc[drop]

    return summary, (table, grouping)


# def simulate_permanova(num_samples, wdist, wspread, bdist, bspread, counts_lim):
# def simulate_permanova(mu_lim, sigma_lim, count_lim=100, distance=None,
#     num_features=100):
#     """Makes a distance matrix with specified mean distance and spread

#    Parameters
#    ----------
#     mu_lim : list, float
#         The limits for selecting a mean of the distributions being transformed
#         into a distance matrix
#     sigma_lim : list, float
#         The limits for selecting a standard deivation for the distributions
#         transformed into a distance matrix
#     count_lim : list, float
#         the number of observations which should be drawn for each sample
#     distance : function
#         A distance metric function. By default, Euclidean distance is used.
#     simulate : function
#         A function which accepts a mean, standard deviation, and sample size
#         argument and returns parameters and distributions

#     Returns
#     -------
#     list:
#         The means, variance and sample size used in the simulation of the
#         underlying data.
#     DistanceMatrix
#         The simulated distance matrix. Within-group distances are described by
#         a normal distribution * means and variances described by `wdist` and
#         `wspread`, respective. Between group distances are described by a
#         normal distribution with means and variances described by `bdist` and
#         `bspread`.
#     DataFrame
#         A dataframe with a simulated mapping file corresponding to the groups
#         in the data.

#     """

#     # Handles the distance
#     if distance is None:
#         distance = scipy.spatial.distance.braycurtis

#     # Simulates the samples
#     params1, sample1 = simulate_anova(mu_lim, sigma_lim, count_lim,
#                                       num_pops=num_features)
#     params2, sample2 = simulate_anova(mu_lim, sigma_lim, count_lim,
#                                       num_pops=num_features)
#     sample1 = np.vstack(sample1)
#     sample2 = np.vstack(sample2)

#     samples = [sample1, sample2]

#     labels = np.hstack([i * np.ones(s.shape[1])
#                        for i, s in enumerate(samples)])
#     names = ['s.%i' % (i + 1) for i in range(len(labels))]

#     dm = skbio.DistanceMatrix.from_iterable(np.hstack(samples).T,
#                                             distance,
#                                             keys=names)
#     grouping = pd.Series(labels.astype(int), index=names, name='groups')

#     return [params1, params1], [dm, grouping]


def simulate_correlation(slope_lim, intercept_lim, sigma_lim, count_lim,
    x_lim):
    """Simulates data for a simple correlation

    Parameters
    ----------
    slope_lim : list, float
        The limits for selecting a slope
    intercept_lim : list, float
        the limits on values for the intercept
    sigma_lim : list, float
        The limits for selecting a variance
    count_lim : list, float
        the number of observations which should be drawn for the sample
    x_lim : list
        sets limits on the x values

    Returns
    -------
    simulation_params : list
        The values for `[sigma, number of samples, slope and intercept]`
        for the sample.
    simulation_results : list
        Vectors of coordinates for the x and y values

    Raises
    ------
    TypeError
        If any of the parameters are not a float or list.
    """

    # Calculates the distribution for the residuals
    sigma = _check_param(sigma_lim, 'sigma lim', np.random.randint)
    n = int(_check_param(count_lim, 'count lim', np.random.randint))
    # Calculates the parameters for the line
    m = _check_param(slope_lim, 'slope lim', np.random.randint)
    b = _check_param(intercept_lim, 'intercept lim', np.random.randint)

    x = np.random.uniform(*x_lim, size=n)
    y = m * x + b + np.random.randn(n) * sigma

    return [sigma, n, m, b], [x, y]


def simulate_mantel(slope_lim, intercept_lim, sigma_lim, count_lim, x_lim,
    distance=None):
    """Simulates two correlated matrices

    Parameters
    ----------
    slope_lim : list, float
        The limits for selecting a slope
    intercept_lim : list, float
        the limits on values for the intercept
    sigma_lim : list, float
        The limits for selecting a variance
    count_lim : list, float
        the number of observations which should be drawn for the sample
    x_lim : list
        sets limits on the x values
    distance : function, optional
        defines the distance between the two samples. If no metric is provided,
        euclidean distance will be used.

    Returns
    -------
    simulation_params : list
        The values for `[sigma, number of samples, slope and intercept]`
        for the sample.
    simulation_results : list
        Vectors of coordinates for the x and y values

    Raises
    ------
    TypeError
        If the `slope_lim`, `intercept_lim`, `sigma_lim`, `count_lim, or
        `x_lim` of the parameters are not a float or list.

    """

    # Handles the distance
    if distance is None:
        distance = scipy.spatial.distance.euclidean

    [sigma, n, m, b], [x_vec, y_vec] = simulate_correlation(slope_lim,
                                                            intercept_lim,
                                                            sigma_lim,
                                                            count_lim,
                                                            x_lim=x_lim,
                                                            )

    # Simulates the distance matrices
    names = ['s.%i' % (i + 1) for i in range(n)]
    x = skbio.DistanceMatrix.from_iterable(x_vec, distance, keys=names)
    y = skbio.DistanceMatrix.from_iterable(y_vec, distance, keys=names)

    return [sigma, n, m, b], [x, y]


def zero_inflated_nb(n, p, phi=0, size=None):
    """Models a zero-inflated negative binomial

    Something about hte negative binomail model here...

    This basically just wraps the numpy negative binomial generator,
    where the probability of a zero is additionally inflated by
    some probability, psi...

    Parameters
    ----------
    n : int
        Parameter, > 0.
    p : float
        Parameter, 0 <= p <= 1.
    phi : float, optional
        The probability of obtaining an excess zero in the model,
        where 0 <= phi <= 1. When `phi = 0`, the distribution collapses
        to a negative binomial model.
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    int or ndarray of ints
        Drawn samples

    Also See
    --------
    np.random.negative_binomial

    References
    ----------
    ..[1] Kutz, Z.D. et al. (2015) "Sparse and Compositionally Robust Inference
    of Microbial Ecological Networks." PLoS Compuational Biology. 11: e10004226
    http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004226

    """
    zeros = (np.random.binomial(1, phi, size) == 1)
    nb_ = np.random.negative_binomial(n, p, size=size)
    nb_[zeros] = 0

    return nb_


def _check_param(param, param_name, random=np.random.uniform, size=None):
    """Checks a parameter is sane"""
    if isinstance(param, list):
        return random(*param, size=size)
    elif not isinstance(param, (int, float)):
        raise TypeError('%s must be a list or a float' % param_name)
    else:
        return param
