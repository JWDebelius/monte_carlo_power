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


def simulate_permanova(num_samples, wdist, wspread, bdist, bspread, num0=None):
    """Makes a distance matrix with specified mean distance and spread

    Paramaters
    ----------
    num_samples : int
        The number of samples which should be described in the distance matrix
    wdist, bdist : list, float, optional
        A value or range of the distance offset for the within (`wdist`) or
        between (`bdist`) sample distances. If a list is supplied, a value
        will be drawn between the first to values.
    wspread, bspread : list, float, optional
        A value or range for the distance spread for the within (`wspread`) or
        between (`bspread`) sample distances. If a list is supplied, a value
        will be drawn between the first to values.
    num0: int, optional
        The number of samples in the first group. (The size of the second
        group will be given by `num_samples - num0`). If no value is supplied,
        the value will be simulated using a binomial distribution with `p=0.5`.

    Returns
    -------
    DistanceMatrix
        The simulated distance matrix. Within-group distances are described by
        a normal distribution * means and variances described by `wdist` and
        `wspread`, respective. Between group distances are described by a
        normal distribution with means and variances described by `bdist` and
        `bspread`.
    DataFrame
        A dataframe with a simulated mapping file corresponding to the groups
        in the data.

    Raises
    ------
    TypeError
        If `wdist`, `wspread`, `bdist`, or `bspread` is not a float or list

    """

    # Gets the group sizes
    if num0 is None:
        num0 = np.random.binomial(1, 0.5, (num_samples)).sum()
    num1 = num_samples - num0

    # Simulates the withi n and between sample distance
    wdist0 = _check_param(wdist, 'wdist')
    wdist1 = _check_param(wdist, 'wdist')
    bdist_ = _check_param(bdist, 'bdist')

    # Simulates the within and between sample spread
    wspread0 = _check_param(wspread, 'wspread')
    wspread1 = _check_param(wspread, 'wspread')
    bspread_ = _check_param(bspread, 'bspread')

    dist = [wdist0, wdist1, bdist_]
    spread = [wspread0, wspread1, bspread]

    # Simulates the distances
    vec0 = _simulate_gauss_vec(wdist0, wspread0, _vec_size(num0))
    vec1 = _simulate_gauss_vec(wdist1, wspread1, _vec_size(num1))
    vecb = _simulate_gauss_vec(bdist_, bspread_, (num0, num1))

    # Reshapes the within distance vectors
    dm0 = _convert_to_mirror(num0, vec0)
    dm1 = _convert_to_mirror(num1, vec1)

    # Creates the distance array
    dm = np.zeros((num_samples, num_samples)) * np.nan
    dm[0:num0, 0:num0] = dm0
    dm[num0:num_samples, num0:num_samples] = dm1
    dm[0:num0, num0:num_samples] = vecb
    dm[num0:num_samples, 0:num0] = vecb.transpose()

    # Simulates the mapping data
    groups = np.hstack((np.zeros(num0), np.ones(num1))).astype(int)

    # Simulates the sample ids
    ids = np.array(['s.%i' % (i + 1) for i in np.arange(num_samples)])

    # Makes the distance matrix and mapping file
    dm = skbio.DistanceMatrix(dm, ids=ids)
    grouping = pd.Series(groups, index=ids, name='groups')

    return [dist, spread], [dm, grouping]


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


def _convert_to_mirror(length, vec):
    """Converts a condensed 1D array to a mirror 2D array
    """

    vec = np.hstack(vec)
    # Creates the output matrix
    dm = np.zeros((length, length))
    # Adds a counter to watch the positon
    pos_count = 0

    # Populates the distance matrix
    for idx in range(length-1):
        # Gets the position for the two dimensional matrix
        pos2 = np.arange(idx+1, length)
        # Gets the postion for hte one dimensional matrix
        pos1 = np.arange(idx, length-1) + pos_count
        pos_count = pos_count + len(pos1) - 1
        # Updates the data in the matrices
        dm[idx, pos2] = vec[pos1]
        dm[pos2, idx] = vec[pos1]

    return dm


def _check_param(param, param_name, random=np.random.uniform, size=1):
    """Checks a parameter is sane"""
    if isinstance(param, list):
        return random(*param, size=size)
    elif not isinstance(param, (int, float)):
        raise TypeError('%s must be a list or a float' % param_name)
    else:
        return param


def _simulate_gauss_vec(mean, std, size):
    """Makes a modified gaussian vector bounded between 0 and 1"""
    vec = np.random.normal(loc=mean, scale=std, size=size)
    vec[vec < 0] = np.absolute(vec[vec < 0])
    vec[vec > 1] = 1
    return vec


def _vec_size(length):
    """Defines a group size for a distance matrix"""
    return int(((length) * (length - 1))/2)
