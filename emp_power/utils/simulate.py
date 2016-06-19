import numpy as np
import pandas as pd
import scipy
import skbio
from skbio.stats.composition import closure


def ttest_1_simulate(mu_lim, sigma_lim, count_lims):
    """Simulates data for a one sample t test compared to 0.

    Parameters
    ----------
    mu_lim : list
        The limits for selecting a mean
    sigma_lim : list
        The limits for selecting a standard deivation
    counts_lim : list
        the number of observations which should be drawn for the sample

    Returns
    -------
    simulation_params : list
        The values for `[mu, sigma, n]` for the sample.
    simulation_results : list
        The simulated normal distribution

    """

    # Gets the distribution parameters
    mu = np.random.randint(*mu_lim)
    sigma = np.random.randint(*sigma_lim)
    n = np.random.randint(*count_lims)

    # Draws a sample that fits the parameters
    return [mu, sigma, n], [mu + np.random.randn(n) * sigma]


def ttest_ind_simulate(mu_lim, sigma_lim, counts_lims):
    """Simulates data for an independent sample t test

    Parameters
    ----------
    mu_lim : list
        The limits for selecting a mean
    sigma_lim : list
        The limits for selecting a standard deivation
    counts_lim : list
        the number of observations which should be drawn for the sample

    Returns
    -------
    simulation_params : list
        The values for `[mu1, mu2, sigma1, sigma2, n]` for the sample.
    simulation_results : list
        The simulated normal distribution

    """
    # Gets the distribution paramters
    mu1, mu2 = np.random.randint(*mu_lim, size=2)
    sigma1, sigma2 = np.random.randint(*sigma_lim, size=2)
    n = np.random.randint(*counts_lims)

    # Returns a pair of distributions
    samples = [mu1 + np.random.randn(n) * sigma1,
               mu2 + np.random.randn(n) * sigma2]

    return [mu1, mu2, sigma1, sigma2, n], samples


def simulate_distance_matrix(num_samples, num0=None, wdist=[0, 0.5],
                             wspread=[0, 0.5], bdist=[0, 0.5],
                             bspread=[0, 0.5]):
    """Makes a distance matrix with specified mean distance and spread

    Paramaters
    ----------
    num_samples : int
        The number of samples which should be described in the distance matrix
    num0: int, optional
        The number of samples in the first group. (The size of the second
        group will be given by `num_samples - num0`). If no value is supplied,
        the value will be simulated using a binomial distribution with `p=0.5`.
    wdist, bdist : list, float, optional
        A value or range of the distance offset for the within (`wdist`) or
        between (`bdist`) sample distances. If a list is supplied, a value
        will be drawn between the first to values.
    wspread, bspread : list, float, optional
        A value or range for the distance spread for the within (`wspread`) or
        between (`bspread`) sample distances. If a list is supplied, a value
        will be drawn between the first to values.

    Returns
    -------
    dm : DistanceMatrix
        A scikit-bio distance matrix object with the simulated distances. The
        within-group distances are described by a normal distribution *
        means and variances described by `wdist` and `wspread`, respective.
        Between group distances are described by a normal distribution with
        means and variances described by `bdist` and `bspread`.
    grouping : DataFrame
        A dataframe with a simulated mapping file corresponding to the groups
        in the data.

    Raises
    ------
    ValueError
        If `wdist`, `wspread`, `bdist`, or `bspread` is not a float or list, a
        ValueError is raised.

    """

    # Gets the group sizes
    if num0 is None:
        num0 = np.binomial(1, 0.5, (num_samples)).sum()
    num1 = num_samples - num0

    # Simulates the withi n and between sample distance
    wdist0 = _check_param(wdist, 'wdist')
    wdist1 = _check_param(wdist, 'wdist')
    bdist_ = _check_param(bdist, 'bdist')

    # Simulates the within and between sample spread
    wspread0 = _check_param(wspread, 'wspread')
    wspread1 = _check_param(wspread, 'wspread')
    bspread_ = _check_param(bspread, 'bspread')

    # Simulates the distances
    vec0 = _simulate_gauss_vec(wdist0, wspread0, _vec_size(num0))
    vec1 = _simulate_gauss_vec(wdist1, wspread1, _vec_size(num1))
    vecb = _simulate_gauss_vec(bdist_, bspread_, (num0, num1))

    # Reshapes the within distance vectors
    dm0 = convert_to_mirror(num0, vec0)
    dm1 = convert_to_mirror(num1, vec1)

    # Creates the distance array
    dm = np.zeros((num_samples, num_samples)) * np.nan
    dm[0:num0, 0:num0] = dm0
    dm[num0:num_samples, num0:num_samples] = dm1
    dm[0:num0, num0:num_samples] = vecb
    dm[num0:num_samples, 0:num0] = vecb.transpose()

    # Simulates the mapping data
    groups = np.hstack((np.zeros(num0), np.ones(num1)))

    # Simulates the sample ids
    ids = np.array(['s.%i' % (i + 1) for i in np.arange(num_samples)])

    # Makes the distance matrix and mapping file
    dm = skbio.DistanceMatrix(dm, ids=ids)
    grouping = pd.Series(groups, index=ids, name='groups')

    return dm, grouping


def convert_to_mirror(length, vec):
    """Converts a condensed 1D array to a mirror 2D array

    Inputs
    ------
    length : int
        The length of the distance matrix
    vec : array
        A one-dimensional condensed array of the values to populate the
        distance matrix

    Returns
    -------
    dm : array
        A two dimensional symetrical matrix of length x length.
    """

    vec = np.hstack(vec)
    # Creates the output matrix
    dm = np.zeros((length, length))
    # Adds a counter to watch the positon
    pos_count = 0

    # Populates the distance matrix
    for idx in np.arange(length-1):
        # Gets the position for the two dimensional matrix
        pos2 = np.arange(idx+1, length)
        # Gets the postion for hte one dimensional matrix
        pos1 = np.arange(idx, length-1) + pos_count
        pos_count = pos_count + len(pos1) - 1
        # Updates the data in the matrices
        dm[idx, pos2] = vec[pos1]
        dm[pos2, idx] = vec[pos1]

    return dm


def _check_param(param, param_name):
    """Checks a parameter is sane"""
    if isinstance(param, list):
        return np.random.uniform(*param)
    elif not isinstance(param, float):
        raise ValueError('%s must be a list or a float' % param_name)
    else:
        return param


def _simulate_gauss_vec(mean, std, size):
    """Makes a modified gaussian vector bounded between 0 and 1"""
    if isinstance(size, tuple):
        size = tuple([int(s) for s in size])
    else:
        size = int(size)
    vec = np.random.normal(loc=mean, scale=std, size=size)
    vec[vec < 0] = np.absolute(vec[vec < 0])
    vec[vec > 1] = 1
    return vec


def _vec_size(length):
    """Defines a group size for a distance matrix"""
    return ((length) * (length - 1))/2


def build_groups(num_groups=2, obs_per_group=100):
    """Builds a group vector

    Parameters
    ----------
    num_groups : int
        The number of groups to test
    obs_per_group : int
        The number of observations per group

    Returns
    -------
    groups : ndarray


    """
    groups = np.hstack([np.ones(obs_per_group) * i
                        for i in np.arange(num_groups)])
    return groups


def generate_vector(num_groups=2, obs_per_group=100, offset=0.1):
    """Generates a vector with no signficiatn differences

    Parameters
    ----------
    num_groups : int
        The number of groups
    obs_per_group : int
        The number of samples per group
    offset : float
        An amount to substract to create sparse data.

    Returns
    -------
    counts : ndarray of floats
        The number of unscaled counts per sample
    params : dict
        A dictionary of the parameters used to simulte the data
    """
    counts = np.random.rand(num_groups * obs_per_group) - offset
    counts[counts < 0] = 0
    params = {'offset': offset,
              'num_groups': num_groups,
              'obs_per_group': obs_per_group}

    return counts, params


def generate_diff_vector(num_groups=2, obs_per_group=100, mu_lim=[0, 5],
                         sigma_lim=[1, 6], scale=0.75, offset=0.1):
    """Generates a vector with signficant differences

    Parameters
    ----------
    num_groups : int
        The number of groups
    obs_per_group : int
        The number of samples per group
    mu_lim : list
        The lower and upper limit for the mean. Multiple means will be
        calculated.
    sigma_lim : list
        The lower and upper limits for the standard deviation. A population
        standard deviation will be used.
    scale : float
        How large the normal distribution should be compared to the random
        noise
    offset : float
        An amount to substract to create sparse data.

    Returns
    -------
    counts : ndarray of floats
        The number of unscaled counts per sample
    params : dict
        A dictionary of the parameters used to simulte the data
    """

    p = 1
    while p > 0.001:
        means = np.random.randint(*mu_lim, size=num_groups)
        sigma = np.random.randint(*sigma_lim)

        counts = []
        for mean in means:
            c = (np.random.randn(obs_per_group) * sigma + mean) + \
                (np.random.rand(obs_per_group) - offset) / scale
            c[c < 0] = 0
            counts.append(c)

        p = scipy.stats.f_oneway(*counts)[1]

    params = {'mus': means,
              'sigma': sigma,
              'num_groups': num_groups,
              'obs_per_group': obs_per_group,
              'scale': scale,
              'offset': offset,
              'p-value': p,
              }

    return np.hstack(counts), params


def simulate_table(num_groups=2, obs_per_group=100, num_features=100,
    num_sig=5, depth=10000, mu_lim=[0, 5], sigma_lim=[1, 6],
    scale=0.75, offset=0.1):
    """Simulates a compositional table and groups for ANCOM

    Parameters
    ----------
    num_groups : int
        The number of groups
    obs_per_group : int
        The number of samples per group
    num_features : int
        The number of features which should be included in the table
    num_sig : int
        The number of features which should be significant
    depth : int
        An average number of pseudo sequences used in each sample
    mu_lim : list
        The lower and upper limit for the mean. Multiple means will be
        calculated.
    sigma_lim : list
        The lower and upper limits for the standard deviation. A population
        standard deviation will be used.
    scale : float
        How large the normal distribution should be compared to the random
        noise
    offset : float
        An amount to substract to create sparse data.

    Returns
    -------
    closed : DataFrame
        The scaled, offset sample by feature table
    groups : Series
        Identifiers for the samples providing the groups
    params : list
        The parameters used to simulate the feature distribution.
    """

    # Simulates the significantly different vectors
    simulations = []
    for idx in np.arange(num_sig):
        simulations.append(
            generate_diff_vector(num_groups=num_groups,
                                 obs_per_group=obs_per_group,
                                 mu_lim=mu_lim,
                                 sigma_lim=sigma_lim,
                                 scale=scale,
                                 offset=offset)
        )
    for idx in np.arange(num_sig, num_features):
        simulations.append(
            generate_vector(num_groups=num_groups,
                            obs_per_group=obs_per_group,
                            offset=offset
                            )
        )
    groups = build_groups(num_groups=num_groups, obs_per_group=obs_per_group)
    features = np.array(['f.%i' % (i + 1) for i in np.arange(num_features)])
    samples = np.array(['s.%i' % (i + 1) for i in
                       np.arange(num_groups * obs_per_group)])

    table, params = zip(*simulations)

    closed = pd.DataFrame(
        data=closure(np.round(np.vstack(table).transpose() * depth, 0) + 1),
        columns=features,
        index=samples
    )
    groups = pd.Series(groups, index=samples, name='grouping')

    return closed, groups, params
