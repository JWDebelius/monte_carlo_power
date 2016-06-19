import numpy as np
import pandas as pd
import scipy
import skbio


def simulate_parametric_distance_matrix(num_samples, num0=None, wdist=[0, 0.5],
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
    for idx in xrange(length-1):
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
    vec = np.random.normal(loc=mean, scale=std, size=size)
    vec[vec < 0] = np.absolute(vec[vec < 0])
    vec[vec > 1] = 1
    return vec


def _vec_size(length):
    """Defines a group size for a distance matrix"""
    return ((length) * (length - 1))/2


def _permanova_ssw(dm, groups):
    """Calculates the within distance sum of squares"""
    total_samples = groups.shape[0]
    metadata_dm = skbio.DistanceMatrix.from_iterable(
        groups,
        metric=scipy.spatial.distance.cityblock,
        keys=groups.index.values
        )
    return np.sum((metadata_dm.condensed_form() == 0).astype(int) *
                  np.square(dm.condensed_form())) / total_samples


def _permanova_sst(dm, groups):
    """Calculates the total sum of squares"""
    total_samples = groups.shape[0]
    return np.sum(np.square(dm.condensed_form())) / total_samples


def coeffient_of_determination(dm, groups):
    """Calculates the coeffecient of determination"""
    # Gets the coeffecients
    ssw = _permanova_ssw(dm, groups)
    sst = _permanova_sst(dm, groups)
    return float(ssw) / float(sst)


def cohens_f2(dm, groups):
    """Calculates cohens f"""
    r2 = coeffient_of_determination(dm, groups)
    return r2 / float(1 - r2)


def omega2(dm, groups):
    """Calculates omega-square, a less biased effect size estimator

    Based on the implementation in R's micropower package
    https://github.com/brendankelly/micropower/blob/master/R/micropower.R
    """

    df_treat = len(set(groups)) - 1
    df_total = groups.shape[0] - df_treat
    df_error = df_total - df_treat

    ssw = _permanova_ssw(dm, groups)
    sst = _permanova_sst(dm, groups)
    ssa = sst - ssw

    error_ms = ssw / df_total

    omega = (ssa - (df_treat * error_ms))/(sst+error_ms)

    return omega
