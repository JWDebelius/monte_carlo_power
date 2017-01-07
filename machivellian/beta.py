from functools import partial

import numpy as np
import scipy
import skbio


def expand_otu_ids(ids, counts):
    """Lists the otu id the number of times provided in count

    Paramaters
    ----------
    ids: iterable
        A list of the ids, corresponding to the value in `counts`
    counts : iterable
        A list of count values, named by the id in `ids`.

    Returns
    -------
    1D-ndarray
        the id value listed the number of times for which it has a count
    1D-ndarray
        the list of ids which have zero counts

    """
    ids = ids.astype('object')
    id_list = np.hstack([np.array(count * [id_])
                         for (count, id_) in zip(list(counts), list(ids))])
    return id_list


def collapse_otu_ids(id_list, order=None):
    """Collapses a list of ids appearing in a sample to counts

    Parameters
    ----------
    id_list: ndarray
        the id value listed the number of times for which it has a count
    order: ndarray, optional
        the order in which the final OTU counts should be returned. If no
        order is included, then OTUs will be returned sorted by ID. If an
        order is supplied and does not appear in `id_list`, a value of 0
        will be returned for that ID.

    Returns
    -------
    1D-ndarray
        A list of the ids, corresponding to the value in `counts`
    1D-ndarray
        A list of count values, named by the id in `ids`.

    """
    if order is None:
        order = np.unique(id_list)

    counts = np.array([np.count_nonzero(id_list == id_)
                       for id_ in order])

    return order, counts


def subsample_ids(counts, depth, feature_ids=None, bootstrap=True):
    """Generates a subsampled vector of values for each row in a table

    Parameters
    ----------
    counts : ndarray
        An m x n array containing c total counts. Subsampling will be performed
        on the rows.
    depth : int
        The number of observations to draw per row
    feature_ids: 1D-ndarray, optional
        A 1D-ndarray of length m which names the rows
    bootstrap: bool, optional
        When `true`, subsampling with replacement will be performed.

    Returns
    -------
    ndarray
        An m x n array, where each row sums to `depth`.

    """
    if feature_ids is None:
        feature_ids = np.arange(0, counts.shape[1])

    new_table = []

    for sample in counts:
        expanded = expand_otu_ids(feature_ids, sample)
        subsampled = np.random.choice(expanded, depth, replace=bootstrap)
        new_table.append(collapse_otu_ids(subsampled, order=feature_ids)[1])

    return np.vstack(new_table)


def bootstrap_permanova(obs_ids, obs, depth, grouping, column=None,
                        bootstrap=True, metric=None, permutations=99,
                        metric_kws=None):
    """Calculates a bootstrapped permanova for samples within the OTU table

    Parameters
    ----------
    obs_ids: array-like
        A list of ids in the observation table and grouping. The ids do not
        have to be unique. Must be a subset of the ids in both `obs` and
        `grouping`.
    obs: ndarray
        A pandas dataframe of the observational data where the rows are the
        observations and the columns are the features. Note that if this is
        transformed from a biom object, the object will need to be transposed.
    depth : int
        The number of observations to draw for each observation
    grouping : 1D array-like, DataFrame
        Vector indicating the assignment of objects to groups. For example,
        these could be strings or integers denoting which group an object
        belongs to. If `grouping` is `1-D array_like`, it must be the same
        length and in the same order as the objects in `distance_matrix`.
        If `grouping` is a `DataFrame`, the column specified by `column` will
        be used as the grouping vector. The `DataFrame` must be indexed by the
        IDs in `distance_matrix` (i.e., the row labels must be distance matrix
        IDs), but the order of IDs between `distance_matrix` and the
        `DataFrame` need not be the same. All IDs in the distance matrix must
        be present in the `DataFrame`. Extra IDs in the `DataFrame` are
        allowed (they are ignored in the calculations).
    column, string, optional
        Column name to use as the grouping vector if `grouping` is a
        `DataFrame`. Must be provided if grouping is a `DataFrame`. Cannot
        be provided if `grouping` is 1-D array_like.
    bootstrap: bool, optional
        When `true`, feature counts can be drawn with replacement for each
        observation.
    metric: bool, optional
        The distance metric to be used for the distance matrix calculation. If
        no metric is specified, bray-curtis distance will be used.
    permutations : int, optional
        Number of permutations to use when assessing statistical
        significance. Must be greater than or equal to zero. If zero,
        statistical significance calculations will be skipped and the p-value
        will be ``np.nan``.
    metric_kws: dict, optional
        A key/value pair of keyword arguments for the distance calculation.


    Returns
    -------
    float
        The p-value for the permutation test

    Also See
    --------
    scipy.spatial.distance.braycurtis
    skbio.stats.distance.permanova
    """

    if metric is None:
        metric = scipy.spatial.distance.braycurtis
    elif metric_kws is not None:
        metric = partial(metric, **metric_kws)
    feature_ids = obs.columns

    # Gets the rarified table
    rare = subsample_ids(obs.loc[obs_ids].values,
                         depth=depth,
                         feature_ids=feature_ids,
                         bootstrap=bootstrap)
    grouping = grouping.loc[obs_ids]

    # Calculates the distance matrix from the bootstrapped feature x
    # observation table
    dm = skbio.DistanceMatrix.from_iterable(rare, metric=metric,
                                            keys=obs_ids)

    # Performs the permanova on the distance matrix.
    permanova_res = skbio.stats.distance.permanova(dm, grouping,
                                                   permutations=permutations)

    return permanova_res, dm
