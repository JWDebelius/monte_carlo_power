import numpy as np
import skbio


def bootstrap_permanova(ids, dm, grouping, permutations=99):
    """Performs a permanova using boostrapped data

    Based on the scikit-bio implementation of a permanova

    Parameters
    ----------
    ids: array-like
        A list of ids in the distance matrix and grouping. The ids do not have
        to be unique. Must be a subset of the ids in both `dm` and `grouping`.
    dm : DistanceMatrix
        Distance matrix containing distances between objects (e.g., distances
        between samples of microbial communities).
    grouping : Series
        Vector indicating the assignment of objects to groups. If `grouping`
        is a DataFrame, `column` must be specified.
    permutations : int, optional
        Number of permutations to use when assessing statistical
        significance. Must be greater than or equal to zero. If zero,
        statistical significance calculations will be skipped and the p-value
        will be ``np.nan``.

    Returns
    -------
    pandas.Series
        Results of the statistical test, including ``test statistic`` and
        ``p-value``.

    Also See
    --------
    skbio.stats.distance.permanova

    """
    ids = np.hstack(ids)
    dm_p = _bootstrap_dm(ids, dm)
    gr_p = grouping.loc[ids]
    gr_p.index = ['%i' for i in np.arange(len(ids))]

    res = skbio.stats.distance.permanova(dm_p, gr_p, permutations=permutations)

    return res


def bootstrap_mantel(ids, dm1, dm2, permutations=99, **kwargs):
    """Performs a mantel test using bootstrapped data

    This is based on the scikit-bio implementation of the mantel test.

    Parameters
    ----------
    ids: array-like
        A list of ids in the distance matrix. These do not have
        to be unique.
    dm1, dm2 : DistanceMatrix
        The input distance matrices to be compared.
    permutations : int, optional
        Number of permutations to use when assessing statistical
        significance. Must be greater than or equal to zero. If zero,
        statistical significance calculations will be skipped and the p-value
        will be ``np.nan``.

    Returns
    -------
    corr_coeff : float
        Correlation coefficient of the test (depends on `method`).
    p_value : float
        p-value of the test.
    n : int
        Number of rows/columns in each of the distance matrices, after any
        reordering/matching of IDs. If ``strict=False``, nonmatching IDs may
        have been discarded from one or both of the distance matrices prior to
        running the Mantel test, so this value may be important as it indicates
        the *actual* size of the matrices that were compared.

    Also See
    --------
    skbio.stats.distance.mantel

    """
    ids = np.hstack(ids)
    dm_1p = _bootstrap_dm(ids, dm1)
    dm_2p = _bootstrap_dm(ids, dm2)

    return skbio.stats.distance.mantel(dm_1p,
                                       dm_2p,
                                       permutations=permutations,
                                       **kwargs)


def _bootstrap_dm(ids, dm, new_names=None):
    """Makes a bootstrapped distance matrix

    Parameters
    ----------
    ids: array-like
        A list of ids in the distance matrix. These do not have
        to be unique.
    dm : DistanceMatrix
        The distance matrix object to resample.
    new_names: array_like, optional
        The names to be used in the new array. Note, this must be
        unique. If nothing is specified, a numeric index will be
        used.

    Returns
    -------
        A DistanceMatrix with the samples above and the index
        names

    """
    if new_names is None:
        new_names = np.arange(0, len(ids))
    dm_ids = dm.ids
    id_pos = [dm_ids.index(id_) for id_ in ids]
    dm_data = dm.data[id_pos][:, id_pos]

    return skbio.DistanceMatrix(dm_data, new_names)
