from __future__ import division

import numpy as np
import scipy.special as sp
import scipy.stats as stats


def calc_ttest_1(sample, x0, counts, alpha=0.05):
    """Calculates statistical power for a one-sample t test

    This is based on [1]_.

    Parameters
    ----------
    sample : array
        The sample to be compared
    x0 : float
        The mean value compared to the distribution
    counts : array
        The sample sizes used to calculate power
    alpha : float, optional
        The critical value for power. Default is 0.05.

    Returns
    -------
    ndarray
        This describes the probability of seeing a signifigant difference
        between the sample and mean for the specified number of observations
        (count) and critical value based on the one sample t test.

    References
    ----------
    .. [1] Lui, X.S. (2014) *Statistical power analysis for the social and
    behavioral sciences: basic and advanced techniques.* New York: Routledge.
    378 pg.
    """
    # Gets the distribution paramteres
    [x, s] = _get_vitals(sample)
    # Gets the degrees of freedom
    df = counts - 1
    # Gets the noncentrality paramter
    noncentrality = np.absolute(x - x0) / s * np.sqrt(counts)
    # Gets the t statistic
    tsu = stats.t.ppf(1 - alpha / 2, df)
    tsl = stats.t.ppf(alpha / 2, df)
    # Calculates the power
    power = 1 - (sp.nctdtr(df, noncentrality, tsu) +
                 sp.nctdtr(df, noncentrality, tsl))

    return power


def calc_ttest_ind(sample1, sample2, counts, alpha=0.05):
    """Calculates statistical power for a two sample t test

    This is based on [1]_.

    Parameters
    ----------
    sample1, sample2 : array
        The samples being tested
    counts : array
        the number of observations per sample to be used to test the power
    alpha : float
        The critical value for power calculations

    Returns
    -------
    ndrray
        This describes the probability of seeing a signifigant difference
        between the samples for the specified number of observations
        (count) and critical value based on the independent two sample t test.

    References
    ----------
    .. [1] Lui, X.S. (2014) *Statistical power analysis for the social and
    behavioral sciences: basic and advanced techniques.* New York: Routledge.
    378 pg.
    """
    # Gets the distribuation characterization
    [x1, s1] = _get_vitals(sample1)
    [x2, s2] = _get_vitals(sample2)

    # Calculates the degrees of freedom
    df = (counts - 1) * np.square(s1 + s2)/(np.square(s1) + np.square(s2))

    # Calculates the non centrality parameter
    noncentrality = (np.absolute(x1 - x2) /
                     (np.sqrt(np.square(s1) / counts + np.square(s2) / counts))
                     )

    tsu = stats.t.ppf(1 - alpha / 2, df)
    tsl = stats.t.ppf(alpha / 2, df)
    power = 1 - (sp.nctdtr(df, noncentrality, tsu) +
                 sp.nctdtr(df, noncentrality, tsl))

    return power


def calc_anova(*samples, **kwargs):
    """Calculates statistical power for a one way ANOVA

    This is based on
        Lui, X.S. (2014) *Statistical power analysis for the social and
        behavioral sciences: basic and advanced techniques.* New York:
        Routledge. 378 pg.

    Parameters
    ----------
    samples : ndarrays
        Arrays of observations to be tested.
    counts : array
        the number of observations per sample to be used to test the power
    alpha : float
        The critical value for power calculations

    Returns
    -------
    ndarray
        This describes the probability of seeing a signifigant difference
        between the samples for the specified number of observations
        (count) and critical value.
    """

    # Checks the keywords
    kwds = {'counts': None,
            'alpha': 0.05}
    for k, v in kwargs.items():
        kwds[k] = v
    if kwds['counts'] is None:
        raise ValueError('counts is undefined!')
    counts = kwds['counts']
    alpha = kwds['alpha']

    # Converts the samples to arrays
    samples = [np.asarray(sample) for sample in samples]

    k = len(samples)
    grand_mean = np.hstack(samples).mean()
    pooled = np.sqrt(
        np.sum([np.square(x.std()) * (len(x) - 1) for x in samples]) /
        (np.sum([len(x) for x in samples]) - 2)
        )

    df1 = k - 1
    df2 = k * (counts - 1)

    # Calculates the noncentrality paramter
    noncentrality = np.array([
        np.square((sample.mean() - grand_mean) / pooled)
        for sample in samples
        ]).sum() * counts
    # noncentrality = cohen_f2(*samples) * counts

    fu = stats.f.ppf(1 - alpha, df1, df2)

    # Calculates the power using the non-central F distribution
    power = (1 - sp.ncfdtr(df1, df2, noncentrality, fu))
    # the non central F distribution does not return a value of 1,
    # so we replace nans with a value of 1.
    power[np.isnan(power)] = 1

    return power


def calc_pearson(sample1, sample2, counts, alpha=0.05):
    """Calculates power for pearsons R

    This is based on [1]_.

    Parameters
    ----------
    sample1, sample2 : ndarrays
        Arrays of observations to be tested. The samples must be of the same
        length, and sample positions should match samples.
    counts : array
        the number of observations per sample to be used to test the power
    alpha : float
        The critical value for power calculations

    Returns
    -------
    ndarray
        This describes the probability of seeing a signifigant difference
        between the samples for the specified number of observations
        (count) and critical value based on the pearson method.

    References
    ----------
    .. [1] Lui, X.S. (2014) *Statistical power analysis for the social and
    behavioral sciences: basic and advanced techniques.* New York: Routledge.
    378 pg.
    """
    r = stats.pearsonr(sample1, sample2)[0]

    noncentrality = r / np.sqrt(1 - np.square(r)) * np.sqrt(counts)
    df = counts - 2
    ts = stats.t.ppf(1 - alpha / 2, df)

    power = (1 - sp.nctdtr(df, noncentrality, ts) +
             sp.nctdtr(df, noncentrality, -ts))

    return power


def _get_vitals(sample):
    """Returns a summary of the sample"""
    return sample.mean(), sample.std()
