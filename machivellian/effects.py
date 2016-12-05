#-----------------------------------------------------------------------------
# Copyright (c) 2016, Machiavellian Project.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
from scipy.stats import norm as z


def z_effect(counts, power, alpha=0.05):
    """Estimates the effect size for power based on the z distribution

    This is based on the equations in [1]_, the equation assumes a positive
    magnitude to the effect size and a one-tailed test.

    Parameters
    ----------
    counts : 1D-ndarray
        The number of observations for each power depth. This can be the same
        length as `power`, or can contain as many elements as there are in
        `power`.
    power : ndarray
        The statistical power at the depth specified by `counts`. If `power`
        is a 1D array, it should have as many elements as found in `counts`. If
        `power` is a 2D ndarray, it is assumed the columns correspond to
        the sampling depths in counts.
    alpha : float, optional
        The critial value used to calculate the power

    Returns
    -------
    1D-ndarray
        A standard measure of the difference between the underlying
        populations

    Raises
    ------
    ValueError
        If the dimensions in counts and the dimensions in power do not
        line up

    References
    ----------
    .. [1] Lui, X.S. (2014) *Statistical power analysis for the social and
    behavioral sciences: basic and advanced techniques.* New York: Routledge.
    378 pg.
    """
    counts, power = _check_shapes(counts, power)

    z_diff = z.ppf(power) + z.ppf(1 - alpha)
    eff = np.sqrt(np.square(z_diff) / counts)

    eff[power == 1] = np.nan
    eff[np.isinf(eff)] = np.nan

    return eff


def z_power(counts, effect, alpha=0.05):
    """Estimates power for a z distribution from an effect size

    This is based on the equations in [1]_, the equation assumes a positive
    magnitude to the effect size and a one-tailed test.

    Parameters
    ----------
    counts : array
        The number of observations for each power depth
    effect : float
        A standard measure of the difference between the underlying populations
    alpha : float
        The critial value used to calculate the power

    Returns
    -------
    ndarray
        The statistical power at the depth specified by `counts`

    References
    ----------
    .. [1] Lui, X.S. (2014) *Statistical power analysis for the social and
    behavioral sciences: basic and advanced techniques.* New York: Routledge.
    378 pg.

    """
    if isinstance(effect, np.ndarray):
        effect = np.nanmean(effect)
    power = z.cdf(effect * np.sqrt(counts) - z.ppf(1 - alpha))

    return power


def _check_shapes(counts, power):
    """Checks that counts and power have the same shape"""
    counts = np.asarray(counts)
    power = np.asarray(power)
    c_shape = counts.shape
    p_shape = power.shape

    if len(c_shape) > 1:
        raise ValueError('counts must be a one-dimensional array')
    elif (len(p_shape) == 1) and (c_shape == p_shape):
        return counts, power
    elif len(p_shape) == 1:
        raise ValueError('there must be a power value for every resample '
                         'depth')
    elif p_shape[1] == c_shape[0]:
        return np.hstack([counts] * p_shape[0]), np.hstack(power)
    else:
        raise ValueError('there must be a power value for every resample '
                         'depth')
