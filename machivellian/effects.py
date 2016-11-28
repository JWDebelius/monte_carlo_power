#-----------------------------------------------------------------------------
# Copyright (c) 2016, Machiavellian Project.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.stats import norm as z
import sklearn
from sklearn.model_selection import LeaveOneOut


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


def cv_z_effect(counts, e_power, alpha=0.05, counts_min=10,
                power_min=0.1, power_max=0.95, check_outside=True):
    """Checks the quality of the effect size estimate

    Using Leave One Out Cross Validation, evaluates the quality of the
    model prediction using the effect sizes. We'll also check against
    values in the model which were not predicted because they were outside the
    estimation boundaries.

    Parameters
    ----------
    counts : 1D-ndarray
        The number of observations per group for each power depth.
    e_power : ndarray
        The power used in model prediction. If it is a 1D-array, there should
        be a power value for every value in `counts`. If this is a 2D array,
        there should be as many columns as values in `counts`.
        It should have as many columns as
        there are counts
    alpha : float
        The critial value used to calculate the power
    counts_min: int, optional
        The minimum number of observations for effect size calculation.
        Default is 10, based on a high degree of sampling error for smaller
        sample sizes.
    power_min, power_max: float, optional
        The minimum and maximum power values for effect size calculations.
        These are required because of the aspymotic behavior of the culumative
        distribution function.
    check_outside : bool, optional
        Checks the values outside of the boundary conditions

    Returns
    -------
    effect : float
        The mean effect size for the model
    effect_std : float
        The standard error of the mean of the effect size
    effect_n : float
        The number of values that went into the effect size calculation
    dict
        A dictionary summarizing the root mean square error and R2 values
        from the model prediction
    ndarray
        The counts, actual, and predicted power values for the model

    Raises
    ------
    ValueError
        If the dimensions in counts and the dimensions in power do not
        line up

    Also See
    --------
    sklearn.model_selection.LeaveOneOut
    sklearn.metrics.mean_square_error
    sklearn.metrics.r2_score
    machiavellian.effects.z_effect
    machiavellian.effects.z_power

    """

    counts, e_power = _check_shapes(counts, e_power)

    # Calculates the effect size for each power value
    effects = z_effect(counts, e_power, alpha)
    effect_mask = ((counts >= counts_min) &
                   (e_power >= power_min) &
                   (e_power <= power_max))
    effects[~effect_mask] = np.nan
    index = np.arange(0, len(counts))[effect_mask]

    # Performs leave one out cross validation
    loo = LeaveOneOut()
    summary = []

    for train, test in loo.split(index):
        # Identifies the training and test count values because sklearn doesnt
        # know how to index on anything *but* numeric location.
        train_i = index[train]
        test_i = index[test]

        # Gets the training counts and effects
        train_effects = effects[train_i]

        # Gets test counts and actual power value
        test_count = counts[test_i]
        test_power = e_power[test_i]

        #  Predicts the power
        predict_power = z_power(test_count, train_effects, alpha)

        # Updates the summary
        summary.append([test_count, test_power, predict_power])

    summary = np.hstack(summary).T

    # Summarizes effects based on the training
    effect = {
        'effect': np.nanmean(effects),
        'effect_std': np.nanstd(effects),
        'effect_n': len(index),
        'train_r2': sklearn.metrics.r2_score(summary[:, 1], summary[:, 2]),
        'train_rmse': np.square(
            sklearn.metrics.mean_squared_error(summary[:, 1], summary[:, 2])
            ),
        }

    return effect, summary
