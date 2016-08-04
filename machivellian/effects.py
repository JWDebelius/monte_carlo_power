#-----------------------------------------------------------------------------
# Copyright (c) 2016, Machiavellian Project.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
from scipy.stats import norm as z
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower

tt = TTestIndPower()
ft = FTestAnovaPower()


def f_effect(counts, power, alpha=0.05, groups=2):
    """Estimates the effect size based on an F distribution

    Parameters
    ----------
    counts : array
        The number of observations for each power depth
    power : array
        The statistical power at the depth specified by `counts`
    alpha : float, optional
        The critial value used to calculate the power

    Returns
    -------
    ndarray
        A standard measure of the difference between the underlying
        populations

    See Also
    --------
        statsmodels.stats.power
        statsmodels.stats.power.FTestAnovaPower
        statsmodels.stats.power.FTestAnovaPower.solve_power
    """
    power = np.atleast_2d(power)
    eff = np.ones(power.shape) * np.nan
    for idx, pwr in enumerate(power):
        for idy, (count, pwr_) in enumerate(zip(*(counts, pwr))):
            try:
                eff[idx, idy] = ft.solve_power(
                    effect_size=None,
                    nobs=count,
                    alpha=alpha,
                    power=pwr_,
                    k_groups=groups,
                    )
            except ValueError:
                pass
    eff[power == 1] = np.nan
    return eff


def t_effect(counts, power, alpha=0.05, ratio=1):
    """Estimates the effect size based on a two-tail t distribution

    Parameters
    ----------
    counts : array
        The number of observations for each power depth
    power : array
        The statistical power at the depth specified by `counts`
    alpha : float, optional
        The critial value used to calculate the power

    Returns
    -------
    ndarray
        A standard measure of the difference between the underlying
        populations

    See Also
    --------
        statsmodels.stats.power
        statsmodels.stats.power.TTestIndPower
        statsmodels.stats.power.TTestIndPower.solve_power
    """
    power = np.atleast_2d(power)
    eff = np.zeros(power.shape) * np.nan
    for idx, pwr in enumerate(power):
        for idy, (count, pwr_) in enumerate(zip(*(counts, pwr))):
            try:
                eff[idx, idy] = tt.solve_power(
                    effect_size=None,
                    nobs1=count,
                    ratio=ratio,
                    alpha=alpha,
                    power=pwr_
                    )
            except ValueError:
                pass
    eff[power == 1] = np.nan
    return eff


def z_effect(counts, power, alpha=0.05):
    """Estimates the effect size for power based on the z distribution

    This is based on the equations in [1]_, the equation assumes a positive
    magnitude to the effect size and a two-tailed test.

    Parameters
    ----------
    counts : array
        The number of observations for each power depth
    power : array
        The statistical power at the depth specified by `counts`
    alpha : float, optional
        The critial value used to calculate the power

    Returns
    -------
    ndarray
        A standard measure of the difference between the underlying
        populations

    References
    ----------
    .. [1] Lui, X.S. (2014) *Statistical power analysis for the social and
    behavioral sciences: basic and advanced techniques.* New York: Routledge.
    378 pg.
    """
    power = np.atleast_2d(power)
    z_diff = z.ppf(power) + z.ppf(1 - alpha/2)
    eff = np.sqrt(np.square(z_diff) / counts)

    eff[power == 1] = np.nan
    eff[np.isinf(eff)] = np.nan

    return eff


def f_power(counts, effect, alpha=0.05, groups=2):
    """Estimates power from an effect size and the f distribution

    This wraps the statsmodels `statsmodels.stats.power.FTestAnovaPower`

    Parameters
    ----------
    counts : array
        The number of observations for each power depth
    effect : float, ndarray
        A standard measure of the difference between the underlying populations
     alpha : float
        The critial value used to calculate the power

    Returns
    -------
    ndarray
        The statistical power at the depth specified by `counts`

    See Also
    --------
        statsmodels.stats.power
        statsmodels.stats.power.FTestAnovaPower
        statsmodels.stats.power.FTestAnovaPower.solve_power
    """
    if isinstance(effect, np.ndarray):
        effect = np.nanmean(effect)
    power = ft.solve_power(effect_size=effect,
                           nobs=counts,
                           alpha=0.05,
                           power=None,
                           k_groups=groups,
                           )
    return power


def t_power(counts, effect, alpha=0.05, ratio=1):
    """Estimates power based on the two-tailed t distribution

    This wraps the statsmodels `statsmodels.stats.power.TTestIndPower`

    Parameters
    ----------
    counts : array
        The number of observations for each power depth
    effect : float, ndarray
        A standard measure of the difference between the underlying populations
     alpha : float
        The critial value used to calculate the power

    Returns
    -------
    ndarray
        The statistical power at the depth specified by `counts`

    See Also
    --------
        statsmodels.stats.power
        statsmodels.stats.power.TTestIndPower
        statsmodels.stats.power.TTestIndPower.solve_power
    """
    if isinstance(effect, np.ndarray):
        effect = np.nanmean(effect)
    power = tt.solve_power(effect_size=effect,
                           nobs1=counts,
                           alpha=0.05,
                           power=None,
                           ratio=ratio)
    return power


def z_power(counts, effect, alpha=0.05):
    """Estimates power for a z distribution from an effect size

    This is based on the equations in [1]_, the equation assumes a positive
    magnitude to the effect size and a two-tailed test.

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
    power = z.cdf(effect * np.sqrt(counts) - z.ppf(1 - alpha/2))
    return power
