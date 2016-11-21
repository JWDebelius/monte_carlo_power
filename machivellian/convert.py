#-----------------------------------------------------------------------------
# Copyright (c) 2016, Machiavellian Project.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
#-----------------------------------------------------------------------------

from __future__ import division

import numpy as np


def convert_d_to_r(d, sd=None, n1=1, n2=1):
    """Converts a Cohen's d value to an odds ratio

    Parameters
    ----------
    d : float
        The Cohen's d effect size
    sd : float, optional
        The variance in d. Is this is not provided, the variance in r will
        not be calculated.
    n1, n2: float, optional
        If the sample sizes are different, this is used to crrecly weight the
        calculation.

    Returns
    -------
    float
        The correlation between two groups.
    float
        The error in the correlation between d

    Also See
    --------
    convert_r_to_d
    convert_or_to_d
    convert_d_to_or

    References
    ----------
    ..[1] Borenstein, M.; Hedges, L.V.; Higgins, J.P.T; Rothstein, H.R. (2009)
    "Ch. 7: Convering Among Effect Sizes". *Introduction to Meta-Analysis*.
    Hoboken: John Wiley & Sons.
    """
    a = np.square(n1 + n2) / (n1 * n2)
    r = d / np.sqrt(np.square(d) + a)

    if sd is not None:
        sr = (np.square(a) * sd) / np.power((np.square(d) + a), 3)
    else:
        sr = np.nan

    return r, sr


def convert_r_to_d(r, sr=None):
    """Converts between r and cohen's d

    Parameters
    ----------
    r : float
        The correlation between the two samples
    sr: float, optional
        The error for the correaltion

    Returns
    ------
    float
        The cohen's d for the comparison
    float
        The error on the cohen's d for the comparison

    Also See
    --------
    convert_r_to_d
    convert_or_to_d
    convert_d_to_or

    References
    ----------
    ..[1] Borenstein, M.; Hedges, L.V.; Higgins, J.P.T; Rothstein, H.R. (2009)
    "Ch. 7: Convering Among Effect Sizes". *Introduction to Meta-Analysis*.
    Hoboken: John Wiley & Sons.

    """
    d = (2 * r) / np.sqrt(1 - np.square(r))

    if sr is not None:
        sd = (4 * sr) / np.power((1 - np.square(r)), 3)
    else:
        sd = np.nan

    return d, sd


def convert_or_to_d(lor, slor=None):
    """Converts from an odds ratio to Cohen's d

    Parameters
    ----------
    lor_ : float
        The log odds ratio for the sample
    slor_ : float, optional
        The error associated with the log odds ratio

    Returns
    -------
    float:
        The value of cohen's d
    float
        The error associated with the Cohen's d value

    Also See
    --------
    convert_r_to_d
    convert_d_to_r
    convert_d_to_or

    References
    ----------
    ..[1] Borenstein, M.; Hedges, L.V.; Higgins, J.P.T; Rothstein, H.R. (2009)
    "Ch. 7: Convering Among Effect Sizes". *Introduction to Meta-Analysis*.
    Hoboken: John Wiley & Sons.

    """
    d = np.sqrt(3) / np.pi * lor

    if slor is not None:
        sd = slor * 3 / np.square(np.pi)
    else:
        sd = np.nan

    return d, sd


def convert_d_to_or(d, sd=None):
    """Converts from an Cohen's d to an odds ratio

    Parameters
    ----------
    d : float
        The Cohen's d effect size
    sd : float, optional
        The variance in d. Is this is not provided, the variance in r will
        not be calculated.

    Returns
    -------
    float:
        The value of the log odds ratio
    float
        The error associated with the log odds ratio

    Also See
    --------
    convert_r_to_d
    convert_or_to_d
    convert_d_to_r

    References
    ----------
    ..[1] Borenstein, M.; Hedges, L.V.; Higgins, J.P.T; Rothstein, H.R. (2009)
    "Ch. 7: Convering Among Effect Sizes". *Introduction to Meta-Analysis*.
    Hoboken: John Wiley & Sons.

    """
    lor = d * np.pi / np.sqrt(3)
    if sd is None:
        slor = np.nan
    else:
        slor = sd * np.square(np.pi) / 3

    return lor, slor

