import numpy as np

import scipy.stats

from statsmodels.stats.power import FTestAnovaPower
from scipy.stats import norm as z
ft = FTestAnovaPower()


WIDTH = width = 2.5/0.9*0.95 + 0.5


def extrapolate_f(counts, pwr_, cnts, alpha=0.05):
    """Converts emperical power to extrapolated

    Parameters
    ----------
    counts : array
        The number of observations which should be used in the final power
        result.
    pwr_ : array
        The observed power. Each column corresponds to the number of
        observations used in `cnts`. The rows correspond to different runs
    cnts : array
        The number of observations drawn to calculate the observed power.
    alpha : float, optional
        The critical value for power calculations.

    Returns
    -------
    power : array
        The extrapolated power for the number of observations given by `counts`

    """
    # Gets the average emperical effect size
    effs = np.zeros(pwr_.shape) * np.nan
    for idx, pwr in enumerate(pwr_):
        for idy, cnt in enumerate(cnts):
            try:
                effs[idx, idy] = ft.solve_power(None, cnt, alpha, pwr[idy])
            except:
                pass
    eff_mean = np.nanmean(effs)
    # Calculates the extrapolated power curve
    extr_pwr = ft.solve_power(effect_size=eff_mean,
                              nobs=counts,
                              alpha=0.05,
                              power=None)

    return extr_pwr


def linear_confidence(x, y, xpred=None, alpha=0.05):
    """Calculates the confidence interval on the linear regression

    Based on Thomas Holderness's Blog and implementation
    (https://tomholderness.wordpress.com/2013/01/10/confidence_intervals/)

    Parameters
    ----------


    Returns
    -------

    """
    xmin = x.min()
    xmax = x.max()
    xint = (xmax - xmin)/10
    if xpred is None:
        xpred = np.arange(xmin - xint, xmax + 2 * xint, xint)

    # Fits the data to a regression model
    m, b, r, p, se = scipy.stats.linregress(x, y)

    # Calculates the confidence interval for the test data
    mean_x = x.mean()
    n = len(x)
    t = scipy.stats.t.ppf(1 - alpha / 2, n - 2)

    confs = (t * np.sqrt((se/(n-2))*(1.0/n +
                                     (np.power((xpred-mean_x), 2) /
                                      ((np.sum(np.power(x, 2))) - n *
                                      (np.power(mean_x, 2)))))))

    ypred = m * xpred + b

    return xpred, ypred, confs


def format_ax(ax, fig=None, **kwargs):
    """Formats the axis"""
    kwds = {'xlim': [0, 1],
            'ylim': [0, 1],
            'show_x': False,
            'show_y': False,
            'xticks': None,
            'yticks': None,
            'fontsize1': 11,
            'fontsize2': 13,
            'xlabel': '',
            'ylabel': '',
            'title': '',
            'fun_': None,
            'label': '',
            'label_pos': (0.05, 0.85)}
    for k, v in kwargs.iteritems():
        kwds[k] = v

    if fig is not None:
        fig.set_size_inches((WIDTH, WIDTH))
        ax.set_position((0.5/WIDTH, 0.5/WIDTH, 2.5/WIDTH, 2.5/WIDTH))

    # Sets the axis limits
    ax.set_xlim(kwds['xlim'])
    ax.set_ylim(kwds['ylim'])

    # Sets up the xtick labels
    if kwds['xticks'] is None:
        kwds['xticks'] = ax.get_xticks()
    if kwds['yticks'] is None:
        kwds['yticks'] = ax.get_yticks()

    # Adds a label
    ax.text(kwds['label_pos'][0], kwds['label_pos'][1], kwds['label'])

    # Adds the tick labels
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_title(kwds['title'])

    if kwds['show_x'] and kwds['show_y']:
        ax.set_xticklabels(kwds['xticks'], size=kwds['fontsize1'])
        ax.set_yticklabels(kwds['yticks'], size=kwds['fontsize1'])
        ax.set_xlabel(kwds['xlabel'], size=kwds['fontsize2'])
        ax.set_ylabel(kwds['ylabel'], size=kwds['fontsize2'])
        if fig is not None:
            ax.set_position((0.5/WIDTH, 0.5/WIDTH, 2.5/WIDTH, 2.5/WIDTH))
            fig.set_size_inches((WIDTH, WIDTH))
    elif kwds['show_x']:
        ax.set_xticklabels(kwds['xticks'], size=kwds['fontsize1'])
        ax.set_xlabel(kwds['xlabel'], size=kwds['fontsize2'])
        if fig is not None:
            fig.set_size_inches((WIDTH, WIDTH))
            ax.set_position((0.5/WIDTH, 0.5/WIDTH, 2.5/WIDTH, 2.5/WIDTH))
    elif kwds['show_y']:
        ax.set_yticklabels(kwds['yticks'], size=kwds['fontsize1'])
        ax.set_ylabel(kwds['ylabel'], size=kwds['fontsize2'])
        if fig is not None:
            fig.set_size_inches((WIDTH, WIDTH))
            ax.set_position((0.5/WIDTH, 0.5/WIDTH, 2.5/WIDTH, 2.5/WIDTH))


def z_effect(counts, power, alpha=0.05):
    """Estimates the effect size for power based on the z distribution

    This is based on the equations in
        Lui, X.S. (2014) *Statistical power analysis for the social and
        behavioral sciences: basic and advanced techniques.* New York:
        Routledge. 378 pg.
    The equation assumes a positive magnitude to the effect size and a
    two-tailed test.

    Parameters
    ----------
    counts : array
        The number of observations for each power depth
    power : array
        The statistical power at the depth specified by `counts`
    alpha : float
        The critial value used to calculate the power

    Returns
    effect : array
        T A standard measure of the difference between the underlying
        populations
    """
    z_diff = z.ppf(power) + z.ppf(1 - alpha / 2)
    eff = np.sqrt(2 * np.square(z_diff) / counts)

    # eff = eff[~ np.isinf(eff) & ~np.isnan(eff)]

    return eff


def z_power(counts, eff, alpha=0.05):
    """Estimates power for a z distribution from an effect size

    This is based on the equations in
        Lui, X.S. (2014) *Statistical power analysis for the social and
        behavioral sciences: basic and advanced techniques.* New York:
        Routledge. 378 pg.
    The equation assumes a positive magnitude to the effect size and a
    two-tailed test.

    Parameters
    ----------
    counts : array
        The number of observations for each power depth
    effect : float
        A standard measure of the difference between the underlying populations
     alpha : float
        The critial value used to calculate the power

    Returns
    power : array
        The statistical power at the depth specified by `counts`

    """
    power = ((z.cdf(eff * np.sqrt(counts/2) - z.ppf(1 - alpha/2)) +
             (z.cdf(z.ppf(alpha/2) - eff * np.sqrt(counts/2)))))
    return power


def f_effect(counts, power, alpha=0.05):
    """Calculates the f-test based effect size

    Parameters
    ----------
    counts : array
        The number of observations for each power depth
    power : array
        The statistical power at the depth specified by `counts`
    alpha : float
        The critial value used to calculate the power

    Returns
    --------
    effect : array
        A standard measure of the difference between the underlying
        populations
    """

    eff = np.zeros(np.product(power.shape)) * np.nan
    for idx, c in enumerate(counts):
        for idy, p in enumerate(power[:, idx]):
            try:
                f_effect[idy, idx] = ft.solve_power(None, c, 0.05, p)
            except:
                pass
    # eff = eff[~ np.isinf(eff) & ~np.nan(eff)]

    return eff


def f_power(counts, eff, alpha=0.05):
    """Estimates power for a f distribution from an effect size

    Parameters
    ----------
    counts : array
        The number of observations for each power depth
    effect : float
        A standard measure of the difference between the underlying populations
     alpha : float
        The critial value used to calculate the power

    Returns
    power : array
        The statistical power at the depth specified by `counts`

    """
    # Calculates the extrapolated power curve
    power = ft.solve_power(effect_size=eff,
                           nobs=counts,
                           alpha=alpha,
                           power=None)
    return power


def cohen_d_one_sample(sample, x0):
    """Calculates Cohen's D

    Parameters
    ----------
    sample : array
        The set of observations being analyzed
    x0 : float
        The sample mean

    Returns
    d : float
        The effect size for a one sample t test.

    """
    x1, s1 = sample.mean(), sample.std()
    return np.absolute((x1 - x0) / s1)
