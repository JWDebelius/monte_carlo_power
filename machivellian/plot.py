
#-----------------------------------------------------------------------------
# Copyright (c) 2016, Machiavellian Project.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sn
import statsmodels.formula.api as smf

from machivellian.power import confidence_bound


def gradient_regression(ax, x, y, gradient, data, size=40,
    alpha=0.5, line_color='k', **reg_kwargs):
    """Creates a regression plot colored by gradient metadata

    Parameters
    ----------
    ax : Axes
        The axis where the data should be plotted
    x, y, gradient: str
        The names of the columns in `data` which contain the predictor,
        response, and gradient variables respectively to be used in the
        regression plot.
    data : DataFrame
        A pandas dataframe containing columns `x`, `y`, and `gradient`.
    size : int, optional
        The size of the points on the plot
    alpha : float, optional
        The transparency of the plots
    line_color : string; tuple, optional
        The color regression line.

    Also See
    --------
        matplotlib.pyplot.scatter
        seaborn.regplot
        scipy.stats.linregress
    """

    data.dropna(subset=[x, y, gradient], inplace=True)
    scatter_kws = {'c': data[gradient],
                   'color': None,
                   's': size,
                   'alpha': alpha,
                   'edgecolors': 'None'
                   }
    line_kws = {'color': line_color
                }

    # Plots the regression plot
    sn.regplot(y=y, x=x,
               data=data,
               ax=ax,
               scatter_kws=scatter_kws,
               line_kws=line_kws,
               **reg_kwargs)
    m, b, r, p, se = scipy.stats.linregress(data[x], data[y])
    ax.text(0.95, 0.05, 'm = %1.2f\nr2 = %1.3f' % (m, r), ha='right')


def gradient_residuals(ax, x, y, gradient, data, x_resid=None,
    size=40, alpha=0.5):
    """Creates a colored plot of the regression residuals against a category

    Parameters
    ----------
    ax : Axes
        The axis where the data should be plotted
    x, y, gradient: str
        The names of the columns in `data` which contain the predictor,
        response, and gradient variables respectively to be used in the
        regression plot.
    data : DataFrame
        A pandas dataframe containing columns `x`, `y`, and `gradient`.
    x_resid : str, optional
        The name of the column to be used to plot compared to the residual.
        If nothing is specifed, the `x` variable will be used.
    size : int, optional
        The size of the points on the plot
    alpha : float, optional
        The transparency of the plots
    line_color : string; tuple, optional
        The color regression line.

    Also See
    --------
        matplotlib.pyplot.scatter
        seaborn.residplot
        statsmodels.regression.linear_model.OLS
    """

    if x_resid is None:
        x_resid = x
    scatter_kws = {
                   'c': data[gradient],
                   'color': None,
                   's': size,
                   'alpha': alpha,
                   'edgecolors': 'None'
                    }

    #  Fits the regression
    fit = smf.ols('%s ~ %s' % (y, x), data=data).fit()
    resid_vec = data[x_resid]

    ax.scatter(resid_vec.loc[fit.resid.index], fit.resid, **scatter_kws)


def format_regression_axis(ax):
    """Makes a clean, pretty axis for the regression plots

    This sets axis limits at 0, 1, with ticks at 0.25, and removes all axis
    labels. The method is primarily used for the power regression plots.

    Parameters
    ----------
    ax : Axis
        The axis to be formatted

    """
    ax.set_xticks(np.arange(0, 1.1, 0.25))
    ax.set_yticks(np.arange(0, 1.1, 0.25))
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    sn.despine(ax=ax)


def format_residual_axis(ax, xlim=None, ylim=None, num_ticks=5, line_kws=None):
    """Creates a clean axis for the residual plots

    This method is primarily designed to make neatly formatted axes for
    residual axes in a plot.

    Parameters
    ----------
    ax : Axis
        The axis to be formatted
    xlim, ylim : list, optional
        The x and y limits for the residual axis. By default, the current x
        limits will be used.
    ylim : list, optional
        The y limits for the residual axis. By default, these will be set as
        symetrical values at the limits.
    num_ticks : int, optional
        The number of ticks ot use on the x and y axis.
    """
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = _get_symetrical(ax.get_ylim())
    xticks = _set_ticks(xlim, num_ticks)
    yticks = _set_ticks(ylim, num_ticks)

    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel('')

    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels('')
    ax.set_ylabel('')

    ax.plot(ax.get_xticks(), [0] * len(ax.get_xticks()),
            marker='|', color='k', markeredgewidth=1, linestyle='',
            linewidth=0.75)

    sn.despine(ax=ax, top=True, bottom=True, right=True)
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False


def summarize_regression(summary, test_names, titles, x, y, gradient,
    alpha, ylabel, ylim):
    """Makes a figure with regression features

    Parameters
    ----------
    summary : DataFrame
        A summary dataframe of the power values. This must include the columns
        described in `x`, `y`, `gradient`, and a column called `test` which
        contains the tests in `test_names`
    test_name : list
        The test names to display and include in the plot.
    x, y, gradient: str
        The names of the columns in `summary` which contain the predictor,
        response, and gradient variables respectively to be used in the
        regression plot.
    alpha : float, optional
        The transparency of the plots
    ylabel : str
        A descriptor for the regression and residual of the plot
    ylim : list
        The limits for hte residual labels

    Returns
    -------
    Figure

    """

    num_plots = len(test_names)

    # Generates the figure
    fig, (ax_r1, ax_r2) = plt.subplots(2, num_plots)
    fig.set_size_inches(num_plots * 2, 4)

    # ...
    for (test_name, title, ax_reg, ax_res) in \
            zip(*(test_names, titles, ax_r1, ax_r2)):
        # Plots the regression
        gradient_regression(
            ax_reg,
            x=x, y=y,
            gradient=gradient,
            data=summary.loc[summary['test'] == test_name].copy(),
            alpha=alpha
            )
        format_regression_axis(ax_reg)
        ax_reg.set_title(title)

        # Plots the gradient
        gradient_residuals(
            ax_res,
            x=x, y=y,
            gradient=gradient,
            data=summary.loc[summary['test'] == test_name].copy(),
            alpha=alpha
            )
        ax_res.plot([0, 0], [-1, 2], 'k-')
        format_residual_axis(ax_res, xlim=[0, 1], ylim=ylim, num_ticks=4)

    # Adds axis labels
    ax_r1[0].set_yticklabels(ax_r1[0].get_yticks())
    ax_r1[0].set_ylabel('%s regression' % ylabel)
    ax_r2[0].set_yticklabels(ax_r2[0].get_yticks())
    ax_r2[0].set_ylabel('%s residuals' % ylabel)

    return fig


def plot_alternate_t(noncentrality, df, alpha=0.05, ax=None):
    """Creates an under the curve plot of statistical power

    Parameters
    ----------
    noncentrality: float
        The noncentrality parameter, representing the offset between
        the null and alternative hypothesis. This can be calculated
        as the product of `machiavellian.traditional.effect_ttest_1`
        times the square root of the sample size.

    df: int
        The number of degrees of freedom associated with the analysis.
        This should  be the number of observations less 1.

    alpha: float
        The critical value for the analysis. This is a two-tailed test.

    axes : Axes, optional
        A matplotlib axes object where the results should be plotted.

    Returns
    -------
        The joint distributions is plotted on the axes.
    """

    if ax is None:
        ax = plt.axes()

    x, y1, y2, crit = _summarize_t(noncentrality, df, alpha)
    color1, color2 = _get_colors()

    ax.plot(x, y1, color=color1)
    ax.plot(x, y2, color=color2)

    sn.despine(ax=ax, left=True, offset=10)

    return ax


def add_noncentrality(noncentrality, df, alpha, ax):
    """Labels the noncentrality parameter"""
    x, y1, y2, crit = _summarize_t(noncentrality, df, alpha)
    height = np.max(np.hstack([y1, y2]))
    x0 = 0
    x1 = noncentrality

    ax.annotate(s='', xy=(x0, height), xytext=(x1, height),
                arrowprops={'arrowstyle': '<|-|>',
                            'linewidth': 2},
                color='k')
    ax.text(s='$\lambda(n)$', x=(x1 - x0) / 2, y=height * 1.05,
            ha='center', size=15)


def plot_power_curve(ax, counts, power_trace=None, power_scatter=None,
                     color=None, ci_alpha=0.05):
    """Plots power as a function of the number of observations

    Parameters
    ----------


    Returns
    -------


    Raises
    ------


    """
    if ((power_trace is None) and (power_scatter is None)):
        raise ValueError('No power value has been specified.')

    if color is None:
        color = sn.color_palette()[0]

    # Plots the curve
    if power_trace is not None:
        pwr_mean, pwr_lo, pwr_hi = _summarize_trace(power_trace,
                                                    ci_alpha=ci_alpha)
        if not np.isnan(pwr_lo).all():
            ax.fill_between(counts,
                            pwr_lo,
                            pwr_hi,
                            color=color,
                            alpha=0.5)
        ax.plot(counts,
                pwr_mean,
                color=color)

    # Plots the scatter data
    if power_scatter is not None:
        ax.plot(counts,
                power_scatter.T,
                marker='o',
                markerfacecolor='None',
                markeredgecolor=color,
                markeredgewidth=1
                )

    # Cleans up the axes
    sn.despine(ax=ax)


def _summarize_t(noncentrality, df, alpha=0.05):
    """Gets the features for a t-distribution

    Parameters
    ----------
    noncentrality: float
        The noncentrality parameter, representing the offset between
        the null and alternative hypothesis. This can be calculated
        as the product of `machiavellian.traditional.effect_ttest_1`
        times the square root of the sample size.

    df: int
        The number of degrees of freedom associated with the analysis.
        This should  be the number of observations less 1.

    alpha: float
        The critical value for the analysis. This is a two-tailed test.
    """

    x = np.arange(-7.5, 7.6, 0.1)
    y1 = scipy.stats.t.pdf(x, loc=0, scale=1, df=df)
    y2 = scipy.stats.t.pdf(x, loc=noncentrality, scale=1, df=df)

    crit = scipy.stats.t.ppf(1 - alpha/2, df=df)

    return x, y1, y2, crit


def _summarize_trace(power_trace, ci_alpha=None):
    """Summarizes the power curve for plotting"""
    power_trace = np.atleast_2d(power_trace)
    pwr_mean = power_trace.mean(0)

    if power_trace.shape[0] == 1:
        pwr_err = np.zeros(power_trace.shape) * np.nan
    elif ci_alpha is None:
        pwr_err = power_trace.std(0)
    else:
        pwr_err = confidence_bound(power_trace, alpha=ci_alpha, axis=0)

    pwr_lo = pwr_mean - pwr_err
    pwr_hi = pwr_mean + pwr_err

    return pwr_mean, pwr_lo, pwr_hi


def _get_colors():
    """Provides distribution colors"""
    color2 = np.round(sn.color_palette()[2], 5)
    color1 = [0.15, 0.15, 0.15]

    return color1, color2


def _set_ticks(lims, num_ticks):
    """Sets the ticks limits on a plot"""
    range_ = lims[1] - lims[0]
    interval = range_ / num_ticks
    ticks = np.arange(lims[0], lims[1] + interval / 2, interval)
    return ticks


def _get_symetrical(lims):
    """Calculates symetrical limits"""
    lim_max = max(lims)
    lims = [-lim_max, lim_max]
    return lims
