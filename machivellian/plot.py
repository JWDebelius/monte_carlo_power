#-----------------------------------------------------------------------------
# Copyright (c) 2016, Machiavellian Project.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
import scipy
import seaborn as sn
import statsmodels.formula.api as smf


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
    scatter_kws = {
                   'c': data[gradient],
                   's': size,
                   'alpha': alpha,
                   'edgecolors': 'None'
                   }
    line_kws = {'color': line_color
                }

    # Plots the regression plot
    sn.regplot(y, x,
               data=data,
               ax=ax,
               scatter_kws=scatter_kws,
               line_kws=line_kws,
               **reg_kwargs)
    m, b, r, p, se = scipy.stats.linregress(data[x], data[y])
    ax.text(0.95, 0.05, 'm = %1.2f\nr2 = %1.3f' % (m, r), ha='right')


def gradient_residuals(ax, x, y, gradient, data, x_resid=None,
                       size=40, alpha=0.5, line_kws=None):
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
                   's': size,
                   'alpha': alpha,
                   'edgecolors': 'None'
                    }
    if line_kws is None:
        line_kws = {'color': 'k',
                    'linewidth': 0.75,
                    'linestyle': '--',
                    }

    #  Fits the regression
    fit = smf.ols('%s ~ %s' % (x, y), data=data).fit()
    resid_vec = data[x_resid]

    ax.scatter(resid_vec.loc[fit.resid.index], fit.resid, **scatter_kws)
    ax.plot(ax.get_xlim(), [0, 0], **line_kws)


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


def format_residual_ax(ax, xlim=None, ylim=None, num_ticks=5):
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
    ax.set_xticklabels('')
    ax.set_xlabel('')

    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels('')
    ax.set_ylabel('')


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
