
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
    sn.regplot(y, x,
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


def plot_ttest_ind_distributions(axd, axs, mu1, mu2, sigma1, sigma2, sample1,
    sample2, color1=None, color2=None, bins=None):
    """Makes a pretty plot of the distribution and samples

    Parameters
    ----------
    axd, axs: Axes
        The matplotlib axes for the distribution (`axd`) and the samples
        (`axs`)
    mu1, mu2: float
        The means of the underlying normal distributions
    sigma1, sigma2: float
        The standard deviation for the underlying distributions
    sample1, sample2: array-like
        The samples drawn from the distributions
    color1, color2: None, array-like, str
        By default, the values will use the seaborn colorpallet values for the
        first and third colors. (Blue and red if using the default pallet.)
        Otherwise, a valid color representation like a 3 or 4 element iterable,
        hexcode, or color string can be used.
    bins: None, ndarray
        The values which should be used for the data distribution. f None are
        provided, bins will be an ndarray between -30 and 30, counting up by
        2.5.
    """
    if color1 is None:
        color1 = sn.color_palette()[0]
    if color2 is None:
        color2 = sn.color_palette()[2]

    if bins is None:
        bins = np.arange(-30, 30, 2.5)

    # Generates the normal distributions
    x = np.linspace(bins.min(), bins.max(), 250)
    y1 = scipy.stats.norm.pdf(x, loc=mu1, scale=sigma1)
    y2 = scipy.stats.norm.pdf(x, loc=mu2, scale=sigma2)
    # Plots the noraml distributions
    axd.plot(x, y1, color=color1)
    axd.plot(x, y2, color=color2)

    # Plots the samples
    sn.distplot(sample1, ax=axs, norm_hist=True,
                kde=False, bins=bins, color=color1)
    sn.distplot(sample2, ax=axs, norm_hist=True,
                kde=False, bins=bins, color=color2)

    # Cleans up the axes
    axd.set_yticks([-1])
    axd.set_xticklabels('')

    sn.despine(ax=axd, left=True, right=True, top=True, offset=5)
    sn.despine(ax=axs, left=True, right=True, top=True, offset=5)


def plot_t_hypotheses(ax, nct, df, alpha=0.05, color1=None, color2=None):
    """Plots data distribution and hypotheses

    Parameters
    ----------
    ax : axes
        The axis on which the data should be plotted
    nct : float
        The noncentrality parameter for the sample size
    df: float
        The degrees of freedom for the t distribution (usually the number of
        observations - 1)
    alpha: float, optional
        The type I error rate, or critical value for the test. By default, this
        is 0.05.
    color1, color2: None, array-like, str
        By default, the values will use the seaborn colorpallet values for the
        first and third colors. (Blue and red if using the default pallet.)
        Otherwise, a valid color representation like a 3 or 4 element iterable,
        hexcode, or color string can be used.
    """

    # Calculates the curves for the null and alternate hypotheses
    crit = scipy.stats.t.ppf(1 - alpha/2, df=df)
    x = np.arange(-5, 9, 0.1)
    y1 = scipy.stats.t.pdf(x, df=df)
    y2 = scipy.stats.t.pdf(x, loc=nct, df=df)

    # Adds the axis for the plot
    ax.set_ylim([0, 0.5])
    sn.despine(ax=ax, left=True, right=True, top=True, offset=10)
    ax.set_yticks([-1])
    ax.set_xticklabels('')

    if color2 is None:
        color2 = sn.color_palette()[1]
    if color1 is None:
        color1 = [0.15, 0.15, 0.15]

    #  Plots the alternate hypothesis
    ax.plot(x, y2, color=color2)
    ax.plot([nct] * 2, [0.35, 0.42], color=color2)
    ax.fill_between(x[(x < -crit)], y2[(x < -crit)], alpha=0.25, color=color2)
    ax.fill_between(x[(x > crit)], y2[(x > crit)], alpha=0.25, color=color2)

    # Plots the null hypothesis
    ax.plot(x, y1, color=color1)
    ax.plot([0, 0], [0.35, 0.42], color=color1)
    ax.fill_between(x[x <= -crit], y1[x <= -crit], alpha=0.25, color=color1)
    ax.fill_between(x[x >= crit], y1[x >= crit], alpha=0.25, color=color1)

    # Adds annotation
    ax.annotate(s='', xy=(0, 0.4),
                xytext=(nct, 0.4),
                arrowprops={'arrowstyle': '<|-|>',
                            'linewidth': 2,
                            },
                color='k')
    ax.text(s='$\lambda$(n)', x=(nct/2), y=0.42, ha='center', size=15)




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
