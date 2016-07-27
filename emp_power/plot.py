# from matplotlib import use, rcParams
# use('Agg') #noqa
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sn
import statsmodels.formula.api as smf

# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
# rcParams['text.usetex'] = True

# sn.set_style()

# possible_markers = ['o', 's', '*', '^', 'd', 'x']
# marker_sizes = [10, 10, 13, 10, 10, 10]
# possible_colors = sn.color_palette('Set1', n_colors=8)


def gradient_regression(ax, x, y, gradient, data, size=40,
                        alpha=0.5, line_color='k', **reg_kwargs):
    """Some nice doc string"""

    data.dropna(subset=[x, y, gradient], inplace=True)
    scatter_kws = {
                   'c': data[gradient], # Sets the color gradient
                   's': size, # sets the size to 40 points.
                   'alpha': alpha, # Makes the points partially transparent
                   'edgecolors': 'None' # hides the edges
                   }
    line_kws = {'color': line_color # Sets the regression line color
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


def gradient_residuals(ax, x, y, gradient, data, x_resid=None, cmap='Spectral',
                       size=40, alpha=0.5, line_kws=None):
    """..."""
    if x_resid is None:
        x_resid = x
    scatter_kws = {
                   # 'c': data[gradient], # Sets the color gradient
                   's': size, # sets the size to 40 points.
                   'cmap': cmap, # Sets the colormap as specified
                   'alpha': alpha, # Makes the points partially transparent
                   'edgecolors': 'None' # hides the edges
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
    """Makes a clean, pretty axis"""
    ax.set_xticks(np.arange(0, 1.1, 0.25))
    ax.set_yticks(np.arange(0, 1.1, 0.25))
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])


def format_residual_ax(ax, xlim=None, ylim=None, num_ticks=5):
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
    range_ = lims[1] - lims[0]
    interval = range_ / (num_ticks - 1)
    ticks = np.arange(lims[0], lims[1] + interval / 2, interval)
    return ticks


def _get_symetrical(lims):
    lim_max = max(lims)
    lims = [-lim_max, lim_max]
    return lims


def gradient_comparison(x, y, gradient, power, ax_r, alpha=0.5,
    cmap='Set3', resid_y=[-0.5, 0.25]):
    """..."""
    # Sets up the axes
    [ax1, ax2] = ax_r
    fit = sms.OLS(y, sms.add_constant(x)).fit()
    b, m = fit.params
    bs, ms = fit.bse
    ticks = np.arange(0, 1.51, 0.25)

    ax1.set_xlim([0, 1.5])
    ax1.set_ylim([0, 1.5])
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.set_xticklabels('')
    ax1.set_yticklabels('')
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    ax1.text(1.45, 0.05, 'm = %1.2f +/- %1.2f' % (m, ms), ha='right')

    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim(resid_y)
    ax2.set_xticklabels('')
    ax2.set_yticklabels('')
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    ax2.text(0.95, np.min(resid_y) * 1.05, 'R2 = %1.3f' % fit.rsquared, ha='right')

    scatter_kws = {'c': gradient, 's': 40, 'cmap': cmap, 'alpha': alpha, 'edgecolors': 'None'}
    line_kws = {'color': 'k', 'linewidth': 0.5}

    sn.regplot(x=x,
               y=y, 
               scatter_kws=scatter_kws, 
               line_kws=line_kws, 
               ax=ax1)
    ax2.scatter(power, fit.resid, **scatter_kws)
    ax2.plot([-0.5, 1.5], [0, 0], color='k', linewidth=0.75, linestyle=':')


def plot_effect_sizes(x, y, ax=None, markers=None, colors=None, labels=None,
    xlabel=None, ylabel=None, ticks=np.arange(0, 3.1, 0.5)):
    """Plots the effect size comparisons
    """

    num_samples = len(x)

    # Gets the markers
    if markers is None:
        markers = np.floor(num_samples / 6) * possible_markers
        markers.extend(possible_markers[:np.fmod(num_samples, 6)])
        sizes = np.floor(num_samples / 6) * marker_sizes
        sizes.extend(marker_sizes[:np.fmod(num_samples, 6)])
    else:
        sizes = [10] * len(markers)

    # Gets the colors
    if colors is None:
        colors = np.floor(num_samples / 8) * possible_colors
        colors.extend(possible_colors[:np.fmod(num_samples, 8)])

    # Creates an axis, if necessary
    if ax is None:
        ax = plt.axes()

    # Plots the data
    for xx, yy, mark, size, color in zip(*(x, y, markers, sizes, colors)):
        ax.plot(xx, yy,
                marker=mark,
                markeredgecolor=color,
                markerfacecolor='None',
                markeredgewidth=0.5,
                ms=size,
                )

    # Sets the ticks
    ax.set_xlim([ticks.min(), ticks.max()])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, size=12)
    ax.set_xlabel(xlabel, size=15)

    ax.set_ylim([ticks.min(), ticks.max()])
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, size=12)
    ax.set_ylabel(xlabel, size=15)

    sn.despine(ax=ax)

    return ax
