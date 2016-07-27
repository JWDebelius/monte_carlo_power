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
    interval = range_ / num_ticks
    ticks = np.arange(lims[0], lims[1] + interval / 2, interval)
    return ticks


def _get_symetrical(lims):
    lim_max = max(lims)
    lims = [-lim_max, lim_max]
    return lims
