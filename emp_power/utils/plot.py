from matplotlib import use, rcParams
# use('Agg') #noqa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
rcParams['text.usetex'] = True

sn.set_style()

possible_markers = ['o', 's', '*', '^', 'd', 'x']
marker_sizes = [10, 10, 13, 10, 10, 10]
possible_colors = sn.color_palette('Set1', n_colors=8)


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
