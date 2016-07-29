import copy
from functools import partial

import numpy as np
import pandas as pd

import emp_power.effects as eff


def summarize_power(power_summary, sim_num, test, colors, num_groups=2,
    dists=None):
    """Generates a dataframe describing a power calculation run"""

    # Pulls out the required information
    if dists is None:
        dists = ['f', 't', 'z']

    # Builds the initia data frame
    run_summary = _build_summary_frame(power_summary)

    #  Adds static information
    run_summary['test'] = test
    run_summary['alpha'] = power_summary['alpha']
    run_summary['sim_num'] = sim_num

    # Includes the count values to be plotted
    run_summary['colors'] = run_summary['counts'].replace(colors)

    # Calculates the effect sizes
    _calculate_effect_size(run_summary, dists, num_groups)

    # # Calculates the mean effects and the power
    _calculate_power(run_summary, dists, num_groups)

    run_summary['index'] = (
        run_summary['test'] + '.' +
        run_summary['sim_num'].apply(lambda x: '%i' % x) + '.' +
        run_summary['sim_position'].apply(lambda x: '%i' % x)
        )
    run_summary.set_index("index", inplace=True)

    return run_summary


def modify_effect_size(df, drop_index, distributions=None, num_groups=2):
    """Calculates a modified effect size based on dropped values"""
    copy_cols = ['emperical', 'counts', 'color', 'simulation', 'sim_pos',
                 'test']
    copy_cols.extend(['%s_effect' % d for d in distributions])
    df_mod = copy.deepcopy(df)
    df_mod.loc[drop_index, ['%s_effect' % d for d in distributions]] = np.nan
    _calculate_power(df_mod, distributions, num_groups)

    return df_mod


def _build_summary_frame(sim):
    """Builds the intial dataframe summarizing the run"""
    counts = sim['counts']
    emperical = sim['emperical_power']

    # Determines the number of samples handled in the summary
    (empr_r, empr_c) = emperical.shape

    # Draws the traditional power
    if 'traditional_power' in sim.keys():
        traditional = sim['traditional_power']
    else:
        traditional = np.nan * np.ones(counts.shape)

    # Sets up the summary dictionary
    run_summary = pd.DataFrame.from_dict(
        {'counts': np.hstack(np.array([counts] * empr_r)),
         'emperical': np.hstack(emperical),
         'traditional': np.hstack(np.array([traditional] * empr_r)),
         'sim_position': np.hstack([np.arange(empr_c) + i * 10
                                    for i in np.arange(empr_r)]),
         })

    return run_summary


def _calculate_effect_size(df, distributions, num_groups=2):
    """Adds the effect sizes to the dataframe"""
    for dist in distributions:
        f_ = partial(effect_lookup[dist], col2='emperical',
                     num_groups=num_groups)
        df['%s_effect' % dist] = df.apply(f_, axis=1)


def _calculate_power(df, distributions, num_groups=2):
    """Adds power calculations to the dataframe"""
    mean_lookup = (df.groupby('sim_num').mean()
                   [['%s_effect' % d for d in distributions]].to_dict())
    for dist in distributions:
        f_ = partial(power_lookup[dist],
                     col2='%s_mean' % dist,
                     num_groups=num_groups)
        df['%s_mean' % dist] = df['sim_num'].replace(mean_lookup)
        df['%s_power' % dist] = df.apply(f_, axis=1)


def calc_f_effect(x, col2='emperical', num_groups=2):
    """Wraps the f-based power calculation for pandas `apply`"""
    effect = eff.f_effect([x['counts']],
                          [x[col2]],
                          alpha=x['alpha'],
                          groups=num_groups)[0][0]
    return effect


def calc_t_effect(x, col2='emperical', num_groups=2):
    """Wraps the t-based power calculation for pandas `apply`"""
    effect = eff.t_effect([x['counts']],
                          [x[col2]],
                          alpha=x['alpha'])[0][0]
    return effect


def calc_z_effect(x, col2='emperical', num_groups=2):
    """Wraps the z-based power calculation for pandas `apply`"""
    effect = eff.z_effect([x['counts']],
                          [x[col2]],
                          alpha=x['alpha'])[0][0]
    return effect


def calc_f_power(x, col2, num_groups=2):
    """Wraps the f-based power calulation"""
    power = eff.f_power(counts=x['counts'],
                        effect=x[col2],
                        alpha=x['alpha'],
                        groups=num_groups)
    return power


def calc_t_power(x, col2, num_groups=2):
    """Wraps the f-based power calulation"""
    power = eff.t_power(counts=x['counts'],
                        effect=x[col2],
                        alpha=x['alpha'])
    return power


def calc_z_power(x, col2, num_groups=2):
    """Wraps the f-based power calulation"""
    power = eff.z_power(counts=x['counts'],
                        effect=x[col2],
                        alpha=x['alpha'])
    return power


effect_lookup = {'f': calc_f_effect,
                 't': calc_t_effect,
                 'z': calc_z_effect
                 }

power_lookup = {'f': calc_f_power,
                't': calc_t_power,
                'z': calc_z_power}
