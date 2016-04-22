# -*- coding: utf-8 -*-
#

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.patches as mpatches
import math

import pandas as pd
from pandas.tools.plotting import parallel_coordinates
import numpy as np
import config as cf
import functions as f


def quartile_years_in_position(prop_ds, sa_ds, job_levels, num_bins,
                               job_str_list,
                               proposal, proposal_dict, eg_dict, color_list,
                               style='bar', plot_differential=True,
                               custom_color=False, cm_name='Dark2', start=0.0,
                               stop=1.0, flip_x=False, flip_y=False,
                               rotate=False,
                               normalize_yr_scale=False, year_clip=30,
                               xsize=8, ysize=6):
    '''stacked bar or area chart presenting the time spent in the various
    job levels for quartiles of a selected employee group.

inputs
    prop_ds (dataframe)
        proposal dataset to explore

    sa_ds (dataframe)
        standalone dataset

    job_levels
        the number of job levels in the model

    num_bins
        the total number of segments (divisions of the population) to calculate
        and display

    job_str_list
        a list of strings which correspond with the job levels, used for the
        chart legend
        example:
            jobs = ['Capt G4', 'Capt G3', 'Capt G2', ....]

    proposal
        text name of the (proposal) dataset, used as key in the proposal dict

    proposal_dict
        a dictionary of proposal text keys and corresponding proposal text
        descriptions, used for chart titles

    eg_dict
        dictionary used to convert employee group numbers to text, used with
        chart title text display

    color_list
        a list of color codes which control the job level color display

    style
        option to select 'area' or 'bar' to determine the type of chart output.
        default is 'bar'.

    custom_color, cm_name, start, stop
        if custom color is set to True, create a custom color map from
        the cm_name color map style.  A portion of the color map may be
        selected for customization using the start and stop inputs.

    flip_x
        'flip' the chart horizontally if True

    flip_y
        'flip' the chart vertically if True

    rotate
        transpose the chart output

    normalize_yr_scale
        set all output charts to have the same x axis range

    yr_clip
        max x axis value (years) if normalize_yr_scale set True
    '''

    if 'new_order' in prop_ds.columns:
        ds_sel_cols = prop_ds[['mnum', 'eg', 'jnum', 'empkey',
                               'new_order', 'doh', 'retdate']]
        if plot_differential:
            sa_ds['new_order'] = sa_ds['idx']
            sa_sel_cols = sa_ds[['mnum', 'eg', 'jnum', 'empkey',
                                 'new_order', 'doh', 'retdate']].copy()

    else:
        prop_ds['new_order'] = prop_ds['idx']
        ds_sel_cols = prop_ds[['mnum', 'eg', 'jnum', 'empkey',
                               'new_order', 'doh', 'retdate']].copy()
        plot_differential = False

    mnum0 = ds_sel_cols[ds_sel_cols.mnum == 0][[]]
    mnum0['order'] = np.arange(len(mnum0)) + 1

    # ds_sel_cols = ds_sel_cols[(ds_sel_cols.doh > '1989-07-01')]
    egs = sorted(list(set(ds_sel_cols.eg)))

    for eg in egs:

        ds_eg = ds_sel_cols[(ds_sel_cols.eg == eg) & (ds_sel_cols.jnum >= 1)]

        job_counts_by_emp = ds_eg.groupby(['empkey', 'jnum']).size()

        months_in_jobs = job_counts_by_emp.unstack() \
            .fillna(0).sort_index(axis=1, ascending=True).astype(int)

        months_in_jobs = months_in_jobs.join(mnum0[['order']], how='left')
        months_in_jobs.sort_values(by='order', inplace=True)
        months_in_jobs.pop('order')

        bin_lims = pd.qcut(np.arange(len(months_in_jobs)),
                           num_bins,
                           retbins=True,
                           labels=np.arange(num_bins) + 1)[1].astype(int)

        result_arr = np.zeros((num_bins, len(months_in_jobs.columns)))

        cols = list(months_in_jobs.columns)

        labels = []
        colors = []
        for col in cols:

            labels.append(job_str_list[col - 1])
            colors.append(color_list[col - 1])

        for i in np.arange(num_bins):
            bin_avg = \
                np.array(months_in_jobs[bin_lims[i]:bin_lims[i + 1]].mean())
            result_arr[i] = bin_avg

        quantile_mos = pd.DataFrame(result_arr,
                                    columns=months_in_jobs.columns,
                                    index=np.arange(1, num_bins + 1))

        quantile_yrs = quantile_mos / 12

        if plot_differential:

            sa_eg = sa_sel_cols[
                (sa_sel_cols.eg == eg) & (sa_sel_cols.jnum >= 1)]

            sa_job_counts_by_emp = sa_eg.groupby(['empkey', 'jnum']).size()

            sa_months_in_jobs = sa_job_counts_by_emp.unstack() \
                .fillna(0).sort_index(axis=1, ascending=True).astype(int)

            sa_months_in_jobs = sa_months_in_jobs.join(
                mnum0[['order']], how='left')
            sa_months_in_jobs.sort_values(by='order', inplace=True)
            sa_months_in_jobs.pop('order')

            sa_bin_lims = pd.qcut(np.arange(len(sa_months_in_jobs)),
                                  num_bins,
                                  retbins=True,
                                  labels=np.arange(num_bins) + 1)[1] \
                .astype(int)

            sa_result_arr = np.zeros(
                (num_bins, len(sa_months_in_jobs.columns)))

            for i in np.arange(num_bins):
                sa_bin_avg = \
                    np.array(sa_months_in_jobs
                             [sa_bin_lims[i]:sa_bin_lims[i + 1]].mean())
                sa_result_arr[i] = sa_bin_avg

            sa_quantile_mos = pd.DataFrame(sa_result_arr,
                                           columns=sa_months_in_jobs.columns,
                                           index=np.arange(1, num_bins + 1))

            sa_quantile_yrs = sa_quantile_mos / 12

            for col in quantile_yrs:
                if col not in sa_quantile_yrs:
                    sa_quantile_yrs[col] = 0

            sa_quantile_yrs.sort_index(axis=1, inplace=True)

            sa_labels = ['Gain', 'Loss']
            sa_colors = []
            sa_cols = list(sa_quantile_yrs.columns)
            for sa_col in sa_cols:
                sa_labels.append(job_str_list[sa_col - 1])
                sa_colors.append(color_list[sa_col - 1])

        # dict color mapping to job level is currently lost...
        if custom_color:
            num_of_colors = cf.num_of_job_levels + 1
            cm_subsection = np.linspace(start, stop, num_of_colors)
            colormap = eval('cm.' + cm_name)
            colors = [colormap(x) for x in cm_subsection]

        with sns.axes_style('darkgrid'):

            if style == 'area':
                quantile_yrs.plot(kind='area',
                                  stacked=True, color=colors)
            if style == 'bar':
                if rotate:
                    kind = 'barh'
                else:
                    kind = 'bar'
                quantile_yrs.plot(kind=kind, width=1,
                                  stacked=True, color=colors)

            ax = plt.gca()

            if normalize_yr_scale:
                if rotate:
                    plt.xlim(0, year_clip)
                else:
                    plt.ylim(ymax=year_clip)

            if style == 'bar':

                if not flip_y:
                    ax.invert_yaxis()

                if rotate:
                    plt.xlabel('years')
                    plt.ylabel('quartiles')
                else:
                    plt.ylabel('years')
                    plt.xlabel('quartiles')
                    plt.xticks(rotation='horizontal')

            if flip_x:
                ax.invert_xaxis()

            plt.suptitle(proposal_dict[proposal] + ', GROUP ' + eg_dict[eg],
                         fontsize=20, y=1.02)
            plt.title('years in position, ' + str(num_bins) + '-quantiles',
                      y=1.02)

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend((labels), loc='center left', bbox_to_anchor=(1, 0.5))

            fig = plt.gcf()
            fig.set_size_inches(xsize, ysize)
            plt.show()

            if plot_differential and style == 'bar':

                diff = quantile_yrs - sa_quantile_yrs

                if style == 'area':
                    diff.plot(kind='area',
                              stacked=True, color=colors)
                if style == 'bar':
                    if rotate:
                        kind = 'barh'
                    else:
                        kind = 'bar'
                    diff.plot(kind=kind, width=1,
                              stacked=True, color=colors)

                ax = plt.gca()

                if rotate:
                    plt.xlabel('years')
                    plt.ylabel('quartiles')
                    if normalize_yr_scale:
                        plt.xlim(year_clip / -3, year_clip / 3)
                    if not flip_y:
                        ax.invert_yaxis()
                    x_min, x_max = plt.xlim()
                    plt.axvspan(0, x_max, facecolor='g', alpha=0.15)
                    plt.axvspan(0, x_min, facecolor='r', alpha=0.15)
                else:
                    plt.ylabel('years')
                    plt.xlabel('quartiles')
                    if normalize_yr_scale:
                        plt.ylim(year_clip / -3, year_clip / 3)
                    if flip_y:
                        ax.invert_yaxis()
                    ymin, ymax = plt.ylim()
                    plt.axhspan(0, ymax, facecolor='g', alpha=0.15)
                    plt.axhspan(0, ymin, facecolor='r', alpha=0.15)
                    ax.invert_xaxis()
                    plt.xticks(rotation='horizontal')

                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(
                    (sa_labels), loc='center left', bbox_to_anchor=(1, 0.5))

                plt.suptitle(proposal_dict[proposal] +
                             ', GROUP ' + eg_dict[eg],
                             fontsize=20, y=1.02)
                plt.title('years differential vs standalone, ' +
                          str(num_bins) + '-quantiles',
                          y=1.02)

                fig = plt.gcf()
                fig.set_size_inches(xsize, ysize)
                plt.show()


def age_vs_spcnt(df, eg_list, mnum, color_list,
                 eg_dict, proposal, proposal_dict,
                 formatter):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    d_age_pcnt = df[df.mnum == mnum][['age', 'mnum', 'spcnt', 'eg']].copy()

    for grp in eg_list:
        d_for_plot = d_age_pcnt[d_age_pcnt.eg == grp]
        x = d_for_plot['age']
        y = d_for_plot['spcnt']
        ax.scatter(x, y, c=color_list[grp - 1],
                   s=20, linewidth=0.1, edgecolors='w',
                   label=eg_dict[grp])

    plt.ylim(1, 0)
    plt.xlim(25, 65)
    plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.title(proposal_dict[proposal] +
              ' - age vs seniority percentage' +
              ', month ' +
              str(mnum), y=1.02)
    plt.legend(loc=2)
    plt.show()


def multiline_plot_by_emp(df, measure, xax, emp_list, job_levels,
                          color_list, job_str_list, proposal,
                          proposal_dict, formatter):

    frame = df.copy()
    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:
        frame = frame[frame.jnum <= cf.num_of_job_levels]
    if measure in ['mpay']:
        frame = frame[frame.age < 65]

    i = 0

    for emp in emp_list:
        try:
            if len(emp_list) == 3:
                frame[frame.empkey == emp].set_index(xax)[measure] \
                    .plot(color=color_list[i], label=emp)
            else:
                frame[frame.empkey == emp].set_index(xax)[measure] \
                    .plot(label=emp)
            i += 1
        except:
            continue

    fig = plt.gca()

    if measure in ['snum', 'spcnt', 'lspcnt', 'jnum', 'jobp', 'fbff']:
        fig.invert_yaxis()
    if measure in ['lspcnt', 'spcnt']:
        plt.gca().yaxis.set_major_formatter(formatter)
    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:

        plt.yticks(np.arange(0, job_levels + 2, 1))
        plt.ylim(job_levels + 1.5, 0.5)

        yticks = fig.get_yticks().tolist()

        for i in np.arange(1, len(yticks)):
            yticks[i] = job_str_list[i - 1]
        plt.axhspan(job_levels + 1, job_levels + 2, facecolor='.8', alpha=0.9)
        fig.set_yticklabels(yticks)
        plt.axhline(y=job_levels + 1, c='.8', ls='-', alpha=.8, lw=3)

    if xax in ['lspcnt', 'spcnt']:
        plt.xlim(1, 0)

    plt.title(measure + ' - ' + proposal_dict[proposal], y=1.02)
    plt.legend(loc=4)
    plt.show()


def multiline_plot_by_eg(df, measure, xax, eg_list, job_dict,
                         proposal, proposal_dict,
                         job_levels, colors, formatter, mnum=0,
                         scatter=False, exclude_fur=False,
                         full_pcnt_xscale=False):

    frame = df[(df.mnum == mnum)]

    if exclude_fur:
        frame = frame[(frame.jnum >= 1) & (frame.jnum <= job_levels)]

    if measure == 'mpay':
        frame = frame[frame.age < 65]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in eg_list:

        frame_for_plot = frame[frame.eg == i]
        x = frame_for_plot[xax]
        y = frame_for_plot[measure]
        if scatter:
            ax.scatter(x=x, y=y, color=colors[i - 1], label=i, alpha=.5)
        else:
            frame_for_plot.set_index(xax)[measure].plot(label=i,
                                                        color=colors[i - 1],
                                                        alpha=.6)

    fig = plt.gca()

    if measure in ['snum', 'spcnt', 'jnum', 'jobp', 'fbff',
                   'lspcnt', 'rank_in_job']:
        fig.invert_yaxis()
    if measure in ['spcnt', 'lspcnt']:
        fig.yaxis.set_major_formatter(formatter)
    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:

        plt.yticks(np.arange(0, job_levels + 2, 1))
        plt.ylim(job_levels + 2, 0.5)

        yticks = fig.get_yticks().tolist()

        for i in np.arange(1, len(yticks)):
            yticks[i] = job_dict[i - 1]
        plt.axhspan(job_levels + 1, job_levels + 2, facecolor='.8', alpha=0.3)
        fig.set_yticklabels(yticks)
        plt.axhline(y=job_levels + 1, c='.8', ls='-', alpha=.3, lw=3)

    if measure in ['mpay', 'cpay', 'mlong', 'ylong']:
        plt.ylim(ymin=0)

    if xax in ['new_order', 'cat_order', 'lnum', 'snum']:
        plt.xlim(xmax=0)
        plt.xlabel(xax)

    if xax in ['mlong', 'ylong']:
        plt.xlim(xmin=0)
        plt.xlabel(xax)

    if xax not in ['age', 'mlong', 'ylong']:
        fig.invert_xaxis()

    if xax in ['spcnt', 'lspcnt'] and full_pcnt_xscale:
        plt.xlim(1, 0)

    plt.legend(loc=4)

    plt.title(measure.upper() +
              ' ordered by ' + xax + ' - ' +
              proposal_dict[proposal] + ' - Month: ' + str(mnum), y=1.02)

    plt.show()


def violinplot_by_eg(df, measure, proposal, proposal_dict,
                     mnum=0, bw=.1, linewidth=1.5, scale='count'):

    if measure == 'age':
        frame = df[df.mnum == mnum][['eg', measure]].copy()
    else:
        frame = df[df.mnum == mnum][[measure, 'eg', 'age']].copy()
    frame.reset_index(drop=True, inplace=True)

    if measure == 'mpay':
        frame = frame[frame.age < 65]

    sns.violinplot(frame.eg, frame[measure],
                   cut=0, scale=scale, inner='box',
                   bw=.1, linewidth=linewidth,
                   palette=['gray', '#3399ff', '#ff8000'])

    plt.title(proposal_dict[proposal] + ' - ' +
              measure.upper() + ' Distribution - Month ' + str(mnum), y=1.02)
    fig = plt.gca()
    if measure == 'age':
        plt.ylim(25, 70)
    if measure in ['snum', 'spcnt', 'jnum', 'jobp']:
        fig.invert_yaxis()
    plt.show()


def age_kde_dist(df, mnum=0, eg=None):

    frame = df[df.mnum == mnum]

    if not eg:
        eg = [1, 2, 3]

    eg_dict = {1: 'amer',
               2: 'east',
               3: 'west'}

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(sharex=True, figsize=(10, 8))

    for x in eg:
        try:
            if x == 1:
                color = 'k'
            elif x == 2:
                color = 'b'
            elif x == 3:
                color = '#e65c00'
            else:
                color = 'g'

            sns.kdeplot(frame[frame.eg == x].age,
                        shade=True, color=color,
                        bw=.8, ax=ax, label=eg_dict[x])
        except:
            continue

    ax.set_xlim(25, 65)
    fig.set_size_inches(10, 8)
    plt.title('Age Distribution Comparison - Month ' + str(mnum), y=1.02)
    plt.show()


def eg_diff_boxplot(df_list, formatter, measure='spcnt',
                    comparison='standalone', year_clip=2035,
                    xsize=18, ysize=10):
    '''
    df_list
        currently hard-coded for a dataframe list of amer, east, west, and
        standalone dataframes
    formatter
        matplotlib percentage y scale formatter
    measure
        differential data to compare
    comparison
        either 'standalone' or 'p2p' (proposal to proposal)
    year_clip
        only present results through this year
    xsize, ysize
        plot size in inches'''

    yval_dict = {'s_a': 'AASIC PROPOSAL vs. standalone ' + measure.upper(),
                 's_e': 'EAST PROPOSAL vs. standalone ' + measure.upper(),
                 's_w': 'WEST PROPOSAL vs. standalone ' + measure.upper(),
                 'a_e': 'EAST vs. AASIC ' + measure.upper(),
                 'a_w': 'WEST vs. AASIC ' + measure.upper(),
                 'e_a': 'AASIC vs. EAST ' + measure.upper(),
                 'e_w': 'WEST vs. EAST ' + measure.upper(),
                 'w_a': 'AASIC vs. WEST ' + measure.upper(),
                 'w_e': 'EAST vs. WEST ' + measure.upper(),
                 }

    chart_pad = {'jnum': .3,
                 'jobp': .3,
                 'spcnt': .03,
                 'lspcnt': .03,
                 'mpay': 1,
                 'cpay': 10,
                 }

    suffix_list = ['_a', '_e', '_w', '_s']

    if comparison == 'standalone':
        yval_list = ['s_a', 's_e', 's_w']
    elif comparison == 'p2p':
        yval_list = ['e_a', 'w_a', 'a_e', 'w_e', 'a_w', 'e_w']

    colors = ['grey', '#66b3ff', '#ff884d', '#00ff99']

    ds1_ = df_list[0][['empkey', 'mnum', measure]].copy()
    ds2_ = df_list[1][['empkey', 'mnum', measure]].copy()
    ds3_ = df_list[2][['empkey', 'mnum', measure]].copy()
    ds4_ = df_list[3][['empkey', 'mnum', 'date', 'eg', measure]].copy()

    i = 0
    for df in [ds1_, ds2_, ds3_, ds4_]:

        df.rename(columns={measure: measure + suffix_list[i]}, inplace=True)
        df['key'] = (df.empkey * 1000) + df.mnum
        df.drop(['mnum', 'empkey'], inplace=True, axis=1)
        df.set_index('key', inplace=True)
        i += 1

    compare = ds4_.join(ds1_).join(ds2_).join(ds3_)

    if measure in ['mpay', 'cpay']:

        if comparison == 'standalone':
            compare['s_a'] = compare[measure + '_a'] - compare[measure + '_s']
            compare['s_e'] = compare[measure + '_e'] - compare[measure + '_s']
            compare['s_w'] = compare[measure + '_w'] - compare[measure + '_s']

        if comparison == 'p2p':
            # compare['a_s'] = compare[measure + '_s']
            # - compare[measure + '_a']
            compare['a_e'] = compare[measure + '_e'] - compare[measure + '_a']
            compare['a_w'] = compare[measure + '_w'] - compare[measure + '_a']

            # compare['e_s'] = compare[measure + '_s']
            # - compare[measure + '_e']
            compare['e_a'] = compare[measure + '_a'] - compare[measure + '_e']
            compare['e_w'] = compare[measure + '_w'] - compare[measure + '_e']

            # compare['w_s'] = compare[measure + '_s']
            # - compare[measure + '_w']
            compare['w_a'] = compare[measure + '_a'] - compare[measure + '_w']
            compare['w_e'] = compare[measure + '_e'] - compare[measure + '_w']

    else:

        if comparison == 'standalone':
            compare['s_a'] = compare[measure + '_s'] - compare[measure + '_a']
            compare['s_e'] = compare[measure + '_s'] - compare[measure + '_e']
            compare['s_w'] = compare[measure + '_s'] - compare[measure + '_w']

        if comparison == 'p2p':
            # compare['a_s'] = compare[measure + '_a']
            # - compare[measure + '_s']
            compare['a_e'] = compare[measure + '_a'] - compare[measure + '_e']
            compare['a_w'] = compare[measure + '_a'] - compare[measure + '_w']

            # compare['e_s'] = compare[measure + '_e']
            # - compare[measure + '_s']
            compare['e_a'] = compare[measure + '_e'] - compare[measure + '_a']
            compare['e_w'] = compare[measure + '_e'] - compare[measure + '_w']

            # compare['w_s'] = compare[measure + '_w']
            # - compare[measure + '_s']
            compare['w_a'] = compare[measure + '_w'] - compare[measure + '_a']
            compare['w_e'] = compare[measure + '_w'] - compare[measure + '_e']

    compare.drop([measure + '_s',
                  measure + '_a',
                  measure + '_e',
                  measure + '_w'],
                 inplace=True, axis=1)

    compare.set_index('date', drop=True, inplace=True)
    compare['date'] = compare.index.year
    y_clip = compare[compare.date <= year_clip]

    for v in yval_list:
        pad = chart_pad[measure]
        yval = v
        ylimit = max(abs(max(y_clip[yval])), abs(min(y_clip[yval]))) + pad
        sns.boxplot(x='date', y=yval,
                    hue='eg', data=y_clip,
                    palette=colors, width=.8,
                    linewidth=1.0, fliersize=1.5)
        fig = plt.gcf()
        fig.set_size_inches(xsize, ysize)
        plt.axhline(y=0, c='r', zorder=.5, alpha=.3, lw=3)
        plt.ylim(-ylimit, ylimit)
        if measure in ['spcnt', 'lspcnt']:
            plt.gca().yaxis.set_major_formatter(formatter)
        plt.title(yval_dict[yval], y=1.02)
        plt.show()


# DISTRIBUTION WITHIN JOB LEVEL (NBNF effect)
def stripplot_distribution_in_category(df, job_levels, mnum, blk_pcnt,
                                       eg_colors, band_colors, bg_alpha=.12,
                                       adjust_y_axis=False):

    fur_lvl = job_levels + 1
    if job_levels == 16:
        adjust = [0, 0, 0, 0, 0, 0, -50, 50, 0, 0, 0, 0, 0, -140, 0, 120, 0]
        jobs_dict = {1: 'Capt G4 B', 2: 'Capt G4 R', 3: 'Capt G3 B',
                     4: 'Capt G2 B', 5: 'Capt G3 R', 6: 'Capt G2 R',
                     7: 'F/O  G4 B', 8: 'F/O  G4 R', 9: 'F/O  G3 B',
                     10: 'F/O  G2 B', 11: 'Capt G1 B', 12: 'F/O  G3 R',
                     13: 'F/O  G2 R', 14: 'Capt G1 R', 15: 'F/O  G1 B',
                     16: 'F/O  G1 R', 17: 'FUR'}

    if job_levels == 8:
        adjust = [0, 0, 0, 0, 0, 0, -50, 50, 0]
        jobs_dict = {1: 'Capt G4', 2: 'Capt G3', 3: 'Capt G2', 4: 'F/O  G4',
                     5: 'F/O  G3', 6: 'F/O  G2', 7: 'Capt G1', 8: 'F/O  G1',
                     9: 'FUR'}

    eg_dict = {1: 'AMER', 2: 'EAST', 3: 'WEST', 4: 'Standalone'}

    df = df[['mnum', 'cat_order', 'jnum', 'eg']].copy()
    data = df[df.mnum == mnum]

    adjust_y_axis = False
    if adjust_y_axis:
        y_count = len(data)
    else:
        y_count = len(data[data.mnum == mnum])

    cum_job_counts = data.jnum.value_counts().sort_index().cumsum()
    lowest_cat = max(cum_job_counts.index)

    cnts = list(cum_job_counts)
    cnts.insert(0, 0)

    axis2_lbl_locs = []
    axis2_lbls = []

    with sns.axes_style('white'):
        fig, ax1 = plt.subplots()
        ax1 = sns.stripplot(y='cat_order', x='eg', data=data, jitter=.5,
                            order=np.arange(1, 4),
                            palette=eg_colors, size=3,
                            linewidth=0, split=True)

        plt.yticks(np.arange(0, 15000, 1000))
        plt.ylim(y_count, 0)

    i = 0
    for job_zone in cum_job_counts:
        ax1.axhline(job_zone, c='magenta', ls='-', alpha=1, lw=.8)
        ax1.axhspan(cnts[i], cnts[i + 1], facecolor=band_colors[i],
                    alpha=bg_alpha,)
        rsv_lvl = (round((cnts[i + 1] - cnts[i]) * blk_pcnt)) + cnts[i]
        ax1.axhline(rsv_lvl, c='#66ff99', ls='--', alpha=1, lw=1)
        i += 1

    i = 0
    for job_num in cum_job_counts.index:
        axis2_lbl_locs.append(round((cnts[i] + cnts[i + 1]) / 2))
        axis2_lbl_locs[i] += adjust[i]
        axis2_lbls.append(jobs_dict[job_num])
        i += 1

    if lowest_cat == fur_lvl:
        plt.axhspan(cnts[-2], cnts[-1], facecolor='#fbfbea', alpha=0.9)
        axis2_lbls[-1] = 'FUR'

    with sns.axes_style("white"):
        ax2 = ax1.twinx()
        ax2.set_yticks(axis2_lbl_locs)
        yticks = ax2.get_yticks().tolist()

        for i in np.arange(len(yticks)):
            yticks[i] = axis2_lbls[i]

        ax2.set_yticklabels(yticks)
        plt.ylim(y_count, 0)

    xticks = plt.gca().get_xticks().tolist()

    tick_dummies = []
    for tck in xticks:
        tick_dummies.append(eg_dict[tck + 1])

    plt.gca().set_xticklabels(tick_dummies)

    plt.gcf().set_size_inches(7, 12)
    plt.title(
        'Group distribution within job levels, month ' + str(mnum), y=1.04)
    plt.show()


def job_level_progression(ds, emp_list, through_date, job_levels,
                          eg_colors, band_colors,
                          job_counts, job_change_lists, alpha=.12):

    through_date = pd.to_datetime(through_date)
    fur_lvl = job_levels + 1
    if job_levels == 16:
        j_changes = f.convert_job_changes_to16(job_change_lists, cf.jd)
        eg_counts = f.convert_jcnts_to16(job_counts,
                                         cf.intl_blk_pcnt,
                                         cf.dom_blk_pcnt)
        # for adjusting secondary y label positioning
        adjust = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -75, 50, 0, -160, -40, 120, 0]
        jobs_dict = {1: 'Capt G4 B', 2: 'Capt G4 R', 3: 'Capt G3 B',
                     4: 'Capt G2 B', 5: 'Capt G3 R', 6: 'Capt G2 R',
                     7: 'F/O  G4 B', 8: 'F/O  G4 R', 9: 'F/O  G3 B',
                     10: 'F/O  G2 B', 11: 'Capt G1 B', 12: 'F/O  G3 R',
                     13: 'F/O  G2 R', 14: 'Capt G1 R', 15: 'F/O  G1 B',
                     16: 'F/O  G1 R', 17: 'FUR'}

    else:
        j_changes = job_change_lists
        eg_counts = job_counts
        adjust = [0, 0, 0, 0, 0, 0, -50, 50, 0]
        jobs_dict = {1: 'Capt G4', 2: 'Capt G3', 3: 'Capt G2', 4: 'F/O  G4',
                     5: 'F/O  G3', 6: 'F/O  G2', 7: 'Capt G1', 8: 'F/O  G1',
                     9: 'FUR'}

    jcnts_arr = f.make_jcnts(eg_counts)
    table = f.job_gain_loss_table(np.unique(ds.mnum).size,
                                  job_levels,
                                  jcnts_arr,
                                  j_changes,
                                  standalone=False)

    df_table = pd.DataFrame(table[0],
                            columns=np.arange(1, job_levels + 1),
                            index=pd.date_range(cf.starting_date,
                                                periods=table[0].shape[0],
                                                freq='M'))
    # for band areas
    jobs_table = df_table[:through_date]
    # for headcount:
    df_monthly_non_ret = pd.DataFrame(ds[ds.fur == 0].groupby('mnum').size(),
                                      columns=['count'])
    df_monthly_non_ret.set_index(
        pd.date_range(cf.starting_date,
                      periods=np.unique(df_monthly_non_ret.index).size,
                      freq='M'), inplace=True)
    # df_monthly_non_ret = pd.DataFrame(non_ret_counts)
    # df_monthly_non_ret.set_index(pd.date_range('2013-12-31',
    #                                            periods=len(non_ret_counts),
    #                                            freq='M'),
    #                              inplace=True)

    non_ret_count = df_monthly_non_ret[:through_date]
    last_month_jobs_series = jobs_table.loc[through_date].sort_index()

    last_month_counts = pd.DataFrame(last_month_jobs_series,
                                     index=last_month_jobs_series.index
                                     ).sort_index()

    last_month_counts.rename(columns={last_month_counts.columns[0]: 'counts'},
                             inplace=True)
    last_month_counts['cum_counts'] = last_month_counts['counts'].cumsum()

    lowest_cat = max(last_month_counts.index)

    cnts = list(last_month_counts['cum_counts'])
    cnts.insert(0, 0)
    axis2_lbl_locs = []
    axis2_lbls = []

    i = 0
    for job_num in last_month_counts.index:
        axis2_lbl_locs.append(round((cnts[i] + cnts[i + 1]) / 2))
        axis2_lbl_locs[i] += adjust[i]
        axis2_lbls.append(jobs_dict[job_num])
        i += 1

    with sns.axes_style("white"):
        i = 0
        for emp in emp_list:
            ax1 = ds[ds.empkey == emp].set_index('date')[:through_date] \
                .cat_order.plot(lw=3, color=eg_colors[i], label=emp)
            i += 1
        non_ret_count['count'].plot(c='k', ls='--',
                                    label='active count', ax=ax1)
        plt.gca().legend()

    with sns.axes_style("white"):
        ax2 = jobs_table.plot.area(stacked=True,
                                   figsize=(12, 10),
                                   sort_columns=True,
                                   linewidth=2,
                                   color=band_colors,
                                   alpha=alpha,
                                   legend=False,
                                   ax=ax1)

        plt.axvline(cf.imp_date, c='g', ls='--', alpha=1, lw=1)

    plt.gca().invert_yaxis()
    plt.ylim(max(df_monthly_non_ret['count']), 0)

    plt.title('cumulative job level bands, full active count', y=1.02)

    if lowest_cat == fur_lvl:
        plt.axhspan(cnts[-2], cnts[-1], facecolor='#fbfbea', alpha=0.9)
        axis2_lbls[-1] = 'FUR'

    with sns.axes_style("white"):
        ax2 = ax1.twinx()
        ax2.set_yticks(axis2_lbl_locs)
        yticks = ax2.get_yticks().tolist()

        for i in np.arange(len(yticks)):
            yticks[i] = axis2_lbls[i]

        ax2.set_yticklabels(yticks)

        plt.ylim(max(df_monthly_non_ret['count']), 0)

    ax1.xaxis.grid(True)
    ax1.set_axisbelow(True)

    plt.show()


def differential_scatter(sa_ds, proposal_ds_list,
                         measure, filter_measure,
                         filter_val, formatter, prop_order=True,
                         show_scatter=True, show_lin_reg=True,
                         show_mean=True, mean_len=50, eg_list=[1, 2, 3],
                         dot_size=15, lin_reg_order=15, ylimit=False, ylim=5,
                         width=22, height=14, bright_bg=False,
                         chart_style='whitegrid'):

    cols = [measure, 'new_order']

    # try:
    #     p_egs_and_order = prop_ds[prop_ds[filter_measure] == filter_val][
    #         ['eg', 'new_order']]
    # except:
    #     p_egs_and_order = prop_ds[prop_ds[filter_measure] == filter_val][
    #         ['eg', 'idx']]

    df = sa_ds[sa_ds[filter_measure] == filter_val][[measure, 'eg']].copy()
    df.rename(columns={measure: measure + '_s'}, inplace=True)

    yval_list = ['avs', 'evs', 'wvs']

    yval_dict = {'avs': 'AASIC',
                 'evs': 'EAST',
                 'wvs': 'WEST'}

    order_dict = {'avs': 'order1',
                  'evs': 'order2',
                  'wvs': 'order3'}

    i = 1
    for ds in proposal_ds_list:
        ds = ds[ds[filter_measure] == filter_val][cols].copy()
        ds.rename(columns={measure: measure + '_' + str(i),
                           'new_order': 'order' + str(i)}, inplace=True)
        df = df.join(ds)
        i += 1

    df.sort_values(by='order1', inplace=True)
    df['eg_sep_order'] = df.groupby('eg').cumcount() + 1
    eg_sep_order = np.array(df.eg_sep_order)
    eg_arr = np.array(df.eg)
    eg_denom_dict = df.groupby('eg').eg_sep_order.max().to_dict()
    unique_egs = np.unique(df.eg)
    denoms = np.zeros(eg_arr.size)

    for eg in unique_egs:
        np.put(denoms, np.where(eg_arr == eg)[0], eg_denom_dict[eg])

    df['sep_eg_pcnt'] = eg_sep_order / denoms

    if measure in ['spcnt', 'lspcnt', 'snum', 'lnum', 'cat_order',
                   'jobp', 'jnum']:

        df['avs'] = df[measure + '_s'] - df[measure + '_1']
        df['evs'] = df[measure + '_s'] - df[measure + '_2']
        df['wvs'] = df[measure + '_s'] - df[measure + '_3']
    else:
        df['avs'] = df[measure + '_1'] - df[measure + '_s']
        df['evs'] = df[measure + '_2'] - df[measure + '_s']
        df['wvs'] = df[measure + '_3'] - df[measure + '_s']

    with sns.axes_style(chart_style):
        for yval in yval_list:
            df.sort_values(by=order_dict[yval], inplace=True)

            if prop_order:
                xval = order_dict[yval]

            else:
                xval = 'sep_eg_pcnt'

            fig, ax = plt.subplots()

            for eg in eg_list:
                data = df[df.eg == eg].copy()
                x_limit = max(data[xval]) + 100

                if show_scatter:
                    data.plot(x=xval, y=yval, kind='scatter', linewidth=0.1,
                              color=cf.eg_colors[eg - 1], s=dot_size,
                              label=cf.eg_dict[eg],
                              ax=ax)

                if show_mean:
                    data['ma'] = data[yval].rolling(mean_len).mean()
                    data.plot(x=xval, y='ma', lw=5,
                              color=cf.mean_colors[eg - 1],
                              label=cf.eg_dict[eg],
                              alpha=.6, ax=ax)
                    plt.xlim(0, x_limit)

                if show_lin_reg:
                    if show_scatter:
                        lr_colors = cf.lr_colors
                    else:
                        lr_colors = cf.lr_colors2
                    sns.regplot(x=xval, y=yval, data=data,
                                color=lr_colors[eg - 1],
                                label=cf.eg_dict[eg],
                                scatter=False, truncate=True, ci=50,
                                order=lin_reg_order,
                                line_kws={'lw': 20,
                                          'alpha': .4},
                                ax=ax)
                    plt.xlim(0, x_limit)

            if measure == 'jobp':
                ymin = math.floor(min(df[yval]))
                ymax = math.ceil(max(df[yval]))
                scale_lim = max(abs(ymin), ymax)
                plt.yticks(np.arange(-scale_lim, scale_lim + 1, 1))
                if ylimit:
                    plt.ylim(-5, 5)
                else:
                    plt.ylim(-scale_lim, scale_lim)

            plt.gcf().set_size_inches(width, height)
            plt.title(yval_dict[yval] + ' differential: ' + measure)
            plt.xlim(xmin=0)

            if measure in ['spcnt', 'lspcnt']:
                plt.gca().yaxis.set_major_formatter(formatter)

            if xval == 'sep_eg_pcnt':
                plt.xlim(xmax=1)

            ax.axhline(0, c='m', ls='-', alpha=1, lw=2)
            ax.invert_xaxis()
            if bright_bg:
                ax.set_axis_bgcolor('#faf6eb')
            plt.show()


# groupby method:
# only age 65 in group, use .sum
# others, use .mean
def job_grouping_over_time(proposal, prop_text, eg_list, jobs, colors,
                           formatter, plt_kind='bar', rets_only=True,
                           time_group='A', display_yrs=40, legend_loc=4,
                           xsize=12, ysize=10):

    if rets_only:
        df_sub = proposal[proposal.age == 65][['eg', 'date', 'jnum']]
    else:
        df_sub = proposal[proposal.age > 55][['eg', 'date', 'jnum']]

    for eg in eg_list:
        with sns.axes_style("darkgrid"):
            denom = len(proposal[(proposal.mnum == 0) & (proposal.eg == eg)])
            df_eg = df_sub[df_sub.eg == eg]
            if rets_only:

                grpby = df_eg.groupby(['date', 'jnum']) \
                    .size().unstack().fillna(0).astype(int)
                df = grpby.resample(time_group).sum()
                if time_group == 'A':
                    df = round(df / denom, 3)
                if time_group == 'Q':
                    df = round(df / .25 * denom, 3)
                ylbl = 'percent of sep list'

            else:

                grpby = df_eg.groupby(['date', 'jnum']) \
                    .size().unstack().fillna(0).astype(int)
                df = grpby.resample(time_group).mean()
                ylbl = 'count'

            df['year'] = df.index.year

            df.set_index('year', drop=True, inplace=True)

            cols = df.columns
            labels = []
            clr = []
            for col in cols:
                labels.append(jobs[col - 1])
                clr.append(colors[col - 1])

            if plt_kind == 'area':
                df.plot(kind=plt_kind, linewidth=0, color=clr, stacked=True)

            if plt_kind == 'bar':
                df.plot(kind=plt_kind, width=1, color=clr, stacked=True)

            if rets_only:
                plt.gca().set_yticks(np.arange(.08, 0, -.01))
                plt.gca().yaxis.set_major_formatter(formatter)
            plt.gca().invert_yaxis()
            plt.xlim(0, display_yrs)
            plt.legend((labels), loc=legend_loc)
            plt.ylabel(ylbl)
            plt.title(cf.proposal_dict[prop_text] + ' group ' +
                      cf.eg_dict[eg], y=1.01)
            plt.gcf().set_size_inches(xsize, ysize)
            plt.show()


def parallel(dsa, dsb, dsc, dsd, eg_list, measure, month_list, job_levels,
             formatter, left='sa', a_stride=50, e_stride=30, w_stride=20,
             xsize=6, ysize=8):

    group_dict = {1: 'AMER', 2: 'EAST', 3: 'WEST'}
    color_dict = {1: 'black', 2: 'blue', 3: 'red'}

    if job_levels == 16:
        jobs = ['Capt G4 B', 'Capt G4 R', 'Capt G3 B', 'Capt G2 B',
                'Capt G3 R', 'Capt G2 R', 'F/O  G4 B', 'F/O  G4 R',
                'F/O  G3 B', 'F/O  G2 B', 'Capt G1 B', 'F/O  G3 R',
                'F/O  G2 R', 'Capt G1 R', 'F/O  G1 B', 'F/O  G1 R', 'FUR']

    if job_levels == 8:
        jobs = ['Capt G4', 'Capt G3', 'Capt G2', 'F/O  G4', 'F/O  G3',
                'F/O  G2', 'Capt G1', 'F/O  G1', 'FUR']

    num_egplots = len(eg_list)
    num_months = len(month_list)

    sns.set_style('whitegrid',
                  {'axes.facecolor': '#f5f5dc',  # f5f5dc
                   'axes.axisbelow': True,
                   'axes.edgecolor': '.2',
                   'axes.linewidth': 1.0,
                   'grid.color': '.7',
                   'grid.linestyle': u'--'})

    fig, ax = plt.subplots(num_months, num_egplots)

    fig = plt.gcf()
    fig.set_size_inches(xsize * num_egplots, ysize * num_months)

    plot_num = 0

    for month in month_list:

        ds1 = dsa[(dsa.mnum == month) & (dsa.fur == 0)][[measure]].copy()
        ds2 = dsb[(dsb.mnum == month) & (dsb.fur == 0)][[measure]].copy()
        ds3 = dsc[(dsc.mnum == month) & (dsc.fur == 0)][[measure]].copy()
        ds4 = dsd[(dsd.mnum == month) & (dsd.fur == 0)][['eg', measure]].copy()

        if left == 'sa':
            df_joined = ds4.join(ds2, rsuffix=('_E')) \
                .join(ds3, rsuffix=('_W')) \
                .join(ds1, rsuffix=('_A'))
            df_joined.columns = ['eg', 'StandAlone', 'EAST', 'WEST', 'AMER']

        if left == 'amer':
            df_joined = ds1.join(ds4, lsuffix=('_A')) \
                .join(ds2, rsuffix=('_E')) \
                .join(ds3, rsuffix=('_W'))
            df_joined.columns = ['AMER', 'eg', 'StandAlone', 'EAST', 'WEST']

        if left == 'east':
            df_joined = ds2.join(ds4, lsuffix=('_E')) \
                .join(ds1, rsuffix=('_A')) \
                .join(ds3, rsuffix=('_W'))
            df_joined.columns = ['EAST', 'eg', 'StandAlone', 'AMER', 'WEST']

        if left == 'west':
            df_joined = ds3.join(ds4, lsuffix=('_W')) \
                .join(ds1, rsuffix=('_A')) \
                .join(ds2, rsuffix=('_E'))
            df_joined.columns = ['WEST', 'eg', 'StandAlone', 'AMER', 'EAST']

        if 1 in eg_list:
            plot_num += 1
            plt.subplot(num_months, num_egplots, plot_num)
            df_1 = df_joined[df_joined.eg == 1]
            df_1 = df_1[::a_stride]
            parallel_coordinates(df_1, 'eg', lw=1.5, alpha=.7,
                                 color=color_dict[1])
            plt.title(group_dict[1].upper() + ' ' + measure.upper() + ' ' +
                      str(month) + ' mths', fontsize=16, y=1.02)

        if 2 in eg_list:
            plot_num += 1
            plt.subplot(num_months, num_egplots, plot_num)
            df_2 = df_joined[df_joined.eg == 2]
            df_2 = df_2[::e_stride]
            parallel_coordinates(df_2, 'eg', lw=1.5, alpha=.7,
                                 color=color_dict[2])
            plt.title(group_dict[2].upper() + ' ' + measure.upper() + ' ' +
                      str(month) + ' mths', fontsize=16, y=1.02)

        if 3 in eg_list:
            plot_num += 1
            plt.subplot(num_months, num_egplots, plot_num)
            df_3 = df_joined[df_joined.eg == 3]
            df_3 = df_3[::w_stride]
            parallel_coordinates(df_3, 'eg', lw=1.5, alpha=.7,
                                 color=color_dict[3])
            plt.title(group_dict[3].upper() + ' ' + measure.upper() + ' ' +
                      str(month) + ' mths', fontsize=16, y=1.02)

    fig = plt.gcf()
    for ax in fig.axes:

        if measure in ['spcnt', 'lspcnt']:
            ax.set_yticks(np.arange(1, 0, -.05))
            ax.invert_yaxis()
            ax.yaxis.set_major_formatter(formatter)

        if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:

            ax.set_yticks(np.arange(0, job_levels + 2, 1))
            ax.set_ylim(job_levels + .5, 0.5)
            yticks = ax.get_yticks().tolist()

            for i in np.arange(1, len(yticks)):
                yticks[i] = jobs[i - 1]

            ax.set_yticklabels(yticks, fontsize=12)

        if measure in ['snum', 'lnum']:
            ax.invert_yaxis()
        ax.grid()
        ax.legend_.remove()

    plt.tight_layout()
    plt.show()


def rows_of_color(prop_text, prop, mnum, measure_list, cmap_colors, cols=200,
                  job_only=False, jnum=1, cell_border=True, border_color='k',
                  jnum_colors=cf.west_color,
                  xsize=14, ysize=12):

    data = prop[prop.mnum == mnum]
    rows = int(len(prop[prop.mnum == 0]) / cols) + 1
    heat_data = np.zeros(cols * rows)

    i = 1

    if 'jnum' in measure_list:
        i = 0
        cmap_colors = jnum_colors

    if job_only:

        eg = np.array(data.eg)
        jnums = np.array(data.jnum)

        for eg_num in np.unique(eg):
            np.put(heat_data, np.where(eg == eg_num)[0], eg_num)
        np.put(heat_data, np.where(jnums != jnum)[0], 0)
        title = cf.proposal_dict[prop_text] + ' month ' + str(mnum) + \
            ':  ' + cf.job_strs[jnum - 1] + '  job distribution'

    else:

        for measure in measure_list:

            if measure in ['eg', 'jnum']:

                measure = np.array(data[measure])

                for val in np.unique(measure):
                    np.put(heat_data, np.where(measure == val)[0], i)
                    i += 1

            else:

                if measure == 'fur':
                    measure = np.array(data[measure])
                    np.put(heat_data, np.where(measure == 1)[0],
                           len(cf.row_colors))

                else:
                    measure = np.array(data[measure])
                    np.put(heat_data, np.where(measure == 1)[0], i)
                    i += 1

        title = cf.proposal_dict[prop_text] + ': month ' + str(mnum)

    heat_data = heat_data.reshape(rows, cols)

    cmap = colors.ListedColormap(cmap_colors)

    with sns.axes_style('ticks'):

        if cell_border:
            sns.heatmap(heat_data, vmin=0, vmax=len(cmap_colors),
                        cbar=False, annot=False,
                        cmap=cmap, linewidths=0.005, linecolor=border_color)
        else:
            sns.heatmap(heat_data, vmin=0, vmax=len(cmap_colors),
                        cbar=False, annot=False,
                        cmap=cmap)

        plt.gcf().set_size_inches(xsize, ysize)
        plt.xticks([])
        plt.yticks(fontsize=min(ysize - 6, 10))
        plt.ylabel(str(cols) + ' per row', fontsize=12)
        plt.title(title, fontsize=18, y=1.01)
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.show()


def quartile_bands_over_time(df, eg, measure, formatter, bins=20,
                             clip=True, year_clip=2035, kind='area',
                             quartile_ticks=False,
                             custom_color=True, cm_name='Set1',
                             quartile_alpha=.75, grid_alpha=.5,
                             custom_start=0, custom_finish=.75,
                             xsize=10, ysize=8, alt_bg_color=False,
                             bg_color='#faf6eb'):

    if custom_color:
        cm_subsection = np.linspace(custom_start, custom_finish, bins)
        colormap = eval('cm.' + cm_name)
        quartile_colors = [colormap(x) for x in cm_subsection]
    else:
        quartile_colors = cf.color1

    if clip:
        eg_df = df[(df.eg == eg) & (df.date.dt.year <= year_clip)]
    else:
        eg_df = df[df.eg == eg]

    eg_df = eg_df[['date', 'empkey', measure]]
    eg_df['year'] = eg_df.date.dt.year

    bin_lims = np.linspace(0, 1, num=bins + 1, endpoint=True, retstep=False)
    years = np.unique(eg_df.year)

    result_arr = np.zeros((years.size, bin_lims.size - 1))

    if measure in ['spcnt', 'lspcnt']:
        filler = 1
    else:
        filler = 0

    grouped = eg_df.groupby(['year', 'empkey'])[measure].mean().reset_index()[
        ['year', measure]].fillna(filler)

    i = 0
    denom = len(grouped[grouped.year == min(eg_df.year)])

    for year in years:
        this_year = grouped[grouped.year == year][measure]
        these_bins = pd.cut(this_year, bin_lims)
        these_counts = this_year.groupby(these_bins).count()
        these_pcnts = these_counts / denom
        result_arr[i, :] = these_pcnts
        i += 1

    frm = pd.DataFrame(result_arr, columns=np.arange(1, bins + 1), index=years)

    with sns.axes_style('white'):

        if custom_color:
            cm_subsection = np.linspace(custom_start, custom_finish, bins)
            colormap = eval('cm.' + cm_name)
            quartile_colors = [colormap(x) for x in cm_subsection]
        else:
            quartile_colors = cf.color1

        step = 1 / bins
        offset = step / 2
        legend_pos_adj = 0
        quartiles = np.arange(1, bins + 1)

        ymajor_ticks = np.arange(offset, 1, step)
        yminor_ticks = np.arange(0, 1, step)

        xmajor_ticks = np.arange(min(df.date.dt.year), year_clip, 1)

        if kind == 'area':
            frm.plot(kind=kind, linewidth=1, stacked=True,
                     color=quartile_colors, alpha=quartile_alpha)
        elif kind == 'bar':
            frm.plot(kind=kind, width=1, stacked=True,
                     color=quartile_colors, alpha=quartile_alpha)

        ax = plt.gca()
        plt.ylim(0, 1)

        if quartile_ticks:

            if kind == 'area':
                ax.set_xticks(xmajor_ticks, minor=True)
            ax.set_yticks(ymajor_ticks)
            ax.set_yticks(yminor_ticks, minor=True)
            ax.invert_yaxis()
            plt.ylabel('original quartile')

            if bins > 20:
                plt.gca().legend_ = None
            if bins < 41:

                if kind != 'area':
                    ax.grid(which='major', color='grey',
                            ls='dotted', alpha=grid_alpha)
                else:
                    ax.grid(which='minor', color='grey', alpha=grid_alpha)
                ax.yaxis.set(ticklabels=np.arange(1, bins + 1))
            else:

                ax.grid(which='minor', color='grey', alpha=0.1)
                plt.tick_params(axis='y', which='both', left='off',
                                right='off', labelleft='off')

            legend_labels = quartiles
            legend_title = 'result quartile'

        else:
            ax.set_yticks(ymajor_ticks + offset)

            if kind == 'area':
                ax.set_xticks(xmajor_ticks, minor=True)
                ax.grid(which='both', color='grey', alpha=grid_alpha)
            else:
                ax.grid(which='major', color='grey',
                        ls='dotted', alpha=grid_alpha)

            ax.invert_yaxis()
            plt.ylabel('original percentage')
            plt.xlabel('year')
            plt.gca().yaxis.set_major_formatter(formatter)
            legend_labels = ['{percent:.1f}'
                             .format(percent=((quart * step) - step) * 100) +
                             ' - ' +
                             '{percent:.1%}'.format(percent=quart * step)
                             for quart in quartiles]
            legend_title = 'result_pcnt'
            legend_pos_adj = .1

        if alt_bg_color:
            ax.set_axis_bgcolor(bg_color)

        plt.title('<group name> quartile change over time',
                  fontsize=16, y=1.02)

        quartiles = np.arange(1, bins + 1)

        recs = []
        patch_alpha = min(quartile_alpha + .1, 1)
        legend_font_size = np.clip(int(bins / 1.65), 12, 14)
        legend_cols = int(bins / 30) + 1
        legend_position = 1 + (legend_cols * .17) + legend_pos_adj

        for i in np.arange(bins, dtype='int'):
            recs.append(mpatches.Rectangle((0, 0), 1, 1,
                                           fc=quartile_colors[i],
                                           alpha=patch_alpha))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 1.0, box.height])
        ax.legend(recs, legend_labels, bbox_to_anchor=(legend_position, 1),
                  title=legend_title, fontsize=legend_font_size,
                  ncol=legend_cols)

        fig = plt.gcf()
        fig.set_size_inches(xsize, ysize)
        fig.tight_layout()
        plt.show()
