# -*- coding: utf-8 -*-
#

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from ipywidgets import interactive, Button, widgets
from IPython.display import display, Javascript
import math
from os import system

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
                               rotate=False, gain_loss_bg=False, bg_alpha=.05,
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
            the total number of segments (divisions of the population) to
            calculate and display

        job_str_list
            a list of strings which correspond with the job levels, used for
            the chart legend
            example:
                jobs = ['Capt G4', 'Capt G3', 'Capt G2', ....]

        proposal
            text name of the (proposal) dataset, used as key in the
            proposal dict

        proposal_dict
            a dictionary of proposal text keys and corresponding proposal text
            descriptions, used for chart titles

        eg_dict
            dictionary used to convert employee group numbers to text,
            used with chart title text display

        color_list
            a list of color codes which control the job level color display

        style
            option to select 'area' or 'bar' to determine the type
            of chart output. default is 'bar'.

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
            max x axis value (years) if normalize_yr_scale set True'''

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
    legend_font_size = np.clip(int(ysize * 1.65), 12, 16)
    ytick_fontsize = np.clip(int(ysize * 1.55), 9, 14)

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
            sa_cols = list(sa_quantile_yrs.columns)

            if gain_loss_bg:
                sa_labels = ['Loss', 'Gain']
                sa_colors = ['r', 'g']

            else:
                sa_labels = []
                sa_colors = []

            for sa_col in sa_cols:
                sa_labels.append(job_str_list[sa_col - 1])
                sa_colors.append(color_list[sa_col - 1])

            # patch_alpha = min(quartile_alpha + .1, 1)
            # legend_font_size = np.clip(int(bins / 1.65), 12, 14)
            # legend_cols = int(bins / 30) + 1
            # legend_position = 1 + (legend_cols * .17) + legend_pos_adj
            if gain_loss_bg:
                recs = []
                for i in np.arange(len(sa_cols) + 2):
                    if i <= 1:
                        patch_alpha = .2
                    else:
                        patch_alpha = 1
                    recs.append(mpatches.Rectangle((0, 0), 1, 1,
                                                   fc=sa_colors[i],
                                                   alpha=patch_alpha))

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
            ax.legend((labels), loc='center left',
                      bbox_to_anchor=(1, 0.5),
                      fontsize=legend_font_size)
            plt.yticks(fontsize=ytick_fontsize)
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
                    if gain_loss_bg:
                        plt.axvspan(0, x_max, facecolor='g', alpha=bg_alpha)
                        plt.axvspan(0, x_min, facecolor='r', alpha=bg_alpha)
                else:
                    plt.ylabel('years')
                    plt.xlabel('quartiles')
                    if normalize_yr_scale:
                        plt.ylim(year_clip / -3, year_clip / 3)
                    if flip_y:
                        ax.invert_yaxis()
                    ymin, ymax = plt.ylim()
                    if gain_loss_bg:
                        plt.axhspan(0, ymax, facecolor='g', alpha=bg_alpha)
                        plt.axhspan(0, ymin, facecolor='r', alpha=bg_alpha)
                    ax.invert_xaxis()
                    plt.xticks(rotation='horizontal')

                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                if gain_loss_bg:
                    ax.legend(
                        recs, (sa_labels), loc='center left',
                        bbox_to_anchor=(1, 0.5),
                        fontsize=legend_font_size)
                else:
                    ax.legend(
                        (sa_labels), loc='center left',
                        bbox_to_anchor=(1, 0.5),
                        fontsize=legend_font_size)

                plt.suptitle(proposal_dict[proposal] +
                             ', GROUP ' + eg_dict[eg],
                             fontsize=20, y=1.02)
                plt.title('years differential vs standalone, ' +
                          str(num_bins) + '-quantiles',
                          y=1.02)
                plt.yticks(fontsize=ytick_fontsize)
                fig = plt.gcf()
                fig.set_size_inches(xsize, ysize)
                plt.show()


def age_vs_spcnt(df, eg_list, mnum, color_list,
                 eg_dict, proposal_text, proposal_dict,
                 formatter, xsize=10, ysize=8, chart_example=False):
    '''scatter plot with age on x axis and list percentage on y axis.
    note: input df may be prefiltered to plot focus attributes, i.e.
    filter to include only employees at a certain job level, hired
    between certain dates, with a particular age range, etc.

    inputs

        df
            input dataset

        eg_list
            list of employee groups to include
            example: [1, 2]

        mnum
            month number to study from dataset

        color_list
            color codes for plotting each employee group

        eg_dict
            dictionary, numerical eg code to string description

        proposal_text
            string representation of df variable

        proposal_dict
            dictionary proposal_text to proposal text description

        formatter
            matplotlib chart formatter for percentage axis display
    '''

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    d_age_pcnt = df[df.mnum == mnum][['age', 'mnum', 'spcnt', 'eg']].copy()

    for grp in eg_list:
        d_for_plot = d_age_pcnt[d_age_pcnt.eg == grp]
        x = d_for_plot['age']
        y = d_for_plot['spcnt']
        if chart_example:
            ax.scatter(x, y, c=color_list[grp - 1],
                       s=20, linewidth=0.1, edgecolors='w',
                       label=str(grp))
        else:
            ax.scatter(x, y, c=color_list[grp - 1],
                       s=20, linewidth=0.1, edgecolors='w',
                       label=eg_dict[grp])

    plt.ylim(1, 0)
    plt.xlim(25, 65)
    plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.yticks(np.arange(0, 1.05, .05))
    if chart_example:
        plt.title('Proposal 1' +
                  ' - age vs seniority percentage' +
                  ', month ' +
                  str(mnum), y=1.02)
    else:
        plt.title(proposal_dict[proposal_text] +
                  ' - age vs seniority percentage' +
                  ', month ' +
                  str(mnum), y=1.02)
    plt.legend(loc=2)
    fig = plt.gcf()
    fig.set_size_inches(xsize, ysize)
    plt.ylabel('spcnt')
    plt.xlabel('age')
    plt.show()


def multiline_plot_by_emp(df, measure, xax, emp_list, job_levels,
                          color_list, job_str_list, proposal,
                          proposal_dict, formatter,
                          chart_example=False):
    '''select example individual employees and plot career measure
    from selected dataset attribute, i.e. list percentage, career
    earnings, job level, etc.

    inputs

        df
            dataset to examine

        measure
            dataset attribute to plot

        xax
            dataset attribute for x axis

        emp_list
            list of employee numbers or ids

        job_levels
            number of job levels in model

        color list
            list of colors for plotting

        job_str_list
            list of string jog descriptions corresponding to
            number of job levels

        proposal
            string representation of df variable

        proposal_dict
            dictionary proposal to proposal text description

        formatter
            matplotlib chart formatter for proper percentage axis display
    '''

    frame = df.copy()
    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:
        frame = frame[frame.jnum <= cf.num_of_job_levels]
    if measure in ['mpay']:
        frame = frame[frame.age < 65]

    i = 0

    for emp in emp_list:
        if chart_example:
            frame[frame.empkey == emp].set_index(xax)[measure] \
                .plot(label='Employee ' + str(i + 1))
            i += 1
        else:
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
        plt.yticks(np.arange(0, 1.05, .05))
    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:

        plt.yticks(np.arange(0, job_levels + 2, 1))
        plt.ylim(job_levels + 1.5, 0.5)

        yticks = fig.get_yticks().tolist()

        for i in np.arange(1, len(yticks)):
            yticks[i] = job_str_list[i - 1]
        plt.axhspan(job_levels + 1, job_levels + 2, facecolor='.8', alpha=0.9)
        fig.set_yticklabels(yticks)
        plt.axhline(y=job_levels + 1, c='.8', ls='-', alpha=.8, lw=3)

    if xax in ['spcnt', 'lspcnt']:
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xticks(np.arange(0, 1.1, .1))
        plt.xlim(1, 0)

    if chart_example:
        plt.title(measure + ' - ' + 'proposal 1', y=1.02)
    else:
        plt.title(measure + ' - ' + proposal_dict[proposal], y=1.02)
    plt.ylabel(measure)
    plt.legend(loc=4)
    plt.show()


def multiline_plot_by_eg(df, measure, xax, eg_list, job_dict,
                         proposal, proposal_dict,
                         job_levels, colors, formatter, mnum=0,
                         scatter=False, exclude_fur=False,
                         full_pcnt_xscale=False, chart_example=False):

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
                   'lspcnt', 'rank_in_job', 'cat_order']:
        fig.invert_yaxis()
    if measure in ['spcnt', 'lspcnt']:
        fig.yaxis.set_major_formatter(formatter)
        plt.yticks(np.arange(0, 1.05, .05))
    if xax in ['spcnt', 'lspcnt']:
        fig.xaxis.set_major_formatter(formatter)
        plt.xticks(np.arange(0, 1.1, .1))

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
    if chart_example:
        plt.title(measure.upper() +
                  ' ordered by ' + xax + ' - ' +
                  'Proposal 3' + ' - Month: ' + str(mnum), y=1.02)
        if measure == 'ylong':
            plt.ylim(0, 40)
    else:
        plt.title(measure.upper() +
                  ' ordered by ' + xax + ' - ' +
                  proposal_dict[proposal] + ' - Month: ' + str(mnum), y=1.02)
    plt.ylabel(measure)
    plt.xlabel(xax)
    plt.show()


def violinplot_by_eg(df, measure, proposal, proposal_dict, formatter,
                     mnum=0, bw=.1, linewidth=1.5, chart_example=False,
                     scale='count'):

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
    if chart_example:
        plt.title('Proposal 3' + ' - ' +
                  measure.upper() + ' Distribution - Month ' +
                  str(mnum), y=1.02)
    else:
        plt.title(proposal_dict[proposal] + ' - ' +
                  measure.upper() + ' Distribution - Month ' +
                  str(mnum), y=1.02)
    fig = plt.gca()
    if measure == 'age':
        plt.ylim(25, 70)
    if measure in ['snum', 'spcnt', 'jnum', 'jobp']:
        fig.invert_yaxis()
        if measure in ['spcnt', 'lspcnt']:
            fig.yaxis.set_major_formatter(formatter)
            plt.yticks(np.arange(0, 1.05, .05))
            plt.ylim(1.04, -.04)
    plt.show()


def age_kde_dist(df, mnum=0, eg=None, chart_example=False):

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

            if chart_example:
                sns.kdeplot(frame[frame.eg == x].age,
                            shade=True, color=color,
                            bw=.8, ax=ax, label='Group ' + str(x))
            else:
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
                    xsize=18, ysize=10, chart_example=False):
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
        if chart_example:
            plt.title('PROPOSAL 3 vs. standalone ' + measure.upper(),
                      y=1.02)
        else:
            plt.title(yval_dict[yval], y=1.02)
        plt.show()


# DISTRIBUTION WITHIN JOB LEVEL (NBNF effect)
def stripplot_distribution_in_category(df, job_levels, mnum, blk_pcnt,
                                       eg_colors, band_colors, bg_alpha=.12,
                                       adjust_y_axis=False,
                                       chart_example=False):

    fur_lvl = job_levels + 1
    if job_levels == 16:
        adjust = [0, 0, 0, 0, 0, 0, -50, 50, 0, 0, -50, 50, 75, 0, 0, 0, 0]
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
        axis2_lbl_locs[i] += adjust[job_num - 1]
        axis2_lbls.append(jobs_dict[job_num])
        i += 1

    counter = 0
    for i in np.arange(1, len(axis2_lbl_locs)):
        this_diff = axis2_lbl_locs[i] - axis2_lbl_locs[i - 1]
        # space labels if overlapping...
        if this_diff < 220:
            counter += 1
            if counter == 1:
                axis2_lbl_locs[i - 1] = axis2_lbl_locs[i] - 220
            else:
                axis2_lbl_locs[i] = axis2_lbl_locs[i - 1] + 220

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
    if chart_example:
        for tck in xticks:
            tick_dummies.append('Group ' + str(tck + 1))
    else:
        for tck in xticks:
            tick_dummies.append(eg_dict[tck + 1])

    plt.gca().set_xticklabels(tick_dummies)

    plt.gcf().set_size_inches(7, 12)
    plt.title(
        'Group distribution within job levels, month ' + str(mnum), y=1.04)
    plt.xlabel('test')
    plt.show()


def job_level_progression(ds, emp_list, through_date, job_levels,
                          eg_colors, band_colors,
                          job_counts, job_change_lists, alpha=.12,
                          chart_example=False):

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
        if chart_example:
            for emp in emp_list:
                ax1 = ds[ds.empkey == emp].set_index('date')[:through_date] \
                    .cat_order.plot(lw=3, color=eg_colors[i],
                                    label='Employee ' + str(i + 1))
                i += 1
        else:
            for emp in emp_list:
                ax1 = ds[ds.empkey == emp].set_index('date')[:through_date] \
                    .cat_order.plot(lw=3, color=eg_colors[i], label=emp)
                i += 1
        non_ret_count['count'].plot(c='k', ls='--',
                                    label='active count', ax=ax1)
        ax1.legend()

        plt.axvline(cf.imp_date, c='g', ls='--', alpha=1, lw=1)

    with sns.axes_style("white"):
        ax2 = jobs_table.plot.area(stacked=True,
                                   figsize=(12, 10),
                                   sort_columns=True,
                                   linewidth=2,
                                   color=band_colors,
                                   alpha=alpha,
                                   legend=False,
                                   ax=ax1)

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


def differential_scatter(base_ds, compare_ds_list,
                         measure, filter_measure,
                         filter_val, formatter, prop_order=True,
                         show_scatter=True, show_lin_reg=True,
                         show_mean=True, mean_len=50, eg_list=[1, 2, 3],
                         dot_size=15, lin_reg_order=15, ylimit=False, ylim=5,
                         width=22, height=14, bright_bg=False,
                         chart_style='whitegrid', chart_example=False):

    cols = [measure, 'new_order']

    # try:
    #     p_egs_and_order = prop_ds[prop_ds[filter_measure] == filter_val][
    #         ['eg', 'new_order']]
    # except:
    #     p_egs_and_order = prop_ds[prop_ds[filter_measure] == filter_val][
    #         ['eg', 'idx']]

    df = base_ds[base_ds[filter_measure] == filter_val][[measure, 'eg']].copy()
    df.rename(columns={measure: measure + '_s'}, inplace=True)

    yval_list = ['avs', 'evs', 'wvs']

    yval_dict = {'avs': 'AASIC',
                 'evs': 'EAST',
                 'wvs': 'WEST'}

    order_dict = {'avs': 'order1',
                  'evs': 'order2',
                  'wvs': 'order3'}

    i = 1
    for ds in compare_ds_list:
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

                if chart_example:
                    label = str(eg)
                else:
                    label = cf.eg_dict[eg]

                if show_scatter:
                    data.plot(x=xval, y=yval, kind='scatter', linewidth=0.1,
                              color=cf.eg_colors[eg - 1], s=dot_size,
                              label=label,
                              ax=ax)

                if show_mean:
                    data['ma'] = data[yval].rolling(mean_len).mean()
                    data.plot(x=xval, y='ma', lw=5,
                              color=cf.mean_colors[eg - 1],
                              label=label,
                              alpha=.6, ax=ax)
                    plt.xlim(0, x_limit)

                if show_lin_reg:
                    if show_scatter:
                        lr_colors = cf.lr_colors
                    else:
                        lr_colors = cf.lr_colors2
                    sns.regplot(x=xval, y=yval, data=data,
                                color=lr_colors[eg - 1],
                                label=label,
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
            if chart_example:
                plt.title('Proposal 1' + ' differential: ' + measure)
            else:
                plt.title(yval_dict[yval] + ' differential: ' + measure)
            plt.xlim(xmin=0)

            if measure in ['spcnt', 'lspcnt']:
                plt.gca().yaxis.set_major_formatter(formatter)

            if xval == 'sep_eg_pcnt':
                plt.gca().xaxis.set_major_formatter(formatter)
                plt.xticks(np.arange(0, 1.1, .1))
                plt.xlim(xmax=1)

            ax.axhline(0, c='m', ls='-', alpha=1, lw=2)
            ax.invert_xaxis()
            if bright_bg:
                ax.set_axis_bgcolor('#faf6eb')
            plt.ylabel('differential')
            plt.show()


# groupby method:
# only age 65 in group, use .sum
# others, use .mean
def job_grouping_over_time(proposal, prop_text, eg_list, jobs, colors,
                           formatter, plt_kind='bar', rets_only=True,
                           time_group='A', display_yrs=40, legend_loc=4,
                           xsize=12, ysize=10, chart_example=False):

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
            if chart_example:
                plt.title('Proposal 1' + ' group ' + str(eg), y=1.01)
            else:
                plt.title(cf.proposal_dict[prop_text] + ' group ' +
                          cf.eg_dict[eg], y=1.01)
            plt.gcf().set_size_inches(xsize, ysize)
            plt.show()


def parallel(dsa, dsb, dsc, dsd, eg_list, measure, month_list, job_levels,
             formatter, left='sa', a_stride=50, e_stride=30, w_stride=20,
             xsize=6, ysize=8, chart_example=False):

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
            if chart_example:
                df_joined.columns = ['eg', 'StandAlone',
                                     'List2', 'List3', 'List1']
            else:
                df_joined.columns = ['eg', 'StandAlone',
                                     'EAST', 'WEST', 'AMER']

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
            if chart_example:
                plt.title('Group 1' + ' ' + measure.upper() + ' ' +
                          str(month) + ' mths', fontsize=16, y=1.02)
            else:
                plt.title(group_dict[1].upper() + ' ' + measure.upper() + ' ' +
                          str(month) + ' mths', fontsize=16, y=1.02)

        if 2 in eg_list:
            plot_num += 1
            plt.subplot(num_months, num_egplots, plot_num)
            df_2 = df_joined[df_joined.eg == 2]
            df_2 = df_2[::e_stride]
            parallel_coordinates(df_2, 'eg', lw=1.5, alpha=.7,
                                 color=color_dict[2])
            if chart_example:
                plt.title('Group 2' + ' ' + measure.upper() + ' ' +
                          str(month) + ' mths', fontsize=16, y=1.02)
            else:
                plt.title(group_dict[2].upper() + ' ' + measure.upper() + ' ' +
                          str(month) + ' mths', fontsize=16, y=1.02)

        if 3 in eg_list:
            plot_num += 1
            plt.subplot(num_months, num_egplots, plot_num)
            df_3 = df_joined[df_joined.eg == 3]
            df_3 = df_3[::w_stride]
            parallel_coordinates(df_3, 'eg', lw=1.5, alpha=.7,
                                 color=color_dict[3])
            if chart_example:
                plt.title('Group 3' + ' ' + measure.upper() + ' ' +
                          str(month) + ' mths', fontsize=16, y=1.02)
            else:
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


def rows_of_color(prop_text, prop, mnum, measure_list, cmap_colors,
                  jnum_colors, cols=200,
                  job_only=False, jnum=1, cell_border=True, border_color='.5',
                  xsize=14, ysize=12, chart_example=False):

    data = prop[prop.mnum == mnum]
    rows = int(len(prop[prop.mnum == 0]) / cols) + 1
    heat_data = np.zeros(cols * rows)

    i = 1

    if ('jnum' in measure_list) and (not job_only):
        i = 0
        if cf.num_of_job_levels == 16:
            cmap_colors = cf.paired_colors16
        elif cf.num_of_job_levels == 8:
            cmap_colors = cf.paired_colors8
        else:
            cmap_colors = jnum_colors

    if job_only:

        eg = np.array(data.eg)
        jnums = np.array(data.jnum)

        for eg_num in np.unique(eg):
            np.put(heat_data, np.where(eg == eg_num)[0], eg_num)
        np.put(heat_data, np.where(jnums != jnum)[0], 0)
        if chart_example:
            title = 'Proposal 1' + ' month ' + str(mnum) + \
                ':  ' + cf.job_strs[jnum - 1] + '  job distribution'
        else:
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
        if chart_example:
            title = 'Proposal 1' + ': month ' + str(mnum)
        else:
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


def job_transfer(p_df, p_text, comp_df, comp_df_text, eg, colors,
                 job_levels,
                 measure='jnum', gb_period='M',
                 custom_color=True, cm_name='Paired',
                 start=0, stop=.95, job_alpha=1, chart_style='white',
                 start_date=cf.starting_date, yticks_lim=5000,
                 ytick_interval=100, legend_xadj=1.62,
                 legend_yadj=.78, annotate=False,
                 xsize=10, ysize=8):

    p_df = p_df[p_df.eg == eg][['date', measure]].copy()
    comp_df = comp_df[comp_df.eg == eg][['date', measure]].copy()

    pg = pd.DataFrame(p_df.groupby(['date', measure]).size()
                      .unstack().fillna(0).resample(gb_period).mean())
    cg = pd.DataFrame(comp_df.groupby(['date', measure]).size()
                      .unstack().fillna(0).resample(gb_period).mean())

    for job_level in np.arange(1, cf.num_of_job_levels + 1):
        if job_level not in cg:
            cg[job_level] = 0
        if job_level not in pg:
            pg[job_level] = 0
    cg.sort_index(axis=1, inplace=True)
    pg.sort_index(axis=1, inplace=True)

    diff2 = pg - cg
    abs_diff2 = np.absolute(diff2.values).astype(int)
    v_crop = (np.amax(np.add.reduce(abs_diff2, axis=1)) / 2) + 75

    if custom_color:
        num_of_colors = job_levels + 1
        cm_subsection = np.linspace(start, stop, num_of_colors)
        colormap = eval('cm.' + cm_name)
        colors = [colormap(x) for x in cm_subsection]

    with sns.axes_style(chart_style):
        ax = diff2.plot(kind='bar', width=1, linewidth=0,
                        color=colors, stacked=True, alpha=job_alpha)
        fig = plt.gcf()

        xtick_locs = ax.xaxis.get_majorticklocs()

        if gb_period == 'M':
            i = 13 - pd.to_datetime(cf.starting_date).month
        elif gb_period == 'Q':
            i = ((13 - pd.to_datetime(cf.starting_date).month) // 3) + 1
        elif gb_period == 'A':
            i = 0

        xtick_dict = {'A': 1,
                      'Q': 4,
                      'M': 12}

        interval = xtick_dict[gb_period]

        xticklabels = [''] * len(diff2.index)

        xticklabels[i::interval] = \
            [item.strftime('%Y') for item in diff2.index[i::interval]]

        if gb_period in ['Q']:
            ax.set_xticklabels(xticklabels, rotation=90, ha='right')
        else:
            ax.set_xticklabels(xticklabels, rotation=90)

        for xmaj in xtick_locs:
            try:
                if i % interval == 0:
                    ax.axvline(xtick_locs[i], ls='-', color='grey',
                               lw=1, alpha=.2, zorder=7)
                i += 1
            except:
                pass
        if gb_period in ['Q', 'A']:
            ax.axvline(xtick_locs[0], ls='-', color='grey',
                       lw=1, alpha=.2, zorder=7)

        yticks = np.arange(-yticks_lim, yticks_lim, ytick_interval)
        ax.set_yticks(yticks)
        plt.ylim(-v_crop, v_crop)
        ymin, ymax = ax.get_ylim()
        for i in yticks:
            ax.axhline(i, ls='-', color='grey', lw=1, alpha=.2, zorder=7)

        recs = []
        job_labels = []
        legend_font_size = 12
        legend_position = 1.12
        legend_title = 'job'

        for i in diff2.columns:
            recs.append(mpatches.Rectangle((0, 0), 1, 1,
                                           fc=colors[i - 1],
                                           alpha=job_alpha))
            job_labels.append(cf.job_strs[i - 1])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * legend_xadj, box.height])
        ax.legend(recs, job_labels, bbox_to_anchor=(legend_position,
                                                    legend_yadj),
                  title=legend_title, fontsize=legend_font_size,
                  ncol=1)

        if annotate:
            if gb_period in ['Q', 'M']:
                ax.annotate('job count increase',
                            xy=(50, v_crop - ytick_interval),
                            xytext=(50, v_crop - ytick_interval), fontsize=14)
                ax.annotate('job count decrease',
                            xy=(50, -v_crop + ytick_interval),
                            xytext=(50, -v_crop + ytick_interval), fontsize=14)

        ax.axhline(color='grey', alpha=.7)
        plt.axhspan(0, ymax, facecolor='g', alpha=0.05, zorder=8)
        plt.axhspan(0, ymin, facecolor='r', alpha=0.05, zorder=8)
        plt.ylabel('change in job count', fontsize=16)
        plt.xlabel('date', fontsize=16, labelpad=15)
        title_string = 'GROUP ' + cf.eg_dict[eg] + \
            ' Jobs Exchange' + '\n' + \
            cf.proposal_dict[p_text] + \
            ' compared to ' + cf.proposal_dict[comp_df_text]
        plt.title(title_string,
                  fontsize=16, y=1.02)

        fig.set_size_inches(xsize, ysize)
        plt.show()


def editor(base_ds, compare_ds_text, prop_order=True,
           mean_len=80, eg_list=[1, 2, 3],
           dot_size=20, lin_reg_order=12, ylimit=False, ylim=5,
           width=17.5, height=10, strip_height=3.5, bright_bg=True,
           chart_style='whitegrid', bg_clr='#fffff0'):
    '''compare specific proposal attributes and interactively adjust
    list order.  may be used to minimize distortions.  utilizes ipywidgets.

    inputs

        base_ds
            baseline dataset

        compare_ds_text
            string representation of comparison dataset variable

        prop_order
            order the output differential chart x axis in proposal
            (or edited dataset) order, necessary to use the interactive
            tool.  If False, the x axis is arranged in native list
            order for each group

        mean_len
            length of rolling mean if 'mean' selected for display

        eg_list
            list of egs(employee groups) to compare and plot

        dot_size
            chart dot size

        lin_reg_order
            polynomial fit order

        ylimit
            limit the y axis scale in scope if outliers exist

        ylim
            limit for ylimit input

        width
            width of chart

        height
            height of chart

        strip_height
            height of stripplot (group density display)

        bright_bg
            fill chart background with alternate color

        chart_style
            seaborn chart style

        bg_clr
            color input for bright_bg option

    '''

    try:
        compare_ds = pd.read_pickle('dill/' + compare_ds_text + '.pkl')
    except:
        compare_ds = pd.read_pickle('dill/ds1.pkl')

    max_month = max(compare_ds.mnum)
    persist = pd.read_pickle('dill/squeeze_vals.pkl')

    chk_scatter = widgets.Checkbox(description='scatter',
                                   value=bool(persist['scat_val'].value))
    chk_fit = widgets.Checkbox(description='poly_fit',
                               value=bool(persist['fit_val'].value))
    chk_mean = widgets.Checkbox(description='mean',
                                value=bool(persist['mean_val'].value))
    drop_measure = widgets.Dropdown(options=['jobp', 'spcnt',
                                             'lspcnt', 'snum', 'lnum',
                                             'cat_order', 'mpay', 'cpay'],
                                    value=persist['drop_msr'].value,
                                    description='msr')
    drop_filter = widgets.Dropdown(options=['age', 'mnum'],
                                   value=persist['drop_filter'].value,
                                   description='fltr')
    int_val = widgets.IntText(min=0,
                              max=max_month,
                              value=persist['int_sel'].value,
                              description='val')

    measure = drop_measure.value
    filter_measure = drop_filter.value
    filter_val = int_val.value

    cols = [measure, 'new_order']
    df = base_ds[base_ds[filter_measure] == filter_val][[measure, 'eg']].copy()
    df.rename(columns={measure: measure + '_b'}, inplace=True)

    yval = 'differential'

    # for stripplot and squeeze:
    data_reorder = compare_ds[compare_ds.mnum == 0][['eg']].copy()
    data_reorder['new_order'] = np.arange(len(data_reorder)).astype(int)

    to_join_ds = compare_ds[
        compare_ds[filter_measure] == filter_val][cols].copy()

    to_join_ds.rename(columns={measure: measure + '_c',
                               'new_order': 'proposal_order'}, inplace=True)
    df = df.join(to_join_ds)
    x_limit = int(max(df.proposal_order) // 100) * 100 + 100
    df.sort_values(by='proposal_order', inplace=True)
    df['squeeze_order'] = np.arange(len(df))
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

        df[yval] = df[measure + '_b'] - df[measure + '_c']

    else:
        df[yval] = df[measure + '_c'] - df[measure + '_b']

    fig, ax = plt.subplots(figsize=(width, height))

    with sns.axes_style(chart_style):

        df.sort_values(by='proposal_order', inplace=True)

        if prop_order:
            xval = 'proposal_order'

        else:
            xval = 'sep_eg_pcnt'

        for eg in eg_list:
            data = df[df.eg == eg].copy()

            if chk_scatter.value:
                ax = data.plot(x=xval, y=yval, kind='scatter', linewidth=0.1,
                               color=cf.eg_colors[eg - 1], s=dot_size,
                               label=cf.eg_dict[eg],
                               ax=ax)

            if chk_mean.value:
                data['ma'] = data[yval].rolling(mean_len).mean()
                ax = data.plot(x=xval, y='ma', lw=5,
                               color=cf.mean_colors[eg - 1],
                               label=cf.eg_dict[eg],
                               alpha=.6, ax=ax)

            if chk_fit.value:
                if chk_scatter.value:
                    lr_colors = cf.lr_colors
                else:
                    lr_colors = cf.lr_colors2
                ax = sns.regplot(x=xval, y=yval, data=data,
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
                plt.ylim(-ylim, ylim)
            else:
                plt.ylim(-scale_lim, scale_lim)

        if measure in ['cat_order', 'snum', 'lnum']:
            ax.invert_yaxis()

        plt.gcf().set_size_inches(width, height)
        plt.title('Differential: ' + measure)
        plt.xlim(xmin=0)

        if measure in ['spcnt', 'lspcnt']:
            formatter = FuncFormatter(f.to_percent)
            plt.gca().yaxis.set_major_formatter(formatter)

        if xval == 'sep_eg_pcnt':
            plt.xlim(xmax=1)

        ax.axhline(0, c='m', ls='-', alpha=1, lw=1.5)
        ax.invert_xaxis()
        if bright_bg:
            # ax.set_axis_bgcolor('#faf6eb')
            ax.set_axis_bgcolor(bg_clr)

        plt.close(fig)

        v1line = ax.axvline(2, color='m', lw=2, ls='dashed')
        v2line = ax.axvline(200, color='c', lw=2, ls='dashed')
        range_list = [0, 0]

        if prop_order:
            try:
                junior_init = persist['junior'].value
                senior_init = persist['senior'].value
            except:
                junior_init = int(.8 * x_limit)
                senior_init = int(.2 * x_limit)
        else:
            try:
                junior_init = persist['junior'].value
                senior_init = persist['senior'].value
            except:
                junior_init = .8
                senior_init = .2

        drop_eg_dict = {'1': 1, '2': 2, '3': 3}
        drop_dir_dict = {'u  >>': 'u', '<<  d': 'd'}
        incr_dir_dict = {'u  >>': -1, '<<  d': 1}

        drop_eg = widgets.Dropdown(options=['1', '2', '3'],
                                   value=persist['drop_eg_val'].value,
                                   description='eg')

        drop_dir = widgets.Dropdown(options=['u  >>', '<<  d'],
                                    value=persist['drop_dir_val'].value,
                                    description='dir')

        drop_squeeze = widgets.Dropdown(options=['log', 'slide'],
                                        value=persist['drop_sq_val'].value,
                                        description='sq')

        slide_factor = widgets.IntSlider(value=persist['slide_fac_val'].value,
                                         min=1,
                                         max=400,
                                         step=1,
                                         description='factor')

        def set_cursor(junior=junior_init, senior=senior_init,):
            v1line.set_xdata((junior, junior))
            v2line.set_xdata((senior, senior))
            range_list[0] = junior
            range_list[1] = senior
            display(fig)

        def perform_squeeze(b):

            jval = v1line.get_xdata()[0]
            sval = v2line.get_xdata()[0]

            direction = drop_dir_dict[drop_dir.value]
            factor = slide_factor.value * .005
            incr_dir_correction = incr_dir_dict[drop_dir.value]
            increment = slide_factor.value * incr_dir_correction

            squeeze_eg = drop_eg_dict[drop_eg.value]

            if drop_squeeze.value == 'log':
                squeezer = f.squeeze_logrithmic(data_reorder,
                                                squeeze_eg,
                                                sval, jval,
                                                direction=direction,
                                                put_segment=1,
                                                log_factor=factor)
            if drop_squeeze.value == 'slide':
                squeezer = f.squeeze_increment(data_reorder,
                                               squeeze_eg,
                                               sval, jval,
                                               increment=increment)

            data_reorder['new_order'] = squeezer
            data_reorder.sort_values('new_order', inplace=True)
            data_reorder['new_order'] = np.arange(len(data_reorder),
                                                  dtype='int')

            fig, ax2 = plt.subplots(figsize=(width, strip_height))

            ax2 = sns.stripplot(x='new_order', y='eg',
                                data=data_reorder, jitter=.5,
                                orient='h', order=np.arange(1, 4, 1),
                                palette=cf.eg_colors, size=3,
                                linewidth=0, split=True)

            if bright_bg:
                # ax2.set_axis_bgcolor('#fff5e5')
                ax2.set_axis_bgcolor(bg_clr)

            plt.xticks(np.arange(0, len(data_reorder), 1000))
            if measure in ['spcnt', 'lspcnt', 'cpay']:
                plt.ylabel('\n\neg\n')
            else:
                plt.ylabel('eg\n')
            plt.xlim(len(data_reorder), 0)
            plt.show()

            data_reorder[['new_order']].to_pickle('dill/new_order.pkl')

            junior_val = range_sel.children[0].value
            senior_val = range_sel.children[1].value

            persist_df = pd.DataFrame({'drop_eg_val': drop_eg.value,
                                       'drop_dir_val': drop_dir.value,
                                       'drop_sq_val': drop_squeeze.value,
                                       'slide_fac_val': slide_factor.value,
                                       'scat_val': chk_scatter.value,
                                       'fit_val': chk_fit.value,
                                       'mean_val': chk_mean.value,
                                       'drop_msr': drop_measure.value,
                                       'drop_filter': drop_filter.value,
                                       'int_sel': int_val.value,
                                       'junior': junior_val,
                                       'senior': senior_val},
                                      index=['value'])

            persist_df.to_pickle('dill/squeeze_vals.pkl')

        def run_cell(ev):
            system('python compute_measures.py df1')
            display(Javascript('IPython.notebook.execute_cell()'))

        def redraw(ev):
            display(Javascript('IPython.notebook.execute_cell()'))

        button_calc = Button(description="calculate",
                             background_color='#99ddff')
        button_calc.on_click(run_cell)

        button_draw = Button(description="redraw",
                             background_color='#dab3ff')
        button_draw.on_click(redraw)
        # display(button)

        if prop_order:
            range_sel = interactive(set_cursor, junior=(0, x_limit),
                                    senior=(0, x_limit))
        else:
            range_sel = interactive(set_cursor, junior=(0, 1, .001),
                                    senior=(0, 1, .001))

        button = Button(description='squeeze',
                        background_color='#b3ffd9')
        button.on_click(perform_squeeze)

        hbox1 = widgets.HBox((button, button_calc, button_draw))
        vbox1 = widgets.VBox((slide_factor, hbox1, range_sel))
        vbox2 = widgets.VBox((chk_scatter, chk_fit, chk_mean))
        vbox3 = widgets.VBox((drop_squeeze, drop_eg, drop_dir))
        vbox4 = widgets.VBox((drop_measure, drop_filter, int_val))
        display(widgets.HBox((vbox1, vbox2, vbox3, vbox4)))


def eg_multiplot_with_cat_order(df, proposal, mnum, measure, xax,
                                formatter, proposal_dict, job_strs,
                                span_colors, job_levels,
                                single_eg=False, num=1, exclude_fur=False,
                                plot_scatter=True, s=20, a=.7, lw=0,
                                width=12, height=12,
                                chart_example=False):
    '''num input options:
                   {1: 'amer_with_twa',
                    2: 'east',
                    3: 'west',
                    4: 'amer_no_twa',
                    5: 'twa_only'
                    }
    '''

    max_count = df.groupby('mnum').size().max()
    mnum_count = np.unique(df.mnum).size
    df = df[df.mnum == mnum].copy()

    if measure == 'cat_order':
        sns.set_style('white')

        if job_levels == 16:
            eg_counts = f.convert_jcnts_to16(cf.eg_counts, cf.intl_blk_pcnt,
                                             cf.dom_blk_pcnt)
            j_changes = f.convert_job_changes_to16(cf.j_changes, cf.jd)

        if job_levels == 8:
            eg_counts = cf.eg_counts
            j_changes = cf.j_changes

        jcnts_arr = f.make_jcnts(eg_counts)

        table = f.job_gain_loss_table(mnum_count, job_levels, jcnts_arr,
                                      j_changes, standalone=False)
        job_ticks = np.cumsum(table[0][mnum])
        job_ticks = np.append(job_ticks, max_count)
        job_ticks = np.insert(job_ticks, 0, 0)
    else:
        sns.set_style('darkgrid')

    if single_eg:
        num = 1
        grp_dict = {1: 'amer_with_twa',
                    2: 'east',
                    3: 'west',
                    4: 'amer_no_twa',
                    5: 'twa_only'
                    }

        cdict = {1: 'black',
                 2: 'blue',
                 3: '#FF6600',
                 4: 'black',
                 5: 'green'
                 }

        if num == 5:
            df = df[(df.eg == 1) & (df.twa == 1)]
            label = 'twa_only'
        elif num == 4:
            df = df[(df.eg == 1) & (df.twa == 0)]
            label = 'amer_no_twa'
        elif num == 2:
            df = df[df.eg == 2]
            label = 'east'
        elif num == 3:
            df = df[df.eg == 3]
            label = 'west'
        elif num == 1:
            df = df[df.eg == 1]
            label = 'amer_with_twa'

        if exclude_fur:
            df = df[df.fur == 0]

        if plot_scatter:
            ax1 = df.plot(x=xax, y=measure, kind='scatter', color=cdict[num],
                          label=label, linewidth=lw, s=s)
        else:
            ax1 = df.set_index(xax, drop=True)[measure].plot(label=label,
                                                             color=cdict[num])
            print('''Ignore the vertical lines.
                  Look right to left within each job level
                  for each group\'s participation''')
        plt.title(grp_dict[num] + ' job disbursement - ' +
                  proposal_dict[proposal] + ' month=' + str(mnum), y=1.02)

    else:

        if exclude_fur:
            df = df[df.fur == 0]

        d1 = df[(df.eg == 1) & (df.twa == 1)]
        d2 = df[(df.eg == 1) & (df.twa == 0)]
        d3 = df[df.eg == 2]
        d4 = df[df.eg == 3]

        if plot_scatter:
            ax1 = d1.plot(x=xax, y=measure, kind='scatter',
                          label='amer_twa_only', color='#5cd65c',
                          alpha=a, s=s, linewidth=lw)
            d2.plot(x=xax, y=measure, kind='scatter',
                    label='amer_no_twa', color='black',
                    alpha=a, s=s, linewidth=lw, ax=ax1)
            d3.plot(x=xax, y=measure, kind='scatter',
                    label='east', color='blue',
                    alpha=a, s=s, linewidth=lw, ax=ax1)
            d4.plot(x=xax, y=measure, kind='scatter',
                    label='west', c='#FF6600',
                    alpha=a, s=s, linewidth=lw, ax=ax1)

        else:
            ax1 = d1.set_index(xax,
                               drop=True)[measure].plot(label='amer_twa_only',
                                                        color='green', alpha=a)
            d2.set_index(xax,
                         drop=True)[measure].plot(label='amer_no_twa',
                                                  color='black', alpha=a,
                                                  ax=ax1)
            d3.set_index(xax,
                         drop=True)[measure].plot(label='east',
                                                  color='blue', alpha=a,
                                                  ax=ax1)
            d4.set_index(xax,
                         drop=True)[measure].plot(label='west',
                                                  color='#FF6600', alpha=a,
                                                  ax=ax1)
            print('''Ignore the vertical lines.  \
                  Look right to left within each job \
                  level for each group\'s participation''')

        if chart_example:
            plt.title('job disbursement - ' +
                      'proposal 1' + ' - month ' + str(mnum), y=1.02)
        else:
            plt.title('job disbursement - ' +
                      proposal_dict[proposal] + ' month ' + str(mnum), y=1.02)

    ax1.legend(loc='center left', bbox_to_anchor=(-0.45, 0.9),
               frameon=True, fancybox=True, shadow=True, markerscale=2)
    plt.ylabel(measure)

    fig = plt.gca()

    if measure in ['snum', 'spcnt', 'lspcnt',
                   'jnum', 'jobp', 'fbff', 'cat_order']:
        plt.gca().invert_yaxis()
        if measure in ['spcnt', 'lspcnt']:
            ax1.set_yticks(np.arange(1, -.05, -.05))
            ax1.yaxis.set_major_formatter(formatter)
            plt.ylim(1, 0)
        else:
            ax1.set_yticks(np.arange(0, max_count, 1000))
            plt.ylim(max_count, 0)
        if measure in ['cat_order']:
            ax2 = plt.gca().twinx()

            axis2_lbl_locs = []
            axis2_lbls = []
            for i in np.arange(1, job_ticks.size):
                axis2_lbl_locs.append(round((job_ticks[i - 1] +
                                             job_ticks[i]) / 2))
                axis2_lbls.append(job_strs[i - 1])

            counter = 0
            for i in np.arange(1, len(axis2_lbl_locs)):
                this_diff = axis2_lbl_locs[i] - axis2_lbl_locs[i - 1]
                # space labels if overlapping...
                if this_diff < 220:
                    counter += 1
                    if counter == 1:
                        axis2_lbl_locs[i - 1] = axis2_lbl_locs[i] - 220
                    else:
                        axis2_lbl_locs[i] = axis2_lbl_locs[i - 1] + 220

            ax2.set_yticks(axis2_lbl_locs)
            ax2.set_yticklabels(axis2_lbls)

            for level in job_ticks:
                ax1.axhline(y=level, c='.8', ls='-', alpha=.8, lw=.5, zorder=0)
            plt.gca().invert_yaxis()

            for i in np.arange(1, job_ticks.size):
                ax2.axhspan(job_ticks[i - 1], job_ticks[i],
                            facecolor=span_colors[i - 1], alpha=.15)
            ax1.grid(ls='dashed', lw=.5)

    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:
        plt.yticks(np.arange(0, job_levels + 2, 1))
        plt.ylim(job_levels + 2, 0.5)

        yticks = fig.get_yticks().tolist()

        for i in np.arange(1, len(yticks)):
            yticks[i] = job_strs[i - 1]
        plt.axhspan(job_levels + 1, job_levels + 2, facecolor='.8', alpha=0.2)
        fig.set_yticklabels(yticks)
        plt.axhline(y=job_levels + 1, c='.8', ls='-', alpha=.8, lw=3)

    if xax in ['snum']:
        plt.xlim(max_count, 0)
    if xax in ['spcnt', 'lspcnt']:
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xticks(np.arange(0, 1.1, .1))
        plt.xlim(1, 0)
    if xax == 'age':
        plt.xlim(xmax=65)
    if xax in ['ylong']:
        plt.xticks(np.arange(0, 55, 5))
        plt.xlim(-0.5, max(df.ylong) + 1)

    plt.gcf().set_size_inches(width, height)
    plt.show()


def diff_range(ds_list, sa_ds, measure, eg_list, proposals_to_plot,
               formatter, gb_period, year_clip=2042,
               show_range=False, show_mean=True, normalize_y=False,
               ysize=6, xsize=6):

    '''Plot a range of differential attributes or a differential
    average over time.  Individual employee groups and proposals may
    be selected.  Each chart indicates the results for one group with
    color bands or average lines indicating the results for that group
    under different proposals.  This is different than the usual method
    of different groups being plotted on the same chart.
    '''

    eg_colors = {1: 'k', 2: 'b', 3: 'r'}
    clrs = []
    for prop in proposals_to_plot:
        clrs.append(eg_colors[prop])
    cmap = colors.ListedColormap(clrs)
    cols = ['date']
    compare_list = []

    sa_ds['eg_order'] = sa_ds.groupby('eg').cumcount()
    sa_ds.sort_values(['eg', 'eg_order'], inplace=True)
    sa_ds = sa_ds[sa_ds.date.dt.year <= year_clip][
        ['eg', 'date', measure]].copy()

    i = 0
    col_list = []
    for ds in ds_list:
        ds['eg_order'] = ds.groupby('eg').cumcount()
        ds.sort_values(['eg', 'eg_order'], inplace=True)
        ds_list[i] = ds[ds.date.dt.year <= year_clip][
            ['eg', 'date', measure]].copy()
        ds_list[i].rename(columns={measure: measure + str(i + 1)},
                          inplace=True)
        col_list.append(measure + str(i + 1))
        i += 1

    i = 0
    for ds in ds_list:
        col = col_list[i]
        sa_ds[col] = ds[col]
        sa_ds[col + '_'] = sa_ds[col] - sa_ds[measure]
        i += 1

    i = 0
    for proposal in proposals_to_plot:
        compare_list.append(measure + str(proposal) + '_')
        i += 1
    cols.extend(compare_list)

    for eg in eg_list:
        if show_range:
            with sns.axes_style('white'):
                ax1 = sa_ds[sa_ds.eg == eg][cols].set_index('date') \
                    .plot(cmap=cmap, alpha=.22)
                plt.grid(lw=1, ls='--', c='grey', alpha=.25)
                if show_mean:
                    ax1.legend_ = None
                    plt.draw()
        if show_mean:
            if show_range:
                sa_ds[sa_ds.eg == eg][cols].set_index('date') \
                    .resample('Q').mean().plot(cmap=cmap, ax=ax1)
            else:
                with sns.axes_style('darkgrid'):
                    sa_ds[sa_ds.eg == eg][cols].set_index('date') \
                        .resample('Q').mean().plot(cmap=cmap)

        if measure in ['spcnt', 'lspcnt', 'jobp', 'jnum']:
            plt.gca().invert_yaxis()

        plt.title('Employee Group ' + str(eg) + ' ' +
                  measure + ' differential')
        plt.axhline(c='m', lw=2, ls='--')
        plt.gcf().set_size_inches(xsize, ysize)
        if measure in ['spcnt', 'lspcnt']:
            plt.gca().yaxis.set_major_formatter(formatter)
            if normalize_y:
                plt.ylim(.5, -.5)
            plt.yticks = np.arange(.5, -.55, .05)
        plt.show()
