# -*- coding: utf-8 -*-

'''built-in plotting functions
'''
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from ipywidgets import interactive, Button, widgets
from IPython.display import display, Javascript
import math
from os import system, path, remove
from collections import OrderedDict as od

import pandas as pd
from pandas.tools.plotting import parallel_coordinates
import numpy as np
import config as cf
import functions as f


# TO_PERCENT (matplotlib percentage axis)
def to_percent(y, position):
    '''matplotlib axis as a percentage...
    Ignores the passed in position variable.
    This has the effect of scaling the default
    tick locations.
    '''
    s = str(np.round(100 * y, 0).astype(int))

    # The percent symbol needs to be escaped in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


pct_format = FuncFormatter(to_percent)


def quartile_years_in_position(dfc, dfb, job_levels,
                               num_bins, job_str_list, eg_dict, color_list,
                               style='bar', plot_differential=True,
                               ds_dict=None,
                               attr1=None, oper1='>=', val1=0,
                               attr2=None, oper2='>=', val2=0,
                               attr3=None, oper3='>=', val3=0,
                               suptitle_fontsize=14, title_fontsize=12,
                               fur_color='#404040',
                               custom_color=False, cm_name='Dark2', start=0.0,
                               stop=1.0, flip_x=False, flip_y=False,
                               rotate=False, gain_loss_bg=False, bg_alpha=.05,
                               normalize_yr_scale=False, year_clip=30,
                               xsize=12, ysize=12):
    '''stacked bar or area chart presenting the time spent in the various
    job levels for quartiles of a selected employee group.

    inputs
        dfc (string or dataframe variable)
            text name of proposal (comparison) dataset to explore (ds_dict key)
            or dataframe
        dfb (string or dataframe variable)
            text name of baseline dataset to explore (ds_dict key)
            or dataframe
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
        eg_dict
            dictionary used to convert employee group numbers to text,
            used with chart title text display
        color_list
            a list of color codes which control the job level color display
        style
            option to select 'area' or 'bar' to determine the type
            of chart output. default is 'bar'.
        plot_differential
            if True, plot the difference between dfc and dfb values
        ds_dict
            variable assigned to the output of the load_datasets function.
            This keyword variable must be set if string dictionary keys are
            used as inputs for the dfc and/or dfb inputs.
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
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

    dsc, df_labelc = determine_dataset(dfc, ds_dict, return_label=True)
    dsb = determine_dataset(dfb, ds_dict, return_label=False)

    d_filtc, t_string = filter_ds(dsc,
                                  attr1=attr1, oper1=oper1, val1=val1,
                                  attr2=attr2, oper2=oper2, val2=val2,
                                  attr3=attr3, oper3=oper3, val3=val3)

    d_filtb = filter_ds(dsb,
                        attr1=attr1, oper1=oper1, val1=val1,
                        attr2=attr2, oper2=oper2, val2=val2,
                        attr3=attr3, oper3=oper3, val3=val3,
                        return_title_string=False)

    if 'new_order' in d_filtc.columns:
        ds_sel_cols = d_filtc[['mnum', 'eg', 'jnum', 'empkey',
                               'new_order', 'doh', 'retdate']].copy()
        if plot_differential:
            d_filtb['new_order'] = d_filtb['idx']
            sa_sel_cols = d_filtb[['mnum', 'eg', 'jnum', 'empkey',
                                  'new_order', 'doh', 'retdate']].copy()

    else:
        d_filtc['new_order'] = d_filtc['idx']
        ds_sel_cols = d_filtc[['mnum', 'eg', 'jnum', 'empkey',
                               'new_order', 'doh', 'retdate']].copy()
        plot_differential = False

    # add color for furloughed employees...
    color_list.append(fur_color)

    mnum0 = ds_sel_cols[ds_sel_cols.mnum == 0][[]]
    mnum0['order'] = np.arange(len(mnum0)) + 1

    egs = sorted(list(set(ds_sel_cols.eg)))

    legend_font_size = np.clip(int(ysize * .8), 12, 18)
    tick_fontsize = (np.clip(int(ysize * .45), 9, 14))
    label_size = (np.clip(int(ysize * .8), 14, 16))

    num_rows = len(egs)
    num_cols = 2 if plot_differential else 1

    fig, ax = plt.subplots(num_rows, num_cols)
    plot_num = 1

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

        if custom_color:
            num_of_colors = cf.num_of_job_levels + 1
            cm_subsection = np.linspace(start, stop, num_of_colors)
            colormap = eval('cm.' + cm_name)
            color_list = [colormap(x) for x in cm_subsection]

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

        with sns.axes_style('darkgrid'):

            ax = plt.subplot(num_rows, num_cols, plot_num)

            if style == 'area':
                quantile_yrs.plot(kind='area',
                                  stacked=True, color=colors, ax=ax)
            if style == 'bar':
                if rotate:
                    kind = 'barh'
                else:
                    kind = 'bar'
                quantile_yrs.plot(kind=kind, width=1,
                                  stacked=True, color=colors, ax=ax)

            if normalize_yr_scale:
                if rotate:
                    plt.xlim(0, year_clip)
                else:
                    plt.ylim(ymax=year_clip)

            if style == 'bar':

                if not flip_y:
                    ax.invert_yaxis()

                if rotate:
                    plt.xlabel('years', fontsize=label_size)
                    plt.ylabel('quartiles', fontsize=label_size)
                else:
                    plt.ylabel('years', fontsize=label_size)
                    plt.xlabel('quartiles', fontsize=label_size)
                    plt.xticks(rotation='horizontal')

            if flip_x:
                ax.invert_xaxis()

            ax.set_title('group ' + str(eg), fontsize=label_size)
            ax.legend_.remove()

            plt.tick_params(axis='y', labelsize=tick_fontsize)
            plot_num += 1

            if plot_differential and style == 'bar':

                ax = plt.subplot(num_rows, num_cols, plot_num)
                diff = quantile_yrs - sa_quantile_yrs

                if style == 'area':
                    diff.plot(kind='area',
                              stacked=True, color=colors, ax=ax)
                if style == 'bar':
                    if rotate:
                        kind = 'barh'
                    else:
                        kind = 'bar'
                    diff.plot(kind=kind, width=1,
                              stacked=True, color=colors, ax=ax)

                if rotate:
                    plt.xlabel('years', fontsize=label_size)
                    plt.ylabel('quartiles', fontsize=label_size)
                    if normalize_yr_scale:
                        plt.xlim(year_clip / -3, year_clip / 3)
                    if not flip_y:
                        ax.invert_yaxis()
                    x_min, x_max = plt.xlim()
                    if gain_loss_bg:
                        plt.axvspan(0, x_max, facecolor='g', alpha=bg_alpha)
                        plt.axvspan(0, x_min, facecolor='r', alpha=bg_alpha)
                else:
                    plt.ylabel('years', fontsize=label_size)
                    plt.xlabel('quartiles', fontsize=label_size)
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

                ax.set_title('group ' + str(eg), fontsize=label_size)
                plt.tick_params(axis='y', labelsize=tick_fontsize)
                ax.legend_.remove()
                plot_num += 1

    try:
        if t_string:
            plt.suptitle(df_labelc + ', ' + t_string,
                         fontsize=suptitle_fontsize, y=1.01)
        else:
            plt.suptitle(df_labelc,
                         fontsize=suptitle_fontsize, y=1.01)
    except:
        if t_string:
            plt.suptitle('proposal' + ', ' + t_string,
                         fontsize=suptitle_fontsize, y=1.01)
        else:
            plt.suptitle('proposal',
                         fontsize=suptitle_fontsize, y=1.01)

    if not plot_differential:
        xsize = xsize * .5
    fig.set_size_inches(xsize, ysize)
    plt.tight_layout()

    if gain_loss_bg:
            legend_labels = ['Loss', 'Gain']
            legend_colors = ['r', 'g']
    else:
        legend_labels = []
        legend_colors = []

    for jnum in np.unique(d_filtb.jnum):
        legend_labels.append(job_str_list[jnum - 1])
        legend_colors.append(color_list[jnum - 1])

    recs = []
    if gain_loss_bg:
        for i in np.arange(len(legend_colors)):
            if i <= 1:
                patch_alpha = .2
            else:
                patch_alpha = 1
            recs.append(mpatches.Rectangle((0, 0), 1, 1,
                                           fc=legend_colors[i],
                                           alpha=patch_alpha))
    else:
        for i in np.arange(len(legend_colors)):
            recs.append(mpatches.Rectangle((0, 0), 1, 1,
                                           fc=legend_colors[i],
                                           alpha=1))

    fig.legend(recs, (legend_labels),
               loc='center left',
               bbox_to_anchor=(1.01, 0.5),
               fontsize=legend_font_size)
    plt.show()


def age_vs_spcnt(df, eg_list, mnum, color_list,
                 eg_dict, ds_dict=None,
                 attr1=None, oper1='>=', val1=0,
                 attr2=None, oper2='>=', val2=0,
                 attr3=None, oper3='>=', val3=0,
                 suptitle_fontsize=14, title_fontsize=12,
                 legend_fontsize=12,
                 xsize=10, ysize=8,
                 chart_example=False):
    '''scatter plot with age on x axis and list percentage on y axis.
    note: input df may be prefiltered to plot focus attributes, i.e.
    filter to include only employees at a certain job level, hired
    between certain dates, with a particular age range, etc.

    inputs
        df (string or dataframe)
            text name of input proposal dataset, also will accept any dataframe
            variable (if a sliced dataframe subset is desired, for example)
            Example:  input can be 'proposal1' (if that proposal exists, of
            course, or could be df[df.age > 50])
        eg_list (list)
            list of employee groups to include
            example: [1, 2]
        mnum (int)
            month number to study from dataset
        color_list (list)
            color codes for plotting each employee group
        eg_dict (dict)
            dictionary, numerical eg code to string description
        ds_dict (dict)
            variable assigned to the output of the load_datasets function,
            reqired when string dictionary key is used as df input
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
        suptitle_fontsize
            text size of chart super title
        title_fontsize
            text size of chart title
        legend_fontsize
            text size of chart legend
        xsize, ysize (integer or float)
            plot size in inches
        chart_example
            if True, remove case-specific descriptions from chart
    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)

    d_filt, t_string = filter_ds(ds,
                                 attr1=attr1, oper1=oper1, val1=val1,
                                 attr2=attr2, oper2=oper2, val2=val2,
                                 attr3=attr3, oper3=oper3, val3=val3)

    d_age_pcnt = d_filt[d_filt.mnum == mnum][
        ['age', 'mnum', 'spcnt', 'eg']].copy()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

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
    plt.xlim(25, cf.ret_age)
    plt.tight_layout()
    ax.yaxis.set_major_formatter(pct_format)
    ax.set_yticks(np.arange(0, 1.05, .05))

    if chart_example:
        plt.suptitle('Proposal 1' +
                     ' - age vs seniority percentage' +
                     ', month ' +
                     str(mnum), fontsize=suptitle_fontsize,
                     y=1.02)
    else:
        try:
            plt.suptitle(df_label +
                         ' - age vs seniority percentage' +
                         ', month ' +
                         str(mnum), fontsize=suptitle_fontsize,
                         y=1.02)
        except:
            plt.suptitle('proposal' +
                         ' - age vs seniority percentage' +
                         ', month ' +
                         str(mnum), fontsize=suptitle_fontsize,
                         y=1.02)
    plt.title(t_string, fontsize=title_fontsize)
    plt.legend(loc=2, markerscale=1.5, fontsize=legend_fontsize)
    fig = plt.gcf()
    fig.set_size_inches(xsize, ysize)
    plt.ylabel('spcnt')
    plt.xlabel('age')
    plt.show()


def multiline_plot_by_emp(df, measure, xax, emp_list, job_levels,
                          color_list, job_str_list,
                          ds_dict=None,
                          legend_fontsize=14,
                          chart_example=False):
    '''select example individual employees and plot career measure
    from selected dataset attribute, i.e. list percentage, career
    earnings, job level, etc.

    inputs
        df
            dataset to examine, may be a dataframe or a string key with the
            ds_dict dictionary object
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
        ds_dict
            output of the load_datasets function, dictionary.  This keyword
            argument must be set if a string key is used as the df input.
        legend_fontsize
            text size of chart legend
        chart_example
            if True, remove case-specific descriptions from chart
    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)

    frame = ds.copy()
    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:
        frame = frame[frame.jnum <= cf.num_of_job_levels]
    if measure in ['mpay']:
        frame = frame[frame.age < cf.ret_age]

    i = 0

    for emp in emp_list:
        if chart_example:
            ax = frame[frame.empkey == emp] \
                .set_index(xax)[measure].plot(label='Employee ' + str(i + 1))
            i += 1
        else:
            if len(emp_list) == 3:
                ax = frame[frame.empkey == emp] \
                    .set_index(xax)[measure].plot(color=color_list[i],
                                                  label=emp)
            else:
                ax = frame[frame.empkey == emp] \
                    .set_index(xax)[measure].plot(label=emp)
            i += 1

    if measure in ['snum', 'spcnt', 'lspcnt', 'jnum',
                   'lnum', 'jobp', 'fbff', 'cat_order']:
        ax.invert_yaxis()
    if measure in ['lspcnt', 'spcnt']:
        ax.yaxis.set_major_formatter(pct_format)
        plt.yticks = (np.arange(0, 1.05, .05))

    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:

        ax.set_yticks(np.arange(0, job_levels + 2, 1))
        ytick_labels = ax.get_yticks().tolist()

        for i in np.arange(1, len(ytick_labels)):
            ytick_labels[i] = job_str_list[i - 1]
        plt.axhspan(job_levels + 1, job_levels + 2, facecolor='.8', alpha=0.9)
        ax.set_yticklabels(ytick_labels)
        plt.axhline(y=job_levels + 1, c='.8', ls='-', alpha=.8, lw=3)
        plt.ylim(job_levels + 1.5, 0.5)

    if xax in ['spcnt', 'lspcnt']:
        ax.xaxis.set_major_formatter(pct_format)
        plt.xticks(np.arange(0, 1.1, .1))
        plt.xlim(1, 0)

    if chart_example:
        plt.title(measure + ' - ' + 'proposal 1', y=1.02)
    else:
        try:
            plt.title(measure + ' - ' + df_label, y=1.02)
        except:
            plt.title(measure + ' - ' + 'proposal', y=1.02)
    plt.ylabel(measure)
    plt.legend(loc=4, markerscale=1.5, fontsize=legend_fontsize)
    plt.show()


def multiline_plot_by_eg(df, measure, xax, eg_list, job_strs,
                         job_levels, colors, ds_dict=None, mnum=0,
                         attr1=None, oper1='>=', val1=0,
                         attr2=None, oper2='>=', val2=0,
                         attr3=None, oper3='>=', val3=0,
                         scatter=False, exclude_fur=False,
                         suptitle_fontsize=14, title_fontsize=12,
                         legend_fontsize=14, xsize=8, ysize=6,
                         full_pcnt_xscale=False, chart_example=False):
    '''plot separate selected employee group data for a specific month.

    chart type may be line or scatter.

    inputs
        df
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        measure
            attribute to plot on y axis
        xax
            x axis attribute
        eg_list
            list of employee groups to plot (integer codes)
        job_strs
            job text labels for y axis when job number measure selected
        job_levels
            number of job levels in model
        colors
            colors for eg plots
        ds_dict
            output of the load_datasets function, dictionary.  This keyword
            argument must be set if a string key is used as the df input.
        mnum
            month number for analysis
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
        scatter
            plot a scatter chart (vs. default line chart)
        exclude_fur
            do not plot furoughed employees
        suptitle_fontsize
            text size of chart super title
        title_fontsize
            text size of chart title
        legend_fontsize
            text size of chart legend
        xsize, ysize (integer or float)
            plot size in inches
        full_pcnt_xscale
            plot x axis percentage from 0 to 100 percent
        chart_example
            remove case-specific text from chart
    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)

    d_filt, t_string = filter_ds(ds,
                                 attr1=attr1, oper1=oper1, val1=val1,
                                 attr2=attr2, oper2=oper2, val2=val2,
                                 attr3=attr3, oper3=oper3, val3=val3,
                                 return_title_string=True)

    frame = d_filt[(d_filt.mnum == mnum)][[xax, measure, 'eg', 'ret_mark']]

    if exclude_fur:
        frame = frame[(frame.jnum >= 1) & (frame.jnum <= job_levels)]

    if measure == 'mpay':
        frame = frame[frame.ret_mark == 0]

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

    if measure in ['snum', 'spcnt', 'jnum', 'jobp', 'fbff',
                   'lspcnt', 'rank_in_job', 'cat_order']:
        ax.invert_yaxis()
    if measure in ['spcnt', 'lspcnt']:
        ax.yaxis.set_major_formatter(pct_format)
        plt.yticks = (np.arange(0, 1.05, .05))
    if xax in ['spcnt', 'lspcnt']:
        ax.xaxis.set_major_formatter(pct_format)
        plt.xticks(np.arange(0, 1.1, .1))

    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:

        ax.set_yticks(np.arange(0, job_levels + 2, 1))
        ytick_labels = list(ax.get_yticks())

        for i in np.arange(1, len(ytick_labels)):
            ytick_labels[i] = job_strs[i - 1]

        plt.axhspan(job_levels + 1, job_levels + 2, facecolor='.8', alpha=0.3)
        ax.set_yticklabels(ytick_labels)
        plt.axhline(y=job_levels + 1, c='.8', ls='-', alpha=.3, lw=3)
        plt.ylim(job_levels + 2, 0.5)

    if measure in ['mpay', 'cpay', 'mlong', 'ylong']:
        plt.ylim(ymin=0)

    if xax in ['new_order', 'cat_order', 'lnum', 'snum']:
        plt.xlim(xmax=0)
        plt.xlabel(xax)

    if xax in ['mlong', 'ylong']:
        plt.xlim(xmin=0)
        plt.xlabel(xax)

    if xax not in ['age', 'mlong', 'ylong']:
        ax.invert_xaxis()

    if xax in ['spcnt', 'lspcnt'] and full_pcnt_xscale:
        plt.xlim(1, 0)

    plt.legend(loc=4, markerscale=1.5, fontsize=legend_fontsize)

    if chart_example:
        prop_text = 'Proposal'
    else:
        prop_text = df_label

    suptitle = measure.upper() + ' ordered by ' + xax + ' - ' + \
        prop_text + ' - Month: ' + str(mnum)

    if t_string:
        plt.suptitle(suptitle, fontsize=suptitle_fontsize, y=1.00)
        plt.title(t_string, fontsize=title_fontsize, y=1.02)
    else:
        plt.title(suptitle, fontsize=title_fontsize)

    if measure == 'ylong':
        plt.ylim(0, 40)
    plt.ylabel(measure)
    plt.xlabel(xax)
    plt.show()


def violinplot_by_eg(df, measure, ds_dict=None,
                     mnum=0, linewidth=1.5, chart_example=False,
                     attr1=None, oper1='>=', val1='0',
                     attr2=None, oper2='>=', val2='0',
                     attr3=None, oper3='>=', val3='0',
                     scale='count', title_fontsize=12):
    '''From the seaborn website:
    Draw a combination of boxplot and kernel density estimate.

    A violin plot plays a similar role as a box and whisker plot.
    It shows the distribution of quantitative data across several
    levels of one (or more) categorical variables such that those
    distributions can be compared. Unlike a box plot, in which all
    of the plot components correspond to actual datapoints, the violin
    plot features a kernel density estimation of the underlying distribution.

    inputs
        df
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        measure
            attribute to plot
        ds_dict (dictionary)
            output from load_datasets function
        mnum
            month number to analyze
        linewidth
            width of line surrounding each violin plot
        chart_example
            remove case-specific text data from chart
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
        scale
            From the seaborn website:
            The method used to scale the width of each violin.
            If 'area', each violin will have the same area.
            If 'count', the width of the violins will be scaled by
            the number of observations in that bin.
            If 'width', each violin will have the same width.
        title_fontsize
            text size of chart title
    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)
    dsm = ds[ds.mnum == mnum]
    dsm = filter_ds(dsm,
                    attr1=attr1, oper1=oper1, val1=val1,
                    attr2=attr2, oper2=oper2, val2=val2,
                    attr3=attr3, oper3=oper3, val3=val3,
                    return_title_string=False).copy()

    title_string = ''

    if attr1:
        title_string = title_string + attr1 + ' ' + oper1 + ' ' + str(val1)
    if attr2:
        title_string = title_string + ', ' + \
            attr2 + ' ' + oper2 + ' ' + str(val2)
    if attr3:
        title_string = title_string + ', ' + \
            attr3 + ' ' + oper3 + ' ' + str(val3)

    if measure == 'age':
        frame = dsm[['eg', measure]].copy()
    else:
        frame = dsm[[measure, 'eg', 'age']].copy()
    frame.reset_index(drop=True, inplace=True)

    if measure == 'mpay':
        frame = frame[frame.age < cf.ret_age]

    sns.violinplot(x=frame.eg, y=frame[measure],
                   cut=0, scale=scale, inner='box',
                   bw=.1, linewidth=linewidth,
                   palette=['gray', '#3399ff', '#ff8000'])
    if chart_example:
        plt.suptitle('Proposal 3' + ' - ' +
                     measure.upper() + ' Distribution - Month ' +
                     str(mnum))
    else:
        try:
            plt.suptitle(df_label + ' - ' +
                         measure.upper() + ' Distribution - Month ' +
                         str(mnum))
        except:
            plt.suptitle('proposal' + ' - ' +
                         measure.upper() + ' Distribution - Month ' +
                         str(mnum))

    plt.title(title_string, fontsize=title_fontsize)
    fig = plt.gca()
    if measure == 'age':
        plt.ylim(25, 70)
    if measure in ['snum', 'spcnt', 'lspcnt', 'jnum', 'jobp', 'cat_order']:
        fig.invert_yaxis()
        if measure in ['spcnt', 'lspcnt']:
            fig.yaxis.set_major_formatter(pct_format)
            plt.gca().set_yticks(np.arange(0, 1.05, .05))
            plt.ylim(1.04, -.04)
    plt.show()


def age_kde_dist(df, color_list, eg_dict, ds_dict=None,
                 mnum=0, title_fontsize=14, chart_example=False):
    '''From the seaborn website:
    Fit and plot a univariate or bivariate kernel density estimate.

    inputs
        df
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        color_list
            list of colors for employee group plots
        eg_dict
            eg to string dict for plot labels
        ds_dict (dictionary)
            output from load_datasets function
        mnum
            month number to analyze
        title_fontsize
            text size of chart title
        chart_example
            remove case-specific text from chart
    '''
    ds = determine_dataset(df, ds_dict)

    frame = ds[ds.mnum == mnum]
    eg_set = pd.unique(frame.eg)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(sharex=True, figsize=(10, 8))

    for x in eg_set:
        try:
            color = color_list[x - 1]

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

    ax.set_xlim(25, cf.ret_age)
    fig.set_size_inches(10, 8)
    plt.title('Age Distribution Comparison - Month ' + str(mnum), y=1.02,
              fontsize=title_fontsize)
    plt.show()


def eg_diff_boxplot(df_list, dfb, eg_list,
                    measure='spcnt',
                    comparison='baseline', ds_dict=None,
                    attr1=None, oper1='>=', val1=0,
                    attr2=None, oper2='>=', val2=0,
                    attr3=None, oper3='>=', val3=0,
                    suptitle_fontsize=14, title_fontsize=12,
                    tick_size=11, label_size=12, year_clip=2035,
                    exclude_fur=False,
                    use_eg_colors=False,
                    width=.8, chart_style='dark',
                    notch=True, linewidth=1.0,
                    job_diff_clip=cf.num_of_job_levels + 1,
                    xsize=10, ysize=6, chart_example=False):
    '''create a differential box plot chart comparing a selected measure from
    computed integrated dataset(s) vs. a baseline (likely standalone) dataset
    or with other integrated datasets.

    inputs
        df_list (list)
            list of datasets to compare, may be ds_dict (output of
            load_datasets function) string keys or dataframe variable(s) or
            mixture of each
        dfb (string or variable)
            baseline dataset, accepts same input types as df_list above
        eg_list (list)
            list of integers for employee groups to be included in analysis
            example: [1, 2, 3]
        measure (string)
            differential data to compare
        comparison (string)
            if 'p2p' (proposal to proposal), will compare proposals within the
            df_list to each other, otherwise will compare proposals to the
            baseline dataset (dfb)
        ds_dict (dictionary)
            output from load_datasets function
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
        suptitle_fontsize
            text size of chart super title
        title_fontsize
            text size of chart title
        tick_size
            text size of x and y tick labels
        label_size
            text size of x and y descriptive labels
        year_clip (integer)
            only present results through this year
        exclude_fur (boolean)
            remove all employees from analysis who are furloughed within the
            data model at any time
        use_eg_colors (boolean)
            use case-specific employee group colors vs. default colors
        width (float)
            plotting width of boxplot or grouped boxplots for each year.
            a width of 1 leaves no gap between groups
        chart_style (string)
            chart styling (string), any valid seaborn chart style
        notch (boolean)
            If True, show boxplots with a notch at median point vs. only a line
        job_diff_clip (integer)
            if measure is jnum or jobp, limit y axis range to +/- this value
        xsize, ysize (integer or float)
            plot size in inches
        chart_example
            if True, remove case-specific descriptions from chart
    '''

    label_dict = {}
    i = 0
    for df in df_list:
        ds, df_label = determine_dataset(df, ds_dict,
                                         return_label=True)
        label_dict[i + 1] = df_label

        df_list[i] = filter_ds(ds,
                               attr1=attr1, oper1=oper1, val1=val1,
                               attr2=attr2, oper2=oper2, val2=val2,
                               attr3=attr3, oper3=oper3, val3=val3,
                               return_title_string=False)
        i += 1

    df_base, dfb_label = determine_dataset(dfb, ds_dict, return_label=True)

    dfb_filt, tb_string = filter_ds(df_base,
                                    attr1=attr1, oper1=oper1, val1=val1,
                                    attr2=attr2, oper2=oper2, val2=val2,
                                    attr3=attr3, oper3=oper3, val3=val3)

    chart_pad = {'spcnt': .03,
                 'lspcnt': .03,
                 'mpay': 1,
                 'cpay': 10}

    if use_eg_colors:
        colors = cf.eg_colors
    else:
        colors = ['grey', '#66b3ff', '#ff884d', '#00ff99']

    # set boxplot color to match employee group(s) color
    color_index = np.array(eg_list) - 1
    color_arr = np.array(colors)
    colors = list(color_arr[color_index])

    ds_dict = od()

    i = 1
    for ds in df_list:
        # filter to only include desired employee groups
        ds = ds[ds['eg'].isin(eg_list)]
        # create ordered dictionary containing input dataframes with
        # columns to create 'key' and the measure column for comparisons
        if exclude_fur:
            idx = np.array(ds.index)
            fur = np.array(ds.fur)
            furs = np.where(fur == 1)[0]
            ds_nofur = ds[~np.in1d(ds.index, pd.unique(idx[furs]))]
            ds_dict[str(i)] = ds_nofur[['empkey', 'mnum', measure]].copy()
        else:
            ds_dict[str(i)] = ds[['empkey', 'mnum', measure]].copy()
        i += 1

    dict_nums = ds_dict.keys()
    yval_list = []
    # make list of comparison columns
    if comparison == 'p2p':
        for num1 in dict_nums:
            for num2 in dict_nums:
                if num1 != num2:
                    yval_list.append(num2 + '_' + num1)
    else:
        for num1 in dict_nums:
            yval_list.append('s_' + num1)

    i = 1
    for df in ds_dict.values():
        # rename measure columuns to be unique in each dataframe
        # and make a unique key column in each dataframe for joining
        df.rename(columns={measure: measure + '_' + str(i)}, inplace=True)
        df['key'] = (df.empkey * 1000) + df.mnum
        df.drop(['mnum', 'empkey'], inplace=True, axis=1)
        df.set_index('key', inplace=True)
        i += 1

    # repeat for standalone dataframe
    baseline = dfb_filt[dfb_filt['eg'].isin(eg_list)][
        ['empkey', 'mnum', 'date', 'eg', measure]].copy()
    baseline.rename(columns={measure: measure + '_s'}, inplace=True)
    baseline['key'] = (baseline.empkey * 1000) + baseline.mnum
    baseline.drop(['mnum', 'empkey'], inplace=True, axis=1)
    baseline.set_index('key', inplace=True)

    # join dataframes (auto-aligned by index)
    for df in ds_dict.values():
        baseline = baseline.join(df)

    # perform the differential calculation
    if measure in ['mpay', 'cpay']:

        for num1 in dict_nums:
            if comparison == 'p2p':

                for num2 in dict_nums:
                    if num2 != num1:
                        baseline[num1 + '_' + num2] = \
                            baseline[measure + '_' + num2] - \
                            baseline[measure + '_' + num1]

            else:
                baseline['s' + '_' + num1] = \
                    baseline[measure + '_' + num1] - baseline[measure + '_s']
    else:

        for num1 in dict_nums:
            if comparison == 'p2p':

                for num2 in dict_nums:
                        if num2 != num1:
                            baseline[num1 + '_' + num2] = \
                                baseline[measure + '_' + num1] -\
                                baseline[measure + '_' + num2]
            else:
                baseline['s' + '_' + num1] = \
                    baseline[measure + '_s'] - baseline[measure + '_' + num1]

    for num1 in dict_nums:
        baseline.drop(measure + '_' + num1, inplace=True, axis=1)

    baseline.drop(measure + '_s', inplace=True, axis=1)

    # make a 'date' column containing date year
    baseline.set_index('date', drop=True, inplace=True)
    baseline['date'] = baseline.index.year
    y_clip = baseline[baseline.date <= year_clip]

    # create a dictionary containing plot titles
    yval_dict = od()

    # proposal to proposal comparison...
    if comparison == 'p2p':

        for num1 in dict_nums:
            if label_dict[int(num1)] == 'Proposal':
                p1_label = 'df_list item ' + num1
            else:
                p1_label = label_dict[int(num1)]
            for num2 in dict_nums:
                if label_dict[int(num2)] == 'Proposal':
                    p2_label = 'df_list item ' + num2
                else:
                    p2_label = label_dict[int(num2)]

                yval_dict[num1 + '_' + num2] = p2_label + \
                    ' vs. ' + p1_label + ' ' + measure.upper()
    # baseline comparison...
    else:
        for num1 in dict_nums:
            if label_dict[int(num1)] == 'Proposal':
                p_label = 'df_list item ' + num1
            else:
                p_label = label_dict[int(num1)]
            yval_dict['s_' + num1] = p_label + ' vs. ' + \
                dfb_label + ' ' + measure.upper()

    for yval in yval_list:
        # determine y axis chart limits
        try:
            pad = chart_pad[measure]
            ylimit = max(abs(max(y_clip[yval])), abs(min(y_clip[yval]))) + pad
        except:
            ylimit = max(abs(max(y_clip[yval])), abs(min(y_clip[yval])))
        # create seaborn boxplot
        with sns.axes_style(chart_style):
            sns.boxplot(x='date', y=yval,
                        hue='eg', data=y_clip,
                        palette=colors, width=width,
                        notch=notch,
                        linewidth=linewidth, fliersize=1.0)
        fig = plt.gcf()
        ax = plt.gca()
        # set chart size
        fig.set_size_inches(xsize, ysize)
        # add zero line
        plt.axhline(y=0, c='r', zorder=.9, alpha=.35, lw=2)
        plt.ylim(-ylimit, ylimit)
        if measure in ['spcnt', 'lspcnt']:
            # format percentage y axis scale
            ax.yaxis.set_major_formatter(pct_format)
        if measure in ['jnum', 'jobp']:
            # if job level measure, set scaling and limit y range
            ax.set_yticks(np.arange(int(-ylimit - 1), int(ylimit + 2)))
            plt.ylim(max(-job_diff_clip, int(-ylimit - 1)),
                     min(job_diff_clip, int(ylimit + 1)))
        if chart_example:
            plt.suptitle('PROPOSAL vs. standalone ' + measure.upper())
        else:
            plt.suptitle(yval_dict[yval], fontsize=suptitle_fontsize)
        plt.title(tb_string, fontsize=title_fontsize)
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
        plt.tick_params(axis='both', which='major', labelsize=tick_size)
        plt.ylabel('differential', fontsize=label_size)
        ax.xaxis.label.set_size(label_size)
        plt.show()


def eg_boxplot(df_list, eg_list,
               measure='spcnt',
               ds_dict=None,
               attr1=None, oper1='>=', val1=0,
               attr2=None, oper2='>=', val2=0,
               attr3=None, oper3='>=', val3=0,
               year_clip=2035,
               exclude_fur=False,
               use_eg_colors=False,
               saturation=.8,
               chart_style='dark',
               width=.7,
               notch=True,
               show_whiskers=True,
               show_xgrid=True,
               show_ygrid=True,
               grid_alpha=.4,
               grid_linestyle='solid',
               job_clip=cf.num_of_job_levels + 1,
               whisker=1.5, fliersize=1.0,
               linewidth=.5,
               suptitle_fontsize=14, title_fontsize=12,
               tick_size=11, label_size=12,
               xsize=10, ysize=6):
    '''create a box plot chart displaying actual attribute values
    (vs. differential values) from a selected dataset(s) for selected
    employee group(s).

    df_list (list)
        list of datasets to compare, may be ds_dict (output of load_datasets
        function) string keys or dataframe variable(s) or mixture of each
    eg_list
        list of integers for employee groups to be included in analysis
        example: [1, 2, 3]
    measure
        attribute for analysis
    ds_dict (dictionary)
        output from load_datasets function
    attr(n)
        filter attribute or dataset column as string
    oper(n)
        operator (i.e. <, >, ==, etc.) for attr1 as string
    val(n)
        attr1 limiting value (combined with oper1) as string
    year_clip
        only present results through this year
    exclude_fur
        remove all employees from analysis who are furloughed within the
        data model at any time (boolean)
    use_eg_colors
        use case-specific employee group colors vs. default colors (boolean)
    chart_style
        chart styling (string), any valid seaborn chart style
    width
        plotting width of boxplot or grouped boxplots for each year.
        a width of 1 leaves no gap between groups
    notch
        If True, show boxplots with a notch at median point
    show_xgrid
        include vertical grid lines on chart
    show_ygrid
        include horizontal grid lines on chart
    grid_alpha
        opacity value for grid lines
    grid_linestyle
        examples: 'solid', 'dotted', 'dashed'
    job_clip
        if measure is jnum or jobp, limit max y axis range to this value
    suptitle_fontsize
        text size of chart super title
    title_fontsize
        text size of chart title
    tick_size
        text size of x and y tick labels
    label_size
        text size of x and y descriptive labels
    xsize, ysize
        plot size in inches
    '''
    label_dict = {}
    filt_title = ''
    i = 0
    for df in df_list:
        ds, df_label = determine_dataset(df, ds_dict,
                                         return_label=True)
        label_dict[i + 1] = df_label

        df_list[i], filt_title = filter_ds(ds,
                                           attr1=attr1, oper1=oper1, val1=val1,
                                           attr2=attr2, oper2=oper2, val2=val2,
                                           attr3=attr3, oper3=oper3, val3=val3,
                                           return_title_string=True)
        i += 1

    chart_pad = {'spcnt': .03,
                 'lspcnt': .03,
                 'mpay': 1,
                 'cpay': 10,
                 'cat_order': 50,
                 'snum': 50,
                 'lnum': 50,
                 'jobp': .5}

    if use_eg_colors:
        colors = cf.eg_colors
    else:
        colors = ['grey', '#66b3ff', '#ff884d', '#00ff99']

    # set boxplot color to match employee group(s) color
    color_index = np.array(eg_list) - 1
    color_arr = np.array(colors)
    colors = list(color_arr[color_index])

    temp_frame = df_list[0][['empkey', 'mnum', 'eg', 'date']].copy()
    temp_frame['year'] = temp_frame.date.dt.year
    temp_frame['key'] = (temp_frame.empkey * 1000) + temp_frame.mnum

    data = {'eg': np.array(temp_frame.eg), 'year': np.array(temp_frame.year)}
    frame = pd.DataFrame(data=data, index=temp_frame.key)
    # filter frame to only include desired employee groups
    frame = frame[frame['eg'].isin(eg_list)]

    yval_list = []
    title_dict = od()

    i = 1
    for ds in df_list:
        this_measure_col = measure + '_' + str(i)
        if exclude_fur:
            ds = ds[ds['eg'].isin(eg_list)][['empkey',
                                             'mnum', 'fur', measure]].copy()
            idx = np.array(ds.index)
            fur = np.array(ds.fur)
            furs = np.where(fur == 1)[0]
            ds = ds[~np.in1d(ds.index, pd.unique(idx[furs]))]

        # filter each ds to only include desired employee groups
        ds = ds[ds['eg'].isin(eg_list)][['empkey', 'mnum', measure]].copy()
        ds['key'] = (ds.empkey * 1000) + ds.mnum

        ds.set_index('key', drop=True, inplace=True)
        frame[this_measure_col] = ds[measure]
        yval_list.append(this_measure_col)
        if label_dict[i] == 'Proposal':
            p_label = 'df_list item ' + str(i)
        else:
            p_label = label_dict[i]
        title_dict[this_measure_col] = p_label + ' ' + measure

        i += 1

    y_clip = frame[frame.year <= year_clip]
    if not show_whiskers:
        whisker = 0
        fliersize = 0

    # make a chart for each selected column
    for yval in yval_list:
        # determine y axis chart limits
        try:
            pad = chart_pad[measure]
            ylimit = max(abs(max(y_clip[yval])), abs(min(y_clip[yval]))) + pad
        except:
            ylimit = max(abs(max(y_clip[yval])), abs(min(y_clip[yval])))
        # create seaborn boxplot
        with sns.axes_style(chart_style):
            sns.boxplot(x='year', y=yval,
                        hue='eg', data=y_clip,
                        palette=colors,
                        notch=notch, whis=whisker,
                        saturation=saturation, width=width,
                        linewidth=linewidth, fliersize=fliersize)
        fig = plt.gcf()
        ax = plt.gca()
        # set chart size
        fig.set_size_inches(xsize, ysize)

        plt.ylim(0, ylimit)
        if measure in ['spcnt', 'lspcnt']:
            # format percentage y axis scale
            ax.yaxis.set_major_formatter(pct_format)
            plt.ylim(ylimit, 0)
        if measure in ['jnum', 'jobp']:
            # if job level measure, set scaling and limit y range
            plt.gca().set_yticks(np.arange(0, int(ylimit + 1)))
            plt.ylim(min(job_clip + 1.5, int(ylimit + 2)), 0.5)
        if measure in ['cat_order', 'snum', 'lnum']:
            plt.gca().invert_yaxis()

        if filt_title:
            plt.suptitle(title_dict[yval], fontsize=suptitle_fontsize)
            plt.title(filt_title, fontsize=title_fontsize)
        else:
            plt.title(title_dict[yval], fontsize=suptitle_fontsize, y=1.01)

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
        plt.ylabel('absolute values', fontsize=label_size)
        plt.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.xaxis.label.set_size(label_size)

        if show_xgrid:
            ax.xaxis.grid(alpha=grid_alpha, ls=grid_linestyle)
        if show_ygrid:
            ax.yaxis.grid(alpha=grid_alpha, ls=grid_linestyle)
        plt.show()


# DISTRIBUTION WITHIN JOB LEVEL (NBNF effect)
def stripplot_distribution_in_category(df, job_levels, mnum, full_time_pcnt,
                                       eg_colors, band_colors, job_strs,
                                       eg_dict, ds_dict=None,
                                       attr1=None, oper1='>=', val1='0',
                                       attr2=None, oper2='>=', val2='0',
                                       attr3=None, oper3='>=', val3='0',
                                       bg_alpha=.12, fur_color='.4',
                                       show_part_time_lvl=True,
                                       size=3, xsize=4, ysize=10,
                                       chart_example=False,
                                       suptitle_fontsize=14, title_fontsize=12,
                                       label_size=12, tick_fontsize=11):
    '''visually display employee group distribution concentration within
    accurately sized job bands for a selected month.

    This chart reveals how evenly or unevenly the employee groups share
    the jobs available within each job category.

    inputs
        df
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        job_levels
            number of job levels in the data model
        mnum
            month number - analyze data from this month
        full_time_pcnt
            percentage of each job level which is full time
        eg_colors
            list of colors for eg plots
        band_colors
            list of colors for background job band colors
        job_strs
            list of job strings for job description labels
        eg_dict
            eg to group string label
        ds_dict (dictionary)
            output from load_datasets function
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
        bg_alpha
            color alpha for background job level color
        fur_color
            color to represent furloughed employees
        show_part_time_lvl
            draw a line within each job band representing the boundry
            between full and part-time jobs
        size
            size of density markers
        chart_example
            remove case-specific text from plot
        suptitle_fontsize
            text size of chart super title
        title_fontsize
            text size of chart title
        label_size
            text size of x and y descriptive labels
        tick_fontsize
            text size of x and y tick labels
    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)
    dsm = ds[ds.mnum == mnum]

    d_filt, t_string = filter_ds(dsm,
                                 attr1=attr1, oper1=oper1, val1=val1,
                                 attr2=attr2, oper2=oper2, val2=val2,
                                 attr3=attr3, oper3=oper3, val3=val3)

    d_filt = d_filt[[]].join(dsm).reindex(dsm.index)
    data = d_filt[['mnum', 'cat_order', 'jnum', 'eg']].copy()

    fur_lvl = job_levels + 1
    if job_levels == 16:
        adjust = [0, 0, 0, 0, 0, 0, -50, 50, 0, 0, -50, 50, 75, 0, 0, 0, 0]

    if job_levels == 8:
        adjust = [0, 0, 0, 0, 0, 0, -50, 50, 0]

    eg_set = pd.unique(data.eg)
    max_eg_plus_one = max(eg_set) + 1

    y_count = len(data)

    cum_job_counts = dsm.jnum.value_counts().sort_index().cumsum()
    lowest_cat = max(cum_job_counts.index)

    cnts = list(cum_job_counts)
    cnts.insert(0, 0)

    axis2_lbl_locs = []
    axis2_lbls = []
    band_colors.append(fur_color)

    with sns.axes_style('white'):
        fig, ax1 = plt.subplots()
        ax1 = sns.stripplot(y='cat_order', x='eg', data=data, jitter=.5,
                            order=np.arange(1, max_eg_plus_one),
                            palette=eg_colors, size=size,
                            linewidth=0, split=True)

        plt.yticks = (np.arange(0, ((len(df) + 1000) % 1000) * 1000, 1000))
        plt.ylim(y_count, 0)
        plt.tick_params(labelsize=tick_fontsize)
        ax1.xaxis.label.set_size(label_size)
        ax1.yaxis.label.set_size(label_size)

    i = 0
    for job_zone in cum_job_counts:
        ax1.axhline(job_zone, c='magenta', ls='-', alpha=1, lw=.8)
        ax1.axhspan(cnts[i], cnts[i + 1],
                    facecolor=band_colors[i],
                    alpha=bg_alpha,)
        if show_part_time_lvl:
            part_time_lvl = (round((cnts[i + 1] - cnts[i]) *
                             full_time_pcnt)) + cnts[i]
            ax1.axhline(part_time_lvl, c='#66ff99', ls='--', alpha=1, lw=1)
        i += 1

    i = 0

    for job_num in cum_job_counts.index:
        axis2_lbl_locs.append(round((cnts[i] + cnts[i + 1]) / 2))
        axis2_lbl_locs[i] += adjust[int(job_num) - 1]
        axis2_lbls.append(job_strs[int(job_num)])
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
    plt.tick_params(labelsize=tick_fontsize)

    plt.gcf().set_size_inches(7, 12)
    plt.suptitle(df_label +
                 ', distribution within job levels, month ' + str(mnum),
                 fontsize=suptitle_fontsize)
    plt.title(t_string, fontsize=title_fontsize, y=1.02)
    # plt.xlabel('test')
    fig.set_size_inches(xsize, ysize)
    plt.show()


def job_level_progression(df, emp_list, through_date,
                          ds_dict=None,
                          job_levels=cf.num_of_job_levels,
                          eg_colors=cf.eg_colors,
                          band_colors=cf.job_colors,
                          job_counts=cf.eg_counts,
                          job_change_lists=cf.j_changes,
                          alpha=.12,
                          max_plots_for_legend=5,
                          xsize=12, ysize=10,
                          chart_example=False):
    '''show employee(s) career progression through job levels regardless of
    actual positioning within integrated seniority list.

    This x axis of this chart represents rank within job category.  There is
    an underlying stacked area chart representing job level bands,
    adjusted to reflect job count changes over time.

    This chart reveals actual career path considering no bump no flush,
    special job assignment rights/restrictions, and furlough/recall events.

    Actual jobs held may not be correlated to jobs normally associated with
    a certain list percentage for many years due to job assignment factors.

    inputs
        df
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        emp_list
            list of empkeys to plot
        through_date
            string representation of y axis date limit, ex. '2025-12-31'
        ds_dict (dictionary)
            output from load_datasets function
        job levels
            number of job levels in model
        eg_colors
            colors to be used for employee line plots corresponding
            to employee group membership
        band_colors
            list of colors to be used for stacked area chart which represent
            job level bands
        job_counts
            list of lists containing job counts for each employee group
        job_change_lists
            list of job changes data (originally from case-specific
            configuration file)
        alpha
            opacity level of background job bands stacked area chart
        max_plots_for_legend (integer)
            if number of plots more than this number, reduce plot linewidth and
            remove legend
        xsize, ysize
            plot size in inches
        chart_example
            set true to remove casse-specific data from chart output
    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)

    through_date = pd.to_datetime(through_date)
    fur_lvl = job_levels + 1
    jobs_dict = cf.jobs_dict

    if cf.enhanced_jobs:
        # use job dictionary from case-specific configuration file
        # for conversion
        eg_counts, j_changes = f.convert_to_enhanced(job_counts,
                                                     job_change_lists,
                                                     cf.jd)
    else:
        j_changes = job_change_lists
        eg_counts = job_counts

    jcnts_arr = f.make_jcnts(eg_counts)
    table = f.job_gain_loss_table(pd.unique(ds.mnum).size,
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
                      periods=pd.unique(df_monthly_non_ret.index).size,
                      freq='M'), inplace=True)

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

    adjust = cf.adjust

    i = 0
    for job_num in last_month_counts.index:
        axis2_lbl_locs.append(round((cnts[i] + cnts[i + 1]) / 2))
        axis2_lbl_locs[i] += adjust[i]
        axis2_lbls.append(jobs_dict[job_num])
        i += 1

    egs = ds[ds.mnum == 0].eg

    if len(emp_list) > max_plots_for_legend:
        lw = 1
    else:
        lw = 3

    with sns.axes_style("white"):
        i = 0
        if chart_example:
            for emp in emp_list:
                c_idx = egs.loc[emp] - 1
                ax1 = ds[ds.empkey == emp].set_index('date')[:through_date] \
                    .cat_order.plot(lw=lw, color=eg_colors[c_idx],
                                    label='Employee ' + str(i + 1))
                i += 1
        else:
            for emp in emp_list:
                c_idx = egs.loc[emp] - 1
                ax1 = ds[ds.empkey == emp].set_index('date')[:through_date] \
                    .cat_order.plot(lw=lw, color=eg_colors[c_idx], label=emp)
                i += 1

        non_ret_count['count'].plot(c='grey', ls='--',
                                    label='active count', ax=ax1)

        if len(emp_list) <= max_plots_for_legend:
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

    plt.title(df_label + ' job level progression', y=1.02)

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
    plt.gcf().set_size_inches(xsize, ysize)
    plt.show()


def differential_scatter(df_list, dfb,
                         measure, filter_measure,
                         filter_val, eg_list, ds_dict=None,
                         attr1=None, oper1='>=', val1=0,
                         attr2=None, oper2='>=', val2=0,
                         attr3=None, oper3='>=', val3=0,
                         prop_order=True,
                         show_scatter=True, show_lin_reg=True,
                         show_mean=True, mean_len=50,
                         dot_size=15, lin_reg_order=15, ylimit=False, ylim=5,
                         suptitle_fontsize=14, title_fontsize=12,
                         legend_fontsize=14,
                         tick_size=11, label_size=12,
                         xsize=12, ysize=8, bright_bg=False,
                         chart_style='whitegrid', chart_example=False):
    '''plot an attribute differential between datasets filtered with another
    attribute if desired.

    Example:  plot the difference in cat_order (job rank number) between all
    integrated datasets vs. standalone for all employee groups, applicable to
    month 57.

    The chart may be set to use proposal order or native list percentage for
    the x axis.

    The scatter markers are selectable on/off, as well as an average line
    and a linear regression line.

    inputs
        df_list (list)
            list of datasets to compare, may be ds_dict (output of
            load_datasets function) string keys or dataframe variable(s)
            or mixture of each
        dfb (string or variable)
            baseline dataset, accepts same input types as df_list above
        measure
            attribute to analyze
        filter_measure
            further reduce or filter the measure by this attribute
        filter_val
            value to apply to the filter_measure
            example: if filter_measure is 'mnum', filter_val would be
            an interger representing the month number
        ds_dict (dictionary)
            output from load_datasets function
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
        eg_list
            a list of employee groups to analyze
        prop_order
            if True, organize x axis by proposal list order,
            otherwise use native list percent
        show_scatter
            if True, draw the scatter chart markers
        show_lin_reg
            if True, draw linear regression lines
        show_mean
            if True, draw average lines
        mean_len
            moving average length for average lines
        dot_size
            scatter marker size
        lin_reg_order
            regression line is actually a polynomial regression
            lin_reg_order is the degree of the fitting polynomial
        ylimit
            if True, set chart y axis limit to ylim (below)
        ylim
            y axis limit positive and negative if ylimit is True
        suptitle_fontsize
            text size of chart super title
        title_fontsize
            text size of chart title
        legend_fontsize
            text size of chart legend labels
        tick_size
            text size of x and y tick labels
        label_size
            text size of x and y descriptive labels
        xsize, ysize
            size of chart
        bright_bg
            use a bright yellow tint chart background
        chart_style
            style for chart, valid inputs are any seaborn chart style
        chart_example
            set True to remove casse-specific data from chart output
    '''

    label_dict = {}
    tb_string = ''
    i = 0
    for df in df_list:
        ds, df_label = determine_dataset(df, ds_dict,
                                         return_label=True)
        label_dict[i + 1] = df_label

        df_list[i] = filter_ds(ds,
                               attr1=attr1, oper1=oper1, val1=val1,
                               attr2=attr2, oper2=oper2, val2=val2,
                               attr3=attr3, oper3=oper3, val3=val3,
                               return_title_string=False)
        i += 1

    df_base, dfb_label = determine_dataset(dfb, ds_dict, return_label=True)

    dfb_filt, tb_string = filter_ds(df_base,
                                    attr1=attr1, oper1=oper1, val1=val1,
                                    attr2=attr2, oper2=oper2, val2=val2,
                                    attr3=attr3, oper3=oper3, val3=val3)

    cols = [measure, 'new_order']

    df = dfb_filt[dfb_filt[filter_measure] == filter_val][
        [measure, 'eg']].copy()
    df.rename(columns={measure: measure + '_s'}, inplace=True)

    order_dict = {}

    i = 1
    for ds in df_list:
        ds = ds[ds[filter_measure] == filter_val][cols].copy()
        ds.rename(columns={measure: measure + '_' + str(i),
                           'new_order': 'order' + str(i)}, inplace=True)
        df = df.join(ds)
        order_dict[i] = 'order' + str(i)
        i += 1

    df.sort_values(by='order1', inplace=True)

    df['eg_sep_order'] = df.groupby('eg').cumcount() + 1
    eg_sep_order = np.array(df.eg_sep_order)
    eg_denom_dict = df.groupby('eg').eg_sep_order.max().to_dict()

    eg_arr = np.array(df.eg)
    eg_set = pd.unique(eg_arr)
    denoms = np.zeros(eg_arr.size)

    for eg in eg_set:
        np.put(denoms, np.where(eg_arr == eg)[0], eg_denom_dict[eg])

    df['sep_eg_pcnt'] = eg_sep_order / denoms

    if measure in ['spcnt', 'lspcnt', 'snum', 'lnum', 'cat_order',
                   'jobp', 'jnum']:

        for key in list(order_dict.keys()):
            df[str(key) + 'vs'] = df[measure + '_s'] - \
                df[measure + '_' + str(key)]
    else:

        for key in list(order_dict.keys()):
            df[str(key) + 'vs'] = df[measure + '_' + str(key)] - \
                df[measure + '_s']

    with sns.axes_style(chart_style):

        for prop_num in np.arange(len(df_list)) + 1:
            df.sort_values(by=order_dict[prop_num], inplace=True)

            if prop_order:
                xax = order_dict[prop_num]
            else:
                xax = 'sep_eg_pcnt'

            fig, ax = plt.subplots()

            for eg in eg_list:
                data = df[df.eg == eg].copy()
                x_limit = max(data[xax]) + 100
                yax = str(prop_num) + 'vs'

                if chart_example:
                    label = str(eg)
                else:
                    label = cf.eg_dict[eg]

                if show_scatter:
                    data.plot(x=xax, y=yax, kind='scatter',
                              linewidth=0.1,
                              color=cf.eg_colors[eg - 1], s=dot_size,
                              label=label,
                              ax=ax)

                if show_mean:
                    data['ma'] = data[eg].rolling(mean_len).mean()
                    data.plot(x=xax, y='ma', lw=5,
                              color=cf.mean_colors[eg - 1],
                              label=label,
                              alpha=.6, ax=ax)
                    plt.xlim(0, x_limit)

                if show_lin_reg:
                    if show_scatter:
                        lin_reg_colors = cf.lin_reg_colors
                    else:
                        lin_reg_colors = cf.lin_reg_colors2
                    sns.regplot(x=xax, y=yax, data=data,
                                color=lin_reg_colors[eg - 1],
                                label=label,
                                scatter=False, truncate=True, ci=50,
                                order=lin_reg_order,
                                line_kws={'lw': 20,
                                          'alpha': .4},
                                ax=ax)
                    plt.xlim(0, x_limit)

            if measure == 'jobp':
                ymin = math.floor(min(df[yax]))
                ymax = math.ceil(max(df[yax]))
                scale_lim = max(abs(ymin), ymax)
                plt.yticks = (np.arange(-scale_lim, scale_lim + 1, 1))
                if ylimit:
                    plt.ylim(-ylim, ylim)
                else:
                    plt.ylim(-scale_lim, scale_lim)

            plt.gcf().set_size_inches(xsize, ysize)
            if chart_example:
                suptitle_str = ('Proposal' + ' differential: ' + measure)
            else:
                if label_dict[prop_num] == 'Proposal':
                    p_label = 'df_list item ' + str(prop_num)
                else:
                    p_label = label_dict[prop_num]
                suptitle_str = (p_label + ' differential: ' + measure)
            if tb_string:
                plt.suptitle(suptitle_str, fontsize=suptitle_fontsize)
                plt.title(tb_string, fontsize=title_fontsize, y=1.005)
            else:
                plt.title(suptitle_str, fontsize=suptitle_fontsize)
            plt.xlim(xmin=0)

            if measure in ['spcnt', 'lspcnt']:
                plt.gca().yaxis.set_major_formatter(pct_format)

            if xax == 'sep_eg_pcnt':
                plt.gca().xaxis.set_major_formatter(pct_format)
                plt.xticks(np.arange(0, 1.1, .1))
                plt.xlim(xmax=1)

            ax.axhline(0, c='m', ls='-', alpha=1, lw=2)
            ax.invert_xaxis()
            if bright_bg:
                ax.set_axis_bgcolor('#faf6eb')
            plt.ylabel('differential', fontsize=label_size)
            plt.tick_params(axis='both', which='major', labelsize=tick_size)
            ax.xaxis.label.set_size(label_size)
            ax.legend(markerscale=1.5, fontsize=legend_fontsize)
            plt.show()


# groupby method:
# only age 65 in group, use .sum
# others, use .mean
def job_grouping_over_time(df, eg_list, jobs, colors,
                           plt_kind='bar', ds_dict=None, rets_only=True,
                           ret_age=cf.ret_age,
                           attr1=None, oper1='>=', val1=0,
                           attr2=None, oper2='>=', val2=0,
                           attr3=None, oper3='>=', val3=0,
                           time_group='A', display_yrs=40, legend_loc=4,
                           suptitle_fontsize=14, title_fontsize=12,
                           legend_fontsize=13,
                           tick_size=11, label_size=12,
                           xsize=12, ysize=10, chart_example=False):

    '''Inverted bar chart display of job counts by group over time.  Various
    filters may be applied to study slices of the datasets.

    The 'rets_only' option will display the count of employees retiring from
    each year grouped by job level.

    developer TODO: fix x axis scaling and labeling when quarterly ("Q") or
    monthly ("M") time group option selected.

    inputs
        df
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        eg_list
            list of unique employee group numbers within the proposal
            Example: [1, 2]
        jobs
            list of job label strings (for plot legend)
        colors
            list of colors to be used for plotting
        plt_kind
            'bar' or 'area' (bar recommended)
        ds_dict
            output from load_datasets function
        rets_only
            calculate for employees at retirement age only
        ret_age
            retirement age in years
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
        time_group
            group counts/percentages by year ('A'), quarter ('Q'),
            or month ('M')
        display_years
            when using the bar chart type display, evenly scale the x axis
            to include the number of years selected for all group charts
        legend_loc
            matplotlib legend location number code
        suptitle_fontsize
            text size of chart super title
        title_fontsize
            text size of chart title
        legend_fontsize
            text size of chart legend labels
        tick_size
            text size of x and y tick labels
        label_size
            text size of x and y descriptive labels
        xsize, ysize
            size of each chart in inches
        chart_example
            produce a chart without case-specific labels (generic example)

    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)

    d_filt, t_string = filter_ds(ds,
                                 attr1=attr1, oper1=oper1, val1=val1,
                                 attr2=attr2, oper2=oper2, val2=val2,
                                 attr3=attr3, oper3=oper3, val3=val3)
    if rets_only:
        try:
            d_filt = d_filt[d_filt.ret_mark == 1][
                ['eg', 'date', 'jnum']].copy()
        except:
            if cf.ret_age_increase:
                print('''Please set "add_ret_mark" option to True
                      in config file''')
                return
            else:
                d_filt = d_filt[d_filt.age == cf.ret_age][
                    ['eg', 'date', 'jnum']].copy()

    colors.append('.7')

    for eg in eg_list:

        with sns.axes_style("darkgrid"):
            denom = len(ds[(ds.mnum == 0) & (ds.eg == eg)])
            df_eg = d_filt[d_filt.eg == eg]

            if rets_only:

                grpby = df_eg.groupby(['date', 'jnum']) \
                    .size().unstack().fillna(0).astype(int)
                df = grpby.resample(time_group).sum()

                if time_group == 'A':
                    df = (df / denom).round(decimals=3)
                if time_group == 'Q':
                    df = (df / (.25 * denom)).round(decimals=3)
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
                df.plot(kind='area', linewidth=0, color=clr, stacked=True)

            if plt_kind == 'bar':
                df.plot(kind='bar', width=1, color=clr, stacked=True)

            ax = plt.gca()
            if rets_only:
                ax.set_yticks(np.arange(.08, 0, -.01))
                ax.yaxis.set_major_formatter(pct_format)

            ax.invert_yaxis()
            if plt_kind == 'bar':
                plt.xlim(0, display_yrs)

            plt.legend((labels), loc=legend_loc, fontsize=legend_fontsize)
            plt.ylabel(ylbl, fontsize=label_size)

            if chart_example:
                suptitle_str = 'Proposal' + ' group ' + str(eg)
            else:
                try:
                    suptitle_str = df_label + ' group ' + cf.eg_dict[eg]
                except:
                    suptitle_str = 'Proposal' + ' group ' + cf.eg_dict[eg]

            if t_string:
                plt.suptitle(suptitle_str, fontsize=suptitle_fontsize)
                plt.title(t_string, fontsize=title_fontsize, y=1.005)
            else:
                plt.title(suptitle_str, fontsize=suptitle_fontsize)

            plt.tick_params(axis='both', which='major', labelsize=tick_size)
            ax.xaxis.label.set_size(label_size)
            plt.gcf().set_size_inches(xsize, ysize)
            plt.show()


def parallel(df_list, dfb, eg_list, measure, month_list, job_levels,
             ds_dict=None,
             attr1=None, oper1='>=', val1=0,
             attr2=None, oper2='>=', val2=0,
             attr3=None, oper3='>=', val3=0,
             left=0, stride_list=None, grid_color='.7',
             suptitle_fontsize=14, title_fontsize=12,
             facecolor='w', xsize=6, ysize=8):
    '''Compare positional or value differences for various proposals
    with a baseline position or value for selected months.

    The vertical lines represent different proposed lists, in the order
    from the df_list list input.

    inputs
        df_list (list)
            list of datasets to compare, may be ds_dict (output of load
            datasets function) string keys or dataframe variable(s) or
            mixture of each
        dfb (string or variable)
            baseline dataset, accepts same input types as df_list above.
            The order of the list is reflected in the chart x axis lables
        eg_list
            list of employee group integer codes to compare
            example: [1, 2]
        measure
            dataset attribute to compare
        month_list
            list of month numbers for analysis.
            the function will plot comparative data from each month listed
        job_levels
            number of job levels in data model
        left
            integer representing the list comparison to plot on left side
            of the chart(s).
            zero (0) represents the standalone results and is the default.
            1, 2, or 3 etc. represent the first, second, third, etc. dataset
            results in df_list input order
        stride_list
            optional list of dataframe strides for plotting every other
            nth result
        grid_color
            string name for horizontal grid color
        facecolor
            chart background color
        xsize, ysize
            size of individual subplots
    '''
    label_dict = {}
    i = 0
    for df in df_list:
        ds, df_label = determine_dataset(df, ds_dict,
                                         return_label=True)
        label_dict[i + 1] = df_label

        df_list[i] = filter_ds(ds,
                               attr1=attr1, oper1=oper1, val1=val1,
                               attr2=attr2, oper2=oper2, val2=val2,
                               attr3=attr3, oper3=oper3, val3=val3,
                               return_title_string=False)
        i += 1

    df_base, dfb_label = determine_dataset(dfb, ds_dict, return_label=True)

    dfb_filt, tb_string = filter_ds(df_base,
                                    attr1=attr1, oper1=oper1, val1=val1,
                                    attr2=attr2, oper2=oper2, val2=val2,
                                    attr3=attr3, oper3=oper3, val3=val3)
    group_dict = cf.eg_dict
    color_dict = dict(enumerate(cf.eg_colors))

    jobs = cf.job_strs

    num_egplots = len(eg_list)
    num_months = len(month_list)

    fig, ax = plt.subplots(num_months, num_egplots)

    fig = plt.gcf()
    fig.set_size_inches(xsize * num_egplots, ysize * num_months)

    plot_num = 0

    for month in month_list:

        ds_dict = od()
        col_dict = od()

        ds_dict[0] = dfb_filt[(dfb_filt.mnum == month) &
                              (dfb_filt.fur == 0)][
                                  ['eg', measure]].copy()
        col_dict[0] = ['eg', 'Baseline']

        i = 1
        for ds in df_list:
            ds = ds[ds['fur'] == 0]
            ds_dict[i] = ds[ds.mnum == month][[measure]].copy()
            if label_dict[i] == "Proposal":
                col_dict[i] = ['List' + str(i)]
            else:
                col_dict[i] = [label_dict[i]]
            i += 1

        dict_nums = list(ds_dict.keys())

        col_list = []
        col_list.extend(col_dict[left])

        df_joined = ds_dict[left]

        i = 1
        for num in dict_nums:
            if num != left:
                df_joined = df_joined.join(ds_dict[num],
                                           rsuffix=('_' + str(num)))
                col_list.extend(col_dict[num])

        df_joined.columns = col_list

        with sns.axes_style('whitegrid', {'axes.facecolor': facecolor,
                                          'axes.axisbelow': True,
                                          'axes.edgecolor': '.2',
                                          'axes.linewidth': 1.0,
                                          'grid.color': grid_color,
                                          'grid.linestyle': u'--'}):

            for eg in eg_list:
                plot_num += 1
                plt.subplot(num_months, num_egplots, plot_num)
                df = df_joined[df_joined.eg == eg]
                try:
                    stride = stride_list[eg - 1]
                    df = df[::stride]
                except:
                    df = df[::(int(len(df) * .015))]
                parallel_coordinates(df, 'eg', lw=1.5, alpha=.7,
                                     color=color_dict[eg - 1])

                plt.title('Group ' + group_dict[eg].upper() + ' ' +
                          measure.upper() +
                          ' ' + str(month) + ' mths',
                          fontsize=title_fontsize, y=1.02)

    fig = plt.gcf()
    for ax in fig.axes:

        if measure in ['spcnt', 'lspcnt']:
            ax.set_yticks(np.arange(1, 0, -.05))
            ax.invert_yaxis()
            ax.yaxis.set_major_formatter(pct_format)

        if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:

            ax.set_yticks(np.arange(0, job_levels + 2, 1))
            ax.set_ylim(job_levels + .5, 0.5)
            yticks = ax.get_yticks().tolist()

            for i in np.arange(1, len(yticks)):
                yticks[i] = jobs[i - 1]

            ax.set_yticklabels(yticks, fontsize=12)

        if measure in ['snum', 'lnum', 'cat_order']:
            ax.invert_yaxis()
        ax.grid()
        ax.legend_.remove()
    plt.suptitle(tb_string, fontsize=title_fontsize, y=1.01)

    plt.tight_layout()
    plt.show()


def rows_of_color(df, mnum, measure_list, eg_colors,
                  jnum_colors, ds_dict=None,
                  attr1=None, oper1='>=', val1=0,
                  attr2=None, oper2='>=', val2=0,
                  attr3=None, oper3='>=', val3=0,
                  cols=150, eg_list=None,
                  job_only=False, jnum=1, cell_border=True,
                  eg_border_color='.3', job_border_color='.8',
                  chart_style='whitegrid',
                  fur_color='.5', empty_color='#ffffff',
                  suptitle_fontsize=14, title_fontsize=12, legend_fontsize=14,
                  xsize=14, ysize=12, chart_example=False):
    '''plot a heatmap with the color of each rectangle representing an
    employee group, job level, or status.

    This chart will show a position snapshot indicating the distribution of
    employees within the entire population, employees holding a certain job,
    or a combination of the two.

    For example, all employees holding a certain job in month 36 may be plotted
    with original group delineated by color.  Or, all employees from one group
    may be shown with the different jobs for that group displayed with
    different colors.

    Also will display any other category such as a special group such as
    furloughed employees.  Input dataframe must have a numerical representation
    of the selected measure, i.e. furloughed indicated by a 1, and others with
    a 0.

    inputs
        df
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        mnum
            month number of dataset to analyze
        measure_list
            list form input, 'categorical' only such as employee group number
            or job number, such as ['jnum'], or ['eg']
            ['eg', 'fur'] is also valid when highlighting furloughees
        eg_colors
            colors to use for plotting the employee groups.
            the first color in the list is used for the plot 'background'
            and is not an employee group color
        jnum_colors
            job level plotting colors, list form
        ds_dict (dictionary)
            output from load_datasets function
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
        cols
            number of columns to construct for the heatmap plot
        eg_list
            employee group integer code list (if used), example: [1, 2]
        job_only
            if True, plot only employees holding the job level identified
            with the jnum input
        jnum
            job level distribution to plot if job_only input is True
        cell_border
            if True, show a border around the heatmap cells
        eg_border_color
            color of cell border if measure_list includes 'eg' (employee group)
        job_border_color
            color of cell border when plotting job information
        chart_style
            underlying chart style, any valid seaborn chart style (string)
        fur_color
            cell color for furloughed employees
        empty_color
            cell color for cells with no data
        suptitle_fontsize
            text size of chart super title
        title_fontsize
            text size of chart title
        legend_fontsize
            text size of chart legend
        xsize, ysize
            size of chart
        chart_example
            if True, produce a chart without case-specific
            labels (generic example)
    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)

    data = ds[ds.mnum == mnum]

    d_filt, t_string = filter_ds(data,
                                 attr1=attr1, oper1=oper1, val1=val1,
                                 attr2=attr2, oper2=oper2, val2=val2,
                                 attr3=attr3, oper3=oper3, val3=val3)

    joined = d_filt[[]].join(data).reindex(data.index)

    rows = int(len(data) / cols) + 1

    heat_data = np.zeros(cols * rows)

    if job_only or measure_list == ['fur']:
        border_color = job_border_color
    else:
        border_color = eg_border_color

    if ('jnum' in measure_list) and (not job_only):
        plot_colors = jnum_colors[:]
    else:
        plot_colors = eg_colors[:]

    plot_colors.append(fur_color)
    plot_colors.insert(0, empty_color)
    fur_integer = len(plot_colors) - 1

    eg = np.array(joined.eg)
    egs = pd.unique(eg)

    if job_only:

        jnums = np.array(joined.jnum)

        for eg_num in egs:
            np.put(heat_data, np.where(eg == eg_num)[0], eg_num)
        np.put(heat_data, np.where(jnums != jnum)[0], 0)
        # if jnum input is not in the list of available job numbers:
        if jnum not in pd.unique(jnums):
            jnum = pd.unique(jnums)[0]

        if chart_example:
            suptitle = 'Proposal' + ' month ' + str(mnum) + \
                ':  ' + cf.job_strs[jnum - 1] + '  job distribution'
        else:
            suptitle = df_label + ' month ' + str(mnum) + \
                ':  ' + cf.job_strs[jnum - 1] + '  job distribution'

    else:

        for measure in measure_list:

            if measure in ['eg', 'jnum']:

                measure = np.array(joined[measure])
                for val in pd.unique(measure):
                    np.put(heat_data, np.where(measure == val)[0], val)

            else:

                if measure == 'fur':
                    measure = np.array(joined[measure])
                    np.put(heat_data, np.where(measure == 1)[0],
                           fur_integer)

                else:
                    measure = np.array(joined[measure])
                    for v in pd.unique(measure):
                        np.put(heat_data, np.where(measure == v)[0], v)

        if chart_example:
            suptitle = 'Proposal' + ': month ' + str(mnum)
        else:
            suptitle = df_label + ': month ' + str(mnum)

    if eg_list:
        np.put(heat_data,
               np.where(np.in1d(eg, np.array(eg_list), invert=True))[0],
               np.nan)

    heat_data = heat_data.reshape(rows, cols)

    cmap = colors.ListedColormap(plot_colors,
                                 name='chart_cmap',
                                 N=len(plot_colors))

    with sns.axes_style(chart_style):

        if cell_border:
            ax = sns.heatmap(heat_data, vmin=0, vmax=len(plot_colors),
                             cbar=False, annot=False,
                             cmap=cmap,
                             linewidths=0.005,
                             linecolor=border_color)
        else:
            ax = sns.heatmap(heat_data, vmin=0, vmax=len(plot_colors),
                             cbar=False, annot=False,
                             cmap=cmap)

        plt.gcf().set_size_inches(xsize, ysize)
        plt.xticks([])
        plt.tick_params(axis='y', labelsize=max(9, (min(12, ysize - 3))))
        plt.ylabel(str(cols) + ' per row',
                   fontsize=max(12, min(ysize + 1, 18)))
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)

    heat_data = heat_data.reshape(rows * cols)

    recs = []
    legend_labels = []

    if 'jnum' in measure_list:

        if job_only:

            label_dict = cf.eg_dict_verbose
            if eg_list:
                heat_unique = np.unique(np.array(eg_list)).astype(int)
            else:
                heat_unique = np.unique(eg).astype(int)

        if not job_only:

            label_dict = cf.jobs_dict
            heat_unique = np.unique(heat_data[~np.isnan(heat_data)]) \
                .astype(int)

    if 'eg' in measure_list:

        label_dict = cf.eg_dict_verbose.copy()
        label_dict[max(egs) + 1] = 'FUR'
        heat_unique = np.unique(heat_data[~np.isnan(heat_data)]) \
            .astype(int)
        heat_unique = heat_unique[heat_unique > 0]

    if measure_list == ['fur']:
        label_dict = {max(egs) + 1: 'FUR'}
        heat_unique = np.unique(heat_data[~np.isnan(heat_data)]).astype(int)

    try:
        heat_unique
    except:
        heat_unique = np.unique(heat_data[~np.isnan(heat_data)]).astype(int)
        label_dict = {}
        for item in heat_unique:
            label_dict[item] = 'value ' + str(item)
    else:
        pass

    for cat in heat_unique:
        if cat > 0:
            try:
                recs.append(mpatches.Rectangle((0, 0), 1, 1,
                            fc=plot_colors[cat],
                            alpha=1))
                legend_labels.append(label_dict[cat])
            except:
                pass

    if t_string:
        plt.suptitle(suptitle, fontsize=suptitle_fontsize)
        plt.title(t_string, fontsize=title_fontsize, y=1.01)
    else:
        plt.title(suptitle, fontsize=suptitle_fontsize)
    ax.legend(recs, legend_labels,
              bbox_to_anchor=(1.2, 0.9),
              fontsize=legend_fontsize)

    plt.show()


def quartile_bands_over_time(df, eg, measure, bins=20, ds_dict=None,
                             clip=True, year_clip=2035, kind='area',
                             quartile_ticks=False,
                             custom_color=True, cm_name='Set1',
                             quartile_alpha=.75, grid_alpha=.5,
                             custom_start=0, custom_finish=.75,
                             xsize=10, ysize=8, alt_bg_color=False,
                             bg_color='#faf6eb', legend_xadj=0):
    '''Visualize quartile distribution for an employee group over time
    for a selected proposal.

    This chart answers the question of where the different employee groups
    will be positioned within the seniority list for future months and years.

    Note:  this is not a comparative study.  It is simply a presentation of
    resultant percentage positioning.

    The chart contains a background grid for reference and may display
    quartiles as integers or percentages, using a bar or area type display,
    and includes several chart color options.

    inputs
        df
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        eg
            employee group number
        measure
            a list percentage input, either 'spcnt' or 'lspcnt'
        bins
            number of quartiles to calculate and display
        ds_dict (dictionary)
            output from load_datasets function
        clip
            if True, limit the chart x axis to year_clip value
        year_clip
            integer, maximum year to display on chart (requires 'clip'
            input to be True)
        kind
            type of chart display, either 'area' or 'bar'
        quartile_ticks
            if True, display integers along y axis and in legend representing
            quartiles.  Otherwise, present percentages.
        custom_color
            If True, use a matplotlib colormap for chart colors
        cm_name
            colormap name (string), example: 'Set1'
        quartile_alpha
            alpha (opacity setting) value for quartile plot
        grid_alpha
            opacity setting for background grid
        custom_start
            custom colormap start level
            (a section of a standard colormap may be used to create
            a custom color mapping)
        custom_finish
            custom colormap finish level
        xsize, ysize
            chart size inputs
        alt_bg_color
            if True, set the background chart color to the bg_color input value
        bg_color
            color for chart background if 'alt_bg_color' is True (string)
        legend_xadj
            (float) small float number (try .2 to start) for use when the
            legend overlaps the chart.  Moves the legend to the right.

    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)

    if custom_color:
        cm_subsection = np.linspace(custom_start, custom_finish, bins)
        colormap = eval('cm.' + cm_name)
        quartile_colors = [colormap(x) for x in cm_subsection]
    else:
        quartile_colors = cf.color1

    if clip:
        eg_df = ds[(ds.eg == eg) & (ds.date.dt.year <= year_clip)]
    else:
        eg_df = ds[ds.eg == eg]

    eg_df = eg_df[['date', 'empkey', measure]]
    eg_df['year'] = eg_df.date.dt.year

    bin_lims = np.linspace(0, 1, num=bins + 1, endpoint=True, retstep=False)
    years = pd.unique(eg_df.year)

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

        xmajor_ticks = np.arange(min(ds.date.dt.year), year_clip, 1)

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
            plt.gca().yaxis.set_major_formatter(pct_format)
            legend_labels = ['{percent:.1f}'
                             .format(percent=((quart * step) - step) * 100) +
                             ' - ' +
                             '{percent:.1%}'.format(percent=quart * step)
                             for quart in quartiles]
            legend_title = 'result_pcnt'
            legend_pos_adj = .1

        if alt_bg_color:
            ax.set_axis_bgcolor(bg_color)

        plt.title(df_label + ', group ' + str(eg) +
                  ' quartile change over time',
                  fontsize=16, y=1.02)

        quartiles = np.arange(1, bins + 1)

        recs = []
        patch_alpha = min(quartile_alpha + .1, 1)
        legend_font_size = np.clip(int(bins / 1.65), 12, 14)
        legend_cols = int(bins / 30) + 1
        legend_position = 1 + (legend_cols * .17) + \
            legend_pos_adj + legend_xadj

        for i in np.arange(bins, dtype='int'):
            recs.append(mpatches.Rectangle((0, 0), 1, 1,
                                           fc=quartile_colors[i],
                                           alpha=patch_alpha))

        ax.legend(recs, legend_labels, bbox_to_anchor=(legend_position, 1),
                  title=legend_title, fontsize=legend_font_size,
                  ncol=legend_cols)

        fig = plt.gcf()
        fig.set_size_inches(xsize, ysize)
        fig.tight_layout()
        plt.show()


def job_transfer(dfc, dfb, eg, colors,
                 job_levels, ds_dict=None,
                 measure='jnum', gb_period='M',
                 custom_color=True, cm_name='Paired',
                 start=0, stop=.95, job_alpha=1, chart_style='white',
                 yticks_lim=5000, fur_color='.5',
                 draw_face_color=True, draw_grid=True,
                 ytick_interval=100, legend_xadj=1.62,
                 legend_yadj=.78, annotate=False,
                 title_fontsize=14,
                 legend_font_size=12,
                 legend_horizontal_position=1.12,
                 xsize=10, ysize=8):
    '''plot a differential stacked bar chart displaying color-coded job
    transfer counts over time.  Result appears to be stacked area chart.

    inputs
        dfc
            proposal (comparison) dataset to examine, may be a dataframe
            variable or a string key from the ds_dict dictionary object
        dfb
            baseline dataset; proposal dataset is compared to this
            dataset, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        eg
            integer code for employee group
        colors
            list of colors for job levels
        job_levels
            number of job levels in data model
        ds_dict (dictionary)
            output from load_datasets function
        measure
            currently only 'jnum' is applicable
        gb_period
            group_by period. default is 'M' for monthly, other options
            are 'Q' for quarterly and 'A' for annual
        custom_color
            create custom color map
        cm_name
            color map name
        start
            custom color linspace start
        stop
            custom color linspace stop
        job_alpha
            chart alpha level for job transfer plotting
        chart_style
            seaborn plotting library style
        yticks_lim
            limit for y tick labeling
        draw_face_color
            apply a transparent background to the chart, red below zero
            and green above zero
        draw_grid
            show major tick label grid lines
        ytick_interval
            ytick spacing
        legend_xadj
            horizontal addjustment for legend placement
        legend_yadj
            vertical adjustment for legend placement
        annotate
            add text to chart, 'job count increase' and 'job count decrease'
        xsize
            horizontal size of chart
        ysize
            vertical size of chart
    '''
    dsc, dfc_label = determine_dataset(dfc, ds_dict, return_label=True)
    dsb, dfb_label = determine_dataset(dfb, ds_dict, return_label=True)

    compare_df = dsc[dsc.eg == eg][['date', measure]].copy()
    base_df = dsb[dsb.eg == eg][['date', measure]].copy()

    cg = pd.DataFrame(compare_df.groupby(['date', measure]).size()
                      .unstack().fillna(0).resample(gb_period).mean())
    bg = pd.DataFrame(base_df.groupby(['date', measure]).size()
                      .unstack().fillna(0).resample(gb_period).mean())

    for job_level in np.arange(1, cf.num_of_job_levels + 1):
        if job_level not in bg:
            bg[job_level] = 0
        if job_level not in cg:
            cg[job_level] = 0

    bg.sort_index(axis=1, inplace=True)
    cg.sort_index(axis=1, inplace=True)

    diff2 = cg - bg
    abs_diff2 = np.absolute(diff2.values).astype(int)
    v_crop = (np.amax(np.add.reduce(abs_diff2, axis=1)) / 2) + 75

    if custom_color:
        num_of_colors = job_levels + 1
        cm_subsection = np.linspace(start, stop, num_of_colors)
        colormap = eval('cm.' + cm_name)
        colors = [colormap(x) for x in cm_subsection]
    colors.append(fur_color)

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

        if gb_period in ['Q', 'A']:
            ax.axvline(xtick_locs[0], ls='-', color='grey',
                       lw=1, alpha=.2, zorder=7)

        yticks = np.arange(-yticks_lim,
                           yticks_lim + ytick_interval,
                           ytick_interval)

        ax.set_yticks(yticks)
        plt.ylim(-v_crop, v_crop)
        ymin, ymax = ax.get_ylim()

        if draw_grid:
            for xmaj in xtick_locs:
                try:
                    if i % interval == 0:
                        ax.axvline(xtick_locs[i], ls='-', color='grey',
                                   lw=1, alpha=.2, zorder=7)
                    i += 1
                except:
                    pass
            for i in yticks:
                ax.axhline(i, ls='-', color='grey', lw=1, alpha=.2, zorder=7)

        recs = []
        job_labels = []
        legend_title = 'job'
        for i in diff2.columns:
            recs.append(mpatches.Rectangle((0, 0), 1, 1,
                                           fc=colors[i - 1],
                                           alpha=job_alpha))
            job_labels.append(cf.job_strs[i - 1])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * legend_xadj, box.height])
        ax.legend(recs, job_labels, bbox_to_anchor=(legend_horizontal_position,
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
        if draw_face_color:
            plt.axhspan(0, ymax, facecolor='g', alpha=0.05, zorder=8)
            plt.axhspan(0, ymin, facecolor='r', alpha=0.05, zorder=8)
        plt.ylabel('change in job count', fontsize=16)
        plt.xlabel('date', fontsize=16, labelpad=15)

        try:
            title_string = 'GROUP ' + cf.eg_dict[eg] + \
                ' Jobs Exchange' + '\n' + \
                dfc_label + \
                ' compared to ' + dfb_label
            plt.title(title_string,
                      fontsize=title_fontsize, y=1.02)
        except:
            pass

        fig.set_size_inches(xsize, ysize)
        plt.show()


def editor(base_ds_text='standalone', compare_ds_text='ds_edit',
           cond_list=None, reset=False, prop_order=True, mean_len=80,
           dot_size=20, lin_reg_order=12, ylimit=False, ylim=5,
           width=17.5, height=10, strip_height=3.5, bright_bg=True,
           chart_style='whitegrid', bg_clr='white', show_grid=True):
    '''compare specific proposal attributes and interactively adjust
    list order.  may be used to minimize distortions.  utilizes ipywidgets.

    The function will recursively use an edited dataset so that an integrated
    list may be incrementally adjusted and examined at each step.  The edited
    dataset is created the first time a calculation is run.  Prior to the
    creation of the edited dataset ('dill/ds_edit.pkl'), the function will use
    the compare_ds_text input to select a previously calculated dataset for
    initial comparison.  The function then will revert to the new edited
    dataset for all subsequent calculations.

    The function will delete the edited dataset and start over with a
    user-defined dataset if the reset input is set to True.  This input must
    be removed or set back to False to allow dataset editing to take place.

    As it stands now, if the user desires to save an edited dataset for further
    analysis, it must be manually copied and saved to another folder.

    inputs
        base_ds_text
            baseline dataset string name
        compare_ds_text
            comparison dataset string name
        cond_list
            conditions to apply when calculating dataset
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
    # define path for edited datasets (ds_edit.pkl):
    edit_file = 'dill/ds_edit.pkl'
    # boolean value, True if ds_edit exists
    edit_file_exists = path.exists(edit_file)

    # test to see if user wishes to start over with another dataset
    if not reset:
        # if not, see if an edited dataset exists:
        if edit_file_exists:
            # if so, use it for calculations
            compare_ds = pd.read_pickle(edit_file)
            title_label = 'edited'

        else:
            # if not, use user input dataset (compare_ds_text) for calculations
            if path.exists('dill/ds_' + compare_ds_text + '.pkl'):
                compare_ds = pd.read_pickle('dill/ds_' + compare_ds_text +
                                            '.pkl')
                title_label = compare_ds_text
            # if user dataset not found, default to first known dataset
            else:
                proposal_list = \
                    list(pd.read_pickle('dill/proposal_names.pkl').proposals)
                compare_ds = pd.read_pickle('dill/ds_' + proposal_list[0] +
                                            '.pkl')
                title_label = '< using ' + proposal_list[0] + '>'

    else:
        # if reset, remove edited dataset file
        if edit_file_exists:
            remove(edit_file)
        try:
            compare_ds = pd.read_pickle('dill/ds_' + compare_ds_text + '.pkl')
            title_label = compare_ds_text
        except:
            proposal_list = \
                list(pd.read_pickle('dill/proposal_names.pkl').proposals)
            compare_ds = pd.read_pickle('dill/ds_' + proposal_list[0] + '.pkl')
            title_label = '< using ' + proposal_list[0] + ' >'

    # set baseline dataset
    try:
        base_ds = pd.read_pickle('dill/_ds' + base_ds_text + '.pkl')
    except:
        try:
            base_ds = pd.read_pickle('dill/standalone.pkl')
        except:
            # exit routine if baseline dataset not found
            print('invalid "base_ds" name input, neither ds_' +
                  base_ds + '.pkl nor standalone.pkl not found\n')
            return

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
                                    description='attr')
    drop_operator = widgets.Dropdown(options=['<', '<=',
                                              '==', '!=', '>=', '>'],
                                     value=persist['drop_opr'].value,
                                     description='operator')
    drop_filter = widgets.Dropdown(options=['jnum', 'mnum', 'eg', 'sg', 'age',
                                            'scale', 's_lmonths',
                                            'orig_job', 'lnum', 'snum', 'mnum',
                                            'rank_in_job', 'mpay', 'cpay',
                                            'ret_mark'],
                                   value=persist['drop_filter'].value,
                                   description='attr filter')

    int_val = widgets.Text(min=0,
                           max=max_month,
                           value=persist['int_sel'].value,
                           description='value', width='150px')

    mnum_operator = widgets.Dropdown(options=['<', '<=',
                                              '==', '!=', '>=', '>'],
                                     value=persist['mnum_opr'].value,
                                     description='mnum')

    mnum_input = widgets.Dropdown(options=list(np.arange(0, max_month)
                                               .astype(str)),
                                  value=persist['int_mnum'].value,
                                  description='value')

    mnum_val = mnum_input.value
    mnum_oper = mnum_operator.value
    measure = drop_measure.value
    filter_measure = drop_filter.value
    filter_operator = drop_operator.value
    filter_val = int_val.value

    cols = [measure, 'new_order']

    df = base_ds[eval('(base_ds[filter_measure]' +
                      filter_operator +
                      filter_val +
                      ') & (base_ds.mnum' +
                      mnum_oper + mnum_val +
                      ')')][[measure, 'eg']].copy()

    df.rename(columns={measure: measure + '_b'}, inplace=True)

    yval = 'differential'

    # for stripplot and squeeze:
    data_reorder = compare_ds[compare_ds.mnum == 0][['eg']].copy()
    data_reorder['new_order'] = np.arange(len(data_reorder)).astype(int)
    # for drop_eg selection widget:
    drop_eg_options = sorted(list(pd.unique(data_reorder.eg).astype(str)))

    to_join_ds = compare_ds[eval('(compare_ds[filter_measure]' +
                                 filter_operator +
                                 filter_val +
                                 ') & (compare_ds.mnum' +
                                 mnum_oper +
                                 mnum_val +
                                 ')')][cols].copy()

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
    eg_set = sorted(list(pd.unique(df.eg)))
    max_eg_plus_one = max(eg_set) + 1
    denoms = np.zeros(eg_arr.size)

    for eg in eg_set:
        np.put(denoms, np.where(eg_arr == eg)[0], eg_denom_dict[eg])

    df['sep_eg_pcnt'] = eg_sep_order / denoms

    if measure in ['spcnt', 'lspcnt', 'snum', 'lnum', 'cat_order',
                   'jobp', 'jnum']:

        df[yval] = df[measure + '_b'] - df[measure + '_c']

    else:
        df[yval] = df[measure + '_c'] - df[measure + '_b']

    with sns.axes_style(chart_style):

        fig, ax = plt.subplots(figsize=(width, height))

        df.sort_values(by='proposal_order', inplace=True)

        if prop_order:
            xval = 'proposal_order'

        else:
            xval = 'sep_eg_pcnt'

        for eg in eg_set:
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
                    lin_reg_colors = cf.lin_reg_colors
                else:
                    lin_reg_colors = cf.lin_reg_colors2
                ax = sns.regplot(x=xval, y=yval, data=data,
                                 color=lin_reg_colors[eg - 1],
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
            plt.yticks = (np.arange(-scale_lim, scale_lim + 1, 1))
            if ylimit:
                plt.ylim(-ylim, ylim)
            else:
                plt.ylim(-scale_lim, scale_lim)

        plt.gcf().set_size_inches(width, height)
        plt.tick_params(labelsize=13)

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(15)

        plt.title(title_label + ' differential: ' + measure, fontsize=16)
        plt.xlim(xmin=0)

        if measure in ['spcnt', 'lspcnt']:
            plt.gca().yaxis.set_major_formatter(pct_format)

        if xval == 'sep_eg_pcnt':
            plt.xlim(xmax=1)

        ax.axhline(0, c='m', ls='-', alpha=1, lw=1.5)
        ax.invert_xaxis()
        ax.legend(markerscale=1.5, fontsize=14)
        if bright_bg:
            ax.set_axis_bgcolor(bg_clr)
        if show_grid:
            ax.grid(lw=1, ls='-', c='grey', alpha=.25)

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

        drop_eg = widgets.Dropdown(options=drop_eg_options,
                                   value=persist['drop_eg_val'].value,
                                   description='emp grp')

        drop_dir = widgets.Dropdown(options=['u  >>', '<<  d'],
                                    value=persist['drop_dir_val'].value,
                                    description='sqz dir')

        drop_squeeze = widgets.Dropdown(options=['log', 'slide'],
                                        value=persist['drop_sq_val'].value,
                                        description='sq type')

        slide_factor = widgets.IntSlider(value=persist['slide_fac_val'].value,
                                         min=1,
                                         max=400,
                                         step=1,
                                         description='squeeze',
                                         margin='15px',
                                         width='600px')

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

            with sns.axes_style(chart_style):
                fig, ax2 = plt.subplots(figsize=(width, strip_height))

            ax2 = sns.stripplot(x='new_order', y='eg',
                                data=data_reorder, jitter=.5,
                                orient='h', order=np.arange(1,
                                                            max_eg_plus_one,
                                                            1),
                                palette=cf.eg_colors, size=3,
                                linewidth=0, split=True)

            for item in ([ax2.xaxis.label, ax2.yaxis.label] +
                         ax2.get_xticklabels() + ax2.get_yticklabels()):
                item.set_fontsize(12)

            if bright_bg:
                ax2.set_axis_bgcolor(bg_clr)

            plt.xticks(np.arange(0, len(data_reorder), 1000))
            if measure in ['spcnt', 'lspcnt', 'cpay']:
                plt.ylabel('\n\neg\n')
            else:
                plt.ylabel('eg\n')
            plt.xlim(len(data_reorder), 0)
            plt.show()

            data_reorder[['new_order']].to_pickle('dill/p_new_order.pkl')

            store_vals()

        def store_vals():
            persist_df = pd.DataFrame({'drop_eg_val': drop_eg.value,
                                       'drop_dir_val': drop_dir.value,
                                       'drop_sq_val': drop_squeeze.value,
                                       'slide_fac_val': slide_factor.value,
                                       'scat_val': chk_scatter.value,
                                       'fit_val': chk_fit.value,
                                       'mean_val': chk_mean.value,
                                       'drop_msr': drop_measure.value,
                                       'drop_opr': drop_operator.value,
                                       'drop_filter': drop_filter.value,
                                       'mnum_opr': mnum_operator.value,
                                       'int_mnum': mnum_input.value,
                                       'int_sel': int_val.value,
                                       'junior': range_sel.children[0].value,
                                       'senior': range_sel.children[1].value},
                                      index=['value'])

            persist_df.to_pickle('dill/squeeze_vals.pkl')

        def run_cell(ev):
            # 'new_order' is simply a placeholder here.
            # This is where ds_edit.pkl is generated (compute_measures script)
            cmd = 'python compute_measures.py new_order edit'
            if cond_list:
                for cond in cond_list:
                    cmd = cmd + ' ' + cond
            # run compute_measures script with conditions
            system(cmd)
            # show results
            display(Javascript('IPython.notebook.execute_cell()'))

        def redraw(ev):
            store_vals()
            display(Javascript('IPython.notebook.execute_cell()'))

        button_calc = Button(description="calculate",
                             background_color='#80ffff', width='80px',
                             margin='15px')
        button_calc.on_click(run_cell)

        button_plot = Button(description="plot",
                             background_color='#dab3ff', width='80px',
                             margin='15px')
        button_plot.on_click(redraw)

        if prop_order:
            range_sel = interactive(set_cursor, junior=(0, x_limit),
                                    senior=(0, x_limit), width='600px')
        else:
            range_sel = interactive(set_cursor, junior=(0, 1, .001),
                                    senior=(0, 1, .001), width='600px')

        button = Button(description='squeeze',
                        background_color='#b3ffd9', width='80px',
                        margin='15px')
        button.on_click(perform_squeeze)

        hbox8 = widgets.HBox((button, button_calc,
                              button_plot))
        hbox1 = widgets.HBox((range_sel, hbox8))

        vbox1 = widgets.VBox((chk_scatter, chk_fit, chk_mean))
        vbox2 = widgets.VBox((drop_squeeze, drop_eg, drop_dir))
        vbox3 = widgets.VBox((drop_measure,
                              mnum_operator, mnum_input))
        vbox4 = widgets.VBox((drop_filter,
                              drop_operator, int_val))
        hbox2 = widgets.HBox((vbox1, vbox2, vbox3, vbox4))
        display(widgets.VBox((slide_factor, hbox2, hbox1)))


def reset_editor():
    '''reset widget selections to default.
    (for use when invalid input is selected resulting in an exception)
    '''
    def reset(x):
        init_editor_vals = pd.DataFrame([['<<  d', '2', 'ret_mark',
                                        'spcnt', 'log',
                                         False, '==', '1',
                                         5000, False, True,
                                         1000, 100, '>=', '0']],
                                        columns=['drop_dir_val', 'drop_eg_val',
                                                 'drop_filter', 'drop_msr',
                                                 'drop_sq_val', 'fit_val',
                                                 'drop_opr',
                                                 'int_sel', 'junior',
                                                 'mean_val',
                                                 'scat_val', 'senior',
                                                 'slide_fac_val',
                                                 'mnum_opr', 'int_mnum'],
                                        index=['value'])

        init_editor_vals.to_pickle('dill/squeeze_vals.pkl')

    button_reset = Button(description='reset editor',
                          background_color='#ffff99', width='120px')
    button_reset.on_click(reset)
    display(button_reset)


def eg_multiplot_with_cat_order(df, mnum, measure, xax, job_strs,
                                span_colors, job_levels,
                                ds_dict=None, fur_color='.5',
                                single_eg=False, num=1, exclude_fur=False,
                                plot_scatter=True, s=20, a=.7, lw=0,
                                xsize=10, ysize=10, title_fontsize=14,
                                tick_fontsize=12,
                                chart_example=False):
    '''num input options:
                   {1: 'eg1_with_sg',
                    2: 'eg2',
                    3: 'eg3',
                    4: 'eg1_no_sg',
                    5: 'sg_only'
                    }

    sg refers to special group - a group with special job rights
    '''

    df, df_label = determine_dataset(df, ds_dict, return_label=True)

    span_colors.append(fur_color)
    max_count = df.groupby('mnum').size().max()
    mnum_count = pd.unique(df.mnum).size
    df = df[df.mnum == mnum].copy()

    if measure == 'cat_order':
        sns.set_style('white')

        if cf.enhanced_jobs:
            # use job dictionary from case-specific configuration file
            # for conversion
            eg_counts, j_changes = f.convert_to_enhanced(cf.eg_counts,
                                                         cf.j_changes,
                                                         cf.jd)
        else:
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

        grp_dict = {1: 'eg1_with_sg',
                    2: 'eg2',
                    3: 'eg3',
                    4: 'eg1_no_sg',
                    5: 'sg_only'
                    }

        cdict = {1: 'black',
                 2: 'blue',
                 3: '#FF6600',
                 4: 'black',
                 5: 'green'
                 }

        if num == 5:
            df = df[(df.eg == 1) & (df.sg == 1)]
            label = 'sg_only'
        elif num == 4:
            df = df[(df.eg == 1) & (df.sg == 0)]
            label = 'eg1_no_sg'
        elif num == 2:
            df = df[df.eg == 2]
            label = 'eg2'
        elif num == 3:
            df = df[df.eg == 3]
            label = 'eg3'
        elif num == 1:
            df = df[df.eg == 1]
            label = 'eg1_with_sg'

        if exclude_fur:
            df = df[df.fur == 0]

        print('\n"num" input codes:\n', grp_dict)

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
                  df_label + ' month=' + str(mnum), y=1.02,
                  fontsize=title_fontsize)

    else:

        if exclude_fur:
            df = df[df.fur == 0]

        d1 = df[(df.eg == 1) & (df.sg == 1)]
        d2 = df[(df.eg == 1) & (df.sg == 0)]
        d3 = df[df.eg == 2]
        d4 = df[df.eg == 3]

        if plot_scatter:
            ax1 = d1.plot(x=xax, y=measure, kind='scatter',
                          label='eg1_sg_only', color='#5cd65c',
                          alpha=a, s=s, linewidth=lw)
            d2.plot(x=xax, y=measure, kind='scatter',
                    label='eg1_no_sg', color='black',
                    alpha=a, s=s, linewidth=lw, ax=ax1)
            d3.plot(x=xax, y=measure, kind='scatter',
                    label='eg2', color='blue',
                    alpha=a, s=s, linewidth=lw, ax=ax1)
            d4.plot(x=xax, y=measure, kind='scatter',
                    label='eg3', c='#FF6600',
                    alpha=a, s=s, linewidth=lw, ax=ax1)

        else:
            ax1 = d1.set_index(xax,
                               drop=True)[measure].plot(label='eg1_sg_only',
                                                        color='green', alpha=a)
            d2.set_index(xax,
                         drop=True)[measure].plot(label='eg1_no_sg',
                                                  color='black', alpha=a,
                                                  ax=ax1)
            d3.set_index(xax,
                         drop=True)[measure].plot(label='eg2',
                                                  color='blue', alpha=a,
                                                  ax=ax1)
            d4.set_index(xax,
                         drop=True)[measure].plot(label='eg3',
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
                      df_label + ' month ' + str(mnum), y=1.02)

    ax1.legend(loc='center left', bbox_to_anchor=(-0.45, 0.9),
               frameon=True, fancybox=True, shadow=True, markerscale=2)
    plt.tick_params(labelsize=tick_fontsize)
    plt.ylabel(measure)

    fig = plt.gca()

    if measure in ['snum', 'spcnt', 'lspcnt',
                   'jnum', 'jobp', 'fbff', 'cat_order']:
        plt.gca().invert_yaxis()
        if measure in ['spcnt', 'lspcnt']:
            ax1.set_yticks(np.arange(1, -.05, -.05))
            ax1.yaxis.set_major_formatter(pct_format)
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

            # plot job band background on chart
            for i in np.arange(1, job_ticks.size):
                ax2.axhspan(job_ticks[i - 1], job_ticks[i],
                            facecolor=span_colors[i - 1], alpha=.15)
            ax1.grid(ls='dashed', lw=.5)

    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:

        ax1.set_yticks(np.arange(0, job_levels + 2, 1))
        yticks = fig.get_yticks().tolist()

        for i in np.arange(1, len(yticks)):
            yticks[i] = job_strs[i - 1]

        plt.axhspan(job_levels + 1, job_levels + 2, facecolor='.8', alpha=0.2)
        ax1.set_yticklabels(yticks)
        plt.axhline(y=job_levels + 1, c='.8', ls='-', alpha=.8, lw=3)
        plt.ylim(job_levels + 1.5, 0.5)

    if xax in ['snum']:
        plt.xlim(max_count, 0)
    if xax in ['spcnt', 'lspcnt']:
        plt.gca().xaxis.set_major_formatter(pct_format)
        plt.xticks(np.arange(0, 1.1, .1))
        plt.xlim(1, 0)
    if xax == 'age':
        plt.xlim(xmax=cf.ret_age)
    if xax in ['ylong']:
        plt.xticks(np.arange(0, 55, 5))
        plt.xlim(-0.5, max(df.ylong) + 1)
    plt.tick_params(labelsize=tick_fontsize)

    plt.gcf().set_size_inches(xsize, ysize)
    plt.show()
    sns.set_style('darkgrid')


def diff_range(df_list, dfb, measure, eg_list,
               ds_dict=None, eg_colors=cf.eg_colors,
               attr1=None, oper1='>=', val1=0,
               attr2=None, oper2='>=', val2=0,
               attr3=None, oper3='>=', val3=0,
               year_clip=2042,
               show_range=False, show_mean=True, normalize_y=False,
               suptitle_fontsize=14, title_fontsize=12,
               tick_size=11, label_size=12, legend_fontsize=13,
               legend_horizontal_position=1.35,
               ysize=6, xsize=8, plot_style='whitegrid'):
    '''Plot a range of differential attributes or a differential
    average over time.  Individual employee groups and proposals may
    be selected.  Each chart indicates the results for one group with
    color bands or average lines indicating the results for that group
    under different proposals.  This is different than the usual method
    of different groups being plotted on the same chart.

    inputs
        df_list
            list of datasets to compare, may be ds_dict (output of
            load_datasets function) string keys or dataframe variable(s)
            or mixture of each
        dfb
            baseline dataset, accepts same input types as df_list above
        measure
            differential data to compare
        eg_list
            list of integers for employee groups to be included in analysis.
            example: [1, 2, 3]
            A chart will be produced for each employee group number.
        ds_dict (dictionary)
            output from load_datasets function
        eg_colors
            list of colors to represent different proposal results
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
        year_clip (integer)
            only plot data up to and including this year
        show_range
            show a transparent background on the chart representing
            the range of values for each measure for each proposal
        show_mean
            plot a line representing the average of the measure values for
            the group under each proposal
        normalize_y
            if measure is 'spcnt' or 'lspcnt', equalize the range of the
            y scale on all charts (-.5 to .5)
        suptitle_fontsize
            text size of chart super title
        title_fontsize
            text size of chart title
        tick_size
            text size of chart tick labels
        label_size
            text size of chart x and y axis labels
        legend_fontsize
            text size of the legend labels
        legend_horizontal_position
            horizontal adjustment of the legend (higher numbers move right)
        xsize, ysize
            size of chart, width and height
        plot_style
            any valid seaborn plotting style (string)
    '''
    print('''NOTE:  Each chart represents a single employee group.
          The lines represent how that group is affected
          by each proposal.  This format is different from other
          charts.''')
    label_dict = {}
    i = 0
    for df in df_list:
        ds, df_label = determine_dataset(df, ds_dict,
                                         return_label=True)
        if df_label == 'Proposal':
            df_label = 'list' + str(i + 1)
        label_dict[i + 1] = df_label

        df_list[i] = filter_ds(ds,
                               attr1=attr1, oper1=oper1, val1=val1,
                               attr2=attr2, oper2=oper2, val2=val2,
                               attr3=attr3, oper3=oper3, val3=val3,
                               return_title_string=False)
        i += 1

    df_base, dfb_label = determine_dataset(dfb, ds_dict, return_label=True)

    dfb_filt, tb_string = filter_ds(df_base,
                                    attr1=attr1, oper1=oper1, val1=val1,
                                    attr2=attr2, oper2=oper2, val2=val2,
                                    attr3=attr3, oper3=oper3, val3=val3)

    cmap = colors.ListedColormap(eg_colors)
    cols = ['date']

    sa_ds = dfb_filt[dfb_filt.date.dt.year <= year_clip][
        ['mnum', 'eg', 'date', measure]].copy()
    sa_ds['eg_order'] = sa_ds.groupby(['mnum', 'eg']).cumcount()
    sa_ds.sort_values(['mnum', 'eg', 'eg_order'], inplace=True)
    sa_ds.pop('eg_order')
    sa_ds.reset_index(inplace=True)
    sa_ds.set_index(['mnum', 'empkey'], drop=True, inplace=True)

    i = 0
    col_list = []
    for ds in df_list:
        col_name = measure + '_' + label_dict[i + 1]
        col_list.append(col_name)
        ds = ds[ds.date.dt.year <= year_clip][['mnum', 'eg',
                                               'date', measure]].copy()
        ds['eg_order'] = ds.groupby(['mnum', 'eg']).cumcount()
        ds.sort_values(['mnum', 'eg', 'eg_order'], inplace=True)
        ds.pop('eg_order')
        ds.reset_index(inplace=True)
        ds.set_index(['mnum', 'empkey'], drop=True, inplace=True)
        ds.rename(columns={measure: col_name}, inplace=True)
        df_list[i] = ds
        i += 1

    i = 0
    for ds in df_list:
        col = col_list[i]
        sa_ds[col] = ds[col]
        sa_ds[col] = sa_ds[col] - sa_ds[measure]
        i += 1

    cols.extend(col_list)

    for eg in eg_list:
        if show_range:
            with sns.axes_style(plot_style):
                ax1 = sa_ds[sa_ds.eg == eg][cols].set_index('date') \
                    .plot(cmap=cmap, alpha=.22)
                plt.grid(lw=1, ls='--', c='grey', alpha=.25)
                if show_mean:
                    ax1.legend_ = None
                    plt.draw()
        if show_mean:
            with sns.axes_style(plot_style):
                if show_range:
                    sa_ds[sa_ds.eg == eg][cols].set_index('date') \
                        .resample('Q').mean().plot(cmap=cmap, ax=ax1)
                else:
                    sa_ds[sa_ds.eg == eg][cols].set_index('date') \
                        .resample('Q').mean().plot(cmap=cmap)

        ax = plt.gca()

        if measure in ['spcnt', 'lspcnt', 'jobp', 'jnum', 'cat_order']:
            ax.invert_yaxis()

        plt.axhline(c='m', lw=2, ls='--')
        plt.gcf().set_size_inches(xsize, ysize)
        if measure in ['spcnt', 'lspcnt']:
            ax.yaxis.set_major_formatter(pct_format)
            if normalize_y:
                plt.ylim(.5, -.5)
            plt.yticks = np.arange(.5, -.55, .05)

        suptitle = 'Employee Group ' + str(eg) + ' ' +\
            measure + ' differential'
        if tb_string:
            plt.suptitle(suptitle, fontsize=suptitle_fontsize)
            plt.title(tb_string, fontsize=title_fontsize)
        else:
            plt.title(suptitle, fontsize=suptitle_fontsize)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels,
                  bbox_to_anchor=(legend_horizontal_position, .8),
                  fontsize=legend_fontsize)

        plt.tick_params(axis='y', labelsize=tick_size)
        plt.tick_params(axis='x', labelsize=tick_size)
        ax.xaxis.label.set_size(label_size)
        plt.show()


def job_count_charts(dfc, dfb, eg_list=None, ds_dict=None,
                     attr1=None, oper1='>=', val1=0,
                     attr2=None, oper2='>=', val2=0,
                     attr3=None, oper3='>=', val3=0,
                     plot_egs_sep=False, plot_total=True, year_max=None,
                     base_ls='solid', prop_ls='dotted',
                     base_lw=1.6, prop_lw=2.5,
                     suptitle_fontsize=14, title_fontsize=12,
                     total_color='g', xsize=6, ysize=5):
    '''line-style charts displaying job category counts over time.

    optionally display employee group results on separate charts or together

    inputs
        dfc
            proposal (comparison) dataset to examine, may be a dataframe
            variable or a string key from the ds_dict dictionary object
        dfb
            baseline dataset; proposal dataset is compared to this
            dataset, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        eg_list
            list of employee group codes to plot
            Example: [1, 2]
        ds_dict
            variable assigned to load_datasets function output
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
        plot_egs_sep
            if True, plot each employee group job level counts separately
        plot_total
            if True, include the combined job counts on chart(s)
        year_max
            maximum year to include on chart
            Example:  if input is 2030, chart would display data from
            beginning of data model through 2030 (integer)
        base_ls
            line style for base job count line(s)
        prop_ls
            line style for comparison (proposal) job count line(s)
        base_lw
            line width for base job count line(s)
        prop_lw
            line width for comparison (proposal) job count lines
        suptitle_fontsize
            text size of chart super title
        title_fontsize
            chart title(s) font size
        total_color
            color for combined job level count from all employee groups
        xsize, ysize
            size of chart display
    '''
    dsc, dfc_label = determine_dataset(dfc, ds_dict, return_label=True)
    dsb, dfb_label = determine_dataset(dfb, ds_dict, return_label=True)

    dfc_filt = filter_ds(dsc,
                         attr1=attr1, oper1=oper1, val1=val1,
                         attr2=attr2, oper2=oper2, val2=val2,
                         attr3=attr3, oper3=oper3, val3=val3,
                         return_title_string=False)

    dfb_filt, t_string = filter_ds(dsb,
                                   attr1=attr1, oper1=oper1, val1=val1,
                                   attr2=attr2, oper2=oper2, val2=val2,
                                   attr3=attr3, oper3=oper3, val3=val3)

    suptitle = dfc_label + ' vs ' + dfb_label + '\n' + t_string

    prop = dfc_filt[['date', 'jnum', 'eg']]
    base = dfb_filt[['date', 'jnum', 'eg']]

    max_mnum = dsb.mnum.max()

    if year_max:
        prop = prop[prop.date.dt.year <= year_max].copy()
        base = base[base.date.dt.year <= year_max].copy()

    jnums = np.unique(np.concatenate((pd.unique(base.jnum),
                                      pd.unique(prop.jnum))))
    num_jobs = jnums.size

    if plot_egs_sep:
        num_egplots = len(eg_list)
    else:
        num_egplots = 1

    fig, ax = plt.subplots(num_jobs, num_egplots)

    fig = plt.gcf()
    fig.set_size_inches(xsize * num_egplots, ysize * num_jobs)

    if year_max:
        min_date = base.date.min()
        max_date = pd.datetime(year_max, 12, 31)
        dates = pd.date_range(min_date, max_date, freq='M')
    else:
        dates = pd.date_range(base.date.min(), periods=max_mnum, freq='M')

    dum = pd.DataFrame(np.zeros(len(dates), dtype=int),
                       columns=['ph'], index=dates)

    # count_plot function:
    def count_plot(df, jnum, dummy, color, ax, lw, alpha, ls):
        try:
            df[df.jnum == jnum].groupby('date').size() \
                .fillna(0).astype(int) \
                .plot(c=color, lw=lw, ls=ls, alpha=alpha, ax=ax)
        except:
            dummy.ph.plot(lw=.1, c='grey', ls='solid', alpha=0)

    plot_idx = 1

    # loop through job levels
    for jnum in jnums:

        # filter for current job number
        base_jobs = base[base.jnum == jnum]
        prop_jobs = prop[prop.jnum == jnum]

        # plot each employee group on separate chart for each job level:
        if plot_egs_sep:

            # loop through employee groups
            for eg in eg_list:

                ax = plt.subplot(num_jobs, num_egplots, plot_idx)
                plt.tick_params(axis='both', which='both', labelsize=10)
                ax.xaxis.label.set_size(12)

                # show combined job level count on chart, otherwise skip over
                if plot_total:
                    # plot base df job total
                    count_plot(base_jobs, jnum, dum, total_color,
                               ax, base_lw, .7, ls=base_ls)

                    # plot proposal df job total
                    count_plot(prop_jobs, jnum, dum, total_color,
                               ax, prop_lw, 1, ls=prop_ls)

                eg_jobs = base_jobs[base_jobs.eg == eg]

                # plot employee group base job count
                count_plot(eg_jobs, jnum, dum, cf.eg_colors[eg - 1],
                           ax, base_lw, 1, ls=base_ls)

                eg_jobs = prop_jobs[prop_jobs.eg == eg]
                # plot employee group proposal job count
                count_plot(eg_jobs, jnum, dum, cf.eg_colors[eg - 1],
                           ax, prop_lw, 1, ls=prop_ls)

                plt.title(cf.eg_dict_verbose[eg] + '  ' +
                          cf.jobs_dict[jnum], fontsize=title_fontsize)
                plt.suptitle(suptitle, fontsize=suptitle_fontsize,
                             y=1.005)
                plot_idx += 1

        # plot all employee groups on same job level chart
        else:

            ax = plt.subplot(num_jobs, num_egplots, plot_idx)
            plt.tick_params(axis='both', which='both', labelsize=10)
            ax.xaxis.label.set_size(12)

            # show combined job level count on chart, otherwise skip over
            if plot_total:
                    # plot base df job total
                    count_plot(base_jobs, jnum, dum, total_color,
                               ax, base_lw, .7, ls=base_ls)

                    # plot proposal df job total
                    count_plot(prop_jobs, jnum, dum, total_color,
                               ax, prop_lw, 1, ls=prop_ls)

            # loop through employee groups
            for eg in eg_list:

                eg_jobs = base_jobs[base_jobs.eg == eg]
                count_plot(eg_jobs, jnum, dum, cf.eg_colors[eg - 1],
                           ax, base_lw, 1, ls=base_ls)

                eg_jobs = prop_jobs[prop_jobs.eg == eg]
                count_plot(eg_jobs, jnum, dum, cf.eg_colors[eg - 1],
                           ax, prop_lw, 1, ls=prop_ls)

            plt.title(cf.jobs_dict[jnum], fontsize=title_fontsize)
            plt.suptitle(suptitle, fontsize=suptitle_fontsize, y=1.005)
            plot_idx += 1

    for ax in fig.axes:
        try:
            ax.legend_.remove()
        except:
            pass

    fig.tight_layout()
    plt.show()


def build_subplotting_order(rows, cols):
    '''build a list of integers to permit passing through subplots by columns
    note:  only used when looping completes one vertical column before
    continuing to next column
    '''
    subplot_order_list = []
    for col in np.arange(1, cols + 1):
        subplot_order_list.extend(np.arange(col, (rows * cols) + 1, cols))

    return subplot_order_list


def emp_quick_glance(empkey, df, ds_dict=None,
                     title_fontsize=14, tick_size=13,
                     xsize=8, ysize=48, lw=4):
    '''view basic stats for selected employee and proposal

    A separate chart is produced for each measure.
    '''

    ds, df_label = determine_dataset(df, ds_dict, return_label=True)

    one_emp = ds[ds.empkey == empkey].set_index('date')

    cols = ['age', 'ylong', 'spcnt', 'lspcnt', 'snum', 'jnum', 'jobp',
            'cat_order', 'rank_in_job', 'job_count', 'mpay', 'cpay']

    with sns.axes_style('dark'):
        one_emp[cols].plot(subplots=True, figsize=(xsize, ysize), lw=lw)

    plt.xticks(rotation=0, horizontalalignment='center')

    fig = plt.gcf()
    i = 0
    for ax in fig.axes:
        if cols[i] in ['new_order', 'jnum', 'snum', 'spcnt', 'lnum',
                       'lspcnt', 'rank_in_job', 'job_count', 'jobp',
                       'cat_order']:
            ax.invert_yaxis()

        if i % 2 == 0:
            ax.yaxis.tick_right()
            if i == 0:
                ax.set_title(df_label + ', emp ' +
                             str(empkey), y=1.1,
                             fontsize=title_fontsize)
            else:
                ax.set_title(df_label + ', emp ' + str(empkey),
                             fontsize=title_fontsize)
        if i == 0:
            ax.xaxis.set_tick_params(labeltop='on')
        ax.grid(c='grey', alpha=.3)
        i += 1

    plt.tick_params(axis='y', labelsize=tick_size)
    plt.tick_params(axis='x', labelsize=tick_size)

    one_emp = ()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.075, wspace=0)
    plt.show()


def quartile_yrs_in_pos_single(dfc, dfb, job_levels, num_bins,
                               job_str_list,
                               eg_dict, color_list, ds_dict=None,
                               attr1=None, oper1='>=', val1=0,
                               attr2=None, oper2='>=', val2=0,
                               attr3=None, oper3='>=', val3=0,
                               fur_color='.5',
                               style='bar', plot_differential=True,
                               custom_color=False, cm_name='Dark2', start=0.0,
                               stop=1.0, flip_x=True, flip_y=False,
                               rotate=True, gain_loss_bg=False, bg_alpha=.05,
                               normalize_yr_scale=False, year_clip=30,
                               suptitle_fontsize=14, title_fontsize=12,
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

    dsc, dfc_label = determine_dataset(dfc, ds_dict, return_label=True)
    dsb, dfb_label = determine_dataset(dfb, ds_dict, return_label=True)

    dfc_filt = filter_ds(dsc,
                         attr1=attr1, oper1=oper1, val1=val1,
                         attr2=attr2, oper2=oper2, val2=val2,
                         attr3=attr3, oper3=oper3, val3=val3,
                         return_title_string=False)

    dfb_filt, t_string = filter_ds(dsb,
                                   attr1=attr1, oper1=oper1, val1=val1,
                                   attr2=attr2, oper2=oper2, val2=val2,
                                   attr3=attr3, oper3=oper3, val3=val3)

    descript = dfc_label + ' vs ' + dfb_label + '\n' + t_string

    prop_ds = dfc_filt
    sa_ds = dfb_filt

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
    tick_fontsize = (np.clip(int(ysize * 1.55), 9, 14))
    color_list.append(fur_color)

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

            plt.suptitle(dfc_label + ', GROUP ' + eg_dict[eg],
                         fontsize=suptitle_fontsize, y=1.02)

            plt.title(descript + 'years in position, ' +
                      str(num_bins) + '-quantiles',
                      fontsize=title_fontsize, y=1.02)

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend((labels), loc='center left',
                      bbox_to_anchor=(1, 0.5),
                      fontsize=legend_font_size)

            plt.tick_params(labelsize=tick_fontsize)
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
                if not flip_x:
                    ax.invert_xaxis()

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

                plt.suptitle(dfc_label +
                             ', GROUP ' + eg_dict[eg],
                             fontsize=suptitle_fontsize, y=1.02)

                plt.title(descript + 'years differential vs standalone, ' +
                          str(num_bins) + '-quantiles',
                          fontsize=title_fontsize, y=1.02)

                plt.tick_params(labelsize=tick_fontsize)
                fig = plt.gcf()
                fig.set_size_inches(xsize, ysize)
    plt.show()


def cond_test(df, opt_sel, ds_dict=None, plot_all_jobs=False, max_mnum=110,
              basic_jobs=[1, 4], enhanced_jobs=[1, 2, 7, 8], use_and=False,
              print_count_months=None, print_all_counts=False,
              xsize=8, ysize=8):
    '''visualize selected job counts applicable to computed condition with
    optional printing of certain data.

    Primary usage is testing, though the function can chart any job level(s)
    to validate proper results of job assignment conditions or to evaluate
    distribution of jobs with various proposals.  Career progression of
    employees who enjoy special job rights may be understood particularily
    well by utilizing the print_all_counts option.


    inputs
        d
            dataset(dataframe) to examine
        opt_sel (list)
            integer input(s) representing the employee group code(s) to
            select for analysis.  This argument also will accept the
            string 'sg' to select a special job rights group(s).  Multiple
            inputs are normally handled as 'or' filters, meaning an input of
            [1, 'sg'] would mean employee group 1 **or** any special job rights
            group, but can be modified to mean only group 1 **and** special job
            rights employees with the 'use_and' input.
        plot_all_jobs
            option to plot all of the job counts within dataset vs only those
            contained within the basic_jobs or enhanced_jobs inputs
        max_mnum
            integer input, only plot data through selected month
        basic_jobs
            job levels to plot if config enhanced_jobs is False
        enhanced_jobs
            job levels to plot if config enhanced_jobs is True
        use_and
            when the opt_sel input has more than one element, require filtered
            dataframe for analysis to be part of all opt_sel input sets.
        print_count_months (list)
            list of month(s) for printing job counts
        print_all_counts
            if True, print the entire job count dataframe.  Jupyter notebook
            parameters may need to be adjusted to view all data at once.

            Example:

            pd.options.display.width = 185
            pd.set_option('max_rows', 500)

    In addition to the print options, the standard output is 2 charts. The
    first chart is a line chart displaying selected job count information
    over time. The second is a stacked area chart displaying all job counts
    for the selected group(s) over time.
    '''

    d, df_label = determine_dataset(df, ds_dict, return_label=True)

    # construct a string which will be evaluated below with the 'eval'
    # statement.  This string is a slicing filter for the input dataframe, and
    # is dependent upon the 'opt_sel' and 'use_and' inputs.
    pre = 'd['
    eg_pre = '(d.eg == '
    eg_suf = ')'
    suf = '].copy()'

    if use_and:
        oper = ' & '
    else:
        oper = ' | '

    i = 1
    if len(opt_sel) > 1:
        s = ''
        while i < len(opt_sel):
            if type(opt_sel[i - 1]) == int:
                s = s + (eg_pre + str(opt_sel[i - 1]) + eg_suf + oper)
                i += 1
            elif opt_sel[i - 1] == 'sg':
                s = s + '(d.sg == 1)' + oper
                i += 1
        if type(opt_sel[i - 1]) == int:
            s = s + (eg_pre + str(opt_sel[i - 1]) + eg_suf)
        elif opt_sel[i - 1] == 'sg':
            s = '(d.sg == 1)'
    else:
        if type(opt_sel[i - 1]) == int:
            s = eg_pre + str(opt_sel[i - 1]) + eg_suf
        elif opt_sel[i - 1] == 'sg':
            s = '(d.sg == 1)'

    segment = pre + s + suf
    # Example s variable: 'd[(d.eg == 2) | (d.eg == 3)].copy()'
    df = eval(segment)

    # This groupby and unstack operation produces a monthly count of all jobs
    all_jcnts = df.groupby(['date', 'jnum']).size() \
        .unstack().fillna(0).astype(int)

    if cf.enhanced_jobs:
        tgt_cols = enhanced_jobs
        info_prefix = 'enhanced'
        info_next_line = 'modify "enhanced_jobs" input to change jobs'
    else:
        tgt_cols = basic_jobs
        info_prefix = 'basic'
        info_next_line = 'modify "basic_jobs" input to change jobs'

    print(info_prefix + ' jobs for analysis: ', tgt_cols)
    print(info_next_line)

    title = s

    # option to print a numerical tally of jobs for targeted months
    if print_count_months:
        for month in print_count_months:
            mdate = df[df.mnum == month]['date'].iloc[0].strftime('%m-%d-%Y')
            print('\nmonth ' + str(month), '(' + mdate + ') ' +
                  title + ' job count:')
            jnum_seg = np.array(df[df.mnum == month]['jnum'])
            for job in tgt_cols:
                job_count = jnum_seg[jnum_seg == job].size
                print(int(job), int(job_count))
    try:
        job_colors = np.array(cf.job_colors)[np.array(tgt_cols) - 1]
    except:
        print('\njob number error - (no matching job color), exiting...\n')
        return

    try:
        if not plot_all_jobs:

            cnd_jcnts = all_jcnts.copy()

            cnd_jcnts['mnum'] = range(len(cnd_jcnts))
            jdf = cnd_jcnts[(cnd_jcnts.mnum >= 0) &
                            (cnd_jcnts.mnum <= max_mnum)]
            jdf[tgt_cols].plot(color=job_colors, title=title)

        if plot_all_jobs:

            outall = []
            for col in all_jcnts.columns:
                try:
                    col + 0
                    outall.append(int(col))
                except:
                    pass

            all_jcnts['mnum'] = range(len(all_jcnts))
            jdf = all_jcnts[(all_jcnts.mnum >= 0) &
                            (all_jcnts.mnum <= max_mnum)]
            jdf[outall].plot(color=cf.job_colors, title=title)

    except:
        print('\n...job number error - exiting...')
        print('> verify job(s) for analysis exist within selected sample? <\n')
        return

    plt.ylim(ymin=0)
    fig = plt.gcf()
    fig.set_size_inches(xsize, ysize)
    plt.show()

    out = []
    for col in all_jcnts.columns:
        try:
            col + 0
            out.append(int(col))
        except:
            pass
    all_jcnts[out][:max_mnum].plot(kind='area',
                                   color=cf.job_colors,
                                   stacked=True,
                                   linewidth=0.1,
                                   alpha=.6)
    plt.title(title)
    ax = plt.gca()
    ax.invert_yaxis()
    fig = plt.gcf()
    fig.set_size_inches(xsize, ysize)
    plt.show()

    # option to print a dataframe containing all job counts for all months:
    if print_all_counts:
        all_jcnts_print = df.groupby(['mnum', 'date', 'jnum']).size() \
            .unstack().fillna(0).astype(int)
        # calculate a total column (total count of all jobs for each month)
        np_total = np.add.accumulate(all_jcnts_print.values, 1).T[-1]
        all_jcnts_print['total'] = np_total
        print('\n', all_jcnts_print)


def single_emp_compare(emp, measure, df_list, xax,
                       job_strs, eg_colors, eg_dict,
                       job_levels, ds_dict=None,
                       standalone_color='#ff00ff',
                       title_fontsize=14,
                       tick_size=12, label_size=13,
                       legend_fontsize=14,
                       chart_example=False):

    '''Select a single employee and compare proposal outcome using various
    calculated measures.

    inputs
        emp
            empkey for selected employee
        measure
            calculated measure to compare
            examples: 'jobp' or 'cpay'
        df_list
            list of calculated datasets to compare
        xax
            dataset column to set as x axis
        job_strs
            string job description list
        eg_colors
            list of colors to be assigned to line plots
        eg_dict
            dictionary containing eg group integer to eg string descriptions
        job_levels
            number of jobs in the model
        ds_dict (dictionary)
            output from load_datasets function
        standalone_color
            color of standalone plot
            (This function assumes one proposal from each group, any additional
            proposal is assumed to be standalone)
        title_fontsize
            text size of chart title
        tick_size
            text size of chart tick labels
        label_size
            text size of x and y axis chart labels
        legend_fontsize
            text size of chart legend
        chart_example
            option to select anonomized results for display purposes
    '''

    label_dict = {}

    i = 0
    for df in df_list:
        ds, df_label = determine_dataset(df, ds_dict,
                                         return_label=True)
        df_list[i] = ds
        label_dict[i] = df_label
        i += 1

    eg_colors.append(standalone_color)
    eg_colors.append('green')

    if chart_example:
        for i in range(0, len(df_list)):
            label = 'proposal ' + str(i + 1)
            ax = df_list[i][df_list[i].empkey == emp].set_index(xax)[measure] \
                .plot(label=label, lw=3,
                      color=eg_colors[i], alpha=.6)
        plt.title('Employee  ' + '123456' + '  -  ' + str.upper(measure),
                  y=1.02, fontsize=title_fontsize)
    else:
        for i in range(0, len(df_list)):
            ax = df_list[i][df_list[i].empkey == emp].set_index(xax)[measure] \
                .plot(label=label_dict[i], lw=3, color=eg_colors[i], alpha=.6)
        plt.title('Employee  ' + str(emp) + '  -  ' + str.upper(measure),
                  y=1.02, fontsize=title_fontsize)

    if measure in ['snum', 'cat_order', 'spcnt', 'lspcnt',
                   'jnum', 'jobp', 'fbff']:
        ax.invert_yaxis()

    if measure in ['spcnt', 'lspcnt']:
        ax.yaxis.set_major_formatter(pct_format)
        plt.axhline(y=1, c='.8', alpha=.8, lw=3)

    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:

        ax.set_yticks(np.arange(0, job_levels + 2, 1))
        yticks = ax.get_yticks().tolist()

        for i in np.arange(1, len(yticks)):
            yticks[i] = job_strs[i - 1]
        plt.axhspan(job_levels + 1, job_levels + 2, facecolor='.8', alpha=0.9)
        ax.set_yticklabels(yticks)
        plt.axhline(y=job_levels + 1, c='.8', ls='-', alpha=.8, lw=3)
        plt.ylim(job_levels + 1.5, 0.5)

    plt.tick_params(axis='y', labelsize=tick_size)
    plt.tick_params(axis='x', labelsize=tick_size)
    plt.xlabel(xax.upper(), fontsize=label_size)
    plt.ylabel(measure.upper(), fontsize=label_size)
    plt.legend(loc='best', markerscale=1.5, fontsize=legend_fontsize)
    plt.show()


def job_time_change(df_list, dfb, eg_list,
                    xax, ds_dict=None,
                    attr1=None, oper1='>=', val1=0,
                    attr2=None, oper2='>=', val2=0,
                    attr3=None, oper3='>=', val3=0,
                    colors=cf.job_colors,
                    job_list=np.arange(cf.num_of_job_levels, 0, -1),
                    jobs_dict=cf.jobs_dict,
                    marker='o', edgecolor='k', linewidth=.05, size=25,
                    alpha=.95, bg_color=None, xmax=1.02,
                    limit_yax=False, ylimit=40, zeroline_color='m',
                    zeroline_width=1.5, pos_neg_face=True,
                    legend_job_strings=True,
                    legend_position=1.18, legend_marker_size=130,
                    suptitle_fontsize=16, title_fontsize=14,
                    tick_size=12,
                    label_fontsize=13, xsize=12, ysize=10):
    '''Plots a scatter plot displaying monthly time in job
    differential, by proposal and employee group.

    inputs
        df_list (list)
            list of datasets to compare, may be ds_dict (output of
            load_datasets function) string keys or dataframe variable(s)
            or mixture of each
        dfb (string or variable)
            baseline dataset, accepts same input types as df_list above
        eg_list (list)
            list of integers for employee groups to be included in analysis
            example: [1, 2, 3]
        xax
            list percentage attrubute, i.e. spcnt or lspcnt
        ds_dict
            output from load_datasets function
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
        colors
            chart color for the job levels
        job_list
            list of integers representing job levels in model
            (a descending list will plot best jobs last, or on
            top of other jobs on chart)
        jobs_dict
            job number to job label dictionary
        marker
            scatter chart matplotlib marker type
        edgecolor
            matplotlib marker edge color
        linewidth
            matplotlib marker edge line size
        size
            size of markers
        alpha
            marker alpha (transparency) value
        bg_color
            background color of chart if not None
        xmax
            high limit of chart x axis
        limit_yax
            if True, restrict plot y scale to this value
            may be used to prevent outliers from exagerating chart scaling
        ylimit
            y axis limit if limit_yax is True
        zeroline_color
            color for zeroline on chart
        zeroline_width
            width of zeroline
        pos_neg_face
            if True, apply a light green tint to the chart area above the
            zero line, and a light red tint below the line
        legend_job_strings
            if True, use job description strings in legend vs. job numbers
        legend_position
            controls the horizontal position of the legend
        legend_marker_size
            adjusts the size of the legend markers
        suptitle_fontsize
            text size of chart super title
        title_fontsize
            text size of chart title
        tick_size
            text size of chart tick labels
        xsize, ysize
            x and y size of each plot
    '''
    label_dict = {}
    i = 0
    for df in df_list:
        ds, df_label = determine_dataset(df, ds_dict,
                                         return_label=True)
        label_dict[i + 1] = df_label

        df_list[i] = filter_ds(ds,
                               attr1=attr1, oper1=oper1, val1=val1,
                               attr2=attr2, oper2=oper2, val2=val2,
                               attr3=attr3, oper3=oper3, val3=val3,
                               return_title_string=False)
        i += 1

    df_base, dfb_label = determine_dataset(dfb, ds_dict, return_label=True)

    dfb_filt, tb_string = filter_ds(df_base,
                                    attr1=attr1, oper1=oper1, val1=val1,
                                    attr2=attr2, oper2=oper2, val2=val2,
                                    attr3=attr3, oper3=oper3, val3=val3)

    df_frames = od()
    df_dict = od()
    # sorts index by empkey, this is base df with key 0
    df_dict[0] = dfb_filt.groupby(['empkey', 'jnum']).size()\
        .unstack().fillna(0)
    i = 1
    for df in df_list:
        df_frames[i] = df[df.mnum == 0][[xax, 'eg']]
        # sorts index by empkey
        df_dict[i] = df.groupby(['empkey', 'jnum']).size().unstack().fillna(0)
        i += 1

    # get keys for comparative dataframes (all but key 0)
    compare_keys = [x for x in list(df_dict.keys()) if x > 0]
    diff_dict = od()
    joined_dict = od()
    for i in compare_keys:
        # create diff dataframes - all comparative dataframes minus base
        diff_dict[i] = df_dict[i] - df_dict[0]
        # bring in index from original dataframes
        empty = df_frames[i][[]]
        # auto-align (order) with index-only dataframes with join
        joined_dict[i] = empty.join(diff_dict[i])
        # sort columns
        joined_dict[i].sort_index(axis=1, inplace=True, ascending=False)
        # add xax and eg columns
        joined_dict[i] = joined_dict[i] \
            .join(df_frames[i])
        # do not include employees with no change (0) in chart
        joined_dict[i] = joined_dict[i].replace(0, np.nan)

    eg_count = len(eg_list)
    diff_keys = list(joined_dict.keys())

    diff_count = len(diff_keys)
    # set up subplot count
    fig, ax = plt.subplots(eg_count * diff_count, 1)
    fig.set_size_inches(xsize, ysize * eg_count * diff_count)

    plot_num = 0
    if legend_job_strings:
        legend_position = legend_position + .05

    for j in diff_keys:
        for eg in eg_list:
            # increment plot number
            plot_num += 1
            ax = plt.subplot(eg_count * diff_count, 1, plot_num)
            # filter for eg
            eg_df = joined_dict[j][joined_dict[j].eg == eg]
            for i in job_list:
                try:
                    eg_df.plot(kind='scatter',
                               x=xax,
                               y=i,
                               color=colors[i - 1],
                               edgecolor=edgecolor,
                               marker=marker,
                               linewidth=linewidth,
                               s=size,
                               alpha=alpha,
                               label=str(i),
                               ax=ax)
                except:
                    pass

            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels (converted to int)
            # from lowest job number(best) to highest (worst) for
            # for legend
            handles, labels = zip(*sorted(zip(handles, labels),
                                  key=lambda x: int(x[1])))
            # use job string descriptions in legend vs. job numbers
            if legend_job_strings:
                label_strs = []
                for label in labels:
                    label_strs.append(jobs_dict[int(label)])
                    labels = label_strs
            # move legend off of chart face to right
            legend_title = 'job'
            # use sorted labels and handlers as legend input
            # and position legend
            lgnd = ax.legend(handles, labels,
                             bbox_to_anchor=(legend_position, 1),
                             title=legend_title, fontsize=12)

            lgnd.get_title().set_fontsize('16')

            # set legend marker size
            for mark in lgnd.legendHandles:
                mark._sizes = [legend_marker_size]

            plt.xlim(xmin=0, xmax=1.02)
            plt.axhline(c=zeroline_color, lw=zeroline_width)

            ax.xaxis.set_major_formatter(pct_format)
            ax.set_xticks(np.arange(0, 1.05, .05))

            plt.tick_params(labelsize=13, labelright=True)
            plt.ylabel('months differential', fontsize=label_fontsize)
            plt.xlabel('proposed list percentage', fontsize=label_fontsize)
            plt.title('Months in job differential, ' +
                      label_dict[j] + ', eg ' + str(eg),
                      fontsize=title_fontsize)
            if limit_yax:
                plt.ylim(-ylimit, ylimit)
            if pos_neg_face:
                ymin, ymax = ax.get_ylim()
                plt.ylim(ymin, ymax)
                plt.axhspan(0, ymax, facecolor='g', alpha=0.02, zorder=8)
                plt.axhspan(0, ymin, facecolor='r', alpha=0.02, zorder=8)
            if bg_color:
                ax.set_axis_bgcolor(bg_color)
            plt.tick_params(axis='y', labelsize=tick_size)
            plt.tick_params(axis='x', labelsize=tick_size)
            plt.grid(linestyle='dotted', lw=1.5)
            ax.invert_xaxis()


# EMPLOYEE_GROUP_ATTRIBUTE_AVERAGE_AND_MEDIAN
def group_average_and_median(dfc, dfb, eg_list, colors,
                             measure, job_levels, job_dict,
                             ds_dict=None,
                             attr1=None, oper1='>=', val1='0',
                             attr2=None, oper2='>=', val2='0',
                             attr3=None, oper3='>=', val3='0',
                             plot_median=False, plot_average=True,
                             compare_to_dfb=True,
                             use_filtered_results=True,
                             job_labels=True,
                             max_date=None, chart_style='whitegrid',
                             legend_horizontal_position=1.4,
                             xsize=12, ysize=8,):
    '''Plot group average and/or median for a selected attribute over time
    for compare and/or base datasets.  Standalone data may be used as compare
    or baseline data.

    Results may be further filtered/sliced by up to 3 constraints,
    such as age, longevity, or job level.

    This function can plot basic data such as average list percentage or could,
    for example, plot the average job category rank for employees hired prior
    to a certain date who are over or under a certain age, for a selected
    integrated dataset and/or standalone data (or for two integrated datasets).

    inputs
        dfc
            comparative dataset to examine, may be a dataframe variable
            or a string key from the ds_dict dictionary object
        dfb
            baseline dataset to plot (likely use standalone
            dataset here for comparison, but may plot and compare any dataset),
            may be a dataframe variable or a string key from the ds_dict
            dictionary object
        eg_list
            list of integers representing the employee groups to analyze
            (i.e. [1, 2])
        colors
            list of colors for plotting, normally cf.eg_colors
        measure
            attribute (column) to compare, such as 'spcnt' or 'jobp'
        job_levels
            number of job levels in the model, normally cf.num_of_job_levels
        job_dict
            dictionary of integer job codes to string description.
            normally cf.jobs_dict
        ds_dict
            dataset dictionary (variable assigned to the output of
            load_datasets function)
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
        plot_meadian
            plot the median of the measure for each employee group
        plot_average
            plot the average(mean) of the measure for each employee group
        compare_to_dfb
            plot average dfb[measure] data as dashed line.
            (likely show standalone data with dfb, or reverse and show
            standalone as primary and integrated as dfb)
            (dfb refers to baseline dataframe or dataset)
        use_filtered_results
            if True, use the same employees from the filtered proposal list.
            For example, if the dfc list is filtered by age only, the
            dfb list could be filtered by the same age and return the
            same employees.  However, if the dfc list is filtered by
            an attribute which diverges from the dfb measurements for
            the same attribute, a different set of employees could be returned.
            This option ensures that the same group of employees from both the
            dfc (filtered first) list and the dfb list are compared.
            (dfc refers to comparisonproposal, dfb refers to baseline)
        job_labels
            if measure input is one of these: 'jnum', 'nbnf', 'jobp', 'fbff',
            use job text description labels vs. number labels on the y axis
            of the chart (boolean)
        max_date
            maximum chart date.  If set to 'None', the maximum chart date will
            be the maximum date within the list data.
        chart_style
            option to specify alternate seaborn chart style
    '''
    # helper function to provide proper input for eval statements

    dsc, dfc_label = determine_dataset(dfc, ds_dict, return_label=True)
    dsb, dfb_label = determine_dataset(dfb, ds_dict, return_label=True)

    if max_date:
        dfc = dfc[dfc.date <= max_date]
        dfb = dfb[dfb.date <= max_date]

    dfc = filter_ds(dsc,
                    attr1=attr1, oper1=oper1, val1=val1,
                    attr2=attr2, oper2=oper2, val2=val2,
                    attr3=attr3, oper3=oper3, val3=val3,
                    return_title_string=False)

    dfb, t_string = filter_ds(dsb,
                              attr1=attr1, oper1=oper1, val1=val1,
                              attr2=attr2, oper2=oper2, val2=val2,
                              attr3=attr3, oper3=oper3, val3=val3)

    if plot_average:
        if plot_median:
            plot_string = ' avg/median '
        else:
            plot_string = ' average '
    elif plot_median:
            plot_string = ' median '
    else:
        plot_string = ' <set plot_average or plot_median to True> '

    title_string = ''

    if measure == 'mpay':
        # eliminate variable employee last month partial pay
        dfc = dfc[dfc.ret_mark == 0].copy()
        dfb = dfb[dfb.ret_mark == 0].copy()

    with sns.axes_style(chart_style):
        fig, ax = plt.subplots()

    for eg in eg_list:
        # for each employee group, group by date and plot avg/median of measure
        try:
            if plot_average:
                dfc[dfc.eg == eg].groupby('date')[measure] \
                    .mean().plot(color=colors[eg - 1], lw=3,
                                 ax=ax, label=dfc_label +
                                 ', ' +
                                 'grp' + cf.eg_dict[eg] + ' avg')
            if plot_median:
                dfc[dfc.eg == eg].groupby('date')[measure] \
                    .median().plot(color=colors[eg - 1],
                                   ax=ax, lw=1,
                                   label=dfc_label +
                                   ', ' +
                                   'grp' + cf.eg_dict[eg] + ' median')
        except:
            print('invalid or missing data - dfc, group ' + str(eg))

    if compare_to_dfb:

        if use_filtered_results:

            title_string = title_string + \
                '  (dfb employees match dfc filtered group)'
            # perform join to grab same employees as filtered proposal list
            try:
                dfb = dfc.set_index(['mnum', 'empkey'],
                                    verify_integrity=False)[['date', 'eg']] \
                    .join(dfb.set_index(['mnum', 'empkey'],
                                        verify_integrity=False)[measure])
            except:
                print('dfb data not found')

        else:
            # if join above not needed...
            title_string = title_string + \
                '  (dfb employees independently filtered)'

        for eg in eg_list:
            try:
                if plot_average:
                    dfb[dfb.eg == eg].groupby('date')[measure]\
                        .mean().plot(label=dfb_label +
                                     ', ' +
                                     'grp' + cf.eg_dict[eg] +
                                     ' avg',
                                     color=colors[eg - 1],
                                     ls='dashed',
                                     lw=2.5,
                                     alpha=.5,
                                     ax=ax)
                if plot_median:
                    dfb[dfb.eg == eg].groupby('date')[measure]\
                        .median().plot(label=dfb_label +
                                       ', ' +
                                       'grp' + cf.eg_dict[eg] +
                                       ' median',
                                       color=colors[eg - 1],
                                       ls='dotted',
                                       lw=2.5,
                                       alpha=.5,
                                       ax=ax)
            except:
                print('invalid or missing data - dfb, group ' + str(eg))

        # snippit below for no label with plots...
        # label='_nolabel_',

    if measure in ['snum', 'spcnt', 'lspcnt', 'jnum',
                   'lnum', 'jobp', 'fbff', 'cat_order']:
        ax.invert_yaxis()

    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:
        if job_labels:

            ax.set_yticks(np.arange(0, job_levels + 2, 1))
            yticks = ax.get_yticks().tolist()

            for i in np.arange(1, len(yticks)):
                yticks[i] = job_dict[i]
            plt.axhspan(job_levels + 1, job_levels + 2,
                        facecolor='.8', alpha=0.9)
            ax.set_yticklabels(yticks)
            plt.axhline(y=job_levels + 1, c='.8', ls='-', alpha=.8, lw=3)
            plt.ylim(job_levels + 1.5, 0.5)
        else:
            plt.ylim(ymax=0)

    if cf.delayed_implementation:
        # plot vertical line at implementation date
        plt.axvline(cf.imp_date, c='#33cc00', ls='dashed', alpha=1, lw=1,
                    label='implementation date', zorder=1)

    if compare_to_dfb:
        suptitle_string = (dfc_label +
                           plot_string.upper() + measure.upper() +
                           ' vs. ' + dfb_label)
    else:
        suptitle_string = (dfc_label +
                           plot_string.upper() + measure.upper())

    plt.suptitle(suptitle_string, fontsize=16)
    plt.title(t_string + ' ' + title_string, y=1.02, fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    # move legend off of chart face to right
    ax.legend(handles, labels,
              bbox_to_anchor=(legend_horizontal_position, .8),
              fontsize=14)
    plt.gcf().set_size_inches(xsize, ysize)

    plt.show()


# EMPLOYEE DENSITY STRIPPLOT (with filtering)
def stripplot_eg_density(df, mnum, eg_colors, ds_dict=None,
                         attr1=None, oper1='>=', val1=0,
                         attr2=None, oper2='>=', val2=0,
                         attr3=None, oper3='>=', val3=0,
                         bg_color='white', title_fontsize=12,
                         suptitle_fontsize=14,
                         xsize=5, ysize=10):
    '''plot a stripplot showing density distribution for each employee group
    separately.

    The input dataframe (df) may be a dictionary key (string) or a pandas
    dataframe.

    The input dataframe may be filtered by attributes using the attr(x),
    oper(x), and val(x) inputs.

    inputs
        df (string or dataframe)
            text name of input proposal dataset, also will accept any dataframe
            variable (if a sliced dataframe subset is desired, for example)
            Example:  input can be 'proposal1' (if that proposal exists, of
            course, or could be df[df.age > 50])
        mnum (int)
            month number to study from dataset
        eg_colors (list)
            color codes for plotting each employee group
        ds_dict (dictionary)
            output from load_datasets function
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
        bg_color
            chart background color
        title_fontsize
            chart title text size
        xsize, ysize
            size of chart width and height
    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)
    ds = ds[ds.mnum == mnum]
    d_filt, t_string = filter_ds(ds,
                                 attr1=attr1, oper1=oper1, val1=val1,
                                 attr2=attr2, oper2=oper2, val2=val2,
                                 attr3=attr3, oper3=oper3, val3=val3)

    d_filt = d_filt[['eg']].copy()
    d_filt.rename(columns={'eg': 'filt_eg'}, inplace=True)

    mnum_p = ds[['eg', 'new_order']].join(d_filt)
    mnum_p['eg'] = mnum_p.filt_eg

    min_eg = min(np.unique(mnum_p.eg))
    max_eg = max(np.unique(mnum_p.eg))

    try:
        with sns.axes_style("whitegrid"):
            ax = sns.stripplot(y='new_order', x='eg', data=mnum_p, jitter=.5,
                               order=np.arange(min_eg, max_eg + 1),
                               palette=eg_colors, size=3, linewidth=0,
                               split=True)
            ax.set_axis_bgcolor(bg_color)
    except:
        print('\nEmpty dataset, nothing to plot. Check filters?\n')
        return

    fig = plt.gcf()
    fig.set_size_inches(xsize, ysize)
    plt.ylim(max(mnum_p.new_order), 0)
    if t_string:
        plt.suptitle(df_label, fontsize=suptitle_fontsize)
        plt.title(t_string, fontsize=title_fontsize, y=1.02)
    else:
        plt.title(df_label, fontsize=suptitle_fontsize)
    plt.show()


def job_count_bands(df_list, eg_list, job_colors,
                    ds_dict=None,
                    attr1=None, oper1='>=', val1=0,
                    attr2=None, oper2='>=', val2=0,
                    attr3=None, oper3='>=', val3=0,
                    fur_color='.5',
                    max_date=None, plot_alpha=.75,
                    legend_alpha=.9,
                    legend_xadj=1.3, legend_yadj=1.0,
                    legend_fontsize=11, title_fontsize=14,
                    tick_size=12, label_size=13, chart_style='darkgrid',
                    xsize=6, ysize=6):
    '''area chart representing count of jobs available over time

    This chart displays the future job opportunities for each employee group
    with various list proposals.

    This is not a comparative chart (for example, with standalone data), it
    is simply displaying job count outcome over time.
    However, the results for the employee groups may be compared and measured
    for equity.

    Inputs
        df_list (list)
            list of datasets to compare, may be ds_dict (output of
            load_datasets function) string keys or dataframe variable(s)
            or mixture of each
        eg_list (list)
            list of integers for employee groups to be included in analysis
            example: [1, 2, 3]
        job_colors (list)
            list of colors to represent job levels
        ds_dict (dictionary)
            output from load_datasets function
        attr(n)
            filter attribute or dataset column as string
        oper(n)
            operator (i.e. <, >, ==, etc.) for attr1 as string
        val(n)
            attr1 limiting value (combined with oper1) as string
        fur_color
            color to use for furlough band
        max_date (date string)
            only include data up to this date
            example input: '1997-12-31'
        plot_alpha (float, 0.0 to 1.0)
            alpha value (opacity) for area plot (job level bands)
        legend_alpha (float, 0.0 to 1.0)
            alpha value (opacity) for legend markers
        legend_xadj, legend_yadj (floats)
            adjustment input for legend horizontal and vertical placement
        legend_fontsize
            text size of legend labels
        title_fontsize
            text size of chart title
        tick_size
            text size of x and y tick labels
        label_size
            text size of x and y descriptive labels
        chart_style (string)
            chart styling (string), any valid seaborn chart style
        xsize, ysize (integer or float)
            plot size in inches
    '''
    label_dict = {}
    i = 0
    for df in df_list:
        ds, df_label = determine_dataset(df, ds_dict,
                                         return_label=True)
        if max_date:
            ds = ds[ds.date <= max_date]

        label_dict[i] = df_label

        df_list[i] = filter_ds(ds,
                               attr1=attr1, oper1=oper1, val1=val1,
                               attr2=attr2, oper2=oper2, val2=val2,
                               attr3=attr3, oper3=oper3, val3=val3,
                               return_title_string=False)
        i += 1

    job_colors.append(fur_color)

    for eg in eg_list:
        i = 0

        for df_object in df_list:

            df = df_object[df_object.eg == eg]
            y = len(df[df.mnum == 0]) + 50

            df = df.groupby(['date', 'jnum']).size()
            df = pd.DataFrame(df.unstack().fillna(0))
            cols = list(df.columns)

            plot_colors = [job_colors[j - 1] for j in cols]

            with sns.axes_style(chart_style):

                df.plot(kind='area', stacked=True,
                        color=plot_colors, linewidth=0, alpha=plot_alpha)

            plt.ylim(y, 0)

            plt.title(label_dict[i] + ', group ' + str(eg),
                      fontsize=title_fontsize, y=1.01)
            fig = plt.gcf()
            ax = plt.gca()
            fig.set_size_inches(xsize, ysize)

            # legend
            recs = []
            job_labels = []
            legend_title = 'job'

            for k in cols:
                recs.append(mpatches.Rectangle((0, 0), 1, 1,
                                               fc=job_colors[k - 1],
                                               alpha=legend_alpha))
                job_labels.append(cf.job_strs[k - 1])

            box = ax.get_position()
            ax.set_position([box.x0, box.y0,
                            box.width * legend_xadj, box.height])
            ax.legend(recs, job_labels, bbox_to_anchor=(legend_xadj,
                                                        legend_yadj),
                      title=legend_title, fontsize=legend_fontsize,
                      ncol=1)
            ax.xaxis.label.set_size(label_size)
            plt.tick_params(axis='y', labelsize=tick_size)
            plt.tick_params(axis='x', labelsize=tick_size)
            plt.show()
            i += 1


def determine_dataset(ds_def, ds_dict, return_label=False):
    '''this function permits either a dictionary key (string) or a dataframe
    variable to be used in functions as a dataframe object.
    '''
    if type(ds_def) == pd.core.frame.DataFrame:
        ds = ds_def
        ds_label = 'Proposal'
    else:
        if ds_dict:
            ds = ds_dict[ds_def][0]
            ds_label = ds_dict[ds_def][1]
        else:
            print('''using a dictionary key as df input, but "ds_dict"
        is undefined, please set "ds_dict" keyword variable''')
            return

    if return_label:
        return ds, ds_label
    else:
        return ds


def numeric_test(value):
    try:
        float(value)
        return True
    except:
        return False


def filter_ds(ds,
              attr1=None, oper1=None, val1=None,
              attr2=None, oper2=None, val2=None,
              attr3=None, oper3=None, val3=None,
              return_title_string=True):

    '''Filter a dataset (dataframe) by attribute(s).

    Filter process is ignored if attr(x) input is None.
    All attr, oper, and val inputs must be strings.
    Up to 3 attribute filters may be combined.

    If return_title_string, returns tuple (ds, title_string), otherwise
    returns ds.
    '''

    title_string = ''

    if any([attr1, attr2, attr3]):

        if attr1:

            title_string = title_string + \
                attr1 + ' ' + oper1 + ' ' + str(val1)

            if not numeric_test(val1):
                val1_text = "'" + val1 + "'"
            else:
                val1_text = str(val1)
            try:
                # slice proposal dataset according to attr1 inputs
                ds = ds[eval('ds[attr1]' + oper1 + val1_text)].copy()
            except:
                print('''attr1 filter error - filter ignored
                      ensure filter inputs are strings''')
        if attr2:

            title_string = title_string + ', ' + \
                attr2 + ' ' + oper2 + ' ' + str(val2)

            if not numeric_test(val2):
                val2_text = "'" + val2 + "'"
            else:
                val2_text = str(val2)
            try:
                ds = ds[eval('ds[attr2]' + oper2 + val2_text)].copy()
            except:
                print('''attr2 filter error - filter ignored
                      ensure filter inputs are strings''')
        if attr3:

            title_string = title_string + ', ' + \
                attr3 + ' ' + oper3 + ' ' + str(val3)

            if not numeric_test(val3):
                val3_text = "'" + val3 + "'"
            else:
                val3_text = str(val3)
            try:
                ds = ds[eval('ds[attr3]' + oper3 + val3_text)].copy()
            except:
                print('''attr3 filter error - filter ignored
                      ensure filter inputs are strings''')

    if return_title_string:
        return ds, title_string
    else:
        return ds


# This is used with the EDITOR_TOOL notebook...
def display_proposals():
    '''print out a list of the proposal names which were generated and stored
    in the dill folder by the build_program_files script

    no inputs
    '''
    print('proposal list:')
    print(list(pd.read_pickle('dill/proposal_names.pkl').proposals))


def slice_ds_by_index_array(df, mnum=0, attr='age',
                            attr_oper='>=', attr_val=50):
    '''filter an entire dataframe by only selecting rows which match
    the filtered results from a target month.  In other words, zero in on
    a slice of data from a particular month, such as employees holding a
    specific job in month 25.  Then, using the index of those results,
    find only those employees within the entire dataset as an input for further
    analysis within the program.

    inputs
        df
            the dataframe (dataset) to be filtered
        mnum
            month number of the data to filter
        attr
            attribute (column) to use during filter
        oper
            operator to use, such as '<=' or '!='
        attr_val
            attribute value, combined with the operator for filtering.
            Example filter:
                jnum >= 7 (in mnum month)
    '''
    mnum = str(mnum)
    try:
        attr_val = str(int(attr_val))
    except:
        pass
    month_slice_indexes = \
        np.array(df[eval('(df[attr]' + attr_oper + attr_val +
                         ') & (df.mnum == ' + mnum + ')')].index)

    df_index = np.array(df.index)

    ds_filter = df[np.in1d(df_index, month_slice_indexes)]

    return ds_filter


def mark_quartiles(df, quartiles=10):
    '''add a column to the input dataframe identifying quartile membership as
    integers (the column is named "quartile").  The quartile membership
    (category) is calculated for each employee group separately, based on
    the employee population in month zero.

    The output dataframe permits attributes for employees within month zero
    quartile categories to be be analyzed throughout all the months of the
    data model.

    The number of quartiles to create within each employee group is selected
    by the "quartiles" input.

    The function utilizes numpy arrays and functions to compute the quartile
    assignments, and pandas index data alignment feature to assign month zero
    quartile membership to the long-form, multi-month output dataframe.

    inputs
        df (pandas dataframe)
            Any pandas dataframe containing an "eg" (employee group) column
        quartiles (integer)
            The number of quartiles to create.

            example:

            If the input is 10, the output dataframe will be a column of
            integers 1 - 10.  The count of each integer will be the same.
            The first quantile members will be marked with a 1, the second
            with 2, etc., through to the last quantile, 10.

    '''
    mult = 1000
    mod = mult / quartiles
    aligned_df = df.copy()
    df = df[df.mnum == 0][['eg']].copy()
    eg_arr = np.array(df.eg)
    bins_arr = np.zeros_like(eg_arr)
    unique_egs = np.arange(eg_arr.max()) + 1
    for eg in unique_egs:
        eg_count = eg_arr[eg_arr == eg].size
        this_eg_arr = np.clip((np.arange(eg_count) + 1) / eg_count, 0, .9999)
        this_bin_arr = (this_eg_arr * mult // mod).astype(int) + 1
        np.put(bins_arr, np.where(eg_arr == eg)[0], this_bin_arr)

    df['quartile'] = bins_arr
    aligned_df['quartile'] = df['quartile']
    return aligned_df

