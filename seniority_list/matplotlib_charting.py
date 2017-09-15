#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# seniority_list is an analytical tool used when seniority-based work
# groups merge. It brings modern data science to the area of labor
# integration, utilizing the powerful data analysis capabilities of Python
# scientific computing.

# Copyright (C) 2016-2017  Robert E. Davison, Ruby Data Systems Inc.
# Please direct consulting inquires to: rubydatasystems@fastmail.net

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
.. module:: matplotlib_charting

   :synopsis: The matplotlib_charting module contains plotting functions
   and supporting utility functions.

.. moduleauthor:: Bob Davison <rubydatasystems@fastmail.net>

'''
import pandas as pd
import numpy as np
import seaborn as sns
import math
from os import system, path, remove, makedirs
import sys

import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib import colors as mplclrs
from matplotlib import dates as mdate
import matplotlib.patches as mpatches

from time import sleep

from cycler import cycler
from ipywidgets import interactive, Button, Layout, Box, HBox, VBox, Label, \
    Checkbox, Dropdown, Text, IntSlider, IntRangeSlider, FloatRangeSlider
from IPython.display import display, Javascript
from collections import OrderedDict as od

from pandas.plotting import parallel_coordinates
import functions as f


# TO_PERCENT (matplotlib percentage axis)
def to_percent(decimal, position, precision=0):
    '''Custom format for matplotlib axis as a percentage.

    Ignores the passed in position variable.  This has the effect of scaling
    the default tick locations.

    inputs
        decimal (axis values)
            no user input
        position
            ignored
        precision (integer)
            number of decimals in output percentage labels
    '''
    fmt_str = '{0:.' + str(precision) + 'f}%'
    pcnt_format = fmt_str.format(decimal * 100)
    return pcnt_format


def pct_format():
    '''Apply "to_percent" custom format for chart tick labels
    '''
    return ticker.FuncFormatter(to_percent)


def quantile_years_in_position(dfc, dfb,
                               job_levels,
                               num_bins,
                               job_str_list,
                               p_dict,
                               color_list,
                               style='bar',
                               plot_differential=True,
                               ds_dict=None,
                               attr1=None, oper1='>=', val1=0,
                               attr2=None, oper2='>=', val2=0,
                               attr3=None, oper3='>=', val3=0,
                               chart_style='darkgrid',
                               grid_alpha=None,
                               custom_color=False,
                               cm_name='Dark2',
                               start=0.0, stop=1.0,
                               fur_color=None,
                               flip_x=False,
                               flip_y=False,
                               rotate=False,
                               gain_loss_bg=False,
                               bg_alpha=.05,
                               normalize_yr_scale=False,
                               year_clip=30,
                               suptitle_size=14,
                               title_size=12,
                               xsize=12, ysize=12,
                               image_dir=None,
                               image_format='png'):
    '''stacked bar or area chart presenting the time spent in the various
    job levels for quantiles of a selected employee group.

    inputs
        dfc (string or dataframe variable)
            text name of proposal (comparison) dataset to explore (ds_dict key)
            or dataframe
        dfb (string or dataframe variable)
            text name of baseline dataset to explore (ds_dict key)
            or dataframe
        job_levels (integer)
            the number of job levels in the model
        num_bins (integer)
            the total number of segments (divisions of the population) to
            calculate and display
        job_str_list (list)
            a list of strings which correspond with the job levels, used for
            the chart legend
            example:
                jobs = ['Capt G4', 'Capt G3', 'Capt G2', ....]
        p_dict (dictionary)
            dictionary used to convert employee group numbers to text,
            used with chart title text display
        color_list (list)
            a list of color codes for the job level color display
        style (string)
            option to select 'area' or 'bar' to determine the type
            of chart output. default is 'bar'.
        plot_differential (boolean)
            if True, plot the difference between dfc and dfb values
        ds_dict (dictionary)
            variable assigned to the output of the load_datasets function.
            This keyword variable must be set if string dictionary keys are
            used as inputs for the dfc and/or dfb inputs.
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (integer, float, date as string, string (as appropriate))
            attr(n) limiting value (combined with oper(n)) as string
        chart_style (string)
            any valid seaborn plotting style name
        custom_color, cm_name, start, stop (boolean, string, float, float)
            if custom color is set to True, create a custom color map from
            the cm_name color map style.  A portion of the color map may be
            selected for customization using the start and stop inputs.
        fur_color (color code in rgba, hex, or string style)
            custom color to signify furloughed employees (otherwise, last
            color in color_list input will be used)
        flip_x (boolean)
            'flip' the chart horizontally if True
        flip_y (boolean)
            'flip' the chart vertically if True
        rotate (boolean)
            transpose the chart output
        gain_loss_bg (boolean)
            if True, apply a green and red background to the
            chart in the gain and loss areas
        bg_alpha (float)
            the alpha of the gain_loss_bg (if selected)
        normalize_yr_scale (boolean)
            set all output charts to have the same x axis range
        yr_clip (integer)
            max x axis value (years) if normalize_yr_scale set True
        suptitle_size (integer or float)
            text size of chart super title
        title_size (integer or float)
            text size of chart title
        xsize, ysize (integer or float)
            size of chart display
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''

    dsc, df_labelc = determine_dataset(dfc, ds_dict,
                                       return_label=True)
    dsb = determine_dataset(dfb, ds_dict,
                            return_label=False)

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

    mnum0 = ds_sel_cols[ds_sel_cols.mnum == 0][[]]
    mnum0['order'] = np.arange(len(mnum0)) + 1

    egs = sorted(list(set(ds_sel_cols.eg)))

    legend_size = np.clip(int(ysize * .8), 12, 18)
    tick_size = (np.clip(int(ysize * .55), 11, 14))
    label_size = (np.clip(int(ysize * .8), 14, 16))

    num_rows = len(egs)
    num_cols = 2 if plot_differential else 1

    fig, ax = plt.subplots(num_rows, num_cols)
    plot_num = 1

    if custom_color:
        num_of_colors = job_levels + 1
        cm_subsection = np.linspace(start, stop, num_of_colors)
        colormap = eval('cm.' + cm_name)
        color_list = [colormap(x) for x in cm_subsection]

    if fur_color:
        # insert custom color for furloughed employees...
        color_list[-1] = fur_color

    for eg in egs:

        ds_eg = ds_sel_cols[(ds_sel_cols.eg == eg) & (ds_sel_cols.jnum >= 1)]

        job_counts_by_emp = ds_eg.groupby([pd.Grouper('empkey'),
                                          'jnum']).size()

        months_in_jobs = job_counts_by_emp.unstack() \
            .fillna(0).sort_index(axis=1, ascending=True).astype(int)

        months_in_jobs = months_in_jobs.join(mnum0[['order']], how='left')
        months_in_jobs.sort_values(by='order', inplace=True)
        months_in_jobs.pop('order')

        bin_lims = pd.qcut(np.arange(len(months_in_jobs)),
                           num_bins,
                           retbins=True,
                           labels=np.arange(num_bins) + 1)[1].astype(int)

        cols = months_in_jobs.columns.values.tolist()
        result_arr = np.zeros((num_bins, len(cols)))

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

            sa_job_counts_by_emp = sa_eg.groupby([pd.Grouper('empkey'),
                                                 'jnum']).size()

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

        with sns.axes_style(chart_style):

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
                                  edgecolor='k', linewidth=.5,
                                  stacked=True, color=colors, ax=ax)

            if normalize_yr_scale:
                if rotate:
                    ax.set_xlim(0, year_clip)
                else:
                    ax.set_ylim(ymax=year_clip)

            if style == 'bar':

                if not flip_y:
                    ax.invert_yaxis()

                if rotate:
                    ax.set_xlabel('years', fontsize=label_size)
                    ax.set_ylabel('quantiles', fontsize=label_size)
                else:
                    ax.set_ylabel('years', fontsize=label_size)
                    ax.set_xlabel('quantiles', fontsize=label_size)
                    ax.set_xticklabels(ax.xaxis.get_ticklabels(),
                                       rotation='horizontal')

            if flip_x:
                ax.invert_xaxis()

            ax.set_title('group ' + str(eg), fontsize=label_size)
            if grid_alpha:
                ax.grid(alpha=grid_alpha)
            ax.legend_.remove()

            ax.tick_params(axis='y', labelsize=tick_size)
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
                              edgecolor='k', linewidth=.5,
                              stacked=True, color=colors, ax=ax)

                if rotate:
                    ax.set_xlabel('years', fontsize=label_size)
                    ax.set_ylabel('quantiles', fontsize=label_size)
                    if normalize_yr_scale:
                        ax.set_xlim(year_clip / -3, year_clip / 3)
                    if not flip_y:
                        ax.invert_yaxis()
                    x_min, x_max = ax.get_xlim()
                    if gain_loss_bg:
                        ax.axvspan(0, x_max, facecolor='g', alpha=bg_alpha)
                        ax.axvspan(0, x_min, facecolor='r', alpha=bg_alpha)
                else:
                    ax.set_ylabel('years', fontsize=label_size)
                    ax.set_xlabel('quantiles', fontsize=label_size)
                    if normalize_yr_scale:
                        ax.set_ylim(year_clip / -3, year_clip / 3)
                    if flip_y:
                        ax.invert_yaxis()
                    ymin, ymax = ax.set_ylim()
                    if gain_loss_bg:
                        ax.axhspan(0, ymax, facecolor='g', alpha=bg_alpha)
                        ax.axhspan(0, ymin, facecolor='r', alpha=bg_alpha)
                    ax.invert_xaxis()

                ax.set_title('group ' + str(eg), fontsize=label_size)
                ax.tick_params(axis='y', labelsize=tick_size)
                ax.legend_.remove()
                if grid_alpha:
                    ax.grid(alpha=grid_alpha)
                plot_num += 1

    fig.suptitle(df_labelc + ', ' + t_string,
                 fontsize=suptitle_size, y=1.01)

    if not plot_differential:
        xsize = xsize * .5
    fig.set_size_inches(xsize, ysize)
    plt.tight_layout()

    for ax in fig.axes:
        if len(ax.get_xticks()) > 20:
            for label in ax.xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)
        if len(ax.get_yticks()) > 20:
            for label in ax.yaxis.get_ticklabels()[1::2]:
                label.set_visible(False)

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
               fontsize=legend_size)

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def age_vs_spcnt(df, eg_list, mnum, color_list,
                 p_dict, ret_age, ds_dict=None,
                 attr1=None, oper1='>=', val1=0,
                 attr2=None, oper2='>=', val2=0,
                 attr3=None, oper3='>=', val3=0,
                 chart_style='darkgrid',
                 suptitle_size=14,
                 title_size=12,
                 legend_size=12,
                 xsize=10, ysize=8,
                 image_dir=None,
                 image_format='png'):
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
        p_dict (dict)
            dictionary, numerical eg code to string description
        ret_age (integer or float)
            chart xaxis limit for plotting
        ds_dict (dict)
            variable assigned to the output of the load_datasets function,
            reqired when string dictionary key is used as df input
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (string, integer, float, date as string as appropriate)
            attr(n) limiting value (combined with oper(n)) as string
        chart_style (string)
            any valid seaborn plotting style
        suptitle_size (integer or font)
            text size of chart super title
        title_size (integer or float)
            text size of chart title
        legend_size (integer or float)
            text size of chart legend
        xsize, ysize (integer or float)
            plot size in inches
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)

    d_filt, t_string = filter_ds(ds,
                                 attr1=attr1, oper1=oper1, val1=val1,
                                 attr2=attr2, oper2=oper2, val2=val2,
                                 attr3=attr3, oper3=oper3, val3=val3)

    d_age_pcnt = d_filt[d_filt.mnum == mnum][
        ['age', 'mnum', 'spcnt', 'eg']].copy()

    with sns.axes_style(chart_style):
        fig, ax = plt.subplots(figsize=(xsize, ysize))

    for grp in eg_list:
        d_for_plot = d_age_pcnt[d_age_pcnt.eg == grp]
        x = d_for_plot['age']
        y = d_for_plot['spcnt']
        ax.scatter(x, y, c=color_list[grp - 1],
                   s=20, linewidth=0.1, edgecolors='w',
                   label=p_dict[grp])

    ax.set_ylim(1, 0)
    ax.set_xlim(25, ret_age)
    ax.yaxis.set_major_formatter(pct_format())
    ax.set_yticks(np.arange(0, 1.05, .05))
    fig.suptitle(df_label +
                 ' - age vs seniority percentage' +
                 ', month ' +
                 str(mnum), fontsize=suptitle_size)

    ax.set_title(t_string, fontsize=title_size)
    ax.legend(loc=2, markerscale=1.5, fontsize=legend_size)
    ax.set_ylabel('seniority list percentage')
    ax.set_xlabel('age')

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def multiline_plot_by_emp(df, measure, xax,
                          emp_list, job_levels,
                          ret_age, color_list,
                          job_str_list, sdict,
                          attr_dict, ds_dict=None,
                          plot_jobp=False,
                          show_implementation_date=True,
                          through_date=None,
                          pcnt_ylimit=1.0,
                          chart_style='ticks',
                          linewidth=3,
                          line_alpha=.7,
                          grid_linestyle='dotted',
                          grid_alpha=.75,
                          legend_size=14,
                          label_size=13,
                          tick_size=13,
                          title_size=18,
                          xsize=12, ysize=9,
                          image_dir=None,
                          image_format='png'):
    '''select example individual employees and plot career measure
    from selected dataset attribute, i.e. list percentage, career
    earnings, job level, etc.

    inputs
        df (dataframe)
            dataset to examine, may be a dataframe or a string key with the
            ds_dict dictionary object
        measure (string)
            dataset attribute to plot.  Usually only one attribute to plot,
            but may be more than one, such as 'jnum' and 'jobp'
        xax (string)
            dataset attribute for x axis
        emp_list (list)
            list of employee numbers or ids
        job_levels (integer)
            number of job levels in model
        ret_age (float)
            retirement age (example: 65.0)
        color list (list)
            list of colors for plotting
        job_str_list (list)
            list of string job descriptions corresponding to
            number of job levels
        sdict (dictionary)
            program settings dictionary
        attr_dict (dictionary)
            dataset column name description dictionary
        ds_dict (dictionary)
            output of the load_datasets function, dictionary.  This keyword
            argument must be set if a string key is used as the df input.
        plot_jobp (boolean)
            if measure input is 'jnum', also plot 'jobp' if set to True
        show_implementation_date (boolean)
            if True and "xax" input is "date", plot a vertical line at the
            implementation date
        chart_style (string)
            any seaborn plotting style name
        linewidth (integer or float)
            width of chart solid lines
        line_alpha (float)
            transparency value of the plotted lines (0.0 to 1.0)
        grid_linestyle (string)
            matplotlib line style for grid, such as "dotted" or "solid"
        grid_alpha
            transparency value for grid (0.0 to 1.0)
        legend_size (integer or float)
            text size of chart legend
        label_size (integer or float)
            font size of x and y axis labels
        tick_size (integer or float)
            font size of chart tick labels
        title_size (integer or float)
            font size of chart title
        xsize, ysize (integer or float)
            plot size in inches
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)

    frame = ds.copy()
    eg_df = frame[frame.mnum == 0]['eg']
    c_list = []
    for emp in emp_list:
        c_list.append(color_list[eg_df[emp] - 1])

    frame_cols = [xax, measure, 'age', 'ret_mark', 'empkey']
    if measure == 'jnum' and plot_jobp:
        frame_cols.append('jobp')
    frame = frame[frame_cols]

    with sns.axes_style(chart_style):
        fig, ax = plt.subplots(figsize=(xsize, ysize))

    if measure in ['mpay']:
        if 'ret_mark' in frame.columns.values.tolist():
            frame = frame[frame.ret_mark != 1]
        else:
            frame = frame[frame.age < ret_age]

    i = 0
    if xax in ['date', 'ldate', 'doh', 'retdate']:
        for emp in emp_list:
            y = frame.loc[emp][measure]
            x = frame.loc[emp][xax]
            ax.plot_date(x=x, y=y, color=c_list[i],
                         label=str(emp), ls='solid', lw=linewidth,
                         markersize=0, alpha=line_alpha)
            if measure == 'jnum' and plot_jobp:
                y = frame.loc[emp]['jobp']
                ax.plot_date(x=x, y=y, color=c_list[i],
                             label='_nolegend_', ls='dashed', markersize=0,
                             alpha=line_alpha, lw=1.5)

            i += 1
        if xax == 'date':
            if through_date:
                ax.set_xlim(xmax=pd.to_datetime(through_date))

        ax.set_xlim(xmin=sdict['starting_date'] + pd.offsets.MonthEnd(-1))
        locator = mdate.YearLocator()
        ax.xaxis.set_major_locator(locator)
        fig.autofmt_xdate()
        plt.xticks(rotation=75, ha='center')

        if len(ax.get_xticks()) > 20:
            for label in ax.xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)

    else:
        frame.set_index(xax, inplace=True, drop=True)
        for emp in emp_list:
            eg_df = frame[frame.empkey == emp]
            eg_df[measure].plot(color=c_list[i], alpha=line_alpha,
                                label=str(emp), lw=linewidth, ax=ax)
            if measure == 'jnum' and plot_jobp:
                eg_df['jobp'].plot(color=c_list[i], alpha=line_alpha,
                                   label='_nolegend_', ax=ax,
                                   ls='dashed', lw=1)

            i += 1

    if measure in ['lspcnt', 'spcnt']:
        if pcnt_ylimit:
            ax.set_yticks(np.arange(0, 1.05, .05))
            pcnt_ylimit = np.clip(pcnt_ylimit, 0.05, 1.0)
            ax.set_ylim(ymax=pcnt_ylimit)
        else:
            ax.set_yticks(np.arange(0, 1.05, .05))
        ax.yaxis.set_major_formatter(pct_format())

    if measure in ['snum', 'spcnt', 'lspcnt', 'jnum',
                   'lnum', 'jobp', 'fbff', 'cat_order']:
        ax.invert_yaxis()

    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:

        ax.set_yticks(np.arange(0, job_levels + 2, 1))
        ytick_labels = ax.get_yticks().tolist()

        for i in np.arange(1, len(ytick_labels)):
            ytick_labels[i] = job_str_list[i - 1]
        ax.axhspan(job_levels + 1, job_levels + 2,
                   facecolor='.8', alpha=0.9)
        ax.set_yticklabels(ytick_labels, va='top')
        ax.axhline(y=job_levels + 1, c='.8', ls='-', alpha=.8, lw=3)
        ax.set_ylim(job_levels + 1.5, 0.5)

    if xax in ['spcnt', 'lspcnt']:
        ax.xaxis.set_major_formatter(pct_format())
        ax.set_xticks(np.arange(0, 1.1, .1))
        ax.set_xlim(1, 0)

    if show_implementation_date:
        if (sdict['delayed_implementation'] and
                sdict['implementation_date'] and
                xax == 'date'):
                    ax.axvline(sdict['implementation_date'],
                               c='g', ls='--', alpha=1, lw=1)

    ax.set_title(attr_dict[measure] + ' - proposal ' + df_label,
                 y=1.02, fontsize=title_size)
    ax.set_ylabel(attr_dict[measure], fontsize=label_size)
    ax.set_xlabel(attr_dict[xax], fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.yaxis.labelpad = 10
    ax.xaxis.labelpad = 10
    ax.grid(ls=grid_linestyle, alpha=grid_alpha)
    ax.legend(loc=4, markerscale=1.5, fontsize=legend_size)

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def violinplot_by_eg(df, measure, ret_age,
                     attr_dict, ds_dict=None,
                     mnum=0, linewidth=1.5,
                     attr1=None, oper1='>=', val1='0',
                     attr2=None, oper2='>=', val2='0',
                     attr3=None, oper3='>=', val3='0',
                     scale='count',
                     title_size=12,
                     chart_style='darkgrid',
                     xsize=12, ysize=10,
                     image_dir=None,
                     image_format='png'):
    '''From the seaborn website:
    Draw a combination of boxplot and kernel density estimate.

    A violin plot plays a similar role as a box and whisker plot.
    It shows the distribution of quantitative data across several
    levels of one (or more) categorical variables such that those
    distributions can be compared. Unlike a box plot, in which all
    of the plot components correspond to actual datapoints, the violin
    plot features a kernel density estimation of the underlying distribution.

    inputs
        df (dataframe)
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        measure (string)
            attribute to plot
        ret_age (float)
            retirement age (example: 65.0)
        attr_dict (dictionary)
            dataset column name description dictionary
        ds_dict (dictionary)
            output from load_datasets function
        mnum (integer)
            month number to analyze
        linewidth (integer or float)
            width of line surrounding each violin plot
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (string, integer, float, date as string as appropriate)
            attr(n) limiting value (combined with oper(n)) as string
        scale (string)
            From the seaborn website:
            The method used to scale the width of each violin.
            If 'area', each violin will have the same area.
            If 'count', the width of the violins will be scaled by
            the number of observations in that bin.
            If 'width', each violin will have the same width.
        title_size (integer or float)
            text size of chart title
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
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
        frame = frame[frame.ret_mark == 0]

    with sns.axes_style(chart_style):
        fig, ax = plt.subplots(figsize=(xsize, ysize))

    sns.violinplot(x=frame.eg, y=frame[measure],
                   cut=0, scale=scale, inner='box',
                   bw=.1, linewidth=linewidth,
                   palette=['gray', '#3399ff', '#ff8000'], ax=ax)

    fig.suptitle(df_label + ' - ' +
                 attr_dict[measure].upper() + ',  Month ' +
                 str(mnum) + ' Distribution')

    ax.set_title(title_string, fontsize=title_size)

    if measure == 'age':
        ax.set_ylim(25, 70)
    if measure in ['snum', 'spcnt', 'lspcnt', 'jnum',
                   'jobp', 'cat_order']:
        ax.invert_yaxis()

        if measure in ['spcnt', 'lspcnt']:
            ax.yaxis.set_major_formatter(pct_format())
            ax.set_yticks(np.arange(0, 1.05, .05))
            ax.set_ylim(1.04, -.04)

    ax.set_xlabel('employee group')
    ax.set_ylabel(attr_dict[measure])

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def age_kde_dist(df, color_list,
                 p_dict, max_age,
                 ds_dict=None,
                 mnum=0,
                 title_size=14,
                 min_age=25,
                 chart_style='darkgrid',
                 xsize=12, ysize=10,
                 image_dir=None,
                 image_format='png'):
    '''From the seaborn website:
    Fit and plot a univariate or bivariate kernel density estimate.

    inputs
        df (dataframe)
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        color_list (list)
            list of colors for employee group plots
        p_dict (dictionary)
            eg to string dict for plot labels
        max_age (float)
            maximum age to plot (x axis limit)
        ds_dict (dictionary)
            output from load_datasets function
        mnum (integer)
            month number to analyze
        title_size (integer or float)
            text size of chart title
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''
    ds = determine_dataset(df, ds_dict)

    frame = ds[ds.mnum == mnum]
    eg_set = pd.unique(frame.eg)

    with sns.axes_style(chart_style):
        fig, ax = plt.subplots(figsize=(xsize, ysize))

    for x in eg_set:
        try:
            color = color_list[x - 1]

            sns.kdeplot(frame[frame.eg == x].age,
                        shade=True, color=color,
                        bw=.8, ax=ax, label=p_dict[x])
        except LookupError:
            print('error plotting for eg:', x)

    ax.set_xlim(min_age, max_age)

    ax.set_title('Age Distribution Comparison - Month ' + str(mnum), y=1.02,
                 fontsize=title_size)

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def eg_diff_boxplot(df_list, dfb, eg_list,
                    eg_colors, job_levels,
                    job_diff_clip, attr_dict,
                    measure='spcnt',
                    comparison='baseline',
                    ds_dict=None,
                    attr1=None, oper1='>=', val1=0,
                    attr2=None, oper2='>=', val2=0,
                    attr3=None, oper3='>=', val3=0,
                    suptitle_size=14,
                    title_size=12,
                    tick_size=11,
                    label_size=12,
                    year_clip=None,
                    exclude_fur=False,
                    width=.8,
                    chart_style='dark',
                    notch=True,
                    linewidth=1.0,
                    xsize=12, ysize=8,
                    image_dir=None,
                    image_format='png'):
    '''create a DIFFERENTIAL box plot chart comparing a selected measure from
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
        eg_colors (list)
            corresponding plot colors for eg_list input
        job_levels (integer)
            number of job levels in the data model (excluding furlough)
        job_diff_clip (integer)
            if measure is jnum or jobp, limit y axis range to +/- this value
        attr_dict (dictionary)
            dataset column name description dictionary
        measure (string)
            differential data to compare
        comparison (string)
            if 'p2p' (proposal to proposal), will compare proposals within the
            df_list to each other, otherwise will compare proposals to the
            baseline dataset (dfb)
        ds_dict (dictionary)
            output from load_datasets function
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (string, integer, float, date as string as appropriate)
            attr(n) limiting value (combined with oper(n)) as string
        suptitle_size (integer or float)
            text size of chart super title
        title_size (integer or float)
            text size of chart title
        tick_size (integer or float)
            text size of x and y tick labels
        label_size (integer or float)
            text size of x and y descriptive labels
        year_clip (integer)
            only present results through this year if not None
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
        xsize, ysize (integer or float)
            plot size in inches
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
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

    # set boxplot color to match employee group(s) color
    color_index = sorted(list(np.array(eg_list) - 1))
    color_arr = np.array(eg_colors)
    eg_clrs = list(color_arr[color_index])

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
    # option to limit display up through a selected year
    if year_clip:
        y_clip = baseline[baseline.date <= year_clip].copy()
    else:
        y_clip = baseline.copy()

    # replace all zero differential results with nan (null) so that only
    # differentials are included in boxplot results (avoid partial year zero
    # differentials averaged with actual differentials)
    y_clip.replace(0, np.nan, inplace=True)

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
        except LookupError:
            pad = 0

        compare_vals = y_clip[yval].values
        max_val = abs(np.nanmax(compare_vals))
        min_val = abs(np.nanmin(compare_vals))
        ylimit = max(min_val, max_val) + pad

        with sns.axes_style(chart_style):
            fig, ax = plt.subplots(figsize=(xsize, ysize))

        sns.boxplot(x='date', y=yval,
                    hue='eg', data=y_clip,
                    palette=eg_clrs, width=width,
                    notch=notch,
                    linewidth=linewidth, fliersize=1.0, ax=ax)

        # add zero line
        ax.axhline(y=0, c='r', zorder=.9, alpha=.35, lw=2)
        # ax.set_ylim(-ylimit, ylimit)
        if measure in ['spcnt', 'lspcnt']:
            # format percentage y axis scale
            ax.yaxis.set_major_formatter(pct_format())
        if measure in ['jnum', 'jobp']:
            # if job level measure, set scaling and limit y range
            ax.set_yticks(np.arange(int(-ylimit - 1), int(ylimit + 2)))
            ax.set_ylim(max(-job_diff_clip, int(-ylimit - 1)),
                        min(job_diff_clip, int(ylimit + 1)))

        ax.set_title(tb_string, fontsize=title_size)
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.set_ylabel('differential', fontsize=label_size)
        ax.xaxis.label.set_size(label_size)
        fig.suptitle(yval_dict[yval], fontsize=suptitle_size)

        if image_dir:
            func_name = sys._getframe().f_code.co_name
            if not path.exists(image_dir):
                makedirs(image_dir)
            plt.savefig(image_dir + '/' + func_name + ' - ' + yval +
                        '.' + image_format,
                        bbox_inches='tight', pad_inches=.25)
        plt.show()


def eg_boxplot(df_list, eg_list,
               eg_colors, job_clip,
               attr_dict,
               measure='spcnt',
               ds_dict=None,
               attr1=None, oper1='>=', val1=0,
               attr2=None, oper2='>=', val2=0,
               attr3=None, oper3='>=', val3=0,
               year_clip=2035,
               exclude_fur=False,
               saturation=.8,
               chart_style='dark',
               width=.7,
               notch=True,
               show_whiskers=True,
               show_xgrid=True,
               show_ygrid=True,
               grid_alpha=.4,
               grid_linestyle='solid',
               whisker=1.5,
               fliersize=1.0,
               linewidth=.75,
               suptitle_size=14,
               title_size=12,
               tick_size=11,
               label_size=12,
               xsize=12, ysize=8,
               image_dir=None,
               image_format='png'):
    '''create a box plot chart displaying ACTUAL attribute values
    (vs. differential values) from a selected dataset(s) for selected
    employee group(s).

    inputs
        df_list (list)
            list of datasets to compare, may be ds_dict (output of
            load_datasets function) string keys or dataframe variable(s)
            or mixture of each
        eg_list (list)
            list of integers for employee groups to be included in analysis
            example: [1, 2, 3]
        measure (string)
            attribute for analysis
        eg_colors (list)
            list of colors for plotting the employee groups
        attr_dict (dictionary)
            dataset column name description dictionary
        ds_dict (dictionary)
            output from load_datasets function
        job_clip (float)
            if measure is jnum or jobp, limit max y axis range to this value
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (string, integer, float, date as string as appropriate)
            attr(n) limiting value (combined with oper(n)) as string
        year_clip (integer)
            only present results through this year
        exclude_fur (boolean)
            remove all employees from analysis who are furloughed within the
            data model at any time (boolean)
        chart_style (string)
            chart styling (string), any valid seaborn chart style
        width (float)
            plotting width of boxplot or grouped boxplots for each year.
            a width of 1 leaves no gap between groups
        notch (boolean)
            If True, show boxplots with a notch at median point
        show_xgrid (boolean)
            include vertical grid lines on chart
        show_ygrid (boolean)
            include horizontal grid lines on chart
        grid_alpha (float)
            opacity value for grid lines
        grid_linestyle (string)
            examples: 'solid', 'dotted', 'dashed'
        suptitle_size (integer or float)
            text size of chart super title
        title_size (integer or float)
            text size of chart title
        tick_size (integer or float)
            text size of x and y tick labels
        label_size (integer or float)
            text size of x and y descriptive labels
        xsize, ysize (integer or float)
            width and hieght of plot in inches
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
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

    # set boxplot color to match employee group(s) color
    color_index = sorted(list(np.array(eg_list) - 1))
    color_arr = np.array(eg_colors)
    eg_clrs = list(color_arr[color_index])

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
        title_dict[this_measure_col] = p_label + ' ' + attr_dict[measure]

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
        except LookupError:
            pad = 0

        compare_vals = y_clip[yval].values
        max_val = abs(np.nanmax(compare_vals))
        min_val = abs(np.nanmin(compare_vals))
        ylimit = max(min_val, max_val) + pad

        with sns.axes_style(chart_style):
            fig, ax = plt.subplots(figsize=(xsize, ysize))

        sns.boxplot(x='year', y=yval,
                    hue='eg', data=y_clip,
                    palette=eg_clrs, width=width,
                    notch=notch,
                    linewidth=linewidth, whis=whisker,
                    fliersize=fliersize, ax=ax)

        ax.set_ylim(0, ylimit)
        if measure in ['spcnt', 'lspcnt']:
            # format percentage y axis scale
            ax.yaxis.set_major_formatter(pct_format())
            ax.set_ylim(ylimit, 0)
        if measure in ['jnum', 'jobp']:
            # if job level measure, set scaling and limit y range
            ax.set_yticks(np.arange(0, int(ylimit + 1)))
            ax.set_ylim(min(job_clip + 1.5, int(ylimit + 2)), 0.5)
        if measure in ['cat_order', 'snum', 'lnum']:
            ax.invert_yaxis()

        if filt_title:
            fig.suptitle(title_dict[yval], fontsize=suptitle_size)
            ax.set_title(filt_title, fontsize=title_size)
        else:
            ax.set_title(title_dict[yval], fontsize=suptitle_size, y=1.01)

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
        ax.set_ylabel('absolute values', fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.xaxis.label.set_size(label_size)

        if show_xgrid:
            ax.xaxis.grid(alpha=grid_alpha, ls=grid_linestyle)
        if show_ygrid:
            ax.yaxis.grid(alpha=grid_alpha, ls=grid_linestyle)

        if image_dir:
            func_name = sys._getframe().f_code.co_name
            if not path.exists(image_dir):
                makedirs(image_dir)
            plt.savefig(image_dir + '/' + func_name + ' - ' + yval +
                        '.' + image_format,
                        bbox_inches='tight', pad_inches=.25)
        plt.show()


# DISTRIBUTION WITHIN JOB LEVEL (NBNF effect)
def stripplot_dist_in_category(df, job_levels,
                               full_time_pcnt,
                               eg_colors,
                               band_colors,
                               job_strs,
                               attr_dict,
                               p_dict,
                               ds_dict=None,
                               rank_metric='cat_order',
                               mnum=None,
                               attr1=None, oper1='>=', val1='0',
                               attr2=None, oper2='>=', val2='0',
                               attr3=None, oper3='>=', val3='0',
                               bg_alpha=.12,
                               fur_color=None,
                               show_part_time_lvl=True,
                               size=3,
                               title_size=14,
                               label_pad=110,
                               label_size=13,
                               tick_size=12,
                               xsize=4, ysize=12,
                               image_dir=None,
                               image_format='png'):

    '''visually display employee group distribution concentration within
    accurately sized job bands for a selected month.

    This chart reveals how evenly or unevenly the employee groups share
    the jobs available within each job category.

    inputs
        df (dataframe)
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        job_levels (integer)
            number of job levels in the data model
        full_time_pcnt (float)
            percentage of each job level which is full time
        eg_colors (list)
            list of colors for eg plots
        band_colors (list)
            list of colors for background job band colors
        job_strs (list)
            list of job strings for job description labels
        attr_dict (dictionary)
            dataset column name description dictionary
        p_dict (dictionary)
            eg to group string label
        ds_dict (dictionary)
            output from load_datasets function
        rank_metric (string)
            rank attribute (currently only accepts 'cat_order')
        mnum (integer)
            month number - if not None, analyze data from this month
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (string, integer, float, date as string as appropriate)
            attr(n) limiting value (combined with oper(n)) as string
        bg_alpha (float)
            color alpha for background job level color
        fur_color (color code in rgba, hex, or string style)
            custom color to signify furloughed job band area (otherwise, last
            color from band_colors list will be used)
        show_part_time_lvl (boolean)
            draw a line within each job band representing the boundry
            between full and part-time jobs
        size (integer or float)
            size of density markers
        title_size (integer or float)
            text size of chart title
        label_size (integer or float)
            text size of x and y descriptive labels
        tick_size (integer or float)
            text size of x and y tick labels
        xsize, ysize (integer or float)
            width and height of chart in inches
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)
    if mnum:
        dsm = ds[ds.mnum == mnum]
    else:
        dsm = ds

    d_filt, t_string = filter_ds(dsm,
                                 attr1=attr1, oper1=oper1, val1=val1,
                                 attr2=attr2, oper2=oper2, val2=val2,
                                 attr3=attr3, oper3=oper3, val3=val3)

    d_filt = d_filt[[]].join(dsm[['mnum',
                                  'jnum',
                                  'eg',
                                  rank_metric]]).reindex(dsm.index)

    data = d_filt.copy()

    eg_set = pd.unique(data.eg)
    max_eg_plus_one = max(eg_set) + 1

    y_count = len(data)

    clr_idx = (np.unique(dsm.jnum) - 1).astype(int)
    cum_job_counts = dsm.jnum.value_counts().sort_index().cumsum()

    cnts = list(cum_job_counts)
    cnts.insert(0, 0)

    axis2_lbl_locs = []
    axis2_lbls = []

    if fur_color:
        band_colors[-1] = fur_color

    with sns.axes_style('white'):
        fig, ax1 = plt.subplots(figsize=(xsize, ysize))
    ax1.tick_params(labelsize=tick_size)

    ax1 = sns.stripplot(y=rank_metric, x='eg', data=data, jitter=.5,
                        order=np.arange(1, max_eg_plus_one),
                        palette=eg_colors, size=size,
                        linewidth=0, split=True)

    ax1.set_yticks = (np.arange(0, ((len(df) + 1000) % 1000) * 1000, 1000))
    ax1.set_ylim(y_count, 0)
    ax1.xaxis.label.set_size(label_size)
    ax1.yaxis.label.set_size(label_size)

    i = 0
    for job_zone in cum_job_counts:
        ax1.axhline(job_zone, c='magenta', ls='-', alpha=1, lw=.8)
        ax1.axhspan(cnts[i], cnts[i + 1],
                    facecolor=band_colors[clr_idx[i]],
                    alpha=bg_alpha)

        if show_part_time_lvl:
            part_time_lvl = (round((cnts[i + 1] - cnts[i]) *
                             full_time_pcnt)) + cnts[i]
            ax1.axhline(part_time_lvl, c='#66ff99', ls='--', alpha=1, lw=1)
        i += 1

    i = 0

    for job_num in cum_job_counts.index:
        axis2_lbl_locs.append(round((cnts[i] + cnts[i + 1]) / 2))
        axis2_lbls.append(job_strs[int(job_num)])
        i += 1

    axis2_lbl_locs = add_pad(axis2_lbl_locs, pad=label_pad)

    with sns.axes_style("white"):
        ax2 = ax1.twinx()
        ax2.set_yticks(axis2_lbl_locs)
        yticks = ax2.get_yticks().tolist()

        for i in np.arange(len(yticks)):
            yticks[i] = axis2_lbls[i]

        ax2.set_yticklabels(yticks)
        ax2.set_ylim(y_count, 0)

    xticks = ax2.get_xticks().tolist()

    tick_dummies = []
    for tck in xticks:
        tick_dummies.append(p_dict[tck + 1])

    ax2.set_xticklabels(tick_dummies)
    ax2.tick_params(labelsize=tick_size)

    title_pt1 = (df_label +
                 ', distribution within job levels, month ' + str(mnum))
    plt.title(title_pt1 + '\n\n' + t_string, fontsize=title_size, y=1.01)
    ax1.set_ylabel(attr_dict[rank_metric])
    ax1.set_xlabel(attr_dict['eg'])

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        fig.set_size_inches(xsize + 1, ysize)
        plt.tight_layout()
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def job_level_progression(df, emp_list,
                          through_date,
                          settings_dict,
                          color_dict,
                          eg_colors,
                          band_colors,
                          ds_dict=None,
                          rank_metric='cat_order',
                          chart_style='white',
                          show_implementation_date=True,
                          job_bands_alpha=.1,
                          max_plots_for_legend=5,
                          xgrid_alpha=.65,
                          xgrid_linestyle='dotted',
                          ygrid_alpha=.5,
                          ygrid_linestyle='dotted',
                          tick_size=13,
                          job_descr_size=12.5,
                          job_descr_pad=115,
                          label_size=15,
                          title_size=18,
                          xsize=12, ysize=10,
                          image_dir=None,
                          image_format='png'):
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
        df (dataframe)
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        emp_list (list)
            list of empkeys to plot
        through_date (date string)
            string representation of y axis date limit, ex. '2025-12-31'
        settings_dict (dictionary)
            program settings dictionary generated by the build_program_files
            script
        color_dict (dictionary)
            dictionary containing color list string titles to lists of color
            values generated by the build_program_files script
        eg_colors (list)
            colors to be used for employee line plots corresponding
            to employee group membership
        band_colors (list)
            list of colors to be used for stacked area chart which represent
            job level bands
        ds_dict (dictionary)
            output from load_datasets function
        rank_metric (string)
            column name for y axis chart ranking.  Currently only 'cat_order'
            is valid.
        chart_style (string)
            any valid seaborn plotting chart style name
        show_implementation_date (boolean)
            plot a vertical dashed line at the implementation date
        job_bands_alpha (float)
            opacity level of background job bands stacked area chart
        max_plots_for_legend (integer)
            if number of plots more than this number, reduce plot linewidth and
            remove legend
        xgrid_alpha, ygrid_alpha (float)
            transparency value for grid.  x and y axis may be set independently
        xgrid_linestyle, ygrid_linestyle (string)
            matplotlib line style for grid, such as "dotted" or "dashed".
            x and y axis may be set independently
        job_descr_size (integer or float)
            font size of job description text labels on right side of chart
        job_descr_pad (integer)
            padding to add between job description labels when they would
            otherwise overlap
        tick_size (intger or float)
            font size of tick labels
        job_descr_size (integer or float)
            font size of job description labels
        label_size (integer or float)
            font size of axis labels
        title_size (integer or label)
            font size of title
        xsize, ysize (integer or float)
            plot size in inches (width, height)
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)

    job_levels = settings_dict['num_of_job_levels']

    through_date = pd.to_datetime(through_date)
    fur_lvl = job_levels + 1
    jobs_dict = settings_dict['job_strs_dict']

    tdict = pd.read_pickle('dill/dict_job_tables.pkl')
    table = tdict['table']

    df_table = pd.DataFrame(table[0],
                            columns=np.arange(1, job_levels + 1),
                            index=pd.date_range(settings_dict['starting_date'],
                                                periods=table[0].shape[0],
                                                freq='M'))
    # for band areas
    jobs_table = df_table[:through_date]
    # for headcount:
    df_monthly_non_ret = pd.DataFrame(ds[ds.fur == 0].groupby('mnum').size(),
                                      columns=['count'])
    df_monthly_non_ret.set_index(
        pd.date_range(settings_dict['starting_date'],
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

    i = 0
    for job_num in last_month_counts.index:
        axis2_lbl_locs.append(round((cnts[i] + cnts[i + 1]) / 2))
        axis2_lbls.append(jobs_dict[job_num])
        i += 1

    axis2_lbl_locs = add_pad(axis2_lbl_locs, pad=job_descr_pad)

    egs = ds[ds.mnum == 0].eg

    if len(emp_list) > max_plots_for_legend:
        lw = 1
    else:
        lw = 3

    with sns.axes_style(chart_style):
        fig, ax1 = plt.subplots(figsize=(xsize, ysize))

    ds = ds[ds.date <= through_date][['empkey', 'date', rank_metric]].copy()
    ds.set_index('date', drop=True, inplace=True)

    i = 0
    for emp in emp_list:
        c_idx = egs.loc[emp] - 1
        ds[ds.empkey == emp][rank_metric].plot(lw=lw, color=eg_colors[c_idx],
                                               label=emp, ax=ax1)
        i += 1

    non_ret_count['count'].plot(c='grey', ls='--',
                                label='active count', ax=ax1)

    if len(emp_list) <= max_plots_for_legend:
        ax1.legend(title='')

    if (settings_dict['delayed_implementation'] and
            show_implementation_date and
            settings_dict['implementation_date']):

                ax1.axvline(settings_dict['implementation_date'],
                            c='g', ls='--', alpha=1, lw=1)

    jobs_table.plot.area(stacked=True,
                         figsize=(xsize, ysize),
                         sort_columns=True,
                         linewidth=2,
                         color=band_colors,
                         alpha=job_bands_alpha,
                         legend=False,
                         ax=ax1)

    ax1.invert_yaxis()
    ax1.set_ylim(max(df_monthly_non_ret['count']), 0)

    ax1.set_title(df_label + ' job level progression',
                  y=1.01, fontsize=title_size)

    if lowest_cat == fur_lvl:
        ax1.axhspan(cnts[-2], cnts[-1], facecolor='#fbfbea', alpha=0.9)
        axis2_lbls[-1] = 'FUR'

    ax2 = ax1.twinx()
    ax2.set_yticks(axis2_lbl_locs)
    yticks = ax2.get_yticks().tolist()

    for i in np.arange(len(yticks)):
        yticks[i] = axis2_lbls[i]

    ax2.set_yticklabels(yticks, fontsize=job_descr_size)
    ax2.grid(False)
    ax2.invert_yaxis()

    ax1.xaxis.grid(True, alpha=xgrid_alpha, ls=xgrid_linestyle)
    ax1.yaxis.grid(True, alpha=ygrid_alpha, ls=ygrid_linestyle)
    ax1.set_axisbelow(True)
    ax1.set_ylabel('global job ranking', fontsize=label_size)
    ax1.set_xlabel('year', fontsize=label_size)
    ax1.tick_params(axis='both', which='major', labelsize=tick_size)

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def differential_scatter(df_list, dfb,
                         measure, eg_list,
                         attr_dict, color_dict,
                         p_dict, ds_dict=None,
                         attr1=None, oper1='>=', val1=0,
                         attr2=None, oper2='>=', val2=0,
                         attr3=None, oper3='>=', val3=0,
                         prop_order=True,
                         show_scatter=True,
                         show_lin_reg=True,
                         show_mean=True,
                         mean_len=50,
                         dot_size=15,
                         lin_reg_order=15,
                         ylimit=False, ylim=5,
                         suptitle_size=14,
                         title_size=12,
                         legend_size=14,
                         tick_size=11,
                         label_size=12,
                         bright_bg=False,
                         bright_bg_color='#faf6eb',
                         chart_style='whitegrid',
                         xsize=12, ysize=8,
                         image_dir=None,
                         image_format='png'):
    '''plot an attribute differential between datasets.

    datasets may be filtered by other attributes if desired.

    Example:  plot the difference in cat_order (job rank number) between all
    integrated datasets vs. standalone for all employee groups, applicable to
    month 57. (optionally add a pre-filter(s), such as all employees hired
    prior to a certain date)

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
        measure (string)
            attribute to analyze
        eg_list (list)
            list of employee group codes
        attr_dict (dictionary)
            dataset column name description dictionary
        color_dict (dictionary)
            dictionary containing color list string titles to lists of color
            values generated by the build_program_files script
        p_dict (dictionary)
            employee group code number to description dictionary
        ds_dict (dictionary)
            output from load_datasets function
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (string, integer, float, date as string as appropriate)
            attr(n) limiting value (combined with oper(n)) as string
        eg_list (list)
            a list of employee groups to analyze
        prop_order (boolean)
            if True, organize x axis by proposal list order,
            otherwise use native list percent
        show_scatter (boolean)
            if True, draw the scatter chart markers
        show_lin_reg (boolean)
            if True, draw linear regression lines
        show_mean (boolean)
            if True, draw average lines
        mean_len (integer)
            moving average length for average lines
        dot_size (integer or float)
            scatter marker size
        lin_reg_order (integer)
            regression line is actually a polynomial regression
            lin_reg_order is the degree of the fitting polynomial
        ylimit (boolean)
            if True, set chart y axis limit to ylim (below)
        ylim (integer or float)
            y axis limit positive and negative if ylimit is True
        suptitle_size (integer or float)
            text size of chart super title
        title_size (integer or float)
            text size of chart title
        legend_size (integer or float)
            text size of chart legend labels
        tick_size (integer or float)
            text size of x and y tick labels
        label_size (integer or float)
            text size of x and y descriptive labels
        bright_bg (boolean)
            use a custom color chart background
        bright_bg_color (color value)
            chart background color if bright_bg input is set to True
        chart_style (string)
            style for chart, valid inputs are any seaborn chart style
        xsize, ysize (integer or float)
            size of chart (width, height)
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
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

    # dfb_filt, tb_string = filter_ds(df_base,
    df, tb_string = filter_ds(df_base,
                              attr1=attr1, oper1=oper1, val1=val1,
                              attr2=attr2, oper2=oper2, val2=val2,
                              attr3=attr3, oper3=oper3, val3=val3)

    cols = [measure, 'new_order']

    # df = dfb_filt[dfb_filt[filter_measure] == filter_val][
    #     [measure, 'eg']].copy()
    df.rename(columns={measure: measure + '_s'}, inplace=True)

    order_dict = {}

    i = 1
    for ds in df_list:
        ds = ds[cols].copy()
        # ds = ds[ds[filter_measure] == filter_val][cols].copy()
        ds.rename(columns={measure: measure + '_' + str(i),
                           'new_order': 'order' + str(i)}, inplace=True)
        df = df.join(ds)
        order_dict[i] = 'order' + str(i)
        i += 1

    df.sort_values(by='order1', inplace=True)

    eg_grouped = df.groupby('eg')
    df['eg_sep_order'] = eg_grouped.cumcount() + 1
    eg_sep_order = np.array(df.eg_sep_order)
    eg_denom_dict = eg_grouped.eg_sep_order.max().to_dict()

    eg_arr = np.array(df.eg)
    eg_set = pd.unique(eg_arr)
    denoms = np.zeros(eg_arr.size)

    for eg in eg_set:
        np.put(denoms, np.where(eg_arr == eg)[0], eg_denom_dict[eg])

    df['separate_eg_percentage'] = eg_sep_order / denoms

    if measure in ['spcnt', 'lspcnt', 'snum', 'lnum', 'cat_order',
                   'jobp', 'jnum']:

        for key in list(order_dict.keys()):
            df[str(key) + 'vs'] = df[measure + '_s'] - \
                df[measure + '_' + str(key)]
    else:

        for key in list(order_dict.keys()):
            df[str(key) + 'vs'] = df[measure + '_' + str(key)] - \
                df[measure + '_s']

    for prop_num in np.arange(len(df_list)) + 1:
        df.sort_values(by=order_dict[prop_num], inplace=True)

        if prop_order:
            xax = order_dict[prop_num]
        else:
            xax = 'separate_eg_percentage'

        with sns.axes_style(chart_style):
            fig, ax = plt.subplots(figsize=(xsize, ysize))

        for eg in eg_list:
            data = df[df.eg == eg].copy()
            x_limit = max(data[xax]) + 100
            yax = str(prop_num) + 'vs'

            label = p_dict[eg]

            if show_scatter:
                data.plot(x=xax, y=yax, kind='scatter',
                          linewidth=0.1,
                          color=color_dict['eg_colors'][eg - 1],
                          s=dot_size,
                          label=label,
                          ax=ax)

            if show_mean:
                data['ma'] = data[eg].rolling(mean_len).mean()
                data.plot(x=xax, y='ma', lw=5,
                          color=color_dict['mean_colors'][eg - 1],
                          label=label,
                          alpha=.6, ax=ax)
                ax.set_xlim(0, x_limit)

            if show_lin_reg:
                if show_scatter:
                    lin_reg_colors = color_dict['lin_reg_colors']
                else:
                    lin_reg_colors = color_dict['lin_reg_colors2']
                sns.regplot(x=xax, y=yax, data=data,
                            color=lin_reg_colors[eg - 1],
                            label=label,
                            scatter=False, truncate=True, ci=50,
                            order=lin_reg_order,
                            line_kws={'lw': 20,
                                      'alpha': .4},
                            ax=ax)
                ax.set_xlim(0, x_limit)

            ax.set_xlabel('order: ' + label_dict[prop_num])

        if measure == 'jobp':
            ymin = math.floor(min(df[yax]))
            ymax = math.ceil(max(df[yax]))
            scale_lim = max(abs(ymin), ymax)
            ax.set_yticks = (np.arange(-scale_lim, scale_lim + 1, 1))
            if ylimit:
                ax.set_ylim(-ylim, ylim)
            else:
                ax.set_ylim(-scale_lim, scale_lim)

        if label_dict[prop_num] == 'Proposal':
            p_label = 'df_list item ' + str(prop_num)
        else:
            p_label = label_dict[prop_num]
        suptitle_str = (p_label + ' differential: ' +
                        attr_dict[measure])

        if tb_string:
            fig.suptitle(suptitle_str, fontsize=suptitle_size)
            ax.set_title(tb_string, fontsize=title_size, y=1.005)
        else:
            ax.set_title(suptitle_str, fontsize=suptitle_size)
        ax.set_xlim(xmin=0)

        if measure in ['spcnt', 'lspcnt']:
            ax.yaxis.set_major_formatter(pct_format())

        if xax == 'separate_eg_percentage':
            ax.xaxis.set_major_formatter(pct_format())
            ax.set_xticks(np.arange(0, 1.1, .1))
            ax.set_xlim(xmax=1)

        ax.axhline(0, c='m', ls='-', alpha=1, lw=2)
        ax.invert_xaxis()
        if bright_bg:
            ax.set_facecolor(bright_bg_color)
        ax.set_ylabel('differential', fontsize=label_size)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.xaxis.label.set_size(label_size)
        ax.legend(markerscale=1.5, fontsize=legend_size)

        if image_dir:
            func_name = sys._getframe().f_code.co_name
            if not path.exists(image_dir):
                makedirs(image_dir)
            plt.savefig(image_dir + '/' + func_name + ' - ' + p_label +
                        '.' + image_format,
                        bbox_inches='tight', pad_inches=.25)
        plt.show()


def job_grouping_over_time(df, eg_list, jobs,
                           job_colors, p_dict,
                           plt_kind='bar',
                           ds_dict=None,
                           rets_only=True,
                           attr1=None, oper1='>=', val1=0,
                           attr2=None, oper2='>=', val2=0,
                           attr3=None, oper3='>=', val3=0,
                           time_group='A',
                           display_yrs=40,
                           legend_loc=4,
                           chart_style='darkgrid',
                           suptitle_size=14,
                           title_size=12,
                           legend_size=13,
                           tick_size=11,
                           label_size=13,
                           xsize=12, ysize=10,
                           image_dir=None,
                           image_format='png'):

    '''Inverted bar chart display of job counts by group over time.  Various
    filters may be applied to study slices of the datasets.

    The 'rets_only' option will display the count of employees retiring from
    each year grouped by job level.

    developer TODO: fix x axis scaling and labeling when quarterly ("Q") or
    monthly ("M") time group option selected.

    inputs
        df (dataframe)
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        eg_list (list)
            list of unique employee group numbers within the proposal
            Example: [1, 2]
        jobs (list)
            list of job label strings (for plot legend)
        job_colors (list)
            list of colors to be used for plotting
        p_dict (dictionary)
            employee group to string description dictionary
        plt_kind (string)
            'bar' or 'area' (bar recommended)
        ds_dict (dictionary)
            output from load_datasets function
        rets_only (boolean)
            calculate for employees at retirement age only
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (string, integer, float, date as string as appropriate)
            attr(n) limiting value (combined with oper(n)) as string
        time_group (string)
            group counts/percentages by year ('A'), quarter ('Q'),
            or month ('M')
        display_years (integer)
            when using the bar chart type display, evenly scale the x axis
            to include the number of years selected for all group charts
        legend_loc (integer)
            matplotlib legend location number code

                +---+----+---+
                | 2 | 9  | 1 |
                +---+----+---+
                | 6 | 10 | 7 |
                +---+----+---+
                | 3 | 8  | 4 |
                +---+----+---+

        suptitle_size (integer or float)
            text size of chart super title
        title_size (integer or float)
            text size of chart title
        legend_size (integer or float)
            text size of chart legend labels
        tick_size (integer or float)
            text size of x and y tick labels
        label_size (integer or float)
            text size of x and y descriptive labels
        xsize, ysize (integer or float)
            size of each chart in inches (width, height)
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)

    d_filt, t_string = filter_ds(ds,
                                 attr1=attr1, oper1=oper1, val1=val1,
                                 attr2=attr2, oper2=oper2, val2=val2,
                                 attr3=attr3, oper3=oper3, val3=val3)
    if rets_only:
        d_filt = d_filt[d_filt.ret_mark == 1][
            ['eg', 'date', 'jnum']].copy()

    for eg in eg_list:

        with sns.axes_style(chart_style):
            fig, ax = plt.subplots(figsize=(xsize, ysize))

        denom = len(ds[(ds.mnum == 0) & (ds.eg == eg)])
        df_eg = d_filt[d_filt.eg == eg]

        grouped = df_eg.groupby(['date', 'jnum'])

        if rets_only:

            grpby = grouped.size().unstack().fillna(0).astype(int)
            df = grpby.resample(time_group).sum()

            if time_group == 'A':
                df = (df / denom).round(decimals=3)
            if time_group == 'Q':
                df = (df / (.25 * denom)).round(decimals=3)
            ylbl = 'percent of sep list'

        else:

            grpby = grouped.size().unstack().fillna(0).astype(int)
            df = grpby.resample(time_group).mean()
            ylbl = 'count'

        df['year'] = df.index.year

        df.set_index('year', drop=True, inplace=True)

        cols = df.columns
        labels = []
        clr = []
        for col in cols:
            labels.append(jobs[col - 1])
            clr.append(job_colors[col - 1])

        if plt_kind == 'area':
            df.plot(kind='area', linewidth=0, color=clr, stacked=True, ax=ax)

        if plt_kind == 'bar':
            df.plot(kind='bar', width=1,
                    edgecolor='k', linewidth=.5,
                    color=clr, stacked=True, ax=ax)

        if rets_only:
            ax.set_yticks(np.arange(.08, 0, -.01))
            ax.yaxis.set_major_formatter(pct_format())

        ax.invert_yaxis()
        if plt_kind == 'bar':
            ax.set_xlim(0, display_yrs)

        ax.legend((labels), loc=legend_loc, fontsize=legend_size)
        ax.set_ylabel(ylbl, fontsize=label_size)

        suptitle_str = df_label + ' group ' + p_dict[eg]

        if t_string:
            fig.suptitle(suptitle_str, fontsize=suptitle_size)
            ax.set_title(t_string, fontsize=title_size, y=1.005)
        else:
            ax.set_title(suptitle_str, fontsize=suptitle_size)

        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.xaxis.label.set_size(label_size)

        if image_dir:
            func_name = sys._getframe().f_code.co_name
            if not path.exists(image_dir):
                makedirs(image_dir)
            plt.savefig(image_dir + '/' + func_name + ' - ' + 'group' +
                        str(eg) + '.' + image_format,
                        bbox_inches='tight', pad_inches=.25)
        plt.show()


def parallel(df_list, dfb,
             eg_list, measure,
             month_list,
             job_levels,
             eg_colors,
             dict_settings,
             attr_dict,
             ds_dict=None,
             attr1=None, oper1='>=', val1=0,
             attr2=None, oper2='>=', val2=0,
             attr3=None, oper3='>=', val3=0,
             left=0,
             stride_list=None,
             chart_style='whitegrid',
             grid_color='.7',
             suptitle_size=14,
             title_size=12,
             facecolor='w',
             xsize=6, ysize=8,
             image_dir=None,
             image_format='png'):
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
        eg_list (list)
            list of employee group integer codes to compare
            example: [1, 2]
        measure (string)
            dataset attribute to compare
        month_list (list)
            list of month numbers for analysis.
            the function will plot comparative data from each month listed
        job_levels (integer)
            number of job levels in data model
        eg_colors (list)
            list of colors to represent the employee groups
        dict_settings (dictionary)
            program settings dictionary generated by the build_program_files
            script
        attr_dict (dictionary)
            dataset column name description dictionary
        ds_dict (dictionary)
            output from load_datasets function
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (string, integer, float, date as string as appropriate)
            attr(n) limiting value (combined with oper(n)) as string
        left (integer)
            integer representing the list comparison to plot on left side
            of the chart(s).
            zero (0) represents the standalone results and is the default.
            1, 2, or 3 etc. represent the first, second, third, etc. dataset
            results in df_list input order
        stride_list (list)
            optional list of dataframe strides for plotting every other
            nth result (must be same length and correspond to eg_list)
        grid_color (string)
            string name for horizontal grid color
        facecolor (color value)
            chart background color
        xsize, ysize (integer or float)
            size of individual subplots (width, height)
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
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

    group_dict = dict_settings['p_dict']
    color_dict = dict(enumerate(eg_colors))

    jobs = dict_settings['job_strs']

    num_egplots = len(eg_list)
    num_months = len(month_list)

    fig, ax = plt.subplots(num_months, num_egplots)
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

        for eg in eg_list:

            plot_num += 1
            with sns.axes_style(chart_style, {'axes.facecolor': facecolor,
                                              'axes.axisbelow': True,
                                              'axes.edgecolor': '.2',
                                              'axes.linewidth': 1.0,
                                              'grid.color': grid_color,
                                              'grid.linestyle': u'--'}):

                ax = plt.subplot(num_months, num_egplots, plot_num)

            df = df_joined[df_joined.eg == eg]
            try:
                stride = stride_list[eg - 1]
                df = df[::stride]
            except (TypeError, LookupError):
                df = df[::(int(len(df) * .015))]
            parallel_coordinates(df, 'eg', lw=1.5, alpha=.7,
                                 color=color_dict[eg - 1], ax=ax)

            ax.set_title('Group ' + group_dict[eg].upper() + ' ' +
                         attr_dict[measure].upper() +
                         ' ' + str(month) + ' mths',
                         fontsize=title_size, y=1.02)

    for ax in fig.axes:

        if measure in ['spcnt', 'lspcnt']:
            ax.set_yticks(np.arange(1, -0.05, -.05))
            ax.invert_yaxis()
            ax.yaxis.set_major_formatter(pct_format())

        if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:

            ax.set_yticks(np.arange(0, job_levels + 2, 1))
            ax.set_ylim(job_levels + .5, 0.5)
            yticks = ax.get_yticks().tolist()

            for i in np.arange(1, len(yticks)):
                yticks[i] = jobs[i - 1]

            ax.set_yticklabels(yticks, va='top', fontsize=12)

        if measure in ['snum', 'lnum', 'cat_order']:
            ax.invert_yaxis()
        ax.grid()
        ax.legend_.remove()

    fig.suptitle(tb_string, fontsize=title_size, y=1.01)
    plt.tight_layout()

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def rows_of_color(df, mnum, measure_list,
                  eg_colors,
                  jnum_colors,
                  dict_settings,
                  ds_dict=None,
                  attr1=None, oper1='>=', val1=0,
                  attr2=None, oper2='>=', val2=0,
                  attr3=None, oper3='>=', val3=0,
                  cols=150,
                  eg_list=None,
                  job_only=False,
                  jnum=1,
                  cell_border=True,
                  eg_border_color='.3',
                  job_border_color='.8',
                  chart_style='whitegrid',
                  fur_color=None,
                  empty_color='#ffffff',
                  suptitle_size=14,
                  title_size=12,
                  legend_size=14,
                  xsize=15, ysize=9,
                  image_dir=None,
                  image_format='png'):
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
        df (dataframe)
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        mnum (integer)
            month number of dataset to analyze
        measure_list (list)
            list form input, 'categorical' only such as employee group number
            or job number, such as ['jnum'], or ['eg']
            ['eg', 'fur'] is also valid when highlighting furloughees
        eg_colors (list)
            colors to use for plotting the employee groups.
            the first color in the list is used for the plot 'background'
            and is not an employee group color
        jnum_colors (list)
            job level plotting colors, list form
        ds_dict (dictionary)
            output from load_datasets function
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (integer, float, date as string, string (as appropriate))
            attr(n) limiting value (combined with oper(n)) as string
        cols (integer)
            number of columns to construct for the heatmap plot
        eg_list (list)
            employee group integer code list (if used), example: [1, 2]
        job_only (boolean)
            if True, plot only employees holding the job level identified
            with the jnum input
        jnum (integer)
            job level distribution to plot if job_only input is True
        cell_border (boolean)
            if True, show a border around the heatmap cells
        eg_border_color (color value)
            color of cell border if measure_list includes 'eg' (employee group)
        job_border_color (color value)
            color of cell border when plotting job information
        chart_style (string)
            underlying chart style, any valid seaborn chart style (string)
        fur_color (color code in rgba, hex, or string style)
            custom color to signify furloughed employees (otherwise, last
            color in jnum_colors input will be used)
        empty_color (color value)
            cell color for cells with no data
        suptitle_size (integer or float)
            text size of chart super title
        title_size (integer or float)
            text size of chart title
        legend_size (integer or float)
            text size of chart legend
        xsize, ysize (integer or float)
            size of chart in inches (width, height)
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
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

    plot_colors.insert(0, empty_color)
    fur_integer = len(plot_colors) - 1
    if fur_color:
        plot_colors[-1] = fur_color

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

        suptitle = df_label + ' month ' + str(mnum) + \
            ':  ' + dict_settings['job_strs'][jnum - 1] + \
            '  job distribution'

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

        suptitle = df_label + ': month ' + str(mnum)

    if eg_list:
        np.put(heat_data,
               np.where(np.in1d(eg, np.array(eg_list), invert=True))[0],
               np.nan)

    heat_data = heat_data.reshape(rows, cols)

    cmap = mplclrs.ListedColormap(plot_colors,
                                  name='chart_cmap',
                                  N=len(plot_colors))

    with sns.axes_style(chart_style):

        fig, ax = plt.subplots(figsize=(xsize, ysize))

        if cell_border:
            sns.heatmap(heat_data, vmin=0, vmax=len(plot_colors),
                        cbar=False, annot=False,
                        cmap=cmap, linewidths=0.005,
                        linecolor=border_color, ax=ax)
        else:
            sns.heatmap(heat_data, vmin=0, vmax=len(plot_colors),
                        cbar=False, annot=False,
                        cmap=cmap, ax=ax)

        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=max(9, (min(12, ysize - 3))))
        ax.set_ylabel(str(cols) + ' per row',
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

            label_dict = dict_settings['p_dict_verbose']
            if eg_list:
                heat_unique = np.unique(np.array(eg_list)).astype(int)
            else:
                heat_unique = np.unique(eg).astype(int)

        if not job_only:

            label_dict = dict_settings['job_strs_dict']
            heat_unique = np.unique(heat_data[~np.isnan(heat_data)]) \
                .astype(int)

    if 'eg' in measure_list:

        label_dict = dict_settings['p_dict_verbose'].copy()
        label_dict[max(egs) + 1] = 'FUR'
        heat_unique = np.unique(heat_data[~np.isnan(heat_data)]) \
            .astype(int)
        heat_unique = heat_unique[heat_unique > 0]

    if measure_list == ['fur']:
        label_dict = {max(egs) + 1: 'FUR'}
        heat_unique = np.unique(heat_data[~np.isnan(heat_data)]).astype(int)

    try:
        heat_unique
    except NameError:
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
            except LookupError:
                pass

    if t_string:
        fig.suptitle(suptitle, fontsize=suptitle_size)
        ax.set_title(t_string, fontsize=title_size, y=1.01)
    else:
        ax.set_title(suptitle, fontsize=suptitle_size)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(recs, legend_labels, loc='center left',
              bbox_to_anchor=(1.01, 0.5),
              fontsize=legend_size)

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def quantile_bands_over_time(df, eg,
                             measure,
                             bins=20,
                             ds_dict=None,
                             year_clip=None,
                             kind='area',
                             quantile_ticks=False,
                             cm_name='Vega20c',
                             chart_style='ticks',
                             quantile_alpha=.75,
                             grid_alpha=.4,
                             custom_start=0.0,
                             custom_finish=1.0,
                             alt_bg_color=False,
                             bg_color='#faf6eb',
                             legend_size=13,
                             label_size=13,
                             xsize=14, ysize=8,
                             image_dir=None,
                             image_format='png'):
    '''Visualize quantile distribution for an employee group over time
    for a selected proposal.

    This chart answers the question of where the different employee groups
    will be positioned within the seniority list for future months and years.

    Note:  this is not a comparative study.  It is simply a presentation of
    resultant percentage positioning.

    The chart contains a background grid for reference and may display
    quantiles as integers or percentages, using a bar or area type display,
    and includes several chart color options.

    inputs
        df (dataframe)
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        eg (integer)
            employee group number
        measure (string)
            a list percentage input, either 'spcnt' or 'lspcnt'
        bins (integer)
            number of quantiles to calculate and display
        ds_dict (dictionary)
            output from load_datasets function
        year_clip (integer)
            maximum year to display on chart (requires 'clip'
            input to be True)
        kind (string)
            type of chart display, either 'area' or 'bar'
        quantile_ticks (boolean)
            if True, display integers along y axis and in legend representing
            quantiles.  Otherwise, present percentages.
        cm_name (string)
            colormap name (string), example: 'Set1'
        chart_style (string)
            style for chart output, any valid seaborn plotting style name
        quantile_alpha (float)
            alpha (opacity setting) value for quantile plot
        grid_alpha (float)
            opacity setting for background grid
        custom_start (float)
            custom colormap start level
            (a section of a standard colormap may be used to create
            a custom color mapping)
        custom_finish (float)
            custom colormap finish level
        alt_bg_color (boolean)
            if True, set the background chart color to the bg_color input value
        bg_color (color value)
            color for chart background if 'alt_bg_color' is True (string)
        legend_size (integer or float)
            text size for chart legend
        label_size (intger or float)
            text size for chart x and y axis labels
        xsize, ysize (integer or float)
            chart size inputs in inches (width, height)
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''
    ds, df_label = determine_dataset(df, ds_dict, return_label=True)

    if bins == 1:
        bins = 2
        print('bins must be 2 or greater, using bins == 2')

    cm_subsection = np.linspace(custom_start, custom_finish, bins)
    colormap = eval('cm.' + cm_name)
    quantile_colors = [colormap(x) for x in cm_subsection]

    quantiles = np.arange(1, bins + 1)
    minor_quantiles = quantiles - .5

    if year_clip:
        eg_df = ds[(ds.eg == eg) & (ds.date.dt.year <= year_clip)]
    else:
        eg_df = ds[ds.eg == eg]

    eg_df = eg_df[['date', 'empkey', measure]]
    eg_df['year'] = eg_df.date.dt.year

    years = pd.unique(eg_df.year)
    year_labels = np.arange(min(years), max(years) + 1, 1)

    bin_lims = np.linspace(0, 1, num=bins + 1, endpoint=True, retstep=False)
    result_arr = np.zeros((years.size, bin_lims.size - 1))

    if measure in ['spcnt', 'lspcnt']:
        filler = 1
    else:
        filler = 0

    grouped = eg_df.groupby(['year', pd.Grouper('empkey')])[measure].mean() \
        .reset_index()[['year', measure]].fillna(filler)

    denom = len(grouped[grouped.year == min(eg_df.year)])

    # in which quantile do we find employees over time?
    i = 0
    for year in years:
        this_year = grouped[grouped.year == year][measure]
        these_bins = pd.cut(this_year, bin_lims)
        these_counts = this_year.groupby(these_bins).count()
        these_pcnts = these_counts / denom
        result_arr[i, :] = these_pcnts
        i += 1

    frm = pd.DataFrame(result_arr, columns=quantiles, index=years)

    with sns.axes_style(chart_style):
        fig, ax = plt.subplots(figsize=(xsize, ysize))

    step = 1 / bins

    if kind == 'area':
        frm.plot(kind=kind, linewidth=1, stacked=True,
                 color=quantile_colors, alpha=quantile_alpha, ax=ax)
    elif kind == 'bar':
        frm.plot(kind=kind, width=1, stacked=True,
                 color=quantile_colors, alpha=quantile_alpha,
                 edgecolor='w', linewidth=.35, ax=ax)

    ax.set_ylim(0, 1)
    raw_yticks = np.arange(0, 1 + step, step)

    if bins > 20:
        raw_yticks = raw_yticks[::2]
    clipped_yticks = np.clip(raw_yticks, 0, 1)
    ax.set_yticks(clipped_yticks)

    ax.yaxis.set_major_formatter(pct_format())
    ax.invert_yaxis()
    if kind == 'area':
        ax.set_xticks(year_labels)
    ax.set_xticklabels(year_labels, rotation=80, ha='center')

    if len(ax.get_xticks()) > 20:
        for label in ax.xaxis.get_ticklabels()[1::2]:
            label.set_visible(False)

    if quantile_ticks:
        ax2 = ax.twinx()

        if bins > 20:
            ax2_yticks = minor_quantiles[::2]
            quantile_labels = quantiles[::2]
        else:
            ax2_yticks = minor_quantiles
            quantile_labels = quantiles
        ax2.set_yticks(quantiles, minor=True)
        ax2.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax2.set_yticks(ax2_yticks)
        ax2.set_yticklabels(quantile_labels)
        ax2.set_ylim(0, bins)

        ax2.invert_yaxis()
        ax2.grid(False)
        ax2.grid(which='minor', color='k', alpha=grid_alpha,
                 linestyle='dotted')

        ax.grid(which='major', color='gray', alpha=grid_alpha)
        ax.tick_params(axis='x', which='both', left='off',
                       right='off', labelleft='off')

        ax2.set_ylabel('original quantile', fontsize=label_size)
        ax2.yaxis.labelpad = 10
        legend_labels = quantiles
        legend_title = 'result quantile'

    else:
        ax.xaxis.grid(which='major', color='k', alpha=grid_alpha, ls='dotted')

        legend_labels = ['{percent:.1f}'
                         .format(percent=((quart * step) - step) * 100) +
                         ' - ' +
                         '{percent:.1%}'.format(percent=quart * step)
                         for quart in quantiles]
        legend_title = 'result_pcnt'

    if alt_bg_color:
        ax.set_facecolor(bg_color)

    ax.set_ylabel('original percentage', fontsize=label_size)
    ax.set_xlabel('year', fontsize=label_size)
    ax.xaxis.labelpad = 10
    ax.set_title(df_label + ', group ' + str(eg) +
                 ' quantile change over time\n' + str(bins) + ' quantiles',
                 fontsize=16, y=1.02)

    recs = []
    patch_alpha = min(quantile_alpha + .1, 1)
    legend_cols = int(bins / 30) + 1

    for i in np.arange(bins, dtype='int'):
        recs.append(mpatches.Rectangle((0, 0), 1, 1,
                                       fc=quantile_colors[i],
                                       alpha=patch_alpha))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    if quantile_ticks:
        ax2.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    handles, labels = ax.get_legend_handles_labels()

    if bins > 50:
        ax.legend_ = None
    else:
        ax.legend(recs, legend_labels, loc='center left',
                  bbox_to_anchor=(1.08, 0.5), ncol=legend_cols,
                  fontsize=legend_size, title=legend_title)

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def job_transfer(dfc, dfb, eg,
                 job_colors,
                 job_levels,
                 job_strs,
                 p_dict,
                 ds_dict=None,
                 gb_period='M',
                 min_date=None,
                 max_date=None,
                 tgt_jobs_list=None,
                 job_alpha=.85,
                 chart_style='whitegrid',
                 fur_color=None,
                 draw_face_color=False,
                 draw_grid=True,
                 grid_alpha=.2,
                 zero_line_color='m',
                 ytick_interval=None,
                 y_limit=None,
                 title_size=14,
                 legend_size=12,
                 xsize=14, ysize=9,
                 image_dir=None,
                 image_format='png'):
    '''plot a differential stacked area chart displaying color-coded job
    transfer counts over time.

    Output chart is actually 2 area charts (one for positive values and one
    for negative values) displayed on a shared axis.

    inputs
        dfc (dataframe)
            proposal (comparison) dataset to examine, may be a dataframe
            variable or a string key from the ds_dict dictionary object
        dfb (dataframe)
            baseline dataset; proposal dataset is compared to this
            dataset, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        eg (integer)
            integer code for employee group
        job_colors (list)
            list of colors for job levels, may be value from color dictionary
        job_levels (integer)
            number of job levels in data model
        job_strs (list)
            list of job descriptions (labels)
        p_dict (dictionary)
            dictionary of employee number codes to short string description

            example: {0: 'sa', 1: '1', 2: '2'}
        ds_dict (dictionary)
            output from load_datasets function
        gb_period (string)
            group_by period. default is 'M' for monthly, other options
            are 'Q' for quarterly and 'A' for annual
        min_date (string date format)
            if set, analyze job transfer data from this date forward
        max_date (string date format)
            if set, analyze job transfer data up to this date
        tgt_jobs_list (list)
            if not None, only plot job level(s) in this list
        job_alpha (float)
            chart alpha level for job transfer plotting (0.0 - 1.0)
        chart_style (string)
            seaborn plotting library style
        fur_color (color code in rgba, hex, or string style)
            custom color to signify furloughed employees (otherwise, last
            color in job_colors input will be used)
        draw_face_color (boolean)
            apply a transparent background to the chart, red below zero
            and green above zero
        draw_grid (boolean)
            show major tick label grid lines
        grid_alpha (float)
            opacity setting for grid lines (0.0 - 1.0)
        zero_line_color (color value)
            color of the horizontal line a zero
        ytick_interval (integer)
            optional manual ytick spacing setting (function has auto-spacing
            built in)
        y_limit (integer)
            optional manual y axis chart limit (enter positive value only).
            This input may be used to "lock" vertical scaling (shut off
            auto_scaling) for comparing gains and losses between proposals
            and employee groups.
        title_size (integer or float)
            chart title text size
        legend_size (integer or float)
            chart legend text size
        xsize (integer or float)
            horizontal size of chart
        ysize (integer or float)
            vertical size of chart
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''
    dsc, dfc_label = determine_dataset(dfc, ds_dict, return_label=True)
    dsb, dfb_label = determine_dataset(dfb, ds_dict, return_label=True)

    if max_date:
        compare_df = dsc[(dsc.eg == eg) & (dsc.date <= max_date)].copy()
        base_df = dsb[(dsb.eg == eg) & (dsb.date <= max_date)].copy()
    else:
        compare_df = dsc[dsc.eg == eg].copy()
        base_df = dsb[dsb.eg == eg].copy()

    if min_date:
        compare_df = compare_df[compare_df.date >= min_date].copy()
        base_df = base_df[base_df.date >= min_date].copy()

    # MAKE JOB COUNTS PER MONTH DATAFRAMES
    cg = compare_df[['date', 'jnum']].groupby(['date', 'jnum'])['jnum'] \
        .count().unstack().fillna(0)
    bg = base_df[['date', 'jnum']].groupby(['date', 'jnum'])['jnum'] \
        .count().unstack().fillna(0)

    for job_level in np.arange(1, job_levels + 1):
        if job_level not in bg:
            bg[job_level] = 0.0
        if job_level not in cg:
            cg[job_level] = 0.0

    bg.sort_index(axis=1, inplace=True)
    cg.sort_index(axis=1, inplace=True)

    # LIMIT JOBS TO TARGET LIST
    if tgt_jobs_list:
        bg = bg[sorted(set(tgt_jobs_list))]
        cg = cg[sorted(set(tgt_jobs_list))]
        # grab the corresponding job colors
        if len(tgt_jobs_list) == 1:
            job_colors = job_colors[tgt_jobs_list[0] - 1]
        else:
            jc = []
            for job in sorted(set(tgt_jobs_list)):
                jc.append(job_colors[job - 1])
            job_colors = jc

    # MAKE DIFFERENTIAL DATAFRAME
    diff2 = cg - bg
    diff2 = diff2.resample(gb_period).mean()
    diff2 = diff2.replace(0., np.nan)

    # CUSTOM FURLOUGH COLOR
    if fur_color and tgt_jobs_list is None:
        job_colors[-1] = fur_color

    # PLOT AX1
    with sns.axes_style(chart_style):
        fig, ax1 = plt.subplots(figsize=(xsize, ysize))

    diff2[diff2 > 0].plot(kind='area', stacked=True, color=job_colors,
                          ax=ax1, lw=0, alpha=job_alpha)
    ylimit1 = (ax1.get_ylim()[1] + 50) // 50 * 50
    ax1.set_ylim(-ylimit1, ylimit1)

    # PLOT AX2
    ax2 = ax1.twinx()
    diff2[diff2 < 0].plot(kind='area', stacked=True, color=job_colors,
                          ax=ax2, sharex=ax1, lw=0, alpha=job_alpha)
    if draw_grid:
        ax1.grid(which='both', c='gray', alpha=grid_alpha, ls='dotted')
        ax2.grid(which='both', c='gray', alpha=grid_alpha, ls='dotted')
    ylimit2 = (ax2.get_ylim()[0] // 50) * -50

    # SET GREATER Y AXIS LIMIT (IF PLOTTING ONLY TARGET JOBS)
    if ylimit2 > ylimit1:
        yl = ylimit2
    else:
        yl = ylimit1

    if y_limit:
        yl = y_limit

    ax1.set_ylim(-yl, yl)
    ax2.set_ylim(-yl, yl)

    # YTICKS
    if ytick_interval:
        interval = ytick_interval
    else:
        if yl > 500:
            interval = ((yl // 1000) + 1) * 100
        else:
            interval = ((yl // 500) + 1) * 50
    neg = np.arange(-interval, -yl - interval, -interval)
    pos = np.arange(0, yl + interval, interval)
    yticks = np.append(neg[::-1], pos)
    ax1.set_yticks(yticks)
    ax2.set_yticks(yticks)

    # REMOVE AX2 TICKS AND LEGEND
    ax2.tick_params(axis='both',
                    which='both',
                    right='off',
                    bottom='off',
                    labelright='off',
                    labelbottom='off')
    ax2.legend_.remove()

    # LEGEND
    job_labels = []
    legend_title = 'job'
    for col in diff2.columns.values.tolist():
        job_labels.append(job_strs[col - 1])

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax2.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, job_labels, title=legend_title, loc='center left',
               bbox_to_anchor=(1.01, 0.5),
               fontsize=legend_size)

    ax1.axhline(color=zero_line_color, alpha=.7, lw=1)

    # GAIN-LOSS BACKGROUND
    if draw_face_color:
        ymin, ymax = ax1.get_ylim()
        ax1.axhspan(0, ymax, facecolor='g', alpha=0.05, zorder=1)
        ax1.axhspan(0, ymin, facecolor='r', alpha=0.05, zorder=1)

    # AXIS LABELS
    ax1.set_ylabel('change in job count', fontsize=16)
    ax1.set_xlabel('date', fontsize=16, labelpad=15)

    # TITLE
    try:
        title_string = 'GROUP ' + p_dict[eg] + \
            ' Jobs Exchange' + '\n' + \
            dfc_label + \
            ' compared to ' + dfb_label
        ax1.set_title(title_string, fontsize=title_size, y=1.02)
    except (NameError, LookupError):
        print('error, problem creating title text')

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '_g' + str(eg) +
                    '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def editor(settings_dict,
           color_dict,
           base='standalone',
           compare='ds_edit',
           cond_list=None,
           eg_list=None,
           mean_len=80,
           size=18,
           alpha=.8,
           lin_reg_order=12,
           trim_xlim=True,
           ylim=None,
           xsize=16,
           ysize=11,
           strip_dot_size=2.5,
           strip_dot_alpha=1,
           strip_height=1,
           title_size=16,
           label_size=14,
           tick_size=13,
           legend_size=13,
           grid_alpha=.25,
           chart_style='whitegrid',
           bg_clr='white',
           show_grid=True):
    '''compare specific proposal attributes and interactively adjust
    list order.  may be used to minimize distortions.  utilizes ipywidgets.

    See the user guide for usage instructions.

    The function will recursively use an edited dataset so that an integrated
    list may be incrementally adjusted and examined at each step.  The edited
    dataset is created the first time a calculation is run.  Prior to the
    creation of the edited dataset ('dill/ds_edit.pkl'), the function will use
    the compare_ds input to select a previously calculated dataset for
    initial comparison.  The function then will revert to the new edited
    dataset for all subsequent calculations.

    If the user desires to save an edited dataset for further
    analysis, it must be manually copied and saved to another folder.

    "mode" dropdown
        reset
            delete the current edited dataset (if it exists) and start
            over with a calculated dataset (the "compare" input).
            No editing will take place with this selection.
        edit
            allow dataset editing to take place

    "x axis" dropdown
        prop order
            the output differential chart x axis will display integrated
            proposal (or edited dataset) order

        eg pcnt
            the x axis is arranged in native list percentage for each group

    files created/updated by the editor tool:

        ds_edit.pkl
        squeeze_vals.pkl
        p_new_order.pkl

    inputs
        settings_dict (dictionary)
            program settings dictionary generated by the build_program_files
            script
        color_dict (dictionary)
            dictionary containing color list string titles to lists of color
            values generated by the build_program_files script
        base (string)
            baseline dataset string name (used only when the display mode is
            set to "diff")
        compare (string)
            comparison dataset string name (also this is the initial dataset
            for attribute measurement when the mode is set to "abs")
        cond_list (list)
            conditions to apply when calculating dataset
        mean_len (integer)
            length of rolling mean if 'mean' selected for display
        eg_list (list)
            list of egs(employee groups) to compare and plot
        size (float or integer)
            chart dot size
        alpha (float or integer)
            chart dot transparency value (0.0 to 1.0)
        lin_reg_order (integer)
            polynomial fit order
        trim_xlim (boolean)
            if True, set x axis scale to match the length of the displayed
            values, otherwise lock x axis chart scale to match length of
            original integrated list length
        ylim (integer or float)
            if not None, limit the y axis scale to this input value.
            Helpful when outliers exist.
        xsize (integer or float)
            width of chart
        ysize (integer or float)
            height of chart
        strip_dot_size (integer or float)
            dot size for stripplot
        strip_dot_alpha (integer or float)
            dot transparency level for stripplot dots (0.0 to 1.0)
        strip_height (integer or float)
            height of stripplot (group density display)
        title_size (integer or float)
            chart title font size
        label_size (integer or float)
            chart x and y label font size
        tick_size (integer or float)
            chart x and y tick label font size
        grid_alpha (integer or float)
            transparency value for chart grid
        chart_style (string)
            seaborn chart style
        bg_clr (None or color value)
            if not None, color input for chart background
        show_grid (boolean)
            if True, show grid on chart

    '''
    persist = pd.read_pickle('dill/squeeze_vals.pkl')

    # ipywidgets layout for dropdowns, text boxes and checkboxes
    cb_layout = Layout(display='flex',
                       flex_flow='row',
                       width='60%',
                       justify_content='center')

    dd_layout = Layout(display='flex',
                       flex_flow='row',
                       width='90%',
                       justify_content='center')

    filt1_layout = Layout(width='20%')
    filt2_layout = Layout(width='40%')

    # --------display style checkboxes-----------------------

    chk_scatter = Checkbox(description='scatter', layout=cb_layout,
                           value=bool(persist['scat_val'].value))

    chk_fit = Checkbox(description='poly_fit', layout=cb_layout,
                       value=bool(persist['fit_val'].value))

    chk_mean = Checkbox(description='mean', layout=cb_layout,
                        value=bool(persist['mean_val'].value))

    # --------top row checkboxes and dropdown-----------------

    chk_ret = Checkbox(description='ret only', layout=Layout(width='25%'),
                       value=bool(persist['ret_val'].value))

    display_attrs = ['jobp', 'cat_order', 'spcnt', 'lspcnt',
                     'jnum', 'mpay', 'cpay', 'snum', 'lnum',
                     'ylong', 'mlong']

    drop_measure = Dropdown(options=display_attrs,
                            value=persist['drop_msr'].value,
                            description='display attr:',
                            layout=Layout(width='50%',
                                          border='solid 2px #99ccff',
                                          margin='1px',
                                          padding='1px'))

    chk_filt = Checkbox(description='filter', layout=Layout(width='25%'),
                        value=bool(persist['filt_val'].value))

    # -----------Chart Display Filter Widgets------------------------
    attr_list = ['', 'cat_order', 'jobp', 'jnum', 'mnum', 'eg',
                 'ldate', 'doh', 'retdate', 'ylong', 'mlong',
                 'sg', 'age', 'scale', 's_lmonths',
                 'lnum', 'snum', 'mnum', 'rank_in_job',
                 'mpay', 'cpay']

    dd1_attr = Dropdown(options=attr_list, value=persist['dd1_attr'].value,
                        layout=filt2_layout)
    dd2_attr = Dropdown(options=attr_list, value=persist['dd2_attr'].value,
                        layout=filt2_layout)
    dd3_attr = Dropdown(options=attr_list, value=persist['dd3_attr'].value,
                        layout=filt2_layout)

    dd1_oper = Dropdown(options=['<', '<=', '==', '!=', '>=', '>'],
                        value=persist['dd1_oper'].value, layout=filt1_layout)
    dd2_oper = Dropdown(options=['<', '<=', '==', '!=', '>=', '>'],
                        value=persist['dd2_oper'].value, layout=filt1_layout)
    dd3_oper = Dropdown(options=['<', '<=', '==', '!=', '>=', '>'],
                        value=persist['dd3_oper'].value, layout=filt1_layout)

    txt1_val = Text(value=persist['txt1_val'].value, layout=filt2_layout)
    txt2_val = Text(value=persist['txt2_val'].value, layout=filt2_layout)
    txt3_val = Text(value=persist['txt3_val'].value, layout=filt2_layout)

    # ----------------------------------------------------------------

    drop_display = Dropdown(options=['diff', 'abs'],
                            value=persist['drop_display'].value,
                            description='display', layout=dd_layout)

    drop_mode = Dropdown(options=['edit', 'reset'],
                         value=persist['drop_mode'].value,
                         description='mode', layout=dd_layout)

    drop_xax = Dropdown(options=['prop', 'eg pcnt'],
                        value=persist['drop_xax'].value,
                        description='x axis', layout=dd_layout)

    if drop_xax.value == 'prop':
        prop_order = True
        xval = 'proposal_order'
    else:
        prop_order = False
        xval = 'separate_eg_percentage'

    if drop_mode.value == 'reset':
        reset = True
    else:
        reset = False

    if drop_display.value == 'diff':
        diff = True
        yval = 'differential'
    else:
        diff = False
        yval = 'absolute value'

    # define path for edited datasets (ds_edit.pkl):
    edit_file = 'dill/ds_edit.pkl'
    # boolean value, True if ds_edit exists
    edit_file_exists = path.exists(edit_file)

    # set the COMPARE dataset...
    # test to see if user wishes to start over with another dataset
    if not reset:
        # if not, see if an edited dataset exists:
        if edit_file_exists:
            # if so, use it for calculations
            compare_ds = pd.read_pickle(edit_file)
            title_label = 'edited'

        else:
            # if not, use user input dataset (compare) for calculations
            if path.exists('dill/ds_' + compare + '.pkl'):
                compare_ds = pd.read_pickle('dill/ds_' + compare +
                                            '.pkl')
                title_label = compare
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
            compare_ds = pd.read_pickle('dill/ds_' + compare + '.pkl')
            title_label = compare
        except OSError:
            proposal_list = \
                list(pd.read_pickle('dill/proposal_names.pkl').proposals)
            compare_ds = pd.read_pickle('dill/ds_' + proposal_list[0] + '.pkl')
            title_label = '< using ' + proposal_list[0] + ' >'

    # set BASELINE dataset
    try:
        base_ds = pd.read_pickle('dill/_ds' + base + '.pkl')
    except OSError:
        try:
            base_ds = pd.read_pickle('dill/standalone.pkl')
        except OSError:
            # exit routine if baseline dataset not found
            print('invalid "base_ds" name input, neither ds_' +
                  base + '.pkl nor standalone.pkl not found\n')
            return

    max_month = max(compare_ds.mnum)

    mnum_caption = Label(value='Month', layout=dd_layout)

    mnum_operator = Dropdown(options=['<', '<=', '==', '!=', '>=', '>'],
                             value=persist['mnum_opr'].value,
                             layout=dd_layout)

    mnum_input = Dropdown(options=list(np.arange(0, max_month)
                                       .astype(str)),
                          value=persist['int_mnum'].value,
                          layout=dd_layout)

    measure = drop_measure.value

    a1 = dd1_attr.value
    a2 = dd2_attr.value
    a3 = dd3_attr.value

    o1 = dd1_oper.value
    o2 = dd2_oper.value
    o3 = dd3_oper.value

    v1 = txt1_val.value
    v2 = txt2_val.value
    v3 = txt3_val.value

    base_cols = [measure, 'mnum']
    comp_cols = [measure, 'mnum', 'new_order', 'eg']
    # add filter columns
    if chk_filt.value:
        filt_cols = list(set(([x for x in
                              [a1, a2, a3]
                              if x])))
        comp_cols = list(set().union(comp_cols, filt_cols))

    mnum_oper = mnum_operator.value
    mnum_val = mnum_input.value
    mnum_filt_str = ' '.join(['mnum', mnum_oper, mnum_val])
    base_str = '(base_ds.' + mnum_filt_str + ')'
    comp_str = '(compare_ds.' + mnum_filt_str + ')'
    if chk_ret.value:
        ret_filt_str = 'ret_mark == 1'
        base_ret_str = '(base_ds.' + ret_filt_str + ')'
        comp_ret_str = '(compare_ds.' + ret_filt_str + ')'
        base_str = ' & '.join([base_str, base_ret_str])
        comp_str = ' & '.join([comp_str, comp_ret_str])

    df = base_ds[eval(base_str)][base_cols].copy()

    df.rename(columns={measure: measure + '_b'}, inplace=True)
    # for stripplot and squeeze (month zero):
    data_reorder = compare_ds[compare_ds.mnum == 0][['eg']].copy()
    idx_df = data_reorder[[]].copy()
    idx_df['orig_order'] = np.arange(len(idx_df)) + 1
    slider_lim = len(data_reorder)
    data_reorder['new_order'] = np.arange(len(data_reorder)).astype(int) + 1

    # for drop_eg selection widget:
    drop_eg_options = sorted(list(pd.unique(data_reorder.eg).astype(str)))

    to_join_ds = compare_ds[eval(comp_str)][comp_cols].copy()
    # drop mnum column in base df to avoid duplicate columns during join
    del df['mnum']
    to_join_ds.rename(columns={measure: measure + '_c',
                               'new_order': 'proposal_order'}, inplace=True)
    df = df.join(to_join_ds)
    df.sort_values(by='proposal_order', inplace=True)
    df['proposal_order'] = np.arange(len(df)) + 1
    if trim_xlim:
        x_limit = int(max(df.proposal_order) // 100) * 100 + 100
    else:
        x_limit = slider_lim // 100 * 100 + 100

    df['eg_sep_order'] = df.groupby('eg').cumcount() + 1
    eg_sep_order = np.array(df.eg_sep_order)
    eg_arr = np.array(df.eg)
    eg_denom_dict = df.groupby('eg').eg_sep_order.max().to_dict()
    eg_set = sorted(list(pd.unique(df.eg)))
    max_eg_plus_one = max(eg_set) + 1
    denoms = np.zeros(eg_arr.size)

    for eg in eg_set:
        np.put(denoms, np.where(eg_arr == eg)[0], eg_denom_dict[eg])

    df['separate_eg_percentage'] = eg_sep_order / denoms

    if diff:
        if measure in ['spcnt', 'lspcnt', 'snum', 'lnum', 'cat_order',
                       'jobp', 'jnum']:

            df[yval] = df[measure + '_b'] - df[measure + '_c']

        else:
            df[yval] = df[measure + '_c'] - df[measure + '_b']
    else:
        df[yval] = df[measure + '_c']

    p_dict = settings_dict['p_dict']

    with sns.axes_style(chart_style):
        fig, ax = plt.subplots(figsize=(xsize, ysize))

    df.sort_values(by='proposal_order', inplace=True)

    filt_str = ''
    if chk_filt.value:
        df_display, filt_str = filter_ds(df,
                                         attr1=a1, oper1=o1, val1=v1,
                                         attr2=a2, oper2=o2, val2=v2,
                                         attr3=a3, oper3=o3, val3=v3)

    else:
        df_display = df

    if eg_list is not None:
        eg_list = [eg for eg in set(eg_list) if eg in eg_set]
        if eg_list:
            eg_set = eg_list
        else:
            print('invalid eg_list input, defaulting to all groups...')

    for eg in eg_set:
        try:
            data = df_display[df_display.eg == eg].copy()

            if chk_scatter.value:
                ax = data.plot(x=xval, y=yval, kind='scatter', linewidth=0.1,
                               color=color_dict['eg_colors'][eg - 1],
                               s=size, alpha=alpha,
                               label=p_dict[eg],
                               ax=ax)

            if chk_mean.value:
                data['ma'] = data[yval].rolling(mean_len).mean()
                ax = data.plot(x=xval, y='ma', lw=5,
                               color=color_dict['mean_colors'][eg - 1],
                               label=p_dict[eg],
                               alpha=.6, ax=ax)

            if chk_fit.value:
                if chk_scatter.value:
                    lin_reg_colors = color_dict['lin_reg_colors']
                else:
                    lin_reg_colors = color_dict['lin_reg_colors2']
                ax = sns.regplot(x=xval, y=yval, data=data,
                                 color=lin_reg_colors[eg - 1],
                                 label=p_dict[eg],
                                 scatter=False, truncate=True, ci=50,
                                 order=lin_reg_order,
                                 line_kws={'lw': 20,
                                           'alpha': .4},
                                 ax=ax)
        except TypeError:
            pass

    ax.set_xlim(0, x_limit)

    if measure in ['jobp', 'jnum']:

        if diff is True:
            ymin = math.floor(min(df[yval]))
            ymax = math.ceil(max(df[yval]))
            scale_lim = max(abs(ymin), ymax)
            ax.set_yticks(np.arange(-scale_lim, scale_lim + 1, 1))

            if ylim is not None:
                ax.set_ylim(-ylim, ylim)
            else:
                ax.set_ylim(-scale_lim, scale_lim)
        else:
            ymax = math.ceil(max(df_display[yval]))
            ax.set_yticks(np.arange(1, ymax + 1, 1))
            if ylim is not None:
                ax.set_ylim(.5, ylim)

    ax.tick_params(labelsize=tick_size)

    for item in ([ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(label_size)

    ax.set_title(title_label + ' ' + yval + ': ' + measure +
                 '  (' + filt_str + ')',
                 fontsize=title_size)
    ax.set_xlim(xmin=0)

    if measure in ['spcnt', 'lspcnt']:
        ax.yaxis.set_major_formatter(pct_format())

    if xval == 'separate_eg_percentage':
        ax.set_xlim(xmax=1)
        ax.xaxis.set_major_formatter(pct_format())
        ax.set_xticks(np.arange(0, 1.05, .05))

    if diff:
        ax.axhline(0, c='m', ls='-', alpha=1, lw=1.5)
    else:
        if measure in ['cat_order', 'jnum', 'jobp', 'spcnt',
                       'lspcnt', 'snum', 'lnum']:
            ax.invert_yaxis()
        if measure in ['jnum', 'jobp']:
            ax.axhline(1, c='m', ls='-', alpha=1, lw=1.5)
        else:
            ax.axhline(0, c='m', ls='-', alpha=1, lw=1.5)
    ax.invert_xaxis()
    ax.legend(markerscale=2, fontsize=legend_size)
    if bg_clr is not None:
        ax.set_facecolor(bg_clr)
    if show_grid:
        ax.grid(lw=1, ls='-', c='grey', alpha=grid_alpha)

    plt.close(fig)

    if prop_order:
        if persist['junior'].value < -1:
            junior_init = persist['junior'].value
            senior_init = persist['senior'].value
        else:
            junior_init = int(.8 * -x_limit)
            senior_init = int(.2 * -x_limit)
    else:
        if persist['junior'].value >= -1:
            junior_init = persist['junior'].value
            senior_init = persist['senior'].value
        else:
            junior_init = -.8
            senior_init = -.2

    drop_p_dict = {'1': 1, '2': 2, '3': 3}
    drop_dir_dict = {'u  >>': 'u', '<<  d': 'd'}
    incr_dir_dict = {'u  >>': -1, '<<  d': 1}

    drop_eg = Dropdown(options=drop_eg_options,
                       value=persist['drop_eg_val'].value,
                       description='emp grp', layout=dd_layout)

    drop_dir = Dropdown(options=['u  >>', '<<  d'],
                        value=persist['drop_dir_val'].value,
                        description='sqz dir', layout=dd_layout)

    drop_squeeze = Dropdown(options=['log', 'slide'],
                            value=persist['drop_sq_val'].value,
                            description='sq type', layout=dd_layout)

    slide_factor = IntSlider(value=persist['slide_fac_val'].value,
                             min=1,
                             max=400,
                             step=1,
                             description='squeeze',
                             layout=Layout(display='flex',
                                           flex='2 1 auto',
                                           width='100%'),
                             margin='10px')

    slide_factor.style.handle_color = '#f7c3a1'

    if prop_order:
        min_val = -slider_lim
        step = 1
        rg = IntRangeSlider(min=min_val, max=-1, step=step,
                            layout=Layout(width='90%'),
                            value=(junior_init, senior_init),
                            continuous_update=True,
                            readout=False)
        jun_caption = str(-junior_init)
        sen_caption = str(-senior_init)

        def range_slider_change(change):
            j_caption.value = str(-rg.value[0])
            s_caption.value = str(-rg.value[1])
    else:
        min_val = -1
        step = .001
        rg = FloatRangeSlider(min=min_val, max=0, step=step,
                              layout=Layout(width='90%'),
                              value=(junior_init, senior_init),
                              continuous_update=True,
                              readout=False)
        jun_caption = "{:.1%}".format(-junior_init)
        sen_caption = "{:.1%}".format(-senior_init)

        def range_slider_change(change):
            j_caption.value = "{:.1%}".format(-rg.value[0])
            s_caption.value = "{:.1%}".format(-rg.value[1])

    j_caption = Label(value=jun_caption, layout=Layout(width='10%'))
    s_caption = Label(value=sen_caption, layout=Layout(width='10%'))

    v1line = ax.axvline(2, color='c', lw=2, ls='dashed')
    v2line = ax.axvline(200, color='m', lw=2, ls='dashed')
    range_list = [0, 0]

    def set_cursor(edit_zone):
        v1line.set_xdata((-rg.value[1], -rg.value[1]))
        v2line.set_xdata((-rg.value[0], -rg.value[0]))
        range_list[0] = -rg.value[1]
        range_list[1] = -rg.value[0]
        sleep(.05)
        display(fig)

    rg.observe(range_slider_change, names='value')

    range_sel = interactive(set_cursor, edit_zone=rg)

    def perform_squeeze(b):

        squeeze_eg = drop_p_dict[drop_eg.value]

        if prop_order:
            jval_line = v2line.get_xdata()[0]
            sval_line = v1line.get_xdata()[0]
            jval, sval = f.find_squeeze_vals(idx_df, df,
                                             [jval_line, sval_line],
                                             col1='orig_order',
                                             col2='proposal_order')
        else:
            ints = data_reorder.new_order.values
            pcnts = ints / ints.size
            p_dict = dict(zip(pcnts, ints))
            v2 = v2line.get_xdata()[0]
            v1 = v1line.get_xdata()[0]
            # find closest key value to match vline pcnt values
            jval = p_dict[v2] if v2 in p_dict \
                else p_dict[min(p_dict.keys(), key=lambda k: abs(k - v2))]
            sval = p_dict[v1] if v1 in p_dict \
                else p_dict[min(p_dict.keys(), key=lambda k: abs(k - v1))]

        direction = drop_dir_dict[drop_dir.value]
        factor = slide_factor.value * .005
        incr_dir_correction = incr_dir_dict[drop_dir.value]
        increment = slide_factor.value * incr_dir_correction

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
        data_reorder['new_order'] = np.arange(1, len(data_reorder) + 1,
                                              dtype='int')

        with sns.axes_style(chart_style):
            fig, ax2 = plt.subplots(figsize=(xsize,
                                             strip_height * len(eg_set)))

        ax2 = sns.stripplot(x='new_order', y='eg',
                            data=data_reorder, jitter=.5,
                            orient='h', order=np.arange(1,
                                                        max_eg_plus_one,
                                                        1),
                            palette=color_dict['eg_colors'],
                            size=strip_dot_size,
                            alpha=strip_dot_alpha,
                            linewidth=0, split=True)

        for item in ([ax2.xaxis.label, ax2.yaxis.label] +
                     ax2.get_xticklabels() + ax2.get_yticklabels()):
            item.set_fontsize(label_size)

        if bg_clr is not None:
            ax2.set_facecolor(bg_clr)

        ax2.set_xticks(np.arange(0, len(data_reorder), 1000))
        ax2.set_ylabel('eg\n')

        ax2.set_xlim(len(data_reorder) + 1, 0)
        plt.show()

        data_reorder[['new_order']].to_pickle('dill/p_new_order.pkl')

        store_vals()

    # grab the widget values, create a dataframe, pickle
    def store_vals():
        persist_df = pd.DataFrame({'drop_eg_val': drop_eg.value,
                                   'drop_dir_val': drop_dir.value,
                                   'drop_sq_val': drop_squeeze.value,
                                   'slide_fac_val': slide_factor.value,
                                   'scat_val': chk_scatter.value,
                                   'fit_val': chk_fit.value,
                                   'filt_val': chk_filt.value,
                                   'ret_val': chk_ret.value,
                                   'mean_val': chk_mean.value,
                                   'drop_msr': drop_measure.value,
                                   'dd1_oper': dd1_oper.value,
                                   'dd2_oper': dd2_oper.value,
                                   'dd3_oper': dd3_oper.value,
                                   'drop_display': drop_display.value,
                                   'drop_mode': drop_mode.value,
                                   'drop_xax': drop_xax.value,
                                   'dd1_attr': dd1_attr.value,
                                   'dd2_attr': dd2_attr.value,
                                   'dd3_attr': dd3_attr.value,
                                   'mnum_opr': mnum_operator.value,
                                   'int_mnum': mnum_input.value,
                                   'txt1_val': txt1_val.value,
                                   'txt2_val': txt2_val.value,
                                   'txt3_val': txt3_val.value,
                                   'junior': rg.value[0],
                                   'senior': rg.value[1]},
                                  index=['value'])

        persist_df.to_pickle('dill/squeeze_vals.pkl')

    def run_cell(ev):
        # 'new_order' is simply a placeholder here.
        # This is where ds_edit.pkl is generated (compute_measures script)
        store_vals()
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

    button_layout = Layout(width='16%', margin='5px')
    buttons_layout = Layout(display='flex', flex_flow='row',
                            justify_content='space-around')

    button_calc = Button(description="calculate", layout=button_layout)
    button_plot = Button(description="plot", layout=button_layout)
    button_squeeze = Button(description='squeeze', layout=button_layout)

    button_calc.on_click(run_cell)
    button_plot.on_click(redraw)
    button_squeeze.on_click(perform_squeeze)

    def drop_mode_change(change):
        store_vals()
    drop_mode.observe(drop_mode_change, names='value')

    button_calc.style.button_color = 'lightblue'
    button_plot.style.button_color = 'plum'
    button_squeeze.style.button_color = 'lightgreen'

    button_items = [Box([j_caption, button_squeeze, button_calc,
                         button_plot, s_caption],
                        layout=buttons_layout)]

    hbuttons = Box(button_items, layout=Layout(display='flex',
                   flex_flow='column', align_items='stretch', width='95%',
                   margin='5px'))

    cb_col_layout = Layout(display='flex',
                           flex_flow='column',
                           width='10%',
                           justify_content='center')

    dd_col_layout1 = Layout(display='flex',
                            flex_flow='column',
                            width='20%',
                            justify_content='center',
                            border='solid 2px #e6e6e6',
                            padding='1px',
                            margin='2px')

    dd_col_layout2 = Layout(display='flex',
                            flex_flow='column',
                            width='10%',
                            justify_content='center',
                            border='solid 2px #e6e6e6',
                            padding='1px',
                            margin='2px')

    form_layout = Layout(display='flex',
                         width='95%',
                         justify_content='space-around')

    vertcap_layout = Layout(display='flex',
                            flex_flow='column',
                            width='5%',
                            justify_content='center')

    filt_layout = Layout(display='flex',
                         flex_flow='row',
                         align_items='stretch',
                         margin='1px')

    filt1_cap = Label(value='Filter 1:', layout=dd_layout)
    filt2_cap = Label(value='Filter 2:', layout=dd_layout)
    filt3_cap = Label(value='Filter 3:', layout=dd_layout)

    filt_cap_box = Box((filt1_cap, filt2_cap, filt3_cap),
                       layout=vertcap_layout)

    filt1_items = [dd1_attr, dd1_oper, txt1_val]
    filt2_items = [dd2_attr, dd2_oper, txt2_val]
    filt3_items = [dd3_attr, dd3_oper, txt3_val]

    filt1_row = Box(children=filt1_items, layout=filt_layout)
    filt2_row = Box(children=filt2_items, layout=filt_layout)
    filt3_row = Box(children=filt3_items, layout=filt_layout)

    filter_box = VBox([filt1_row, filt2_row, filt3_row])

    items = [Box([chk_scatter, chk_fit, chk_mean], layout=cb_col_layout),
             Box([drop_display, drop_mode, drop_xax], layout=dd_col_layout1),
             Box([drop_squeeze, drop_eg, drop_dir], layout=dd_col_layout1),
             Box([mnum_caption, mnum_operator, mnum_input],
                 layout=dd_col_layout2),
             filt_cap_box,
             filter_box]

    top_row_items = [drop_measure, chk_ret, chk_filt]
    top_row = Box(children=top_row_items, layout=filt_layout)

    top_widgets = HBox([slide_factor, top_row], layout=form_layout)
    hdropdowns = Box(items, layout=form_layout)

    display(VBox((top_widgets,
                  hdropdowns,
                  hbuttons,
                  range_sel)))


def reset_editor():
    '''reset widget selections to default.
    (for use when invalid input is selected resulting in an exception)
    '''
    def reset(x):
        init_editor_vals = pd.DataFrame({'dd1_attr': '',
                                         'dd2_attr': '',
                                         'dd3_attr': '',
                                         'dd1_oper': '==',
                                         'dd2_oper': '==',
                                         'dd3_oper': '==',
                                         'drop_dir_val': '<<  d',
                                         'drop_display': 'diff',
                                         'drop_eg_val': '2',
                                         'drop_mode': 'edit',
                                         'drop_msr': 'spcnt',
                                         'drop_sq_val': 'log',
                                         'drop_xax': 'prop',
                                         'filt_val': False,
                                         'fit_val': False,
                                         'int_mnum': '0',
                                         'junior': 1000,
                                         'mean_val': False,
                                         'mnum_opr': '>=',
                                         'ret_val': True,
                                         'scat_val': True,
                                         'senior': 500,
                                         'slide_fac_val': 100,
                                         'txt1_val': '',
                                         'txt2_val': '',
                                         'txt3_val': ''},
                                        index=['value'])

        init_editor_vals.to_pickle('dill/squeeze_vals.pkl')

    button_reset = Button(description='reset editor',
                          background_color='#ffff99', width='120px')
    button_reset.on_click(reset)
    display(button_reset)


def eg_multiplot_with_cat_order(df, mnum, measure,
                                xax, job_strs,
                                span_colors,
                                job_levels,
                                settings_dict,
                                attr_dict,
                                ds_dict=None,
                                fur_color=None,
                                single_eg=False,
                                num=1,
                                exclude_fur=False,
                                plot_scatter=True,
                                s=20, a=.7, lw=0,
                                job_bands_alpha=.3,
                                title_size=14,
                                tick_size=12,
                                label_pad=110,
                                chart_style='whitegrid',
                                remove_ax2_border=True,
                                lgd_h_adj=None,
                                xsize=13, ysize=10,
                                image_dir=None,
                                image_format='png'):
    '''num input options:
                   {1: 'eg1_with_sg',
                    2: 'eg2',
                    3: 'eg3',
                    4: 'eg1_no_sg',
                    5: 'sg_only'
                    }

    num input is used with the single_eg input - output is plot of only
    corresponding employee group.

    sg refers to special group - a group with special job rights

    inputs
        df (dataframe)
            pandas dataframe input
        mnum (integer)
            month number for analysis
        measure (string)
            dataframe column name (attribute for analysis)
        xax (string)
            x axis attribute
        job_strs (list)
            list of job descriptions for labels (normally sdict['job_strs'])
        span_colors (list)
            list of colors for job level zones (normally cdict['job_colors'])
        job_levels (integer)
            number of job levels in model (sdict['num_of_job_levels'])
        settings_dict (dictionary)
            program job settings dictionary
        attr_dict (dictionary)
            program attribute name to attribute description dictionary
        ds_dict (dictionary)
            output from load_datasets function
        fur_color (string color value)
            if not None, color for furlough span color
        single_eg (boolean)
            if True, use the num input options to plot data for one group of
            employees
        num (integer)
            selective plot option number (see docstring before input section)
        exclude_fur (boolean)
            if True, remove furloughed employees from input data
        plot_scatter (boolean)
            if True (default), plot a scatter chart, otherwise plot a line
            chart
        s (integer or float)
            size of scatter markers if a plot_scatter input is True
        a (float)
            transparency value for both line plots and scatter plots
            (0.0 to 1.0)
        lw (integer or float)
            width of maker edge lines with a scatter plot
        job_bands_alpha (float)
            transparency value for job level color spans
        title_size (integer or float)
            text size of chart title
        tick_size (integer or float)
            text size of chart tick labels
        label_pad (integer)
            minimum padding between job description labels that would
            otherwise overlap
        chart_style (string)
            any seaborn plotting style name
        remove_ax2_border (boolean)
            if True, remove axis 2 (ax2) chart spines
        xsize, ysize (integer or float)
            width and height of chart
        lgd_h_adj (float)
            set to a small float value (for example: .02, -.01) to adjust
            the horizontal position of the chart legend if required.  Use
            negative values to move left, positive values to move right
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''
    df, df_label = determine_dataset(df, ds_dict, return_label=True)

    if fur_color:
        span_colors[-1] = fur_color
    max_count = df.groupby('mnum').size().max()
    df = df[df.mnum == mnum].copy()

    with sns.axes_style(chart_style):
        fig, ax1 = plt.subplots(figsize=(xsize, ysize))

    if measure == 'cat_order':

        tdict = pd.read_pickle('dill/dict_job_tables.pkl')
        table = tdict['table']

        job_ticks = np.cumsum(table[0][mnum])
        job_ticks = np.append(job_ticks, max_count)
        job_ticks = np.insert(job_ticks, 0, 0)

    if single_eg:

        grp_dict = {1: 'eg1_with_sg',
                    2: 'eg2',
                    3: 'eg3',
                    4: 'eg1_no_sg',
                    5: 'sg_only'
                    }

        clr_dict = {1: 'black',
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

        clr = clr_dict[num]

        if plot_scatter:
            df.plot(x=xax, y=measure, kind='scatter', color=clr,
                    label=label, linewidth=lw, s=s, ax=ax1)
        else:
            df.set_index(xax, drop=True)[measure].plot(label=label, color=clr,
                                                       ax=ax1)
            print('''Ignore the vertical lines.
                  Look right to left within each job level
                  for each group\'s participation''')
        ax1.set_title(grp_dict[num] + ' job disbursement - ' +
                      df_label + ' month=' + str(mnum), y=1.02,
                      fontsize=title_size)

    else:

        if exclude_fur:
            df = df[df.fur == 0]

        d1 = df[(df.eg == 1) & (df.sg == 1)]
        d2 = df[(df.eg == 1) & (df.sg == 0)]
        d3 = df[df.eg == 2]
        d4 = df[df.eg == 3]

        if plot_scatter:
            d1.plot(x=xax, y=measure, kind='scatter',
                    label='eg1_sg_only', color='#5cd65c',
                    alpha=a, s=s, linewidth=lw, ax=ax1)
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
            d1.set_index(xax, drop=True)[measure].plot(label='eg1_sg_only',
                                                       color='green',
                                                       alpha=a,
                                                       ax=ax1)

            d2.set_index(xax, drop=True)[measure].plot(label='eg1_no_sg',
                                                       color='black',
                                                       alpha=a,
                                                       ax=ax1)

            d3.set_index(xax, drop=True)[measure].plot(label='eg2',
                                                       color='blue',
                                                       alpha=a,
                                                       ax=ax1)

            d4.set_index(xax, drop=True)[measure].plot(label='eg3',
                                                       color='#FF6600',
                                                       alpha=a,
                                                       ax=ax1)
            print('''Ignore the vertical lines.  \
                  Look right to left within each job \
                  level for each group\'s participation''')

        ax1.set_title('job disbursement - ' +
                      df_label + ' month ' + str(mnum), y=1.02)

    ax1.tick_params(labelsize=tick_size)

    if measure in ['snum', 'spcnt', 'lspcnt',
                   'jnum', 'jobp', 'fbff', 'cat_order']:
        ax1.invert_yaxis()

        if measure in ['spcnt', 'lspcnt']:
            ax1.set_yticks(np.arange(1, -.05, -.05))
            ax1.yaxis.set_major_formatter(pct_format())
            ax1.set_ylim(1, 0)
        else:
            ax1.set_yticks(np.arange(0, max_count, 1000))
            ax1.set_ylim(max_count, 0)

        if measure in ['cat_order']:
            with sns.axes_style('white'):
                ax2 = ax1.twinx()
            if remove_ax2_border:
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax2.spines[axis].set_linewidth(0.0)

            ax1_lims = ax1.get_ylim()
            reversed_ax1_lims = (ax1_lims[1], ax1_lims[0])
            ax2.set_ylim(reversed_ax1_lims)

            axis2_lbl_locs = []
            axis2_lbls = []

            for i in np.arange(1, job_ticks.size):
                axis2_lbl_locs.append(round((job_ticks[i - 1] +
                                             job_ticks[i]) / 2))
                axis2_lbls.append(job_strs[i - 1])

            axis2_lbl_locs = add_pad(axis2_lbl_locs, pad=label_pad)

            ax2.set_yticks(axis2_lbl_locs)
            ax2.set_yticklabels(axis2_lbls)

            for level in job_ticks:
                ax1.axhline(y=level, c='.8', ls='-', alpha=.8, lw=.6, zorder=0)

            ax2.invert_yaxis()

            # plot job band background on chart
            for i in np.arange(1, job_ticks.size):
                ax2.axhspan(job_ticks[i - 1], job_ticks[i],
                            facecolor=span_colors[i - 1],
                            alpha=job_bands_alpha)
            ax1.grid(ls='dashed', lw=.5)

    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:

        ax1.set_yticks(np.arange(0, job_levels + 2, 1))
        yticks = ax1.get_yticks().tolist()

        for i in np.arange(1, len(yticks)):
            yticks[i] = job_strs[i - 1]

        ax1.axhspan(job_levels + 1, job_levels + 2, facecolor='.8', alpha=0.2)
        ax1.set_yticklabels(yticks, va='top')
        ax1.axhline(y=job_levels + 1, c='.8', ls='-', alpha=.8, lw=3)
        ax1.set_ylim(job_levels + 1.5, 0.5)

    if xax in ['snum']:
        ax1.set_xlim(max_count, 0)

    if xax in ['spcnt', 'lspcnt']:
        ax1.xaxis.set_major_formatter(pct_format())
        ax1.set_xticks(np.arange(0, 1.1, .1))
        ax1.set_xlim(1, 0)

    if xax in ['age']:
        if settings_dict['ret_age_increase']:
            month_val = 1 / 12
            months_incr = \
                sum(np.array(settings_dict['ret_incr'])[:, -1].astype(int))
            yr_add_decimal = months_incr * month_val
            ret_age_limit = settings_dict['ret_age'] + yr_add_decimal
        else:
            ret_age_limit = settings_dict['ret_age']
        ax1.set_xlim(xmax=ret_age_limit)

    if xax in ['ylong']:
        ax1.set_xticks(np.arange(0, 55, 5))
        ax1.set_xlim(-0.5, max(df.ylong) + 1)

    ax1.tick_params(labelsize=tick_size)

    # LEGEND --------------
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    if measure in ['cat_order']:
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        lgd_adj = 1.04
    else:
        lgd_adj = .99

    # allow manual adjustment of legend horizontal position
    if lgd_h_adj is not None:
        lgd_adj = lgd_adj + lgd_h_adj

    ax1.legend(bbox_to_anchor=(lgd_adj, .5), loc='center left',
               borderaxespad=4, frameon=True, fancybox=True,
               shadow=True, markerscale=2)
    # ---------------------

    ax1.set_ylabel(attr_dict[measure])
    ax1.set_xlabel(attr_dict[xax])

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def diff_range(df_list, dfb,
               measure, eg_list,
               attr_dict,
               ds_dict=None,
               cm_name='Set1',
               attr1=None, oper1='>=', val1=0,
               attr2=None, oper2='>=', val2=0,
               attr3=None, oper3='>=', val3=0,
               year_clip=2042,
               show_range=False,
               range_alpha=.25,
               show_mean=True,
               normalize_y=False,
               suptitle_size=16,
               title_size=16,
               tick_size=13,
               label_size=16,
               legend_size=14,
               chart_style='whitegrid',
               ysize=6, xsize=11,
               image_dir=None,
               image_format='png'):
    '''Plot a range of differential attributes or a differential
    average over time.  Individual employee groups and proposals may
    be selected.  Each chart indicates the results for one group with
    color bands or average lines indicating the results for that group
    under different proposals.  This is different than the usual method
    of different groups being plotted on the same chart.

    inputs
        df_list (list)
            list of datasets to compare, may be ds_dict (output of
            load_datasets function) string keys or dataframe variable(s)
            or mixture of each
        dfb (dataframe, can be proposal string name)
            baseline dataset, accepts same input types as df_list above
        measure (string)
            differential data to compare
        eg_list (list)
            list of integers for employee groups to be included in analysis.
            example: [1, 2, 3]
            A chart will be produced for each employee group number.
        eg_colors (list)
            list of colors to represent different proposal results
        attr_dict (dictionary)
            dataset column name description dictionary
        ds_dict (dictionary)
            output from load_datasets function
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (integer, float, date as string, string (as appropriate))
            attr(n) limiting value (combined with oper(n)) as string
        year_clip (integer)
            only plot data up to and including this year
        show_range (boolean)
            show a transparent background on the chart representing
            the range of values for each measure for each proposal
        range_alpha (float)
            transparancy level for range plotting (0.0 to 1.0)
        show_mean (boolean)
            plot a line representing the average of the measure values for
            the group under each proposal
        normalize_y (boolean)
            if measure is 'spcnt' or 'lspcnt', equalize the range of the
            y scale on all charts (-.5 to .5)
        suptitle_size (integer or font)
            text size of chart super title
        title_size (integer or font)
            text size of chart title
        tick_size (integer or font)
            text size of chart tick labels
        label_size (integer or font)
            text size of chart x and y axis labels
        legend_size (integer or font)
            text size of the legend labels
        chart_style (string)
            any valid seaborn plotting style (string)
        xsize, ysize (integer or font)
            size of chart in inches (width and height)
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
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

    color_list = make_color_list(num_of_colors=len(df_list),
                                 cm_name_list=[cm_name])
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

        with sns.axes_style(chart_style):
            fig, ax = plt.subplots(figsize=(xsize, ysize))

        if show_range:
                sa_ds[sa_ds.eg == eg][cols].set_index('date') \
                    .plot(color=color_list, alpha=range_alpha, ax=ax)
                ax.grid(lw=1, ls='--', c='grey', alpha=.25)
                if show_mean:
                    ax.legend_ = None
                    plt.draw()
        if show_mean:

            if show_range:
                sa_ds[sa_ds.eg == eg][cols].set_index('date') \
                    .resample('Q').mean().plot(color=color_list, ax=ax)
            else:
                sa_ds[sa_ds.eg == eg][cols].set_index('date') \
                    .resample('Q').mean().plot(color=color_list, ax=ax)

        if measure in ['spcnt', 'lspcnt', 'jobp', 'jnum',
                       'cat_order']:
            ax.invert_yaxis()

        ax.axhline(c='m', lw=2, ls='--')
        if measure in ['spcnt', 'lspcnt']:
            ax.yaxis.set_major_formatter(pct_format())
            if normalize_y:
                ax.set_ylim(.5, -.5)
            ax.set_yticks = np.arange(.5, -.55, .05)

        suptitle = 'Employee Group ' + str(eg) + ' ' +\
            attr_dict[measure] + ' differential'
        if tb_string:
            fig.suptitle(suptitle, fontsize=suptitle_size)
            ax.set_title(tb_string, fontsize=title_size)
        else:
            ax.set_title(suptitle, fontsize=title_size)

        plt.tight_layout()

        # LEGEND --------------
        handles, labels = ax.get_legend_handles_labels()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(handles, labels, bbox_to_anchor=(1.01, .5),
                  loc='center left', fontsize=legend_size)
        # ---------------------

        ax.tick_params(axis='y', labelsize=tick_size)
        ax.tick_params(axis='x', labelsize=tick_size)
        ax.xaxis.label.set_size(label_size)

        if image_dir:
            func_name = sys._getframe().f_code.co_name
            if not path.exists(image_dir):
                makedirs(image_dir)
            plt.savefig(image_dir + '/' + func_name + ' - group' + str(eg) +
                        '.' + image_format,
                        bbox_inches='tight', pad_inches=.25)
        plt.show()


def job_count_charts(dfc, dfb,
                     settings_dict,
                     eg_colors,
                     eg_list=None,
                     ds_dict=None,
                     attr1=None, oper1='>=', val1=0,
                     attr2=None, oper2='>=', val2=0,
                     attr3=None, oper3='>=', val3=0,
                     plot_egs_sep=False,
                     plot_total=True,
                     xax='date',
                     year_max=None,
                     chart_style='darkgrid',
                     base_ls='solid',
                     prop_ls='dotted',
                     base_lw=1.6,
                     prop_lw=2.5,
                     suptitle_size=14,
                     title_size=12,
                     total_color='g',
                     xsize=5, ysize=4,
                     image_dir=None,
                     image_format='png'):
    '''line-style charts displaying job category counts over time.

    optionally display employee group results on separate charts or together

    inputs
        dfc (dataframe)
            proposal (comparison) dataset to examine, may be a dataframe
            variable or a string key from the ds_dict dictionary object
        dfb (dataframe)
            baseline dataset; proposal dataset is compared to this
            dataset, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        settings_dict (dictionary)
            program settings dictionary generated by the build_program_files
            script
        eg_colors (list)
            list of color values for plotting the employee groups, length is
            equal to the number of employee groups in the data model
        eg_list (list)
            list of employee group codes to plot
            Example: [1, 2]
        ds_dict (dictionary)
            variable assigned to load_datasets function output
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (integer, float, date as string, string (as appropriate))
            attr(n) limiting value (combined with oper(n)) as string
        plot_egs_sep (boolean)
            if True, plot each employee group job level counts separately
        plot_total (boolean)
            if True, include the combined job counts on chart(s)
        xax (string)
            x axis groupby attribute, options are 'date' or 'mnum', default is
            'date'
        year_max (integer)
            maximum year to include on chart
            Example:  if input is 2030, chart would display data from
            beginning of data model through 2030 (integer)
        base_ls (string)
            line style for base job count line(s)
        prop_ls (string)
            line style for comparison (proposal) job count line(s)
        base_lw (float)
            line width for base job count line(s)
        prop_lw (float)
            line width for comparison (proposal) job count lines
        suptitle_size (integer or float)
            text size of chart super title
        title_size (integer or float)
            chart title(s) font size
        total_color (color value)
            color for combined job level count from all employee groups
        xsize, ysize (integer or float)
            size of chart display in inches (width and height)
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
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

    suptitle = dfc_label + ' vs ' + dfb_label + \
        ' (dotted line is comparison, solid line is base)' + '\n' + t_string

    prop = dfc_filt[[xax, 'jnum', 'eg']]
    base = dfb_filt[[xax, 'jnum', 'eg']]

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
        except TypeError:
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

                with sns.axes_style(chart_style):
                    ax = plt.subplot(num_jobs, num_egplots, plot_idx)

                ax.tick_params(axis='both', which='both', labelsize=10)
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
                count_plot(eg_jobs, jnum, dum, eg_colors[eg - 1],
                           ax, base_lw, 1, ls=base_ls)

                eg_jobs = prop_jobs[prop_jobs.eg == eg]
                # plot employee group proposal job count
                count_plot(eg_jobs, jnum, dum, eg_colors[eg - 1],
                           ax, prop_lw, 1, ls=prop_ls)

                ax.set_title(settings_dict['p_dict_verbose'][eg] + '  ' +
                             settings_dict['job_strs_dict'][jnum],
                             fontsize=title_size)
                plot_idx += 1

        # plot all employee groups on same job level chart
        else:

            with sns.axes_style(chart_style):
                ax = plt.subplot(num_jobs, num_egplots, plot_idx)
            ax.tick_params(axis='both', which='both', labelsize=10)
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
                count_plot(eg_jobs, jnum, dum, eg_colors[eg - 1],
                           ax, base_lw, 1, ls=base_ls)

                eg_jobs = prop_jobs[prop_jobs.eg == eg]
                count_plot(eg_jobs, jnum, dum, eg_colors[eg - 1],
                           ax, prop_lw, 1, ls=prop_ls)

            ax.set_title(settings_dict['job_strs_dict'][jnum],
                         fontsize=title_size)
            plot_idx += 1

    fig = plt.gcf()
    for ax in fig.axes:
        try:
            ax.legend_.remove()
        except AttributeError:
            pass

    fig.set_size_inches(xsize * num_egplots, ysize * num_jobs)
    fig.suptitle(suptitle, fontsize=suptitle_size, y=1.005)
    fig.tight_layout()

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def build_subplotting_order(rows, cols):
    '''build a list of integers to permit passing through subplots by columns
    note:  only used when looping completes one vertical column before
    continuing to next column

    inputs
        rows, cols (integer)
            number of rows and columns in multiple chart output
    '''
    subplot_order_list = []
    for col in np.arange(1, cols + 1):
        subplot_order_list.extend(np.arange(col, (rows * cols) + 1, cols))

    return subplot_order_list


def emp_quick_glance(empkey, df,
                     ds_dict=None,
                     title_size=14,
                     tick_size=13,
                     lw=4,
                     chart_style='dark',
                     xsize=8, ysize=48,
                     image_dir=None,
                     image_format='png'):
    '''view basic stats for selected employee and proposal

    A separate chart is produced for each measure.

    inputs
        empkey (integer)
            employee number (in data model)
        df (dataframe)
            dataset to study, will accept string proposal name
        ds_dict (dictionary)
            variable assigned to load_datasets function output
        title_size (integer or float)
            text size of chart title
        tick_size (integer or font)
            text size of chart tick labels
        lw (integer or float)
            line width of plot lines
        chart_style (string)
            any valid seaborn charting style
        xsize, ysize (integer or float)
            size of chart display
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''

    ds, df_label = determine_dataset(df, ds_dict, return_label=True)

    one_emp = ds[ds.empkey == empkey].set_index('date')

    cols = ['age', 'ylong', 'spcnt', 'lspcnt', 'snum', 'jnum', 'jobp',
            'cat_order', 'rank_in_job', 'job_count', 'mpay', 'cpay']

    with sns.axes_style(chart_style):
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
                             fontsize=title_size)
            else:
                ax.set_title(df_label + ', emp ' + str(empkey),
                             fontsize=title_size)
        if i == 0:
            ax.xaxis.set_tick_params(labeltop='on')
        ax.grid(c='grey', alpha=.3)
        i += 1

    plt.tick_params(axis='y', labelsize=tick_size)
    plt.tick_params(axis='x', labelsize=tick_size)

    one_emp = ()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.075, wspace=0)

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def cond_test(df, grp_sel,
              enhanced_jobs,
              job_colors,
              job_dict,
              basic_jobs=None,
              ds_dict=None,
              plot_all_jobs=False,
              min_mnum=None,
              max_mnum=None,
              limit_to_jobs=None,
              use_and=False,
              print_count_months=None,
              print_all_counts=False,
              plot_job_bands_chart=True,
              only_target_bands=False,
              legend_size=14,
              title_size=16,
              xsize=8, ysize=8,
              image_dir=None,
              image_format='png'):
    '''visualize selected job counts over time applicable to computed
    condition with optional printing of certain data.

    Primary usage is validation of job assignment conditions by charting the
    count(s) of job(s) assigned by the program to particular employee groups
    over time.

    The function may also be used to evaluate distribution of jobs with
    various proposals.  Career progression of employees who enjoy special
    job rights may be understood particularily well by utilizing the
    print_all_counts option.

    The output is 2 charts.  The first chart is a line chart displaying
    selected job count information over time. The second is a stacked area
    chart displaying all job counts for the selected group(s) over time.

    There are additional optional print outputs.  The print_all_counts option
    will print a dataframe containing job count totals for each month.  The
    print_count_months input is a list of months to print the only the plotted
    job counts, primarily for testing purposes.

    inputs
        df (dataframe)
            dataset(dataframe) to examine
        grp_sel (list)
            integer input(s) representing the employee group code(s) to
            select for analysis.  This argument also will accept the
            string 'sg' to select a special job rights group(s).  Multiple
            inputs are normally handled as 'or' filters, meaning an input of
            [1, 'sg'] would mean employee group 1 **or** any special job rights
            group, but can be modified to mean only group 1 **and** special job
            rights employees with the 'use_and' input.
        enhanced_jobs (boolean)
            if True, basic_jobs input job levels will be converted to
            enhanced job levels with reference to the job_dictionary input,
            otherwise basic_jobs input job levels will be used
        job_colors (list)
            list of color values to use for job plots
        job_dict (dictionary)
            dictionary containing basic to enhanced job level conversion data.
            This is likely the settings dictionary "jd" value.
        basic_jobs (list)
            basic job levels to plot.  This list will be converted to the
            corresponding enhanced job list if the enhanced_jobs input is
            set to True.  Defaults to [1] if not assigned.
        ds_dict (dictionary)
            dataset dictionary which allows df input to be a string
            description (proposal name)
        plot_all_jobs (boolean)
            option to plot all of the job counts within the input dataset vs
            only those selected with the basic_jobs input (or as converted to
            enhanced jobs if enhanced_jobs input is True).  The jobs plotted
            may be filtered by the limit_to_jobs input.
        min_mnum (integer)
            integer input, only plot data including this month forward(mnum).
            Defaults to zero.
        max_mnum (integer)
            integer input, only plot data through selected month (mnum).
            Defaults to maximum mnum for input data
        limit_to_jobs (list)
            a list of jobs to plot, allowing focus on target jobs.  Should be
            a subset of normal output, otherwise no filtering of normal output
            occurs
        use_and (boolean)
            when the grp_sel input has more than one element, require filtered
            dataframe for analysis to be part of all grp_sel input sets.
        print_count_months (list)
            list of month(s) for printing job counts
        print_all_counts (boolean)
            if True, print the entire job count dataframe.
        plot_job_bands_chart (boolean)
            if True, plot an area chart beneath the job count chart.  The area
            chart will display all of the jobs available to the selected
            employee group(s) over time with job band areas
        only_target_bands (boolean)
            if True, plot area chart of jobs from job count chart only,
            vs the default of all job levels
        legend_size (integer or float)
            text size of legend labels
        title_size (integer or float)
            text size of chart title
        xsize, ysize (integer or float)
            size of chart display in inches (width and height)
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''

    d, df_label = determine_dataset(df, ds_dict, return_label=True)

    # construct a string which will be evaluated below with the 'eval'
    # statement.  This string is a slicing filter for the input dataframe, and
    # is dependent upon the 'grp_sel' and 'use_and' inputs.
    pre = 'd['
    eg_pre = '(d.eg == '
    eg_suf = ')'
    suf = '].copy()'

    if use_and:
        oper = ' & '
    else:
        oper = ' | '

    if not basic_jobs:
        basic_jobs = [1]

    if enhanced_jobs:
        job_list = []
        for job in basic_jobs:
            job_list.extend([int(job_dict[job][0]), int(job_dict[job][1])])
    else:
        job_list = basic_jobs

    if limit_to_jobs:
        plot_jobs = []
        for job in limit_to_jobs:
            if job in job_list:
                plot_jobs.append(job)
        if plot_jobs:
            job_list = plot_jobs

    job_list.sort()

    i = 1
    if len(grp_sel) > 1:
        s = ''
        while i < len(grp_sel):
            if type(grp_sel[i - 1]) == int:
                s = s + (eg_pre + str(grp_sel[i - 1]) + eg_suf + oper)
                i += 1
            elif grp_sel[i - 1] == 'sg':
                s = s + '(d.sg == 1)' + oper
                i += 1
        if type(grp_sel[i - 1]) == int:
            s = s + (eg_pre + str(grp_sel[i - 1]) + eg_suf)
        elif grp_sel[i - 1] == 'sg':
            s = '(d.sg == 1)'
    else:
        if type(grp_sel[i - 1]) == int:
            s = eg_pre + str(grp_sel[i - 1]) + eg_suf
        elif grp_sel[i - 1] == 'sg':
            s = '(d.sg == 1)'

    segment = pre + s + suf
    # Example s variable: 'd[(d.eg == 2) | (d.eg == 3)].copy()'
    df = eval(segment)

    # This groupby and unstack operation produces a monthly count of all jobs
    all_jcnts = df.groupby(['date', 'jnum']).size() \
        .unstack().fillna(0).astype(int)
    all_jcnts['mnum'] = range(len(all_jcnts))

    if not min_mnum:
        min_mnum = 0
    if not max_mnum:
        max_mnum = all_jcnts.mnum.max()

    if enhanced_jobs:
        info_prefix = 'enhanced'
    else:
        info_prefix = 'basic'

    info_next_line = ('\nmodify "limit_to_jobs" input (as list) ' +
                      'to select target job plots')

    print('"enhanced_jobs" option is >>', bool(enhanced_jobs))
    if not plot_all_jobs:
        print(info_prefix + ' jobs for analysis >>', job_list)
    print(info_next_line)

    title = s

    # option to print a numerical tally of jobs for targeted months
    if print_count_months:
        for month in print_count_months:
            mdate = df[df.mnum == month]['date'].iloc[0].strftime('%m-%d-%Y')
            print('\nmonth ' + str(month), '(' + mdate + ') ' +
                  title + ' job count:')
            jnum_seg = np.array(df[df.mnum == month]['jnum'])
            for job in job_list:
                job_count = jnum_seg[jnum_seg == job].size
                print(int(job), int(job_count))
    try:
        j_colors = np.array(job_colors)[np.array(job_list) - 1]
        if len(job_list) == 1:
            j_colors = j_colors[0]
    except LookupError:
        print('\njob number error - (no matching job color), exiting...\n')
        return

    try:
        if not plot_all_jobs:

            cnd_jcnts = all_jcnts.copy()

            jdf = cnd_jcnts[(cnd_jcnts.mnum >= min_mnum) &
                            (cnd_jcnts.mnum <= max_mnum)]
            jdf_cols = jdf.columns.values.tolist()
            not_found = [job for job in job_list if job not in jdf_cols]
            job_list = [job for job in job_list if job in jdf_cols]
            if not_found:
                print('these jobs are not found and are not plotted >>',
                      not_found)
            jdf[job_list].plot(color=j_colors)

        if plot_all_jobs:

            job_list = []
            for col in all_jcnts.columns:
                try:
                    col + 0
                    job_list.append(int(col))
                except TypeError:
                    pass
            print('\n"plot_all_jobs" option selected')
            print('all job numbers found >>', job_list)

            if limit_to_jobs:
                temp_jobs = []
                for job in limit_to_jobs:
                    if job in job_list:
                        temp_jobs.append(job)
                job_list = temp_jobs
                print('\n"limit_to_jobs" option selected, ' +
                      'output limited to jobs >>',
                      job_list)
                print(' (set "limit_to_jobs" to "None" to stop...)')

            jdf = all_jcnts[(all_jcnts.mnum >= min_mnum) &
                            (all_jcnts.mnum <= max_mnum)]
            jdf[job_list].plot(color=j_colors)

    except LookupError:
        print('\n...job number error - exiting...')
        print('> verify job(s) for analysis exist within selected sample? <\n')
        return

    fig = plt.gcf()
    ax = plt.gca()
    ax.set_ylim(ymin=0)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5),
              fontsize=legend_size)
    fig.set_size_inches(xsize, ysize)
    ax.set_title(title, fontsize=title_size)

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()

    if plot_job_bands_chart:
        out = []
        for col in all_jcnts.columns:
            try:
                col + 0
                out.append(int(col))
            except TypeError:
                pass

        out.sort()
        job_list.sort()

        if only_target_bands:
            df_jcnts = all_jcnts[job_list]
            job_colors = j_colors
        else:
            df_jcnts = all_jcnts[out]

        df_jcnts[(all_jcnts.mnum >= min_mnum) &
                 (all_jcnts.mnum <= max_mnum)].plot(kind='area',
                                                    color=job_colors,
                                                    stacked=True,
                                                    linewidth=0.1,
                                                    alpha=.6)
        fig = plt.gcf()
        ax = plt.gca()
        ax.invert_yaxis()
        fig.set_size_inches(xsize, ysize)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='center left',
                  bbox_to_anchor=(1.01, 0.5),
                  fontsize=legend_size)

        ax.set_title(title, fontsize=title_size)
        ax.grid(linestyle='dotted', lw=1.5)

        if image_dir:
            func_name = sys._getframe().f_code.co_name
            if not path.exists(image_dir):
                makedirs(image_dir)
            plt.savefig(image_dir + '/' + func_name + ' - ' + 'job_bands' +
                        '.' + image_format,
                        bbox_inches='tight', pad_inches=.25)
        plt.show()

    # option to print a dataframe containing all job counts for all months:
    if print_all_counts:
        all_jcnts_print = df.groupby(['mnum', 'date', 'jnum']).size() \
            .unstack().fillna(0).astype(int)
        # calculate a total column (total count of all jobs for each month)
        np_total = np.add.accumulate(all_jcnts_print.values, 1).T[-1]
        all_jcnts_print['total'] = np_total
        print('\n', all_jcnts_print)


def single_emp_compare(emp, measure,
                       df_list, xax,
                       job_strs,
                       eg_colors,
                       p_dict,
                       job_levels,
                       attr_dict,
                       ds_dict=None,
                       chart_style='whitegrid',
                       standalone_color='#ff00ff',
                       title_size=14,
                       tick_size=12,
                       label_size=13,
                       legend_size=14,
                       xsize=12, ysize=8,
                       image_dir=None,
                       image_format='png'):

    '''Select a single employee and compare proposal outcome using various
    calculated measures.

    inputs
        emp (integer)
            empkey for selected employee
        measure (string)
            calculated measure to compare
            examples: 'jobp' or 'cpay'
        df_list (list)
            list of calculated datasets to compare
        xax (string)
            dataset column to set as x axis
        job_strs (list)
            string job description list
        eg_colors (list)
            list of colors to be assigned to line plots
        p_dict (dictionary)
            dictionary containing eg group integer to eg string descriptions
        job_levels (integer)
            number of jobs in the model
        attr_dict (dictionary)
            dataset column name description dictionary
        ds_dict (dictionary)
            output from load_datasets function
        chart_style (string)
            any valid seaborn plotting style
        standalone_color (color value)
            color of standalone plot
            (This function assumes one proposal from each group, any additional
            proposal is assumed to be standalone)
        title_size (integer or float)
            text size of chart title
        tick_size (integer or float)
            text size of chart tick labels
        label_size (integer or float)
            text size of x and y axis chart labels
        legend_size (integer or float)
            text size of chart legend
        xsize, ysize (integer or float)
            width and height of output chart in inches
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
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

    with sns.axes_style(chart_style):
        fig, ax = plt.subplots(figsize=(xsize, ysize))

    for i in range(0, len(df_list)):

        df_list[i][df_list[i].empkey == emp].set_index(xax)[measure] \
            .plot(label=label_dict[i],
                  lw=3,
                  alpha=.6, ax=ax)

    ax.set_title('Employee  ' + str(emp) + '  -  ' + attr_dict[measure],
                 y=1.02, fontsize=title_size)

    if measure in ['snum', 'cat_order', 'spcnt', 'lspcnt',
                   'jnum', 'jobp', 'fbff']:
        ax.invert_yaxis()

    if measure in ['spcnt', 'lspcnt']:
        ax.yaxis.set_major_formatter(pct_format())
        ax.axhline(y=1, c='.8', alpha=.8, lw=3)

    if measure in ['jnum', 'nbnf', 'jobp', 'fbff']:

        ax.set_yticks(np.arange(0, job_levels + 2, 1))
        yticks = ax.get_yticks().tolist()

        for i in np.arange(1, len(yticks)):
            yticks[i] = job_strs[i - 1]
        ax.axhspan(job_levels + 1, job_levels + 2, facecolor='.8', alpha=0.9)
        ax.set_yticklabels(yticks, va='top')
        ax.axhline(y=job_levels + 1, c='.8', ls='-', alpha=.8, lw=3)
        ax.set_ylim(job_levels + 1.5, 0.5)

    ax.tick_params(axis='y', labelsize=tick_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.set_xlabel(attr_dict[xax], fontsize=label_size)
    ax.set_ylabel(attr_dict[measure], fontsize=label_size)
    ax.legend(loc='best', markerscale=1.5, fontsize=legend_size)

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def job_time_change(ds_list, ds_base,
                    eg_list,
                    job_colors,
                    job_strs_dict,
                    job_levels,
                    attr_dict,
                    xax, ds_dict=None,
                    attr1=None, oper1='>=', val1=0,
                    attr2=None, oper2='>=', val2=0,
                    attr3=None, oper3='>=', val3=0,
                    marker='o', edgecolor='k',
                    linewidth=.05, size=25,
                    alpha=.95,
                    bg_color='#ffffff',
                    xmax=1.02,
                    limit_yax=False,
                    ylimit=40,
                    zeroline_color='m',
                    zeroline_width=1.5,
                    pos_neg_face=True,
                    pos_neg_face_alpha=.03,
                    legend_job_strings=True,
                    legend_position=1.18,
                    legend_marker_size=130,
                    suptitle_size=16,
                    title_size=14,
                    tick_size=12,
                    chart_style='whitegrid',
                    label_size=13,
                    xsize=12, ysize=10,
                    image_dir=None,
                    image_format='png',
                    experimental=False):
    '''Plots a scatter plot displaying monthly time in job
    differential, by proposal and employee group.  X axis percentage
    reflects first month within each comparative dataset, which will be the
    same as standalone for all groups unless the data model implementation
    date occurs at month zero.

    inputs
        ds_list (list)
            list of datasets to compare, may be ds_dict (output of
            load_datasets function) string keys or dataframe variable(s)
            or mixture of each
        ds_base (string or variable)
            baseline dataset, accepts same input types as ds_list above
        eg_list (list)
            list of integers for employee groups to be included in analysis
            example: [1, 2, 3]
        job_levels (integer)
            number of job levels in the data model
        job_colors (list)
            list of color values for job level plotting
        job_strs_dict (dictionary)
            dictionary of job code (integer) to job description label
        attr_dict (dictionary)
            dataset column name description dictionary
        xax (string)
            list percentage attrubute, i.e. spcnt or lspcnt
        ds_dict (dictionary)
            output from load_datasets function
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (integer, float, date as string, string (as appropriate))
            attr(n) limiting value (combined with oper(n)) as string
        job_colors (list)
            list of color values for the job level plotting
        job_strs_dict (dictionary)
            job number to job label dictionary
        marker (string)
            scatter chart matplotlib marker type
        edgecolor (color value)
            matplotlib marker edge color
        linewidth (integer or float)
            matplotlib marker edge line size
        size (integer or float)
            size of markers
        alpha (float)
            marker alpha (transparency) value
        bg_color (color value)
            background color of chart if not None
        xmax (integer or float)
            high limit of chart x axis
        limit_yax (integer or float)
            if True, restrict plot y scale to this value
            may be used to prevent outliers from exagerating chart scaling
        ylimit (integer or float)
            y axis limit if limit_yax is True
        zeroline_color (color value)
            color for zeroline on chart
        zeroline_width (integer or float)
            width of zeroline
        pos_neg_face (boolean)
            if True, apply a light green tint to the chart area above the
            zero line, and a light red tint below the line
        legend_job_strings (boolean)
            if True, use job description strings in legend vs. job numbers
        legend_position (float)
            controls the horizontal position of the legend
        legend_marker_size (integer or float)
            adjusts the size of the legend markers
        suptitle_size (integer or float)
            text size of chart super title
        title_size (integer or float)
            text size of chart title
        tick_size (integer or float)
            text size of chart tick labels
        xsize, ysize (integer or float)
            x and y size of each plot in inches
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
        experimental (boolean)
            show additional output under development consisting of a
            table, heatmap, and bar chart
    '''
    label_dict = {}
    i = 0
    for df in ds_list:
        ds, ds_label = determine_dataset(df, ds_dict,
                                         return_label=True)
        label_dict[i + 1] = ds_label

        ds_list[i] = filter_ds(ds,
                               attr1=attr1, oper1=oper1, val1=val1,
                               attr2=attr2, oper2=oper2, val2=val2,
                               attr3=attr3, oper3=oper3, val3=val3,
                               return_title_string=False)
        i += 1

    ds_base, ds_base_label = determine_dataset(ds_base, ds_dict,
                                               return_label=True)

    dsb_filt, tb_string = filter_ds(ds_base,
                                    attr1=attr1, oper1=oper1, val1=val1,
                                    attr2=attr2, oper2=oper2, val2=val2,
                                    attr3=attr3, oper3=oper3, val3=val3)

    ds_frames = od()
    ds_dict = od()
    # sorts index by empkey, this is base df with key 0
    ds_dict[0] = dsb_filt.groupby([pd.Grouper('empkey'), 'jnum']).size()\
        .unstack().fillna(0)

    i = 1
    for ds in ds_list:
        ds_frames[i] = ds[ds.mnum == 0][[xax, 'eg']]
        # sorts index by empkey
        ds_dict[i] = ds.groupby([pd.Grouper('empkey'),
                                'jnum']).size().unstack().fillna(0)
        i += 1

    # get keys for comparative dataframes (all but key 0)
    compare_keys = [x for x in list(ds_dict.keys()) if x > 0]
    diff_dict = od()
    joined_dict = od()

    for i in compare_keys:
        # create diff dataframes - all comparative dataframes minus base df
        diff_dict[i] = ds_dict[i] - ds_dict[0]
        # create empty comparative dataframe (index only)
        empty = ds_frames[i][[]]
        # auto-align (order) diff_dict[i] with index-only dataframes
        # using join method
        joined_dict[i] = empty.join(diff_dict[i])
        # sort columns
        joined_dict[i].sort_index(axis=1, inplace=True, ascending=False)
        # add xax and eg columns
        joined_dict[i] = joined_dict[i].join(ds_frames[i])
        # do not include results for employees with no change (0) in chart
        joined_dict[i] = joined_dict[i].replace(0, np.nan)

    joined_keys = list(joined_dict.keys())

    # make a reversed list of the data model job levels (high to low)
    job_list = np.arange(job_levels, 0, -1)

    for jk in joined_keys:
        for eg in eg_list:
            # filter for eg
            eg_df = joined_dict[jk][joined_dict[jk].eg == eg]

            if experimental:
                db = eg_df.copy()
                db.replace(np.nan, 0, inplace=True)
                db['quantile'] = \
                    np.clip(db[xax] * 100 // 10 + 1, 1, 10).astype(int)
                db.drop([xax, 'eg'], axis=1, inplace=True)
                # db = db.groupby('quantile').sum().divide(len(db) / 10)
                db = db.groupby('quantile').sum().astype(int)
                db = db[db.columns[::-1]]
                db.columns.name = 'job_level'
                print('< proposal',
                      label_dict[jk] + ', eg ' + str(eg),
                      '>', '\n', '--' * 10, '\n')
                print('Job time change table (months)', '\n')
                print(db.T.reindex(columns=list(range(10, 0, -1))))
                sns.heatmap(db.T, cmap='seismic_r', center=0,
                            annot=True, fmt='d')
                plt.gca().invert_xaxis()
                plt.gca().set_title('months-in-job change, by quantile')
                db.plot(kind='bar', stacked=True, width=1,
                        color=job_colors, linewidth=.5, edgecolor='k')
                ax = plt.gca()
                ax.invert_xaxis()
                ax.set_title('Months in job change by quantile')
                handles, labels = ax.get_legend_handles_labels()
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
                lgnd = ax.legend(handles, labels, loc='center left',
                                 bbox_to_anchor=(.99, .5), borderaxespad=4,
                                 title='job_level', fontsize=12)

            with sns.axes_style(chart_style):
                fig, ax = plt.subplots()
            fig.set_size_inches(xsize, ysize)

            for jnum in job_list:
                try:
                    eg_df.plot(kind='scatter',
                               x=xax,
                               y=jnum,
                               color=job_colors[jnum - 1],
                               edgecolor=edgecolor,
                               marker=marker,
                               linewidth=linewidth,
                               s=size,
                               alpha=alpha,
                               label=str(jnum),
                               ax=ax)
                except KeyError:
                    pass

            if xax in ['spcnt', 'lspcnt']:
                ax.set_xlim(xmin=0, xmax=1.02)
                ax.xaxis.set_major_formatter(pct_format())
                ax.set_xticks(np.arange(0, 1.05, .05))
            if xax in ['cat_order']:
                ax.set_xlim(xmin=0)

            ax.axhline(c=zeroline_color, lw=zeroline_width)

            ax.tick_params(labelsize=13, labelright=True)
            ax.set_ylabel('months differential', fontsize=label_size)

            ax.set_xlabel(attr_dict[xax], fontsize=label_size)
            ax.set_title('Months in job differential, ' +
                         label_dict[jk] + ', eg ' + str(eg),
                         fontsize=title_size)
            if limit_yax:
                ax.set_ylim(-ylimit, ylimit)
            if pos_neg_face:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
                ax.axhspan(0, ymax, facecolor='g',
                           alpha=pos_neg_face_alpha, zorder=8)
                ax.axhspan(0, ymin, facecolor='r',
                           alpha=pos_neg_face_alpha, zorder=8)
            if bg_color:
                ax.set_facecolor(bg_color)
            ax.tick_params(axis='y', labelsize=tick_size)
            ax.tick_params(axis='x', labelsize=tick_size)
            ax.grid(linestyle='dotted', lw=1.5)
            ax.invert_xaxis()
            plt.tight_layout()

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
                    label_strs.append(job_strs_dict[int(label)])
                    labels = label_strs
            # move legend off of chart face to right
            legend_title = 'job'
            # use sorted labels and handlers as legend input
            # and position legend

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
            lgnd = ax.legend(handles, labels, loc='center left',
                             bbox_to_anchor=(.99, .5), borderaxespad=4,
                             title=legend_title, fontsize=12)

            lgnd.get_title().set_fontsize('16')

            # set legend marker size
            for mark in lgnd.legendHandles:
                mark._sizes = [legend_marker_size]

            if image_dir:
                func_name = sys._getframe().f_code.co_name
                if not path.exists(image_dir):
                    makedirs(image_dir)
                plt.savefig(image_dir + '/' + func_name + ' - ' + str(jk) +
                            'grp' + str(eg) +
                            '.' + image_format,
                            bbox_inches='tight', pad_inches=.25)
            plt.show()


# EMPLOYEE_GROUP_ATTRIBUTE_AVERAGE_AND_MEDIAN
def group_average_and_median(dfc, dfb,
                             eg_list,
                             eg_colors,
                             measure,
                             job_levels,
                             settings_dict,
                             attr_dict,
                             ds_dict=None,
                             attr1=None, oper1='>=', val1='0',
                             attr2=None, oper2='>=', val2='0',
                             attr3=None, oper3='>=', val3='0',
                             plot_median=False,
                             plot_average=True,
                             compare_to_dfb=True,
                             use_filtered_results=True,
                             show_full_yscale=False,
                             job_labels=True,
                             max_date=None,
                             chart_style='whitegrid',
                             xsize=14, ysize=8,
                             image_dir=None,
                             image_format='png'):
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
        dfc (string or dataframe variable)
            comparative dataset to examine, may be a dataframe variable
            or a string key from the ds_dict dictionary object
        dfb (string or dataframe variable)
            baseline dataset to plot (likely use standalone
            dataset here for comparison, but may plot and compare any dataset),
            may be a dataframe variable or a string key from the ds_dict
            dictionary object
        eg_list (list)
            list of integers representing the employee groups to analyze
            (i.e. [1, 2])
        eg_colors (list)
            list of colors for plotting the employee groups
        measure (string)
            attribute (column) to compare, such as 'spcnt' or 'jobp'
        job_levels (integer)
            number of job levels in the data model
        settings_dict (dictionary)
            program settings dictionary generated by the build_program_files
            script
        attr_dict (dictionary)
            dataset column name description dictionary
        ds_dict (dictionary)
            dataset dictionary (variable assigned to the output of
            load_datasets function)
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (integer, float, date as string, string (as appropriate))
            attr(n) limiting value (combined with oper(n)) as string
        plot_meadian (boolean)
            plot the median of the measure for each employee group
        plot_average (boolean)
            plot the average(mean) of the measure for each employee group
        compare_to_dfb (boolean)
            plot average dfb[measure] data as dashed line.
            (likely show standalone data with dfb, or reverse and show
            standalone as primary and integrated as dfb)
            (dfb refers to baseline dataframe or dataset)
        use_filtered_results (boolean)
            if True, use the same employees from the filtered proposal list.
            For example, if the dfc list is filtered by age only, the
            dfb list could be filtered by the same age and return the
            same employees.  However, if the dfc list is filtered by
            an attribute which diverges from the dfb measurements for
            the same attribute, a different set of employees could be returned.
            This option ensures that the same group of employees from both the
            dfc (filtered first) list and the dfb list are compared.
            (dfc refers to the comparison proposal, dfb refers to baseline)
        show_full_yscale (boolean)
            if measure input is one of these: 'jnum', 'nbnf', 'jobp', 'fbff',
            if True, show all job levels on chart.  Otherwise, allow chart to
            autoscale with plotted data
        job_labels (boolean)
            if measure input is one of these: 'jnum', 'nbnf', 'jobp', 'fbff',
            use job text description labels vs. number labels on the y axis
            of the chart (boolean)
        max_date (date string)
            maximum chart date.  If set to 'None', the maximum chart date will
            be the maximum date within the list data.
        chart_style (string)
            option to specify alternate seaborn chart style
        xsize, ysize (integer or float)
            x and y size of chart in inches
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''
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

    p_dict = settings_dict['p_dict']

    if measure == 'mpay':
        # eliminate variable employee last month partial pay
        dfc = dfc[dfc.ret_mark == 0].copy()
        dfb = dfb[dfb.ret_mark == 0].copy()

    with sns.axes_style(chart_style):
        fig, ax = plt.subplots()

    for eg in eg_list:
        # for each employee group, group by date and plot avg/median of measure
        try:
            date_group = dfc[dfc.eg == eg].groupby('date')
            if plot_average:
                date_group[measure].mean().plot(color=eg_colors[eg - 1],
                                                ax=ax, lw=3,
                                                label=dfc_label +
                                                ', grp' +
                                                p_dict[eg] + ' avg')
            if plot_median:
                date_group[measure].median().plot(color=eg_colors[eg - 1],
                                                  ax=ax, lw=1,
                                                  label=dfc_label +
                                                  ', grp' +
                                                  p_dict[eg] + ' median')
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
                date_group = dfb[dfb.eg == eg].groupby('date')
                if plot_average:
                    date_group[measure].mean().plot(label=dfb_label +
                                                    ', ' +
                                                    'grp' + p_dict[eg] +
                                                    ' avg',
                                                    color=eg_colors[eg - 1],
                                                    ls='dashed',
                                                    lw=2.5,
                                                    alpha=.5,
                                                    ax=ax)
                if plot_median:
                    date_group[measure].median().plot(label=dfb_label +
                                                      ', ' +
                                                      'grp' + p_dict[eg] +
                                                      ' median',
                                                      color=eg_colors[eg - 1],
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
            if show_full_yscale:
                ax.set_yticks(np.arange(0, job_levels + 2, 1))
            yticks = ax.get_yticks().tolist()
            job_strs_dict = settings_dict['job_strs_dict']
            for i in np.arange(1, len(yticks)):
                yticks[i] = job_strs_dict[i]
            ax.set_yticklabels(yticks, va='top')
            ax.set_ylim(ymax=0.75)

        else:
            if show_full_yscale:
                ax.set_yticks(np.arange(0, job_levels + 2, 1))
            ax.set_ylim(ymax=0.75)

        if show_full_yscale:
            ax.set_ylim(ymin=job_levels + 2)
            ax.axhspan(job_levels + 1, job_levels + 2,
                       facecolor='.8', alpha=0.5)
            ax.axhline(y=job_levels + 1, c='.8', ls='-', alpha=.8, lw=3)

    if settings_dict['delayed_implementation']:
        if settings_dict['implementation_date']:
            # plot vertical line at implementation date
            ax.axvline(settings_dict['implementation_date'], c='#33cc00',
                       ls='dashed', alpha=1, lw=1,
                       label='implementation date', zorder=1)

    if compare_to_dfb:
        suptitle_string = (dfc_label +
                           plot_string.upper() + attr_dict[measure].upper() +
                           ' vs. ' + dfb_label)
    else:
        suptitle_string = (dfc_label +
                           plot_string.upper() + attr_dict[measure].upper())

    fig.suptitle(suptitle_string, fontsize=16)
    ax.set_title(t_string + ' ' + title_string, y=1.02, fontsize=14)

    # LEGEND --------------
    # move legend off of chart face to right
    handles, labels = ax.get_legend_handles_labels()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles, labels, loc='center left',
              bbox_to_anchor=(1.01, .5),
              fontsize=14)
    # --------------------

    fig.set_size_inches(xsize, ysize)

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


# EMPLOYEE DENSITY STRIPPLOT (with filtering)
def stripplot_eg_density(df, mnum,
                         eg_colors, attr_dict,
                         ds_dict=None,
                         attr1=None, oper1='>=', val1=0,
                         attr2=None, oper2='>=', val2=0,
                         attr3=None, oper3='>=', val3=0,
                         chart_style='whitegrid',
                         bg_color='white',
                         title_size=12,
                         suptitle_size=14,
                         xsize=5, ysize=10,
                         image_dir=None,
                         image_format='png'):
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
        mnum (integer)
            month number to study from dataset
        eg_colors (list)
            color codes for plotting each employee group
        attr_dict (dictionary)
            dataset column name description dictionary
        ds_dict (dictionary)
            output from load_datasets function
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (integer, float, date as string, string (as appropriate))
            attr(n) limiting value (combined with oper(n)) as string
        bg_color (color value)
            chart background color
        title_size (integer or float)
            chart title text size
        suptitle_size (integer or float)
            chart text size of suptitle
        xsize, ysize (integer or float)
            size of chart width and height in inches
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
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
        with sns.axes_style(chart_style):
            fig, ax = plt.subplots(figsize=(xsize, ysize))

        sns.stripplot(y='new_order', x='eg', data=mnum_p, jitter=.5,
                      order=np.arange(min_eg, max_eg + 1),
                      palette=eg_colors, size=3, linewidth=0,
                      split=True, ax=ax)
        ax.set_facecolor(bg_color)

    except:
        print('\nEmpty dataset, nothing to plot. Check filters?\n')
        return

    ax.set_ylim(max(mnum_p.new_order), 0)

    if t_string:
        fig.suptitle(df_label, fontsize=suptitle_size)
        ax.set_title(t_string, fontsize=title_size)
    else:
        ax.set_title(df_label, fontsize=suptitle_size)

    ax.set_ylabel(attr_dict['eg'])

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def job_count_bands(df_list,
                    eg_list,
                    job_colors,
                    settings_dict,
                    ds_dict=None,
                    emp_list=None,
                    attr1=None, oper1='>=', val1=0,
                    attr2=None, oper2='>=', val2=0,
                    attr3=None, oper3='>=', val3=0,
                    fur_color=None,
                    show_grid=True,
                    max_date=None,
                    plot_alpha=.75,
                    legend_alpha=.9,
                    legend_xadj=1.3,
                    legend_yadj=1.0,
                    legend_size=11,
                    title_size=14,
                    tick_size=12,
                    label_size=13,
                    chart_style='darkgrid',
                    xsize=13, ysize=8,
                    image_dir=None,
                    image_format='png'):
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
        settings_dict (dictionary)
            program settings dictionary generated by the build_program_files
            script
        ds_dict (dictionary)
            output from load_datasets function
        emp_list (list)
            optional list of employee number(s) to plot (empkey attribute)
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (integer, float, date as string, string (as appropriate))
            attr(n) limiting value (combined with oper(n)) as string
        fur_color (color code in rgba, hex, or string style)
            custom color to signify furloughed job level band (otherwise, last
            color in job_colors input will be used)
        max_date (date string)
            only include data up to this date
            example input: '1997-12-31'
        plot_alpha (float, 0.0 to 1.0)
            alpha value (opacity) for area plot (job level bands)
        legend_alpha (float, 0.0 to 1.0)
            alpha value (opacity) for legend markers
        legend_xadj, legend_yadj (floats)
            adjustment input for legend horizontal and vertical placement
        legend_size (integer or float)
            text size of legend labels
        title_size (integer or float)
            text size of chart title
        tick_size (integer or float)
            text size of x and y tick labels
        label_size (integer or float)
            text size of x and y descriptive labels
        chart_style (string)
            chart styling (string), any valid seaborn chart style
        xsize, ysize (integer or float)
            plot size in inches (width and height)
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
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

    if fur_color:
        job_colors[-1] = fur_color

    for eg in eg_list:
        i = 0

        for df_object in df_list:

            with sns.axes_style(chart_style):
                fig, ax = plt.subplots(figsize=(xsize, ysize))

            if show_grid:
                ax.grid(which='major', color='k',
                        alpha=.1, linestyle='solid')
                ax.grid(which='minor', color='k',
                        alpha=.1, linestyle='dotted')
                ax.minorticks_on()
            else:
                ax.grid(False)

            df = df_object[df_object.eg == eg].copy()
            y = len(df[df.mnum == 0]) + 50

            dfg = df.groupby(['date', 'jnum']).size()
            dfg = pd.DataFrame(dfg.unstack().fillna(0))
            cols = dfg.columns.values.tolist()

            plot_colors = [job_colors[j - 1] for j in cols]

            dfg.plot(kind='area', stacked=True,
                     color=plot_colors, linewidth=0,
                     alpha=plot_alpha, ax=ax)

            if emp_list:
                empkey_set = set(df.empkey)

                if any(x in empkey_set for x in emp_list):
                    df['eg_order'] = df.groupby('mnum').cumcount() + 1

                for emp in emp_list:
                    if emp in empkey_set:
                        df[df.empkey == emp].plot('date', 'eg_order',
                                                  ls='dashed',
                                                  lw=2,
                                                  ax=ax)

            ax.set_ylim(y, 0)

            if label_dict[i] in ['standalone', 'award']:
                title_str = label_dict[i] + ', group ' + str(eg)
            else:
                title_str = label_dict[i] + ' proposal, group ' + str(eg)

            ax.set_title(title_str, fontsize=title_size, y=1.01)

            # legend-----------
            recs = []
            job_labels = []

            for k in cols:
                recs.append(mpatches.Rectangle((0, 0), 1, 1,
                                               fc=job_colors[k - 1],
                                               alpha=legend_alpha))
                job_labels.append(settings_dict['job_strs'][k - 1])

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(recs, job_labels, loc='center left',
                      bbox_to_anchor=(1.01, 0.5),
                      fontsize=legend_size, title='job')
            # -----------------

            ax.xaxis.label.set_size(label_size)
            ax.yaxis.label.set_size(label_size)
            ax.tick_params(axis='y', labelsize=tick_size)
            ax.tick_params(axis='x', labelsize=tick_size)
            ax.set_ylabel('job count', fontsize=10)

            if image_dir:
                func_name = sys._getframe().f_code.co_name
                if not path.exists(image_dir):
                    makedirs(image_dir)
                plt.savefig(image_dir + '/' + func_name + ' - ' +
                            label_dict[i] + ' grp' + str(eg) +
                            '.' + image_format,
                            bbox_inches='tight', pad_inches=.25)
            plt.show()
            i += 1


def determine_dataset(ds_def,
                      ds_dict=None,
                      return_label=False):
    '''this function permits either a dictionary key (string) or a dataframe
    variable to be used in functions as a dataframe object.

    inputs
        ds_def (dataframe or string)
            A pandas dataframe or a string representing a key for a dictionary
            which contains dataframe(s) as values
        ds_dict (dictionary)
            A dictionary containing string to dataframes, used if ds_def input
            is not a dataframe
        return_label (boolean)
            If True, return a descriptive dataframe label if the
            ds_dict was referenced, otherwise return a generic "Proposal"
            string
    '''
    if type(ds_def) == pd.core.frame.DataFrame:
        ds = ds_def
        ds_label = 'Proposal'
    else:
        if ds_dict:
            try:
                ds = ds_dict[ds_def]
                ds_label = ds_def
            except (NameError, LookupError):
                print('error:\n invalid dataframe or ds_dict key ' +
                      '(first argument)')
                return
        else:
            print('error:\n it appears that you may be using a dictionary' +
                  ' key as input, but "ds_dict" is undefined,' +
                  ' please set "ds_dict" keyword variable')
            return

    if return_label:
        return ds, ds_label
    else:
        return ds


def numeric_test(value):
    '''determine if a variable is numeric

    returns a boolean value

    input
        value
            any variable
    '''
    try:
        float(value)
        return True
    except ValueError:
        return False


def filter_ds(ds,
              attr1=None, oper1=None, val1=None,
              attr2=None, oper2=None, val2=None,
              attr3=None, oper3=None, val3=None,
              return_title_string=True):

    '''Filter a dataset (dataframe) by attribute(s).

    Filter process is ignored if attr(n) input is None.
    All attr, oper, and val inputs must be strings.
    Up to 3 attribute filters may be combined.

    Attr, oper, and val inputs are combined and then evaluated as expressions.

    If return_title_string is set to True, returns tuple (ds, title_string),
    otherwise returns ds.

    inputs
        ds (dataframe)
            the dataframe to filter
        attr(n) (string)
            an attribute (column) to filter.  Example: 'ldate'
        oper(n) (string)
            an operator to apply to the attr(n) input.  Example:  '<='
        val(n) (integer, float, date as string, string (as appropriate))
            attr(n) limiting value (combined with oper(n)) as string
        return_title_string (boolean)
            If True, returns a string which dexcribes the filter(s) applied to
            the dataframe (ds)
    '''

    # title_string = ''

    if any([attr1, attr2, attr3]):

        str1 = ''
        str2 = ''
        str3 = ''

        if attr1:

            if not numeric_test(val1):
                val1_text = "'" + val1 + "'"
            else:
                val1_text = str(val1)
            try:
                # slice proposal dataset according to attr1 inputs
                ds = ds[eval('ds[attr1]' + oper1 + val1_text)].copy()
                str1 = attr1 + ' ' + oper1 + ' ' + str(val1)

            except:
                print('''attr1 filter error - filter ignored
                      ensure filter inputs are strings''')
        if attr2:

            if not numeric_test(val2):
                val2_text = "'" + val2 + "'"
            else:
                val2_text = str(val2)
            try:
                ds = ds[eval('ds[attr2]' + oper2 + val2_text)].copy()
                str2 = attr2 + ' ' + oper2 + ' ' + str(val2)

            except:
                print('''attr2 filter error - filter ignored
                      ensure filter inputs are strings''')
        if attr3:

            if not numeric_test(val3):
                val3_text = "'" + val3 + "'"
            else:
                val3_text = str(val3)
            try:
                ds = ds[eval('ds[attr3]' + oper3 + val3_text)].copy()
                str3 = attr3 + ' ' + oper3 + ' ' + str(val3)

            except:
                print('''attr3 filter error - filter ignored
                      ensure filter inputs are strings''')

    else:
        if return_title_string:
            title_string = ''
            return ds, title_string
        else:
            return ds

    if return_title_string:
        title_string = ', '.join([x for x in [str1, str2, str3] if x])
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


def slice_ds_by_filtered_index(df, ds_dict=None,
                               mnum=0, attr='age',
                               attr_oper='>=',
                               attr_val=50):
    '''filter an entire dataframe by only selecting rows which match
    the filtered results from a target month.  In other words, zero in on
    a slice of data from a particular month, such as employees holding a
    specific job in month 25.  Then, using the index of those results,
    find only those employees within the entire dataset as an input for further
    analysis within the program.

    The output may be used as an input to a plotting function or for other
    analysis.  This function may also be used repeatedly with various filters,
    using output of one execution as input for another execution.

    inputs
        df (dataframe, can be proposal string name)
            the dataframe (dataset) to be filtered
        ds_dict (dictionary)
            A dictionary containing string to dataframes, used if ds_def input
            is not a dataframe
        mnum (integer)
            month number of the data to filter
        attr (string)
            attribute (column) to use during filter
        oper (string)
            operator to use, such as '<=' or '!='
        attr_val (integer, float, date as string, string (as appropriate))
            attr1 limiting value (combined with oper) as string

            Example filter:
                jnum >= 7 (in mnum month)
    '''
    ds = determine_dataset(df, ds_dict)

    mnum = str(mnum)
    try:
        attr_val = str(float(attr_val))
    except ValueError:
        pass

    # get the indexes (employee numbers) of the filtered data
    month_slice_indexes = \
        np.array(ds[eval('(ds[attr]' + attr_oper + attr_val +
                         ') & (ds.mnum == ' + mnum + ')')].index)

    ds_index = np.array(ds.index)
    # get all of the dataset rows with an index (employee number) which exists
    # within the month_slice_indexes array
    ds_filter = ds[np.in1d(ds_index, month_slice_indexes)]

    return ds_filter


def mark_quantiles(df, quantiles=10):
    '''add a column to the input dataframe identifying quantile membership as
    integers (the column is named "quantile").  The quantile membership
    (category) is calculated for each employee group separately, based on
    the employee population in month zero.

    The output dataframe permits attributes for employees within month zero
    quantile categories to be be analyzed throughout all the months of the
    data model.

    The number of quantiles to create within each employee group is selected
    by the "quantiles" input.

    The function utilizes numpy arrays and functions to compute the quantile
    assignments, and pandas index data alignment feature to assign month zero
    quantile membership to the long-form, multi-month output dataframe.

    This function is used within the quantile_groupby function.

    inputs
        df (dataframe)
            Any pandas dataframe containing an "eg" (employee group) column
        quantiles (integer)
            The number of quantiles to create.

            example:

            If the input is 10, the output dataframe will be a column of
            integers 1 - 10.  The count of each integer will be the same.
            The first quantile members will be marked with a 1, the second
            with 2, etc., through to the last quantile, 10.
    '''
    mult = 1000
    mod = mult / quantiles
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

    df['quantile'] = bins_arr
    aligned_df['quantile'] = df['quantile']
    return aligned_df


def quantile_groupby(df, eg_list,
                     measure, quantiles,
                     eg_colors,
                     band_colors,
                     settings_dict,
                     attr_dict,
                     groupby_method='median',
                     xax='date',
                     ds_dict=None,
                     through_date=None,
                     show_job_bands=True,
                     show_grid=True,
                     plot_implementation_date=True,
                     custom_color=False,
                     cm_name='Set1',
                     start=0.0,
                     stop=1.0,
                     exclude=None,
                     reverse=False,
                     chart_style='whitegrid',
                     remove_ax2_border=True,
                     line_width=1,
                     bg_color='.98',
                     job_bands_alpha=.15,
                     line_alpha=.7,
                     grid_alpha=.3,
                     title_size=14,
                     tick_size=12,
                     label_size=13,
                     label_pad=110,
                     xsize=12,
                     ysize=10,
                     image_dir=None,
                     image_format='png'):
    '''Plot representative values of a selected attribute measure for each
    employee group quantile over time.

    Multiple employee groups may be plotted at the same time.  Job bands may
    be plotted as a chart background to display job level progression when
    the measure input is set to "cat_order".

    Example use case: plot the average job category rank of each employee
    quantile group, from the start date though the life of the data model.

    The quantile group attribute may be analyzed with any of the following
    methods:

        [mean, median, first, last, min, max]

    If the eg_list input list contains a single employee group code and
    the custom_color input is set to "True", the color of the plotted
    quantile result lines will be a spectrum of colors. The following inputs
    are related to the custom color generation:

        [cm_name, start, stop, exclude, reverse]

    The above inputs will be used by the make_color_list function located
    within this module to produce a list of colors with a length equal to
    the quantiles input.  (Please see the docstring for the make_color_list
    function for further explaination).  If the quantiles input is set to a
    relatively high value (100-200), the impact on the career profiles of
    the employee groups is easily discernible when using a qualitative
    color map.

    inputs
        df (dataframe)
            Any long-form dataframe which contains "date" (and "mnum" if xax
            input is set to "mnum") and "eg" columns and at least one
            attribute column for analysis.  The normal input is a calculated
            dataset with many attribute columns.
        eg_list (list)
            List of eg (employee group) codes for analysis.  The order of the
            employee codes will determine the z-order of the plotted lines,
            last employee group plotted on top of the others.
        measure (string)
            Attribute column name
        quantiles (integer)
            The number of quantiles to create and plot for each employee
            group in the eg_list input.
        eg_colors (list)
            list of color values for plotting the employee groups
        band_colors (list)
            list of color values for plotting the background job level
            color bands when the using a measure of 'cat_order' with the
            'show_job_bands' variable set to True
        settings_dict (dictionary)
            program settings dictionary generated by the build_program_files
            script
        attr_dict (dictionary)
            dataset column name description dictionary
        groupby_method (string)
            The method applied to the attribute data within each quantile.  The
            allowable methods are listed in the description above.  Default is
            'median'.
        xax (string)
            The first groupby level and x axis value for the analysis.  This
            value defaults to "date" which represents each month of the model.
            Alternatively, "mnum" may be used.
        job_levels (integer)
            The number of job levels (excluding the furlough level) in the data
            model.
        ds_dict (dictionary)
            A dictionary containing string to dataframes, used if df input
            is not a dataframe but a string key (examples: 'standalone', 'p1')
        through_date (date string)
            If set as a date string, such as '2020-12-31', only show results
            up to and including this date.
        show_job_bands
            If measure is set to "cat_order", plot properly scaled job level
            color bands on chart background
        show_grid (boolean)
            if True, plot a grid on the chart
        plot_implementation_date
            If True and the xax argument is set to "date", plot a dashed
            vertical line at the implementation date.
        custom_color (boolean)
            If set to True, will permit a custom color spectrum to be produced
            for plotting a single employee group "cat_order" result (color map
            is selected with the cm_name input)
        cm_name (string)
            The colormap name to be used for the custom color option
        start (float)
            The starting point of the colormap to begin a custom color list
            generation (0.0 to less than 1.0)
        stop (float)
            The ending point of the colormap to finish a custom color list
            generation (greater than 0.0 to 1.0)
        exclude (list)
            A list of 2 floats between 0.0 and 1.0 describing a section of
            the original colormap to exclude from a custom color list
            generation.  (Example [.45, .55], the middle of the list excluded)
        reverse (boolean)
            If True, reverse the sequence of the custom color list
        chart_style (string)
            set the chart plot style for ax1 from the avialable seaborn
            plotting themes:

                ["darkgrid", "whitegrid", "dark", "white", and "ticks"]

            The default is "whitegrid".
        remove_ax2_border (boolean)
            if "cat_order" is set as the measure input and the show_job_bands
            input is set True, a second axis is generated to be the container
            for the job level labels.  The chart style for
            ax2 is "white" which avoids unwanted grid lines but includes a
            black solid chart border by default.  This ax2 border may be
            removed if this input is set to True.  (The border may be
            displayed if the chart_style input (for ax1) is set to "white"
            or "ticks").
        line_width (float)
            The width of the plotted lines.  Default is .75
        bg_color (color value)
            The background color for the chart.  May be a color name, color
            abreviation, hex value, or decimal between 0 and 1
            (shades of black)
        job_bands_alpha (float)
            If show_job_bands input is set to True and measure is set to
            "cat_order", this input controls the alpha or transparency of
            the background job level bands. (0.0 to 1.0)
        line_alpha (float)
            Transparency value of plotted lines (0.0 to 1.0)
        grid_alpha (float)
            Transparency value of grid lines (0.0 to 1.0)
        title_size (integer or float)
            Font size value for title
        tick_size (integer or float)
            Font size value for chart tick (value) labels
        label_size (integer or float)
            Font size value for x and y unit labels
        xsize, ysize (integers or floats)
            Width and height of chart in inches
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''
# **************
    df, df_label = determine_dataset(df, ds_dict, return_label=True)

    # limit the scope of the plot to a selected month in the future if the
    # through_date argument is assigned an integer
    if through_date:
        through_date = pd.to_datetime(through_date)
        df = df[df.date <= through_date]
    else:
        through_date = max(df.date)

    # make a dataframe with an added column ('quantile') indicating quantile
    # membership number (integer) for each employee, each employee group
    # calculated separately...
    bin_df = mark_quantiles(df, quantiles)

    # if mpay is selected, remove employee monthly pay data for retirement
    # months to exclude partial pay amounts
    if measure == 'mpay':
        bin_df = bin_df[bin_df.ret_mark == 0]

    if len(eg_list) > 1:
        multiplot = True
    else:
        multiplot = False
        clrs = make_color_list(num_of_colors=quantiles, start=start,
                               stop=stop, exclude=exclude, reverse=reverse,
                               cm_name_list=[cm_name],
                               return_list=True)

    if measure == 'cat_order' and (groupby_method not in
                                   ['mean', 'median', 'first',
                                    'last', 'min', 'max']):
        print('\nError:\n\n' +
              '  When measure is set to "cat_order", groupby_method input ' +
              'must be in ["mean", "median", "first", "last", "min", "max"]' +
              '\n\n' + '  Current groupby_method value is "' +
              groupby_method + '".\n\n')
        return

    with sns.axes_style(chart_style):
        fig, ax1 = plt.subplots(figsize=(xsize, ysize))

    job_levels = settings_dict['num_of_job_levels']
    job_strs = settings_dict['job_strs_dict']

# ************

    # create the job bands and labels on ax2
    if measure in ['cat_order'] and show_job_bands:

        bg_color = '#ffffff'
        job_strs_dict = settings_dict['job_strs_dict']
        starting_date = settings_dict['starting_date']

        tdict = pd.read_pickle('dill/dict_job_tables.pkl')
        table = tdict['table']

        df_table = pd.DataFrame(table[0], columns=np.arange(1, job_levels + 1),
                                index=pd.date_range(starting_date,
                                                    periods=table[0].shape[0],
                                                    freq='M'))
        # for band areas
        jobs_table = df_table[:through_date]

        jobs_table.plot.area(stacked=True,
                             figsize=(12, 10),
                             sort_columns=True,
                             linewidth=2,
                             color=band_colors,
                             alpha=job_bands_alpha,
                             legend=False,
                             ax=ax1)
        # for headcount:
        df_monthly_non_ret = \
            pd.DataFrame(df[df.fur == 0].groupby('mnum').size(),
                         columns=['count'])

        df_monthly_non_ret.set_index(
            pd.date_range(starting_date,
                          periods=pd.unique(df_monthly_non_ret.index).size,
                          freq='M'), inplace=True)

        non_ret_count = df_monthly_non_ret[:through_date]
        last_month_jobs_series = jobs_table.loc[through_date].sort_index()

        l_mth_counts = pd.DataFrame(last_month_jobs_series,
                                    index=last_month_jobs_series.index
                                    ).sort_index()

        l_mth_counts.rename(columns={l_mth_counts.columns[0]: 'counts'},
                            inplace=True)
        l_mth_counts['cum_counts'] = l_mth_counts['counts'].cumsum()

        cnts = list(l_mth_counts['cum_counts'])
        cnts.insert(0, 0)
        axis2_lbl_locs = []
        axis2_lbls = []

        ax2 = ax1.twinx()
        ax2.grid(False)
        if remove_ax2_border:
            for axis in ['top', 'bottom', 'left', 'right']:
                ax2.spines[axis].set_linewidth(0.0)
        ax1.invert_yaxis()

        i = 0
        for job_num in l_mth_counts.index:
            axis2_lbl_locs.append(round((cnts[i] + cnts[i + 1]) / 2))
            axis2_lbls.append(job_strs_dict[job_num])
            i += 1

        axis2_lbl_locs = add_pad(axis2_lbl_locs, pad=label_pad)
        ax2.set_yticks(axis2_lbl_locs)
        ax2.set_yticklabels(axis2_lbls)

        non_ret_count['count'].plot(c='grey', ls='--',
                                    label='active count', ax=ax1)

        ax2.set_ylim(ax1.get_ylim())

# .............................................................
    # this section is for the quantile line plots:
    y_limit = 0

    for eg in eg_list:
        frame = bin_df[bin_df.eg == eg]
        # group frame for eg by xax and quantile category and include
        # measure attribute
        gb = frame.groupby([xax, 'quantile'])[measure]
        # apply a groupby method to the groups
        gb = getattr(gb, groupby_method)()
        # unstack and plot
        gb = gb.unstack()
        y_limit = max(y_limit, np.nanmax(gb.values))
        if multiplot or not custom_color:
            gb.plot(c=eg_colors[eg - 1], lw=line_width,
                    ax=ax1, alpha=line_alpha)
        else:
            ax1.set_prop_cycle(cycler('color', clrs))
            gb.plot(lw=line_width, ax=ax1, alpha=line_alpha)

    # set "dense" tick labels
    if measure in ['cat_order']:
        try:
            y_limit = (y_limit + 500) // 50 * 50
            ax1.set_ylim(y_limit, 0)
            tick_stride = min(y_limit / 10 // 10 * 10, 500)
            ax1.set_yticks(np.arange(0, y_limit, tick_stride))
            if show_job_bands:
                ax2.set_ylim(ax1.get_ylim())
        except:
            pass

    if measure in ['fbff', 'jobp', 'jnum', 'orig_job']:
        jnums = np.arange(1, job_levels + 2, 1)
        ax1.set_yticks(jnums)
        yticks = []
        for i in jnums:
            yticks.append(job_strs[i])
        ax1.set_yticklabels(yticks, va='top')
        ax1.set_ylim(job_levels + 1.25, 0.5)

    if measure in ['spcnt', 'lspcnt']:
        ax1.yaxis.set_major_formatter(pct_format())
        ax1.set_yticks(np.arange(0, 1.05, .05))

    ax1.legend_.remove()

    try:
        ax2.tick_params(axis='both', left='off',
                        which='both', labelsize=tick_size)
    except:
        pass

    ax1.set_facecolor(bg_color)
    if show_grid:
        ax1.grid(b=True, c='grey', lw=.5, alpha=grid_alpha)
    else:
        ax1.grid(b=False)

    m_list = ['spcnt', 'lspcnt', 'lnum', 'snum', 'fbff',
              'orig_job', 'rank_in_job']

    if (measure in m_list) and (groupby_method not in ['size', 'count']):
        ax1.set_ylim(ymin=0)
        ax1.invert_yaxis()

    ax1.tick_params(axis='both', which='both', labelsize=tick_size)
    ax1.set_ylabel(attr_dict[measure] + ' for each quantile',
                   fontsize=label_size)
    ax1.set_xlabel(xax, fontsize=label_size)
    ax1.yaxis.labelpad = 10
    ax1.xaxis.labelpad = 7

    if (settings_dict['delayed_implementation'] and
            settings_dict['implementation_date'] and
            plot_implementation_date and xax == 'date'):

                ax1.axvline(settings_dict['implementation_date'],
                            c='g', ls='--', alpha=1, lw=1)

    ax1.set_title('egs: ' + str(eg_list) + '    ' + str(quantiles) +
                  ' quantile ' + attr_dict[measure] + ' by ' + groupby_method,
                  fontsize=title_size)

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)

        plt.savefig(image_dir + '/' + func_name + ' - ' + df_label +
                    ' grp ' + str(eg_list) +
                    '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def make_color_list(num_of_colors=10,
                    start=0.0,
                    stop=1.0,
                    exclude=None,
                    reverse=False,
                    cm_name_list=['Set1'],
                    return_list=True,
                    return_dict=False,
                    print_all_names=False,
                    palplot_cm_name=False,
                    palplot_all=False):
    '''Utility function to generate list(s) of colors (rgba format),
    any length and any from any section of any matplotlib colormap.

    The function can return a list of colors, a dictionary of colormaps
    to color lists, plot result(s) as seaborn palplot(s), and print out
    the names of all of the colormaps available.

    The end goal of this function is to provide customized color lists
    for plotting.

    inputs
        num_of_colors (integer)
            number of colors to produce for the output color list(s),
            used within the cm_subsection data calculation
        start (float)
            the starting point within the selected colormap to begin
            the spectrum color selection (0.0 to 1.0), used within the
            cm_subsection data calculation
        stop (float)
            the ending point within the selected colormap to end
            the spectrum color selection (0.0 to 1.0), used within the
            cm_subsection data calculation
        exclude (list)
            list of 2 floats representing a section of the colormap(s) to
            remove before calculating the result list(s).
        reverse (boolean)
            reverse the color list order which reverses the color spectrum
        cm_name_list (list)
            any matplotlib colormap name(s)
        return_list (boolean)
            if True, return a list of rgba color codes for the cm_name_list
            colormap input only, or (if the return_dict input is set to
            True) a dictionary of all colormap names to all of the
            resultant corresponding calculated color lists using the
            cm_subsection data
        return_dict (boolean)
            if True (and return_list is True), return a dictionary of
            all colormap names to all of the resultant corresponding
            calculated color lists
        print_all_names (boolean)
            if True (and return_list is False), print all the names of
            available matplotlib colormaps
        palplot_cm_name (boolean)
            if True (and return_list is set to False), plot a seaborn palplot
            of the color list produced with the cm_name_list colormap input
            using the cm_subsection data
        palplot_all (boolean)
            if True (and return_list and palplot_cm_name are False),
            plot a seaborn palplot for all of the color lists produced
            from all available matplotlib colormaps using the
            cm_subsection data
    '''
    # get a list of all of the matplotlib colormaps
    maps = sorted(m for m in plt.cm.datad if not m.endswith("_r"))
    # swap start and stop values if reverse input is True
    if reverse:
        start, stop = stop, start
        if exclude:
            exclude = exclude[::-1]
    # make the decimal array(s) to grab colormap sections
    if exclude:
        num_of_colors1 = num_of_colors // 2
        num_of_colors2 = num_of_colors - num_of_colors1
        ex_stop = exclude[0]
        ex_start = exclude[1]
        cmap_section1 = np.linspace(start, ex_stop, num_of_colors1)
        cmap_section2 = np.linspace(ex_start, stop, num_of_colors2)
        cm_subsection = np.concatenate((cmap_section1, cmap_section2))
    else:
        cm_subsection = np.linspace(start, stop, num_of_colors)

    lists_of_colors = []
    # make list of color lists (all colormaps) using cm_subsection data
    for m in maps:
        colormap = eval('cm.' + m)
        color_list = [colormap(x) for x in cm_subsection]
        lists_of_colors.append(color_list)
    # make a dictionary of map to list of colors from map
    color_dict = od(zip(maps, lists_of_colors))
    # returning a color list has priority over plots and printing
    if return_list:
        if return_dict:
            # return the entire color_dict (dictionary) and stop
            return color_dict
        else:
            # return only the cm_name color list (list) and stop
            color_lists = []
            if len(cm_name_list) == 1:
                return color_dict[cm_name_list[0]]
            else:
                for cmn in cm_name_list:
                    if cmn in color_dict.keys():
                        color_lists.append(color_dict[cmn])
                return color_lists
    else:
        # plot all lists of colors as seaborn palplots
        if palplot_all:
            for key in color_dict.keys():
                sns.palplot(color_dict[key])
                if print_all_names:
                    print(key)
                plt.show()
        # only plot cm_name_list as palplot
        else:
            if cm_name_list:
                for cmn in cm_name_list:
                    if cmn in color_dict.keys():
                        if exclude:
                            print('\n', 'palplot colormap:', cmn,
                                  ' [', start,
                                  ' >> ', ex_stop, '] [*exclude*] [',
                                  ex_start,
                                  ' >> ', stop, ']')
                        else:
                            print('\n', 'palplot colormap:', cmn,
                                  ' [ ', start, ' >> ', stop, ' ]')
                        sns.palplot(color_dict[cmn])
                        plt.show()
                    else:
                        print('"' + str(cmn) + '"',
                              'not found in color dictionary')
            else:
                print('set a cm_name_list argument for a palplot(s)')
        # print all of the colormap names
        if print_all_names and not palplot_all:
            print('All colormaps:', '\n')
            i = 0
            for key in color_dict.keys():
                print(i, key)
                i += 1


def add_pad(list_in, pad=100):
    '''Separate all elements in a monotonic list by a minimum pad value.

    Used by plotting functions to prevent overlapping tick labels.

    inputs
        list_in (list)
            a monotonic list of numbers
        pad (integer)
            the minimum separation required between list elements

    If the function is unable to produce a list with the pad between all
    elements (excluding the last list spacing), the original list is returned.
    The function will permit the final list padding (between the last two
    elements) to be less than the pad value.
    '''
    a = list_in[:]
    diff_list = []
    i = 0
    for tick in a[1:]:
        diff_list.append(tick - a[i])
        i += 1
    max_idx = diff_list.index(max(diff_list))
    sec1 = a[:max_idx]
    len_sec1 = len(sec1)

    if len_sec1 > 1:
        for i in np.arange(1, len_sec1):
            if sec1[i] - sec1[i - 1] < pad:
                sec1[i] = sec1[i - 1] + pad

    a[:len_sec1] = sec1

    sec2 = a[max_idx:]
    len_sec2 = len(sec2)

    if len_sec2 > 0:
        for i in np.arange(len_sec2 - 1, 1, -1):
            if sec2[i] - sec2[i - 1] < pad:
                sec2[i - 1] = sec2[i] - pad

    a[-len_sec2:] = sec2

    if f.monotonic(a):
        return a
    else:
        for i in np.arange(1, len(a[:-1])):
            if a[i] - a[i - 1] < pad:
                a[i - 1] = a[i] - pad

    if f.monotonic(a):
        return a
    else:
        return list_in


def add_editor_list_to_excel(case=None):
    '''save editor tool list order to the excel input file, "proposals.xlsx".

    The list order will be saved to a new worksheet, "edit".  Subsequent saved
    lists will overwrite previous worksheets.  Change the worksheet name of
    previously saved worksheets from "edit" to something else prior to running
    this function if they are to be preserved within the workbook.

    The routine reads the case_dill.pkl file - this provides a write path to
    the correct case study folder and excel "proposals.xlsx" file.
    Then the routine reads the editor-produced p_new_order.pkl file and writes
    it to the new worksheet "edit" in the proposals.xlsx file.

    input
        case (string)
            The case study name (and consequently, the write file path).
            This variable will default to the stored case study name contained
            within the "dill/case_dill.pkl" file if no input is supplied by
            the user.
    '''
    if not case:
        try:
            case = pd.read_pickle('dill/case_dill.pkl').case.value
        except OSError:
            print('case variable not found,',
                  'tried to find it in "dill/case_dill.pkl"',
                  'without success\n')
            import sys
            sys.exit()

    xl_str = 'excel/' + case + '/proposals.xlsx'
    df = pd.read_pickle('dill/p_new_order.pkl')
    df = df.reset_index()[['empkey']]
    df.index = df.index + 1
    df.index.name = 'order'

    ws_dict = pd.read_excel(xl_str, index_col=0, sheetname=None)
    ws_dict['edit'] = df

    with pd.ExcelWriter(xl_str, engine='xlsxwriter') as writer:

        for ws_name, df_sheet in ws_dict.items():
            df_sheet.to_excel(writer, sheet_name=ws_name)


# Pretty print a dictionary...
def pprint_dict(dct, marker1='#',
                marker2='',
                skip_line=True):
    '''print the key-value pairs in a horizontal, organized fashion.

    inputs
        dct (dictionary)
            the dictionary to print
        marker1, marker2
            prefix and suffix for the dictionary key headers
    '''
    for el in sorted(dct.items()):
        print(marker1, el[0], marker2)
        if skip_line:
            print('  ', el[1], '\n')
        else:
            print('  ', el[1])


def percent_bins(eg, base,
                 compare,
                 measure='spcnt',
                 by_year=True,
                 quantiles=20,
                 time_col='date',
                 agg_method='median'):
    '''Return a tuple of two dataframes containing differential percentage
    bin counts, one containing positive counts and another containing negative
    counts.

    This function first compares list percentage between two datasets on a
    grouped time period basis (annual or monthly), then counts the number of
    employees within specified percentage gain or loss quantiles.

    The counts are returned in dataframes with indexes reflecting the quantiles
    and columns representing the grouped time period.

    This function is used in the percent_diff_bins plotting function.

    inputs
        eg (integer)
            employee group code
        base (dataframe)
            baseline dataframe (dataset) containing a list percentage column
        compare (dataframe)
            comparison dataframe (dataset) containing a list percentage column
        measure (string)
            dataset percentage attribute column ('spcnt' or 'lspcnt')
        by_year (boolean)
            if True, group employee percentage differentials by year, otherwise
            by time_col input
        quantiles (integer)
            number of quantiles to measure.  An input of 20 would translate to
            quantiles of 5% each (100 / 20).
        time_col (string)
            if by_year is False, group percentage differentials by this time
            unit.  Inputs may be "mnum" or "date".
        agg_method (string)
            quantile bin aggregation method.  Inputs may be "mean" or "median"
    '''
    bins = np.linspace(0, 1, quantiles + 1)
    neg_bins = np.linspace(-1, 0, quantiles + 1)
    neg_bins[-1] = -.001
    bins[0] = .001

    c = compare[['mnum', 'date', 'eg', measure]]
    b = base[['mnum', 'date', 'eg', measure]]
    eg_c = c[c.eg == eg].copy()
    eg_b = b[b.eg == eg].copy()

    if by_year:
        eg_c['year'] = eg_c.date.dt.year
        pcnt_df = eg_c[[measure, 'year']].copy()
    else:
        pcnt_df = eg_c[[measure, time_col]].copy()

    pcnt_df[measure + '_b'] = eg_b[measure]
    pcnt_df['out'] = pcnt_df[measure + '_b'] - pcnt_df[measure]
    pcnt_df['out'].replace(to_replace=0.0, value=np.nan, inplace=True)
    pcnt_df['empkey'] = pcnt_df.index

    if by_year:
        grouped = pcnt_df[['empkey', 'year', 'out']] \
            .groupby([pd.Grouper('empkey'), 'year'])
    else:
        grouped = pcnt_df[['empkey', time_col, 'out']] \
            .groupby([pd.Grouper('empkey'), time_col])

    if agg_method == 'mean':
        pc_df = grouped.mean().unstack().fillna(0)
    if agg_method == 'median':
        pc_df = grouped.median().unstack().fillna(0)

    pc_df.columns = pc_df.columns.droplevel(0)
    pc_df_col_list = pc_df.columns.values.tolist()

    pos_df = pd.DataFrame(index=np.arange(1, quantiles + 1),
                          columns=pc_df_col_list)
    neg_df = pd.DataFrame(index=np.arange(1, quantiles + 1),
                          columns=pc_df_col_list)

    for time_period in pc_df_col_list:
        count, division = np.histogram(pc_df[time_period],
                                       bins=bins)
        neg_count, neg_division = np.histogram(pc_df[time_period],
                                               bins=neg_bins)
        pos_df[time_period] = count
        neg_df[time_period] = neg_count[::-1]

    return pos_df, neg_df * -1


# DIFFERENTIAL PERCENTAGE BINS
def percent_diff_bins(compare,
                      base, eg,
                      measure='spcnt',
                      kind='bar',
                      quantiles=40,
                      num_display_colors=25,
                      area_xax='date',
                      ds_dict=None,
                      attr1=None, oper1='>=', val1=0,
                      attr2=None, oper2='>=', val2=0,
                      attr3=None, oper3='>=', val3=0,
                      man_plotlim=None,
                      invert_barh=False,
                      chart_style='ticks',
                      cmap_pos='Vega20c',
                      cmap_neg='Vega20c',
                      zero_line_color='m',
                      bright_bg=False,
                      bg_color='#ffffe6',
                      title_size=14,
                      legend_size=12.5,
                      xsize=16, ysize=10,
                      image_dir=None,
                      image_format='png'):
    '''Display employee group counts within differential list
    percentage bins over time.

    Chart style options include bar, barh, and area.

    Selectable inputs include the number of percentile bins, chart colors and
    the number of colors in the color cycle representing the bins.

    The analysis groups may be targeted by up to three attribute value filters.

    inputs
        compare (dataframe)
            comparison dataframe (dateset)
        base (dataframe)
            baseline dataframe (dataset)
        eg (integer)
            employee group code
        measure (string)
            list percentage attribute for comparison ('spcnt' or 'lspcnt')
        kind (string)
            chart style ('bar', 'barh', or 'area')
        quantiles (integer)
            the number of differential percentage bins.  If the input is 40,
            each bin width will be 2.5% (100 / 40)
        num_display_colors (integer)
            the number of distinct colors to create from the cmap inputs.  If
            the input is less than the number of bins found for display, the
            colors display will cycle or repeat as necessary.
        area_xax (string)
            attribute to use for the chart when the kind input is set to
            'area'.  Inputs may be 'mnum' or 'date'.
        ds_dict (dictionary)
            variable assigned to the output of the load_datasets function.
            This keyword variable must be set if string dictionary keys are
            used as inputs for the dfc and/or dfb inputs.
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (integer, float, date as string, string (as appropriate))
            attr(n) limiting value (combined with oper(n)) as string
        man_plotlim (integer)
            if not None, restrict chart differential axis to this value.
            Otherwise, limit is set by an algorithm.
        invert_barh (boolean)
            If 'kind' input is set to 'barh', if True, invert the chart y axis
        chart_style (string)
            any valid seaborn plotting style name
        cmap_pos (string)
            any matplotlib colormap name representing colors to be applied to
            positive chart values
        cmap_neg (string)
            any matplotlib colormap name representing colors to be applied to
            negative chart values
        zero_line_color (color value)
            color to be applied to the chart zero line
        bright_bg (boolean)
            if True, color the chart background with the 'bg_color' color value
        bg_color (color value)
            color to use for the chart background if 'bright_bg' is True
        title_size (integer or float)
            text size for the chart title
        legend_size (integer or float)
            text size for the chart legend
        xsize, ysize (integers or floats)
            Width and height of chart in inches
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''

    b, b_label = determine_dataset(base, ds_dict, return_label=True)
    c, c_label = determine_dataset(compare, ds_dict, return_label=True)

    d_filtb = filter_ds(b,
                        attr1=attr1, oper1=oper1, val1=val1,
                        attr2=attr2, oper2=oper2, val2=val2,
                        attr3=attr3, oper3=oper3, val3=val3,
                        return_title_string=False)

    d_filtc, t_string = filter_ds(c,
                                  attr1=attr1, oper1=oper1, val1=val1,
                                  attr2=attr2, oper2=oper2, val2=val2,
                                  attr3=attr3, oper3=oper3, val3=val3)

    b = d_filtb[['mnum', 'date', 'eg', measure]]
    c = d_filtc[['mnum', 'date', 'eg', measure]]

    with sns.axes_style(chart_style):
        fig, ax1 = plt.subplots(figsize=(xsize, ysize))

    pos_colors = make_color_list(num_of_colors=num_display_colors,
                                 cm_name_list=[cmap_pos])
    neg_colors = make_color_list(num_of_colors=num_display_colors,
                                 cm_name_list=[cmap_neg])

    eg_label = 'Group ' + str(eg) + ',  '
    proposal_str = c_label + ' vs ' + b_label

    if t_string:
        t_string = t_string + '\n'

    if kind == 'area':
        title = (eg_label + proposal_str + '\n' + measure.upper() +
                 ' percent differential bin counts\n' + t_string)
        y_label = '<< LOSS         GAIN >>'

        if any(x in ['ret_mark'] for x in [attr1, attr2, attr3]):
            by_year = True
        else:
            by_year = False

    if kind == 'barh':
        title = (eg_label + proposal_str + '\n' + measure.upper() +
                 ' percent differential bin counts\n' +
                 t_string + '\n<< LOSS' +
                 (' ' * 12) + (' ' * 12) + 'GAIN >>')
        by_year = True
    if kind == 'bar':
        title = (eg_label + proposal_str + '\n' + measure.upper() +
                 ' percent differential bin counts\n' + t_string)
        y_label = '<< LOSS         GAIN >>'
        by_year = True

    pos, neg = percent_bins(eg, b, c, by_year=by_year,
                            quantiles=quantiles, measure=measure,
                            agg_method='median', time_col=area_xax)

    pv = pos.values
    nv = neg.values

    raw_xlim = max([np.amax(np.add.reduce(pv, 0)),
                   (np.abs(np.amin(np.add.reduce(nv, 0))))])
    try:
        pos_bins_found = np.amax(np.nonzero(pv)[0])
    except ValueError:
        pos_bins_found = 0
    try:
        neg_bins_found = np.amax(np.nonzero(nv)[0])
    except ValueError:
        neg_bins_found = 0

    bins_found = max(pos_bins_found, neg_bins_found) + 1

    label_arr = (np.linspace(0, 1, quantiles + 1) * 100)[1:]
    pos_labels = [str(x) + '%' for x in label_arr][:bins_found]
    neg_labels = [str(x) + '%' for x in label_arr * -1][:bins_found]

    str_labels = pos_labels + neg_labels

    if man_plotlim:
        plotlim = man_plotlim
    else:
        if raw_xlim <= 1500:
            rounder = 100
        else:
            rounder = 200
        plotlim = (raw_xlim + rounder) // 100 * 100

    if kind == 'bar':
        pos.T.plot(kind='bar', stacked=True, color=pos_colors,
                   width=1, edgecolor='k', linewidth=0.5, ax=ax1)
        ax1.set_ylim(-plotlim, plotlim)
    if kind == 'barh':
        pos.T.plot(kind='barh', stacked=True, color=pos_colors,
                   width=1, edgecolor='k', linewidth=0.5, ax=ax1)
        ax1.set_xlim(-plotlim, plotlim)
    if kind == 'area':
        pos.T.plot(kind='area', stacked=True, color=pos_colors,
                   linewidth=0, ax=ax1)
        ax1.set_ylim(-plotlim, plotlim)

    ax2 = ax1.twiny()
    ax2.tick_params(axis='both',
                    which='both',
                    right='off',
                    left='off',
                    bottom='off',
                    top='off',
                    labelright='off',
                    labelbottom='off',
                    labeltop='off',
                    labelleft='off')

    if kind == 'barh':
        neg.T.plot(kind='barh', stacked=True, color=neg_colors,
                   width=1, edgecolor='k', linewidth=0.5, ax=ax2)
        ax2.set_xlim(-plotlim, plotlim)
        if invert_barh:
            ax2.invert_yaxis()
        ax1.axvline(c=zero_line_color, lw=2, ls='dotted')
        ax2.axvline(c=zero_line_color, lw=2, ls='dotted')
        which = 'major'
    if kind == 'bar':
        neg.T.plot(kind='bar', stacked=True, color=neg_colors,
                   width=1, edgecolor='k', linewidth=0.5, ax=ax2)
        ax2.set_ylim(-plotlim, plotlim)
        ax1.axhline(c=zero_line_color, lw=2, ls='dotted')
        ax2.axhline(c=zero_line_color, lw=2, ls='dotted')
        which = 'major'
        ax1.set_ylabel(y_label, fontsize=14)
    if kind == 'area':
        neg.T.plot(kind='area', stacked=True, color=neg_colors,
                   linewidth=0, ax=ax2)
        ax2.set_ylim(-plotlim, plotlim)
        ax1.axhline(c='k', lw=2, ls='dotted')
        ax2.axhline(c='k', lw=2, ls='dotted')
        which = 'both'
        ax1.minorticks_on()
        ax1.set_ylabel(y_label, fontsize=14)

    ax1.grid(ls='dotted', color='gray', alpha=.25, which=which)
    # turn off ax2 grid lines...
    ax2.grid(False)

    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend_.remove()

    handles2 = handles2[:bins_found]
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    handles1, labels1 = ax1.get_legend_handles_labels()

    handles1 = handles1[:bins_found]
    handles = handles1 + handles2
    ax1.legend(handles, str_labels, loc='center left',
               bbox_to_anchor=(1.01, 0.5),
               fontsize=legend_size, ncol=2,
               title='Percentage Bins  \n\nGain            Loss')
    if bright_bg:
        ax1.set_facecolor(bg_color)

    ax1.set_title(title, fontsize=title_size)

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def cohort_differential(ds, base,
                        sdict, cdict,
                        adict,
                        measure='ldate',
                        compare_value='2010-12-31',
                        mnum=None,
                        ds_dict=None,
                        single_eg_compare=None,
                        sort_xax_by_measure=False,
                        attr1=None, oper1='>=', val1=0,
                        attr2=None, oper2='>=', val2=0,
                        attr3=None, oper3='>=', val3=0,
                        pos_color='g',
                        neg_color='r',
                        pos_alpha=.25,
                        neg_alpha=.25,
                        bg_color=None,  # #ffffe6
                        zero_line_color='m',
                        title_size=16,
                        label_size=14,
                        tick_size=12.5,
                        legend_size=12.5,
                        xsize=14, ysize=10,
                        image_dir=None,
                        image_format='png'):
    '''Compare proposed integrated list locations of employees from different
    groups who share a similar attribute value.

    This function is best used with date-type attributes, such as longevity
    date or date of hire.

    The comparative list locations are a continuous list of index locations
    determined by finding the last list position within an attribute column
    from another employee group which is less than or equal to a corresponding
    column from the base employee group.  A variance or differential is
    calculated by comparing the base and comparative locations.

    Attributes (measures) are sorted within each employee group prior to
    comparison.  The x axis may be arranged to display proposed list ordering
    or the attribute value range (typically a date range).

    Differences in list position are shown with a line above or below zero.
    One employee group (base) is compared to other group(s) in the proposed
    list within a selected month.  When the line is above zero, it means
    that the base group cohort at a particular x axis position is on the list
    ahead of another group cohort by an amount equal to the y displacement
    of the line.  The line colors correspond to the employee group color
    codes.

    The default behavior is to compare the base group with all other groups
    at once, but single group comparison may be accomplished as well.

    When the x axis is set to display list location (not attribute values),
    the user may designate a compare value.  The list location of employees
    from each group who share the comparison attribute value will be marked
    on the chart with a color-coded vertical line.

    inputs
        ds (dataframe)
            dataset for analysis
        base (integer)
            employee group number code
        sdict (dictionary)
            program settings dictionary
        cdict (dictionary)
            program color dictionary
        adict (dictionary)
            program attribute dictionary
        measure (string)
            attribute column for list location comparison, likely 'ldate' or
            'doh'
        compare_value (type to match measure input dtype)
            value to mark on chart if "sort_xax_by_measure" input is False.
            Likely a date string, such as "2001-01-31"
        mnum (integer)
            data model month number to study
        ds_dict (dictionary)
            dictionary of datasets, likely generated by the "load_datasets"
            function
        single_eg_compare (integer)
            if not None, compare base employee group to this group only
        sort_xax_by_measure (boolean)
            if True, use an x axis for the chart based on the selected measure.
            if False, use list location for the x axis
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (integer, float, date as string, string (as appropriate))
            attr(n) limiting value (combined with oper(n)) as string
        pos_color, neg_color (color value string)
            color used for the positive and negative area shading
        pos_alpha, neg_alpha (integer or float)
            transparency value assigned to the positive and negative color
            shading areas (0.0 to 1.0)
        bg_color (color value string)
            if not None, the color for the chart background
        zero_line_color (color value string)
            color for the zero line
        title_size (integer or float)
            text size for the chart title
        label_size (integer or float)
            text size for the chart axis labels
        tick_size (integer or float)
            text size for the chart tick labels
        legend_size (integer or float)
            text size for the chart legend
        xsize, ysize (integer or float)
            size of the chart in inches (width, height)
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''
    d, d_label = determine_dataset(ds, ds_dict, return_label=True)

    if mnum is None:
        mnum = 0

    df0 = d[d.mnum == mnum].copy()
    df0['list_order'] = np.arange(len(df0))

    df0 = filter_ds(df0,
                    attr1=attr1, oper1=oper1, val1=val1,
                    attr2=attr2, oper2=oper2, val2=val2,
                    attr3=attr3, oper3=oper3, val3=val3,
                    return_title_string=False)

    df0 = df0[['eg', measure, 'list_order']]

    eg_colors = cdict['eg_colors']

    eg_dict = {}
    color_dict = {}
    for eg in set(df0.eg):
        df = df0[df0.eg == eg].copy()
        df[measure] = np.sort(df[measure])
        eg_dict[eg] = df
        color_dict[eg] = eg_colors[eg - 1]

    other_egs = [eg for eg in list(eg_dict.keys()) if eg != base]
    dfb = eg_dict[base]

    asof_dict = {}

    for eg in other_egs:
        dfc = eg_dict[eg]
        asof = pd.merge_asof(dfb[[measure, 'list_order']],
                             dfc[[measure, 'list_order']],
                             on=measure,
                             suffixes=('_grp' + str(base), '_compare'),
                             allow_exact_matches=True)
        asof['cohort_diff'] = asof['list_order_compare'] -\
            asof['list_order_grp' + str(base)]
        if sort_xax_by_measure:
            asof.set_index(measure, inplace=True)
            asof_dict[eg] = asof
        else:
            asof.set_index('list_order_grp' + str(base), inplace=True)
            asof_dict[eg] = asof

    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(figsize=(xsize, ysize))

    if single_eg_compare:
        if single_eg_compare == base:
            print('''single_eg_compare input must be different than base eg,\n
                  reverting to plotting all employee groups...''')
            plot_egs = list(asof_dict.keys())
        else:
            plot_egs = [single_eg_compare]
    else:
        plot_egs = list(asof_dict.keys())

    for eg in plot_egs:
        yvals = asof_dict[eg].cohort_diff
        yvals.plot(color=color_dict[eg],
                   lw=1.5,
                   label=sdict['p_dict_verbose'][eg])

        ax.invert_xaxis()
        ax.fill_between(asof.index, 0, yvals,
                        where=yvals > 0, facecolor=pos_color,
                        alpha=pos_alpha, interpolate=True)
        ax.fill_between(asof.index, 0, yvals,
                        where=yvals < 0, facecolor=neg_color,
                        alpha=neg_alpha, interpolate=True)

    if sort_xax_by_measure:
        ax.set_xlim(max(df0[measure]), min(df0[measure]))
        ax.set_xlabel(adict[measure], fontsize=label_size)
    else:
        ax.set_xlim(max(df0.list_order), 0)
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_size)
    ax.legend(fontsize=legend_size, loc=0)
    ax.axhline(lw=1.5, color='m')

    ax.set_title('Proposal ' + d_label +
                 ' - [ group ' + str(base) +
                 ' ] - ' + adict[measure] +
                 ' differential - month ' +
                 str(mnum), fontsize=title_size)

    ax.set_ylabel('list position compared to cohorts',
                  fontsize=label_size)

    ax.tick_params(axis='both', labelsize=tick_size)
    ax.grid(alpha=.15, ls='dotted', color='k')
    if sort_xax_by_measure and measure in ['date', 'ldate', 'doh', 'retdate']:
        locator = mdate.YearLocator()
        ax.xaxis.set_major_locator(locator)
        fig.autofmt_xdate()
        plt.xticks(rotation=75, ha='center')
        if len(ax.get_xticks()) > 20:
            for label in ax.xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)
    ax.yaxis.labelpad = 10
    ax.xaxis.labelpad = 10

    if not sort_xax_by_measure and compare_value:

        try:
            marker_dict = {}
            base_and_plot_egs = [base] + plot_egs
            print('Base group is group < ' + str(base) + ' >')
            print('finding last list locations for:\n\n  ' +
                  measure + ' <= ' + str(compare_value) + '\n')
            print('Results:')
            for eg in base_and_plot_egs:
                df = eg_dict[eg]
                if compare_value:
                    try:
                        marker_dict[eg] = (df[df[measure] <= compare_value]
                                           .list_order.values[-1])
                    except IndexError:
                        marker_dict[eg] = np.nan

                print('Group ' + str(eg) +
                      ' location: ' + str(marker_dict[eg]))
                if eg == base:
                    ls = 'solid'
                else:
                    ls = 'dotted'
                ax.axvline(marker_dict[eg], color=color_dict[eg], ls=ls, lw=2)

            print('')
            for eg in plot_egs:
                print('Relative to group ' + str(eg) + ' cohort: ' +
                      str(marker_dict[eg] - marker_dict[base]))

        except (ValueError, TypeError):

            print('''Error plotting comparative location "compare_value"
                  vertical lines.\n perhaps check type equivalence?\n
                  ("compare_value" input vs. "measure" input)''')

    if bg_color:
        ax.set_facecolor(bg_color)

    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)
    plt.show()


def eg_attributes(ds, xmeasure, ymeasure,
                  sdict,
                  adict,
                  cdict,
                  eg_list=None,
                  mnum=None,
                  ret_only=False,
                  ds_dict=None,
                  attr1=None, oper1='>=', val1=0,
                  attr2=None, oper2='>=', val2=0,
                  attr3=None, oper3='>=', val3=0,
                  q_eglist_only=True,
                  xquant_lines=True,
                  x_quantiles=10,
                  xl_alpha=1,
                  xl_ls='dashed',
                  xl_lw=1,
                  xl_color='.7',
                  x_bands=True,
                  xb_fc='.3',
                  xb_alpha=.09,
                  yquant_lines=True,
                  y_quantiles=10,
                  yl_alpha=1,
                  yl_ls='dashed',
                  yl_lw=1,
                  yl_color='.7',
                  y_bands=True,
                  yb_fc='#66ffb3',
                  yb_alpha=.09,
                  linestyle='',
                  linewidth=0,
                  markersize=5,
                  marker_alpha=.7,
                  grid_alpha=.25,
                  chart_style='ticks',
                  full_xpcnt=True,
                  full_ypcnt=True,
                  xax_rotate=70,
                  label_size=13,
                  qtick_size=12,
                  tick_size=12,
                  border_size=.5,
                  legend_size=14,
                  title_size=18,
                  y_title_pos=1.12,
                  box_height=.95,
                  xsize=15,
                  ysize=11,
                  image_dir=None,
                  image_format='png'):
    '''Plot selected employee group(s) attribute data.

    Chart x and y axes may be any dataset attributes, including date
    attributes.

    Quantile membership for the x and/or y attribute may also be displayed.
    Membership may be relative to the entire integrated population or only
    to the employee group(s) selected for display (q_eglist_only input).

    inputs
        ds (dataframe)
            dataset to examine, may be a dataframe variable or a string key
            from the ds_dict dictionary object
        xmeasure (string)
            attribute to plot on x axis
        ymeasure (string)
            attribute to plot on y axis
        sdict (dictionary)
            program settings dictionary
        adict (dictionary)
            dataset column name description dictionary
        cdict (dictionary)
            program colors dictionary
        eg_list (list)
            list of employee groups to plot (integer codes)
        mnum (integer)
            month number for analysis
        ret_only (boolean)
            if True, mnum input is ignored and results are displayed for
            all employees at retirement
        ds_dict (dictionary)
            output of the load_datasets function, dictionary.  This keyword
            argument must be set if a string key is used as the df input.
        attr(n) (string)
            filter attribute or dataset column as string
        oper(n) (string)
            operator (i.e. <, >, ==, etc.) for attr(n) as string
        val(n) (string, integer, float, date as string as appropriate)
            attr(n) limiting value (combined with oper(n)) as string
        q_eglist_only (boolean)
            if set to True:

                if quantile bands are displayed, show membership based on
                selected employee groups (eg_list input).

            if set to False:

                if quantile bands are displayed, show membership based on
                the integrated group population (all groups).
        xquant_lines (boolean)
            if True, show quantile membership for x axis attribute
        x_quantiles (integer)
            number of quantiles to display if xquant_lines input is True
        xl_alpha (float)
            transparency value of x axis quantile lines (0.0 to 1.0)
        xl_ls (string)
            x axis quantile lines linestyle ('dashed', 'dotted', etc.)
        xl_lw (integer or float)
            x axis quantile lines line width
        xl_color (string color value)
            x axis quantile lines color
        x_bands (boolean)
            if True, show a background color within every other x axis
            quantile membership area
        xb_fc (string color value)
            x axis quantile bands background color
        xb_alpha (float)
            x axis quantile bands color transparency value (0.0 to 1.0)
        yquant_lines (boolean)
            if True, show quantile membership for y axis attribute
        y_quantiles (integer)
            number of quantiles to display if yquant_lines input is True
        yl_alpha (float)
            transparency value of y axis quantile lines (0.0 to 1.0)
        yl_ls (string)
            y axis quantile lines linestyle ('dashed', 'dotted', etc.)
        yl_lw (integer or float)
            y axis quantile lines line width
        yl_color (string color value)
            y axis quantile lines color
        y_bands (boolean)
            if True, show a background color within every other y axis
            quantile membership area
        yb_fc (string color value)
            y axis quantile bands background color
        yb_alpha (float)
            y axis quantile bands color transparency value (0.0 to 1.0)
        markersize (integer or float)
            size of chart scatter points
        marker_alpha (integer or float)
            transparency setting for plot lines or points (0.0 to 1.0)
        grid_alpha (float)
            transparency value for the chart grid corresponding to the x and
            y attribute values (not the quantile membership lines)
        chart_style (string)
            any valid seaborn chart style name
        full_xpcnt (boolean)
            if True, show full range percentage (0 to 100 percent) when
            a percentage attribute is displayed on the x axis
        full_ypcnt (boolean)
            if True, show full range percentage (0 to 100 percent) when
            a percentage attribute is displayed on the y axis
        xax_rotate (integer)
            rotation value (in degrees) for the x axis tick labels
        qtick_size (integer or float)
            text size of the quantile membership tick labels
        tick_size (integer or float)
            text size of the x and y attribute tick labels
        label_size (integer or float)
            text size of x and y axis labels
        border_size (integer or float)
            width of the chart border line (chart spines)
        legend_size (integer or float)
            text size of chart legend
        title_size (integer or float)
            text size of chart title
        y_title_pos (float)
            vertical position of the chart title when attribute filtering has
            been applied.  (typical values are 1.1 to 1.2)
        box_height (float)
            chart height multiplier which slightly shrinks vertical chart
            area for proper printing (saving) purposes.  This input does not
            affect the displayed values.
        xsize, ysize (integer or float)
            plot size in inches
        image_dir (string)
            if not None, name of a directory in which to save an image of the
            chart output.  If the directory does not exist, it will be
            created.
        image_format (string)
            file extension string for a saved chart image if the image_dir
            input is not None

            Examples:

                'svg', 'png'
    '''
    d, d_label = determine_dataset(ds, ds_dict, return_label=True)

    # filter for ret_only or specific month
    if ret_only:
        d = d[d.ret_mark == 1].copy()
    else:
        if mnum is not None:
            d = d[d.mnum == mnum].copy()

    # additional user-defined filters
    df, filt_title = filter_ds(d,
                               attr1=attr1, oper1=oper1, val1=val1,
                               attr2=attr2, oper2=oper2, val2=val2,
                               attr3=attr3, oper3=oper3, val3=val3,
                               return_title_string=True)

    # reduce data to specific employee group(s)
    if q_eglist_only:
        egs = df.eg.values
        if eg_list is None:
            eg_list = np.unique(egs)
        df = df[np.in1d(egs, eg_list)].copy()

    # filter to include only active employees if an "active only" attribute
    # is selected
    no_fur_list = ['spcnt', 'snum']
    if xmeasure in no_fur_list or ymeasure in no_fur_list:
        df = df[df.spcnt >= 0]

    # list of attributes where a lower value should be at the top or right
    # of chart display
    invert_attr_list = ['cat_order', 'ldate', 'doh', 'retdate', 'date',
                        'snum', 'lnum', 'spcnt', 'lspcnt', 'jnum', 'jobp',
                        'fbff', 'orig_job', 'rank_in_job', 'new_order']

    job_attr_list = ['jnum', 'jobp', 'orig_job', 'fbff']
    date_attr_list = ['ldate', 'doh', 'retdate', 'date']

    # define chart axis limits
    if xmeasure in ['snum', 'lnum', 'spcnt', 'lspcnt', 'cat_order']:
        minx = 0
        if full_xpcnt and xmeasure in ['spcnt', 'lspcnt']:
            maxx = 1
        else:
            maxx = max(df[xmeasure])
    elif xmeasure in job_attr_list:
        minx = .75
        maxx = sdict['num_of_job_levels'] + 2
    else:
        minx = min(df[xmeasure])
        maxx = max(df[xmeasure])

    if ymeasure in ['snum', 'lnum', 'spcnt', 'lspcnt', 'cat_order']:
        miny = 0
        if full_ypcnt and ymeasure in ['spcnt', 'lspcnt']:
            maxy = 1
        else:
            maxy = max(df[ymeasure])
    elif ymeasure in job_attr_list:
        miny = .75
        maxy = sdict['num_of_job_levels'] + 2
    else:
        miny = min(df[ymeasure])
        maxy = max(df[ymeasure])

    dflen = len(df)

    num_of_job_levels = sdict['num_of_job_levels']
    eg_colors = cdict['eg_colors']

    # set employee group list
    if eg_list is None:
        eg_list = list(set(df.eg))
    else:
        eg_list = list(set(eg_list).intersection(set(df.eg)))

    with sns.axes_style(chart_style):
        fig, ax1 = plt.subplots(figsize=(xsize, ysize))

    # set x and y chart values - format values if date attribute
    if xmeasure in date_attr_list:
        xto_dates = pd.to_datetime(df[xmeasure])
        dtx = list(xto_dates.dt.strftime('%Y-%b-%d'))
        x = mdate.datestr2num(dtx)
        max_yearx = max(df[xmeasure].dt.year)
        min_yearx = min(df[xmeasure].dt.year)
        yrngx = max_yearx - min_yearx
    else:
        x = df[xmeasure].values

    if ymeasure in date_attr_list:
        yto_dates = pd.to_datetime(df[ymeasure])
        dty = list(yto_dates.dt.strftime('%Y-%b-%d'))
        y = mdate.datestr2num(dty)

        max_yeary = max(df[ymeasure].dt.year)
        min_yeary = min(df[ymeasure].dt.year)
        yrngy = max_yeary - min_yeary
    else:
        y = df[ymeasure].values

    # plot chart
    for eg in eg_list:
        mask = (df.eg.values == eg)
        egy = y[mask]
        egx = x[mask]
        ax1.plot(egx, egy,
                 color=eg_colors[eg - 1],
                 ls=linestyle,
                 lw=linewidth,
                 marker='o',
                 markersize=markersize,
                 alpha=marker_alpha,
                 label=eg)

    # more date attribute handling:
    if xmeasure in date_attr_list:
        ax1.xaxis_date()

        ax1.xaxis.set_major_locator(mdate.YearLocator())
        ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y'))
        if yrngx <= 3:
            ax1.xaxis.set_minor_locator(mdate.MonthLocator())
            ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
        fig.autofmt_xdate()
        plt.xticks(rotation=75, ha='center')
        if len(ax1.get_xticks()) > 20:
            for label in ax1.xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)

    if ymeasure in date_attr_list:
        ax1.yaxis_date()
        ax1.yaxis.set_major_locator(mdate.YearLocator())
        ax1.yaxis.set_major_formatter(mdate.DateFormatter('%Y'))
        if yrngy <= 3:
            ax1.yaxis.set_minor_locator(mdate.MonthLocator())
            ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
        if len(ax1.get_yticks()) > 20:
            for label in ax1.yaxis.get_ticklabels()[1::2]:
                label.set_visible(False)

    # axis setup if percentage or job-level related
    if xmeasure in ['spcnt', 'lspcnt']:
        ax1.set_xticks(np.arange(0, 1.05, .05))
        ax1.xaxis.set_major_formatter(pct_format())
    if ymeasure in ['spcnt', 'lspcnt']:
        ax1.set_yticks(np.arange(0, 1.05, .05))
        ax1.yaxis.set_major_formatter(pct_format())
    if xmeasure in ['jnum', 'jobp', 'orig_job', 'fbff']:
        ax1.set_xticks(np.arange(0, num_of_job_levels + 2).astype(int))
    if ymeasure in ['jnum', 'jobp', 'orig_job', 'fbff']:
        ax1.set_yticks(np.arange(0, num_of_job_levels + 2).astype(int))

    # set chart axis limits
    ax1.set_xlim(minx, maxx)
    ax1.set_ylim(miny, maxy)

    # quantile lines/bands start--------------
    if xquant_lines:

        with sns.axes_style(chart_style):
            x_ax = ax1.twiny()

        xdiv_list = np.linspace(0, 1, x_quantiles + 1)
        qx_labels = ["{0:.1f}%".format(f * 100) for f in xdiv_list]
        xlines = np.linspace(0, dflen, x_quantiles + 1).astype(int)
        xlines[-1] = xlines[-1] - 1
        sorted_xdf = df[[xmeasure]].sort_values(xmeasure, ascending=False)
        x_locations = []
        for line in xlines:
            x_locations.append(sorted_xdf.iloc[line][xmeasure])
        x_ax.set_xticks(x_locations)
        x_ax.set_xlim(ax1.get_xlim())
        x_ax.grid(ls=xl_ls, lw=xl_lw, color=xl_color)

        # x quantile bands
        if x_bands:
            x1 = x_locations[0:-1:2]
            x2 = x_locations[1::2][:len(x1)]
            for x1, x2 in zip(x1, x2):
                x_ax.axvspan(x1, x2, facecolor=xb_fc, alpha=xb_alpha)

    if yquant_lines:

        with sns.axes_style(chart_style):
            y_ax = ax1.twinx()

        ydiv_list = np.linspace(0, 1, y_quantiles + 1)
        qy_labels = ["{0:.1f}%".format(f * 100) for f in ydiv_list]
        ylines = np.linspace(0, dflen, y_quantiles + 1).astype(int)
        ylines[-1] = ylines[-1] - 1
        sorted_ydf = df[[ymeasure]].sort_values(ymeasure, ascending=False)

        y_locations = []
        for line in ylines:
            y_locations.append(sorted_ydf.iloc[line][ymeasure])

        y_ax.set_ylim(ax1.get_ylim())
        y_ax.set_yticks(y_locations)
        y_ax.grid(ls=yl_ls, lw=yl_lw, color=xl_color)

        # y quantile bands
        if y_bands:
            y1 = y_locations[0:-1:2]
            y2 = y_locations[1::2][:len(y1)]
            for y1, y2 in zip(y1, y2):
                y_ax.axhspan(y1, y2, facecolor=yb_fc, alpha=yb_alpha)
    # quantile lines/bands end--------------

    # invert/label section start......................
    if ymeasure in invert_attr_list:

        ax1.invert_yaxis()
        if yquant_lines:
            if ymeasure in invert_attr_list:
                y_ax.invert_yaxis()
                y_ax.set_yticklabels(qy_labels[::-1], rotation=-4,
                                     fontsize=qtick_size, va='top')
            else:
                y_ax.set_yticklabels(qy_labels, rotation=-4,
                                     fontsize=qtick_size, va='top')

            if len(ax1.get_yticks()) > 20:
                for label in ax1.yaxis.get_ticklabels()[1::2]:
                    label.set_visible(False)

    else:
        if yquant_lines:
            y_ax.set_yticklabels(qy_labels, rotation=-4)

    if xmeasure in invert_attr_list:

        ax1.invert_xaxis()
        if xquant_lines:
            if xmeasure in invert_attr_list:
                x_ax.invert_xaxis()
                x_ax.set_xticklabels(qx_labels[::-1], rotation=75,
                                     fontsize=qtick_size, ha='left')
            else:
                x_ax.set_xticklabels(qx_labels, rotation=75,
                                     fontsize=qtick_size, ha='left')

            if len(ax1.get_xticks()) > 20:
                for label in ax1.xaxis.get_ticklabels()[1::2]:
                    label.set_visible(False)
    else:
        if xquant_lines:
            x_ax.set_xticklabels(qx_labels, rotation=75)
    # invert/label section end........................

    if ymeasure in job_attr_list:
        job_labels = ['']
        job_str_dict = sdict['job_strs_dict']
        y_ticks = ax1.get_yticks()
        for tick in y_ticks:
            try:
                job_labels.append(job_str_dict[tick])
            except:
                pass
        # move labels slightly down for y axis job-level attributes
        ax1.set_yticklabels(job_labels, va='top')

    if xmeasure in job_attr_list:
        job_labels = ['']
        job_str_dict = sdict['job_strs_dict']
        x_ticks = ax1.get_xticks()
        for tick in x_ticks:
            try:
                job_labels.append(job_str_dict[tick])
            except:
                pass
        ax1.set_xticklabels(job_labels, va='top', ha='right', rotation=80)

    # title position adjustment
    if xquant_lines:
        y_pos = y_title_pos
    else:
        y_pos = 1.01

    if yquant_lines:
        legend_pad = 5
    else:
        legend_pad = 1

    title = (d_label + ' ' + adict[xmeasure] + ' vs. ' + adict[ymeasure] +
             ', groups ' + str(eg_list))
    if mnum is not None and ret_only is False:
        title = title + ' ,month ' + str(mnum)
    if ret_only:
        title = title + ' at retirement'
    if filt_title:
        title = title + '\n' + filt_title

    ax1.set_title(title, fontsize=title_size, y=y_pos)

    if xmeasure not in date_attr_list:
        for tick in ax1.get_xticklabels():
            tick.set_rotation(xax_rotate)
    ax1.tick_params(axis='both', labelsize=tick_size)
    ax1.grid(alpha=grid_alpha)
    ax1.set_xlabel(adict[xmeasure], fontsize=label_size)
    ax1.set_ylabel(adict[ymeasure], fontsize=label_size)
    ax1.xaxis.labelpad = 5

    # legend
    legend_title = 'eg'
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0,
                     box.width * 0.92, box.height * box_height])
    if xquant_lines:
        x_ax.set_position([box.x0, box.y0,
                          box.width * 0.92, box.height * box_height])
    if yquant_lines:
        y_ax.set_position([box.x0, box.y0,
                          box.width * 0.92, box.height * box_height])

    handles, labels = ax1.get_legend_handles_labels()
    lg = ax1.legend(handles, labels, title=legend_title, loc='center left',
                    bbox_to_anchor=(1.0, 0.5),
                    borderaxespad=legend_pad,
                    frameon=True,
                    fancybox=True,
                    markerscale=2,
                    fontsize=legend_size)
    lg.get_frame().set_linewidth(1.5)

    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(border_size)

    # saving chart image:
    if image_dir:
        func_name = sys._getframe().f_code.co_name
        if not path.exists(image_dir):
            makedirs(image_dir)
        plt.savefig(image_dir + '/' + func_name + '.' + image_format,
                    bbox_inches='tight', pad_inches=.25)

    plt.show()
