#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# seniority_list is an analytical tool used when seniority-based work
# groups merge. It brings modern data science to the area of labor
# integration, utilizing the powerful data analysis capabilities of Python
# scientific computing.

# Copyright (C) 2016-2017  Robert E. Davison, Ruby Data Systems Inc.
# Please direct inquires to: rubydatasystems@fastmail.net

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

#.. moduleauthor:: Bob Davison <rubydatasystems@fastmail.net>

'''
    Create general statistical reports for all program datasets and present
    as charts and spreadsheets within the reports folder.

'''

import numpy as np
import pandas as pd
import seaborn as sns
import openpyxl as op
import matplotlib.pyplot as plt
import functions as f
import matplotlib_charting as mp
from os import path, makedirs


def stats_to_excel(ds_dict,
                   quantiles=10,
                   date_grouper='ldate',
                   fixed_col_name='eg_initQ',
                   running_col_name='eg_runQ'):
    '''Create a set of basic statistics for each calculated dataset and
    write the results as spreadsheets within the **reports** folder.

    There are 2 spreadsheets produced, one related to retirement data and the
    other related to annual data.annual

    The retirement information is grouped by employees retiring in future
    years, further grouped for longevity or initial job.

    The annual information is grouped by the model year, and further grouped
    by 10% quantiles, by initial quantile membership and also by an annual
    quantile adjustment of remaining employees.

    inputs
        ds_dict (dictionary)
            output of load_datasets function, a dictionary of datasets
        quantiles (integer)
            the number of binning quantiles to measure for the initial and
            running (annually updated) quantile membership analysis
            (default is 10)
        date_grouper (string)
            column name representing a column of dates within a dataframe.
            Year membership of this column will be used for grouping.
            Input is limited to 'ldate' or 'doh'.
        fixed_col_name (string)
            label to use for quantile number column when calculating using
            the initial quantile membership for all results
        running_col_name (string)
            label to use for quantile number column when calculating using
            a continuously updated quantile membership for all results
    '''
    ret_dict = {}
    ann_dict = {}
    try:
        # get current case study name
        case_df = pd.read_pickle('dill/case_dill.pkl')
        case_name = case_df.at['prop', 'case']
    except OSError:
        print('unable to retrieve case name, try setting case input')
        return
    # remove the skeleton dataset from consideration for this function
    ds_list = [key for key in ds_dict.keys() if key != 'skeleton']
    attrs = ['spcnt', 'snum', 'cat_order', 'jobp', 'cpay', 'ylong']
    ret_attrs = [attr for attr in attrs if attr not in ['mlong', 'ylong']]

    # find number of employee groups in case study
    num_egs = np.unique(ds_dict[ds_list[0]].eg).size
    # find column numbers for each attribute measure
    ret_start = np.array(range(2, (num_egs * 6) + 1, num_egs))
    ret_end = ret_start + (num_egs - 1)
    ann_start = np.array(range(1, (num_egs * 6) + 1, num_egs))
    ann_end = ann_start + (num_egs - 1)
    # make an array of excel column letters
    xlcols = list('abcdefghijklmnopqrstuvwxyz'.upper())
    next_cols = ['A' + el for el in xlcols]
    xlcols.extend(next_cols)
    cols = np.array(xlcols)
    # slice column letter array according to column numbers
    ret_startxl_cols = cols[ret_start]
    ret_endxl_cols = cols[ret_end]
    ann_startxl_cols = cols[ann_start]
    ann_endxl_cols = cols[ann_end]
    # make excel column ranges for formatting (example: 'C:E')
    ret_col_dict = {}
    i = 0
    for idx in range(len(ret_startxl_cols)):
        ret_col_dict[i] = ret_startxl_cols[idx] + ':' + ret_endxl_cols[idx]
        i += 1
    ann_col_dict = {}
    i = 0
    for idx in range(len(ann_startxl_cols)):
        ann_col_dict[i] = ann_startxl_cols[idx] + ':' + ann_endxl_cols[idx]
        i += 1

    # loop through datasets
    for key in ds_list:

        # copy to avoid altering ds_dict values
        df = ds_dict[key].copy()
        # create employee group percentage columns, one fixed and the other
        # recalculated each month
        f.make_eg_pcnt_column(df, fixed_col_name=fixed_col_name)
        f.make_eg_pcnt_column(df, recalc_each_month=True,
                              running_col_name=running_col_name)
        # create a column assigning the initial model job for each
        # employee for all months
        f.make_starting_val_column(df, 'jnum')
        # make grouping values
        yr = df.date.dt.year
        lyr = df.ldate.dt.year
        rq = ((df[running_col_name] * quantiles // 1) + 1).astype(int)
        iq = ((df[fixed_col_name] * quantiles // 1) + 1).astype(int)

        retp = df[df.ret_mark == 1]
        dateyr = retp[date_grouper].dt.year
        job = retp.start_jnum
        ret = retp.retdate.dt.year
        # add grouped dataframes to dictionaries
        try:
            ret_dict[date_grouper + '_' + key] = \
                retp.groupby([dateyr, ret, 'eg'])[ret_attrs].mean().unstack()
            ret_dict['job' + '_' + key] = \
                retp.groupby([job, ret, 'eg'])[ret_attrs].mean().unstack()
            ann_dict['A_' + key] = \
                df.groupby([yr, 'eg'])[attrs].mean().unstack()
            ann_dict['Long_' + key] = \
                df.groupby([lyr, yr, 'eg'])[attrs].mean().unstack()
            ann_dict['RQ_' + key] = \
                df.groupby([rq, yr, 'eg'])[attrs].mean().unstack()
            ann_dict['IQ_' + key] = \
                df.groupby([iq, yr, 'eg'])[attrs].mean().unstack()
        except:
            print(key, 'report fail')

    # make a count worksheet
    for key in ds_dict.keys():
        if key != 'skeleton':
            found_key = key
            break
        else:
            continue
    dfk = ds_dict[found_key].copy()
    p = dfk[dfk.ret_mark == 1]
    yr = p.date.dt.year
    dateyr = p[date_grouper].dt.year
    retyr = p.retdate.dt.year

    ret_dict['retirement_count'] = p.groupby([dateyr, retyr, 'eg'])['mnum'] \
        .count().unstack().fillna(0).astype(int)
    ann_dict['retirement_count'] = p.groupby([retyr, 'eg'])['mnum'] \
        .count().unstack().fillna(0).astype(int)

    # write grouped retirement dataframes to excel workbook
    with pd.ExcelWriter('reports/' + case_name + '/ret_stats.xlsx',
                        engine='xlsxwriter') as writer:

        for ws_name, df_sheet in sorted(ret_dict.items()):
            df_sheet.to_excel(writer, sheet_name=ws_name)

            if ws_name not in ['retirement_count']:

                # prepare to format worksheet
                workbook = writer.book
                worksheet = writer.sheets[ws_name]

                format0 = workbook.add_format({'num_format': '#0',
                                               'align': 'center'})
                format2 = workbook.add_format({'num_format': '#0.00',
                                               'align': 'center'})
                formatpcnt = workbook.add_format({'num_format': '#0.0%'})
                # format each worksheet attribute column range
                worksheet.set_column('A:A', 10, None)
                worksheet.set_column('B:B', 7, None)
                worksheet.set_column(ret_col_dict[0], 7, formatpcnt)
                worksheet.set_column(ret_col_dict[1], 7, format0)
                worksheet.set_column(ret_col_dict[2], 7, format0)
                worksheet.set_column(ret_col_dict[3], 6, format2)
                worksheet.set_column(ret_col_dict[4], 7, format0)
                # freeze worksheet for scrolling with headers visible
                worksheet.freeze_panes('A4')

            else:
                # format retirement count worksheet
                workbook = writer.book
                worksheet = writer.sheets[ws_name]
                format0 = workbook.add_format({'num_format': '#0',
                                               'align': 'center'})

                worksheet.set_column('A:A', 10, None)
                worksheet.set_column('B:B', 7, None)
                worksheet.set_column(ret_col_dict[0], 7, format0)
                worksheet.freeze_panes('A2')

    # write grouped annual dataframes to excel workbook
    with pd.ExcelWriter('reports/' + case_name + '/annual_stats.xlsx',
                        engine='xlsxwriter') as writer:

        for ws_name, df_sheet in sorted(ann_dict.items()):
            df_sheet.to_excel(writer, sheet_name=ws_name)

            if ws_name not in ['retirement_count']:

                workbook = writer.book
                worksheet = writer.sheets[ws_name]

                format0 = workbook.add_format({'num_format': '#0',
                                               'align': 'center'})
                format1 = workbook.add_format({'num_format': '#0.0',
                                               'align': 'center'})
                format2 = workbook.add_format({'num_format': '#0.00',
                                               'align': 'center'})
                formatpcnt = workbook.add_format({'num_format': '#0.0%'})

                if ws_name.startswith('A_'):
                    col_dict = ann_col_dict
                else:
                    col_dict = ret_col_dict

                worksheet.set_column('A:A', 10, None)
                worksheet.set_column(col_dict[0], 7, formatpcnt)
                worksheet.set_column(col_dict[1], 7, format0)
                worksheet.set_column(col_dict[2], 7, format0)
                worksheet.set_column(col_dict[3], 6, format2)
                worksheet.set_column(col_dict[4], 7, format0)
                worksheet.set_column(col_dict[5], 6, format1)
                worksheet.freeze_panes('A4')

            else:

                workbook = writer.book
                worksheet = writer.sheets[ws_name]
                format0 = workbook.add_format({'num_format': '#0',
                                               'align': 'center'})

                worksheet.set_column('A:A', 10, None)
                worksheet.set_column(ann_col_dict[0], 7, format0)
                worksheet.freeze_panes('A2')


def retirement_charts(ds_dict,
                      adict, cdict,
                      plot_year_group=True,
                      date_grouper='ldate',
                      plot_job_group=True,
                      plot_init_quarter=True,
                      plot_running_quarter=True,
                      quantiles=10,
                      pcnt_ylim=.75,
                      cpay_stride=500,
                      fixed_col_name='eg_initQ',
                      running_col_name='eg_runQ',
                      figsize=None,
                      chartstyle='ticks',
                      verbose_status=True,
                      tick_size=13,
                      legend_size=14,
                      label_size=14,
                      title_size=14,
                      adjust_chart_top=.85):
    '''Generates multiple charts representing general attribute statistics
    of all calculated datasets for all employee groups AT RETIREMENT ONLY.

    The user may select grouping analysis by any or all of the following:

        1. longevity or date of hire year

        2. job level

        3. initial employee group list quantile membership

        4. annual employee group list quantile membership

    Stores the output as images in multiple folders within the
    **reports/<case_name>/ret_charts** folder.

    inputs
        ds_dict (dictionary)
            output of load_datasets function, a dictionary of datasets
        adict (dictionary)
            dataset column name description dictionary
        cdict (dictionary)
            program colors dictionary
        plot_year_group (boolean)
            if True, create chart images grouped by the date_grouper input
            year
        date_grouper (string)
            column name representing a column of dates within a dataframe.
            Year membership of this column will be used for grouping.
            Input is limited to 'ldate' or 'doh'.
        plot_job_group (boolean)
            if True,  create chart images grouped by job level held by
            employees
        quantiles (integer)
            the number of binning quantiles to measure for the initial and
            running (annually updated) quantile membership analysis
            (default is 10)
        plot_init_quarter (boolean)
            if True, produce output grouped by initial list quantile
            membership, for each employee group
        plot_running_quarter (boolean)
            if True, produce output grouped by annual list quantile
            membership, for each employee group
        pcnt_ylim (float)
            output chart maximum y axis value for percentage attribute charts
            as a float,  example: .75 equals max displayed chart value of 75%
        cpay_stride (integer)
            y axis chart tick interval (in thousands) for charts displaying
            cpay (career pay)
        fixed_col_name (string)
            label to use for quantile number column when calculating using
            the initial quantile membership for all results
        running_col_name (string)
            label to use for quantile number column when calculating using
            a continuously updated quantile membership for all results
        figsize (tuple)
            optional size of all generated chart images.  Default is None.
            This input will allow creation of larger chart images than the
            default small charts, at the price of an increase
            in the time required to run the function.
        date_grouper (string)
            'ldate' or 'doh' date column grouping attribute used when
            plot_year_group input is True
        chartstyle (string)
            any valid seaborn charting style ('ticks', 'dark', 'white',
            'darkgrid', 'whitegrid'), defalut is 'ticks'
        verbose_status (boolean)
            if True, print status of calculations as function is running
        tick_size (integer or float)
            text size of tick labels on the output chart images
        legend_size (integer or float)
            text size of the legend on the output chart images
        label_size (integer or float)
            text size of the x and y axis labels on the output chart images
        title_size (integer or float)
            text size of the title on the output chart images
        adjust_chart_top (float)
            input to permit adjustment of the top location of the generated
            charts - used to ensure full chart title is captured by the save
            chart figure code.  Defalt top position is 1.0, default vaule for
            this input is .85 which "shrinks" the charts slightly vertically
            so that the two-line chart titles are captured when saving the
            charts to file as images.
    '''
    try:
        # get current case study name
        case_df = pd.read_pickle('dill/case_dill.pkl')
        case_name = case_df.at['prop', 'case']
    except OSError:
        print('unable to retrieve case name, try setting case input')
        return
    attrs = ['spcnt', 'cat_order', 'jobp', 'cpay']
    pcnt_attributes = ['spcnt', 'lspcnt']
    job_attrs = ['jnum', 'jobp']
    eg_colors = cdict['eg_color_dict']
    im_prefix = 'reports/' + case_name + '/ret_charts/ret_charts_'

    # set directory names, make directories, set title prefixes
    if plot_year_group:

        date_grouper_string = adict[date_grouper]
        # set path to image directories
        long_image_dir = (im_prefix + date_grouper_string + '/')
        # make directory if it does not already exist
        if not path.exists(long_image_dir):
            makedirs(long_image_dir)
        long_title_pre = date_grouper_string + ' '

    if plot_job_group:

        job_image_dir = (im_prefix + 'init_job/')
        # make directory if it does not already exist
        if not path.exists(job_image_dir):
            makedirs(job_image_dir)
        job_title_pre = 'initial job '

    if plot_init_quarter:

        iq_image_dir = im_prefix + 'init_qntl/'
        # make directory if it does not already exist
        if not path.exists(iq_image_dir):
            makedirs(iq_image_dir)
        iq_title_pre = 'initial qntl '

    if plot_init_quarter:

        rq_image_dir = im_prefix + 'run_qntl/'
        # make directory if it does not already exist
        if not path.exists(rq_image_dir):
            makedirs(rq_image_dir)
        rq_title_pre = 'running_qntl '

    # remove the skeleton dataset from consideration for this function
    ds_list = [key for key in ds_dict.keys() if key != 'skeleton']

    # get any computed dataset to find intial chart values
    dummy_ds = ds_dict[ds_list[0]]
    # get a sorted array of unique employee group codes
    eg_nums = np.unique(dummy_ds.eg)
    # get the first and last data model years
    dt_min = min(dummy_ds.date.dt.year)
    dt_max = max(dummy_ds.date.dt.year)

    for attr in attrs:
        if verbose_status:
            print('\nPreparing', attr.upper(), 'charts...')
        # set chart style
        with sns.axes_style(chartstyle):
            # set global figure size (if figsize input is not None)
            if figsize is None:
                fig, ax = plt.subplots()
            else:
                fig, ax = plt.subplots(figsize=figsize)

        if attr not in pcnt_attributes:
            ylim = max(dummy_ds[attr]) * .7
        if attr in job_attrs:
            jlim = int(max(dummy_ds[attr]) + 1)
        if attr == 'cpay':
            ylim = max(dummy_ds[attr])
        ldict = {}
        # initialize chart lines for each employee group
        for eg_num in eg_nums:
            ldict[eg_num], = ax.plot([], [], eg_colors[eg_num],
                                     label='eg' + str(eg_num),
                                     marker='o', markersize=4)
        # format y axis tick labels, axis labels, and y axis limit
        if attr in pcnt_attributes:
            ax.yaxis.set_major_formatter(mp.pct_format())
        ax.set_xlabel('retirement year', fontsize=label_size)
        if attr == 'cpay':
            ax.set_ylabel(adict[attr] + ' ($K)', fontsize=label_size)
        else:
            ax.set_ylabel(adict[attr], fontsize=label_size)
        if attr in pcnt_attributes:
            ax.set_yticks(np.arange(0, pcnt_ylim, .05))
            for label in ax.yaxis.get_ticklabels()[1::2]:
                label.set_visible(False)
            ax.set_ylim(ymin=-.01, ymax=pcnt_ylim)
        elif attr in ['ylong']:
            ax.set_ylim(ymin=-.75, ymax=ylim)
        elif attr in job_attrs:
            ax.set_yticks(np.arange(1, jlim))
            ax.set_ylim(ymin=.75, ymax=ylim)
        elif attr in ['snum', 'cat_order']:
            ax.set_ylim(ymin=-100, ymax=ylim)
        elif attr in ['cpay']:
            ax.set_yticks(np.arange(0, ylim, cpay_stride))
            ax.set_ylim(ymin=-100, ymax=ylim)
        min_xtick_yr = dt_min // 10 * 10
        max_xtick_yr = dt_max // 10 * 10 + 10
        ax.set_xticks(np.arange(min_xtick_yr, max_xtick_yr, 5))
        ax.set_xlim(xmin=dt_min, xmax=dt_max)
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.grid(alpha=.6)
        # invert y axis if appropriate for attribute
        if attr not in ['cpay', 'ylong']:
            ax.invert_yaxis()
        ax.legend(loc=4, fontsize=legend_size)
        plt.tight_layout()
        fig.subplots_adjust(top=adjust_chart_top)

        # loop through each calculated dataset
        for key in ds_list:
            ds = ds_dict[key]
            # create a column assigning the initial model job for each
            # employee for all months
            f.make_starting_val_column(ds, 'jnum')
            # create employee group percentage columns, one fixed and the other
            # recalculated each month
            f.make_eg_pcnt_column(ds, fixed_col_name=fixed_col_name)
            f.make_eg_pcnt_column(ds, recalc_each_month=True,
                                  running_col_name=running_col_name)
            # make grouping values
            ds['yr'] = ds.date.dt.year
            # running quantile:
            ds['rq'] = ((ds[running_col_name] *
                         quantiles // 1) + 1).astype(int)
            # initial quantile:
            ds['iq'] = ((ds[fixed_col_name] * quantiles // 1) + 1).astype(int)
            # create a dataframe containing each employee in retirement month
            ret_df = ds[ds.ret_mark == 1]
            yr = ret_df.date.dt.year
            rq = ret_df.rq
            iq = ret_df.iq
            dateyr = ret_df[date_grouper].dt.year
            job = ret_df.start_jnum
            retyr = ret_df.retdate.dt.year

            if plot_year_group:
                # group by date grouper year (likely longevity date), data
                # model year, and employee group
                gb = ret_df.groupby([dateyr, retyr, 'eg'])[attr]
                # find attribute averages and unstack to create employee group
                # attribute value columns
                unstkd = gb.mean().unstack()
                # get a sorted array of unique data model years
                yrs = np.unique(unstkd.index.get_level_values(level=0))
                # loop through model years
                for year in yrs:
                    # reset the chart line data for the current data model year
                    # and employee group
                    for eg_num in eg_nums:
                        ldict[eg_num].set_data(unstkd.loc[year].index.values,
                                               unstkd.loc[year][eg_num].values)
                    # create the chart title
                    ax.set_title(long_title_pre + str(year) + '\n' +
                                 key + ' retirees',
                                 fontsize=title_size)
                    # draw the chart
                    fig.canvas.draw()
                    # save the chart
                    fig.savefig(path.join(long_image_dir,
                                          attr + '_' +
                                          str(year) + ' ' +
                                          date_grouper +
                                          '_' + key + '.png'))

            if plot_job_group:
                # group by initial job (from make starting value function),
                # data model year, and employee group
                gb = ret_df.groupby([job, retyr, 'eg'])[attr]
                # find attribute averages and unstack to create employee group
                # attribute value columns
                unstkd = gb.mean().unstack()
                # get a sorted array of unique data model years
                jobs = np.unique(unstkd.index.get_level_values(level=0))
                # loop through model years
                for job in jobs:
                    # reset the chart line data for the current data model job
                    # and employee group
                    for eg_num in eg_nums:
                        ldict[eg_num].set_data(unstkd.loc[job].index.values,
                                               unstkd.loc[job][eg_num].values)
                    # create the chart title
                    ax.set_title(job_title_pre + str(job) + '\n' +
                                 key + ' retirees',
                                 fontsize=title_size)
                    # draw the chart
                    fig.canvas.draw()
                    # save the chart
                    fig.savefig(path.join(job_image_dir,
                                          attr + '_' + 'job' +
                                          str(job) + '_' + key + '.png'))

            if plot_init_quarter:
                # group by initial quantile, data model year, and employee
                # group
                gb = ret_df.groupby([iq, yr, 'eg'])[attr]
                unstkd = gb.mean().unstack()
                # get a sorted array of quantiles
                qrtls = np.unique(unstkd.index.get_level_values(level=0))
                # loop through intial quantile groups
                for qt in qrtls:
                    # reset the chart line data for the data model
                    # initial quantile and employee group
                    for eg_num in eg_nums:
                        ldict[eg_num].set_data(unstkd.loc[qt].index.values,
                                               unstkd.loc[qt][eg_num].values)
                    # create the chart title
                    ax.set_title(iq_title_pre + str(qt) + '\n' +
                                 key + ' retirees',
                                 fontsize=title_size)
                    # draw the chart
                    fig.canvas.draw()
                    # save the chart
                    fig.savefig(path.join(iq_image_dir,
                                          attr + '_' + 'iq' +
                                          str(qt) + '_' + key + '.png'))

            if plot_running_quarter:
                # group by running quantile, data model year, and employee
                # group
                gb = ret_df.groupby([rq, yr, 'eg'])[attr]
                # get a sorted array of quantiles
                unstkd = gb.mean().unstack()
                # get a sorted array of unique data model years
                qrtls = np.unique(unstkd.index.get_level_values(level=0))
                # loop through running quantile groups
                for qt in qrtls:
                    # reset the chart line data for the current data
                    # running quantile and employee group
                    for eg_num in eg_nums:
                        ldict[eg_num].set_data(unstkd.loc[qt].index.values,
                                               unstkd.loc[qt][eg_num].values)
                    # create the chart title
                    ax.set_title(rq_title_pre + str(qt) + '\n' +
                                 key + ' retirees',
                                 fontsize=title_size)
                    # draw the chart
                    fig.canvas.draw()
                    # save the chart
                    fig.savefig(path.join(rq_image_dir,
                                          attr + '_' + 'rq' +
                                          str(qt) + '_' + key + '.png'))

            if verbose_status:
                print('  ', key, ' done!')

        plt.close()

    if verbose_status:
        print('\nALL JOBS COMPLETE\n\nChart images are located in the "' +
              case_name + '" folder within the "reports" folder.\n\n' +
              'Note: The charts produced from this routine are intended ' +
              'for basic overview of the calculated datasets only.\n' +
              'The built-in plotting functions provide much more detailed ' +
              'and comprehensive analysis of integrated list outcomes.\n')


def annual_charts(ds_dict,
                  adict, cdict,
                  plot_year_group=True,
                  plot_job_group=True,
                  quantiles=10,
                  plot_init_quarter=True,
                  plot_running_quarter=True,
                  pcnt_ylim=.75,
                  cpay_stride=500,
                  fixed_col_name='eg_initQ',
                  running_col_name='eg_runQ',
                  figsize=None,
                  date_grouper='ldate',
                  chartstyle='ticks',
                  verbose_status=True,
                  tick_size=13,
                  legend_size=14,
                  label_size=14,
                  title_size=14,
                  adjust_chart_top=.85):
    '''Generates multiple charts representing general annual attribute
    statistics of all calculated datasets for all employee groups FOR ALL
    ACTIVE EMPLOYEES (annual results for all employees).

    The user may select grouping analysis by any or all of the following:

        1. longevity or date of hire year

        2. job level

        3. initial employee group list quantile membership

        4. annual employee group list quantile membership

    Stores the output as images in multiple folders within the
    **reports/<case_name>/ann_charts** folder.

    inputs
        ds_dict (dictionary)
            output of load_datasets function, a dictionary of datasets
        adict (dictionary)
            dataset column name description dictionary
        cdict (dictionary)
            program colors dictionary
        plot_year_group (boolean)
            if True, create chart images grouped by the date_grouper input
            year
        date_grouper (string)
            column name representing a column of dates within a dataframe.
            Year membership of this column will be used for grouping.
            Input is limited to 'ldate' or 'doh'.
        plot_job_group (boolean)
            if True,  create chart images grouped by job level held by
            employees
        quantiles (integer)
            the number of binning quantiles to measure for the initial and
            running (annually updated) quantile membership analysis
            (default is 10)
        plot_init_quarter (boolean)
            if True, produce output grouped by initial list quantile
            membership, for each employee group
        plot_running_quarter (boolean)
            if True, produce output grouped by annual list quantile
            membership, for each employee group
        pcnt_ylim (float)
            output chart maximum y axis value for percentage attribute charts
            as a float,  example: .75 equals max displayed chart value of 75%
        cpay_stride (integer)
            y axis chart tick interval (in thousands) for charts displaying
            cpay (career pay)
        fixed_col_name (string)
            label to use for quantile number column when calculating using
            the initial quantile membership for all results
        running_col_name (string)
            label to use for quantile number column when calculating using
            a continuously updated quantile membership for all results
        figsize (tuple)
            optional size of all generated chart images.  Default is None.
            This input will allow creation of larger chart images than the
            default small charts, at the price of an increase
            in the time required to run the function.
        date_grouper (string)
            'ldate' or 'doh' date column grouping attribute used when
            plot_year_group input is True
        chartstyle (string)
            any valid seaborn charting style ('ticks', 'dark', 'white',
            'darkgrid', 'whitegrid'), defalut is 'ticks'
        verbose_status (boolean)
            if True, print status of calculations as function is running
        tick_size (integer or float)
            text size of tick labels on the output chart images
        legend_size (integer or float)
            text size of the legend on the output chart images
        label_size (integer or float)
            text size of the x and y axis labels on the output chart images
        title_size (integer or float)
            text size of the title on the output chart images
        adjust_chart_top (float)
            input to permit adjustment of the top location of the generated
            charts - used to ensure full chart title is captured by the save
            chart figure code.  Defalt top position is 1.0, default vaule for
            this input is .85 which "shrinks" the charts slightly vertically
            so that the two-line chart titles are captured when saving the
            charts to file as images.
    '''
    try:
        # get current case study name
        case_df = pd.read_pickle('dill/case_dill.pkl')
        case_name = case_df.at['prop', 'case']
    except OSError:
        print('unable to retrieve case name, try setting case input')
        return
    attrs = ['spcnt', 'cat_order', 'jobp', 'cpay']
    pcnt_attributes = ['spcnt', 'lspcnt']
    job_attrs = ['jnum', 'jobp']
    eg_colors = cdict['eg_color_dict']
    im_prefix = 'reports/' + case_name + '/annual_charts/ann_charts_'

    # set directory names, make directories, set title prefixes
    if plot_year_group:

        date_grouper_string = adict[date_grouper]
        # set path to image directories
        long_image_dir = (im_prefix + date_grouper_string + '/')
        # make directory if it does not already exist
        if not path.exists(long_image_dir):
            makedirs(long_image_dir)
        long_title_pre = date_grouper_string + ' '

    if plot_job_group:

        job_image_dir = (im_prefix + 'init_job/')
        # make directory if it does not already exist
        if not path.exists(job_image_dir):
            makedirs(job_image_dir)
        job_title_pre = 'initial job '

    if plot_init_quarter:

        iq_image_dir = im_prefix + 'init_qntl/'
        # make directory if it does not already exist
        if not path.exists(iq_image_dir):
            makedirs(iq_image_dir)
        iq_title_pre = 'initial qntl '

    if plot_init_quarter:

        rq_image_dir = im_prefix + 'run_qntl/'
        # make directory if it does not already exist
        if not path.exists(rq_image_dir):
            makedirs(rq_image_dir)
        rq_title_pre = 'running_qntl '

    # remove the skeleton dataset from consideration for this function
    ds_list = [key for key in ds_dict.keys() if key != 'skeleton']

    # get any computed dataset to find intial chart values
    dummy_ds = ds_dict[ds_list[0]]
    # get a sorted array of unique employee group codes
    eg_nums = np.unique(dummy_ds.eg)
    # get the first and last data model years
    dt_min = min(dummy_ds.date.dt.year)
    dt_max = max(dummy_ds.date.dt.year)

    for attr in attrs:
        if verbose_status:
            print('\nPreparing', attr.upper(), 'charts...')
        # set chart style
        with sns.axes_style(chartstyle):
            # set global figure size (if figsize input is not None)
            if figsize is None:
                fig, ax = plt.subplots()
            else:
                fig, ax = plt.subplots(figsize=figsize)

        if attr not in pcnt_attributes:
            ylim = max(dummy_ds[attr]) * .7
        if attr in job_attrs:
            jlim = int(max(dummy_ds[attr]) + 1)
        if attr == 'cpay':
            ylim = max(dummy_ds[attr])
        ldict = {}
        # initialize chart lines for each employee group
        for eg_num in eg_nums:
            ldict[eg_num], = ax.plot([], [], eg_colors[eg_num],
                                     label='eg' + str(eg_num),
                                     marker='o', markersize=4)
        # format y axis tick labels, axis labels, and y axis limit
        if attr in pcnt_attributes:
            ax.yaxis.set_major_formatter(mp.pct_format())
        ax.set_xlabel('year', fontsize=label_size)
        if attr == 'cpay':
            ax.set_ylabel(adict[attr] + ' ($K)', fontsize=label_size)
        else:
            ax.set_ylabel(adict[attr], fontsize=label_size)
        if attr in pcnt_attributes:
            ax.set_yticks(np.arange(0, pcnt_ylim, .05))
            for label in ax.yaxis.get_ticklabels()[1::2]:
                label.set_visible(False)
            ax.set_ylim(ymin=-.01, ymax=pcnt_ylim)
        elif attr in ['ylong']:
            ax.set_ylim(ymin=-.75, ymax=ylim)
        elif attr in ['jobp', 'jnum']:
            ax.set_yticks(np.arange(1, jlim))
            ax.set_ylim(ymin=.75, ymax=ylim)
        elif attr in ['snum', 'cat_order']:
            ax.set_ylim(ymin=-100, ymax=ylim)
        elif attr in ['cpay']:
            ax.set_yticks(np.arange(0, ylim, cpay_stride))
            ax.set_ylim(ymin=-100, ymax=ylim)
        min_xtick_yr = dt_min // 10 * 10
        max_xtick_yr = dt_max // 10 * 10 + 10
        ax.set_xticks(np.arange(min_xtick_yr, max_xtick_yr, 5))
        ax.set_xlim(xmin=dt_min, xmax=dt_max)
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.grid(alpha=.6)
        # invert y axis if appropriate for attribute
        if attr not in ['cpay', 'ylong']:
            ax.invert_yaxis()
        ax.legend(loc=4, fontsize=legend_size)
        plt.tight_layout()
        fig.subplots_adjust(top=adjust_chart_top)

        # loop through each calculated dataset
        for key in ds_list:
            ds = ds_dict[key]
            # create a column assigning the initial model job for each
            # employee for all months
            f.make_starting_val_column(ds, 'jnum')
            # create employee group percentage columns, one fixed and the other
            # recalculated each month
            f.make_eg_pcnt_column(ds, fixed_col_name=fixed_col_name)
            f.make_eg_pcnt_column(ds, recalc_each_month=True,
                                  running_col_name=running_col_name)
            # make grouping values
            yr = ds.date.dt.year
            rq = ((ds[running_col_name] * quantiles // 1) + 1).astype(int)
            iq = ((ds[fixed_col_name] * quantiles // 1) + 1).astype(int)

            dateyr = ds[date_grouper].dt.year
            job = ds.start_jnum
            # retyr = ds.retdate.dt.year

            if plot_year_group:
                # group by date grouper year (likely longevity date), data
                # model year, and employee group
                gb = ds.groupby([dateyr, yr, 'eg'])[attr]
                # find attribute averages and unstack to create employee group
                # attribute value columns
                unstkd = gb.mean().unstack()
                # get a sorted array of unique data model years
                yrs = np.unique(unstkd.index.get_level_values(level=0))
                # loop through model years
                for year in yrs:
                    # reset the chart line data for the current data model year
                    # and employee group
                    for eg_num in eg_nums:
                        ldict[eg_num].set_data(unstkd.loc[year].index.values,
                                               unstkd.loc[year][eg_num].values)
                    # create the chart title
                    ax.set_title(long_title_pre + str(year) + '\n' +
                                 key + ' actives',
                                 fontsize=title_size)
                    # draw the chart
                    fig.canvas.draw()
                    # save the chart
                    fig.savefig(path.join(long_image_dir,
                                          attr + '_' +
                                          str(year) + ' ' +
                                          date_grouper +
                                          '_' + key + '.png'))

            if plot_job_group:
                # group by initial job (from make starting value function),
                # data model year, and employee group
                gb = ds.groupby([job, yr, 'eg'])[attr]
                # find attribute averages and unstack to create employee group
                # attribute value columns
                unstkd = gb.mean().unstack()
                # get a sorted array of unique data model years
                jobs = np.unique(unstkd.index.get_level_values(level=0))
                # loop through model years
                for job in jobs:
                    # reset the chart line data for the current data model job
                    # and employee group
                    for eg_num in eg_nums:
                        ldict[eg_num].set_data(unstkd.loc[job].index.values,
                                               unstkd.loc[job][eg_num].values)
                    # create the chart title
                    ax.set_title(job_title_pre + str(job) + '\n' +
                                 key + ' actives',
                                 fontsize=title_size)
                    # draw the chart
                    fig.canvas.draw()
                    # save the chart
                    fig.savefig(path.join(job_image_dir,
                                          attr + '_' + 'job' +
                                          str(job) + '_' + key + '.png'))

            if plot_init_quarter:
                # group by initial quantile, data model year, and employee
                # group
                gb = ds.groupby([iq, yr, 'eg'])[attr]
                unstkd = gb.mean().unstack()
                # get a sorted array of quantiles
                qrtls = np.unique(unstkd.index.get_level_values(level=0))
                # loop through intial quantile groups
                for qt in qrtls:
                    # reset the chart line data for the data model
                    # initial quantile and employee group
                    for eg_num in eg_nums:
                        ldict[eg_num].set_data(unstkd.loc[qt].index.values,
                                               unstkd.loc[qt][eg_num].values)
                    # create the chart title
                    ax.set_title(iq_title_pre + str(qt) + '\n' +
                                 key + ' actives',
                                 fontsize=title_size)
                    # draw the chart
                    fig.canvas.draw()
                    # save the chart
                    fig.savefig(path.join(iq_image_dir,
                                          attr + '_' + 'iq' +
                                          str(qt) + '_' + key + '.png'))

            if plot_running_quarter:
                # group by running quantile, data model year, and employee
                # group
                gb = ds.groupby([rq, yr, 'eg'])[attr]
                # get a sorted array of quantiles
                unstkd = gb.mean().unstack()
                # get a sorted array of unique data model years
                qrtls = np.unique(unstkd.index.get_level_values(level=0))
                # loop through running quantile groups
                for qt in qrtls:
                    # reset the chart line data for the current data
                    # running quantile and employee group
                    for eg_num in eg_nums:
                        ldict[eg_num].set_data(unstkd.loc[qt].index.values,
                                               unstkd.loc[qt][eg_num].values)
                    # create the chart title
                    ax.set_title(rq_title_pre + str(qt) + '\n' +
                                 key + ' actives',
                                 fontsize=title_size)
                    # draw the chart
                    fig.canvas.draw()
                    # save the chart
                    fig.savefig(path.join(rq_image_dir,
                                          attr + '_' + 'rq' +
                                          str(qt) + '_' + key + '.png'))

            if verbose_status:
                print('  ', key, ' done!')

        plt.close()

    if verbose_status:
        print('\nALL JOBS COMPLETE\n\nChart images are located in the "' +
              case_name + '" folder within the "reports" folder.\n\n' +
              'Note: The charts produced from this routine are intended ' +
              'for basic overview of the calculated datasets only.\n' +
              'The built-in plotting functions provide much more detailed ' +
              'and comprehensive analysis of integrated list outcomes.\n')


def job_diff_to_excel(base_ds,
                      compare_ds,
                      ds_dict,
                      add_cpay=True,
                      diff_color=True,
                      row_color=True,
                      lighten_factor=.65,
                      neg_color='red',
                      pos_color='blue',
                      zero_color='white',
                      id_cols=['lname', 'ldate', 'retdate']):
    '''Generates a spreadsheet which reports the differential number of
    months spent at each job level between two outcome datasets.  Results
    are reported for every employee.

    The order of the employees shown will be the order from the "compare"
    dataset input.

    The user may choose to apply formatting to the output spreadsheet.  The
    generation of the output with formatting is much slower than without,
    however.

    Stores the output within the **reports/<case_name>/by_employee** folder.

    inputs
        base_ds (dataframe)
            baseline dataset
        compare_ds (dataframe)
            comparison dataset
        add_cpay (boolean)
            if True, add a "cpay_diff" column to show data model pay
            differential (compare vs. base)
        diff_color (boolean)
            if True, use the neg_color, pos_color, and zero_color inputs
            to color the spreadsheet job differential output
        row_color (boolean)
            color spreadsheet rows by employee group if True.  Color will
            be a tint (lighter color version) of the colors used to
            represent the employee groups in chart output.
        lighten_factor (float)
            when the "row_color" input is True, this input controls the
            tint of the normal employee group colors to use for the cell
            background row coloring.  The input is limited from 0.0 to 1.0
            and a higher value will make the coloring lighter.
        neg_color, pos_color, zero_color (color values)
            this input will determine the font colors to use for negative,
            positive, and zero job differential values within the spreadsheet
            output.  Inputs may by string hex values, or rgb values within
            tuples or lists
        id_cols (list)
            list of columns to include within the spreadsheet output which
            are in addition to the job level columns.  This list (with the
            addition of the "order" column) will also be colored according
            to employee group when the "row_color" input is set to True.
            '''

    sdict = pd.read_pickle('dill/dict_settings.pkl')
    hex_dict = f.hex_dict()
    if compare_ds == 'edit':
        order_col = 'idx'
    else:
        order_col = 'new_order'

    def lighten(color, hex_dict, factor=.8):
        '''This function will return a lightened color.  The "factor" input
        determines how much lighter.  The "color" input may be a string or
        rgb input.
        '''
        try:
            if type(color) is str:
                if color.startswith('#'):
                    h = color.lstrip('#')
                else:
                    h = hex_dict[color].lstrip('#')

                rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
                list_rgb = list(rgb)
                lighter = (((255 - np.array(list_rgb)) * factor) +
                           list_rgb).astype(int)
                hexcolor = '#%02x%02x%02x' % tuple(lighter)
                return hexcolor

            if type(color) is tuple:
                list_rgb = list(color)[:3]
            if type(color) is list:
                list_rgb = color[:3]

            lighter = (((255 - np.array(list_rgb)) * factor) +
                       list_rgb).astype(int)
            return type(color)(lighter)
        except:
            print('''invalid color input for "lighten" function,
                  must be an RGB tuple, RGB list, or string hex value''')

    def cpay_diff(base, compare):
        base_cpay = base[base.ret_mark == 1][['cpay']].copy()
        compare_cpay = compare[compare.ret_mark == 1][['cpay']].copy()
        compare_cpay['base'] = base_cpay['cpay']
        compare_cpay['cpay_diff'] = \
            ((compare_cpay.cpay - compare_cpay.base) * 1000).astype(int)
        return compare_cpay[['cpay_diff']]

    def job_months(df):
        job_months = df.groupby([df.empkey, 'jnum'])['jnum']. \
            count().unstack().fillna(0)
        return job_months.astype(int)

    def fill_job_columns(df, sdict):
        num_of_job_levels = sdict['num_of_job_levels']
        job_levels = list(range(1, num_of_job_levels + 2))

        for level in job_levels:
            if level not in df.columns:
                df[level] = 0

        df.columns = sorted(df.columns)

        return df

    def resort_df(df, order_df):
        m0 = order_df[order_df.mnum == 0][[]].copy()
        m0[order_col] = range(1, len(m0) + 1)
        df['order'] = m0[order_col]
        df.sort_values('order', inplace=True)
        df.drop('order', inplace=True, axis=1)
        return df.copy()

    def add_cols(df, order_df, col_list=id_cols):
        cols = [order_col]
        if 'eg' not in col_list:
            col_list.append('eg')
        m0 = order_df[order_df.mnum == 0][col_list].copy()
        m0[order_col] = range(1, len(m0) + 1)
        cols.extend(col_list)
        m0 = m0[cols].copy()
        m0 = m0.join(df.copy())
        for col in col_list:
            try:
                m0[col] = m0[col].dt.date
            except:
                pass
        return m0.copy()

    def color_vals(val):
        """
        Takes a scalar and returns a string with
        the css property 'color: red' for negative
        strings, 'color: blue' for positive numbers
        and 'color: white' for zero values.
        """
        color = ((pos_color, neg_color), (zero_color, ))[val == 0][val < 0]
        return 'color: %s' % color

    def color_rows(df, hex_dict, light_factor=.85):
        sdict = pd.read_pickle('dill/dict_color.pkl')
        eg_dict = sdict['eg_color_dict']
        for key in eg_dict.keys():
            eg_dict[key] = lighten(eg_dict[key], hex_dict, factor=light_factor)
        prefix = 'background-color: '
        egs = df.eg.values
        color_strs = np.empty(egs.size, dtype=object)
        for eg in np.unique(egs):
            np.put(color_strs, np.where(egs == eg)[0], prefix + eg_dict[eg])
        return color_strs

    try:
        # get current case study name
        case_df = pd.read_pickle('dill/case_dill.pkl')
        case_name = case_df.at['prop', 'case']
    except OSError:
        print('unable to retrieve case name, try setting case input')
        return

    # make a folder for the report if it does not exist
    folder = 'reports/' + case_name + '/by_employee/'
    if not path.exists(folder):
        makedirs(folder)

    # assign the output spreadsheet name and path
    file_name = 'job_months_' + compare_ds + '_vs_' + base_ds + '.xlsx'
    path_name = folder + file_name

    # get the base and compare dataframes from the dataset dictionary
    base = ds_dict[base_ds]
    compare = ds_dict[compare_ds]

    # make the groupby job count dataframes
    base_jobs = job_months(base)
    compare_jobs = job_months(compare)

    # add any missing job level columns
    base_jobs_filled = fill_job_columns(base_jobs, sdict)
    compare_jobs_filled = fill_job_columns(compare_jobs, sdict)

    # sort both datasets by compare dataset order
    base_sorted = resort_df(base_jobs_filled, compare)
    compare_sorted = resort_df(compare_jobs_filled, compare)

    # create the differential dataframe
    df_diff = compare_sorted - base_sorted

    # add other columns to the differential dataframe (name, ldate, etc.)
    final = add_cols(df_diff, compare)

    # add a career pay differential column to the final dataframe
    if add_cpay:
        cpay = cpay_diff(base, compare)
        final = final.join(cpay)

    # define the row color function
    apply_row_color = color_rows(final, hex_dict, light_factor=lighten_factor)

    job_levels = list(range(1, sdict['num_of_job_levels'] + 2))

    id_cols.append(order_col)

    # allow the career pay difference column to have color formatting
    if add_cpay:
        job_levels.append('cpay_diff')

    # determine the formatting to be applied
    if diff_color and row_color:
        frame = final.style.apply(lambda s: apply_row_color,
                                  subset=id_cols).applymap(color_vals,
                                                           subset=job_levels)
    elif diff_color and not row_color:
        frame = final.style.applymap(color_vals, subset=job_levels)
    elif not diff_color and row_color:
        frame = final.style.apply(lambda s: apply_row_color, subset=id_cols)
    else:
        frame = final

    # write the spreadsheet file to disk
    frame.to_excel(path_name, engine='openpyxl', freeze_panes=(1, 0))

