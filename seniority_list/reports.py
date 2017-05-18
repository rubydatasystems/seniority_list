#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''This module builds general statistical reports for all of the program
datasets and presents the results as spreadsheets and chart images within
the *reports* folder.
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import functions as f
import matplotlib_charting as mp
from os import path, makedirs


def stats_to_excel(ds_dict,
                   fixed_col_name='strt_quar',
                   running_col_name='cur_quar'):
    '''Create a set of basic statistics for each calculated dataset and
    write the results as spreadsheets within the **reports** folder.

    There are 2 spreadsheets produced, one related to retirement data and the
    other related to annual data.annual

    The retirement information is grouped by employees retiring in future
    years, further grouped for longevity or initial job.

    The annual information is grouped by the model year, and further grouped
    by 10% quartiles, either by initial quartile membership or by an annual
    quartile adjustment of remaining employees.

    inputs
        ds_dict (dictionary)
            output of load_datasets function, a dictionary of datasets
        fixed_col_name (string)
            label to use for quartile number column when calculating using
            the initial quartile membership for all results
        running_col_name (string)
            label to use for quartile number column when calculating using
            a continuously updated quartile membership for all results
    '''
    ret_dict = {}
    ann_dict = {}
    try:
        # get current case study name
        case_df = pd.read_pickle('dill/case_dill.pkl')
        case_name = case_df.case.value
    except OSError:
        print('unable to retrieve case name, try setting case input')
        return
    # define attributes for plotting
    attrs = ['spcnt', 'snum', 'cat_order', 'jobp', 'cpay', 'ylong']
    # remove the skeleton dataset from consideration for this function
    ds_list = [key for key in ds_dict.keys() if key != 'skeleton']

    # loop through datasets
    for key in ds_list:

        # copy to avoid altering ds_dict values
        p = ds_dict[key].copy()
        # rename columns for excel column appearance
        p.rename(columns={'ldate': 'longevity',
                          'retdate': 'retire',
                          'jnum': 'job'}, inplace=True)
        # create employee group percentage columns, one fixed and the other
        # recalculated each month
        f.make_eg_pcnt_column(p, fixed_col_name=fixed_col_name)
        f.make_eg_pcnt_column(p, recalc_each_month=True,
                              running_col_name=running_col_name)
        # create a column assigning the initial model job for each
        # employee for all months
        f.make_starting_val_column(p, 'job')
        # make grouping values
        yr = p.date.dt.year
        mthq = ((p[running_col_name] * 1000 // 100) + 1).astype(int)
        smthq = ((p[fixed_col_name] * 1000 // 100) + 1).astype(int)

        retp = p[p.ret_mark == 1]
        lyr = retp.longevity.dt.year
        job = retp.start_job
        retyr = retp.retire.dt.year
        # add grouped dataframes to dictionaries
        try:
            ret_dict['longevity' + '_' + key] = \
                retp.groupby([lyr, retyr, 'eg'])[attrs].mean().unstack()
            ret_dict['job' + '_' + key] = \
                retp.groupby([job, retyr, 'eg'])[attrs].mean().unstack()
            ann_dict['A_' + key] = \
                p.groupby([yr, 'eg'])[attrs].mean().unstack()
            ann_dict['RQ_' + key] = \
                p.groupby([yr, mthq, 'eg'])[attrs].mean().unstack()
            ann_dict['IQ_' + key] = \
                p.groupby([yr, smthq, 'eg'])[attrs].mean().unstack()
        except:
            print(key, 'report fail')

    # make a count worksheet
    for key in ds_dict.keys():
        if key != 'skeleton':
            found_key = key
            break
        else:
            continue
    df = ds_dict[found_key].copy()
    p = df[df.ret_mark == 1]
    yr = p.date.dt.year
    lyr = p.ldate.dt.year
    retyr = p.retdate.dt.year

    ret_dict['retirement_count'] = p.groupby([lyr, retyr, 'eg'])['mnum'] \
        .count().unstack().fillna(0).astype(int)
    ann_dict['retirement_count'] = p.groupby([retyr, 'eg'])['mnum'] \
        .count().unstack().fillna(0).astype(int)

    # write grouped retirement dataframes to excel workbook
    with pd.ExcelWriter('reports/' + case_name + '/stats_retirement.xlsx',
                        engine='xlsxwriter',
                        datetime_format='yyyy-mm-dd',
                        date_format='yyyy-mm-dd') as writer:

        for ws_name, df_sheet in sorted(ret_dict.items()):
            df_sheet.to_excel(writer, sheet_name=ws_name)

    # write grouped annual dataframes to excel workbook
    with pd.ExcelWriter('reports/' + case_name + '/stats_annual.xlsx',
                        engine='xlsxwriter',
                        datetime_format='yyyy-mm-dd',
                        date_format='yyyy-mm-dd') as writer:

        for ws_name, df_sheet in sorted(ann_dict.items()):
            df_sheet.to_excel(writer, sheet_name=ws_name)


def ret_stats_charts(ds_dict,
                     adict, cdict,
                     plot_longevity=True,
                     plot_job=True,
                     figsize=None,
                     date_grouper='ldate',
                     chartstyle='ticks',
                     verbose_status=True,
                     adjust_chart_top=.85):
    '''Generates multiple charts representing general attribute statistics of
    all calculated datasets for all employee groups.  Stores the output as
    images in multiple folders within the **reports** folder.
    '''
    try:
        # get current case study name
        case_df = pd.read_pickle('dill/case_dill.pkl')
        case_name = case_df.case.value
    except OSError:
        print('unable to retrieve case name, try setting case input')
        return
    ret_attrs = ['spcnt', 'cat_order', 'jobp', 'cpay']
    pcnt_attributes = ['spcnt', 'lspcnt']
    eg_colors = cdict['eg_color_dict']
    # remove the skeleton dataset from consideration for this function
    ds_list = [key for key in ds_dict.keys() if key != 'skeleton']

    # get any computed dataset to find intial chart values
    dummy_ds = ds_dict[ds_list[0]]
    # get a sorted array of unique employee group codes
    eg_nums = np.unique(dummy_ds.eg)
    # get the first and last data model years
    dt_min = min(dummy_ds.date.dt.year)
    dt_max = max(dummy_ds.date.dt.year)
    x = np.arange(dt_min, dt_max + 1)

    for attr in ret_attrs:
        # set chart style
        with sns.axes_style(chartstyle):
            # set global figure size (if figsize input is not None)
            if figsize is None:
                fig, ax = plt.subplots()
            else:
                fig, ax = plt.subplots(figsize=figsize)
        # set up y axis dummy data (sets intitial axis ranges)
        if attr in pcnt_attributes:
            y = np.linspace(0, .75, len(x))
        else:
            ylim = max(dummy_ds[attr]) * .7
            y = np.linspace(0, ylim, len(x))
        ldict = {}
        # initialize chart lines for each employee group
        for eg_num in eg_nums:
            ldict[eg_num], = ax.plot(x, y, eg_colors[eg_num],
                                     label='eg' + str(eg_num),
                                     marker='o', markersize=4)
        # format y axis tick labels, axis labels, and y axis limit
        if attr in pcnt_attributes:
            ax.yaxis.set_major_formatter(mp.pct_format())
        ax.set_xlabel('retirement year', fontsize=14)
        if attr == 'cpay':
            ax.set_ylabel(adict[attr] + ' ($K)', fontsize=14)
        else:
            ax.set_ylabel(adict[attr], fontsize=14)
        if attr in pcnt_attributes:
            ax.set_yticks(np.arange(0, .75, .05))
            for label in ax.yaxis.get_ticklabels()[1::2]:
                label.set_visible(False)
            ax.set_ylim(ymin=-.01, ymax=.75)
        elif attr in ['ylong']:
            ax.set_ylim(ymin=-.75, ymax=ylim)
        elif attr in ['jobp', 'jnum']:
            ax.set_yticks(np.arange(1, int(max(y) + 1)))
            ax.set_ylim(ymin=.75, ymax=ylim)
        elif attr in ['snum', 'cat_order']:
            ax.set_ylim(ymin=-100, ymax=ylim)
        elif attr in ['cpay']:
            ax.set_yticks(np.arange(0, ylim + 500, 500))
            ax.set_ylim(ymin=-100, ymax=ylim)
        min_xtick = dt_min // 10 * 10
        max_xtick = dt_max // 10 * 10 + 10
        ax.set_xticks(np.arange(min_xtick, max_xtick, 5))
        ax.set_xlim(xmin=dt_min, xmax=dt_max)
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(alpha=.6)
        # invert y axis if appropriate for attribute
        if attr not in ['cpay', 'ylong']:
            ax.invert_yaxis()
        ax.legend(loc=4, fontsize=14)
        plt.tight_layout()
        fig.subplots_adjust(top=adjust_chart_top)

        if plot_longevity:

            date_grouper_string = adict[date_grouper]
            # set path to image directories
            long_image_dir = ('reports/' + case_name + '/ret_images_' +
                              date_grouper_string + '/')
            # make directory if it does not already exist
            if not path.exists(long_image_dir):
                makedirs(long_image_dir)
            title_pre = date_grouper_string + ' '

            if verbose_status:
                print('\nworking:',
                      'Grouping |' + adict[date_grouper].title() + '|\n',
                      '        Attribute: < ' + attr.upper() + ' >')
            # loop through each calculated dataset
            for key in ds_list:
                p = ds_dict[key]
                # only include employees in last working month
                # (retirement month)
                ret_df = p[p.ret_mark == 1]
                # group by date grouper year (likely longevity date), data
                # model year, and employee group
                gb = ret_df.groupby([ret_df[date_grouper].dt.year,
                                    ret_df.retdate.dt.year, 'eg'])[attr]
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
                    ax.set_title(title_pre + str(year) + '\n' + key,
                                 fontsize=14)
                    # draw the chart
                    fig.canvas.draw()
                    # save the chart
                    fig.savefig(path.join(long_image_dir,
                                          attr + '_' +
                                          str(year) + ' ' +
                                          date_grouper +
                                          '_' + key + '.png'))
                if verbose_status:
                    print('  ', key, ' done!')

        if plot_job:

            job_image_dir = 'reports/' + case_name + '/ret_images_init_job/'
            # make directory if it does not already exist
            if not path.exists(job_image_dir):
                makedirs(job_image_dir)
            title_pre = 'initial job '

            if verbose_status:
                print('\nworking:', 'Grouping |Init Job|\n',
                      '        Attribute: < ' + attr.upper() + ' >')
            # loop through each calculated dataset
            for key in ds_list:
                p = ds_dict[key]
                # create a column assigning the initial model job for each
                # employee for all months
                f.make_starting_val_column(p, 'jnum')
                # only include employees in last working month
                # (retirement month)
                ret_df = p[p.ret_mark == 1]

                job = ret_df.start_jnum
                retyr = ret_df.retdate.dt.year
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
                    ax.set_title(title_pre + str(job) + '\n' + key,
                                 fontsize=14)
                    # draw the chart
                    fig.canvas.draw()
                    # save the chart
                    fig.savefig(path.join(job_image_dir,
                                          attr + '_' + 'job' +
                                          str(job) + '_' + key + '.png'))
                if verbose_status:
                    print('  ', key, ' done!')

    if verbose_status:
        print('\nALL JOBS COMPLETE\n\nChart images are located in the "' +
              case_name + '" folder within the "reports" folder.\n\n' +
              'Note: The charts produced from this routine are intended ' +
              'for basic overview of the calculated datasets only.\n' +
              'The built-in plotting functions provide much more detailed ' +
              'and comprehensive analysis of integrated list outcomes.\n')
