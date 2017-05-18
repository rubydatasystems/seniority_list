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

    # loop through datasets
    for key in ds_dict.keys():
        # exclude skeleton (not a complete calculated dataset)
        if key not in ['skeleton']:
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


def stats_to_charts(ds_dict, adict, cdict):
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
    ret_attrs = ['snum', 'spcnt', 'cat_order', 'jobp', 'cpay']
    attrs = ['snum', 'spcnt', 'cat_order', 'jobp', 'cpay', 'ylong']
    pcnt_attributes = ['spcnt', 'lspcnt']
    eg_colors = cdict['eg_color_dict']
    date_attr = 'ldate'
    #attr = 'spcnt'
    ds_list = [key for key in ds_dict.keys() if key != 'skeleton']

    title_pre = adict[date_attr] + ' '

    dummy_ds = ds_dict[ds_list[0]]
    eg_nums = np.unique(dummy_ds.eg)
    dt_min = min(dummy_ds.date.dt.year)
    dt_max = max(dummy_ds.date.dt.year)
    x = np.arange(dt_min, dt_max + 1)
    for attr in ret_attrs:
        with sns.axes_style('ticks'):
            fig, ax = plt.subplots(figsize=(8, 6))
        if attr in pcnt_attributes:
            y = np.linspace(0, .8, len(x))
        else:
            y = np.linspace(0, max(dummy_ds[attr]) * .8, len(x))
        ldict = {}
        for eg_num in eg_nums:
            ldict[eg_num], = ax.plot(x, y, eg_colors[eg_num],
                                     label='eg' + str(eg_num))
        if attr in pcnt_attributes:
            ax.yaxis.set_major_formatter(mp.pct_format())
        ax.set_xlabel('retirement year', fontsize=14)
        ax.set_ylabel(adict[attr], fontsize=14)
        if attr in pcnt_attributes:
            ax.set_ylim(ymin=-.01)
        elif attr in ['jobp', 'ylong']:
            ax.set_ylim(ymin=.75)
        elif attr in ['snum', 'cat_order']:
            ax.set_ylim(ymin=200)

        ax.tick_params(axis='both', labelsize=13)
        ax.grid(alpha=.6)
        if attr not in ['cpay', 'ylong']:
            ax.invert_yaxis()
        ax.legend(loc=4, fontsize=14)

        ret_image_dir = 'reports/' + case_name + '/ret_images/'

        if not path.exists(ret_image_dir):
            makedirs(ret_image_dir)
        for key in ds_list:

            p = ds_dict[key]
            ret_df = p[p.ret_mark == 1]
            gb = ret_df.groupby([ret_df[date_attr].dt.year,
                                ret_df.retdate.dt.year, 'eg'])[attr]
            unstkd = gb.mean().unstack()
            yrs = np.unique(unstkd.index.get_level_values(level=0))
            for year in yrs:
                for eg_num in eg_nums:
                    ldict[eg_num].set_data(unstkd.loc[year].index.values,
                                           unstkd.loc[year][eg_num].values)
                ax.set_title(title_pre + str(year) + '\n' + key, fontsize=14)
                fig.canvas.draw()

                fig.savefig(path.join(ret_image_dir,
                                      attr + '_' +
                                      str(year) + '_' + key + '.png'))

