#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''This module builds general statistical reports for all of the program
datasets and presents the results as spreadsheets and chart images within
the *reports* folder.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as f


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
    attrs = ['spcnt', 'snum', 'cat_order', 'jobp', 'cpay', 'ylong']
    for key in ds_dict.keys():
        if key not in ['skeleton']:
            p = ds_dict[key]
            p.rename(columns={'ldate': 'longevity',
                              'retdate': 'retire',
                              'jnum': 'job'}, inplace=True)
            f.make_eg_pcnt_column(p, fixed_col_name=fixed_col_name)
            f.make_eg_pcnt_column(p, recalc_each_month=True,
                                  running_col_name=running_col_name)
            f.make_starting_val_column(p, 'job')
            yr = p.date.dt.year
            lyr = p.longevity.dt.year
            job = p.start_job
            mthq = ((p[running_col_name] * 1000 // 100) + 1).astype(int)
            smthq = ((p[fixed_col_name] * 1000 // 100) + 1).astype(int)
            retyr = p.retire.dt.year
            try:
                ret_dict[key + '_' + 'longevity'] = \
                    p.groupby([lyr, retyr, 'eg'])[attrs].mean().unstack()
                ret_dict[key + '_' + 'job'] = \
                    p.groupby([job, retyr, 'eg'])[attrs].mean().unstack()
                ann_dict[key] = \
                    p.groupby([yr, 'eg'])[attrs].mean().unstack()
                ann_dict[key + '_' + 'RQ'] = \
                    p.groupby([yr, mthq, 'eg'])[attrs].mean().unstack()
                ann_dict[key + '_' + 'IQ'] = \
                    p.groupby([yr, smthq, 'eg'])[attrs].mean().unstack()
            except:
                print(key, 'report fail')

    with pd.ExcelWriter('reports/' + case_name + '/retirement_stats.xlsx',
                        engine='xlsxwriter',
                        datetime_format='yyyy-mm-dd',
                        date_format='yyyy-mm-dd') as writer:

        for ws_name, df_sheet in sorted(ret_dict.items()):
            df_sheet.to_excel(writer, sheet_name=ws_name)

    with pd.ExcelWriter('reports/' + case_name + '/annual_stats.xlsx',
                        engine='xlsxwriter',
                        datetime_format='yyyy-mm-dd',
                        date_format='yyyy-mm-dd') as writer:

        for ws_name, df_sheet in sorted(ann_dict.items()):
            df_sheet.to_excel(writer, sheet_name=ws_name)


def stats_to_charts(ds_dict):
    pass
