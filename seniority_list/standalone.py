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

'''create the standalone dataset
output is a single dataset containing independent results for each
employee group
'prex' may be included as a condition argument if the dataset should be
constructed with pre-existing job rights conditions
'''

import os
import pandas as pd
import numpy as np

import functions as f

from sys import argv, exit


def main():

    script, *conditions = argv

    input_skel = 'skeleton'

    pre, suf = 'dill/', '.pkl'

    skeleton_path_string = (pre + input_skel + suf)

    try:
        ds = pd.read_pickle(skeleton_path_string)
    except OSError:
        print('\nSkeleton file not found. ' +
              'Run build_program_files script?\n\n' +
              'Standalone build failed.\n\n' +
              '  >>> exiting routine.\n')
        exit()

    if os.path.isdir('dill/'):
        try:
            os.remove('dill/standalone.pkl')
        except OSError:
            pass

    sdict = pd.read_pickle('dill/dict_settings.pkl')
    tdict = pd.read_pickle('dill/dict_job_tables.pkl')

    num_of_job_levels = sdict['num_of_job_levels']
    egs = np.unique(ds.eg).astype(int)
    start_month = 0

    # make prex True or False
    # (for input to assign_standalone_job_changes function)
    prex = 'prex' in conditions

    table = tdict['s_table']
    jcnts_arr = tdict['jcnts_arr']
    j_changes = tdict['j_changes']

    job_change_months = f.get_job_change_months(j_changes)
    job_reduction_months = f.get_job_reduction_months(j_changes)

    # sort the skeleton by employee group, month, and index
    # (preserves each group's list order)
    ds.sort_values(['eg', 'mnum', 'idx'])

    ds_dict = {}
    short_ds_dict = {}

    for grp in egs:
        ds_dict[grp] = ds[ds.eg == grp].copy()

    for grp in egs:
        short_ds_dict[grp] = ds_dict[grp][ds_dict[grp].mnum == 0].copy()

    ds = pd.DataFrame()

    for eg in egs:

        df_long = ds_dict[eg]
        df_short = short_ds_dict[eg]
        jcnts = jcnts_arr[0][eg - 1]
        short_len = len(df_short)

        # ORIG_JOB*
        cmonths_this_ds = \
            f.career_months(df_short, sdict['starting_date'])
        this_ds_nonret_each_month = f.count_per_month(cmonths_this_ds)
        high_limits = this_ds_nonret_each_month.cumsum()
        low_limits = f.make_lower_slice_limits(high_limits)
        all_months = np.sum(this_ds_nonret_each_month)

        this_eg_table = f.add_zero_col(table[0][eg - 1])
        this_eg_month_counts = table[1][eg - 1]

        df_align_cols = ['fur']
        if 'sg' in df_long:
            df_align_cols.append('sg')

        df_align = df_long[df_align_cols]

        # pre-existing employee group special job assignment is included within
        # the job assignment function below...
        results = f.assign_standalone_job_changes(eg,
                                                  df_align,
                                                  low_limits,
                                                  high_limits,
                                                  all_months,
                                                  this_eg_table,
                                                  this_eg_month_counts,
                                                  this_ds_nonret_each_month,
                                                  job_change_months,
                                                  job_reduction_months,
                                                  start_month,
                                                  sdict,
                                                  tdict,
                                                  apply_sg_cond=prex)

        jnums = results[0]
        count_col = results[1]
        held = results[2]
        fur = results[3]
        orig_jobs = results[4]
        # HELD JOB
        # job from previous month
        df_long['held'] = held
        # JOB_COUNT
        df_long['job_count'] = count_col

        # ORIG_JOB
        df_short['orig_job'] = orig_jobs
        df_long['orig_job'] = df_short['orig_job']

        # ASSIGN JOBS - (stovepipe method only since only
        # assigning within each employee group separately)

        # JNUM

        df_long['jnum'] = jnums

        # SNUM, SPCNT, LNUM, LSPCNT

        monthly_job_counts = table[1][eg - 1]
        lspcnt_calc = sdict['lspcnt_calc_on_remaining_population']

        df_long['snum'], df_long['spcnt'], \
            df_long['lnum'], df_long['lspcnt'] = \
            f.create_snum_and_spcnt_arrays(jnums, num_of_job_levels,
                                           this_ds_nonret_each_month,
                                           monthly_job_counts,
                                           lspcnt_calc)

        # RANK in JOB

        df_long['rank_in_job'] = \
            df_long.groupby(['mnum', 'jnum']).cumcount() + 1

        # JOBP
        # make last percentage position in each job category .99999 vs 1.0
        # so that jobp calculations are correct
        jpcnt = (df_long.rank_in_job / df_long.job_count).values
        np.put(jpcnt, np.where(jpcnt == 1.0)[0], .99999)

        df_long['jobp'] = df_long['jnum'] + jpcnt

        # PAY - merge with pay table - provides monthly pay
        if sdict['compute_pay_measures']:

            if sdict['discount_longev_for_fur']:
                # skel provides non-discounted scale data
                # flip ones and zeros...
                df_long['non_fur'] = 1 - fur
                df_long['fur'] = fur

                non_fur = \
                    (df_long.groupby([pd.Grouper('empkey')])
                     ['non_fur'].cumsum().values)
                df_long.pop('non_fur')
                starting_mlong = df_long.s_lmonths.values
                cum_active_months = non_fur + starting_mlong
                df_long['mlong'] = cum_active_months
                df_long['ylong'] = df_long['mlong'] / 12
                df_long['scale'] = \
                    np.clip((cum_active_months / 12) + 1, 1,
                            sdict['top_of_scale']).astype(int)

            # SCALE
            df_pt_index = pd.DataFrame(
                index=(df_long['scale'] * 100) + df_long['jnum'] +
                (df_long['year'] * 100000))

            if sdict['enhanced_jobs']:
                df_pt = pd.read_pickle(
                    'dill/pay_table_enhanced.pkl')
            else:
                df_pt = pd.read_pickle(
                    'dill/pay_table_basic.pkl')

            df_pt_index['monthly'] = df_pt['monthly']

            df_long['monthly'] = df_pt_index['monthly'].values

            # MPAY
            # adjust monthly pay for any raise and last month pay percent if
            # applicable
            df_long['mpay'] = (
                (df_long['pay_raise'] *
                 df_long['mth_pcnt'] *
                 df_long['monthly'])) / 1000

            # print(df_long['pay_raise'])
            # print(df_long['monthly'])

            df_long.pop('monthly')

            # CPAY

            df_long['cpay'] = df_long.groupby('idx')['mpay'].cumsum()

        ds = pd.concat([ds, df_long], ignore_index=True)

    ds.sort_values(by=['mnum', 'idx'], inplace=True)
    ds.set_index('empkey', drop=False, verify_integrity=False, inplace=True)

    # CAT_ORDER
    # global job ranking

    if sdict['compute_job_category_order']:
        table = tdict['table']
        ds['cat_order'] = f.make_cat_order(ds, table[0])

    # save to file
    if sdict['save_to_pickle']:
        ds.to_pickle('dill/standalone.pkl', protocol=4)


if __name__ == "__main__":
    main()
