#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import functions as f
import config as cf

from sys import argv

script, *conditions = argv

input_skel = 'skel'

pre, suf = 'dill/', '.pkl'

skeleton_path_string = (pre + input_skel + suf)

ds = pd.read_pickle(skeleton_path_string)

num_of_job_levels = cf.num_of_job_levels
fur_counts = cf.furlough_count
num_of_months = pd.unique(ds.mnum).size
egs = pd.unique(ds.eg)
start_month = 0

# if cf.actives_only:
#     ds = ds[ds.fur == 0].copy()

if cf.enhanced_jobs:
    eg_counts = f.convert_jcnts_to_enhanced(cf.eg_counts,
                                            cf.full_time_pcnt1,
                                            cf.full_time_pcnt2)
    j_changes = f.convert_job_changes_to_enhanced(cf.j_changes, cf.jd)
else:
    eg_counts = cf.eg_counts
    j_changes = cf.j_changes

jcnts_arr = f.make_jcnts(eg_counts)

table = f.job_gain_loss_table(num_of_months,
                              num_of_job_levels,
                              jcnts_arr,
                              j_changes,
                              standalone=True)

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

for i in egs - 1:

    df_long = ds_dict[i + 1]
    df_short = short_ds_dict[i + 1]
    jcnts = jcnts_arr[0][i]
    short_len = len(df_short)

    # ORIG_JOB*
    cmonths_this_ds = f.career_months_df_in(df_short)
    this_ds_nonret_each_month = f.count_per_month(cmonths_this_ds)
    uppers = this_ds_nonret_each_month.cumsum()
    lowers = f.make_lower_slice_limits(uppers)
    all_months = np.sum(this_ds_nonret_each_month)

    this_table = table[0][i]
    this_month_counts = table[1][i]

    df_align_cols = ['fur']
    if 'sg' in df_long:
        df_align_cols.append('sg')

    df_align = df_long[df_align_cols]
    fur_codes = np.array(df_align.fur)

    # pre-existing employee group special job assignment is included within
    # the job assignment function below...
    if cf.compute_with_job_changes:

        results = f.assign_standalone_job_changes(df_align,
                                                  lowers,
                                                  uppers,
                                                  all_months,
                                                  this_table,
                                                  this_month_counts,
                                                  this_ds_nonret_each_month,
                                                  job_change_months,
                                                  job_reduction_months,
                                                  start_month,
                                                  i,
                                                  fur_return=cf.recall)

        jnums = results[0]
        count_col = results[1]
        held = results[2]
        fur = results[3]
        orig_jobs = results[4]
        # HELD JOB
        df_long['held'] = held
        # JOB_COUNT
        df_long['job_count'] = count_col

    else:

        orig_jobs = f.make_stovepipe_jobs_from_jobs_arr(jcnts,
                                                        short_len)

        jnums = f.assign_jobs_full_flush_skip_furs(this_ds_nonret_each_month,
                                                   orig_jobs,
                                                   fur_codes,
                                                   num_of_job_levels)
        # JOB_COUNT
        fur_count = fur_counts[i]
        df_long['job_count'] = f.put_map(jnums, jcnts, fur_count).astype(int)

    df_short['orig_job'] = orig_jobs

    # ORIG_JOB

    df_long['orig_job'] = df_short['orig_job']

    # ASSIGN JOBS - (stovepipe method only since only
    # assigning within each employee group separately)

    # JNUM

    df_long['jnum'] = jnums

    # LNUM, LSPCNT (includes fur)

    df_long['lnum'] = f.create_snum_array(jnums, this_ds_nonret_each_month)

    df_long['lspcnt'] = df_long['lnum'] / max(df_long.lnum)

    # SNUM, SPCNT (excludes fur)

    df_long['snum'], df_long['spcnt'] = \
        f.snum_and_spcnt(np.array(df_long.jnum),
                         num_of_job_levels,
                         lowers,
                         uppers,
                         this_month_counts,
                         all_months)

    # RANK in JOB

    df_long['rank_in_job'] = df_long.groupby(['mnum', 'jnum']).cumcount() + 1

    # jobp

    df_long['jobp'] = (df_long['rank_in_job'] /
                       df_long['job_count']) + (df_long['jnum'] - .0001)

    # if cf.compute_job_category_order:
    #     df_long['cat_order'] = df_long.groupby('mnum', sort=False)['jobp'] \
    #         .rank(method='first') * len(ds) / len(df_long)

    # PAY - merge with pay table - provides monthly pay
    if cf.compute_pay_measures:

        if cf.discount_longev_for_fur:
            # flip ones and zeros...
            df_long['non_fur'] = 1 - fur
            df_long['fur'] = fur

            non_fur = np.array(df_long.groupby('empkey')['non_fur'].cumsum())
            starting_mlong = np.array(df_long.s_lmonths)
            cum_active_months = non_fur + starting_mlong

            df_long['scale'] = np.clip((cum_active_months / 12) + 1,
                                       1, cf.top_of_scale).astype(int)
            df_long.pop('non_fur')

        df_pt_index = pd.DataFrame(
            index=(df_long['scale'] * 100) + df_long['jnum'] +
            (df_long['year'] * 100000))

        if cf.enhanced_jobs:
            df_pt = pd.read_pickle(
                'dill/pay_table_enhanced_with_fur_indexed.pkl')
        else:
            df_pt = pd.read_pickle(
                'dill/pay_table_basic_with_fur_indexed.pkl')

        df_pt_index['monthly'] = df_pt['monthly']

        df_long['monthly'] = np.array(df_pt_index.monthly)

        # MPAY
        # adjust monthly pay for any raise and last month pay percent if
        # applicable

        df_long['mpay'] = (
            (df_long['pay_raise'] *
             df_long['mth_pcnt'] *
             df_long['monthly'])) / 1000

        df_long.pop('monthly')

        # CPAY

        df_long['cpay'] = df_long.groupby('idx')['mpay'].cumsum()

    ds = pd.concat([ds, df_long], ignore_index=True)

if cf.compute_job_category_order:
        ds['cat_order'] = ds.groupby('mnum', sort=False)['jobp'] \
            .rank(method='first')

ds.sort_values(by=['mnum', 'snum'], inplace=True)
ds.set_index('empkey', drop=False, verify_integrity=False, inplace=True)

# save to file
if cf.save_to_pickle:
    ds.to_pickle('dill/stand.pkl')
