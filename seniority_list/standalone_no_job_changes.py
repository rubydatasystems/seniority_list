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
num_of_months = np.unique(ds.mnum).size
egs = pd.unique(ds.eg)

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
    fur_count = fur_counts[i]
    fur_codes = np.array(df_long.fur)

    # ORIG_JOB*
    cmonths_this_ds = f.career_months_df_in(df_short)
    this_ds_nonret_each_month = f.count_per_month(cmonths_this_ds)
    upper = this_ds_nonret_each_month.cumsum()
    lower = f.make_lower_slice_limits(upper)
    all_months = np.sum(this_ds_nonret_each_month)

    this_table = table[0][i]
    this_month_counts = table[1][i]

    if (i == 0) and ('prex' in conditions):  # i == 0 >> eg1 from skeleton

        sg_rights = np.array(cf.sg_rights)
        sg_jobs = np.transpose(sg_rights)[1]
        sg_counts = np.transpose(sg_rights)[2]
        sg_dict = dict(zip(sg_jobs, sg_counts))

        # calc sg sup c condition month range and concat
        sg_month_range = np.arange(np.min(sg_rights[:, 3]),
                                   np.max(sg_rights[:, 4]))

        sg_codes = np.array(df_long.sg)

        df_align = df_long[['sg', 'fur']]

        amer_job_counts = jcnts_arr[0][0]

        sg = f.make_amer_standalone_long_prex(lower,
                                              upper,
                                              df_align,
                                              amer_job_counts,
                                              sg_jobs,
                                              sg_dict,
                                              sg_month_range)

        orig_jobs = sg[1]
        jnums = sg[0]

    else:

        orig_jobs = f.make_stovepipe_jobs_from_jobs_arr(jcnts,
                                                        short_len)

        jnums = f.assign_jobs_full_flush_skip_furs(this_ds_nonret_each_month,
                                                   orig_jobs,
                                                   fur_codes,
                                                   num_of_job_levels)

    df_short['orig_job'] = orig_jobs

    # ORIG_JOB

    df_long['orig_job'] = df_short['orig_job']

    # ASSIGN JOBS - (stovepipe method only since only
    # assigning within each employee group separately)

    # JNUM

    df_long['jnum'] = jnums

    # LIST, LSPCNT (includes fur)

    df_long['list'] = f.create_snum_array(jnums, this_ds_nonret_each_month)

    df_long['lspcnt'] = df_long['list'] / max(df_long.list)

    # SNUM, SPCNT (excludes fur)

    df_long['snum'], df_long['spcnt'] = \
        f.snum_and_spcnt(np.array(df_long.jnum),
                         num_of_job_levels,
                         lower,
                         upper,
                         this_month_counts,
                         all_months)

    # RANK in JOB

    df_long['rank_in_job'] = df_long.groupby(['mnum', 'jnum']).cumcount() + 1

    # JOB_COUNT*

    jnums = np.array(df_long.jnum)

    # JOB_COUNT

    df_long['job_count'] = f.put_map(jnums, jcnts, fur_count).astype(int)

    # JOBP

    df_long['jobp'] = (df_long['rank_in_job'] /
                       df_long['job_count']) + (df_long['jnum'] - .0001)

    # PAY - merge with pay table - provides monthly pay
    if cf.compute_pay_measures:

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

ds.sort_values(by=['mnum', 'snum'], inplace=True)
ds.set_index('empkey', drop=False, verify_integrity=False, inplace=True)

# save to file
if cf.save_to_pickle:
    ds.to_pickle('dill/stand.pkl')
