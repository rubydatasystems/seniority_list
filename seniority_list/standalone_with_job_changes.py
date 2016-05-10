# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import functions as f
import config as cf

from sys import argv

script, input_skel = argv

pre, suf = 'dill/', '.pkl'

skeleton_path_string = (pre + input_skel + suf)

ds = pd.read_pickle(skeleton_path_string)

num_of_job_levels = cf.num_of_job_levels
fur_counts = cf.furlough_count
num_of_months = np.unique(ds.mnum).size
start_month = 0

# if cf.actives_only:
#     ds = ds[ds.fur == 0].copy()

# grab the job counts from the config file
# todo: change in config to job count array...
# todo: allow any number of job counts...
# jcnts_arr = f.make_array_of_job_lists(cf.eg1_job_count,
#                                       cf.eg2_job_count,
#                                       cf.eg3_job_count)

if num_of_job_levels == 16:
    eg_counts = f.convert_jcnts_to16(cf.eg_counts,
                                     cf.intl_blk_pcnt,
                                     cf.dom_blk_pcnt)
    j_changes = f.convert_job_changes_to16(cf.j_changes, cf.jd)
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

# jcnts_list = [jcnts_arr[0][0], jcnts_arr[0][1], jcnts_arr[0][2]]
# sort the skeleton by employee group, month, and index
# (preserves each group's list order)

ds.sort_values(['eg', 'mnum', 'idx'])

ds1 = ds[ds.eg == 1].copy()
ds2 = ds[ds.eg == 2].copy()
ds3 = ds[ds.eg == 3].copy()

short_ds1 = ds1[ds1.mnum == 0].copy()
short_ds2 = ds2[ds2.mnum == 0].copy()
short_ds3 = ds3[ds3.mnum == 0].copy()

ds = pd.DataFrame()

ds_list = [ds1, ds2, ds3]
short_ds_list = [short_ds1, short_ds2, short_ds3]


for i in range(len(ds_list)):

    df_long = ds_list[i]
    df_short = short_ds_list[i]
    jcnts = jcnts_arr[0][i]
    # jcnts = np.take(jcnts, np.where(jcnts != 0)[0])
    short_len = len(short_ds_list[i])

    # ORIG_JOB*

    cmonths_this_ds = f.career_months_df_in(df_short)
    this_ds_nonret_each_month = f.count_per_month(cmonths_this_ds)
    uppers = this_ds_nonret_each_month.cumsum()
    lowers = f.make_lower_slice_limits(uppers)
    all_months = np.sum(this_ds_nonret_each_month)

    this_table = table[0][i]
    this_month_counts = table[1][i]
    df_align = df_long[['twa', 'fur']]
    fur_codes = np.array(df_align.fur)

    # if i == 0 and cf.apply_supc:  # i == 0 >> eg1 from skeleton

    #     twa_rights = np.array(cf.twa_rights)
    #     twa_jobs = np.transpose(twa_rights)[1]
    #     sup_c_counts = np.transpose(twa_rights)[2]
    #     twa_dict = dict(zip(twa_jobs, sup_c_counts))

    #     # calc twa sup c condition month range and concat
    #     twa_month_range = np.arange(np.min(twa_rights[:, 3]),
    #                                 np.max(twa_rights[:, 4]))

    #     twa_codes = np.array(df_long.twa)

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
                                       1, 12).astype(int)
            df_long.pop('non_fur')

        df_pt_index = pd.DataFrame(
            index=(df_long['scale'] * 100) + df_long['jnum'] +
            (df_long['year'] * 100000))

        if num_of_job_levels == 8:
            df_pt = pd.read_pickle('dill/idx_pay_table_no_rsv_with_fur.pkl')
        if num_of_job_levels == 16:
            df_pt = pd.read_pickle('dill/idx_pay_table_with_rsv_with_fur.pkl')

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

ds.sort_values(by=['mnum', 'idx'], inplace=True)
ds.set_index('empkey', drop=False, verify_integrity=False, inplace=True)

# save to file
if cf.save_to_pickle:
    ds.to_pickle('dill/stand.pkl')
