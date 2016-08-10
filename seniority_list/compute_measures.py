#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import functions as f
import config as cf

from sys import argv

script, proposal_name, *conditions = argv

print(conditions)

pre, suf = 'dill/', '.pkl'

skeleton_path_string = (pre + 'skel' + suf)

proposal_order_string = (pre + proposal_name + suf)
stand_path_string = (pre + 'stand' + suf)

if proposal_name == 'hybrid':
    output_name = 'dsh'
else:
    output_name = 'ds' + proposal_name[-1:]

ds = pd.read_pickle(skeleton_path_string)
df_order = pd.read_pickle(proposal_order_string)
df_master = pd.read_pickle(pre + 'master' + suf)

start_date = pd.to_datetime(cf.starting_date)

# # only include pilots that are not retired prior to the starting_month
# df_master = df_master[
#     df_master.retdate > start_date - pd.DateOffset(months=1)]

if cf.actives_only:
    df_master = df_master[df_master.line == 1].copy()
    ds = ds[ds.fur == 0].copy()
else:
    df_master = df_master[
        (df_master.line == 1) | (df_master.fur == 1)].copy()

population = len(df_master)
num_of_job_levels = cf.num_of_job_levels
lspcnt_calc = cf.lspcnt_calc_on_remaining_population

if cf.enhanced_jobs:
    eg_counts = f.convert_jcnts_to_enhanced(cf.eg_counts,
                                            cf.full_time_pcnt1,
                                            cf.full_time_pcnt2)
    j_changes = f.convert_job_changes_to_enhanced(cf.j_changes, cf.jd)
else:
    eg_counts = cf.eg_counts
    j_changes = cf.j_changes
# grab the job counts from the config file and insert into array (and
# compute totals)

jcnts_arr = f.make_jcnts(eg_counts)

# >>> OPTIMIZATION start here (above not repeated...) <<<

# ORDER the skeleton df
# df_skel can initially be in any order.
# use the empkey index (contains duplicates) to align
# the idx (order) column from the proposed list or the
# new_order column from an edited list to the skeleton

if 'edit' in conditions:
    df_new_order = pd.read_pickle(proposal_order_string)
    ds['new_order'] = df_new_order['new_order']
    dataset_path_string = (pre + 'ds_edit' + suf)
else:
    order_key = df_order.idx
    ds['new_order'] = order_key
    dataset_path_string = (pre + output_name + suf)

# sort the skeleton by month and proposed list order
ds.sort_values(['mnum', 'new_order'], inplace=True)

# ORIG_JOB*

eg_sequence = np.array(df_master.eg)
fur_sequence = np.array(df_master.fur)

if 'prex' in conditions:

    eg2_stove = f.make_stovepipe_jobs_from_jobs_arr(jcnts_arr[0][1])
    eg3_stove = f.make_stovepipe_jobs_from_jobs_arr(jcnts_arr[0][2])
    sg = np.array(df_master[df_master.eg == 1]['sg'])
    eg1_fur = np.array(df_master[df_master.eg == 1]['fur'])
    eg1_ojob_array = f.make_amer_stovepipe_short_prex(
        jcnts_arr[0][0], sg, cf.sg_rights, eg1_fur)

    eg1_prex_stove = np.take(eg1_ojob_array, np.where(eg1_fur == 0)[0])

    sp_arr = np.array((eg1_prex_stove, eg2_stove, eg3_stove))
    # total of jobs per eg
    eg_job_counts = np.add.reduce(jcnts_arr[0], axis=1)

    orig_jobs = f.make_intgrtd_from_sep_stove_lists(
        sp_arr, eg_sequence, fur_sequence, eg_job_counts, num_of_job_levels)

else:

    orig_jobs = f.make_original_jobs_from_counts(
        jcnts_arr[0], eg_sequence, fur_sequence, num_of_job_levels).astype(int)

# insert stovepipe job result into new column of proposal (month_form)
# this indexes the jobs with empkeys (orig_jobs is an ndarray only)

df_master['orig_job'] = orig_jobs

# ASSIGN JOBS - flush and no flush option*

# cmonths - career length in months for each employee.
#   length is equal to number of employees
cmonths = f.career_months_df_in(df_master)

# nonret_each_month: count of non-retired employees remaining
# in each month until no more remain -
# length is equal to longest career length
nonret_each_month = f.count_per_month(cmonths)
all_months = np.sum(nonret_each_month)
cumulative = nonret_each_month.cumsum()
np_low_limits = f.make_lower_slice_limits(cumulative)

job_level_counts = np.array(jcnts_arr[1])

if cf.delayed_implementation:

    imp_month = cf.imp_month
    imp_low = np_low_limits[imp_month]
    imp_high = cumulative[imp_month]

    dstand = pd.read_pickle(stand_path_string)
    # ds_option = dstand[['job_count', 'lspcnt',
    #                     'spcnt', 'rank_in_job', 'jobp']]
    dstand = dstand[['mnum', 'empkey', 'jnum', 'fur', 'cat_order']][:imp_high]
    dstand.rename(columns={'jnum': 'stand_jobs',
                           'cat_order': 'stand_cat_order'}, inplace=True)
    dstand['key'] = (dstand.empkey * 1000) + dstand.mnum
    dstand.drop(['mnum', 'empkey'], inplace=True, axis=1)

    ds_temp = ds[['mnum', 'empkey']][:imp_high]
    ds_temp['key'] = (ds_temp.empkey * 1000) + ds_temp.mnum
    ds_temp.drop(['mnum', 'empkey'], inplace=True, axis=1)

    ds_temp = pd.merge(ds_temp, dstand, on='key')

    dstand = []

    temp_jnums = np.array(ds_temp.stand_jobs)
    delayed_jnums = np.zeros(all_months)
    delayed_jnums[:imp_high] = temp_jnums

    temp_fur = np.array(ds_temp.fur)
    delayed_fur = np.zeros(all_months)
    delayed_fur[:imp_high] = temp_fur

    aligned_jnums = f.align_fill_down(imp_low,
                                      imp_high,
                                      ds[[]],
                                      delayed_jnums[imp_low:imp_high],
                                      delayed_jnums)

    aligned_fur = f.align_fill_down(imp_low,
                                    imp_high,
                                    ds[[]],
                                    delayed_fur[imp_low:imp_high],
                                    delayed_fur)

    delayed_jnums[imp_low:] = aligned_jnums[imp_low:]
    ds['orig_job'] = delayed_jnums

    delayed_fur[imp_low:] = aligned_fur[imp_low:]
    ds['fur'] = delayed_fur

    # CAT_ORDER preliminary
    # grab standalone data for pre-implementation period
    if cf.compute_job_category_order:
        temp_cat = np.array(ds_temp.stand_cat_order)
        delayed_cat = np.zeros(all_months)
        delayed_cat[:imp_high] = temp_cat

    # this function assigns combined job counts wiping out
    # standalone counts...refactor to add (skip this?) option
    # to keep standalone job counts
    standalone_preimp_job_counts = \
        f.make_delayed_job_counts(imp_month,
                                  delayed_jnums,
                                  np_low_limits,
                                  cumulative)
else:

    imp_month = 0
    # ORIG_JOB
    # transfer proposal stovepipe jobs (month_form) to long_form via index
    # (empkey) alignment...
    ds['orig_job'] = df_master['orig_job']

# grab long_form indexed stovepipe jobs (int)
orig = np.array(ds['orig_job'])

if cf.compute_with_job_changes:

    table = f.job_gain_loss_table(nonret_each_month.size,
                                  num_of_job_levels,
                                  jcnts_arr,
                                  j_changes)

    job_change_months = f.get_job_change_months(j_changes)
    reduction_months = f.get_job_reduction_months(j_changes)

    df_align = ds[['eg', 'sg', 'fur', 'orig_job']].copy()

    jobs_and_counts = f.assign_jobs_nbnf_job_changes(df_align, np_low_limits,
                                                     cumulative, all_months,
                                                     table[0], table[1],
                                                     job_change_months,
                                                     reduction_months,
                                                     imp_month,
                                                     proposal_name,
                                                     conditions,
                                                     fur_return=cf.recall)

    nbnf = jobs_and_counts[0]
    # if job_changes, replace original fur column...
    ds['fur'] = jobs_and_counts[3]
    # remove this assignment when testing complete...
    ds['func_job'] = jobs_and_counts[2]
else:

    nbnf = f.assign_jobs_nobump_noflush(orig, np.array(ds.fur), np_low_limits,
                                        cumulative, all_months,
                                        job_level_counts)

# FBFF*

if cf.compute_with_job_changes:
    job_col = f.assign_jobs_full_flush_with_job_changes(
        nonret_each_month, table[1], num_of_job_levels)
else:
    integrated_stovepipe_jobs = f.make_stovepipe_jobs_from_jobs_arr(
        jcnts_arr[1], population)
    job_col = f.assign_jobs_full_flush(
        nonret_each_month, integrated_stovepipe_jobs, num_of_job_levels)


# JNUM, NBNF, FBFF

if cf.no_bump:
    ds['jnum'] = nbnf
    ds['fbff'] = job_col
else:
    ds['jnum'] = job_col
    ds['nbnf'] = nbnf


jnum_jobs = np.array(ds['jnum']).astype(int)

# SNUM, SPCNT, LNUM, LSPCNT

if cf.compute_with_job_changes:
    job_count_each_month = table[1]
    # insert if delayed implementation here...
    # this code will not consider delayed job counts
    # alignment needed with sep results...
    ds['snum'], ds['spcnt'], ds['lnum'], ds['lspcnt'] = \
        f.create_snum_and_spcnt_arrays(jnum_jobs, num_of_job_levels,
                                       nonret_each_month,
                                       job_count_each_month,
                                       lspcnt_calc)
else:
    ds['snum'] = f.create_snum_array(nbnf, nonret_each_month)
    ds['spcnt'] = ds['snum'] / max(ds.snum)


# RANK in JOB

ds['rank_in_job'] = ds.groupby(['mnum', 'jnum'], sort=False).cumcount() + 1


# JOB_COUNT
if cf.compute_with_job_changes:
    if cf.delayed_implementation:
        delayed_job_counts = np.zeros(len(ds))
        delayed_job_counts[:imp_high] = standalone_preimp_job_counts
        delayed_job_counts[imp_high:] = jobs_and_counts[1][imp_high:]
        ds['job_count'] = delayed_job_counts.astype(int)
    else:
        ds['job_count'] = jobs_and_counts[1]
else:
    jnums = np.array(ds.jnum)
    job_level_counts = np.array(jcnts_arr[1])
    ds['job_count'] = f.put_map(jnums,
                                job_level_counts,
                                sum(cf.furlough_count)).astype(int)

# JOBP

ds['jobp'] = (ds['rank_in_job'] / ds['job_count']) + (ds['jnum'] - .001)

# CAT_ORDER final
# rank integrated jobp data then assign from implementation date forward
if cf.compute_job_category_order:
    cat_arr = np.array(ds.groupby('mnum', sort=False)['jobp']
                       .rank(method='first'))
    delayed_cat[imp_high:] = cat_arr[imp_high:]
    ds['cat_order'] = delayed_cat

# PAY - merge with pay table - provides monthly pay
if cf.compute_pay_measures:

    # account for furlough time (only count active months)
    if cf.discount_longev_for_fur:

        # flip ones and zeros...
        ds['non_fur'] = 1 - ds.fur

        non_fur = np.array(ds.groupby('empkey')['non_fur'].cumsum())
        ds.pop('non_fur')
        starting_mlong = np.array(ds.s_lmonths)
        cum_active_months = non_fur + starting_mlong
        ds['mlong'] = cum_active_months
        ds['ylong'] = ds['mlong'] / 12
        ds['scale'] = np.clip((cum_active_months / 12) + 1, 1,
                              cf.top_of_scale).astype(int)

    # make a new long_form dataframe and assign a combination of
    # pay-related ds columns from large dataset as its index...
    # the dataframe is empty - we are only making an index-alignment
    # vehicle to use with indexed pay table....
    # the dataframe index contains specific scale, job, and contract year for
    # each line in long_form ds
    df_pt_index = pd.DataFrame(
        index=(ds['scale'] * 100) + ds['jnum'] + (ds['year'] * 100000))

    if cf.enhanced_jobs:
        df_pt = pd.read_pickle(
            'dill/pay_table_enhanced_with_fur_indexed.pkl')
    else:
        df_pt = pd.read_pickle(
            'dill/pay_table_basic_with_fur_indexed.pkl')

    # 'data-align' small indexed pay_table to long_form df:
    df_pt_index['monthly'] = df_pt['monthly']

    ds['monthly'] = np.array(df_pt_index.monthly)

    # MPAY
    # adjust monthly pay for any raise and last month pay percent if applicable

    ds['mpay'] = ((ds['pay_raise'] * ds['mth_pcnt'] * ds['monthly'])) / 1000

    ds.pop('monthly')

    # CPAY

    ds['cpay'] = ds.groupby('new_order')['mpay'].cumsum()

# save to file
if cf.save_to_pickle:
    ds.to_pickle(dataset_path_string)
