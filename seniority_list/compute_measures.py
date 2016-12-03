#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''create integrated datasets

The editor output order list (p_new_order.pkl) may be used as an input by
including "edit" within the conditions list.  The resultant dataframe will
be stored as "ds_edit.pkl"
'''

import pandas as pd
import numpy as np

import functions as f
import config as cf

from sys import argv
from collections import OrderedDict as od


script, proposal_name, *conditions = argv

pre, suf = 'dill/', '.pkl'

skeleton_path_string = (pre + 'skeleton' + suf)

proposal_order_string = (pre + 'p_' + proposal_name + suf)
stand_path_string = (pre + 'standalone' + suf)

output_name = 'ds_' + proposal_name

ds = pd.read_pickle(skeleton_path_string)
df_order = pd.read_pickle(proposal_order_string)
df_master = pd.read_pickle(pre + 'master' + suf)

start_date = pd.to_datetime(cf.starting_date)

# do not include inactive employees (other than furlough) in data model
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

# create list of employee group codes from the master data
egs = sorted(pd.unique(eg_sequence))

if 'prex' in conditions:

    sg_rights = cf.sg_rights
    sg_eg_list = []
    sg_dict = od()
    stove_dict = od()

    # Find the employee groups which have pre-existing job rights...
    # grab the eg code from each sg (special group) job right description
    # and add to sg_eg_list
    for line_item in sg_rights:
        sg_eg_list.append(line_item[0])
    # place unique eg codes into sorted list
    sg_eg_list = sorted(pd.unique(sg_eg_list))

    # Make a dictionary containing the special group data for each group with
    # special rights
    for eg in sg_eg_list:
        sg_data = []
        for line_item in sg_rights:
            if line_item[0] == eg:
                sg_data.append(line_item)
        sg_dict[eg] = sg_data

    for eg in egs:

        if eg in sg_eg_list:
            # (run prex stovepipe routine with eg dict key and value)
            sg = np.array(df_master[df_master.eg == eg]['sg'])
            fur = np.array(df_master[df_master.eg == eg]['fur'])
            ojob_array = f.make_stovepipe_prex_shortform(
                jcnts_arr[0][eg - 1], sg, sg_dict[eg], fur)
            prex_stove = np.take(ojob_array, np.where(fur == 0)[0])
            stove_dict[eg] = prex_stove
        else:
            # (run make_stovepipe routine with eg dict key and value)
            stove_dict[eg] = f.make_stovepipe_jobs_from_jobs_arr(
                jcnts_arr[0][eg - 1])

    # use dict values as inputs to sp_arr,
    # ordered dict maintains proper sequence...
    sp_arr = list(np.array(list(stove_dict.values())))
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
high_limits = nonret_each_month.cumsum()
low_limits = f.make_lower_slice_limits(high_limits)

job_level_counts = np.array(jcnts_arr[1])

if cf.delayed_implementation:

    imp_month = cf.imp_month
    imp_low = low_limits[imp_month]
    imp_high = high_limits[imp_month]

    dstand = pd.read_pickle(stand_path_string)

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

    # TODO
    # Refactor section below with loop and dictionary so all column data may be
    # passed for delayed implementation...
    # possibly make function to incorporate section above as well.
    # ***********************************************************************

    temp_jnums = np.array(ds_temp.stand_jobs)
    delayed_jnums = np.zeros(all_months)
    delayed_jnums[:imp_high] = temp_jnums

    temp_fur = np.array(ds_temp.fur)
    delayed_fur = np.zeros(all_months)
    delayed_fur[:imp_high] = temp_fur

    aligned_jnums = f.align_fill_down(imp_low,
                                      imp_high,
                                      ds[[]],  # indexed with empkeys
                                      delayed_jnums)

    aligned_fur = f.align_fill_down(imp_low,
                                    imp_high,
                                    ds[[]],
                                    delayed_fur)

    delayed_jnums[imp_low:] = aligned_jnums[imp_low:]
    ds['orig_job'] = delayed_jnums

    delayed_fur[imp_low:] = aligned_fur[imp_low:]
    ds['fur'] = delayed_fur

    # ***********************************************************************

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
                                  low_limits,
                                  high_limits)
    ds_temp = []

else:

    imp_month = 0
    # ORIG_JOB
    # transfer proposal stovepipe jobs (month_form) to long_form via index
    # (empkey) alignment...
    ds['orig_job'] = df_master['orig_job']

# grab long_form indexed stovepipe jobs (int)
orig = np.array(ds['orig_job'])

table = f.job_gain_loss_table(nonret_each_month.size,
                              num_of_job_levels,
                              jcnts_arr,
                              j_changes)

job_change_months = f.get_job_change_months(j_changes)
reduction_months = f.get_job_reduction_months(j_changes)

df_align = ds[['eg', 'sg', 'fur', 'orig_job']].copy()

# this is the main job assignment function.  It loops through all of the
# months in the model and assigns jobs
jobs_and_counts = f.assign_jobs_nbnf_job_changes(df_align, low_limits,
                                                 high_limits, all_months,
                                                 table[0], table[1],
                                                 job_change_months,
                                                 reduction_months,
                                                 imp_month,
                                                 conditions,
                                                 fur_return=cf.recall)

nbnf = jobs_and_counts[0]
# if job_changes, replace original fur column...
ds['fur'] = jobs_and_counts[3]
# remove this assignment when testing complete...
ds['func_job'] = jobs_and_counts[2]

job_col = f.assign_jobs_full_flush_with_job_changes(
    nonret_each_month, table[1], num_of_job_levels)


# JNUM, NBNF, FBFF

if cf.no_bump:
    ds['jnum'] = nbnf
    ds['fbff'] = job_col
else:
    ds['jnum'] = job_col
    ds['nbnf'] = nbnf


jnum_jobs = np.array(ds['jnum']).astype(int)

# SNUM, SPCNT, LNUM, LSPCNT
job_count_each_month = table[1]

ds['snum'], ds['spcnt'], ds['lnum'], ds['lspcnt'] = \
    f.create_snum_and_spcnt_arrays(jnum_jobs, num_of_job_levels,
                                   nonret_each_month,
                                   job_count_each_month,
                                   lspcnt_calc)


# RANK in JOB

ds['rank_in_job'] = ds.groupby(['mnum', 'jnum'], sort=False).cumcount() + 1


# JOB_COUNT
if cf.delayed_implementation:
    delayed_job_counts = np.zeros(len(ds))
    delayed_job_counts[:imp_high] = standalone_preimp_job_counts
    delayed_job_counts[imp_high:] = jobs_and_counts[1][imp_high:]
    ds['job_count'] = delayed_job_counts.astype(int)
else:
    ds['job_count'] = jobs_and_counts[1]

# JOBP

ds['jobp'] = (ds['rank_in_job'] / ds['job_count']) + (ds['jnum'] - .001)

# CAT_ORDER final
# rank integrated jobp data then assign from implementation date forward
if cf.compute_job_category_order:
    cat_arr = np.array(ds.groupby('mnum', sort=False)['jobp']
                       .rank(method='first'))
    if cf.delayed_implementation:
        delayed_cat[imp_high:] = cat_arr[imp_high:]
        ds['cat_order'] = delayed_cat
    else:
        ds['cat_order'] = cat_arr

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
        df_pt = pd.read_pickle('dill/pay_table_enhanced.pkl')
    else:
        df_pt = pd.read_pickle('dill/pay_table_basic.pkl')

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
