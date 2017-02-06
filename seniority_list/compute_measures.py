#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''create integrated datasets

output is a single, integrated dataset based on an input integrated list
order and any conditions

The editor output order list (p_new_order.pkl) may be used as an input by
including "edit" within the conditions list.  The resultant dataframe will
be stored as "ds_edit.pkl"
'''

import pandas as pd
import numpy as np

import functions as f

from sys import argv
from collections import OrderedDict as od


script, proposal_name, *conditions = argv

pre, suf = 'dill/', '.pkl'

skeleton_path_string = (pre + 'skeleton' + suf)

proposal_order_string = (pre + 'p_' + proposal_name + suf)
stand_path_string = (pre + 'standalone' + suf)

output_name = 'ds_' + proposal_name

try:
    df_master = pd.read_pickle(pre + 'master' + suf)
except:
    print('Master list not found.  Run build_program_files script?')

ds = pd.read_pickle(skeleton_path_string)
df_order = pd.read_pickle(proposal_order_string)

sdict = pd.read_pickle('dill/dict_settings.pkl')

start_date = pd.to_datetime(sdict['starting_date'])

# do not include inactive employees (other than furlough) in data model
df_master = df_master[
    (df_master.line == 1) | (df_master.fur == 1)].copy()

population = len(df_master)
num_of_job_levels = sdict['num_of_job_levels']
lspcnt_calc = sdict['lspcnt_calc_on_remaining_population']

if sdict['enhanced_jobs']:
    # use job dictionary from case-specific configuration file for conversion
    eg_counts, j_changes = f.convert_to_enhanced(sdict['eg_counts'],
                                                 sdict['j_changes'],
                                                 sdict['jd'])
else:
    eg_counts = sdict['eg_counts']
    j_changes = sdict['j_changes']

# grab the job counts from the config file and insert into array (and
# compute totals)
jcnts_arr = f.make_jcnts(eg_counts)

# ORDER the skeleton df according to INTEGRATED list order.
# df_skel can initially be in any integrated order, each employee group must
# be in proper order relative to itself.
# Use the short-form 'idx' (order) column from either the proposed
# list or the new_order column from an edited list to create a new column,
# 'new_order', within the long-form df_skel.  The new order column is created
# by data alignment using the common empkey indexes.  The skeleton may then
# be sorted by month and new_order.  (note: duplicate df_skel empkey index
# empkeys (from different months) are assigned the same order value)

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

    sg_rights = sdict['sg_rights']
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
cmonths = f.career_months_df_in(df_master, sdict['starting_date'])

# nonret_each_month: count of non-retired employees remaining
# in each month until no more remain -
# length is equal to longest career length
nonret_each_month = f.count_per_month(cmonths)
all_months = np.sum(nonret_each_month)
high_limits = nonret_each_month.cumsum()
low_limits = f.make_lower_slice_limits(high_limits)

job_level_counts = np.array(jcnts_arr[1])

if sdict['delayed_implementation']:

    imp_month = sdict['imp_month']
    imp_low = low_limits[imp_month]
    imp_high = high_limits[imp_month]

    # read the standalone dataset (info is not in integrated order)
    ds_stand = pd.read_pickle(stand_path_string)

    # get standalone data and order it the same as the integrated dataset.
    # create a unique key column in the standalone data df and a temporary df
    # which is ordered according to the integrated dataset
    imp_cols, arr_dict, col_array = \
        f.make_preimp_array(ds_stand, ds,
                            imp_high, sdict['compute_job_category_order'],
                            sdict['compute_pay_measures'])

    # # select columns to use as pre-implementation data for integrated dataset
    # # data is limited to the pre-implementation months

    # aligned_jnums and aligned_fur arrays are the same as standalone data
    # up to the end of the implementation month, then the standalone value for
    # the implementation month is passed down unchanged for the remainder of
    # months in the model.  These arrays carry over standalone data for each
    # employee group to be honored until and when the integrated list is
    # implemented.
    # These values from the standalone datasets (furlough status and standalone
    # job held at the implementation date) are needed for subsequent
    # integrated dataset job assignment calculations.  Other standalone values
    # are simply copied and inserted into the pre-implementation months of the
    # integrated dataset.

    delayed_jnums = col_array[arr_dict['jnum']]
    delayed_fur = col_array[arr_dict['fur']]

    aligned_jnums = f.align_fill_down(imp_low,
                                      imp_high,
                                      ds[[]],  # indexed with empkeys
                                      delayed_jnums)

    aligned_fur = f.align_fill_down(imp_low,
                                    imp_high,
                                    ds[[]],
                                    delayed_fur)

    # now assign "filled-down" job numbers to numpy array
    delayed_jnums[imp_low:] = aligned_jnums[imp_low:]
    delayed_fur[imp_low:] = aligned_fur[imp_low:]

    # ORIG_JOB and FUR (delayed implementation)
    # then assign numpy array values to orig_job column of integrated dataset
    # as starting point for integrated job assignments
    ds['orig_job'] = delayed_jnums
    ds['fur'] = delayed_fur

    if sdict['int_job_counts']:
        # assign combined job counts prior to the implementation date.
        # (otherwise, separate employee group counts will be used when
        # data is transferred from col_array at end of script)
        # NOTE:  this data is the actual number of jobs held within each
        # category; could be less than the number of jobs available as
        # attrition occurs
        standalone_preimp_job_counts = \
            f.make_delayed_job_counts(imp_month,
                                      delayed_jnums,
                                      low_limits,
                                      high_limits)
        col_array[arr_dict['job_count']][:imp_high] = \
            standalone_preimp_job_counts

else:
    # set implementation month at zero for job assignment routine
    imp_month = 0

    # ORIG_JOB and FUR (no delayed implementation)
    # transfer proposal stovepipe jobs (month_form) to long_form via index
    # (empkey) alignment...
    ds['orig_job'] = df_master['orig_job']
    # developer note:  test to verify this is not instantiated elsewhere...
    ds['fur'] = df_master['fur']

# grab long_form indexed stovepipe jobs (int)
orig = np.array(ds['orig_job'])

table = f.job_gain_loss_table(nonret_each_month.size,
                              num_of_job_levels,
                              jcnts_arr,
                              j_changes)

job_change_months = f.get_job_change_months(j_changes)
reduction_months = f.get_job_reduction_months(j_changes)
# copy selected columns from ds for job assignment function input below.
# note:  if delayed implementation, the 'fur' and 'orig_job' columns contain
# standalone data through the implementation month.
df_align = ds[['eg', 'sg', 'fur', 'orig_job']].copy()

# this is the main job assignment function.  It loops through all of the
# months in the model and assigns jobs
jobs_and_counts = \
    f.assign_jobs_nbnf_job_changes(df_align,
                                   sdict['num_of_job_levels'],
                                   sdict['delayed_implementation'],
                                   low_limits,
                                   high_limits, all_months,
                                   table[0], table[1],
                                   job_change_months,
                                   reduction_months,
                                   imp_month,
                                   conditions,
                                   sdict['sg_rights'],
                                   sdict['ratio_cond'],
                                   sdict['count_cond'],
                                   sdict['quota_dict'],
                                   sdict['recalls'],
                                   fur_return=sdict['recall'])

# if job_changes, replace original fur column...
ds['fur'] = jobs_and_counts[3]

# JNUM, NBNF, FBFF

nbnf = jobs_and_counts[0]

job_col = f.assign_jobs_full_flush_with_job_changes(
    nonret_each_month, table[1], num_of_job_levels)

if sdict['no_bump']:
    ds['jnum'] = nbnf
    ds['fbff'] = job_col
else:
    ds['nbnf'] = nbnf
    ds['jnum'] = job_col

# SNUM, SPCNT, LNUM, LSPCNT

jnum_jobs = np.array(ds['jnum']).astype(int)
job_count_each_month = table[1]

ds['snum'], ds['spcnt'], ds['lnum'], ds['lspcnt'] = \
    f.create_snum_and_spcnt_arrays(jnum_jobs, num_of_job_levels,
                                   nonret_each_month,
                                   job_count_each_month,
                                   lspcnt_calc)

# RANK in JOB

ds['rank_in_job'] = ds.groupby(['mnum', 'jnum'], sort=False).cumcount() + 1

# JOB_COUNT

ds['job_count'] = jobs_and_counts[1]

# JOBP

jpcnt = np.array(ds.rank_in_job / ds.job_count)
np.put(jpcnt, np.where(jpcnt == 1.0)[0], .99999)

# ds['jobp'] = (ds['rank_in_job'] / ds['job_count']) + (ds['jnum'] - .001)
ds['jobp'] = ds['jnum'] + jpcnt

# PAY - merge with pay table - provides monthly pay
if sdict['compute_pay_measures']:

    # account for furlough time (only count active months)
    if sdict['discount_longev_for_fur']:
        # skel(ds) provides pre-calculated non-discounted scale data
        # flip ones and zeros...
        ds['non_fur'] = 1 - ds.fur

        non_fur = np.array(ds.groupby('empkey')['non_fur'].cumsum())
        ds.pop('non_fur')
        starting_mlong = np.array(ds.s_lmonths)
        cum_active_months = non_fur + starting_mlong
        ds['mlong'] = cum_active_months
        ds['ylong'] = ds['mlong'] / 12
        ds['scale'] = np.clip((cum_active_months / 12) + 1, 1,
                              sdict['top_of_scale']).astype(int)

    # make a new long_form dataframe and assign a combination of
    # pay-related ds columns from large dataset as its index...
    # the dataframe is empty - we are only making an index-alignment
    # vehicle to use with indexed pay table....
    # the dataframe index contains specific scale, job, and contract year for
    # each line in long_form ds
    df_pt_index = pd.DataFrame(
        index=(ds['scale'] * 100) + ds['jnum'] + (ds['year'] * 100000))

    if sdict['enhanced_jobs']:
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

if sdict['delayed_implementation']:
    ds_cols = ds.columns
    # grab each imp_col (column to insert standalone or pre-implementation
    # date data) and replace integrated data up through implementation date
    for col in imp_cols:
        if col in ds_cols:
            arr = np.array(ds[col])
            arr[:imp_high] = col_array[arr_dict[col]][:imp_high]
            ds[col] = arr

# CAT_ORDER
# global job ranking
if sdict['compute_job_category_order']:
    ds['cat_order'] = f.make_cat_order(ds, table[0])

# save to file
if sdict['save_to_pickle']:
    ds.to_pickle(dataset_path_string)
