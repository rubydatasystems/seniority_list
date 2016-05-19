# -*- coding: utf-8 -*-

'''produces a dataframe with information for each employee that is not
dependent on the ORDER of the list...
some column(s) are dependent on config variables, such as
pay raise beyond the contract last year.'''

import pandas as pd
import numpy as np

import functions as f
import config as cf

from sys import argv

script, input_list = argv

# read prepared list dataframe - proper column headers, column formats...
# normally this would be master.pkl, order-independent, concatenated list data
pre, suf = 'dill/', '.pkl'
list_path_string = (pre + input_list + suf)
df_list = pd.read_pickle(list_path_string)

output_name = 'skel'
skel_path_string = (pre + output_name + suf)

# only include pilots that are not retired prior to the starting_month
start_date = pd.to_datetime(cf.starting_date)
df_list = df_list[
    df_list.retdate >= start_date - pd.DateOffset(months=1)]

# include furloughees by default
df = df_list[(df_list.line == 1) | (df_list.fur == 1)].copy()
df_list = []

# MNUM*
# calculate the number of career months for each employee (short_form)
# cmonths is used for mnum, idx, and mth_pcnt calculations

cmonths = f.career_months_df_in(df)

# convert the python cmonths list to a numpy array and
# use that array as input for the count_per_month function.
# The count_per_month function output array is input for
# other functions (month_form)

nonret_each_month = f.count_per_month(cmonths)

# calculate the total of all career months (sum)

total_months = np.sum(nonret_each_month)

# first long form data generation.
# month numbers, same month number repeated for each month length (long_form)

long_form_skeleton = f.gen_month_skeleton(nonret_each_month)

# this is making a dataframe out of the
# long_form_skeleton (months) created above.
# this is the basis for the long_form dataframe...

# MNUM
# (month number)

skel = pd.DataFrame(long_form_skeleton.astype(int), columns=['mnum'])

freeze_date = pd.to_datetime('2013-12-31')
month_offset = ((start_date.year - freeze_date.year) * 12) - \
    (freeze_date.month - start_date.month)

skel.mnum = skel.mnum + month_offset

# IDX*
# grab emp index for each remaining
# employee for each month - used for merging dfs later

long_index = f.gen_emp_skeleton_index(nonret_each_month, cmonths)

# IDX
skel['idx'] = long_index.astype(int)

# EMPKEY
skel['empkey'] = pd.Series.map(skel.idx, df.reset_index(drop=True).empkey)

# grab retdates from df column (short_form)
# used for mth_pcnt and age calc (also mapping retdates)

rets = list(df['retdate'])

# calculate last month pay percentage for each employee (short_form)

# MTH_PCNT*
# slow calc - refactor candidate

# lmonth_pcnt = f.last_month_mpay_pcnt(rets)

# Eliminated last_month_mpay_pcnt function.
# Replaced with precalculated last month percentage dataframe...

df_last = pd.read_pickle('dill/last_month.pkl')

df.set_index('retdate', inplace=True)
df['lmonth_pcnt'] = df_last.last_pay
df.reset_index(inplace=True)
df.set_index('empkey', inplace=True, verify_integrity=False, drop=False)

lmonth_pcnt = np.array(df.lmonth_pcnt)

df_dict = {'mth_pcnt': lmonth_pcnt, 'final_month': cmonths}

df_last_month = pd.DataFrame(df_dict)

df_last_month['idx'] = df_last_month.index

df_last_month.set_index(['idx', 'final_month'], inplace=True)

skel = pd.merge(skel, df_last_month, right_index=True,
                left_on=['idx', 'mnum'], how='outer')

# MTH_PCNT
skel['mth_pcnt'] = skel.mth_pcnt.fillna(1)

# DATE, YEAR, PAY RAISE*

# set up date_range - end of month dates

df_dates = pd.DataFrame(pd.date_range(cf.starting_date,
                                      periods=len(nonret_each_month),
                                      freq='M'), columns=['date'])

# This is an input for the contract_pay_and_year_and_raise function

date_series = pd.to_datetime(list(df_dates['date']))

# this function produces a 2-column array.
# First column is the year value of the date list passed as an input.
# The second column is either 1.0 or
# a calculated percentage pay raise after the last contract year.

year_and_scale = \
    f.contract_pay_year_and_raise(date_series,
                                  exception=True,
                                  future_raise=cf.pay_raise,
                                  date_exception='2014-12-31',
                                  year_additive=.1,
                                  annual_raise=cf.annual_pcnt_raise,
                                  last_contract_year=2019.0)

df_dates['year'] = year_and_scale[0]

df_dates['pay_raise'] = year_and_scale[1]

# the merge below brings in 3 columns - date, year, and pay_raise
# - from month_form to long_form

# DATE, YEAR, PAY RAISE
skel = pd.merge(skel, df_dates, right_index=True, left_on=['mnum'])

# AGE, SCALE*
# calculate and assign starting age and
# starting longevity.
# Assign to columns in df and then data align merge into skeleton df.
# These columns are used later for age and scale calculations.
# Merged here so that they could be done together
# after setting indexes to match.

s_age = f.starting_age(rets)
df['s_age'] = s_age

# data alignment magic...set index to empkey
skel.set_index('empkey', inplace=True, verify_integrity=False, drop=False)


# AGE, RETDATE, EG, DOH, LDATE, LNAME, FUR to long_form skeleton
skel['s_age'] = df.s_age

if cf.add_eg_col:
    skel['eg'] = df.eg
if cf.add_retdate_col:
    skel['retdate'] = df.retdate
if cf.add_doh_col:
    skel['doh'] = df.doh
if cf.add_ldate_col:
    skel['ldate'] = df.ldate
if cf.add_lname_col:
    skel['lname'] = df.lname
if cf.add_line_col:
    skel['line'] = df.line
if cf.add_sg_col:
    skel['sg'] = df.sg

if not cf.actives_only:
    skel['fur'] = df.fur


# SCALE*

if cf.compute_pay_measures:

    df['s_lyears'] = f.longevity_at_startdate(list(df['ldate']))
    skel['s_lyears'] = df.s_lyears

    month_inc = (1 / 12)

    # scale is payrate longevity level
    # compute scale for each employee for each month
    # begin with s_lyears (starting longevity years)
    # add a monthly increment based on the month number (mnum)
    # convert to an integer which rounds toward zero
    # clip to min of 1 and max of top_of_scale (max pay longevity scale)
    skel['scale'] = np.clip(
        ((skel['mnum'] *
          month_inc) +
         skel['s_lyears']).astype(int),
        1,
        cf.top_of_scale)
    skel.pop('s_lyears')

    # this column is only used for calculating furloughed employee pay
    # longevity in compute_measures routine.
    # ...could be an option if recalls are not part of model
    df['s_lmonths'] = f.longevity_at_startdate(list(df['ldate']),
                                               return_months=True)
    skel['s_lmonths'] = df.s_lmonths

# AGE

# calculate monthly age using starting age and month number

age_list = np.array(skel.s_age)

corr_ages = f.age_correction(long_form_skeleton, age_list)

skel['age'] = corr_ages

skel.pop('s_age')

# empkey index (keep empkey column)
# this is for easy data alignment with different list order keys

# save results to pickle
if cf.save_to_pickle:
    skel.to_pickle(skel_path_string)
