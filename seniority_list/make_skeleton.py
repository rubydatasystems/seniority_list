#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# seniority_list is an analytical tool used when seniority-based work
# groups merge. It brings modern data science to the area of labor
# integration, utilizing the powerful data analysis capabilities of Python
# scientific computing.

# Copyright (C) 2016-2017  Robert E. Davison, Ruby Data Systems Inc.
# Please direct consulting inquires to: rubydatasystems@fastmail.net

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

'''produces a dataframe with information for each employee that is not
dependent on the ORDER of the list...

some column(s) are dependent on the settings dictionary values, such as
pay raise beyond the contract last year.
'''

import pandas as pd
import numpy as np

import functions as f


def main():

    # read prepared list dataframe - proper column headers, column formats...
    # this is master.pkl, order-independent, concatenated list data
    pre, suf = 'dill/', '.pkl'
    master_list = 'master'
    master_path = (pre + master_list + suf)

    try:
        df_mlist = pd.read_pickle(master_path)
    except OSError:
        print('\nMaster list not found.  Run build_program_files script?\n\n' +
              'Skeleton build failed.\n\n' +
              '  >>> exiting routine.\n')
        import sys
        sys.exit()

    output_name = 'skeleton'
    skel_path_string = (pre + output_name + suf)

    sdict = pd.read_pickle('dill/dict_settings.pkl')

    # only include pilots that are not retired prior to the starting_month
    start_date = sdict['starting_date']

    df_mlist = df_mlist[
        df_mlist.retdate >= start_date - pd.DateOffset(months=1)]

    # include furloughees by default
    df = df_mlist[(df_mlist.line == 1) | (df_mlist.fur == 1)].copy()

    df_mlist = []

    # MNUM*
    # calculate the number of career months for each employee (short_form)
    # cmonths is used for mnum, idx, and mth_pcnt calculations

    cmonths = f.career_months(df, sdict['starting_date'])
    # convert the python cmonths list to a numpy array and
    # use that array as input for the count_per_month function.
    # The count_per_month function output array is input for
    # other functions (month_form)

    nonret_each_month = f.count_per_month(cmonths)

    # first long form data generation.
    # month numbers, same month number repeated for each
    # month length (long_form)

    long_form_skeleton = f.gen_month_skeleton(nonret_each_month)

    # this is making a dataframe out of the
    # long_form_skeleton (months) created above.
    # this is the basis for the long_form dataframe...

    # MNUM
    # (month number)

    skel = pd.DataFrame(long_form_skeleton.astype(int), columns=['mnum'])

    # IDX*
    # grab emp index for each remaining
    # employee for each month - used for merging dfs later

    empkey_arr = df.empkey.values

    long_index, long_emp = f.gen_skel_emp_idx(nonret_each_month,
                                              cmonths, empkey_arr)

    # IDX
    skel['idx'] = long_index.astype(int)

    # EMPKEY
    skel['empkey'] = long_emp.astype(int)

    # grab retdates from df column (short_form)
    # used for mth_pcnt and age calc (also mapping retdates)
    dobs = list(df['dob'])

    df_last = pd.read_pickle('dill/last_month.pkl')

    df.set_index('retdate', inplace=True)
    df['lmonth_pcnt'] = df_last.last_pay
    df.reset_index(inplace=True)
    df.set_index('empkey', inplace=True, verify_integrity=False, drop=False)

    lmonth_pcnt = df.lmonth_pcnt.values

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

    df_dates = pd.DataFrame(pd.date_range(sdict['starting_date'],
                                          periods=len(nonret_each_month),
                                          freq='M'), columns=['date'])

    # This is an input for the contract_pay_and_year_and_raise function

    # date_series = pd.to_datetime(list(df_dates['date']))

    # this function produces a 2-column array.
    # First column is the year value of the date list passed as an input.
    # The second column is either 1.0 or
    # a calculated percentage pay raise after the last contract year.

    if sdict['compute_pay_measures']:
        df_dates = f.contract_year_and_raise(df_dates, sdict)

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

    s_age = f.starting_age(dobs, sdict['starting_date'])
    df['s_age'] = s_age

    # data alignment magic...set index to empkey
    skel.set_index('empkey', inplace=True, verify_integrity=False, drop=False)

    # AGE, RETDATE, EG, DOH, LDATE, LNAME,
    # FUR, RET_MONTH to long_form skeleton
    skel['s_age'] = df.s_age
    skel['fur'] = df.fur

    if sdict['add_eg_col']:
        skel['eg'] = df.eg
    if sdict['add_retdate_col']:
        skel['retdate'] = df.retdate
    if sdict['add_doh_col']:
        skel['doh'] = df.doh
    if sdict['add_ldate_col']:
        skel['ldate'] = df.ldate
    if sdict['add_lname_col']:
        skel['lname'] = df.lname
    if sdict['add_line_col']:
        skel['line'] = df.line
    if sdict['add_sg_col']:
        skel['sg'] = df.sg

    # RET_MARK
    # add last month number to df
    df['ret_month'] = cmonths
    # data align to long-form skel
    skel['ret_mark'] = df.ret_month
    mnums = skel.mnum.values
    lmonth_arr = np.zeros(mnums.size).astype(int)
    ret_month = skel.ret_mark.values
    # mark array where retirement month is equal to month number
    np.put(lmonth_arr, np.where(ret_month == mnums)[0], 1)
    skel['ret_mark'] = lmonth_arr

    # SCALE*

    if sdict['compute_pay_measures']:

        df['s_lyears'] = f.longevity_at_startdate(list(df['ldate']),
                                                  sdict['starting_date'])
        skel['s_lyears'] = df.s_lyears

        month_inc = (1 / 12)

        # scale is payrate longevity level
        # compute scale for each employee for each month
        # begin with s_lyears (starting longevity years)
        # add a monthly increment based on the month number (mnum)
        # convert to an integer which rounds toward zero
        # clip to min of 1 and max of top_of_scale (max pay longevity scale)
        skel['scale'] = np.clip(((skel['mnum'] * month_inc) +
                                skel['s_lyears']).astype(int),
                                1,
                                sdict['top_of_scale'])
        skel.pop('s_lyears')

        # this column is only used for calculating furloughed employee pay
        # longevity in compute_measures routine.
        # ...could be an option if recalls are not part of model
        df['s_lmonths'] = f.longevity_at_startdate(list(df['ldate']),
                                                   sdict['starting_date'],
                                                   return_as_months=True)
        skel['s_lmonths'] = df.s_lmonths

    # AGE

    # calculate monthly age using starting age and month number

    age_list = skel.s_age.values

    corr_ages = f.age_correction(long_form_skeleton,
                                 age_list,
                                 sdict['ret_age'])

    if sdict['ret_age_increase']:
        skel['age'] = f.clip_ret_ages(sdict['ret_incr_dict'],
                                      sdict['init_ret_age'],
                                      skel.date.values, corr_ages)
    else:
        skel['age'] = corr_ages

    skel.pop('s_age')

    # empkey index (keep empkey column)
    # this is for easy data alignment with different list order keys

    # save results to pickle
    if sdict['save_to_pickle']:
        skel.to_pickle(skel_path_string)

    # END OF SKELETON GENERATION


if __name__ == "__main__":
    main()
