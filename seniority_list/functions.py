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

'''The functions module contains core program routines related
to building and working with the data model and associated files.
General definitions:
dataset "month_form" is length n months in model
"short_form" data has a length equal to the number of employees
"long_form" data is the length of the cumulative sum non-retired employees
for all months in the data model (could be millions of rows,
depending on workgroup size and age)
'''

import os
import shutil
import copy
import random
import pandas as pd
import numpy as np
import scipy.stats as st
from numba import jit
from collections import OrderedDict as od


# CAREER MONTHS
def career_months(ret_input, start_date):
    '''(Short_Form)
    Determine how many months each employee will work
    including retirement partial month
    "ret_input" (retirement dates) may be in the form of a pandas dataframe,
    pandas series, array, list, or string
    Output is a numpy array of integers containing the number of months
    between the start_date and each date in the ret_input (months from
    start date to retirement for each employee)
    inputs
        ret_input (dataframe, series, array, list, or string)
            retirement dates input
        start_date (string date)
            comparative date for the retirement dates input, normally the
            data model starting date
    '''
    start_date = pd.to_datetime(start_date)
    s_year = start_date.year
    s_month = start_date.month

    ret_list = convert_to_datetime(ret_input, 'retdate')

    retyears = ret_list.year.values
    retmonths = ret_list.month.values

    cmths = ((retyears - s_year) * 12) - (s_month - retmonths)

    return cmths


# LONGEVITY AT STARTDATE (for pay purposes)
def longevity_at_startdate(ldate_input,
                           start_date,
                           return_as_months=False):
    ''' (Short_Form)
    determine how much longevity (years) each employee has accrued as of the
    start date
    float output is longevity in years (+1 added to reflect current 1-based
    pay year)
    inputs
        ldate_input (dataframe, series, list, or string)
            list of longevity dates in datetime format
        start_date (string date)
            comparative date for longevity dates, normally the data model
            starting date
        return_as_months (boolean)
            option to return result as month value instead of year value
    '''
    start_date = pd.to_datetime(start_date)
    s_year = start_date.year
    # subtract one month so pay increase begins
    # in month after anniversary month
    s_month = start_date.month - 1

    ldate_list = convert_to_datetime(ldate_input, 'ldate')

    longevity_list = []

    if return_as_months:
        for ldate in ldate_list:
            longevity_list.append((((s_year - ldate.year) * 12) -
                                   (ldate.month - s_month)) + 1)
    else:
        for ldate in ldate_list:
            longevity_list.append(((((s_year - ldate.year) * 12) -
                                    (ldate.month - s_month)) / 12) + 1)

    return longevity_list


# AGE AT START DATE
def starting_age(dob_input, start_date):
    '''Short_Form
    Returns decimal age at given date.
    "dob_input" (birth dates) may be in the form of a pandas dataframe,
    pandas series, list, or string
    inputs
        dob_list (dataframe, series, list, or string)
            birth dates input
        start_date
            comparative date for the birth dates, normally the data model
            starting date
    '''
    start_date = pd.to_datetime(start_date)
    s_year = start_date.year
    s_month = start_date.month
    s_day = start_date.day
    m_val = 1 / 12

    dob_list = convert_to_datetime(dob_input, 'dob')

    age_list = []
    for dob in dob_list:
        age_list.append(m_val * (((s_year - dob.year) * 12) -
                                 (dob.month - s_month) +
                                 ((s_day - dob.day) / s_day)))

    return age_list


def convert_to_datetime(date_data, attribute):
    '''Convert a dataframe column, series, list, or string input into an array
    of datetimes
    inputs
        data_data (dataframe, series, array, list, or string)
            pandas dataframe with a date column containing string dates or
            datetime objects, pandas series of dates
            (strings or datetime objects), a list/array of date strings or
            datetime objects, or a single comma-separated string containing
            date information.
        attribute (string)
            if the date_data type is a dataframe, the name of the column
            containing the date information.  Otherwise, this input is
            ignored.
    '''
    in_type = type(date_data)

    if in_type == pd.core.frame.DataFrame:
        date_list = pd.to_datetime(date_data[attribute].values)
    if in_type == pd.core.frame.Series:
        date_list = pd.to_datetime(date_data.values)
    if in_type in [np.ndarray, list]:
        try:
            date_list = pd.to_datetime(date_data)
        except:
            print('\nError:\n\n longevity_at_startdate function\n')
    if in_type == str:
        try:
            stripped = [x.strip("[]Timestamp(' '', freq='M')")
                        for x in date_data.split(',')]
            str_list = list(filter(None, stripped))
            date_list = pd.to_datetime(str_list)
        except:
            print('\nError:\n\n longevity_at_startdate function\n')

    return date_list


# COUNT PER MONTH
def count_per_month(career_months_array):
    '''Month_Form
    Returns number of employees remaining for each month (not retired).
    Cumulative sum of career_months_array input (np array) that are
    greater or equal to each incremental loop month number.
    Note: alternate method to this function is value count of mnums:
    df_actives_each_month = pd.DataFrame(df_idx.mnum.value_counts())
    df_actives_each_month.columns = ['count']
    input
        career_months_array
            output of career_months function.  This input is an array
            containing the number of months each employee will work until
            retirement.
    '''
    max_career = career_months_array.max() + 1
    emp_count_array = np.zeros(max_career)

    for i in range(0, max_career):
        emp_count_array[i] = np.count_nonzero(career_months_array >= i)

    return emp_count_array.astype(int)


# GENERATE MONTH SKELETON
@jit(nopython=True, cache=True)
def gen_month_skeleton(month_count_array):
    '''Long_Form
    Create an array of month numbers with each month number
    repeating n times for n non-retired employees in each month.
    i.e. the first month section of the array will be all zeros
    (month: 0) repeating for the number of non-retired employees.
    The next section of the array will be all ones (month: 1)
    repeating for the number of employees remaining in month 1.
    Output is a 1d ndarray.
    This funtion creates the first column and the basic form
    of the skeleton dataframe which is the basis for the dataset dataframes.
    input
        month_count_array
            a numpy array containing the number of employees remaining or
            not retired for each month.  This input is the result of the
            count_per_month function.
    '''
    total_months = int(np.sum(month_count_array))
    mnum_skeleton_array = np.zeros(total_months)
    i = 0
    j = 0
    for mcount in month_count_array:
        for slot in np.arange(j, int(mcount) + j):
            mnum_skeleton_array[slot] = i
        i += 1
        j = slot + 1

    return mnum_skeleton_array


# GENERATE THE EMPLOYEE INDEX SKELETON
@jit(nopython=True, cache=True)
def gen_skel_emp_idx(monthly_count_array,
                     career_mths_array,
                     empkey_source_array):
    '''Long_Form
    For each employee who remains for each month,
    grab that employee index number.
    This index will be the key to merging in other data using data alignment.
    Input is the result of the count_per_month function (np.array)
    and the result of the career_months function
    inputs
        monthly_count_array (numpy array)
            count of non-retired active employees for each month in the model,
            the ouput from the count_per_month function.
        career_mths_array (numpy array)
            career length in months for each employee, output of
            career_months functions.
        empkey_source_array (numpy array)
            empkey column data as array
    Returns tuple (skel_idx_array, skel_empkey_array)
    '''
    total_months = int(np.sum(monthly_count_array))
    skel_idx_array = np.empty(total_months)
    skel_empkey_array = np.empty(total_months)
    emp_idx = np.arange(0, career_mths_array.size)

    k = 0
    # look in career months list for each month
    for j in range(int(np.max(career_mths_array)) + 1):
        idx = 0
        for i in emp_idx:
            if career_mths_array[i] >= j:
                skel_idx_array[k] = idx
                skel_empkey_array[k] = empkey_source_array[idx]
                k += 1
            idx += 1

    return skel_idx_array, skel_empkey_array


# AGE FOR EACH MONTH (correction to starting age)
def age_correction(month_nums_array,
                   ages_array,
                   retage):
    '''Long_Form
    Returns a long_form (all months) array of employee ages by
    incrementing starting ages according to month number.
    Note:  Retirement age increases are handled by the build_program_files
    script by incrementing retirement dates and by the clip_ret_ages
    function within the make_skeleton script.
    inputs
        month_nums_array (array)
            gen_month_skeleton function output (ndarray)
        ages_array (array)
            starting_age function output aligned with long_form (ndarray)
            i.e. s_age is starting age (aligned to empkeys)
            repeated each month.
        retage (integer or float)
            output clip upper limit for retirement age
    Output is s_age incremented by a decimal month value according to month_num
    (this is candidate for np.put refactored function)
    '''
    month_val = 1 / 12
    result_array = (month_nums_array * month_val) + ages_array
    result_array = np.clip(result_array, 0, retage)

    return result_array


# FIND CONTRACT PAY YEAR AND RAISE (contract pay year
# and optional raise multiplier)
def contract_year_and_raise(df, settings_dict):
    '''(Month_Form)
    Generate the contract pay year for indexing into the pay table.
    Pay year is clipped to last year of contract.
    Also create an annual assumed raise column applicable to the time period
    beyond the contract duration.  This is a multiplier column with a
    compounded value each subsequent year.  If no raise is elected (via the
    settings.xlsx input file, "scalars" worksheet), then this column will be
    all ones.  The annual raise percentage is designated on the same worksheet.
    The input df must be a single column dataframe containing end-of-month
    dates, one for each month of the data model.
    NOTE: (this function can accept any number of pay exception periods
    through the pay_exceptions dictionary, populated by the "pay_exceptions"
    worksheet values within the *settings.xlsx* input file, see the
    program documentation for more information)
    inputs
        df (dataframe)
            a single column dataframe containing end-of-month dates,
            one for each month of the data model
        settings_dict (dictionary)
            dictionary of program settings generated by the
            *build_program_files* script
    '''
    df.columns = ['date']
    df.date = pd.to_datetime(df.date)
    df['year'] = df.date.dt.year
    exception_dict = settings_dict['pay_exceptions']
    annual_raise = settings_dict['annual_pcnt_raise']
    pay_table_years = settings_dict['contract_years']
    last_contract_yr = settings_dict['contract_end']
    future_raise = settings_dict['future_raise']

    if annual_raise and future_raise:
        df['pay_raise'] = (1 + annual_raise) ** \
            (df.year - int(last_contract_yr))
        df.pay_raise = np.clip(df.pay_raise, 1, 100000)
    else:
        df['pay_raise'] = 1

    df['year'] = df.date.dt.year

    if exception_dict:
        for year_code in exception_dict.keys():
            if year_code in pay_table_years:
                df.loc[(df['date'] >= exception_dict[year_code][0]) &
                       (df['date'] <= exception_dict[year_code][1]),
                       'year'] = year_code
            else:
                print('\nPay year exception error:\n\n',
                      'The pay table input data does not contain rates for: "',
                      year_code, '"\n'
                      'mpay and cpay calculations skipped for',
                      'the affected time period.\n\n',
                      'The excel pay table "rates" worksheet contains\n',
                      'information for the following year codes:\n',
                      pay_table_years, '\n\n'
                      'Check the settings.xlsx "pay_exceptions" worksheet',
                      'inputs \nand/or update the pay_tables.xlsx "rates"',
                      'worksheet\n')
        df.year = np.clip(df.year, 0, last_contract_yr)

    return df


# MAKE eg INITIAL JOB LIST from job_count_array (Stovepipe)
def make_stovepipe_jobs_from_jobs_arr(jobs_arr,
                                      total_emp_count=0):
    '''Month_Form
    Compute a stovepipe job list derived from the total
    count of jobs in each job level.
    This function is for one eg (employee group) and one jobs_arr (list).
    Creates an array of job numbers from a
    job count list (converted to np.array).
    Result is an array with each job number repeated n times for n job count.
    - job count list like : job_counts = [334, 222, 701, 2364]
    - jobs_array = np.array(job_counts)
    inputs
        jobs_arr (numpy array)
            job counts starting with job level 1
        total_emp_count
            if zero (normal input), sum of jobs_arr elements,
            otherwise user-defined size of result_jobs_arr
    '''
    if total_emp_count == 0:
        result_jobs_arr = np.zeros(sum(jobs_arr))
    else:
        result_jobs_arr = np.zeros(total_emp_count)

    i = 1
    j = 0

    # this loop is faster than a np.repeat routine...
    for job_quant in jobs_arr:

        if job_quant > 0:
            result_jobs_arr[j: j + job_quant] = i
            j = j + job_quant

        # increment job number for next loop
        i += 1

    return result_jobs_arr.astype(int)


# MAKE integrated INITIAL JOB LIST from eg stovepipe job arrays
def make_intgrtd_from_sep_stove_lists(job_lists_arr,
                                      eg_arr,
                                      fur_arr,
                                      eg_total_jobs,
                                      num_levels,
                                      skip_fur=True):
    '''Month_Form
    Compute an integrated job list built from multiple
    independent eg stovepiped job lists.
    This function is for multiple egs (employee groups) - multiple lists in
    one job_lists_arr.
    Creates an ndarray of job numbers.
    Function takes independent job number lists and an array of eg codes
    which represent the eg ordering in the proposed list.
    Job numbers from the separate lists are added to the result array
    according to the eg_arr order.  Jobs on each list do not have to be
    in any sort of order.  The routine simply adds items from the list(s)
    to the result array slots in list order.
    inputs
        job_lists_arr
            array of the input job number arrays.
            represents the jobs that would be assigned to each employee
            in a list form.
            each list within the array will be the length of the
            respective eg.
        eg_arr
            short_form array of eg codes (proposal eg ordering)
        fur_arr
            short_form array of fur codes from proposal
        eg_total_jobs
            list length n egs
            sums of total jobs available for each eg, form: [n,n,n]
        num_levels
            number of job levels in model (excluding furlough level)
        skip_fur (boolean)
        skip_fur option:
            ignore or skip furloughs when assigning stovepipe jobs.
            If True, employees who are originally marked as furloughed are
            assigned the furlough level number which is 1 greater
            than the number of job levels.
            If False, jobs are assigned within each employee group in a
            stovepipe fashion, including those employees who are marked
            as furloughed
    '''
    result_jobs_arr = np.zeros(eg_arr.size)

    if skip_fur:

        for i in range(len(job_lists_arr)):

            job_indexes = np.where((eg_arr == (i + 1)) & (fur_arr == 0))[0]

            np.put(result_jobs_arr,
                   job_indexes[:eg_total_jobs[i]],
                   job_lists_arr[i])

            np.put(result_jobs_arr,
                   np.where(result_jobs_arr == 0)[0],
                   num_levels + 1)

    else:

        for i in range(len(job_lists_arr)):

            job_indexes = np.where(eg_arr == (i + 1))[0]

            np.put(result_jobs_arr,
                   job_indexes[:eg_total_jobs[i]],
                   job_lists_arr[i])

            np.put(result_jobs_arr,
                   np.where(result_jobs_arr == 0)[0],
                   num_levels + 1)

    return result_jobs_arr.astype(int)


# MAKE_STOVEPIPE_JOBS_WITH_PRE=EXISTING CONDITION
# (Stovepipe with internal condition stovepiped, SHORT_FORM)
def make_stovepipe_prex_shortform(job_list,
                                  sg_codes,
                                  sg_rights,
                                  fur_codes):
    '''Short_Form
    Creates a 'stovepipe' job assignment within a single eg including a
    special job assignment condition for a subgroup.  The subgroup is
    identified with a 1 in the sg_codes array input, originating with
    the sg column in the master list.
    This function applies a pre-existing (prior to the merger)
    contractual job condition, which is likely the result of a previous
    seniority integration.
    The subset group will have proirity assignment for the first n jobs
    in the affected job category, the remainding jobs
    are assigned in seniority order.
    The subgroup jobs are assigned in subgroup stovepipe order.
    This function is applicable to a condition with known job counts.
    The result of this function is used with standalone calculations or
    combined with other eg lists to form an integrated original
    job assignment list.
    inputs
        job_list
            list of job counts for eg, like [23,34,0,54,...]
        sg_codes
            ndarray
            eg group members entitled to job condition
            (marked with 1, others marked 0)
            length of this eg population
        sg_rights
            list of lists (from settings dictionary) including job numbers and
            job counts for condition.
            Columns 2 and 3 are extracted for use.
        fur_codes
            array of ones and zeros, one indicates furlough status
    '''
    o_job = np.zeros(sg_codes.size)
    this_count = 0
    job = 0
    sg_jobs_and_counts = [
        np.array(sg_rights).astype(int)[:, 1],
        np.array(sg_rights).astype(int)[:, 2]]

    for i in job_list:

        job += 1

        if job in sg_jobs_and_counts[0]:

            sg_allotment = sg_jobs_and_counts[1][this_count]

            np.put(o_job,
                   np.where((sg_codes == 1) &
                            (o_job == 0) &
                            (fur_codes == 0))[0]
                   [:sg_allotment],
                   job)

            np.put(o_job,
                   np.where((o_job == 0) & (fur_codes == 0))[0]
                   [:(i - sg_allotment)],
                   job)

            this_count += 1

        else:
            np.put(o_job, np.where((o_job == 0) &
                                   (fur_codes == 0))[0][:i], job)

    return o_job.astype(int)


# MAKE LIST OF ORIGINAL JOBS
def make_original_jobs_from_counts(jobs_arr_arr,
                                   eg_array,
                                   fur_array,
                                   num_levels):
    '''Short_Form
    This function grabs jobs from standalone job count
    arrays (normally stovepiped) for each employee group and inserts
    those jobs into a proposed integrated list, or a standalone list.
    Each eg (employee group) is assigned jobs from their standalone
    list in order top to bottom.
    Result is a combined list of jobs with each eg maintaining ordered
    independent stovepipe jobs within the combined list of jobs
    jobs_arr_arr is an array of arrays, likely output[0] from
    make_array_of_job_lists function.
    Order of job count arrays within jobs_arr_arr input
    must match emp group codes order (1, 2, 3, etc.).
    If total group counts of job(s) is less than slots available to that group,
    remaining slots will be assigned (remain) a zero job number (0).
    eg_array is list (order sequence) of employee group codes from proposed
    list with length equal to length of proposed list.
    Result of this function is ultimately merged into long form
    for no bump no flush routine.
    employees who are originally marked as furloughed are assigned the furlough
    level number which is 1 greater than the number of job levels.
    inputs
        jobs_arr_arr (numpy array of arrays)
            lists of job counts for each job level within each employee
            group, each list in order starting with job level one.
        eg_array (numpy array)
            employee group (eg) column data from master list source
        fur_array
            furlough (fur) column data from master list source
        num_levels
            number of job levels (without furlough level) in the model
    '''
    result_jobs_arr = np.zeros(eg_array.size)
    eg = 0

    for job_arr in jobs_arr_arr:

        eg += 1
        this_job_list = np.repeat((np.arange(len(job_arr)) + 1), job_arr)

        np.put(result_jobs_arr,
               np.where((eg_array == eg) &
                        (fur_array == 0))[0][:sum(job_arr)],
               this_job_list)

        np.put(result_jobs_arr,
               np.where(result_jobs_arr == 0)[0],
               num_levels + 1)

    return result_jobs_arr.astype(int)


# ASSIGN JOBS STANDALONE WITH JOB CHANGES and prex option
def assign_standalone_job_changes(eg,
                                  df_align,
                                  lower,
                                  upper,
                                  total_months,
                                  job_counts_each_month,
                                  total_monthly_job_count,
                                  nonret_each_month,
                                  job_change_months,
                                  job_reduction_months,
                                  start_month,
                                  sdict,
                                  tdict,
                                  apply_sg_cond=True):
    '''(Long_Form)
    Uses the job_gain_or_loss_table job count array for job assignments.
    Jobs counts may change up or down in any category for any time period.
    Handles furlough and return of employees, prior rights/conditions and
    restrictions and recall of initially furloughed employees.
    Inputs are precalculated outside of function to the extent possible.
    Returns tuple (long_assign_column, long_count_column, held_jobs,
    fur_data, orig_jobs)
    inputs
        eg (integer)
            input from an incremental loop which is used to select the proper
            employee group recall scedule
        df_align (dataframe)
            dataframe with ['sg', 'fur'] columns
        num_of_job_levels (integer)
            number of job levels in the data model (excluding a furlough
            level)
        lower (1d array)
            ndarry from make_lower_slice_limits function
            (calculation derived from cumsum of count_per_month function)
        upper (1d array)
            cumsum of count_per_month function
        total_months (integer or float)
            sum of count_per_month function output
        job_counts_each_month (array)
            output of job_gain_loss_table function[0]
            (precalculated monthly count of jobs in each job category,
            size (months,jobs))
        total_monthly_job_count (array)
            output of job_gain_loss_table function[1]
            (precalculated monthly total count of all job categories,
            size (months))
        nonret_each_month (1d array)
            output of count_per_month function
        job_change_months (list)
            the min start month and max ending month found within the
            array of job_counts_each_month inputs
            (find the range of months to apply consideration for
            any job changes - prevents unnecessary looping)
        job_reduction_months (list)
            months in which the number of jobs is decreased (list).
            from the get_job_reduction_months function
        start_month (integer)
            starting month for calculations, likely implementation month
            from settings dictionary
        sdict (dictionary)
            the program settings dictionary (produced by the
            build_program_files script)
        tdict (dictionary)
            job tables dictionary (produced by the build_program_files script)
        apply_sg_cond (boolean)
            compute with pre-existing special job quotas for certain
            employees marked with a one in the sg column (special group)
            according to a schedule defined in the settings dictionary
    Assigns jobs so that original standalone jobs are assigned
    each month (if available) unless a better job is available
    through attrition of employees.
    Each month loop starts with the lowest job number.
    For each month and for each job level:
        1. assigns nbnf (orig) job if job array (long_assign_column) element
        is zero (unassigned) and orig job number is less than or
        equal to the job level in current loop, then
        2. assigns job level in current loop to unassigned slots from
        top to bottom in the job array (up to the count of that
        job level remaining after step one above)
    Each month range is determined by slicing using the lower and upper inputs.
    A comparison is made each month between the original job numbers and the
    current job loop number.
    Job assignments are placed into the monthly segment
    (assign_range) of the long_assign_column.
    The long_assign_column eventually becomes the job number (jnum) column
    in the dataset.
    Original job numbers of 0 indicate no original job and are
    treated as furloughed employees - no jobs are assigned
    to furloughees unless furlough_return option is selected.
    '''
    num_of_job_levels = sdict['num_of_job_levels']
    fur_return = sdict['recall']
    sg_rights = sdict['sg_rights']
    recalls = sdict['recalls']

    sg_ident = df_align.sg.values
    fur_data = df_align.fur.values
    index_data = df_align.index.values

    lower_next = lower[1:]
    lower_next = np.append(lower_next, lower_next[-1])

    upper_next = upper[1:]
    upper_next = np.append(upper_next, upper_next[-1])

    # job assignment result array/column
    long_assign_column = np.zeros(total_months, dtype=int)
    # job counts result array/column
    long_count_column = np.zeros(total_months, dtype=int)
    # job held col
    held_jobs = np.zeros(total_months, dtype=int)

    num_of_months = upper.size
    this_eg_sg = None

    if apply_sg_cond:

        sg_rights = np.array(sg_rights)
        sg_egs = np.unique(np.transpose(sg_rights)[0])
        if eg in sg_egs:
            this_eg_sg = True
            sg_jobs = np.transpose(sg_rights)[1]
            sg_counts = np.transpose(sg_rights)[2]
            sg_dict = dict(zip(sg_jobs, sg_counts))

            # calc sg prex condition month range and concat
            sg_month_range = np.arange(np.min(sg_rights[:, 3]),
                                       np.max(sg_rights[:, 4]))
            job_change_months = np.concatenate((job_change_months,
                                                sg_month_range))

    if fur_return:

        recall_months = get_recall_months(recalls)
        job_change_months = np.concatenate((job_change_months,
                                            recall_months))

    job_change_months = np.unique(job_change_months)

    for month in range(start_month, num_of_months):

        L = lower[month]
        U = upper[month]

        L_next = lower_next[month]
        U_next = upper_next[month]

        held_job_range = held_jobs[L:U]
        assign_range = long_assign_column[L:U]
        job_count_range = long_count_column[L:U]
        fur_range = fur_data[L:U]
        sg_range = sg_ident[L:U]
        index_range = index_data[L:U]
        index_range_next = index_data[L_next:U_next]

        # use numpy arrays for job assignment process for each month
        # use pandas for data alignment 'job position forwarding'
        # to future months

        # this_job_col = 0
        job = 1

        if month in job_reduction_months:
            mark_for_furlough(held_job_range, fur_range, month,
                              total_monthly_job_count, num_of_job_levels)

        if fur_return and (month in recall_months):
            mark_for_recall(held_job_range, num_of_job_levels,
                            fur_range, month, recalls,
                            total_monthly_job_count,
                            standalone=True,
                            eg_index=eg - 1)

        while job <= num_of_job_levels:

            this_job_count = job_counts_each_month[month, job]

            if month in job_change_months:

                if this_eg_sg:

                    if month in sg_month_range and job in sg_jobs:

                        # assign prex condition jobs to sg employees
                        sg_jobs_avail = min(sg_dict[job], this_job_count)
                        np.put(assign_range,
                               np.where((assign_range == 0) &
                                        (sg_range == 1) &
                                        (fur_range == 0))[0][:sg_jobs_avail],
                               job)

            # TODO, (for developer) code speedup...
            # use when not in condition month and monotonic is true
            # (all nbnf distortions gone, no job count changes)
            # if (month > max(job_change_months))
            # and monotonic(assign_range):
            #     quick_stopepipe_assign()

            jobs_avail = count_avail_jobs(assign_range,
                                          job,
                                          this_job_count)
            np.put(assign_range,
                   np.where((assign_range == 0) &
                            (held_job_range <= job) &
                            (fur_range == 0))[0][:jobs_avail],
                   job)

            jobs_avail = count_avail_jobs(assign_range,
                                          job,
                                          this_job_count)
            np.put(assign_range,
                   np.where((assign_range == 0) &
                            (fur_range == 0))[0][:jobs_avail],
                   job)

            assign_job_counts(job_count_range,
                              assign_range,
                              job,
                              this_job_count)

            # this_job_col += 1
            job += 1

        # AFTER MONTHLY JOB LOOPS DONE, PRIOR TO NEXT MONTH:

        # pass down assign_range
        held_jobs[L_next:U_next] = \
            align_next(index_range, index_range_next, assign_range)

        # pass down fur_range
        #  TODO (for developer)**
        # eliminate this furlough pass down...
        # all fur data including future month fur status
        # can be precalculated with headcount,
        # job counts and fur return schedule

        # unassigned marked as fur
        mark_fur_range(assign_range, fur_range, num_of_job_levels)

        np.put(job_count_range,
               np.where(fur_range == 1)[0],
               np.count_nonzero(fur_range == 1))

        fur_data[L_next:U_next] = \
            align_next(index_range, index_range_next, fur_range)

    long_assign_column[long_assign_column == 0] = num_of_job_levels + 1
    held_jobs[held_jobs == num_of_job_levels + 1] = 0
    orig_jobs = long_assign_column[lower[0]:upper[0]]

    return long_assign_column.astype(int), long_count_column.astype(int), \
        held_jobs.astype(int), fur_data.astype(int), orig_jobs.astype(int)


# ASSIGN JOBS FULL FLUSH with JOB COUNT CHANGES
def assign_jobs_full_flush_job_changes(nonret_counts,
                                       job_counts,
                                       num_job_levels):
    '''(Long_Form)
    Using the nonret counts for each month:
      a. determine the long form slice for assignment, and
      b. slice the jobs list from the top to create job assignment column
      c. create a corresponding furlough column
      d. create a job count column
    Uses the job_counts (job_gain_loss_table function)[0] to
    build stovepiped job lists allowing for job count changes each month
    and a furlough status column
    Unassigned employees (not enough jobs), are left at job number zero
    This is the full bump and full flush version
    inputs
        nonret_counts (numpy array)
            array containing the number of non-retired employees
            for each month
        job_counts (numpy array)
            array containing the monthly counts of jobs for each job level
        num_job_levels (integer)
            the number of job levels in the model (excluding furlough level)
    '''
    long_job_array = np.zeros(sum(nonret_counts))
    long_count_array = np.zeros(long_job_array.size)
    jobs_arr = np.arange(1, num_job_levels + 1)
    # build low and high array indexes
    cumsum = nonret_counts.cumsum()
    cumsum_w0 = np.append(np.array(0), cumsum)
    lows = cumsum_w0[:-1]
    highs = cumsum_w0[1:]

    for i in range(nonret_counts.size):
        # set current loop range indexes
        L = lows[i]
        H = highs[i]
        job_range = long_job_array[L:H]
        count_range = long_count_array[L:H]
        # build stovepipe job array
        job_list = np.repeat(jobs_arr, job_counts[i])
        idx = min(job_range.size, job_list.size)
        job_range[:idx] = job_list[:idx]
        # build job count array
        count_list = np.repeat(job_counts[i], job_counts[i])
        count_range[:idx] = count_list[:idx]
        # count unassigned workers
        fur_count = np.count_nonzero(job_range == 0)
        count_range[count_range == 0] = fur_count

    long_fur_array = np.zeros(long_job_array.size).astype(int)
    # mark unassigned workers with a 1 in furlough array
    long_fur_array[long_job_array == 0] = 1

    long_job_array[long_job_array == 0] = num_job_levels + 1
    return long_job_array.astype(int), long_count_array.astype(int), \
        long_fur_array.astype(int)


# ASSIGN JOBS NBNF JOB CHANGES
def assign_jobs_nbnf_job_changes(df,
                                 lower,
                                 upper,
                                 total_months,
                                 job_reduction_months,
                                 start_month,
                                 condition_list,
                                 sdict,
                                 tdict,
                                 fur_return=False):
    '''(Long_Form)
    Uses the job_gain_or_loss_table job count array for job assignments.
    Jobs counts may change up or down in any category for any time period.
    Handles furlough and return of employees, prior rights/conditions and
    restrictions and recall of initially furloughed employees.
    Inputs are precalculated outside of function to the extent possible.
    Returns tuple (long_assign_column, long_count_column, orig jobs, fur_data)
    inputs
        df (dataframe)
            long-form dataframe with ['eg', 'sg', 'fur', 'orig_job']
            columns.
        lower (array)
            ndarry from make_lower_slice_limits function
            (calculation derived from cumsum of count_per_month function)
        upper (array)
            cumsum of count_per_month function
        total_months (integer or float)
            sum of count_per_month function output
        job_reduction_months (list)
            months in which the number of jobs is decreased
            from the get_job_reduction_months function
        start_month (integer)
            integer representing the month number to begin calculations,
            likely month of integration when there exists a delayed
            integration (from settings dictionary)
        condition_list (list)
            list of special job assignment conditions to apply,
            example: ['prex', 'count', 'ratio']
        sdict (dictionary)
            the program settings dictionary (produced by the
            build_program_files script)
        tdict (dictionary)
            job tables dictionary (produced by the build_program_files script)
        fur_return (boolean)
            model employee recall from furlough if True using recall
            schedule from settings dictionary (allows call to
            mark_for_recall function)
    Assigns jobs so that original standalone jobs are assigned
    each month (if available) unless a better job is available
    through attrition of employees.
    Each month loop starts with the lowest job number.
    For each month and for each job level:
        1. assigns nbnf (orig) job if job array (long_assign_column) element
           is zero (unassigned) and orig job number is less than or
           equal to the job level in current loop, then
        2. assigns job level in current loop to unassigned slots from
           top to bottom in the job array (up to the count of that
           job level remaining after step one above)
    Each month range is determined by slicing using the lower and upper inputs.
    A comparison is made each month between the original job numbers and the
    current job loop number.
    Job assignments are placed into the monthly segment (assign_range)
    of the long_assign_column.
    The long_assign_column eventually becomes the job number (jnum) column
    in the dataset.
    Original job numbers of 0 indicate no original job and are
    treated as furloughed employees.  No jobs are assigned to
    furloughees unless furlough_return option is selected.
    '''
    orig = df.orig_job.values
    eg_data = df.eg.values
    sg_ident = df.sg.values
    fur_data = df.fur.values
    index_data = df.index.values

    lower_next = lower[1:]
    lower_next = np.append(lower_next, lower_next[-1])

    upper_next = upper[1:]
    upper_next = np.append(upper_next, upper_next[-1])

    # job assignment result array/column
    long_assign_column = np.zeros(total_months, dtype=int)
    # job counts result array/column
    long_count_column = np.zeros(total_months, dtype=int)

    num_of_months = upper.size
    num_of_job_levels = sdict['num_of_job_levels']

    job_counts_each_month = add_zero_col(tdict['table'][0])
    total_monthly_job_count = tdict['table'][1]
    loop_check = tdict['loop_check']

    if sdict['delayed_implementation']:
        long_assign_column[:upper[start_month]] = \
            orig[:upper[start_month]]

    job_change_months = sdict['jc_months']

    if 'prex' in condition_list:

        sg_rights = np.array(sdict['sg_rights'])

        sg_jobs = np.transpose(sg_rights)[1].astype(int)
        sg_counts = np.transpose(sg_rights)[2].astype(int)
        sg_dict = dict(zip(sg_jobs, sg_counts))

        # calc sg prex condition month range and concat
        sg_month_range = sdict['prex_month_range']
        job_change_months = job_change_months.union(sg_month_range)

    # calc ratio condition month range and concat to
    # job_change_months
    if 'ratio' in condition_list:
        ratio_dict = sdict['ratio_dict']
        ratio_jobs = list(ratio_dict.keys())
        ratio_month_range = sdict['ratio_month_range']
        r_mdict = {}
        for job in ratio_dict.keys():
            r_mdict[job] = set(range(ratio_dict[job][2],
                                     ratio_dict[job][3] + 1))

        # ratio_cond_month = min(ratio_month_range)
        job_change_months = job_change_months.union(ratio_month_range)

        snap_ratio_dict = sdict['snap_ratio_dict']
        ratio_items = snap_ratio_dict.items()
        ratio_months = set(snap_ratio_dict.values())
        ratio_on = sdict['snap_ratio_on_off_dict']

        # calc capped count condition month range and concat
    if 'count' in condition_list:
        count_dict = sdict['count_ratio_dict']
        count_jobs = sorted(count_dict.keys())
        cr_mdict = {}
        for job in count_jobs:
            cr_mdict[job] = set(range(count_dict[job][3],
                                      count_dict[job][4] + 1))

        dkeys = {'grp': 0, 'wgt': 1, 'cap': 2}

        count_month_range = sdict['count_ratio_month_range']
        job_change_months = job_change_months.union(count_month_range)

        snap_count_dict = sdict['snap_count_dict']
        count_items = snap_count_dict.items()
        count_months = set(snap_count_dict.values())
        count_on = sdict['snap_count_on_off_dict']

    if fur_return:
        recalls = sdict['recalls']
        recall_months = set(get_recall_months(recalls))
        job_change_months = job_change_months.union(recall_months)
    # convert job_change_months array to a set for faster membership test
    job_change_months = set(job_change_months)

    # loop through model integrated months:
    for month in range(start_month, num_of_months):

        L = lower[month]
        U = upper[month]

        L_next = lower_next[month]
        U_next = upper_next[month]

        orig_job_range = orig[L:U]
        assign_range = long_assign_column[L:U]
        job_count_range = long_count_column[L:U]
        fur_range = fur_data[L:U]
        eg_range = eg_data[L:U]
        sg_range = sg_ident[L:U]
        index_range = index_data[L:U]
        index_range_next = index_data[L_next:U_next]

        job = 1

        if month in job_reduction_months:
            mark_for_furlough(orig_job_range, fur_range, month,
                              total_monthly_job_count, num_of_job_levels)

        if fur_return and (month in recall_months):
            mark_for_recall(orig_job_range, num_of_job_levels,
                            fur_range, month, recalls,
                            total_monthly_job_count, standalone=False)
        # use numpy arrays for job assignment process for each month
        # loop_check is a pre-processed boolean array which prevents looping
        # through job levels after all remaining employees have been assigned
        while job <= num_of_job_levels and loop_check[month, job]:

            this_job_count = job_counts_each_month[month, job]

            if month in job_change_months:

                # **pre-existing condition**
                if 'prex' in condition_list:

                    if (month in sg_month_range) and (job in sg_jobs):

                        sg_jobs_avail = min(sg_dict[job], this_job_count)
                        np.put(assign_range,
                               np.where((assign_range == 0) &
                                        (sg_range == 1) &
                                        (fur_range == 0))[0][:sg_jobs_avail],
                               job)

                # assign ratio count condition jobs
                if 'count' in condition_list:

                    if month in count_months and (job, month) in count_items:

                        if count_on[job]:
                            # adjust count dict values to capture job counts
                            count_dict = set_snapshot_weights(job,
                                                              count_dict,
                                                              orig_job_range,
                                                              eg_range)

                    if (job in count_jobs) and (month in cr_mdict[job]):

                        cap = min(count_dict[job][dkeys['cap']],
                                  this_job_count)

                        assign_cond_ratio(job,
                                          this_job_count,
                                          count_dict,
                                          orig_job_range,
                                          assign_range,
                                          eg_range,
                                          fur_range,
                                          cap=cap)

                # assign ratio condition jobs
                if 'ratio' in condition_list:

                    if month in ratio_months and (job, month) in ratio_items:
                        if ratio_on[job]:
                            # adjust ratio dict values to capture job counts
                            ratio_dict = set_snapshot_weights(job,
                                                              ratio_dict,
                                                              orig_job_range,
                                                              eg_range)

                    if (job in ratio_jobs) and (month in r_mdict[job]):

                        assign_cond_ratio(job,
                                          this_job_count,
                                          ratio_dict,
                                          orig_job_range,
                                          assign_range,
                                          eg_range,
                                          fur_range)

            # TODO, (for developer) code speedup...
            # use when not in condition month and monotonic is true
            # (all nbnf distortions gone, no job count changes)
            # if (month > max(job_change_months))
            # and monotonic(assign_range):
            #     quick_stopepipe_assign()

            # assign no bump, no flush jobs...
            jobs_avail = count_avail_jobs(assign_range,
                                          job,
                                          this_job_count)

            np.put(assign_range,
                   np.where((assign_range == 0) &
                            (orig_job_range <= job) &
                            (fur_range == 0))[0][:jobs_avail],
                   job)

            # assign remaining jobs by list order
            jobs_avail = count_avail_jobs(assign_range,
                                          job,
                                          this_job_count)

            np.put(assign_range,
                   np.where((assign_range == 0) &
                            (fur_range == 0))[0][:jobs_avail],
                   job)

            # insert corresponding job count
            assign_job_counts(job_count_range,
                              assign_range,
                              job,
                              this_job_count)

            job += 1

        # AFTER MONTHLY JOB LOOPS DONE, PRIOR TO NEXT MONTH:

        # pass down assign_range
        orig[L_next:U_next] = \
            align_next(index_range, index_range_next, assign_range)

        # pass down fur_range
        #  TODO (for developer) **
        # eliminate this furlough pass down...
        # all fur data including future month fur status
        # can be precalculated with headcount,
        # job counts and fur return schedule

        # unassigned marked as fur
        mark_fur_range(assign_range, fur_range, num_of_job_levels)

        np.put(job_count_range,
               np.where(fur_range == 1)[0],
               np.count_nonzero(fur_range == 1))

        fur_data[L_next:U_next] = \
            align_next(index_range, index_range_next, fur_range)

    # not part of month loops, cleaning up fur data for output
    long_assign_column[long_assign_column == 0] = num_of_job_levels + 1
    # orig is no longer a function output...
    # orig[orig == num_of_job_levels + 1] = 0

    return long_assign_column.astype(int), long_count_column.astype(int), \
        fur_data.astype(int)


# MAKE LOWER SLICE LIMITS
def make_lower_slice_limits(month_counts_cumsum):
    '''for use when working with unique month data
    within larger array (slice).
    The top of slice is cumulative sum, bottom of each slice
    will be each value of this function output array.
    Output is used as input for nbnf functions.
    input
        month_counts_cumsum (numpy array)
            cumsum of count_per_month function output (employee count
            each month)
    '''
    lower_list = sorted(month_counts_cumsum, reverse=True)
    lower_list.append(0)
    lower_list.sort()
    lower_list.pop()
    return np.array(lower_list).astype(int)


# SNUM, SPCNT, LNUM, LSPCNT with JOB CHANGES
def create_snum_and_spcnt_arrays(jnums,
                                 job_level_count,
                                 monthly_population_counts,
                                 monthly_job_counts,
                                 lspcnt_remaining_only):
    '''Calculates:
    long_form seniority number ('snum', only active employees),
    seniority percentage ('spcnt', only active employees),
    list number ('lnum', includes furlougees),
    list percentage ('lspcnt', includes furloughees).
    Iterate through monthly jobs count data, capturing monthly_job_counts
    to be used as the denominator for percentage calculations.
    This function produces four ndarrays which will make up four columns
    in the long_form pandas dataset.
    Returns tuple (long_snum_array, long_spcnt_array, long_list_array,
    long_lspcnt_array)
    inputs
        jnums
            the long_form jnums result
        job_level_count
            number of job levels in model
        monthly_population_counts
            count_per_month function output
        monthly_job_counts
            total of all jobs each month derived from
            job_gain_loss_table function (table)
            >>> np.add.reduce(table, axis=1)
        lspcnt_remaining_only
            calculate list percentage based on employees remaining
            in each month including furloughees, otherwise
            percentage calculation denominator is the greater of
            employees remaining (incl fur) or jobs available
    '''
    fur_level = job_level_count + 1
    seq_nums = np.arange(1, monthly_population_counts[0] + 1)

    # TODO (for developer)
    # consider np.sum if monthly_population_counts is always np array
    monthly_population = sum(monthly_population_counts)

    long_snum_array = np.zeros(monthly_population)
    long_denom_array = np.zeros(monthly_population)
    long_list_array = np.zeros(monthly_population)
    long_lspcnt_array = np.zeros(monthly_population)
    long_lspcnt_denom = np.zeros(monthly_population)
    long_spcnt_array = np.zeros(monthly_population)

    L = 0

    for i in range(len(monthly_population_counts)):

        this_month_count = monthly_population_counts[i]
        H = this_month_count + L

        if lspcnt_remaining_only:
            monthly_list_denom = monthly_population_counts[i]
        else:
            monthly_list_denom = max(monthly_job_counts[i],
                                     monthly_population_counts[i])

        jnum_range = jnums[L: H]
        snum_range = long_snum_array[L: H]
        denom_range = long_denom_array[L: H]
        posit_range = long_list_array[L: H]
        lspcnt_denom_range = long_lspcnt_denom[L: H]
        spcnt_range = long_spcnt_array[L: H]

        non_fur_indexes = np.where((jnum_range > 0) &
                                   (jnum_range < fur_level))[0]

        np.put(snum_range,
               non_fur_indexes,
               seq_nums[:this_month_count])
        np.copyto(denom_range,
                  monthly_job_counts[i])
        np.copyto(posit_range,
                  seq_nums[:posit_range.size])
        np.copyto(lspcnt_denom_range,
                  monthly_list_denom)
        np.copyto(spcnt_range,
                  snum_range / monthly_job_counts[i])

        L += this_month_count

    long_spcnt_array = long_snum_array / long_denom_array
    long_spcnt_array[long_spcnt_array == 0] = np.nan
    long_snum_array[long_snum_array == 0] = np.nan
    long_lspcnt_array = long_list_array / long_lspcnt_denom

    return long_snum_array, long_spcnt_array, \
        long_list_array.astype(int), long_lspcnt_array


# MAKE JOB COUNTS (without furlough counts)
def make_jcnts(job_count_lists):
    '''Make two arrays:
    1. array of n lists of job counts for n number
    of eg job count input lists
    2. array of one summation list of first array
    (total count of all eg jobs)
    The arrays above will not contain a furlough count.
    Returns tuple (eg_job_counts, combined_job_count)
    inputs
        job_count_lists
            list of the employee job count list(s).
            If the program is using the enhanced jobs option, this input
            will be the output of the convert_jcnts_to_enhanced function.
            Otherwise, it will be the eg_counts variable from the
            settings dictionary.
    Example return:
    .. code:: python
      (array([
      [ 237,  158,  587, 1373,  352,  739,  495,  330,  784,
       1457,    0,  471,  785,    0,    0,    0],
      [  97,   64,  106,  575,   64,  310,  196,  130,  120,
       603,    71,   72,  325,   38,   86,   46],
      [  0,     0,   33,  414,   20,  223,    0,    0,   46,
       395,     0,   28,  213,    0,    0,    0]]),
      array(
      [ 334,  222,  726, 2362,  436, 1272,  691,  460,  950,
       2455,   71,  571, 1323,   38,   86,   46]))
    '''
    eg_job_counts = []

    for job_list in job_count_lists:
        j = list(job_list)
        eg_job_counts.append(j)

    eg_job_counts = np.array(eg_job_counts)
    combined_job_count = sum(np.array(eg_job_counts))

    return eg_job_counts.astype(int), combined_job_count.astype(int)


# SQUEEZE
def squeeze_increment(data,
                      eg,
                      low_num,
                      high_num,
                      increment):
    '''Move members of a selected eg (employee group) within
    a list according to an increment input (positive or negative)
    while retaining relative ordering within all eg groups.
    inputs
        data (dataframe)
            dataframe with empkey as index which at
            minimum includes an order column and an eg column
        eg (integer)
            employee group number code
        low_num and high_num
            indexes for the beginning and end of the list zone to be
            reordered
        increment (integer)
            the amount to add or subrtract from the appropriate eg order
            number increment can be positive (move down list) or
            negative (move up list - toward zero)
    Selected eg order numbers within the selected zone
    (as a numpy array) are incremented - then
    the entire group order numbers are reset within
    the zone using scipy.stats.rankdata.
    The array is then assigned to a dataframe with empkeys as index.
    '''
    L = low_num - 1
    H = high_num
    if H <= L:
        return

    if L < 0:
        L = 0

    idx_arr = np.array(data.new_order).astype(int)
    eg_arr = np.array(data.eg).astype(int)

    np.putmask(idx_arr[L:H], eg_arr[L:H] == eg, idx_arr[L:H] + increment)
    idx_arr[L:H] = st.rankdata(idx_arr[L:H], method='ordinal') + L

    return idx_arr


# SQUEEZE_LOGRITHMIC
def squeeze_logrithmic(data,
                       eg,
                       low_value,
                       high_value,
                       log_factor=1.5,
                       put_segment=1,
                       direction='d'):
    '''perform a log squeeze (logrithmic-based movement of
    one eg (employee group), determine the closest
    matching indexes within the rng to fit the squeeze,
    put the affected group in those indexes,
    then fill in the remaining slots with the
    other group(s), maintaining orig ordering
    within each group at all times
    inputs
        data (dataframe)
            a dataframe indexed by empkey with at least 2 columns:
            employee group (eg) and order (order)
        eg (employee code integer)
            the employee group to move
        low_val and high_val (integers)
            integers marking the boundries (rng)
            for the operation
            (H must be greater than L)
        log_factor (float)
            determines the degree of 'logrithmic packing'
        put_segment (float)
            allows compression of the squeeze result (values under 1)
        direction (string)
            squeeze direction:
            "u" - move up the list (more senior)
            "d" - move down the list (more junior)
    '''
    H = high_value
    L = low_value

    if put_segment <= 0:
        return

    if L < 0:
        L = 0

    if H > len(data):
        H = len(data)

    rng = H - L
    if rng < 2:
        return

    rng_dummy = np.arange(L, H, dtype=np.int_)

    order_arr = np.array(data.new_order, dtype=np.float_)
    eg_arr = np.array(data.eg, dtype=np.int_)

    order_segment = order_arr[L - 1:H]
    eg_segment = eg_arr[L - 1:H]

    eg_count = np.count_nonzero(eg_segment == eg)
    if eg_count == 0:
        # if there are no members of the selected eg within the list segment,
        # do nothing and return the current order:
        return data.new_order.values

    log_result = np.logspace(0, log_factor, eg_count, endpoint=False)
    log_result = log_result - log_result[0]
    pcnt_result = (log_result / log_result[-1])
    additive_arr = (pcnt_result * rng) * put_segment
    additive_arr = np.int16(additive_arr)

    if direction == 'd':
        put_nums = (H - additive_arr[::-1])
        put_nums = get_indexes(put_nums)
        additive_arr = H - get_indexes(additive_arr)[::-1] - L
    else:
        put_nums = (additive_arr + L)
        put_nums = get_indexes(put_nums)
        additive_arr = get_indexes(additive_arr)

    np.put(order_segment, np.where(eg_segment == eg)[0], put_nums)

    rng_dummy = np.delete(rng_dummy, additive_arr)

    np.put(order_segment, np.where(eg_segment != eg)[0], rng_dummy)

    return order_arr.astype(int)


# GET_INDEXES
@jit(nopython=True, cache=True)
def get_indexes(in_arr):
    rank_bot_up = np.arange(in_arr.size) + np.min(in_arr)
    rank_top_dn = np.arange(in_arr.size - 1, -1, -1)

    c_up_arr = rank_bot_up - in_arr
    c_up = np.maximum(c_up_arr, 0)
    r_up = in_arr + c_up

    c_dn_arr = r_up + rank_top_dn - np.max(r_up)
    c_dn = np.maximum(c_dn_arr, 0)
    idx_arr = r_up - c_dn

    return idx_arr


# MAKE_DECILE_BANDS
def make_decile_bands(num_bands=40,
                      num_returned_bands=10):
    '''creates an array of lower and upper percentile values surrounding
    a consistent schedule of percentile markers.  If the user desires to
    sample data at every 10th percentile, this function provides selectiable
    bottom and top percentile limits surrounding each 10th percentile, or
    variable width sample ranges.
    num_bands input must be multiple of 5 greater than or equal to 10
    and less than 10000.
    num_returned_bands input must be multiple of 5, equal to or less than
    the num_bands input, and num_bands/num_returned_bands must have no
    remainder.
    Used for selecting sample employees surrounding deciles
    (0, 10, 20 etc. percent levels).
    Top and bottom bands will be half of normal size.
    inputs
        num_bands
            Width of bands in percentage is determined by num_bands input.
            Input of 40 would mean bands 2.5% wide. (100/40)
            Top and bottom bands would be 1.25% wide.
            Ex. 0-1.25%,
            8.75-11.25%,
            ... 98.75-100%
        num_returned_bands
            number of returned delineated sections.  Must be a multiple of 5
            less than or equal to the num_bands value
            with no remainder when divided into the num_bands value.
            (note:  an input of 10 would result in 11 actual segments,
            one-half band at the top and bottom of list (0% and 100%),
            and 9 full bands surrounding each decile, 10% to 90%)
    '''
    if num_bands < 10:
        print('input must be multiple of 5 greater than or equal to 10')
        return
    if num_bands % 5 != 0:
        print('input must be multiple of 5 greater than or equal to 10')
        return
    if (num_returned_bands > num_bands) or \
       (num_bands % num_returned_bands != 0):
        print('num_returned_bands input must be <= num_bands and ' +
              'num_bands / num_returned_bands must have no remainder')
        return
    cutter = (num_bands * 2) + 1
    cuts = np.round(np.linspace(0, 1, cutter) * 100, 2)
    strider = 2
    lower = list(cuts[strider - 1::strider])
    upper = list(cuts[1::strider])
    upper.append(100)
    lower = sorted(lower, reverse=True)
    lower.append(0)
    lower = sorted(lower)
    band_limits = np.array((lower, upper)) / 100
    stride = int(num_bands / num_returned_bands)
    return band_limits.T[::stride]


# MONOTONIC TEST
def monotonic(sequence):
    '''test for stricly increasing array-like input
    May be used to determine when need for no bump,
    no flush routine is no longer required.
    If test is true, and there are no job changes,
    special rights, or furlough recalls,
    then a straight stovepipe job assignment routine may
    be implemented (fast).
    input
        sequence
            array-like input (list or numpy array ok)
    '''
    seq_diff = np.diff(sequence)
    return np.all(seq_diff >= 0)


# GET_MONTH_SLICE
def get_month_slice(df, l, h):
    '''Convenience function to extract data for a particular month.
    Input is low and high indexes of target month data (within dataset
    containing many months)
    The input may also be an array (not limited to a dataframe).
    inputs
        df
            dataframe (or array) to be sliced
        l
            lower index of slice
        h
            upper index of slice
    '''
    segment = df[l:h]
    return segment


# GET_RECALL_MONTHS (refactor to provide for no recall list)
def get_recall_months(list_of_recall_schedules):
    '''extract a sorted list of only the unique months containing a recall
    as defined within the settings dictionary recall schedules.
    input
        list_of_recall_schedules
            list of recall schedule lists, normally equal to the recalls
            variable from the settings dictionary
    '''
    recall_months = []
    for recall_sched in list_of_recall_schedules:
        recall_months.extend(list(range(recall_sched[2], recall_sched[3])))
        recall_months = sorted(list(set(recall_months)))
    return np.array(recall_months).astype(int)


# GET_JOB_CHANGE_MONTHS
def get_job_change_months(job_changes):
    '''extract a sorted list of only the unique months containing a change in
    any job count as defined within the settings dictionary job change
    schedules
    input
        job_changes
            list of job change schedule lists, normally equal to the j_changes
            variable from the settings dictionary
    '''
    month_list = []
    for change in job_changes:
        month_list = np.concatenate((month_list,
                                     np.arange(change[1][0],
                                               change[1][1])))
        month_list = np.unique(month_list)
    return month_list.astype(int)


# GET_REDUCTION_MONTHS
def get_job_reduction_months(job_changes):
    '''extract a sorted list of only the unique months containing a reduction
    in any job count as defined within the settings dictionary job change
    schedules
    input
        job_changes
            list of job change schedule lists, normally equal to the j_changes
            variable from the settings dictionary
    '''
    month_list = []
    for change in job_changes:
        if change[2] < 0:
            month_list = np.concatenate((month_list,
                                         np.arange(change[1][0],
                                                   change[1][1])))
        month_list = np.unique(month_list).astype(int)
    return month_list


# SET SNAPSHOT RATIO WEIGHTINGS
def set_snapshot_weights(job,
                         ratio_dict,
                         orig_rng,
                         eg_range):
    '''Determine the job distribution ratios to carry forward during
    the ratio condition application period using actual jobs held ratios.
    likely called at implementation month by main job assignment function
    Count the number of jobs held by each of the ratio groups for each of the
    affected job level numbers.  Set the weightings in the distribute function
    accordingly.

    inputs
        ratio_dict (dictionary)
            dictionary containing job levels as keys and ratio groups,
            weightings, month_start and month end as values.
        orig_rng (numpy array)
            month slice of original job array
        eg_range (numpy array)
            month slice of employee group code array
    '''
    ratio_dict = copy.deepcopy(ratio_dict)
    # job_nums = list(ratio_dict.keys())
    # for job in job_nums:
    wgt_list = []
    for ratio_group in ratio_dict[job][0]:
        wgt_list.append(np.count_nonzero((orig_rng == job) &
                                         (np.isin(eg_range, ratio_group))))
    ratio_dict[job][1] = tuple(wgt_list)
    return ratio_dict


# ASSIGN JOBS BY RATIO CONDITION
def assign_cond_ratio(job,
                      this_job_count,
                      ratio_dict,
                      orig_range,
                      assign_range,
                      eg_range,
                      fur_range,
                      cap=None):
    ''' Apply a job ratio assignment condition

    The main job assignment function calls this function at the appropriate
    months.

    This function applies a ratio for job assignment between ratio groups.
    Each ratio group may contain one or more employee groups.  The number
    of jobs affected may be limited with the "cap" input.

    The ratio for job assignment is set with the inputs on the "ratio_cond"
    worksheet of the *settings.xlsx* input spreadsheet, using the "group"
    columns and the corresponding "weights" columns.

    Optionally, the ratio of jobs which exists during the "month_start"
    spreadsheet input may be captured and used for job assignment during the
    data model months when the ratio job assignment condition is applicable
    ("month_start" through "month_end").  The existing ratios are captured and
    used by setting the "snapshot" input cell to True for the appropriate
    basic job row.  When using the snapshot option, any weightings designated
    within the "weights" columns will be ignored.

    There may be a mix of snapshot ratios and ratios set by the "weight"
    columns for use within the program.  There may also be count-capped
    ratio assignments and straight ratio assignments within the same data
    model as long as the effective months and jobs do not overlap, but there
    may only be one row of ratio data for a job level within the same
    input worksheet.

    No bump, no flush rules apply when assigning jobs by ratio, meaning only
    job openings due to retirements, increases in job counts, or other openings
    will be assigned according to the ratio schedule.  Employees previously
    holding a job affected by the ratio condition will not be displaced to
    allow an employee from a different ratio group to have that job when
    the ratio assignment period begins.  Therefore, it may take
    some time for the desired ratio of job assignments to be achieved if it
    differs significantly from the actual ratio(s) when the time period of
    conditional job assignment begins.

    inputs
        job (integer or float)
            job level number
        this_job_count (integer or float)
            number of jobs available
        ratio_dict (dictionary)
            ratio condition dictionary, constructed with the
            build_program_files script and possibly modified by the
            set_snapshot_weights function if the "snapshot" option is set
            to True on the "ratio_cond" worksheet of the "settings.xlsx"
            input spreadsheet.
        orig_range (1d array)
            original job range
            Month slice of the orig_job column array (normally pertaining a
            specific month).
        assign_range (1d array)
            job assignment range
            Month slice of the assign_range column array
        eg_range (1d array)
            employee group range
            Month slice of the eg_range column array
        fur_range (1d array)
            furlough range
            Month slice of the fur_range column array
        cap (integer (or whole float))
            if a count ratio job assignment is being used, this number
            represents the number of jobs affected by the conditional
            assignment.  Available jobs above this amount are not affected.
    '''
    ratio_groups = ratio_dict[job][0]
    weights = ratio_dict[job][1]
    quota = distribute(this_job_count, weights, cap=cap)

    mask_index = []
    actual = []
    assign_filter = ((assign_range == 0) &
                     (fur_range == 0) &
                     (orig_range <= job))

    # find the indexes of each ratio group
    for grp in ratio_groups:
        grp_mask = np.isin(eg_range, grp)
        mask_index.append(grp_mask)
        actual.append(np.count_nonzero(assign_filter & grp_mask))

    # the counts variable is a list of job counts to be assigned the
    # ratio groups.  The counts include employees already holding the job
    # from the previous month plus additional to try to meet quotas.
    if cap:
        counts = eg_quotas(quota, actual, cap=cap,
                           this_job_count=this_job_count)
    else:
        counts = eg_quotas(quota, actual)

    # if job == 1:
    #     print(job, ratio_groups, weights, cap, quota, actual, counts)

    i = 0
    for i in range(len(ratio_groups)):
        # assign jobs to employees within each ratio group who already hold
        # that job or better (no bump no flush)
        np.put(assign_range,
               np.where(assign_filter & mask_index[i])[0][:counts[i]],
               job)
        # count how many jobs were assigned to this ratio group by no bump
        # no flush
        used = np.count_nonzero((assign_range == job) &
                                (np.isin(eg_range, ratio_groups[i])))
        # determine how many remain for assignment within the ratio group
        remaining = counts[i] - used

        # assign the remaining jobs by seniority within the ratio group
        np.put(assign_range,
               np.where((assign_range == 0) &
                        (fur_range == 0) &
                        (mask_index[i]))[0][:remaining],
               job)
        i += 1


# RECALL
def mark_for_recall(orig_range,
                    num_of_job_levels,
                    fur_range,
                    month,
                    recall_sched,
                    jobs_avail,
                    standalone=True,
                    eg_index=0,
                    method='sen_order',
                    stride=2):
    '''change fur code to non-fur code for returning employees
    according to selected method (seniority order,
    every nth furloughee, or random)
    note: function assumes it is only being called
    during a recall month
    inputs
        orig_range
            original job range
        num_of_job_levels
            number of job levels in model, normally from settings dictionary
        fur_range
            current month slice of fur data
        month
            current month (loop) number
        recall sched
            list(s) of recall schedule
            (recall amount/month, recall start month, recall end month)
        jobs_avail
            total number of jobs for each month
            array, job_gain_loss_table function output [1]
        standalone (boolean)
            This function may be used with both standalone and integrated
            dataset generation.
            Set this variable to True for use within standalone dataset
            calculation, False for integrated dataset calculation routine.
        eg_index (integer)
            selects the proper recall schedule for standalone dataset
            generation, normally from a loop increment.  The recall schedule
            is defined in the settings dictionary.
            set to zero for an integrated routine (integrated routine
            uses a global recall schedule)
        method
            means of selecting employees to be recalled
            default is by seniority order, most senior recalled first
            other options are:
                stride
                    i.e. every other nth employee.
                    (note: could be multiple strides per month
                    if multiple recall lists are designated).
                random
                    use shuffled list of furloughees
        stride
            set stride if stride option for recall selected.
            default is 2.
    '''
    active_count = np.count_nonzero(fur_range == 0)
    excess_job_slots = jobs_avail[month] - active_count

    if excess_job_slots > 0:

        for sched in recall_sched:

            if month not in np.arange(sched[2], sched[3]):
                continue

            if standalone:

                this_eg_recall_amount = sched[1][eg_index]

                if this_eg_recall_amount == 0:
                    continue

                recalls_this_month = min(this_eg_recall_amount,
                                         excess_job_slots)

            else:
                recalls_this_month = min(sched[0],
                                         excess_job_slots)

            fur_indexes = np.where(fur_range == 1)[0]

            if method == 'sen_order':

                np.put(fur_range,
                       fur_indexes[:recalls_this_month],
                       0)
                np.put(orig_range,
                       fur_indexes[:recalls_this_month],
                       num_of_job_levels + 1)

            if method == 'stride':
                np.put(fur_range,
                       fur_indexes[::stride][:recalls_this_month],
                       0)
                np.put(orig_range,
                       fur_indexes[::stride][:recalls_this_month],
                       num_of_job_levels + 1)

            if method == 'random':
                fur_indexes == np.random.shuffle(fur_indexes)
                fur_range[fur_indexes[:recalls_this_month]] = 0
                orig_range[fur_indexes[:recalls_this_month]] = \
                    num_of_job_levels + 1

            excess_job_slots -= recalls_this_month

            if excess_job_slots == 0:
                return


# RECALL
def mark_for_furlough(orig_range,
                      fur_range,
                      month,
                      jobs_avail,
                      num_of_job_levels):
    '''Assign fur code to employees when count of jobs is
    less than count of active employees in inverse seniority
    order and assign furloughed job level number.
    note: normally only called during a job change month though it
    will do no harm if called in other months
    inputs
        orig_range
            current month slice of jobs held
        fur_range
            current month slice of fur data
        month
            current month (loop) number
        jobs_avail
            total number of jobs for each month
            array, job_gain_loss_table function output [1]
        num_of_job_levels
            from settings dictionary, used to mark fur job level as
            num_of_job_levels + 1
    '''
    active_count = np.count_nonzero(fur_range == 0)

    excess_job_slots = jobs_avail[month] - active_count

    if excess_job_slots >= 0:
        return

    elif excess_job_slots < 0:

        non_fur_indexes = np.where(fur_range == 0)[0]

        np.put(fur_range,
               non_fur_indexes[excess_job_slots:],
               1)
        np.put(orig_range,
               non_fur_indexes[excess_job_slots:],
               num_of_job_levels + 1)


@jit(nopython=True, cache=True)
def mark_fur_range(assign_range,
                   fur_range,
                   job_levels):
    '''apply fur code to current month fur_range based on job assignment status
    inputs
        assign_range
            current month assignment range
            (array of job numbers, 0 indicates no job)
        fur_range
            current month fur status (1 means furloughed,
            0 means not furloughed)
        job_levels
            number of job levels in model (from settings dictionary)
    '''

    for i in range(assign_range.size):
        if assign_range[i] == 0:
            fur_range[i] = 1
        if assign_range[i] > 0:
            if assign_range[i] <= job_levels:
                fur_range[i] = 0


# ALIGN FILL DOWN (all future months)
def align_fill_down(l, u,
                    long_indexed_df,
                    long_array):
    '''Data align current values to all future months
    (short array segment aligned to long array)
    This function is used to set the values from the last standalone month as
    the initial data for integrated dataset computation when a delayed
    implementation exists.
    uses pandas df auto align - relatively slow
    TODO (for developer) - consider an all numpy solution
    inputs
        l, u (integers)
            current month slice indexes (from long df)
        long_indexed_df (dataframe)
            empty long dataframe with empkey indexes
        long_array (array)
            long array of multiple month data
            (orig_job, fur_codes, etc)
    declare long indexed df outside of function (input).
    grab current month slice for array insertion (copy).
    chop long df to begin with current month (copy).
    assign array to short df.
    data align short df to long df (chopped to current month and future).
    copy chopped df column as array to long_array
    return long_array
    '''
    short_df = long_indexed_df[l:u].copy()
    short_df['x'] = long_array[l:u]
    # chopped_df begins with a defined index (row), normally the begining of
    # a delayed implementation month
    chopped_df = long_indexed_df[l:].copy()
    # data align short_df to chopped_df
    chopped_df['x'] = short_df['x']
    result_array = chopped_df.x.values
    result_size = result_array.size
    np.copyto(long_array[-result_size:], result_array)

    return long_array


# ALIGN NEXT (month)
@jit
def align_next(this_index_arr,
               next_index_arr,
               these_vals_arr):
    '''"Carry forward" data from one month to the next.
    Use the numpy isin function to compare indexes (empkeys) from one month
    to the next month and return a boolean mask.  Apply the mask to current
    month data column (slice) and assign results to next month slice.
    Effectively finds the remaining employees (not retired) in the next month
    and copies the target column data for them from current month into the
    next month.
    inputs
        this_index_arr (array)
            current month index of unique employee keys
        next_index_arr (array)
            next month index of unique employee keys
            (a subset of this_index_arr)
        arr (array)
            the data column segment (attribute) to carry forward
    '''
    this_len = this_index_arr.size
    next_len = next_index_arr.size
    result = np.empty(next_len, dtype=int)
    j = 0
    for i in range(this_len):
        if this_index_arr[i] == next_index_arr[j]:
            result[j] = these_vals_arr[i]
            j += 1
    return result


# COUNT_AVAILABLE_JOBS
@jit(nopython=True, cache=True)
def count_avail_jobs(assign_range,
                     job,
                     this_job_count):
    '''use numba to loop through the job assignment range and count the
    number of jobs in a specified job level previously assigned from the
    previous month, then subtract result from the total job level
    positions count.  This result identifies the number of openings
    available for the current month.
    inputs
        assign_range (array)
            monthly slice of job assignment array
        job (integer)
            job level being tested
        this_job_count (integer)
            total job positions count for the job being tested
    '''
    count = 0
    for j in assign_range:
        if j == job:
            count += 1

    return this_job_count - count


# ASSIGN JOB COUNTS
@jit(nopython=True, cache=True)
def assign_job_counts(job_count_range,
                      assign_range,
                      job,
                      this_job_count):
    '''assign job counts to job count array month slice
    inputs
        job_count_range (array)
            month slice of long job count array
        assign_range (array)
            month slice of long job assignment array
        job (integer)
            job level number
        this_job_count (integer)
            job count alloted for job level
    '''

    for i in range(assign_range.size):
        if assign_range[i] == job:
            job_count_range[i] = this_job_count


# DISTRIBUTE (simple)
def distribute(available,
               weights,
               cap=None):
    '''proportionally distribute 'available' according to 'weights'
    usage example:
    .. code:: python
      distribute(334, [2.48, 1])
    returns distribution as a list, rounded as integers:
    .. code:: python
      [238, 96]
    inputs
        available (integer)
            the count (number) to divide
        weights (list)
            relative weighting to be applied to available count
            for each section.
            numbers may be of any size, integers or floats.
            the number of resultant sections is the same as the number of
            weights in the list.
        cap (integer)
            limit distribution total to this amount, if less than the
            "available" input.
    '''
    if cap:
        available = min(available, cap)
    bin_counts = []
    total_weights = sum(weights)
    for weight in weights:
        if weight:
            p = weight / total_weights
            this_bin_count = int(round(p * available))
        else:
            this_bin_count = 0
        bin_counts.append(this_bin_count)
        total_weights -= weight
        available -= this_bin_count

    return bin_counts


# MAKE PARTIAL JOB COUNT LIST (prior to implementation month)
def make_delayed_job_counts(imp_month,
                            delayed_jnums,
                            lower,
                            upper):
    '''Make an array of job counts to be inserted into the long_form job counts
    array of the job assignment function.  The main assignment function calls
    this function prior to the implementation month. The array output of this
    function is inserted into what will become the job count column.
    These jobs are from the standalone job results.
    The job count column displays a total monthly count of the job in the
    corresponding jnum (job number) column.
    inputs
        imp_month (integer)
            implementation month, defined by settings dictionary
        delayed_jnums (numpy array)
            array of job numbers, normally data from the start of the model
            through the implementation month
        lower (numpy array)
            array of indexes marking the beginning of data for each month
            within a larger array of stacked, multi-month data
        upper (numpy array)
            array of indexes marking the end of data for each month
    '''
    imp_high = upper[imp_month]
    stand_job_counts = np.zeros(imp_high)
    job_numbers = sorted(list(set(delayed_jnums[:imp_high])))

    for month in range(imp_month + 1):
        low = lower[month]
        high = upper[month]
        jnums_range = delayed_jnums[low:high]
        stand_range = stand_job_counts[low:high]

        for job in job_numbers:
            job_indexes = np.where(jnums_range == job)[0]
            np.put(stand_range,
                   job_indexes,
                   job_indexes.size)

    return stand_job_counts


# MAKE GAIN_LOSS_TABLE
def job_gain_loss_table(months,
                        job_levels,
                        init_job_counts,
                        job_changes,
                        standalone=False):
    '''Make two arrays of job tally information.
    The first array has a row for each month in the model, and a column for
    each job level (excluding furlough).  This array provides a count for each
    job for each month of the model accounting for changes provided by the
    job change schedules defined by the settings dictionary.
    The second array is a one-dimensional array containing the sum for all jobs
    for each month of the model.
    inputs
        months (integer)
            number of months in model
        job_levels (integer)
            number of job levels in model (excluding furlough level)
        init_job_counts (tuple of two numpy arrays)
            initial job counts.
            Output from the make_jcnts function, essentially an array of the
            job count lists for each employee group and an array of the
            combined counts.
        job_changes (list)
            The list of job changes from the settings dictionary.
        standalone (boolean)
            if True, use the job count lists for the separate employee groups,
            otherwise use the combined job count
    Returns tuple (job_table, monthly_job_totals)
    '''
    table_list = []
    monthly_totals = []

    if standalone:
        this_list_of_counts = init_job_counts[0]
    else:
        this_list_of_counts = [init_job_counts[1]]

    sep_index = 0
    for counts in this_list_of_counts:

        this_job_table = np.zeros((months, job_levels))

        this_job_table[:] = counts

        job_list = []
        start = []
        end = []
        gain_loss = []

        for change in job_changes:

            jnum = int(change[0])
            start_mth = int(change[1][0])
            end_mth = int(change[1][1])
            total_change = change[2]
            eg_dist = change[3]

            job_list.append(jnum)
            start.append(start_mth)
            end.append(end_mth)
            if standalone:
                delta = eg_dist[sep_index]
            else:
                delta = total_change
            gain_loss.append(delta)
            if this_job_table[0][jnum - 1] + delta < 0:
                print('Group ' + str(sep_index + 1) +
                      ' ERROR: job_gain_loss_table function: \n' +
                      'job reduction below zero, job ' +
                      str(jnum) +
                      ', final job total is ' +
                      str(this_job_table[0][jnum - 1] + delta) +
                      ', fur delta input: ' + str(delta) +
                      ', start count: ' +
                      str(this_job_table[0][jnum - 1]) +
                      ' job_levels: ' + str(job_levels))

        for i in range(len(job_changes)):
            col = job_list[i] - 1
            col_change_range = this_job_table[start[i]:end[i], col]
            fill_down_col_range = this_job_table[end[i]:, col]

            calculated_additives = \
                np.linspace(0,
                            gain_loss[i],
                            end[i] - start[i] + 1)[1:].astype(int)

            np.copyto(this_job_table[start[i]:end[i], col],
                      col_change_range + calculated_additives)

            this_job_table[end[i]:, col] = fill_down_col_range + \
                calculated_additives[-1:]

        job_total_each_month = np.add.reduce(this_job_table, axis=1)

        table_list.append(this_job_table)
        monthly_totals.append(job_total_each_month)
        sep_index += 1

    job_table = np.array(table_list)
    monthly_job_totals = np.array(monthly_totals)

    if not standalone:
        job_table = job_table[0]
        monthly_job_totals = monthly_job_totals[0]

    return job_table.astype(int), monthly_job_totals.astype(int)


# Convert to enhanced from basic job levels
def convert_to_enhanced(eg_job_counts,
                        j_changes,
                        job_dict):
    '''Convert employee basic job counts to enhanced job counts (includes
    full-time and part-time job level counts) and convert basic job change
    schedules to enhanced job change schedules.
    Returns tuple (enhanced_job_counts, enhanced_job_changes)
    inputs
        eg_job_counts
            A list of lists of the basic level job counts for each employee
            group.  Each nested list has a length equal to the number of
            basic job levels.
            example:
            .. code:: python
              [[197, 470, 1056, 412, 628, 1121, 0, 0],
              [80, 85, 443, 163, 96, 464, 54, 66],
              [0, 26, 319, 0, 37, 304, 0, 0]]
        j_changes
            input from the settings dictionary describing change of job
            quantity over months of time (list)
            example:
            .. code:: python
              [1, [35, 64], 87, [80, 7, 0]]
            [[job level, [start and end month],
            total job count change,
            [eg allotment of change for standalone calculations]]
        job_dict
            conversion dictionary for an enhanced model.
            This is the "jd" key value from the settings dictionary.
            It uses the basic job levels as the keys, and lists as values
            which containin the new full- and part-time job level numbers
            and the percentage of basic job counts to be converted to
            full-time jobs.
            example:
            .. code:: python
              {1: [1, 2, 0.6],
              2: [3, 5, 0.625],
              3: [4, 6, 0.65],
              4: [7, 8, 0.6],
              5: [9, 12, 0.625],
              6: [10, 13, 0.65],
              7: [11, 14, 0.65],
              8: [15, 16, 0.65]}
    '''
    # job changes section
    # ..................................................
    enhanced_job_changes = []

    for jc in j_changes:
        job = jc[0]
        temp1 = []
        temp2 = []
        # ft refers to full-time, pt is part-time
        ft = job_dict[job][2]
        pt = 1 - ft

        # full-time calculation for this job change
        temp1 = list([job_dict[job][0],
                      jc[1], np.around(jc[2] * ft).astype(int),
                      list(np.around(np.array(jc[3]) * ft).astype(int))])

        # part-time calculation for this job change
        temp2 = list([job_dict[job][1],
                      jc[1], np.around(jc[2] * pt).astype(int),
                      list(np.around(np.array(jc[3]) * pt).astype(int))])

        # add full-time change to changes list
        enhanced_job_changes.append(temp1)
        # add part-time changes to changes list
        enhanced_job_changes.append(temp2)

    # job counts section
    # ...............................................
    enhanced_job_counts = []

    for job_list in eg_job_counts:
        this_list = []
        new_dict = {}
        for job in list(job_dict.keys()):
            # grab full-time job number as key, calculate count, set as value
            new_dict[job_dict[job][0]] = \
                np.around(job_list[job - 1] *
                          job_dict[job][2]).astype(int)
            # same for part-time
            new_dict[job_dict[job][1]] = \
                np.around(job_list[job - 1] *
                          (1 - job_dict[job][2])).astype(int)
        # sort keys and then assign corresponding values to list
        for key in sorted(new_dict.keys()):
            this_list.append(new_dict[key])
        # add list to list of lists
        enhanced_job_counts.append(this_list)

    return enhanced_job_counts, enhanced_job_changes


def print_settings():
    '''grab settings dictionary data settings and put it in a dataframe and then
    print it for a quick summary of scalar settings dictionary inputs
    '''
    sdict = pd.read_pickle('dill/dict_settings.pkl')
    try:
        case_study = pd.read_pickle('dill/case_dill.pkl')
    except OSError:
        case_study = 'error, no case_dill.pkl file found'

    config_dict = {'case_study': case_study,
                   'starting_date': sdict['starting_date'],
                   'enhanced_jobs': sdict['enhanced_jobs'],
                   'enhanced_jobs_full_suffix':
                   sdict['enhanced_jobs_full_suffix'],
                   'enhanced_jobs_part_suffix':
                   sdict['enhanced_jobs_part_suffix'],
                   'delayed_implementation': sdict['delayed_implementation'],
                   'implementation_date': sdict['implementation_date'],
                   'imp_month': sdict['imp_month'],
                   'recall': sdict['recall'],
                   'compute_job_category_order':
                   sdict['compute_job_category_order'],
                   'compute_pay_measures': sdict['compute_pay_measures'],
                   'compute_with_job_changes':
                   sdict['compute_with_job_changes'],
                   'discount_longev_for_fur': sdict['discount_longev_for_fur'],
                   'future_raise': sdict['future_raise'],
                   'init_ret_age': sdict['init_ret_age'],
                   'init_ret_age_months': sdict['init_ret_age_months'],
                   'init_ret_age_years': sdict['init_ret_age_years'],
                   'ret_age': sdict['ret_age'],
                   'ret_age_increase': sdict['ret_age_increase'],
                   'job_levels_basic': sdict['job_levels_basic'],
                   'job_levels_enhanced': sdict['job_levels_enhanced'],
                   'contract_end': sdict['contract_end'],
                   'lspcnt_calc_on_remaining_population':
                   sdict['lspcnt_calc_on_remaining_population'],
                   'no_bump': sdict['no_bump'],
                   'num_of_job_levels': sdict['num_of_job_levels'],
                   'pay_table_longevity_sort':
                   sdict['pay_table_longevity_sort'],
                   'pay_table_year_sort': sdict['pay_table_year_sort'],
                   'save_to_pickle': sdict['save_to_pickle'],
                   'dist_count': sdict['dist_count'],
                   'dist_ratio': sdict['dist_ratio'],
                   'dist_sg': sdict['dist_sg'],
                   'top_of_scale': sdict['top_of_scale'],
                   'add_doh_col': sdict['add_doh_col'],
                   'add_eg_col': sdict['add_eg_col'],
                   'add_ldate_col': sdict['add_ldate_col'],
                   'add_line_col': sdict['add_line_col'],
                   'add_lname_col': sdict['add_lname_col'],
                   'add_ret_mark': sdict['add_ret_mark'],
                   'add_retdate_col': sdict['add_retdate_col'],
                   'add_sg_col': sdict['add_sg_col'],
                   'annual_pcnt_raise': sdict['annual_pcnt_raise']}

    settings = pd.DataFrame(config_dict, index=['setting']).stack()
    df = pd.DataFrame(settings, columns=['setting'])
    df.index = df.index.droplevel(0)
    df.index.name = 'option'

    return df


def max_of_nested_lists(nested_list,
                        return_min=False):
    '''Find the maximum value within a list of lists (or tuples or arrays).
    The function may optionally return the minimum value within nested
    containers.
    inputs
        nested_list (list, tuple, or array)
            nested container input
        return_min (boolean)
            if True, return minimum of nested_list input (vs. max)
    '''
    result_list = []
    if not return_min:
        for lst in nested_list:
            result_list.append(max(lst))
        return max(result_list)
    else:
        for lst in nested_list:
            result_list.append(min(lst))
        return min(result_list)


def clip_ret_ages(ret_age_dict,
                  init_ret_age,
                  dates_long_arr,
                  ages_long_arr):
    '''Clip employee ages in employee final month to proper retirement age if
    the model includes an increasing retirement age over time
    inputs
        ret_age_dict (dictionary)
            dictionary of retirement increase date to new retirement age as
            defined in settings dictionary
        init_ret_age
            initial retirement age prior to any increase
        dates_long_arr (numpy array)
            array of month dates (long form, same value during each month)
        ages_long_arr (numpy array)
            array of employee ages (long form)
    '''
    date_list = []
    ret_age_list = [init_ret_age]
    prev = 0

    for date, month_add in ret_age_dict.items():
        month_yrs = month_add * (1 / 12)
        date_list.append(np.datetime64(pd.to_datetime(date)))
        ret_age_list.append(month_yrs + init_ret_age + prev)
        prev += month_yrs
    date_list.append(np.datetime64(pd.to_datetime(dates_long_arr.max())))
    date_arr = np.array(date_list)
    ret_age_arr = np.array(ret_age_list)

    for date, age in zip(date_arr, ret_age_arr):
        clip_count = np.where(dates_long_arr < date)[0].size
        ages_long_arr[:clip_count] = np.clip(ages_long_arr[:clip_count],
                                             0, age)

    return ages_long_arr


def clear_dill_files():
    '''remove all files from 'dill' folder.
    used when changing case study, avoids possibility of file
    from previos calculations being used by new study
    '''
    if os.path.isdir('dill/'):
        filelist = [f for f in os.listdir('dill/') if f.endswith('.pkl')]
        for f in filelist:
            os.remove('dill/' + f)


def load_datasets(other_datasets=['standalone', 'skeleton', 'edit', 'hybrid']):
    '''Create a dictionary of proposal names to corresponding datasets.
    The datasets are generated with the RUN_SCRIPTS notebook.  This routine
    reads the names of the case study proposals from a pickled dataframe
    ('dill/proposal_names.pkl'), created by the build_program_files.py script.
    It then looks for the matching stored datasets within the dill folder.
    The datasets are loaded into a dictionary, using the proposal names as
    keys.
    The dictionary allows easy reference to datasets from the Jupyter notebook
    and from within functions.
    input
        other_datasets (list)
            list of datasets to load in addition to those computed from the
            proposals (from the case-specific proposals.xlsx Excel file)
    '''

    # create ordered dictionary
    ds_dict = od()
    # read stored dataframe
    proposals_df = pd.read_pickle('dill/proposal_names.pkl')
    # make a list of the proposal names
    proposal_names = list(proposals_df.proposals)
    # add the other dataset names
    proposal_names.extend(other_datasets)

    # read and assign the datasets to the dictionary
    for ws in proposal_names:
        if ws not in other_datasets or ws in ['edit', 'hybrid']:
            ws_ref = 'ds_' + ws
        else:
            ws_ref = ws

        try:
            ds_dict[ws] = pd.read_pickle('dill/' + ws_ref + '.pkl')
        except OSError:
            # if dataset doesn't exist, pass and notify user
            print('dataset for proposal "' + ws + '" not found in dill folder')
            if ws == 'edit':
                print('"edit" proposal is produced with the editor tool.\n')
            if ws == 'hybrid':
                print('"hybrid" proposal is generated with the "build_list"' +
                      ' function within the list_builder.py module\n')

    print('datasets loaded (dictionary keys):', list(ds_dict.keys()), '\n')

    return ds_dict


def make_preimp_array(ds_stand,
                      ds_integrated,
                      imp_high,
                      compute_cat,
                      compute_pay):
    '''Create an ordered numpy array of pre-implementation data gathered from
    the pre-calculated standalone dataset and a dictionary to keep track of the
    information.  This data will be joined with post_implementation integrated
    data and then copied into the appropriate columns of the final integrated
    dataset.
    inputs
        ds_stand (dataframe)
            standalone dataset
        ds_integrated (dataframe)
            dataset ordered for proposal
        imp_high
            highest index (row number) from implementation month data
            (from long-form dataset)
        compute_cat (boolean)
            if True, compute and append a job category order column
        compute_pay (boolean)
            if True, compute and append a monthly pay column and a career
            pay column
    '''
    key_cols = ['mnum', 'empkey']
    imp_cols = ['mnum', 'empkey', 'job_count', 'orig_job', 'jnum', 'lnum',
                'lspcnt', 'snum', 'spcnt', 'rank_in_job', 'jobp', 'fur']
    if compute_cat:
        imp_cols.append('cat_order')
    if compute_pay:
        imp_cols.extend(['mpay', 'cpay'])
    # only include columns from col_list which exist in ds_stand
    filtered_cols = list(set(imp_cols).intersection(ds_stand.columns))

    # grab appropriate columns from standalone dataset up to end of
    # implementation month initiate a 'key' column to save assignment
    # time below
    ds_stand = ds_stand[filtered_cols][:imp_high].copy()

    # grab the 'mnum' and 'empkey' columns from the ordered dataset to
    # form a 'key' column with unique values.
    # The ds_temp dataframe is used to sort the ds_stand dataframe.
    ds_temp = ds_integrated[key_cols][:imp_high].copy()

    # make numpy arrays out of column values for fast 'key' column generation
    stand_emp = ds_stand.empkey.values * 1000
    stand_mnum = ds_stand.mnum.values
    temp_emp = ds_temp.empkey.values * 1000
    temp_mnum = ds_temp.mnum.values

    # assign 'key' columns
    ds_stand['key'] = stand_emp + stand_mnum
    ds_temp['key'] = temp_emp + temp_mnum

    # now that the 'key' columns are in place, we don't need or
    # want the key making columns.
    # get ds_stand columns except for key making columns ('mnum', 'empkey')
    stand_cols = list(set(ds_stand.columns).difference(key_cols))

    # redefine ds_stand to include original columns less key making columns
    ds_stand = ds_stand[stand_cols]

    # redefine ds_temp to only include 'key' column (retains index)
    ds_temp = ds_temp[['key']]

    # merge standalone data to integrated list ordered ds_temp df,
    # using the unique 'key' column values.
    # this will generate standalone data ordered to match the employee order
    # from the integrated dataset
    ds_temp = pd.merge(ds_temp, ds_stand, on='key')

    # now get rid of the 'key' column
    temp_cols = list(set(ds_temp.columns).difference(['key']))

    # re-order the ds_temp columns according to the imp_cols order
    ordered_cols = []
    for col in imp_cols:
        if col in temp_cols:
            ordered_cols.append(col)
    ds_temp = ds_temp[ordered_cols]

    # convert the ds_temp dataframe to a 2d numpy array for fast
    # indexing and retrieval
    stand_arr = ds_temp.values.T

    # construct a dictionary of columns to numpy row indexes
    values = np.arange(len(ordered_cols))
    delay_dict = od(zip(ordered_cols, values))

    # make a numpy array as wide as the stand_arr and
    # as long as the integrated dataset
    final_array = np.zeros((len(ordered_cols), len(ds_integrated)))

    # assign the standalone data to the final_array.  The data will extend
    # in each column up to the imp_high index
    for col in ordered_cols:
        final_array[delay_dict[col]][:imp_high] = stand_arr[delay_dict[col]]

    return ordered_cols, delay_dict, final_array


def make_cat_order(ds,
                   table):
    '''make a long-form "cat_order" (global job ranking) column
    This function assigns a global job position value to each employee,
    considering the modeled job level hierarchy and the job count within
    each level.  For example, if a case study contains 3 job levels with
    100 jobs in each level, an employee holding a job in the middle of
    job level 2 would be assigned a cat_order value of 150.
    Category order for standalone employee groups is "normalized" to an
    integrated scale by applying *standalone* job level percentage
    (relative position within a job level) to the *integrated* job level
    counts.  This process allows "apples to apples" comparison between
    standalone and integrated job progression.
    Standalone cat_order will only reflect job levels available within the
    standalone scenario.  If the integrated model contains job levels which
    do not exist within a standalone employee group model, standalone
    cat_order results will exclude the respective job level rank segments
    and will rank the existing standalone data according to the integrated
    ranking scale.
    The routine creates numpy array lookup tables from integrated job
    level count data for each month of the model.  The tables are the source
    for count and additive information which is used to calculate a rank number
    within job level and cumulative job count additives.
    Month number and job number arrays (from the input ds (dataset)) are used
    to index into the numpy lookup arrays, producing the count and additive
    arrays.
    A simple formula is then applied to the percentage, count, and additive
    arrays to produce the cat_order array.
    inputs
        ds (dataframe)
            a dataset containing ['jobp', 'mnum', 'jnum'] columns
        table (numpy array)
            the first output from the job_gain_loss_table function which
            is a numpy array with total job counts for each job level for
            each month of the data model
    '''

    ds = ds[['jobp', 'mnum', 'jnum']].copy()

    zero_col = np.zeros((table.shape[0], 1)).T

    cat_counts = table.T
    cat_counts = np.concatenate((cat_counts, zero_col), axis=0)

    cat_add = np.add.accumulate(table, axis=1).T
    cat_add = np.concatenate((zero_col, cat_add), axis=0)
    cat_add = cat_add[0:-1]
    cat_add = np.concatenate((cat_add, zero_col), axis=0)

    cat_add[-1] = np.nan
    cat_counts[-1] = np.nan

    mnum_arr = ds.mnum.values
    jnum_arr = ds.jnum.values - 1
    jpcnt_arr = np.array(ds.jobp % 1)

    cnt_arr = cat_counts[jnum_arr, mnum_arr]
    add_arr = cat_add[jnum_arr, mnum_arr]

    cat_arr = (jpcnt_arr * cnt_arr) + add_arr

    return cat_arr


def make_tuples_from_columns(df,
                             col_list,
                             return_as_list=True,
                             date_cols=[],
                             return_dates_as_strings=False,
                             date_format='%Y-%m-%d'):
    '''Combine row values from selected columns to form tuples.
    Returns a list of tuples which may be assigned to a new column.
    The length of the list is equal to the length of the input dataframe.
    Date columns may be first converted to strings before adding to output
    tuples if desired.
    inputs
        df (dataframe)
            input dataframe
        col_list (list)
            columns from which to create tuples
        return_as_list (boolean)
            if True, return a list of tuples
        date_cols (list)
            list of columns to treat as dates
        return_dates_as_strings (boolean)
            if True, for columns within the data_cols input, convert date
            values to string format
        date_format (string)
            string format of converted date columns
    '''
    i = 0
    col_list = col_list[:]
    for col in col_list:
        if col in date_cols and return_dates_as_strings:
            col_list[i] = list(df[col].dt.strftime(date_format))
        else:
            col_list[i] = list(df[col])
        i += 1
    zipped = zip(*col_list)
    if return_as_list:
        return list(zipped)
    else:
        return tuple(zipped)


def make_dict_from_columns(df, key_col, value_col):
    '''Make a dictionary from two dataframe columns.  One column will be the
    keys and the other the values.
    Unique key column values will be assigned dictionary values.  If the
    key_col input contains duplicates, only the last duplicate key-value pair
    will exist within the returned dictionary.
    inputs
        df (dataframe)
            pandas dataframe containing the columns
        key_col (string (or possibly integer))
            dataframe column which will become dictionary keys
        value_col (string (or possibly integer))
            dataframe column which will become dictionary values
    '''
    keys = df[key_col]
    values = df[value_col]

    return dict(zip(keys, values))


def make_lists_from_columns(df,
                            columns,
                            remove_zero_values=False,
                            try_integers=False,
                            as_tuples=False):
    '''combine columns row-wise into separate lists, return a list of lists

    example:
               +----+----+----+----+
               | A  | B  | C  | D  |
               +----+----+----+----+
               | 1  | 6  | 0  | 2  |
               +----+----+----+----+
               | 8  | 4  | 5  | 3  |
               +----+----+----+----+

        .. code:: python
          make_lists_from_columns(df, ["A", "B", "C"])
          [[1, 6, 0], [8, 4, 5]]
          make_lists_from_columns(df, ["A", "B", "C"],
                                  remove_zero_values=True,
                                  as_tuples=True)
          [(1, 6), (8, 4, 5)]

    inputs
        df (dataframe)
            pandas dataframe containing columns to combine
        columns (list)
            list of column names
        try_integers (boolean)
            if True, if all column values are numerical, the output will
            be converted to integers
        remove_zero_values (boolean)
            if True, remove zero values from list or tuple outputs.  The
            routine checks for zeros as a zero value or a list with a
            single zero value
        as_tuples (boolean)
            if True, output will be a list of tuples instead of a list of lists
    '''
    df_cols = df[columns]

    arrays = list(df_cols.values)

    if try_integers:
        try:
            arrays = list(df_cols.values.astype(int))
        except ValueError:
            pass

    column_list = []
    for e in arrays:
        e = list(e)
        column_list.append(e)

    if remove_zero_values:
        for i in range(len(column_list)):
            column_list[i] = [grp for grp in column_list[i]
                              if grp not in [[0], 0]]

    if as_tuples:
        column_list = [tuple(x) for x in column_list]
    return column_list


def make_group_lists(df,
                     column_name):
    '''this function is used with Excel input to convert string objects and
    integers into Python lists containing integers.  This function is used
    with the count_ratio_dict dictionary construction.
    The function works with one column at a time.
    Output is a list of lists which may be reinserted into a column of the
    dataframe.
        example:
               +----+----+----+-------+
               | A  | B  | C  |   D   |
               +----+----+----+-------+
               | 1  | 6  | 0  | "2,3" |
               +----+----+----+-------+
               | 8  | 4  | 5  |  "5"  |
               +----+----+----+-------+
        .. code:: python
          make_group_lists(df, ["D"])
          [[2, 3], [5]]
    This function allows the user to type the string 2,3 into an Excel
    worksheet cell and have it interpreted by seniority_list as [2, 3]
    inputs
        df (dataframe)
            dataframe containing Excel employee group codes
        column_name
            dataframe column name to convert
    '''
    col = df[column_name]
    col_list = []
    for item in col:
        this_list = []
        try:
            for el in item.strip("'").split(","):
                this_list.append(int(el))
        except AttributeError:
            if type(item) == list:
                this_list = item
            else:
                this_list.append(int(item))
        col_list.append(this_list)
    return col_list


def make_eg_pcnt_column(df, recalc_each_month=False, mnum=0,
                        inplace=True, trim_ones=True,
                        fixed_col_name='eg_start_pcnt',
                        running_col_name='eg_pcnt'):
    '''make an array derived from the input df reflecting one of the following
    options:
    Option A:
        The percentage of each employee within his/her original employee
        group for **a selected month**.  The array values will be data-aligned
        with the df input index.  This option is useful for tracking
        percentile cohorts throughout the model.
    Option B:
        The percentage of each employee within his/her original employee
        group **recalculated each month**.  This has the effect of adjusting
        each group relative percentage for population changes due to
        retirements, furlough, etc.  This option is useful for monthly
        percentile cohort comparisons.
    This function either adds a column to the input dataframe or returns an
    array of values, the same length as the input dataframe.
    Note: This function calculations include any furloughed employees
    assign to long-form dataframe (with default month 0 values aligned):
    .. code:: python
      make_eg_pcnt_column(df)
    input
        df (dataframe)
            pandas dataframe containing an employee group code column ('eg')
            and a month number column ('mnum').  The dataframe must be
            indexed with employee number code integers ('empkey')
        recalc_each_month (boolean)
            if True:
                recalculate separate employee group percentage each month of
                data model
            if False:
                calculate values for one month only - align those values
                by employee number (empkey) to the entire data model
        mnum (integer)
            if recalc_each_month is True, calculate values for this selected
            month number
        inplace (boolean)
            if True, add a column to the input dataframe with the calculated
            values.  If False, return a numpy array of the calculated values.
        trim_ones (boolean)
            if True, replace 100% values (1.0) with a value slightly under
            1.0 (.9999).  This action assists construction of percentile
            quantiles for membership grouping purposes.
        exclude_fur (boolean)
            if True, remove furloughed employees from percentage calculations
        fixed_col_name (string)
            manually designated name for dataframe column when
            recalc_each_month input is False and inplace input is True.
        running_col_name (string)
            manually designated name for dataframe column when
            recalc_each_month input is True and inplace input is True.
    '''

    if not recalc_each_month:
        # grab the first month of the input dataframe, only 'eg' column
        m0df = df[df.mnum == mnum][['eg']].copy()
        # make a running total for each employee group and assign to column
        m0df['eg_count'] = m0df.groupby('eg').cumcount() + 1
        # make another column with the total count for each respective group
        m0df['eg_total'] = m0df.groupby('eg')['eg'].transform('count')
        # calculate the group percentage and assign to column
        m0df['eg_pcnt'] = m0df.eg_count / m0df.eg_total

        if trim_ones:
            m0df['eg_pcnt'].replace(to_replace=1, value=.9999, inplace=True)
        if inplace:
            # data align results to long_form input dataframe
            df[fixed_col_name] = m0df.eg_pcnt
        else:
            return df.eg_start_pcnt.values

    else:

        df_egs = df[['mnum', 'eg']].copy()
        df_egs['eg_count'] = df_egs.groupby(['mnum', 'eg']).cumcount() + 1
        df_egs['eg_total'] = df_egs.groupby(['mnum', 'eg'])['eg'] \
            .transform('count')
        df_egs['eg_pcnt'] = df_egs.eg_count / df_egs.eg_total

        if trim_ones:
            df_egs['eg_pcnt'].replace(to_replace=1, value=.9999, inplace=True)
        if inplace:
            df[running_col_name] = df_egs.eg_pcnt
        else:
            return df.eg_pcnt.values


def make_starting_val_column(df, attr, inplace=True):
    '''make an array of values derived from the input dataframe which will
    reflect the starting value (month zero) of a selected attribute.  Each
    employee will be assigned the zero-month attribute value specific to
    that employee, duplicated in each month of the data model.
    This column allows future attribute analysis with a constant starting
    point for all employees.  For example, retirement job position may be
    compared to initial list percentage.
    assign to long-form dataframe:
    .. code:: python
      df['start_attr'] = make_starting_val_column(df, attr)
    input
        df (dataframe)
            pandas dataframe containing the attr input column and a month
            number coulumn.  The dataframe must be indexed with employee
            number code integers ('empkey')
        attr (column name in df)
            selected zero-month attribute (column) from which to assign
            values to the remaining data model months
    '''
    all_mths_df = df[['mnum', attr]].copy()
    m0df = all_mths_df[all_mths_df.mnum == 0]
    all_mths_df['starting_value'] = m0df[attr]

    if inplace:
        df['start_' + attr] = all_mths_df.starting_value.values
    else:
        return all_mths_df.starting_value.values


def save_and_load_dill_folder(save_as=None,
                              load_case=None,
                              print_saved=False):
    '''Save the current "dill" folder to the "saved_dill_folders" folder, or
    load a saved dill folder as the "dill" folder if it exists.
    This function allows calculated case study pickle files
    (including the calculated datasets) to be saved to or loaded loaded from
    a "saved_dill_folders" folder.
    The "saved_dill_folders" folder is created if it does not already exist.
    The load_case input is a case study name.  If the load_case input is set to
    None, the function will only save the current dill folder and do nothing
    else.  If a load_case input is given, but is incorrect or no matching
    folder exists, the function will only save the current dill folder and do
    nothing else.
    The user may print a list of available saved dill folders (for loading)
    by setting the print_saved input to True.  No other action will take place
    when this option is set to True.
    If an award has conditions which differ from proposed conditions, the
    settings dictionary must be modified and the dataset rebuilt.
    This function allows previously calculated datasets to be quickly
    retrieved and eliminates continual adjustment of the settings spreadsheet
    if the user switches between case studies (assuming the award has been
    determined and no more input adjustment will be made).
    input
        save_as (string)
            A user-specified folder prefix.  If None, the current "dill" folder
            will be saved using the current case study name as a prefix.  If
            set to a string value, the current dill folder will be saved with
            the "save_as" string value prefix.
            Example with the save_as variable set to "test1".  The existing
            dill folder would be saved as:
            .. code:: python
              saved_dill_folders/test1_dill_folder
        load_case (string)
            The name of a case study.  If None, the only action performed will
            be to save the current "dill" folder to the "saved_dill_folders"
            folder.
            If the load_case variable is a valid case study name and a saved
            dill folder for that case study exists, the saved dill folder will
            become the current dill folder (contents of the saved dill folder
            will be copied into the current dill folder).  This action will
            occur after the contents of the current dill folder are copied into
            the "saved_dill_folders" folder.
        print_saved (boolean)
            option to print the saved folder prefixes only.  This provides a
            quick check of the folders available to be loaded.  No other action
            will take place with this option set to True.
    '''
    os.makedirs('saved_dill_folders/',
                exist_ok=True)

    if print_saved:
        # get all saved folder prefixes and print
        this_dir = 'saved_dill_folders/'
        print('The saved dill folders available to load are:\n')
        print('   ' + str([name.replace('_dill_folder', '') for name
              in os.listdir(this_dir) if
              os.path.isdir(os.path.join(this_dir, name))]))
        print('\nNothing changed, set print_saved input to "False" if ' +
              'you wish to save and/or load a folder\n')
        return

    try:
        # get current case study name
        case_df = pd.read_pickle('dill/case_dill.pkl')
        current_case_name = case_df.case.value
    except OSError:
        current_case_name = 'copy'

    if save_as is None:
        # use case study name as prefix
        save_name = current_case_name
        if current_case_name == 'copy':
            print('"dill/case_dill.pkl" not found, ' +
                  'copying dill folder as "copy_dill_folder"\n')
    else:
        # set user-defined prefix
        save_name = save_as

    dill = 'dill/'
    dst = 'saved_dill_folders/' + save_name + '_dill_folder'

    # delete destination folder if it already exists
    if os.path.exists(dst):
        shutil.rmtree(dst)
    # copy dill folder to destination folder
    shutil.copytree(dill, dst)
    print('"' + current_case_name + '" dill folder copied to:\n\n    ' +
          dst + '\n')

    if load_case:
        try:
            load_dill = 'saved_dill_folders/' + load_case + '_dill_folder'
            # if both a saved load_case folder and a dill folder exist,
            # delete the dill folder in preparation for the paste
            if os.path.exists(load_dill):
                if os.path.exists(dill):
                    shutil.rmtree(dill)
            # copy load_case folder to dill folder (will fail if load_case
            # folder does not exist)
            shutil.copytree(load_dill, dill)
            # update case_dill.pkl file
            case_dill = pd.DataFrame({'case': load_case}, index=['value'])
            case_dill.to_pickle('dill/case_dill.pkl')
            # read the proposal names (dataset names) and print for user
            prop_df = pd.read_pickle('dill/proposal_names.pkl')
            proposal_names = list(prop_df.proposals)
            print('The dill folder contains the files previously saved as "' +
                  load_case + '".')
            print('The "' + load_case +
                  '" proposal names are:\n\n    ' +
                  str(proposal_names) + '\n')
        except OSError:
            print('\nError >>>  problem finding a saved dill folder with a "' +
                  load_case + '" prefix in ' +
                  'the "saved_dill_folders" folder.')
            print('\nThe dill folder contents remain unchanged.\n')


# ADD COLUMN OF ZEROS TO 2D ARRAY
def add_zero_col(arr):
    '''Add a column of zeros as the first column in a 2d array.
    Output will be a numpy array.
    example:
        input array:
        .. code:: python
          array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
                 [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                 [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                 [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                 [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
        output array:
        .. code:: python
          array([[ 0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
                 [ 0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                 [ 0, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                 [ 0, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                 [ 0, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
    input
        arr (array)
            2-dimensional numpy array
    '''
    zero_col = np.zeros((arr.shape[0], 1))
    arr = np.append(zero_col, arr, 1)
    return arr.astype(int)


def update_excel(case,
                 file,
                 ws_dict={},
                 sheets_to_remove=None):
    '''Read an excel file, optionally remove worksheet(s), add worksheets
    or overwrite worksheets with a dictionary of ws_name, dataframe key, value
    pairs, and write the excel file back to disk
    inputs
        case (string)
            the data model case name
        file (string)
            the excel file name without the .xlsx extension
        ws_dict (dictionary)
            dictionary of worksheet names as keys and pandas dataframes as
            values.  The items in this dictionary will be passed into the
            excel file as worksheets. The worksheet name keys may be the
            same as some or all of the worksheet names in the excel file.
            In the case of matching names, the data from the input dict will
            overwrite the existing data (worksheet) in the excel file.
            Non-overlapping worksheet names/dataframe values will be added
            as new worksheets.
        sheets_to_remove (list)
            a list of worksheet names (strings) representing worksheets to
            remove from the excel workbook.  It is not necessary to remove
            sheets which are being replaced by worksheet with the same name.
    '''
    # read a single or multi-sheet excel file
    # (returns dict of sheetname(s), dataframe(s))
    path = 'excel/' + case + '/' + file + '.xlsx'

    # make a copy of file before modifying
    copy_excel_file(case, file, verbose=False)

    # get a dictionary from the excel file consisting of worksheet name keys
    # and worksheet contents as values (as dataframes)
    try:
        dict0 = pd.read_excel(path, sheetname=None)
    except OSError:
        print('Error: Unable to find "' + path + '"')
        return
    # all worksheets are now accessible as dataframes.
    # drop worksheets which match an element in the sheets_to_remove:
    if sheets_to_remove is not None:
        for ws_name in sheets_to_remove:
            dict0.pop(ws_name, None)

    # update worksheet dictionary with ws_dict (ws_dict values will override
    # existing values in the case of matching worksheet name keys):
    dict0.update(ws_dict)

    # write the updated dictionary back to excel...
    with pd.ExcelWriter(path,
                        engine='xlsxwriter',
                        datetime_format='yyyy-mm-dd',
                        date_format='yyyy-mm-dd') as writer:

        for sheet_name, df_sheet in dict0.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name)


def copy_excel_file(case,
                    file,
                    return_path_and_df=False,
                    revert=False,
                    verbose=True):
    '''Copy an excel file and add '_orig' to the file name, or restore an
    excel file from the '_orig' copy.
    inputs
        case (string)
            the data model case name
        file (string)
            the excel file name without the .xlsx extension
        return_path_and_df (boolean)
            if True, return a tuple containing the file path as a string and
            the worksheet designated by the "file" input as a dataframe
        revert (boolean)
            if False, copy the excel file and add '_orig' to the file name.
            if True, restore the copied file and drop the '_orig' suffix
        verbose (boolean)
            if True, print a brief summary of the operation result
    '''
    path_orig = 'excel/' + case + '/' + file + '_orig.xlsx'
    path = 'excel/' + case + '/' + file + '.xlsx'

    # copy file as <file>_orig.xlsx
    if not revert:
        try:
            if not os.path.isfile(path_orig):
                shutil.copyfile(path, path_orig)
                if verbose:
                    print('\n"' + path + '" *copied* as "' + path_orig + '"')
            else:
                if verbose:
                    print('\n"' + path_orig +
                          '" file already exists (nothing copied)')
        except OSError:
            if verbose:
                print('\n"' + path + '" file *not found*')

        if return_path_and_df:
            try:
                return path, pd.read_excel(path, sheetname=None)
            except OSError:
                print('\nproblem reading/returning ' + path + '"')
        return

    # restore <file>_orig.xlsx as <file>.xlsx
    if revert:
        try:
            if os.path.isfile(path_orig):
                os.remove(path)
                os.rename(path_orig, path)
                if verbose:
                    print('\n"' + path_orig + '" *restored* as "' + path + '"')
            else:
                if verbose:
                    print('\n"' + path_orig + '" file *not found*')
        except OSError:
            if verbose:
                print('\n' + path + ' file not found')


def anon_names(length=10,
               min_seg=3,
               max_seg=3,
               add_rev=False,
               df=None,
               inplace=False):
    '''Generate a list of random strings
    Output may be used to anonymize a dataset name column
    The length of the output strings will be determined by the min_seg and
    max_seg inputs.  The segments (seg) are random 2-letter combinations of
    a consonant and a vowel.  An additional random consonant or vowel will
    be added to the segment combinations, so the length of the output strings
    will always be an odd number.  The min and max may be the same value to
    produce a list of strings of uniform length.
    Example:
    If the min_seg input is 1 and the max_seg input is 3, the output list will
    contain strings from 3 (2-letter seg + 1 random letter) to 7 characters.
    inputs
        length (integer)
            the length of the output list
        min_seg (integer)
            the minimum number of 2 letter segments to include in the output
            list
        max_seg (integer)
            the maximum number of 2 letter segments to include in the output
            list (must be => "min_seg" input)
        add_rev (boolean)
            add vowel-consonant combinations to the consonant-vowel segments.
            (this is not normally needed to produce random and readable
            strings)
        df (dataframe)
            optional short-form pandas dataframe input.  If not None, use the
            length of the dataframe as the "length" input value
        inplace (boolean)
            if the "df" input is not None, insert the results directly into
            the input "lname" column.  Caution: make a copy first!
    '''
    segs = []
    vowels = 'aeiou'
    letters = 'abcdefghijklmnopqrstuvwxyz'
    anon_list = []

    for l in letters:
        for v in vowels:
            segs.append(l + v)
    if add_rev:
        rev_segs = [el[::-1] for el in segs if el[::-1] not in segs]
        segs.extend(rev_segs)

    segs.remove('fu')
    segs.remove('hi')
    segs.remove('cu')
    segs.remove('co')
    segs.remove('mo')

    rnd = len(segs) - 1

    if df is not None:
        length = len(df)

    for n in range(length):
        anon = ''
        num_segs = random.randint(min_seg, max_seg)
        for i in range(num_segs):
            anon = anon + segs[random.randint(1, rnd)]
        anon = anon + letters[random.randrange(0, 25)]
        anon_list.append(anon)

    if not inplace:
        return anon_list
    else:
        df['lname'] = anon_list


def anon_empkeys(df,
                 seq_start=10001,
                 frame_num=10000000,
                 inplace=False):
    '''Produce a list of unique, randomized employee numbers, catogorized
    by employee group number code.  Output may be used to anonymize a dataset
    empkey column.
    Dataframe input (df) must contain an employee group (eg) column.
    inputs
        df (dataframe)
            short-form (master list) pandas dataframe containing an employee
            group code column
        seq_start (integer)
            this number will be added to each employee group cumulative count
            to "seed" the random employee numbers.  These numbers will be
            shuffled within employee groups by the function for the output
        frame_num (integer)
            This number will be multiplied by each employee code and added to
            the employee group cumulative counts (added to the seq_start
            number), and should be much larger than the data model population
            to provide a constant length employee number (empkey) for all
            employees.
        inplace (boolean)
            if True, insert the results directly into the "empkey" column
            of the input dataframe.  Caution: make a copy first!
    '''
    df0 = df[['eg']].copy()
    df0['new_emp'] = (df0.groupby('eg').cumcount() + seq_start) +\
        (df0.eg * frame_num)

    eg = df0.eg.values
    emp = df0.new_emp.values

    for eg_num in np.unique(eg):
        emp_slice = emp[np.where(eg == eg_num)[0]]
        shuffled = np.random.permutation(emp_slice)
        np.put(emp,
               np.where(eg == eg_num)[0],
               shuffled)
    if inplace:
        df.empkey = emp
    else:
        return emp


def anon_dates(df,
               date_col_list,
               max_adj=5,
               positive_only=True,
               inplace=False):
    '''Add (or optionally, add or subtract) a random number of days to each
    element of a date attribute column.
    inputs
        df (dataframe)
            short-form (master list) pandas dataframe containing a date
            attribute column
        date_col_list (list)
            name(s) of date attribute column(s) to be adjusted (as a list
            of strings)
            Example:
                ['ldate', 'doh', 'dob']
        max_adj (integer)
            the maximum number of days to add (or optionally subtract) from
            each element within the date column
        positive_only (boolean)
            if True limit the range of adjustment days from zero to the
            max_adj value.  If False, limit the range of adjustment from
            negative max_adj value to positive max_adj value.
        inplace (boolean)
            if True, insert the results directly into the date column(s)
            of the input dataframe.  Caution: make a copy first!
    '''
    # p_adj is only positive adjustment
    # pn_adj is both positive and negative adjustment
    df0 = df.copy()
    if positive_only:
        adj_range = range(max_adj)
        # make arrays slightly longer than dataframe length
    else:
        adj_range = range(-max_adj, max_adj)

    adj = np.random.permutation(np.repeat(adj_range,
                                          int(len(df) / len(adj_range)) + 1))

    df0['adjust'] = adj[:len(df)]
    days_adjust = pd.TimedeltaIndex(df0['adjust'], unit='D')

    for col in date_col_list:
        if inplace:
            df[col] = df0[col] + days_adjust
        else:
            df0[col] = df0[col] + days_adjust

    if not inplace:
        return df0[date_col_list]


def anon_pay(df,
             proportional=True,
             mult=1.0,
             inplace=False):
    '''Substitute pay table baseline rate information a proportional method
    or with a non-linear, non-proportional method.
    inputs
        df (dataframe)
            pandas dataframe containing pay rate date (dataframe
            representation of the "rates" worksheet from the pay_tables.xlsx
            workbook)
        proportional (boolean)
            if True, use the mult input to increase or decrease all of the
            "rates" worksheet pay data proportionally.  If False, use a fixed
            algorithm to disproportionally adjust the pay rates.
        mult (integer or float)
            if the proportional input is True, multiply all pay rate values
            by this input value
        inplace (boolean)
            if True, replace the values within the original dataframe with
            the "anonomized" values.
    '''
    df0 = df.copy()

    all_cols = df.columns.values.tolist()
    val_cols = [col for col in all_cols if type(col) == int]
    data = df0[val_cols].values

    if proportional:
        arr_mod = data * mult
    else:
        arr_mod = np.where(data > 0,
                           (((((data - 17) / 1.4) + 20) / 1.3) - 5) * 1.1, 0)

    if inplace:
        df[val_cols] = arr_mod
    else:
        df0[val_cols] = arr_mod

    if not inplace:
        return df0[val_cols]


def sample_dataframe(df,
                     n=None,
                     frac=None,
                     reset_index=False):
    '''Get a random sample of a dataframe by rows, with the number of rows
    in the returned sample defined by a count or fraction input.
    inputs
        df (dataframe)
            pandas dataframe for sampling
        n (integer)
            If not None, the count of the rows in the returned sample
            dataframe.
            The "n" input will override the "frac" input if both are not None.
            Will be clipped between zero and len(df) if input exceeds these
            boundries.
        frac (float)
            If not None, the size of the returned sample dataframe relative to
            the input dataframe.  Will be ignored if "n" input is not None.
            Will be clipped between 0.0 and 1.0 if input exceeds these
            boundries.  An input of .3 would randomly select 30% of the rows
            from the input dataframe.
        reset_index (boolean)
            If True, reset the output dataframe index
    If both the "n" and "frac" inputs are None, a random single row will be
    returned.
    The rows in the output dataframe will be sorted according to original
    order.
    '''
    # set "frac" input to None if "n" input is not None (cannot use both
    # inputs at once)
    if (frac is None) and (n is None):
        return df

    if (frac is not None) and (n is not None):
        frac = None

    # make an order column for sorting after sample operation
    df['df_order'] = np.arange(len(df)) + 1

    if n is not None:
        n = np.clip(n, 0, len(df))
        df = df.sample(n=n)

    else:
        if frac is not None:
            frac = np.clip(frac, 0.0, 1.0)
        df = df.sample(frac=frac)

    # use order column to restore original order to sampled data
    df.sort_values('df_order', inplace=True)
    # get rid of utility order column
    df.pop('df_order')

    if reset_index:
        df.reset_index(drop=True, inplace=True)

    # recalculate "eg_order" column if it exists in dataframe
    try:
        df['eg_order'] = df.groupby('eg').cumcount() + 1
    except:
        pass

    return df


def anon_master(case,
                empkey=True,
                name=True,
                date=False,
                sample=False,
                # empkey
                seq_start=10001, frame_num=10000000,
                # name
                min_seg=3, max_seg=3, add_rev=False,
                # date
                date_col_list=['ldate', 'doh'],
                max_adj=5, positive_only=True,
                date_col_list_sec=['dob'],
                max_adj_sec=5, positive_only_sec=True,
                # sample
                n=None, frac=None, reset_index=False):
    '''Specialized function to anonymize selected columns from a master.xlsx
    file and/or select a subset.  All operations are inplace.  The original
    master file is copied and saved as master_orig.xlsx.
    The default parameters will replace last names and employee keys with
    substitute values.  Date columns, (doh, ldate, dob) will also be adjusted
    if the date input is set True and the proper column names are set as
    column list inputs.
    The function reads the original excel file, copies and saves it, modifies
    the original file as directed, and writes the results back to the original
    file.  Subsequent dataset creation runs will use the modified data.
    The output master list will be sorted according to the original master
    list order.
    inputs
        case (string)
            the case study name
        empkey (boolean)
            if True, anonymize the empkey column
        name (boolean)
            if True, anonymize the lname column
        date (boolean)
            if True, anonymize date columns as disignated with the
            date_col_list and the date_col_list_sec inputs
        sample (boolean)
            if True, sample the dataframe if the n or frac inputs is/are not
            None
        seq_start (integer)
            beginning anonymous employee number portion of empkey
        frame_num (integer)
            large frame number which will contain all generated employee
            numbers.  This number will be adjusted to begin with the
            appropriate employee group code
        min_seg (integer)
            minimum number of 2-character segments to include in the generated
            substitute last names.
        max_seg (integer)
            maximum number of 2-character segments to include in the generated
            substitute last names.
        add_rev (boolean)
            if True, add reversed, non-duplicated 2-character segments to the
            pool of strings for name construction.  This is normally not
            necessary and will construct output strings with multiple
            consecutive consonants/vowels.
        date_col_list (list)
            list of date value columns to adjust.  All columns in this list
            will be adjusted in a syncronized fashion, meaning a random day
            adjustment for each row will be applied to each row member of all
            columns.
        max_adj (integer)
            maximum random adjustment deviation, in days, from the original
            date(s)
        positive_only (boolean)
            if True, only adjust dates forward in time
        date_col_list_sec (list)
            a secondary list of date column(s) which will be adjusted
            independently from the date columns in the date_col_list
        max_adj_sec (integer)
            maximum random adjustment deviation, in days, from the original
            date(s) in the date_col_list_sec columns
        positive_only_sec (boolean)
            if True, only adjust dates forward in time (for secondary cols)
        n (integer or None)
            number of rows to sample if the sample input is True.  This input
            will override the frac input
        frac (float (0.0 - 1.0) or None)
            decimal fraction (0.0 to 1.0) of the master list to sample, if
            the sample input is True and the n input is None
        reset_index (boolean)
            if True, reset the index of the output master list (zero-based
            integer index).  Do not use this option normally because it will
            wipe out the empkey index of the master list.
    '''
    inplace = True
    anon_cols = []
    attributes = [empkey, name, date]

    if any(attributes):
        path, d = copy_excel_file(case, 'master', return_path_and_df=True)
        df = d['master']

        if sample:
            df = sample_dataframe(df, n=n, frac=frac,
                                  reset_index=reset_index).copy()
            print('\n  input master length:', len(d['master']))
            print('sampled master length:', len(df))

        if empkey:
            anon_empkeys(df, seq_start=seq_start,
                         frame_num=frame_num,
                         inplace=inplace)
            anon_cols.append('empkey')

        if name:
            anon_names(min_seg=min_seg,
                       max_seg=max_seg,
                       add_rev=add_rev,
                       df=df,
                       inplace=inplace)
            anon_cols.append('name')

        if date:
            if date_col_list:
                anon_dates(df,
                           date_col_list,
                           max_adj=max_adj,
                           positive_only=positive_only,
                           inplace=inplace)
                anon_cols.extend(date_col_list)

            if date_col_list_sec:
                anon_dates(df,
                           date_col_list_sec,
                           max_adj=max_adj_sec,
                           positive_only=positive_only_sec,
                           inplace=inplace)
                anon_cols.extend(date_col_list_sec)

        d['master'] = df
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:

            for ws_name, df_sheet in d.items():
                df_sheet.to_excel(writer, sheet_name=ws_name)

        print('\n"' + path + '" anonymized!\n', '   columns: ',
              anon_cols, '\n')


def anon_pay_table(case,
                   proportional=True,
                   mult=1.0,):
    '''Anonymize the "rates" worksheet of the "pay_tables.xlsx" input file.
    The rates may be proportionally adjusted (larger or smaller) or
    disproportionally adjusted with a fixed algorithm.
    A copy of the original excel file is copied and saved as
    "pay_tables_orig.xlsx".
    All modifications are inplace.
    inputs
        case (string)
            the case name
        proportional (boolean)
            if True, use the mult input to increase or decrease all of the
            "rates" worksheet pay data proportionally.  If False, use a fixed
            algorithm to disproportionally adjust the pay rates.
        mult (integer or float)
            if the proportional input is True, multiply all pay rate values
            by this input value
    '''
    inplace = True

    path, d = copy_excel_file(case, 'pay_tables', return_path_and_df=True)
    df = d['rates']
    anon_pay(df,
             proportional=proportional,
             mult=mult,
             inplace=inplace)
    d['rates'] = df

    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:

        for ws_name, df_sheet in d.items():
            df_sheet.to_excel(writer, sheet_name=ws_name)

    print('\nanon_pay_table routine complete')


def find_index_val(df1, df2, df2_vals, col1=None, col2=None):
    '''find a value in another dataframe with the same index of
    another given value in a dataframe.
    df1 index, df2 index, and the value columns must contain unique values.
    inputs
        df1 (dataframe)
            the first dataframe containing values to index match in another
            dataframe
        df2 (dataframe)
            the second dataframe with corresponding index values
        df2_vals (list)
            values to match
    '''

    # initiate df1_vals list
    df1_vals = []

    # set df1 value column, use first column if None
    if col1 is not None:
        column1 = df1[col1]
    else:
        column1 = df1[df1.columns[0]]

    # set df2 value column, use first column if None
    if col2 is not None:
        column2 = df2[col2]
    else:
        column2 = df2[df2.columns[0]]

    # find index for df1 value
    # find value in df2 with corresponding index from df1
    for v in df2_vals:

        try:
            df1_idx = df2[column2 == v].index[0]
            result_val = column1.loc[df1_idx]
            # append df2 value to df1_vals list
            df1_vals.append(result_val)

        except KeyError:
            print('value ' + str(v) + ' error:',
                  'no corresponding index')
            return

    return df1_vals


def convert_to_hex(rgba_input):
    '''convert float rgba color values to string hex color values
    rgba = color values expressed as:
        red, green, blue, and (optionally) alpha float values
    rgba_input may be:
    1. a single rgba list or tuple
    2. a list or tuple containing rgba lists or rgba tuples
    3. a dictionary of key: rgba values
    output is string hex color values in place of rgba values
    Examples:
    input single rgba value:
    .. code:: python
      sample_value = (.5, .3, .2)
      convert_to_hex(sample_value)
      '#7f4c33'
    input list:
    .. code:: python
      sample_list = [[0.65, 0.81, 0.89, 1.0],
                     [0.31, 0.59, 0.77, 1.0],
                     [0.19, 0.39, 0.70, 1.0],
                     [0.66, 0.85, 0.55, 1.0]]
      convert_to_hex(sample_list)
      ['#a5cee2', '#4f96c4', '#3063b2', '#a8d88c']
    input dict:
    .. code:: python
      sample_dict = {1: (.65, .45, .45, 1.),
                     2: [.60, .45, .45, 1.],
                     3: (.55, .45, .45, 1.)}
      convert_to_hex(sample_dict)
      {1: '#a57272', 2: '#99593a', 3: '#8c7249'}
    inputs
        rgba_input (tuple, list, or dictionary)
            input may be a single list or tuple OR a list of float rgba
            values as lists or tuples OR a dictionary with values as
            lists or tuples.  Valid string hex values may also be
            passed as inputs and will be returned unchanged.
    '''
    if type(rgba_input) in [list, tuple]:
        if type(rgba_input[0]) not in [list, tuple]:
            c0 = int(rgba_input[0] * 255)
            c1 = int(rgba_input[1] * 255)
            c2 = int(rgba_input[2] * 255)
            return '#%02x%02x%02x' % (c0, c1, c2)
        color_list = []
        for rgba in rgba_input:
            if type(rgba) in [tuple, list]:
                c0 = int(rgba[0] * 255)
                c1 = int(rgba[1] * 255)
                c2 = int(rgba[2] * 255)
                color_list.append('#%02x%02x%02x' % (c0, c1, c2))
            else:
                color_list.append(rgba)
        return color_list
    elif type(rgba_input) is dict:
        for key in rgba_input.keys():
            if type(rgba_input[key]) in [tuple, list]:
                c = rgba_input[key]
                c0 = int(c[0] * 255)
                c1 = int(c[1] * 255)
                c2 = int(c[2] * 255)
                rgba_input[key] = '#%02x%02x%02x' % (c0, c1, c2)
        return rgba_input
    else:
        print('invalid convert_to_hex function input')
        return rgba_input


@jit(cache=True)
def find_nearest(col_arr, value):
    idx = np.searchsorted(col_arr, value)
    if idx > 0 and (idx == len(col_arr) or
                    np.fabs(value - col_arr[idx - 1]) <
                    np.fabs(value - col_arr[idx])):
        return col_arr[idx - 1]
    else:
        return col_arr[idx]


# for stripplot, col2 will always be the integrated list original order values
@jit(cache=True)
def cross_val(col1, value, col2):
    idx = np.searchsorted(col1, value)
    if idx > 0 and (idx == col1.size or
                    np.fabs(value - col1[idx - 1]) <
                    np.fabs(value - col1[idx])):
        nearest = col1[idx - 1]
    else:
        nearest = col1[idx]
    return col2[np.where(col1 == nearest)[0]][0]


def hex_dict():
    hex_dict = {'white': '#ffffff',
                'azure': '#f0ffff',
                'mintcream': '#f5fffa',
                'snow': '#fffafa',
                'ivory': '#fffff0',
                'ghostwhite': '#f8f8ff',
                'floralwhite': '#fffaf0',
                'aliceblue': '#f0f8ff',
                'lightcyan': '#e0ffff',
                'honeydew': '#f0fff0',
                'lightyellow': '#ffffe0',
                'seashell': '#fff5ee',
                'lavenderblush': '#fff0f5',
                'whitesmoke': '#f5f5f5',
                'oldlace': '#fdf5e6',
                'cornsilk': '#fff8dc',
                'linen': '#faf0e6',
                'lightgoldenrodyellow': '#fafad2',
                'lemonchiffon': '#fffacd',
                'beige': '#f5f5dc',
                'lavender': '#e6e6fa',
                'papayawhip': '#ffefd5',
                'mistyrose': '#ffe4e1',
                'antiquewhite': '#faebd7',
                'blanchedalmond': '#ffebcd',
                'bisque': '#ffe4c4',
                'paleturquoise': '#afeeee',
                'moccasin': '#ffe4b5',
                'gainsboro': '#dcdcdc',
                'peachpuff': '#ffdab9',
                'navajowhite': '#ffdead',
                'palegoldenrod': '#eee8aa',
                'wheat': '#f5deb3',
                'powderblue': '#b0e0e6',
                'aquamarine': '#7fffd4',
                'lightgrey': '#d3d3d3',
                'pink': '#ffc0cb',
                'lightblue': '#add8e6',
                'thistle': '#d8bfd8',
                'lightpink': '#ffb6c1',
                'lightskyblue': '#87cefa',
                'palegreen': '#98fb98',
                'lightsteelblue': '#b0c4de',
                'khaki': '#f0d58c',
                'skyblue': '#87ceeb',
                'aqua': '#00ffff',
                'cyan': '#00ffff',
                'silver': '#c0c0c0',
                'plum': '#dda0dd',
                'gray': '#bebebe',
                'lightgreen': '#90ee90',
                'violet': '#ee82ee',
                'yellow': '#ffff00',
                'turquoise': '#40e0d0',
                'burlywood': '#deb887',
                'greenyellow': '#adff2f',
                'tan': '#d2b48c',
                'mediumturquoise': '#48d1cc',
                'lightsalmon': '#ffa07a',
                'mediumaquamarine': '#66cdaa',
                'darkgray': '#a9a9a9',
                'orchid': '#da70d6',
                'darkseagreen': '#8fbc8f',
                'deepskyblue': '#00bfff',
                'sandybrown': '#f4a460',
                'gold': '#ffd700',
                'mediumspringgreen': '#00fa9a',
                'darkkhaki': '#bdb76b',
                'cornflowerblue': '#6495ed',
                'hotpink': '#ff69b4',
                'darksalmon': '#e9967a',
                'darkturquoise': '#00ced1',
                'springgreen': '#00ff7f',
                'lightcoral': '#f08080',
                'rosybrown': '#bc8f8f',
                'salmon': '#fa8072',
                'chartreuse': '#7fff00',
                'mediumpurple': '#9370db',
                'lawngreen': '#7cfc00',
                'dodgerblue': '#1e90ff',
                'yellowgreen': '#9acd32',
                'palevioletred': '#db7093',
                'mediumslateblue': '#7b68ee',
                'mediumorchid': '#ba55d3',
                'coral': '#ff7f50',
                'cadetblue': '#5f9ea0',
                'lightseagreen': '#20b2aa',
                'goldenrod': '#daa520',
                'orange': '#ffa500',
                'lightslategray': '#778899',
                'fuchsia': '#ff00ff',
                'magenta': '#ff00ff',
                'mediumseagreen': '#3cb371',
                'peru': '#cd853f',
                'steelblue': '#4682b4',
                'royalblue': '#4169e1',
                'slategray': '#708090',
                'tomato': '#ff6347',
                'darkorange': '#ff8c00',
                'slateblue': '#6a5acd',
                'limegreen': '#32cd32',
                'lime': '#00ff00',
                'indianred': '#cd5c5c',
                'darkorchid': '#9932cc',
                'blueviolet': '#8a2be2',
                'deeppink': '#ff1493',
                'darkgoldenrod': '#b8860b',
                'chocolate': '#d2691e',
                'darkcyan': '#008b8b',
                'dimgray': '#696969',
                'olivedrab': '#6b8e23',
                'seagreen': '#2e8b57',
                'teal': '#008080',
                'darkviolet': '#9400d3',
                'mediumvioletred': '#c71585',
                'orangered': '#ff4500',
                'olive': '#808000',
                'sienna': '#a0522d',
                'darkslateblue': '#483d8b',
                'darkolivegreen': '#556b2f',
                'forestgreen': '#228b22',
                'crimson': '#dc143c',
                'blue': '#0000ff',
                'darkmagenta': '#8b008b',
                'darkslategray': '#2f4f4f',
                'saddlebrown': '#8b4513',
                'brown': '#a52a2a',
                'firebrick': '#b22222',
                'purple': '#800080',
                'green': '#008000',
                'red': '#ff0000',
                'mediumblue': '#0000cd',
                'indigo': '#4b0082',
                'midnightblue': '#191970',
                'darkgreen': '#006400',
                'darkblue': '#00008b',
                'navy': '#000080',
                'darkred': '#8b0000',
                'maroon': '#800000',
                'black': '#000000'}
    return hex_dict


def remove_zero_groups(ratio_dict):
    '''remove data related to a "dummy" group represented by a zero

    example:

    ..code python
      {2: [([2], [0], [1]), [0, 2, 6], 34, 120]}

    becomes:

    ..code python
      {2: [([2], [1]), [0, 6], 34, 120]}

    inputs
        ratio_dict (dictionary)
            the ratio dictionary produced by the build_program_files script
            originating from the "ratio_cond" worksheet of the
            *settings.xlsx* input file
    '''
    for job in ratio_dict.keys():
        idx_list = []
        groups = []
        these_jobs = ratio_dict[job][0]
        for i, ratio_group in enumerate(these_jobs):
            if 0 in ratio_group:
                idx_list.append(i)
            else:
                groups.append(ratio_group)
        ratio_dict[job][0] = tuple(groups)
        if idx_list:
            weight_list = ratio_dict[job][1]
            idx_list = idx_list[::-1]
            for idx in idx_list:
                del weight_list[idx]
            ratio_dict[job][1] = weight_list
    return ratio_dict


def eg_quotas(quota, actual, cap=None, this_job_count=None):
    '''determine the job counts to be assigned to each ratio group during
    a ratio condition job assignment routine

    inputs
        quota (list or list-like)
            the desired job counts for each employee group
        actual (list or list-like)
            the actual job counts for each employee group
        cap (integer (or whole float))
            if a count ratio routine is being used, this is
            the total count of jobs to be affected by the ratio
        this_job_count (integer (or whole float))
            the monthly count of the applicable job
    '''

    quota_weights = []
    quota_range = range(len(quota))
    if cap:
        excess_avail = this_job_count - sum(actual)
    else:
        excess_avail = sum(quota) - sum(actual)

    actual = np.array(actual)
    # decide how to assign job openings, based on quotas and count of
    # jobs already occupied by each ratio group (the "actual" input)
    if excess_avail > 0:
        for i in quota_range:
            quota_weights.append(quota[i] - actual[i])
    if excess_avail <= 0:
        for i in quota_range:
            quota_weights.append(actual[i] - quota[i])

    for i, f in enumerate(quota_weights):
        if quota_weights[i] > 0:
            quota_weights[i] = f
        else:
            quota_weights[i] = 0

    # calculate the distribution of job openings
    additives = np.array(distribute(excess_avail, quota_weights))

    if cap:
        shortfall = []
        for i in quota_range:
            temp = quota[i] - actual[i]
            if temp > 0:
                shortfall.append(temp)
            else:
                shortfall.append(0)

        additives = np.minimum(additives, np.array(shortfall))

    # combine the above distribution with the original actual counts
    grp_assign_count = actual + additives

    return grp_assign_count
