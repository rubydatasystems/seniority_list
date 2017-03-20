# -*- coding: utf-8 -*-

'''month_form is length n months in model

short_form is length n employees

long_form is length cumulative sum non-retired each month

(could be millions of rows, depending on workgroup
size and age)
'''
import os
import shutil
import copy
import pandas as pd
import numpy as np
import scipy.stats as st
from numba import jit
from collections import OrderedDict as od


# CAREER MONTHS
def career_months_list_in(ret_list, start_date):
    '''Short_Form

    Determine how many months each employee will work
    including retirement partial month.

    This version takes a list of retirement dates

    inputs
        ret_list
            list of retirement dates in datetime format
        start_date
            comparative date for retirement dates, starting date for the
            data model
    '''
    start_date = pd.to_datetime(start_date)
    s_year = start_date.year
    s_month = start_date.month
    cmths = []
    for retdate in ret_list:
        cmths.append(((retdate.year - s_year) * 12) -
                     (s_month - retdate.month))

    return np.array(cmths)


# CAREER MONTHS
def career_months_df_in(df, startdate):
    '''Short_Form

    Determine how many months each employee will work
    including retirement partial month.
    This version has a df as input
    - df must have 'retdate' column of retirement dates

    inputs
        df
            dataframe containing a column of retirement dates
            in datetime format
        start_date
            comparative date for retirement dates, starting date for the
            data model
    '''
    start_date = pd.to_datetime(startdate)
    rets = list(df.retdate)
    cmths = []
    s_year = start_date.year
    s_month = start_date.month

    for mth in rets:
        cmths.append(((mth.year - s_year) * 12) - (s_month - mth.month))

    return np.array(cmths)


# LONGEVITY AT STARTDATE (for pay purposes)
def longevity_at_startdate(ldates_list, start_date,
                           return_months=False):
    ''' Short_Form

    - determine how much longevity (years) each employee has accrued
      as of the start date
    - input is list of longevity dates
    - float output is longevity in years
      (+1 added to reflect current 1-based pay year)
    - option for output in months

    inputs
        ldates_list
            list of longevity dates in datetime format
        start_date
            comparative date for retirement dates, starting date for the
            data model
        return_months (boolean)
            option to return result as month value instead of year value
    '''
    start_date = pd.to_datetime(start_date)
    s_year = start_date.year
    # subtract one month so pay increase begins
    # in month after anniversary month
    s_month = start_date.month - 1
    longevity_list = []

    if return_months:
        for ldate in ldates_list:
            longevity_list.append((((s_year - ldate.year) * 12) -
                                   (ldate.month - s_month)) + 1)
    else:
        for ldate in ldates_list:
            longevity_list.append(((((s_year - ldate.year) * 12) -
                                    (ldate.month - s_month)) / 12) + 1)

    return longevity_list


# AGE AT START DATE
def starting_age(dob_list, start_date):
    '''Short_Form

    Returns decimal age at given date.
    Input is list of birth dates.

    inputs
        dob_list
            list of birth dates in datetime format
        start_date
            comparative date for retirement dates, starting date for the
            data model
    '''
    start_date = pd.to_datetime(start_date)
    s_year = start_date.year
    s_month = start_date.month
    s_day = start_date.day
    m_val = 1 / 12
    ages = []
    for dob in dob_list:
        ages.append(m_val * (((s_year - dob.year) * 12) -
                             (dob.month - s_month) +
                             ((s_day - dob.day) / s_day)))

    return ages


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
            output of career_months_list_in or career_months_df_in
            functions.  This input is an array containing the number of
            months each employee will work until retirement.

    '''
    max_career = np.max(career_months_array) + 1
    emp_count_array = np.zeros(max_career)

    for i in np.arange(0, max_career):
        emp_count_array[i] = np.sum(career_months_array >= i)

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
    and the result of the career_months_df_in (or ...list_in)
    function

    inputs
        monthly_count_array (numpy array)
            count of non-retired active employees for each month in the model,
            the ouput from the count_per_month function.
        career_mths_array (numpy array)
            career length in months for each employee, output of
            career_months_list_in or career_months_list_in functions.
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
    for j in np.arange(0, int(np.max(career_mths_array)) + 1):
        idx = 0
        for i in emp_idx:
            if career_mths_array[i] >= j:
                skel_idx_array[k] = idx
                skel_empkey_array[k] = empkey_source_array[idx]
                k += 1
            idx += 1

    return skel_idx_array, skel_empkey_array


# AGE FOR EACH MONTH (correction to starting age)
# @jit  (jit broken with numba version update 0.28.1, np111py35_0)
def age_correction(month_nums_array, ages_array, retage):
    '''Long_Form

    Returns a long_form (all months) array of employee ages by
    incrementing starting ages according to month number.

    inputs
        month_nums_array
            gen_month_skeleton function output (ndarray)
        ages_array
            starting_age function output aligned with long_form (ndarray)
            i.e. s_age is starting age (aligned to empkeys)
            repeated each month.
        retage
            output clip upper limit

    Output is s_age incremented by a decimal month value according to month_num
    (this is candidate for np.put refactored function)
    '''
    month_val = 1 / 12
    array_len = month_nums_array.size
    result_array = np.ndarray(array_len)
    for i in np.arange(array_len):
        result_array[i] = ((month_nums_array[i] * month_val) + ages_array[i])
    result_array = np.clip(result_array, 0, retage)

    return result_array


# FIND CONTRACT PAY YEAR AND RAISE (contract pay year
# and optional raise multiplier)
def contract_pay_year_and_raise(date_list, future_raise,
                                date_exception_start,
                                date_exception_end,
                                exception_year,
                                annual_raise,
                                last_contract_year):
    '''Month_Form

    Generate the contract pay year for indexing into the pay table.
    Pay year is clipped to last year of contract.
    If desired, an annual assumed raise beyond the contract time frame
    may be elected.

    Result is an array with two columns - column [0] contains the pay year,
    column [1] contains the multiplier for any raise (default is 1.0,
    which remains from the np.ones initial array.

    Usage example:

        ::

            year_scale = find_scale(series_years,
                                    future_raise = True,
                                    annual_raise = .02)

    NOTE: **(this function can accept a pay exception time period...)**

    inputs
        date_list
            time series format list of dates
        future_raise
            option for pay calculations to apply a
            percentage increase for each year beyond
            the last contract year
        date_exception_start
            date representing the first month of the outlier pay month
            range as a string, example: '2014-12-31'
        date_exception_end
            date representing the final month of the outlier pay month
            range as a string, example: '2014-12-31', can be identical to
            date_exception_start input for a single month exception
        exception_year
            year value (float) representing an exception pay rate.  This
            value must match exception year float number from pay table
            sheets 'year' columns within the Excel input workbook,
            pay_tables.xlsx.
            This value is simply a placeholder value to mark months with an
            contract exception pay table.
        annual_raise
            yearly raise to calculate beyond the last contract year
            if future_raise option is selected
        last_contract_year
            last year of contract pay rate changes
    '''
    float_years = np.ones(len(date_list) * 2)
    float_years = float_years.reshape(len(date_list), 2)
    date_exception_range = pd.date_range(date_exception_start,
                                         date_exception_end,
                                         freq='M')

    for i in np.arange(0, len(date_list)):

        if future_raise:
            float_years[i][1] = \
                np.clip((1 + annual_raise) **
                        (date_list[i].year - int(last_contract_year)),
                        1, 1000)
            float_years[i][0] = np.clip(date_list[i].year,
                                        0.0, last_contract_year)
        else:
            float_years[i][0] = np.clip(date_list[i].year,
                                        0.0, last_contract_year)

        if exception_year:
            if date_list[i] in date_exception_range:
                float_years[i][0] = exception_year

    return float_years.T


# MAKE eg INITIAL JOB LIST from job_count_array (Stovepipe)
def make_stovepipe_jobs_from_jobs_arr(jobs_arr, total_emp_count=0):
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
def make_intgrtd_from_sep_stove_lists(job_lists_arr, eg_arr,
                                      fur_arr, eg_total_jobs,
                                      num_levels, skip_fur=True):
    '''Month_Form

    Compute an integrated job list built from multiple
    independent eg stovepiped job lists.

    (old name: make_jobs_arr_from_job_lists)

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
        skip_fur (option)
            ignore or skip furloughs when assigning stovepipe jobs

    This function is for multiple egs (employee groups) - multiple lists in
    one job_lists_arr.

    Creates an ndarray of job numbers.

    Function takes independent job number lists and an array of eg codes
    which represent the eg ordering in the proposed list.

    Job numbers from the separate lists are added to the result array
    according to the eg_arr order.  Jobs on each list do not have to be
    in any sort of order.  The routine simply adds items from the list(s)
    to the result array slots in list order.

    skip_fur option:
        Employees who are originally marked as furloughed are
        assigned the furlough level number which is 1 greater
        than the number of job levels.
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
def make_stovepipe_prex_shortform(job_list, sg_codes,
                                  sg_rights, fur_codes):
    '''Short_Form

    Creates a 'stovepipe' job assignment within a single eg including a
    special job assignment condition for a subgroup.  The subgroup is
    identified with a 1 in the sg_codes array input, originating with
    the sg column in the master list.  This function applies a pre-existing
    (prior to the merger) contractual job condition, which is likely the
    result of a previous seniority integration.

    *old name: make_amer_stovepipe_short_prex*

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

    The subset group will have proirity assignment for the first n jobs
    in the affected job category, the remainding jobs
    are assigned in seniority order.

    The subgroup jobs are assigned in subgroup stovepipe order.

    This function is applicable to a condition with known job counts.
    The result of this function is used with standalone calculations or
    combined with other eg lists to form an integrated original
    job assignment list.
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
def make_original_jobs_from_counts(jobs_arr_arr, eg_array,
                                   fur_array, num_levels):
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


# ASSIGN JOBS FULL FLUSH
def assign_jobs_full_flush(monthly_nonret_counts, job_list, job_level_count):
    '''Long_Form

    Uses the nonret counts for each month to:
      a. determine the long form monthly slice for assignment, and
      b. slice the jobs list from the top for assignment

    The job_list input is the output from the make_stovepipe_jobs function
    using the totals of all eg job categories as input

    monthly_nonret_counts input is the monthly list of job counts from the
    count_per_month function

    This is the full bump and full flush version
    Extremely fast...

    Note:  this function has no adjustment for job changes over time

    inputs
        monthly_nonret_counts
            count of active, non-retired employees for each month
        job_list (numpy array)
            list of job level codes derived from the job counts, each job code
            is repeated for its respective count, and stacked with the other
            job codes - result is monotonic
        job_level_count
            number of active job levels in the model (do not count the
            furlough level)
    '''
    long_job_array = np.zeros(sum(monthly_nonret_counts))
    tcount = 0
    for i in np.arange(0, len(monthly_nonret_counts)):
        long_job_array[tcount:monthly_nonret_counts[i] + tcount] = \
            job_list[0:monthly_nonret_counts[i]]
        tcount += monthly_nonret_counts[i]

    long_job_array[long_job_array == 0] = job_level_count + 1
    return long_job_array.astype(int)


# ASSIGN JOBS FULL FLUSH - SKIP FUROUGHED EMPS
def assign_jobs_full_flush_skip_furs(monthly_nonret_counts,
                                     job_list,
                                     fur_arr,
                                     job_level_count):
    '''Long_Form

    Using the nonret counts for each month:

        a. determine the long form monthly slice for assignment
        b. slice the jobs list from the top for assignment,
           skipping furloughees

    This function is used within the standalone computation

    inputs
        monthly_nonret_counts
            monthly list of job counts from the count_per_month function
        job_list
            output from the make_stovepipe_jobs_from_jobs_arr function
        fur_arr
            long_form furlough codes (same size as long_job_array)
        job_level_count
            num_of_job_levels (ultimately from settings.xlsx)

    This is bump and flush (skipping furloughed employees)
    '''
    long_job_array = np.zeros(sum(monthly_nonret_counts))
    tcount = 0

    for i in np.arange(0, len(monthly_nonret_counts)):

        target_slice = long_job_array[tcount:monthly_nonret_counts[i] + tcount]
        fur_slice = fur_arr[tcount:monthly_nonret_counts[i] + tcount]
        jobs_segment = job_list[0:monthly_nonret_counts[i]]

        np.put(target_slice, np.where(fur_slice == 0)[0], jobs_segment)

        tcount += monthly_nonret_counts[i]

    long_job_array[long_job_array == 0] = job_level_count + 1

    return long_job_array.astype(int)


# ASSIGN JOBS FULL FLUSH with JOB COUNT CHANGES
def assign_jobs_full_flush_with_job_changes(monthly_nonret_counts,
                                            job_counts_each_month,
                                            job_level_count):
    '''Long_Form

    Using the nonret counts for each month:

      a. determine the long form slice for assignment, and
      b. slice the jobs list from the top for assignment

    Uses the job_counts_each_month (job_gain_loss_table function)[0] to
    build stovepiped job lists allowing for job count changes each month

    Unassigned employees (not enough jobs), are left at job number zero

    This is the full bump and full flush version

    inputs
        monthly_nonret_counts (numpy array)
            array containing the number of non-retired employees
            for each month
        job_counts_each_month (numpy array)
            array containing the monthly counts of jobs for each job level
        job_level_count (integer)
            the number of job levels in the model (excluding furlough)
    '''
    long_job_array = np.zeros(sum(monthly_nonret_counts)).astype(int)
    tcount = 0
    jc_skel = np.arange(job_counts_each_month[0].size)
    monthly_nonret_counts = monthly_nonret_counts.astype(int)

    for i in np.arange(0, len(monthly_nonret_counts)):

        job_list = np.repeat(jc_skel, job_counts_each_month[i]) + 1
        np.put(long_job_array,
               np.arange(tcount,
                         monthly_nonret_counts[i] + tcount)[:job_list.size],
               job_list)
        tcount += monthly_nonret_counts[i]

    long_job_array[long_job_array == 0] = job_level_count + 1
    return long_job_array.astype(int)


# ASSIGN JOBS NBNF JOB CHANGES
def assign_jobs_nbnf_job_changes(df,
                                 lower,
                                 upper,
                                 total_months,
                                 job_counts_each_month,
                                 total_monthly_job_count,
                                 job_reduction_months,
                                 start_month,
                                 condition_list,
                                 sdict,
                                 fur_return=False):
    '''Long_Form

    Uses the job_gain_or_loss_table job count array for job assignments.
    Jobs counts may change up or down in any category for any time period.
    Handles furlough and return of employees.
    Handles prior rights/conditions and restrictions.
    Handles recall of initially furloughed employees.

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
        total_months
            sum of count_per_month function output
        job_counts_each_month
            output of job_gain_loss_table function[0]
            (precalculated monthly count of jobs in each job category,
            size (months,jobs))
        total_monthly_job_count
            output of job_gain_loss_table function[1]
            (precalculated monthly total count of all job categories,
            size (months))
        job_reduction_months
            months in which the number of jobs is decreased (list).
            from the get_job_reduction_months function
        start_month
            integer representing the month number to begin calculations,
            likely month of integration when there exists a delayed
            integration (from settings dictionary)
        condition_list (list)
            list of special job assignment conditions to apply,
            example: ['prex', 'count', 'ratio']
        sdict (dictionary)
            the program settings dictionary (produced by the
            build_program_files script)
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
    orig = np.array(df.orig_job)
    eg_data = np.array(df.eg)
    sg_ident = np.array(df.sg)
    fur_data = np.array(df.fur)
    index_data = np.array(df.index)

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

        ratio_cond_month = min(ratio_month_range)
        job_change_months = job_change_months.union(ratio_month_range)

        # calc capped count condition month range and concat
    if 'count' in condition_list:
        count_dict = sdict['count_ratio_dict']
        count_jobs = sorted(count_dict.keys())
        cr_mdict = {}
        for job in count_dict.keys():
            cr_mdict[job] = set(range(count_dict[job][3],
                                      count_dict[job][4] + 1))

        dkeys = {'grp': 0, 'wgt': 1, 'cap': 2}

        count_month_range = sdict['count_ratio_month_range']
        job_change_months = job_change_months.union(count_month_range)

    if fur_return:
        recalls = sdict['recalls']
        recall_months = set(get_recall_months(recalls))
        job_change_months = job_change_months.union(recall_months)
    # convert job_change_months array to a set for faster membership test
    job_change_months = set(job_change_months)

    # loop through model integrated months:
    for month in np.arange(start_month, num_of_months):

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

        this_job_col = 0
        job = 1

        if month in job_reduction_months:
            mark_for_furlough(orig_job_range, fur_range, month,
                              total_monthly_job_count, num_of_job_levels)

        if fur_return and (month in recall_months):
            mark_for_recall(orig_job_range, num_of_job_levels,
                            fur_range, month, recalls,
                            total_monthly_job_count, standalone=False)

        # use numpy arrays for job assignment process for each month
        while job <= num_of_job_levels:

            this_job_count = job_counts_each_month[month, this_job_col]

            if month in job_change_months:

                # **pre-existing condition**
                if 'prex' in condition_list:

                    if (month in sg_month_range) and (job in sg_jobs):

                        # assign prex condition jobs to sg employees
                        sg_jobs_avail = min(sg_dict[job], this_job_count)
                        np.put(assign_range,
                               np.where((assign_range == 0) &
                                        (sg_range == 1) &
                                        (fur_range == 0))[0][:sg_jobs_avail],
                               job)

                # assign ratio condition jobs
                if 'ratio' in condition_list:

                    if month == ratio_cond_month:

                        ratio_cond_dict = set_snapshot_weights(ratio_dict,
                                                               orig_job_range,
                                                               eg_range)

                    if (job in ratio_jobs) and (month in r_mdict[job]):

                        assign_cond_ratio(job,
                                          this_job_count,
                                          ratio_cond_dict,
                                          orig_job_range,
                                          assign_range,
                                          eg_range,
                                          fur_range)

                # assign ratio count condition jobs
                if 'count' in condition_list:
                    if (job in count_jobs) and (month in cr_mdict[job]):
                        cap = count_dict[job][dkeys['cap']]
                        weights = count_dict[job][dkeys['wgt']]
                        ratio_groups = count_dict[job][dkeys['grp']]

                        assign_cond_ratio_capped(job,
                                                 this_job_count,
                                                 ratio_groups,
                                                 weights,
                                                 cap,
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
            jobs_avail = this_job_count - np.sum(assign_range == job)

            np.put(assign_range,
                   np.where((assign_range == 0) &
                            (orig_job_range <= job) &
                            (fur_range == 0))[0][:jobs_avail],
                   job)

            # assign remaining jobs by list order
            jobs_avail = this_job_count - np.sum(assign_range == job)
            np.put(assign_range,
                   np.where((assign_range == 0) &
                            (fur_range == 0))[0][:jobs_avail],
                   job)

            # insert corresponding job count
            np.put(job_count_range,
                   np.where(assign_range == job)[0],
                   this_job_count)

            this_job_col += 1
            job += 1

        # AFTER MONTHLY JOB LOOPS DONE, PRIOR TO NEXT MONTH:

        # pass down assign_range
        orig_next = align_next(index_range, index_range_next, assign_range)
        np.copyto(orig[L_next:U_next], orig_next)

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
               np.sum(fur_range == 1))

        fur_next = align_next(index_range, index_range_next, fur_range)
        np.copyto(fur_data[L_next:U_next], fur_next)

    # not part of month loops, cleaning up fur data for output
    long_assign_column[long_assign_column == 0] = num_of_job_levels + 1
    orig[orig == num_of_job_levels + 1] = 0

    return long_assign_column.astype(int), long_count_column.astype(int), \
        orig.astype(int), fur_data.astype(int)


# PUT-MAP function
def put_map(jobs_array, job_cnts):
    '''Any_Form (Practical application is Long_Form)

    Best use when values array is limited set of integers.

    10x faster than lambda function.
    dictionary-like value-key lookup using np.put and np.where

    inputs
        jobs_array
            long_form jnums
        jobs_cnts
            array of job counts from settings dictionary
            example: with 3 egs, array of 3 lists of counts

    Example:

        function call:

            ::

                map_jobs = put_map(no_bump_jnums, job_level_counts)

        assigned to df:

            ::

                df['nbnf_job_count'] = map_jobs.astype(int)


    length of set(jobs_array) must equal length of job_cnts.
    '''
    target_array = np.zeros(jobs_array.size)

    job_cnts = np.take(job_cnts, np.where(job_cnts != 0))[0]

    i = 0

    for job in sorted(list(set(jobs_array))):
        np.put(target_array,
               np.where(jobs_array == job),
               job_cnts[i])

        i += 1

    return target_array


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


def make_lower_and_upper_slice_limits(mnum_arr):
    '''calculate the monthly slice indexes for a
    long_form dataset.  Result is used to perform
    operations within month ranges of the dataset.

    Returns tuple (lower, upper)

    input
        mnum_arr
            nd.array of a long_form df mnum (month number) column
    '''
    mnum_data = np.unique(mnum_arr, return_counts=True)[1]
    lower = make_lower_slice_limits(mnum_data)
    upper = mnum_data.cumsum()
    return lower, upper


def snum_and_spcnt(jnum_arr, job_levels, low_limits, high_limits,
                   table_counts, all_mths):
    '''Calculates:

    long_form seniority number ('snum', only active employees),
    seniority percentage ('spcnt', only active employees),

    Iterate through monthly jobs count data, capturing monthly_job_counts
    to be used as the denominator for percentage calculations.

    This function produces two ndarrays which will make up two columns
    in the long_form pandas dataset.

    Note:  This function has been updated.  The new version produces lnum and
    lspcnt.  It is the create_snum_and_spcnt_arrays function...

    Returns tuple (long_snum, long_spcnt)

    inputs
        jnum_arr (numpy array)
            the long_form jnums (job numbers) column result
        job_levels (integer)
            number of job levels in model
        low_limits
            array of long-form start of each month data indexes
        high_limits
            array of long-form end of each month data indexes
        table_counts
            job_gain_loss_table function output[1],
            job counts for each job level,
            one row of counts for each month
        all_mths
            total sum of monthly active, non-retired employees (sum of
            all months in model)
    '''
    fur_level = job_levels + 1
    seq_nums = np.arange(high_limits[0] + high_limits[1]) + 1
    # all_months = np.sum(high_limits)
    long_snum = np.zeros(all_mths)
    long_spcnt = np.zeros(all_mths)
    num_of_months = high_limits.size
    for month in np.arange(num_of_months):

        L = low_limits[month]
        H = high_limits[month]
        jnum_range = jnum_arr[L:H]
        snum_range = long_snum[L:H]
        spcnt_range = long_spcnt[L:H]

        non_fur_indexes = np.where(jnum_range < fur_level)[0]

        np.put(snum_range,
               non_fur_indexes,
               seq_nums)
        np.put(snum_range,
               np.where(snum_range == 0)[0],
               None)
        np.copyto(spcnt_range, snum_range / table_counts[month])

    return long_snum, long_spcnt


# SNUMS
def create_snum_array(jobs_held, monthly_population_counts):
    '''Create an array of seniority numbers repeating for each month.

    Much faster than groupby cumcount...

    Furloughees are not assigned a seniority number.

    Returns ndarray for use in seniority number (snum) column.

    inputs
        jobs_held
            long_form array of jnums (job numbers) with unassigned employees
            (furloughed) indicated with a zero.
        monthly_population_counts
            array of non-retired employee counts for each month in model
    '''
    seq_nums = np.arange(1, monthly_population_counts[0] + 1)
    # TODO (for developer) consider np.sum vs sum below
    # (is input always np array?)
    long_snum_array = np.zeros(sum(monthly_population_counts))
    tcount = 0

    for i in np.arange(0, len(monthly_population_counts)):
        assign_range = \
            long_snum_array[tcount: monthly_population_counts[i] + tcount]
        jobs_held_range = \
            jobs_held[tcount: monthly_population_counts[i] + tcount]

        np.put(assign_range,
               np.where(jobs_held_range > 0)[0],
               seq_nums[0: monthly_population_counts[i]])

        tcount += monthly_population_counts[i]

    long_snum_array[long_snum_array == 0] = np.nan

    return long_snum_array.astype(int)


# SNUM, SPCNT, LNUM, LSPCNT with JOB CHANGES
def create_snum_and_spcnt_arrays(jnums, job_level_count,
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

    for i in np.arange(0, len(monthly_population_counts)):

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
               seq_nums[0: this_month_count])
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


# MAKE JOB COUNTS
def make_job_counts(furlough_list, *job_count_lists):
    '''Make two arrays:

    1. array of n lists of job counts for n number
    of eg job count input lists

    2. array of one summation list of first array
    (total count of all eg jobs)

    The last element of each array above will be a furlough count.

    Returns tuple (eg_job_counts, combined_job_count)

    inputs
        furlough_list
            a list of integers holding any furlough count for each eg
        job_count_lists
            the eg job count list(s)
    '''
    eg_job_counts = []
    i = 0

    for job_list in job_count_lists:

        j = list(job_list)
        j.append(furlough_list[i])
        i += 1

        eg_job_counts.append(j)

    eg_job_counts = np.array(eg_job_counts)
    combined_job_count = sum(np.array(eg_job_counts))

    return eg_job_counts.astype(int), combined_job_count.astype(int)


# MAKE JOB COUNTS (without furlough counts)
def make_array_of_job_lists(*job_count_lists):
    '''Make two arrays:

    1. array of n lists of job counts for n number
    of eg job count input lists

    2. array of one summation list of first array
    (total count of all eg jobs)

    (old function name: make_job_counts_without_fur)

    The arrays above will not contain a furlough count.

    Returns tuple (eg_job_counts, combined_job_count)

    inputs
        job_count_lists
            the eg job count list(s)
    '''
    eg_job_counts = []

    for job_list in job_count_lists:
        j = list(job_list)
        eg_job_counts.append(j)

    eg_job_counts = np.array(eg_job_counts)
    combined_job_count = sum(np.array(eg_job_counts))

    return eg_job_counts.astype(int), combined_job_count.astype(int)


# MAKE JOB COUNTS (without furlough counts)
def make_jcnts(job_count_lists):
    '''Make two arrays:

    1. array of n lists of job counts for n number
    of eg job count input lists

    2. array of one summation list of first array
    (total count of all eg jobs)

    (old function name: make_job_counts_without_fur)

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

        ::

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
def squeeze_increment(data, eg, senior_num, junior_num, increment):
    '''Move members of a selected eg (employee group) within
    a list according to an increment input (positive or negative)
    while retaining relative ordering within all eg groups.

    inputs
        data
            dataframe with empkey as index which at
            minimum includes an order column and an eg column
        eg
            employee group number
        senior_num and junior_num
            indexes for the beginning and end of the list zone to be
            reordered
        increment
            the amount to add or subrtract from the appropriate eg order
            number increment can be positive (move down list) or
            negative (move up list - toward zero)

    Selected eg order numbers within the selected zone
    (as a numpy array) are incremented - then
    the entire group order numbers are reset within
    the zone using scipy.stats.rankdata.
    The array is then assigned to a dataframe with empkeys as index.
    '''
    L = senior_num
    H = junior_num

    if H <= L:
        return

    if L < 0:
        L = 0

    idx_arr = np.array(data.new_order).astype(int)
    eg_arr = np.array(data.eg).astype(int)

    np.putmask(idx_arr[L:H], eg_arr[L:H] == eg, idx_arr[L:H] + increment)
    idx_arr[L:H] = st.rankdata(idx_arr[L:H], method='ordinal') - 1 + L

    return idx_arr


# SQUEEZE_LOGRITHMIC
def squeeze_logrithmic(data, eg, senior_num, junior_num,
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
        data
            a dataframe indexed by empkey with at least 2 columns:
            employee group (eg) and order (order)
        eg
            the employee group to move
        senior_num and junior_num
            integers marking the boundries (rng)
            for the operation
            (H must be greater than L)
        log_factor
            determines the degree of 'logrithmic packing'
        put_segment
            allows compression of the squeeze result (values under 1)
        direction
            squeeze direction:
            "u" - move up the list (more senior)
            "d" - move down the list (more junior)
    '''
    H = junior_num
    L = senior_num

    if put_segment <= 0:
        return

    if H <= L:
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

    order_segment = order_arr[L:H]
    eg_segment = eg_arr[L:H]

    eg_count = np.sum(eg_segment == eg)
    if eg_count == 0:
        return

    log_result = np.logspace(0, log_factor, eg_count, endpoint=False)
    log_result = log_result - log_result[0]
    pcnt_result = (log_result / log_result[-1])
    additive_arr = (pcnt_result * rng) * put_segment
    additive_arr = np.int16(additive_arr)

    if direction == 'd':
        put_nums = (H - additive_arr[::-1])
        put_nums = get_indexes_down(put_nums)
        additive_arr = H - get_indexes_up(additive_arr)[::-1] - L
    else:
        put_nums = (additive_arr + L)
        put_nums = get_indexes_up(put_nums)
        additive_arr = get_indexes_up(additive_arr)

    np.put(order_segment, np.where(eg_segment == eg)[0], put_nums)

    rng_dummy = np.delete(rng_dummy, additive_arr)

    np.put(order_segment, np.where(eg_segment != eg)[0], rng_dummy)

    return order_arr.astype(int)


# GET_INDEXES_UP
@jit(nopython=True, cache=True)
def get_indexes_up(list_of_positions):
    '''"FIT" a sample array to a list of unique index positions
    by incrementing any duplicates by one
    example:
    input > [0,0,1,2,5,9]
    output > [0,1,2,3,5,9]

    input
        list_of_positions
            list of index numbers
    '''
    for i in np.arange(1, list_of_positions.size):
        if list_of_positions[i] <= list_of_positions[i - 1]:
            list_of_positions[i] = list_of_positions[i - 1] + 1
    return list_of_positions


# GET_INDEXES_DOWN
@jit(nopython=True, cache=True)
def get_indexes_down(list_of_positions):
    '''"FIT" a sample array to a list of unique index positions
    by reducing any duplicates by one
    example:
    input > [0,1,2,8,9,9]
    output > [0,1,2,7,8,9]

    input
        list_of_positions
            list of index numbers
    '''
    for i in np.arange(list_of_positions.size - 2, -1, -1):
        if list_of_positions[i] >= list_of_positions[i + 1]:
            list_of_positions[i] = list_of_positions[i + 1] - 1
    return list_of_positions


# MAKE_DECILE_BANDS
def make_decile_bands(num_bands=40, num_returned_bands=10):
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


# PRECALCULATE FURLOUGHS
def precalculate_fur_without_recalls(monthly_job_totals,
                                     head_counts,
                                     fur_data, lows, highs):
    '''add monthly fur data to existing fur data if total job count
    is less than headcount for future months

    initial future furloughs may be precalculated
    based on monthly job changes and non_ret employee counts.

    This data is used to populate the furlough data and will
    be modified during the job assignment function if the recall
    option is incorporated.

    inputs
        monthly_job_totals
            job_gain_loss_table function output[1]
            short_form, job counts for each job level
            one row of counts for each month
        head_counts
            count_per_month function output
            short_form, one total for each month
        fur data
            array of initial furlough data from long_form df
        lows
            array of starting indexes for each month within long_form
            make_lower_slice_limits(head_counts)
        highs
            array of ending indexes for each month within long_form
            (cumsum of head_counts)
    '''
    for i in np.arange(head_counts.size):
        L = lows[i]
        U = highs[i]
        surplus = monthly_job_totals[i] - head_counts[i]
        if surplus < 0:
            np.put(fur_data[L:U],
                   np.where(fur_data[L:U] == 0)[0]
                   [monthly_job_totals[i] - head_counts[i]:],
                   1)
    return fur_data


# GET_RECALL_MONTHS (refactor to provide for no recall list)
def get_recall_months(list_of_recall_schedules):
    '''extract a sorted list of only the unique months containing a recall
    as defined within the settings dictionary recall schedules

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
def set_snapshot_weights(ratio_dict, orig_rng, eg_range):
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
    job_nums = list(ratio_dict.keys())
    for job in job_nums:
        wgt_list = []
        for ratio_group in ratio_dict[job][0]:
            wgt_list.append(np.sum((orig_rng == job) &
                                   (np.in1d(eg_range, ratio_group))))
        ratio_dict[job][1] = tuple(wgt_list)

    return ratio_dict


# ASSIGN JOBS BY RATIO CONDITION
def assign_cond_ratio(job, this_job_count,
                      ratio_dict, orig_range, assign_range,
                      eg_range, fur_range):
    ''' Apply a job ratio condition

    Main job assignment function calls this function at the appropriate month
    and with appropriate job data

    As written, this function applies a ratio for job assignment between
    one group and one or more other groups. The function code may be modified
    to permit other employee group ratio combinations.

    inputs
        job
            job level number
        this_job_count
            number of jobs available
        ratio_dict
            ratio condition dictionary, output of set_ratio_cond_dict function
        orig_range
            original job range
            Month slice of the orig_job column array (normally pertaining a
            specific month).
        assign_range
            job assignment range
            Month slice of the assign_range column array
        eg_range
            employee group range
            Month slice of the eg_range column array
        fur_range
            furlough range
            Month slice of the fur_range column array
    '''
    ratio_groups = ratio_dict[job][0]
    weights = ratio_dict[job][1]
    cond_assign_counts = distribute(this_job_count, weights)

    mask_index = []

    # find the indexes of each ratio group
    for grp in ratio_groups:
        mask_index.append(np.in1d(eg_range, grp))

    i = 0
    for i in np.arange(len(ratio_groups)):
        # assign jobs to employees within each ratio group who already hold
        # that job (no bump no flush)
        np.put(assign_range,
               np.where((assign_range == 0) &
                        (fur_range == 0) &
                        (orig_range <= job) &
                        (mask_index[i]))[0][:cond_assign_counts[i]],
               job)
        # count how many jobs were assigned to this ratio group by no bump
        # no flush
        used = np.sum((assign_range == job) &
                      (np.in1d(eg_range, ratio_groups[i])))
        # determine how many remain for assignment within the ratio group
        remaining = cond_assign_counts[i] - used
        # assign the remaining jobs by seniority within the ratio group
        np.put(assign_range,
               np.where((assign_range == 0) &
                        (fur_range == 0) &
                        (mask_index[i]))[0][:remaining],
               job)
        i += 1


# ASSIGN JOBS BY RATIO for FIRST n JOBS
def assign_cond_ratio_capped(job, this_job_count,
                             ratio_groups, weights, cap,
                             orig_range, assign_range,
                             eg_range, fur_range):
    '''distribute job assignments to employee groups by ratio for the first
    n jobs specified. Any jobs remaining are not distributed with
    this function.

    inputs
        job
            job level number
        this_job_count
            count of jobs at the current level available to be assigned for
            the current month
        ratio_groups (array-like)
            employee group(s) to be assigned to each ratio group.  This data
            originates with the "count_ratio_dict" spreadsheet within the
            settings.xlsx input file, as designated within a "group" column.
            For example a "1" in the "group1" column and a "2,3" in the
            group2 column would produce the following tuple:

                ::

                    ([1], [2, 3])

            Conditional job assignments would be ratioed between employees
            from group 1 and employees in group 2 or group 3 combined.
        weights (array-like)
            The weightings to use for proportional job counts for the ratio
            groups.  The elements may be any positive numbers.

            Example:

                ::

                    (2.48, 1.0)

        cap (integer)
            The maximum number of jobs to which the conditional assignments
            apply.  After assigning the first jobs (up to the cap count)
            according to the ratio group weightings, jobs available above the
            cap amount will be assigned without reference to this condition.
            If there are fewer jobs available than the cap amount, all jobs
            will be assigned to the ratio groups in accordance with the
            weighting (ratio) input.
        orig_range
            current month slice of original job array
        assign_range
            current month slice of job assignment array
        eg_range
            current month slice of employee group codes array
        fur_range
            current month slice of furlough data
    '''
    mask_index = []

    # find the indexes of each ratio group
    for grp in ratio_groups:
        mask_index.append(np.in1d(eg_range, grp))

    cond_assign_counts = distribute(this_job_count, weights, cap)

    i = 0
    for i in np.arange(len(ratio_groups)):
        np.put(assign_range,
               np.where((assign_range == 0) &
                        (fur_range == 0) &
                        (mask_index[i]))[0][:cond_assign_counts[i]],
               job)
        i += 1


# RECALL
def mark_for_recall(orig_range, num_of_job_levels,
                    fur_range, month, recall_sched,
                    jobs_avail, standalone=True,
                    eg_index=0,
                    method='sen_order', stride=2):
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
    active_count = np.sum(fur_range == 0)
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
def mark_for_furlough(orig_range, fur_range, month,
                      jobs_avail, num_of_job_levels):
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
    active_count = np.sum(fur_range == 0)

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


# MARK_FUR_RANGE
def mark_fur_range(assign_range, fur_range, job_levels):
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
    np.put(fur_range, np.where(assign_range == 0)[0], 1)
    np.put(fur_range, np.where((assign_range > 0) &
                               (assign_range <= job_levels))[0], 0)


# ALIGN FILL DOWN (all future months)
def align_fill_down(l, u, long_indexed_df, long_array):
    '''data align current values to all future months
    (short array segment aligned to long array)

    This function is used to set the values from the last standalone month as
    the initial data for integrated dataset computation when a delayed
    implementation exists.

    uses pandas df auto align - relatively slow

    TODO (for developer) - consider an all numpy solution

    inputs
        l, u
            current month slice indexes (from long df)
        long_indexed_df
            empty long dataframe with empkey indexes
        long_array
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
    result_array = np.array(chopped_df.x)
    result_size = result_array.size
    np.copyto(long_array[-result_size:], result_array)

    return long_array


# ALIGN NEXT (month)
def align_next(long_index_arr, short_index_arr, arr):
    '''"carry forward" data from one month to the next.

    Use the numpy in1d function to compare indexes (empkeys) from one month
    to the next month and return a boolean mask.  Apply the mask to current
    month data column (slice) and assign results to next month slice.

    Effectively finds the remaining employees (not retired) in the next month
    and copies the target column data for them from current month into the
    next month.

    inputs
        long_index_arr
            current month index of unique employee keys
        short_index_arr
            next month index of unique employee keys
            (a subset of long_index_arr)
        arr
            the data column (attribute) to carry forward
    '''

    arr = arr[np.in1d(long_index_arr, short_index_arr, assume_unique=True)]

    return arr


# DISTRIBUTE (simple)
def distribute(available, weights, cap=None):
    '''proportionally distribute 'available' according to 'weights'

    usage example:

        ::

            distribute(334, [2.48, 1])

    returns distribution as a list, rounded as integers:

        ::

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
        p = weight / total_weights
        this_bin_count = int(round(p * available))
        bin_counts.append(this_bin_count)
        total_weights -= weight
        available -= this_bin_count

    return bin_counts


# DISTRIBUTE VACANCIES BY WEIGHTS (CONTRACTUAL RATIOS)
def distribute_vacancies_by_weights(available, eg_counts, weights, cap=None):
    '''Determine how vacancies are assigned to employee groups
    with a given distribution ratio, total count of jobs, and a
    pre-existing and likely uneven initial job distribution.

    inputs
        available (integer)
            total count of jobs in distribution pool
            includes count of jobs already held by affected employee groups
    eg_counts (list of ints)
            count of jobs already assigned to each affected employee group
    weights (list (ints or floats))
            relative weighting between the employee groups
            examples:

                ::
                    [2.5, 3, 1.1]

        The length of the eg_counts list and the weights list must be the
        same.
        If there are zero or less vacancies, the function will
        return an array of zeros with a length equal to the eg_counts

        ...no displacements if no vacancies

        If any group(s) is already over their quota, the remaining
        vacancies will be distributed to the remaining group(s) according
        to the weightings (up to the quota for each group)
    '''

    if cap:
        max_allocations = distribute(cap, weights)
        add_limits = np.array(max_allocations) - np.array(eg_counts)
        add_limits[add_limits < 0] = 0
        available = min(sum(add_limits), available)
        variance = add_limits
    else:
        current_count = sum(eg_counts)
        balanced_distribution = distribute(current_count + available, weights)
        variance = balanced_distribution - np.array(eg_counts)
        variance[variance < 0.] = 0.
        variance = variance / balanced_distribution

    if min(variance) <= 0:
        i = 0
        list_loc = []
        for num in variance:
            if num > 0:
                list_loc.append(i)
            i += 1

        variance[list_loc] = distribute(available, variance[list_loc])

    return variance.astype(int)


# MAKE PARTIAL JOB COUNT LIST (prior to implementation month)
def make_delayed_job_counts(imp_month, delayed_jnums,
                            lower, upper):
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

    for month in np.arange(imp_month + 1):
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


# GEN_DELAYED_JOB_CHANGES_PER_MONTH
def delayed_monthly_sep_job_tables(job_levels,
                                   eg_job_counts,
                                   imp_job_counts,
                                   imp_month,
                                   allocation):
    '''make a job count table for each eg prior to a delayed
    implementation date. (eg = employee group).

    The difference between the initial total job counts and the job counts
    at the implementation date is proportionally spread out over the months
    between the starting date and the implementation date.
    A job dict determines the allocation of jobs amoung egs.

    inputs
        job_levels
            the number of job levels in the model
            (from the settings dictionary)
        eg_job_counts
            numpy array of the job count lists for the egs
        imp_job_counts
            the total of the jobs available within each job level on the
            implementation date (array)
        allocation
            array of job levels to eg weighting lists.  Key to determine
            the job allocation per level and month until implementation
            date.
            Total of each list must equal 1.

            example:

                ::

                    [[1.00, 0.00, 0.00],  # c4

                    [.50, 0.25, 0.25],   # c3

                    [.88, 0.09, 0.03],   # c2

                    [1.00, 0.00, 0.00],  # f4

                    [.50, 0.25, 0.25],   # f3

                    [.88, 0.09, 0.03],   # f2

                    [0.00, 1.00, 0.00],  # c1

                    [0.00, 1.00, 0.00]]  # f1

            using the above, if there were 4 additional jobs for job
            level 2 in a given month, eg 1 would get 2 and eg 2 and 3,
            1 each.

                ::

                    [.50, 0.25, 0.25]

    '''
    sum_of_initial_jobs = sum(eg_job_counts)
    job_change_totals = imp_job_counts - sum_of_initial_jobs

    monthly_job_changes = job_change_totals / imp_month

    # first number is imp month, second is num of job levels
    temp_tables = np.zeros(imp_month * job_levels).reshape(imp_month,
                                                           job_levels)

    temp_changes = np.zeros(imp_month * job_levels).reshape(imp_month,
                                                            job_levels)

    sep_tables = np.array((temp_tables, temp_tables, temp_tables))

    sep_changes = np.array((temp_changes, temp_changes, temp_changes))

    result_list = []
    # create initial sep job tables
    # create cumulative additives for each sep_table
    for i in np.arange(eg_job_counts.shape[0]):
        sep_tables[i][:] = eg_job_counts[i]
        sep_changes[i][:] = monthly_job_changes * allocation.T[i]
        sep_changes[i] = np.cumsum(sep_changes[i], axis=0)
        sep_tables[i] = sep_tables[i] + sep_changes[i]
        result_list.append(list(sep_tables[i]))
        print('start', int(np.sum(sep_tables[i][0])),
              'final', int(np.sum(sep_tables[i][-1:])))
    result_array = np.around(np.array(result_list), decimals=0).astype(int)
    result_array = np.clip(result_array, 0, 1000000)

    return result_array.astype(int)


# MAKE GAIN_LOSS_TABLE
def job_gain_loss_table(months, job_levels, init_job_counts,
                        job_changes, standalone=False):
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

        for i in np.arange(len(job_changes)):
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
def convert_to_enhanced(eg_job_counts, j_changes, job_dict):
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

                ::

                    [[197, 470, 1056, 412, 628, 1121, 0, 0],
                    [80, 85, 443, 163, 96, 464, 54, 66],
                    [0, 26, 319, 0, 37, 304, 0, 0]]

        j_changes
            input from the settings dictionary describing change of job
            quantity over months of time (list)

            example:

                ::

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

                ::

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


# ASSIGN JOBS STANDALONE WITH JOB CHANGES and prex option
def assign_standalone_job_changes(df_align,
                                  num_of_job_levels,
                                  lower,
                                  upper,
                                  total_months,
                                  job_counts_each_month,
                                  total_monthly_job_count,
                                  nonret_each_month,
                                  job_change_months,
                                  job_reduction_months,
                                  start_month,
                                  eg,
                                  sg_rights,
                                  recalls,
                                  apply_sg_cond=True,
                                  fur_return=False):
    '''Long_Form

    Uses the job_gain_or_loss_table job count array for job assignments.
    Jobs counts may change up or down in any category for any time period.
    Handles furlough and return of employees.
    Handles prior rights/conditions and restrictions.
    Handles recall of initially furloughed employees.

    Inputs are precalculated outside of function to the extent possible.

    Returns tuple (long_assign_column, long_count_column, held_jobs,
    fur_data, orig_jobs)

    inputs
        df_align (dataframe)
            dataframe with ['sg', 'fur'] columns
        num_of_job_levels (integer)
            number of job levels in the data model (excluding a furlough
            level)
        lower
            ndarry from make_lower_slice_limits function
            (calculation derived from cumsum of count_per_month function)
        upper
            cumsum of count_per_month function
        total_months
            sum of count_per_month function output
        job_counts_each_month
            output of job_gain_loss_table function[0]
            (precalculated monthly count of jobs in each job category,
            size (months,jobs))
        total_monthly_job_count
            output of job_gain_loss_table function[1]
            (precalculated monthly total count of all job categories,
            size (months))
        nonret_each_month
            output of count_per_month function
        job_change_months
            the min start month and max ending month found within the
            array of job_counts_each_month inputs
            (find the range of months to apply consideration for
            any job changes - prevents unnecessary looping)
        job_reduction_months
            months in which the number of jobs is decreased (list).
            from the get_job_reduction_months function
        start_month (integer)
            starting month for calculations, likely implementation month
            from settings dictionary
        eg (integer)
            input from an incremental loop which is used to select the proper
            employee group recall scedule
        sg_rights (list)
            list of 5-element lists for a pre-existing job assignment
            condition calculation (special group)

            Formant: [employee group number, job number, count, start_month,
                end_month]
        recalls (list)
            lists of integers and a nested list for recall calculations.

            Format: [total monthly_recall_count,
                [employee group recall allocation],
                start_month, end_month]
        apply_sg_cond (boolean)
            compute with pre-existing special job quotas for certain
            employees marked with a one in the sg column (special group)
            according to a schedule defined in the settings dictionary
        fur_return (boolean)
            compute with a recall schedule(s) defined in the settings
            dictionary

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
    sg_ident = np.array(df_align.sg)
    fur_data = np.array(df_align.fur)
    index_data = np.array(df_align.index)

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

    # for month in np.arange(num_of_months):
    for month in np.arange(start_month, num_of_months):

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

        this_job_col = 0
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

            this_job_count = job_counts_each_month[month, this_job_col]

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

            jobs_avail = this_job_count - np.sum(assign_range == job)
            np.put(assign_range,
                   np.where((assign_range == 0) &
                            (held_job_range <= job) &
                            (fur_range == 0))[0][:jobs_avail],
                   job)

            jobs_avail = this_job_count - np.sum(assign_range == job)
            np.put(assign_range,
                   np.where((assign_range == 0) &
                            (fur_range == 0))[0][:jobs_avail],
                   job)

            np.put(job_count_range,
                   np.where(assign_range == job)[0],
                   this_job_count)

            this_job_col += 1
            job += 1

        # AFTER MONTHLY JOB LOOPS DONE, PRIOR TO NEXT MONTH:

        # pass down assign_range
        # held_jobs = align_fill_down(L, U, long_df, assign_range, held_jobs)
        held_next = align_next(index_range, index_range_next, assign_range)
        np.copyto(held_jobs[L_next:U_next], held_next)

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
               np.sum(fur_range == 1))

        # fur_data = align_fill_down(L, U, long_df, fur_range, fur_data)
        fur_next = align_next(index_range, index_range_next, fur_range)
        np.copyto(fur_data[L_next:U_next], fur_next)

    long_assign_column[long_assign_column == 0] = num_of_job_levels + 1
    held_jobs[held_jobs == num_of_job_levels + 1] = 0
    orig_jobs = held_jobs[lower[0]:upper[0]]

    return long_assign_column.astype(int), long_count_column.astype(int), \
        held_jobs.astype(int), fur_data.astype(int), orig_jobs.astype(int)


def print_config_selections():
    '''grab settings dictionary data settings and put it in a dataframe and then
    print it for a quick summary of scalar settings dictionary inputs
    '''
    sdict = pd.read_pickle('dill/dict_settings.pkl')
    try:
        case_study = pd.read_pickle('dill/case_dill.pkl')
    except:
        case_study = 'error, no case_dill.pkl file found'

    config_dict = {'case_study': case_study,
                   'start': sdict['start'],
                   'starting_date': sdict['starting_date'],
                   'enhanced_jobs': sdict['enhanced_jobs'],
                   'enhanced_jobs_full_suffix':
                   sdict['enhanced_jobs_full_suffix'],
                   'enhanced_jobs_part_suffix':
                   sdict['enhanced_jobs_part_suffix'],
                   'delayed_implementation': sdict['delayed_implementation'],
                   'implementation_date': sdict['implementation_date'],
                   'imp_date': sdict['imp_date'],
                   'imp_month': sdict['imp_month'],
                   'recall': sdict['recall'],
                   'annual_pcnt_raise': sdict['annual_pcnt_raise'],
                   'compute_job_category_order':
                   sdict['compute_job_category_order'],
                   'compute_pay_measures': sdict['compute_pay_measures'],
                   'compute_with_job_changes':
                   sdict['compute_with_job_changes'],
                   'count_final_month': sdict['count_final_month'],
                   'date_exception_end': sdict['date_exception_end'],
                   'date_exception_start': sdict['date_exception_start'],
                   'discount_longev_for_fur': sdict['discount_longev_for_fur'],
                   'end_date': sdict['end_date'],
                   'future_raise': sdict['future_raise'],
                   'init_ret_age': sdict['init_ret_age'],
                   'init_ret_age_months': sdict['init_ret_age_months'],
                   'init_ret_age_years': sdict['init_ret_age_years'],
                   'ret_age': sdict['ret_age'],
                   'ret_age_increase': sdict['ret_age_increase'],
                   'int_job_counts': sdict['int_job_counts'],
                   'job_levels_basic': sdict['job_levels_basic'],
                   'job_levels_enhanced': sdict['job_levels_enhanced'],
                   'last_contract_year': sdict['last_contract_year'],
                   'lspcnt_calc_on_remaining_population':
                   sdict['lspcnt_calc_on_remaining_population'],
                   'no_bump': sdict['no_bump'],
                   'num_of_job_levels': sdict['num_of_job_levels'],
                   'pay_table_exception_year':
                   sdict['pay_table_exception_year'],
                   'pay_table_longevity_sort':
                   sdict['pay_table_longevity_sort'],
                   'pay_table_year_sort': sdict['pay_table_year_sort'],
                   'save_to_pickle': sdict['save_to_pickle'],
                   'count_dist': sdict['count_dist'],
                   'ratio_dist': sdict['ratio_dist'],
                   'sg_dist': sdict['sg_dist'],
                   'stripplot_full_time_pcnt':
                   sdict['stripplot_full_time_pcnt'],
                   'top_of_scale': sdict['top_of_scale'],
                   'add_doh_col': sdict['add_doh_col'],
                   'add_eg_col': sdict['add_eg_col'],
                   'add_ldate_col': sdict['add_ldate_col'],
                   'add_line_col': sdict['add_line_col'],
                   'add_lname_col': sdict['add_lname_col'],
                   'add_ret_mark': sdict['add_ret_mark'],
                   'add_retdate_col': sdict['add_retdate_col'],
                   'add_sg_col': sdict['add_sg_col']}

    settings = pd.DataFrame(config_dict, index=['setting']).stack()
    df = pd.DataFrame(settings, columns=['setting'])
    df.index = df.index.droplevel(0)
    df.index.name = 'option'

    return df


def max_of_nested_lists(nested_list, return_min=False):
    '''find the maximum value within a list of lists (or tuples or arrays)

    return_min input will find minimum of nested containers
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


def eval_strings(args):
    arg_list = []
    for arg in args:
        arg_list.append(eval(arg))

    return arg_list


def clip_ret_ages(ret_age_dict, init_ret_age, dates_long_arr, ages_long_arr):
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
            ds_dict[ws] = pd.read_pickle('dill/' + ws_ref + '.pkl'), ws
        except:
            # if dataset doesn't exist, pass and notify user
            print('dataset for proposal "' + ws + '" not found in dill folder')
            if ws == 'edit':
                print('"edit" proposal is produced with the editor tool.\n')
            if ws == 'hybrid':
                print('"hybrid" proposal is generated with the "build_list"' +
                      ' function within the list_builder.py module\n')

    print('datasets loaded (dictionary keys):', list(ds_dict.keys()), '\n')

    return ds_dict


def assign_preimp_standalone(ds_stand, ds_integrated, col_list,
                             imp_high, return_array_and_dict=False):
    '''Copy standalone data to an integrated dataset up to the implementation
    date.

    inputs
        ds_stand (dataframe)
            standalone dataset
        ds_integrated (dataframe)
            integrated dataset
        col_list
            common columns in standalone and integrated datasets.  These
            are calculated columns and have different results in each dataset.
        imp_high
            highest index (row number) from implementation month data
            (from long-form dataset)
        return_array_and_dict
            if True, return the standalone data array and column-name to numpy
            array index dictionary
    '''

    # only include columns from col_list which exist in both datasets
    col_list = list(set(col_list).intersection(ds_stand.columns))
    col_list = list(set(col_list).intersection(ds_integrated.columns))
    key_cols = ['mnum', 'empkey']

    # grab appropriate columns from standalone dataset up to end of
    # implementation month initiate a 'key' column to save assignment
    # time below
    ds_stand = ds_stand[col_list][:imp_high].copy()

    # grab the 'mnum' and 'empkey' columns from the ordered dataset to
    # form a 'key' column with unique values
    ds_temp = ds_integrated[key_cols][:imp_high].copy()

    # make numpy arrays out of column values for fast 'key' column generation
    stand_emp = np.array(ds_stand.empkey) * 1000
    stand_mnum = np.array(ds_stand.mnum)
    temp_emp = np.array(ds_temp.empkey) * 1000
    temp_mnum = np.array(ds_temp.mnum)
    # make the 'key' columns
    stand_key = stand_emp + stand_mnum
    temp_key = temp_emp + temp_mnum
    # assign to 'key' columns
    ds_stand['key'] = stand_key
    ds_temp['key'] = temp_key
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
    ds_temp = ds_temp[temp_cols]
    # convert the ds_temp dataframe to a 2d numpy array for fast
    # indexing and retrieval
    stand_array = ds_temp.values.T

    # construct a dictionary of columns to numpy row indexes
    keys = ds_temp.columns
    values = np.arange(len(ds_temp.columns))
    stand_dict = od(zip(keys, values))

    if return_array_and_dict:
        return stand_array, stand_dict
    else:
        for col in keys:
            col_arr = np.array(ds_integrated[col])
            col_arr[:imp_high] = stand_array[stand_dict[col]]
            ds_integrated[col] = col_arr

    return ds_integrated


def make_preimp_array(ds_stand, ds_integrated, imp_high,
                      compute_cat, compute_pay):
    '''Create an ordered numpy array of pre-implementation data gathered from
    the pre-calculated standalone dataset and a dictionary to keep track of the
    information.  This data will be joined by post_implementation integrated
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
    stand_emp = np.array(ds_stand.empkey) * 1000
    stand_mnum = np.array(ds_stand.mnum)
    temp_emp = np.array(ds_temp.empkey) * 1000
    temp_mnum = np.array(ds_temp.mnum)
    # make the 'key' columns
    stand_key = stand_emp + stand_mnum
    temp_key = temp_emp + temp_mnum
    # assign to 'key' columns
    ds_stand['key'] = stand_key
    ds_temp['key'] = temp_key
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


def make_cat_order(ds, table):
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

    mnum_arr = np.array(ds.mnum)
    jnum_arr = np.array(ds.jnum) - 1
    jpcnt_arr = np.array(ds.jobp % 1)

    cnt_arr = cat_counts[jnum_arr, mnum_arr]
    add_arr = cat_add[jnum_arr, mnum_arr]

    cat_arr = (jpcnt_arr * cnt_arr) + add_arr

    return cat_arr


def make_tuples_from_columns(df, columns, return_as_list=True,
                             return_dates_as_strings=False,
                             date_cols=[]):
    '''Combine row values from selected columns to form tuples.

    Returns a list of tuples which may be assigned to a new column.

    The length of the list is equal to the length of the input dataframe.

    inputs
        df (dataframe)
            input dataframe
        columns (list)
            columns from which to create tuples
    '''
    i = 0
    col_list = columns[:]
    for col in columns:
        if col in date_cols and return_dates_as_strings:
            col_list[i] = list(df[col].dt.strftime('%Y-%m-%d'))
        else:
            col_list[i] = list(df[col])
        i += 1
    zipped = zip(*col_list)
    if return_as_list:
        return list(zipped)
    else:
        return tuple(zipped)


def make_dict_from_columns(df, key_col, value_col):
    '''
    '''
    keys = df[key_col]
    values = df[value_col]

    return dict(zip(keys, values))


def make_lists_from_columns(df, columns,
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

        ::

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
        as_tuples (boolean)
            if True, output will be a list of tuples instead of a list of lists
    '''
    df_cols = df[columns]

    arrays = list(df_cols.values)

    if try_integers:
        try:
            arrays = list(df_cols.values.astype(int))
        except:
            pass

    column_list = []
    for e in arrays:
        e = list(e)
        column_list.append(e)

    if remove_zero_values:
        for i in np.arange(len(column_list)):
            column_list[i] = [grp for grp in column_list[i]
                              if grp not in [[0], 0]]

    if as_tuples:
        column_list = [tuple(x) for x in column_list]
    return column_list


def make_group_lists(df, column_name):
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

            ::

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
        except:
            if type(item) == list:
                this_list = item
            else:
                this_list.append(int(item))
        col_list.append(this_list)
    return col_list


def make_eg_pcnt_column(df):
    '''make an array derived from the input df reflecting the
    starting (month zero) percentage of each employee within his/her
    original employee group.  The array values have been data-aligned with
    the df input index.

    returns an array of values, same length as input df

    assign to long-form dataframe:

        ::

            df['eg_start_pcnt'] = make_eg_pcnt_column(df)

    input
        df (dataframe)
            pandas dataframe containing an employee group code column ('eg')
            and a month number column ('mnum').  The dataframe must be
            indexed with employee number code integers ('empkey')
    '''
    # grab the first month of the input dataframe, only 'eg' column
    m0df = df[df.mnum == 0][['eg']].copy()
    # make a running total for each employee group and assign to column
    m0df['eg_count'] = m0df.groupby('eg').cumcount() + 1
    # make another column with the total count for each respective group
    m0df['eg_total'] = m0df.groupby('eg')['eg'].transform('count')
    # calculate the group percentage and assign to column
    m0df['eg_pcnt'] = m0df.eg_count / m0df.eg_total
    # data align results to long_form input dataframe
    df['eg_start_pcnt'] = m0df.eg_pcnt

    return df.eg_start_pcnt.values


def make_starting_val_column(df, attr):
    '''make an array of values derived from the input dataframe which will
    reflect the starting value (month zero) of a selected attribute.  Each
    employee will be assigned the zero-month attribute value specific to
    that employee, duplicated in each month of the data model.

    This column allows future attribute analysis with a constant starting
    point for all employees.  For example, retirement job position may be
    compared to initial list percentage.

    assign to long-form dataframe:

        ::

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

    return all_mths_df.starting_value.values


def save_and_load_dill_folder(save_as=None,
                              load_case=None,
                              print_saved=False):
    '''Save the current "dill" folder to the "saved_dill_folders" folder.
    Load a saved dill folder as the "dill" folder if it exists.

    This function allows previously calculated pickle files (including the
    datasets) to be loaded into the dill folder for quick review.

    The "saved_dill_folders" folder is created if it does not already exist.
    The load_case input is a case study name.  If the load_case input is set to
    None, the function will only save the current dill folder and do nothing
    else.  If a load_case input is given, but is incorrect or no matching
    folder exists, the function will only save the current dill folder and do
    nothing else.

    The user may print a list of available saved dill folders (for loading)
    by setting the print_saved input to True.  No other action will take place
    with this option.

    If an award has conditions which differ from proposed conditions, the
    settings dictionary must be modified prior to calculating the award
    dataset.

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

                ::

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
    os.makedirs('saved_dill_folders/', exist_ok=True)

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
    except:
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
        except:
            print('\nError >>>  problem finding a saved dill folder with a ' +
                  load_case + ' prefix in ' +
                  'the "saved_dill_folders" folder.')
            print('\nThe dill folder contents remain unchanged.\n')
