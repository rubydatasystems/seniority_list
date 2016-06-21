# -*- coding: utf-8 -*-

'''month_form is length n months in model

short_form is length n employees

long_form is length cumulative sum non-retired each month

(could be millions of rows, depending on workgroup
size and age)
'''

import pandas as pd
import numpy as np
from numba import jit
import scipy.stats as st

import config as cf


# CAREER MONTHS
def career_months_list_in(ret_list, start_date=cf.starting_date):
    '''Determine how many months each employee will work
    including retirement partial month.

    This version takes a list of retirement dates
    Short_Form
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
def career_months_df_in(df, startdate=cf.starting_date):
    '''Determine how many months each employee will work
    including retirement partial month.
    This version has a df as input
    - df must have 'retdate' column of retirement dates
    Short_Form
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
def longevity_at_startdate(ldates_list, return_months=False,
                           start_date=cf.starting_date):
    ''' Short_Form
    - determine how much longevity (years) each employee has accrued
    as of the start date
    - input is list of longevity dates
    - float output is longevity in years
        (+1 added to reflect current 1-based pay year)
    - option for output in months
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


# LAST MONTH PARTIAL MONTH FOR PAY
def last_month_mpay_pcnt(ret_list):
    '''Short_Form
    Retirement decimal month for last month pay calculation
    Input is a list of dates
    (vectorize candidate??)
    Note: this function was replaced with a precalculated
    month percentage dataframe (pickled) which is read during
    the "make_skeleton" routine.  Much faster...
    '''
    last_month_pcnt = []
    for retdate in ret_list:
        last_month_pcnt.append(retdate.day /
                               (retdate + pd.offsets.MonthEnd(0)).day)
    return last_month_pcnt


# AGE AT START DATE
def starting_age(ret_list, start_date=cf.starting_date, retage=cf.ret_age):
    '''Short_Form
    Returns decimal age at given date.
    Input is list of retirement dates
    '''
    start_date = pd.to_datetime(start_date)
    s_year = start_date.year
    s_month = start_date.month
    s_day = start_date.day
    m_val = 1 / 12
    ages = []
    for retdate in ret_list:
        ages.append(m_val *
                    (((s_year - (retdate.year - retage)) * 12) -
                     (retdate.month - s_month) +
                     ((s_day - retdate.day) / s_day)))
    return ages


# COUNT PER MONTH
def count_per_month(career_months_array):
    '''Month_Form
    Returns number of employees remaining for each month (not retired).
    Cumulative sum of career_months_array input (np array)
    that are greater or equal to each incremental loop month number.

    Note: alternate method to this function is value count of mnums:
    df_actives_each_month = pd.DataFrame(df_idx.mnum.value_counts())
    df_actives_each_month.columns = ['count']
    '''
    max_career = np.max(career_months_array) + 1
    emp_count_array = np.zeros(max_career)

    for i in np.arange(0, max_career):
        emp_count_array[i] = np.sum(career_months_array >= i)

    return emp_count_array.astype(int)


# GENERATE MONTH SKELETON
@jit(nopython=True, cache=True)
def gen_month_skeleton(input_array):
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
    Input is the result of the count_per_month function as an np.array.
    '''
    total_months = int(np.sum(input_array))
    month_counts_array = np.zeros(total_months)
    i = 0
    j = 0
    for mcount in input_array:
        for slot in np.arange(j, int(mcount) + j):
            month_counts_array[slot] = i
        i += 1
        j = slot + 1
    return month_counts_array


# GENERATE THE EMPLOYEE INDEX SKELETON
@jit(nopython=True, cache=True)
def gen_emp_skeleton_index(monthly_active_count_nparay, career_mths_nparray):
    '''Long_Form
    For each employee who remains for each month,
    grab that employee index number.
    This index will be the key to merging in other data using data alignment.
    Input is the result of the count_per_month function (np.array)
    and the result of the career_months_df_in (or ...list_in)
    function
    '''
    total_months = int(np.sum(monthly_active_count_nparay))
    emp_idx_array = np.empty(total_months)
    total_emps = career_mths_nparray.size

    k = 0
    # look in career months list for each month
    for j in np.arange(0, int(np.max(career_mths_nparray)) + 1):
        cnt = 0
        for i in np.arange(0, total_emps):
            if career_mths_nparray[i] >= j:
                emp_idx_array[k] = cnt
                k += 1
            cnt += 1

    return emp_idx_array


# AGE FOR EACH MONTH (correction to starting age)
@jit
def age_correction(month_nums_array, ages_array, retage=cf.ret_age):
    '''Long_Form
    Returns a long_form (all months) array of employee ages by
    incrementing starting ages according to month number.

    Inputs

        mnum
            gen_month_skel function output (ndarray)

        ages_array
            starting_age function output aligned with long_form (ndarray)
            i.e. s_age is starting age (aligned to empkeys)
            repeated each month.

        retage option
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
def contract_pay_year_and_raise(date_list, future_raise=False, exception=False,
                                date_exception='2014-12-31', year_additive=.1,
                                annual_raise=.02, last_contract_year=2019.0):
    '''Month_Form
    Generate the contract pay year for indexing into the pay table.
    Pay year is clipped to last year of contract.
    If desired, an annual assumed raise beyond the contract time frame
    may be elected.

    Result is an array with two columns - column [0] contains the pay year,
    column [1] contains the multiplier for any raise (default is 1.0,
    which remains from the np.ones initial array.

    Usage example:
    year_scale = find_scale(series_years,
        future_raise = True, annual_raise = .02)

    NOTE: **(this function can accept a one-month pay exception
    for an outlier pay month...)**

    inputs

        date_list
            time series format list of dates

        future_raise
            option for pay calculations to apply a
            percentage increase for each year beyond
            the last contract year

        exception
            allows for an outlier pay month to be calculated

        date_exception
            date representing the month of the outlier pay month

        year_additive
            add this float amount to the regular year float
            so that it can be distinguished as an outlier pay month

        annual_raise
            yearly raise to calculate beyond the last contract year
            if future_raise option is selected

        last_contract_year
            last year of contract pay rate changes
    '''
    float_years = np.ones(len(date_list) * 2)
    float_years = float_years.reshape(len(date_list), 2)

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

        if exception:
            if date_list[i] == pd.to_datetime(date_exception):
                float_years[i][0] = date_list[i].year + year_additive

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

    Inputs

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


# MAKE AMER_STOVEPIPE_JOBS_WITH_SUP_C
# (Stovepipe with internal condition stovepiped, SHORT_FORM)
def make_amer_stovepipe_short_supc(job_list, sg_codes,
                                   sg_rights, fur_codes):
    '''Creates a 'stovepipe' job assignment within a single eg (american)
    which also includes a condition of certain job counts allocated
    to an eg subgroup, marked by a code array (sg_codes).

    Inputs

        job_list
            list of job counts for eg, like [23,34,0,54,...]
        sg_codes
            ndarray
            eg group members entitled to job condition
            (marked with 1, others marked 0)
            length of this eg population
        sg_rights
            list of lists from config file including job numbers and
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
        np.array(sg_rights)[:, 1],
        np.array(sg_rights)[:, 2]]

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


# MAKE AMER_STOVEPIPE_JOBS_WITH_SUP_C
# (Stovepipe with internal condition stovepiped, LONG_FORM)
def make_amer_standalone_long_supc(lower, upper,
                                   df_align,
                                   amer_job_counts,
                                   sg_job_nums,
                                   sg_dict, sg_months):
    '''Creates a 'stovepipe' job assignment within a single eg (american)
    which also includes a condition of certain job counts allocated
    to an eg subgroup, marked by a code array (sg_codes).

    Does not account for any job changes or furlough recall.

    Inputs

        lower
            ndarry from make_lower_slice_limits function
            (calculation derived from cumsum of count_per_month function)
        upper
            cumsum of count_per_month function
        df_align
            indexed dataframe to be used with align function.

            allows monthly result to be passed to next month
            with data alignment.
            Also includes sg and fur code data for processing.
        amer_job_counts
            either an array of lists (job_changes=True) or
            a single list of job counts (job_changes=False)
        sg_job_nums
            job levels included within amer supc condition
        sg_dict
            dictionary
            sg job to allotment dictionary
        sg_months
            list of month numbers when the condition is in effect

    The subset group will have proirity assignment for the first n jobs
    in the affected job category, the remainding jobs
    are assigned in seniority order.

    The subgroup jobs are assigned in subgroup stovepipe order.

    This function is applicable to a condition with known job counts.

    The result of this function is used with standalone calculations or
    combined with other eg lists to form an integrated original
    job assignment list.
    '''
    num_of_months = upper.size
    fur_level = cf.num_of_job_levels + 1

    fur_arr = np.array(df_align.fur, dtype=int)
    sg_arr = np.array(df_align.sg, dtype=int)
    long_assign_column = np.zeros(sg_arr.size, dtype=int)

    long_supc = np.zeros(sg_arr.size, dtype=int)
    index_data = np.array(df_align.index)

    lower_next = lower[1:]
    lower_next = np.append(lower_next, lower_next[-1])

    upper_next = upper[1:]
    upper_next = np.append(upper_next, upper_next[-1])

    for month in np.arange(num_of_months):

        L = lower[month]
        U = upper[month]

        L_next = lower_next[month]
        U_next = upper_next[month]

        assign_range = long_assign_column[L:U]
        long_range = long_supc[L:U]
        fur_range = fur_arr[L:U]
        sg_range = sg_arr[L:U]
        index_range = index_data[L:U]
        index_range_next = index_data[L_next:U_next]

        job = 0

        for count in amer_job_counts:

            job += 1

            if count > 0:

                if (job in sg_job_nums) and (month in sg_months):

                    sg_allotment = sg_dict[job]
                    # assign to unassigned, non-furloughed sg emps...
                    np.put(assign_range,
                           np.where((sg_range == 1) &
                                    (assign_range == 0) &
                                    (fur_range == 0))[0][:sg_allotment],
                           job)

                    remaining = count - np.sum(assign_range == job)
                    # assign to nbnf emps
                    np.put(assign_range,
                           np.where((assign_range == 0) &
                                    (fur_range == 0) &
                                    (long_range <= job))[0][:remaining],
                           job)

                    remaining = count - np.sum(assign_range == job)
                    # assign to remaining in seniority order
                    np.put(assign_range,
                           np.where((assign_range == 0) &
                                    (fur_range == 0))[0][:remaining],
                           job)

                else:
                    # nbnf...
                    np.put(assign_range,
                           np.where((assign_range == 0) &
                                    (fur_range == 0) &
                                    (long_range <= job))[0][:count],
                           job)

                    remaining = count - np.sum(assign_range == job)
                    # remaining...
                    np.put(assign_range,
                           np.where((assign_range == 0) &
                                    (fur_range == 0))[0][:remaining],
                           job)

        # long_supc = align(L, U, long_df, assign_range, long_supc)
        supc_next = align_next(index_range, index_range_next, assign_range)
        np.copyto(long_supc[L_next:U_next], supc_next)

    # assign fur job level to unassigned
    np.put(long_supc, np.where(long_supc == 0)[0], fur_level)

    return long_supc.astype(int), long_supc[:upper[0]]


# MAKE LIST OF ORIGINAL JOBS
def make_original_jobs_from_counts(jobs_arr_arr, eg_array,
                                   fur_array, num_levels):
    '''Short_Form

    This function grabs jobs from standalone job count
    arrays (normally stovepiped) for each employee group and inserts
    those jobs into a proposed integrated list, or a standalone list.

    Each eg (employee group) is assigned jobs from their standalone
    list in order top to bottom.

    Resut is a combined list of jobs with each eg maintaining ordered
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
    Uses the nonret counts for each month to:
      a. determine the long form monthly slice for assignment, and
      b. slice the jobs list from the top for assignment, skipping furloughees

    This function is used within the standalone computation

    inputs

        monthly_nonret_counts
            monthly list of job counts from the count_per_month function

        job_list
            output from the make_stovepipe_jobs_from_jobs_arr function

        fur_arr
            long_form furlough codes (same size as long_job_array)

        job_level_count
            num_of_job_levels (ultimately from config file)

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
    use the nonret counts for each month to:
      a. determine the long form slice for assignment, and
      b. slice the jobs list from the top for assignment

    Uses the job_counts_each_month (job_gain_loss_table function)[0] to
    build stovepiped job lists allowing for job count changes each month

    Unassigned employees (not enough jobs), are left at job number zero

    This is the full bump and full flush version
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


# ASSIGN JOBS NO BUMP NO FLUSH
def assign_jobs_nobump_noflush(orig, fur,
                               lower, upper, total_months, job_counts):
    '''Long_Form
    Assigns jobs so that original standalone jobs are assigned each month
    unless a better job is available through attrition of employees.
    Each month loop starts with the lowest job number.

    For each month and for each job level:

      1. assigns nbnf (orig) job if job array (ja) element
      is zero (unassigned) and orig job number is less than or
      equal to the job level in current loop, then

      2. assigns job level in current loop to unassigned slots from
      top to bottom in the job array (up to the count of that
      job level remaining after step one above)

    Each month range is determined by slicing using the lower and upper inputs.

    A comparison is made each month between the original job numbers and the
    current job loop number.

    Job assignments are placed into the monthly segment
    (assign_range) of the ja(job array).

    The ja eventually becomes the job number (jnum) column in the dataset.

    Original job numbers of 0 indicate no original job and are
    treated as furloughed employees - no jobs are assigned to furloughees.

    (add option to this function to allow furloughees to return?)
    (covered in assign_jobs_nbnf_job_changes function)
    '''
    ja = np.zeros(total_months, dtype=int)
    num_of_months = upper.size
    num_of_job_levels = job_counts.size

    for i in np.arange(num_of_months):

        L = lower[i]
        U = upper[i]

        orig_range = orig[L:U]
        assign_range = ja[L:U]
        fur_range = fur[L:U]

        j = 1

        while j <= num_of_job_levels:

            np.putmask(assign_range,
                       ((assign_range == 0) &
                        (orig_range <= j) &
                        (orig_range != 0) &
                        (fur_range == 0)),
                       j)

            used = (assign_range == j).sum()
            remaining_bid_count = job_counts[j - 1] - used

            np.put(assign_range,
                   np.where(
                       (assign_range == 0) &
                       (orig_range != 0) &
                       (fur_range == 0)
                   )[0]
                   [:remaining_bid_count],
                   j)

            j += 1

    return ja.astype(int)


# ASSIGN JOBS NBNF JOB CHANGES
def assign_jobs_nbnf_job_changes(df,
                                 lower,
                                 upper,
                                 total_months,
                                 job_counts_each_month,
                                 total_monthly_job_count,
                                 job_change_months,
                                 job_reduction_months,
                                 start_month,
                                 proposal_name_text,
                                 condition_list,
                                 fur_return=False):
    '''Long_Form
    Uses the job_gain_or_loss_table job count array for job assignments.
    Jobs counts may change up or down in any category for any time period.
    Handles furlough and return of employees.
    Handles prior rights/conditions and restrictions.
    Handles recall of initially furloughed employees.

    Inputs are precalculated outside of function to the extent possible.

    Inputs

        df
            long-form dataframe with ['eg', 'sg', 'fur', 'orig_job'] columns.
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
        job_change_months
            the min start month and max ending month found within the
            array of job_counts_each_month inputs
            (find the range of months to apply consideration for
                any job changes - prevents unnecessary looping)
        job_reduction_months
            months in which the number of jobs is decreased (list).
            from the get_job_reduction_months function
        proposal_name_text
            text of proposal file name without the extension.
            This should be the command line input 'input_proposal'
            if file name of proposal is df3.pkl, proposal_name_text = 'df3'

        furlough_return option -
            allows call to function XXX TODO...

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

    The long_assign_column eventually becomes the job number
    (jnum) column in the dataset.

    Original job numbers of 0 indicate no original job and are
    treated as furloughed employees - no jobs are assigned to
    furloughees unless furlough_return option is selected.
    '''
    num_of_job_levels = cf.num_of_job_levels
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

    if cf.delayed_implementation:
        long_assign_column[:upper[start_month]] = \
            orig[:upper[start_month]]

    if 'supc' in condition_list:

        sg_rights = np.array(cf.sg_rights)

        sg_jobs = np.transpose(sg_rights)[1]
        sup_c_counts = np.transpose(sg_rights)[2]
        sg_dict = dict(zip(sg_jobs, sup_c_counts))

        # calc sg sup c condition month range and concat
        sg_month_range = np.arange(np.min(sg_rights[:, 3]),
                                   np.max(sg_rights[:, 4]))
        job_change_months = np.concatenate((job_change_months,
                                            sg_month_range))

    # calc amer grp4 condition month range and concat to
    # job_change_months
    if 'ratio' in condition_list:
        amer_cond = np.array(cf.amr_g4_cond)
        amer_jobs = np.transpose(amer_cond)[1]
        ratio_cond_month = amer_cond[0][2]
        ratio_cond_job = 1
        amr_month_range = np.arange(np.min(amer_cond[:, 2]),
                                    np.max(amer_cond[:, 3]))
        job_change_months = np.concatenate((job_change_months,
                                            amr_month_range))

        # calc east grp4 condition month range and concat
    if 'count' in condition_list:
        east_cond = np.array(cf.east_g4_cond)
        east_jobs = np.transpose(east_cond)[1]
        east_cond_start_month = east_cond[0][3]
        east_month_range = np.arange(np.min(east_cond[:, 3]),
                                     np.max(east_cond[:, 4]))
        job_change_months = np.concatenate((job_change_months,
                                            east_month_range))
        west_nbnf_jobs = np.zeros(total_months, dtype=int)

    if fur_return:

        recall_months = get_recall_months(cf.recalls)
        job_change_months = np.concatenate((job_change_months,
                                            recall_months))

    job_change_months = np.unique(job_change_months)

    # for month in np.arange(num_of_months):
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

        # use numpy arrays for job assignment process for each month
        # use pandas for data alignment 'job position forwarding'
        # to future months

        this_job_col = 0
        job = 1

        if month in job_reduction_months:
            mark_for_furlough(orig_job_range, fur_range, month,
                              total_monthly_job_count, num_of_job_levels)

        if fur_return and (month in recall_months):
            mark_for_recall(orig_job_range, num_of_job_levels,
                            fur_range, month, cf.recalls,
                            total_monthly_job_count, standalone=False)

        while job <= num_of_job_levels:

            this_job_count = job_counts_each_month[month, this_job_col]

            if month in job_change_months:

                if 'supc' in condition_list:

                    if month in sg_month_range and job in sg_jobs:

                        # assign SupC condition jobs to sg employees
                        sg_jobs_avail = min(sg_dict[job], this_job_count)
                        np.put(assign_range,
                               np.where((assign_range == 0) &
                                        (sg_range == 1) &
                                        (fur_range == 0))[0][:sg_jobs_avail],
                               job)

                # **AMR GRP4 condition**
                if 'ratio' in condition_list:
                    # TODO refactor cond_dict below so it only runs once...
                    # instead of 8 or 16 times...
                    if month == ratio_cond_month and job == ratio_cond_job:

                        ratio_cond_dict = set_ratio_cond_dict(1,
                                                              amer_jobs,
                                                              orig_job_range,
                                                              eg_range)

                    if month in amr_month_range and job in amer_jobs:

                        assign_cond_ratio(job,
                                          this_job_count,
                                          1,
                                          ratio_cond_dict,
                                          orig_job_range,
                                          assign_range,
                                          eg_range,
                                          fur_range)

                # **EAST GRP4 condition**
                if 'count' in condition_list:

                    if month in east_month_range and job in east_jobs:

                        # this is for the first month of cond only.
                        # mark the west employees holding an affected job
                        # and pass down to long_form west nbnf array.
                        # this is future reference for assignment function
                        # below.
                        if month == east_cond_start_month and job == 1:

                            west_range = west_nbnf_jobs[L:U]

                            for j in east_jobs:
                                np.put(west_range,
                                       np.where((orig_job_range == j) &
                                                (eg_range == 3))[0],
                                       j)

                            west_next = align_next(index_range,
                                                   index_range_next,
                                                   west_range)
                            np.copyto(west_nbnf_jobs[L_next:U_next],
                                      west_next)

                        west_range = west_nbnf_jobs[L:U]
                        assign_cond_ratio_capped(job,
                                                 this_job_count,
                                                 np.array((1)),
                                                 np.array((2)),
                                                 orig_job_range,
                                                 assign_range,
                                                 eg_range,
                                                 fur_range,
                                                 west_range)

            # TODO, code speedup...
            # use when not in condition month and monotonic is true
            # (all nbnf distortions gone, no job count changes)
            # if (month > max(job_change_months))
            # and monotonic(assign_range):
            #     quick_stopepipe_assign()

            jobs_avail = this_job_count - np.sum(assign_range == job)
            np.put(assign_range,
                   np.where((assign_range == 0) &
                            (orig_job_range <= job) &
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
        orig_next = align_next(index_range, index_range_next, assign_range)
        np.copyto(orig[L_next:U_next], orig_next)

        # pass down fur_range
        #  TODO **
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

    long_assign_column[long_assign_column == 0] = num_of_job_levels + 1
    orig[orig == num_of_job_levels + 1] = 0

    return long_assign_column.astype(int), long_count_column.astype(int), \
        orig.astype(int), fur_data.astype(int)


# PUT-MAP function
def put_map(jobs_array, job_cnts, fur_count):
    '''best use when values array is limited set of integers
    Any_Form (Practical application is Long_Form).
    10x faster than lambda function.
    dictionary-like value-key lookup using np.put and np.where

    inputs

        jobs_array
            long_form jnums

        jobs_count_array
            array of job counts from config file
            example: with 3 egs, array of 3 lists of counts

    Example:

          function call:
            map_jobs = put_map(no_bump_jnums, job_level_counts)

          assigned to df:
            df['nbnf_job_count'] = map_jobs.astype(int)

    len(set(jobs_array)) must equal length of jobs_count_array.
    '''
    target_array = np.zeros(jobs_array.size)

    counts_arr = np.append(job_cnts, fur_count)

    counts_arr = np.take(counts_arr, np.where(counts_arr != 0))[0]

    i = 0

    for job in sorted(list(set(jobs_array))):
        np.put(target_array,
               np.where(jobs_array == job),
               counts_arr[i])

        i += 1

    return target_array


# MAKE LOWER SLICE LIMITS
def make_lower_slice_limits(month_counts_cumsum):
    '''for use when working with unique month data
    within larger array (slice).

    The top of slice is cumulative sum, bottom of each slice
    will be each value of this function output array.

    Output is used as input for nbnf functions.
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

    input

        mnum_arr
            nd.array of a long_form df mnum column
    '''
    mnum_data = np.unique(mnum_arr, return_counts=True)[1]
    lower = make_lower_slice_limits(mnum_data)
    upper = mnum_data.cumsum()
    return lower, upper


def snum_and_spcnt(jnum_arr, job_levels, low_limits, high_limits,
                   table_counts, all_mths):
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
    '''
    seq_nums = np.arange(1, monthly_population_counts[0] + 1)
    # TODO consider np.sum vs sum below (is input always np array?)
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


# SNUMS and SPCNT with JOB CHANGES
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

    Inputs

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
            pilots remaining (incl fur) or jobs available
    '''
    fur_level = job_level_count + 1
    seq_nums = np.arange(1, monthly_population_counts[0] + 1)

    # TODO consider np.sum if monthly_population_counts is always np array
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

    Inputs

        furlough_list
            a list of integers holding any furlough count for each eg
        *job_count_lists
            the eg job count list(s)

    Returns tuple of two ndarrays.
    '''
    eg_job_counts = []
    i = 0

    for job_list in job_count_lists:

        if not cf.actives_only:
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

    Inputs

        *job_count_lists
            the eg job count list(s)

    Returns tuple of two ndarrays.
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

    Inputs

        *job_count_lists
            the eg job count list(s)

    Returns tuple of two ndarrays.
    '''
    eg_job_counts = []

    for job_list in job_count_lists:
        j = list(job_list)
        eg_job_counts.append(j)

    eg_job_counts = np.array(eg_job_counts)
    combined_job_count = sum(np.array(eg_job_counts))

    return eg_job_counts.astype(int), combined_job_count.astype(int)


# SQUEEZE
def squeeze_increment(data, eg, senior_num, junior_num,
                      increment, pack_factor=1):
    '''Move members of a selected eg (employee group) within
    a list according to an increment input (positive or negative)
    while retaining relative ordering within all eg groups.

    pack_factor undeveloped at present...

    Inputs

        data
            dataframe with empkey as index which at
            minimum includes an order column and an eg column
        L and H
            indexes for the beginning and end of the list zone to be reordered
        increment
            the amount to add or subrtract from the appropriate eg order number
            increment can be positive (move down list) or
            negative (move up list - toward zero)

    Selected eg order numbers within the selected zone
    (as a numpy array) are incremented - then
    the entire group order numbers are reset within
    the zone using scipy.stats.rankdata.
    The array is then assigned to a dataframe with empkeys as index.

    This function could be modified to return the original input array
    resorted with the new order ready to be resqueezed!
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

    Inputs

        data
            a dataframe indexed by empkey with at least 2 columns:
            employee group (eg) and order (order)

        eg
            the employee group to move

        H and L
            integers marking the boundries (rng)
            for the operation
            (H must be greater than L)

        log_factor
            determines the degree of 'logrithmic packing'

        put_segment
            allows compression of the squeeze result (values under 1)

        direction input
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
    '''
    for i in np.arange(list_of_positions.size - 2, -1, -1):
        if list_of_positions[i] >= list_of_positions[i + 1]:
            list_of_positions[i] = list_of_positions[i + 1] - 1
    return list_of_positions


# MAKE_DECILE_BANDS
def make_decile_bands(num_bands):
    '''creates lower and upper arrays of percentile values.

    num_bands input must be multiple of 5 greater than or equal to 10
    and less than 10000

    Used for selecting sample employees surrounding deciles
    (0, 10, 20 etc. percent levels).

    Top and bottom bands will be half of normal size.

    Width of bands in percentage is determined by num_bands input.

    Input of 40 would mean bands 2.5% wide.
    Top and bottom bands would be 1.25% wide.

    Ex. 0-1.25%,

    8.75-11.25%,

    ... 98.75-100%
    '''
    if num_bands < 10:
        print('input must be multiple of 5 greater than or equal to 10')
        return
    if num_bands % 5 != 0:
        print('input must be multiple of 5 greater than or equal to 10')
        return
    cutter = (num_bands * 2) + 1
    cuts = np.round(np.linspace(0, 1, cutter) * 100, 2)
    width = 100 / num_bands
    strider = np.sum(cuts < 10)
    lower = list(cuts[strider - 1::strider])
    upper = list(cuts[1::strider])
    upper.append(100)
    lower = sorted(lower, reverse=True)
    lower.append(0)
    lower = sorted(lower)
    band_limits = np.array((lower, upper)) / 100
    return band_limits.T, width


# MONOTONIC TEST
def monotonic(x):
    '''test for stricly increasing array-like input

    Used to determine when need for no bump,
    no flush routine is no longer required.

    If test is true, and there are no job changes,
    special rights, or furlough recalls,
    then a straight stovepipe job assignment routine may
    be implemented (fast).
    '''
    dx = np.diff(x)
    return np.all(dx >= 0)


# GET_MONTH_SLICE
def get_month_slice(long_df, l, h):
    '''use low and high indexes to slice df
    month input is index of low and high index arrays
    '''
    out_df = long_df[l:h]
    return out_df


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
    recall_months = []
    for recall_sched in list_of_recall_schedules:
        recall_months.extend(list(range(recall_sched[2], recall_sched[3])))
        recall_months = sorted(list(set(recall_months)))
    return np.array(recall_months).astype(int)


# GET_JOB_CHANGE_MONTHS
def get_job_change_months(job_changes):
    month_list = []
    for change in job_changes:
        month_list = np.concatenate((month_list,
                                     np.arange(change[1][0],
                                               change[1][1])))
        month_list = np.unique(month_list).astype(int)
    return month_list


# GET_REDUCTION_MONTHS
def get_job_reduction_months(job_changes):
    month_list = []
    for change in job_changes:
        if change[2] < 0:
            month_list = np.concatenate((month_list,
                                         np.arange(change[1][0],
                                                   change[1][1])))
        month_list = np.unique(month_list).astype(int)
    return month_list


# ASSIGN JOBS BY RATIO CONDITION
def assign_cond_ratio(job, this_job_count, eg_num,
                      c_dict, orig_rng, assign_rng,
                      eg_rng, fur_rng):
    ''' Apply a job ratio condition
    # main job assignment function calls this function in appropriate month
    and with appropriate job data

    # as written, ratios one group with one or more groups, may be recoded
    to permit other combinations of ratios.
    '''
    eg1_job_count = int(round(c_dict[job] * this_job_count))

    not_eg1_job_count = int(this_job_count - eg1_job_count)
    np.put(assign_rng,
           np.where((assign_rng == 0) &
                    (eg_rng == eg_num) &
                    (fur_rng == 0))[0][:eg1_job_count],
           job)
    # assign not_eg1 nbnf jobs
    np.put(assign_rng,
           np.where((assign_rng == 0) &
                    (eg_rng != eg_num) &
                    (fur_rng == 0) &
                    (orig_rng <= job))[0][:not_eg1_job_count],
           job)

    used_jobs = np.where((assign_rng == job) & (eg_rng > 1))[0].size
    # then assign any remaining non_eg1 jobs by seniority
    np.put(assign_rng,
           np.where((assign_rng == 0) &
                    (eg_rng != eg_num) &
                    (fur_rng == 0))[0][:not_eg1_job_count - used_jobs],
           job)


# ASSIGN JOBS PER EAST Group 4 CONDITION
def assign_cond_ratio_capped(job, this_job_count, eg_arr1, eg_arr2,
                             orig_range, assign_range,
                             eg_range, fur_range, exclude_eg_range):
    '''distribute job assignments to employee groups by ratio for the first
    n jobs specified. Any jobs remaining are not distributed with
    this function.

    inputs

        job
            job number
        this_job_count
            count of job
        eg_arr1
            np.array containing the employee group codes within ratio group 1
        eg_arr2
            np.array containing the employee group codes within ratio group 2
        orig_range
            current month slice of original job array
        assign_range
            current month slice of job assignment array
        eg_range
            current month slice of employee group codes array
        fur_range
            current month slice of furlough data
        exclude_eg_range
            current month slice of excluded group array

    exclude_eg_range input is an array marking an employee group(s) which does
    not participate in the conditional job distribution.  (this group(s) is
    protected with no bump no flush)
    '''
    eg_1_count = 0
    eg_2_count = 0
    block_pcnt = cf.intl_blk_pcnt
    reserve_pcnt = 1 - block_pcnt

    # find the indexes of each ratio group
    eg_1_indexes = np.in1d(eg_range, eg_arr1)
    eg_2_indexes = np.in1d(eg_range, eg_arr2)
    # combine
    affected_indexes = np.in1d(eg_range, np.append(eg_arr1, eg_arr2))

    # find the first n indexes of unassigned employees
    assign_range_indexes = np.where(assign_range == 0)[0][:this_job_count]

    # first assign nbnf jobs to excluded employees
    exclude_nbnf = np.where(
        exclude_eg_range[assign_range_indexes] == job)[0]

    # assign jobs to exclude_eg_range
    np.put(assign_range, assign_range_indexes[exclude_nbnf], job)

    # count the number of jobs assigned to excluded employees
    exclude_count = np.where(assign_range == job)[0].size

    # initial nbnf assignment to ratio condition affected groups
    np.put(assign_range,
           np.where((assign_range == 0) &
                    (orig_range <= job) &
                    (fur_range == 0) &
                    (affected_indexes))[0][:this_job_count - exclude_count],
           job)
    # count the job assignments per eg
    eg_1_count = np.where((assign_range == job) & (eg_1_indexes))[0].size
    eg_2_count = np.where((assign_range == job) & (eg_2_indexes))[0].size

    # calculate 'available' (total job count to split between egs
    # affected by cond)
    # and assign proper weighting per proposal
    if cf.num_of_job_levels == 16:

        if job in [1, 2]:

            weights = [2.48, 1]
            limit = 637

            if job == 1:
                limit = limit * block_pcnt

            if job == 2:
                limit = limit * reserve_pcnt

        elif job in [7, 8]:

            weights = [2.46, 1]
            limit = 1162

            if job == 7:
                limit = limit * block_pcnt

            if job == 8:
                limit = limit * reserve_pcnt

        max_quota = min(this_job_count, round(limit))

    elif cf.num_of_job_levels == 8:

        if job == 1:
            weights = [2.48, 1]
            limit = 637

        elif job == 4:
            weights = [2.46, 1]
            limit = 1162

        max_quota = min(this_job_count, limit)

    available = max_quota - exclude_count
    # make list for function below
    eg_counts = [eg_1_count, eg_2_count]
    # run function to determine disposition of vacancies (nbnf assignment
    # already run above)

    assignment_list = distribute_vacancies_by_weights(
        available, eg_counts, weights).astype(int)

    # if jobs held are less than available count (total alloted by condition)
    if np.sum(assignment_list) > 0:

        eg_1_quota = assignment_list[0]
        eg_2_quota = assignment_list[1]
        # assign vacancies to proper groups to move toward or maintain proper
        # weightings
        if eg_1_quota > 0:
            np.put(assign_range,
                   np.where((assign_range == 0) &
                            (eg_1_indexes) &
                            (fur_range == 0))[0][:eg_1_quota],
                   job)

        if eg_2_quota > 0:
            np.put(assign_range,
                   np.where((assign_range == 0) &
                            (eg_2_indexes) &
                            (fur_range == 0))[0][:eg_2_quota],
                   job)

    else:

        # ADD CODE FOR CONDITION WHEN TOTAL COUNT OF JOBS IS LESS THAN
        # ALLOTMENT AND ONE EG IS BELOW QUOTA AND THE OTHER IS ABOVE...
        # if count of jobs held by both egs is higher than condition
        # check to see if one group has not yet met quota
        # if true, give vacant positions to underrepresented eg until quota met
        eg_quotas = distribute(limit, weights)

        open_jobs = this_job_count - exclude_count - eg_1_count - eg_2_count

        eg_1_shortfall = max(0, eg_quotas[0] - eg_1_count)
        eg_2_shortfall = max(0, eg_quotas[1] - eg_2_count)

        if eg_1_shortfall > 0:
            eg_1_to_add = min(eg_1_shortfall, open_jobs)
            np.put(assign_range,
                   np.where((assign_range == 0) &
                            (eg_1_indexes) &
                            (fur_range == 0))[0][:eg_1_to_add],
                   job)

        if eg_2_shortfall > 0:
            eg_2_to_add = min(eg_2_shortfall, open_jobs)
            np.put(assign_range,
                   np.where((assign_range == 0) &
                            (eg_2_indexes) &
                            (fur_range == 0))[0][:eg_2_to_add],
                   job)


# SET_RATIO_COND_DICT
def set_ratio_cond_dict(eg_num, job_list, orig_rng, eg_range):
    '''Determine the job distribution ratios to carry forward during
    the ratio condition application period

    likely called at implementation month by main job assignment function
    '''
    ratio_cond_dict = {}
    for job in job_list:

        total_this_job_count = np.sum(orig_rng == job)
        eg_count = np.sum((orig_rng == job) & (eg_range == eg_num))
        eg_ratio = round(eg_count / total_this_job_count, 2)
        ratio_cond_dict[job] = eg_ratio

    return ratio_cond_dict


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
            from config file, used to mark fur job level as
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
            number of job levels in model (from config file)
    '''
    np.put(fur_range, np.where(assign_range == 0)[0], 1)
    np.put(fur_range, np.where((assign_range > 0) &
                               (assign_range <= job_levels))[0], 0)


# ALIGN FILL DOWN (all future months)
def align_fill_down(l, u, long_indexed_df, short_array, long_array):
    '''data align current values to all future months
    (short array segment aligned to long array)

    use pandas df auto align - relatively slow

    inputs

        short_array
            current month values (job assignment, fur code, etc)
            numpy array
            variable length depending on number of
            remaining non_ret employees for this month

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
    chopped_df = long_indexed_df[l:].copy()
    short_df['x'] = short_array
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
def distribute(available, weights):
    '''proportionally distribute 'available' according to 'weights'
    usage: distribute(334, [2.48, 1])
    '''
    bin_counts = []
    total_weights = sum(weights)
    for weight in weights:
        p = weight / total_weights
        this_bin_count = round(p * available)
        bin_counts.append(this_bin_count)
        total_weights -= weight
        available -= this_bin_count
    return bin_counts


# DISTRIBUTE VACANCIES BY WEIGHTS (CONTRACTUAL RATIOS)
def distribute_vacancies_by_weights(available, eg_counts, weights):
    '''Determine how vacancies are assigned to employee groups
    with a given distribution ratio, total count of jobs, and a
    pre-existing and likely uneven initial job distribution.

    inputs

        available (integer)
            total count of jobs in distribution pool
            includes count of jobs already held by affected employee groups
            does not include jobs held by other non-affected employee groups

        eg_counts (list of ints)
            count of jobs already assigned to each affected employee group

        weights (list (ints or floats))
            relative weighting between the employee groups
            examples: [2.5, 3, 1.1]

        The length of the eg_counts list and the weights list must be the same.
        If there are zero or less vacancies, the function will
        return an array of zeros with a length equal to the eg_counts

        ...no displacements if no vacancies

        If any group(s) is already over their quota, the remaining vacancies
        will be distributed to the remaining group(s) according
        to the weightings
    '''
    bin_counts = []
    total_weights = sum(weights)
    vacancies = available - sum(eg_counts)
    if vacancies <= 0:
        additives = np.repeat(0, len(eg_counts)).astype(int)
        return additives

    for weight in weights:
        p = weight / total_weights
        this_bin_count = round(p * available)
        bin_counts.append(this_bin_count)
        total_weights -= weight
        available -= this_bin_count
    eg_counts = np.array(eg_counts)
    bin_counts = np.array(bin_counts)
    additives = bin_counts - eg_counts
    if min(additives) < 0:
        i = -1
        list_loc = []
        for num in additives:
            i += 1
            if num > 0:
                list_loc.append(i)
            else:
                additives[i] = 0
        weights = np.array(weights).astype(int)

        additives = np.array(additives).astype(int)
        additives[list_loc] = distribute(vacancies, weights[list_loc])

    return additives


# To 8 (job levels) from 16 (job_levels)
def convert_enhanced_to_basic(j):
    '''Convert blockholder and reserve job levels(16) to
    group job counts (CA and FO counts, total of 8 counts
    for the 4 groups)

        Input is a list of 16 integers.
        Returns a list of 8 integers.

    Inputs

        j
            A list of the 16 level job counts
            example: [97, 64, 102, 575, 68, 310, 196, 130, 115,
                603, 71, 77, 325, 38, 86, 46]
    '''
    k1 = j[0] + j[1]
    k2 = j[2] + j[4]
    k3 = j[3] + j[5]
    k4 = j[6] + j[7]
    k5 = j[8] + j[11]
    k6 = j[9] + j[12]
    k7 = j[10] + j[13]
    k8 = j[14] + j[15]

    k = [k1, k2, k3, k4, k5, k6, k7, k8]
    return k


# MAKE PAY TABLE
def make_pay_table(wb_address, rates_sheetname, hours_sheetname):
    '''Create an indexed dataframe with one column of monthly
    compensation.  Index is a combination of contract year and
    pay year longevity and job level (jnum).

    Note:  this is a manual pay_table creation function - there is a
    make_pay_tables_from_excel script that is normally used...

    Inputs

        wb_address (string)
            folder/file_name of excel wb containing pay charts and
            monthly hours
            Example:  'excel/pay_tables.xlsx'

        rates_sheetname (string)
            excel workbook sheetname containing pay charts
            Example: 'no_rsv_with_fur'
            layout example (partial table...more rows and columns...):

            +-------+-------+-------+-------+-------+-------+
            | year  | jnum  |   1   |   2   |   3   |   4   |
            +-------+-------+-------+-------+-------+-------+
            |2013.0 |   1   | 40.00 | 197.04| 198.64| 200.24|
            +-------+-------+-------+-------+-------+-------+
            |2013.0 |   2   | 40.00 | 197.04| 198.64| 200.24|
            +-------+-------+-------+-------+-------+-------+
            |2013.0 |   3   | 40.00 | 167.20| 168.56| 169.91|
            +-------+-------+-------+-------+-------+-------+
            |2013.0 |   4   | 40.00 | 155.10| 156.36| 157.62|
            +-------+-------+-------+-------+-------+-------+
            |2013.0 |   5   | 40.00 | 167.20| 168.56| 169.91|
            +-------+-------+-------+-------+-------+-------+
            |2013.0 |   6   | 40.00 | 155.10| 156.36| 157.62|
            +-------+-------+-------+-------+-------+-------+
            |2013.0 |   7   | 40.00 | 98.52 | 119.18| 122.15|
            +-------+-------+-------+-------+-------+-------+


        hours_sheetname (string)
            excel workbook sheetname containing monthly hours
            credited to each job level
            Example: 'block_only_hours'
            layout example (2 columns, partial example):

                    +-----+-----+
                    |jnumm|hours|
                    +-----+-----+
                    |  1  | 85  |
                    +-----+-----+
                    |  2  | 74  |
                    +-----+-----+
                    |  3  | 85  |
                    +-----+-----+
                    |  4  | 85  |
                    +-----+-----+

        Data is contained in 2 columns of worksheet

    Function output is 2 pickled files:

        1. multi-indexed pay table (scale, year, jnum) with
        corresponding monthly rates

        2. single-indexed pay table with the index combining
        the multi-index above for faster retrieval

    This function creates or over-writes files directly with no return value
    '''
    pay_table = pd.read_excel(wb_address,
                              sheetname=rates_sheetname)

    pay_hours = pd.read_excel(wb_address,
                              sheetname=hours_sheetname,
                              index_col='jnum')

    pay_melt = pd.melt(pay_table, id_vars=['year', 'jnum'],
                       var_name='scale',
                       value_name='rate')

    pay_melt.rate = np.round(pay_melt.rate, 2)

    df_hours_added = pd.merge(pay_melt, pay_hours,
                              right_index=True, left_on=['jnum'])

    df_hours_added.sort_index(inplace=True)

    df_hours_added['monthly'] = df_hours_added.rate * df_hours_added.hours

    df_hours_added.drop(['rate', 'hours'], inplace=True, axis=1)

    df_hours_added.set_index(['scale', 'year', 'jnum'], inplace=True)

    file_string1 = 'dill/pay_table_' + rates_sheetname + '.pkl'

    df_hours_added.to_pickle(file_string1)

    # cells below convert pay table multi-index to single column combined
    # index which allows for much faster data align merge...

    df_hours_added.reset_index(drop=False, inplace=True)

    df_hours_added['ptindex'] = (df_hours_added.year *
                                 100000 + df_hours_added.scale *
                                 100 + df_hours_added.jnum)

    df_hours_added.drop(['scale', 'year', 'jnum'], axis=1, inplace=True)

    df_hours_added.sort_values('ptindex', inplace=True)

    df_hours_added.set_index('ptindex', drop=True, inplace=True)

    file_string2 = 'dill/idx_pay_table_' + rates_sheetname + '.pkl'

    df_hours_added.to_pickle(file_string2)


# MAKE PARTIAL JOB COUNT LIST (prior to implementation month)
def make_delayed_job_counts(imp_month, delayed_jnums,
                            lower, upper):
    '''STANDALONE JOB COUNTS
    Make an array of job counts to be inserted into the long_form job counts
    array of the job assignment function.  The main assignment function calls
    this function prior to the implementation month. The array output of this
    function is inserted into what will become the job count column.
    These jobs are from the standalone job results.
    '''
    imp_high = upper[imp_month]
    stand_job_counts = np.zeros(imp_high)
    job_numbers = sorted(list(set(delayed_jnums[:imp_high])))

    for month in np.arange(imp_month + 1):
        l = lower[month]
        h = upper[month]
        jnums_range = delayed_jnums[l:h]
        stand_range = stand_job_counts[l:h]

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
            the number of job levels in the model (from the config file)

        eg_job_counts
            numpy array of the job count lists for the egs

        imp_job_counts
            the total of the jobs available within each job level on the
            implementation date (array)

        allocation
            array of job levels to eg weighting lists.  Key to determine
            the job allocation per level and month until implementation date.
            Total of each list must equal 1.

            example:

            [[1.00, 0.00, 0.00],  # c4

            [.50, 0.25, 0.25],   # c3

            [.88, 0.09, 0.03],   # c2

            [1.00, 0.00, 0.00],  # f4

            [.50, 0.25, 0.25],   # f3

            [.88, 0.09, 0.03],   # f2

            [0.00, 1.00, 0.00],  # c1

            [0.00, 1.00, 0.00]]  # f1

            using the above, if there were 4 additional jobs for job level 2 in
            a given month, eg 1 would get 2 and eg 2 and 3, 1 each.

            ([.50, 0.25, 0.25])
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
    '''
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
            job_list.append(change[0])
            start.append(change[1][0])
            end.append(change[1][1])
            if standalone:
                delta = (change[3][sep_index])
            else:
                delta = (change[2])
            gain_loss.append(delta)
            if this_job_table[0][change[0] - 1] + delta < 0:
                print('Group ' + str(sep_index + 1) +
                      ' ERROR: job reduction below zero, job ' +
                      str(change[0]) +
                      ', final job total is ' +
                      str(this_job_table[0][change[0] - 1] + delta) +
                      ', fur delta input: ' + str(delta) +
                      ', start count: ' +
                      str(this_job_table[0][change[0] - 1]) +
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


def convert_job_changes_to_enhanced(j_changes, job_dict):
    '''Converts job changes based on an 8-level model into a 16-level
    list of job changes using a job dictionary and blk percentages.
    This function allows all job changes to be made at the group level
    rather than constantly recalculating equivalent distribution when adjusting
    job changes.

    Also allows block vs reserve percentages to be adjusted
    with corresponding job change table updating. (TODO)

    inputs

        j_changes
            input from config file describing change of job quantity over
            months of time (list)

            example:

            [1, [35, 64], 87, [80, 7, 0]]:

            [job level, [start and end month],
            total job count change,
            [eg allotment of change for standalone calculations]]

        job_dict
            conversion dictionary for equivalent block and reserve job
            levels with a 16-level model, and block_percentage per group

        (blk inputs)
            ratio of blockholders to reserves
            (most likely international vs domestic)
    '''
    k = []

    for jc in j_changes:
        job = jc[0]
        temp1 = []
        temp2 = []
        blk = job_dict[job][2]
        rsv = 1 - blk

        temp1 = list([job_dict[job][0],
                      jc[1], np.around(jc[2] * blk).astype(int),
                      list(np.around(np.array(jc[3]) * blk).astype(int))])

        temp2 = list([job_dict[job][1],
                      jc[1], np.around(jc[2] * rsv).astype(int),
                      list(np.around(np.array(jc[3]) * rsv).astype(int))])

        k.append(temp1)
        k.append(temp2)

    return k


# To 16 (job levels) from 8 (job levels)
def convert_jcnts_to_enhanced(eg_cnts, blk_int_pcnt, blk_dom_pcnt):
    '''Convert job groups to include blockholder and reserve levels.
    Order is by compensation (precalculated).

        Input is a list of 8 integers.
        Returns a list of 16 integers.

    Inputs

        eg_cnts
            A list of lists of the 8 level job counts
            example: [[161, 170, 885, 326, 192, 928, 109, 132], [...], [...]]

        blk_int_pcnt
            decimal percentage of international jobs (group 3 and 4) that
            are designated as blockholders (schedule holders with more
            pay hours per month than reserve employees)

        blk_dom_pcnt
            decimal percentage of domestic jobs (group 1 and 2) that
            are designated as blockholders (schedule holders with more
            pay hours per month than reserve employees)
    '''
    bi = blk_int_pcnt
    bd = blk_dom_pcnt
    ba = (bi + bd) / 2  # average (group 3 only)
    k = []

    for j in eg_cnts:

        k1 = round(j[0] * bi)
        k2 = j[0] - k1
        k3 = round(j[1] * ba)
        k4 = round(j[2] * bd)
        k5 = j[1] - k3
        k6 = j[2] - k4
        k7 = round(j[3] * bi)
        k8 = j[3] - k7
        k9 = round(j[4] * ba)
        k10 = round(j[5] * bd)
        k11 = round(j[6] * bd)
        k12 = j[4] - k9
        k13 = j[5] - k10
        k14 = j[6] - k11
        k15 = round(j[7] * bd)
        k16 = j[7] - k15

        temp = list([k1, k2, k3, k4, k5, k6, k7, k8,
                     k9, k10, k11, k12, k13, k14, k15, k16])
        k.append(temp)

    return k


# ASSIGN JOBS STANDALONE WITH JOB CHANGES and SUPC option
def assign_standalone_job_changes(df_align,
                                  lower,
                                  upper,
                                  total_months,
                                  job_counts_each_month,
                                  total_monthly_job_count,
                                  nonret_each_month,
                                  job_change_months,
                                  job_reduction_months,
                                  start_month,
                                  df_index,
                                  fur_return=False):
    '''Long_Form
    Uses the job_gain_or_loss_table job count array for job assignments.
    Jobs counts may change up or down in any category for any time period.
    Handles furlough and return of employees.
    Handles prior rights/conditions and restrictions.
    Handles recall of initially furloughed employees.

    Inputs are precalculated outside of function to the extent possible.

    Inputs

        df_align
            dataframe with ['sg', 'fur'] columns
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
        proposal_name_text
            text of proposal order file name without the extension.
            This should be the command line input 'input_proposal'
            if file name of proposal is p3.pkl, proposal_name_text = 'p3'

        furlough_return option -
            allows call to function XXX TODO...

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
    num_of_job_levels = cf.num_of_job_levels
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

    if cf.apply_supc:

        sg_rights = np.array(cf.sg_rights)

        sg_jobs = np.transpose(sg_rights)[1]
        sup_c_counts = np.transpose(sg_rights)[2]
        sg_dict = dict(zip(sg_jobs, sup_c_counts))

        # calc sg sup c condition month range and concat
        sg_month_range = np.arange(np.min(sg_rights[:, 3]),
                                   np.max(sg_rights[:, 4]))
        job_change_months = np.concatenate((job_change_months,
                                            sg_month_range))

    if fur_return:

        recall_months = get_recall_months(cf.recalls)
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
                            fur_range, month, cf.recalls,
                            total_monthly_job_count,
                            standalone=True,
                            eg_index=df_index)

        while job <= num_of_job_levels:

            this_job_count = job_counts_each_month[month, this_job_col]

            if month in job_change_months:

                if cf.apply_supc:

                    if month in sg_month_range and job in sg_jobs:

                        # assign SupC condition jobs to sg employees
                        sg_jobs_avail = min(sg_dict[job], this_job_count)
                        np.put(assign_range,
                               np.where((assign_range == 0) &
                                        (sg_range == 1) &
                                        (fur_range == 0))[0][:sg_jobs_avail],
                               job)

            # TODO, code speedup...
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
        #  TODO **
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
    '''grab config file data settings and put it in a dataframe
    '''
    config_dict = {'lspcnt_calc_on_remaining_population':
                   cf.lspcnt_calc_on_remaining_population,
                   'case_study': cf.case_study,
                   'edit_mode': cf.edit_mode,
                   'enhanced_jobs': cf.enhanced_jobs,
                   'apply_supc': cf.apply_supc,
                   'apply_east_cond': cf.apply_east_cond,
                   'apply_amer_cond': cf.apply_ratio_cond,
                   'starting_date': cf.starting_date,
                   'delayed_implementation': cf.delayed_implementation,
                   'intl_blk_pcnt': cf.intl_blk_pcnt,
                   'dom_blk_pcnt': cf.dom_blk_pcnt,
                   'implementation_date': cf.implementation_date,
                   'no_bump': cf.no_bump,
                   'recall': cf.recall,
                   'actives_only': cf.actives_only,
                   'pay_raise': cf.pay_raise,
                   'annual_pcnt_raise': cf.annual_pcnt_raise,
                   'top_of_scale': cf.top_of_scale,
                   'compute_job_category_order': cf.compute_job_category_order,
                   'compute_pay_measures': cf.compute_pay_measures,
                   'num_of_job_levels': cf.num_of_job_levels}

    settings = pd.DataFrame(config_dict, index=['setting']).stack()
    df = pd.DataFrame(settings, columns=['setting'])
    df.index = df.index.droplevel(0)
    df.index.name = 'option'

    return df


def max_of_nested_lists(nested_list):
    '''
    '''
    max_list = []
    for lst in nested_list:
        x = max(lst)
        max_list.append(x)
    return max(max_list)


