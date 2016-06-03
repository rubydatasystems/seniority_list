# -*- coding: utf-8 -*-

'''Build list orderings from master list data as a starting point for
further analysis and/or list editing.  Lists may be built by various
weighting and sorting methods.

Typical workflow:

prepare_master_list - add columns to master list which can be used as hybrid
list factors.  These columns are longevity, job, and percentage related.

build_list - select, apply weighting, organize and sort a "hybrid" list.

Note: the sort_eg_attributes and sort_and_rank functions are helper functions
which may be used as standalone functions as well.

sort_eg_attributes - normally used within the prepare_master_list function.
Sort date-type attributes by employee group to form a chronological order
within each group without disturbing other columns order.  (also works with
any other attribute if needed).  Typical date columns to prepare in this
manner would be doh and ldate.

The sort_and_rank is a helper function for the build_list function.

The build_list function stores a pickle file that can then be used as an
input to the compute_measures script.
Example:
%run compute_measures.py hybrid

'''

import pandas as pd
import numpy as np

import config as cf
import functions as f


def prepare_master_list(name_int_demo=False, pre_sort=True):
    '''Add attribute columns to master list which can be used as factors
    to construct a list ordering.

    Employee groups must be listed in seniority order in relation to employees
    from the same group.  Order between groups is uninmportant at this step.

    New columns added: ['age', 's_lmonths', 'jnum', 'job_count', 'rank_in_job',
    'jobp', 'eg_number', 'eg_spcnt']

    input

        name_int_demo
            if True, lname strings are converted to an integer then a
            corresponding alpha-numeric percentage for constructing lists by
            last name.  This is a demo only to show that any attribute
            may be used as a list weighting factor.

        pre_sort
            sort the master data dataframe doh and ldate columns prior to
            beginning any calculations.  This sort has no effect on the other
            columns.  The s_lmonths coulumn will be calculated on the sorted
            ldate data.

    Job-related attributes are referenced to job counts from the config file.
    '''

    if cf.sample_mode:
        sample_prefix = cf.sample_prefix
        master_ = pd.read_pickle('sample_data/' + sample_prefix + 'master.pkl')
    else:
        master_ = pd.read_pickle('dill/master.pkl')

    if pre_sort:
        sort_eg_attributes(master_)

    master = master_[(master_.line == 1) | (master_.fur == 1)].copy()

    # AGE and LONGEVITY
    master['age'] = f.starting_age(master.retdate)
    master['s_lmonths'] = f.longevity_at_startdate(list(master['ldate']),
                                                   return_months=True)

    jobs_list = []

    if cf.enhanced_jobs:
        eg_counts = f.convert_jcnts_to_enhanced(cf.eg_counts,
                                                cf.intl_blk_pcnt,
                                                cf.dom_blk_pcnt)
    else:
        eg_counts = cf.eg_counts

    # make a list of stovepipe jobs for each group (from config job counts)
    i = 1
    for jobs in eg_counts:
        # the second input determines the length of the zero
        # array formed (possible excess)
        jobs_list.append(
            f.make_stovepipe_jobs_from_jobs_arr(jobs,
                                                sum((master.eg == i) &
                                                    ((master.line == 1) |
                                                     (master.fur == 1)))))
        i += 1

    fur_level = f.max_of_nested_lists(jobs_list) + 1
    jobs = np.array(jobs_list)

    # mark unassigned as furloughed (from zero to fur_level)
    for job_arr in jobs:
        np.put(job_arr, np.where(job_arr == 0)[0], fur_level)

    egs = np.array(master.eg)
    jnums = np.zeros(egs.size)
    job_count = np.zeros(egs.size)

    # JNUM and JOB_COUNT data prep
    i = 1
    for job_arr in jobs:
        data = np.unique(job_arr, return_counts=True)
        zipped = zip(data[0], data[1])
        for job, count in zipped:
            np.put(job_count,
                   np.where((jnums == 0) & (egs == i))[0][:count],
                   count)
            np.put(jnums, np.where((jnums == 0) & (egs == i))[0][:count], job)
        i += 1

    # Employee group count (for spcnt column)
    eg_counts = np.zeros(egs.size)
    data = np.unique(master.eg, return_counts=True)
    zipped = zip(data[0], data[1])
    for eg, count in zipped:
        np.put(eg_counts, np.where(egs == eg)[0], count)

    # Attribute columns assignment
    master['jnum'] = jnums.astype(int)
    master['job_count'] = job_count.astype(int)
    master['rank_in_job'] = master.groupby(['eg', 'jnum']).cumcount() + 1
    master['jobp'] = (master.rank_in_job /
                      master.job_count) + master.jnum - .0001
    master['eg_number'] = master.groupby('eg').cumcount() + 1
    master['eg_count'] = eg_counts.astype(int)
    master['eg_spcnt'] = master.eg_number / master.eg_count
    if name_int_demo:
        master['name_int'] = names_to_integers(master.lname)[2]

    master.pop('eg_count')

    return master


def build_list(df, measure_list, weight_list, show_weightings=False,
               return_df=False, hide_rank_cols=True):
    '''Construct a "hybrid" list ordering.

    Combine and sort various attributes according to variable multipliers to
    produce a list order. The list order output is based on a sliding scale
    of the priority assigned amoung the attributes.  The attribute values
    from the employee groups may be evenly ratioed together or combined
    on an absolute basis where the actual values determine the positioning.

    The output is the resultant dataframe and also a list order is written to
    disk as 'dill/hybrid.pkl'.

    input
        df
            the prepared dataframe output of the prepare_master_list function

        measure_list
            a list of attributes that form the basis of the final sorted list.
            The employee groups will be combined, sorted, and numbered
            according to these attributes one by one.  Each time the current
            attribute numbered list is formed, a weighting is applied to that
            order column.  The final result number will be the rank of the
            cummulative total of the weighted attribute columns.

        weight_list
            a list of decimal weightings to apply to each corresponding
            measure within the measure_list.  Normally the total of the
            weight_list should be 1, but any numbers may be used as weightings
            since the final result is a ranking of a cumulative total.

        show_weightings
            add columns to display the product of the weight/column
            mutiplcation

        return_df
            option to return the new sorted hybrid dataframe as output.
            Normally, the function produces a list ordering file which is
            written to disk and used as an input by the compute measures
            script.

        hide_rank_cols
            remove the attrubute rank columns from the dataframe unless
            visual review is desired
    '''

    # options todo:
    #  , absolute=True,
    # invert=False, include_inactives=False, include_fur=True,
    # cut=False, qcut=False, remove_retired=True):

    df = df.copy()
    df['hybrid'] = 0
    for i in np.arange(len(measure_list)):

        if show_weightings:
            sort_and_rank(df, measure_list[i])
            df[measure_list[i] + '_wgt'] = \
                df[measure_list[i] + '_rank'] * weight_list[i]
            df['hybrid'] += df[measure_list[i] + '_wgt']
        else:
            sort_and_rank(df, measure_list[i])
            hybrid = np.array(df[measure_list[i] + '_rank'] * weight_list[i])
            df['hybrid'] += hybrid

    df = sort_and_rank(df, 'hybrid')
    if hide_rank_cols:
        for measure in measure_list:
            df.pop(measure + '_rank')
        df['idx'] = df.hybrid_rank
        df.pop('hybrid_rank')
    else:
        df['idx'] = np.arange(len(df), dtype=int) + 1
    df.set_index('empkey', drop=True, inplace=True)
    df[['idx']].to_pickle('dill/hybrid.pkl')

    if return_df:
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('idx')))
        df = df.reindex(columns=cols)
        return df


def sort_eg_attributes(df, attributes=['doh', 'ldate'],
                       reverse_list=[0, 0], add_columns=False):
    '''Sort master list attribute columns by employee group in preparation
    for list construction.  The overall master list structure and order is
    unaffected, only the selected attribute columns are sorted (normally
    date-related columns such as doh or ldate)

    inputs

        df
            The master data dataframe (does not need to be sorted)

        attributes
            columns to sort by eg (inplace)

        reverse_list
            If an attribute is to be sorted in reverse order (descending),
            use a '1' in the list position corresponding to the position of
            the attribute within the attributes input

        add_columns
            If True, an additional column for each sorted attribute will be
            added to the resultant dataframe, with the suffix '_sort' added
            to it.
    '''
    date_cols = []
    for col in df:
        if (df[col]).dtype == 'datetime64[ns]':
            date_cols.append(col)
    try:
        df.sort_values(['eg', 'eg_number'], inplace=True)
    except:
        df.sort_values(['eg', 'eg_order'], inplace=True)

    egs = np.array(df.eg)
    i = 0
    for measure in attributes:
        data = np.array(df[measure])
        measure_col = np.empty_like(data)
        for eg in np.unique(df.eg):
            measure_slice = data[egs == eg]
            measure_slice_index = np.where(egs == eg)[0]
            measure_slice_sorted = np.sort(measure_slice, axis=0)

            if reverse_list[i]:
                measure_slice_invert = measure_slice_sorted[::-1]
                measure_slice_sorted = measure_slice_invert
            np.put(measure_col, measure_slice_index, measure_slice_sorted)

        if add_columns:
            col_name = measure + '_sort'
        else:
            col_name = measure

        df[col_name] = measure_col

        if measure in date_cols:
            df[col_name] = pd.to_datetime(df[col_name].dt.date)
        i += 1

    return df


def sort_and_rank(df, col, tiebreaker1='eg_spcnt', tiebreaker2='ldate',
                  reverse=False):
    '''Sort a datframe by a specified attribute and insert a column indicating
    the resultant ranking.  Tiebreaker inputs select columns to be used for
    secondary ordering in the event of value ties. Reverse ordering may be
    selected as an option.

    inputs

        df
            input dataframe

        col
            dataframe column to sort

        tiebreaker1, tiebreaker2
            second and third sort columns to break ties with primary col sort

        reverse
            If True, reverses sort (descending values)
    '''

    if reverse:
        df.sort_values([col, tiebreaker1, tiebreaker2],
                       ascending=False, inplace=True)
    else:
        df.sort_values([col, tiebreaker1, tiebreaker2],
                       inplace=True)

    df[col + '_rank'] = np.arange(len(df), dtype=float) + 1

    return df


def names_to_integers(names, leading_precision=5, normalize_alpha=True):
    '''convert a list of string names (i.e. last names) into integers
    for numerical sorting

    input

        names
            List of strings for conversion to integers
        leading_precision
            Number of characters to use with full numeric precision, remainder
            of characters will be assigned a rounded single digit between
            0 and 9
        normalize_alpha
            If True, insert 'aaaaaaaaaa' and 'zzzzzzzzzz' as bottom and
            top names. Otherwise, bottom and top names will be calculated
            from within the names input
    output

        1. an array of the name integers
        2. the range of the name integers,
        3. an array of corresponding percentages for each name integer
           relative to the range of name integers array

    Note: This function demonstrates the possibility of constructing
    a list using any type or combination of attributes.
    '''
    if normalize_alpha:
        names = np.append(names, ['aaaaaaaaaa', 'zzzzzzzzzz'])
    int_names = np.zeros_like(names)
    max_str_len = len(max(names, key=len))
    alpha_numer = {'a': '01', 'b': '04', 'c': '08', 'd': '12', 'e': '16',
                   'f': '20', 'g': '24', 'h': '28', 'i': '32', 'j': '36',
                   'k': '40', 'l': '44', 'm': '48', 'n': '52', 'o': '56',
                   'p': '60', 'q': '64', 'r': '68', 's': '72', 't': '76',
                   'u': '80', 'v': '83', 'w': '87', 'x': '91', 'y': '95',
                   'z': '99'}

    j = 0

    for name in names:
        num_convert = ''
        name = name.lower()
        for i in np.arange(max_str_len):
            if i < leading_precision:
                try:
                    num_convert += alpha_numer[name[i]]
                except:
                    num_convert += '00'
            else:
                try:
                    num_convert += str(int(int(alpha_numer[name[i]]) * .1))
                except:
                    num_convert += '0'
        num_convert = int(num_convert)
        int_names[j] = num_convert
        j += 1

    int_names = int_names.astype(float)
    name_min = np.amin(int_names)
    name_max = np.amax(int_names)
    int_range = name_max - name_min
    name_percentages = (int_names - name_min) / int_range

    if normalize_alpha:
        int_names = int_names[:-2]
        name_percentages = name_percentages[:-2]

    return int_names, int_range, name_percentages
