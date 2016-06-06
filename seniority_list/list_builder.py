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


def find_row_orphans(base_df, compare_df, col,
                     ignore_case=True, print_output=False):
    '''Given two columns (series) with the same column label in separate pandas
    dataframes, return values which are unique to one or the other column,
    not common to both series.
    Will also work with dataframe indexes.

    Returns separate dataframse with the series orphans.

    inputs

        base_df
            first dataframe to compare

        compare_df
            second dataframe to compare

        col
            column label of the series to compare.
            routine will compare the dataframe indexes with the
            input of 'index'.

        ignore_case
            convert col to lowercase prior to comparison

        print_output
            print results instead of returning results
    '''

    col = col.lower()
    base_df.columns = map(str.lower, base_df.columns)
    compare_df.columns = map(str.lower, compare_df.columns)

    if col == 'index':
        base_series = base_df.index
        compare_series = compare_df.index
    else:
        if (col not in base_df) or (col not in compare_df):
            print(col + ' is not a column in both dataframes...')
            return
        else:
            base_series = base_df[col]
            compare_series = compare_df[col]

    if ignore_case:
        try:
            base_series = base_series.str.lower()
            compare_series = compare_series.str.lower()
            base_df[col] = base_series
            compare_df[col] = compare_series
        except:
            pass

    base_orphans = list(base_series[~base_series.isin(compare_series)])
    compare_orphans = list(compare_series[~compare_series.isin(base_series)])
    base_col_name = 'base_orphans'
    compare_col_name = 'compare_orphans'

    base_loners = pd.DataFrame(base_orphans,
                               columns=[base_col_name])
    compare_loners = pd.DataFrame(compare_orphans,
                                  columns=[compare_col_name])

    def find_label_locs(df, orphans):

        loc_list = []
        for orphan in orphans:
            loc_list.append(df.index.get_loc(orphan))
        return loc_list

    def find_val_locs(df, orphans):

        loc_list = []
        for orphan in orphans:
            if base_df[col].dtype == 'datetime64[ns]':
                loc_list.append(list(df[col]).index(pd.to_datetime(orphan)))
            else:
                loc_list.append(list(df[col]).index(orphan))
        return loc_list

    if base_orphans:
        if col == 'index':
            base_loners['index_loc'] = find_label_locs(base_df, base_orphans)
        else:
            base_loners['index_loc'] = find_val_locs(base_df, base_orphans)

    if compare_orphans:
        if col == 'index':
            compare_loners['index_loc'] = find_label_locs(compare_df,
                                                          compare_orphans)
        else:
            compare_loners['index_loc'] = find_val_locs(compare_df,
                                                        compare_orphans)

    if print_output:
        print('BASE:\n', base_loners, '\nCOMPARE:\n', compare_loners)
    else:
        return base_loners, compare_loners


def compare_dataframes(base, compare, return_orpahns=True,
                       ignore_case=True, print_info=False,
                       convert_np_timestamps=True):
    """
    Compare all common index and common column DataFrame values and
    report if any value is not equal in a returned dataframe.

    Values are compared only by index and column label, not order.
    Therefore, the only values compared are within common index rows
    and common columns.  The routine will report the common columns and
    any unique index rows when the print_info option is selected (True).

    Inputs are pandas dataframes and/or pandas series

    inputs

        base
            baseline dataframe or series

        compare
            dataframe or series to compare against the baseline (base)

        return_orphans
            separately calculate and return the rows which are unique to
            base and compare

        ignore_case
            convert the column labels and column data to be compared to
            lowercase - this will avoid differences detected based on string
            case

        print_info
            option to print out to console verbose statistical information
            and the dataframe(s) instead of returning dataframe(s)

        convert_np_timestamps
            numpy returns datetime64 objects when the source is a datetime
            date-only object.
            this option will convert back to a date-only object for comparison.

    """
    try:
        assert ((isinstance(base, pd.DataFrame)) |
                (isinstance(base, pd.Series))) and \
            ((isinstance(compare, pd.DataFrame)) |
             (isinstance(compare, pd.Series)))
    except:
        print('Routine aborted. Inputs must be a pandas dataframe or series.')
        return

    if isinstance(base, pd.Series):
        base = pd.DataFrame(base)
    if isinstance(compare, pd.Series):
        compare = pd.DataFrame(compare)

    common_rows = list(base.index[base.index.isin(compare.index)])

    if print_info:
        print('\nROW AND INDEX INFORMATION:\n')
        print('base length:', len(base))
        print('comp length:', len(compare))
        print('common index count:', len(common_rows), '\n')

    # ------------------------------------------------------------
    if return_orpahns:
        base_orphans = list(base.index[~base.index.isin(compare.index)])
        compare_orphans = list(compare.index[~compare.index.isin(base.index)])
        base_col_name = 'base_orphans'
        compare_col_name = 'compare_orphans'

        base_loners = pd.DataFrame(base_orphans,
                                   columns=[base_col_name])
        compare_loners = pd.DataFrame(compare_orphans,
                                      columns=[compare_col_name])

        def find_label_locs(df, orphans):

            loc_list = []
            for orphan in orphans:
                loc_list.append(df.index.get_loc(orphan))
            return loc_list

        if base_orphans:
            base_loners['index_loc'] = find_label_locs(base, base_orphans)
            if print_info:
                print('BASE LONERS (rows, by index):')
                print(base_loners, '\n')
        else:
            if print_info:
                print('''There are no unique index rows in the base input vs.
                      the compare input.\n''')

        if compare_orphans:
            compare_loners['index_loc'] = find_label_locs(compare,
                                                          compare_orphans)
            if print_info:
                print('COMPARE LONERS (rows, by index):')
                print(compare_loners, '\n')
        else:
            if print_info:
                print('''There are no unique index rows in the compare input
                      vs. the base input.\n''')
    # -----------------------------------------------------------------------

    base = base.loc[common_rows]
    compare = compare.loc[common_rows]

    unequal_cols = []
    equal_cols = []

    if ignore_case:
        base.columns = map(str.lower, base.columns)
        compare.columns = map(str.lower, compare.columns)

    common_cols = list(base.columns[base.columns.isin(compare.columns)])
    base_only_cols = list(base.columns[~base.columns.isin(compare.columns)])
    comp_only_cols = list(compare.columns[~compare.columns.isin(base.columns)])

    if print_info:
        same_col_list = []
        print('\nCOMMON COLUMN equivalency:\n')
    for col in common_cols:
        if ignore_case:
            try:
                base[col] = base[col].str.lower()
                compare[col] = compare[col].str.lower()
            except:
                pass
        same_col = base[col].sort_index().equals(compare[col].sort_index())
        if print_info:
            same_col_list.append(same_col)
        if not same_col:
            unequal_cols.append(col)
        else:
            equal_cols.append(col)

    base = base[unequal_cols]
    compare = compare[unequal_cols]

    if print_info:
        same_col_df = pd.DataFrame(list(zip(common_cols, same_col_list)),
                                   columns=['common_col', 'equivalent'])
        same_col_df.sort_values(['equivalent', 'common_col'], inplace=True)
        same_col_df.reset_index(drop=True, inplace=True)
        print(same_col_df, '\n')
        print('\nCOLUMN INFORMATION:')
        print('\ncommon columns:\n', common_cols)
        print('\ncommon and equal columns:\n', equal_cols)
        print('\ncommon but unequal columns:\n', unequal_cols)
        print('\ncols only in base:\n', base_only_cols)
        print('\ncols only in compare:\n', comp_only_cols, '\n')

    zipped = []
    col_counts = []
    for col in base:
        base_np = np.array(base[col])
        compare_np = np.array(compare[col])
        row_ = np.where(base_np != compare_np)[0]
        index_ = base.iloc[np.where(base_np != compare_np)[0]].index
        col_ = np.array([col] * row_.size)
        base_ = base_np[np.where(base_np != compare_np)]
        compare_ = compare_np[np.where(base_np != compare_np)]
        if (base[col]).dtype == 'datetime64[ns]' and convert_np_timestamps:
            try:
                base_ = base_.astype('M8[D]').astype('O')
                compare_ = compare_.astype('M8[D]').astype('O')
            except:
                pass
        zipped.extend(list(zip(row_, index_, col_, base_, compare_)))
        col_counts.append(row_.size)

    diffs = pd.DataFrame(
        zipped, columns=['row', 'index', 'column', 'base', 'compare'])
    diffs.sort_values('row', inplace=True)
    diffs.reset_index(drop=True, inplace=True)

    if print_info:
        print('\nDIFFERENTIAL DATAFRAME:\n')
        print(diffs)
        print('\nSUMMARY:\n')
        print('''{!r} total differences found in
              common rows and columns\n'''.format(len(zipped)))

        if len(zipped) == 0:
            print('''Comparison complete, dataframes are
                  equivalent. \nIndex and Column order may be different\n''')
        else:
            print('Breakdown by column:\n',
                  pd.DataFrame(list(zip(base.columns, col_counts)),
                               columns=['column', 'diff_count']),
                  '\n')

    else:
        if return_orpahns:
            return diffs, base_loners, compare_loners
        else:
            return diffs