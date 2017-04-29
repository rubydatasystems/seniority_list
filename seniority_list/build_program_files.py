#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Create the necessary program support files from the input Excel files
for the program operation.  The Excel files are read from the case-named
folder within the excel folder.

# -------------------------------------------------

Excel files

from proposals.xlsx:
    <worksheet name>.pkl for each proposal
    proposal_names.pkl

from master.xlsx:
    master.pkl,
    last_month.pkl
    dict_job_tables.pkl

from pay_tables.xlsx:

    pay_table_basic.pkl
    pay_table_enhanced.pkl
    pay_table_data.xlsx

from settings.xlsx

    dict_settings.pkl
    dict_attr.pkl

the .pkl files will be stored in the dill folder.  the dill folder is
created if it does not exist.

# -------------------------------------------------

initialized with this script (independent of input files):

    squeeze_vals.pkl
    case_dill.pkl
    squeeze_vals.pkl
    dict_color.pkl
    case-study-named folder in the **reports** folder
        (if it doesn't already exist)

example usage to run this script from the jupyter notebook:
    %run build_program_files sample3
'''
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from collections import OrderedDict as od
import os
import functions as f
from matplotlib_charting import make_color_list as mcl
from converter import convert as cnv

from sys import argv


def main():

    script, case = argv

    os.makedirs('dill/', exist_ok=True)

    try:
        # check to see if file exists and get value if it does
        case_dill_value = pd.read_pickle('dill/case_dill.pkl').case.value
    except OSError:
        case_dill_value = 'empty_placeholder'

    if case_dill_value == case:
        # if stored value is same as case study name, remove the files
        # which will be replaced.  Removal of old files then writing the
        # new files to disk is faster than overwriting the old files.
        if os.path.isdir('dill/'):

            clear_files = ['squeeze_vals.pkl', 'last_month.pkl',
                           'dict_color.pkl', 'dict_settings.pkl',
                           'dict_attr.pkl', 'master.pkl',
                           'pay_table_enhanced.pkl', 'pay_table_basic.pkl']

            filelist = \
                [pkl for pkl in os.listdir('dill/') if pkl in clear_files]
            for pkl in filelist:
                os.remove('dill/' + pkl)
    else:
        # if the case name is different, delete all dill files (stored
        # calculated files).
        # create new case_dill.pkl file
        f.clear_dill_files()
        case_dill = pd.DataFrame({'case': case}, index=['value'])
        case_dill.to_pickle('dill/case_dill.pkl')

    # START THE SETTINGS DICTIONARY - POPULATE WITH THE SCALARS ONLY
    # some of these values will be used for pay data calculation
    # Then some of the calculated pay data is used to further populate the
    # settings dictionary

    xl = pd.read_excel('excel/' + case + '/settings.xlsx',
                       sheetname=None)
    settings = defaultdict(int)
    # ## scalars
    settings.update(f.make_dict_from_columns(xl['scalars'], 'option', 'value'))

    # PAY TABLES $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    xl_pay_path = 'excel/' + case + '/pay_tables.xlsx'

    # read pay table data from excel file
    pay_rates = pd.read_excel(xl_pay_path, sheetname='rates')
    # read monthly pay hours per job level and job description from excel file
    pay_hours = pd.read_excel(xl_pay_path, sheetname='hours')

    # inputs to determine global sorting master year and
    # longevity...function parameters
    # include second set of parameters for enhanced model and set to None
    # check for not None to alter for enhanced sort...
    year = settings['pay_table_year_sort']
    longevity = settings['pay_table_longevity_sort']

    # instantiate dict values to None
    basic_compen = None
    full_mth_compen = None
    part_mth_compen = None
    job_key_enhan = None
    job_key_basic = None
    basic = None
    enhanced = None
    job_dict_df = None

    # numpy unique returns a SORTED array of unique elements
    contract_years = np.unique(pay_rates.year)

    # extract integer column names (represents years of pay longevity)
    longevity_cols = []
    for col in pay_rates.columns.values.tolist():
        try:
            int(col)
            longevity_cols.append(col)
        except ValueError:
            pass

    table_cols = ['year', 'jnum']
    table_cols.extend(longevity_cols)

    basic = pd.merge(pay_rates, pay_hours)

    # For enhanced_jobs:
    enhanced_full = basic.copy()
    enhanced_part = basic.copy()

    # SELECTED COLUMNS MULTIPLIED BY A DESIGNATED COLUMN ROW VALUE

    basic[longevity_cols] = basic[longevity_cols]\
        .multiply(basic['basic_hours'], axis="index")

    # sort by year and job level and only keep columns: 'year', 'jnum',
    # and all year longevity (integer) columns

    basic_compen = basic.sort_values(['year', 'jnum'])[table_cols]\
        .set_index('year', drop=True)

    # create small dataframes for furloughed pay data (no pay)
    fur_rows = pd.DataFrame(0., index=np.arange(len(contract_years)),
                            columns=basic.columns)

    basic_fur_rows = fur_rows.copy()
    basic_fur_rows.jnum = basic.jnum.max() + 1
    basic_fur_rows.year = contract_years
    basic_fur_rows.jobstr = 'FUR'

    # CONCATENATE the furlough pay data to the basic and enhanced pay data
    basic = pd.concat([basic, basic_fur_rows])

    # select a SECTION OF THE PAY DATA TO USE AS A MASTER ORDER
    # for entire pay dataframe(s).
    # In other words, the job level order of the entire pay
    # dataframe will match the selected year and pay longevity
    # order, even if certain year and pay level compensation
    # amounts are not in descending order.
    # The order must be consistent for the data model.
    order_basic = basic[basic.year == year][['jnum', longevity, 'jobstr']]\
        .sort_values(longevity, ascending=False)

    order_basic['order'] = np.arange(len(order_basic)) + 1

    job_key_basic = order_basic[['order', 'jobstr', 'jnum']].copy()

    # make a dataframe to save the job level hierarchy

    job_key_basic.set_index('order', drop=True, inplace=True)
    job_key_basic.rename(columns={'jnum': 'orig_order'}, inplace=True)

    # this is the way to sort each job level heirarchy for each year.
    # this dataframe is merged with the 'enhanced' dataframe
    # then enhanced is sorted by year and order columns
    order_basic = order_basic.reset_index()[['jnum', 'order']]

    basic = pd.merge(basic, order_basic).sort_values(['year', 'order'])\
        .reset_index(drop=True)

    basic.jnum = basic.order

    basic_df = basic[table_cols].copy()

    # MELT AND INDEX - CREATING INDEXED MONTHLY PAY DATAFRAME(S)
    melt_basic = pd.melt(basic_df, id_vars=['year', 'jnum'],
                         var_name='scale',
                         value_name='monthly')

    melt_basic['ptindex'] = (melt_basic.year * 100000 +
                             melt_basic.scale * 100 +
                             melt_basic.jnum)

    melt_basic.drop(['scale', 'year', 'jnum'], axis=1, inplace=True)
    melt_basic.sort_values('ptindex', inplace=True)
    melt_basic.set_index('ptindex', drop=True, inplace=True)
    melt_basic.to_pickle('dill/pay_table_basic.pkl')

    # Calculate for enhanced_jobs and write to workbook
    # ENHANCED JOBS

    # calculate monthly compensation for each job level and pay longevity
    enhanced_full[longevity_cols] = enhanced_full[longevity_cols]\
        .multiply(enhanced_full['full_hours'], axis="index")

    enhanced_part[longevity_cols] = enhanced_part[longevity_cols]\
        .multiply(enhanced_part['part_hours'], axis="index")

    # ENHANCED TABLE SUFIXES, COLUMNS, JNUMS(ENHANCED_PART)

    # make enhanced_part (fewer hours per position per month)
    # jnums begin with maximum enhanced_full jnum + 1 and
    # increment upwards
    enhanced_part.jnum = enhanced_part.jnum + enhanced_part.jnum.max()

    # sort by year and job level and only keep columns: 'year', 'jnum',
    # and all year longevity (integer) columns

    full_mth_compen = enhanced_full.sort_values(['year',
                                                'jnum'])[table_cols]\
        .set_index('year', drop=True)
    part_mth_compen = enhanced_part.sort_values(['year',
                                                'jnum'])[table_cols]\
        .set_index('year', drop=True)

    # add appropriate suffixes to jobstr columns for full
    # and part enhanced tables
    full_suf = settings['enhanced_jobs_full_suffix']
    part_suf = settings['enhanced_jobs_part_suffix']
    enhanced_full.jobstr = enhanced_full.jobstr.astype(str) + full_suf
    enhanced_part.jobstr = enhanced_part.jobstr.astype(str) + part_suf

    # CONCATENATE the full and part(-time) enhanced jobs dataframes
    enhanced = pd.concat([enhanced_full, enhanced_part])

    enhan_fur_rows = fur_rows.copy()
    enhan_fur_rows.jnum = enhanced.jnum.max() + 1
    enhan_fur_rows.year = contract_years
    enhan_fur_rows.jobstr = 'FUR'

    # CONCATENATE the furlough pay data to the basic and
    # enhanced pay data
    enhanced = pd.concat([enhanced, enhan_fur_rows])

    # select a SECTION OF THE PAY DATA TO USE AS A MASTER ORDER
    # for entire pay dataframe(s).
    order_enhan = \
        enhanced[enhanced.year == year][['jnum', longevity, 'jobstr']]\
        .sort_values(longevity, ascending=False)

    order_enhan['order'] = np.arange(len(order_enhan)) + 1
    job_key_enhan = order_enhan[['order', 'jobstr', 'jnum']].copy()

    # make a dataframe to assist with job dictionary construction
    # (case_specific config file variable 'jd')

    s = job_key_enhan['jnum'].reset_index(drop=True)
    jobs = np.arange((s.max() - 1) / 2) + 1
    j_cnt = jobs.max()
    idx_list1 = []
    idx_list2 = []
    for job_level in jobs:
        idx_list1.append(s[s == job_level].index[0] + 1)
        idx_list2.append(s[s == job_level + j_cnt].index[0] + 1)

    dict_data = (('job', jobs.astype(int)),
                 ('full', idx_list1),
                 ('part', idx_list2),
                 ('jobstr', list(job_key_basic.jobstr[:int(j_cnt)])),
                 ('full_pcnt', list(pay_hours.full_pcnt)))
    # use of ordered dict preserves column order
    job_dict_df = pd.DataFrame(data=od(dict_data)).set_index('job',
                                                             drop=True)

    # make a dataframe to save the job level hierarchy

    job_key_enhan.set_index('order', drop=True, inplace=True)
    job_key_enhan.rename(columns={'jnum': 'concat_order'}, inplace=True)
    order_enhan = order_enhan.reset_index()[['jnum', 'order']]
    enhanced = pd.merge(enhanced,
                        order_enhan).sort_values(['year', 'order'])\
        .reset_index(drop=True)

    enhanced.jnum = enhanced.order
    enhanced_df = enhanced[table_cols].copy()

    # MELT AND INDEX - CREATING INDEXED MONTHLY PAY DATAFRAME(S)

    melt_enhan = pd.melt(enhanced_df, id_vars=['year', 'jnum'],
                         var_name='scale',
                         value_name='monthly')

    melt_enhan['ptindex'] = (melt_enhan.year * 100000 +
                             melt_enhan.scale * 100 +
                             melt_enhan.jnum)

    melt_enhan.drop(['scale', 'year', 'jnum'], axis=1, inplace=True)
    melt_enhan.sort_values('ptindex', inplace=True)
    melt_enhan.set_index('ptindex', drop=True, inplace=True)
    melt_enhan.to_pickle('dill/pay_table_enhanced.pkl')

    # WRITE PAY DATA TO EXCEL FILE - WITHIN CASE-NAMED FOLDER
    # WITHIN THE 'REPORTS' FOLDER

    path = 'reports/' + case + '/'
    os.makedirs(path, exist_ok=True)

    writer = pd.ExcelWriter(path + 'pay_table_data.xlsx')
    # string to dataframe items for ws_dict
    dict_items = (('basic (no sort)', basic_compen),
                  ('enhanced full (no sort)', full_mth_compen),
                  ('enhanced part (no sort)', part_mth_compen),
                  ('basic ordered', basic),
                  ('enhanced ordered', enhanced),
                  ('basic job order', job_key_basic),
                  ('enhanced job order', job_key_enhan),
                  ('job dict', job_dict_df))

    ws_dict = od(dict_items)
    # write pay data dataframes to workbook
    for key, value in ws_dict.items():
        try:
            value.to_excel(writer, key)
        except:
            pass

    writer.save()

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # dict items from calculated pay data - refactor to eliminate reading file.
    # just use variables from above...

    xl_pay = pd.read_excel('reports/' + case + '/pay_table_data.xlsx',
                           sheetname=['basic job order',
                                      'enhanced job order',
                                      'job dict'])
    df_jd = xl_pay['job dict']
    df_jd['list_cols'] = f.make_lists_from_columns(xl_pay['job dict'],
                                                   ['full', 'part',
                                                   'full_pcnt'])

    settings['jd'] = f.make_dict_from_columns(df_jd, 'job', 'list_cols')

    if settings['enhanced_jobs']:
        descr_df = xl_pay['enhanced job order']
    else:
        descr_df = xl_pay['basic job order']

    job_strings = list(descr_df.jobstr)
    settings['job_strs'] = job_strings
    settings['job_strs_dict'] = od(enumerate(job_strings, 1))

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # ADD MORE ITEMS TO SETTINGS DICTIONARY //////////////////////////////////

    settings['ret_incr'] = \
        f.make_tuples_from_columns(xl['ret_incr'],
                                   ['month_start', 'month_increase'],
                                   return_as_list=False,
                                   return_dates_as_strings=True,
                                   date_cols=['month_start'])

    # ## init_ret_age

    settings['init_ret_age'] = settings['init_ret_age_years'] + \
        (settings['init_ret_age_months'] * 1 / 12)

    # ## ret_incr_dict

    settings['ret_incr_dict'] = od(settings['ret_incr'])

    # ## ret_age

    init_ret_age = settings['init_ret_age']
    if settings['ret_age_increase']:
        ret_dict = settings['ret_incr_dict']
        ret_age = init_ret_age + sum(ret_dict.values()) * (1 / 12)
    else:
        ret_age = init_ret_age

    settings['ret_age'] = ret_age

    # ## start

    settings['start'] = pd.to_datetime(settings['starting_date'])

    # ## imp_date

    settings['imp_date'] = \
        pd.to_datetime(settings['implementation_date'])

    # ## imp_month

    imp_date = settings['imp_date']
    start = settings['start']

    settings['imp_month'] = ((imp_date.year - start.year) * 12) - \
        (start.month - imp_date.month)

    # ## num_of_job_levels

    if settings['enhanced_jobs']:
        settings['num_of_job_levels'] = settings['job_levels_enhanced']
    else:
        settings['num_of_job_levels'] = settings['job_levels_basic']

    df = xl['job_counts']
    filter_cols = \
        [col for col in df.columns.values.tolist() if col.startswith('eg')]
    df_filt = df[filter_cols]
    eg_counts = []
    for col in df_filt:
        eg_counts.append(list(df_filt[col]))
    settings['eg_counts'] = eg_counts

    # ## j_changes

    df = xl['job_changes']

    start = list(df.month_start)
    end = list(df.month_end)
    jc_set = set()
    for i in np.arange(len(start)):
        jc_set = jc_set.union(set(range(start[i], end[i] + 1)))
    settings['jc_months'] = jc_set

    df['lister1'] = f.make_lists_from_columns(df, ['month_start', 'month_end'])
    filter_cols = \
        [col for col in df.columns.values.tolist() if col.startswith('eg')]
    df['lister2'] = f.make_lists_from_columns(df, filter_cols)
    settings['j_changes'] = f.make_lists_from_columns(df, ['job', 'lister1',
                                                           'total_change',
                                                           'lister2'])

    # ## recalls

    df = xl['recall']
    filter_cols = \
        [col for col in df.columns.values.tolist() if col.startswith('eg')]
    df['lister'] = f.make_lists_from_columns(df, filter_cols)
    settings['recalls'] = f.make_lists_from_columns(df, ['total_monthly',
                                                         'lister',
                                                         'month_start',
                                                         'month_end'])

    # ## sg_rights

    df = xl['prex']

    # make count ratio condition month range
    month_start = df.month_start.min()
    month_end = df.month_end.max()
    settings['prex_month_range'] = set(range(month_start, month_end + 1))

    sg_col_list = ['eg', 'job', 'count', 'month_start', 'month_end']
    filter_cols = \
        [col for col in df.columns.values.tolist() if col in sg_col_list]
    settings['sg_rights'] = f.make_lists_from_columns(df, filter_cols)

    # ## ratio_cond

    df = xl['ratio_cond']

    # make count ratio condition month range
    month_start = df.month_start.min()
    month_end = df.month_end.max()
    settings['ratio_month_range'] = set(range(month_start, month_end + 1))

    df_cols = df.columns.values.tolist()
    group_cols = [col for col in df_cols if col.startswith('group')]
    weight_cols = [col for col in df_cols if col.startswith('weight')]
    for col in group_cols:
        df[col] = f.make_group_lists(df, col)

    df['grp_tup'] = f.make_lists_from_columns(df, group_cols,
                                              remove_zero_values=True,
                                              as_tuples=True)
    df['wgt_tup'] = f.make_lists_from_columns(df,
                                              weight_cols,
                                              remove_zero_values=True,
                                              as_tuples=True)
    df = df[['basic_job', 'grp_tup', 'wgt_tup',
             'month_start', 'month_end']].copy()
    cols = [col for col in df if col != 'basic_job']
    comb = f.make_lists_from_columns(df, cols)
    df = pd.DataFrame({'job': df.basic_job, 'data': comb})
    settings['ratio_dict'] = f.make_dict_from_columns(df, 'job', 'data')

    # ## count_ratio_dict

    df = xl['ratio_count_capped_cond']

    # make count ratio condition month range
    month_start = df.month_start.min()
    month_end = df.month_end.max()
    settings['count_ratio_month_range'] = set(range(month_start,
                                                    month_end + 1))

    df_cols = df.columns.values.tolist()
    group_cols = [col for col in df_cols if col.startswith('group')]
    weight_cols = [col for col in df_cols if col.startswith('weight')]
    for col in group_cols:
        df[col] = f.make_group_lists(df, col)
    df['grp_tup'] = f.make_lists_from_columns(df, group_cols,
                                              remove_zero_values=True,
                                              as_tuples=True)
    df['wgt_tup'] = f.make_lists_from_columns(df,
                                              weight_cols,
                                              remove_zero_values=True,
                                              as_tuples=True)
    df = df[['basic_job', 'grp_tup', 'wgt_tup',
             'cap', 'month_start', 'month_end']].copy()
    cols = [col for col in df if col != 'basic_job']
    comb = f.make_lists_from_columns(df, cols)
    df = pd.DataFrame({'job': df.basic_job, 'data': comb})
    settings['count_ratio_dict'] = f.make_dict_from_columns(df, 'job', 'data')

    # ## p_dict, p_dict_verbose

    df = xl['proposal_dictionary']
    df.short_descr = df.short_descr.astype(str)
    settings['p_dict'] = f.make_dict_from_columns(df,
                                                  'proposal',
                                                  'short_descr')
    settings['p_dict_verbose'] = f.make_dict_from_columns(df, 'proposal',
                                                          'long_descr')

    if settings['enhanced_jobs']:
        jd = settings['jd']
        sg_rights = settings['sg_rights']
        # ratio_cond = settings['ratio_cond']
        count_dict = settings['count_ratio_dict']
        ratio_dict = settings['ratio_dict']

        dist_sg = settings['sg_dist']
        dist_ratio = settings['ratio_dist']
        dist_count = settings['count_dist']

        sg_rights, count_dict, ratio_dict = cnv(job_dict=jd,
                                                sg_list=sg_rights,
                                                # ratio_list=ratio_cond,
                                                count_ratio_dict=count_dict,
                                                ratio_dict=ratio_dict,
                                                dist_sg=dist_sg,
                                                dist_ratio=dist_ratio,
                                                dist_count_ratio=dist_count)

        settings['sg_rights'] = sg_rights
        # settings['ratio_cond'] = ratio_cond
        settings['count_ratio_dict'] = count_dict
        settings['ratio_dict'] = ratio_dict

    # ///////////////////////////////////////////////////////////////////

    # COLOR DICTIONARY~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    color_dict = mcl(num_of_colors=settings['num_of_job_levels'] + 1,
                     return_dict=True)

    if settings['enhanced_jobs']:
        df = xl['enhanced_job_colors']
    else:
        df = xl['basic_job_colors']

    job_colors = f.make_lists_from_columns(df, ['red', 'green',
                                                'blue', 'alpha'])

    color_dict['job_colors'] = job_colors

    # ## eg_colors, lin_reg_colors, lin_reg_colors2, mean_colors

    short_colors = xl['eg_colors']
    short_cols = [col for col in list(short_colors) if col != 'eg']
    short_colors = xl['eg_colors'][short_cols]
    for col in list(short_colors):
        color_dict[col] = list(short_colors[col])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ATTRIBUTE DICTIONARY >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    df = xl['attribute_dict']
    attribute_dict = dict(zip(df.col_name, df.col_description))
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # OLD CODE STARTS HERE:  (making pickle files...) *********************
    start_date = pd.to_datetime(settings['starting_date'])

    # MASTER FILE:
    master = pd.read_excel('excel/' + case + '/master.xlsx')

    master.set_index('empkey', drop=False, inplace=True)

    master['retdate'] = master['dob'] + \
        pd.DateOffset(years=settings['init_ret_age_years']) + \
        pd.DateOffset(months=settings['init_ret_age_months'])
    # calculate future retirement age increase(s)
    if settings['ret_age_increase']:
        ret_incr_dict = settings['ret_incr_dict']
        for date, add_months in ret_incr_dict.items():
            master.loc[master.retdate > pd.to_datetime(date) +
                       pd.offsets.MonthEnd(-1), 'retdate'] = \
                master.retdate + pd.DateOffset(months=add_months)

    # only include employees who retire during or after the starting_month
    # (remove employees who retire prior to analysis period)
    master = master[master.retdate >= start_date -
                    pd.DateOffset(months=1) +
                    pd.DateOffset(days=1)]

    master.to_pickle('dill/master.pkl')

    # ACTIVE EACH MONTH (no consideration for job changes or recall, only
    # calculated on retirements of active employees as of start date)
    emps_to_calc = master[master.line == 1].copy()
    cmonths = f.career_months_df_in(emps_to_calc, settings['starting_date'])

    # LIST ORDER PROPOSALS
    # Read the list ordering proposals from an Excel workbook, add an index
    # column ('idx'), and store each proposal as a dataframe in a pickled file.
    # The proposals are contained on separate worksheets.
    # The routine below will loop through the worksheets.
    # The worksheet tab names are important for the function.
    # The pickle files will be named like the workbook sheet names.

    xl = pd.ExcelFile('excel/' + case + '/proposals.xlsx')

    sheets = xl.sheet_names
    # make dataframe containing proposal names and store it
    # (will be utilized by load_datasets function)
    sheets_df = pd.DataFrame(sheets, columns=['proposals'])
    sheets_df.to_pickle('dill/proposal_names.pkl')

    for ws in sheets:
        try:
            df = xl.parse(ws)[['empkey']]
            df.set_index('empkey', inplace=True)
            df['idx'] = np.arange(len(df)).astype(int) + 1
            df.to_pickle('dill/p_' + ws + '.pkl')
        except:
            print('proposal worksheet', ws, 'skipped during processing')
            continue

    # LAST MONTH
    # percent of month for all employee retirement dates.
    # Used for retirement month pay.

    df_dates = master[['retdate']].copy()
    df_dates['day_of_month'] = df_dates.retdate.dt.day
    df_dates['days_in_month'] = (df_dates.retdate +
                                 pd.offsets.MonthEnd(0)).dt.day
    df_dates['last_pay'] = df_dates.day_of_month / df_dates.days_in_month

    df_dates.set_index('retdate', inplace=True)
    df_dates = df_dates[['last_pay']]
    df_dates.sort_index(inplace=True)
    df_dates = df_dates[~df_dates.index.duplicated()]
    df_dates.to_pickle('dill/last_month.pkl')
    # ********************************************************************

    # JOB TABLES AND RELATED DICTIONARY___________________________________
    # create job tables (standalone and integrated), store as dictionary
    # (also job changes and job counts input arrays)

    # JOB_ASSIGN_FILTER_TABLE 1
    master_copy = master[['retdate', 'line', 'fur']].copy()
    # only active employees...
    df_actives = master_copy[master_copy.line == 1]
    # only furloughees...
    df_fur = master_copy[master_copy.fur == 1]
    cmonths = f.career_months_df_in(df_actives, settings['starting_date'])
    cmonths_fur = f.career_months_df_in(df_fur, settings['starting_date'])
    active_each_month = f.count_per_month(cmonths)
    fur_left_each_month = f.count_per_month(cmonths_fur)
    num_of_months = active_each_month.size

    num_of_job_levels = settings['num_of_job_levels']

    if settings['enhanced_jobs']:
        # use job dictionary(jd) from settings dictionary for conversion
        eg_counts, j_changes = f.convert_to_enhanced(settings['eg_counts'],
                                                     settings['j_changes'],
                                                     settings['jd'])
    else:
        eg_counts = settings['eg_counts']
        j_changes = settings['j_changes']

    # compute job counts array
    jcnts_arr = f.make_jcnts(eg_counts)

    s_table = f.job_gain_loss_table(num_of_months,
                                    num_of_job_levels,
                                    jcnts_arr,
                                    j_changes,
                                    standalone=True)

    table = f.job_gain_loss_table(num_of_months,
                                  num_of_job_levels,
                                  jcnts_arr,
                                  j_changes,
                                  standalone=False)

    # JOB_ASSIGN_FILTER_TABLE 2
    # this array will contain the number of originally furloughed employees
    # who remain under the retirement age
    fur_arr = np.zeros(num_of_months)
    np.put(fur_arr, np.arange(fur_left_each_month.size), fur_left_each_month)

    # this array will hold the cumulative furlough recall counts
    recall_arr = np.zeros(num_of_months)
    # loop through each recall schedule and make an array of of cumulative
    # recall counts
    for recall in settings['recalls']:
        recall_add = np.zeros(num_of_months)
        np.put(recall_add, np.arange(recall[2], recall[3]), recall[0])
        np.cumsum(recall_add, out=recall_add)
        # add this recall cumsum to main recall_arr (for each recall schedule)
        recall_arr = recall_arr + recall_add

    # limit each months cumulative recall count if monthly count of remaining
    # furloughed employees is less
    additive_arr = np.minimum(fur_arr, recall_arr)

    # add 2 zero columns in front of job count table
    zero_table = f.add_zero_col(f.add_zero_col(table[0]))

    # create accumulative table of job counts, left to right for comparison
    accum_table = np.add.accumulate(zero_table, axis=1)

    # create employee count limit array to compare with cumulative job counts
    if settings['recall']:
        limit_arr = (active_each_month + additive_arr).astype(int)
    else:
        limit_arr = active_each_month.astype(int)

    limit_arr = limit_arr[:, None]

    # perform a truth test on accum_table, False results will cause job
    # loop(s) for a month to be skipped with the assign_standalone_job_changes
    # function
    loop_check = np.less_equal(accum_table, limit_arr)

    table_dict = {'s_table': s_table,
                  'table': table,
                  'j_changes': j_changes,
                  'jcnts_arr': jcnts_arr,
                  'loop_check': loop_check}

    # ___________________________________________________________________

    # SQUEEZE_VALS ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # initial values for editor tool widgets.
    # The values stored within this file will be replaced and
    # updated by the editor tool when it is utilized.
    rows = len(master)
    low = int(.2 * rows)
    high = int(.8 * rows)

    init_editor_vals = pd.DataFrame([['<<  d', '2', 'ret_mark', 'spcnt', 'log',
                                    False, '==', '1', high, False, True,
                                    low, 100, '>=', '0']],
                                    columns=['drop_dir_val', 'drop_eg_val',
                                             'drop_filter', 'drop_msr',
                                             'drop_sq_val', 'fit_val',
                                             'drop_opr',
                                             'int_sel', 'junior', 'mean_val',
                                             'scat_val', 'senior',
                                             'slide_fac_val',
                                             'mnum_opr', 'int_mnum'],
                                    index=['value'])
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # WRITE DICTIONARIES TO DISC ==========================================
    init_editor_vals.to_pickle('dill/squeeze_vals.pkl')

    with open('dill/dict_settings.pkl', 'wb') as handle:
        pickle.dump(settings,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    with open('dill/dict_color.pkl', 'wb') as handle:
        pickle.dump(color_dict,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    with open('dill/dict_attr.pkl', 'wb') as handle:
        pickle.dump(attribute_dict,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    with open('dill/dict_job_tables.pkl', 'wb') as handle:
        pickle.dump(table_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
