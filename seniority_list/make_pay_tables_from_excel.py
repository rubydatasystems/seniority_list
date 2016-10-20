#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Create the indexed compensation dataframes from the pay_tables.xlsx
Excel input file.  Also create an excel file, pay_table_data.xlsx which
displays the compensation data being used by the program data model
in a tabular (spreadsheet) form.

"Indexed" means that the index of the dataframe(s) contains a unique value
representing the year, longevity step, and job level.  The only column
("monthly") contains the corresponding monthly compensation value.

The Excel file is read from the case-specific named folder located
within the excel folder.

from pay_tables.xlsx this script generates:

    In 'dill' folder:
        pay_table_basic.pkl
        pay_table_enhanced.pkl
            (if 'enhanced_jobs' = True in the config.py file)
    In 'reports' folder (within case-specific named folder):
        pay_table_data.xlsx
    '''

import pandas as pd
import numpy as np
import os
from collections import OrderedDict as od
import config as cf


case = cf.case_study

enhanced_jobs = cf.enhanced_jobs

# inputs to determine global sorting master year and
# longevity...function parameters
# include second set of parameters for enhanced model and set to None
# check for not None to alter for enhanced sort...
year = cf.pay_table_year_sort
longevity = cf.pay_table_longevity_sort

# instantiate dict values to None
basic_compen = None
full_mth_compen = None
part_mth_compen = None
job_key_enhan = None
job_key_basic = None
basic = None
enhanced = None
job_dict_df = None

# read pay table data from excel file
pay_rates = pd.read_excel('excel/sample3/pay_tables.xlsx', sheetname='rates')
# read monthly pay hours per job level and job description from excel file
pay_hours = pd.read_excel('excel/sample3/pay_tables.xlsx', sheetname='hours')

# numpy unique returns a SORTED array of unique elements
contract_years = np.unique(pay_rates.year)

# extract integer column names (represents years of pay longevity)
longevity_cols = []
for col in list(pay_rates.columns):
    try:
        int(col)
        longevity_cols.append(col)
    except:
        pass

table_cols = ['year', 'jnum']
table_cols.extend(longevity_cols)

basic = pd.merge(pay_rates, pay_hours)
print(cf.enhanced_jobs)
if enhanced_jobs:
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
# In other words, the job level order of the entire pay dataframe
# will match the selected year and pay longevity order, even if certain year
# and pay level compensation amounts are not in descending order.
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

if enhanced_jobs:
    # calculate monthly compensation for each job level and pay longevity
    enhanced_full[longevity_cols] = enhanced_full[longevity_cols]\
        .multiply(enhanced_full['full_hours'], axis="index")

    enhanced_part[longevity_cols] = enhanced_part[longevity_cols]\
        .multiply(enhanced_part['part_hours'], axis="index")

    # ENHANCED TABLE SUFIXES, COLUMNS, JNUMS(ENHANCED_PART)

    # make enhanced_part (fewer hours per position per month) jnums begin
    # with maximum enhanced_full jnum + 1 and increment upwards
    enhanced_part.jnum = enhanced_part.jnum + enhanced_part.jnum.max()

    # sort by year and job level and only keep columns: 'year', 'jnum',
    # and all year longevity (integer) columns

    full_mth_compen = enhanced_full.sort_values(['year', 'jnum'])[table_cols]\
        .set_index('year', drop=True)
    part_mth_compen = enhanced_part.sort_values(['year', 'jnum'])[table_cols]\
        .set_index('year', drop=True)

    # add appropriate suffixes to jobstr columns for full
    # and part enhanced tables
    full_suf = cf.enhanced_jobs_full_suffix
    part_suf = cf.enhanced_jobs_part_suffix
    enhanced_full.jobstr = enhanced_full.jobstr.astype(str) + full_suf
    enhanced_part.jobstr = enhanced_part.jobstr.astype(str) + part_suf

    # CONCATENATE the full and part(-time) enhanced jobs dataframes
    enhanced = pd.concat([enhanced_full, enhanced_part])

    enhan_fur_rows = fur_rows.copy()
    enhan_fur_rows.jnum = enhanced.jnum.max() + 1
    enhan_fur_rows.year = contract_years
    enhan_fur_rows.jobstr = 'FUR'

    # CONCATENATE the furlough pay data to the basic and enhanced pay data
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

    dict_data = (('base', jobs.astype(int)),
                 ('full', idx_list1),
                 ('part', idx_list2),
                 ('jobstr', job_key_basic.jobstr[:int(j_cnt)]))
    # use of ordered dict preserves column order
    job_dict_df = pd.DataFrame(data=od(dict_data)).set_index('base', drop=True)

    # make a dataframe to save the job level hierarchy

    job_key_enhan.set_index('order', drop=True, inplace=True)
    job_key_enhan.rename(columns={'jnum': 'concat_order'}, inplace=True)
    order_enhan = order_enhan.reset_index()[['jnum', 'order']]
    enhanced = pd.merge(enhanced, order_enhan).sort_values(['year', 'order'])\
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
dict_items = (('basic (no sort)', basic_compen),
              ('enhanced full (no sort)', full_mth_compen),
              ('enhanced part (no sort)', part_mth_compen),
              ('basic ordered', basic),
              ('enhanced ordered', enhanced),
              ('basic job order', job_key_basic),
              ('enhanced job order', job_key_enhan),
              ('job dict (jd) helper', job_dict_df))

ws_dict = od(dict_items)

for key, value in ws_dict.items():
    try:
        value.to_excel(writer, key)
    except:
        pass

writer.save()
