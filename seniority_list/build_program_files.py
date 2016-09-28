#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Create the necessary support files from the input Excel files
for the program operation.  The Excel files are read from the folder
named for the case within the excel folder.

####################################################
Excel files

from proposals.xlsx:
    <worksheet name>.pkl for each proposal

from master.xlsx:
    master.pkl,
    fur.pkl,
    sg.pkl,
    last_month.pkl,
    active_each_month.pkl

from pay_tables.xlsx:
    pay_table_with_rsv_with_fur.pkl,
    pay_table_no_rsv_with_fur.pkl,
    idx_pay_table_with_rsv_with_fur.pkl,
    idx_pay_table_no_rsv_with_fur.pkl

#####################################################

initialized with this script:
    slider_vals.pkl*,

created by editor tool (when run):
     ds_edit.pkl,
     slider_vals.pkl*,
     squeeze_vals.pkl,
     new_order.pkl

*persistent values stored when editor tool is used

example usage to run this script from the jupyter notebook:
    %run build_program_files
'''
import pandas as pd
import numpy as np

import functions as f
import config as cf

case = cf.case_study

try:
    # check to see if file exists and get value if it does
    case_dill_value = pd.read_pickle('dill/case_dill.pkl').case.value
except:
    case_dill_value = 'empty_placeholder'

if case_dill_value == case:
    # if stored value is same as case study name, pass and keep dill files
    pass
else:
    # if the case name is different, time to start over.  Delete dill files
    # create new case_dill.pkl file
    f.clear_dill_files()
    case_dill = pd.DataFrame({'case': cf.case_study}, index=['value'])
    case_dill.to_pickle('dill/case_dill.pkl')

start_date = pd.to_datetime(cf.starting_date)

# MASTER FILE:
master = pd.read_excel('excel/' + case + '/master.xlsx')

master.set_index('empkey', drop=False, inplace=True)

retage = cf.ret_age
master['retdate'] = master['dob'] + \
    pd.DateOffset(years=cf.init_ret_age_years) + \
    pd.DateOffset(months=cf.init_ret_age_months)
# calculate future retirement age increase(s)
if cf.ret_age_increase:
    ret_incr_dict = cf.ret_incr_dict
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
cmonths = f.career_months_df_in(emps_to_calc)
nonret_each_month = f.count_per_month(cmonths)
# code below subject to removal pending test period completion
# actives = pd.DataFrame(nonret_each_month, columns=['count'])
# actives.to_pickle('dill/active_each_month.pkl')

# LIST ORDER PROPOSALS
# Read the list ordering proposals from an Excel workbook, add an index
# column ('idx'), and store each proposal as a dataframe in a pickled file.
# The proposals are contained on separate worksheets.
# The routine below will loop through the worksheets.
# The worksheet tab names are important for the function.
# The pickle files will be named like the workbook sheet names.

xl = pd.ExcelFile('excel/' + case + '/proposals.xlsx')

sheets = xl.sheet_names
for ws in sheets:
    try:
        df = xl.parse(ws)
        df.set_index('empkey', inplace=True)
        df['idx'] = np.arange(len(df)).astype(int) + 1
        df.to_pickle('dill/p_' + ws + '.pkl')
    except:
        continue

# LAST MONTH
# percent of month for all employee retirement dates.
# Used for retirement month pay.

df_dates = master[['retdate']].copy()
df_dates['day_of_month'] = df_dates.retdate.dt.day
df_dates['days_in_month'] = (df_dates.retdate + pd.offsets.MonthEnd(0)).dt.day
df_dates['last_pay'] = df_dates.day_of_month / df_dates.days_in_month

df_dates.set_index('retdate', inplace=True)
df_dates = df_dates[['last_pay']]
df_dates.sort_index(inplace=True)
df_dates = df_dates[~df_dates.index.duplicated()]
df_dates.to_pickle('dill/last_month.pkl')

# SQUEEZE_VALS
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

init_editor_vals.to_pickle('dill/squeeze_vals.pkl')
