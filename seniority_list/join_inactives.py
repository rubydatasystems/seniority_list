#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Merge and order a master dataframe containing all employees (including
inactive employees), with a proposed list dataframe which may contain all
employees or only active (and optionally furloughed) employees.

Writes the result list/dataframe to a pickle file (within 'dill' folder) and
an Excel file (within the case-specific folder located in
the 'reports' folder).

arguments (all strings with no quotes)
    script
        'join inactives'
    master_name
        name of the master file, normally 'master'
    proposed_order_df
        the name of the proposal to be used as the final ordering from the
        dill folder, without the "p_" prefix and ".pkl" file extension.
        A file named "p_p1.pkl" would have a proposed_order_df input name
        of "p1"
    output_name
        name to assign to output, normally "final".  A prefix equal to the
        case_study name and a ".pkl" file extension will be added.  If the
        output_name input is "final", the output file will be named
        "<case_study>_final.pkl"
    fill_style
        "ffill" or "bfill"

        If the proposed list ordering does not contain the inactive employees,
        the 'fill_style' argument determines where the inactive employees will
        be placed within the combined list relative to their same employee
        group active cohorts, just senior to the closest junior cohort or
        just junior to closest senior cohort.

            "ffill" - inactives attached to just *senior* same-group cohort

            "bfill" - inactives attached to just *junior* same-group cohort

example jupyter notebook usage:
    %run join_inactives.py master p1 final bfill
'''

import pandas as pd
import numpy as np

import os
from sys import argv

script, master_name, proposed_order_df, output_name, fill_style = argv

try:
    case = pd.read_pickle('dill/case_dill.pkl').case.value
except:
    print('case variable not found')

dill_pre, pkl_suf = 'dill/', '.pkl'

# name of this case-specific folder within 'reports' folder
file_path = 'reports/' + case + '/'
# create the folder if it doesn't already exist
os.makedirs(file_path, exist_ok=True)

master_path_string = (dill_pre + master_name + pkl_suf)
order_path_string = (dill_pre + 'p_' + proposed_order_df + pkl_suf)

excel_file_name = case + '_' + output_name + '.xlsx'
write_xl_path = (file_path + excel_file_name)

df_master = pd.read_pickle(master_path_string)
df_order = pd.read_pickle(order_path_string)

joined = df_master.join(df_order, how='outer')

eg_set = pd.unique(joined.eg)

final = pd.DataFrame()

for eg in eg_set:

    eg_df = joined[joined.eg == eg].copy()
    eg_df.sort_values('eg_order', inplace=True)

    if fill_style == 'ffill':
        eg_df.iloc[0, eg_df.columns.get_loc('idx')] = eg_df.idx.min()
        eg_df.idx.fillna(method='ffill', inplace=True)

    if fill_style == 'bfill':
        eg_df.iloc[-1, eg_df.columns.get_loc('idx')] = eg_df.idx.max()
        eg_df.idx.fillna(method='bfill', inplace=True)

    final = pd.concat([final, eg_df])

final = final.sort_values(['idx', 'eg_order'])
final['snum'] = np.arange(len(final)).astype(int) + 1
final.pop('idx')

final.to_pickle(dill_pre + case + '_' + output_name + pkl_suf)

final.set_index('snum', drop=True, inplace=True)

writer = pd.ExcelWriter(write_xl_path,
                        engine='xlsxwriter',
                        datetime_format='yyyy-mm-dd',
                        date_format='yyyy-mm-dd')

final.to_excel(writer, sheet_name='final')
writer.save()
