# -*- coding: utf-8 -*-

'''orders a master dataframe containing all employees according to a
proposed list dataframe containing only active employees, "attaching" the
inactive employees to active employees within the same employee group.  The
inactives may be attached to same group active employees just senior or
just junior to them within the overall combined list'''

import pandas as pd
import numpy as np

from sys import argv

script, input_order_df, fill_style = argv

pre, suf = 'dill/', '.pkl'

master_name = 'master'
output_name = 'final'

master_path_string = (pre + master_name + suf)
order_path_string = (pre + input_order_df + suf)
final_path_string = (pre + output_name + suf)

df_master = pd.read_pickle(master_path_string)
df_order = pd.read_pickle(order_path_string)

joined = df_master.join(df_order, how='outer')

eg_set = np.unique(joined.eg)

final = pd.DataFrame()

for eg in eg_set:

    eg_df = joined[joined.eg == eg].copy()
    eg_df.sort_values('eg_order', inplace=True)

    if fill_style == 'ffill':
        eg_df.iloc[0, eg_df.columns.get_loc('idx')] = eg_df.idx.min()
        eg_df.idx = eg_df.idx.fillna(method='ffill').astype(int)

    if fill_style == 'bfill':
        eg_df.iloc[-1, eg_df.columns.get_loc('idx')] = eg_df.idx.max()
        eg_df.idx = eg_df.idx.fillna(method='bfill').astype(int)

    final = pd.concat([final, eg_df])

final = final.sort_values(['idx', 'eg_order'])
final['snum'] = np.arange(len(final)).astype(int) + 1

final.to_pickle('dill/final.pkl')

final.set_index('snum', drop=True, inplace=True)

writer = pd.ExcelWriter('excel/final.xlsx',
                        engine='xlsxwriter',
                        datetime_format='yyyy-mm-dd',
                        date_format='yyyy-mm-dd')

final.to_excel(writer, sheet_name='final')
writer.save()
