# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import config as cf

# todo: allow this script to take variables to match proper excel tables for number of job levels
# add block and reserve hours/month option
# add international override pay option
pay_table = pd.read_excel('excel/pay_tables.xlsx', sheetname = 'with_reserve_with_fur')
pay_hours = pd.read_excel('excel/pay_tables.xlsx', sheetname = 'block_and_reserve_hours', index_col = 'jnum')

pay_melt = pd.melt(pay_table, id_vars = ['year','jnum'], var_name='scale', value_name='rate')
pay_melt.rate = np.round(pay_melt.rate, 2)

pay_melt = pd.merge(pay_melt, pay_hours, right_index = True, left_on = ['jnum'])
pay_melt.sort_index(inplace = True)

pay_melt['monthly'] = pay_melt.rate * pay_melt.hours

pay_melt.drop(['rate','hours'],inplace=True,axis=1)
pay_melt.set_index(['scale', 'year','jnum'], inplace = True)

# save 'traditional' multi-index monthly pay table to file:
if cf.save_to_pickle:
	pay_melt.to_pickle('dill/pay_table_with_reserve.pkl')

# ### cells below convert pay table multi-index to 
# single column combined index which allows for much faster data align merge...

pay_melt.reset_index(drop = False, inplace = True)

pay_melt['ptindex'] = (pay_melt.year * 100000 + pay_melt.scale * 100 + pay_melt.jnum)

pay_melt.drop(['scale','year','jnum'], axis = 1, inplace = True)
pay_melt.sort_values('ptindex', inplace = True)
pay_melt.set_index('ptindex', drop = True, inplace = True)

# write to file
if cf.save_to_pickle:
	pay_melt.to_pickle('dill/indexed_pay_table.pkl')





