#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import config as cf

table_sheets = ['basic', 'enhanced']
hours_sheets = ['basic_hours', 'enhanced_hours']

save_multi_index_version = False
case = cf.case_study

wb = 'excel/' + case + '/pay_tables.xlsx'

for i in np.arange(len(table_sheets)):
    # add international override pay option
    pay_table = pd.read_excel(wb,
                              sheetname=table_sheets[i])
    pay_hours = pd.read_excel(wb,
                              sheetname=hours_sheets[i],
                              index_col='jnum')

    pay_melt = pd.melt(pay_table,
                       id_vars=['year', 'jnum'],
                       var_name='scale',
                       value_name='rate')

    # pay_melt.rate = np.round(pay_melt.rate, 2)

    pay_melt = pd.merge(pay_melt, pay_hours,
                        right_index=True, left_on=['jnum'])
    pay_melt.sort_index(inplace=True)

    pay_melt['monthly'] = pay_melt.rate * pay_melt.hours

    pay_melt.drop(['rate', 'hours'], inplace=True, axis=1)

    pay_melt.set_index(['scale', 'year', 'jnum'], inplace=True)

    # save 'traditional' multi-index monthly pay table to file:
    if save_multi_index_version:
        pay_melt.to_pickle('dill/pay_table_' + table_sheets[i] + '.pkl')

    # cells below convert pay table multi-index to
    # single column combined index which allows for much
    # faster data align merge...

    pay_melt.reset_index(drop=False, inplace=True)

    pay_melt['ptindex'] = (pay_melt.year *
                           100000 + pay_melt.scale * 100 +
                           pay_melt.jnum)

    pay_melt.drop(['scale', 'year', 'jnum'], axis=1, inplace=True)
    pay_melt.sort_values('ptindex', inplace=True)
    pay_melt.set_index('ptindex', drop=True, inplace=True)

    # write to file
    if cf.save_to_pickle:
        pay_melt.to_pickle('dill/pay_table_' +
                           table_sheets[i] + '.pkl')
