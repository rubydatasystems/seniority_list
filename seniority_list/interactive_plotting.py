#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
.. module:: interactive_plotting

   :synopsis: The bokeh module contains interactive plotting functions.

.. moduleauthor:: Bob Davison <rubydatasystems@fastmail.net>

'''

from bokeh.plotting import figure, ColumnDataSource
# from bokeh.models import (HoverTool, BoxZoomTool, WheelZoomTool, ResetTool,
#                           PanTool, SaveTool, UndoTool, RedoTool)
from bokeh.models import NumeralTickFormatter, Range1d, Label
from bokeh.models.widgets import Slider, Button, Select
from bokeh.layouts import column, row, widgetbox
from bokeh.models.layouts import Spacer

import numpy as np
import pandas as pd

'''
TODO:
add stacked area for cat_order
test source.date update using groupby groups/precalculated ColumnDataSources
add size, alpha sliders
make tabs for right side controls
background color selection, alpha control
add datatable
add save underlying data (reports?)
add mark selected employees
add dataset selection
add diff comparison
add hover (with user selection)
add tools (crosshair, etc)
add dataset selection
add dataset group compare
add dataset employee compare
add ret_only
add other chart types
make this the only display??
add persist df
'''


def bk_basic_interactive(doc, df=None,
                         plot_height=700, plot_width=900,
                         dot_size=5):
    '''run a basic interactive chart as a server app - powered by the bokeh
    plotting library.  Run the app in the jupyter notebook as follows:

    .. code:: python

        from functools import partial
        import pandas as pd

        import interactive_plotting as ip

        from bokeh.io import show, output_notebook

        from bokeh.application.handlers import FunctionHandler
        from bokeh.application import Application

        output_notebook()

        proposal = 'p1'
        df = pd.read_pickle('dill/ds_' + proposal + '.pkl')

        handler = FunctionHandler(partial(ip.bk_basic_interactive, df=df))

        app = Application(handler)
        show(app)

    inputs
        doc (required input)
            do not change this input

        df (dataframe)
            calculated dataset input, this is a required input

        plot_height (integer)
            height of plot in pixels

        plot_width (integer)
            width of plot in pixels

    Add plot_height and/or plot_width parameters as kwargs within the partial
    method:

    .. code:: python

        handler = FunctionHandler(partial(ip.bk_basic_interactive,
                                          df=df,
                                          plot_height=450,
                                          plot_width=625))

    Note: the "df" argument is not optional, a valid dataset variable must
    be assigned.
    '''

    class CallbackID():

        def __init__(self, identifier):
            self.identifier = identifier

    max_month = df['mnum'].max()
    # set up color column
    egs = df['eg'].values
    sdict = pd.read_pickle('dill/dict_settings.pkl')
    cdict = pd.read_pickle('dill/dict_color.pkl')
    eg_cdict = cdict['eg_color_dict']
    clr = np.empty(len(df), dtype='object')
    for eg in eg_cdict.keys():
        np.put(clr, np.where(egs == eg)[0], eg_cdict[eg])
    df['c'] = clr
    df['a'] = .7
    df['s'] = dot_size

    # date list for animation label background
    date_list = list(pd.date_range(start=sdict['starting_date'],
                                   periods=max_month, freq='M'))
    date_list = [x.strftime('%Y %b') for x in date_list]
    slider_height = plot_height - 200

    # create empty data source template
    source = ColumnDataSource(data=dict(x=[], y=[], c=[], s=[], a=[]))

    slider_month = Slider(start=0, end=max_month,
                          value=0, step=1,
                          title='month',
                          height=slider_height,
                          width=15,
                          tooltips=False,
                          bar_color='#ffe6cc',
                          direction='rtl',
                          orientation='vertical',)

    display_attrs = ['age', 'jobp', 'cat_order', 'spcnt', 'lspcnt',
                     'jnum', 'mpay', 'cpay', 'snum', 'lnum',
                     'ylong', 'mlong', 'idx', 'retdate', 'ldate',
                     'doh', 's_lmonths', 'new_order']

    sel_x = Select(options=display_attrs,
                   value='age',
                   title='x axis attribute:',
                   width=115, height=45)
    sel_y = Select(options=display_attrs,
                   value='spcnt',
                   title='y axis attribute:',
                   width=115, height=45)

    label = Label(x=20, y=plot_height - 150,
                  x_units='screen', y_units='screen',
                  text='', text_alpha=.25,
                  text_color='#b3b3b3',
                  text_font_size='70pt')

    spacer1 = Spacer(height=plot_height, width=30)

    but_fwd = Button(label='FWD', width=60)
    but_back = Button(label='BACK', width=60)
    add_sub = widgetbox(but_fwd, but_back, height=50, width=30)

    def make_plot():
        this_df = get_df()
        xcol = sel_x.value
        ycol = sel_y.value
        source.data = dict(x=this_df[sel_x.value],
                           y=this_df[sel_y.value],
                           c=this_df['c'],
                           a=this_df['a'],
                           s=this_df['s'])

        non_invert = ['age', 'idx', 's_lmonths', 'mlong',
                      'ylong', 'cpay', 'mpay']
        if xcol in non_invert:
            xrng = Range1d(df[xcol].min(), df[xcol].max())
        else:
            xrng = Range1d(df[xcol].max(), df[xcol].min())

        if ycol in non_invert:
            yrng = Range1d(df[ycol].min(), df[ycol].max())
        else:
            yrng = Range1d(df[ycol].max(), df[ycol].min())

        p = figure(plot_width=plot_width,
                   plot_height=plot_height,
                   x_range=xrng,
                   y_range=yrng,
                   title='')

        p.circle(x='x', y='y', color='c', size='s', alpha='a',
                 line_color=None, source=source)

        pcnt_cols = ['spcnt', 'lspcnt']
        if xcol in pcnt_cols:
            p.x_range.end = -.001
            p.xaxis[0].formatter = NumeralTickFormatter(format="0.0%")
        if ycol in pcnt_cols:
            p.y_range.end = -.001
            p.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")

        if xcol in ['cat_order']:
            p.x_range.end = -50
        if ycol in ['cat_order']:
            p.y_range.end = -50

        if xcol in ['jobp', 'jnum']:
            p.x_range.end = .95
        if ycol in ['jobp', 'jnum']:
            p.y_range.end = .95

        p.xaxis.axis_label = sel_x.value
        p.yaxis.axis_label = sel_y.value
        p.add_layout(label)
        label.text = date_list[slider_month.value]

        return p

    def get_df():
        filter_df = df[df.mnum == slider_month.value][[sel_x.value,
                                                       sel_y.value,
                                                       'c', 's', 'a']]

        return filter_df

    def update_data(attr, old, new):
        this_df = get_df()

        source.data = dict(x=this_df[sel_x.value],
                           y=this_df[sel_y.value],
                           c=this_df['c'],
                           a=this_df['a'],
                           s=this_df['s'])

        label.text = date_list[new]

    controls = [sel_x, sel_y]
    wb_controls = [sel_x, sel_y, slider_month]

    for control in controls:
        control.on_change('value', lambda attr, old, new: insert_plot())

    slider_month.on_change('value', update_data)

    sizing_mode = 'fixed'

    inputs = widgetbox(*wb_controls, width=190, height=60,
                       sizing_mode=sizing_mode)

    def insert_plot():
        lo.children[0] = make_plot()

    def animate_update():
        mth = slider_month.value + 1
        if mth > max_month:
            mth = 0
        slider_month.value = mth

    def fwd():
        slider_val = slider_month.value
        if slider_val < max_month:
            slider_month.value = slider_val + 1

    def back():
        slider_val = slider_month.value
        if slider_val > 0:
            slider_month.value = slider_val - 1

    but_back.on_click(back)
    but_fwd.on_click(fwd)

    cb = CallbackID(None)

    def animate():
        if play_button.label == '► Play':
            play_button.label = '❚❚ Pause'
            cb.identifier = doc.add_periodic_callback(animate_update, 350)
        else:
            play_button.label = '► Play'
            doc.remove_periodic_callback(cb.identifier)

    def reset():
        slider_month.value = 0

    play_button = Button(label='► Play', width=60)
    play_button.on_click(animate)

    reset_button = Button(label='Reset', width=60)
    reset_button.on_click(reset)

    lo = row(make_plot(), spacer1, inputs, column(play_button,
                                                  reset_button,
                                                  add_sub))

    doc.add_root(lo)
