#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# seniority_list is an analytical tool used when seniority-based work
# groups merge. It brings modern data science to the area of labor
# integration, utilizing the powerful data analysis capabilities of Python
# scientific computing.

# Copyright (C) 2016-2017  Robert E. Davison, Ruby Data Systems Inc.
# Please direct inquires to: rubydatasystems@fastmail.net

# This program is free software: you can redistribute it and/or modiffy
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''bokeh_editor.py

EDITOR TOOL

requires bokeh 0.12.13+ - uses bokeh server

'''

import numpy as np
import pandas as pd
import os
import sys
import pickle
from functools import partial
from collections import OrderedDict as od
from types import SimpleNamespace as sn
import scipy.stats as st
from scipy.signal import savgol_filter as sf
from numpy.polynomial import Polynomial as poly

from bokeh.plotting import figure
from bokeh.layouts import column, row, layout
from bokeh.models import ColumnDataSource, DataRange1d, \
    Span, Panel, Tabs, Label, NumeralTickFormatter, \
    Jitter, DatetimeTickFormatter, HoverTool, CrosshairTool
from bokeh.models.layouts import Spacer
from bokeh.models.widgets import Slider, Button, Select, \
    RangeSlider, TextInput, CheckboxGroup
from bokeh.models.glyphs import Line
from bokeh.models.annotations import BoxAnnotation

import functions as f
from matplotlib_charting import filter_ds


class Data():

    def __init__(self, data=None):
        self.data = data

    def update_data(self, d):
        self.data = d


class PropOrder():

    def __init__(self, list_order=None, name=None):
        self.list_order = list_order
        self.name = name

    def update_order(self, new_order):
        self.list_order = new_order

    def update_name(self, new_name):
        self.name = new_name


class Kwargs():

    def __init__(self, kdict=None):
        self.kdict = kdict
        if self.kdict is None:
            self.kdict = {}

    def update(self, other_dict):
        self.kdict.update(other_dict)

    def add(self, key, value):
        self.kdict[key] = value

    def remove(self, key):
        self.kdict.pop(key)

    def clear(self):
        self.kdict.clear()


class CallbackID():

    def __init__(self, identifier):
        self.identifier = identifier


def editor(doc,
           poly_dim=15,
           ema_len=25,
           savgol_window=35,
           savgol_fit=1,
           animate_speed=350,
           plot_width=1100,
           plot_height=500,
           strip_eg_height=50,
           start_dot_size=4.75,
           max_dot_size=25,
           start_marker_alpha=.65,
           marker_edge_color=None,
           marker_edge_width=0.0):
    '''create the editor tool

    use the following code to run within the notebook:

    .. code:: python

        import editor_function as ef
        from functools import partial

        from bokeh.io import show, output_notebook
        from bokeh.application.handlers import FunctionHandler
        from bokeh.application import Application

        output_notebook()

        handler = FunctionHandler(partial(ef.editor,
                                  # optional kwargs,
                                  ))

        app = Application(handler)
        show(app)

    inputs
        doc (variable)
            a variable representing the bokeh document, do not modify
        poly_dim (integer)
            the order of the polynomial fit line
        ema_len (integer)
            the smoothing length to use when constructing the exponential
            moving average line
        savgol_window (positive odd integer)
            Savitzky-Golay filter window length
        savgol_fit (integer)
            The order of the polynomial used to fit the samples.
            This value must be less than the savgol_window value.
        animate_speed (integer)
            Number of milliseconds between each animated month display
        plot_width (integer)
            width of main and density charts in pixels
        plot_height (integer)
            height of main chart in pixels
        strip_eg_height (integer)
            height alloted for each employee group when constructing
            the density chart
        start_dot_size (float)
            initial scatter marker size for main chart
        max_dot_size (integer)
            maximum scatter marker size for the main chart display, set
            to size sliders
        start_marker_alpha (float)
            initial scatter marker alpha (transparency) for main chart
            display
        marker_edge_color (color value string or None)
            color of scatter marker edge color for main chart when
            marker edge width value is greater than zero
        marker_edge_width (float)
            width of scatter marker edge width when marker_edge_color is
            not None
    '''

    # ------START variable assignment------------------------------
    try:
        settings_dict = pd.read_pickle('dill/dict_settings.pkl')
        color_dict = pd.read_pickle('dill/dict_color.pkl')
    except OSError:
        print('dict_settings.pkl and/or dict_color.pkl not found ' +
              '\nperhaps run build_program_files.py?')

    # the editor dictionary contains values representing the current state
    # of widget values and other variables.
    # The values are stored as a pickled dictionary file between sessions.
    # The editor dictionary is converted to a SimpleNamespace object
    # for use within the routine.
    # This way dot notation and global access is provided.
    # sn is the alias for SimpleNamespace

    ed = sn(**pd.read_pickle('dill/editor_dict.pkl'))

    # grab proposal names for sel_base and sel_proposal dropdowns
    p_list = list(pd.read_pickle('dill/proposal_names.pkl').proposals.values)
    # limit proposal names to 10 characters to maintain layout integrity
    p_list = [x[:10] for x in p_list]
    # add hybrid if a hybrid dataset exists
    if os.path.exists('dill/ds_hybrid.pkl'):
        p_list.append('hybrid')
    # make a list for baseline selection (add standalone)
    base_p_list = [p for p in p_list if p != 'edit']
    base_p_list.append('standalone')
    # add edit to p_list
    if 'edit' not in p_list:
        p_list.append('edit')

    max_month = ed.num_of_months
    mth_str_list = list(np.arange(0, max_month).astype(str))
    # date list for animation label background
    date_list = list(pd.date_range(start=settings_dict['starting_date'],
                                   periods=max_month, freq='M'))
    date_list = [x.strftime('%Y %b') for x in date_list]

    # cover the possibility of rgba values in eg_color_dict values:
    eg_cdict = f.convert_to_hex(color_dict['eg_color_dict'])
    eg_list = list(eg_cdict.keys())
    # used for stripplot source (see callbacks update_scat_size_p2/alpha2)
    num_dots = ed.total_count
    str_eg_list = [str(eg) for eg in eg_list]

    # desc = Div(text=open(os.path.join(os.path.dirname(__file__),
    #                                   'description.html')).read(),
    #            width=800)

    # slider steps for marker size and alpha
    size_step = .25
    alpha_step = .025

    strip_height = len(eg_list) * strip_eg_height
    aux_slider_height = 160
    aux_slider_width = 22
    panel1_width = 460
    panel2_width = max(450, 210 + (2 * aux_slider_width * len(eg_list)))

    slider_edit_width = plot_width - 80

    all_colors = color_list()
    alphas = alpha_list()
    widths = line_widths()

    # layout variables
    controls_height = 220
    chart_sel_height = 140
    but_space_width = 50
    but_save_width = 260
    sel_height = 40
    sel_width = 90
    txt_height = 52
    main_but_width = 120
    toggle_but_width = 25
    toggle_space_width = 20
    toggle_center_width = 110
    but_height = 35

    # squeeze tab
    drop_dir_dict = {'u  >>': 'u', '<<  d': 'd'}
    incr_dir_dict = {'u  >>': -1, '<<  d': 1}

    # these items are referenced when datasets are created
    # baseline datasets are created and stored with the RUN_SCRIPTS notebook
    # edited datasets are created with the editor tool for analysis
    cond_dict = {'none': [],
                 'prex': ['prex'],
                 'count': ['count'],
                 'ratio': ['ratio'],
                 'pc': ['prex', 'count'],
                 'pr': ['prex', 'ratio'],
                 'cr': ['count', 'ratio'],
                 'pcr': ['prex', 'count', 'ratio']}

    pcnt_cols = ['spcnt', 'lspcnt']
    float_cols = ['jobp', 'mpay', 'cpay', 'ylong', 'mlong', 'age']
    date_cols = ['date', 'doh', 'ldate', 'retdate']
    no_invert = ['mnum', 'date', 'year', 'retdate', 'doh', 'ldate',
                 'scale', 's_lmonths', 'age', 'job_count', 'mlong',
                 'ylong', 'mpay', 'cpay']

    p1_tools = 'pan, box_zoom, wheel_zoom, reset, undo, redo, save'
    p2_tools = 'wheel_zoom, box_zoom, reset, save'

    # Select widget arguments
    sel_size_kwargs = {'width': sel_width, 'height': sel_height}

    # density tab
    aux_slider_kwargs = {'height': aux_slider_height,
                         'width': aux_slider_width,
                         'direction': 'rtl',
                         'orientation': 'vertical',
                         'tooltips': False,
                         'show_value': False}

    size_alpha_kwargs = {'width': 30,
                         'height': 20}

    # extra filters and display tabs
    opers = ['<', '<=', '==', '!=', '>=', '>']
    opers2 = opers + ['']

    # extra filters options
    attr_list = ['', 'cat_order', 'jobp', 'jnum', 'mnum', 'eg',
                 'date', 'ldate', 'doh', 'retdate', 'ylong', 'mlong',
                 'sg', 'age', 'scale', 's_lmonths',
                 'lnum', 'snum', 'mnum', 'rank_in_job',
                 'mpay', 'cpay']

    # add or remove keys and values here for hover selection generation
    hdict = {0: ('lname', '@lname'),
             1: ('empkey', '@empkey'),
             2: ('ldate', '@ldate{%F}'),
             3: ('retdate', '@retdate{%F}'),
             4: ('spcnt', '@spcnt{.000}'),
             5: ('ylong', '@ylong{0.00}'),
             6: ('age', '@age{0.0}')}

    # default string for tooltip formatting
    # the tuples from the dictionary above are added as appropriate for
    # proper hover names and value formatting further in the routine
    html_str = ('<div>' +
                '<span style=' +
                '"font-size: 13px; font-weight: bold; ' +
                'color: @c;">%s:</span>' +
                '<span style="font-size: 13px;">%s</span>' +
                '</div>')

    # display attribute options
    display_attrs = ['jobp', 'cat_order', 'spcnt', 'lspcnt',
                     'jnum', 'mpay', 'cpay', 'snum', 'lnum',
                     'ylong', 'mlong', 'age', 's_lmonths',
                     'ldate', 'doh']

    # size_alpha tab vars
    sl_size_dict = {}
    sl_alpha_dict = {}
    slider_list = []

    # plot_note label and calc_note label arguments
    note_kwargs = dict(x=40, y=40, x_units='screen',
                       y_units='screen',
                       border_line_color='black',
                       border_line_alpha=.5,
                       background_fill_alpha=1.0,
                       text_font_size='15pt',
                       visible=False)

    plot_kwargs = dict(text='..filtering data... ',
                       background_fill_color='#ffcc80',
                       **note_kwargs)

    calc_kwargs = dict(text='..calculating new dataset... ',
                       background_fill_color='#99ddff',
                       **note_kwargs)

    # ------END variable assignment---------------------------------

    # ------START widget declarations-------------------------------

    # squeeze tab
    sel_sqz_type = Select(options=['log', 'slide'],
                          value=ed.sel_sqz_type,
                          title='sqz type',
                          **sel_size_kwargs)

    sel_emp_grp = Select(options=str_eg_list,
                         value=ed.sel_emp_grp,
                         title='emp group',
                         **sel_size_kwargs)

    sel_sqz_dir = Select(options=['u  >>', '<<  d'],
                         value=ed.sel_sqz_dir,
                         title='sqz dir',
                         **sel_size_kwargs)

    slider_squeeze = Slider(start=1, end=400,
                            value=ed.slider_squeeze,
                            step=1,
                            title='squeeze',
                            width=450, height=40,
                            bar_color='#ffe6cc')

    but_0add = Button(label='<', width=toggle_but_width)
    but_0sub = Button(label='>', width=toggle_but_width)

    but_squeeze = Button(label='SQUEEZE', width=main_but_width,
                         height=but_height, button_type='success')

    but_1add = Button(label='<', width=toggle_but_width)
    but_1sub = Button(label='>', width=toggle_but_width)

    # extra filters tab
    sel_filt1 = Select(options=attr_list,
                       value=ed.sel_filt1,
                       title='Filter 1', width=115, height=sel_height)

    sel_filt2 = Select(options=attr_list,
                       value=ed.sel_filt2,
                       title='Filter 2', width=115, height=sel_height)

    sel_filt3 = Select(options=attr_list,
                       value=ed.sel_filt3,
                       title='Filter 3', width=115, height=sel_height)

    sel_oper1 = Select(options=opers2,
                       value=ed.sel_oper1,
                       title='Oper 1', **sel_size_kwargs)

    sel_oper2 = Select(options=opers2,
                       value=ed.sel_oper2,
                       title='Oper 2', **sel_size_kwargs)

    sel_oper3 = Select(options=opers2,
                       value=ed.sel_oper3,
                       title='Oper 3', **sel_size_kwargs)

    txt_input1 = TextInput(value=ed.txt_input1,
                           title='Val 1', height=txt_height)
    txt_input2 = TextInput(value=ed.txt_input2,
                           title='Val 2', height=txt_height)
    txt_input3 = TextInput(value=ed.txt_input3,
                           title='Val 3', height=txt_height)

    # animate tab
    slider_animate = Slider(start=0, end=max_month - 1,
                            value=int(ed.sel_mth_num),
                            step=1, title='Month',
                            width=350,
                            orientation='horizontal',
                            tooltips=False,
                            show_value=True,
                            bar_color='#a6a6a6')

    but_play = Button(label='► Play', width=90)
    but_reset = Button(label='Reset', width=90)

    # This commented section is on hold for future development...
    # chk_trails = CheckboxGroup(labels=['show_trails'],
    #                            active=ed.chk_trails,
    #                            height=35, width=130, inline=False)

    # trails_list = ['all']
    # trails_list.extend(mth_str_list)
    # sel_trails = Select(options=trails_list,
    #                     value=ed.sel_trails, title='trail_mths',
    #                     width=sel_width, height=sel_height)

    but_fwd = Button(label='FWD', width=90)
    but_back = Button(label='BACK', width=90)

    but_refresh = Button(label='refresh size_alpha',
                         width=120)

    label = Label(x=20, y=plot_height - 150,
                  x_units='screen', y_units='screen',
                  text='', text_alpha=.25,
                  text_color='#b3b3b3',
                  text_font_size='70pt')

    # proposal_save tab
    but_save_edit = Button(label='SAVE EDITED DATASET',
                           button_type='warning',
                           width=but_save_width)

    but_save_order = Button(label='SAVE EDITED ORDER to proposals.xlsx',
                            button_type='danger',
                            width=but_save_width)

    sel_base = Select(options=base_p_list,
                      value=ed.sel_base,
                      title='baseline:',
                      width=sel_width, height=sel_height + 5)

    condition_options = list(cond_dict.keys())
    sel_cond = Select(options=condition_options,
                      value=ed.sel_cond,
                      title='conditions:',
                      width=sel_width, height=sel_height + 5)

    sel_proposal = Select(options=p_list,
                          value=ed.sel_proposal,
                          title='proposal:',
                          **sel_size_kwargs)

    # center column
    sel_measure = Select(options=display_attrs,
                         value=ed.sel_measure,
                         title='display attr:',
                         width=sel_width, height=sel_height + 15)

    but_calc = Button(label='CALC', width=sel_width + 12,
                      height=but_height, button_type='primary')

    but_plot = Button(label='PLOT', width=sel_width + 12,
                      height=but_height, button_type='warning')

    # display tab
    chk_filter = CheckboxGroup(labels=['use extra filters', 'at_retire_only'],
                               active=ed.chk_filter,
                               height=35, width=130, inline=False)

    sel_mth_oper = Select(options=opers,
                          value=ed.sel_mth_oper,
                          title='month oper',
                          **sel_size_kwargs)

    sel_mth_num = Select(options=mth_str_list,
                         value=ed.sel_mth_num,
                         title='month num',
                         **sel_size_kwargs)

    chk_display = CheckboxGroup(labels=['scatter', 'poly_fit',
                                        'mean', 'savgol'],
                                active=ed.chk_display,
                                height=40,
                                width=70,
                                inline=False)

    sel_ytype = Select(options=['diff', 'abs'],
                       value=ed.sel_ytype,
                       title='ytype',
                       **sel_size_kwargs)

    sel_xtype = Select(options=['prop_s', 'prop_r',
                                'pcnt_s', 'pcnt_r'],
                       value=ed.sel_xtype,
                       title='xtype',
                       **sel_size_kwargs)

    # size_alpha tab:
    for eg in eg_list:
        sl_size_dict[eg] = Slider(start=.5,
                                  end=max_dot_size,
                                  value=start_dot_size,
                                  step=size_step, title='S',
                                  bar_color=eg_cdict[eg],
                                  **aux_slider_kwargs)

        sl_alpha_dict[eg] = Slider(start=0.0, end=1.0,
                                   value=start_marker_alpha,
                                   step=alpha_step, title='A',
                                   bar_color=eg_cdict[eg],
                                   **aux_slider_kwargs)

        slider_list.extend([sl_size_dict[eg], sl_alpha_dict[eg]])

    but_slider_reset = Button(label='Reset', width=50)

    but_slider_big = Button(label='S >', **size_alpha_kwargs)
    but_slider_sml = Button(label='< S', **size_alpha_kwargs)
    but_slider_aup = Button(label='A >', **size_alpha_kwargs)
    but_slider_adn = Button(label='< A', **size_alpha_kwargs)

    # grid_bg tab
    sel_bgc = Select(options=all_colors,
                     value=ed.sel_bgc,
                     title='chart / edit_fill',
                     width=115, height=sel_height)

    sel_gridc = Select(options=all_colors,
                       value=ed.sel_gridc,
                       title='grid / edit_line',
                       width=115, height=sel_height)

    sel_bgc_alpha = Select(options=alphas,
                           value=ed.sel_bgc_alpha,
                           title='alpha',
                           width=40, height=sel_height)

    sel_gridc_alpha = Select(options=alphas,
                             value=ed.sel_gridc_alpha,
                             title='alpha',
                             width=40, height=sel_height)

    but_reset_colors = Button(label='Reset', width=60)

    chk_minor_grid = CheckboxGroup(labels=['minor grid lines'],
                                   active=ed.chk_minor_grid)

    chk_color_apply = CheckboxGroup(labels=['chart bg/grid',
                                            'edit zone'],
                                    active=ed.chk_color_apply,
                                    height=50)

    sel_box_line_width = Select(options=widths,
                                value=ed.box_line_width,
                                title='edit_line_width',
                                width=60, height=sel_height)

    # hover tab
    chk_hover_on = CheckboxGroup(labels=['hover ON'],
                                 active=ed.chk_hover_on,
                                 width=150)

    # get column names from hdict (first value of each tuple)
    hover_labels = [val[0] for val in hdict.values()]

    chk_hover_sel = CheckboxGroup(labels=hover_labels,
                                  active=ed.chk_hover_sel,
                                  width=120)

    # density tab (stripplot):
    slider_strip_size = Slider(start=.05, end=15.0,
                               value=ed.p2_marker_size,
                               step=.05, title='S',
                               height=40, width=200,
                               tooltips=False,
                               show_value=True,
                               bar_color='#e6e6e6')

    slider_strip_alpha = Slider(start=.025, end=1.0,
                                value=ed.p2_marker_alpha,
                                step=.025, title='A',
                                height=40, width=200,
                                tooltips=False,
                                show_value=True,
                                bar_color='#e6e6e6')

    slider_edit_zone = RangeSlider(start=0.0, end=ed.ez_end,
                                   value=(float(ed.x_low), float(ed.x_high)),
                                   step=ed.ez_step,
                                   title='edit range values',
                                   width=slider_edit_width,
                                   bar_color='#a6a6a6', direction='rtl',
                                   show_value=True)

    plot_note = Label(**plot_kwargs)
    calc_note = Label(**calc_kwargs)

    # Spacer Widgets...................

    # display tab:
    spacer_top_disp = Spacer(width=200, height=45)
    spacer_disp_mth1 = Spacer(width=55)
    spacer_disp_mth2 = Spacer(width=35)
    spacer_disp_ax1 = Spacer(width=55)
    spacer_disp_ax2 = Spacer(width=35)

    # squeeze tab
    spacer_sqz_but1 = Spacer(width=but_space_width)
    spacer_sqz_but2 = Spacer(width=but_space_width)
    spacer_sqz_but3 = Spacer(width=but_space_width)

    spacer_toggle_1 = Spacer(width=toggle_space_width)
    spacer_toggle_center1 = Spacer(width=toggle_center_width)
    spacer_toggle_center2 = Spacer(width=toggle_center_width)
    spacer_toggle_2 = Spacer(width=toggle_space_width)

    # animate tab (commented for future use)
    # spacer_anim = Spacer(width=40, height=aux_slider_height)

    # proposal_save tab
    spacer_top_save1 = Spacer(width=but_save_width, height=85)
    spacer_middle_save = Spacer(width=50, height=aux_slider_height)

    # above sel_measure dropdown (center column)
    spacer_top_center_col = Spacer(height=80, width=sel_width)

    # size_alpha tab
    spacer_top_size_alpha = Spacer(width=50, height=50)
    spacer_size_buts = Spacer(width=30)
    spacer_alpha_buts = Spacer(width=30)

    # grid_bg tab
    spacer_linesbg_col = Spacer(width=75)
    spacer_linesbg_col2 = Spacer(width=5)
    spacer_top_color_apply = Spacer(width=70, height=40)
    spacer_linesbg_bottom = Spacer(width=75)

    # animate tab
    spacer_anim1 = Spacer(width=60, height=but_height)
    spacer_anim_refresh = Spacer(width=60, height=but_height)
    spacer_anim2 = Spacer(width=60, height=but_height)

    # layout column spacers
    spacer_controls1 = Spacer(width=50)
    spacer_controls2 = Spacer(width=50)

    # edit zone slider (left margin)
    spacer_edit = Spacer(width=40)

    # ------END widget declarations---------------------------------

    # ------START Class instantiations------------------------------
    proposal = PropOrder()
    diff_str = Data()
    filt_str = Data()

    skel = Data()
    ds_stand = Data()
    base_ds = Data()
    calc_ds = Data()
    idx_df = Data()
    filt_df = Data()
    strip_df = Data()
    reorder_df = Data()
    anim_df = Data()

    mgrps_gb = Data()

    filt_xax = Data()
    idx_xax = Data()

    alpha_filt_arr = Data()
    eg_filt_arr = Data()
    zero_filt_arr = Data()
    size_filt_arr = Data()

    tool_tips = Data()
    hover_tool = Data()
    crosshair_tool = Data()

    polys = Kwargs()
    means = Kwargs()
    savgols = Kwargs()
    src_dict = Kwargs()

    cb = CallbackID(None)

    # ------figures, sources, tool classes----------------------------

    p1 = figure(min_border_left=50, tools=p1_tools)
    p2 = figure(width=plot_width, height=strip_height,
                x_range=DataRange1d(flipped=True, range_padding=0.0),
                y_range=DataRange1d(flipped=True, range_padding=0.05),
                tools=p2_tools)

    source1 = ColumnDataSource(data=dict(a=[], c=[], s=[], x=[], y=[]))
    source2 = ColumnDataSource(data=dict(a=[], c=[], eg=[], s=[], x=[]))

    # --------------------------------------------------------------

    box_kwargs = dict(fill_alpha=float(ed.box_fill_alpha),
                      fill_color=ed.box_fill_color,
                      line_color=ed.box_line_color,
                      line_alpha=float(ed.box_line_alpha),
                      line_width=float(ed.box_line_width),
                      level='underlay',
                      )

    box1 = BoxAnnotation(**box_kwargs.copy())
    box2 = BoxAnnotation(**box_kwargs.copy())

    # ------polyfit, mean, and savgol smoothing line glyphs-------
    # dummy nan dict
    nan_dict = dict(x=np.full(1, np.nan), y=np.full(1, np.nan))
    # line glyphs arguments
    poly_kwargs = dict(x="x", y="y",
                       line_width=15, line_alpha=0.7)
    mean_kwargs = dict(x="x", y="y",
                       line_width=6, line_alpha=0.7)
    savgol_kwargs = dict(x="x", y="y",
                         line_width=8, line_alpha=0.7)

    for eg in eg_list:
        # ----make line glyphs------------------------
        polys.kdict['p' + str(eg)] = Line(line_color=eg_cdict[eg],
                                          **poly_kwargs)
        means.kdict['m' + str(eg)] = Line(line_color=eg_cdict[eg],
                                          **mean_kwargs)
        savgols.kdict['s' + str(eg)] = Line(line_color=eg_cdict[eg],
                                            **savgol_kwargs)

        # ----line glyphs data source instantiation----
        src_dict.kdict['sp' + str(eg)] = \
            ColumnDataSource(data=nan_dict.copy())
        src_dict.kdict['sm' + str(eg)] = \
            ColumnDataSource(data=nan_dict.copy())
        src_dict.kdict['ss' + str(eg)] = \
            ColumnDataSource(data=nan_dict.copy())

    # hover and crosshair tools
    hover_tool.data = HoverTool(formatters={'ldate': 'datetime',
                                            'retdate': 'datetime'},
                                show_arrow=False)
    hover_cols = Data()

    crosshair_tool.data = CrosshairTool(dimensions='both',
                                        line_alpha=.3,
                                        line_color='red',
                                        line_width=.75)

    # ------END Class instantiations------------------------------

    # ------START Callback functions------------------------------

    # squeeze source
    def sqz_type_change(attr, old, new):
        ed.sel_sqz_type = new

    def emp_group_change(attr, old, new):
        ed.sel_emp_grp = new

    def sqz_dir_change(attr, old, new):
        ed.sel_sqz_dir = new

    def update_squeeze(attr, old, new):
        ed.slider_squeeze = new

    # toggle line adjustment:
    def line1_add():
        low_slider = ed.x_low
        high_slider = ed.x_high
        if ed.sel_xtype in ['prop_s', 'prop_r']:
            high_slider += 1
        else:
            if high_slider < 1.0:
                high_slider += .001
        slider_edit_zone.value = (low_slider, high_slider)

    def line1_sub():
        low_slider = ed.x_low
        high_slider = ed.x_high
        if high_slider > low_slider:
            if ed.sel_xtype in ['prop_s', 'prop_r']:
                high_slider -= 1
            else:
                high_slider -= .001
            slider_edit_zone.value = (low_slider, high_slider)

    def line0_add():
        low_slider = ed.x_low
        high_slider = ed.x_high
        if low_slider < high_slider:
            if ed.sel_xtype in ['prop_s', 'prop_r']:
                low_slider += 1
            else:
                low_slider += .001
            slider_edit_zone.value = (low_slider, high_slider)

    def line0_sub():
        low_slider = ed.x_low
        high_slider = ed.x_high
        if ed.sel_xtype in ['prop_s', 'prop_r']:
            low_slider -= 1
        else:
            if low_slider > 0.0:
                low_slider -= .001
        slider_edit_zone.value = (low_slider, high_slider)

    def perform_squeeze():  # make new order for sripplot and/or skeleton

        if ed.sel_proposal != 'edit':
            sel_proposal.value = 'edit'

        squeeze_eg = int(ed.sel_emp_grp)

        ed.x_low = slider_edit_zone.value[0]
        ed.x_high = slider_edit_zone.value[1]

        low_val = f.cross_val(filt_xax.data, ed.x_low, idx_xax.data)
        high_val = f.cross_val(filt_xax.data, ed.x_high, idx_xax.data)

        if sel_sqz_type.value == 'log':
            direction = drop_dir_dict[ed.sel_sqz_dir]
            factor = slider_squeeze.value * .005
            squeezer = f.squeeze_logrithmic(reorder_df.data,
                                            squeeze_eg,
                                            low_val, high_val,
                                            direction=direction,
                                            put_segment=1,
                                            log_factor=factor)

        if sel_sqz_type.value == 'slide':
            incr_dir_correction = incr_dir_dict[ed.sel_sqz_dir]
            increment = slider_squeeze.value * incr_dir_correction
            squeezer = f.squeeze_increment(reorder_df.data,
                                           squeeze_eg,
                                           low_val, high_val,
                                           increment=increment)

        strip_df.update_data(reorder_df.data.copy())
        strip_df.data['prop_s'] = squeezer

        strip_df.data.drop(['new_order'], axis=1, inplace=True)
        for col in ['c', 'eg']:
            strip_df.data[col] = source2.data[col]

        strip_df.data['a'] = ed.p2_marker_alpha
        strip_df.data['s'] = ed.p2_marker_size

        strip_df.data.sort_values('prop_s', inplace=True)

        reorder_df.data['new_order'] = squeezer
        reorder_df.data.sort_values('new_order', inplace=True)
        reorder_df.data['new_order'] = np.arange(1, len(reorder_df.data) + 1,
                                                 dtype='int')

        proposal.update_order(reorder_df.data[['new_order']])

        update_stripplot()

    # extra filters
    def update_sel_filt1(attr, old, new):
        ed.sel_filt1 = new

    def update_sel_filt2(attr, old, new):
        ed.sel_filt2 = new

    def update_sel_filt3(attr, old, new):
        ed.sel_filt3 = new

    def update_oper1(attr, old, new):
        ed.sel_oper1 = new

    def update_oper2(attr, old, new):
        ed.sel_oper2 = new

    def update_oper3(attr, old, new):
        ed.sel_oper3 = new

    def update_txt_input1(attr, old, new):
        ed.txt_input1 = new

    def update_txt_input2(attr, old, new):
        ed.txt_input2 = new

    def update_txt_input3(attr, old, new):
        ed.txt_input3 = new

    # animate
    def animate_source(attr, old, new):

        use_hover = ed.chk_hover_on and ed.chk_hover_sel
        if mgrps_gb.data:
            hover_dict = {}
            # try to find data for selected month group, if none found, stop
            try:
                mth = mgrps_gb.data.get_group(new)
            except:
                label.text = 'NO DATA'
                return
            x = mth[ed.sel_xtype].values
            y = mth[ed.sel_ytype].values
            c = mth['c'].values
            a = mth['a'].values
            s = mth['s'].values
            eg = mth['eg'].values

            s1_dict = {'x': x, 'y': y, 'c': c,
                       'a': a, 's': s, 'eg': eg}

            if use_hover:
                for idx in ed.chk_hover_sel:
                    col = hdict[idx][0]
                    if col != ed.sel_measure:
                        hover_dict[col] = mth[col].values

            s1_dict.update(hover_dict)

            source1.update(data=s1_dict)
            label.text = date_list[new]
            sel_mth_num.value = str(new)
            # reset "running" values for edit zone value conversion using
            # the cross_val function (use current month values, not
            # the values from the last time the "plot" button was used)
            if ed.sel_xtype in ['prop_r', 'pcnt_r']:
                filt_xax.data = x
                idx_xax.data = mth['prop_s'].values

    def animate():
        box1.right, box1.left = None, None
        if but_play.label == '► Play':
            but_play.label = '❚❚ Pause'
            cb.identifier = \
                doc.add_periodic_callback(animate_update, animate_speed)
        else:
            but_play.label = '► Play'
            doc.remove_periodic_callback(cb.identifier)

    def reset():
        box1.right, box1.left = None, None
        slider_animate.value = 0
        sel_mth_num.value = '0'
        sel_mth_oper.value = '=='

    def refresh():
        eg_arr = anim_df.data['eg'].values
        for eg, slider in sl_size_dict.items():
            np.put(anim_df.data['s'], np.where(eg_arr == eg)[0], slider.value)
        for eg, slider in sl_alpha_dict.items():
            np.put(anim_df.data['a'], np.where(eg_arr == eg)[0], slider.value)
        # capture the new size and alpha values for the month groupby data
        mgrps_gb.update_data(anim_df.data.groupby('mnum'))

    def fwd1():
        box1.right, box1.left = None, None
        new_val = slider_animate.value + 1
        if new_val < max_month:
            slider_animate.value = new_val
            sel_mth_num.value = str(new_val)

    def back1():
        box1.right, box1.left = None, None
        new_val = slider_animate.value - 1
        if new_val >= 0:
            slider_animate.value = new_val
            sel_mth_num.value = str(new_val)

    def animate_update():
        box1.right, box1.left = None, None
        mth = slider_animate.value + 1
        if mth > max_month:
            mth = 0
        slider_animate.value = mth
        sel_mth_num.value = str(mth)

    # def prepare_animate(attr, old, new):
    #     pass
        # future development...trails

    # proposal_save
    # grab the widget values, create a dictionary, pickle
    def store_vals():

        with open('dill/editor_dict.pkl', 'wb') as handle:
            pickle.dump(vars(ed),
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def save_edited_df():
        store_vals()
        calc_ds.data.to_pickle('dill/ds_edit.pkl')
        save_edited_order()

    def save_edited_order():
        reorder_df.data[['new_order']].to_pickle('dill/p_edit.pkl')

    def save_order_to_excel():
        xl_str = 'excel/' + ed.case + '/proposals.xlsx'
        df = reorder_df.data[['new_order']]
        df = df.reset_index()[['empkey']]
        df.index = df.index + 1
        df.index.name = 'order'

        ws_dict = pd.read_excel(xl_str, index_col=0, sheet_name=None)
        ws_dict['edit'] = df

        with pd.ExcelWriter(xl_str, engine='xlsxwriter') as writer:

            for ws_name, df_sheet in ws_dict.items():
                df_sheet.to_excel(writer, sheet_name=ws_name)

    def base_change(attr, old, new):
        ed.sel_base = new

    def cond_change(attr, old, new):
        ed.sel_cond = new

    def find_order():
        try:  # look for edit list or compare list (determined by sel_proposal)
            if ed.sel_proposal == 'edit':  # edit order
                prop_name = 'edit'
                if proposal.list_order is not None:
                    df_order = proposal.list_order
                else:
                    df_order = pd.read_pickle('dill/p_edit.pkl')
            else:  # reset to compare order
                prop_name = ed.sel_proposal
                df_order = pd.read_pickle('dill/p_' + ed.sel_proposal + '.pkl')
        except OSError:  # above not found, default to first found
            df_order, prop_name = use_first_proposal_found('edit')

        proposal.update_order(df_order)
        proposal.update_name(prop_name)

    def proposal_change(attr, old, new):
        ed.sel_proposal = new
        # set the proposal.list_order
        find_order()

    # Center Column
    def measure_change(attr, old, new):
        ed.sel_measure = new

    def calc_button():

        label.text = ''
        calc_note.visible = True
        find_order()
        calc_dataset()
        join_dataset()
        update_main_plot()
        update_stripplot()
        calc_note.visible = False

    def plot_button():
        label.text = ''
        plot_note.visible = True
        join_dataset()
        update_main_plot()
        plot_note.visible = False

    def calc_dataset():
        # this routine creates a new integrated dataset based on a given
        # list order and list of job assignment conditions

        # to change calculation order,
        # update the proposal.list_order property...

        # save the input list order (not every time a squeeze is done) if
        # the edit proposal is selected (sel_proposal).
        # if the proposal is not edit, the order column is 'idx',
        # not 'new_order'.
        # This avoids saving a non-edit proposal list as an edited list.
        if 'new_order' in proposal.list_order.columns:
            proposal.list_order.to_pickle('dill/p_edit.pkl')
        # save the widget settings
        store_vals()
        # calling the main integrated dataset generation routine...
        ds = make_dataset(proposal_name=proposal.name,
                          df_order=proposal.list_order,
                          conditions=cond_dict[ed.sel_cond],
                          ds=skel.data,
                          ds_stand=ds_stand.data)

        calc_ds.update_data(ds)  # set to instance of Data class

    def update_axis_formats():
        if len(filt_df.data):
            if ed.sel_ytype == 'abs':
                if ed.sel_measure in ['cpay', 'mpay', 'ylong', 'mlong',
                                      'age', 'scale', 's_lmonths']:
                    ed.cht_yflipped = False
                else:
                    ed.cht_yflipped = True
            else:
                ed.cht_yflipped = False

            p1.y_range.update(flipped=ed.cht_yflipped)
            if ed.sel_measure in pcnt_cols:
                p1.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")
            else:
                if ed.sel_measure in float_cols:
                    p1.yaxis[0].formatter = NumeralTickFormatter(format="0.0")
                elif ed.sel_measure in date_cols:
                    p1.yaxis[0].formatter = DatetimeTickFormatter(years=['%Y'])
                else:
                    p1.yaxis[0].formatter = NumeralTickFormatter(format="0")

            if ed.sel_xtype in ['pcnt_s', 'pcnt_r']:
                p1.xaxis[0].formatter = NumeralTickFormatter(format="0.0%")
                if (slider_edit_zone.value[1] > 1 or
                        slider_edit_zone.value[1] > max(filt_xax.data)):
                    ed.x_high = .65 * max(filt_xax.data)
                    ed.x_low = .45 * max(filt_xax.data)
                    ed.ez_step = .001
            else:
                p1.xaxis[0].formatter = NumeralTickFormatter(format="0")
                if (slider_edit_zone.value[1] <= 1 or
                        slider_edit_zone.value[1] > max(filt_xax.data)):
                    ed.x_high = int(.65 * max(filt_xax.data))
                    ed.x_low = int(.45 * max(filt_xax.data))
                    ed.ez_step = 1
            slider_edit_zone.update(end=max(filt_xax.data),
                                    step=ed.ez_step,
                                    value=(ed.x_low, ed.x_high))

    def join_dataset():

        label.text = ''
        ret_only = 1 in ed.chk_filter
        extra_filter = 0 in ed.chk_filter

        base_cols = [ed.sel_measure, 'mnum']
        calc_ds_cols = [ed.sel_measure, 'mnum', 'new_order', 'eg']
        if ret_only:
            calc_ds_cols.append('ret_mark')

        # if extra filters are to be used, the appropriate columns are
        # added to the dataframe
        if extra_filter:  # this means filter is checked
            a1 = ed.sel_filt1
            a2 = ed.sel_filt2
            a3 = ed.sel_filt3

            o1 = ed.sel_oper1
            o2 = ed.sel_oper2
            o3 = ed.sel_oper3

            v1 = ed.txt_input1
            v2 = ed.txt_input2
            v3 = ed.txt_input3

            # filtlist means "filter list"
            attr_filtlist = [a1, a2, a3]
            oper_filtlist = [o1, o2, o3]
            vals_filtlist = [v1, v2, v3]

            # add filter columns
            filt_cols = []
            for i, attr in enumerate(attr_filtlist):
                if attr_filtlist[i] and oper_filtlist[i] and vals_filtlist[i]:
                    filt_cols.append(attr)
            filt_cols = list(set(filt_cols))
            calc_ds_cols = list(set().union(calc_ds_cols, filt_cols))

        if ed.chk_hover_on and ed.chk_hover_sel:

            hover_cols.data = []
            for key in ed.chk_hover_sel:
                col = hdict[key][0]
                if col != ed.sel_measure:
                    hover_cols.data.append(col)
            if hover_cols.data:
                calc_ds_cols = list(set().union(calc_ds_cols,
                                                hover_cols.data))

        # -----------------------------------------------------------------

        # BASE DATAFRAME (not filtered)
        # assign base_ds - check if stored dataset must be read from disc
        # or current base_ds may be used
        if ed.sel_base == 'standalone':
            base_ds.data = ds_stand.data
        else:
            if ed.sel_base != ed.base_ds_name:
                base_ds.data = pd.read_pickle('dill/ds_' +
                                              ed.sel_base + '.pkl')
                ed.base_ds_name = ed.sel_base

        df = base_ds.data[base_cols].copy()
        df.rename(columns={ed.sel_measure: ed.sel_measure + '_b'},
                  inplace=True)

        # for stripplot and squeeze (month zero):
        data_reorder = calc_ds.data[calc_ds.data.mnum == 0][['eg']].copy()
        data_reorder['new_order'] = \
            np.arange(len(data_reorder)).astype(int) + 1

        # set the df attribute of the reorder_df Data object:
        reorder_df.update_data(data_reorder)

        # index df for range values conversion (integrated ds month zero order)
        idx_df.update_data(reorder_df.data[[]].copy())
        idx_df.data['orig_order'] = np.arange(len(idx_df.data)) + 1

        join_ds = calc_ds.data[calc_ds_cols].copy()

        # add mnum to index
        df.set_index('mnum', append=True, inplace=True)
        join_ds.set_index('mnum', append=True, inplace=True)

        join_ds.rename(columns={ed.sel_measure: 'abs', 'new_order': 'prop_s'},
                       inplace=True)

        # JOIN BASE and COMPARE
        df = df.join(join_ds)

        df.reset_index(level='mnum', inplace=True)

        df.sort_values(['mnum', 'prop_s'], inplace=True)

        strip_df.update_data(df[df.mnum == 0][['prop_s', 'eg']].copy())
        # set up color column - Note rgba values do not work with this
        egs = strip_df.data['eg'].values
        clr = np.empty(len(strip_df.data), dtype='object')
        for eg in eg_list:
            np.put(clr, np.where(egs == eg)[0], eg_cdict[eg])
        strip_df.data['c'] = clr
        strip_df.data['a'] = ed.p2_marker_alpha
        strip_df.data['s'] = ed.p2_marker_size

        # running (monthly) proposal list ordering
        df['prop_r'] = df.groupby('mnum').cumcount() + 1

        prop_r = df.prop_r.values
        eg_vals = df.eg.values
        eg_denom_dict = df.groupby('eg').prop_r.max().to_dict()

        denoms = np.zeros(eg_vals.size)

        for eg in eg_list:
            np.put(denoms, np.where(eg_vals == eg)[0], eg_denom_dict[eg])

        df['pcnt_r'] = prop_r / denoms
        df['pcnt_s'] = f.make_starting_val_column(df, 'pcnt_r',
                                                  inplace=False)

        # FILTERING

        # ret only filter
        if ret_only:  # this means ret_only is checked

            df = df[eval('(df.ret_mark == 1)')].copy()

        if extra_filter:
            df, filt_str.data = filter_ds(df,
                                          attr1=a1, oper1=o1, val1=v1,
                                          attr2=a2, oper2=o2, val2=v2,
                                          attr3=a3, oper3=o3, val3=v3)
            filt_str.data = ', with filter: [ ' + filt_str.data + ' ]'
        else:
            filt_str.data = ''

        df = add_source_columns(df)
        df.sort_values(by='prop_s', inplace=True)

        # make dataframe and groupby source for animation
        anim_df.data = df.copy()
        mgrps_gb.update_data(anim_df.data.groupby('mnum'))

        # month filter
        mnum_oper = ed.sel_mth_oper
        mnum_val = ed.sel_mth_num
        mnum_filt_str = ' '.join(['mnum', mnum_oper, mnum_val])
        mnum_str = '(df.' + mnum_filt_str + ')'
        df_display = df[eval(mnum_str)].copy()

        if len(df_display):

            filt_df.update_data(df_display)

            # make arrays from filt_df
            filt_xax.update_data(filt_df.data[ed.sel_xtype].values)
            idx_xax.update_data(filt_df.data['prop_s'].values)
            alpha_filt_arr.update_data(filt_df.data['a'].values)
            eg_filt_arr.update_data(filt_df.data['eg'].values)
            zero_filt_arr.update_data(np.full(len(filt_df.data), 0.0))
            size_filt_arr.update_data(filt_df.data['s'].values)
            slider_edit_zone.update(end=max(filt_xax.data))

        else:
            # if df_display is empty (through use of extra filters):
            label.text = 'NO DATA: mth ' + ed.sel_mth_num

    def add_source_columns(df):
        # set up color column - Note rgba values do not work with this
        egs = df['eg'].values
        clr = np.empty(len(df), dtype='object')
        alph = np.zeros(len(df))
        sze = np.zeros(len(df))

        # set colors from eg_cdict and set size and alpha from slider values
        for eg in eg_list:
            these_idx = np.where(egs == eg)[0]
            np.put(clr, these_idx, eg_cdict[eg])
            np.put(alph, these_idx, sl_alpha_dict[eg].value)
            np.put(sze, these_idx, sl_size_dict[eg].value)
        df['c'] = clr
        df['a'] = alph
        df['s'] = sze

        # add "diff" column if selected by sel_ytype dropdown widget input
        if ed.sel_ytype == 'diff':
            diff_str.data = ' vs ' + ed.sel_base + ' '
            if ed.sel_measure not in no_invert:
                df['diff'] = df[ed.sel_measure + '_b'] - df['abs']
            else:
                df['diff'] = df['abs'] - df[ed.sel_measure + '_b']
        else:
            diff_str.data = ' '

        return df

    def update_main_plot():

        p1.title.text = (proposal.name + diff_str.data +
                         ed.sel_measure.upper() +
                         ' ' + ed.sel_ytype + ' values' +
                         filt_str.data)

        if 0 in ed.chk_display:
            acol = alpha_filt_arr.data
        else:
            acol = zero_filt_arr.data

        source1.data = {'x': filt_df.data[ed.sel_xtype].values,
                        'y': filt_df.data[ed.sel_ytype].values,
                        'c': filt_df.data['c'].values,
                        'a': acol,
                        's': filt_df.data['s'].values,
                        'eg': filt_df.data['eg'].values}

        if ed.chk_hover_on and ed.chk_hover_sel:

            for key in ed.chk_hover_sel:
                col = hdict[key][0]
                if col != ed.sel_measure:
                    source1.add(data=filt_df.data[col].values,
                                name=col)

        xl = float(ed.x_low)
        xh = float(ed.x_high)

        box1.left, box1.right = xh, xl

        xl2 = f.cross_val(filt_xax.data, xl, idx_xax.data)
        xh2 = f.cross_val(filt_xax.data, xh, idx_xax.data)

        box2.left, box2.right = xh2, xl2

        clear_line_data()
        update_axis_formats()
        update_line_data()

    def make_plots(return_plots=False):

        p1.plot_width = plot_width  # ed.cht_xsize
        p1.plot_height = plot_height  # ed.cht_ysize
        p1.y_range = DataRange1d(range_padding=0.0)

        p1.x_range = DataRange1d(end=0.0, flipped=True, range_padding=0.0)
        p1.title.text = (proposal.name + diff_str.data +
                         ed.sel_measure.upper() +
                         ' ' + ed.sel_ytype + ' values' +
                         filt_str.data)
        p1.background_fill_color = ed.sel_bgc
        p1.background_fill_alpha = float(ed.sel_bgc_alpha)
        p1.add_tools(crosshair_tool.data)
        p1.add_tools(hover_tool.data)
        p1.toolbar.active_inspect = [hover_tool.data]
        # p1.output_backend = 'webgl'
        box1.left, box1.right = ed.x_high, ed.x_low

        p2.background_fill_color = ed.sel_bgc
        p2.background_fill_alpha = float(ed.sel_bgc_alpha)
        # p2.output_backend = 'webgl'
        box2.left, box2.right = ed.x_high, ed.x_low

        # source1 dictionary assignment
        src1_dict = {'x': filt_df.data[ed.sel_xtype],
                     'y': filt_df.data[ed.sel_ytype],
                     'c': filt_df.data['c'],
                     'a': filt_df.data['a'],
                     's': filt_df.data['s'],
                     'eg': filt_df.data['eg']}

        if ed.chk_hover_on and ed.chk_hover_sel:
            hover_dict = {}
            for idx in ed.chk_hover_sel:
                col = hdict[idx][0]
                if col != ed.sel_measure:
                    hover_dict[col] = filt_df.data[col].values
            src1_dict.update(hover_dict)

        source1.update(data=src1_dict)

        # source2 dictionary assignment
        src2_dict = {'x': strip_df.data['prop_s'],
                     # 'y': df_display[yval],
                     'c': strip_df.data['c'],
                     'a': strip_df.data['a'],
                     's': strip_df.data['s'],
                     'eg': strip_df.data['eg']}

        # set ColumnDataSource data
        source1.data = src1_dict
        source2.data = src2_dict

        # ------------------------------------------------------------------

        p1.grid.grid_line_color = ed.sel_gridc
        p1.grid.grid_line_alpha = float(ed.sel_gridc_alpha)
        p1.toolbar.logo = None
        p1.grid.minor_grid_line_color = ed.sel_gridc
        p1.grid.minor_grid_line_alpha = ed.minor_grid_alpha
        p1.grid.minor_grid_line_dash = 'dotted'

        p1.circle('x', 'y', color='c', size='s',
                  alpha='a',
                  line_color=marker_edge_color,
                  line_width=marker_edge_width,
                  source=source1)

        p2.circle(x='x',
                  y={'field': 'eg', 'transform': Jitter(width=0.92)},
                  color='c',
                  size='s',
                  alpha='a',
                  line_color=None,
                  source=source2)

        p2.yaxis[0].ticker.desired_num_ticks = len(eg_list)
        p2.yaxis.minor_tick_line_color = None
        p2.ygrid.grid_line_color = None
        p2.xgrid.grid_line_color = ed.sel_gridc
        p2.xgrid.grid_line_alpha = float(sel_gridc_alpha.value)
        p2.toolbar.logo = None

        p1.add_layout(box1)
        p2.add_layout(box2)
        # p2.add_glyph(quad2_source, quad2)

        add_line_glyphs(eg_list)
        update_line_data()
        update_axis_formats()

        # zeroline
        zeroline = Span(location=0, dimension='width',
                        line_dash='dashed',
                        line_color='red', line_width=1)
        p1.add_layout(zeroline)
        p1.add_layout(label)

        if return_plots:
            return p1, p2

    # display tab
    def filter_change(attr, old, new):
        ed.chk_filter = list(chk_filter.active)
        if 1 not in ed.chk_filter:
            sel_mth_oper.value = '=='

    def add_line_glyphs(eg_list):

        for eg in eg_list:
            p1.add_glyph(src_dict.kdict['sp' + str(eg)],
                         glyph=polys.kdict['p' + str(eg)])
            p1.add_glyph(src_dict.kdict['sm' + str(eg)],
                         glyph=means.kdict['m' + str(eg)])
            p1.add_glyph(src_dict.kdict['ss' + str(eg)],
                         glyph=savgols.kdict['s' + str(eg)])

    def update_line_data():

        chkd = set.intersection(set([1, 2, 3]), set(ed.chk_display))

        # scatter markers
        if 0 in ed.chk_display:
            source1.data.update(a=alpha_filt_arr.data)
        else:
            source1.data.update(a=zero_filt_arr.data)

        if chkd:
            for eg in pd.unique(filt_df.data['eg']):
                eg_df = filt_df.data[filt_df.data['eg'] == eg].copy()
                xlvals = eg_df[ed.sel_xtype].values
                ylvals = eg_df[ed.sel_ytype].values
                idx = np.isfinite(xlvals) & np.isfinite(ylvals)
                xlvals = xlvals[idx]
                ylvals = ylvals[idx]

                # poly_fit
                if 1 in chkd:
                    pdata = poly.fit(xlvals, ylvals, poly_dim).linspace()

                    src_dict.kdict['sp' + str(eg)].data = \
                        dict(x=list(pdata[0]), y=list(pdata[1]))

                # mean
                if 2 in chkd:
                    yma = ema(ylvals, ema_len)
                    src_dict.kdict['sm' + str(eg)].data = dict(x=xlvals,
                                                               y=yma)

                # Savitzky–Golay filter
                if 3 in chkd:
                    sf_data = sf(ylvals, savgol_window, savgol_fit)
                    sf_data[sf_data == np.nan] = 0
                    src_dict.kdict['ss' + str(eg)].data.update(x=xlvals,
                                                               y=sf_data)

    def clear_line_data():
        for eg in eg_list:

            # poly_fit
            src_dict.kdict['sp' + str(eg)].data.update(**nan_dict)

            # mean
            src_dict.kdict['sm' + str(eg)].data.update(**nan_dict)

            # savgol
            src_dict.kdict['ss' + str(eg)].data.update(**nan_dict)

    def ema(arr, n):
        """
        compute an n period exponential moving average.
        """
        x = np.asarray(arr)
        weights = np.exp(np.linspace(-1., 0., n))
        weights /= weights.sum()

        a = np.convolve(x, weights, mode='full')[:len(x)]
        a[:n] = a[n]
        return a

    def display_change(attr, old, new):
        ed.chk_display = list(chk_display.active)
        clear_line_data()
        update_line_data()

    def month_oper_change(attr, old, new):
        ed.sel_mth_oper = new

    def month_num_change(attr, old, new):
        ed.sel_mth_num = new

    def ytype_change(attr, old, new):
        ed.sel_ytype = new

    def xtype_change(attr, old, new):
        ed.sel_xtype = new

    # size_alpha
    def reset_sliders():
        for s_slider in sl_size_dict.values():
            s_slider.value = start_dot_size
        for a_slider in sl_alpha_dict.values():
            a_slider.value = start_marker_alpha

    def slider_big():
        for s_slider in sl_size_dict.values():
            if s_slider.value < max_dot_size:
                s_slider.value += size_step

    def slider_sml():
        for s_slider in sl_size_dict.values():
            if s_slider.value >= size_step:
                s_slider.value -= size_step

    def slider_aup():
        for a_slider in sl_alpha_dict.values():
            if a_slider.value <= 1 - alpha_step:
                a_slider.value += alpha_step

    def slider_adn():
        for slider in sl_alpha_dict.values():
            if slider.value > 0:
                slider.value -= alpha_step

    # size_alpha source
    def update_scat_size_p1(attr, old, new, eg):
        s = sl_size_dict[eg].value
        s_arr = np.array(source1.data['s'])
        eg_arr = np.array(source1.data['eg'])
        np.put(s_arr, np.where(eg_arr == eg)[0], s)
        source1.data.update({'s': s_arr})

    def update_scat_alpha_p1(attr, old, new, eg):
        a = sl_alpha_dict[eg].value
        a_arr = np.array(source1.data['a'])
        eg_arr = np.array(source1.data['eg'])
        np.put(a_arr, np.where(eg_arr == eg)[0], a)
        source1.data.update({'a': a_arr})

    # grid_bg
    def update_bg_color(attr, old, new):
        float_alpha = float(sel_bgc_alpha.value)
        if 0 in ed.chk_color_apply:
            p1.background_fill_color = sel_bgc.value
            p1.background_fill_alpha = float_alpha
            p2.background_fill_color = sel_bgc.value
            p2.background_fill_alpha = float_alpha
            ed.sel_bgc = sel_bgc.value
            ed.sel_bgc_alpha = sel_bgc_alpha.value
        if 1 in ed.chk_color_apply:
            box1.fill_color = sel_bgc.value
            box1.fill_alpha = float_alpha
            box2.fill_color = sel_bgc.value
            box2.fill_alpha = float_alpha
            ed.box_fill_color = sel_bgc.value
            ed.box_fill_alpha = float_alpha

    def update_grid_color(attr, old, new):
        float_alpha = float(sel_gridc_alpha.value)
        if 0 in ed.chk_color_apply:
            p1.grid.grid_line_color = sel_gridc.value
            p1.grid.grid_line_alpha = float_alpha
            p2.xgrid.grid_line_color = sel_gridc.value
            p2.xgrid.grid_line_alpha = float_alpha
            ed.sel_gridc = sel_gridc.value
            ed.sel_gridc_alpha = sel_gridc_alpha.value
        if 1 in ed.chk_color_apply:
            box1.line_color = sel_gridc.value
            box1.line_alpha = float_alpha
            box2.line_color = sel_gridc.value
            box2.line_alpha = float_alpha
            ed.box_line_color = sel_gridc.value
            ed.box_line_alpha = float_alpha

    def reset_colors():
        temp_chk_color_apply = ed.chk_color_apply
        ed.chk_color_apply = [0, 1]
        sel_bgc.value = 'White'
        sel_bgc_alpha.value = '.10'
        sel_gridc.value = 'Gray'
        sel_gridc_alpha.value = '.20'
        ed.sel_bgc = 'White'
        ed.sel_bgc_alpha = '.10'
        ed.sel_gridc = 'Gray'
        ed.sel_gridc_alpha = '.20'

        if chk_minor_grid.active:
            p1.grid.minor_grid_line_color = 'Gray'
            p1.grid.minor_grid_line_alpha = .20
        else:
            p1.grid.minor_grid_line_alpha = 0.0

        sel_box_line_width.value = '1.0'
        box1.line_color = 'black'
        box1.line_alpha = .8
        box2.line_color = 'black'
        box2.line_alpha = .8
        box1.fill_color = 'black'
        box1.fill_alpha = .05
        box2.fill_color = 'black'
        box2.fill_alpha = .05
        ed.chk_color_apply = temp_chk_color_apply

    def minor_grid(attr, old, new):
        if chk_minor_grid.active:
            p1.grid.minor_grid_line_color = ed.sel_gridc
            p1.grid.minor_grid_line_alpha = float(ed.sel_gridc_alpha)
        else:
            p1.grid.minor_grid_line_alpha = 0.0
        ed.chk_minor_grid = list(chk_minor_grid.active)

    def color_apply(attr, old, new):
        ed.chk_color_apply = list(chk_color_apply.active)

    def edit_line_width(attr, old, new):
        ed.box_line_width = sel_box_line_width.value
        box1.line_width = float(new)
        box2.line_width = float(new)

    # hover
    def hover_tool_control(attr, old, new):
        ed.chk_hover_sel = list(chk_hover_sel.active)
        ed.chk_hover_on = list(chk_hover_on.active)
        manage_hover_tool()

    # make html for tooltip formatting
    def manage_hover_tool():
        if ed.chk_hover_on and ed.chk_hover_sel:

            pre_div = ('<div style="background-color:' +
                       'rgba(0, 0, 0, 0.03);' +
                       'overflow: auto;">')
            mid_div = ''
            suf_div = '</div>'

            for key in ed.chk_hover_sel:
                col = hdict[key][0]
                if col != ed.sel_measure:
                    mid_div += html_str % (col, ' ' + hdict[key][1])

            hover_tool.data.tooltips = pre_div + mid_div + suf_div

        else:
            hover_tool.data.tooltips = None
            tool_tips.data = None
            hover_cols.data = []

    # density (jitter stripplot)
    def update_stripplot():
        source2.data = dict(x=strip_df.data['prop_s'],
                            c=strip_df.data['c'],
                            a=strip_df.data['a'],
                            s=strip_df.data['s'],
                            eg=strip_df.data['eg'])

    def update_scat_size_p2(attr, old, new):
        size_arr = np.full(num_dots, new)
        source2.data.update({'s': size_arr})
        ed.p2_marker_size = new

    def update_scat_alpha_p2(attr, old, new):
        size_arr = np.full(num_dots, new)
        source2.data.update({'a': size_arr})
        ed.p2_marker_alpha = new

    # edit range
    def update_edit_range(attr, old, new):
        # Get slider values
        xl = slider_edit_zone.value[0]
        xh = slider_edit_zone.value[1]

        box1.left, box1.right = xh, xl

        xl2 = f.cross_val(filt_xax.data, xl, idx_xax.data)
        xh2 = f.cross_val(filt_xax.data, xh, idx_xax.data)

        box2.left, box2.right = xl2, xh2

        # update editor dict namespace
        ed.x_low = xl
        ed.x_high = xh

    # -----END Callback functions-------------------------------

    # -----START Callback actions-------------------------------

    # squeeze
    sel_sqz_type.on_change('value', sqz_type_change)
    sel_sqz_dir.on_change('value', sqz_dir_change)
    sel_emp_grp.on_change('value', emp_group_change)
    slider_squeeze.on_change('value', update_squeeze)
    but_squeeze.on_click(perform_squeeze)

    but_0add.on_click(line0_add)
    but_0sub.on_click(line0_sub)
    but_1add.on_click(line1_add)
    but_1sub.on_click(line1_sub)

    # extra filters
    sel_filt1.on_change('value', update_sel_filt1)
    sel_filt2.on_change('value', update_sel_filt2)
    sel_filt3.on_change('value', update_sel_filt3)
    sel_oper1.on_change('value', update_oper1)
    sel_oper2.on_change('value', update_oper2)
    sel_oper3.on_change('value', update_oper3)
    txt_input1.on_change('value', update_txt_input1)
    txt_input2.on_change('value', update_txt_input2)
    txt_input3.on_change('value', update_txt_input3)

    # animate:
    but_play.on_click(animate)
    but_reset.on_click(reset)
    but_refresh.on_click(refresh)
    but_back.on_click(back1)
    but_fwd.on_click(fwd1)
    slider_animate.on_change('value', animate_source)
    # commented for future development...
    # chk_trails.on_change('active', prepare_animate)

    # proposal_save
    but_save_edit.on_click(save_edited_df)
    but_save_order.on_click(save_order_to_excel)
    sel_base.on_change('value', base_change)
    sel_cond.on_change('value', cond_change)
    sel_proposal.on_change('value', proposal_change)

    # center column
    sel_measure.on_change('value', measure_change)
    but_plot.on_click(plot_button)
    but_calc.on_click(calc_button)

    # display
    chk_filter.on_change('active', filter_change)
    chk_display.on_change('active', display_change)
    sel_mth_oper.on_change('value', month_oper_change)
    sel_mth_num.on_change('value', month_num_change)
    sel_ytype.on_change('value', ytype_change)
    sel_xtype.on_change('value', xtype_change)

    # size_alpha
    for eg, slider in sl_size_dict.items():
        slider.on_change('value', partial(update_scat_size_p1, eg=eg))

    for eg, slider in sl_alpha_dict.items():
        slider.on_change('value', partial(update_scat_alpha_p1, eg=eg))

    but_slider_reset.on_click(reset_sliders)
    but_slider_big.on_click(slider_big)
    but_slider_sml.on_click(slider_sml)
    but_slider_aup.on_click(slider_aup)
    but_slider_adn.on_click(slider_adn)

    # grid_bg
    sel_bgc.on_change('value', update_bg_color)
    sel_bgc_alpha.on_change('value', update_bg_color)
    sel_gridc.on_change('value', update_grid_color)
    sel_gridc_alpha.on_change('value', update_grid_color)
    but_reset_colors.on_click(reset_colors)
    chk_minor_grid.on_change('active', minor_grid)
    chk_color_apply.on_change('active', color_apply)
    sel_box_line_width.on_change('value', edit_line_width)

    # hover
    chk_hover_on.on_change('active', hover_tool_control)
    chk_hover_sel.on_change('active', hover_tool_control)

    # density (stripplot):
    slider_strip_size.on_change('value', update_scat_size_p2)
    slider_strip_alpha.on_change('value', update_scat_alpha_p2)

    # edit range slider
    slider_edit_zone.on_change('value', update_edit_range)

    # ------END Callback Actions-------------------------------------

    # ------START Initial Computations-------------------------------
    # Read skeleton dataset
    try:
        skel.data = pd.read_pickle('dill/skeleton.pkl')
    except OSError:
        # exit routine if baseline dataset not found
        print('skeleton.pkl not found, run make_skeleton.py?\n')
        print('\n  >>> exiting routine.\n')
        sys.exit()

    # Read standalone dataset/assign baseline dataset
    try:
        ds_stand.data = pd.read_pickle('dill/standalone.pkl')
        if ed.sel_base == 'standalone':
            base_ds.data = ds_stand.data.copy()
        else:
            # set BASELINE dataset if something other than standalone
            try:
                base_ds.data = pd.read_pickle('dill/ds_' +
                                              ed.sel_base + '.pkl')
            except OSError:
                base_ds.data = ds_stand.data.copy()
                print('invalid "base_ds" name input?\n' +
                      'standalone set as base\n')
    except OSError:
        # exit routine if baseline dataset not found
        print('standalone.pkl or selected baseline dataset not found...\n' +
              'run standalone.py?\n')
        print('\n  >>> exiting routine.\n')
        sys.exit()

    # initial order and dataset generation
    find_order()
    calc_dataset()
    join_dataset()
    p1, p2 = make_plots(return_plots=True)
    manage_hover_tool()
    # --------END Initial Computations-------------------------------

    # --------START Widget Layout------------------------------------

    # PANEL1
    # squeeze tab items
    squeeze_widgets = column(row(spacer_sqz_but1,
                                 sel_sqz_type,
                                 spacer_sqz_but2,
                                 sel_emp_grp,
                                 spacer_sqz_but3,
                                 sel_sqz_dir),
                             row(slider_squeeze),
                             row(but_1add, spacer_toggle_1, but_1sub,
                                 spacer_toggle_center1,
                                 but_squeeze,
                                 spacer_toggle_center2,
                                 but_0add, spacer_toggle_2, but_0sub))

    # extra filters tab items
    filter_widgets = row(column(sel_filt1, sel_filt2, sel_filt3),
                         column(sel_oper1, sel_oper2, sel_oper3),
                         column(txt_input1,
                                txt_input2,
                                txt_input3))

    # animate tab items
    anim_row1 = row(but_play, spacer_anim1, but_reset)
    anim_row2 = row(slider_animate)
    anim_row3 = row(but_back, spacer_anim2, but_fwd,
                    spacer_anim_refresh, but_refresh)
    anim_col1 = column(anim_row1, anim_row2, anim_row3)
    # the commented items below are on hold for future development...
    # anim_col2 = column(chk_trails, sel_trails)
    # anim_items = row(anim_col1, spacer_anim, anim_col2)
    anim_items = row(anim_col1)

    # proposal_save
    save_widgets = row(column(spacer_top_save1, but_save_edit, but_save_order),
                       column(spacer_middle_save),
                       column(sel_base, sel_cond, sel_proposal))

    # make panels for main tab group
    panel1_tab1 = Panel(child=squeeze_widgets, title='squeeze')
    panel1_tab2 = Panel(child=filter_widgets, title='extra filters')
    panel1_tab3 = Panel(child=anim_items, title='animate')
    panel1_tab4 = Panel(child=save_widgets, title='proposal_save')
    # combine main panels into panel1 tab object
    panel1 = Tabs(tabs=[panel1_tab1, panel1_tab2,
                        panel1_tab3, panel1_tab4], width=panel1_width,
                  height=controls_height)

    # CENTER COLUMN
    buttons_and_attr_sel = column(spacer_top_center_col,
                                  sel_measure,
                                  but_plot,
                                  but_calc,
                                  height=controls_height,
                                  width=sel_width)

    # PANEL2
    # display
    but_row1 = row(spacer_disp_mth1, sel_mth_oper,
                   spacer_disp_mth2, sel_mth_num)
    but_row2 = row(spacer_disp_ax1, sel_ytype,
                   spacer_disp_ax2, sel_xtype)
    but_col = column(spacer_top_disp, but_row1, but_row2)

    chk_col = column(chk_filter, chk_display)

    # display tab items
    display_widgets = row(chk_col, but_col)

    # size_alpha tab items
    szal_sliders = row(slider_list)

    sz_buttons = row(but_slider_sml, spacer_size_buts, but_slider_big)
    al_buttons = row(but_slider_adn, spacer_alpha_buts, but_slider_aup)

    szal_but_col = column(spacer_top_size_alpha, but_slider_reset,
                          sz_buttons, al_buttons,
                          width=120)
    szal_items = row(szal_sliders, szal_but_col)

    # grid_bg tab items
    gbg_col1 = column(sel_bgc, sel_gridc,
                      height=chart_sel_height)

    gbg_col2 = column(sel_bgc_alpha, sel_gridc_alpha,
                      height=chart_sel_height)
    gbg_col12 = row(gbg_col1, spacer_linesbg_col,
                    gbg_col2, spacer_linesbg_col2)
    gbg_bottom_row = row(but_reset_colors,
                         spacer_linesbg_bottom,
                         chk_minor_grid, width=200)
    gbg_left = column(gbg_col12, gbg_bottom_row, width=300)
    gbg_col3 = column(spacer_top_color_apply, chk_color_apply,
                      sel_box_line_width)

    gbg_items = row(gbg_left, gbg_col3)

    # hover tab items
    hover_row = row(chk_hover_on, chk_hover_sel)

    # make panels for aux tab group
    panel2_tab1 = Panel(child=display_widgets, title='display')
    panel2_tab2 = Panel(child=szal_items, title='size_alpha')
    panel2_tab3 = Panel(child=gbg_items, title='grid_bg')
    panel2_tab4 = Panel(child=hover_row, title='hover')
    panel2_tab5 = Panel(child=column(slider_strip_size, slider_strip_alpha),
                        title='density')

    # combine aux panels into panel2 tab object
    panel2 = Tabs(tabs=[panel2_tab1, panel2_tab2, panel2_tab3,
                        panel2_tab4, panel2_tab5],
                  height=controls_height, width=panel2_width)

    # --------END Widget Layout--------------------------------------

    # --------START Main Layout--------------------------------------

    p1_row = row(p1)
    p2_row = row(p2)

    p1.add_layout(calc_note)
    p1.add_layout(plot_note)

    l_o = layout(row(panel1,
                     spacer_controls1,
                     buttons_and_attr_sel,
                     spacer_controls2,
                     panel2),
                 row(spacer_edit, slider_edit_zone),
                 p1_row,
                 p2_row)

    doc.add_root(l_o)
    return doc

    # --------END Main Layout----------------------------------------


def color_list():
    '''provides a list of string color names for editor grid_bg tab
    color selectors
    '''
    colors = ['AliceBlue', 'AntiqueWhite', 'Aqua', 'Aquamarine',
              'Azure', 'Beige', 'Bisque', 'Black', 'BlanchedAlmond',
              'Blue', 'BlueViolet', 'Brown', 'BurlyWood', 'CadetBlue',
              'Chartreuse', 'Chocolate', 'Coral', 'CornflowerBlue',
              'Cornsilk', 'Crimson', 'Cyan', 'DarkBlue', 'DarkCyan',
              'DarkGoldenRod', 'DarkGray', 'DarkGrey', 'DarkGreen',
              'DarkKhaki', 'DarkMagenta', 'DarkOliveGreen', 'Darkorange',
              'DarkOrchid', 'DarkRed', 'DarkSalmon', 'DarkSeaGreen',
              'DarkSlateBlue', 'DarkSlateGray', 'DarkSlateGrey',
              'DarkTurquoise', 'DarkViolet', 'DeepPink', 'DeepSkyBlue',
              'DimGray', 'DimGrey', 'DodgerBlue', 'FireBrick',
              'FloralWhite', 'ForestGreen', 'Fuchsia', 'Gainsboro',
              'GhostWhite', 'Gold', 'GoldenRod', 'Gray', 'Grey',
              'Green', 'GreenYellow', 'HoneyDew', 'HotPink', 'IndianRed',
              'Indigo', 'Ivory', 'Khaki', 'Lavender', 'LavenderBlush',
              'LawnGreen', 'LemonChiffon', 'LightBlue', 'LightCoral',
              'LightCyan', 'LightGoldenRodYellow', 'LightGray',
              'LightGrey', 'LightGreen', 'LightPink', 'LightSalmon',
              'LightSeaGreen', 'LightSkyBlue', 'LightSlateGray',
              'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
              'LimeGreen', 'Linen', 'Magenta', 'Maroon',
              'MediumAquaMarine', 'MediumBlue', 'MediumOrchid',
              'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue',
              'MediumSpringGreen', 'MediumTurquoise', 'MediumVioletRed',
              'MidnightBlue', 'MintCream', 'MistyRose', 'Moccasin',
              'NavajoWhite', 'Navy', 'OldLace', 'Olive', 'OliveDrab',
              'Orange', 'OrangeRed', 'Orchid', 'PaleGoldenRod',
              'PaleGreen', 'PaleTurquoise', 'PaleVioletRed', 'PapayaWhip',
              'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
              'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Salmon',
              'SandyBrown', 'SeaGreen', 'SeaShell', 'Sienna', 'Silver',
              'SkyBlue', 'SlateBlue', 'SlateGray', 'SlateGrey', 'Snow',
              'SpringGreen', 'SteelBlue', 'Tan', 'Teal', 'Thistle',
              'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
              'WhiteSmoke', 'Yellow', 'YellowGreen']

    return colors


def alpha_list():
    '''provides a list of string decimals for editor grid_bg tab
    alpha selectors
    '''
    alphas = ['.00', '.01', '.02', '.03', '.04', '.05', '.06', '.07',
              '.08', '.09', '.10', '.11', '.12', '.13', '.14', '.15',
              '.16', '.17', '.18', '.19', '.20', '.21', '.22', '.23',
              '.24', '.25', '.26', '.27', '.28', '.29', '.30', '.31',
              '.32', '.33', '.34', '.35', '.36', '.37', '.38', '.39',
              '.40', '.41', '.42', '.43', '.44', '.45', '.46', '.47',
              '.48', '.49', '.50', '.51', '.52', '.53', '.54', '.55',
              '.56', '.57', '.58', '.59', '.60', '.61', '.62', '.63',
              '.64', '.66', '.68', '.70', '.72', '.75', '.77', '.80',
              '.82', '.85', '.87', '.90', '.92', '.95', '.97', '1.0']

    return alphas


def line_widths():
    '''provides a list of string decimals for editor grid_bg tab
    edit line width selector
    '''
    widths = ['0.1', '0.2', '0.3', '0.4', '0.5',
              '0.6', '0.7', '0.8', '0.9', '1.0',
              '1.1', '1.2', '1.3', '1.4', '1.5',
              '1.6', '1.7', '1.8', '1.9', '2.0']

    return widths


def use_first_proposal_found(proposal_name):
    '''find and return the first list order found in 'dill/proposal_names.pkl'.
    This function is used when another proposal name is designated by another
    section of the program but does not exist.

    inputs
        proposal_name (string)
            the name of the proposal which was not found
    '''
    try:
        prop_names = \
            pd.read_pickle('dill/proposal_names.pkl').proposals.tolist()
        this_prop_name = prop_names[0]
        stored_case = pd.read_pickle('dill/case_dill.pkl').case.value
        print('\nerror : proposal name "' +
              str(proposal_name) + '" not found...\n')
        print('available proposal names are ', prop_names,
              'for case study:',
              stored_case)
        print('< using ' + this_prop_name + '>')

        return pd.read_pickle('dill/p_' + this_prop_name + '.pkl'), \
            this_prop_name

    except OSError:
        print('dill/proposal_names.pkl' + ' or ' +
              'dill/case_dill.pkl' + ' not found')
        print('\n  >>> exiting routine.\n')
        sys.exit()


def make_dataset(proposal_name='',
                 df_order=None,  # list order
                 conditions=[],
                 ds=None,  # skeleton input
                 ds_stand=None):  # used to calculate pre-implementation data

    pre, suf = 'dill/', '.pkl'

    order_name = 'p_' + proposal_name
    # dataset_name = 'ds_' + proposal_name

    order_file = (pre + order_name + suf)

    sdict = pd.read_pickle('dill/dict_settings.pkl')
    tdict = pd.read_pickle('dill/dict_job_tables.pkl')

    num_of_job_levels = sdict['num_of_job_levels']
    lspcnt_calc = sdict['lspcnt_calc_on_remaining_population']

    try:
        df_master = pd.read_pickle(pre + 'master' + suf)
    except OSError:
        print('Master list not found.  Run build_program_files script?')
        sys.exit()

    # do not include inactive employees (other than furlough) in data model
    df_master = df_master[
        (df_master.line == 1) | (df_master.fur == 1)].copy()

    # ORDER the skeleton df according to INTEGRATED list order.
    # df_skel can initially be in any integrated order, each employee
    # group must be in proper order relative to itself.
    # Use the short-form 'idx' (order) column from either the proposed
    # list or the new_order column from an edited list to create a new column,
    # 'new_order', within the long-form df_skel.  The new order column
    # is created by data alignment using the common empkey indexes.
    # The skeleton may then be sorted by month and new_order.
    # (note: duplicate df_skel empkey index empkeys (from different months)
    # are assigned the same order value)

    if proposal_name == 'edit':
        df_new_order = pd.read_pickle(order_file)
        ds['new_order'] = df_new_order['new_order']
        # dataset_file = (pre + 'ds_edit' + suf)
    else:
        ds_index = ds[ds.mnum == 0].index.values
        df_order_index = df_order.index.values
        # mask will remove any inactive employees existing
        # within the list df_order proposal
        mask = np.isin(df_order_index, ds_index)

        df_order = df_order[mask].copy()
        df_order_vals = df_order['idx'].values

        # assign back to df_order column to permit index data alignment...
        df_order['idx'] = st.rankdata(df_order_vals).astype(int)
        ds['new_order'] = df_order['idx']
        # dataset_file = (pre + dataset_name + suf)

    # sort the skeleton by month and proposed list order
    ds.sort_values(['mnum', 'new_order'], inplace=True)

    # ORIG_JOB*

    eg_sequence = df_master.eg.values
    fur_sequence = df_master.fur.values

    # create list of employee group codes from the master data
    egs = sorted(pd.unique(eg_sequence))
    # retrieve job counts array
    jcnts_arr = tdict['jcnts_arr']

    if 'prex' in conditions:

        sg_rights = sdict['sg_rights']
        sg_eg_list = []
        sg_dict = od()
        stove_dict = od()

        # Find the employee groups which have pre-existing job rights...
        # grab the eg code from each sg (special group) job right description
        # and add to sg_eg_list
        for line_item in sg_rights:
            sg_eg_list.append(line_item[0])
        # place unique eg codes into sorted list
        sg_eg_list = sorted(pd.unique(sg_eg_list))

        # Make a dictionary containing the special group data for each
        # group with special rights
        for eg in sg_eg_list:
            sg_data = []
            for line_item in sg_rights:
                if line_item[0] == eg:
                    sg_data.append(line_item)
            sg_dict[eg] = sg_data

        for eg in egs:

            if eg in sg_eg_list:
                # (run prex stovepipe routine with eg dict key and value)
                sg = df_master[df_master.eg == eg]['sg'].values
                fur = df_master[df_master.eg == eg]['fur']
                ojob_array = f.make_stovepipe_prex_shortform(
                    jcnts_arr[0][eg - 1], sg, sg_dict[eg], fur)
                prex_stove = np.take(ojob_array, np.where(fur == 0)[0])
                stove_dict[eg] = prex_stove
            else:
                # (run make_stovepipe routine with eg dict key and value)
                stove_dict[eg] = f.make_stovepipe_jobs_from_jobs_arr(
                    jcnts_arr[0][eg - 1])

        # use dict values as inputs to sp_arr,
        # ordered dict maintains proper sequence...
        sp_arr = list(np.array(list(stove_dict.values())))
        # total of jobs per eg
        eg_job_counts = np.add.reduce(jcnts_arr[0], axis=1)

        orig_jobs = f.make_intgrtd_from_sep_stove_lists(sp_arr,
                                                        eg_sequence,
                                                        fur_sequence,
                                                        eg_job_counts,
                                                        num_of_job_levels)

    else:

        orig_jobs = f.make_original_jobs_from_counts(
            jcnts_arr[0], eg_sequence,
            fur_sequence, num_of_job_levels).astype(int)

    # insert stovepipe job result into new column of proposal (month_form)
    # this indexes the jobs with empkeys (orig_jobs is an ndarray only)

    df_master['orig_job'] = orig_jobs

    # ASSIGN JOBS - flush and no flush option*

    # cmonths - career length in months for each employee.
    #   length is equal to number of employees
    cmonths = f.career_months(df_master, sdict['starting_date'])

    # nonret_each_month: count of non-retired employees remaining
    # in each month until no more remain -
    # length is equal to longest career length
    nonret_each_month = f.count_per_month(cmonths)
    all_months = np.sum(nonret_each_month)
    high_limits = nonret_each_month.cumsum()
    low_limits = f.make_lower_slice_limits(high_limits)

    # job_level_counts = np.array(jcnts_arr[1])

    if sdict['delayed_implementation']:

        imp_month = sdict['imp_month']
        imp_low = low_limits[imp_month]
        imp_high = high_limits[imp_month]

        # # read the standalone dataset (info is not in integrated order)
        # ds_stand = pd.read_pickle(stand_path_string)

        # get standalone data and order it the same as the integrated dataset.
        # create a unique key column in the standalone data df and a temporary
        # df which is ordered according to the integrated dataset
        imp_cols, arr_dict, col_array = \
            f.make_preimp_array(ds_stand, ds,
                                imp_high, sdict['compute_job_category_order'],
                                sdict['compute_pay_measures'])

        # select columns to use as pre-implementation data for integrated
        # dataset data is limited to the pre-implementation months

        # aligned_jnums and aligned_fur arrays are the same as standalone data
        # up to the end of the implementation month, then the standalone value
        # for the implementation month is passed down unchanged for the
        # remainder of months in the model.  These arrays carry over
        # standalone data for each employee group to be honored until and when
        # the integrated list is implemented.
        # These values from the standalone datasets (furlough status and
        # standalone job held at the implementation date) are needed for
        # subsequent integrated dataset job assignment calculations.  Other
        # standalone values are simply copied and inserted into the
        # pre-implementation months of the integrated dataset.

        delayed_jnums = col_array[arr_dict['jnum']]
        delayed_fur = col_array[arr_dict['fur']]

        aligned_jnums = f.align_fill_down(imp_low,
                                          imp_high,
                                          ds[[]],  # indexed with empkeys
                                          delayed_jnums)

        aligned_fur = f.align_fill_down(imp_low,
                                        imp_high,
                                        ds[[]],
                                        delayed_fur)

        # now assign "filled-down" job numbers to numpy array
        delayed_jnums[imp_low:] = aligned_jnums[imp_low:]
        delayed_fur[imp_low:] = aligned_fur[imp_low:]

        # ORIG_JOB and FUR (delayed implementation)
        # then assign numpy array values to orig_job column of integrated
        # dataset as starting point for integrated job assignments
        ds['orig_job'] = delayed_jnums
        ds['fur'] = delayed_fur

        if sdict['integrated_counts_preimp']:
            # assign combined job counts prior to the implementation date.
            # (otherwise, separate employee group counts will be used when
            # data is transferred from col_array at end of script)
            # NOTE:  this data is the actual number of jobs held within each
            # category; could be less than the number of jobs available as
            # attrition occurs
            standalone_preimp_job_counts = \
                f.make_delayed_job_counts(imp_month,
                                          delayed_jnums,
                                          low_limits,
                                          high_limits)
            col_array[arr_dict['job_count']][:imp_high] = \
                standalone_preimp_job_counts

    else:
        # set implementation month at zero for job assignment routine
        imp_month = 0

        # ORIG_JOB and FUR (no delayed implementation)
        # transfer proposal stovepipe jobs (month_form) to long_form via index
        # (empkey) alignment...
        ds['orig_job'] = df_master['orig_job']
        # developer note:  test to verify this is not instantiated elsewhere...
        ds['fur'] = df_master['fur']

    table = tdict['table']
    j_changes = tdict['j_changes']

    reduction_months = f.get_job_reduction_months(j_changes)
    # copy selected columns from ds for job assignment function input below.
    # note:  if delayed implementation, the 'fur' and 'orig_job' columns
    # contain standalone data through the implementation month.
    df_align = ds[['eg', 'sg', 'fur', 'orig_job']].copy()

    # JNUM, FUR, JOB_COUNT
    if sdict['no_bump']:

        # No bump, no flush option (includes conditions, furlough/recall,
        # job changes schedules)
        # this is the main job assignment function.  It loops through all of
        # the months in the model and assigns jobs
        nbnf, job_count, fur = \
            f.assign_jobs_nbnf_job_changes(df_align,
                                           low_limits,
                                           high_limits,
                                           all_months,
                                           reduction_months,
                                           imp_month,
                                           conditions,
                                           sdict,
                                           tdict,
                                           fur_return=sdict['recall'])

        ds['jnum'] = nbnf
        ds['job_count'] = job_count
        ds['fur'] = fur
        # for create_snum_and_spcnt_arrays function input...
        jnum_jobs = nbnf

    else:

        # Full flush and bump option (no conditions or
        # furlough/recall schedulue considered, job changes are included)
        # No bump, no flush applied up to implementation date
        fbff, job_count, fur = f.assign_jobs_full_flush_job_changes(
            nonret_each_month, table[0], num_of_job_levels)

        ds['jnum'] = fbff
        ds['job_count'] = job_count
        ds['fur'] = fur
        # for create_snum_and_spcnt_arrays function input...
        jnum_jobs = fbff

    # SNUM, SPCNT, LNUM, LSPCNT

    monthly_job_counts = table[1]

    ds['snum'], ds['spcnt'], ds['lnum'], ds['lspcnt'] = \
        f.create_snum_and_spcnt_arrays(jnum_jobs, num_of_job_levels,
                                       nonret_each_month,
                                       monthly_job_counts,
                                       lspcnt_calc)

    # RANK in JOB

    ds['rank_in_job'] = ds.groupby(['mnum', 'jnum'],
                                   sort=False).cumcount() + 1

    # JOBP

    jpcnt = (ds.rank_in_job / ds.job_count).values
    np.put(jpcnt, np.where(jpcnt == 1.0)[0], .99999)

    ds['jobp'] = ds['jnum'] + jpcnt

    # PAY - merge with pay table - provides monthly pay
    if sdict['compute_pay_measures']:

        # account for furlough time (only count active months)
        if sdict['discount_longev_for_fur']:
            # skel(ds) provides pre-calculated non-discounted scale data
            # flip ones and zeros...
            ds['non_fur'] = 1 - ds.fur.values

            non_fur = ds.groupby([pd.Grouper('empkey')])['non_fur'] \
                .cumsum().values
            ds.pop('non_fur')
            starting_mlong = ds.s_lmonths.values
            cum_active_months = non_fur + starting_mlong
            ds['mlong'] = cum_active_months
            ds['ylong'] = ds['mlong'].values / 12
            ds['scale'] = np.clip((cum_active_months / 12) + 1, 1,
                                  sdict['top_of_scale']).astype(int)

        # make a new long_form dataframe and assign a combination of
        # pay-related ds columns from large dataset as its index...
        # the dataframe is empty - we are only making an index-alignment
        # vehicle to use with indexed pay table....
        # the dataframe index contains specific scale, job, and contract year
        # for each line in long_form ds
        df_pt_index = pd.DataFrame(index=((ds['scale'].values * 100) +
                                          ds['jnum'].values +
                                          (ds['year'].values * 100000)))

        if sdict['enhanced_jobs']:
            df_pt = pd.read_pickle('dill/pay_table_enhanced.pkl')
        else:
            df_pt = pd.read_pickle('dill/pay_table_basic.pkl')

        # 'data-align' small indexed pay_table to long_form df:
        df_pt_index['monthly'] = df_pt['monthly']

        ds['monthly'] = df_pt_index.monthly.values

        # MPAY
        # adjust monthly pay for any raise and last month pay percent if
        # applicable
        ds['mpay'] = ((ds['pay_raise'].values *
                       ds['mth_pcnt'].values *
                       ds['monthly'].values)) / 1000

        ds.pop('monthly')

        # CPAY

        ds['cpay'] = ds.groupby('new_order')['mpay'].cumsum()

    if sdict['delayed_implementation']:
        ds_cols = ds.columns
        # grab each imp_col (column to insert standalone or pre-implementation
        # date data) and replace integrated data up through implementation
        # date
        for col in imp_cols:
            if col in ds_cols:
                arr = ds[col].values
                arr[:imp_high] = col_array[arr_dict[col]][:imp_high]
                ds[col] = arr

    # CAT_ORDER
    # global job ranking
    if sdict['compute_job_category_order']:
        ds['cat_order'] = f.make_cat_order(ds, table[0])

    return(ds)
