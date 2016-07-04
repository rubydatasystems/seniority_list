# -*- coding: utf-8 -*-

import pandas as pd
import importlib

compute_with_job_changes = True
discount_longev_for_fur = True
lspcnt_calc_on_remaining_population = False

case_study = 'sample3'

enhanced_jobs = True

edit_mode = False

# ********************************************************************
# This impport must come after any variables that need to be used by the
# case-specific modules...
case = importlib.import_module(case_study)
# ********************************************************************

num_of_job_levels = case.num_of_job_levels

full_time_pcnt1 = case.full_time_pcnt1
full_time_pcnt2 = case.full_time_pcnt2

# CONDITIONS
apply_supc = True
apply_count_cond = True
apply_ratio_cond = True

starting_date = '2013-12-31'
start = pd.to_datetime(starting_date)
# for separate job assignment calculations:
delayed_implementation = True
# imp_month = 33

# for 8 to 16 job level conversion.
# full_time_pcnt1 also used for count group 4 cond calculation when
# model contains 16 job levels.
# full_time_pcnt1 = .6
# full_time_pcnt2 = .65
# full_time_avg_pcnt = (full_time_pcnt1 + full_time_pcnt2) / 2

# this is used for proposal conditions even if
# 'delayed implementation above is False'
implementation_date = '2016-10-31'
imp_date = pd.to_datetime(implementation_date)
imp_month = ((imp_date.year - start.year) * 12) - \
    (start.month - imp_date.month)

if apply_ratio_cond:

    end_date = pd.to_datetime('2020-01-31')
    ratio_final_month = ((end_date.year - start.year) * 12) - \
        (start.month - end_date.month)

no_bump = True
recall = True

ret_age = 65
actives_only = False
save_to_pickle = True

add_eg_col = True
add_retdate_col = True
add_doh_col = True
add_ldate_col = True
add_lname_col = True
add_line_col = True
add_sg_col = True

# PAY RELATED (skeleton):

pay_raise = False
annual_pcnt_raise = case.annual_pcnt_raise
top_of_scale = case.top_of_scale

compute_job_category_order = True

# PAY (measures)

compute_pay_measures = True

j_changes = case.j_changes
eg_counts = case.eg_counts
furlough_count = case.furlough_count
recalls = case.recalls

adjust = case.adjust

if enhanced_jobs:
    # Job dictionary for enhanced jobs conversion:
    jd = case.jd

    if apply_ratio_cond:
        # eg1 group 4 C&R
        # sequence = [eg, jnum, pcnt, start_month, end_month]
        eg1_cr1 = [1, 1, imp_month, ratio_final_month]
        eg1_cr2 = [1, 2, imp_month, ratio_final_month]
        eg1_cr7 = [1, 7, imp_month, ratio_final_month]
        eg1_cr8 = [1, 8, imp_month, ratio_final_month]

        ratio_cond = [eg1_cr1, eg1_cr2, eg1_cr7, eg1_cr8]

    if case_study == 'sample3':  # use sample data, enhanced jobs...

        if apply_supc:
            # eg1 sg sup cc award (all reserve...)
            # sequence = [eg, jnum, count, start_month, end_month]
            eg1_sg5 = [1, 5, 43, 0, 67]
            eg1_sg6 = [1, 6, 130, 0, 67]
            eg1_sg12 = [1, 12, 43, 0, 67]
            eg1_sg13 = [1, 13, 130, 0, 67]

        if apply_count_cond:
            # eg2 group 4 C&R (split block and reserve...):
            # sequence = [eg, jnum, count, start_month, end_month]
            eg2_cr1 = [2, 1, 55, imp_month, imp_month + 60]
            eg2_cr2 = [2, 2, 37, imp_month, imp_month + 60]
            eg2_cr7 = [2, 7, 101, imp_month, imp_month + 60]
            eg2_cr8 = [2, 8, 67, imp_month, imp_month + 60]

    if case_study == 'aa_us':

        if apply_supc:
            # eg1 sg sup cc award (all reserve...)
            # sequence = [eg, jnum, count, start_month, end_month]
            eg1_sg5 = [1, 5, 86, 0, 67]
            eg1_sg6 = [1, 6, 260, 0, 67]
            eg1_sg12 = [1, 12, 86, 0, 67]
            eg1_sg13 = [1, 13, 260, 0, 67]

        if apply_count_cond:
            # eg2 group 4 C&R (split block and reserve...):
            # sequence = [eg, jnum, count, start_month, end_month]
            eg2_cr1 = [2, 1, 110, imp_month, imp_month + 60]
            eg2_cr2 = [2, 2, 73, imp_month, imp_month + 60]
            eg2_cr7 = [2, 7, 202, imp_month, imp_month + 60]
            eg2_cr8 = [2, 8, 134, imp_month, imp_month + 60]

    sg_rights = [eg1_sg5, eg1_sg6, eg1_sg12, eg1_sg13]
    count_cond = [eg2_cr1, eg2_cr2, eg2_cr7, eg2_cr8]

else:  # basic job levels only:

    if apply_ratio_cond:
        # eg1 group 4 C&R
        # sequence = [eg, jnum, pcnt, start_month, end_month]
        eg1_cr1 = [1, 1, imp_month, ratio_final_month]
        eg1_cr4 = [1, 4, imp_month, ratio_final_month]

        ratio_cond = [eg1_cr1, eg1_cr4]

    if case_study == 'sample3':  # use sample data, enhanced jobs...

        if apply_supc:
            # eg1 sg sup cc award
            # sequence = [eg, jnum, count, start_month, end_month]
            eg1_sg2 = [1, 2, 43, 0, 67]
            eg1_sg3 = [1, 3, 130, 0, 67]
            eg1_sg5 = [1, 5, 43, 0, 67]
            eg1_sg6 = [1, 6, 130, 0, 67]

        if apply_count_cond:
            # eg2 group 4 C&R
            # sequence = [eg, jnum, count, start_month, end_month]
            eg2_cr1 = [2, 1, 92, imp_month, imp_month + 60]
            eg2_cr4 = [2, 4, 168, imp_month, imp_month + 60]

    if case_study == 'aa_us':

        if apply_supc:
            # eg1 sg sup cc award
            # sequence = [eg, jnum, count, start_month, end_month]
            eg1_sg2 = [1, 2, 86, 0, 67]
            eg1_sg3 = [1, 3, 260, 0, 67]
            eg1_sg5 = [1, 5, 86, 0, 67]
            eg1_sg6 = [1, 6, 260, 0, 67]

        if apply_count_cond:
            # eg2 group 4 C&R
            # sequence = [eg, jnum, count, start_month, end_month]
            eg2_cr1 = [2, 1, 183, imp_month, imp_month + 60]
            eg2_cr4 = [2, 4, 336, imp_month, imp_month + 60]

    sg_rights = [eg1_sg2, eg1_sg3, eg1_sg5, eg1_sg6]
    count_cond = [eg2_cr1, eg2_cr4]

############################################
# colors, dicts

eg_dict = case.eg_dict
eg_dict_verbose = case.eg_dict_verbose
proposal_dict = case.proposal_dict

job_strs = case.job_strs
jobs_dict = case.jobs_dict

color_lists = case.color_lists
job_colors = case.job_colors


eg_colors = case.eg_colors
lin_reg_colors = case.lin_reg_colors
lin_reg_colors2 = case.lin_reg_colors2
mean_colors = case.mean_colors

row_colors = case.row_colors

# alternate white, grey
white_grey = case.white_grey

color1 = case.color1
color2 = case.color2
color3 = case.color3


