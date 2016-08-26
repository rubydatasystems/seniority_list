# -*- coding: utf-8 -*-

import importlib

compute_with_job_changes = True
discount_longev_for_fur = True
lspcnt_calc_on_remaining_population = False

case_study = 'sample3'

this_module = 'case_files.' + case_study

enhanced_jobs = True

# ********************************************************************
# This import must come after any variable assignment that is used by the
# case-specific modules...
case = importlib.import_module(this_module)
# ********************************************************************

num_of_job_levels = case.num_of_job_levels

# full_time_pcnt1 also used for count group 4 cond calculation when
# model contains 16 job levels (assign_cond_ratio_capped function)
full_time_pcnt1 = case.full_time_pcnt1
full_time_pcnt2 = case.full_time_pcnt2

starting_date = case.starting_date
start = case.start

# for separate job assignment calculations:
delayed_implementation = True

# implementation date is used for proposal conditions even if
# 'delayed implementation' above is False
implementation_date = case.implementation_date

# imp below refers to 'implementation'
imp_date = case.imp_date
imp_month = case.imp_month
end_date = case.end_date
ratio_final_month = case.ratio_final_month

no_bump = True
recall = True

init_ret_age_years = case.init_ret_age_years
init_ret_age_months = case.init_ret_age_months

init_ret_age = case.init_ret_age
ret_age = case.ret_age
ret_age_increase = case.ret_age_increase
ret_incr_dict = case.ret_incr_dict

actives_only = False
save_to_pickle = True

add_eg_col = True
add_retdate_col = True
add_doh_col = True
add_ldate_col = True
add_lname_col = True
add_line_col = True
add_sg_col = True
add_ret_mark = True

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

# Job dictionary for enhanced jobs conversion:
if enhanced_jobs:
    jd = case.jd

# JOB ASSIGNMENT CONDITIONS DATA
############################################
ratio_cond = case.ratio_cond
sg_rights = case.sg_rights
count_cond = case.count_cond
quota_dict = case.quota_dict
############################################

# chart label adjustment
adjust = case.adjust

# colors, dicts (matches case jobs and proposal descriptions)
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

white_grey = case.white_grey

color1 = case.color1
color2 = case.color2
color3 = case.color3
