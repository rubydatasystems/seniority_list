# -*- coding: utf-8 -*-

import importlib

# OK to adjust variables within area marked by asterisks:
# *****************************************************************************
enhanced_jobs = True
case_study = 'aa_award'
# *****************************************************************************


# Do not change this code within area marked by hyphens:
# -----------------------------------------------------------------------------
# This import must come after any variable assignment that is imported
# by the case-specific modules...(i.e. 'enhanced jobs')
# Do not change this code
this_module = 'case_files.' + case_study
case = importlib.import_module(this_module)
# -----------------------------------------------------------------------------


# OK to adjust variables within area marked by asterisks:
# *****************************************************************************
compute_with_job_changes = True
no_bump = True

recall = True
discount_longev_for_fur = True

# calculate list percentage based on employees remaining
# in each month including furloughees (True), otherwise
# percentage calculation denominator is the greater of
# pilots remaining (incl fur) or jobs available (False)
# for 'lspcnt' column
lspcnt_calc_on_remaining_population = False

# for unmerged job assignment calculations:
delayed_implementation = True

save_to_pickle = True

add_eg_col = True
add_retdate_col = True
add_doh_col = True
add_ldate_col = True
add_lname_col = True
add_line_col = True
add_sg_col = True
add_ret_mark = True
compute_job_category_order = True

# PAY (measures)
compute_pay_measures = True
# pay_raise = False

# *****************************************************************************


# DO NOT CHANGE ANY CODE BELOW.
# These variable assignments are imported from the case-specific config file:
# -----------------------------------------------------------------------------

num_of_job_levels = case.num_of_job_levels

# full_time_pcnt1 also used for count cond calculation when
# model contains 16 job levels (assign_cond_ratio_capped function)
full_time_pcnt1 = case.full_time_pcnt1
full_time_pcnt2 = case.full_time_pcnt2

starting_date = case.starting_date
start = case.start

# implementation date is used for proposal conditions even if
# 'delayed implementation' above is False
implementation_date = case.implementation_date

pay_table_exception_year = case.pay_table_exception_year
date_exception_start = case.date_exception_start
date_exception_end = case.date_exception_end
last_contract_year = case.last_contract_year
future_raise = case.future_raise
annual_pcnt_raise = case.annual_pcnt_raise

# imp below refers to 'implementation'
imp_date = case.imp_date
imp_month = case.imp_month
# end_date = case.end_date
ratio_final_month = case.ratio_final_month

init_ret_age_years = case.init_ret_age_years
init_ret_age_months = case.init_ret_age_months

init_ret_age = case.init_ret_age
ret_age = case.ret_age
ret_age_increase = case.ret_age_increase
ret_incr_dict = case.ret_incr_dict


# PAY RELATED (skeleton):

annual_pcnt_raise = case.annual_pcnt_raise
top_of_scale = case.top_of_scale

try:
    if compute_with_job_changes:
        j_changes = case.j_changes
    else:
        j_changes = case.j_changes[:]
        replace_1 = 0
        replace_2 = [0, 0, 0]

        for i in range(len(j_changes)):
            j_changes[i][2] = replace_1
            j_changes[i][3] = replace_2
except:
    j_changes = [[1, [10, 20], 0, [0, 0, 0]]]

eg_counts = case.eg_counts
furlough_count = case.furlough_count
recalls = case.recalls

# Job dictionary for enhanced jobs conversion:
if enhanced_jobs:
    jd = case.jd

# JOB ASSIGNMENT CONDITIONS DATA
ratio_cond = case.ratio_cond
sg_rights = case.sg_rights
count_cond = case.count_cond
quota_dict = case.quota_dict

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

white_grey = case.white_grey

color1 = case.color1
color2 = case.color2
color3 = case.color3
# -----------------------------------------------------------------------------
