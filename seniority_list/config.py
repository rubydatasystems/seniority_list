# -*- coding: utf-8 -*-

import pandas as pd

compute_with_job_changes = True
discount_longev_for_fur = True
lspcnt_calc_on_remaining_population = False

edit_mode = False
# CONDITIONS
apply_supc = True
apply_east_cond = True
apply_amer_cond = True

starting_date = '2013-12-31'
start = pd.to_datetime(starting_date)
# for separate job assignment calculations:
delayed_implementation = True
# imp_month = 33

# for 8 to 16 job level conversion.
# intl_blk_pcnt also used for east group 4 cond calculation when
# model contains 16 job levels.
intl_blk_pcnt = .6
dom_blk_pcnt = .65
blk_average = (intl_blk_pcnt + dom_blk_pcnt) / 2

# this is used for proposal conditions even if
# 'delayed implementation above is False'
implementation_date = '2016-10-31'
imp_date = pd.to_datetime(implementation_date)
imp_month = ((imp_date.year - start.year) * 12) - \
    (start.month - imp_date.month)

if apply_amer_cond:

    end_date = pd.to_datetime('2020-01-31')
    amer_final_month = ((end_date.year - start.year) * 12) - \
        (start.month - end_date.month)

no_bump = True

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
annual_pcnt_raise = .02

# number of job levels for model excluding furlough
num_of_job_levels = 16

compute_job_category_order = True

# PAY (measures)

compute_pay_measures = True

# if delayed_implementation:
#     g4 = [1.00, 0.00, 0.00]
#     g3 = [.83, 0.15, 0.02]
#     g2 = [.79, 0.18, 0.03]
#     g1 = [0.00, 1.00, 0.00]

# job changes
# [job level affected, [start and end month], total change,
# [standalone allocation]]
# MASTER ENTRY...16 job level changes are calculated from these 8-level lists
# group 4 additions...
jc1 = [1, [35, 64], 87, [80, 7, 0]]
jc2 = [4, [35, 64], 145, [133, 12, 0]]
# group3 reductions...
jc3 = [2, [1, 52], -816, [-754, -46, -16]]
jc4 = [5, [1, 52], -1020, [-948, -52, -20]]
# group 2 additions...
jc5 = [3, [1, 61], 822, [747, 57, 18]]
jc6 = [6, [1, 61], 822, [747, 57, 18]]

j_changes = [jc1, jc2, jc3, jc4, jc5, jc6]

# testing
# group 4 additions...
# jc1 = [1, [35, 64], 87, [80, 7, 0]]
# jc2 = [4, [35, 64], 145, [133, 12, 0]]
# group3 reductions...
# jc3 = [2, [1, 52], -816, [-754, -46, -16]]
# jc4 = [5, [1, 52], -1020, [-948, -52, -20]]
# group 2 reductions...
# jc5 = [3, [1, 61], -822, [-747, -57, -18]]
# jc6 = [6, [1, 61], -822, [-747, -57, -18]]

# j_changes = [jc1, jc2, jc3, jc4, jc5, jc6]

# eg job count lists
# MASTER ENTRY...16 job level counts are calculated from these 8-level lists
eg1_job_count = [395, 939, 2112, 825, 1255, 2242, 0, 0]
eg2_job_count = [161, 170, 885, 326, 192, 928, 109, 132]
eg3_job_count = [0, 53, 637, 0, 74, 608, 0, 0]
furlough_count = [681, 0, 45]

eg_counts = [eg1_job_count, eg2_job_count, eg3_job_count]

# RECALL
recall = True
# recall schedule
# format [total monthly_recall_count, eg recall allocation,
#           start_month, end_month]
recall_1 = [8, [6, 0, 2], 50, 75]
recall_2 = [10, [10, 0, 0], 75, 150]

recalls = [recall_1, recall_2]

top_of_scale = 12

if num_of_job_levels == 8:

    if apply_amer_cond:
        # eg1 group 4 C&R
        # sequence = [eg, jnum, pcnt, start_month, end_month]
        eg1_cr1 = [1, 1, imp_month, amer_final_month]
        eg1_cr4 = [1, 4, imp_month, amer_final_month]

        amr_g4_cond = [eg1_cr1, eg1_cr4]

    if apply_supc:
        # eg1 sg sup cc award
        # sequence = [eg, jnum, start_month, end_month]
        eg1_sg2 = [1, 2, 86, 0, 67]
        eg1_sg3 = [1, 3, 260, 0, 67]
        eg1_sg5 = [1, 5, 86, 0, 67]
        eg1_sg6 = [1, 6, 260, 0, 67]

        sg_rights = [eg1_sg2, eg1_sg3, eg1_sg5, eg1_sg6]

    if apply_east_cond:
        # eg2 group 4 C&R
        # sequence = [eg, jnum, count, start_month, end_month]
        eg2_cr1 = [2, 1, 183, imp_month, imp_month + 60]
        eg2_cr4 = [2, 4, 336, imp_month, imp_month + 60]

        east_g4_cond = [eg2_cr1, eg2_cr4]

if num_of_job_levels == 16:

    # job dictionary for 8 to 16 job level conversion
    jd = {
        1: [1, 2, intl_blk_pcnt],
        2: [3, 5, blk_average],
        3: [4, 6, dom_blk_pcnt],
        4: [7, 8, intl_blk_pcnt],
        5: [9, 12, blk_average],
        6: [10, 13, dom_blk_pcnt],
        7: [11, 14, dom_blk_pcnt],
        8: [15, 16, dom_blk_pcnt]
    }

    # eg1_job_count = [237, 158, 563, 1373, 376, 739, 495, 330, 753,
    #                  1457, 0, 502, 785, 0, 0, 0]
    # eg2_job_count = [97, 64, 102, 575, 68, 310, 196, 130, 115, 603,
    #                  71, 77, 325, 38, 86, 46]
    # eg3_job_count = [0, 0, 32, 414, 21, 223, 0, 0, 44, 395, 0, 30,
    #                  213, 0, 0, 0]
    # furlough_count = [681, 0, 45]
    # eg_counts = [eg1_job_count, eg2_job_count, eg3_job_count]

    # with east starting counts (approx.):
    # eg1_job_count = [272, 182, 548, 1373, 365, 739 ,497,
    # 329, 734, 1464,  0, 487, 778,  0,  0  ,0]
    # eg2_job_count = [110,  73,  93,  576,  62, 309, 202,
    # 134, 106,  599, 71,  69, 329, 38, 74, 58]

    if apply_amer_cond:
        # eg1 group 4 C&R
        # sequence = [eg, jnum, pcnt, start_month, end_month]
        eg1_cr1 = [1, 1, imp_month, amer_final_month]
        eg1_cr2 = [1, 2, imp_month, amer_final_month]
        eg1_cr7 = [1, 7, imp_month, amer_final_month]
        eg1_cr8 = [1, 8, imp_month, amer_final_month]

        amr_g4_cond = [eg1_cr1, eg1_cr2, eg1_cr7, eg1_cr8]

    if apply_supc:
        # eg1 sg sup cc award (all reserve...)
        # sequence = [eg, jnum, start_month, end_month]
        eg1_sg5 = [1, 5, 86, 0, 67]
        eg1_sg6 = [1, 6, 260, 0, 67]
        eg1_sg12 = [1, 12, 86, 0, 67]
        eg1_sg13 = [1, 13, 260, 0, 67]

        sg_rights = [eg1_sg5, eg1_sg6, eg1_sg12, eg1_sg13]

    if apply_east_cond:
        # eg2 group 4 C&R (split block and reserve...):
        # sequence = [eg, jnum, count, start_month, end_month]
        eg2_cr1 = [2, 1, 110, imp_month, imp_month + 60]
        eg2_cr2 = [2, 2, 73, imp_month, imp_month + 60]
        eg2_cr7 = [2, 7, 202, imp_month, imp_month + 60]
        eg2_cr8 = [2, 8, 134, imp_month, imp_month + 60]

        east_g4_cond = [eg2_cr1, eg2_cr2, eg2_cr7, eg2_cr8]

# This is not used:

# job_num,job_code
# 1.0,c4b
# 2.0,c4r
# 3.0,c3b
# 4.0,c2b
# 5.0,c3r
# 6.0,c2r
# 7.0,f4b
# 8.0,f4r
# 9.0,f3b
# 10.0,f2b
# 11.0,c1b
# 12.0,f3r
# 13.0,f2r
# 14.0,c1r
# 15.0,f1b
# 16.0,f1r

# job_num,job_code
# 1.0,c4
# 2.0,c3
# 3.0,c2
# 4.0,f4
# 5.0,f3
# 6.0,f2
# 7.0,c1
# 8.0,f1

############################################

eg_dict = {1: 'A', 2: 'E', 3: 'W', 4: 'S'}
eg_dict_verbose = {1: 'AMER', 2: 'EAST', 3: 'WEST', 4: 'Standalone'}
proposal_dict = {'ds1': 'APSIC PROPOSAL', 'ds2': 'EAST PROPOSAL',
                 'ds3': 'WEST PROPOSAL', 'ds4': 'Standalone Data'}

if num_of_job_levels == 16:

    job_strs = ['Capt G4 B', 'Capt G4 R', 'Capt G3 B', 'Capt G2 B',
                'Capt G3 R', 'Capt G2 R', 'F/O  G4 B', 'F/O  G4 R',
                'F/O  G3 B', 'F/O  G2 B', 'Capt G1 B', 'F/O  G3 R',
                'F/O  G2 R', 'Capt G1 R', 'F/O  G1 B', 'F/O  G1 R', 'FUR']
    jobs_dict = {1: 'Capt G4 B', 2: 'Capt G4 R', 3: 'Capt G3 B',
                 4: 'Capt G2 B', 5: 'Capt G3 R', 6: 'Capt G2 R',
                 7: 'F/O  G4 B', 8: 'F/O  G4 R', 9: 'F/O  G3 B',
                 10: 'F/O  G2 B', 11: 'Capt G1 B', 12: 'F/O  G3 R',
                 13: 'F/O  G2 R', 14: 'Capt G1 R', 15: 'F/O  G1 B',
                 16: 'F/O  G1 R', 17: 'FUR'}

if num_of_job_levels == 8:

    job_strs = ['Capt G4', 'Capt G3', 'Capt G2', 'F/O  G4', 'F/O  G3',
                'F/O  G2', 'Capt G1', 'F/O  G1', 'FUR']
    jobs_dict = {1: 'Capt G4', 2: 'Capt G3', 3: 'Capt G2', 4: 'F/O  G4',
                 5: 'F/O  G3', 6: 'F/O  G2', 7: 'Capt G1', 8: 'F/O  G1',
                 9: 'FUR'}

# colors

# eg_colors = ['k','b','#FF6600', '#CC00FF']#ff6600
# eg_colors = ["#505050", "#0081ff", "#FF4500", '#CC00FF']
eg_colors = ["#505050", "#0081ff", "#ff6600", '#CC00FF']
lr_colors = ['#00b300', '#0086b3', '#cc5200']
lr_colors2 = ['grey', '#0086b3', '#cc5200']
mean_colors = ['#4d4d4d', '#3399ff', '#ff8000']
row_colors = ['#ffffe6', '#404040', '#3399ff', '#ff8000', '#00cc44',
              '#b800e6', '#ff0000', '#996633', '#ff99ff']

# alternate white, grey
white_grey = ['#999999', '#ffffff', '#999999', '#ffffff', '#999999',
              '#ffffff', '#999999', '#ffffff', '#999999', '#ffffff',
              '#999999', '#ffffff', '#999999', '#ffffff', '#999999',
              '#ffffff', '#999999']

color1 = ['#00ff00', '#80ff80', '#cc3300', '#005ce6', '#ff8c66',
          '#3384ff', '#00ff00', '#80ff80', '#cc3300', '#005ce6',
          '#ffff00', '#ff8c66', '#3384ff', '#ffff00', '#e600e5',
          '#ff66fe', '#ff0000']

color2 = ['#ff0066', '#ff4d94', '#ffff00', '#00b2b3', '#ffff99',
          '#00e4e6', '#ff0066', '#ff4d94', '#ffff00', '#00b2b3',
          '#6600cc', '#ffff99', '#00e4e6', '#ffff00', '#8c1aff',
          '#8c1aff', '#ff0000']

color3 = ['#ff0066', '#ff4d94', '#0033cc', '#ffff00', '#0040ff',
          '#ffff99', '#ff0066', '#ff4d94', '#0033cc', '#ffff00',
          '#00cc00', '#0040ff', '#ffff99', '#00e600', '#00cc00',
          '#00e600', '#333333']

# amer color
if num_of_job_levels == 16:
    amer_color = ['#008f00', '#006600', '#7b1979', '#0433ff', '#551154',
                  '#4d6dff', '#ff7c00', '#ffa34d', '#ff2600', '#fffb00',
                  '#9ec2e9', '#ff8080', '#fffd99', '#c1d8f1', '#e2a8a5',
                  '#f3dad8', '#ffb3ff']
if num_of_job_levels == 8:
    amer_color = ['#008f00', '#7b1979', '#0433ff', '#ff7c00', '#ff2600',
                  '#fffb00', '#9ec2e9', '#e2a8a5', '#ffb3ff']

# east_color
if num_of_job_levels == 16:
    east_color = ['#5091cf', '#2492f9', '#b33932', '#9aba5a', '#da160b',
                  '#88cc00', '#00b2ec', '#37cafb', '#b0a2cb', '#b6dde9',
                  '#ef994d', '#a57ef1', '#a6b9bf', '#ffa64d', '#c7bb9a',
                  '#c8b79d', '#fefc05']
if num_of_job_levels == 8:
    east_color = ['#5091cf', '#b33932', '#9aba5a', '#00b2ec', '#b0a2cb',
                  '#b6dde9', '#ef994d', '#c7bb9a', '#fefc05']

# west color
if num_of_job_levels == 16:
    west_color = ['#3064a4', '#3399ff', '#aa322f', '#81a33a', '#ff1a1a',
                  '#88cc00', '#66488a', '#9679b9', '#2c93af', '#dd7621',
                  '#7b95c1', '#5cbcd6', '#e89f64', '#ffa64d', '#c67c7b',
                  '#c8b79d', '#fefc05']
if num_of_job_levels == 8:
    west_color = ['#3064a4', '#aa322f', '#81a33a', '#66488a', '#2c93af',
                  '#dd7621', '#7b95c1', '#c67c7b', '#fefc05']

if num_of_job_levels == 16:
    job_colors = [
        [0.65, 0.81, 0.89, 1.0],
        [0.31, 0.59, 0.77, 1.0],
        [0.19, 0.39, 0.7, 1.0],
        [0.66, 0.85, 0.55, 1.0],
        [0.41, 0.73, 0.32, 1.0],
        [0.22, 0.6, 0.23, 1.0],
        [0.93, 0.61, 0.57, 1.0],
        [0.93, 0.32, 0.32, 1.0],
        [0.75, 0.1, 0.1, 1.0],
        [0.99, 0.79, 0.49, 1.0],
        [0.95, 0.65, 0.19, 1.0],
        [0.82, 0.42, 0.12, 1.0],
        [0.82, 0.67, 0.71, 1.0],
        [0.6, 0.47, 0.72, 1.0],
        [0.5, 0.35, 0.6, 1.0],
        [0.9, 0.87, 0.6, 1.0],
        [0.4, 0.4, 0.4, 1.0]]

if num_of_job_levels == 8:
    job_colors = [[0.65, 0.8, 0.89, 1.],
                  [0.14, 0.48, 0.7, 1.],
                  [0.66, 0.85, 0.51, 1.],
                  [0.28, 0.62, 0.21, 1.],
                  [0.97, 0.53, 0.53, 1.],
                  [0.9, 0.21, 0.16, 1.],
                  [0.99, 0.79, 0.49, 1.],
                  [0.94, 0.54, 0.2, 1.],
                  [0.4, 0.4, 0.4, 1.]]
