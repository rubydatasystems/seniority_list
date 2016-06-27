# -*- coding: utf-8 -*-

import config as cf

intl_blk_pcnt = .6
dom_blk_pcnt = .65
blk_average = (intl_blk_pcnt + dom_blk_pcnt) / 2

if cf.enhanced_jobs:
    num_of_job_levels = 16

    # Job dictionary for enhanced jobs conversion:
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
else:
    num_of_job_levels = 8

annual_pcnt_raise = .02
top_of_scale = 12

# JOB COUNTS
eg1_job_count = [197, 470, 1056, 412, 628, 1121, 0, 0]
eg2_job_count = [80, 85, 443, 163, 96, 464, 54, 66]
eg3_job_count = [0, 26, 319, 0, 37, 304, 0, 0]
furlough_count = [340, 0, 23]

# JOB CHANGES
# group 4 additions...
jc1 = [1, [35, 64], 43, [40, 3, 0]]
jc2 = [4, [35, 64], 72, [66, 6, 0]]
# group3 reductions...
jc3 = [2, [1, 52], -408, [-377, -23, -8]]
jc4 = [5, [1, 52], -510, [-474, -26, -10]]
# group 2 additions...
jc5 = [3, [1, 61], 411, [376, 26, 9]]
jc6 = [6, [1, 61], 411, [376, 26, 9]]

recall_1 = [8, [6, 0, 2], 50, 75]
recall_2 = [10, [10, 0, 0], 75, 150]

j_changes = [jc1, jc2, jc3, jc4, jc5, jc6]
eg_counts = [eg1_job_count, eg2_job_count, eg3_job_count]
recalls = [recall_1, recall_2]

eg_dict = {1: '1', 2: '2', 3: '3', 4: 'sa'}
eg_dict_verbose = {1: 'Group 1', 2: 'Group 2', 3: 'Group 3',
                   4: 'Standalone'}
proposal_dict = {'ds1': 'Group 1 PROPOSAL', 'ds2': 'Group 2 PROPOSAL',
                 'ds3': 'Group 3 PROPOSAL', 'ds4': 'Standalone Data'}

# detailed job labels...
if cf.enhanced_jobs:

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

# basic job labels...
else:

    job_strs = ['Capt G4', 'Capt G3', 'Capt G2', 'F/O  G4', 'F/O  G3',
                'F/O  G2', 'Capt G1', 'F/O  G1', 'FUR']

    jobs_dict = {1: 'Capt G4', 2: 'Capt G3', 3: 'Capt G2', 4: 'F/O  G4',
                 5: 'F/O  G3', 6: 'F/O  G2', 7: 'Capt G1', 8: 'F/O  G1',
                 9: 'FUR'}

if cf.enhanced_jobs:  # enhanced job level chart colors (16)

    grp1_color = ['#008f00', '#006600', '#7b1979', '#0433ff', '#551154',
                  '#4d6dff', '#ff7c00', '#ffa34d', '#ff2600', '#fffb00',
                  '#9ec2e9', '#ff8080', '#fffd99', '#c1d8f1', '#e2a8a5',
                  '#f3dad8', '#ffb3ff']

    grp2_color = ['#5091cf', '#2492f9', '#b33932', '#9aba5a', '#da160b',
                  '#88cc00', '#00b2ec', '#37cafb', '#b0a2cb', '#b6dde9',
                  '#ef994d', '#a57ef1', '#a6b9bf', '#ffa64d', '#c7bb9a',
                  '#c8b79d', '#fefc05']

    grp3_color = ['#3064a4', '#3399ff', '#aa322f', '#81a33a', '#ff1a1a',
                  '#88cc00', '#66488a', '#9679b9', '#2c93af', '#dd7621',
                  '#7b95c1', '#5cbcd6', '#e89f64', '#ffa64d', '#c67c7b',
                  '#c8b79d', '#fefc05']

    color_lists = [grp1_color, grp2_color, grp3_color]

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

else:  # basic 8-level chart colors

    grp1_color = ['#008f00', '#7b1979', '#0433ff', '#ff7c00', '#ff2600',
                  '#fffb00', '#9ec2e9', '#e2a8a5', '#ffb3ff']

    grp2_color = ['#5091cf', '#b33932', '#9aba5a', '#00b2ec', '#b0a2cb',
                  '#b6dde9', '#ef994d', '#c7bb9a', '#fefc05']

    grp3_color = ['#3064a4', '#aa322f', '#81a33a', '#66488a', '#2c93af',
                  '#dd7621', '#7b95c1', '#c67c7b', '#fefc05']

    color_lists = [grp1_color, grp2_color, grp3_color]

    job_colors = [[0.65, 0.8, 0.89, 1.],
                  [0.14, 0.48, 0.7, 1.],
                  [0.66, 0.85, 0.51, 1.],
                  [0.28, 0.62, 0.21, 1.],
                  [0.97, 0.53, 0.53, 1.],
                  [0.9, 0.21, 0.16, 1.],
                  [0.99, 0.79, 0.49, 1.],
                  [0.94, 0.54, 0.2, 1.],
                  [0.4, 0.4, 0.4, 1.]]

eg_colors = ["#505050", "#0081ff", "#ff6600", '#CC00FF']
lin_reg_colors = ['#00b300', '#0086b3', '#cc5200']
lin_reg_colors2 = ['grey', '#0086b3', '#cc5200']
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

# for chart lable adjustment (secondary y label positioning)
if cf.enhanced_jobs:
    adjust = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -75, 50, 0, -160, -40, 120, 0]
else:
    adjust = [0, 0, 0, 0, 0, 0, -50, 50, 0]

# reference only:

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