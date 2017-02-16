#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''test doc
'''

import numpy as np


def convert(sg_list, ratio_list, count_list,
            quota_dict, job_dict,
            sg_dist='part', ratio_dist='split',
            count_dist='split', quota_dist='split'):
    '''convert data relating to job assignment conditions from basic job
    level inputs to enhanced job level inputs

    inputs are the basic job level inputs from the conditions section within
    the case-specific configuration file.  this function is called within the
    case-specific configuration files when the "enhanced_job" variable
    within the general configuration file is set to "True".

    inputs
        sg_list (list)
            case-specific sg_rights variable
        ratio_list (list)
            case-specific ratio_cond variable
        count_list (list)
            case-specific count_cond variable
        quota_dict (dictionary)
            case_specific quota_dict variable
        job_dict (dictionary)
            case_specific jd variable
        sg_dist, ratio_dist, count_dist, quota_dist (string)
            options are: 'split', 'full', 'part'

            determines how jobs are distributed to the enhanced job levels.

            'split' - distribute basic job count to full- and part-time
            enhanced job levels according to the ratios set in the job
            dictionary (jd) variable

            'full' - distribute basic job count to corresponding enhanced
            full-time job levels only

            'part' - distribute basic job count to corresponding enhanced
            part-time job levels only.  This option could be selected if the
            employees with special job rights are placed in a relatively low
            position on the integrated list, eliminating the option of
            obtaining a full-time job position

            The distribution type for each condition input is independent of
            the other condition distributions.
    '''

    # these are the output variables
    enhan_sg_cond = []
    enhan_ratio_cond = []
    enhan_count_cond = []
    enhan_quota_dict = {}

    # helper dictionaries
    sg_dict = {}
    ratio_dict = {}
    count_dict = {}

    if sg_list:
        for sg_cond in sg_list:

            eg = sg_cond[0]
            job = sg_cond[1]
            count = sg_cond[2]
            start_month = sg_cond[3]
            end_month = sg_cond[4]

            full_job = int(job_dict[job][0])
            part_job = int(job_dict[job][1])
            full_pcnt = job_dict[job][2]
            part_pcnt = 1 - job_dict[job][2]

            if sg_dist == 'split':
                # grab full-time job number as key,
                # calculate count, set as value
                full_count = np.around(count * full_pcnt).astype(int)
                part_count = np.around(count * part_pcnt).astype(int)
                sg_dict[(eg, full_job)] = [eg, full_job,
                                           full_count, start_month, end_month]

                # same for part-time
                sg_dict[(eg, part_job)] = [eg, part_job,
                                           part_count, start_month, end_month]

            elif sg_dist == 'full':
                # apply entire count to full-time jobs only
                sg_dict[(eg, full_job)] = [eg, full_job,
                                           count, start_month, end_month]

            elif sg_dist == 'part':
                # apply entire count to part-time jobs only
                sg_dict[(eg, part_job)] = [eg, part_job,
                                           count, start_month, end_month]

        # sort keys and then assign corresponding values to list
        for key in sorted(sg_dict.keys()):
            enhan_sg_cond.append(sg_dict[key])

    if ratio_list:
        for r_cond in ratio_list:

            eg = r_cond[0]
            job = r_cond[1]
            start_month = r_cond[2]
            end_month = r_cond[3]

            full_job = int(job_dict[job][0])
            part_job = int(job_dict[job][1])
            full_pcnt = job_dict[job][2]
            part_pcnt = 1 - job_dict[job][2]

            # no count with the ratio data...
            if ratio_dist == 'split':
                ratio_dict[(eg, full_job)] = [eg, full_job,
                                              start_month, end_month]
                ratio_dict[(eg, part_job)] = [eg, part_job,
                                              start_month, end_month]

            elif ratio_dist == 'full':
                ratio_dict[(eg, full_job)] = [eg, full_job,
                                              start_month, end_month]

            elif ratio_dist == 'part':
                ratio_dict[(eg, part_job)] = [eg, part_job,
                                              start_month, end_month]

        for key in sorted(ratio_dict.keys()):
            enhan_ratio_cond.append(ratio_dict[key])

    if count_list:
        for job_list in count_list:

            eg = job_list[0]
            job = job_list[1]
            count = job_list[2]
            start_month = job_list[3]
            end_month = job_list[4]

            full_job = int(job_dict[job][0])
            part_job = int(job_dict[job][1])
            full_pcnt = job_dict[job][2]
            part_pcnt = 1 - job_dict[job][2]

            if count_dist == 'split':
                full_count = np.around(count * full_pcnt).astype(int)
                part_count = np.around(count * part_pcnt).astype(int)
                count_dict[(eg, full_job)] = [eg, full_job, full_count,
                                              start_month, end_month]
                count_dict[(eg, part_job)] = [eg, part_job, part_count,
                                              start_month, end_month]

            elif count_dist == 'full':
                count_dict[(eg, full_job)] = [eg, full_job, count,
                                              start_month, end_month]

            elif count_dist == 'part':
                count_dict[(eg, part_job)] = [eg, part_job, count,
                                              start_month, end_month]

        for key in sorted(count_dict.keys()):
            enhan_count_cond.append(count_dict[key])

    if quota_dict:
        for job in quota_dict.keys():

            ratio = quota_dict[job][0]
            count = quota_dict[job][1]

            full_job = int(job_dict[job][0])
            part_job = int(job_dict[job][1])
            full_pcnt = job_dict[job][2]
            part_pcnt = 1 - job_dict[job][2]

            if quota_dist == 'split':
                # grab full-time job number as key,
                # calculate count, set as value
                full_count = np.around(count * full_pcnt).astype(int)
                part_count = np.around(count * part_pcnt).astype(int)
                enhan_quota_dict[job_dict[job][0]] = (ratio, full_count)

                # same for part-time
                enhan_quota_dict[job_dict[job][1]] = (ratio, part_count)

            elif quota_dist == 'full':
                # apply entire count to full-time jobs only
                enhan_quota_dict[job_dict[job][0]] = (ratio, count)

            elif quota_dist == 'part':
                # apply entire count to part-time jobs only
                enhan_quota_dict[job_dict[job][1]] = (ratio, count)

    return (enhan_sg_cond, enhan_ratio_cond,
            enhan_count_cond, enhan_quota_dict)
