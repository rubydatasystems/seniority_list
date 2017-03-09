#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''test doc
'''

import numpy as np


def convert(job_dict=None,
            sg_list=None,
            ratio_list=None,
            count_list=None,
            quota_dict=None,
            count_ratio_dict=None,
            dist_sg=None,
            dist_ratio=None,
            dist_count=None,
            dist_quota=None,
            dist_count_ratio=None):
    '''Convert data relating to job assignment conditions from basic job
    level inputs to enhanced job level inputs

    Inputs are the basic job level values for the various conditions, the
    job dictionary, and the distribution methods used during conversion.

    This function is called within the build_program_files script when the
    "enhanced_job" key value within the settings dictionary is set to "True".

    inputs
        job_dict (dictionary)
            case_specific jd variable.  This input contains full-time job
            level conversion percentages
        sg_list (list)
            case-specific sg_rights variable
        ratio_list (list)
            case-specific ratio_cond variable
        count_list (list)
            case-specific count_cond variable
        quota_dict (dictionary)
            case_specific quota_dict variable
        count_ratio_dict (dictionary)
            dictionary containing all data related to a capped ratio or
            count ratio condition (will replace count list and quota_dict
            as program is developed further)

        dist_sg, dist_ratio, dist_count, dist_quota (string)
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

            If these variables are not assigned, the program will default
            to "split".
    '''

    if job_dict is None:
        print('Please set job_dict variable - the job dictionary is ' +
              'required to convert from basic to enhanced jobs')
        return

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

            if dist_sg is None:
                dist_sg = 'split'

            if dist_sg == 'split':
                # grab full-time job number as key,
                # calculate count, set as value
                full_count = np.around(count * full_pcnt).astype(int)
                part_count = np.around(count * part_pcnt).astype(int)
                sg_dict[(eg, full_job)] = [eg, full_job,
                                           full_count, start_month, end_month]

                # same for part-time
                sg_dict[(eg, part_job)] = [eg, part_job,
                                           part_count, start_month, end_month]

            elif dist_sg == 'full':
                # apply entire count to full-time jobs only
                sg_dict[(eg, full_job)] = [eg, full_job,
                                           count, start_month, end_month]

            elif dist_sg == 'part':
                # apply entire count to part-time jobs only
                sg_dict[(eg, part_job)] = [eg, part_job,
                                           count, start_month, end_month]

        # sort keys and then assign corresponding values to list
        for key in sorted(sg_dict.keys()):
            enhan_sg_cond.append(sg_dict[key])
    else:
        enhan_sg_cond = 0

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

            if dist_ratio is None:
                dist_ratio = 'split'

            # no count with the ratio data...
            if dist_ratio == 'split':
                ratio_dict[(eg, full_job)] = [eg, full_job,
                                              start_month, end_month]
                ratio_dict[(eg, part_job)] = [eg, part_job,
                                              start_month, end_month]

            elif dist_ratio == 'full':
                ratio_dict[(eg, full_job)] = [eg, full_job,
                                              start_month, end_month]

            elif dist_ratio == 'part':
                ratio_dict[(eg, part_job)] = [eg, part_job,
                                              start_month, end_month]

        for key in sorted(ratio_dict.keys()):
            enhan_ratio_cond.append(ratio_dict[key])
    else:
        enhan_ratio_cond = 0

# count_ratio_dict

    if count_ratio_dict:
        temp_dict = {}
        enhan_count_dict = {}
        for job in count_ratio_dict.keys():

            job_data = count_ratio_dict[job]

            grp = job_data[0]
            wgt = job_data[1]
            cap = job_data[2]
            start_month = job_data[3]
            end_month = job_data[4]

            full_job = int(job_dict[job][0])
            part_job = int(job_dict[job][1])
            full_pcnt = job_dict[job][2]
            part_pcnt = 1 - job_dict[job][2]

            if dist_count_ratio is None:
                dist_count_ratio = 'split'

            # no count with the ratio data...
            if dist_count_ratio == 'split':
                temp_dict[full_job] = [grp, wgt,
                                       int(round(cap * full_pcnt)),
                                       start_month, end_month]
                temp_dict[part_job] = [grp, wgt,
                                       int(round(cap * part_pcnt)),
                                       start_month, end_month]

            elif dist_count_ratio == 'full':
                temp_dict[full_job] = [grp, wgt, cap,
                                       start_month, end_month]

            elif dist_count_ratio == 'part':
                temp_dict[part_job] = [grp, wgt, cap,
                                       start_month, end_month]

        for key in sorted(temp_dict.keys()):
            enhan_count_dict[key] = temp_dict[key]
    else:
        enhan_count_dict = 0

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

            if dist_count is None:
                dist_count = 'split'

            if dist_count == 'split':
                full_count = np.around(count * full_pcnt).astype(int)
                part_count = np.around(count * part_pcnt).astype(int)
                count_dict[(eg, full_job)] = [eg, full_job, full_count,
                                              start_month, end_month]
                count_dict[(eg, part_job)] = [eg, part_job, part_count,
                                              start_month, end_month]

            elif dist_count == 'full':
                count_dict[(eg, full_job)] = [eg, full_job, count,
                                              start_month, end_month]

            elif dist_count == 'part':
                count_dict[(eg, part_job)] = [eg, part_job, count,
                                              start_month, end_month]

        for key in sorted(count_dict.keys()):
            enhan_count_cond.append(count_dict[key])
    else:
        enhan_count_cond = 0

    if quota_dict:
        for job in quota_dict.keys():

            ratio = quota_dict[job][0]
            count = quota_dict[job][1]

            full_job = int(job_dict[job][0])
            part_job = int(job_dict[job][1])
            full_pcnt = job_dict[job][2]
            part_pcnt = 1 - job_dict[job][2]

            if dist_quota is None:
                dist_quota = 'split'

            if dist_quota == 'split':
                # grab full-time job number as key,
                # calculate count, set as value
                full_count = np.around(count * full_pcnt).astype(int)
                part_count = np.around(count * part_pcnt).astype(int)
                enhan_quota_dict[job_dict[job][0]] = (ratio, full_count)

                # same for part-time
                enhan_quota_dict[job_dict[job][1]] = (ratio, part_count)

            elif dist_quota == 'full':
                # apply entire count to full-time jobs only
                enhan_quota_dict[job_dict[job][0]] = (ratio, count)

            elif dist_quota == 'part':
                # apply entire count to part-time jobs only
                enhan_quota_dict[job_dict[job][1]] = (ratio, count)
    else:
        enhan_quota_dict = 0

    return (enhan_sg_cond, enhan_ratio_cond,
            enhan_count_cond, enhan_quota_dict, enhan_count_dict)
