# -*- coding: utf-8 -*-

'''Build list orderings from master list data as a starting point for
further analysis and/or list editing.  Lists may be built by various
weighting and sorting methods.
'''

import pandas as pd
import numpy as np

import config as cf
import functions as f


def add_attributes_to_master_list():

    '''Add attribute columns to master list which can be used as factors
    to construct a list ordering.

    Adds: ['age', 's_lmonths', 'jnum', 'job_count', 'rank_in_job',
           'jobp', 'eg_number', 'eg_spcnt']

    Job-related attributes are referenced to job counts from the config file.
    '''

    if cf.sample_mode:
        sample_prefix = cf.sample_prefix
        master_ = pd.read_pickle('sample_data/' + sample_prefix + 'master.pkl')
    else:
        master_ = pd.read_pickle('dill/master.pkl')

    master = master_[(master_.line == 1) | (master_.fur == 1)].copy()

    # AGE and LONGEVITY
    master['age'] = f.starting_age(master.retdate)
    master['s_lmonths'] = f.longevity_at_startdate(list(master['ldate']),
                                                   return_months=True)

    jobs_list = []

    if cf.enhanced_jobs:
        eg_counts = f.convert_jcnts_to_enhanced(cf.eg_counts,
                                                cf.intl_blk_pcnt,
                                                cf.dom_blk_pcnt)
    else:
        eg_counts = cf.eg_counts

    # make a list of stovepipe jobs for each group (from config job counts)
    i = 1
    for jobs in eg_counts:
        # the second input determines the length of the zero
        # array formed (possible excess)
        jobs_list.append(
            f.make_stovepipe_jobs_from_jobs_arr(jobs,
                                                sum((master.eg == i) &
                                                    ((master.line == 1) |
                                                     (master.fur == 1)))))
        i += 1

    fur_level = f.max_of_nested_lists(jobs_list) + 1
    jobs = np.array(jobs_list)

    # mark unassigned as furloughed (from zero to fur_level)
    for job_arr in jobs:
        np.put(job_arr, np.where(job_arr == 0)[0], fur_level)

    egs = np.array(master.eg)
    jnums = np.zeros(egs.size)
    job_count = np.zeros(egs.size)

    # JNUM and JOB_COUNT data prep
    i = 1
    for job_arr in jobs:
        data = np.unique(job_arr, return_counts=True)
        zipped = zip(data[0], data[1])
        for job, count in zipped:
            np.put(job_count,
                   np.where((jnums == 0) & (egs == i))[0][:count],
                   count)
            np.put(jnums, np.where((jnums == 0) & (egs == i))[0][:count], job)
        i += 1

    # Employee group count (for spcnt column)
    eg_counts = np.zeros(egs.size)
    data = np.unique(master.eg, return_counts=True)
    zipped = zip(data[0], data[1])
    for eg, count in zipped:
        np.put(eg_counts, np.where(egs == eg)[0], count)

    # Attribute columns assignment
    master['jnum'] = jnums.astype(int)
    master['job_count'] = job_count.astype(int)
    master['rank_in_job'] = master.groupby(['eg', 'jnum']).cumcount() + 1
    master['jobp'] = (master.rank_in_job /
                      master.job_count) + master.jnum - .0001
    master['eg_number'] = master.groupby('eg').cumcount() + 1
    master['eg_count'] = eg_counts.astype(int)
    master['eg_spcnt'] = master.eg_number / master.eg_count

    master.pop('eg_count')

    return master


def build_list():

    print('under construction')


