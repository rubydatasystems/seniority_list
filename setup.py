#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def calculate_version():
    initpy = open('seniority_list/_version.py').read().split('\n')
    version = list(filter(lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version

package_version = calculate_version()

setup(
    name='seniority_list',
    version=package_version,
    author='Bob Davison',
    author_email='rubydatasystems@fastmail.net',
    packages=find_packages(),
    url='https://github.com/rubydatasystems/seniority_list',
    license='GNU/GPLv3',
    # If you specify a main() function in seniority_list.py, people can call your program from the
    # command line
    #entry_points={'console_scripts': ['seniority_list=seniority_list:main', ]},
    description=('seniority_list is an analytical tool used when seniority-based workgroups merge'),
    long_description='''
seniority_list is an analytical tool used when seniority-based workgroups merge

Contact
=============
If you have any questions or comments about seniority_list, please feel free to contact us via e-mail: rubydatasystems@fastmail.net

This project is hosted at https://github.com/rubydatasystems/seniority_list
''',
    zip_safe=True,
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'numba', 'ipython', 'seaborn', 'notebook'],
    # Choose more classifiers from here: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords=['workgroups', 'seniority', 'list', 'merge', 'merger', 'pilots'],
)
