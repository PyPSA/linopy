#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:04:43 2021

@author: fabulous
"""


from setuptools import setup, find_packages
from codecs import open

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='linopy',
    author='Fabian Hofmann',
    author_email='hofmann@fias.uni-frankfurt.de',
    description='Linear optimization with N-D labeled arrays in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # url='https://github.com/PyPSA/linopy',
    license='GPLv3',
    packages=find_packages(exclude=['doc', 'test']),
    include_package_data=True,
    python_requires='~=3.7',
    use_scm_version={'write_to': 'linopy/version.py'},
    setup_requires=['setuptools_scm'],
    install_requires=['numpy',
                      'scipy',
                      'bottleneck',
                      'toolz',
                      'numexpr',
                      'cplex',
                      'xarray>=0.16',
                      'dask>=0.18.0'],
    extras_require = {
        "docs": ["numpydoc",
                 "sphinx", "sphinx_rtd_theme", "nbsphinx"]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ])