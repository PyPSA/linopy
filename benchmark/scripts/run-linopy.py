#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:40:33 2021.

@author: fabian
"""

from benchmark_linopy import model

n = int(snakemake.wildcards.N)
solver = snakemake.wildcards.solver
integerlabels = snakemake.params.integerlabels
model(n, solver, integerlabels)
