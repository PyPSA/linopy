#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 22:36:38 2021

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model


def test_variable_repr():
    m = Model()
    x = m.add_variables()
    x.__repr__()
    x._repr_html_()


def test_variable_bound_accessor():
    m = Model()
    x = m.add_variables(0, 10)
    assert x.get_upper_bound().item() == 10
    assert x.get_lower_bound().item() == 0
