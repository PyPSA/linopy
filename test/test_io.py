#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 09:03:35 2021

@author: fabian
"""

import xarray as xr
import numpy as np
import pandas as pd

from linopy import LinearExpression, Model
from linopy.io import to_int_str


def test_str_arrays():
    m = Model()

    x = m.add_variables('x', 4, pd.Series([8,10]))
    y = m.add_variables('y', 0, pd.DataFrame([[1,2], [3,4], [5,6]]).T)

    da = to_int_str(x)
    assert da.dtype == object


def test_str_arrays_chunked():
    m = Model(chunk=-1)

    x = m.add_variables('x', 4, pd.Series([8,10]))
    y = m.add_variables('y', 0, pd.DataFrame([[1,2], [3,4], [5,6]]).T)

    da = to_int_str(y).compute()
    assert da.dtype == object



def test_str_arrays_with_nans():
    m = Model()

    x = m.add_variables('x', 4, pd.Series([8,10]))
    # now expand the second dimension, expended values of x will be nan
    y = m.add_variables('y', 0, pd.DataFrame([[1,2], [3,4], [5,6]]))
    assert not m['x'].notnull().all()

    da = to_int_str(m['x'])
    assert da.dtype == object


# def test_to_file()

