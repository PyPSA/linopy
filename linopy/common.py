#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linopy common module.
This module contains commonly used functions.
"""

from xarray import apply_ufunc, merge


def _merge_inplace(self, attr, da, name, **kwargs):
    """
    Assign a new dataarray to the dataset `attr` by merging.

    This takes care of all coordinate alignments, instead of a direct
    assignment like self.variables[name] = var
    """
    ds = merge([getattr(self, attr), da.rename(name)], **kwargs)
    setattr(self, attr, ds)


def _remap(array, mapping):
    return mapping[array.ravel()].reshape(array.shape)


def replace_by_map(ds, mapping):
    "Replace values in a DataArray by a one-dimensional mapping."
    return apply_ufunc(_remap, ds, kwargs=dict(mapping=mapping), dask="allowed")
