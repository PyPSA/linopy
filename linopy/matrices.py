#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 13:33:55 2022.

@author: fabian
"""

import numpy as np
import pandas as pd
import xarray as xr


def get_lower_bounds(m, filter_missings=True):
    """
    Get vector of all lower variable bounds.

    Parameters
    ----------
    m : linopy.Model
    filter_missings : bool, optional
        Whether to filter out missing variables. The default is True.

    Returns
    -------
    lb
        One-dimensional numpy array containing all lower bounds.
    """
    return m.variables.ravel("lower", filter_missings=filter_missings)


def get_upper_bounds(m, filter_missings=True):
    """
    Get vector of all upper variable bounds.

    Parameters
    ----------
    m : linopy.Model
    filter_missings : bool, optional
        Whether to filter out missing variables. The default is True.

    Returns
    -------
    ub
        One-dimensional numpy array containing all upper bounds.
    """
    return m.variables.ravel("upper", filter_missings=filter_missings)


def get_variable_labels(m, filter_missings=True):
    """
    Get vector of all variable labels.

    Parameters
    ----------
    m : linopy.Model
    filter_missings : bool, optional
        Whether to filter out missing variables. The default is True.

    Returns
    -------
    labels
        One-dimensional numpy array containing all variable labels.
    """
    return m.variables.ravel("labels", filter_missings=filter_missings)


def get_variable_types(m, filter_missings=True):
    """
    Get vector of all variable types.

    'C' -> continuous
    'B -> binary

    Parameters
    ----------
    m : linopy.Model
    filter_missings : bool, optional
        Whether to filter out missing variables. The default is True.

    Returns
    -------
    labels
        One-dimensional numpy array containing all variable types.
    """
    specs = {name: "B" if name in m.binaries else "C" for name in m.variables}
    specs = xr.Dataset({k: xr.DataArray(v) for k, v in specs.items()})
    return m.variables.ravel(specs, filter_missings=True)


def get_objective_coefficients(m, filter_missings=True):
    """
    Get variable objective coefficients.

    Parameters
    ----------
    m : linopy.Model
    filter_missings : bool, optional
        Whether to filter out missing variables. The default is True.

    Returns
    -------
    coeffs
        One-dimensional numpy array containing coefficients of all variables.
    """
    vars = np.asarray(m.objective.vars)
    coeffs = np.asarray(m.objective.coeffs)
    if np.unique(m.objective.vars).shape != vars.shape:
        ds = pd.Series(coeffs).groupby(vars).sum()
        vars, coeffs = ds.index.values, ds.values
    c = np.zeros(m._xCounter)
    c[vars] = coeffs
    return c[get_variable_labels(m, filter_missings)]


def get_constraint_matrix(m, filter_missings=True):
    """
    Get the constraint matrix of a linopy model.

    Parameters
    ----------
    m : linopy.Model
    filter_missings : bool, optional
        Whether to filter out missing variables. The default is True.

    Returns
    -------
    A
        Sparse constraint matrix.
    """
    return m.constraints.to_matrix(filter_missings=filter_missings)


def get_sense(m, filter_missings=True):
    """
    Get the constraint senses of a linopy model.

    Parameters
    ----------
    m : linopy.Model
    filter_missings : bool, optional
        Whether to filter out missing variables. The default is True.

    Returns
    -------
    sense
        One-dimensional numpy array containing senses of all constraints.
    """
    return m.constraints.ravel("sign", filter_missings=filter_missings).astype(
        np.dtype("<U1")
    )


def get_rhs(m, filter_missings=True):
    """
    Get the constraint right hand sides of a linopy model.

    Parameters
    ----------
    m : linopy.Model
    filter_missings : bool, optional
        Whether to filter out missing variables. The default is True.

    Returns
    -------
    sense
        One-dimensional numpy array containing rhs of all constraints.
    """
    return m.constraints.ravel("rhs", filter_missings=filter_missings)


def get_constraint_labels(m, filter_missings=True):
    """
    Get the constraint labels of a linopy model.

    Parameters
    ----------
    m : linopy.Model
    filter_missings : bool, optional
        Whether to filter out missing variables. The default is True.

    Returns
    -------
    labels
        One-dimensional numpy array containing labels of all constraints.
    """
    m.constraints.sanitize_missings()
    return m.constraints.ravel("labels", filter_missings=filter_missings)


class MatrixAccessor:
    """
    Helper class to quickly access model related vectors and matrices.
    """

    def __init__(self, model):
        self._parent = model

    @property
    def vlabels(self):
        "Vector of labels of all non-missing variables."
        return get_variable_labels(self._parent)

    @property
    def vtypes(self):
        "Vector of types of all non-missing variables."
        return get_variable_types(self._parent)

    @property
    def lb(self):
        "Vector of lower bounds of all non-missing variables."
        return get_lower_bounds(self._parent)

    @property
    def ub(self):
        "Vector of upper bounds of all non-missing variables."
        return get_upper_bounds(self._parent)

    @property
    def clabels(self):
        "Vector of labels of all non-missing constraints."
        return get_constraint_labels(self._parent)

    @property
    def A(self):
        "Constraint matrix of all non-missing constraints and variables."
        return get_constraint_matrix(self._parent)

    @property
    def sense(self):
        "Vector of senses of all non-missing constraints."
        return get_sense(self._parent)

    @property
    def b(self):
        "Vector of right-hand-sides of all non-missing constraints."
        return get_rhs(self._parent)

    @property
    def c(self):
        "Vector of objective coefficients of all non-missing variables."
        return get_objective_coefficients(self._parent)
