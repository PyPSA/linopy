#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 13:33:55 2022.

@author: fabian
"""

import numpy as np
import pandas as pd
import xarray as xr


def get_lower_bounds(m):
    """
    Get vector of all lower variable bounds.

    Parameters
    ----------
    m : linopy.Model

    Returns
    -------
    lb
        One-dimensional numpy array containing all lower bounds.
    """
    return m.variables.flat.lower


def get_upper_bounds(m):
    """
    Get vector of all upper variable bounds.

    Parameters
    ----------
    m : linopy.Model

    Returns
    -------
    ub
        One-dimensional numpy array containing all upper bounds.
    """
    return m.variables.flat.upper


def get_variable_labels(m):
    """
    Get vector of all variable labels.

    Parameters
    ----------
    m : linopy.Model

    Returns
    -------
    labels
        One-dimensional numpy array containing all variable labels.
    """
    return m.variables.flat.labels


def get_variable_types(m):
    """
    Get vector of all variable types.

    'C' -> continuous
    'B -> binary
    'I -> integer

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
    specs = []
    for name in m.variables:
        if name in m.binaries:
            val = "B"
        elif name in m.integers:
            val = "I"
        else:
            val = "C"
        specs.append(pd.Series(val, index=m.variables[name].flat.index))

    return pd.concat(specs, ignore_index=True)


def get_objective_coefficients(m):
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
    return c[get_variable_labels(m)]


def get_constraint_matrix(m):
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
    return m.constraints.to_matrix()


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
    return m.constraints.flat.sign.astype(np.dtype("<U1"))


def get_rhs(m):
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
    return m.constraints.flat.rhs


def get_constraint_labels(m):
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
    return m.constraints.flat.labels


class MatrixAccessor:
    """
    Helper class to quickly access model related vectors and matrices.
    """

    def __init__(self, model):
        self._parent = model

    @property
    def vlabels(self):
        "Vector of labels of all non-missing variables."
        m = self._parent
        return m.variables.flat.labels.values

    @property
    def vtypes(self):
        "Vector of types of all non-missing variables."
        m = self._parent
        specs = []
        for name in m.variables:
            if name in m.binaries:
                val = "B"
            elif name in m.integers:
                val = "I"
            else:
                val = "C"
            specs.append(pd.Series(val, index=m.variables[name].flat.index))

        return pd.concat(specs, ignore_index=True)

    @property
    def lb(self):
        "Vector of lower bounds of all non-missing variables."
        m = self._parent
        return m.variables.flat.lower.values

    @property
    def ub(self):
        "Vector of upper bounds of all non-missing variables."
        m = self._parent
        return m.variables.flat.upper.values

    @property
    def clabels(self):
        "Vector of labels of all non-missing constraints."
        m = self._parent
        return m.constraints.flat.labels.values

    @property
    def A(self):
        "Constraint matrix of all non-missing constraints and variables."
        m = self._parent
        return m.constraints.to_matrix()

    @property
    def sense(self):
        "Vector of senses of all non-missing constraints."
        m = self._parent
        return m.constraints.flat.sign.values

    @property
    def b(self):
        "Vector of right-hand-sides of all non-missing constraints."
        m = self._parent
        return m.constraints.flat.rhs.values

    @property
    def c(self):
        "Vector of objective coefficients of all non-missing variables."
        m = self._parent
        ds = m.objective.flat
        res = np.zeros(m._xCounter)
        res[ds.vars] = ds.coeffs
        return res[self.vlabels]
