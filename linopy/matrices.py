#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 13:33:55 2022.

@author: fabian
"""

import numpy as np
import pandas as pd


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
    return m.constraints.ravel("labels", filter_missings=filter_missings)


def is_documented_by(original):
    def wrapper(target):
        target.__doc__ = original.__doc__.splitlines()
        return target

    return wrapper


class MatrixAccessor:
    """
    Helper class to quickly access model related vectors and matrices.
    """

    def __init__(self, model):
        self._parent = model

    @property
    @is_documented_by(get_variable_labels)
    def vlabels(self):
        return get_variable_labels(self._parent)

    @property
    @is_documented_by(get_lower_bounds)
    def lb(self):
        return get_lower_bounds(self._parent)

    @property
    @is_documented_by(get_upper_bounds)
    def ub(self):
        return get_upper_bounds(self._parent)

    @property
    @is_documented_by(get_constraint_labels)
    def clabels(self):
        return get_constraint_labels(self._parent)

    @is_documented_by(get_constraint_matrix)
    @property
    def A(self):
        return get_constraint_matrix(self._parent)

    @property
    @is_documented_by(get_sense)
    def sense(self):
        return get_sense(self._parent)

    @property
    @is_documented_by(get_rhs)
    def b(self):
        return get_rhs(self._parent)

    @property
    @is_documented_by(get_objective_coefficients)
    def c(self):
        return get_objective_coefficients(self._parent)
