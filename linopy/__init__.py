#!/usr/bin/env python3
"""
Created on Wed Mar 10 11:03:06 2021.

@author: fabulous
"""

from importlib.metadata import version

__version__ = version("linopy")

# Note: For intercepting multiplications between xarray dataarrays, Variables and Expressions
# we need to extend their __mul__ functions with a quick special case
import linopy.monkey_patch_xarray  # noqa: F401
from linopy.common import align
from linopy.config import options
from linopy.constants import EQUAL, GREATER_EQUAL, LESS_EQUAL
from linopy.constraints import Constraint, Constraints
from linopy.expressions import LinearExpression, QuadraticExpression, merge
from linopy.io import read_netcdf
from linopy.model import Model, Variable, Variables, available_solvers
from linopy.objective import Objective
from linopy.remote import OetcHandler, RemoteHandler

__all__ = (
    "Constraint",
    "Constraints",
    "EQUAL",
    "GREATER_EQUAL",
    "LESS_EQUAL",
    "LinearExpression",
    "Model",
    "Objective",
    "OetcHandler",
    "QuadraticExpression",
    "RemoteHandler",
    "Variable",
    "Variables",
    "available_solvers",
    "align",
    "merge",
    "options",
    "read_netcdf",
)
