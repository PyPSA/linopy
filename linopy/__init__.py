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
from linopy import model, remote, types
from linopy.config import options
from linopy.constants import EQUAL, GREATER_EQUAL, LESS_EQUAL
from linopy.constraints import Constraint
from linopy.expressions import LinearExpression, QuadraticExpression, merge
from linopy.io import read_netcdf
from linopy.model import Model, Variable, available_solvers
from linopy.objective import Objective
from linopy.remote import RemoteHandler

__all__ = (
    "Constraint",
    "EQUAL",
    "GREATER_EQUAL",
    "LESS_EQUAL",
    "LinearExpression",
    "Model",
    "Objective",
    "QuadraticExpression",
    "RemoteHandler",
    "Variable",
    "available_solvers",
    "merge",
    "model",
    "options",
    "read_netcdf",
    "remote",
    "types",
)
