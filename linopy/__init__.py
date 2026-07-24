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
from linopy.alignment import align
from linopy.config import LinopySemanticsWarning, options
from linopy.constants import (
    EQUAL,
    GREATER_EQUAL,
    LESS_EQUAL,
    EvolvingAPIWarning,
    PerformanceWarning,
)
from linopy.constraints import (
    Constraint,
    ConstraintBase,
    Constraints,
    CSRConstraint,
)
from linopy.expressions import LinearExpression, QuadraticExpression, merge
from linopy.io import read_netcdf
from linopy.model import Model, Variable, Variables
from linopy.objective import Objective
from linopy.piecewise import (
    PiecewiseFormulation,
    Slopes,
    breakpoints,
    segments,
    tangent_lines,
)
from linopy.remote import RemoteHandler
from linopy.solvers import SolverFeature, available_solvers, licensed_solvers

try:
    from linopy.remote import OetcCredentials, OetcHandler, OetcSettings  # noqa: F401
except ImportError:
    pass

__all__ = (
    "CSRConstraint",
    "ConstraintBase",
    "Constraints",
    "Constraint",
    "EQUAL",
    "PerformanceWarning",
    "EvolvingAPIWarning",
    "GREATER_EQUAL",
    "LESS_EQUAL",
    "LinearExpression",
    "LinopySemanticsWarning",
    "Model",
    "Objective",
    "OetcHandler",
    "PiecewiseFormulation",
    "QuadraticExpression",
    "RemoteHandler",
    "Slopes",
    "SolverFeature",
    "Variable",
    "Variables",
    "align",
    "available_solvers",
    "licensed_solvers",
    "breakpoints",
    "merge",
    "options",
    "read_netcdf",
    "segments",
    "tangent_lines",
)
