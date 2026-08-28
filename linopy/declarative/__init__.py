"""
Linopy declarative math interface.

Build a linopy model from a declarative math definition and an xarray dataset of input data, via :func:`declarative_model`.

This directory is adapted from the calliope Apache-2.0 licensed math backend module:
https://github.com/calliope-project/calliope/tree/9916116a06ec8c1feaf3c2606bdb8941b916ce85/src/calliope/backend
"""

from linopy.declarative.build import DeclarativeModelBuilder, declarative_model
from linopy.declarative.helpers import HelperFunction
from linopy.declarative.latex import LatexModelBuilder, latex_math_doc
from linopy.declarative.schema import ConfigModel, MathModel

__all__ = [
    "ConfigModel",
    "DeclarativeModelBuilder",
    "HelperFunction",
    "LatexModelBuilder",
    "MathModel",
    "declarative_model",
    "latex_math_doc",
]
