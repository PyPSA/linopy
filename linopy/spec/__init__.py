"""
Build linopy models from math-spec programs.

The package needs the ``math-spec`` distribution (import name ``math_spec``,
Python >= 3.12). It is imported here and nowhere else in linopy, so
``import linopy`` never pulls it in.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec

if find_spec("math_spec") is None:
    message = (
        "linopy.spec needs the math-spec package. Install it from a checkout "
        "with `uv sync --group spec` or `pip install --group spec`."
    )
    if sys.version_info < (3, 12):
        message = "linopy.spec needs Python >= 3.12. " + message
    raise ImportError(message)

from math_spec import merge, override

from linopy.spec.accessor import (
    Declaration,
    ModelSpec,
    NamedExpression,
    NamedExpressions,
    SpecLike,
    Unspecified,
)
from linopy.spec.attach import Attached, Retain, attach
from linopy.spec.errors import SpecDataError

__all__ = [
    "Attached",
    "Declaration",
    "ModelSpec",
    "NamedExpression",
    "NamedExpressions",
    "Retain",
    "SpecDataError",
    "SpecLike",
    "Unspecified",
    "attach",
    "merge",
    "override",
]
