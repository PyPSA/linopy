"""
Build linopy models from mathspec programs.

The package needs the ``mathspec`` distribution (import name ``mathspec``,
Python >= 3.12). It is imported here and nowhere else in linopy, so
``import linopy`` never pulls it in.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec

if find_spec("mathspec") is None:
    message = (
        "linopy.spec needs the mathspec package. Install it with "
        "`pip install 'linopy[spec]'`."
    )
    if sys.version_info < (3, 12):
        message = "linopy.spec needs Python >= 3.12. " + message
    raise ImportError(message)

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
]
