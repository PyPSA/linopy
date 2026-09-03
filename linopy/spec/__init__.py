"""
Build linopy models from math-spec programs.

The package needs the ``math-spec`` distribution (import name ``math_spec``,
Python >= 3.12). It is imported here and nowhere else in linopy, so
``import linopy`` never pulls it in.
"""

from __future__ import annotations

from importlib.util import find_spec

if find_spec("math_spec") is None:
    raise ImportError(
        "linopy.spec needs the math-spec package. Install it with "
        "`pip install math-spec` (Python >= 3.12) and try again."
    )

from linopy.spec.binder import Bound, Retain, bind
from linopy.spec.errors import SpecDataError

__all__ = ["Bound", "Retain", "SpecDataError", "bind"]
