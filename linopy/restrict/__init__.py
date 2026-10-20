"""
Rolling-horizon primitives, backed by a model's flattened matrices.

:meth:`linopy.Model.restrict` returns a :class:`~linopy.restrict.WindowModel`,
a window of the parent model expressed as a row slice and column range of its
flattened ``A``, ``b``, ``c``, ``lb`` and ``ub``. Terms whose variables fall
outside the window are folded into the right-hand side at their actual matrix
coefficient:

    ``A_WW @ x_W ~ b_W - A_WB @ x_bar_B``

with ``x_bar_B`` the fixed boundary values, i.e. the subproblem-conditioning
primitive of decomposition methods.

Working on the matrices means a window costs a slice, not a rebuild of every
variable and constraint. The window materialises a variable or constraint only
when its values are read, and solves by handing the flat arrays to the solver.

The package is organized around:

* :mod:`linopy.restrict.matrix` - :class:`MatrixWindow`, the flattened view.
* :mod:`linopy.restrict.window` - :class:`WindowModel`, the user-facing window.
* :mod:`linopy.restrict.api` - the :func:`restrict_model` entry point.
"""

from __future__ import annotations

from linopy.restrict.api import restrict_model
from linopy.restrict.errors import RestrictionError
from linopy.restrict.matrix import MatrixWindow, WindowMatrices
from linopy.restrict.window import WindowModel, WindowObjective

__all__ = [
    "MatrixWindow",
    "RestrictionError",
    "WindowMatrices",
    "WindowModel",
    "WindowObjective",
    "restrict_model",
]
