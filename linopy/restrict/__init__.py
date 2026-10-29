"""
Rolling-horizon primitives, backed by a model's flattened matrices.

:meth:`linopy.Model.restrict` returns a :class:`~linopy.restrict.WindowModel`,
the rows and columns of the parent's flattened ``A``, ``b``, ``c``, ``lb`` and
``ub`` that belong to the selected labels. Terms whose variables fall outside
the window are folded into the right-hand side at their actual coefficient:

    ``A_WW @ x_W ~ b_W - A_WB @ x_bar_B``

with ``x_bar_B`` the fixed boundary values, i.e. the subproblem-conditioning
primitive of decomposition methods.

* :mod:`linopy.restrict.matrix` - :class:`MatrixWindow`, the flattened view.
* :mod:`linopy.restrict.window` - :class:`WindowModel`, the user-facing window.
"""

from __future__ import annotations

from linopy.restrict.matrix import MatrixWindow, RestrictionError, WindowMatrices
from linopy.restrict.window import WindowModel, WindowObjective, restrict_model

__all__ = [
    "MatrixWindow",
    "RestrictionError",
    "WindowMatrices",
    "WindowModel",
    "WindowObjective",
    "restrict_model",
]
