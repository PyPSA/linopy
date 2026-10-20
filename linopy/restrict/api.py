"""
Rolling-horizon entry point, backed by the flattened matrices.

:meth:`Model.restrict` delegates here. A window is a slice of the parent's
flattened matrices (:class:`linopy.restrict.MatrixWindow`), returned as a
:class:`~linopy.restrict.window.WindowModel` that materialises variables and
constraints only when their values are read.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from linopy.restrict.errors import RestrictionError

if TYPE_CHECKING:
    from linopy.model import Model
    from linopy.restrict.window import WindowModel

__all__ = ["RestrictionError", "restrict_model"]


def restrict_model(
    model: Model,
    fix: Mapping[str, Any] | None,
    selection: Mapping[str, Any],
) -> WindowModel:
    """
    Build a matrix-backed window of ``model``.

    See :meth:`linopy.Model.restrict` for the documented behavior.
    """
    from linopy.restrict.window import WindowModel

    if len(selection) != 1:
        raise NotImplementedError(
            "Model.restrict supports exactly one selected dimension, got "
            f"{sorted(selection)}."
        )
    dim = next(iter(selection))
    carried = any(dim in item.sizes for item in model._coordinate_carriers())
    if not carried and dim not in model.parameters.sizes:
        raise RestrictionError(
            f"Cannot restrict the model along '{dim}': the dimension does not "
            "exist in the model."
        )
    return WindowModel(model, str(dim), dict(selection), dict(fix) if fix else None)
