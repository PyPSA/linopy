from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper
from typing import Any

from xarray import DataArray

from linopy import expressions, variables

_LINOPY_TYPES = (
    variables.Variable,
    variables.ScalarVariable,
    expressions.LinearExpression,
    expressions.ScalarLinearExpression,
    expressions.QuadraticExpression,
)


def _make_patched_op(op_name: str) -> None:
    """Patch a DataArray operator to return NotImplemented for linopy types, enabling reflected operators."""
    original = getattr(DataArray, op_name)

    def patched(
        da: DataArray, other: Any, unpatched_method: Callable = original
    ) -> Any:
        if isinstance(other, _LINOPY_TYPES):
            return NotImplemented
        return unpatched_method(da, other)

    update_wrapper(patched, original)
    setattr(DataArray, op_name, patched)


for _op in (
    "__mul__",
    "__add__",
    "__sub__",
    "__truediv__",
    "__le__",
    "__ge__",
    "__eq__",
):
    _make_patched_op(_op)
del _op
