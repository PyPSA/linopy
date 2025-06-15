from __future__ import annotations

from collections.abc import Callable
from functools import partialmethod, update_wrapper
from types import NotImplementedType
from typing import Any

from xarray import DataArray

from linopy import expressions, variables


def monkey_patch(cls: type[DataArray], pass_unpatched_method: bool = False) -> Callable:
    def deco(func: Callable) -> Callable:
        func_name = func.__name__
        wrapped = getattr(cls, func_name)
        update_wrapper(func, wrapped)
        if pass_unpatched_method:
            func = partialmethod(func, unpatched_method=wrapped)  # type: ignore
        setattr(cls, func_name, func)
        return func

    return deco


@monkey_patch(DataArray, pass_unpatched_method=True)
def __mul__(
    da: DataArray, other: Any, unpatched_method: Callable
) -> DataArray | NotImplementedType:
    if isinstance(
        other,
        variables.Variable
        | expressions.LinearExpression
        | expressions.QuadraticExpression,
    ):
        return NotImplemented
    return unpatched_method(da, other)
