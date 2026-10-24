from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeAlias, Union, get_args

import numpy
from pandas import DataFrame, Index, Series
from xarray import DataArray
from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from linopy.constraints import (
        AnonymousScalarConstraint,
        ConstraintBase,
    )
    from linopy.expressions import (
        LinearExpression,
        QuadraticExpression,
        ScalarLinearExpression,
    )
    from linopy.variables import ScalarVariable, Variable

CoordsLike: TypeAlias = (
    Sequence[Sequence | Index] | Mapping | DataArrayCoordinates | DatasetCoordinates
)
DimsLike: TypeAlias = str | Iterable[Hashable]

# polars is optional. When it is missing, `pl` is None and the polars entries
# drop out of the aliases and the runtime type tuples below, so `isinstance`
# checks keep working and simply never match a polars input.
POLARS_SERIES_TYPES: tuple[type, ...] = (pl.Series,) if pl is not None else ()

if pl is None:  # pragma: no cover
    ConstantLike: TypeAlias = (
        int
        | float
        | numpy.floating
        | numpy.integer
        | numpy.ndarray
        | DataArray
        | Series
        | DataFrame
    )
    UnlabeledLike: TypeAlias = numpy.ndarray | list
else:
    ConstantLike: TypeAlias = (  # type: ignore[misc,no-redef]
        int
        | float
        | numpy.floating
        | numpy.integer
        | numpy.ndarray
        | DataArray
        | Series
        | DataFrame
        | pl.Series
    )
    UnlabeledLike: TypeAlias = numpy.ndarray | list | pl.Series  # type: ignore[misc,no-redef]

CONSTANT_TYPES: tuple[type, ...] = get_args(ConstantLike)
UNLABELED_TYPES = get_args(UnlabeledLike)

SignLike: TypeAlias = str | numpy.ndarray | DataArray | Series | DataFrame
MaskLike: TypeAlias = numpy.ndarray | DataArray | Series | DataFrame

# These reference types only available under TYPE_CHECKING, so use Union with strings
VariableLike: TypeAlias = Union["ScalarVariable", "Variable"]
ExpressionLike: TypeAlias = Union[
    "ScalarLinearExpression", "LinearExpression", "QuadraticExpression"
]
ConstraintLike = Union["ConstraintBase", "AnonymousScalarConstraint"]
LinExprLike = Union["Variable", "LinearExpression"]
SideLike = Union[ConstantLike, VariableLike, ExpressionLike]  # noqa: UP007
