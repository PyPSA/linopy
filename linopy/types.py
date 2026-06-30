from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, Union, get_args

import numpy
import polars as pl
from pandas import DataFrame, Index, Series
from xarray import DataArray
from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates

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

ConstantLike: TypeAlias = (
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
CONSTANT_TYPES: tuple[type, ...] = get_args(ConstantLike)
SignLike: TypeAlias = str | numpy.ndarray | DataArray | Series | DataFrame
MaskLike: TypeAlias = numpy.ndarray | DataArray | Series | DataFrame
PathLike: TypeAlias = str | Path

# These reference types only available under TYPE_CHECKING, so use Union with strings
VariableLike: TypeAlias = Union["ScalarVariable", "Variable"]
ExpressionLike: TypeAlias = Union[
    "ScalarLinearExpression", "LinearExpression", "QuadraticExpression"
]
ConstraintLike = Union["ConstraintBase", "AnonymousScalarConstraint"]
LinExprLike = Union["Variable", "LinearExpression"]
SideLike = Union[ConstantLike, VariableLike, ExpressionLike]  # noqa: UP007
