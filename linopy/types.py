from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeAlias, Union, get_args

import numpy
import polars as pl
from pandas import DataFrame, Index, Series
from xarray import DataArray
from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates

from linopy.table import Table

if TYPE_CHECKING:
    from linopy.constraints import AnonymousScalarConstraint, Constraint
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
    | Table
)
CONSTANT_TYPES: tuple[type, ...] = get_args(ConstantLike)

UnlabeledLike: TypeAlias = numpy.ndarray | list | pl.Series
UNLABELED_TYPES = get_args(UnlabeledLike)

SignLike: TypeAlias = str | numpy.ndarray | DataArray | Series | DataFrame
MaskLike: TypeAlias = numpy.ndarray | DataArray | Series | DataFrame

# These reference types only available under TYPE_CHECKING, so use Union with strings
VariableLike: TypeAlias = Union["ScalarVariable", "Variable"]
ExpressionLike: TypeAlias = Union[
    "ScalarLinearExpression", "LinearExpression", "QuadraticExpression"
]
ConstraintLike = Union["Constraint", "AnonymousScalarConstraint"]
LinExprLike = Union["Variable", "LinearExpression"]
SideLike = Union[ConstantLike, VariableLike, ExpressionLike]  # noqa: UP007
