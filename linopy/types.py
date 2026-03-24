from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

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
    from linopy.piecewise import PiecewiseConstraintDescriptor
    from linopy.variables import ScalarVariable, Variable

CoordsLike: TypeAlias = (
    Sequence[Sequence | Index | DataArray]
    | Mapping
    | DataArrayCoordinates
    | DatasetCoordinates
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
SignLike: TypeAlias = str | numpy.ndarray | DataArray | Series | DataFrame
VariableLike: TypeAlias = ScalarVariable | Variable
ExpressionLike: TypeAlias = (
    ScalarLinearExpression | LinearExpression | QuadraticExpression
)
ConstraintLike: TypeAlias = (
    ConstraintBase | AnonymousScalarConstraint | PiecewiseConstraintDescriptor
)
LinExprLike: TypeAlias = Variable | LinearExpression
MaskLike: TypeAlias = numpy.ndarray | DataArray | Series | DataFrame
SideLike: TypeAlias = ConstantLike | VariableLike | ExpressionLike
PathLike: TypeAlias = str | Path
