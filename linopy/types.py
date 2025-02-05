from __future__ import annotations

import sys
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy
import numpy.typing
from pandas import DataFrame, Index, Series
from xarray import DataArray
from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates

if sys.version_info >= (3, 10):
    from types import EllipsisType, NotImplementedType
else:
    EllipsisType = type(Ellipsis)
    NotImplementedType = type(NotImplemented)

if TYPE_CHECKING:
    from linopy.constraints import AnonymousScalarConstraint, Constraint
    from linopy.expressions import (
        LinearExpression,
        QuadraticExpression,
        ScalarLinearExpression,
    )
    from linopy.variables import ScalarVariable, Variable

# Type aliases using Union for Python 3.9 compatibility
CoordsLike = Union[
    Sequence[Union[Sequence, Index, DataArray]],
    Mapping,
    DataArrayCoordinates,
    DatasetCoordinates,
]
DimsLike = Union[str, Iterable[Hashable]]

ConstantLike = Union[
    int,
    float,
    numpy.floating,
    numpy.integer,
    numpy.ndarray,
    DataArray,
    Series,
    DataFrame,
]
SignLike = Union[str, numpy.ndarray, DataArray, Series, DataFrame]
VariableLike = Union["ScalarVariable", "Variable"]
ExpressionLike = Union[
    "ScalarLinearExpression",
    "LinearExpression",
    "QuadraticExpression",
]
ConstraintLike = Union["Constraint", "AnonymousScalarConstraint"]
MaskLike = Union[numpy.ndarray, DataArray, Series, DataFrame]
SideLike = Union[ConstantLike, VariableLike, ExpressionLike]
PathLike = Union[str, Path]
