from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy
import polars as pl
from pandas import DataFrame, Index, Series
from xarray import DataArray
from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates

if TYPE_CHECKING:
    from linopy.constraints import AnonymousScalarConstraint, Constraint
    from linopy.expressions import (
        LinearExpression,
        QuadraticExpression,
        ScalarLinearExpression,
    )
    from linopy.variables import ScalarVariable, Variable

# Type aliases using Union for Python 3.9 compatibility
CoordsLike = Union[  # noqa: UP007
    Sequence[Sequence | Index | DataArray],
    Mapping,
    DataArrayCoordinates,
    DatasetCoordinates,
]
DimsLike = Union[str, Iterable[Hashable]]  # noqa: UP007

ConstantLike = Union[  # noqa: UP007
    int,
    float,
    numpy.floating,
    numpy.integer,
    numpy.ndarray,
    DataArray,
    Series,
    DataFrame,
    pl.Series,
]
SignLike = Union[str, numpy.ndarray, DataArray, Series, DataFrame]  # noqa: UP007
VariableLike = Union["ScalarVariable", "Variable"]
ExpressionLike = Union[
    "ScalarLinearExpression",
    "LinearExpression",
    "QuadraticExpression",
]
ConstraintLike = Union["Constraint", "AnonymousScalarConstraint"]
MaskLike = Union[numpy.ndarray, DataArray, Series, DataFrame]  # noqa: UP007
SideLike = Union[ConstantLike, VariableLike, ExpressionLike]  # noqa: UP007
PathLike = Union[str, Path]  # noqa: UP007
