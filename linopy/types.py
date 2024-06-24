from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy
import numpy.typing
from pandas import DataFrame, Series
from xarray import DataArray

if TYPE_CHECKING:
    from linopy.constraints import AnonymousScalarConstraint, Constraint
    from linopy.expressions import (
        LinearExpression,
        QuadraticExpression,
        ScalarLinearExpression,
    )
    from linopy.variables import ScalarVariable, Variable

ConstantLike = Union[int, float, numpy.ndarray, DataArray, Series, DataFrame]
SignLike = Union[str, numpy.ndarray, DataArray, Series, DataFrame]
VariableLike = Union["ScalarVariable", "Variable"]
ExpressionLike = Union[
    "ScalarLinearExpression", "LinearExpression", "QuadraticExpression"
]
ConstraintLike = Union["Constraint", "AnonymousScalarConstraint"]
MaskLike = Union[numpy.ndarray, DataArray, Series, DataFrame]
LhsLike = Union[VariableLike, ExpressionLike, ConstraintLike]


PathLike = Union[str, Path]
