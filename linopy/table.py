"""
Long-form input tables.

A :class:`Table` holds values in long form, one row per key combination,
and pivots onto the target coordinates of the position it is used in.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from xarray import DataArray


def to_pandas(frame: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    """``frame`` as a pandas DataFrame, a polars one converted column by column."""
    if isinstance(frame, pl.DataFrame):
        return pd.DataFrame(frame.to_dict(as_series=False))
    return frame


@dataclass(frozen=True, eq=False)
class Table:
    """
    A constant in long form: key columns plus one value column.

    Every column but ``value`` is a key column named after a dimension. Where
    a constant is accepted (variable bounds, coefficients, arithmetic
    constants, constraint right-hand sides), the table is pivoted onto the
    target coordinates: key combinations not listed take ``default``, and
    the values broadcast over target dimensions without a key column.

    The pivot materialises the full cube over the key dimensions, which is
    what the expression stores hold anyway; a sparsely filled table costs
    the memory of its cube.

    Parameters
    ----------
    frame : pandas.DataFrame or polars.DataFrame
        The long-form data.
    value : str
        Name of the value column.
    default : float
        Value of key combinations missing from ``frame``. Defaults to NaN.

    Examples
    --------
    >>> import pandas as pd
    >>> import linopy
    >>> m = linopy.Model()
    >>> i = pd.RangeIndex(3, name="i")
    >>> t = linopy.Table(pd.DataFrame({"i": [0, 2], "ub": [5.0, 7.0]}), "ub", 0)
    >>> m.add_variables(upper=t, coords=[i], name="x").upper.values
    array([5., 0., 7.])
    """

    frame: pd.DataFrame | pl.DataFrame
    value: str
    default: float = np.nan

    def __neg__(self) -> Table:
        frame = to_pandas(self.frame)
        negated = frame.assign(**{self.value: -frame[self.value]})
        return Table(negated, self.value, -self.default)

    def to_dataarray(self, coords: Mapping[Hashable, pd.Index]) -> DataArray:
        """
        Pivot the table onto ``coords``, one dim per key column in column
        order. Raises for key columns without a target dimension, key
        values outside the target coordinates and duplicate keys.
        """
        frame = to_pandas(self.frame)
        if self.value not in frame.columns:
            raise ValueError(f"Table has no value column {self.value!r}.")
        keys = [k for k in frame.columns if k != self.value]
        unknown = [k for k in keys if k not in coords]
        if unknown or not keys:
            raise ValueError(
                f"Table key columns {keys} must name target dimensions "
                f"{list(coords)}, got unmatched {unknown}."
            )
        index = pd.MultiIndex.from_frame(frame[keys])
        if index.has_duplicates:
            duplicates = index[index.duplicated()].unique().tolist()
            raise ValueError(f"Table has duplicate keys {duplicates}.")
        target = pd.MultiIndex.from_product([coords[k] for k in keys], names=keys)
        outside = index.difference(target)
        if len(outside):
            raise ValueError(
                f"Table keys {outside.tolist()} lie outside the target coordinates."
            )
        values = pd.Series(frame[self.value].to_numpy(), index=index)
        values = values.reindex(target, fill_value=self.default)
        shape = [len(coords[k]) for k in keys]
        return DataArray(
            values.to_numpy().reshape(shape),
            coords={k: coords[k] for k in keys},
            dims=keys,
        )
