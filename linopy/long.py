"""
Long-form inputs.

:func:`from_long` pivots a long-form frame of key columns plus one value
column into a DataArray, :class:`Relation` maps the members of a dimension
to groups for :meth:`~linopy.expressions.LinearExpressionGroupby.sum`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd
import polars as pl
from xarray import DataArray

from linopy.alignment import _as_index, _coords_to_dict
from linopy.types import CoordsLike

Frame: TypeAlias = pl.DataFrame | pd.DataFrame
Column: TypeAlias = pl.Series | pd.Series

KIND_FAMILIES = {
    "b": "boolean",
    "i": "numeric",
    "u": "numeric",
    "f": "numeric",
    "M": "datetime",
    "m": "duration",
    "O": "string",
    "U": "string",
    "S": "string",
}


def _family(column: Column | pd.Index) -> str:
    """
    The dtype family of ``column``: numeric, datetime with its time zone,
    duration, string or boolean.
    """
    dtype = column.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        return f"datetime[{dtype.tz}]"
    if not isinstance(dtype, pl.DataType):
        return KIND_FAMILIES[dtype.kind]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return f"datetime[{dtype.time_zone}]"
    if isinstance(dtype, pl.Date | pl.Datetime):
        return "datetime"
    if isinstance(dtype, pl.Duration):
        return "duration"
    if dtype.is_numeric():
        return "numeric"
    if dtype == pl.Boolean:
        return "boolean"
    if dtype == pl.String or isinstance(dtype, pl.Categorical | pl.Enum):
        return "string"
    raise TypeError(f"Column {column.name!r} has unsupported dtype {dtype}.")


def _as_polars(index: pd.Index) -> pl.Series:
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        utc = pl.Series(index.asi8).cast(pl.Datetime(index.unit))
        return utc.dt.replace_time_zone("UTC").dt.convert_time_zone(str(index.tz))
    return pl.Series(index.to_numpy())


def _has_nulls(column: Column) -> bool:
    if isinstance(column, pl.Series):
        return column.null_count() > 0
    return bool(column.isna().any())


def _is_float_numpy(column: Column) -> bool:
    return isinstance(column.dtype, np.dtype) and column.dtype.kind == "f"


def _check_frame(frame: Frame) -> None:
    if not isinstance(frame, Frame):
        raise TypeError(
            f"Expected a polars or pandas DataFrame, got {type(frame).__name__}."
        )
    if isinstance(frame, pd.DataFrame) and (
        frame.columns.nlevels > 1 or frame.index.names != [None]
    ):
        raise TypeError(
            "Expected flat columns and an unnamed index, call `reset_index()` "
            "to turn index levels into key columns."
        )


def _positions(index: pd.Index, column: Column) -> pl.Series:
    """
    The positions of the values of ``column`` in ``index``. Raises for null
    values, a dtype family other than the one of ``index`` and values not in
    ``index``.
    """
    if _has_nulls(column):
        raise ValueError(f"Column {column.name!r} has null values.")
    if _family(column) != _family(index):
        raise TypeError(
            f"Column {column.name!r} of dtype {column.dtype} does not match "
            f"the coordinates of dimension {index.name!r} of dtype {index.dtype}."
        )
    if isinstance(column, pd.Series):
        pos = pl.Series(index.get_indexer(column))
        pos = pos.set(pos == -1, None)
    elif (
        isinstance(index, pd.RangeIndex)
        and index.step == 1
        and column.dtype.is_integer()
    ):
        pos = column.cast(pl.Int64, strict=False) - index.start
        pos = pos.set((pos < 0) | (pos >= len(index)), None)
    else:
        lookup = _as_polars(index)
        key = column.cast(lookup.dtype, strict=False)
        if lookup.dtype != column.dtype:
            key = key.set(key.cast(column.dtype, strict=False).ne_missing(column), None)
        table = pl.DataFrame({"key": lookup, "pos": np.arange(len(index))})
        pos = key.to_frame("key").join(
            table, on="key", how="left", maintain_order="left"
        )["pos"]
    if pos.null_count():
        missing = pos.is_null()
        outside = (
            column.filter(missing).unique(maintain_order=True).to_list()
            if isinstance(column, pl.Series)
            else column[missing.to_numpy()].unique().tolist()
        )
        raise ValueError(
            f"Values {outside[:5]} of column {column.name!r} are "
            f"not in the coordinates of dimension {index.name!r}."
        )
    return pos.cast(pl.Int64)


def from_long(
    frame: Frame, value: str, coords: CoordsLike, default: float = np.nan
) -> DataArray:
    """
    Pivot a long-form frame onto ``coords``.

    Every column of ``frame`` but ``value`` is a key column named after a
    dimension of ``coords``. The result spans the key dimensions only, in
    column order, with the full coordinates of ``coords``: key combinations
    not listed take ``default``. Dimensions of ``coords`` without a key column
    are not part of the result, linopy broadcasts it over them where it is
    used. The result is the full cube over the key dimensions, a sparsely
    filled frame costs the memory of its cube.

    Parameters
    ----------
    frame : polars.DataFrame or pandas.DataFrame
        The long-form data, one row per key combination. A pandas frame needs
        an unnamed index, its keys are columns.
    value : str
        Name of the value column, numeric without null values; NaN values
        are kept.
    coords : list/dict/xarray.Coordinates
        The coordinates the keys are positioned in, e.g. the coordinates of
        the variable the result is used with.
    default : float
        Value of key combinations missing from ``frame``. Defaults to NaN.

    Returns
    -------
    xarray.DataArray
        Float values over the key dimensions.

    Raises
    ------
    ValueError
        For a missing value column, no key column, a key column not naming a
        dimension of ``coords``, null keys or values, duplicate keys and keys
        outside ``coords``.
    TypeError
        For a frame type other than a polars or pandas DataFrame, a non-numeric
        value column and key dtypes (numeric, datetime with its time zone,
        duration, string) differing from the ones of ``coords``.

    Examples
    --------
    >>> import pandas as pd
    >>> import linopy
    >>> m = linopy.Model()
    >>> i = pd.RangeIndex(3, name="i")
    >>> ub = linopy.from_long(
    ...     pd.DataFrame({"i": [0, 2], "ub": [5.0, 7.0]}), "ub", coords=[i], default=0
    ... )
    >>> m.add_variables(upper=ub, coords=[i], name="x").upper.values
    array([5., 0., 7.])
    """
    _check_frame(frame)
    if value not in frame.columns:
        raise ValueError(f"Frame has no value column {value!r}.")
    keys = [k for k in frame.columns if k != value]
    if not keys:
        raise ValueError("Frame has no key column.")
    target = {d: _as_index(v).rename(d) for d, v in _coords_to_dict(coords).items()}
    unknown = [k for k in keys if k not in target]
    if unknown:
        raise ValueError(
            f"Key columns {unknown} do not name a dimension of {list(target)}."
        )
    if _family(frame[value]) != "numeric":
        raise TypeError(
            f"Value column {value!r} has non-numeric dtype {frame[value].dtype}."
        )
    if _has_nulls(frame[value]) and not _is_float_numpy(frame[value]):
        raise ValueError(f"Value column {value!r} has null values.")
    flat = pl.Series(np.zeros(frame.shape[0], dtype=np.int64))
    for k in keys:
        flat = flat * len(target[k]) + _positions(target[k], frame[k])
    if flat.n_unique() != len(flat):
        raise ValueError(f"Frame has duplicate keys over {keys}.")
    shape = [len(target[k]) for k in keys]
    size = int(np.prod(shape))
    values = frame[value].to_numpy().astype(float)
    if len(flat) == size and flat.is_sorted():
        out = values
    else:
        out = np.full(size, default, dtype=float)
        out[flat.to_numpy()] = values
    return DataArray(out.reshape(shape), coords=[target[k] for k in keys], dims=keys)


@dataclass(frozen=True, eq=False)
class Relation:
    """
    A many-to-many map from the members of a dimension to groups.

    Each row of ``frame`` relates the member in column ``member``, a
    coordinate of the dimension of the same name, to the group in column
    ``group``; a member may be listed in several rows.
    ``expr.groupby(relation).sum()`` adds each row's member to its group, so a
    member listed in several rows lands in several groups and one not listed
    in none.

    Parameters
    ----------
    frame : polars.DataFrame or pandas.DataFrame
        The member-group pairs.
    member : str
        Name of the member column, a dimension of the grouped expression.
    group : str
        Name of the group column, the dimension of the result.

    Examples
    --------
    >>> import pandas as pd
    >>> import linopy
    >>> m = linopy.Model()
    >>> link = pd.Index(["l0", "l1"], name="link")
    >>> p = m.add_variables(coords=[link], name="p")
    >>> ports = pd.DataFrame({"link": ["l0", "l0", "l1"], "bus": ["b0", "b1", "b1"]})
    >>> p.groupby(linopy.Relation(ports, "link", "bus")).sum().nterm
    2
    """

    frame: Frame
    member: str
    group: str

    def __post_init__(self) -> None:
        _check_frame(self.frame)
        missing = {self.member, self.group} - set(self.frame.columns)
        if missing or self.member == self.group:
            raise ValueError(
                f"Relation needs distinct columns {self.member!r} and "
                f"{self.group!r}, missing {sorted(missing)}."
            )
        if _has_nulls(self.frame[self.group]):
            raise ValueError(f"Relation group column {self.group!r} has null values.")

    def positions(self, index: pd.Index) -> np.ndarray:
        """The positions of the members in ``index``."""
        return _positions(index, self.frame[self.member]).to_numpy()

    def groups(self) -> np.ndarray:
        """The group of each row."""
        return self.frame[self.group].to_numpy()
