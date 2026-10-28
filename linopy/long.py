"""
Long-form inputs.

:func:`from_long` pivots a long-form frame of key columns plus one value
column into a DataArray, :class:`Relation` maps the members of a dimension
to groups for :meth:`~linopy.expressions.LinearExpressionGroupby.sum`.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd
import polars as pl
from xarray import DataArray

from linopy.alignment import _as_index, _coords_to_dict
from linopy.constants import _warn_evolving_api
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
    "U": "string",
    "S": "string",
}


def _family(column: Column | pd.Index) -> str:
    """
    The dtype family of ``column``: numeric, datetime with its time zone,
    date, duration, string or boolean, a pandas categorical classified by
    its categories.
    """
    dtype = column.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return _family(dtype.categories.rename(column.name))
    if isinstance(dtype, pd.DatetimeTZDtype):
        return f"datetime[{dtype.tz}]"
    if not isinstance(dtype, pl.DataType) and dtype.kind == "O":
        return "date" if pd.api.types.infer_dtype(column) == "date" else "string"
    if not isinstance(dtype, pl.DataType) and dtype.kind in KIND_FAMILIES:
        return KIND_FAMILIES[dtype.kind]
    if not isinstance(dtype, pl.DataType):
        raise TypeError(f"Column {column.name!r} has unsupported dtype {dtype}.")
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return f"datetime[{dtype.time_zone}]"
    if isinstance(dtype, pl.Datetime):
        return "datetime"
    if isinstance(dtype, pl.Date):
        return "date"
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
    if _family(index) == "date":
        return pl.Series(index.to_numpy(dtype="datetime64[D]")).cast(pl.Date)
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
    values, a dtype family other than the one of ``index``, values not in
    ``index`` and duplicate labels of ``index``.
    """
    if not index.is_unique:
        raise ValueError(
            f"Coordinates of dimension {index.name!r} have duplicate labels."
        )
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


def _flat(
    frame: Frame,
    keys: Sequence[str],
    coords: CoordsLike,
    strict: bool,
    hint: str = "",
    duplicate_hint: str = "",
) -> tuple[dict[Hashable, pd.Index], pl.Series, list[int], int]:
    """
    The flat positions of the key combinations of ``frame`` in the cube of
    ``coords`` over ``keys``, with the target indexes, the cube shape and size.
    Raises for duplicate key combinations and, if ``strict``, for key
    combinations of the cube missing from ``frame``.
    """
    target = {d: _as_index(v).rename(d) for d, v in _coords_to_dict(coords).items()}
    absent = [k for k in keys if k not in target]
    if absent:
        raise ValueError(
            f"Coords lack the dimensions {absent} of columns {list(keys)}."
        )
    flat = pl.Series(np.zeros(frame.shape[0], dtype=np.int64))
    for k in keys:
        flat = flat * len(target[k]) + _positions(target[k], frame[k])
    if flat.n_unique() != len(flat):
        raise ValueError(f"Frame has duplicate keys over {list(keys)}{duplicate_hint}.")
    shape = [len(target[k]) for k in keys]
    size = int(np.prod(shape))
    if strict and len(flat) < size:
        missing = np.ones(size, dtype=bool)
        missing[flat.to_numpy()] = False
        at = np.unravel_index(int(missing.argmax()), shape)
        first = [target[k][[i]].item() for k, i in zip(keys, at)]
        raise ValueError(
            f"Frame misses {int(missing.sum())} of {size} key combinations over "
            f"{list(keys)}, first {first}{hint}."
        )
    return target, flat, shape, size


def from_long(
    frame: Frame, coords: CoordsLike, default: float | None = None
) -> DataArray:
    """
    Pivot a long-form frame onto ``coords``.

    The columns of ``frame`` named after a dimension of ``coords`` are the key
    columns, exactly one other column is the value column. The result is
    named after the value column and spans the key dimensions only, in column
    order, with the full coordinates of ``coords``: key combinations not
    listed take ``default`` or raise without it. Dimensions of ``coords``
    without a key column are not part of the result, linopy broadcasts it over
    them where it is used.
    The result is the full cube over the key dimensions, a sparsely filled
    frame costs the memory of its cube.

    Parameters
    ----------
    frame : polars.DataFrame or pandas.DataFrame
        The long-form data, one row per key combination: key columns plus one
        value column, numeric without null values; NaN values are kept. A
        pandas frame needs an unnamed index, its keys are columns.
    coords : list/dict/xarray.Coordinates
        The coordinates the keys are positioned in, e.g. ``var.coords`` of
        the variable the result is used with.
    default : float, optional
        Value of key combinations missing from ``frame``; without it, gaps raise.

    Returns
    -------
    xarray.DataArray
        Float values over the key dimensions, named after the value column.

    Raises
    ------
    ValueError
        For no key column, not exactly one column besides the key columns (a
        misspelled key column counts as a second value column), null keys or
        values, duplicate keys, key combinations missing from ``frame``
        without ``default``, keys outside ``coords`` and key coordinates with
        duplicate labels.
    TypeError
        For a frame type other than a polars or pandas DataFrame, a non-numeric
        value column and key dtypes (numeric, datetime with its time zone,
        date, duration, string) differing from the ones of ``coords``.

    Warns
    -----
    EvolvingAPIWarning
        :func:`from_long` is part of the newly-added long-form API; its signature and
        input format may change in minor releases. Silence with
        ``warnings.filterwarnings("ignore", message=r"^long-form:",
        category=linopy.EvolvingAPIWarning)``.

    Examples
    --------
    >>> import pandas as pd
    >>> import linopy
    >>> m = linopy.Model()
    >>> i = pd.RangeIndex(3, name="i")
    >>> ub = linopy.from_long(
    ...     pd.DataFrame({"i": [0, 2], "ub": [5.0, 7.0]}), coords=[i], default=0
    ... )
    >>> ub.name
    'ub'
    >>> m.add_variables(upper=ub, coords=[i], name="x").upper.values
    array([5., 0., 7.])
    """
    _warn_evolving_api(
        "from_long",
        "long-form: from_long is a new API; its signature and input format may "
        "change in minor releases. Please share your use cases at "
        "https://github.com/PyPSA/linopy/issues. This warning fires once per "
        "session.",
    )
    _check_frame(frame)
    dims = list(_coords_to_dict(coords))
    keys = [k for k in frame.columns if k in dims]
    if not keys:
        raise ValueError(f"Frame has no key column naming a dimension of {dims}.")
    values = [c for c in frame.columns if c not in dims]
    if len(values) != 1:
        raise ValueError(
            f"Frame needs exactly one value column besides the key columns "
            f"{keys} naming dimensions of {dims}, got {values}."
        )
    (value,) = values
    if _family(frame[value]) != "numeric":
        raise TypeError(
            f"Value column {value!r} has non-numeric dtype {frame[value].dtype}."
        )
    if _has_nulls(frame[value]) and not _is_float_numpy(frame[value]):
        raise ValueError(f"Value column {value!r} has null values.")
    target, flat, shape, size = _flat(
        frame,
        keys,
        coords,
        strict=default is None,
        hint="; pass `default=` to fill them",
    )
    data = frame[value].to_numpy().astype(float)
    if len(flat) == size and flat.is_sorted():
        out = data
    else:
        out = np.full(size, default, dtype=float)
        out[flat.to_numpy()] = data
    return DataArray(
        out.reshape(shape), coords=[target[k] for k in keys], dims=keys, name=value
    )


@dataclass(frozen=True, eq=False)
class Relation:
    """
    A many-to-many map from the member cells of one or several dimensions to
    groups.

    Each row of ``frame`` relates the member cell in the ``member`` columns,
    each a coordinate of the dimension of the same name, to the group in
    column ``group``; a member cell may be listed in several rows.
    ``expr.groupby(relation).sum()`` adds each row's member cell to its
    group, so a member cell listed in several rows lands in several groups
    and one not listed in none.

    Parameters
    ----------
    frame : polars.DataFrame or pandas.DataFrame
        The member-group pairs, exactly the ``member`` columns and ``group``.
    member : str or sequence of str
        Name of the member column or columns, each a dimension of the grouped
        expression.
    group : str
        Name of the group column, the dimension of the result.
    groups : pandas.Index, optional
        The group labels the result spans, groups without a member are zero.
        Without it, the result spans the groups listed in ``frame``.

    Warns
    -----
    EvolvingAPIWarning
        :class:`Relation` is part of the newly-added long-form API; its signature and
        input format may change in minor releases. Silence with
        ``warnings.filterwarnings("ignore", message=r"^long-form:",
        category=linopy.EvolvingAPIWarning)``.

    Examples
    --------
    >>> import pandas as pd
    >>> import linopy
    >>> m = linopy.Model()
    >>> link = pd.Index(["l0", "l1"], name="link")
    >>> port = pd.Index([0, 1], name="port")
    >>> p = m.add_variables(coords=[link, port], name="p")
    >>> ports = pd.DataFrame(
    ...     {"link": ["l0", "l0", "l1"], "port": [0, 1, 1], "bus": ["b0", "b1", "b1"]}
    ... )
    >>> p.groupby(linopy.Relation(ports, ["link", "port"], "bus")).sum().nterm
    2
    """

    frame: Frame
    member: str | Sequence[str]
    group: str
    groups: pd.Index | None = None

    def __post_init__(self) -> None:
        _warn_evolving_api(
            "Relation",
            "long-form: Relation is a new API; its signature and input format may "
            "change in minor releases. Please share your use cases at "
            "https://github.com/PyPSA/linopy/issues. This warning fires once per "
            "session.",
            stacklevel=4,
        )
        _check_frame(self.frame)
        members, group = self.members, self.group
        columns = list(self.frame.columns)
        if group in members:
            raise ValueError(
                f"Relation member and group columns must differ, both are {group!r}."
            )
        missing = [c for c in (*members, group) if c not in columns]
        if missing:
            raise ValueError(f"Relation frame has no columns {missing}.")
        if len(columns) != len(members) + 1:
            raise ValueError(
                f"Relation frame must have exactly the columns {[*members, group]}, "
                f"got {columns}."
            )
        if _has_nulls(self.frame[group]):
            raise ValueError(f"Relation group column {group!r} has null values.")
        if self.groups is not None:
            _positions(self.groups.rename(group), self.frame[group])

    @property
    def members(self) -> tuple[str, ...]:
        """The member columns."""
        member = self.member
        return (member,) if isinstance(member, str) else tuple(member)

    def group_of(self, coords: CoordsLike) -> DataArray:
        """
        The group of each member cell over the member dims of ``coords``.

        Every member cell of ``coords`` must be listed in exactly one row,
        the result keeps the dtype of the group column.

        Examples
        --------
        >>> import pandas as pd
        >>> import linopy
        >>> gen = pd.Index(["g0", "g1"], name="gen")
        >>> frame = pd.DataFrame({"gen": ["g1", "g0"], "bus": ["b1", "b0"]})
        >>> linopy.Relation(frame, "gen", "bus").group_of([gen]).values
        array(['b0', 'b1'], dtype=object)
        """
        members = self.members
        target, flat, shape, _ = _flat(
            self.frame,
            members,
            coords,
            strict=True,
            duplicate_hint="; group_of needs exactly one row per member cell",
        )
        values = self.frame[self.group].to_numpy()
        out = np.empty_like(values)
        out[flat.to_numpy()] = values
        return DataArray(
            out.reshape(shape),
            coords=[target[m] for m in members],
            dims=members,
            name=self.group,
        )
