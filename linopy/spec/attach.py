"""
Attach user data to a mathspec program.

The language fixes three attachment rules and this module enforces them: a
dimension's members come only from the source keyed by the dimension's
name, their order is the source's order and is never sorted, and a
parameter or relation source is read for values, never for labels. Parameters
are resolved from ``sources`` on demand and aligned onto the master
coordinates without copying an already aligned array. A coordinate a table
leaves out is a missing row, and the parameter's ``missing:`` says what it
means: refused, absent (NaN) or a value. A
parameter with a ``where:`` is absent outside it, whatever its ``missing:``.
A NaN in a grid (a DataArray or a Dataset variable) is a missing row too,
while a table says a row is missing by leaving it out. Read leaves a missing
row a hole, and :func:`fill` applies ``missing:``.

Each kind of data takes one shape, and anything else is refused rather than
guessed at. A table is a pandas or polars DataFrame carrying its labels in
columns named after the dimensions or roles, never in its index, and no
column beyond those it is read for. A grid is a DataArray over the declared
dimensions. A dimension is a flat sequence of unique labels.
"""

from __future__ import annotations

import difflib
import functools
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from mathspec import program as ms
from pandas.api.extensions import ExtensionArray

from linopy.spec.errors import (
    SpecDataError,
    coordinate,
    coordinates_shown,
    first_coordinates,
    shown,
    unknown,
)

_ACCEPTED_KINDS: dict[str, frozenset[str]] = {
    "float": frozenset("fiu"),
    "int": frozenset("iu"),
    "bool": frozenset("b"),
    "str": frozenset("OUS"),
    "datetime": frozenset("M"),
}
_KIND_NAMES = {k: n for n, kinds in _ACCEPTED_KINDS.items() for k in kinds}
_EMPTY_DTYPES: dict[str, Any] = {
    "float": float,
    "int": int,
    "bool": bool,
    "str": object,
}
_SCALARS = (bool, int, float, str, np.number, np.bool_)
_DIMENSION_SHAPES = "a pandas Index, a list, a tuple, a 1-D numpy or pandas array, or a pandas or polars Series"
_UNIQUE_HINT = (
    "To take a dimension's labels from a table column, pass its unique values in order of "
    "first appearance: table['{dim}'].unique() for pandas, "
    "table['{dim}'].unique(maintain_order=True) for polars."
)


class Parameter(NamedTuple):
    """
    A parameter's values on the master coordinates beside where the data has a row.

    :meth:`Attached.read` leaves a missing row a NaN hole, an ``int`` or
    ``bool`` parameter with one then held as ``float``; :func:`fill` makes it
    read the way the declaration's ``missing:`` says. ``present`` keeps a
    missing row distinguishable from a supplied value equal to the fill.
    """

    values: xr.DataArray
    present: xr.DataArray


def attach(program: ms.Program, sources: Mapping[str, Any] | xr.Dataset) -> Attached:
    """
    Attach *sources* to *program*: master coordinates and relations now, parameters on demand.

    Parameters
    ----------
    program
        The lowered spec: ``mathspec.to_spec(spec).expand("piecewise").program``.
    sources
        Data keyed by declared name. Any mapping works: membership is
        read off ``keys()``, never by iterating, and values by key. An
        ``xr.Dataset`` is read the way ``store=True`` writes
        ``model.parameters``: its indexes are dimension sources, its data
        variables parameters and relations.

    Raises
    ------
    SpecDataError
        A program reading what it does not build (``given:``), a reached
        dimension or a relation with no source, a duplicated dimension
        member, or a relation breaking the rules a map has.
    """
    if program.given:
        raise SpecDataError(
            "the spec reads names under given: that another model would provide, and "
            "linopy builds a spec on its own, with no host model to look them up in."
        )
    if isinstance(sources, xr.Dataset):
        sources = _from_dataset(program, sources)
    keys = frozenset(sources.keys())
    _warn_near_misses(keys, program)
    coords = _master_coords(program, sources, keys)
    relations = _relations(program, sources, keys, coords)
    return Attached(program, coords, relations, sources, keys)


@dataclass(frozen=True, eq=False)
class Attached:
    """
    A program attached to its data.

    Attributes
    ----------
    program
        The lowered spec the data is attached to.
    coords
        Master coordinates by dimension, in source order, each index
        named after its dimension. A declared dimension nothing reaches
        and nothing supplies is absent.
    relations
        By relation name, its rows checked against the master coordinates.
    sources
        The caller's data, read by key on demand, one ``__getitem__`` per read.
    keys
        The keys of ``sources``, read once at attach time.
    """

    program: ms.Program
    coords: Mapping[str, pd.Index]
    relations: Mapping[str, Table]
    sources: Mapping[str, Any]
    keys: frozenset[str]

    def read(self, name: str) -> Parameter:
        """
        The parameter *name* resolved from ``sources`` and aligned to ``coords``, unfilled.

        Resolved on every call and never cached. A missing row is a NaN hole
        in ``values``, which :func:`fill` makes read the way the parameter's
        ``missing:`` says. ``present`` is where the data has a row.

        Raises
        ------
        KeyError
            The program declares no parameter *name*.
        SpecDataError
            No data, a shape no reader accepts, a rank other
            than declared, a label its dimension lacks, two rows for one
            coordinate, a null value in a table row, values of another type
            than declared.
        """
        return self.check(name)()

    def require(self, names: Iterable[str]) -> None:
        """
        Refuse the first of the parameters *names* that ``sources`` holds no data for, reading none.

        Raises
        ------
        SpecDataError
            A parameter without data.
        """
        for name in names:
            if name not in self.keys:
                raise SpecDataError(f"no data provided for parameter '{name}'")

    def check(self, name: str) -> Callable[[], Parameter]:
        """
        Refuse the data of parameter *name* as :meth:`read` would, and return the call laying it out on ``coords``.

        The check reads the data's rows or cells and leaves the grid to the
        call, so a parameter no declaration reads is checked without a grid.

        Raises
        ------
        KeyError, SpecDataError
            What :meth:`read` raises.
        """
        if name not in self.program.parameters:
            raise unknown("parameter", name, self.program.parameters)
        if name not in self.keys:
            raise SpecDataError(f"no data provided for parameter '{name}'")
        declared = self.program.parameters[name]
        lay_out = _as_array(name, declared, self.sources[name], self.coords)
        if declared.dtype != "bool":
            return lay_out

        def read() -> Parameter:
            arr, present = lay_out()
            return Parameter(arr if bool(present.all()) else arr.astype(float), present)

        return read


# ---------------------------------------------------------------------------
# sources and keys
# ---------------------------------------------------------------------------


def _from_dataset(program: ms.Program, data: xr.Dataset) -> dict[str, Any]:
    """
    *data* keyed by name, read the way ``model.parameters`` stores a spec's data.

    Its indexes are dimension sources and its data variables parameters. A
    relation of one key and one value column is stored as its value column
    over the key's dimension, NaN where a label is unmapped, and is read back
    into its rows, its values at the dtype of the dimension they label, which
    the NaN of an unmapped label turns to float. Any other relation variable
    stays as it is and is refused.
    """
    out: dict[str, Any] = {str(k): v for k, v in data.indexes.items()}
    out.update({str(k): v for k, v in data.data_vars.items()})
    for name, rel in program.relations.items():
        if name not in out or len(rel.key) != 1 or len(rel.values) != 1:
            continue
        (key,), (value,) = rel.key, rel.values
        if out[name].dims == (rel.dim(key),):
            rows = out[name].to_series().dropna().rename(value)
            if (into := out.get(rel.dim(value))) is not None:
                rows = rows.astype(into.dtype)
            out[name] = rows.rename_axis(key).reset_index()
    return out


def _warn_near_misses(keys: frozenset[str], program: ms.Program) -> None:
    """Warn about a key naming nothing the spec declares and close to a name it does that has no key: a typo, which leaves that name without data."""
    declared = {*program.parameters, *program.dimensions, *program.relations}
    unsupplied = sorted(declared.difference(keys))
    typos = {
        k: near[0]
        for k in sorted(keys.difference(declared))
        if (near := difflib.get_close_matches(k, unsupplied, n=1, cutoff=0.6))
    }
    if not typos:
        return
    warnings.warn(
        f"source key(s) {shown(sorted(typos))} name neither a parameter, a dimension nor "
        f"a relation this spec declares, and read like a typo: "
        f"{shown([f'{k} -> {near}' for k, near in sorted(typos.items())])}. "
        f"An unknown key is ignored, so a mistyped one leaves its declaration without data.",
        UserWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# dimensions
# ---------------------------------------------------------------------------


def reached_dimensions(program: ms.Program) -> set[str]:
    p = program
    sections = (p.parameters, p.variables, p.constraints, p.relations)
    return {d for section in sections for decl in section.values() for d in decl.dims}


def _master_coords(
    program: ms.Program, sources: Mapping[str, Any], keys: frozenset[str]
) -> dict[str, pd.Index]:
    reached = reached_dimensions(program)
    coords: dict[str, pd.Index] = {}
    for dim in program.dimensions:
        if dim in keys:
            coords[dim] = _index(dim, sources[dim], program.dimensions[dim])
        elif dim in reached:
            raise SpecDataError(
                f"dimension '{dim}' has no index: pass its labels under key '{dim}' as "
                f"{_DIMENSION_SHAPES}. The index is what says which labels exist, and without "
                f"one a mistyped label is indistinguishable from a new one. "
                + _UNIQUE_HINT.format(dim=dim)
            )
    return coords


def _index(dim: str, obj: Any, declared: ms.Dimension) -> pd.Index:
    obj = _pandas(obj)
    if isinstance(obj, (set, frozenset)):
        raise SpecDataError(
            f"index for dimension '{dim}' arrived as a {type(obj).__name__}, which has no "
            f"order, and a dimension's order is the source's. Pass an ordered sequence, "
            f"e.g. sorted(...)."
        )
    if isinstance(obj, (pd.Series, np.ndarray)):
        if obj.ndim != 1:
            raise SpecDataError(
                f"index for dimension '{dim}' is {obj.ndim}-dimensional; pass {_DIMENSION_SHAPES}."
            )
        values: Any = np.asarray(obj)
    elif isinstance(obj, (pd.Index, list, tuple, ExtensionArray)):
        values = obj
    else:
        raise SpecDataError(
            f"index for dimension '{dim}': cannot read labels out of {type(obj).__name__}; pass {_DIMENSION_SHAPES}."
        )
    flat = values if isinstance(values, pd.Index) else pd.Index(values)
    if isinstance(flat, pd.MultiIndex):
        raise SpecDataError(
            f"index for dimension '{dim}' is a MultiIndex; a spec dimension is one flat axis, "
            f"flatten or split it."
        )
    if not isinstance(flat.dtype, (np.dtype, pd.StringDtype)):
        flat = pd.Index(flat.to_numpy())
    index = flat.rename(dim)
    _check_value_dtype(dim, declared.dtype, index.dtype, what="dimension")
    if declared.one_of is not None and (
        strays := _outside(index, pd.Index(declared.one_of))
    ):
        raise SpecDataError(
            f"dimension '{dim}' has label(s) {shown(strays)} that its one_of does not "
            f"allow, which are {shown(list(declared.one_of))}. Fix the labels, or add them "
            f"to the one_of of '{dim}'."
        )
    if index.has_duplicates:
        twice = _twice(index)
        raise SpecDataError(
            f"dimension '{dim}' lists {shown(twice)} more than once. A dimension's members are a set: "
            f"each label appears once, in the order the source gives it. "
            + _UNIQUE_HINT.format(dim=dim)
        )
    return index


def _pandas(obj: Any) -> Any:
    """*obj*, a polars DataFrame or Series, as its pandas counterpart; anything else as it is."""
    if isinstance(obj, (pl.DataFrame, pl.Series)):
        return obj.to_pandas()
    return obj


def _check_table(what: str, df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Refuse a table carrying labels in a named index, or whose columns are not exactly *columns*."""
    named = [n for n in df.index.names if n is not None]
    if named:
        raise SpecDataError(
            f"{what} arrived as a DataFrame indexed by {named}, and a table carries its labels in "
            f"columns, never in its index. Move them into columns with reset_index(), so that the "
            f"table has the columns {list(columns)}."
        )
    if len(df.columns) != len(columns) or set(df.columns) != set(columns):
        raise SpecDataError(
            f"{what} arrived as a DataFrame with the columns {list(df.columns)}, and it is read "
            f"from exactly the columns {list(columns)}. Select them, e.g. table[{list(columns)}]."
        )


# ---------------------------------------------------------------------------
# relations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Table:
    """
    A relation's rows as the data gives them: one column per role of the
    declaration, each holding labels of the role's dimension, one row per
    key tuple.

    The rows are the relation: a bare relation is never laid out as a grid
    over its key, which for two columns over one dimension is that dimension
    squared. A value column reads over the key's dimensions through
    :meth:`column`.
    """

    declared: ms.Relation
    rows: pd.DataFrame
    coords: Mapping[str, pd.Index]
    _columns: dict[str, xr.DataArray] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    @property
    def key_dims(self) -> list[str]:
        return [self.declared.dim(role) for role in self.declared.key]

    def column(self, role: str) -> xr.DataArray:
        """The value column *role* over the key's dimensions, NaN where the key has no row."""
        if role not in self._columns:
            self._columns[role] = self._on_key(self.rows[role])
        return self._columns[role]

    @functools.cached_property
    def defined(self) -> xr.DataArray:
        """Where the relation has a row, over the key's dimensions."""
        return self._on_key(pd.Series(np.ones(len(self.rows)))).notnull()

    def _on_key(self, values: pd.Series) -> xr.DataArray:
        dims = self.key_dims
        key = pd.MultiIndex.from_frame(self.rows[list(self.declared.key)], names=dims)
        arr = _numpy(xr.DataArray.from_series(values.set_axis(key)))
        return arr.reindex({d: self.coords[d] for d in dims})


def _relations(
    program: ms.Program,
    sources: Mapping[str, Any],
    keys: frozenset[str],
    coords: Mapping[str, pd.Index],
) -> dict[str, Table]:
    out: dict[str, Table] = {}
    for name, rel in program.relations.items():
        if name not in keys:
            raise SpecDataError(
                f"no data provided for relation '{name}'. Pass it under key '{name}' as "
                f"{_relation_shapes(rel)}, with a row for each key tuple it holds and "
                f"none for one it does not."
            )
        rows = _relation_rows(name, rel, sources[name])
        _check_relation(rows, name, rel, coords)
        out[name] = Table(rel, rows, coords)
        _refuse_unmapped(name, out[name])
    return out


def _relation_shapes(rel: ms.Relation) -> str:
    return f"a pandas or polars DataFrame with exactly the columns {list(rel.roles)}"


def _named(roles: Sequence[Any]) -> str:
    return repr(roles[0]) if len(roles) == 1 else repr(tuple(roles))


def _numpy(arr: xr.DataArray) -> xr.DataArray:
    """
    *arr* backed by a numpy array.

    xarray keeps a pandas extension array as it arrives, and pandas 3 hands
    strings over as one. Its ``dtype`` is no ``np.dtype``, so nothing
    downstream that records or restores a dtype can name it, and xarray's
    positional indexing refuses the Arrow-backed variant.
    """
    if isinstance(arr.dtype, np.dtype):
        return arr
    return arr.copy(data=arr.to_numpy())


def _twice(index: pd.Index) -> list[Any]:
    return index[index.duplicated()].unique().tolist()


def _outside(labels: pd.Index, known: pd.Index) -> list[Any]:
    return labels[~labels.isin(known)].unique().tolist()


def _relation_rows(name: str, rel: ms.Relation, obj: Any) -> pd.DataFrame:
    """*obj* as the relation's rows, one column per role."""
    roles = list(rel.roles)
    obj = _pandas(obj)
    if not isinstance(obj, pd.DataFrame):
        raise SpecDataError(
            f"relation '{name}': cannot read rows out of {type(obj).__name__}; pass "
            f"{_relation_shapes(rel)}, one row per key tuple."
        )
    _check_table(f"relation '{name}'", obj, roles)
    return obj[roles].reset_index(drop=True)


def _check_relation(
    rows: pd.DataFrame,
    name: str,
    rel: ms.Relation,
    coords: Mapping[str, pd.Index],
) -> None:
    key = list(rel.key)
    for role in rel.roles:
        holes = rows[role].isna()
        if holes.any():
            nulls = rows.loc[holes, key].head(5).itertuples(index=False, name=None)
            raise SpecDataError(
                f"relation '{name}' carries {int(holes.sum())} row(s) with a null in "
                f"'{role}': {coordinates_shown(key, nulls)}. A relation is partial by "
                f"leaving a row out, not by a null in it: drop the row and the key is "
                f"unmapped, which is what every operator reading the relation already "
                f"means by it."
            )
    keys = (
        pd.MultiIndex.from_frame(rows[key]) if len(key) > 1 else pd.Index(rows[key[0]])
    )
    if keys.has_duplicates:
        twice = _twice(keys)
        raise SpecDataError(
            f"relation '{name}' maps {len(twice)} {_named(key)} label(s) more than once: "
            f"{shown(twice)}. A relation has one row per key tuple, so each label it maps "
            f"takes exactly one row."
        )
    for role in rel.roles:
        _refuse_stray_labels(rows[role], name, rel, role, coords)


def _refuse_stray_labels(
    column: pd.Series,
    name: str,
    rel: ms.Relation,
    role: str,
    coords: Mapping[str, pd.Index],
) -> None:
    dim = rel.dim(role)
    labels, known = pd.Index(column), coords[dim]
    strays = _outside(labels, known)
    if not strays:
        return
    _refuse_label_type(f"relation '{name}'", dim, labels, known)
    if role in rel.key:
        raise SpecDataError(
            f"relation '{name}' maps {shown(strays)}, which are not labels of '{dim}'. "
            f"'{dim}' takes its labels from sources['{dim}'], and they are "
            f"{shown(known.tolist(), 8)}. A map maps the labels that exist: a key matching "
            f"none of them would place its terms nowhere, so it is a typo on one side or a label "
            f"missing from the other."
        )
    raise SpecDataError(
        f"relation '{name}' has value(s) that are not '{dim}' labels: "
        f"{shown(strays)}. Every value must be a declared '{dim}' label, otherwise "
        f"a sum through {name} drops those terms in the join that places them, and the "
        f"model builds and solves without them."
    )


def _refuse_unmapped(name: str, table: Table) -> None:
    """Refuse a ``missing: refused`` relation lacking a key tuple, a keyed one naming the first few."""
    rel = table.declared
    grid = int(np.prod([len(table.coords[d]) for d in table.key_dims]))
    lacking = grid - len(table.rows)
    if rel.missing != "refused" or not lacking:
        return
    first = (
        f", the first at {first_coordinates(~table.defined, 3)}" if rel.values else ""
    )
    raise SpecDataError(
        f"relation '{name}' has no row for {lacking} {_named(rel.key)} label(s){first}. "
        f"Every key tuple needs a row, or the relation declares missing: absent."
    )


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


def _as_array(
    name: str,
    declared: ms.Parameter,
    obj: Any,
    coords: Mapping[str, pd.Index],
) -> Callable[[], Parameter]:
    """*obj* checked as the data of parameter *name*, and the call laying it out on *coords*."""
    obj = _pandas(obj)
    dims = declared.dims
    if isinstance(obj, xr.DataArray):
        return _from_dense(name, declared, obj, coords)
    if dims and isinstance(obj, pd.DataFrame):
        return _from_table(name, declared, obj, coords)
    if not dims and isinstance(obj, _SCALARS):
        return _from_scalar(name, declared, obj)
    if not dims:
        raise SpecDataError(
            f"parameter '{name}' is declared with no dims, which means one value, and it "
            f"arrived as {type(obj).__name__}. Pass one number, or declare the dims it is "
            f"indexed by."
        )
    raise SpecDataError(
        f"parameter '{name}': cannot read {type(obj).__name__} as values over {list(dims)}; "
        f"pass a pandas or polars DataFrame with exactly the columns {[*dims, name]}, or a "
        f"DataArray over {list(dims)}."
    )


def _from_scalar(
    name: str, declared: ms.Parameter, obj: Any
) -> Callable[[], Parameter]:
    if pd.isna(obj):
        raise SpecDataError(
            f"parameter '{name}' is one value and that value is a hole (null or NaN). "
            f"A number was meant, or the parameter has no data and should not be passed."
        )
    value = np.asarray(obj)
    _check_value_dtype(name, declared.dtype, value.dtype)
    if declared.dtype == "float":
        value = value.astype(float)
    arr = xr.DataArray(value, name=name)
    parameter = Parameter(arr, xr.ones_like(arr, dtype=bool))
    return lambda: parameter


def _from_dense(
    name: str,
    declared: ms.Parameter,
    arr: xr.DataArray,
    coords: Mapping[str, pd.Index],
) -> Callable[[], Parameter]:
    dims = declared.dims
    _check_dense_dtype(name, declared.dtype, arr)
    if set(arr.dims) != set(dims):
        raise SpecDataError(
            f"parameter '{name}' arrived as a DataArray over {list(arr.dims)}, and '{name}' is over "
            f"{list(dims)}. The dims must be the declared ones, in any order."
        )
    for d in dims:
        if d not in arr.indexes:
            raise SpecDataError(
                f"parameter '{name}' has no coordinate labels along '{d}'. A parameter is read for "
                f"values against its labels, so every dimension needs an index coordinate."
            )
        _refuse_duplicate_coordinates(name, (d,), arr.indexes[d])
        if not arr.indexes[d].equals(coords[d]):
            _refuse_strangers(name, d, arr.indexes[d], coords[d])
    onto = {d: coords[d] for d in dims}
    return functools.partial(_aligned, arr.transpose(*dims), onto)


def _check_dense_dtype(name: str, declared: str, arr: xr.DataArray) -> None:
    """Refuse values of another type than declared, accepting floats with a hole for an ``int`` or ``bool`` grid."""
    held = {"int": "whole numbers", "bool": "0 or 1"}
    if declared not in held or arr.dtype.kind != "f" or not bool(arr.isnull().any()):
        _check_value_dtype(name, declared, arr.dtype)
        return
    present = arr.values[~np.isnan(arr.values)]
    fits = np.isin(present, (0.0, 1.0)) if declared == "bool" else present % 1 == 0
    if fits.all():
        return
    raise SpecDataError(
        f"parameter '{name}' is declared '{declared}' and arrived as floats with a hole, which is "
        f"how a grid holds an '{declared}' with missing rows, but {shown(present[~fits].tolist())} "
        f"are not {held[declared]}. Fix the values, or declare what the data has: {{dtype: float}}"
    )


def _from_table(
    name: str,
    declared: ms.Parameter,
    df: pd.DataFrame,
    coords: Mapping[str, pd.Index],
) -> Callable[[], Parameter]:
    """
    A table of rows, one column per dim and the values under the parameter's name.

    A coordinate the table leaves out is a missing row; a row holding no value
    is refused, since it says the coordinate exists and denies it at once.
    """
    dims = declared.dims
    _check_table(f"parameter '{name}'", df, [*dims, name])
    series = df.set_index(list(dims))[name]
    if series.empty:
        series = series.astype(_EMPTY_DTYPES[declared.dtype])
    holes = series.isna()
    if holes.any():
        raise SpecDataError(
            f"parameter '{name}' carries {int(holes.sum())} row(s) with no value, null or NaN: "
            f"{coordinates_shown(dims, series.index[holes][:3])}. In a table the absence of a "
            f"value is the absence of the row, and such a row says the coordinate exists and denies "
            f"it in the same breath. Drop those rows, or supply the values."
        )
    _check_value_dtype(name, declared.dtype, series.dtype)
    _refuse_duplicate_coordinates(name, dims, series.index)
    onto = {d: coords[d] for d in dims}
    positions = []
    for d, index in onto.items():
        labels = series.index.get_level_values(d)
        _refuse_strangers(name, d, labels, index)
        positions.append(_positions(name, d, labels, index))
    at = np.ravel_multi_index(positions, tuple(len(i) for i in onto.values()))
    return functools.partial(_on_grid, at, series.to_numpy(), onto)


def _positions(name: str, dim: str, labels: pd.Index, index: pd.Index) -> np.ndarray:
    """The position of each of *labels* in *index*, refusing a label equal to a coordinate in value only."""
    found = index.get_indexer(labels)
    if (found >= 0).all():
        return found
    raise SpecDataError(
        f"parameter '{name}' has label(s) in dimension '{dim}' that equal coordinates of it in "
        f"value but not in type: {shown(labels[found < 0].unique().tolist())}, and '{dim}' holds "
        f"'{_kind_name(index.dtype)}' labels. Cast them to the dimension's type."
    )


def _on_grid(
    at: np.ndarray, values: np.ndarray, onto: Mapping[str, pd.Index]
) -> Parameter:
    """
    The *values* placed at the flat positions *at* of the grid *onto*, a cell no value reaches a hole.

    One array the size of the grid is allocated, beside the mask of where a
    row is: the grid of a table is its dims' full product, whatever few rows
    it holds.
    """
    shape = tuple(len(index) for index in onto.values())
    present = np.zeros(int(np.prod(shape)), dtype=bool)
    present[at] = True
    if len(at) == present.size:
        grid = np.empty(present.size, dtype=values.dtype)
    else:
        grid = np.full(present.size, *_hole(values.dtype))
    grid[at] = values
    dims = list(onto)
    return Parameter(
        xr.DataArray(grid.reshape(shape), coords=onto, dims=dims),
        xr.DataArray(present.reshape(shape), coords=onto, dims=dims),
    )


def _hole(dtype: np.dtype) -> tuple[float, Any]:
    """A hole, NaN, and the dtype of an array holding *dtype*'s values beside one: float for a number, object for a string."""
    if dtype.kind == "f":
        return np.nan, dtype
    if dtype.kind in "iub":
        return np.nan, np.float64
    return np.nan, object


def _refuse_duplicate_coordinates(
    name: str, dims: tuple[str, ...], index: pd.Index
) -> None:
    duplicated = index.duplicated()
    if not duplicated.any():
        return
    counts = index[duplicated].value_counts()
    shown = "; ".join(
        f"{coordinate(dims, key)} ({n + 1} rows)" for key, n in counts.iloc[:3].items()
    )
    raise SpecDataError(
        f"parameter '{name}' has more than one row for a coordinate: {shown}. A parameter is a "
        f"function of its dims, so which value applies is undefined; aggregate the source to one "
        f"row per {list(dims)} before attaching it."
    )


def _refuse_strangers(name: str, dim: str, labels: pd.Index, known: pd.Index) -> None:
    strangers = _outside(labels, known)
    if not strangers:
        return
    _refuse_label_type(f"parameter '{name}'", dim, labels, known)
    raise SpecDataError(
        f"parameter '{name}' has label(s) in dimension '{dim}' that are not coordinates of it: "
        f"{shown(strangers)}.\n  {dim} has: {shown(known.tolist(), 10)}\n"
        f"A label that is not a coordinate is a typo: its row joins nothing, so the coordinate it "
        f"was meant for is left uncovered. Fix the label, or add it to sources['{dim}']."
    )


def _refuse_label_type(what: str, dim: str, labels: pd.Index, known: pd.Index) -> None:
    arrived, expected = _kind_name(labels.dtype), _kind_name(known.dtype)
    if arrived == expected or labels.isin(known).any():
        return
    hint = (
        " Parse them with pd.to_datetime, or with parse_dates= when reading the table."
        if expected == "datetime"
        else ""
    )
    raise SpecDataError(
        f"{what} has '{arrived}' label(s) in dimension '{dim}', and '{dim}' holds '{expected}' "
        f"labels: {shown(_outside(labels, known))} match none of {shown(known.tolist())}, as "
        f"they differ in type, not spelling.{hint}"
    )


def _aligned(arr: xr.DataArray, onto: Mapping[str, pd.Index]) -> Parameter:
    """The grid *arr* on the master coordinates *onto*, without copying one already on them."""
    arr = _numpy(arr)
    present = arr.notnull()
    if all(arr.indexes[d].equals(index) for d, index in onto.items()):
        stale = {d: i for d, i in onto.items() if arr.indexes[d].dtype != i.dtype}
        if stale:
            return Parameter(arr.assign_coords(stale), present.assign_coords(stale))
        return Parameter(arr, present)
    return Parameter(
        arr.reindex(onto, fill_value=np.nan), present.reindex(onto, fill_value=False)
    )


def fill(
    program: ms.Program,
    name: str,
    parameter: Parameter,
    admitted: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    The values of *parameter* with its missing rows reading the way *program* declares *name* ``missing:``.

    *parameter* is what :meth:`Attached.read` returns, so
    ``fill(attached.program, name, attached.read(name))`` reads a parameter
    filled. An ``absent`` row is NaN and a value row that value; an ``int`` or ``bool`` parameter is back at its declared
    dtype unless a row stays NaN. A ``refused`` row stays NaN too: whether a
    declaration reads it is for :func:`linopy.spec.evaluator.refuse_holes` to
    decide. *admitted* is where the parameter's ``where:`` holds: outside it,
    the parameter is absent and its rows of data are dropped.
    """
    arr, present = parameter
    declared = program.parameters[name]
    missing = declared.missing
    if missing not in ("refused", "absent") and not bool(present.all()):
        arr = _with_fill(arr, present, missing, declared.dtype)
    if admitted is None:
        return arr
    return arr.where(admitted).transpose(*arr.dims)


def _with_fill(
    arr: xr.DataArray, present: xr.DataArray, value: Any, dtype: str
) -> xr.DataArray:
    filled = arr.where(present, value)
    if dtype in ("int", "bool"):
        return filled.astype(_EMPTY_DTYPES[dtype])
    return filled


def _check_value_dtype(
    name: str, declared: str, dtype: Any, what: str = "parameter"
) -> None:
    if str(dtype.kind) in _ACCEPTED_KINDS[declared]:
        return
    arrived = _kind_name(dtype)
    raise SpecDataError(
        f"{what} '{name}' is declared '{declared}' and its values arrived as '{arrived}'. "
        f"A declared dtype is a claim about the values, and it is checked here: the file says what "
        f"the values are, or the values are not attached.\n"
        f"  Cast the values to {declared}, if the declaration is what you meant\n"
        f"  Or declare what the data has: {{dtype: {arrived}}}"
    )


def _kind_name(dtype: Any) -> str:
    """The declared-dtype name of *dtype*'s kind, or the dtype spelled out when no declaration has it."""
    return _KIND_NAMES.get(str(dtype.kind), str(dtype))
