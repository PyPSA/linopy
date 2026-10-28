"""
Attach user data to a mathspec program.

The language fixes three attachment rules and this module enforces them: a
dimension's members come only from the source keyed by the dimension's
name, their order is the source's order and is never sorted, and a
parameter or relation source is read for values, never for labels. Parameters
are resolved from ``sources`` on demand and aligned onto the master
coordinates without copying an already aligned array. A coordinate a table
leaves out is a missing row, and the parameter's ``missing:`` says what it
means: refused, absent (NaN, ``False`` for ``bool``), neutral (0) or a value.
A NaN in a grid (a DataArray, a Dataset variable or a wide DataFrame) is a
missing row too, while a table says a row is missing by leaving it out. Read
leaves a missing row a hole, and :func:`fill` applies ``missing:``.
"""

from __future__ import annotations

import difflib
import functools
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr
from mathspec import program as ms
from mathspec.program import parameters_of, walk
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
_NEUTRAL: dict[str, Any] = {"float": 0.0, "int": 0, "bool": False}
_SCALARS = (bool, int, float, str, np.number, np.bool_)
_DIMENSION_SHAPES = (
    "a pandas Index, a list, a tuple, a set (read sorted), a 1-D numpy or pandas array, "
    "a pandas Series or a 1-D DataArray"
)
_PARAMETER_SHAPES = (
    "a DataArray over {dims}, a pandas Series whose (Multi)Index levels are {dims}, "
    "a DataFrame with columns {columns} or in wide form, a dict keyed by label, or one number"
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
        ``xr.Dataset`` is accepted too: its indexes are dimension sources,
        its data variables parameters and relations.

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
        tabular = (*sources.indexes.items(), *sources.data_vars.items())
        sources = {str(k): v for k, v in tabular}
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
        if name not in self.program.parameters:
            raise unknown("parameter", name, self.program.parameters)
        if name not in self.keys:
            raise SpecDataError(f"no data provided for parameter '{name}'")
        declared = self.program.parameters[name]
        arr, present = _as_array(name, declared, self.sources[name], self.coords)
        if declared.dtype == "bool" and not bool(present.all()):
            arr = arr.astype(float)
        return Parameter(arr, present)


def strict_parameters(program: ms.Program) -> frozenset[str]:
    """The parameters standing where no neutral value exists: a divisor, a bound, a whole comparison side."""
    p = program
    bounds = [b for v in p.variables.values() for b in (v.lower, v.upper)]
    sides = [_bare(s) for c in p.constraints.values() for s in (c.lhs, c.rhs)]
    bodies = [e.expression for e in p.expressions.values()]
    divisors = [n.divisor for n in walk(*p.roots, *bodies) if isinstance(n, ms.Divide)]
    references = [n for n in bounds + sides if isinstance(n, ms.ParameterReference)]
    return frozenset({r.name for r in references} | parameters_of(*divisors))


def _bare(node: ms.Expression) -> ms.Expression:
    """*node* looked through its references, powers and negations."""
    if isinstance(node, ms.ExpressionReference):
        return _bare(node.body)
    if isinstance(node, ms.Power):
        return _bare(node.base)
    if isinstance(node, ms.Negate):
        return _bare(node.operand)
    return node


# ---------------------------------------------------------------------------
# sources and keys
# ---------------------------------------------------------------------------


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
                f"one a mistyped label is indistinguishable from a new one."
            )
    return coords


def _index(dim: str, obj: Any, declared: ms.Dimension) -> pd.Index:
    if isinstance(obj, (set, frozenset)):
        obj = _sorted(dim, obj)
    if isinstance(obj, (pd.Series, xr.DataArray, np.ndarray)):
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
    if index.has_duplicates:
        twice = _twice(index)
        raise SpecDataError(
            f"dimension '{dim}' lists {shown(twice)} more than once. A dimension's members are a set: "
            f"each label appears once, in the order the source gives it."
        )
    return index


def _sorted(dim: str, members: set[Any] | frozenset[Any]) -> list[Any]:
    try:
        return sorted(members)
    except TypeError as error:
        raise SpecDataError(
            f"index for dimension '{dim}' arrived as a {type(members).__name__}, which has no "
            f"order, and its labels {shown(list(members))} do not sort; pass an ordered sequence."
        ) from error


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
    table = f"a DataFrame with the columns {list(rel.roles)}"
    if len(rel.values) != 1:
        return table
    key, dims = _named(rel.key), [rel.dim(role) for role in rel.key]
    return (
        f"{table}, a pandas Series indexed by {key}, a dict keyed by {key} labels, "
        f"or a DataArray over {dims}"
    )


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
    if isinstance(obj, pd.DataFrame):
        tidy = obj.reset_index() if set(obj.index.names) & set(roles) else obj
        if not set(roles) <= set(tidy.columns):
            raise SpecDataError(
                f"relation '{name}' arrived as a DataFrame with columns {list(tidy.columns)}, "
                f"and a table for '{name}' carries the columns {roles}, one per column "
                f"of its declaration and named after it."
            )
        return tidy[roles].reset_index(drop=True)
    if len(rel.values) != 1:
        raise SpecDataError(
            f"relation '{name}': cannot adapt {type(obj).__name__} to a table; pass "
            f"{_relation_shapes(rel)}."
        )
    series = _keyed_series(name, rel, obj)
    return series.rename(rel.values[0]).reset_index()[roles]


def _keyed_series(name: str, rel: ms.Relation, obj: Any) -> pd.Series:
    """*obj*, a map into one value column, as a Series whose index levels are the key's roles."""
    key = list(rel.key)
    if isinstance(obj, xr.DataArray):
        dims = [rel.dim(role) for role in key]
        if set(obj.dims) != set(dims) or not set(dims) <= set(obj.indexes):
            raise SpecDataError(
                f"relation '{name}' arrived as a DataArray over {list(obj.dims)}, and it is a "
                f"map out of {dims}: pass a DataArray with {dims} as its labelled dimensions."
            )
        series = obj.to_series().dropna()
        return series.rename_axis([key[dims.index(d)] for d in series.index.names])
    if isinstance(obj, Mapping):
        obj = pd.Series(dict(obj))
    if not isinstance(obj, pd.Series):
        raise SpecDataError(
            f"relation '{name}': cannot adapt {type(obj).__name__} to a map; pass "
            f"{_relation_shapes(rel)}."
        )
    names = list(obj.index.names)
    if all(n is None for n in names) and len(names) == len(key):
        return obj.rename_axis(key)
    if set(names) != set(key):
        raise SpecDataError(
            f"relation '{name}' is a Series indexed by {_named(names)}, and it is a map out "
            f"of {_named(key)}: index it by {_named(key)} labels."
        )
    return obj


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
) -> Parameter:
    if isinstance(obj, xr.DataArray):
        return _from_dense(name, declared, obj, coords)
    if isinstance(obj, pd.DataFrame):
        return _from_frame(name, declared, obj, coords)
    if isinstance(obj, pd.Series):
        return _from_rows(name, declared, obj, coords)
    if isinstance(obj, Mapping):
        return _from_rows(name, declared, pd.Series(dict(obj)), coords)
    if isinstance(obj, _SCALARS):
        return _from_scalar(name, declared, obj, coords)
    dims = declared.dims
    raise SpecDataError(
        f"parameter '{name}': cannot adapt {type(obj).__name__} to an array over {list(dims)}; "
        f"pass {_parameter_shapes(name, dims)}."
    )


def _parameter_shapes(name: str, dims: Sequence[str]) -> str:
    return _PARAMETER_SHAPES.format(dims=list(dims), columns=_columns(name, dims))


def _columns(name: str, dims: Sequence[str]) -> str:
    return f"{[*dims, name]} or {[*dims, 'value']}"


def _from_scalar(
    name: str,
    declared: ms.Parameter,
    obj: Any,
    coords: Mapping[str, pd.Index],
) -> Parameter:
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
    if declared.dims:
        arr = arr.expand_dims({d: coords[d] for d in declared.dims})
    return Parameter(arr, xr.ones_like(arr, dtype=bool))


def _from_dense(
    name: str,
    declared: ms.Parameter,
    arr: xr.DataArray,
    coords: Mapping[str, pd.Index],
) -> Parameter:
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
    onto = {d: coords[d] for d in dims}
    return _aligned(name, _numpy(arr.transpose(*dims)), onto)


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


def _from_frame(
    name: str,
    declared: ms.Parameter,
    df: pd.DataFrame,
    coords: Mapping[str, pd.Index],
) -> Parameter:
    dims = declared.dims
    tidy = df if set(dims) <= set(df.columns) else df.reset_index()
    column = name if name in tidy.columns else "value"
    if column in tidy.columns and set(dims) <= set(tidy.columns):
        indexed = tidy.set_index(list(dims)) if dims else tidy
        return _from_rows(name, declared, indexed[column], coords)
    if len(dims) == 2:
        wide = xr.DataArray(_wide(name, dims, df))
        return _from_dense(name, declared, wide, coords)
    raise SpecDataError(
        f"parameter '{name}' arrived as a DataFrame with columns {list(df.columns)}; a table for "
        f"'{name}' carries columns {_columns(name, dims)}."
    )


def _wide(name: str, dims: tuple[str, ...], df: pd.DataFrame) -> pd.DataFrame:
    names = (df.index.name, df.columns.name)
    if names == (None, None):
        if len(df.index) == len(df.columns):
            raise SpecDataError(
                f"parameter '{name}' arrived as a {len(df.index)}x{len(df.columns)} wide DataFrame "
                f"with neither axis named, and '{name}' is over {list(dims)}. A square frame does "
                f"not say which axis is which. Name the index and columns after the two dims, or "
                f"pass a table with columns {_columns(name, dims)}."
            )
        return df.rename_axis(index=dims[0], columns=dims[1])
    if set(names) == set(dims):
        return df
    raise SpecDataError(
        f"parameter '{name}' arrived as a wide DataFrame with index '{names[0]}' and columns "
        f"'{names[1]}', and '{name}' is over {list(dims)}. Name the index and columns after the "
        f"two dims, or pass a table with columns {_columns(name, dims)}."
    )


def _from_rows(
    name: str,
    declared: ms.Parameter,
    series: pd.Series,
    coords: Mapping[str, pd.Index],
) -> Parameter:
    dims = declared.dims
    if not dims:
        if len(series) != 1:
            raise SpecDataError(
                f"parameter '{name}' is declared with no dims, which means one value broadcast "
                f"everywhere, but its source has {len(series)} rows. Declare the dims it is indexed "
                f"by, or pass one number."
            )
        return _from_scalar(name, declared, series.iloc[0], coords)
    series = _with_dims(name, dims, series)
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
    for d in dims:
        _refuse_strangers(name, d, series.index.get_level_values(d), coords[d])
    arr = _numpy(xr.DataArray.from_series(series.rename(name)))
    return _aligned(name, arr.transpose(*dims), {d: coords[d] for d in dims})


def _with_dims(name: str, dims: tuple[str, ...], series: pd.Series) -> pd.Series:
    index = series.index
    if index.nlevels != len(dims):
        raise SpecDataError(
            f"parameter '{name}': a Series or dict carries one label per level, and its index has "
            f"{index.nlevels} level(s) where '{name}' is over {list(dims)}. "
            f"Pass {_parameter_shapes(name, dims)}."
        )
    names = list(index.names)
    if all(n is None for n in names):
        return series.set_axis(index.set_names(list(dims)))
    if set(names) != set(dims):
        raise SpecDataError(
            f"parameter '{name}' is indexed by {names}, and '{name}' is over {list(dims)}. "
            f"Name the index levels after the declared dims."
        )
    if tuple(names) != dims:
        series = series.reorder_levels(list(dims))
    return series


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


def _aligned(name: str, arr: xr.DataArray, onto: Mapping[str, pd.Index]) -> Parameter:
    present = arr.notnull()
    if all(arr.indexes[d].equals(index) for d, index in onto.items()):
        stale = {d: i for d, i in onto.items() if arr.indexes[d].dtype != i.dtype}
        if stale:
            return Parameter(arr.assign_coords(stale), present.assign_coords(stale))
        return Parameter(arr, present)
    for d, index in onto.items():
        _refuse_strangers(name, d, arr.indexes[d], index)
    return Parameter(
        arr.reindex(onto, fill_value=np.nan), present.reindex(onto, fill_value=False)
    )


def fill(
    program: ms.Program,
    name: str,
    parameter: Parameter,
    strict: frozenset[str] | None = None,
) -> xr.DataArray:
    """
    The values of *parameter* with its missing rows reading the way *program* declares *name* ``missing:``.

    *parameter* is what :meth:`Attached.read` returns, so
    ``fill(attached.program, name, attached.read(name))`` reads a parameter
    filled. An ``absent`` row is NaN (``False`` for ``bool``), a ``neutral``
    one 0 and a value that value; an ``int`` or ``bool`` parameter is back at
    its declared dtype unless a row stays NaN. A ``refused`` row stays NaN
    too: whether a declaration reads it is for
    :func:`linopy.spec.evaluator.refuse_holes` to decide. *strict* names the parameters
    standing where no neutral value exists, worked out from *program* when
    not given; pass it to skip that walk on repeated calls.

    Raises
    ------
    SpecDataError
        A ``neutral`` missing row where no neutral value exists: as a
        divisor, a bound, or the whole constant side of a comparison.
    """
    arr, present = parameter
    declared = program.parameters[name]
    missing = declared.missing
    if missing == "refused" or bool(present.all()):
        return arr
    if missing == "absent":
        if declared.dtype == "bool":
            return _with_fill(arr, present, False, "bool")
        return arr
    if missing == "neutral":
        if name in (strict_parameters(program) if strict is None else strict):
            raise SpecDataError(
                f"parameter '{name}' is missing: neutral and has no row at "
                f"{first_coordinates(~present, 3)}, and it stands where no neutral value "
                f"exists: as a divisor, in bounds:, or as the whole constant side of a "
                f"comparison. Supply the rows, or declare missing: absent."
            )
        return _with_fill(arr, present, _NEUTRAL[declared.dtype], declared.dtype)
    return _with_fill(arr, present, missing, declared.dtype)


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
