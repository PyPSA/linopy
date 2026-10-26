"""
Attach user data to a mathspec program.

The language fixes three attachment rules and this module enforces them: a
dimension's members come only from the source keyed by the dimension's
name, their order is the source's order and is never sorted, and a
parameter or relation source is read for values, never for labels. Parameters
are resolved from ``sources`` on demand and aligned onto the master
coordinates without copying an already aligned array. A coordinate a table
leaves out is a missing row, and the parameter's ``missing:`` says what it
means: refused, absent (NaN beside a presence mask), neutral (0) or a value.
"""

from __future__ import annotations

import difflib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from mathspec import program as ms
from mathspec.errors import did_you_mean
from mathspec.program import carries_variable, parameters_of, walk

from linopy.constants import _warn_evolving_api
from linopy.spec.errors import (
    SpecDataError,
    coordinate,
    coordinates_shown,
    first_coordinates,
    shown,
)

_ACCEPTED_KINDS: dict[str, frozenset[str]] = {
    "float": frozenset("fiu"),
    "int": frozenset("iu"),
    "bool": frozenset("b"),
    "str": frozenset("OUS"),
    "datetime": frozenset("M"),
}
_KIND_NAMES: dict[str, str] = {
    "f": "float",
    "i": "int",
    "u": "int",
    "b": "bool",
    "O": "str",
    "U": "str",
    "S": "str",
    "M": "datetime",
}
_EMPTY_DTYPES: dict[str, Any] = {
    "float": float,
    "int": int,
    "bool": bool,
    "str": object,
}
_NEUTRAL: dict[str, Any] = {"float": 0.0, "int": 0, "bool": False}
_SCALARS = (bool, int, float, str, np.number, np.bool_)
_DIMENSION_SHAPES = "a pandas Index, a list, a tuple, a 1-D numpy array, a pandas Series or a 1-D DataArray"
_RELATION_SHAPES = "a pandas Series indexed by '{key}', a dict keyed by '{key}' labels, or a 1-D DataArray over '{key}'"
_PARAMETER_SHAPES = (
    "a DataArray over {dims}, a pandas Series whose (Multi)Index levels are {dims}, "
    "a DataFrame with columns {columns} or in wide form, a dict keyed by label, or one number"
)

EVOLVING_MESSAGE = (
    "spec: Model.add_spec, Model.from_spec, model.spec and linopy.spec.attach are "
    "newly added and their details may change in minor releases. Silence with "
    '`warnings.filterwarnings("ignore", category=linopy.EvolvingAPIWarning)`.'
)

Parameter = tuple[xr.DataArray, xr.DataArray]


def attach(program: ms.Program, sources: Mapping[str, Any] | xr.Dataset) -> Attached:
    """
    Attach *sources* to *program*: master coordinates and relations now, parameters on demand.

    Parameters
    ----------
    program
        The lowered spec.
    sources
        Data keyed by declared name. Any mapping works; ``keys()`` is
        called once, and everything after that is read by key. An
        ``xr.Dataset`` is accepted too: its indexes are dimension sources,
        its data variables parameters and relations.

    Raises
    ------
    SpecDataError
        A program reading what it does not build (``given:``), a reached
        dimension or a relation with no source, a duplicated dimension
        member, or a relation breaking the rules a map has.
    TypeError
        *sources* offers no ``keys()``.
    """
    _warn_evolving_api("spec", EVOLVING_MESSAGE)
    if program.given:
        raise SpecDataError(
            "the spec reads names under given: that another model would provide, and "
            "linopy builds a spec on its own, with no host model to look them up in."
        )
    if isinstance(sources, xr.Dataset):
        sources = _dataset_sources(sources)
    read = Sources(sources, _attachable(program))
    coords = _master_coords(program, read)
    relations = _relations(program, read, coords)
    return Attached(program, coords, relations, read)


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
        By relation name, the map as an array over its key dimension's
        master coordinates, NaN where a label is unmapped.
    sources
        The caller's data, read by key on demand.
    """

    program: ms.Program
    coords: Mapping[str, pd.Index]
    relations: Mapping[str, xr.DataArray]
    sources: Sources

    def parameter(self, name: str) -> xr.DataArray:
        """
        The parameter *name* resolved from ``sources`` and aligned to ``coords``.

        Resolved on every call and never cached. The array is filled the way
        the parameter's ``missing:`` says: a ``refused`` row is refused here,
        an ``absent`` one is NaN (``False`` for ``bool``), a ``neutral`` one
        is 0 and a value is that value.

        Raises
        ------
        SpecDataError
            No data, a shape no reader accepts, a rank other
            than declared, a label its dimension lacks, two rows for one
            coordinate, a null value in a row, values of another type
            than declared, or a missing row the declaration refuses.
        """
        return self.read(name)[0]

    def read(self, name: str) -> Parameter:
        """The parameter *name* as :meth:`parameter` gives it, beside where the data has a row for it."""
        declared = self._declaration(name)
        if name not in self.sources:
            raise SpecDataError(f"no data provided for parameter '{name}'")
        arr, present = _as_array(name, declared, self.sources[name], self.coords)
        onto = {d: self.coords[d] for d in declared.dims}
        arr, present = _aligned(name, _numpy(arr), present, onto, _fill(declared))
        return _filled(name, declared, arr, present, name in self._strict), present

    def retained(self) -> tuple[xr.Dataset, xr.Dataset]:
        """The relations plus the parameters the named expressions read, and the presence mask of each parameter a row is missing from."""
        arrays: dict[str, xr.DataArray] = {}
        presence: dict[str, xr.DataArray] = {}
        for name in self._retained_names():
            arrays[name], present = self.read(name)
            if not bool(present.all()):
                presence[name] = present
        arrays.update(self.relations)
        return xr.Dataset(arrays), xr.Dataset(presence)

    def _declaration(self, name: str) -> ms.Parameter:
        if name not in self.program.parameters:
            raise SpecDataError(
                f"unknown parameter '{name}'. {did_you_mean(name, self.program.parameters)}"
            )
        return self.program.parameters[name]

    def _retained_names(self) -> list[str]:
        keep = _report_closure(self.program)
        return [n for n in self.program.parameters if n in keep]

    @cached_property
    def _strict(self) -> frozenset[str]:
        """The parameters standing where no neutral value exists: a divisor, a bound, or the whole constant side of a comparison."""
        return _strict_positions(self.program)


def _strict_positions(program: ms.Program) -> frozenset[str]:
    names: set[str] = set()
    for variable in program.variables.values():
        for bound in (variable.lower, variable.upper):
            if isinstance(bound, ms.ParameterReference):
                names.add(bound.name)
    bodies = [e.expression for e in program.expressions.values()]
    for node in walk(*program.roots, *bodies):
        if isinstance(node, ms.Divide):
            names |= parameters_of(node.divisor)
    for constraint in program.constraints.values():
        for side in (constraint.lhs, constraint.rhs):
            side = _bare(side)
            if isinstance(side, ms.ParameterReference) and not carries_variable(side):
                names.add(side.name)
    return frozenset(names)


def _bare(node: ms.Expression) -> ms.Expression:
    """*node* looked through its references, powers and negations."""
    if isinstance(node, ms.ExpressionReference):
        return _bare(node.body)
    if isinstance(node, ms.Power):
        return _bare(node.base)
    if isinstance(node, ms.Negate):
        return _bare(node.operand)
    return node


def amounts_of(node: ms.Expression) -> list[str]:
    """The parameters *node* names as an amount: a translation's offset or a window's width."""
    if isinstance(node, ms.Translate) and isinstance(node.offset, str):
        return [node.offset]
    if isinstance(node, ms.WindowSum) and isinstance(node.width, str):
        return [node.width]
    return []


def _report_closure(program: ms.Program) -> set[str]:
    """Every parameter a named expression reads, by node or by name."""
    bodies = tuple(d.expression for d in program.expressions.values())
    names = set(parameters_of(*bodies))
    for node in walk(*bodies):
        names.update(amounts_of(node))
        if isinstance(node, ms.Cases):
            for region in node.regions:
                names |= region.when.names_read
    return names & set(program.parameters)


# ---------------------------------------------------------------------------
# sources and keys
# ---------------------------------------------------------------------------


def _dataset_sources(ds: xr.Dataset) -> dict[str, Any]:
    sources: dict[str, Any] = {str(d): index for d, index in ds.indexes.items()}
    sources.update({str(n): ds[n] for n in ds.data_vars})
    return sources


def _attachable(program: ms.Program) -> dict[str, str]:
    kinds = {n: "parameter" for n in program.parameters}
    kinds.update({d: "dimension" for d in program.dimensions})
    kinds.update({name: "relation" for name in program.relations})
    return kinds


class Sources:
    """
    The caller's data, read by key.

    ``keys()`` is called once, to see which declared names have data;
    everything after that goes through ``__getitem__``, so a source is
    read when a declaration reaches it and never otherwise. A key naming
    nothing the spec declares is ignored, so one mapping can feed several
    specs; one close to a declared name is a typo and is warned about.
    """

    def __init__(self, mapping: Mapping[str, Any], declared: Mapping[str, str]) -> None:
        self._mapping = mapping
        try:
            self._keys = frozenset(mapping.keys())
        except AttributeError:
            raise TypeError(
                f"sources must offer keys(): it is called once, to see which declared names "
                f"have data, and everything after that is read by key. "
                f"{type(mapping).__name__} offers no keys()."
            ) from None
        _warn_near_misses(self._keys, declared)

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def __getitem__(self, key: str) -> Any:
        return self._mapping[key]


def _warn_near_misses(keys: frozenset[str], declared: Mapping[str, str]) -> None:
    unknown = sorted(keys - set(declared))
    typos = {k: near for k in unknown if (near := _near(k, declared)) is not None}
    if not typos:
        return
    warnings.warn(
        f"source key(s) {shown(sorted(typos))} name neither a parameter, a dimension nor "
        f"a relation this spec declares, and read like a typo: "
        f"{shown([f'{k} -> {near}' for k, near in sorted(typos.items())])}. "
        f"An unknown key is ignored, so a mistyped one leaves its declaration without data.",
        UserWarning,
        stacklevel=4,
    )


def _near(key: str, declared: Mapping[str, str]) -> str | None:
    """The one declared name *key* is close enough to be a typo of."""
    near = difflib.get_close_matches(key, sorted(declared), n=1, cutoff=0.6)
    return near[0] if near else None


# ---------------------------------------------------------------------------
# dimensions
# ---------------------------------------------------------------------------


def _reached(program: ms.Program) -> set[str]:
    dims: set[str] = set()
    for declared in (program.parameters, program.variables, program.constraints):
        dims.update(d for decl in declared.values() for d in decl.dims)
    for rel in program.relations.values():
        dims.update(rel.dims)
    return dims


def _master_coords(program: ms.Program, sources: Sources) -> dict[str, pd.Index]:
    reached = _reached(program)
    coords: dict[str, pd.Index] = {}
    for dim in program.dimensions:
        if dim in sources:
            coords[dim] = _index(dim, sources[dim], program.dimensions[dim])
        elif dim in reached:
            raise SpecDataError(
                f"dimension '{dim}' has no index: pass its labels under key '{dim}' as "
                f"{_DIMENSION_SHAPES}. The index is what says which labels exist, and without "
                f"one a mistyped label is indistinguishable from a new one."
            )
    return coords


def _index(dim: str, obj: Any, declared: ms.Dimension) -> pd.Index:
    if isinstance(obj, (pd.Series, xr.DataArray, np.ndarray)):
        if obj.ndim != 1:
            raise SpecDataError(
                f"index for dimension '{dim}' is {obj.ndim}-dimensional; pass {_DIMENSION_SHAPES}."
            )
        values: Any = np.asarray(obj)
    elif isinstance(obj, (pd.Index, list, tuple)):
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
    index = flat.rename(dim)
    _check_value_dtype(dim, declared.dtype, index.dtype, what="dimension")
    if index.has_duplicates:
        twice = index[index.duplicated()].unique().tolist()
        raise SpecDataError(
            f"dimension '{dim}' lists {shown(twice)} more than once. A dimension's members are a set: "
            f"each label appears once, in the order the source gives it."
        )
    return index


# ---------------------------------------------------------------------------
# relations
# ---------------------------------------------------------------------------


def _key_value(name: str, rel: ms.Relation) -> tuple[str, str]:
    """The key dimension and the value dimension of a relation with one key column mapping into one value column."""
    if len(rel.key) != 1 or len(rel.values) != 1:
        raise SpecDataError(
            f"relation '{name}' keys {list(rel.key)} into {list(rel.values)}, and linopy "
            f"reads a relation over two columns: one key column mapping into one value "
            f"column. A bare relation, several key columns or several value columns are "
            f"not supported yet; split it into single-valued relations."
        )
    return rel.dim(rel.key[0]), rel.dim(rel.values[0])


def _relations(
    program: ms.Program, sources: Sources, coords: Mapping[str, pd.Index]
) -> dict[str, xr.DataArray]:
    out: dict[str, xr.DataArray] = {}
    for name, rel in program.relations.items():
        over, target = _key_value(name, rel)
        if name not in sources:
            raise SpecDataError(
                f"no data provided for relation '{name}'. Pass it under key '{name}' as "
                f"{_RELATION_SHAPES.format(key=over)}, holding a '{target}' value for each "
                f"'{over}' label it maps and nothing for a label it does not."
            )
        series = _relation_series(name, over, sources[name])
        _check_relation(series, name, over, target, coords)
        padded = series.reindex(coords[over])
        unmapped = padded.isna()
        if rel.missing == "refused" and unmapped.any():
            raise SpecDataError(
                f"relation '{name}' has no row for {int(unmapped.sum())} '{over}' label(s): "
                f"{shown(padded.index[unmapped].tolist())}. Every '{over}' label needs a row, "
                f"or the relation declares missing: absent."
            )
        out[name] = _numpy(xr.DataArray(padded, name=name))
    return out


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


def _relation_series(name: str, over: str, obj: Any) -> pd.Series:
    if isinstance(obj, xr.DataArray):
        if obj.dims != (over,) or over not in obj.indexes:
            raise SpecDataError(
                f"relation '{name}' arrived as a DataArray over {list(obj.dims)}, and it is a map "
                f"out of '{over}': pass a 1-D DataArray with '{over}' as its labelled dimension."
            )
        return obj.to_series()
    if isinstance(obj, Mapping):
        return pd.Series(dict(obj)).rename_axis(over)
    if isinstance(obj, pd.Series):
        if obj.index.name not in (None, over):
            raise SpecDataError(
                f"relation '{name}' is a Series indexed by '{obj.index.name}', and it is a map out of "
                f"'{over}': index it by '{over}' labels."
            )
        return obj.rename_axis(over)
    raise SpecDataError(
        f"relation '{name}': cannot adapt {type(obj).__name__} to a map; pass {_RELATION_SHAPES.format(key=over)}."
    )


def _check_relation(
    series: pd.Series,
    name: str,
    over: str,
    target: str,
    coords: Mapping[str, pd.Index],
) -> None:
    holes = series.isna()
    if holes.any():
        at = coordinates_shown((over,), series.index[holes][:5])
        raise SpecDataError(
            f"relation '{name}' carries {int(holes.sum())} row(s) with a null in '{target}': {at}. A map is "
            f"partial by leaving a label out, not by mapping it to nothing: drop the row and the "
            f"label is unmapped, which is what every operator reading the relation already means by it."
        )
    if series.index.has_duplicates:
        twice = series.index[series.index.duplicated()].unique().tolist()
        raise SpecDataError(
            f"relation '{name}' maps {len(twice)} '{over}' label(s) more than once: {shown(twice)}. "
            f"This relation is single-valued, so each label it maps takes exactly one row."
        )
    strays = series.index[~series.index.isin(coords[over])].tolist()
    if strays:
        raise SpecDataError(
            f"relation '{name}' maps {shown(strays)}, which are not labels of '{over}'. "
            f"'{over}' takes its labels from sources['{over}'], and they are "
            f"{shown(coords[over].tolist(), 8)}. A map maps the labels that exist: a key matching "
            f"none of them would place its terms nowhere, so it is a typo on one side or a label "
            f"missing from the other."
        )
    values = pd.Index(series.to_numpy())
    foreign = values[~values.isin(coords[target])].unique().tolist()
    if foreign:
        raise SpecDataError(
            f"relation '{name}' has value(s) that are not '{target}' labels: "
            f"{shown(foreign)}. Every value must be a declared '{target}' label, otherwise "
            f"a sum through {name} drops those terms in the join that places them, and the "
            f"model builds and solves without them."
        )


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


def _fill(declared: ms.Parameter) -> Any:
    return False if declared.dtype == "bool" else np.nan


def _as_array(
    name: str,
    declared: ms.Parameter,
    obj: Any,
    coords: Mapping[str, pd.Index],
) -> Parameter:
    if isinstance(obj, xr.DataArray):
        return _from_dense(name, declared, obj)
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
        f"pass {_parameter_shapes(dims)}."
    )


def _parameter_shapes(dims: Sequence[str]) -> str:
    return _PARAMETER_SHAPES.format(dims=list(dims), columns=[*dims, "value"])


def _everywhere(arr: xr.DataArray) -> Parameter:
    """*arr* beside the presence of every one of its rows."""
    return arr, xr.ones_like(arr, dtype=bool)


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
    return _everywhere(arr)


def _from_dense(name: str, declared: ms.Parameter, arr: xr.DataArray) -> Parameter:
    dims = declared.dims
    _check_value_dtype(name, declared.dtype, arr.dtype)
    if set(arr.dims) != set(dims) or len(arr.dims) != len(dims):
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
    holes = arr.isnull()
    if bool(holes.any()):
        raise SpecDataError(
            f"parameter '{name}' holds {int(holes.sum())} null or NaN value(s), the first at "
            f"{first_coordinates(holes, 3)}. A coordinate with no value has no row: leave it "
            f"out of the table, and the declaration's missing: says what that means."
        )
    return _everywhere(arr.transpose(*dims))


def _from_frame(
    name: str,
    declared: ms.Parameter,
    df: pd.DataFrame,
    coords: Mapping[str, pd.Index],
) -> Parameter:
    dims = declared.dims
    tidy = df
    if not set(dims) <= set(df.columns) and set(dims) <= _headers(df):
        tidy = df.reset_index()
    if "value" in tidy.columns and set(dims) <= set(tidy.columns):
        indexed = tidy.set_index(list(dims)) if dims else tidy
        return _from_rows(name, declared, indexed["value"], coords)
    if len(dims) == 2:
        return _from_dense(name, declared, xr.DataArray(_wide(name, dims, df)))
    raise SpecDataError(
        f"parameter '{name}' arrived as a DataFrame with columns {list(df.columns)}; a table for "
        f"'{name}' carries columns {[*dims, 'value']}."
    )


def _headers(df: pd.DataFrame) -> set[Any]:
    return set(df.columns) | set(df.index.names)


def _wide(name: str, dims: tuple[str, ...], df: pd.DataFrame) -> pd.DataFrame:
    names = (df.index.name, df.columns.name)
    if names == (None, None):
        if len(df.index) == len(df.columns):
            raise SpecDataError(
                f"parameter '{name}' arrived as a {len(df.index)}x{len(df.columns)} wide DataFrame "
                f"with neither axis named, and '{name}' is over {list(dims)}. A square frame does "
                f"not say which axis is which. Name the index and columns after the two dims, or "
                f"pass a table with columns {[*dims, 'value']}."
            )
        return df.rename_axis(index=dims[0], columns=dims[1])
    if set(names) == set(dims):
        return df
    raise SpecDataError(
        f"parameter '{name}' arrived as a wide DataFrame with index '{names[0]}' and columns "
        f"'{names[1]}', and '{name}' is over {list(dims)}. Name the index and columns after the "
        f"two dims, or pass a table with columns {[*dims, 'value']}."
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
    onto = [coords[d] for d in dims]
    full = onto[0] if len(dims) == 1 else pd.MultiIndex.from_product(onto, names=dims)
    shape = tuple(len(index) for index in onto)
    values = series.reindex(full, fill_value=_fill(declared)).to_numpy()
    rows = pd.Series(True, index=series.index).reindex(full, fill_value=False)
    on = dict(zip(dims, onto))
    arr = xr.DataArray(values.reshape(shape), dims=dims, coords=on, name=name)
    present = xr.DataArray(rows.to_numpy().reshape(shape), dims=dims, coords=on)
    return arr, present


def _with_dims(name: str, dims: tuple[str, ...], series: pd.Series) -> pd.Series:
    index = series.index
    if index.nlevels != len(dims):
        raise SpecDataError(
            f"parameter '{name}': a Series or dict carries one label per level, and its index has "
            f"{index.nlevels} level(s) where '{name}' is over {list(dims)}. "
            f"Pass {_parameter_shapes(dims)}."
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
    strangers = labels[~labels.isin(known)].unique().tolist()
    if not strangers:
        return
    raise SpecDataError(
        f"parameter '{name}' has label(s) in dimension '{dim}' that are not coordinates of it: "
        f"{shown(strangers)}.\n  {dim} has: {shown(known.tolist(), 10)}\n"
        f"A label that is not a coordinate is a typo: its row joins nothing, so the coordinate it "
        f"was meant for is left uncovered. Fix the label, or add it to sources['{dim}']."
    )


def _aligned(
    name: str,
    arr: xr.DataArray,
    present: xr.DataArray,
    onto: Mapping[str, pd.Index],
    fill: Any,
) -> Parameter:
    if all(arr.indexes[d].equals(index) for d, index in onto.items()):
        stale = {d: i for d, i in onto.items() if arr.indexes[d].dtype != i.dtype}
        if stale:
            return arr.assign_coords(stale), present.assign_coords(stale)
        return arr, present
    for d, index in onto.items():
        _refuse_strangers(name, d, arr.indexes[d], index)
    return arr.reindex(onto, fill_value=fill), present.reindex(onto, fill_value=False)


def _filled(
    name: str,
    declared: ms.Parameter,
    arr: xr.DataArray,
    present: xr.DataArray,
    strict: bool,
) -> xr.DataArray:
    """*arr* with its missing rows reading the way the declaration's ``missing:`` says."""
    missing = declared.missing
    if bool(present.all()):
        return arr
    if missing == "refused":
        raise SpecDataError(
            f"parameter '{name}' has no row at {int((~present).sum())} coordinate(s), the "
            f"first at {first_coordinates(~present, 3)}. Every coordinate of {list(declared.dims)} "
            f"needs a row, or the declaration says what a missing row means with missing:."
        )
    if missing == "absent":
        return arr
    if missing == "neutral":
        if strict:
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
    arrived = _KIND_NAMES.get(str(dtype.kind), str(dtype))
    raise SpecDataError(
        f"{what} '{name}' is declared '{declared}' and its values arrived as '{arrived}'. "
        f"A declared dtype is a claim about the values, and it is checked here: the file says what "
        f"the values are, or the values are not attached.\n"
        f"  Cast the values to {declared}, if the declaration is what you meant\n"
        f"  Or declare what the data has: {{dtype: {arrived}}}"
    )
