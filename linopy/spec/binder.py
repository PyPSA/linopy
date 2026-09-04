"""
Bind user data to a math-spec program.

The language fixes three binding rules and this module enforces them: a
dimension's members come only from the source keyed by the dimension's
name, their order is the source's order and is never sorted, and a
parameter or lookup source is read for values, never for labels. Parameters
are resolved from ``sources`` on demand and aligned onto the master
coordinates without copying an already aligned array. A coordinate a table
leaves out becomes NaN (``False`` for a ``bool`` parameter); what that means
is the builder's question, not this module's.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
import xarray as xr
from math_spec import did_you_mean
from math_spec import program as ms

from linopy.spec.errors import SpecDataError

Retain = Literal["report", "all", "none"]
_RETAIN: tuple[str, ...] = get_args(Retain)

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
_SCALARS = (bool, int, float, str, np.number, np.bool_)
_DIMENSION_SHAPES = "a pandas Index, a list, a tuple, a 1-D numpy array, a pandas Series or a 1-D DataArray"
_LOOKUP_SHAPES = "a pandas Series indexed by '{over}', a dict keyed by '{over}' labels, or a 1-D DataArray over '{over}'"
_PARAMETER_SHAPES = (
    "a DataArray over {dims}, a pandas Series whose (Multi)Index levels are {dims}, "
    "a DataFrame with columns {columns} or in wide form, a dict keyed by label, or one number"
)


def bind(
    program: ms.Program,
    sources: Mapping[str, Any] | xr.Dataset,
    *,
    retain: Retain = "report",
) -> Bound:
    """
    Bind *sources* to *program*: master coordinates now, parameters on demand.

    Args:
        program: The lowered spec.
        sources: Data keyed by declared name. Any mapping works; it is read by
            key and never iterated beyond ``sources.keys()``. An ``xr.Dataset``
            is accepted too: its indexes are dimension sources, its data
            variables parameters and lookups.
        retain: Which parameters :meth:`Bound.retained` persists.

    Raises:
        SpecDataError: A ``retain`` outside its three values, a key naming
            nothing the spec declares, a reached dimension or a lookup with
            no source, a duplicated dimension member, or a lookup breaking
            the rules a map has.
    """
    if retain not in _RETAIN:
        raise SpecDataError(
            f"retain={retain!r} is not one of {_shown(_RETAIN)}. {did_you_mean(retain, _RETAIN)}"
        )
    if isinstance(sources, xr.Dataset):
        sources = _dataset_sources(sources)
    keys = frozenset(sources.keys())
    _check_keys(program, keys)
    coords = _master_coords(program, sources, keys)
    lookups = _lookups(program, sources, keys, coords)
    return Bound(program, coords, lookups, retain, sources, keys)


@dataclass(frozen=True, eq=False)
class Bound:
    """
    A program bound to its data.

    Attributes:
        program: The lowered spec the data is bound to.
        coords: Master coordinates by dimension, in source order, each index
            named after its dimension. A declared dimension nothing reaches
            and nothing supplies is absent.
        lookups: By dimension, by lookup name, the map as an array over the
            dimension's master coordinates, NaN where a label is unmapped.
        retain: Which parameters :meth:`retained` persists.
        sources: The caller's data, read by key on demand.
    """

    program: ms.Program
    coords: Mapping[str, pd.Index]
    lookups: Mapping[str, Mapping[str, xr.DataArray]]
    retain: Retain
    sources: Mapping[str, Any]
    _keys: frozenset[str] = field(repr=False)

    def parameter(self, name: str) -> xr.DataArray:
        """
        The parameter *name* resolved from ``sources`` and aligned to ``coords``.

        Resolved on every call and never cached. An already aligned array is
        returned without a copy; a mismatching one is reindexed onto the
        master coordinates, leaving NaN (``False`` for ``bool``) where no row
        was supplied.

        Raises:
            SpecDataError: No data, a shape no reader accepts, a rank other
                than declared, a label its dimension lacks, two rows for one
                coordinate, a null value in a row, or values of another type
                than declared.
        """
        declared = self._declaration(name)
        if name not in self._keys:
            raise SpecDataError(f"no data provided for parameter '{name}'")
        arr = _as_array(name, declared, self.sources[name], self.coords)
        onto = {d: self.coords[d] for d in declared.dims}
        return _aligned(name, arr, onto, _fill(declared))

    def retained(self) -> xr.Dataset:
        """The lookups plus the parameters ``retain`` keeps, as one dataset."""
        arrays = {n: self.parameter(n) for n in self._retained_names()}
        for by_name in self.lookups.values():
            arrays.update(by_name)
        return xr.Dataset(arrays)

    def _declaration(self, name: str) -> ms.ParameterDeclaration:
        if name not in self.program.parameters:
            raise SpecDataError(
                f"unknown parameter '{name}'. {did_you_mean(name, self.program.parameters)}"
            )
        declared = self.program.parameters[name]
        if declared.derivation is not None:
            raise SpecDataError(
                f"parameter '{name}' is emitted by piecewise block '{declared.derivation.block}' "
                f"and is filled from the block's own breakpoints, not bound from sources."
            )
        return declared

    def _retained_names(self) -> list[str]:
        if self.retain == "none":
            return []
        parameters = self.program.parameters
        keep = (
            set(parameters) if self.retain == "all" else _report_closure(self.program)
        )
        return [n for n, p in parameters.items() if p.derivation is None and n in keep]


def _report_closure(program: ms.Program) -> set[str]:
    """Every parameter a named expression reads, by node or by name."""
    bodies = tuple(program.named_expressions.values())
    names = set(ms.parameters_of(*bodies))
    for node in ms.walk(*bodies):
        if isinstance(node, ms.Translate) and isinstance(node.offset, str):
            names.add(node.offset)
        elif isinstance(node, ms.Window) and isinstance(node.width, str):
            names.add(node.width)
        elif isinstance(node, ms.Cases):
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
    kinds = {
        n: "parameter" for n, p in program.parameters.items() if p.derivation is None
    }
    kinds.update({d: "dimension" for d in program.dimensions})
    kinds.update({lk.name: "lookup" for _, lk in program.lookups})
    return kinds


def _check_keys(program: ms.Program, keys: frozenset[str]) -> None:
    known = _attachable(program)
    unknown = sorted(keys - set(known))
    if not unknown:
        return
    one = len(unknown) == 1
    lead = f"source key {unknown[0]!r} names" if one else f"source keys {unknown} name"
    raise SpecDataError(
        f"{lead} neither a parameter, a dimension nor a lookup this spec declares. "
        f"{did_you_mean(unknown[0], known)} Pass only what the spec takes."
    )


# ---------------------------------------------------------------------------
# dimensions
# ---------------------------------------------------------------------------


def _reached(program: ms.Program) -> set[str]:
    dims: set[str] = set()
    for declared in (program.parameters, program.variables, program.constraints):
        dims.update(d for decl in declared.values() for d in decl.dims)
    dims.update(pw.over for pw in program.piecewise.values())
    for over, lk in program.lookups:
        dims.add(over)
        if lk.target is not None:
            dims.add(lk.target)
    return dims


def _master_coords(
    program: ms.Program, sources: Mapping[str, Any], keys: frozenset[str]
) -> dict[str, pd.Index]:
    reached = _reached(program)
    coords: dict[str, pd.Index] = {}
    for dim in program.dimensions:
        if dim in keys:
            coords[dim] = _index(dim, sources[dim])
        elif dim in reached:
            raise SpecDataError(
                f"dimension '{dim}' has no index: pass its labels under key '{dim}' as "
                f"{_DIMENSION_SHAPES}. The index is what says which labels exist, and without "
                f"one a mistyped label is indistinguishable from a new one."
            )
    return coords


def _index(dim: str, obj: Any) -> pd.Index:
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
    index = pd.Index(values, name=dim)
    if index.has_duplicates:
        twice = index[index.duplicated()].unique().tolist()
        raise SpecDataError(
            f"dimension '{dim}' lists {_shown(twice)} more than once. A dimension's members are a set: "
            f"each label appears once, in the order the source gives it."
        )
    return index


# ---------------------------------------------------------------------------
# lookups
# ---------------------------------------------------------------------------


def _lookups(
    program: ms.Program,
    sources: Mapping[str, Any],
    keys: frozenset[str],
    coords: Mapping[str, pd.Index],
) -> dict[str, dict[str, xr.DataArray]]:
    out: dict[str, dict[str, xr.DataArray]] = {}
    for over, lk in program.lookups:
        space = lk.target or lk.name
        if lk.name not in keys:
            raise SpecDataError(
                f"no data provided for lookup '{lk.name}'. Pass it under key '{lk.name}' as "
                f"{_LOOKUP_SHAPES.format(over=over)}, holding a '{space}' value for each "
                f"'{over}' label it maps and nothing for a label it does not."
            )
        series = _lookup_series(lk.name, over, sources[lk.name])
        _check_lookup(series, lk, over, coords)
        padded = series.reindex(coords[over])
        out.setdefault(over, {})[lk.name] = xr.DataArray(padded, name=lk.name)
    return out


def _lookup_series(name: str, over: str, obj: Any) -> pd.Series:
    if isinstance(obj, xr.DataArray):
        if obj.dims != (over,) or over not in obj.indexes:
            raise SpecDataError(
                f"lookup '{name}' arrived as a DataArray over {list(obj.dims)}, and it is a map "
                f"out of '{over}': pass a 1-D DataArray with '{over}' as its labelled dimension."
            )
        return obj.to_series()
    if isinstance(obj, Mapping):
        return pd.Series(dict(obj)).rename_axis(over)
    if isinstance(obj, pd.Series):
        if obj.index.name not in (None, over):
            raise SpecDataError(
                f"lookup '{name}' is a Series indexed by '{obj.index.name}', and it is a map out of "
                f"'{over}': index it by '{over}' labels."
            )
        return obj.rename_axis(over)
    raise SpecDataError(
        f"lookup '{name}': cannot adapt {type(obj).__name__} to a map; pass {_LOOKUP_SHAPES.format(over=over)}."
    )


def _check_lookup(
    series: pd.Series,
    lk: ms.LookupDeclaration,
    over: str,
    coords: Mapping[str, pd.Index],
) -> None:
    space = lk.target or lk.name
    holes = series.isna()
    if holes.any():
        at = _coordinates_shown((over,), series.index[holes][:5])
        raise SpecDataError(
            f"lookup '{lk.name}' carries {int(holes.sum())} row(s) with a null in '{space}': {at}. A map is "
            f"partial by leaving a label out, not by mapping it to nothing: drop the row and the "
            f"label is unmapped, which is what every operator reading the lookup already means by it."
        )
    if series.index.has_duplicates:
        twice = series.index[series.index.duplicated()].unique().tolist()
        raise SpecDataError(
            f"lookup '{lk.name}' maps {len(twice)} '{over}' label(s) more than once: {_shown(twice)}. "
            f"A lookup is single-valued, so each label it maps takes exactly one row."
        )
    strays = series.index[~series.index.isin(coords[over])].tolist()
    if strays:
        raise SpecDataError(
            f"lookup '{lk.name}' maps {_shown(strays)}, which are not labels of '{over}'. "
            f"'{over}' takes its labels from sources['{over}'], and they are "
            f"{_shown(coords[over].tolist(), 8)}. A map maps the labels that exist: a key matching "
            f"none of them would place its terms nowhere, so it is a typo on one side or a label "
            f"missing from the other."
        )
    if lk.dtype is not None:
        _check_value_dtype(lk.name, lk.dtype, series.dtype, kind="lookup")
    if lk.target is None:
        return
    values = pd.Index(series.to_numpy())
    foreign = values[~values.isin(coords[lk.target])].unique().tolist()
    if foreign:
        raise SpecDataError(
            f"dimension '{over}' lookup '{lk.name}' has value(s) that are not '{lk.target}' labels: "
            f"{_shown(foreign)}. Every value must be a declared '{lk.target}' label, otherwise "
            f"sum(by={lk.name}) drops those terms in the join that places them, and the model "
            f"builds and solves without them."
        )


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


def _fill(declared: ms.ParameterDeclaration) -> Any:
    return False if declared.dtype == "bool" else np.nan


def _as_array(
    name: str,
    declared: ms.ParameterDeclaration,
    obj: Any,
    coords: Mapping[str, pd.Index],
) -> xr.DataArray:
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


def _from_scalar(
    name: str,
    declared: ms.ParameterDeclaration,
    obj: Any,
    coords: Mapping[str, pd.Index],
) -> xr.DataArray:
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
    return arr


def _from_dense(
    name: str, declared: ms.ParameterDeclaration, arr: xr.DataArray
) -> xr.DataArray:
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
    return arr.transpose(*dims)


def _from_frame(
    name: str,
    declared: ms.ParameterDeclaration,
    df: pd.DataFrame,
    coords: Mapping[str, pd.Index],
) -> xr.DataArray:
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
    declared: ms.ParameterDeclaration,
    series: pd.Series,
    coords: Mapping[str, pd.Index],
) -> xr.DataArray:
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
            f"{_coordinates_shown(dims, series.index[holes][:3])}. In a table the absence of a "
            f"value is the absence of the row, and such a row says the coordinate exists and denies "
            f"it in the same breath. Drop those rows, or supply the values."
        )
    _check_value_dtype(name, declared.dtype, series.dtype)
    _refuse_duplicate_coordinates(name, dims, series.index)
    for d in dims:
        _refuse_strangers(name, d, series.index.get_level_values(d), coords[d])
    onto = [coords[d] for d in dims]
    full = onto[0] if len(dims) == 1 else pd.MultiIndex.from_product(onto, names=dims)
    values = series.reindex(full, fill_value=_fill(declared)).to_numpy()
    dense = values.reshape(tuple(len(index) for index in onto))
    return xr.DataArray(dense, dims=dims, coords=dict(zip(dims, onto)), name=name)


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
        f"{_coordinate(dims, key)} ({n + 1} rows)" for key, n in counts.iloc[:3].items()
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
        f"{_shown(strangers)}.\n  {dim} has: {_shown(known.tolist(), 10)}\n"
        f"A missing row is a zero coefficient, but a label that is not a coordinate is a typo: its "
        f"row joins nothing, so the coordinate it was meant for silently reads as absent. Fix the "
        f"label, or add it to sources['{dim}']."
    )


def _aligned(
    name: str, arr: xr.DataArray, onto: Mapping[str, pd.Index], fill: Any
) -> xr.DataArray:
    if all(arr.indexes[d].equals(index) for d, index in onto.items()):
        stale = {d: i for d, i in onto.items() if arr.indexes[d].dtype != i.dtype}
        return arr.assign_coords(stale) if stale else arr
    for d, index in onto.items():
        _refuse_strangers(name, d, arr.indexes[d], index)
    return arr.reindex(onto, fill_value=fill)


def _check_value_dtype(
    name: str, declared: str, dtype: Any, kind: str = "parameter"
) -> None:
    if str(dtype.kind) in _ACCEPTED_KINDS[declared]:
        return
    arrived = _KIND_NAMES.get(str(dtype.kind), str(dtype))
    raise SpecDataError(
        f"{kind} '{name}' is declared '{declared}' and its values arrived as '{arrived}'. "
        f"A declared dtype is a claim about the values, and it is checked here: the file says what "
        f"the values are, or the values are not attached.\n"
        f"  Cast the values to {declared}, if the declaration is what you meant\n"
        f"  Or declare what the data has: {{dtype: {arrived}}}"
    )


# ---------------------------------------------------------------------------
# wording
# ---------------------------------------------------------------------------


def _shown(labels: Sequence[Any], limit: int = 5) -> str:
    head = ", ".join(repr(x) for x in labels[:limit])
    return head + (f" (and {len(labels) - limit} more)" if len(labels) > limit else "")


def _coordinate(dims: Sequence[str], key: Hashable) -> str:
    row = key if isinstance(key, tuple) else (key,)
    return ", ".join(f"{d}={v!r}" for d, v in zip(dims, row))


def _coordinates_shown(dims: Sequence[str], rows: Iterable[Hashable]) -> str:
    return "; ".join(_coordinate(dims, row) for row in rows)
