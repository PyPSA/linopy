#!/usr/bin/env python3
"""
Linopy common module.

This module contains commonly used functions.
"""

from __future__ import annotations

import operator
import os
from collections.abc import Callable, Generator, Hashable, Iterable, Mapping, Sequence
from functools import cached_property, partial, reduce, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar, overload
from warnings import warn

import numpy as np
import pandas as pd
import polars as pl
from numpy import arange, nan, signedinteger
from polars.datatypes import DataTypeClass
from xarray import Coordinates, DataArray, Dataset, apply_ufunc, broadcast
from xarray import align as xr_align
from xarray.core import dtypes, indexing
from xarray.core.coordinates import CoordinateValidationError
from xarray.core.types import JoinOptions, T_Alignable
from xarray.namedarray.utils import is_dict_like

from linopy.config import options
from linopy.constants import (
    HELPER_DIMS,
    SIGNS,
    EvolvingAPIWarning,
    SIGNS_alternative,
    SIGNS_pretty,
    sign_replace_dict,
)
from linopy.types import (
    CONSTANT_TYPES,
    CoordsLike,
    DimsLike,
    SideLike,
)

if TYPE_CHECKING:
    from linopy.constraints import ConstraintBase
    from linopy.expressions import LinearExpression, QuadraticExpression
    from linopy.variables import Variable


def set_int_index(series: pd.Series) -> pd.Series:
    """
    Convert string index to int index.
    """

    if not series.empty and not pd.api.types.is_integer_dtype(series.index):
        cutoff = count_initial_letters(str(series.index[0]))
        try:
            series.index = series.index.str[cutoff:].astype(int)
        except ValueError:
            series.index = series.index.str.replace(".*#", "", regex=True).astype(int)
    return series


def maybe_replace_sign(sign: str) -> str:
    """
    Replace the sign with an alternative sign if available.

    Parameters
    ----------
        sign (str): The sign to be replaced.

    Returns
    -------
        str: The replaced sign.

    Raises
    ------
        ValueError: If the sign is not in the available signs.
    """
    if sign in SIGNS_alternative:
        return sign_replace_dict[sign]
    elif sign in SIGNS:
        return sign
    else:
        raise ValueError(f"Sign {sign} not in {SIGNS} or {SIGNS_alternative}")


def maybe_replace_signs(sign: DataArray) -> DataArray:
    """
    Replace signs with alternative signs if available.

    Parameters
    ----------
        sign (np.ndarray): The signs to be replaced.

    Returns
    -------
        np.ndarray: The replaced signs.
    """
    func = np.vectorize(maybe_replace_sign)
    return apply_ufunc(func, sign, dask="parallelized", output_dtypes=[sign.dtype])


def format_string_as_variable_name(name: Hashable) -> str:
    """
    Format a string to a valid python variable name.

    Parameters
    ----------
        name (str): The name to be converted.

    Returns
    -------
        str: The formatted name.
    """
    return str(name).replace(" ", "_").replace("-", "_")


def get_from_iterable(lst: DimsLike | None, index: int) -> Any | None:
    """
    Returns the element at the specified index of the list, or None if the index
    is out of bounds.
    """
    if lst is None:
        return None
    if isinstance(lst, Sequence | Iterable):
        lst = list(lst)
    else:
        lst = [lst]
    return lst[index] if 0 <= index < len(lst) else None


def pandas_to_dataarray(
    arr: pd.DataFrame | pd.Series,
    coords: CoordsLike | None = None,
    dims: DimsLike | None = None,
    **kwargs: Any,
) -> DataArray:
    """
    Convert a pandas DataFrame or Series to a DataArray.

    As pandas objects already have a concept of coordinates, the
    coordinates (index, columns) will be used as coordinates for the DataArray.
    Solely the dimension names can be specified.

    Parameters
    ----------
        arr (Union[pd.DataFrame, pd.Series]):
            The input pandas DataFrame or Series.
        coords (Union[dict, list, None]):
            The coordinates for the DataArray. If None, default coordinates will be used.
        dims (Union[list, None]):
            The dimensions for the DataArray. If None, the column names of the DataFrame or the index names of the Series will be used.
        **kwargs:
            Additional keyword arguments to be passed to the DataArray constructor.

    Returns
    -------
        DataArray:
            The converted DataArray.
    """
    dims = [
        axis.name or get_from_iterable(dims, i) or f"dim_{i}"
        for i, axis in enumerate(arr.axes)
    ]
    return DataArray(arr, coords=None, dims=dims, **kwargs)


def numpy_to_dataarray(
    arr: np.ndarray,
    coords: CoordsLike | None = None,
    dims: DimsLike | None = None,
    **kwargs: Any,
) -> DataArray:
    """
    Convert a numpy array to a DataArray.

    Parameters
    ----------
        arr (np.ndarray):
            The input numpy array.
        coords (Union[dict, list, None]):
            The coordinates for the DataArray. If None, default coordinates will be used.
        dims (Union[list, None]):
            The dimensions for the DataArray. If None, the dimensions will be automatically generated.
        **kwargs:
            Additional keyword arguments to be passed to the DataArray constructor.

    Returns
    -------
        DataArray:
            The converted DataArray.
    """
    # fallback case for zero dim arrays
    if arr.ndim == 0:
        if dims is None and is_dict_like(coords):
            dims = list(coords.keys())
        return DataArray(arr.item(), coords=coords, dims=dims, **kwargs)

    if isinstance(dims, Iterable | Sequence):
        dims = list(dims)
    elif dims is not None:
        dims = [dims]

    if dims is not None and len(dims):
        dims = [get_from_iterable(dims, i) or f"dim_{i}" for i in range(arr.ndim)]

    if dims is not None and len(dims) and coords is not None:
        if isinstance(coords, list):
            coords = dict(zip(dims, coords[: arr.ndim]))
        elif is_dict_like(coords):
            coords = {k: v for k, v in coords.items() if k in dims}

    return DataArray(arr, coords=coords, dims=dims, **kwargs)


def as_dataarray(
    arr: Any,
    coords: CoordsLike | None = None,
    dims: DimsLike | None = None,
    **kwargs: Any,
) -> DataArray:
    """
    Convert ``arr`` to a DataArray.

    Picks the right constructor for each supported input type (pandas,
    polars, numpy, scalar, DataArray) and labels positional axes with
    ``dims`` / ``coords``. The result is not reshaped against ``coords``:
    dims are neither expanded, reordered, nor projected onto MultiIndex
    dims. Use :func:`broadcast_to_coords` or :func:`align_to_coords` when
    ``coords`` should govern the result's shape.
    """
    if isinstance(arr, pd.Series | pd.DataFrame):
        arr = pandas_to_dataarray(arr, coords=coords, dims=dims, **kwargs)
    elif isinstance(arr, np.ndarray):
        arr = numpy_to_dataarray(arr, coords=coords, dims=dims, **kwargs)
    elif isinstance(arr, pl.Series):
        arr = numpy_to_dataarray(arr.to_numpy(), coords=coords, dims=dims, **kwargs)
    elif isinstance(arr, np.number | int | float | str | bool | list):
        if isinstance(arr, np.number):
            arr = float(arr)
        if dims is None:
            if isinstance(coords, Coordinates):
                dims = coords.dims
            elif is_dict_like(coords) and np.ndim(arr) == 0:
                dims = list(coords.keys())
        arr = DataArray(arr, coords=coords, dims=dims, **kwargs)

    elif not isinstance(arr, DataArray):
        supported_types = [
            np.number,
            str,
            bool,
            list,
            pd.Series,
            pd.DataFrame,
            np.ndarray,
            DataArray,
            pl.Series,
        ]
        supported_types_str = ", ".join([t.__name__ for t in supported_types])
        raise TypeError(
            f"Unsupported type of arr: {type(arr)}. Supported types are: {supported_types_str}"
        )

    arr = fill_missing_coords(arr)
    return arr


def _as_index(coord_values: Any) -> pd.Index:
    return (
        coord_values if isinstance(coord_values, pd.Index) else pd.Index(coord_values)
    )


def _as_multiindex(coord_values: Any) -> pd.MultiIndex | None:
    """Return the backing ``pd.MultiIndex`` of a coords entry, or ``None``."""
    if isinstance(coord_values, pd.MultiIndex):
        return coord_values
    if isinstance(coord_values, DataArray):
        idx = coord_values.to_index()
        if isinstance(idx, pd.MultiIndex):
            return idx
    return None


class _LevelProjection(NamedTuple):
    """Record of one MultiIndex-level projection performed by ``_broadcast_to_coords``."""

    dim: Hashable
    levels: list[Hashable]
    is_partial: bool  # input carried only a subset of the MI's levels
    has_gap: bool  # projection left entries of the MI dim uncovered (NaN)


def _project_onto_multiindex_levels(
    arr: DataArray,
    expected: dict[Hashable, Any],
) -> tuple[DataArray, list[_LevelProjection]]:
    """
    Map ``arr`` dims that name levels of a stacked-MultiIndex coords dim onto it.

    For every entry of the MultiIndex dim, select the ``arr`` value at that
    entry's level values. A subset of levels broadcasts across the remaining
    ones; the full set aligns element-wise. ``arr`` is returned unchanged
    when it carries no level dims.

    Raises ``ValueError`` only on structural errors: a level name owned by
    two MI dims, or a level value missing from ``arr``. Partial projections
    and coverage gaps are recorded in the returned ``_LevelProjection`` list;
    the caller decides how to treat them.
    """
    level_owner: dict[Hashable, Hashable] = {}
    owner_mi: dict[Hashable, pd.MultiIndex] = {}
    for dim, coord_values in expected.items():
        mi = _as_multiindex(coord_values)
        if mi is None:
            continue
        owner_mi[dim] = mi
        for level in mi.names:
            if level is None:
                continue
            if level in level_owner:
                raise ValueError(
                    f"Level {level!r} is shared by MultiIndex dimensions "
                    f"{level_owner[level]!r} and {dim!r}; cannot resolve which "
                    f"to align to."
                )
            level_owner[level] = dim

    groups: dict[Hashable, list[Hashable]] = {}
    for d in arr.dims:
        if d in expected:
            continue
        owner = level_owner.get(d)
        if owner is not None:
            groups.setdefault(owner, []).append(d)

    projections: list[_LevelProjection] = []
    for dim, levels in groups.items():
        mi = owner_mi[dim]
        selectors = {
            level: DataArray(np.asarray(mi.get_level_values(level)), dims=[dim])
            for level in levels
        }
        try:
            arr = arr.sel(selectors)
        except KeyError as err:
            raise ValueError(
                f"Cannot align level(s) {levels} onto MultiIndex dimension "
                f"{dim!r}: value {err} is missing."
            ) from err
        arr = arr.assign_coords(Coordinates.from_pandas_multiindex(mi, dim))
        projections.append(
            _LevelProjection(
                dim=dim,
                levels=levels,
                is_partial=len(levels) < sum(name is not None for name in mi.names),
                has_gap=bool(arr.isnull().any()),
            )
        )

    return arr, projections


def _broadcast_to_coords(
    arr: Any,
    coords: CoordsLike | None = None,
    dims: DimsLike | None = None,
    **kwargs: Any,
) -> tuple[DataArray, list[_LevelProjection]]:
    """
    Convert ``arr`` and broadcast it against ``coords`` (shared mechanics).

    Returns the broadcast DataArray together with the MultiIndex-level
    projections performed along the way, so the public entry points can
    apply their own policy (warn or raise) to partial projections and
    coverage gaps.
    """
    if coords is None:
        return as_dataarray(arr, coords, dims, **kwargs), []

    if isinstance(coords, list | tuple) and any(isinstance(c, tuple) for c in coords):
        # xarray reads bare `(a, b)` as `(dim_name, values)`; normalize so a
        # coords entry passed as a tuple behaves identically to a list.
        coords = [list(c) if isinstance(c, tuple) else c for c in coords]

    expected = _coords_to_dict(coords, dims=dims)
    if not expected:
        return as_dataarray(arr, coords, dims, **kwargs), []

    if isinstance(arr, pd.Series | pd.DataFrame):
        converted = _named_pandas_to_dataarray(arr)
        if converted is not None:
            arr = converted

    if not isinstance(arr, DataArray):
        # numpy/polars/unnamed-pandas inputs are positional — their only
        # meaningful information is the values; any axis labels are
        # auto-generated. Default dims to coords' keys so the conversion
        # labels axes correctly (instead of dim_0/dim_1), then re-assign
        # coords from expected so positional inputs align to coords by
        # position. A shape mismatch surfaces here as a clear xarray
        # "conflicting sizes" error rather than a confusing
        # "coordinates do not match" further down.
        if dims is None:
            dims = list(expected)
        arr = as_dataarray(arr, coords, dims=dims, **kwargs)
        # Skip MultiIndex dims — re-assigning a PandasMultiIndex coord emits
        # a FutureWarning and isn't needed (the conversion already used it).
        arr = arr.assign_coords(
            {
                d: expected[d]
                for d in arr.dims
                if d in expected and not isinstance(arr.indexes.get(d), pd.MultiIndex)
            }
        )

    arr, projections = _project_onto_multiindex_levels(arr, expected)

    for dim, coord_values in expected.items():
        if dim not in arr.dims:
            continue
        if isinstance(arr.indexes.get(dim), pd.MultiIndex):
            continue
        expected_idx = _as_index(coord_values)
        actual_idx = arr.coords[dim].to_index()
        if actual_idx.equals(expected_idx):
            continue
        # Same values, different order → reindex to match expected order.
        # Different value sets are left alone for downstream xarray alignment.
        if len(actual_idx) == len(expected_idx) and set(actual_idx) == set(
            expected_idx
        ):
            arr = arr.reindex({dim: expected_idx})

    # expand_dims prepends new dimensions and their coordinate variables;
    # the subsequent transpose restores coords order. Both are no-ops when
    # the array already matches. Reconstruct so the DataArray's coords
    # iteration order also follows coords (a Dataset built from this picks
    # up its dim order from coord insertion).
    expand = {k: v for k, v in expected.items() if k not in arr.dims}
    if expand:
        # expand_dims drops the level coords of a MultiIndex-backed dim,
        # leaving a degenerate flat index that fails to align downstream.
        # Broadcast against a proper Coordinates template instead.
        plain = {}
        for dim, coord_values in expand.items():
            mi = _as_multiindex(coord_values)
            # Fall back to expand_dims when arr already carries one of the
            # MultiIndex's level names as its own coord: broadcasting against
            # the level coords would raise on the conflicting index.
            if mi is None or set(mi.names) & (set(arr.coords) | set(arr.dims)):
                plain[dim] = coord_values
                continue
            template = DataArray(
                np.zeros(len(mi)),
                coords=Coordinates.from_pandas_multiindex(mi, dim),
                dims=[dim],
            )
            arr, _ = broadcast(arr, template)
        if plain:
            arr = arr.expand_dims(plain)

    target_dims = tuple(d for d in expected if d in arr.dims) + tuple(
        d for d in arr.dims if d not in expected
    )
    arr = arr.transpose(*target_dims)

    coord_order = [c for c in target_dims if c in arr.coords] + [
        c for c in arr.coords if c not in target_dims
    ]
    if list(arr.coords) != coord_order:
        arr = DataArray(
            arr.variable,
            coords={c: arr.coords[c] for c in coord_order},
            name=arr.name,
        )

    return arr, projections


def broadcast_to_coords(
    arr: Any,
    coords: CoordsLike | None = None,
    dims: DimsLike | None = None,
    **kwargs: Any,
) -> DataArray:
    """
    Convert ``arr`` to a DataArray and broadcast it against ``coords``.

    When ``coords`` carries named dimensions, the result is aligned with
    them: positional inputs are labeled by position, shared dims with equal
    values in a different order are reindexed, dims missing from ``arr``
    are expanded, dims naming levels of a stacked-MultiIndex coords dim are
    projected onto it, and the result is transposed to ``coords`` order.

    Dims of ``arr`` not present in ``coords``, and shared dims with
    disagreeing value sets, pass through unchanged so downstream xarray
    alignment can handle them. Use :func:`align_to_coords` to enforce that
    ``arr`` stays within ``coords``.

    Implicit MultiIndex-level projections (a level subset, or one that
    leaves entries uncovered) emit an :class:`~linopy.EvolvingAPIWarning`;
    the v1 arithmetic convention will require them to be explicit.
    """
    da, projections = _broadcast_to_coords(arr, coords, dims, **kwargs)
    for p in projections:
        if not p.is_partial and not p.has_gap:
            continue
        kind = (
            f"broadcasting level subset {p.levels}"
            if p.is_partial
            else f"filling uncovered entries with NaN (from level(s) {p.levels})"
        )
        warn(
            f"multiindex-projection: implicitly {kind} onto MultiIndex "
            f"dimension {p.dim!r}. The v1 arithmetic convention will require "
            f"this to be explicit; reindex onto the dimension or use a "
            f"named method with `join=` to keep current behavior.",
            EvolvingAPIWarning,
            stacklevel=2,
        )
    return da


def validate_alignment(
    arr: DataArray,
    coords: CoordsLike | None,
    dims: DimsLike | None = None,
    *,
    label: str | None = None,
) -> None:
    """
    Raise ``ValueError`` if ``arr`` is incompatible with ``coords``.

    ``arr`` is compatible with ``coords`` when both of the following hold:

    - every dim in ``arr.dims`` is also a dim in ``coords`` (no extras);
    - for every dim shared between ``arr`` and ``coords``, the coord
      values are equal.

    ``dims`` mirrors the ``dims`` argument of ``as_dataarray``: it names
    unnamed entries in a sequence-form ``coords`` by position, so
    ``coords=[[1, 2, 3]], dims=["x"]`` is enforced the same way as
    ``coords={"x": [1, 2, 3]}``.

    ``label`` names the argument in error messages (e.g. ``"lower bound"``).

    No-op when ``coords`` is ``None`` or carries no named dimensions.
    """
    if coords is None:
        return
    expected = _coords_to_dict(coords, dims=dims)
    if not expected:
        return
    subject = label or "Value"
    expected_dims = set(expected)
    extra = set(arr.dims) - expected_dims
    if extra:
        raise ValueError(
            f"{subject} has dimension(s) {sorted(extra, key=str)} not declared in coords "
            f"({sorted(expected_dims, key=str)}). Add them to coords or remove them from "
            f"{subject.lower()}."
        )
    for dim, coord_values in expected.items():
        if dim not in arr.dims:
            continue
        expected_is_mi = isinstance(coord_values, pd.MultiIndex)
        actual_is_mi = isinstance(arr.indexes.get(dim), pd.MultiIndex)
        if expected_is_mi or actual_is_mi:
            if expected_is_mi and actual_is_mi:
                if not arr.indexes[dim].equals(coord_values):
                    raise ValueError(
                        f"{subject}: MultiIndex for dimension {dim!r} does not "
                        f"match coords."
                    )
            continue
        expected_idx = _as_index(coord_values)
        actual_idx = arr.coords[dim].to_index()
        if not actual_idx.equals(expected_idx):
            raise ValueError(
                f"{subject}: coordinate values for dimension {dim!r} do not match "
                f"coords — expected {expected_idx.tolist()}, got "
                f"{actual_idx.tolist()}."
            )


def align_to_coords(
    value: Any,
    coords: CoordsLike | None,
    *,
    label: str,
    dims: DimsLike | None = None,
    **kwargs: Any,
) -> DataArray:
    """
    Convert and broadcast ``value`` against ``coords``, enforcing the coords contract.

    On top of :func:`broadcast_to_coords` this requires that ``value`` stays
    within ``coords``: no extra dims, no disagreeing coord values, and no
    MultiIndex coverage gaps. Errors are raised as :class:`ValueError` /
    :class:`TypeError` naming ``label``; coords-parsing errors propagate
    unchanged.
    """
    if coords is not None:
        _coords_to_dict(coords, dims=dims)
    try:
        da, projections = _broadcast_to_coords(value, coords, dims=dims, **kwargs)
    except TypeError as err:
        raise TypeError(f"{label} could not be aligned to coords: {err}") from err
    except (ValueError, CoordinateValidationError) as err:
        raise ValueError(f"{label} could not be aligned to coords: {err}") from err
    for p in projections:
        if p.has_gap:
            raise ValueError(
                f"{label} could not be aligned to coords: input does not cover "
                f"every entry of MultiIndex dimension {p.dim!r} (aligned from "
                f"level(s) {p.levels})."
            )
    validate_alignment(da, coords, dims=dims, label=label)
    return da


def _coords_to_dict(
    coords: Sequence[Sequence | pd.Index] | Mapping,
    dims: DimsLike | None = None,
) -> dict[Hashable, Any]:
    """
    Normalize coords to a dict mapping dim names to coordinate values.

    Container forms:

    - ``xarray.Coordinates``  → kept dim entries only (MultiIndex level
      coords dropped).
    - ``Mapping``             → returned as a shallow ``dict`` copy.
    - sequence-of-entries     → each entry handled per the rules below.

    Sequence-entry rules (``i`` is the position in ``coords``, ``dims[i]``
    is the matching entry in ``dims`` when one exists). An entry is
    *unlabeled* if it's an unnamed ``pd.Index`` or a bare ``list`` /
    ``tuple`` / ``range`` / ``ndarray``.

    +---------------------------------+-----------------------+-----------+
    | Entry                           | Naming source         | Outcome   |
    +=================================+=======================+===========+
    | ``pd.Index`` with ``.name``     | ``.name``             | accepted  |
    +---------------------------------+-----------------------+-----------+
    | unlabeled entry                 | ``dims[i]``           | accepted  |
    +---------------------------------+-----------------------+-----------+
    | unlabeled entry                 | — (no ``dims[i]``)    | skipped   |
    |                                 |                       | — xarray  |
    |                                 |                       | assigns   |
    |                                 |                       | ``dim_0`` |
    |                                 |                       | etc.      |
    +---------------------------------+-----------------------+-----------+
    | ``pd.MultiIndex`` with ``.name``| ``.name``             | accepted  |
    +---------------------------------+-----------------------+-----------+
    | ``pd.MultiIndex`` w/o ``.name`` | ``dims[i]``           | accepted  |
    |                                 |                       | (named on |
    |                                 |                       | a copy)   |
    +---------------------------------+-----------------------+-----------+
    | ``pd.MultiIndex`` w/o ``.name`` | — (no ``dims[i]``)    | TypeError |
    +---------------------------------+-----------------------+-----------+
    | anything else (e.g. DataArray)  | —                     | TypeError |
    +---------------------------------+-----------------------+-----------+
    """
    if isinstance(coords, Coordinates):
        # Coordinates iterates over every coord variable, including
        # MultiIndex level coords. Keep only the entries that are dims.
        return {d: coords[d] for d in coords.dims if d in coords}
    if isinstance(coords, Mapping):
        return dict(coords)
    dim_names: list[Any] | None = None
    if dims is not None:
        dim_names = list(dims) if isinstance(dims, list | tuple) else [dims]
    result: dict[Hashable, Any] = {}
    for i, c in enumerate(coords):
        if isinstance(c, pd.MultiIndex):
            name = c.name or (
                dim_names[i] if dim_names and i < len(dim_names) else None
            )
            if name is None:
                raise TypeError(
                    "MultiIndex coords entries must have .name set so "
                    "xarray can use it as the dimension name. Set it via "
                    "`idx.name = 'my_dim'`, or pass `dims=[...]` to name "
                    "entries by position."
                )
            if c.name is None:
                c = c.copy()
                c.name = name
            result[name] = c
        elif isinstance(c, pd.Index):
            name = (
                c.name
                if c.name
                else (dim_names[i] if dim_names and i < len(dim_names) else None)
            )
            if name is not None:
                result[name] = c
        elif isinstance(c, list | tuple | range | np.ndarray):
            if dim_names and i < len(dim_names):
                result[dim_names[i]] = pd.Index(c, name=dim_names[i])
        else:
            raise TypeError(
                f"coords entries must be pd.Index or an unnamed sequence "
                f"(list / tuple / range / numpy.ndarray); got "
                f"{type(c).__name__}. For an xarray DataArray coord, pass "
                f"`variable.indexes[<dim>]` (a pd.Index) instead."
            )
    return result


def _named_pandas_to_dataarray(arr: pd.Series | pd.DataFrame) -> DataArray | None:
    """
    Convert a pandas Series or DataFrame with fully named axes to a DataArray.

    Returns ``None`` if any axis (or MultiIndex level) is unnamed or
    non-string, so the caller can fall back to ``as_dataarray``.
    """
    names = list(arr.index.names)
    if isinstance(arr, pd.DataFrame):
        names += list(arr.columns.names)
    if any(not isinstance(n, str) for n in names):
        return None

    if isinstance(arr, pd.DataFrame):
        if isinstance(arr.index, pd.MultiIndex) or isinstance(
            arr.columns, pd.MultiIndex
        ):
            arr = arr.stack(list(range(arr.columns.nlevels)), future_stack=True)
            return arr.to_xarray()
        return DataArray(arr)

    return arr.to_xarray()


# TODO: rename to to_pandas_dataframe
def to_dataframe(
    ds: Dataset,
    mask_func: Callable[[dict[Hashable, np.ndarray]], pd.Series] | None = None,
) -> pd.DataFrame:
    """
    Convert an xarray Dataset to a pandas DataFrame.

    This is an memory efficient alternative implementation to the built-in `to_dataframe` method, which
    does not create a multi-indexed DataFrame.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to convert to a DataFrame.
    """
    data = broadcast(ds)[0]
    datadict = {k: v.values.reshape(-1) for k, v in data.items()}

    if mask_func is not None:
        mask = mask_func(datadict)
        for k, v in datadict.items():
            datadict[k] = v[mask]

    return pd.DataFrame(datadict, copy=False)


def check_has_nulls(df: pd.DataFrame, name: str) -> None:
    any_nan = df.isna().any()
    if any_nan.any():
        fields = ", ".join(df.columns[any_nan].to_list())
        raise ValueError(f"Fields {name} contains nan's in field(s) {fields}")


def infer_schema_polars(ds: Dataset) -> dict[str, DataTypeClass]:
    """
    Infer the polars data schema from a xarray dataset.

    Args:
    ----
        ds (polars.DataFrame): The Polars DataFrame for which to infer the schema.

    Returns:
    -------
        dict: A dictionary mapping column names to their corresponding Polars data types.
    """
    schema: dict[str, DataTypeClass] = {}
    np_major_version = int(np.__version__.split(".")[0])
    use_int32 = os.name == "nt" and np_major_version < 2
    for name, array in ds.items():
        name = str(name)
        if np.issubdtype(array.dtype, np.integer):
            schema[name] = pl.Int32 if use_int32 else pl.Int64
        elif np.issubdtype(array.dtype, np.floating):
            schema[name] = pl.Float64
        elif np.issubdtype(array.dtype, np.bool_):
            schema[name] = pl.Boolean
        elif np.issubdtype(array.dtype, np.object_):
            schema[name] = pl.Object
        else:
            schema[name] = pl.Utf8
    return schema


def to_polars(ds: Dataset, **kwargs: Any) -> pl.DataFrame:
    """
    Convert an xarray Dataset to a polars DataFrame.

    This is an memory efficient alternative implementation
    of `to_dataframe`.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to convert to a DataFrame.
    kwargs : dict
        Additional keyword arguments to be passed to the
        DataFrame constructor.
    """
    data = broadcast(ds)[0]
    return pl.DataFrame({k: v.values.reshape(-1) for k, v in data.items()}, **kwargs)


def check_has_nulls_polars(df: pl.DataFrame, name: str = "") -> None:
    """
    Checks if the given DataFrame contains any null or NaN values and raises a ValueError if it does.

    Args:
    ----
        df (pl.DataFrame): The DataFrame to check for null or NaN values.
        name (str): The name of the data container being checked.

    Raises:
    ------
        ValueError: If the DataFrame contains null or NaN values,
        a ValueError is raised with a message indicating the name of the constraint and the fields containing null/NaN values.
    """
    # Check for null values in all columns
    has_nulls = df.select(pl.col("*").is_null().any())
    null_columns = [col for col in has_nulls.columns if has_nulls[col][0]]

    # Check for NaN values only in numeric columns (avoid enum/categorical columns)
    numeric_cols = [
        col for col, dtype in zip(df.columns, df.dtypes) if dtype.is_numeric()
    ]

    nan_columns = []
    if numeric_cols:
        has_nans = df.select(pl.col(numeric_cols).is_nan().any())
        nan_columns = [col for col in has_nans.columns if has_nans[col][0]]

    invalid_columns = list(set(null_columns + nan_columns))
    if invalid_columns:
        raise ValueError(f"{name} contains nan's in field(s) {invalid_columns}")


def filter_nulls_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter out rows containing "empty" values from a polars DataFrame.

    Args:
    ----
        df (pl.DataFrame): The DataFrame to filter.

    Returns:
    -------
        pl.DataFrame: The filtered DataFrame.
    """
    cond = []
    varcols = [c for c in df.columns if c.startswith("vars")]
    if varcols:
        cond.append(reduce(operator.or_, [pl.col(c).ne(-1) for c in varcols]))
    if "coeffs" in df.columns:
        cond.append(pl.col("coeffs").ne(0))
    if "labels" in df.columns:
        cond.append(pl.col("labels").ne(-1))

    cond = reduce(operator.and_, cond)  # type: ignore[arg-type]
    return df.filter(cond)


def group_terms_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Groups terms in a polars DataFrame.

    Args:
    ----
        df (pl.DataFrame): The input DataFrame containing the terms.

    Returns:
    -------
        pl.DataFrame: The DataFrame with grouped terms.

    """
    varcols = [c for c in df.columns if c.startswith("vars")]
    agg_list = [pl.col("coeffs").sum().alias("coeffs")]
    for col in set(df.columns) - set(["coeffs", "labels", *varcols]):
        agg_list.append(pl.col(col).first().alias(col))

    by = [c for c in ["labels"] + varcols if c in df.columns]
    df = df.group_by(by, maintain_order=True).agg(agg_list)
    return df


def maybe_group_terms_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Group terms only if there are duplicate (labels, vars) pairs.

    This avoids the expensive group_by operation when terms already
    reference distinct variables (e.g. ``x - y`` has ``_term=2`` but
    no duplicates). When skipping, columns are reordered to match the
    output of ``group_terms_polars``.
    """
    varcols = [c for c in df.columns if c.startswith("vars")]
    keys = [c for c in ["labels"] + varcols if c in df.columns]
    key_count = df.select(pl.struct(keys).n_unique()).item()
    if key_count < df.height:
        return group_terms_polars(df)
    # Match column order of group_terms (group-by keys, coeffs, rest)
    rest = [c for c in df.columns if c not in keys and c != "coeffs"]
    return df.select(keys + ["coeffs"] + rest)


def save_join(*dataarrays: DataArray, integer_dtype: bool = False) -> Dataset:
    """
    Join multiple xarray Dataarray's to a Dataset and warn if coordinates are not equal.
    """
    try:
        arrs = xr_align(*dataarrays, join="exact")
    except ValueError:
        warn(
            "Coordinates across variables not equal. Perform outer join.",
            UserWarning,
        )
        arrs = xr_align(*dataarrays, join="outer")
        if integer_dtype:
            arrs = tuple([ds.fillna(-1).astype(int) for ds in arrs])
    return Dataset({ds.name: ds for ds in arrs})


def assign_multiindex_safe(ds: Dataset, **fields: Any) -> Dataset:
    """
    Assign a field to a xarray Dataset while being safe against warnings about multiindex corruption.

    See https://github.com/PyPSA/linopy/issues/303 for more information

    Parameters
    ----------
    ds : Dataset
        Dataset to assign the field to
    keys : Union[str, List[str]]
        Keys of the fields
    to_assign : Union[List[DataArray], DataArray, Dataset]
        New values added to the dataset

    Returns
    -------
    Dataset
        Merged dataset with the new field added
    """
    remainders = list(set(ds) - set(fields))
    return Dataset({**ds[remainders], **fields}, attrs=ds.attrs)


@overload
def fill_missing_coords(ds: DataArray, fill_helper_dims: bool = False) -> DataArray: ...


@overload
def fill_missing_coords(ds: Dataset, fill_helper_dims: bool = False) -> Dataset: ...


def fill_missing_coords(
    ds: DataArray | Dataset, fill_helper_dims: bool = False
) -> Dataset | DataArray:
    """
    Fill coordinates of a xarray Dataset or DataArray with integer coordinates.

    This function fills in the integer coordinates for all dimensions of a
    Dataset or DataArray that have no coordinates assigned yet.

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
    fill_helper_dims : bool, optional
        Whether to fill in integer coordinates for helper dimensions, by default False.

    """
    ds = ds.copy()
    if not isinstance(ds, Dataset | DataArray):
        raise TypeError(f"Expected xarray.DataArray or xarray.Dataset, got {type(ds)}.")

    skip_dims = [] if fill_helper_dims else HELPER_DIMS

    # Fill in missing integer coordinates
    for dim in ds.dims:
        if dim not in ds.coords and dim not in skip_dims:
            ds.coords[dim] = arange(ds.sizes[dim])

    return ds


T = TypeVar("T", Dataset, "Variable", "LinearExpression", "ConstraintBase")


@overload
def iterate_slices(
    ds: Dataset,
    slice_size: int | None = 10_000,
    slice_dims: list | None = None,
) -> Generator[Dataset, None, None]: ...


@overload
def iterate_slices(
    ds: Variable,
    slice_size: int | None = 10_000,
    slice_dims: list | None = None,
) -> Generator[Variable, None, None]: ...


@overload
def iterate_slices(
    ds: LinearExpression,
    slice_size: int | None = 10_000,
    slice_dims: list | None = None,
) -> Generator[LinearExpression, None, None]: ...


@overload
def iterate_slices(
    ds: ConstraintBase,
    slice_size: int | None = 10_000,
    slice_dims: list | None = None,
) -> Generator[ConstraintBase, None, None]: ...


def iterate_slices(
    ds: T,
    slice_size: int | None = 10_000,
    slice_dims: list | None = None,
) -> Generator[T, None, None]:
    """
    Generate slices of an xarray Dataset or DataArray with a specified soft maximum size.

    The slicing is performed on the largest dimension of the input object.
    If the maximum size is larger than the total size of the object, the function yields
    the original object.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The input xarray Dataset or DataArray to be sliced.
    slice_size : int
        The maximum number of elements in each slice. If the maximum size is too small to accommodate any slice,
        the function splits the largest dimension.
    slice_dims : list, optional
        The dimensions to slice along. If None, all dimensions in `coord_dims` are used if
        `coord_dims` is an attribute of the input object. Otherwise, all dimensions are used.

    Yields
    ------
    xarray.Dataset or xarray.DataArray
        A slice of the input Dataset or DataArray.

    """
    if slice_dims is None:
        slice_dims = list(getattr(ds, "coord_dims", ds.dims))

    if not set(slice_dims).issubset(ds.dims):
        raise ValueError(
            "Invalid slice dimensions. Must be a subset of the dataset dimensions."
        )

    # Calculate the total number of elements in the dataset
    size = np.prod([ds.sizes[dim] for dim in ds.dims], dtype=int)

    if slice_size is None or size <= slice_size:
        yield ds
        return

    # number of slices
    n_slices = max((size + slice_size - 1) // slice_size, 1)

    # leading dimension (the dimension with the largest size)
    sizes = {dim: ds.sizes[dim] for dim in slice_dims}
    if not sizes:
        yield ds
        return

    leading_dim = max(sizes, key=sizes.get)  # type: ignore
    size_of_leading_dim = ds.sizes[leading_dim]

    if size_of_leading_dim < n_slices:
        n_slices = size_of_leading_dim

    chunk_size = (ds.sizes[leading_dim] + n_slices - 1) // n_slices

    # Iterate over the Cartesian product of slice indices
    for i in range(n_slices):
        start = i * chunk_size
        end = min(start + chunk_size, size_of_leading_dim)
        slice_dict = {leading_dim: slice(start, end)}
        yield ds.isel(slice_dict)  # type: ignore[attr-defined]


def _remap(array: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return mapping[array.ravel()].reshape(array.shape)


def count_initial_letters(word: str) -> int:
    """
    Count the number of initial letters in a word.
    """
    count = 0
    for char in word:
        if char.isalpha():
            count += 1
        else:
            break
    return count


def replace_by_map(ds: DataArray, mapping: np.ndarray) -> DataArray:
    """
    Replace values in a DataArray by a one-dimensional mapping.
    """
    return apply_ufunc(
        _remap,
        ds,
        kwargs=dict(mapping=mapping),
        dask="parallelized",
        output_dtypes=[mapping.dtype],
    )


def to_path(path: str | Path | None) -> Path | None:
    """
    Convert a string to a Path object.
    """
    return Path(path) if path is not None else None


def best_int(max_value: int) -> type[signedinteger[Any]]:
    """
    Get the minimal int dtype for storing values <= max_value.
    """
    for t in (np.int8, np.int16, np.int32, np.int64):
        if max_value <= np.iinfo(t).max:
            return t
    raise ValueError(f"Value {max_value} is too large for int64.")


def get_index_map(*arrays: Sequence[Hashable]) -> dict[tuple, int]:
    """
    Given arrays of hashable objects, create a map from unique combinations to unique integers.
    """
    # Create unique combinations
    unique_combinations = set(zip(*arrays))

    return {combination: i for i, combination in enumerate(unique_combinations)}


def generate_indices_for_printout(
    dim_sizes: Sequence[int], max_lines: int
) -> Generator[tuple[int | np.signedinteger[Any], ...] | None, None, None]:
    total_lines = int(np.prod(dim_sizes))
    lines_to_skip = total_lines - max_lines + 1 if total_lines > max_lines else 0
    if lines_to_skip > 0:
        half_lines = max_lines // 2
        for i in range(half_lines):
            yield np.unravel_index(i, dim_sizes)
        yield None
        for i in range(total_lines - half_lines, total_lines):
            yield tuple(np.unravel_index(i, dim_sizes))
    else:
        for i in range(total_lines):
            yield tuple(np.unravel_index(i, dim_sizes))


def align_lines_by_delimiter(lines: list[str], delimiter: str | list[str]) -> list[str]:
    # Determine the maximum position of the delimiter
    if isinstance(delimiter, str):
        delimiter = [delimiter]
    try:
        max_pos = max(line.index(d) for line in lines for d in delimiter if d in line)
    except ValueError:
        return lines

    # Create the formatted lines
    formatted_lines = []
    for line in lines:
        formatted_line = line
        for d in delimiter:
            if d in line:
                parts = line.split(d)
                formatted_line = f"{parts[0]:<{max_pos}}{d} {parts[1].strip()}"
        formatted_lines.append(formatted_line)
    return formatted_lines


def get_dims_with_index_levels(
    ds: Dataset, dims: Sequence[Hashable] | None = None
) -> list[str]:
    """
    Get the dimensions of a Dataset with their index levels.

    Example usage with a dataset that has:
    - regular dimension 'time'
    - multi-indexed dimension 'station' with levels ['country', 'city']
    The output would be: ['time', 'station (country, city)']
    """
    dims_with_levels = []
    if dims is None:
        dims = list(ds.dims)

    for dim in dims:
        if isinstance(ds.indexes[dim], pd.MultiIndex):
            # For multi-indexed dimensions, format as "dim (level0, level1, ...)"
            names = ds.indexes[dim].names
            dims_with_levels.append(f"{dim} ({', '.join(names)})")
        else:
            # For regular dimensions, just add the dimension name
            dims_with_levels.append(str(dim))

    return dims_with_levels


class LabelPositionIndex:
    """
    Index for fast O(log n) lookup of label positions using binary search.

    This class builds a sorted index of label ranges and uses binary search
    to find which container (variable/constraint) a label belongs to.

    Parameters
    ----------
    obj : Any
        Container object with items() method returning (name, val) pairs,
        where val has .labels and .range attributes.
    """

    __slots__ = ("_starts", "_names", "_obj", "_built")

    def __init__(self, obj: Any) -> None:
        self._obj = obj
        self._starts: np.ndarray | None = None
        self._names: list[str] | None = None
        self._built = False

    def _build_index(self) -> None:
        """Build the sorted index of label ranges."""
        if self._built:
            return

        ranges = []
        for name, val in self._obj.items():
            start, stop = val.range
            ranges.append((start, name))

        # Sort by start value
        ranges.sort(key=lambda x: x[0])
        self._starts = np.array([r[0] for r in ranges])
        self._names = [r[1] for r in ranges]
        self._built = True

    def invalidate(self) -> None:
        """Invalidate the index (call when items are added/removed)."""
        self._built = False
        self._starts = None
        self._names = None

    def find_single(self, value: int) -> tuple[str, dict] | tuple[None, None]:
        """Find the name and coordinates for a single label value."""
        if value == -1:
            return None, None

        self._build_index()
        starts = self._starts
        names = self._names
        assert starts is not None and names is not None

        # Binary search to find the right range
        idx = int(np.searchsorted(starts, value, side="right")) - 1

        if idx < 0 or idx >= len(starts):
            raise ValueError(f"Label {value} is not existent in the model.")

        name = names[idx]
        val = self._obj[name]
        start, stop = val.range

        # Verify the value is in range
        if value < start or value >= stop:
            raise ValueError(f"Label {value} is not existent in the model.")

        labels = val.labels
        index = np.unravel_index(value - start, labels.shape)
        coord = {dim: labels.indexes[dim][i] for dim, i in zip(labels.dims, index)}
        return name, coord

    def find_single_with_index(
        self, value: int
    ) -> tuple[str, dict, tuple[int, ...]] | tuple[None, None, None]:
        """
        Find name, coordinates, and raw numpy index for a single label value.

        Returns (name, coord, index) where index is a tuple of integers that
        can be used for direct numpy indexing (e.g., arr.values[index]).
        This avoids the overhead of xarray's .sel() method.
        """
        if value == -1:
            return None, None, None

        self._build_index()
        starts = self._starts
        names = self._names
        assert starts is not None and names is not None

        # Binary search to find the right range
        idx = int(np.searchsorted(starts, value, side="right")) - 1

        if idx < 0 or idx >= len(starts):
            raise ValueError(f"Label {value} is not existent in the model.")

        name = names[idx]
        val = self._obj[name]
        start, stop = val.range

        # Verify the value is in range
        if value < start or value >= stop:
            raise ValueError(f"Label {value} is not existent in the model.")

        labels = val.labels
        index = np.unravel_index(value - start, labels.shape)
        coord = {dim: labels.indexes[dim][i] for dim, i in zip(labels.dims, index)}
        return name, coord, index


def _get_label_position_linear(
    obj: Any, values: int | np.ndarray
) -> (
    tuple[str, dict]
    | tuple[None, None]
    | list[tuple[str, dict] | tuple[None, None]]
    | list[list[tuple[str, dict] | tuple[None, None]]]
):
    """
    Get tuple of name and coordinate for variable labels.

    This is the original O(n) implementation that scans through all items.
    Used only for testing/benchmarking comparisons.
    """

    def find_single(value: int) -> tuple[str, dict] | tuple[None, None]:
        if value == -1:
            return None, None
        for name, val in obj.items():
            labels = val.labels
            start, stop = val.range

            if value >= start and value < stop:
                index = np.unravel_index(value - start, labels.shape)

                # Extract the coordinates from the indices
                coord = {
                    dim: labels.indexes[dim][i] for dim, i in zip(labels.dims, index)
                }
                # Add the name of the DataArray and the coordinates to the result list
                return name, coord
        raise ValueError(f"Label {value} is not existent in the model.")

    if isinstance(values, int):
        return find_single(values)

    values = np.array(values)
    ndim = values.ndim
    if ndim == 0:
        return find_single(values.item())
    elif ndim == 1:
        return [find_single(v) for v in values]
    elif ndim == 2:
        return [[find_single(v) for v in _] for _ in values.T]
    else:
        raise ValueError("Array's with more than two dimensions is not supported")


class VariableLabelIndex:
    """
    Index for O(1) mapping between variable labels and dense positions.

    Both arrays are computed lazily and cached:
    - ``vlabels``: active variable labels in encounter order, shape (n_active_vars,)
    - ``label_to_pos``: derived from vlabels; size _xCounter, maps label -> position (-1 if masked)

    Invalidated by clearing the instance ``__dict__`` when variables are added or removed.
    """

    def __init__(self, variables: Any) -> None:
        self._variables = variables

    @cached_property
    def vlabels(self) -> np.ndarray:
        """Active variable labels in encounter order, shape (n_active_vars,)."""
        label_lists = []
        for _, var in self._variables.items():
            labels = var.labels.values.ravel()
            mask = labels != -1
            label_lists.append(labels[mask])
        return (
            np.concatenate(label_lists) if label_lists else np.array([], dtype=np.intp)
        )

    @cached_property
    def label_to_pos(self) -> np.ndarray:
        """
        Mapping from variable label to dense position, shape (_xCounter,).

        Position i in the active variable array corresponds to label vlabels[i].
        Masked or unused labels map to -1.
        """
        vlabels = self.vlabels
        n = self._variables.model._xCounter
        label_to_pos = np.full(n, -1, dtype=np.intp)
        label_to_pos[vlabels] = np.arange(len(vlabels), dtype=np.intp)
        return label_to_pos

    @property
    def n_active_vars(self) -> int:
        """Number of active (non-masked) variables."""
        return len(self.vlabels)

    def invalidate(self) -> None:
        """Clear cached arrays so they are recomputed on next access."""
        self.__dict__.pop("vlabels", None)
        self.__dict__.pop("label_to_pos", None)


class ConstraintLabelIndex:
    """
    Index for O(1) mapping between constraint labels and dense positions.

    Mirrors VariableLabelIndex on the constraint side, but without building
    the full constraint matrix — only labels and the row mask are computed.
    """

    def __init__(self, constraints: Any) -> None:
        self._constraints = constraints

    @cached_property
    def clabels(self) -> np.ndarray:
        """Active constraint labels in build order, shape (n_active_cons,)."""
        label_lists = [c.active_labels() for c in self._constraints.data.values()]
        return (
            np.concatenate(label_lists) if label_lists else np.array([], dtype=np.intp)
        )

    @cached_property
    def label_to_pos(self) -> np.ndarray:
        """Mapping from constraint label to dense position, shape (_cCounter,)."""
        clabels = self.clabels
        n = self._constraints.model._cCounter
        label_to_pos = np.full(n, -1, dtype=np.intp)
        label_to_pos[clabels] = np.arange(len(clabels), dtype=np.intp)
        return label_to_pos

    @property
    def n_active_cons(self) -> int:
        return len(self.clabels)

    def invalidate(self) -> None:
        self.__dict__.pop("clabels", None)
        self.__dict__.pop("label_to_pos", None)


def get_label_position(
    obj: Any,
    values: int | np.ndarray,
    index: LabelPositionIndex | None = None,
) -> (
    tuple[str, dict]
    | tuple[None, None]
    | list[tuple[str, dict] | tuple[None, None]]
    | list[list[tuple[str, dict] | tuple[None, None]]]
):
    """
    Get tuple of name and coordinate for variable labels.

    Uses O(log n) binary search with a cached index for fast lookups.

    Parameters
    ----------
    obj : Any
        Container object with items() method (Variables or Constraints).
    values : int or np.ndarray
        Label value(s) to look up.
    index : LabelPositionIndex, optional
        Pre-built index for fast lookups. If None, one will be created.

    Returns
    -------
    tuple or list
        (name, coord) tuple for single values, or list of tuples for arrays.
    """
    if index is None:
        index = LabelPositionIndex(obj)

    if isinstance(values, int):
        return index.find_single(values)

    values = np.array(values)
    ndim = values.ndim
    if ndim == 0:
        return index.find_single(values.item())
    elif ndim == 1:
        return [index.find_single(int(v)) for v in values]
    elif ndim == 2:
        return [[index.find_single(int(v)) for v in col] for col in values.T]
    else:
        raise ValueError("Array's with more than two dimensions is not supported")


def format_coord(coord: dict[str, Any] | Iterable[Any]) -> str:
    """
    Format coordinates into a string representation.

    Args:
        coord: Dictionary or iterable containing coordinate values.
              Values can be numbers, strings, or nested iterables.

    Returns:
        Formatted string representation of coordinates in brackets,
        with nested coordinates grouped in parentheses.

    Examples:
        >>> format_coord({"x": 1, "y": 2})
        '[1, 2]'
        >>> format_coord([1, 2, 3])
        '[1, 2, 3]'
        >>> format_coord([(1, 2), (3, 4)])
        '[(1, 2), (3, 4)]'
    """
    # Handle empty input
    if not coord:
        return ""

    # Extract values if input is dictionary
    values = coord.values() if isinstance(coord, dict) else coord

    # Convert each coordinate component to string
    formatted = []
    for value in values:
        if isinstance(value, list | tuple):
            formatted.append(f"({', '.join(str(x) for x in value)})")
        else:
            formatted.append(str(value))

    return f"[{', '.join(formatted)}]"


def format_single_variable(model: Any, label: int) -> str:
    if label == -1:
        return "None"

    variables = model.variables
    name, coord, index = variables.get_label_position_with_index(label)

    var = variables[name]
    # Use direct numpy indexing instead of .sel() for performance
    lower = var.lower.values[index]
    upper = var.upper.values[index]

    if var.attrs["binary"]:
        bounds = " ∈ {0, 1}"
    elif var.attrs["integer"]:
        bounds = f" ∈ Z ⋂ [{lower:.4g},...,{upper:.4g}]"
    else:
        bounds = f" ∈ [{lower:.4g}, {upper:.4g}]"

    return f"{name}{format_coord(coord)}{bounds}"


def format_single_expression(
    c: np.ndarray,
    v: np.ndarray,
    const: float,
    model: Any,
) -> str:
    """
    Print a single linear expression based on the coefficients and variables.
    """
    c, v = np.atleast_1d(c), np.atleast_1d(v)

    # catch case that to many terms would be printed
    def format_line(
        expr: list[tuple[float, tuple[str, Any] | list[tuple[str, Any]]]], const: float
    ) -> str:
        res = []
        for i, (coeff, var) in enumerate(expr):
            coeff_string = f"{coeff:+.4g}"
            if i:
                # split sign and coefficient
                coeff_string = f"{coeff_string[0]} {coeff_string[1:]}"

            if isinstance(var, list):
                var_string = ""
                for name, coords in var:
                    if name is not None:
                        coord_string = format_coord(coords)
                        var_string += f" {name}{coord_string}"
            else:
                name, coords = var
                coord_string = format_coord(coords)
                var_string = f" {name}{coord_string}"

            res.append(f"{coeff_string}{var_string}")

        if not np.isnan(const) and not (const == 0.0 and len(res) >= 1):
            const_string = f"{const:+.4g}"
            if len(res):
                res.append(f"{const_string[0]} {const_string[1:]}")
            else:
                res.append(const_string)
        return " ".join(res) if len(res) else "None"

    if v.ndim == 1:
        mask = v != -1
        c, v = c[mask], v[mask]
    else:
        # case for quadratic expressions
        mask = (v != -1).any(0)
        c = c[mask]
        v = v[:, mask]

    max_terms = options.get_value("display_max_terms")
    if len(c) > max_terms:
        truncate = max_terms // 2
        positions = model.variables.get_label_position(v[..., :truncate])
        expr = list(zip(c[:truncate], positions))
        res = format_line(expr, const)
        res += " ... "
        expr = list(
            zip(
                c[-truncate:],
                model.variables.get_label_position(v[-truncate:]),
            )
        )
        residual = format_line(expr, const)
        if residual != " None":
            res += residual
        return res
    expr = list(zip(c, model.variables.get_label_position(v)))
    return format_line(expr, const)


def format_single_constraint(model: Any, label: int) -> str:
    constraints = model.constraints
    name, coord = constraints.get_label_position(label)

    coeffs = model.constraints[name].coeffs.sel(coord).values
    vars = model.constraints[name].vars.sel(coord).values
    sign = model.constraints[name].sign.sel(coord).item()
    rhs = model.constraints[name].rhs.sel(coord).item()

    expr = format_single_expression(coeffs, vars, 0, model)
    sign = SIGNS_pretty[sign]

    return f"{name}{format_coord(coord)}: {expr} {sign} {rhs:.12g}"


def has_optimized_model(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Check if a reference model is set.
    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if self.model is None:
            raise AttributeError("No reference model set.")
        if self.model.status != "ok":
            raise AttributeError("Underlying model not optimized.")
        return func(self, *args, **kwargs)

    return wrapper


def require_constant(func: Callable[..., Any]) -> Callable[..., Any]:
    from linopy import expressions, variables

    @wraps(func)
    def wrapper(self: Any, arg: Any) -> Any:
        if isinstance(
            arg,
            variables.Variable
            | variables.ScalarVariable
            | expressions.LinearExpression
            | expressions.QuadraticExpression,
        ):
            raise TypeError(f"Assigned rhs must be a constant, got {type(arg)}).")
        return func(self, arg)

    return wrapper


def forward_as_properties(**routes: list[str]) -> Callable[[type], type]:
    #
    def add_accessor(cls: Any, item: str, attr: str) -> None:
        def get(self: Any) -> Any:
            return getattr(getattr(self, item), attr)

        setattr(cls, attr, property(get))

    def deco(cls: Any) -> Any:
        for item, attrs in routes.items():
            for attr in attrs:
                add_accessor(cls, item, attr)
        return cls

    return deco


def check_common_keys_values(list_of_dicts: list[dict[str, Any]]) -> bool:
    """
    Check if all common keys among a list of dictionaries have the same value.

    Parameters
    ----------
    list_of_dicts : list of dict
        A list of dictionaries.

    Returns
    -------
    bool
        True if all common keys have the same value across all dictionaries, False otherwise.
    """
    common_keys = set.intersection(*(set(d.keys()) for d in list_of_dicts))
    return all(len({d[k] for d in list_of_dicts if k in d}) == 1 for k in common_keys)


def align(
    *objects: LinearExpression | QuadraticExpression | Variable | T_Alignable,
    join: JoinOptions = "inner",
    copy: bool = True,
    indexes: Any = None,
    exclude: str | Iterable[Hashable] = frozenset(),
    fill_value: Any = dtypes.NA,
) -> tuple[LinearExpression | QuadraticExpression | Variable | T_Alignable, ...]:
    """
    Given any number of Variables, Expressions, Dataset and/or DataArray objects,
    returns new objects with aligned indexes and dimension sizes.

    Array from the aligned objects are suitable as input to mathematical
    operators, because along each dimension they have the same index and size.

    Missing values (if ``join != 'inner'``) are filled with ``fill_value``.
    The default fill value is NaN.

    This functions essentially wraps the xarray function
    :py:func:`xarray.align`.

    Parameters
    ----------
    *objects : Variable, LinearExpression, Dataset or DataArray
        Objects to align.
    join : {"outer", "inner", "left", "right", "exact", "override"}, optional
        Method for joining the indexes of the passed objects along each
        dimension:

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
        aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
        those of the first object with that dimension. Indexes for the same
        dimension must have the same size in all objects.

    copy : bool, default: True
        If ``copy=True``, data in the return values is always copied. If
        ``copy=False`` and reindexing is unnecessary, or can be performed with
        only slice operations, then the output may share memory with the input.
        In either case, new xarray objects are always returned.
    indexes : dict-like, optional
        Any indexes explicitly provided with the `indexes` argument should be
        used in preference to the aligned indexes.
    exclude : str, iterable of hashable or None, optional
        Dimensions that must be excluded from alignment
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.

    Returns
    -------
    aligned : tuple of DataArray or Dataset
        Tuple of objects with the same type as `*objects` with aligned
        coordinates.


    """
    from linopy.expressions import LinearExpression, QuadraticExpression
    from linopy.variables import Variable

    finisher: list[partial[Any] | Callable[[Any], Any]] = []
    das: list[Any] = []
    for obj in objects:
        if isinstance(obj, LinearExpression | QuadraticExpression):
            finisher.append(partial(obj.__class__, model=obj.model))
            das.append(obj.data)
        elif isinstance(obj, Variable):
            finisher.append(
                partial(
                    obj.__class__,
                    model=obj.model,
                    name=obj.data.attrs["name"],
                    skip_broadcast=True,
                )
            )
            das.append(obj.data)
        else:
            finisher.append(lambda x: x)
            das.append(obj)

    exclude = frozenset(exclude).union(HELPER_DIMS)
    aligned = xr_align(
        *das,
        join=join,
        copy=copy,
        indexes=indexes,
        exclude=exclude,
        fill_value=fill_value,
    )
    return tuple([f(da) for f, da in zip(finisher, aligned)])


LocT = TypeVar(
    "LocT",
    "Dataset",
    "Variable",
    "LinearExpression",
    "QuadraticExpression",
    "ConstraintBase",
)


class LocIndexer(Generic[LocT]):
    __slots__ = ("object",)
    object: LocT

    def __init__(self, obj: LocT) -> None:
        self.object = obj

    def __getitem__(
        self, key: dict[Hashable, Any] | tuple | slice | int | list
    ) -> LocT:
        if not is_dict_like(key):
            # expand the indexer so we can handle Ellipsis
            labels = indexing.expanded_indexer(key, self.object.ndim)
            key = dict(zip(self.object.dims, labels))
        return self.object.sel(key)  # type: ignore[attr-defined]


class EmptyDeprecationWrapper:
    """
    Temporary wrapper for a smooth transition from .empty() to .empty

    Use `bool(expr.empty)` to explicitly unwrap.

    See Also
    --------
    https://github.com/PyPSA/linopy/pull/425
    """

    __slots__ = ("value",)

    def __init__(self, value: bool):
        self.value = value

    def __bool__(self) -> bool:
        return self.value

    def __repr__(self) -> str:
        return repr(self.value)

    def __call__(self) -> bool:
        warn(
            "Calling `.empty()` is deprecated, use `.empty` property instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.value


def coords_to_dataset_vars(coords: list[pd.Index]) -> dict[str, DataArray]:
    """
    Serialize a list of pd.Index (including MultiIndex) to a DataArray dict.

    Suitable for embedding coordinate metadata as plain data variables in a
    Dataset that has its own unrelated dimensions (e.g. CSR netcdf format).
    Reconstruct with :func:`coords_from_dataset`.
    """
    data_vars: dict[str, DataArray] = {}
    for c in coords:
        if isinstance(c, pd.MultiIndex):
            for level_name, level_values in zip(c.names, c.levels):
                data_vars[f"_coord_{c.name}_level_{level_name}"] = DataArray(
                    np.array(level_values),
                    dims=[f"_coorddim_{c.name}_level_{level_name}"],
                )
            data_vars[f"_coord_{c.name}_codes"] = DataArray(
                np.array(c.codes).T,
                dims=[f"_coorddim_{c.name}", f"_coorddim_{c.name}_nlevels"],
            )
        else:
            data_vars[f"_coord_{c.name}"] = DataArray(
                np.array(c), dims=[f"_coorddim_{c.name}"]
            )
    return data_vars


def coords_from_dataset(ds: Dataset, coord_dims: list[str]) -> list[pd.Index]:
    """
    Deserialize a list of pd.Index (including MultiIndex) from a Dataset.

    Reconstructs coordinates previously serialized by :func:`coords_to_dataset_vars`.
    """
    coords = []
    for d in coord_dims:
        if f"_coord_{d}_codes" in ds:
            codes_2d = ds[f"_coord_{d}_codes"].values.T
            level_names = [
                str(k)[len(f"_coord_{d}_level_") :]
                for k in ds
                if str(k).startswith(f"_coord_{d}_level_")
            ]
            arrays = [
                ds[f"_coord_{d}_level_{ln}"].values[codes_2d[i]]
                for i, ln in enumerate(level_names)
            ]
            mi = pd.MultiIndex.from_arrays(arrays, names=level_names)
            mi.name = d
            coords.append(mi)
        else:
            coords.append(pd.Index(ds[f"_coord_{d}"].values, name=d))
    return coords


def is_constant(x: SideLike) -> bool:
    """
    Check if the given object is a constant type or an expression type without
    any variables.

    Note that an expression such as ``x - x + 1`` will evaluate to ``False`` as
    the expression is not simplified before evaluation.

    Parameters
    ----------
    x : SideLike
        The object to check.

    Returns
    -------
    bool
        True if the object is constant-like, False otherwise.
    """
    from linopy.expressions import (
        LinearExpression,
        QuadraticExpression,
    )
    from linopy.variables import ScalarVariable, Variable

    if isinstance(x, Variable | ScalarVariable):
        return False
    if isinstance(x, LinearExpression | QuadraticExpression):
        return x.is_constant
    if isinstance(x, CONSTANT_TYPES):
        return True
    raise TypeError(
        "Expected a constant, variable, or expression on the constraint side, "
        f"got {type(x)}."
    )


def values_to_lookup_array(
    values: np.ndarray, labels: np.ndarray, size: int | None = None
) -> np.ndarray:
    """
    Build a dense NaN-padded lookup array from values and integer labels.

    Non-negative labels are placed at their corresponding positions; negative
    labels are skipped. Gaps are filled with NaN.

    Parameters
    ----------
    values : np.ndarray
        Values to place into the lookup array.
    labels : np.ndarray
        Integer labels giving the target position for each value.
    size : int, optional
        Length of the returned array. Defaults to ``max(labels) + 1`` if any
        non-negative label is present, otherwise 0.

    Returns
    -------
    np.ndarray
        Dense float lookup array.
    """
    labels = np.asarray(labels, dtype=int)
    mask = labels >= 0
    if size is None:
        size = int(labels[mask].max()) + 1 if mask.any() else 0
    arr = np.full(size, nan, dtype=float)
    arr[labels[mask]] = values[mask]
    return arr
