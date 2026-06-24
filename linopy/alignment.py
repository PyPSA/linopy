#!/usr/bin/env python3
"""
Conversion, broadcasting, and alignment of user input against coordinates.

This module owns the seam between what users pass (scalars, numpy arrays,
pandas / polars objects, DataArrays) and what linopy stores (labelled
DataArrays conforming to a model's coordinates):

- :func:`as_dataarray` — convert only (type dispatch + positional labeling).
- :func:`broadcast_to_coords` — convert and broadcast against ``coords``;
  ``strict=True`` (default) raises on any mismatch, ``strict=False`` passes
  mismatches through for downstream xarray alignment.
- :func:`validate_alignment` — the validation primitive behind the strict mode.
- :func:`align` — the symmetric counterpart, wrapping :func:`xarray.align`
  for any number of linopy objects.

Terminology for stacked MultiIndex dimensions: a dim has *levels* (its
component index names, e.g. ``period`` / ``timestep``) and *level
combinations* (its elements — one tuple per position, e.g. ``(2030, 't1')``).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, overload

import numpy as np
import pandas as pd
import polars as pl
from numpy import arange
from xarray import Coordinates, DataArray, Dataset, broadcast
from xarray import align as xr_align
from xarray.core import dtypes
from xarray.core.types import JoinOptions, T_Alignable
from xarray.namedarray.utils import is_dict_like

try:
    from xarray.core.coordinates import CoordinateValidationError
except ImportError:
    # Added in xarray 2025.6.0; it subclasses ValueError on newer versions.
    CoordinateValidationError = ValueError  # type: ignore[assignment, misc]

from linopy.constants import HELPER_DIMS
from linopy.types import UNLABELED_TYPES, CoordsLike, DimsLike


def as_constant(other: Any) -> Any:
    """
    Normalize a degenerate operand for arithmetic on entry.

    Two normalizations let the operators treat every numeric operand the
    same way downstream:

    - a Python ``list`` carries array data but no numeric operators
      (``-[1, 2]`` is a ``TypeError``, ``[1, 2] * x`` repeats the list), so
      it becomes a numpy array;
    - a 0-d numpy array (``np.array(1)``) is unwrapped to a Python scalar so
      it takes the scalar fast-path instead of size-pairing.

    Everything else passes through unchanged — typed constants, DataArrays,
    Variables, and Expressions all already behave.
    """
    if isinstance(other, list):
        other = np.asarray(other)
    if isinstance(other, np.ndarray) and other.ndim == 0:
        return other.item()
    return other


if TYPE_CHECKING:
    from linopy.expressions import LinearExpression, QuadraticExpression
    from linopy.variables import Variable


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
    ``range`` / ``ndarray``. A ``tuple`` is **not** unlabeled: following
    xarray, it is read as ``(dim_name, values[, attrs])`` — the first
    element names the dimension.

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
    | ``(name, values)`` tuple        | ``name`` (1st elem)   | accepted  |
    |                                 |                       | (xarray   |
    |                                 |                       | form)     |
    +---------------------------------+-----------------------+-----------+
    | tuple of length < 2             | —                     | TypeError |
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
                result[name] = c if c.name == name else c.rename(name)
        elif isinstance(c, tuple):
            if (
                len(c) < 2
                or not isinstance(c[0], Hashable)
                or isinstance(c[0], list | tuple | np.ndarray)
            ):
                raise TypeError(
                    f"tuple coords entries follow xarray's (dim_name, values) "
                    f"convention; got {c!r}. Pass a list for a bare sequence "
                    f"of coordinate values."
                )
            name, values = c[0], c[1]
            try:
                result[name] = pd.Index(values, name=name)
            except TypeError as err:
                raise TypeError(
                    f"tuple coords entries follow xarray's (dim_name, values) "
                    f"convention with array-like values; got {c!r}. Pass a "
                    f"list for a bare sequence of coordinate values."
                ) from err
        elif isinstance(c, list | range | np.ndarray):
            if dim_names and i < len(dim_names):
                result[dim_names[i]] = pd.Index(c, name=dim_names[i])
        else:
            raise TypeError(
                f"coords entries must be pd.Index, an unlabeled sequence "
                f"(list / range / numpy.ndarray), or a (dim_name, values) "
                f"tuple; got {type(c).__name__}. For an xarray DataArray "
                f"coord, pass `variable.indexes[<dim>]` (a pd.Index) instead."
            )
    return result


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
    dims. Use :func:`broadcast_to_coords` when
    ``coords`` should govern the result's shape.

    Parameters
    ----------
    arr
        The input to convert.
    coords
        Coordinate values used to label positional axes.
    dims
        Dimension names used to label positional axes.
    **kwargs
        Forwarded to the underlying DataArray construction.

    Returns
    -------
    DataArray
        The converted input, dims and entries as ``arr`` provides them.
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
            # A scalar broadcasts over the coords' dims, but never over a
            # helper dim (e.g. ``_term``) — those are storage book-keeping,
            # not user axes.
            if isinstance(coords, Coordinates):
                dims = [d for d in coords.dims if d not in HELPER_DIMS]
            elif is_dict_like(coords) and np.ndim(arr) == 0:
                dims = [d for d in coords.keys() if d not in HELPER_DIMS]
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


class _LevelProjection(NamedTuple):
    """
    Record of one MultiIndex-level projection performed by ``_broadcast_to_coords``.

    Terminology: a stacked MultiIndex dim has *levels* (its component index
    names, e.g. ``period`` / ``timestep``) and *level combinations* (its
    elements — one tuple per position, e.g. ``(2030, 't1')``).
    """

    dim: Hashable
    levels: list[Hashable]
    is_partial: bool  # input carried only a subset of the MI's levels
    has_gap: bool  # some level combinations of the MI dim got no value (NaN)
    missing: list[Any]  # the level combinations that got no value


def _project_onto_multiindex_levels(
    arr: DataArray,
    expected: dict[Hashable, Any],
) -> tuple[DataArray, list[_LevelProjection]]:
    """
    Map ``arr`` dims that name levels of a stacked-MultiIndex coords dim onto it.

    For every level combination of the MultiIndex dim, select the ``arr``
    value at that combination's level values. A subset of levels broadcasts
    across the remaining ones; the full set aligns element-wise. ``arr`` is
    returned unchanged when it carries no level dims.

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
        # A level combination is "missing" when the projection gave it no
        # value at any position of the other dims.
        null_mask = arr.isnull()
        other_dims = [d for d in arr.dims if d != dim]
        if other_dims:
            null_mask = null_mask.any(other_dims)
        has_gap = bool(null_mask.any())
        missing = list(arr.indexes[dim][null_mask.values]) if has_gap else []
        projections.append(
            _LevelProjection(
                dim=dim,
                levels=levels,
                is_partial=len(levels) < sum(name is not None for name in mi.names),
                has_gap=has_gap,
                missing=missing,
            )
        )

    return arr, projections


def _enforce_implicit_projections(projections: list[_LevelProjection]) -> None:
    """
    Semantics policy for implicit MultiIndex-level projections.

    Implicit projection is legacy-only behavior (scenario B of the #732 /
    #737 discussion): under legacy semantics it emits a deprecation warning
    (#738: via ``warn_legacy``); under the v1 convention it raises — a dim
    naming a level of a stacked MultiIndex dim is a shared-dim / aux-coord
    concern (sections 8 and 11), and the projection must be written
    explicitly by the caller.

    The strict path raises on coverage gaps before reaching here, so only
    partial levels arrive there; the non-strict path sees both.
    """
    # Deferred import: linopy.semantics imports xarray/pandas machinery that
    # in turn may import this module's consumers; keep the seam lazy.
    from linopy.semantics import is_v1, warn_legacy

    for p in projections:
        if p.is_partial or p.has_gap:
            kind = (
                f"broadcasting level subset {p.levels}"
                if p.is_partial
                else f"filling uncovered level combinations with NaN "
                f"(from level(s) {p.levels})"
            )
            if is_v1():
                raise ValueError(
                    f"multiindex-projection: implicitly {kind} onto MultiIndex "
                    f"dimension {p.dim!r} is not supported under the v1 "
                    f"convention (sections 8 and 11). Project the input onto "
                    f"the dimension explicitly, e.g. select with the "
                    f"dimension's level values."
                )
            warn_legacy(
                f"multiindex-projection: implicitly {kind} onto MultiIndex "
                f"dimension {p.dim!r}. This is deprecated and will raise under "
                f"the v1 convention; project the input onto the dimension "
                f"explicitly (select with the dimension's level values) to "
                f"keep current behavior."
            )


def _pair_axes_by_size(
    shape: tuple[int, ...], sizes: dict[Hashable, int]
) -> tuple[list[Hashable] | None, str | None]:
    """
    Pair each axis of an unlabeled array with the operand dim of matching size.

    The pairing must be determined by the sizes alone (v1 convention,
    coordinate-alignment intro): every axis size must match exactly one
    operand dim, and no two axes may share a size. Returns
    ``(dims, None)`` on success or ``(None, problem)`` where ``problem``
    describes why the pairing is impossible or ambiguous.
    """
    by_size: dict[int, list[Hashable]] = {}
    for d, n in sizes.items():
        by_size.setdefault(n, []).append(d)

    axes_per_size: dict[int, int] = {}
    for s in shape:
        axes_per_size[s] = axes_per_size.get(s, 0) + 1

    for s, n_axes in axes_per_size.items():
        candidates = by_size.get(s, [])
        if len(candidates) < n_axes:
            return None, (
                f"no unambiguous dimension match for an axis of length {s}: "
                f"the operand has dimensions {dict(sizes)}."
            )
        if len(candidates) > 1 or n_axes > 1:
            return None, (
                f"axis of length {s} could pair with any of "
                f"{sorted(candidates, key=str)} — sizes alone cannot decide."
            )

    return [by_size[s][0] for s in shape], None


def _dims_for_unlabeled_operand(
    shape: tuple[int, ...], expected: dict[Hashable, Any]
) -> list[Hashable]:
    """
    Choose dim names for an unlabeled (numpy / list / polars) input.

    Used everywhere an unlabeled array meets a known set of dims — bounds
    and masks in ``add_variables`` / ``add_constraints``, and arithmetic
    operands (#736).

    v1 (convention, coordinate-alignment intro): axes pair with the dims by
    size; ambiguity or a missing match raises, with wrap-in-a-DataArray as
    the documented resolution. Legacy: axes pair with the leading dims
    positionally; a deprecation warning fires whenever the v1 pairing would
    differ from or reject the positional one.
    """
    from linopy.semantics import is_v1, warn_legacy

    # A 0-d operand has no axes to pair — it broadcasts over every dim, so it
    # carries no dim names (matching a bare scalar).
    if len(shape) == 0:
        return []

    # Helper dims (e.g. ``_term``) are storage book-keeping, never user axes,
    # so they are not pairing candidates.
    candidates = {d: v for d, v in expected.items() if d not in HELPER_DIMS}
    sizes = {d: len(_as_index(v)) for d, v in candidates.items()}
    paired, problem = _pair_axes_by_size(shape, sizes)
    positional = list(candidates)[: len(shape)]

    if is_v1():
        if problem is not None:
            raise ValueError(
                f"Cannot pair an unlabeled array of shape {tuple(shape)} with "
                f"the operand's dimensions: {problem} Wrap the array in an "
                f"xarray.DataArray with explicit dims to name its axes."
            )
        assert paired is not None
        return paired

    # LEGACY: remove at 1.0 — positional pairing plus the transition warning.
    if problem is not None:
        warn_legacy(
            f"An unlabeled array of shape {tuple(shape)} was paired with the "
            f"operand's leading dimension(s) {positional} by position. Under "
            f"the v1 convention this raises: {problem} Wrap the array in an "
            f"xarray.DataArray with explicit dims to keep it working."
        )
    elif paired != positional:
        warn_legacy(
            f"An unlabeled array of shape {tuple(shape)} was paired with the "
            f"operand's leading dimension(s) {positional} by position. Under "
            f"the v1 convention it pairs by size instead — with {paired} — "
            f"which gives a different result. Wrap the array in an "
            f"xarray.DataArray with explicit dims to make the pairing explicit."
        )
    return positional


def _matmul_operand_to_dataarray(
    other: Any, coords: Coordinates, coord_dims: tuple[Hashable, ...]
) -> DataArray:
    """
    Convert a non-expression ``@`` operand, pairing unlabeled axes by size.

    Shared by ``LinearExpression.__matmul__`` and
    ``QuadraticExpression.__matmul__``: :func:`_dims_for_positional_input`
    decides which dims the contraction collapses (#736) — by size under v1,
    positionally with a warning under legacy. Unlike the broadcast pipeline
    this only converts (no reindex / expand / transpose).
    """
    expected = {d: coords[d] for d in coord_dims}
    dims = _dims_for_positional_input(other, expected, None)
    return as_dataarray(other, coords=coords, dims=dims)


def _dims_for_positional_input(
    arr: Any, expected: dict[Hashable, Any], dims: DimsLike | None
) -> DimsLike | None:
    """
    Resolve the dim names a non-DataArray input's axes adopt.

    An explicit ``dims`` is honored as given. Otherwise an unlabeled
    array (numpy / list / polars) pairs its axes with ``expected`` by
    size (#736); any other input falls back to the coords dims, minus
    helper dims like ``_term`` which are never user axes.
    """
    if dims is not None:
        return dims
    if isinstance(arr, UNLABELED_TYPES) and np.ndim(arr) >= 1:
        return _dims_for_unlabeled_operand(np.shape(arr), expected)
    return [d for d in expected if d not in HELPER_DIMS]


def _label_input(
    arr: Any, expected: dict[Hashable, Any], dims: DimsLike | None, **kwargs: Any
) -> DataArray:
    """
    Convert a non-DataArray input to a DataArray labelled against ``expected``.

    The converter is handed ``expected`` (the normalized name→values dict),
    not the raw sequence-form coords, so it selects coords by name — a
    sequence would zip dims to coords by position, which is wrong once
    size-pairing has chosen a non-leading dim.
    """
    dims = _dims_for_positional_input(arr, expected, dims)
    arr = as_dataarray(arr, expected, dims=dims, **kwargs)
    # Re-assign non-MultiIndex coords from ``expected`` (a MultiIndex coord
    # re-assignment emits a FutureWarning and the conversion already used it).
    return arr.assign_coords(
        {
            d: expected[d]
            for d in arr.dims
            if d in expected and not isinstance(arr.indexes.get(d), pd.MultiIndex)
        }
    )


def _reindex_reordered_dims(arr: DataArray, expected: dict[Hashable, Any]) -> DataArray:
    """
    Reindex shared dims whose values match ``expected`` in a different order.

    Disagreeing value *sets* are left for downstream xarray alignment;
    only a pure reordering of the same values is conformed here.
    """
    for dim, coord_values in expected.items():
        if dim not in arr.dims or isinstance(arr.indexes.get(dim), pd.MultiIndex):
            continue
        expected_idx = _as_index(coord_values)
        actual_idx = arr.coords[dim].to_index()
        if actual_idx.equals(expected_idx):
            continue
        if len(actual_idx) == len(expected_idx) and set(actual_idx) == set(
            expected_idx
        ):
            arr = arr.reindex({dim: expected_idx})
    return arr


def _expand_missing_dims(arr: DataArray, expected: dict[Hashable, Any]) -> DataArray:
    """
    Broadcast ``arr`` over ``expected`` dims it does not yet carry.

    A MultiIndex-backed dim is broadcast against a proper ``Coordinates``
    template: plain ``expand_dims`` would drop its level coords and leave a
    degenerate flat index that fails to align downstream. The exception is
    when ``arr`` already carries one of the MultiIndex's level names —
    broadcasting would then raise on the conflicting index, so fall back to
    ``expand_dims``.
    """
    expand = {k: v for k, v in expected.items() if k not in arr.dims}
    if not expand:
        return arr
    plain = {}
    for dim, coord_values in expand.items():
        mi = _as_multiindex(coord_values)
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
    return arr


def _order_like_coords(arr: DataArray, expected: dict[Hashable, Any]) -> DataArray:
    """
    Transpose ``arr`` to ``coords`` dim order, then match coord iteration order.

    The reconstruction makes a Dataset built from ``arr`` pick up its dim
    order from coord insertion, not just the transpose.
    """
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
    return arr


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
    coverage gaps. Unlabeled inputs pair their axes with the coords dims by
    size (#736); see :func:`_label_input`.
    """
    if coords is None:
        return as_dataarray(arr, coords, dims, **kwargs), []

    expected = _coords_to_dict(coords, dims=dims)
    if not expected:
        return as_dataarray(arr, coords, dims, **kwargs), []

    if isinstance(arr, pd.Series | pd.DataFrame):
        converted = _named_pandas_to_dataarray(arr)
        if converted is not None:
            arr = converted

    if not isinstance(arr, DataArray):
        arr = _label_input(arr, expected, dims, **kwargs)

    arr, projections = _project_onto_multiindex_levels(arr, expected)
    arr = _reindex_reordered_dims(arr, expected)
    arr = _expand_missing_dims(arr, expected)
    arr = _order_like_coords(arr, expected)
    return arr, projections


@overload
def broadcast_to_coords(
    arr: Any,
    coords: CoordsLike | None = ...,
    dims: DimsLike | None = ...,
    *,
    strict: Literal[True] = ...,
    label: str,
    **kwargs: Any,
) -> DataArray: ...


@overload
def broadcast_to_coords(
    arr: Any,
    coords: CoordsLike | None = ...,
    dims: DimsLike | None = ...,
    *,
    strict: Literal[False],
    label: None = ...,
    **kwargs: Any,
) -> DataArray: ...


def broadcast_to_coords(
    arr: Any,
    coords: CoordsLike | None = None,
    dims: DimsLike | None = None,
    *,
    strict: bool = True,
    label: str | None = None,
    **kwargs: Any,
) -> DataArray:
    """
    Convert ``arr`` to a DataArray and broadcast it against ``coords``.

    When ``coords`` carries named dimensions, the result is aligned with
    them: positional inputs are labeled by position, shared dims with equal
    values in a different order are reindexed, dims missing from ``arr``
    are expanded, dims naming levels of a stacked-MultiIndex coords dim are
    projected onto it, and the result is transposed to ``coords`` order.

    ``strict`` decides what happens to anything broadcasting alone cannot
    resolve — extra dims, disagreeing coord values, and MultiIndex coverage
    gaps:

    - ``strict=True`` (default): raise, naming ``label`` in the error.
    - ``strict=False``: pass through unchanged so downstream xarray
      alignment can handle them.

    A stacked-MultiIndex dim of ``coords`` has *levels* (its component
    index names, e.g. ``period`` / ``timestep``) and *level combinations*
    (its elements — one tuple per position, e.g. ``(2030, 't1')``). Inputs
    indexed by levels instead of the dim itself are implicitly projected
    onto the dim's level combinations. These projections are legacy-only:
    under legacy semantics they emit a :class:`~linopy.LinopySemanticsWarning`
    deprecation; under the v1 convention they raise. Two cases:

    - input misses a whole level → broadcasts across it; warns in both modes.
    - input gives some level combinations no value (a *coverage gap*) →
      warns under ``strict=False``, raises under ``strict=True`` (the error
      lists the missing combinations).

    Parameters
    ----------
    arr
        The input to convert and broadcast.
    coords
        Coordinate values the result is broadcast against. ``None`` falls
        back to plain conversion.
    dims
        Dimension names used to label positional axes.
    strict
        Check that the result stays within ``coords`` (raise on violation)
        instead of passing violations through.
    label
        Name of the input in error messages (e.g. ``"lower bound"``).
        Required when ``strict=True``, not accepted otherwise.
    **kwargs
        Forwarded to the underlying DataArray construction.

    Returns
    -------
    DataArray
        Broadcast against ``coords``.
    """
    if not strict:
        da, projections = _broadcast_to_coords(arr, coords, dims, **kwargs)
        _enforce_implicit_projections(projections)
        return da

    if label is None:
        raise TypeError(
            "broadcast_to_coords(strict=True) requires `label` to name the "
            "input in error messages, e.g. label='lower bound'."
        )
    subject = label
    if coords is not None:
        _coords_to_dict(coords, dims=dims)
    try:
        da, projections = _broadcast_to_coords(arr, coords, dims=dims, **kwargs)
    except TypeError as err:
        raise TypeError(f"{subject} could not be aligned to coords: {err}") from err
    except (ValueError, CoordinateValidationError) as err:
        raise ValueError(f"{subject} could not be aligned to coords: {err}") from err
    for p in projections:
        if p.has_gap:
            preview = ", ".join(str(c) for c in p.missing[:5])
            if len(p.missing) > 5:
                preview += f", … ({len(p.missing)} in total)"
            raise ValueError(
                f"{subject} could not be aligned to coords: no value for "
                f"{len(p.missing)} level combination(s) of MultiIndex dimension "
                f"{p.dim!r}: {preview}. The input is indexed by level(s) "
                f"{p.levels} and must cover every combination."
            )
    _enforce_implicit_projections(projections)
    validate_alignment(da, coords, dims=dims, label=label)
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
        expected_mi = _as_multiindex(coord_values)
        actual_mi = _as_multiindex(arr.indexes.get(dim))
        if expected_mi is not None or actual_mi is not None:
            if (
                expected_mi is None
                or actual_mi is None
                or not actual_mi.equals(expected_mi)
            ):
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
