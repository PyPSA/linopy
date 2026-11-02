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
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse
from numpy import arange
from xarray import Coordinates, DataArray, Dataset
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
from linopy.semantics import (
    _multiindex_message,
    _shared_dim_mismatch_message,
    check_user_nan,
    enforce_aux_conflict,
    enforce_no_multiindex,
    first_mismatched_dim,
)
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
    from linopy.csr import AuxCoords
    from linopy.expressions import LinearExpression, QuadraticExpression
    from linopy.variables import Variable


def _coords_to_dict(
    coords: Sequence[Sequence | pd.Index] | Mapping,
    dims: DimsLike | None = None,
) -> dict[Hashable, Any]:
    """
    Normalize coords to a dict mapping dim names to coordinate values.

    Container forms:

    - ``xarray.Coordinates``  → kept dim entries only (auxiliary coords
      dropped).
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
    | anything else (e.g. DataArray)  | —                     | TypeError |
    +---------------------------------+-----------------------+-----------+
    """
    if isinstance(coords, Coordinates):
        # Coordinates iterates over every coord variable, including
        # auxiliary coords. Keep only the entries that are dims.
        return _reject_multiindex_coords(
            {d: coords[d] for d in coords.dims if d in coords}
        )
    if isinstance(coords, Mapping):
        return _reject_multiindex_coords(dict(coords))
    dim_names: list[Any] | None = None
    if dims is not None:
        dim_names = list(dims) if isinstance(dims, list | tuple) else [dims]
    result: dict[Hashable, Any] = {}
    for i, c in enumerate(coords):
        if isinstance(c, pd.Index):
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
    return _reject_multiindex_coords(result)


def _reject_multiindex_coords(coords: dict[Hashable, Any]) -> dict[Hashable, Any]:
    for dim, values in coords.items():
        index = values.to_index() if isinstance(values, DataArray) else values
        if isinstance(index, pd.MultiIndex):
            raise ValueError(_multiindex_message(str(dim), "coords"))
    return coords


def _as_index(coord_values: Any) -> pd.Index:
    if isinstance(coord_values, DataArray):
        return coord_values.to_index()
    if isinstance(coord_values, pd.Index):
        return coord_values
    return pd.Index(coord_values)


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

    Returns ``None`` if any axis (or index level) is unnamed or
    non-string, so the caller can fall back to ``as_dataarray``.
    """
    names = list(arr.index.names)
    if isinstance(arr, pd.DataFrame):
        names += list(arr.columns.names)
    if any(not isinstance(n, str) for n in names):
        return None

    if isinstance(arr, pd.DataFrame):
        if arr.index.nlevels > 1 or arr.columns.nlevels > 1:
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
    if not isinstance(ds, Dataset | DataArray):
        raise TypeError(f"Expected xarray.DataArray or xarray.Dataset, got {type(ds)}.")

    skip_dims = [] if fill_helper_dims else HELPER_DIMS
    missing = [d for d in ds.dims if d not in ds.coords and d not in skip_dims]
    if not missing:
        return ds

    ds = ds.copy()
    for dim in missing:
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
    dims are neither expanded nor reordered. Use :func:`broadcast_to_coords` when
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
        arr = pandas_to_dataarray(arr, dims=dims, **kwargs)
    elif isinstance(arr, np.ndarray):
        arr = numpy_to_dataarray(arr, coords=coords, dims=dims, **kwargs)
    elif isinstance(arr, pl.Series):
        arr = numpy_to_dataarray(arr.to_numpy(), coords=coords, dims=dims, **kwargs)
    elif isinstance(arr, np.number | int | float | str | bool | list):
        if isinstance(arr, np.number):
            arr = float(arr)
        if dims is None:
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

    enforce_no_multiindex(arr, context="The constant")
    return fill_missing_coords(arr)


def _pair_axes_by_size(
    shape: tuple[int, ...], sizes: dict[Hashable, int]
) -> tuple[list[Hashable] | None, str | None]:
    """
    Pair each axis of an unlabeled array with the operand dim of matching size.

    The pairing must be determined by the sizes alone (convention,
    coordinate-alignment intro): every axis size must match exactly one
    operand dim, and no two axes may share a size. Returns
    ``(dims, None)`` on success or ``(None, problem)`` where ``problem``
    describes why the pairing is impossible or ambiguous.
    """
    by_size: dict[int, list[Hashable]] = {}
    for d, n in sizes.items():
        by_size.setdefault(n, []).append(d)

    for s, n_axes in Counter(shape).items():
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

    Axes pair with the dims by size (convention, coordinate-alignment
    intro); ambiguity or a missing match raises, with wrap-in-a-DataArray as
    the documented resolution. A 0-d operand has no axes to pair and
    broadcasts over every dim, like a bare scalar; helper dims (``_term``)
    are storage book-keeping and never pairing candidates.
    """
    if len(shape) == 0:
        return []

    candidates = {d: v for d, v in expected.items() if d not in HELPER_DIMS}
    sizes = {d: len(_as_index(v)) for d, v in candidates.items()}
    paired, problem = _pair_axes_by_size(shape, sizes)
    if paired is None:
        raise ValueError(
            f"Cannot pair an unlabeled array of shape {tuple(shape)} with "
            f"the operand's dimensions: {problem} Wrap the array in an "
            f"xarray.DataArray with explicit dims to name its axes."
        )
    return paired


def _matmul_operand_to_dataarray(
    other: Any, coords: Coordinates, coord_dims: tuple[Hashable, ...]
) -> DataArray:
    """
    Convert a non-expression ``@`` operand, pairing unlabeled axes by size.

    Shared by ``LinearExpression.__matmul__`` and
    ``QuadraticExpression.__matmul__``: :func:`_dims_for_positional_input`
    decides which dims the contraction collapses (#736) — by size. Unlike the
    broadcast pipeline this only converts (no reindex / expand / transpose).
    """
    expected = {d: coords[d] for d in coord_dims}
    dims = _dims_for_positional_input(other, expected, None)
    return as_dataarray(other, coords=coords, dims=dims)


def _matmul_operand_to_matrix(
    other: DataArray,
    contracted: Sequence[str],
    new_dims: Sequence[str],
    indexes: Mapping[str, pd.Index],
    aux_coords: AuxCoords,
) -> scipy.sparse.csr_array:
    """
    Flatten a ``@`` constant to the ``(contracted, new)`` matrix of the sparse
    contraction, enforcing the same rules ``*`` enforces on a constant: §5 on
    NaN, §8 on the labels of the shared (contracted) dims and §11 on auxiliary
    coordinates. The expression side is represented by its grid labels alone
    and never broadcasts.
    """
    if other.isnull().any():
        check_user_nan()
    reference = Dataset(coords={d: indexes[d] for d in contracted} | dict(aux_coords))
    mismatch = first_mismatched_dim(reference, other)
    if mismatch is not None:
        raise ValueError(_shared_dim_mismatch_message(*mismatch))
    enforce_aux_conflict([reference, other])
    values = other.transpose(*contracted, *new_dims).values
    n_contracted = int(np.prod([len(indexes[d]) for d in contracted], dtype=np.int64))
    n_new = int(np.prod([other.sizes[d] for d in new_dims], dtype=np.int64))
    return scipy.sparse.csr_array(values.reshape(n_contracted, n_new))


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
    return arr.assign_coords({d: expected[d] for d in arr.dims if d in expected})


def _reindex_reordered_dims(arr: DataArray, expected: dict[Hashable, Any]) -> DataArray:
    """
    Reindex shared dims whose values match ``expected`` in a different order.

    Disagreeing value *sets* are left for downstream xarray alignment;
    only a pure reordering of the same values is conformed here.
    """
    for dim, coord_values in expected.items():
        if dim not in arr.dims:
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
    """Broadcast ``arr`` over ``expected`` dims it does not yet carry."""
    expand = {k: v for k, v in expected.items() if k not in arr.dims}
    return arr.expand_dims(expand) if expand else arr


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
    *,
    reindex_reordered: bool = True,
    **kwargs: Any,
) -> DataArray:
    """
    Convert ``arr`` and broadcast it against ``coords`` (shared mechanics).

    Unlabeled inputs pair their axes with the coords dims by size (#736);
    see :func:`_label_input`. ``reindex_reordered`` is False on the
    arithmetic path, so a reorder reaches the §8 exact check instead of
    being conformed here.
    """
    if coords is None:
        return as_dataarray(arr, coords, dims, **kwargs)

    expected = _coords_to_dict(coords, dims=dims)
    if not expected:
        return as_dataarray(arr, coords, dims, **kwargs)

    if isinstance(arr, pd.Series | pd.DataFrame):
        converted = _named_pandas_to_dataarray(arr)
        if converted is not None:
            arr = converted

    if not isinstance(arr, DataArray):
        arr = _label_input(arr, expected, dims, **kwargs)

    if reindex_reordered:
        arr = _reindex_reordered_dims(arr, expected)
    arr = _expand_missing_dims(arr, expected)
    return _order_like_coords(arr, expected)


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
    values in a different order are reindexed (``strict=True`` only), dims
    missing from ``arr`` are expanded, and the result is transposed to
    ``coords`` order.

    ``strict`` decides what happens to anything broadcasting alone cannot
    resolve — extra dims and disagreeing coord values:

    - ``strict=True`` (default): raise, naming ``label`` in the error.
    - ``strict=False``: pass through unchanged so downstream xarray
      alignment can handle them.

    A pandas MultiIndex dimension on ``arr`` or ``coords`` is rejected.

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
    enforce_no_multiindex(arr, context=label or "The constant")
    if not strict:
        return _broadcast_to_coords(
            arr, coords, dims, reindex_reordered=False, **kwargs
        )

    if label is None:
        raise TypeError(
            "broadcast_to_coords(strict=True) requires `label` to name the "
            "input in error messages, e.g. label='lower bound'."
        )
    subject = label
    if coords is not None:
        _coords_to_dict(coords, dims=dims)
    try:
        da = _broadcast_to_coords(arr, coords, dims=dims, **kwargs)
    except TypeError as err:
        raise TypeError(f"{subject} could not be aligned to coords: {err}") from err
    except (ValueError, CoordinateValidationError) as err:
        raise ValueError(f"{subject} could not be aligned to coords: {err}") from err
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
