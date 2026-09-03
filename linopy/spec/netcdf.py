"""
Persist the spec of a spec-built model in its netcdf file.

Variables, constraints and the solution round trip through :mod:`linopy.io`
already. Besides them a spec-built model carries the spec text, the master
coordinates and the lookups; the program is re-lowered from the text on read,
so no lowered ``Program`` ever reaches the file.

Labels are the delicate part. A partial lookup holds NaN in an array of
labels, and neither the labels nor their holes survive a netcdf type: the
engines hand back ``<U`` or object, a hole comes back as an empty string, and
an int narrows to int32. So every lookup and every array of labels is written
as integer codes into its own table of categories, ``-1`` where a label is
missing, and each array carries the dtype it had in memory. Decoding indexes
the table and fills the holes back in, which reproduces what the binder
built, values and dtype alike.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from linopy.model import Model
from linopy.spec.accessor import ModelSpec, restore

PREFIX = "spec-"
COORD = "coords__"
CODES = "codes__"
CATEGORIES = "cats__"
CATEGORY_DIM = "category__"
DTYPE = "dtype"

LABEL_KINDS = frozenset("OUS")
HOLES: dict[str, Any] = {"f": np.nan, "O": np.nan, "M": np.datetime64("NaT")}


def encode(spec: ModelSpec) -> tuple[xr.Dataset, xr.Dataset]:
    """
    The model's parameters without the coded arrays, and the spec's own dataset.

    The spec dataset holds one array of labels per master coordinate and, per
    coded array, its codes and its categories. It carries no coordinates of
    its own: an index coordinate is dropped on read together with the
    dimension it indexes once no data variable is left over that dimension,
    and a master coordinate nothing else reaches has exactly that shape.
    """
    parameters = spec.parameters
    arrays: dict[str, xr.DataArray] = {
        COORD + dim: _array(index.to_numpy(), (dim,))
        for dim, index in spec.coords.items()
    }
    for name in _coded(spec):
        arrays.update(_encode(name, parameters[name]))
        parameters = parameters.drop_vars(name)
    return parameters, _prefixed(xr.Dataset(arrays))


def decode(model: Model, ds: xr.Dataset, text: str) -> ModelSpec:
    """
    Re-lower *text* onto *model* and put its coded arrays and coordinates back.

    The parameters read from the file are the retained ones minus what
    :func:`encode` took out; together with the master coordinates and the
    decoded arrays they are the dataset :func:`linopy.spec.accessor.attach`
    left on the model when it was built.
    """
    sub = _unprefixed(ds)
    coords = {
        _stripped(name, COORD): _index(sub[name])
        for name in sub.data_vars
        if str(name).startswith(COORD)
    }
    coded = {
        _stripped(name, CODES): _decode(sub, _stripped(name, CODES), coords)
        for name in sub.data_vars
        if str(name).startswith(CODES)
    }
    model.parameters = model.parameters.assign_coords(coords).assign(coded)
    return restore(model, text)


def _coded(spec: ModelSpec) -> list[str]:
    """The parameters written as codes: every lookup and every array of labels."""
    lookups = {name for by_name in spec.lookups.values() for name in by_name}
    return [
        str(name)
        for name, arr in spec.parameters.items()
        if name in lookups or arr.dtype.kind in LABEL_KINDS
    ]


def _encode(name: str, arr: xr.DataArray) -> dict[str, xr.DataArray]:
    codes, categories = pd.factorize(arr.to_numpy().ravel())
    written = {
        CODES + name: _array(
            codes.astype(np.int32).reshape(arr.shape), arr.dims, str(arr.dtype)
        )
    }
    if len(categories):
        written[CATEGORIES + name] = _array(
            np.asarray(categories), (CATEGORY_DIM + name,)
        )
    return written


def _decode(sub: xr.Dataset, name: str, coords: dict[str, pd.Index]) -> xr.DataArray:
    codes = sub[CODES + name]
    dtype = np.dtype(codes.attrs[DTYPE])
    categories = _categories(sub, name, dtype)
    positions = codes.to_numpy().astype(int)
    mapped = positions >= 0
    if mapped.all():
        values = categories[positions]
    else:
        values = np.full(positions.shape, HOLES[dtype.kind], dtype=dtype)
        values[mapped] = categories[positions[mapped]]
    dims = tuple(str(d) for d in codes.dims)
    return xr.DataArray(
        values, coords={d: coords[d] for d in dims}, dims=dims, name=name
    )


def _categories(sub: xr.Dataset, name: str, dtype: np.dtype) -> np.ndarray:
    """
    The table a coded array indexes.

    A map that leaves every label unmapped has no table: netCDF3 writes a
    zero-length dimension as the unlimited one, of which a file holds one.
    """
    written = CATEGORIES + name
    if written in sub.data_vars:
        return _values(sub[written])
    return np.empty(0, dtype=dtype)


def _array(
    values: np.ndarray, dims: tuple[Any, ...], dtype: str | None = None
) -> xr.DataArray:
    return xr.DataArray(values, dims=dims, attrs={DTYPE: dtype or str(values.dtype)})


def _prefixed(ds: xr.Dataset) -> xr.Dataset:
    return ds.rename({k: PREFIX + str(k) for k in (*ds.dims, *ds.data_vars)})


def _unprefixed(ds: xr.Dataset) -> xr.Dataset:
    sub = ds[[k for k in ds.data_vars if str(k).startswith(PREFIX)]]
    return sub.rename({k: str(k)[len(PREFIX) :] for k in (*sub.dims, *sub.data_vars)})


def _stripped(name: Any, prefix: str) -> str:
    return str(name)[len(prefix) :]


def _values(arr: xr.DataArray) -> np.ndarray:
    """The array as it was in memory, undoing what the netcdf type could not hold."""
    return arr.to_numpy().astype(np.dtype(arr.attrs[DTYPE]))


def _index(arr: xr.DataArray) -> pd.Index:
    return pd.Index(_values(arr), name=_stripped(arr.name, COORD))
