"""
Persist the spec of a spec-built model in its netcdf file.

Variables, constraints and the solution round trip through :mod:`linopy.io`
already. Besides them a spec-built model carries the spec text, the master
coordinates and the lookups; the program is re-lowered from the text on read,
so no lowered ``Program`` ever reaches the file.

No netcdf type holds a dtype as written, so every array carries the dtype it
had in memory (:func:`linopy.io.record_dtypes`) and is cast back to it on
read. That is enough for a parameter, but not for a partial lookup, which
holds NaN in an array of labels: a hole in a string array comes back as an
empty string, indistinguishable from a label. So a lookup, and any array of
objects, is written instead as integer codes into its own table of
categories, ``-1`` where a label is missing. Decoding indexes the table and
fills the holes back in, which reproduces what attach built, values and
dtype alike.

The master coordinates are canonical: a container's coordinates for a
dimension are re-stamped from them on read, so the whole model agrees on one
dtype per dimension however the engine returned it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from linopy.io import (
    DTYPE_ATTR,
    SPEC_ATTR,
    get_prefix,
    restamp_coords,
    with_prefix,
)
from linopy.model import Model
from linopy.spec.accessor import ModelSpec, restore

PREFIX = "spec"
OBJECTIVE_ATTR = "_linopy_spec_objective_replaced"
COORD = "coords__"
PARAM = "param__"
CODES = "codes__"
CATEGORIES = "cats__"
CATEGORY_DIM = "category__"

HOLES: dict[str, Any] = {"f": np.nan, "O": np.nan, "M": np.datetime64("NaT")}


def encode(spec: ModelSpec) -> xr.Dataset:
    """
    The spec's own dataset: its text, its master coordinates and its parameters.

    The spec text is the dataset's one attribute, which the merge lifts to the
    file's. Beside it sits one array of labels per master coordinate and, per
    parameter, either its values or -- where it is coded -- its codes and its
    categories. The dataset carries no coordinates of its own: an index
    coordinate is dropped on read together with the dimension it indexes once
    no data variable is left over that dimension, and a master coordinate
    nothing else reaches has exactly that shape. So a parameter is written
    over bare dimensions and put back on the master coordinates on read.
    """
    arrays: dict[str, xr.DataArray] = {
        COORD + dim: _array(index.to_numpy(), (dim,))
        for dim, index in spec.coords.items()
    }
    coded = _coded(spec)
    for name, arr in spec.parameters.items():
        if str(name) in coded:
            arrays.update(_encode(str(name), arr))
        else:
            arrays[PARAM + str(name)] = _array(arr.to_numpy(), arr.dims, str(arr.dtype))
    written = with_prefix(xr.Dataset(arrays), PREFIX).assign_attrs(
        {SPEC_ATTR: spec.text}
    )
    if spec.unspecified.objective:
        # Only when true, so a file written from an untouched spec is unchanged.
        written = written.assign_attrs({OBJECTIVE_ATTR: 1})
    return written


def decode(model: Model, ds: xr.Dataset, text: str) -> ModelSpec:
    """
    Re-lower *text* onto *model* and read back the dataset :func:`encode` wrote.

    The master coordinates, the plainly written parameters and the coded ones
    together are the dataset :func:`linopy.spec.accessor.attach` gave the spec
    when the model was built. ``model.parameters`` is not touched: it holds
    what the caller put there and nothing of the spec.
    """
    sub = get_prefix(ds, PREFIX)
    coords = {
        _stripped(name, COORD): _index(sub[name])
        for name in sub.data_vars
        if str(name).startswith(COORD)
    }
    arrays = {
        _stripped(name, PARAM): _plain(sub[name], _stripped(name, PARAM), coords)
        for name in sub.data_vars
        if str(name).startswith(PARAM)
    }
    arrays.update(
        {
            _stripped(name, CODES): _decode(sub, _stripped(name, CODES), coords)
            for name in sub.data_vars
            if str(name).startswith(CODES)
        }
    )
    restamp_coords(model, coords)
    return restore(
        model,
        text,
        xr.Dataset(arrays).assign_coords(coords),
        bool(ds.attrs.get(OBJECTIVE_ATTR, 0)),
    )


def _coded(spec: ModelSpec) -> set[str]:
    """The parameters written as codes: every lookup and every array of objects."""
    lookups = {name for by_name in spec.lookups.values() for name in by_name}
    return {
        str(name)
        for name, arr in spec.parameters.items()
        if name in lookups or arr.dtype == object
    }


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


def _plain(arr: xr.DataArray, name: str, coords: dict[str, pd.Index]) -> xr.DataArray:
    """A parameter written as its own values, back on the master coordinates at its own dtype."""
    dims = tuple(str(d) for d in arr.dims)
    return xr.DataArray(
        _values(arr), coords={d: coords[d] for d in dims}, dims=dims, name=name
    )


def _decode(sub: xr.Dataset, name: str, coords: dict[str, pd.Index]) -> xr.DataArray:
    codes = sub[CODES + name]
    dtype = np.dtype(codes.attrs[DTYPE_ATTR])
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
    return xr.DataArray(
        values, dims=dims, attrs={DTYPE_ATTR: dtype or str(values.dtype)}
    )


def _stripped(name: Any, prefix: str) -> str:
    return str(name)[len(prefix) :]


def _values(arr: xr.DataArray) -> np.ndarray:
    """The array as it was in memory, undoing what the netcdf type could not hold."""
    return arr.to_numpy().astype(np.dtype(arr.attrs[DTYPE_ATTR]))


def _index(arr: xr.DataArray) -> pd.Index:
    return pd.Index(_values(arr), name=_stripped(arr.name, COORD))
