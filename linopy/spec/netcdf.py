"""
Persist the spec of a spec-built model in its netcdf file.

Variables, constraints and the solution round trip through :mod:`linopy.io`
already. Besides them a spec-built model carries the spec text, the master
coordinates, the relations and the parameters, a missing row a NaN hole;
the program is re-lowered from the text on read, so no lowered
``Program`` ever reaches the file. The dataset names the mathspec version
that lowered the text, and a read under another version warns.

No netcdf type holds a dtype as written, so every array carries the dtype it
had in memory and is cast back to it on read. That is enough for a
parameter, but not for a partial relation, which holds NaN in an array of
labels: a hole in a string array comes back as an empty string,
indistinguishable from a label. So a relation, and any array of objects, is
written instead as integer codes into its own table of categories, ``-1``
where a label is missing. Decoding indexes the table and fills the holes
back in, which reproduces what attach built, values and dtype alike.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from mathspec import __version__ as MATHSPEC_VERSION

from linopy.model import Model
from linopy.spec.accessor import ModelSpec, lower

TEXT_ATTR = "text"
NAME_ATTR = "name"
VERSION_ATTR = "mathspec"
DTYPE_ATTR = "_linopy_dtype"
COORD = "coords__"
PARAM = "param__"
CODES = "codes__"
CATEGORIES = "cats__"
CATEGORY_DIM = "category__"

HOLES: dict[str, Any] = {"f": np.nan, "O": np.nan, "M": np.datetime64("NaT")}


def record_dtype(arr: xr.DataArray, dtype: str | None = None) -> xr.DataArray:
    """*arr* with its in-memory dtype, or the *dtype* it stands for, written as an attribute; a bool array, which no netcdf type holds, as int8."""
    recorded = arr.assign_attrs({DTYPE_ATTR: dtype or str(arr.dtype)})
    return recorded.astype(np.int8) if arr.dtype == bool else recorded


def restore_dtype(arr: xr.DataArray) -> xr.DataArray:
    """*arr* back at the dtype :func:`record_dtype` recorded, the attribute consumed."""
    return arr.astype(np.dtype(arr.attrs.pop(DTYPE_ATTR)))


def encode(spec: ModelSpec) -> xr.Dataset:
    """
    The spec's own dataset: its text, its name, its master coordinates and its parameters.

    The spec text, the mathspec version and the name are the dataset's
    attributes, which the merge lifts to the file's. Beside them sits one
    array of labels per master coordinate and, per parameter, either its
    values or, where it is coded, its codes and its categories. The dataset
    carries no coordinates of its own: a parameter is written over bare
    dimensions and put back on the master coordinates on read.
    """
    arrays: dict[str, xr.DataArray] = {
        COORD + dim: record_dtype(xr.DataArray(index.to_numpy(), dims=(dim,)))
        for dim, index in spec.coords.items()
    }
    for name, arr in spec.parameters.items():
        if name in spec.program.relations or arr.dtype == object:
            arrays.update(_encode(str(name), arr))
        else:
            arrays[PARAM + str(name)] = record_dtype(_bare(arr))
    attrs = {
        TEXT_ATTR: spec.text,
        VERSION_ATTR: MATHSPEC_VERSION,
        NAME_ATTR: spec.name,
    }
    return xr.Dataset(arrays, attrs=attrs)


def decode(model: Model, sub: xr.Dataset) -> ModelSpec:
    """
    Re-lower the text onto *model* and read back the dataset :func:`encode` wrote, its prefix already removed.

    The master coordinates, the plainly written parameters and the coded
    ones together are what :func:`linopy.spec.attach`
    gave the spec when the model was built. ``model.parameters`` is not
    touched: it holds what the caller put there and nothing of the spec.

    Warns
    -----
    UserWarning
        The file names another mathspec version than this reader's.
    """
    if sub.attrs[VERSION_ATTR] != MATHSPEC_VERSION:
        warnings.warn(
            f"the file's spec was lowered by mathspec {sub.attrs[VERSION_ATTR]} and is "
            f"re-lowered by {MATHSPEC_VERSION}; the same text may lower differently.",
            UserWarning,
            stacklevel=4,
        )
    coords = {
        dim: pd.Index(restore_dtype(arr).to_numpy(), name=dim)
        for dim, arr in _group(sub, COORD).items()
    }
    arrays = {
        name: _plain(arr, name, coords) for name, arr in _group(sub, PARAM).items()
    }
    arrays |= {name: _decode(sub, name, coords) for name in _group(sub, CODES)}
    text = str(sub.attrs[TEXT_ATTR])
    parameters = xr.Dataset(arrays).assign_coords(coords)
    return ModelSpec(model, str(sub.attrs[NAME_ATTR]), lower(text), text, parameters)


def _group(sub: xr.Dataset, prefix: str) -> dict[str, xr.DataArray]:
    """The arrays of *sub* written under *prefix*, by their names without it."""
    return {
        str(name)[len(prefix) :]: sub[name]
        for name in sub.data_vars
        if str(name).startswith(prefix)
    }


def _bare(arr: xr.DataArray) -> xr.DataArray:
    return xr.DataArray(arr.to_numpy(), dims=arr.dims)


def _encode(name: str, arr: xr.DataArray) -> dict[str, xr.DataArray]:
    codes, categories = pd.factorize(arr.to_numpy().ravel())
    coded = xr.DataArray(codes.astype(np.int32).reshape(arr.shape), dims=arr.dims)
    written = {CODES + name: record_dtype(coded, str(arr.dtype))}
    if len(categories):
        table = xr.DataArray(np.asarray(categories), dims=(CATEGORY_DIM + name,))
        written[CATEGORIES + name] = record_dtype(table)
    return written


def _plain(arr: xr.DataArray, name: str, coords: dict[str, pd.Index]) -> xr.DataArray:
    """An array written as its own values, back on the master coordinates at its own dtype."""
    return _on_master(restore_dtype(arr).to_numpy(), arr.dims, name, coords)


def _decode(sub: xr.Dataset, name: str, coords: dict[str, pd.Index]) -> xr.DataArray:
    """
    A coded array, its codes looked up in its table of categories.

    A map that leaves every label unmapped has no table: netCDF3 writes a
    zero-length dimension as the unlimited one, of which a file holds one.
    """
    codes = sub[CODES + name]
    dtype = np.dtype(codes.attrs[DTYPE_ATTR])
    table = CATEGORIES + name
    if table in sub.data_vars:
        categories = restore_dtype(sub[table]).to_numpy()
    else:
        categories = np.empty(0, dtype=dtype)
    positions = codes.to_numpy().astype(int)
    mapped = positions >= 0
    if mapped.all():
        values = categories[positions]
    else:
        values = np.full(positions.shape, HOLES[dtype.kind], dtype=dtype)
        values[mapped] = categories[positions[mapped]]
    return _on_master(values, codes.dims, name, coords)


def _on_master(
    values: np.ndarray,
    dims: tuple[Hashable, ...],
    name: str,
    coords: dict[str, pd.Index],
) -> xr.DataArray:
    """*values* over *dims*, put back on the master coordinates."""
    framed = tuple(str(d) for d in dims)
    return xr.DataArray(
        values, coords={d: coords[d] for d in framed}, dims=framed, name=name
    )
