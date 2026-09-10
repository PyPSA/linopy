"""
Persist the spec layers of a model in its netcdf file.

Variables, constraints and the solution round trip through :mod:`linopy.io`
already. Besides them each spec layer carries its text, the names it binds,
its master coordinates and its lookups; the program is re-lowered from the
text on read, so no lowered ``Program`` ever reaches the file. The layer
order, whether the layers describe the whole model and which layer owns the
objective are attributes of the file itself.

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

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from linopy.io import (
    DTYPE_ATTR,
    LAYER_BOUND_ATTR,
    LAYER_TEXT_ATTR,
    SPEC_ATTR,
    SPEC_LAYERS_ATTR,
    SPEC_OBJECTIVE_ATTR,
    SPEC_WHOLE_ATTR,
    get_prefix,
    restamp_coords,
    with_prefix,
)
from linopy.model import Model
from linopy.spec.accessor import Layer, ModelSpec, restore, restore_layer

PREFIX = "spec"
LEGACY_NAME = "spec"
LEGACY_OBJECTIVE_ATTR = "_linopy_spec_objective_replaced"
COORD = "coords__"
PARAM = "param__"
CODES = "codes__"
CATEGORIES = "cats__"
CATEGORY_DIM = "category__"

HOLES: dict[str, Any] = {"f": np.nan, "O": np.nan, "M": np.datetime64("NaT")}


def encode(layer: Layer) -> xr.Dataset:
    """
    The layer's own dataset: its master coordinates and its parameters, its text and bindings as attributes.

    Everything is written under the prefix ``spec-<name>``, and the two
    attributes carry the layer's name too, so the merge of several layers
    lifts every one of them to the file's. Beside them sits one array of
    labels per master coordinate and, per parameter, either its values or --
    where it is coded -- its codes and its categories. The dataset carries no
    coordinates of its own: an index coordinate is dropped on read together
    with the dimension it indexes once no data variable is left over that
    dimension, and a master coordinate nothing else reaches has exactly that
    shape. So a parameter is written over bare dimensions and put back on the
    master coordinates on read.

    Raises
    ------
    ValueError
        An array name holds a ``-``: the prefix is split off at the last
        one on read, so such a name would be silently dropped.
    """
    arrays: dict[str, xr.DataArray] = {
        COORD + dim: _array(index.to_numpy(), (dim,))
        for dim, index in layer.coords.items()
    }
    coded = _coded(layer)
    for name, arr in layer.parameters.items():
        if str(name) in coded:
            arrays.update(_encode(str(name), arr))
        else:
            arrays[PARAM + str(name)] = _array(arr.to_numpy(), arr.dims, str(arr.dtype))
    dashed = sorted(name for name in arrays if "-" in name)
    if dashed:
        raise ValueError(
            f"spec layer '{layer.name}' would write arrays {dashed}, and a netcdf name "
            f"is split from its prefix at the last '-'. A dimension or parameter name "
            f"cannot hold one."
        )
    written = with_prefix(xr.Dataset(arrays), f"{PREFIX}-{layer.name}")
    return written.assign_attrs(
        {
            LAYER_TEXT_ATTR.format(layer.name): layer.text,
            LAYER_BOUND_ATTR.format(layer.name): json.dumps(dict(layer.names)),
        }
    )


def read(model: Model, ds: xr.Dataset) -> ModelSpec:
    """
    The spec layers a file holds, restored onto *model* in their order.

    A file written before layers existed holds one spec under the bare
    ``spec`` prefix and its text in one attribute; it reads as a single layer
    named ``"spec"`` that describes the whole model.
    """
    if SPEC_LAYERS_ATTR in ds.attrs:
        layers = [
            decode(
                model,
                get_prefix(ds, f"{PREFIX}-{name}"),
                name,
                ds.attrs[LAYER_TEXT_ATTR.format(name)],
                json.loads(ds.attrs[LAYER_BOUND_ATTR.format(name)]),
            )
            for name in json.loads(ds.attrs[SPEC_LAYERS_ATTR])
        ]
        whole = bool(ds.attrs[SPEC_WHOLE_ATTR])
        return restore(model, layers, whole, json.loads(ds.attrs[SPEC_OBJECTIVE_ATTR]))
    layer = decode(model, get_prefix(ds, PREFIX), LEGACY_NAME, ds.attrs[SPEC_ATTR], {})
    replaced = bool(ds.attrs.get(LEGACY_OBJECTIVE_ATTR, 0))
    owner = (
        LEGACY_NAME if layer.program.objective is not None and not replaced else None
    )
    return restore(model, [layer], whole=True, objective_owner=owner)


def decode(
    model: Model, sub: xr.Dataset, name: str, text: str, names: Mapping[str, str]
) -> Layer:
    """
    Re-lower *text* onto *model* as the layer *name* and read back the dataset :func:`encode` wrote.

    *sub* is the layer's part of the file with its prefix given back. The
    master coordinates, the plainly written parameters and the coded ones
    together are the dataset :func:`linopy.spec.accessor.attach` gave the
    layer when it was built. ``model.parameters`` is not touched: it holds
    what the caller put there and nothing of the spec.
    """
    coords = {
        _stripped(var, COORD): _index(sub[var])
        for var in sub.data_vars
        if str(var).startswith(COORD)
    }
    arrays = {
        _stripped(var, PARAM): _plain(sub[var], _stripped(var, PARAM), coords)
        for var in sub.data_vars
        if str(var).startswith(PARAM)
    }
    arrays.update(
        {
            _stripped(var, CODES): _decode(sub, _stripped(var, CODES), coords)
            for var in sub.data_vars
            if str(var).startswith(CODES)
        }
    )
    restamp_coords(model, coords)
    parameters = xr.Dataset(arrays).assign_coords(coords)
    return restore_layer(model, name, text, parameters, names)


def _coded(layer: Layer) -> set[str]:
    """The parameters written as codes: every lookup and every array of objects."""
    lookups = {name for by_name in layer.lookups.values() for name in by_name}
    return {
        str(name)
        for name, arr in layer.parameters.items()
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
