"""
v1 semantics helpers.

Single home for the predicates, validators, and storage-invariant
enforcement that the v1 arithmetic convention requires. Importing from
here keeps ``expressions.py`` focused on the operator dispatch and lets a
future legacy removal be a single-file delete.

See ``arithmetics-design/convention.md`` for the rules these helpers
implement and ``arithmetics-design/goals.md`` for the design intent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from warnings import warn

import numpy as np
from xarray import DataArray, Dataset

from linopy.config import (
    LEGACY_SEMANTICS_MESSAGE,
    V1_SEMANTICS,
    LinopySemanticsWarning,
    options,
)
from linopy.constants import HELPER_DIMS

_USER_NAN_MESSAGE = (
    "NaN in a user-supplied constant. Resolve it explicitly with .fillna(...) "
    "or .where(...) before passing it to linopy."
)


def is_v1() -> bool:
    """True iff the current semantics is v1."""
    return options["semantics"] == V1_SEMANTICS


def check_user_nan_scalar() -> None:
    """Enforce §5 for a scalar: v1 raises, legacy warns once."""
    if is_v1():
        raise ValueError(_USER_NAN_MESSAGE)
    warn(LEGACY_SEMANTICS_MESSAGE, LinopySemanticsWarning, stacklevel=4)


def check_user_nan_array() -> None:
    """Enforce §5 for a DataArray operand: v1 raises, legacy warns once."""
    if is_v1():
        raise ValueError(_USER_NAN_MESSAGE)
    warn(LEGACY_SEMANTICS_MESSAGE, LinopySemanticsWarning, stacklevel=4)


def dim_coords_differ(a: DataArray, b: DataArray) -> bool:
    """True if a and b share a dimension whose coordinate labels disagree."""
    for dim in set(a.dims) & set(b.dims):
        if dim in a.coords and dim in b.coords:
            if not a.coords[dim].equals(b.coords[dim]):
                return True
    return False


def merge_shared_user_coords_differ(
    datasets: Sequence[Dataset], concat_dim: str
) -> bool:
    """
    True if the datasets disagree on the labels of any shared user dim.

    Helper dims (``_term``, ``_factor``) and the concat dim itself are
    excluded — those legitimately vary across the operands being merged.
    Compares the bare dimension index (``d.indexes[k]``) so non-dim
    (auxiliary) coords are ignored — those are §11's job.
    """
    skip = set(HELPER_DIMS) | {concat_dim}
    per_ds = [
        {k: d.indexes[k] for k in d.dims if k not in skip and k in d.indexes}
        for d in datasets
    ]
    shared = set.intersection(*(set(p.keys()) for p in per_ds)) if per_ds else set()
    for d_name in shared:
        ref = per_ds[0][d_name]
        for p in per_ds[1:]:
            if not ref.equals(p[d_name]):
                return True
    return False


def conflicting_aux_coord(datasets: Sequence[Any]) -> str | None:
    """
    Return the name of an auxiliary (non-dim) coord that two or more
    operands carry with disagreeing values — None if no conflict.

    Per §11, an auxiliary coord either propagates (values agree across
    operands) or surfaces as an error; xarray's default silently drops
    the conflict and is what this check intercepts under v1.
    """
    if not datasets:
        return None
    all_names: set[str] = set()
    for d in datasets:
        all_names.update(d.coords)
    for name in all_names:
        present = [
            d.coords[name].values
            for d in datasets
            if name in d.coords and name not in d.dims
        ]
        if len(present) < 2:
            continue
        ref = present[0]
        # ``equal_nan`` is only meaningful (and only well-defined) for
        # float dtypes — string/object coord values would crash isnan.
        equal_nan = np.issubdtype(ref.dtype, np.floating)
        for vals in present[1:]:
            if ref.shape != vals.shape or not np.array_equal(
                ref, vals, equal_nan=equal_nan
            ):
                return str(name)
    return None


def absorb_absence(ds: Dataset) -> Dataset:
    """
    Enforce the v1 dead-term invariant on a merged dataset.

    ``const.isnull()`` at a slot ⇒ every term at that slot must have
    ``coeffs = NaN`` and ``vars = -1``. After ``merge`` concatenates two
    expressions along ``_term``, a slot that's absent in one operand
    still carries the *other* operand's valid term in its row; this
    helper masks those away so the §1/§2 storage invariant holds.
    """
    if "const" not in ds or "coeffs" not in ds or "vars" not in ds:
        return ds
    mask = ds["const"].isnull()
    if not bool(mask.any()):
        return ds
    coeffs = ds["coeffs"].where(~mask, np.nan)
    vars_ = ds["vars"].where(~mask, -1)
    return ds.assign(coeffs=coeffs, vars=vars_)
