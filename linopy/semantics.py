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
    V1_SEMANTICS,
    LinopySemanticsWarning,
    options,
)
from linopy.constants import HELPER_DIMS


def _user_nan_message() -> str:
    """User-NaN error text — distinguishes the two intents a user might have."""
    return (
        "NaN found in a user-supplied constant. linopy treats this as "
        "ambiguous: if you meant a *data error*, fix it with .fillna(value); "
        "if you meant *absent at this slot*, mark it on the variable "
        "instead (mask=, .where(cond), .reindex(...), .shift(...))."
    )


def _shared_dim_mismatch_message(dim: str, left: Any, right: Any) -> str:
    """Shared-dim error text — names the dim and shows the disagreeing labels."""
    return (
        f"Coordinate mismatch on shared dimension {dim!r}: "
        f"left={_short_repr(left)}, right={_short_repr(right)}. "
        "Resolve with `.sel(...)` / `.reindex(...)` to align before "
        "combining, with `.assign_coords(...)` to relabel one side "
        "(positional alignment, made explicit), with `linopy.align(...)` "
        "to pre-align several operands at once, or by passing an explicit "
        "`join=` argument to `.add` / `.sub` / `.mul` / `.div` / `.le` / "
        "`.ge` / `.eq` (accepts inner / outer / left / right / override)."
    )


def _aux_conflict_message(name: str, left: Any, right: Any) -> str:
    """Aux-coord error text — names the coord and shows the disagreeing values."""
    return (
        f"Auxiliary coordinate {name!r} has conflicting values across "
        f"operands: left={_short_repr(left)}, right={_short_repr(right)}. "
        "xarray would silently drop the conflict; linopy raises so the "
        f"caller resolves it. Use `.drop_vars({name!r})` to remove the "
        f"coord, `.assign_coords({name}=...)` to relabel one side, or "
        "`.isel(..., drop=True)` if the coord was introduced by a "
        "scalar isel."
    )


# ---------------------------------------------------------------------------
# Legacy-deprecation warnings — actionable, per-site (goal #2 in goals.md:
# tell the user *what* will change for the op they just ran).
# Each helper returns the message; ``warn_legacy(msg)`` issues it.
# ---------------------------------------------------------------------------


_OPT_IN_HINT = (
    "\n  Opt in:    linopy.options['semantics'] = 'v1'"
    "\n  Silence:   warnings.filterwarnings('ignore', "
    "category=LinopySemanticsWarning)"
)


def _legacy_nan_constant_add_message() -> str:
    """``+`` / ``-`` legacy fill-with-0 for NaN constants."""
    return (
        "NaN in the constant operand was silently treated as 0 by legacy"
        " (additive identity). Under v1 this raises ValueError."
        "\n  Resolve:   `.fillna(value)` (data error)"
        "\n             or `mask=` / `.where(cond)` / `.reindex(...)` "
        "on the variable (intended absence)." + _OPT_IN_HINT
    )


def _legacy_nan_constant_mul_message() -> str:
    """``*`` legacy fill-with-0 for NaN constants."""
    return (
        "NaN in the multiplicative factor was silently treated as 0 by"
        " legacy (so the variable was zeroed out at that slot)."
        " Under v1 this raises ValueError."
        "\n  Resolve:   `.fillna(value)` (data error)"
        "\n             or `mask=` / `.where(cond)` / `.reindex(...)` "
        "on the variable (intended absence)." + _OPT_IN_HINT
    )


def _legacy_nan_constant_div_message() -> str:
    """``/`` legacy fill-with-1 for NaN constants."""
    return (
        "NaN in the divisor was silently treated as 1 by legacy (a"
        " different fill from `+`/`*` which use 0). Under v1 this raises"
        " ValueError."
        "\n  Resolve:   `.fillna(value)` (data error)"
        "\n             or `mask=` / `.where(cond)` / `.reindex(...)` "
        "on the variable (intended absence)." + _OPT_IN_HINT
    )


def _legacy_coord_mismatch_message(context: str) -> str:
    """Mismatched dim coords silently aligned (positional or left-join)."""
    return (
        f"Coordinate mismatch in {context} silently aligned by legacy"
        " (positional when sizes match, otherwise left-join)."
        " Under v1 this raises ValueError."
        "\n  Resolve:   `.sel(...)` / `.reindex(...)` to align"
        "\n             `.assign_coords(...)` to relabel one side"
        "\n             `linopy.align(...)` to pre-align several operands"
        "\n             or pass an explicit `join=` argument." + _OPT_IN_HINT
    )


def _legacy_aux_conflict_message(name: str) -> str:
    """Conflicting aux coord silently dropped by xarray under legacy."""
    return (
        f"Auxiliary coordinate {name!r} was conflicting across operands"
        " and silently dropped by legacy (xarray's default)."
        " Under v1 this raises ValueError."
        f"\n  Resolve:   `.drop_vars({name!r})`"
        f"\n             `.assign_coords({name}=...)` to relabel one side"
        "\n             or `.isel(..., drop=True)` if a scalar isel "
        "introduced it." + _OPT_IN_HINT
    )


def _legacy_nan_rhs_constraint_message() -> str:
    """Constraint RHS NaN silently kept as 'no constraint at this row'."""
    return (
        "NaN in the constraint RHS was silently kept as 'no constraint"
        " at this row' by legacy auto-mask. Under v1 this raises"
        " ValueError."
        "\n  Resolve:   `mask=` on the variable for explicit per-row "
        "masking"
        "\n             or `.fillna(value)` if the NaN was a data error." + _OPT_IN_HINT
    )


def _legacy_masked_variable_message(name: str) -> str:
    """A masked/shifted/reindexed variable used in arithmetic under legacy."""
    return (
        f"Variable {name!r} has absent slots (from `mask=` / `.where()`"
        " / `.shift()` / `.reindex()`). Under legacy each absent slot"
        " contributes 0 to the resulting expression's terms (so `x + y"
        " >= 10` reduces to `x >= 10` there). Under v1 the absence"
        " propagates through arithmetic instead (`x + y` becomes absent"
        " at the slot and the constraint drops)."
        f"\n  Resolve:   wrap with `{name}.fillna(0)` for the legacy"
        " behaviour under v1"
        "\n             (no fix needed if you only use the variable in a"
        " constraint LHS alone — `y >= 0` drops the same way in both)." + _OPT_IN_HINT
    )


def warn_legacy(message: str, *, stacklevel: int = 3) -> None:
    """Emit a `LinopySemanticsWarning` with the given site-specific message."""
    warn(message, LinopySemanticsWarning, stacklevel=stacklevel)


def _short_repr(values: Any, limit: int = 6) -> str:
    """Render an array-like as a short, readable string for error messages."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        return repr(arr.item())
    flat = arr.ravel()
    if flat.size <= limit:
        return repr(flat.tolist())
    head = ", ".join(repr(v) for v in flat[:limit].tolist())
    return f"[{head}, ... ({flat.size} total)]"


def is_v1() -> bool:
    """True iff the current semantics is v1."""
    return options["semantics"] == V1_SEMANTICS


def check_user_nan_scalar(*, op_kind: str = "add") -> None:
    """
    Enforce §5 for a scalar: v1 raises, legacy warns with op-specific text.

    ``op_kind`` is one of ``"add"`` (covers +/-), ``"mul"``, ``"div"``.
    """
    if is_v1():
        raise ValueError(_user_nan_message())
    warn_legacy(_legacy_message_for_op(op_kind), stacklevel=5)


def check_user_nan_array(*, op_kind: str = "add") -> None:
    """Enforce §5 for a DataArray operand: v1 raises, legacy warns once."""
    if is_v1():
        raise ValueError(_user_nan_message())
    warn_legacy(_legacy_message_for_op(op_kind), stacklevel=5)


def _legacy_message_for_op(op_kind: str) -> str:
    """Pick the per-operator legacy NaN-fill message."""
    return {
        "add": _legacy_nan_constant_add_message,
        "mul": _legacy_nan_constant_mul_message,
        "div": _legacy_nan_constant_div_message,
    }[op_kind]()


def dim_coords_differ(a: DataArray, b: DataArray) -> bool:
    """True if a and b share a dimension whose coordinate labels disagree."""
    for dim in set(a.dims) & set(b.dims):
        if dim in a.coords and dim in b.coords:
            if not a.coords[dim].equals(b.coords[dim]):
                return True
    return False


def merge_shared_user_coord_mismatch(
    datasets: Sequence[Dataset], concat_dim: str
) -> tuple[str, Any, Any] | None:
    """
    Find a shared user dim where the operands' labels disagree.

    Returns ``(dim_name, left_labels, right_labels)`` for the first
    mismatch found, or ``None`` if all operands agree. Helper dims
    (``_term``, ``_factor``) and the concat dim itself are excluded —
    those legitimately vary across the operands being merged. Compares
    bare dimension indexes (``d.indexes[k]``) so non-dim (auxiliary)
    coords are ignored — those are §11's job.
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
                return str(d_name), ref.values, p[d_name].values
    return None


def conflicting_aux_coord(
    datasets: Sequence[Any],
) -> tuple[str, Any, Any] | None:
    """
    Find an auxiliary (non-dim) coord that two or more operands carry with
    disagreeing values.

    Returns ``(name, left_values, right_values)`` for the first conflict
    found, or ``None`` if every shared aux coord agrees. Per §11, an aux
    coord either propagates (values agree across operands) or surfaces as
    an error; xarray's default silently drops the conflict and is what
    this check intercepts under v1.
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
                return str(name), ref, vals
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
