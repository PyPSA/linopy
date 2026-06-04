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

import os
import sys
from collections.abc import Sequence
from typing import Any
from warnings import warn

import numpy as np
import pandas as pd
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


def _aux_conflict_message(name: str, left: Any, right: Any, kind: str) -> str:
    """
    Aux-coord error text — names the coord, the failure mode (shape vs
    value), and shows the disagreeing values.
    """
    if kind == "shape":
        problem = (
            f"Auxiliary coordinate {name!r} has differing shapes across "
            f"operands: left.shape={np.shape(left)}, "
            f"right.shape={np.shape(right)}. "
        )
    else:
        problem = (
            f"Auxiliary coordinate {name!r} has conflicting values across "
            f"operands: left={_short_repr(left)}, right={_short_repr(right)}. "
        )
    return (
        problem + "xarray would silently drop the conflict; linopy raises so the "
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


# Per-op opening clause for ``_legacy_nan_constant_message`` — operand
# noun and the historical fill value (`+`/`*` filled with 0; `/` filled
# with 1, a different fill that's worth calling out at the warn site).
_LEGACY_NAN_FILL_CLAUSE = {
    "add": (
        "NaN in the constant operand was silently treated as 0 by legacy"
        " (additive identity)."
    ),
    "mul": (
        "NaN in the multiplicative factor was silently treated as 0 by"
        " legacy (so the variable was zeroed out at that slot)."
    ),
    "div": (
        "NaN in the divisor was silently treated as 1 by legacy (a"
        " different fill from `+`/`*` which use 0)."
    ),
}


def _legacy_nan_constant_message(op_kind: str) -> str:
    """Legacy NaN-fill warning for `+`/`*`/`/`, keyed by ``op_kind``."""
    return (
        _LEGACY_NAN_FILL_CLAUSE[op_kind] + " Under v1 this raises ValueError."
        "\n  Resolve:   `.fillna(value)` (data error)"
        "\n             or `mask=` / `.where(cond)` / `.reindex(...)` "
        "on the variable (intended absence)." + _OPT_IN_HINT
    )


def _legacy_coord_mismatch_message(
    context: str,
    dim: str | None = None,
    left: Any = None,
    right: Any = None,
) -> str:
    """
    Mismatched dim coords silently aligned (positional or left-join).

    When ``dim`` / ``left`` / ``right`` are given, the message names the
    offending dim and shows the diff — same shape as the v1-raise text
    so the user sees the same information at warn time as at raise time.
    """
    diff = (
        f"\n  Dim:       {dim!r}: left={_short_repr(left)}, right={_short_repr(right)}"
        if dim is not None
        else ""
    )
    return (
        f"Coordinate mismatch in {context} silently aligned by legacy"
        " (positional when sizes match, otherwise left-join)."
        " Under v1 this raises ValueError."
        + diff
        + "\n  Resolve:   `.sel(...)` / `.reindex(...)` to align"
        "\n             `.assign_coords(...)` to relabel one side"
        "\n             `linopy.align(...)` to pre-align several operands"
        "\n             or pass an explicit `join=` argument." + _OPT_IN_HINT
    )


def _legacy_coord_reorder_message(context: str, dim: str, left: Any, right: Any) -> str:
    """Same labels, different order — aligned positionally by legacy; v1 reindexes."""
    return (
        f"Coordinate order mismatch in {context} aligned positionally by legacy."
        " Under v1 the same labels in a different order align by label (a"
        " reindex), giving a different result."
        f"\n  Dim:       {dim!r}: left={_short_repr(left)}, right={_short_repr(right)}"
        "\n  Resolve:   `.sel(...)` / `.reindex(...)` to align"
        "\n             `.assign_coords(...)` to relabel one side"
        "\n             or pass an explicit `join=` argument." + _OPT_IN_HINT
    )


def _legacy_aux_conflict_message(name: str, left: Any, right: Any, kind: str) -> str:
    """
    Conflicting aux coord silently dropped by xarray under legacy.

    The diff line names the failure mode (shape vs value) — same shape
    as the v1-raise text so the user sees the same information at warn
    time as at raise time.
    """
    if kind == "shape":
        diff = f"\n  Shapes:    left={np.shape(left)}, right={np.shape(right)}"
    else:
        diff = f"\n  Values:    left={_short_repr(left)}, right={_short_repr(right)}"
    return (
        f"Auxiliary coordinate {name!r} was conflicting across operands"
        " and silently dropped by legacy (xarray's default)."
        " Under v1 this raises ValueError."
        + diff
        + f"\n  Resolve:   `.drop_vars({name!r})`"
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


_LINOPY_ROOT = os.path.dirname(os.path.abspath(__file__))


def warn_legacy(message: str, *, stacklevel: int | None = None) -> None:
    """
    Emit a `LinopySemanticsWarning` whose source-frame points at the
    first call-stack frame *outside* the linopy package.

    Static ``stacklevel`` doesn't fit here — the call-chain depth from
    ``warn_legacy`` to the user's code varies per site (e.g. masked-var
    via ``__add__`` is 5 frames deep, via ``Variable.fillna`` is 4). On
    Python 3.12+ we use the stdlib ``skip_file_prefixes`` argument
    (implemented and tested in CPython); on 3.11 we fall back to a
    static ``stacklevel=5``, good enough for the common merge chain.
    Pass an explicit ``stacklevel`` to override (e.g. for tests).
    """
    if stacklevel is not None:
        warn(message, LinopySemanticsWarning, stacklevel=stacklevel)
    elif sys.version_info >= (3, 12):
        warn(
            message,
            LinopySemanticsWarning,
            skip_file_prefixes=(_LINOPY_ROOT,),
        )
    else:
        warn(message, LinopySemanticsWarning, stacklevel=5)


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


def check_user_nan(*, op_kind: str = "add") -> None:
    """
    Enforce §5 for a user-supplied constant (scalar or array).

    v1 raises ``ValueError`` with the generic user-NaN message; legacy
    warns with operator-specific text (``"add"`` covers +/-, ``"mul"``,
    ``"div"`` — they differ in which fill value legacy applied).
    """
    if is_v1():
        raise ValueError(_user_nan_message())
    warn_legacy(_legacy_nan_constant_message(op_kind), stacklevel=5)


def enforce_aux_conflict(datasets: Sequence[Any], *, stacklevel: int = 5) -> None:
    """
    Enforce §11 across the given operands: v1 raises on aux-coord
    conflict, legacy warns (xarray would silently drop it).
    """
    conflict = conflicting_aux_coord(datasets)
    if conflict is None:
        return
    if is_v1():
        raise ValueError(_aux_conflict_message(*conflict))
    warn_legacy(_legacy_aux_conflict_message(*conflict), stacklevel=stacklevel)


def dim_coords_differ(a: DataArray, b: DataArray) -> bool:
    """True if a and b share a dimension whose coordinate labels disagree."""
    return first_mismatched_dim(a, b) is not None


def first_mismatched_dim(a: DataArray, b: DataArray) -> tuple[str, Any, Any] | None:
    """
    Return ``(dim, a_labels, b_labels)`` for the first shared dim that
    disagrees on coordinate labels OR size, or ``None`` if all agree.

    Uses ``indexes[dim]`` (the bare pandas Index) rather than
    ``coords[dim]`` — a coord DataArray's ``equals`` compares attached
    aux coords too, which gives a false positive when only one operand
    carries an aux coord on the shared dim (§11's territory, not §8's).
    """
    for dim in set(a.dims) & set(b.dims):
        if dim in a.indexes and dim in b.indexes:
            if not a.indexes[dim].equals(b.indexes[dim]):
                return str(dim), a.indexes[dim].values, b.indexes[dim].values
        elif a.sizes[dim] != b.sizes[dim]:
            return str(dim), None, None
    return None


def conform_merge_dims(
    datasets: Sequence[Dataset], concat_dim: str
) -> tuple[list[Dataset], tuple[str, Any, Any] | None, tuple[str, Any, Any] | None]:
    """
    Inspect shared user dims for a merge, in a single pass over the operands.

    Returns ``(data, mismatch, reorder)``. A shared user dim whose labels are
    the first operand's in a different order (same set, including a stacked
    MultiIndex's tuples) is a *reorder*; one whose label set differs is a
    *mismatch* — each reported as ``(dim, first_labels, other_labels)`` (first
    found). Under v1, reorders are aligned to the first operand's order in the
    returned data (via positional ``isel`` — ``reindex`` cannot reorder a
    MultiIndex by tuple) and ``reorder`` is ``None``; the caller raises on
    ``mismatch``. Under legacy, nothing is aligned and the caller warns:
    ``reorder`` (v1 would align by label) or ``mismatch`` (v1 would raise).
    Helper dims (``_term``, ``_factor``) and the concat dim are excluded; bare
    dimension indexes are compared, so auxiliary coords stay §11's job.
    """
    datasets = list(datasets)
    if len(datasets) < 2:
        return datasets, None, None
    skip = set(HELPER_DIMS) | {concat_dim}
    indexed = [
        {k: d.indexes[k] for k in d.dims if k not in skip and k in d.indexes}
        for d in datasets
    ]
    shared = set.intersection(*(set(p) for p in indexed))
    if not shared:
        return datasets, None, None

    permute = is_v1()
    out = [datasets[0]]
    mismatch: tuple[str, Any, Any] | None = None
    reorder: tuple[str, Any, Any] | None = None
    for i in range(1, len(datasets)):
        plan: dict[Any, Any] = {}
        for d in shared:
            ref, idx = indexed[0][d], indexed[i][d]
            if ref.equals(idx):
                continue
            positions = idx.get_indexer(ref) if len(idx) == len(ref) else None
            if positions is not None and (positions >= 0).all():
                if permute:
                    plan[d] = positions
                elif reorder is None:
                    reorder = (str(d), ref.values, idx.values)
            elif mismatch is None:
                mismatch = (str(d), ref.values, idx.values)
        out.append(datasets[i].isel(plan) if plan else datasets[i])
    return out, mismatch, reorder


def conflicting_aux_coord(
    datasets: Sequence[Any],
) -> tuple[str, Any, Any, str] | None:
    """
    Find an auxiliary (non-dim) coord that two or more operands carry with
    disagreeing values.

    Returns ``(name, left_values, right_values, kind)`` for the first
    conflict found, or ``None`` if every shared aux coord agrees. ``kind``
    is ``"shape"`` if the two operands carry differently-shaped values for
    the coord (e.g. one is a vector, the other a scalar), or ``"value"``
    if shapes agree but the values themselves disagree. The two failure
    modes get different error text downstream.

    Per §11, an aux coord either propagates (values agree across operands)
    or surfaces as an error; xarray's default silently drops the conflict
    and is what this check intercepts under v1. When only one operand
    carries the coord (``len(present) < 2``), it propagates from that
    operand unchanged.
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
        # §11 asymmetric-presence: when only one operand carries the coord,
        # it propagates unchanged — no conflict to surface.
        if len(present) < 2:
            continue
        ref = present[0]
        for vals in present[1:]:
            if ref.shape != vals.shape:
                return str(name), ref, vals, "shape"
            if not _aux_values_equal(ref, vals):
                return str(name), ref, vals, "value"
    return None


def _aux_values_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Equality for aux-coord value arrays with NaN-equal-NaN semantics
    on every dtype.

    ``np.array_equal(..., equal_nan=True)`` only works on float dtypes
    (it calls ``isnan`` which crashes on object/string). Aux coords on
    object dtype can embed ``np.nan`` placeholders (e.g. ragged category
    labels), and we want two operands with identical NaN placement to
    compare equal — pandas' element-equality already treats NaN as
    self-equal for object dtypes, so route through ``pd.Series.equals``.
    """
    if np.issubdtype(a.dtype, np.floating):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(pd.Series(a.ravel()).equals(pd.Series(b.ravel())))


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
