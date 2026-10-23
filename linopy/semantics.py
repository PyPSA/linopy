"""
Arithmetic convention helpers.

Single home for the predicates, validators, and storage-invariant
enforcement that the arithmetic convention requires. Importing from
here keeps ``expressions.py`` focused on the operator dispatch.

See ``doc/design/convention.rst`` for the rules these helpers
implement and ``doc/design/goals.rst`` for the design intent.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any, NoReturn, TypeAlias
from warnings import warn

import numpy as np
import pandas as pd
from xarray import DataArray, Dataset

from linopy.constants import HELPER_DIMS


def _user_nan_message(context: str = "constant") -> str:
    """User-NaN error text — distinguishes the two intents a user might have."""
    return (
        f"NaN found in a user-supplied {context}. linopy treats this as "
        "ambiguous: if you meant a *data error*, fix it with .fillna(value); "
        "if you meant *absent at this slot*, mark it on the variable "
        "instead (mask=, .where(cond), .reindex(...), .shift(...))."
    )


def _shared_dim_mismatch_message(
    dim: str, left: Any, right: Any, *, joinable: bool = True
) -> str:
    """
    Shared-dim error text — names the dim and shows the disagreeing labels.
    ``joinable=False`` drops the ``join=`` hint for operations that reject it.
    """
    join_hint = (
        ", or by passing an explicit `join=` argument to `.add` / `.sub` /"
        " `.mul` / `.div` / `.le` / `.ge` / `.eq` (accepts inner / outer /"
        " left / right / override)."
        if joinable
        else "."
    )
    return (
        f"Coordinate mismatch on shared dimension {dim!r}: "
        f"left={_short_repr(left)}, right={_short_repr(right)}. "
        "Resolve with `.sel(...)` / `.reindex(...)` to align before "
        "combining, with `.assign_coords(...)` to relabel one side "
        "(positional alignment, made explicit), with `linopy.align(...)` "
        "to pre-align several operands at once" + join_hint
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


_RAGGED_RESOLVE_HINT = (
    "\n  Ragged:    declare which slots hold a breakpoint —"
    "\n             `add_piecewise_formulation(..., mask=x_pts.notnull())`"
    "\n  Data error: `.fillna(value)`"
)


def _user_nan_breakpoint_message(context: str) -> str:
    """
    User-NaN error text for breakpoint tables.

    The generic §5 message points at `mask=` on a *variable*, which is the
    wrong remedy here — breakpoints are plain data, and raggedness is the
    reason a NaN usually shows up. Name that case instead.
    """
    return (
        f"NaN found in user-supplied {context} (§5). linopy trusts NaN only "
        "from its own structural operations (§4): here it cannot tell a "
        "shorter curve (a padded slot) from a data error, and guessing would "
        "silently change the model." + _RAGGED_RESOLVE_HINT
    )


def check_user_nan_breakpoints(*, context: str = "breakpoints") -> NoReturn:
    """Enforce §5 for a breakpoint table whose padding was not declared."""
    raise ValueError(_user_nan_breakpoint_message(context))


def _multiindex_message(dim: str, context: str) -> str:
    """A pandas MultiIndex dim-coord, rejected."""
    return (
        f"{context} uses dimension {dim!r}, a `pd.MultiIndex`, which the convention"
        f" does not support. Decompose it to a flat dim with the levels as auxiliary"
        f" coords first — e.g. `reset_index({dim!r})`."
    )


_LINOPY_ROOT = os.path.dirname(os.path.abspath(__file__))


def warn_outside_linopy(message: str, category: type[Warning]) -> None:
    """
    Emit a warning whose source-frame points at the first call-stack frame
    *outside* the linopy package, as the call-chain depth varies per site.
    Python 3.12+ uses the stdlib ``skip_file_prefixes``; older versions walk
    the stack to compute the ``stacklevel``.
    """
    if sys.version_info >= (3, 12):
        warn(message, category, skip_file_prefixes=(_LINOPY_ROOT,))
        return
    frame = sys._getframe()
    stacklevel = 1
    while frame.f_back is not None and frame.f_code.co_filename.startswith(
        _LINOPY_ROOT
    ):
        frame = frame.f_back
        stacklevel += 1
    warn(message, category, stacklevel=stacklevel)


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


def is_nan_scalar(value: Any) -> bool:
    """
    True for a scalar float NaN of any dtype — Python ``float`` or any
    ``np.floating`` (``np.float32`` / ``np.float16`` included).

    ``np.float64`` subclasses ``float``, but ``np.float32`` does not, so an
    ``isinstance(value, float)`` guard alone lets a ``np.float32('nan')`` slip
    past the §5 user-NaN check on a scalar fast path. The float-type guard is
    still required to keep ``np.isnan`` from raising on non-numeric scalars.
    """
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def check_user_nan(context: str = "constant") -> NoReturn:
    """Enforce §5 for a user-supplied constant (scalar or array): raise."""
    raise ValueError(_user_nan_message(context))


class AbsentType:
    """
    Sentinel for ``fill_value=``: create the label, leave the slot absent.

    A reindexing join otherwise fills what it creates (§10). Passing
    ``linopy.ABSENT`` instead keeps the join's coordinates but makes every
    created position absent, so §6 absorbs it and §12 drops the row. It is a
    dedicated object rather than ``np.nan`` because a user-supplied NaN raises
    under §5, and rather than ``None`` because that already means "the
    operator's default fill".
    """

    def __repr__(self) -> str:
        return "linopy.ABSENT"

    def __neg__(self) -> AbsentType:
        return self


ABSENT = AbsentType()

FillValueLike: TypeAlias = float | AbsentType | None


def join_fill(fill_value: FillValueLike, default: float) -> float:
    """Numeric fill a join applies — ``ABSENT`` fills with NaN, ``None`` with ``default``."""
    if fill_value is None:
        return default
    if isinstance(fill_value, AbsentType):
        return np.nan
    return fill_value


def check_join_fill_value(fill_value: FillValueLike, join: str | None) -> None:
    """
    ``fill_value=`` fills what a join creates, so it requires an explicit join.

    A NaN fill is rejected for the §5 reason a NaN constant is: absence has a
    name of its own (``ABSENT``).
    """
    if fill_value is None:
        return
    if join is None:
        raise ValueError(
            "fill_value= fills the positions a join creates, so it requires an "
            "explicit join= (e.g. join='outer'). To fill absent slots of an "
            "operand, call `.fillna(...)` on it before the operation."
        )
    if is_nan_scalar(fill_value):
        raise ValueError(
            "fill_value=NaN is ambiguous: pass linopy.ABSENT to leave the "
            "positions the join creates absent, or a number to fill them."
        )


def enforce_no_multiindex(obj: Any, *, context: str = "input") -> None:
    """Reject any MultiIndex dimension on ``obj``."""
    indexes = getattr(obj, "indexes", None) or {}
    dim = next((d for d, i in indexes.items() if isinstance(i, pd.MultiIndex)), None)
    if dim is not None:
        raise ValueError(_multiindex_message(str(dim), context))


def first_mismatched_dim(
    a: DataArray | Dataset, b: DataArray | Dataset
) -> tuple[str, Any, Any] | None:
    """
    Return ``(dim, a_labels, b_labels)`` for the first shared dim that
    disagrees on coordinate labels OR size, or ``None`` if all agree.

    Label equality is order-sensitive (§8 exact): the same labels in a
    different order are a mismatch, like a differing label *set*.

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


def reindex_like_if_needed(
    arr: DataArray, ref: DataArray, fill_value: Any
) -> DataArray:
    """
    ``arr.reindex_like(ref, fill_value=...)`` without the copy when already aligned.
    Only the values take ``fill_value``; auxiliary coordinates are filled with NaN.

    ``broadcast_to_coords`` leaves an operand spanning ``ref``'s dims but with
    fresh index objects (equal, not identical), so a plain ``reindex_like`` would
    deep-copy the whole array to no effect on the exact-match hot path. Skip it
    when every ``ref`` dim is present and ``first_mismatched_dim`` finds no
    disagreement — the same shared-dim check the alignment rules use, so ``None``
    means the reindex changes nothing.
    """
    if set(ref.dims) <= set(arr.dims) and first_mismatched_dim(ref, arr) is None:
        return arr
    return arr.reindex_like(ref, fill_value={arr.name: fill_value})


def enforce_merge_dims(
    datasets: Sequence[Dataset],
    *,
    concat_dim: str,
    joinable: bool = True,
) -> None:
    """
    Enforce §8 across the operands of a merge: raise on the first shared user
    dim whose labels differ from the first operand's, in set or in order.
    Helper dims and the concat dim are excluded; bare dimension indexes are
    compared, so auxiliary coords stay §11's job. ``joinable`` says whether
    the operation accepts ``join=``.
    """
    if len(datasets) < 2:
        return
    skip = set(HELPER_DIMS) | {concat_dim}
    indexed = [
        {k: d.indexes[k] for k in d.dims if k not in skip and k in d.indexes}
        for d in datasets
    ]
    shared = set.intersection(*(set(p) for p in indexed))
    for other in indexed[1:]:
        for d in shared:
            ref, idx = indexed[0][d], other[d]
            if not ref.equals(idx):
                raise ValueError(
                    _shared_dim_mismatch_message(
                        str(d), ref.values, idx.values, joinable=joinable
                    )
                )


def enforce_aux_conflict(
    datasets: Sequence[Any], *, concat_dim: str | None = None
) -> None:
    """
    Enforce §11 across the operands: raise on an auxiliary (non-dim) coord
    that two or more operands carry with disagreeing values, which xarray
    would silently drop.

    A coord shared with differently-shaped values (e.g. a vector and a
    scalar) and one with equal shapes but differing values get different
    error text. When only one operand carries the coord it propagates from
    that operand unchanged.

    ``concat_dim`` names the dimension a merge concatenates along. A coord
    that lies solely along that dim is concatenated with it rather than
    reconciled, so differing values are no conflict.
    """
    all_names: set[str] = set()
    for d in datasets:
        all_names.update(d.coords)
    for name in sorted(all_names):
        carriers = [d for d in datasets if name in d.coords and name not in d.dims]
        if concat_dim is not None and all(
            d.coords[name].dims == (concat_dim,) for d in carriers
        ):
            continue
        present = [d.coords[name].values for d in carriers]
        if len(present) < 2:
            continue
        ref = present[0]
        for vals in present[1:]:
            if ref.shape != vals.shape:
                raise ValueError(_aux_conflict_message(str(name), ref, vals, "shape"))
            if not _aux_values_equal(ref, vals):
                raise ValueError(_aux_conflict_message(str(name), ref, vals, "value"))


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
    Enforce the dead-term invariant on a merged dataset.

    ``const.isnull()`` at a slot ⇒ every term at that slot must have
    ``coeffs = NaN`` and ``vars = -1``. After ``merge`` concatenates two
    expressions along ``_term``, a slot that's absent in one operand
    still carries the *other* operand's valid term in its row; this
    helper masks those away so the §1/§2 storage invariant holds.
    """
    if "const" not in ds or "coeffs" not in ds or "vars" not in ds:
        return ds
    if not np.isnan(ds["const"].values.sum()):
        return ds
    mask = ds["const"].isnull()
    coeffs = ds["coeffs"].where(~mask, np.nan)
    vars_ = ds["vars"].where(~mask, -1)
    return ds.assign(coeffs=coeffs, vars=vars_)
