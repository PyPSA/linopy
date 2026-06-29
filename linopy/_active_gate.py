"""
Legacy helper for padding a partial piecewise ``active`` gate.

This module is a temporary stopgap.  Under the planned v1 arithmetic
semantics (#717) the bare idiom ``active.reindex(coords).fillna(fill_value)``
is correct on its own, so :func:`active_gate` is expected to be deprecated
and this file removed once v1 lands.  Keeping it isolated makes that
removal a single-file delete.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from linopy.piecewise import _to_linexpr, _warn_evolving_api

if TYPE_CHECKING:
    from linopy.expressions import LinearExpression
    from linopy.types import LinExprLike


def active_gate(
    active: LinExprLike,
    coords: Mapping[Any, Any],
    fill_value: float = 1,
) -> LinearExpression:
    r"""
    Pad a partial ``active`` gate to full coverage for piecewise gating.

    Reindexes ``active`` to ``coords`` and fills missing/masked entries with
    ``fill_value`` (``1`` = always active, ``0`` = always off), so a gate
    defined over only a subset of :meth:`~linopy.Model.add_piecewise_formulation`'s
    coordinate does not force the uncovered entries to zero. Equivalent to the
    v1 idiom ``active.reindex(coords).fillna(fill_value)`` but correct under
    legacy too (see the module docstring).

    .. code-block:: python

        gate = active_gate(status, {"component": components})
        m.add_piecewise_formulation((power, xs), (fuel, ys), active=gate)

    Parameters
    ----------
    active : Variable or LinearExpression
        The (possibly partial) gate expression.
    coords : mapping of dim to labels
        Reindex target, passed straight to ``reindex``; unlisted dims
        broadcast.
    fill_value : float, default 1
        Value for missing/masked entries (``1`` = on, ``0`` = off).

    Returns
    -------
    LinearExpression
        The padded gate, suitable to pass as ``active=``.

    Warns
    -----
    EvolvingAPIWarning
        Part of the evolving piecewise API; may be refined.
    """
    _warn_evolving_api(
        "active_gate",
        "piecewise: active_gate is a new API; its signature and the way it "
        "resolves missing/masked entries may be refined in minor releases.  "
        "It is primarily a legacy stopgap and may be removed once legacy "
        "semantics are dropped.  This warning fires once per session; "
        "silence with "
        '`warnings.filterwarnings("ignore", category=linopy.EvolvingAPIWarning)`.',
    )
    gate = _to_linexpr(active).reindex(coords)
    term_dims = [d for d in gate.vars.dims if d not in gate.coord_dims]
    present = (gate.vars >= 0).any(term_dims)
    return gate.where(present, fill_value)
