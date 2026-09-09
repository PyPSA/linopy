# SPDX-FileCopyrightText: Contributors to linopy <https://github.com/PyPSA/linopy>
#
# SPDX-License-Identifier: MIT
"""
Partition algebra for decomposing a linopy model into Plasmo graph nodes.

A :class:`Partition` is an ordered mapping ``{node_name: predicate}`` over the
*constraints* of a model. Each constraint is assigned to the first node whose
predicate matches it (first-match-wins), so the partition is disjoint and
exhaustive over constraints. Variables are *not* partitioned here; their node
membership is derived from the constraints that reference them (see
:mod:`linopy.contrib.plasmo.build`), and a variable appearing in more than one
node becomes a linking variable.

Predicates are built from atoms composed with ``~ & |``:

- ``has(dim)`` -- scalar: the constraint is dimensioned over ``dim``.
- ``name(cns)`` -- scalar: the constraint has this linopy name.
- ``group(dim)`` -- scattering: label = the coordinate value along ``dim``;
  fans the node into one sub-node per distinct value.
- ``by_size(dim, n)`` -- scattering: label = position-along-``dim`` // ``n``;
  buckets a fine dimension into slices.

A *scalar* predicate answers yes/no per constraint. A *scattering* predicate,
when it matches, additionally carries a label so the node expands into
``name[label]``. Combining scatterers with ``&`` crosses their labels;
combining a scalar with a scatterer lets the scalar *gate* (filter out
non-matching rows) while the scatterer labels the rest.

Implementation note (axis-separable evaluation)
-----------------------------------------------
A predicate over a constraint's ``labels`` grid is separable *by axis*:
``has``/``name`` ignore coordinates entirely; ``group``/``by_size`` each depend
on a single axis. So each atom returns one *per-axis* selector -- either ``ALL``
(every position on that axis matches, no label) or a length-``dim_size`` object
array of labels (``NOMATCH`` where a position fails). These are compounded into
the full grid only at the end, and mapped to CSR rows via the constraint's own
``labels`` grid (ravel, drop ``-1``). This makes an atom O(sum of dim sizes)
rather than O(active rows), a large speedup on fine grids.

The trade-off: only *rectangular* selections are representable per axis. ``&``
of rectangles is a rectangle (intersect per axis), but ``~scatter`` and
``scatter | scatter`` are unions/complements that are not single rectangles --
these raise. This is not a real restriction for decomposition: a block is
coupled to the master along index boundaries, which are rectangles; disjoint
unions of index regions belong in *separate* nodes (expressed by scattering into
``name[label]``), not merged into one. ``|`` of *scalars* (e.g.
``~has(a) | ~has(b)`` for the master) stays rectangular and is supported.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from linopy.model import Model

NodeKey = Hashable


# Sentinel label meaning "this position does not match". Distinct from a real
# label; ``True`` is the trivial "matched, no label" label used by scalars.
class _NoMatch:
    _singleton: _NoMatch | None = None

    def __new__(cls) -> _NoMatch:
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

    def __repr__(self) -> str:
        return "NOMATCH"


NOMATCH = _NoMatch()


# Per-axis selector sentinel: this axis places no constraint and carries no
# label -- every position on it matches. (Distinct from a length-n label array.)
class _All:
    _singleton: _All | None = None

    def __new__(cls) -> _All:
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

    def __repr__(self) -> str:
        return "ALL"


ALL = _All()

# An AxisResult is either ``None`` (predicate matches no row of the constraint)
# or a ``dict[str, selector]`` mapping each dim to a selector that is ``ALL`` or
# a 1-D object array of length dim_size (entries: NOMATCH, or a hashable label).
# A missing dim key is treated as ``ALL``.


class Predicate:
    """Base class for constraint-selection predicates. Compose with ``~ & |``."""

    def _eval_axes(
        self, model: Model, name: str, dims: tuple[str, ...]
    ) -> dict[str, object] | None:
        """
        Return per-axis selectors for one constraint, or ``None`` if the
        predicate matches no row of it at all.

        Each returned dict maps a subset of ``dims`` to a selector; a missing dim
        or an ``ALL`` value means "all positions on that axis". Non-``ALL`` values
        are length-``dim_size`` object arrays of labels / ``NOMATCH``.
        """
        raise NotImplementedError

    @property
    def is_scattering(self) -> bool:
        """
        Whether this predicate carries per-position labels (``group``/``by_size``
        or any combination containing one). Scattering predicates select
        *rectangles with labels*; their negation/union is not a single rectangle,
        so ``~`` and ``|`` reject them (see module docstring).
        """
        return False

    def __invert__(self) -> Predicate:
        if self.is_scattering:
            raise TypeError(
                f"cannot apply ~ to the scattering predicate {self!r}: the "
                "complement of a group()/by_size() selection is not a single "
                "rectangle. Negate the scalar part instead (e.g. "
                "~has(dim), not ~group(dim))."
            )
        return _Not(self)

    def __and__(self, other: Predicate) -> Predicate:
        return _And(self, other)

    def __or__(self, other: Predicate) -> Predicate:
        if self.is_scattering and other.is_scattering:
            raise TypeError(
                f"cannot apply | to two scattering predicates ({self!r} | "
                f"{other!r}): their union is not a single rectangle. Put disjoint "
                "index regions in separate nodes (scatter into name[label]) "
                "rather than OR-ing them into one; | is only for scalars "
                "(has/name), e.g. ~has(a) | ~has(b)."
            )
        return _Or(self, other)


@dataclass(frozen=True)
class _Has(Predicate):
    dim: str

    def _eval_axes(self, model, name, dims):
        return {} if self.dim in dims else None


@dataclass(frozen=True)
class _Name(Predicate):
    cns: str

    def _eval_axes(self, model, name, dims):
        return {} if name == self.cns else None


@dataclass(frozen=True)
class _Group(Predicate):
    dim: str

    is_scattering = True

    def _eval_axes(self, model, name, dims):
        if self.dim not in dims:
            return None
        index = model.constraints[name].labels.get_index(self.dim)
        # label = coordinate value along dim (kept as-is: str or int)
        return {self.dim: np.asarray(index, dtype=object)}


@dataclass(frozen=True)
class _BySize(Predicate):
    dim: str
    n: int

    is_scattering = True

    def _eval_axes(self, model, name, dims):
        if self.dim not in dims:
            return None
        index = model.constraints[name].labels.get_index(self.dim)
        codes = pd.factorize(np.asarray(index), sort=True)[0]
        return {self.dim: (codes // self.n).astype(object)}


@dataclass(frozen=True)
class _Not(Predicate):
    inner: Predicate

    def _eval_axes(self, model, name, dims):
        inner = self.inner._eval_axes(model, name, dims)
        if inner is None:
            # inner matched nothing -> negation matches everything
            return {}
        # A negated scatterer is a complement that is not axis-separable; only a
        # pure scalar (all axes ALL) has a rectangular complement (= nothing).
        if any(v is not ALL for v in inner.values()):
            raise ValueError(
                "~ of a scattering predicate (group/by_size) is not supported."
            )
        return None


def _cross_axis(sa, sb):
    """AND two per-axis label selectors position-wise, crossing their labels."""
    ma, mb = sa != NOMATCH, sb != NOMATCH
    both = ma & mb
    out = np.full(len(sa), NOMATCH, dtype=object)
    for i in np.flatnonzero(both):
        lbls = [x for x in (sa[i], sb[i]) if x is not True]
        out[i] = True if not lbls else lbls[0] if len(lbls) == 1 else tuple(lbls)
    return out, both.any()


@dataclass(frozen=True)
class _And(Predicate):
    left: Predicate
    right: Predicate

    @property
    def is_scattering(self) -> bool:
        return self.left.is_scattering or self.right.is_scattering

    def _eval_axes(self, model, name, dims):
        a = self.left._eval_axes(model, name, dims)
        if a is None:
            return None
        b = self.right._eval_axes(model, name, dims)
        if b is None:
            return None
        merged: dict[str, object] = {}
        for d in dims:
            sa = a.get(d, ALL)
            sb = b.get(d, ALL)
            if sa is ALL and sb is ALL:
                continue
            if sa is ALL:
                merged[d] = sb
            elif sb is ALL:
                merged[d] = sa
            else:
                crossed, nonempty = _cross_axis(sa, sb)
                if not nonempty:
                    return None
                merged[d] = crossed
        return merged


@dataclass(frozen=True)
class _Or(Predicate):
    left: Predicate
    right: Predicate

    def _eval_axes(self, model, name, dims):
        a = self.left._eval_axes(model, name, dims)
        b = self.right._eval_axes(model, name, dims)
        if a is None:
            return b
        if b is None:
            return a
        # OR of two rectangles is a single rectangle only when one side is a
        # pure scalar (all-or-nothing): a scalar-True side matches every row, so
        # the union is everything.
        if all(v is ALL for v in a.values()) or all(v is ALL for v in b.values()):
            return {}
        raise ValueError(
            "| of two scattering predicates is not supported "
            "(their union is not a single rectangle)."
        )


# -- public atom constructors ------------------------------------------------


def has(dim: str) -> Predicate:
    """Scalar: constraint is dimensioned over ``dim``."""
    return _Has(dim)


def name(cns: str) -> Predicate:
    """Scalar: constraint has the linopy name ``cns``."""
    return _Name(cns)


def group(dim: str) -> Predicate:
    """Scattering: one sub-node per distinct coordinate value along ``dim``."""
    return _Group(dim)


def by_size(dim: str, n: int) -> Predicate:
    """Scattering: bucket ``dim`` into slices of ``n`` consecutive positions."""
    return _BySize(dim, n)


# -- the Partition ------------------------------------------------------------


def _compound_labels(res, dims, shape, active_pos):
    """
    Compound per-axis selectors into a flat label array over the *active* grid
    positions. Returns an object array (len == active_pos.size): ``NOMATCH``
    where a row is not selected, else the compounded label (``True`` if no
    per-axis labels contributed).
    """
    out = np.full(active_pos.size, True, dtype=object)
    fail = np.zeros(active_pos.size, dtype=bool)
    multi = np.unravel_index(active_pos, shape)  # per-axis index of each row

    for ax, d in enumerate(dims):
        sel = res.get(d, ALL)
        if sel is ALL:
            continue
        per_pos = sel[multi[ax]]  # label / NOMATCH for each active row on axis
        axis_fail = per_pos == NOMATCH
        fail |= axis_fail
        for i in np.flatnonzero(~axis_fail):
            lb = per_pos[i]
            if lb is True:
                continue
            la = out[i]
            if la is True:
                out[i] = lb
            elif isinstance(la, tuple):
                out[i] = (*la, lb)
            else:
                out[i] = (la, lb)

    out[fail] = NOMATCH
    return out


class Partition:
    """
    Ordered mapping ``{node_name: predicate}`` over a model's constraints.

    First-match-wins: every constraint is assigned to the first node whose
    predicate matches it. A scattering predicate expands its node into
    ``name[label]``. A constraint matching no node raises.

    Construct with named nodes as keyword arguments (declaration order is the
    match order, so the first is the Benders master)::

        Partition(
            top=~has("year") | ~has("nodes"),
            sub=group("year") & has("nodes"),
        )

    A single ``dict`` positional argument is also accepted (useful when node
    names are not valid Python identifiers).
    """

    def __init__(
        self,
        nodes: dict[str, Predicate] | None = None,
        /,
        **named: Predicate,
    ):
        if nodes is not None and named:
            raise TypeError("pass nodes either as a dict or as keywords, not both")
        self.nodes: dict[str, Predicate] = dict(nodes) if nodes is not None else named
        if not self.nodes:
            raise ValueError("Partition needs at least one node")

    def assign(self, model: Model) -> tuple[np.ndarray, list[NodeKey]]:
        """
        Assign every active constraint to a node.

        Returns
        -------
        node_of_cns : numpy.ndarray of int, shape (n_active_cons,)
            Node index per constraint, aligned to
            ``model.constraints.label_index.clabels`` order (= CSR rows).
        node_keys : list of NodeKey
            ``node_of_cns[i]`` indexes into this list. Order is the discovery
            order: nodes appear in partition declaration order, and within a
            scattering node in first-seen label order. So ``node_keys[0]`` is
            the first-declared node's first key -- the Benders master.
        """
        cli = model.constraints.label_index
        clabels = cli.clabels
        label_to_pos = cli.label_to_pos  # constraint label -> CSR row

        node_of_cns = np.full(len(clabels), -1, dtype=np.int64)
        node_keys: list[NodeKey] = []
        key_to_idx: dict[NodeKey, int] = {}

        def key_index(key: NodeKey) -> int:
            idx = key_to_idx.get(key)
            if idx is None:
                idx = len(node_keys)
                key_to_idx[key] = idx
                node_keys.append(key)
            return idx

        for cname in model.constraints:
            con = model.constraints[cname]
            labels = con.labels
            if labels.size == 0:
                continue
            dims = labels.dims
            shape = labels.shape

            # Active rows in grid order: take the labels grid, ravel it, keep the
            # non -1 entries; those label values map straight to CSR rows.
            lab_flat = labels.values.ravel()
            active_pos = np.flatnonzero(lab_flat != -1)
            if active_pos.size == 0:
                continue
            rows = label_to_pos[lab_flat[active_pos]]

            taken = np.zeros(active_pos.size, dtype=bool)
            for node_name, pred in self.nodes.items():
                res = pred._eval_axes(model, cname, dims)
                if res is None:
                    continue
                labels_flat = _compound_labels(res, dims, shape, active_pos)
                sel = (labels_flat != NOMATCH) & ~taken
                if not sel.any():
                    continue
                for i in np.flatnonzero(sel):
                    lab = labels_flat[i]
                    key: NodeKey = node_name if lab is True else (node_name, lab)
                    node_of_cns[rows[i]] = key_index(key)
                taken |= sel
                if taken.all():
                    break

        if (node_of_cns == -1).any():
            n = int((node_of_cns == -1).sum())
            missing = clabels[node_of_cns == -1][:5]
            raise ValueError(
                f"{n} constraint(s) matched no partition node "
                f"(labels {missing.tolist()}...). The partition must be "
                f"exhaustive over constraints."
            )

        return node_of_cns, node_keys
