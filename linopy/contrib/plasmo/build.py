# SPDX-FileCopyrightText: Contributors to linopy <https://github.com/PyPSA/linopy>
#
# SPDX-License-Identifier: MIT
"""
Turn a linopy model plus a constraint :class:`~linopy.contrib.plasmo.partition.Partition`
into the flat, per-node arrays that ``LinopyPlasmo.jl`` needs to build a Plasmo
``OptiGraph``.

Two stages, kept separate so blocks can be *streamed* into Julia:

- :meth:`Plan.from_model` -- the cheap global part. Assigns constraints to nodes,
  derives the node x variable incidence, the per-node variable membership, the
  linking equalities, and the owner of each variable. Holds only index data and
  one bool incidence matrix, never the float CSR blocks.
- :meth:`Plan.iter_blocks` -- a *generator* that slices one :class:`NodeBlock` out of
  the constraint matrix at a time (cheap CSR row-slice + column remap, never
  densified). The caller consumes each block (hands it to Julia) and drops it
  before the next is built, so peak Python memory is one block, not all of them.

Everything here is numpy/scipy; no Julia. Strings never cross to Julia -- only
integer positions and float data.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_array

from linopy.contrib.plasmo.partition import NodeKey, Partition


@dataclass
class NodeBlock:
    """One node's local LP block, all arrays in node-local column indexing."""

    node: int  # node index (into plan.node_keys)
    # CSR of this node's constraint rows over this node's variable columns
    indptr: np.ndarray  # int64, len n_rows+1
    colval: np.ndarray  # int64, node-local column of each nonzero
    nzval: np.ndarray  # float64
    b: np.ndarray  # float64, len n_rows
    sense: np.ndarray  # 'U1' one of '<' '=' '>', len n_rows
    lb: np.ndarray  # float64, len n_cols
    ub: np.ndarray  # float64, len n_cols
    c: np.ndarray  # float64, len n_cols, objective coeff (0 unless owned here)
    vlabels: np.ndarray  # int64, global variable label of each local column


@dataclass
class Links:
    """
    All cross-node equality links ``x[owner] == x[other]`` as parallel int
    vectors (node-local columns), ready to hand to Julia without per-link
    Python iteration.
    """

    owner_node: np.ndarray  # int64, node index of the owner instance
    owner_col: np.ndarray  # int64, node-local column in owner_node
    other_node: np.ndarray  # int64
    other_col: np.ndarray  # int64

    def __len__(self) -> int:
        return len(self.owner_node)


@dataclass
class Plan:
    """
    The cheap, global result of partitioning -- everything except the per-node
    float blocks (which :meth:`iter_blocks` streams). Small: index data + links.
    """

    node_keys: list[NodeKey]  # node index -> key; node_keys[0] is the master
    n_vars: int  # global active variable count
    links: Links  # cross-node equality links (star from owner)
    # per node: global variable label of each node-local column, so the solution
    # read back from Julia (node-local vectors) can be scattered by label.
    node_vlabels: list[np.ndarray]

    # -- internal, consumed by iter_blocks (kept here to avoid recomputation) --
    _A: object  # cleaned CSR constraint matrix
    _node_of_cns: np.ndarray  # node index per constraint row
    _node_vars: list[np.ndarray]  # per node: sorted global var columns
    _owner_node: np.ndarray  # global var -> owner node index
    _b: np.ndarray
    _sense: np.ndarray
    _lb: np.ndarray
    _ub: np.ndarray
    _c: np.ndarray
    _vlabels: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.node_keys)

    def node_n_cons(self, node: int) -> int:
        """Number of constraint rows assigned to ``node``."""
        return int(np.count_nonzero(self._node_of_cns == node))

    def node_n_vars(self, node: int) -> int:
        """Number of variable columns (local or linked) touching ``node``."""
        return len(self.node_vlabels[node])

    @classmethod
    def from_model(cls, model, partition: Partition) -> Plan:
        """
        Assign constraints to nodes and derive membership, links and owners.

        Does not slice any per-node block -- see :meth:`iter_blocks`.
        """
        mat = model.matrices
        A = mat.A  # CSR, rows = clabels order, cols = vlabels order
        if A is None:
            raise ValueError("Model has no constraints to decompose.")
        n_vars = A.shape[1]

        node_of_cns, node_keys = partition.assign(model)
        b, sense = mat.b, mat.sense

        n_nodes = len(node_keys)

        # -- node x variable incidence (see README "Variable membership & linking")
        nnz_node = np.repeat(node_of_cns, np.diff(A.indptr))
        M = coo_array(
            (np.ones(A.nnz, dtype=bool), (nnz_node, A.indices)),
            shape=(n_nodes, n_vars),
        ).tocsr()
        Mc = M.tocsc()
        deg = np.diff(Mc.indptr)  # number of nodes each variable appears in

        c_full = mat.c  # objective coeff aligned to vlabels
        # A variable with a nonzero cost but no constraint has nowhere to be
        # placed for Benders -- reject. Cost-free unconstrained variables are
        # inert (dropped: absent from every node, NaN in the solution).
        orphan = (deg == 0) & (c_full != 0.0)
        if orphan.any():
            raise ValueError(
                f"{int(orphan.sum())} variable(s) carry an objective coefficient "
                "but appear in no constraint, so cannot be assigned to a node."
            )

        # -- owner node per variable: earliest node it appears in. CSC stores
        # each column's row (node) indices ascending, so the first stored entry
        # is the min. Node order places the master at 0, so this is "earliest in
        # declared order". deg==0 variables get -1 (no owner; dropped).
        owner_node = np.full(n_vars, -1, dtype=np.int64)
        present = deg > 0
        owner_node[present] = Mc.indices[Mc.indptr[:-1][present]]

        # -- per-node membership: rows of M give each node's (sorted) columns
        node_vars = [M.indices[M.indptr[k] : M.indptr[k + 1]] for k in range(n_nodes)]
        node_vlabels = [mat.vlabels[cols] for cols in node_vars]

        # -- links: every (node, var) incidence where the node is not the var's
        # owner links back to the owner. Vectorised over all Mc entries.
        entry_node = Mc.indices  # node of each (var, node) incidence
        entry_var = np.repeat(np.arange(n_vars), deg)  # its variable
        entry_owner = owner_node[entry_var]
        is_other = entry_node != entry_owner  # skip the owner's own entry
        ln_var = entry_var[is_other]
        ln_owner = entry_owner[is_other]
        ln_other = entry_node[is_other]
        links = Links(
            owner_node=ln_owner,
            owner_col=_local_cols(node_vars, ln_owner, ln_var),
            other_node=ln_other,
            other_col=_local_cols(node_vars, ln_other, ln_var),
        )

        return cls(
            node_keys=node_keys,
            n_vars=n_vars,
            links=links,
            node_vlabels=node_vlabels,
            _A=A,
            _node_of_cns=node_of_cns,
            _node_vars=node_vars,
            _owner_node=owner_node,
            _b=b,
            _sense=sense,
            _lb=mat.lb,
            _ub=mat.ub,
            _c=c_full,
            _vlabels=mat.vlabels,
        )

    def iter_blocks(self, order: list[int] | None = None) -> Iterator[NodeBlock]:
        """
        Yield one :class:`NodeBlock` at a time, in ``order`` (default: node-index
        order, master first). ``order`` lets the caller stream parents before
        children for nested topologies.

        Each block is sliced fresh from the constraint matrix and yielded; the
        caller is expected to consume it (hand it to Julia) before requesting the
        next, so only one block's float data is alive at a time.
        """
        A = self._A
        for k in range(self.n_nodes) if order is None else order:
            cols = self._node_vars[k]  # sorted global columns of this node

            rows = np.flatnonzero(self._node_of_cns == k)
            Ablk = A[rows]  # CSR row-slice
            # remap global columns to node-local via binary search (cols sorted)
            colval = np.searchsorted(cols, Ablk.indices)

            # objective coeff on this node: only for variables OWNED here
            c_local = np.zeros(cols.size, dtype=np.float64)
            owned_mask = self._owner_node[cols] == k
            c_local[owned_mask] = self._c[cols[owned_mask]]

            yield NodeBlock(
                node=k,
                indptr=Ablk.indptr.astype(np.int64),
                colval=colval.astype(np.int64),
                nzval=Ablk.data.astype(np.float64),
                b=self._b[rows].astype(np.float64),
                sense=self._sense[rows].astype("U1"),
                lb=self._lb[cols].astype(np.float64),
                ub=self._ub[cols].astype(np.float64),
                c=c_local,
                vlabels=self._vlabels[cols].astype(np.int64),
            )


def _local_cols(
    node_vars: list[np.ndarray], nodes: np.ndarray, gvars: np.ndarray
) -> np.ndarray:
    """
    Node-local column of each ``gvars[i]`` within node ``nodes[i]``. Each node's
    ``node_vars`` is sorted, so the local column is a binary search. Grouped by
    node to search each node's column set at once.
    """
    out = np.empty(len(nodes), dtype=np.int64)
    for k in np.unique(nodes):
        sel = nodes == k
        out[sel] = np.searchsorted(node_vars[k], gvars[sel])
    return out
