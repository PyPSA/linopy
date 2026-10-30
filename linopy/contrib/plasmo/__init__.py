# SPDX-FileCopyrightText: Contributors to linopy <https://github.com/PyPSA/linopy>
#
# SPDX-License-Identifier: MIT
"""
Decompose a linopy model onto a Plasmo.jl ``OptiGraph`` and solve it with a
graph-based algorithm (Benders, or a plain monolithic solve; more as Plasmo
grows).

A :class:`Partition` over the model's *constraints* defines the decomposition
cells (:mod:`~linopy.contrib.plasmo.partition`); :class:`PlasmoModel` streams one
per-node LP block at a time into a Julia ``OptiGraph``
(:mod:`~linopy.contrib.plasmo.build` + the ``LinopyPlasmo.jl`` module), arranged
into a subgraph tree by a :mod:`~linopy.contrib.plasmo.topology`. The *graph* is
the artifact; the *algorithm* is a separate free function over it, so the same
graph supports Benders, a flat solve, or (later) other Plasmo algorithms.

Different algorithms constrain the topology: :func:`benders` requires a **flat**
topology (master and subproblems as sibling subgraphs under the root), while
:func:`optimize` works with any (flat or nested) topology.

Only integer positions and float data cross to Julia, via shared numpy arrays
through juliacall. Continuous variables only (PlasmoBenders does not support
integers).

Example:
-------
>>> import linopy
>>> from linopy.contrib.plasmo import Partition, has, group, solve_benders
>>> m = linopy.read_netcdf("testProblem.nc")  # doctest: +SKIP
>>> partition = Partition(
...     {  # doctest: +SKIP
...         "top": ~has("set_time_steps_yearly") | ~has("set_nodes"),
...         "sub": group("set_time_steps_yearly") & has("set_nodes"),
...     }
... )
>>> solution = solve_benders(m, partition)  # doctest: +SKIP

For more control, build the model and pick the algorithm explicitly:

>>> pm = PlasmoModel(m, partition)  # doctest: +SKIP
>>> benders(pm)  # or optimize(pm)
>>> pm.value("capacity")  # a solution DataArray
"""

from __future__ import annotations

import os

import numpy as np

from linopy.constants import Result, Solution, Status
from linopy.contrib.plasmo.build import Plan
from linopy.contrib.plasmo.partition import (
    Partition,
    Predicate,
    by_size,
    group,
    has,
    name,
)
from linopy.contrib.plasmo.topology import ManualTopology, Topology, _node_name, flat

__all__ = [
    "Partition",
    "Predicate",
    "has",
    "name",
    "group",
    "by_size",
    "flat",
    "PlasmoModel",
    "optimize",
    "benders",
    "solve_benders",
]

_MODULE_JL = os.path.join(os.path.dirname(__file__), "LinopyPlasmo.jl")


def _jl():
    """
    Import juliacall, ``include`` the LinopyPlasmo Julia module once, and return
    ``(Main, LinopyPlasmo)``.
    """
    from juliacall import Main as jl

    mod = getattr(_jl, "_mod", None)
    if mod is None:
        mod = jl.seval(f'include("{_MODULE_JL}")')  # returns the module
        _jl._mod = mod  # type: ignore[attr-defined]
    return jl, mod


class PlasmoModel:
    """
    A linopy model built as a Plasmo ``OptiGraph``, ready for a graph algorithm.

    Construction is cheap (it only *plans* the partition); the Julia graph is
    built lazily on first use, streaming one block at a time. With
    ``topology="manual"`` the subgraph tree must be declared via
    :meth:`add_subgraph` before the graph is built.
    """

    def __init__(self, model, partition: Partition, *, topology: Topology | str = None):
        self._model = model
        self._plan: Plan = Plan.from_model(model, partition)
        if topology is None or topology == "flat":
            self._topology: Topology = flat()
        elif topology == "manual":
            self._topology = ManualTopology(self._plan.node_keys)
        elif isinstance(topology, Topology):
            self._topology = topology
        else:
            raise ValueError(f"unknown topology {topology!r}")
        self._jl = None
        self._mod = None
        self._builder = None  # Julia GraphBuilder
        self._result = None  # algorithm object / graph for read-back
        self._solved = False

    def __repr__(self) -> str:
        plan = self._plan
        graph = "built" if self._builder is not None else "not built"
        solved = "solved" if self._solved else "not solved"
        header = (
            f"PlasmoModel ({plan.n_nodes} nodes, {len(plan.links)} links, "
            f"{graph}, {solved})"
        )
        # Once built, self._parents is cached and authoritative; pass it along
        # so describe() doesn't need to re-resolve (or re-raise) the topology.
        parents = self._parents if self._builder is not None else None

        def label(i, key):
            return (
                f"{_node_name(key)} "
                f"({plan.node_n_vars(i)} vars, {plan.node_n_cons(i)} cons)"
            )

        tree = self._topology.describe(plan.node_keys, parents, label)
        return "\n".join([header, tree])

    # -- manual topology --------------------------------------------------

    def add_subgraph(self, cns_node_name):
        """
        (manual topology) Attach the cell with key head ``cns_node_name`` under
        the root, returning a handle whose ``.add_subgraph`` nests further cells.
        """
        if not isinstance(self._topology, ManualTopology):
            raise TypeError("add_subgraph requires topology='manual'")
        return self._topology._attach(cns_node_name, -1)

    # -- lazy build -------------------------------------------------------

    def _build(self):
        if self._builder is not None:
            return
        self._jl, self._mod = _jl()
        jl, mod = self._jl, self._mod
        parents = self._topology.parents(self._plan.node_keys)
        self._parents = parents  # -1 everywhere == flat

        builder = mod.new_builder()
        # Stream blocks parents-first so a nested block's parent subgraph exists.
        # jl_index[k] = 1-based Julia node index assigned to plan node k.
        jl_index = np.zeros(self._plan.n_nodes, dtype=np.int64)
        for block in self._plan.iter_blocks(order=_parents_first(parents)):
            k = block.node
            par = parents[k]
            par_jl = 0 if par < 0 else int(jl_index[par])
            jl_index[k] = int(
                mod.add_block_b(
                    builder,
                    par_jl,
                    _node_name(self._plan.node_keys[k]),
                    jl.Vector[jl.Int](block.indptr),
                    jl.Vector[jl.Int](block.colval),
                    block.nzval,
                    block.b,
                    jl.Vector[jl.String]([str(s) for s in block.sense]),
                    block.lb,
                    block.ub,
                    block.c,
                )
            )
            del block  # drop the Python copy before building the next

        # links: node indices already parallel int vectors; remap plan node ->
        # Julia node (0-based) with a single fancy-index, no per-link Python loop.
        lk = self._plan.links
        if len(lk):
            mod.add_links_b(
                builder,
                jl.Vector[jl.Int](jl_index[lk.owner_node] - 1),
                jl.Vector[jl.Int](lk.owner_col),
                jl.Vector[jl.Int](jl_index[lk.other_node] - 1),
                jl.Vector[jl.Int](lk.other_col),
            )
        mod.finalize_b(builder)

        self._builder = builder
        self._jl_index = jl_index  # plan node -> 1-based Julia node

    # -- graph handles (escape hatch) ------------------------------------

    @property
    def is_flat(self) -> bool:
        """Whether every cell sits directly under the root (no nesting)."""
        self._build()
        return all(p < 0 for p in self._parents)

    @property
    def graph(self):
        """The Julia ``OptiGraph`` (builds it on first access)."""
        self._build()
        return self._builder.graph

    def master(self):
        """The master subgraph (the first-declared node's subgraph)."""
        self._build()
        # plan node 0 is the master; find its Julia subgraph
        return self._builder.subgraphs[self._jl_index[0] - 1]

    # -- read-back --------------------------------------------------------

    def _record_result(self, alg):
        self._result = alg
        self._solved = True

    def result(self) -> Result:
        """
        The solution as a :class:`linopy.constants.Result`. ``solution.primal``
        is a dense array indexed by variable label (``NaN`` where a variable was
        not part of the decomposed LP). ``dual`` is left empty for now.
        """
        if not self._solved:
            raise RuntimeError("no solution yet -- run optimize(pm) or benders(pm)")

        mod = self._mod
        n_labels = self._model.variables.label_index.label_to_pos.shape[0]
        primal = np.full(n_labels, np.nan)
        # Retrieve each node's full value vector on the Julia side (few nodes,
        # not many variables), then scatter by that node's global labels.
        for k in range(self._plan.n_nodes):
            vals = np.asarray(
                mod.node_values(self._result, self._builder, int(self._jl_index[k]) - 1)
            )
            primal[self._plan.node_vlabels[k]] = vals

        status = Status.from_termination_condition("optimal")
        mat = self._model.matrices  # c is vlabels-position-indexed
        objective = float(np.nansum(mat.c * primal[mat.vlabels]))
        return Result(
            status=status, solution=Solution(primal=primal, objective=objective)
        )

    def assign_to_model(self):
        """
        Write the solution back onto the linopy model via ``assign_result`` so
        ``model.solution`` and ``variable.solution`` work natively. Returns the
        model. ``dual`` is empty, so no constraint duals are written.
        """
        self._model.assign_result(self.result())
        return self._model

    def solution(self):
        """The model's solution ``Dataset`` (assigns to the model first)."""
        return self.assign_to_model().solution

    def value(self, var_name):
        """Solution ``DataArray`` for a single variable."""
        return self.solution()[var_name]


def _parents_first(parents: list[int]) -> list[int]:
    """Order node indices so every parent precedes its children."""
    order: list[int] = []
    placed = [False] * len(parents)

    def visit(k):
        if placed[k]:
            return
        p = parents[k]
        if p >= 0:
            visit(p)
        placed[k] = True
        order.append(k)

    for k in range(len(parents)):
        visit(k)
    return order


# -- algorithms (free functions over a PlasmoModel) --------------------------


def optimize(pm: PlasmoModel, *, solver: str = "highs") -> PlasmoModel:
    """Solve the whole graph monolithically (all subgraphs, links hard)."""
    pm._build()
    jl, mod = pm._jl, pm._mod
    jl.seval(f"using {_solver_pkg(solver)}")
    opt = getattr(jl, _solver_pkg(solver)).Optimizer
    graph = mod.run_optimize(pm.graph, solver=opt)
    pm._record_result(graph)  # solution lives on the graph backend
    return pm


def benders(
    pm: PlasmoModel,
    *,
    master=None,
    solver: str = "highs",
    max_iters: int = 1000,
    **benders_options,
) -> PlasmoModel:
    """
    Run PlasmoBenders with ``master`` (default: the first-declared node).

    ``max_iters`` and ``**benders_options`` are forwarded to
    ``PlasmoBenders.jl``'s ``BendersAlgorithm`` (e.g. ``tol``, ``regularize``,
    ``add_slacks``, ``multicut``, ``strengthened``, ``warm_start``, ...) --
    see the `BendersAlgorithm docs
    <https://plasmo-dev.github.io/PlasmoAlgorithms.jl/dev/PlasmoBenders/introduction/>`_
    for the full list.
    ``add_slacks=True`` and ``regularize=True`` are linopy's defaults (unlike
    PlasmoBenders' own, both ``False``); pass either as ``False`` to override.
    """
    pm._build()
    if not pm.is_flat:
        raise ValueError(
            "benders requires a flat topology: the master and subproblems must be "
            "sibling subgraphs under the root, but this model nests subgraphs. "
            "Rebuild with the default topology=flat() for Benders (nested "
            "topologies work with optimize())."
        )
    jl, mod = pm._jl, pm._mod
    jl.seval(f"using {_solver_pkg(solver)}")
    opt = getattr(jl, _solver_pkg(solver)).Optimizer
    alg = mod.run_benders(
        pm.graph,
        pm.master() if master is None else master,
        solver=opt,
        max_iters=max_iters,
        **benders_options,
    )
    pm._record_result(alg)
    return pm


def solve_benders(
    model, partition: Partition, *, solver="highs", max_iters=1000, **benders_options
):
    """Convenience: build, run Benders, return the solution dict."""
    pm = PlasmoModel(model, partition)
    benders(pm, solver=solver, max_iters=max_iters, **benders_options)
    return pm.solution()


def _solver_pkg(solver: str) -> str:
    return {"highs": "HiGHS", "gurobi": "Gurobi"}[solver.lower()]
