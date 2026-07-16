.. _plasmo-benders:

=====================================
Benders Decomposition with Plasmo.jl
=====================================

.. warning::

   This feature is **experimental**, lives in ``linopy.contrib.plasmo``, and is not part of linopy's stable API.
   It requires a Julia installation and the packages below; it is not covered by CI and may change without notice.

``linopy.contrib.plasmo`` decomposes a linopy model into a `Plasmo.jl <https://github.com/plasmo-dev/Plasmo.jl>`_ ``OptiGraph`` and solves it with `PlasmoBenders.jl <https://github.com/plasmo-dev/PlasmoAlgorithms.jl/tree/main/lib/PlasmoBenders>`_'s Benders algorithm.

Only continuous variables are supported for now -- ``linopy.contrib.plasmo`` doesn't yet build integer/binary variables into the decomposed graph, though PlasmoBenders itself supports MIPs (``is_MIP``, ``strengthened`` cuts).

Why decompose at all
=====================

Benders decomposition splits a large LP into a **master** problem and one or more **subproblems** that only communicate through a small set of shared (*linking*) variables.
Each iteration solves the subproblems given the master's current guess, then adds a cut to the master from the subproblems' sensitivity information.
For models with natural block structure -- investment decisions per year, coupled only lightly to per-year operational detail, for example -- this can be far cheaper than solving the whole LP at once, and lets each subproblem be solved in parallel.

The trade-off is engineering complexity: you must decide which constraints belong to which block, and the master/subproblem split must respect the model's coupling structure or the algorithm won't converge quickly.
That decision is what a :class:`~linopy.contrib.plasmo.Partition` encodes.

Installing the Julia side
===========================

``linopy.contrib.plasmo`` talks to Julia via `juliacall <https://github.com/JuliaPy/PythonCall.jl>`_ (``pyjuliacall``), which manages its own private Julia environment through `pyjuliapkg <https://github.com/JuliaPy/pyjuliapkg>`_ -- no separate Julia install or ``Pkg.add`` step needed.
The Julia packages themselves (``JuMP``, ``Plasmo``, ``PlasmoBenders``, ``HiGHS``) are declared in a ``juliapkg.json`` file that ``pyjuliapkg`` discovers automatically the first time ``juliacall`` is imported, and resolves once into that private environment.

.. code-block:: bash

    pip install pyjuliacall pyjuliapkg

A ``juliapkg.json`` next to your working directory (or anywhere importable on ``sys.path``) is enough -- see ``experiment/juliapkg.json`` in the linopy repository for a working example pinning ``JuMP``/``Plasmo``/``PlasmoBenders``/``HiGHS``.

Verify the Julia side resolves correctly:

.. code-block:: python

    from juliacall import Main as jl

    jl.seval("using JuMP, Plasmo, PlasmoBenders, HiGHS")

The first import triggers Julia package resolution and precompilation, which can take a few minutes; subsequent imports are fast.

Quick start
===========

.. code-block:: python

    import linopy
    from linopy.contrib.plasmo import Partition, has, group, solve_benders

    m = linopy.read_netcdf("model.nc")

    partition = Partition(
        {
            "top": ~has("year") | ~has("region"),  # master: coupling constraints
            "sub": group("year") & has("region"),  # one subproblem per year
        }
    )

    solution = solve_benders(m, partition)

For more control over the build/solve split (e.g. to inspect the graph before solving, or to run a monolithic solve instead), use :class:`~linopy.contrib.plasmo.PlasmoModel` directly:

.. code-block:: python

    from linopy.contrib.plasmo import PlasmoModel, benders, optimize

    pm = PlasmoModel(m, partition)
    benders(pm)  # or: optimize(pm) for a monolithic graph solve
    pm.value("capacity")  # a solution DataArray, same shape as m.variables["capacity"]

See :doc:`plasmo-benders-decomposition` for a full worked example.

Concepts
========

Partition
---------

A :class:`~linopy.contrib.plasmo.Partition` is an ordered mapping ``{node_name: predicate}`` over the model's **constraints**.
Every constraint is assigned to the first node whose predicate matches it (first-match-wins), so the partition must be disjoint and exhaustive -- a constraint matching no node raises.
Node order matters: the first-declared node becomes the Benders master.

Predicates are built from two kinds of atom, composed with ``~`` / ``&`` / ``|``:

- **scalar** -- matches or doesn't, no label: :func:`~linopy.contrib.plasmo.has` (is the constraint dimensioned over this set?) and :func:`~linopy.contrib.plasmo.name` (select by linopy constraint name).
- **scattering** -- matches and fans the node out into one sub-node per label: :func:`~linopy.contrib.plasmo.group` (one sub-node per distinct coordinate value) and :func:`~linopy.contrib.plasmo.by_size` (bucket a fine dimension into slices of *n* consecutive positions, e.g. hours into weeks).

Combining two scatterers with ``&`` crosses their labels.
Combining a scalar with a scatterer lets the scalar *gate* rows (filter out non-matching ones) while the scatterer labels the rest.
Negating (``~``) or ``|``-combining two scattering predicates is not supported: their complement/union is not a single rectangle of the constraint's coordinate grid, and disjoint index regions belong in *separate* nodes rather than merged into one.

Variables are **not** partitioned directly -- a variable belongs to every node whose constraints reference it, derived automatically.
A variable referenced from more than one node is a *linking variable*; the build step adds an equality constraint pinning every non-owning copy to the (first-declared) owning node's copy, which is exactly what Benders cuts on.

Topology
--------

By default (:func:`~linopy.contrib.plasmo.flat`), every partition cell becomes a sibling subgraph directly under the root -- the layout :func:`~linopy.contrib.plasmo.benders` requires.
Pass ``topology="manual"`` to :class:`~linopy.contrib.plasmo.PlasmoModel` to nest subgraphs (for Plasmo algorithms other than Benders that expect a tree), building it with ``PlasmoModel.add_subgraph``.

Algorithms
----------

Two free functions operate on a built :class:`~linopy.contrib.plasmo.PlasmoModel`:

- :func:`~linopy.contrib.plasmo.benders` -- runs ``PlasmoBenders.jl``'s ``BendersAlgorithm``.
  Requires a flat topology.
- :func:`~linopy.contrib.plasmo.optimize` -- solves the whole graph monolithically (every subgraph, links enforced as hard constraints).
  Works with any topology; useful as a correctness cross-check against :func:`~linopy.contrib.plasmo.benders`, or against ``model.solve()`` on the original, undecomposed model.

:func:`~linopy.contrib.plasmo.benders` and :func:`~linopy.contrib.plasmo.solve_benders` accept ``max_iters`` explicitly plus arbitrary further keyword arguments, forwarded verbatim to `BendersAlgorithm <https://plasmo-dev.github.io/PlasmoAlgorithms.jl/dev/PlasmoBenders/introduction/>`_ (``tol``, ``regularize``, ``add_slacks``, ``multicut``, ``strengthened``, ``warm_start``, and the rest of that constructor's options -- see its docs for the full, authoritative list).
linopy defaults ``add_slacks`` and ``regularize`` to ``True`` (both default to ``False`` in PlasmoBenders itself, but subproblems in a linopy-derived graph are more prone to infeasibility without slacks); pass either explicitly as ``False`` to opt back out:

.. code-block:: python

    benders(pm, max_iters=200, tol=1e-6, multicut=False)
    solve_benders(m, partition, add_slacks=False, regularize_param=0.3)

Both read the solution back into a dense array indexed by linopy's variable labels (``PlasmoModel.result()``), which ``PlasmoModel.assign_to_model()`` writes onto the original model so ``model.solution`` and ``variable.solution`` work as usual.
Constraint duals are not populated (Benders subproblems don't yield duals for the original, undecomposed model).

How data crosses to Julia
==========================

Only integer positions and float data cross the Python/Julia boundary -- never linopy objects, xarray, or strings (set element names stay in Python).
Internally, ``Plan`` slices each node's constraint rows and variable columns directly out of the model's CSR ``Model.matrices``, remapping to node-local column indices, and streams one node's arrays (``indptr``, ``colval``, ``nzval``, bounds, objective coefficients) at a time into the Julia ``GraphBuilder`` -- so peak Python memory is one block's float data, not the whole decomposed problem at once.

Limitations
===========

- Continuous variables only -- not a PlasmoBenders restriction (it supports MIPs), but ``linopy.contrib.plasmo`` doesn't build integer/binary variables into the graph yet.
- Node/subgraph granularity is one cell per :class:`~linopy.contrib.plasmo.Partition` node; there is no explicit variable-side partition yet (a variable's node membership is always derived from its constraints).
- ``experiment/juliapkg.json`` (see *Installing the Julia side* above) is only discovered when Python is run from the ``experiment/`` directory; using ``linopy.contrib.plasmo`` from elsewhere needs its own ``juliapkg.json`` on the resolution path.

References
==========

- `Plasmo.jl documentation <https://plasmo-dev.github.io/Plasmo.jl/stable/>`_
- `PlasmoBenders.jl documentation <https://plasmo-dev.github.io/PlasmoAlgorithms.jl/dev/PlasmoBenders/introduction/>`_ and `source <https://github.com/plasmo-dev/PlasmoAlgorithms.jl/tree/main/lib/PlasmoBenders>`_ -- ``PlasmoBenders`` is registered and installed on its own (``Pkg.add("PlasmoBenders")``), but lives as a subdirectory of the ``PlasmoAlgorithms.jl`` monorepo, alongside sibling algorithm packages such as ``PlasmoSchwarz``.
- `Linopy2Plasmo.jl <https://github.com/leonardgoeke/Linopy2Plasmo.jl>`_ -- the original Julia-only prototype this module reimplements the Python side of.
- The design log and implementation notes live in ``experiment/README.md`` in the linopy repository.
