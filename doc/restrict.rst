.. _restrict:

Restricting a Model
===================

:meth:`linopy.Model.restrict` cuts a subproblem out of a model. You select
labels along one or several dimensions, and pass fixed values for the
variables the selection leaves out. Linopy returns a window of the model:
the selected variables and constraints, with every term on a left-out
variable moved to the right-hand side.

This one operation serves several patterns:

* **Rolling-horizon optimization**: solve a long horizon one time slice at a
  time, carrying the end state into the next slice.
* **Spatial or sectoral decomposition**: solve one region or one sector with
  the exchange to its neighbours held fixed.
* **Fix-and-optimize heuristics**: reoptimize a part of the model while the
  rest stays at a known solution.
* **Decomposition subproblems**: condition a subproblem on master-problem
  values, as in Benders-type or ADMM-type schemes.

Three methods cover the workflow:

* :meth:`linopy.Model.restrict` returns a window of the model,
* :meth:`linopy.restrict.WindowModel.solve` solves it,
* :meth:`linopy.restrict.WindowModel.advance` moves it to a new selection.

A window is a row and column slice of the model's matrices. Creating or
moving it does not rebuild variables or constraints.

.. contents::
   :local:
   :depth: 2

Selecting a window
------------------

:meth:`~linopy.Model.restrict` follows :meth:`xarray.Dataset.sel` semantics:
every container's labels are selected along the dimensions it holds, containers
without them stay whole. Labels, lists, slices and scalars work, and several
dimensions can be selected at once, e.g.
``m.restrict(snapshot=slice(t0, t1), bus=["north"])``. The parent model is
unchanged.

Boundary values
---------------

A constraint in the window can reference a variable outside it.
:meth:`~linopy.Model.restrict` moves those terms to the right-hand side,
multiplied by their coefficient:

.. math::

    A_{WW}\, x_W \sim b_W - A_{WB}\, x_B

where :math:`x_W` are the window's variables, :math:`x_B` the variables
outside it, and :math:`x_B` on the right-hand side the values you pass in
``fix``. In words, each outside term adds ``coefficient * fixed value`` to the
right-hand side. Rows that lose every term are checked and dropped when they
are satisfied.

Without ``fix``, a window that cuts a coupling raises
:class:`~linopy.restrict.RestrictionError` naming the constraints. A window
with no outside terms needs no ``fix``.

``fix`` values are keyed by variable name, aligned with the variable by
coordinate label and broadcast over the dimensions they lack, so a
DataArray over the full model, a ``pandas.Series`` or a scalar works directly.
Every coordinate of the variable must be present, otherwise ``restrict``
raises. Values for in-window variables are ignored. A ``NaN`` at an outside
label counts as missing.

Example: one region of a network
--------------------------------

Two regions balance their demand with local generation and imports. The
exchange constraint couples the regions: what one imports, the other exports.

.. code-block:: python

    import pandas as pd
    import xarray as xr
    import linopy

    buses = pd.Index(["north", "south"], name="bus")
    sns = pd.RangeIndex(24, name="snapshot")
    demand = xr.DataArray([[30.0] * 24, [50.0] * 24], coords=[buses, sns])
    cost = xr.DataArray([1.0, 2.0], coords=[buses])

    m = linopy.Model()
    gen = m.add_variables(0, 100, coords=[buses, sns], name="gen")
    imports = m.add_variables(-20, 20, coords=[buses, sns], name="imports")
    m.add_constraints(gen + imports == demand, name="balance")
    m.add_constraints(imports.sum("bus") == 0, name="exchange")
    m.add_objective((cost * gen).sum())

    # solve the north with the south importing 15 in every snapshot
    south_imports = xr.DataArray([0.0, 15.0], coords=[buses])
    north = m.restrict(bus=["north"], fix={"imports": south_imports})
    north.solve(solver_name="highs")

The ``exchange`` constraint has no ``bus`` dimension, so it stays whole. Its
term on the south's imports moves to the right-hand side, which forces the
north to export 15.

Example: rolling horizon
------------------------

A long-horizon problem is often solved one time slice at a time: solve a slice
(say, one day), carry the end state into the next slice, then move forward.

.. code-block:: python

    import pandas as pd
    import xarray as xr
    import linopy

    sns = pd.date_range("2026-01-01", periods=8760, freq="h", name="snapshot")
    m = linopy.Model()
    soc = m.add_variables(0, 100, coords=[sns], name="soc")
    flow = m.add_variables(0, 1000, coords=[sns], name="flow")
    # the balance references the previous snapshot
    m.add_constraints(soc - 0.5 * soc.shift(snapshot=1) - flow <= 10, name="balance")
    m.add_objective(flow.sum(), sense="min")

    # initial state of charge over the full horizon
    initial = xr.DataArray([50.0] * 8760, dims=["snapshot"])

    window = m.restrict(snapshot=sns[:24], fix={"soc": initial})
    window.solve(solver_name="highs")

    for start in range(24, 8760 - 24, 24):
        window.advance(snapshot=sns[start : start + 24], fix={"soc": initial})
        window.solve(solver_name="highs")

        # containers are built when you read them
        solution = window.solution

In the first row of each window, the balance references the previous
snapshot's state of charge, which lies outside the window. That term moves to
the right-hand side with the value from ``fix``.

.. note::

   Folding uses the constraint's *actual* coefficient on the outside term.
   Hand-written rolling-horizon code often uses a coefficient of 1 for the
   initial state of charge, instead of the loss coefficient used on the other
   rows. ``restrict`` reproduces a slightly different model.

Working with a window
---------------------

:class:`~linopy.restrict.WindowModel` hands the solver the window's matrices
directly:

* :attr:`~linopy.restrict.WindowModel.matrices` holds ``A``, ``b``, ``sense``,
  ``c``, ``lb``, ``ub`` and the labels of its rows and columns.
* :attr:`~linopy.restrict.WindowModel.solution` and
  :attr:`~linopy.restrict.WindowModel.dual` are ``Dataset`` objects with one
  array per variable or constraint. They are built when you read them.
* :attr:`~linopy.restrict.WindowModel.objective` is the last solve's value.

Terms outside the window are dropped from the objective, so the window's value
differs from the full model's by a constant.

Moving the window
-----------------

:meth:`~linopy.restrict.WindowModel.advance` slices the matrices for a new
selection along the same dimensions and moves the new boundary values to the
right-hand side. When the new window's matrix has the same sparsity pattern,
the solver updates its existing model instead of building a new one, so a
chain of equal-sized windows keeps that fast path. A changed pattern, e.g. a
coefficient that is zero in one window only, rebuilds the solver model, or
raises :class:`~linopy.persistent.errors.RebuildRequiredError` with
``disallow_rebuild=True``.

Limits
------

Every container holding a selected dimension must contain the selected labels;
otherwise ``restrict`` raises :class:`~linopy.restrict.RestrictionError`. A
slice selects whatever labels each container holds within its bounds.

Indicators and quadratic objectives raise ``NotImplementedError``.

Native SOS constraints also raise: they are not part of the matrices.
Reformulate them first with :meth:`Model.apply_sos_reformulation`. A
cardinality row from the reformulation that spans the selected dimension
still cannot be windowed.

Piecewise formulations work; they expand into ordinary variables and constraints
when added.

A window is not a :class:`~linopy.Model`: it cannot take new variables or
constraints.
