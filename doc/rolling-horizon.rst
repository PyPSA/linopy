.. _rolling-horizon:

Rolling-Horizon Optimization
============================

A long-horizon problem is often solved one window at a time: solve a slice of
the horizon (say, one day), carry the end state into the next window, then move
the window forward. Linopy has three methods for this:

* :meth:`linopy.Model.restrict` returns a window of the model,
* :meth:`linopy.restrict.WindowModel.solve` solves it,
* :meth:`linopy.restrict.WindowModel.advance` moves it forward.

A window is a row and column slice of the model's matrices. Moving it does not
rebuild variables or constraints.

.. contents::
   :local:
   :depth: 2

The rolling-horizon recipe
--------------------------

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

:meth:`~linopy.Model.restrict` follows :meth:`xarray.Dataset.sel` semantics:
every container holding the selected dimension is sliced, containers without it
stay whole. The parent model is unchanged.

Boundary values
---------------

The balance constraint references the previous snapshot's state of charge. In
the first row of a window, that reference points outside the window.
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

.. note::

   Folding uses the constraint's *actual* coefficient on the outside term.
   Hand-written rolling-horizon code often uses a coefficient of 1 for the
   initial state of charge, instead of the loss coefficient used on the other
   rows. ``restrict`` reproduces a slightly different model.

``fix`` values are keyed by variable name and broadcast over the variable's
coordinates, so a full-horizon DataArray works directly. Values for in-window
variables are ignored. A ``NaN`` at an outside label counts as missing.

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

:meth:`~linopy.restrict.WindowModel.advance` slices the matrices for the new
selection and moves the new boundary values to the right-hand side. When the
new window has the same shape, the solver updates its existing model instead of
building a new one. Nothing is rebuilt, so a chain of equal-width windows
keeps that fast path.

Limits
------

A window slices one dimension. Every container holding that dimension must
contain the window's labels; otherwise ``restrict`` raises.

Several selected dimensions, indicators and quadratic objectives raise
:class:`~linopy.restrict.RestrictionError` or ``NotImplementedError``.

Native SOS constraints also raise: they are not part of the matrices. Reformulate
them first with :meth:`Model.apply_sos_reformulation`, though the cardinality row
that adds spans the horizon and still cannot be windowed.

Piecewise formulations work; they expand into ordinary variables and constraints
when added.
