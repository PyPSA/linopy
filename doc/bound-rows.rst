.. _bound-rows:

Constraints as Variable Bounds
==============================

Many models write variable bounds as constraints, one variable per row,
e.g. ``p <= p_nom`` for every generator and snapshot. As rows they enlarge
the matrix handed to the solver, the LP files and the presolve. With
``as_bound=True``, linopy keeps such a constraint in the model as usual, but
lowers its rows to column bounds instead of matrix rows:

.. code-block:: python

    m.add_constraints(p <= p_nom, name="p-upper", as_bound=True)
    m.add_constraints(p >= p_min, name="p-lower", as_bound=True)

Every active row must hold exactly one variable term ``c * x`` with a
nonzero coefficient, otherwise ``add_constraints`` raises a ``ValueError``.
When the model is lowered, in :attr:`linopy.Model.matrices`, the direct
solver APIs and the LP and MPS writers, a row ``c * x (sign) b`` tightens the
bounds of ``x`` to ``b / c`` (the direction flips for ``c < 0``; ``=`` sets
both bounds). Several rows on one variable and the variable's own bounds
combine to the tightest bound. The constraint does not appear in
``m.matrices.clabels``.

Duals
-----

The duals of these rows are recovered after the solve and stored in
``Constraint.dual`` as for any other constraint. The solver reports duals
for the remaining rows only; linopy rebuilds the reduced cost of each
column as ``d = c + Q x - A^T y`` in user units and assigns ``d / c`` to the
row that sets the active bound, the side being given by the sign of ``d``
and the objective sense. Every other row on the column is not binding and
gets a dual of 0. This is the dual the row would get as an ordinary
constraint, with the same sign convention, for variable, constraint and
objective scaling alike.

Ties are resolved in favour of rows: when a row and the variable's own
bound give the same value, the row is reported as binding; between equal
rows, the first one added. In a degenerate optimum the dual of an ordinary
row is not unique either, so in such cases the reported dual can differ
from the one the solver would pick for an ordinary row.

Dual recovery has been verified against ordinary rows for HiGHS and Gurobi,
with the direct API and LP and MPS files, for minimisation and
maximisation, linear and quadratic objectives and with scaling. For other
solvers, the duals of rows lowered to bounds are set to NaN with a warning.
Mixed-integer problems report no duals, as usual.

Limitations
-----------

* ``as_bound=True`` cannot be combined with ``penalty`` or
  :meth:`linopy.constraints.Constraint.soften`.
* Rows on semi-continuous variables raise a ``ValueError`` when the model
  is lowered, since a bound changes their semantics.
* A constraint modified later is checked again when the model is lowered;
  a row with more than one term then raises.
* A solver built with ``track_updates=True`` rebuilds the model on every
  update instead of updating it in place
  (``RebuildReason.BOUND_ROWS``).
* :meth:`linopy.Model.dualize` treats these constraints as ordinary rows.
* In a :ref:`restricted window <restrict>` the rows act as bounds but their
  duals are not recovered.
* Infeasibility analysis (:meth:`linopy.Model.compute_infeasibilities`)
  only reports matrix rows, so a row lowered to a bound never appears in an
  IIS; the conflicting variable bound does.
