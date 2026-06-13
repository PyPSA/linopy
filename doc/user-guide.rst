.. _user-guide:

Overview
========

In :doc:`Getting Started <prerequisites>` you installed linopy, built
a first scalar model, and saw N-D variables on coordinates. The User
Guide reopens each of those pieces in depth and adds the rest of the
modelling surface.

Each page is a runnable Jupyter notebook — read it top to bottom, or
use it as a reference once you know what you're looking for.


Core building blocks
--------------------

The four notebooks below cover the model object you'll interact with
most. Read them in order the first time; come back to them whenever
you're unsure what a particular operator or argument does.

- :doc:`creating-variables` — declaring decision variables, with bounds
  and coordinates. Continuous, integer, binary, and semi-continuous.
- :doc:`creating-expressions` — combining variables into linear (and
  quadratic) expressions; arithmetic, broadcasting, ``sum``,
  ``groupby``, ``rolling``, ``where``.
- :doc:`creating-constraints` — turning expressions into ``≤`` / ``≥``
  / ``==`` constraints, and the ``CSRConstraint`` memory-efficient
  alternative.
- :doc:`coordinate-alignment` — how linopy lines up operands that live
  on different coordinates, and how to control it with ``join``.

After these four you can build any LP/MIP/QP linopy supports.


Numerical scaling
-----------------

Large or very small coefficients can make mathematical programs harder
for solvers to process. Linopy therefore accepts optional positive,
finite ``scaling`` factors when adding variables, constraints, and the
objective.

For continuous and semi-continuous variables, ``scaling=s`` means the
solver-side column represents ``s * x``. Bounds are multiplied by ``s``,
and coefficients that reference ``x`` are divided by ``s``. Binary and
integer variable scaling is stored and round-tripped, but their solver
columns remain ordinary discrete columns.

For constraints, ``scaling=s`` divides both the left-hand-side
coefficients and the right-hand side by ``s``. For objectives,
``scaling=s`` divides linear and quadratic objective coefficients by
``s``. Primal values, dual values, and objective values are transformed
back to the original user units after solving.

.. code-block:: python

   x = m.add_variables(lower=0, scaling=1e3, name="x")
   m.add_constraints(2 * x >= 10, scaling=10, name="demand")
   m.add_objective(5 * x, scaling=100)


Working with an existing model
------------------------------

Once you've built a model, you'll often want to inspect it, change a
bound, swap a constraint, or copy it for what-if analysis.

- :doc:`manipulating-models` — modifying or removing variables and
  constraints in place; ``Model.copy()``; ``fix`` / ``relax`` for
  variables.


Where to go next
----------------

- **Examples** — end-to-end problem walkthroughs:
  :doc:`transport-tutorial`, :doc:`migrating-from-pyomo`.
- **Advanced features** — :doc:`sos-constraints`,
  :doc:`piecewise-linear-constraints`, and the
  :doc:`testing-framework` for asserting structural properties of a
  model.
- **Solving** — :doc:`solve-on-remote` (SSH),
  :doc:`solve-on-oetc` (OET Cloud), :doc:`gpu-acceleration` (cuPDLPx).
- **Troubleshooting** — :doc:`infeasible-model` (diagnosing infeasible
  problems), :doc:`gurobi-double-logging` (and other solver quirks).
- **Reference** — the full :doc:`api` listing.
