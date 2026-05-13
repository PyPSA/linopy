.. _user-guide:

User Guide
==========

This guide takes you from a working install to building, modifying, and
solving real models. Each page is a runnable Jupyter notebook — read it
top to bottom, or use it as a reference once you know what you're
looking for.

If you haven't yet, run through the :doc:`Getting Started <prerequisites>`
section first: it installs linopy and a solver, then walks through a
first scalar model and the move to N-D variables with coordinates.


Core building blocks
--------------------

The four notebooks below cover the model object users interact with
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


Working with an existing model
------------------------------

Once you've built a model, you'll often want to inspect it, change a
bound, swap a constraint, or copy it for what-if analysis.

- :doc:`manipulating-models` — modifying or removing variables and
  constraints in place; ``Model.copy()``; ``fix`` / ``relax`` for
  variables.


Where to go next
----------------

- **Advanced features** — :doc:`sos-constraints`,
  :doc:`piecewise-linear-constraints`, and the
  :doc:`testing-framework` for asserting structural properties of a
  model.
- **Tutorials** — end-to-end problem walkthroughs:
  :doc:`transport-tutorial`, :doc:`migrating-from-pyomo`.
- **Solving** — :doc:`solve-on-remote` (SSH),
  :doc:`solve-on-oetc` (OET Cloud), :doc:`gpu-acceleration` (cuPDLPx).
- **Troubleshooting** — :doc:`infeasible-model` (diagnosing infeasible
  problems), :doc:`gurobi-double-logging` (and other solver quirks).
- **Reference** — the full :doc:`api` listing.
