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
