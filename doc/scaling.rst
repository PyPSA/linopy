.. _scaling:

Pre-solve Scaling
=================

Linopy can rescale your model before handing it to a solver. Scaling can
improve numerical robustness when constraint coefficients or bounds span
different orders of magnitude.

How it works
------------

- **Row scaling** (default) rescales constraint rows by their largest (or
  RMS) coefficient and applies the same factor to the RHS and duals.
- **Column scaling** (optional) rescales continuous variables so column
  norms are close to 1. Bounds, objective coefficients, and primal values
  are adjusted accordingly. Integer/binary variables stay unscaled by default
  to preserve integrality.
- Scaling is undone automatically on primal/dual values and the objective
  before they are stored on the model.

API
---

Enable scaling via ``Model.solve(scale=...)``:

.. code-block:: python

    import pandas as pd
    import linopy
    from linopy.scaling import ScaleOptions

    m = linopy.Model()

    hours = pd.RangeIndex(24, name="h")
    p = m.add_variables(lower=0, name="prod", coords=[hours])

    # Coefficients with very different magnitudes
    m.add_constraints(1e3 * p <= 5e4, name="capacity")
    m.add_constraints(0.01 * p.sum() >= 10, name="energy")

    # Default: row scaling using max-norm
    m.solve(scale=True)

    # Custom: row+column scaling on continuous variables, RMS norm
    m.solve(
        scale=ScaleOptions(
            enabled=True,
            method="row-l2",
            variable_scaling=True,
            scale_integer_variables=False,
        )
    )

Options
-------

``scale`` accepts:

- ``False`` / ``None``: disable scaling (default behavior prior to this feature)
- ``True``: enable row scaling with max-norm
- ``"row-max"`` or ``"row-l2"``: select norm for row scaling
- ``ScaleOptions``: full control (row/column scaling, integer handling,
  target magnitude, zero-floor)

Notes
-----

- Remote execution currently disables scaling.
- Many solvers also perform internal scaling; Linopy’s scaling is optional
  and aims to help when you need deterministic pre-conditioning on the
  problem you pass to the solver.
