.. _piecewise-linear-constraints:

Piecewise Linear Constraints
============================

Piecewise linear (PWL) constraints approximate nonlinear functions as connected
linear segments, allowing you to model cost curves, efficiency curves, or
production functions within a linear programming framework.

linopy offers two tools:

- :py:meth:`~linopy.model.Model.add_piecewise_constraints` ---
  exact equality on the piecewise curve (creates auxiliary variables).
- :func:`~linopy.piecewise.tangent_lines` ---
  one-sided bounds via tangent lines (pure LP, no auxiliary variables).

.. contents::
   :local:
   :depth: 2


Equality vs Inequality
----------------------

``add_piecewise_constraints`` --- exact equality on the curve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this when variables must lie **exactly on** the piecewise curve
(:math:`y = f(x)`).  It creates auxiliary variables (lambda weights or
delta fractions) and combinatorial constraints (SOS2 or binary indicators)
to enforce that the operating point is interpolated between adjacent
breakpoints.

.. code-block:: python

    m.add_piecewise_constraints(
        (power, [0, 30, 60, 100]),
        (fuel, [0, 36, 84, 170]),
    )

This is the only way to enforce exact piecewise equality.  It requires
a MIP or SOS2-capable solver.

``tangent_lines`` --- one-sided bound, pure LP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this when a variable must be **bounded above or below** by the
piecewise curve (:math:`y \le f(x)` or :math:`y \ge f(x)`).  It
computes one tangent line per segment and returns them as a regular
:class:`~linopy.expressions.LinearExpression` with a segment dimension.
**No auxiliary variables are created.**

.. code-block:: python

    t = linopy.tangent_lines(power, x_pts, y_pts)
    m.add_constraints(fuel <= t)  # fuel bounded above by f(power)
    m.add_constraints(fuel >= t)  # fuel bounded below by f(power)

xarray broadcasting creates one linear constraint per segment per
coordinate entry.  The result is solvable by **any LP solver** ---
no SOS2, no binaries.

.. warning::

   ``tangent_lines`` does **not** work with equality.  Writing
   ``fuel == tangent_lines(...)`` would require fuel to simultaneously
   satisfy every tangent line, which is infeasible except at breakpoints.
   Use ``add_piecewise_constraints`` for equality.

**When is the bound tight?** The tangent-line bound is exact (tight at
every point on the curve) when the function has the right convexity:

- :math:`y \le f(x)` is tight when *f* is **concave** (slopes decrease)
- :math:`y \ge f(x)` is tight when *f* is **convex** (slopes increase)

For other combinations the bound is valid but loose (a relaxation).


Overview
--------

``add_piecewise_constraints`` takes ``(expression, breakpoints)`` tuples as
positional arguments.  All tuples share the same interpolation weights,
coupling the expressions on the same curve segment.

**2 variables:**

.. code-block:: python

    m.add_piecewise_constraints(
        (power, [0, 30, 60, 100]),
        (fuel, [0, 36, 84, 170]),
    )

**N variables (e.g. CHP plant):**

.. code-block:: python

    m.add_piecewise_constraints(
        (power, [0, 30, 60, 100]),
        (fuel, [0, 40, 85, 160]),
        (heat, [0, 25, 55, 95]),
    )


Mathematical Background
-----------------------

Core formulation
~~~~~~~~~~~~~~~~

The piecewise linear formulation links *N* expressions
:math:`e_1, e_2, \ldots, e_N` through a shared set of breakpoints.

Given :math:`n+1` breakpoints :math:`B_{j,0}, B_{j,1}, \ldots, B_{j,n}` for
each expression :math:`j`, the SOS2 formulation introduces interpolation
weights :math:`\lambda_i \in [0, 1]`:

.. math::

   &\sum_{i=0}^{n} \lambda_i = 1
   \qquad \text{(convexity)}

   &e_j = \sum_{i=0}^{n} \lambda_i \, B_{j,i}
   \qquad \text{for each expression } j
   \qquad \text{(linking)}

   &\text{SOS2}(\lambda_0, \lambda_1, \ldots, \lambda_n)
   \qquad \text{(adjacency)}

The SOS2 constraint ensures at most two *adjacent* :math:`\lambda_i` are
non-zero, so every expression is interpolated within the same segment.  All
expressions share the same :math:`\lambda` weights, which is what couples them.


Tangent lines (inequality)
~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~linopy.piecewise.tangent_lines` computes the tangent line for
each segment of the piecewise function:

.. math::

   \text{tangent}_k(x) = m_k \cdot x + c_k \quad \text{for each segment } k

where :math:`m_k = (y_{k+1} - y_k) / (x_{k+1} - x_k)` is the slope and
:math:`c_k = y_k - m_k \cdot x_k` is the intercept.  The result is a
:class:`~linopy.expressions.LinearExpression` with a segment dimension ---
one linear expression per segment, no auxiliary variables.


Formulation Methods
-------------------

SOS2 (Convex Combination)
~~~~~~~~~~~~~~~~~~~~~~~~~

The default formulation, using Special Ordered Sets of type 2.  Works for any
breakpoint ordering.

.. note::

   SOS2 is a combinatorial constraint handled via branch-and-bound.
   Prefer ``method="incremental"`` or ``method="auto"`` when breakpoints are
   monotonic.

Incremental (Delta) Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **strictly monotonic** breakpoints :math:`b_0 < b_1 < \cdots < b_n`, the
incremental formulation uses fill-fraction variables:

.. math::

   \delta_i \in [0, 1], \quad
   \delta_{i+1} \le \delta_i, \quad
   e_j = B_{j,0} + \sum_{i=1}^{n} \delta_i \, (B_{j,i} - B_{j,i-1})

Binary indicators enforce segment ordering.  This avoids SOS2 constraints
entirely, using only standard MIP constructs.

**Limitation:** All breakpoint sequences must be strictly monotonic.

Disjunctive (Disaggregated Convex Combination)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **disconnected segments** (with gaps), binary indicators select exactly one
segment and SOS2 applies within it.  No big-M constants are needed.

.. math::

   z_k \in \{0, 1\}, \quad \sum_{k} z_k = 1, \quad
   \sum_{i} \lambda_{k,i} = z_k, \quad
   e_j = \sum_{k} \sum_{i} \lambda_{k,i} \, B_{j,k,i}


.. _choosing-a-formulation:

Choosing a Formulation
~~~~~~~~~~~~~~~~~~~~~~

Pass ``method="auto"`` (the default) and linopy picks the best formulation:

- **All breakpoints monotonic** -> incremental
- Otherwise -> SOS2
- Disjunctive (segments) -> always SOS2 with binary selection
- **Inequality** -> use ``tangent_lines`` + regular constraints


Usage Examples
--------------

2-variable equality
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    m.add_piecewise_constraints(
        (power, linopy.breakpoints([0, 30, 60, 100])),
        (fuel, linopy.breakpoints([0, 36, 84, 170])),
    )

N-variable linking
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    m.add_piecewise_constraints(
        (power, [0, 30, 60, 100]),
        (fuel, [0, 40, 85, 160]),
        (heat, [0, 25, 55, 95]),
    )

Inequality via tangent lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    t = linopy.tangent_lines(power, x_pts, y_pts)
    m.add_constraints(fuel <= t)  # upper bound (concave function)
    m.add_constraints(fuel >= t)  # lower bound (convex function)

Disjunctive (disconnected segments)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    m.add_piecewise_constraints(
        (x, linopy.segments([(0, 10), (50, 100)])),
        (y, linopy.segments([(0, 15), (60, 130)])),
    )

Choosing a method
~~~~~~~~~~~~~~~~~

.. code-block:: python

    m.add_piecewise_constraints((x, xp), (y, yp), method="sos2")
    m.add_piecewise_constraints((x, xp), (y, yp), method="incremental")
    m.add_piecewise_constraints((x, xp), (y, yp), method="auto")  # default

Active parameter (unit commitment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``active`` parameter gates the piecewise function with a binary variable.
When ``active=0``, all auxiliary variables are forced to zero.

.. code-block:: python

    commit = m.add_variables(name="commit", binary=True, coords=[time])
    m.add_piecewise_constraints(
        (power, x_pts),
        (fuel, y_pts),
        active=commit,
    )


Breakpoints and Segments Factories
-----------------------------------

:func:`~linopy.piecewise.breakpoints` creates DataArrays with the correct
``_breakpoint`` dimension.  Accepts lists, Series, DataFrames, dicts, or
DataArrays:

.. code-block:: python

    linopy.breakpoints([0, 50, 100])  # from list
    linopy.breakpoints({"gen1": [0, 50], "gen2": [0, 80]}, dim="gen")  # per-entity
    linopy.breakpoints(slopes=[1.2, 1.4], x_points=[0, 30, 60], y0=0)  # from slopes

:func:`~linopy.piecewise.segments` creates DataArrays with both ``_segment``
and ``_breakpoint`` dimensions for disjunctive formulations:

.. code-block:: python

    linopy.segments([(0, 10), (50, 100)])  # from list
    linopy.segments({"gen1": [(0, 10)], "gen2": [(0, 80)]}, dim="gen")  # per-entity


Auto-broadcasting
-----------------

Breakpoints are automatically broadcast to match expression dimensions.
You don't need ``expand_dims`` when your variables have extra dimensions:

.. code-block:: python

    time = pd.Index([1, 2, 3], name="time")
    x = m.add_variables(name="x", lower=0, upper=100, coords=[time])
    y = m.add_variables(name="y", coords=[time])

    # 1D breakpoints auto-expand to match x's time dimension
    m.add_piecewise_constraints((x, [0, 50, 100]), (y, [0, 70, 150]))


Generated Variables and Constraints
------------------------------------

Given base name ``name``, the following objects are created:

**SOS2 method:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Name
     - Type
     - Description
   * - ``{name}_lambda``
     - Variable
     - Interpolation weights :math:`\lambda_i \in [0, 1]` (SOS2).
   * - ``{name}_convex``
     - Constraint
     - :math:`\sum_i \lambda_i = 1`.
   * - ``{name}_x_link``
     - Constraint
     - Linking: :math:`e_j = \sum_i \lambda_i \, B_{j,i}` for all expressions.

**Incremental method:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Name
     - Type
     - Description
   * - ``{name}_delta``
     - Variable
     - Fill-fraction variables :math:`\delta_i \in [0, 1]`.
   * - ``{name}_inc_binary``
     - Variable
     - Binary indicators for each segment.
   * - ``{name}_fill``
     - Constraint
     - :math:`\delta_{i+1} \le \delta_i` (fill order).
   * - ``{name}_x_link``
     - Constraint
     - Linking: :math:`e_j = B_{j,0} + \sum_i \delta_i \, \Delta B_{j,i}`.

**Disjunctive method:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Name
     - Type
     - Description
   * - ``{name}_binary``
     - Variable
     - Segment indicators :math:`z_k \in \{0, 1\}`.
   * - ``{name}_select``
     - Constraint
     - :math:`\sum_k z_k = 1`.
   * - ``{name}_lambda``
     - Variable
     - Per-segment interpolation weights (SOS2).
   * - ``{name}_convex``
     - Constraint
     - :math:`\sum_i \lambda_{k,i} = z_k`.
   * - ``{name}_x_link``
     - Constraint
     - :math:`e_j = \sum_k \sum_i \lambda_{k,i} \, B_{j,k,i}`.


See Also
--------

- :doc:`piecewise-linear-constraints-tutorial` -- Worked examples (notebook)
- :doc:`sos-constraints` -- Low-level SOS1/SOS2 constraint API
