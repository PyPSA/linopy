.. _piecewise-linear-constraints:

Piecewise Linear Constraints
============================

Piecewise linear (PWL) constraints approximate nonlinear functions as connected
linear segments, allowing you to model cost curves, efficiency curves, or
production functions within a linear programming framework.

.. contents::
   :local:
   :depth: 2


Quick Start
-----------

.. code-block:: python

    import linopy

    m = linopy.Model()
    power = m.add_variables(name="power", lower=0, upper=100)
    fuel = m.add_variables(name="fuel")

    # Link power and fuel via a piecewise linear curve
    m.add_piecewise_constraints(
        (power, [0, 30, 60, 100]),
        (fuel, [0, 36, 84, 170]),
    )

Each ``(expression, breakpoints)`` tuple pairs a variable with its
breakpoint values.  All tuples share interpolation weights, so at any
feasible point, every variable is interpolated between the *same* pair
of adjacent breakpoints.


API
---

``add_piecewise_constraints``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    m.add_piecewise_constraints(
        (expr1, breakpoints1),
        (expr2, breakpoints2),
        ...,
        method="auto",  # "auto", "sos2", or "incremental"
        active=None,  # binary variable to gate the constraint
        name=None,  # base name for generated variables/constraints
        skip_nan_check=False,
    )

Creates auxiliary variables and constraints that enforce all expressions
to lie exactly on the piecewise curve.  Requires a MIP or SOS2-capable
solver.

``tangent_lines``
~~~~~~~~~~~~~~~~~

.. code-block:: python

    t = linopy.tangent_lines(x, x_points, y_points)

Returns a :class:`~linopy.expressions.LinearExpression` with one tangent
line per segment.  **No variables are created** --- the result is pure
linear algebra.  Use it with regular ``add_constraints``:

.. code-block:: python

    t = linopy.tangent_lines(power, x_pts, y_pts)
    m.add_constraints(fuel <= t)  # upper bound
    m.add_constraints(fuel >= t)  # lower bound

``breakpoints`` and ``segments``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Factory functions that create DataArrays with the correct dimension names:

.. code-block:: python

    linopy.breakpoints([0, 50, 100])  # list
    linopy.breakpoints({"gen1": [0, 50], "gen2": [0, 80]}, dim="gen")  # per-entity
    linopy.breakpoints(slopes=[1.2, 1.4], x_points=[0, 30, 60], y0=0)  # from slopes
    linopy.segments([(0, 10), (50, 100)])  # disjunctive
    linopy.segments({"gen1": [(0, 10)], "gen2": [(0, 80)]}, dim="gen")  # per-entity


When to Use What
----------------

linopy provides two distinct tools for piecewise linear modelling.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - ``add_piecewise_constraints``
     - ``tangent_lines``
   * - **Constraint type**
     - Equality: :math:`y = f(x)`
     - Inequality: :math:`y \le f(x)` or :math:`y \ge f(x)`
   * - **Creates variables?**
     - Yes (lambdas, deltas, binaries)
     - No
   * - **Solver requirement**
     - MIP or SOS2-capable
     - Any LP solver
   * - **N-variable support**
     - Yes
     - No (2-variable only)

.. warning::

   ``tangent_lines`` does **not** work with equality.  Writing
   ``fuel == tangent_lines(...)`` creates one equality per segment,
   which is overconstrained (infeasible except at breakpoints).
   Use ``add_piecewise_constraints`` for equality.

**When is the tangent-line bound tight?**

- :math:`y \le f(x)` is tight when *f* is **concave** (slopes decrease)
- :math:`y \ge f(x)` is tight when *f* is **convex** (slopes increase)

For other combinations the bound is valid but loose (a relaxation).


Breakpoint Construction
-----------------------

From lists
~~~~~~~~~~

The simplest form --- pass Python lists directly in the tuple:

.. code-block:: python

    m.add_piecewise_constraints(
        (power, [0, 30, 60, 100]),
        (fuel, [0, 36, 84, 170]),
    )

With the ``breakpoints()`` factory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Equivalent, but explicit about the DataArray construction:

.. code-block:: python

    m.add_piecewise_constraints(
        (power, linopy.breakpoints([0, 30, 60, 100])),
        (fuel, linopy.breakpoints([0, 36, 84, 170])),
    )

From slopes
~~~~~~~~~~~

When you know marginal costs (slopes) rather than absolute values:

.. code-block:: python

    m.add_piecewise_constraints(
        (power, [0, 50, 100, 150]),
        (
            cost,
            linopy.breakpoints(
                slopes=[1.1, 1.5, 1.9], x_points=[0, 50, 100, 150], y0=0
            ),
        ),
    )
    # cost breakpoints: [0, 55, 130, 225]

Per-entity breakpoints
~~~~~~~~~~~~~~~~~~~~~~

Different generators can have different curves.  Pass a dict to
``breakpoints()`` with entity names as keys:

.. code-block:: python

    m.add_piecewise_constraints(
        (
            power,
            linopy.breakpoints(
                {"gas": [0, 30, 60, 100], "coal": [0, 50, 100, 150]}, dim="gen"
            ),
        ),
        (
            fuel,
            linopy.breakpoints(
                {"gas": [0, 40, 90, 180], "coal": [0, 55, 130, 225]}, dim="gen"
            ),
        ),
    )

Ragged lengths are NaN-padded automatically.  Breakpoints are
auto-broadcast over remaining dimensions (e.g. ``time``).

Disjunctive segments
~~~~~~~~~~~~~~~~~~~~~

For disconnected operating regions (e.g. forbidden zones), use
``segments()``:

.. code-block:: python

    m.add_piecewise_constraints(
        (power, linopy.segments([(0, 0), (50, 80)])),
        (cost, linopy.segments([(0, 0), (125, 200)])),
    )

The disjunctive formulation is selected automatically when breakpoints
have a segment dimension.

N-variable linking
~~~~~~~~~~~~~~~~~~

Link any number of variables through shared breakpoints.  All variables
are symmetric --- there is no distinguished "x" or "y":

.. code-block:: python

    m.add_piecewise_constraints(
        (power, [0, 30, 60, 100]),
        (fuel, [0, 40, 85, 160]),
        (heat, [0, 25, 55, 95]),
    )


Formulation Methods
-------------------

Pass ``method="auto"`` (the default) and linopy picks the best
formulation automatically:

- **All breakpoints monotonic** --- incremental
- **Otherwise** --- SOS2
- **Disjunctive** (segments) --- always SOS2 with binary selection

SOS2 (Convex Combination)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Works for any breakpoint ordering.  Introduces interpolation weights
:math:`\lambda_i` with an SOS2 adjacency constraint:

.. math::

   &\sum_{i=0}^{n} \lambda_i = 1, \qquad
   \text{SOS2}(\lambda_0, \ldots, \lambda_n)

   &e_j = \sum_{i=0}^{n} \lambda_i \, B_{j,i}
   \quad \text{for each expression } j

The SOS2 constraint ensures at most two adjacent :math:`\lambda_i` are
non-zero, so every expression is interpolated within the same segment.

.. note::

   SOS2 is handled via branch-and-bound, similar to integer variables.
   Prefer ``method="incremental"`` when breakpoints are monotonic.

.. code-block:: python

    m.add_piecewise_constraints((power, xp), (fuel, yp), method="sos2")

Incremental (Delta) Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **strictly monotonic** breakpoints.  Uses fill-fraction variables
:math:`\delta_i` with binary indicators --- no SOS2 needed:

.. math::

   &\delta_i \in [0, 1], \quad \delta_{i+1} \le \delta_i

   &e_j = B_{j,0} + \sum_{i=1}^{n} \delta_i \, (B_{j,i} - B_{j,i-1})

.. code-block:: python

    m.add_piecewise_constraints((power, xp), (fuel, yp), method="incremental")

**Limitation:** All breakpoint sequences must be strictly monotonic.

Disjunctive (Disaggregated Convex Combination)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **disconnected segments** (gaps between operating regions).  Binary
indicators :math:`z_k` select exactly one segment; SOS2 applies within it:

.. math::

   &z_k \in \{0, 1\}, \quad \sum_{k} z_k = 1

   &\sum_{i} \lambda_{k,i} = z_k, \qquad
   e_j = \sum_{k} \sum_{i} \lambda_{k,i} \, B_{j,k,i}

No big-M constants are needed, giving a tight LP relaxation.

Tangent lines
~~~~~~~~~~~~~

For inequality bounds.  Computes one tangent line per segment:

.. math::

   \text{tangent}_k(x) = m_k \cdot x + c_k

where :math:`m_k` is the slope and :math:`c_k` the intercept of
segment :math:`k`.  Returns a ``LinearExpression`` --- no variables
created.

.. code-block:: python

    t = linopy.tangent_lines(power, x_pts, y_pts)
    m.add_constraints(fuel <= t)


Advanced Features
-----------------

Active parameter (unit commitment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``active`` parameter gates the piecewise function with a binary
variable.  When ``active=0``, all auxiliary variables (and thus the
linked expressions) are forced to zero:

.. code-block:: python

    commit = m.add_variables(name="commit", binary=True, coords=[time])
    m.add_piecewise_constraints(
        (power, [30, 60, 100]),
        (fuel, [40, 90, 170]),
        active=commit,
    )

- ``commit=1``: power operates in [30, 100], fuel = f(power)
- ``commit=0``: power = 0, fuel = 0

Auto-broadcasting
~~~~~~~~~~~~~~~~~

Breakpoints are automatically broadcast to match expression dimensions.
You don't need ``expand_dims``:

.. code-block:: python

    time = pd.Index([1, 2, 3], name="time")
    x = m.add_variables(name="x", lower=0, upper=100, coords=[time])
    y = m.add_variables(name="y", coords=[time])

    # 1D breakpoints auto-expand to match x's time dimension
    m.add_piecewise_constraints((x, [0, 50, 100]), (y, [0, 70, 150]))

NaN masking
~~~~~~~~~~~

Trailing NaN values in breakpoints mask the corresponding lambda/delta
variables.  This is useful for per-entity breakpoints with ragged
lengths:

.. code-block:: python

    # gen1 has 3 breakpoints, gen2 has 2 (NaN-padded)
    bp = linopy.breakpoints({"gen1": [0, 50, 100], "gen2": [0, 80]}, dim="gen")

Interior NaN values (gaps in the middle) are not supported and raise
an error.

Generated variables and constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given base name ``name``:

**SOS2:**
``{name}_lambda`` (variable), ``{name}_convex`` (constraint),
``{name}_x_link`` (constraint)

**Incremental:**
``{name}_delta`` (variable), ``{name}_inc_binary`` (variable),
``{name}_fill`` (constraint), ``{name}_x_link`` (constraint)

**Disjunctive:**
``{name}_binary`` (variable), ``{name}_select`` (constraint),
``{name}_lambda`` (variable), ``{name}_convex`` (constraint),
``{name}_x_link`` (constraint)


See Also
--------

- :doc:`piecewise-linear-constraints-tutorial` --- Worked examples (notebook)
- :doc:`sos-constraints` --- Low-level SOS1/SOS2 constraint API
