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

linopy provides two distinct tools for piecewise linear modelling.
Understanding when to use which is the key design decision.

``add_piecewise_constraints`` — exact equality on the curve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this when variables must lie **exactly on** the piecewise curve
(:math:`y = f(x)`).  It creates auxiliary variables (lambda weights or
delta fractions) and combinatorial constraints (SOS2 or binary indicators)
to enforce that the operating point is interpolated between adjacent
breakpoints.

.. code-block:: python

    m.add_piecewise_constraints(
        x=power,
        y=fuel,
        x_points=x_pts,
        y_points=y_pts,
    )

This is the only way to enforce exact piecewise equality.  It requires
a MIP or SOS2-capable solver.

``tangent_lines`` — one-sided bound, pure LP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

``add_piecewise_constraints`` supports two calling conventions:

**N-variable (general form):** Link any number of expressions through shared
breakpoints.  All expressions are symmetric --- they are jointly constrained to
lie on the interpolated breakpoint curve.

.. code-block:: python

    m.add_piecewise_constraints(
        exprs={"power": power, "fuel": fuel, "heat": heat},
        breakpoints=bp,
    )

**2-variable (convenience form):** A shorthand for linking two expressions
``x`` and ``y`` via separate x/y breakpoints.

.. code-block:: python

    m.add_piecewise_constraints(
        x=power,
        y=fuel,
        x_points=x_pts,
        y_points=y_pts,
    )


Mathematical Background
-----------------------

Core formulation (N-variable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The general piecewise linear formulation links *N* expressions
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

**Example:** A CHP plant with fuel input, electrical output, and heat output at
four operating points:

.. code-block:: python

    bp = linopy.breakpoints(
        {"fuel": [0, 50, 120, 200], "power": [0, 15, 50, 100], "heat": [0, 25, 45, 55]},
        dim="var",
    )
    m.add_piecewise_constraints(
        exprs={"fuel": fuel, "power": power, "heat": heat},
        breakpoints=bp,
    )

At any feasible point, fuel, power, and heat are interpolated between the
*same* pair of adjacent breakpoints.


2-variable case: equality
~~~~~~~~~~~~~~~~~~~~~~~~~

The 2-variable equality constraint :math:`y = f(x)` is the most common use
case.  Mathematically, it is equivalent to the N-variable form with two
expressions:

.. math::

   x = \sum_i \lambda_i \, x_i, \qquad
   y = \sum_i \lambda_i \, y_i, \qquad
   \sum_i \lambda_i = 1

Internally, the 2-variable equality form builds a dict and delegates to the
same N-variable code path.

.. code-block:: python

    # These two are equivalent:
    m.add_piecewise_constraints(x=x, y=y, x_points=xp, y_points=yp)

    m.add_piecewise_constraints(
        exprs={"x": x, "y": y},
        breakpoints=linopy.breakpoints({"x": xp, "y": yp}, dim="var"),
    )


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

The user then adds their own constraint:

.. code-block:: python

    t = linopy.tangent_lines(power, x_pts, y_pts)
    m.add_constraints(fuel <= t)  # one constraint per segment per timestep


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

**Limitation:** Breakpoints must be strictly monotonic along the breakpoint
dimension.

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

- **Equality + monotonic breakpoints** -> incremental
- Otherwise -> SOS2
- Disjunctive (segments) -> always SOS2 with binary selection
- **Inequality** -> use ``tangent_lines`` + regular constraints

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20

   * - Property
     - SOS2
     - Incremental
     - Disjunctive
   * - Segments
     - Connected
     - Connected
     - Disconnected
   * - Constraint type
     - Equality
     - Equality
     - Equality
   * - Breakpoint order
     - Any
     - Strictly monotonic
     - Any (per segment)
   * - Variable types
     - Continuous + SOS2
     - Continuous + binary
     - Binary + SOS2
   * - N-variable support
     - Yes
     - Yes
     - 2-var only


Usage Examples
--------------

2-variable equality
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    m.add_piecewise_constraints(
        x=power,
        y=fuel,
        x_points=linopy.breakpoints([0, 30, 60, 100]),
        y_points=linopy.breakpoints([0, 36, 84, 170]),
    )

Inequality via envelope
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # fuel <= f(power): y bounded above (concave function)
    envelope = linopy.tangent_lines(power, x_pts, y_pts)
    m.add_constraints(fuel <= envelope)

    # fuel >= f(power): y bounded below (convex function)
    envelope = linopy.tangent_lines(power, x_pts, y_pts)
    m.add_constraints(fuel >= envelope)

N-variable linking
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    bp = linopy.breakpoints(
        {"power": [0, 30, 60, 100], "fuel": [0, 40, 85, 160], "heat": [0, 25, 55, 95]},
        dim="var",
    )
    m.add_piecewise_constraints(
        exprs={"power": power, "fuel": fuel, "heat": heat},
        breakpoints=bp,
    )

Disjunctive (disconnected segments)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    x_seg = linopy.segments([(0, 10), (50, 100)])
    y_seg = linopy.segments([(0, 15), (60, 130)])

    m.add_piecewise_constraints(
        x=x,
        y=y,
        x_points=x_seg,
        y_points=y_seg,
    )

Choosing a method
~~~~~~~~~~~~~~~~~

.. code-block:: python

    m.add_piecewise_constraints(x=x, y=y, x_points=xp, y_points=yp, method="sos2")
    m.add_piecewise_constraints(
        x=x, y=y, x_points=xp, y_points=yp, method="incremental"
    )
    m.add_piecewise_constraints(
        x=x, y=y, x_points=xp, y_points=yp, method="auto"
    )  # default

Active parameter (unit commitment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``active`` parameter gates the piecewise function with a binary variable.
When ``active=0``, all auxiliary variables are forced to zero.

.. code-block:: python

    commit = m.add_variables(name="commit", binary=True, coords=[time])
    m.add_piecewise_constraints(
        x=power,
        y=fuel,
        x_points=x_pts,
        y_points=y_pts,
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
    m.add_piecewise_constraints(x=x, y=y, x_points=[0, 50, 100], y_points=[0, 70, 150])


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
