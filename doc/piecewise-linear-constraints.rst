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

**Equality — lock variables onto the piecewise curve:**

.. code-block:: python

    import linopy

    m = linopy.Model()
    power = m.add_variables(name="power", lower=0, upper=100)
    fuel = m.add_variables(name="fuel")

    # fuel = f(power) on the piecewise curve defined by these breakpoints
    m.add_piecewise_formulation(
        (power, [0, 30, 60, 100]),
        (fuel, [0, 36, 84, 170]),
    )

**Inequality — bound one expression by the curve:**

.. code-block:: python

    # fuel <= f(power).  "auto" picks the cheapest correct formulation
    # (pure LP with chord constraints when the curve's curvature matches
    # the requested sign; SOS2/incremental otherwise).
    m.add_piecewise_formulation(
        (fuel, [0, 20, 30, 35]),  # bounded output listed FIRST
        (power, [0, 10, 20, 30]),  # input always on the curve
        sign="<=",
    )

Each ``(expression, breakpoints)`` tuple pairs a variable with its breakpoint
values.  All tuples share interpolation weights, so at any feasible point every
variable corresponds to the *same* point on the piecewise curve.


API
---

``add_piecewise_formulation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    m.add_piecewise_formulation(
        (expr1, breakpoints1),
        (expr2, breakpoints2),
        ...,
        sign="==",  # "==", "<=", or ">="
        method="auto",  # "auto", "sos2", "incremental", or "lp"
        active=None,  # binary variable to gate the constraint
        name=None,  # base name for generated variables/constraints
    )

Creates auxiliary variables and constraints that enforce either an equality
(``sign="=="``, default) or a one-sided bound (``sign="<="`` / ``">="``) of the
first expression by the piecewise function of the rest.

``breakpoints`` and ``segments``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Factory functions that create DataArrays with the correct dimension names:

.. code-block:: python

    linopy.breakpoints([0, 50, 100])  # list
    linopy.breakpoints({"gen1": [0, 50], "gen2": [0, 80]}, dim="gen")  # per-entity
    linopy.breakpoints(slopes=[1.2, 1.4], x_points=[0, 30, 60], y0=0)  # from slopes
    linopy.segments([(0, 10), (50, 100)])  # disjunctive
    linopy.segments({"gen1": [(0, 10)], "gen2": [(0, 80)]}, dim="gen")


The ``sign`` parameter — equality vs inequality
------------------------------------------------

The ``sign`` argument of ``add_piecewise_formulation`` chooses whether all
expressions are locked onto the curve or whether the first one is bounded:

- ``sign="=="`` (default): every expression lies *exactly* on the piecewise
  curve — joint equality.  All tuples are symmetric.
- ``sign="<="``: the **first** tuple's expression is bounded **above** by its
  interpolated value; the remaining tuples are forced to equality (inputs on
  the curve).  Reads as *"first expression ≤ f(the rest)"*.
- ``sign=">="``: same but the first is bounded **below**.

This is the *first-tuple convention* — a single inequality direction applies to
one designated output; the other tuples parameterise the curve.

**When is a one-sided bound wanted?**

The primary reason to reach for ``sign="<="`` / ``">="`` is to unlock the
**LP chord formulation** — no SOS2, no binaries, just pure LP.  On a
convex/concave curve with a matching sign, the chord inequalities are as
tight as SOS2, so you get the same optimum with a cheaper model.

Beyond that: fuel-on-efficiency-envelope modelling (extra burn above the
curve is admissible, cost is still bounded), emissions caps where the curve
is itself a convex overestimator, or any situation where the curve bounds a
variable that need not sit *on* it.

If the curvature doesn't match the sign (convex + ``"<="``, or concave +
``">="``), LP is not applicable — ``method="auto"`` falls back to
SOS2/incremental with the signed output link, which gives a valid but
much more expensive model.  In that case prefer ``sign="=="`` unless you
genuinely need the one-sided semantics; the equality formulation is
typically simpler to reason about and no more expensive than the SOS2
inequality variant.

**Math (2-variable ``sign="<="``, concave :math:`f`).**  The feasible region is
the **hypograph** of :math:`f` restricted to the breakpoint range:

.. math::

   \{ (x, y) \ :\ x_0 \le x \le x_n,\ y \le f(x) \}.

For convex :math:`f` with ``sign=">="``, the feasible region is the epigraph.
Mismatched sign+curvature (convex + ``<=``, or concave + ``>=``) describes a
*non-convex* region — ``method="auto"`` will fall back to SOS2/incremental and
``method="lp"`` will raise.  See the
:doc:`piecewise-inequality-bounds-tutorial` notebook for a full walkthrough.

.. warning::

   With ``sign="<="`` and ``active=0``, the output is only bounded **above** by
   ``0`` — the lower side still comes from the output variable's own lower
   bound.  In the common case of non-negative outputs (fuel, cost, heat), set
   ``lower=0`` on that variable: combined with the ``y ≤ 0`` constraint from
   deactivation, this forces ``y = 0`` automatically.  See the docstring for
   the full recipe.


Breakpoint Construction
-----------------------

From lists
~~~~~~~~~~

The simplest form — pass Python lists directly in the tuple:

.. code-block:: python

    m.add_piecewise_formulation(
        (power, [0, 30, 60, 100]),
        (fuel, [0, 36, 84, 170]),
    )

With the ``breakpoints()`` factory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Equivalent, but explicit about the DataArray construction:

.. code-block:: python

    m.add_piecewise_formulation(
        (power, linopy.breakpoints([0, 30, 60, 100])),
        (fuel, linopy.breakpoints([0, 36, 84, 170])),
    )

From slopes
~~~~~~~~~~~

When you know marginal costs (slopes) rather than absolute values:

.. code-block:: python

    m.add_piecewise_formulation(
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

    m.add_piecewise_formulation(
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

Ragged lengths are NaN-padded automatically.  Breakpoints are auto-broadcast
over remaining dimensions (e.g. ``time``).

Disjunctive segments
~~~~~~~~~~~~~~~~~~~~

For disconnected operating regions (e.g. forbidden zones), use ``segments()``:

.. code-block:: python

    m.add_piecewise_formulation(
        (power, linopy.segments([(0, 0), (50, 80)])),
        (cost, linopy.segments([(0, 0), (125, 200)])),
    )

The disjunctive formulation is selected automatically when breakpoints have a
segment dimension.  ``sign="<="`` / ``">="`` also works here; the signed link
is applied to the first tuple as usual.

N-variable linking
~~~~~~~~~~~~~~~~~~

Link any number of variables through shared breakpoints:

.. code-block:: python

    m.add_piecewise_formulation(
        (power, [0, 30, 60, 100]),
        (fuel, [0, 40, 85, 160]),
        (heat, [0, 25, 55, 95]),
    )

With ``sign="=="`` (default) all variables are symmetric.  With a non-equality
sign the first tuple is the bounded output and the rest are forced to
equality.


Formulation Methods
-------------------

Pass ``method="auto"`` (the default) and linopy picks the cheapest correct
formulation based on ``sign``, curvature and breakpoint layout:

- **2-variable inequality on a convex/concave curve** → ``lp`` (chord lines,
  no auxiliary variables)
- **All breakpoints monotonic** → ``incremental``
- **Otherwise** → ``sos2``
- **Disjunctive (segments)** → always ``sos2`` with binary segment selection

The resolved choice is exposed on the returned ``PiecewiseFormulation`` via
``.method`` (and ``.convexity`` when well-defined).  An ``INFO``-level log line
explains the resolution whenever ``method="auto"`` is in play.

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

With ``sign != "=="`` the input tuples still use the equality above; the
**first** tuple's link is replaced by a one-sided ``e_1 \ \text{sign}\ \sum_i
\lambda_i B_{1,i}`` constraint.

.. note::

   SOS2 is handled via branch-and-bound, similar to integer variables.
   Prefer ``method="incremental"`` when breakpoints are monotonic.

.. code-block:: python

    m.add_piecewise_formulation((power, xp), (fuel, yp), method="sos2")

Incremental (Delta) Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **strictly monotonic** breakpoints.  Uses fill-fraction variables
:math:`\delta_i` with binary indicators :math:`z_i`:

.. math::

   &\delta_i \in [0, 1], \quad z_i \in \{0, 1\}

   &\delta_{i+1} \le \delta_i, \quad z_{i+1} \le \delta_i, \quad \delta_i \le z_i

   &e_j = B_{j,0} + \sum_{i=1}^{n} \delta_i \, (B_{j,i} - B_{j,i-1})

With ``sign != "=="`` the same sign split as SOS2 applies: inputs use the
equality above; the first tuple's link uses the requested sign.

.. code-block:: python

    m.add_piecewise_formulation((power, xp), (fuel, yp), method="incremental")

**Limitation:** breakpoint sequences must be strictly monotonic.

LP (chord-line) Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **2-variable inequality** on a **convex** or **concave** curve.  Adds one
chord inequality per segment plus a domain bound — no auxiliary variables and
no MIP relaxation:

.. math::

   &y \ \text{sign}\ m_k \cdot x + c_k
   \quad \forall\ \text{segments } k

   &x_0 \le x \le x_n

where :math:`m_k = (y_{k+1} - y_k)/(x_{k+1} - x_k)` and
:math:`c_k = y_k - m_k\, x_k`.  For concave :math:`f` with ``sign="<="``,
the intersection of all chord inequalities equals the hypograph of
:math:`f` on its domain.

The LP dispatch requires curvature and sign to match: ``sign="<="`` needs
concave (or linear); ``sign=">="`` needs convex (or linear).  A mismatch
is *not* just a loose bound — it describes the wrong region (see the
:doc:`piecewise-inequality-bounds-tutorial`).  ``method="auto"`` detects
this and falls back; ``method="lp"`` raises.

.. code-block:: python

    # y <= f(x) on a concave f — auto picks LP
    m.add_piecewise_formulation((y, yp), (x, xp), sign="<=")

    # Or explicitly:
    m.add_piecewise_formulation((y, yp), (x, xp), sign="<=", method="lp")

**Not supported with** ``method="lp"``: ``sign="=="``, more than 2 tuples,
and ``active``.  ``method="auto"`` falls back to SOS2/incremental in all
three cases.

The underlying chord expressions are also exposed as a standalone helper,
``linopy.tangent_lines(x, x_pts, y_pts)``, which returns the per-segment
chord as a :class:`~linopy.expressions.LinearExpression` with no variables
created.  Use it directly if you want to compose the chord bound with other
constraints by hand, without the domain bound that ``method="lp"`` adds
automatically.

Disjunctive (Disaggregated Convex Combination)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **disconnected segments** (gaps between operating regions).  Binary
indicators :math:`z_k` select exactly one segment; SOS2 applies within it:

.. math::

   &z_k \in \{0, 1\}, \quad \sum_{k} z_k = 1

   &\sum_{i} \lambda_{k,i} = z_k, \qquad
   e_j = \sum_{k} \sum_{i} \lambda_{k,i} \, B_{j,k,i}

No big-M constants are needed, giving a tight LP relaxation.


Advanced Features
-----------------

Active parameter (unit commitment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``active`` parameter gates the piecewise function with a binary variable.
When ``active=0``, all auxiliary variables (and thus the linked expressions)
are forced to zero:

.. code-block:: python

    commit = m.add_variables(name="commit", binary=True, coords=[time])
    m.add_piecewise_formulation(
        (power, [30, 60, 100]),
        (fuel, [40, 90, 170]),
        active=commit,
    )

- ``commit=1``: power operates in [30, 100], fuel = f(power)
- ``commit=0``: power = 0, fuel = 0

Not supported with ``method="lp"``.

.. note::

   With a non-equality ``sign``, deactivation only pushes the signed bound to
   ``0`` — the complementary side comes from the output variable's own
   lower/upper bound.  Set ``lower=0`` on naturally non-negative outputs
   (fuel, cost, heat) to pin the output to zero on deactivation.  See the
   ``sign`` section above for details.

Auto-broadcasting
~~~~~~~~~~~~~~~~~

Breakpoints are automatically broadcast to match expression dimensions — you
don't need ``expand_dims``:

.. code-block:: python

    time = pd.Index([1, 2, 3], name="time")
    x = m.add_variables(name="x", lower=0, upper=100, coords=[time])
    y = m.add_variables(name="y", coords=[time])

    # 1D breakpoints auto-expand to match x's time dimension
    m.add_piecewise_formulation((x, [0, 50, 100]), (y, [0, 70, 150]))

NaN masking
~~~~~~~~~~~

Trailing NaN values in breakpoints mask the corresponding lambda / delta
variables (and, for LP, the corresponding chord constraints).  This is useful
for per-entity breakpoints with ragged lengths:

.. code-block:: python

    # gen1 has 3 breakpoints, gen2 has 2 (NaN-padded)
    bp = linopy.breakpoints({"gen1": [0, 50, 100], "gen2": [0, 80]}, dim="gen")

Interior NaN values (gaps in the middle) are not supported and raise an error.

Generated variables and constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a base name ``N`` (either user-supplied or auto-assigned like ``pwl0``),
each formulation creates a predictable set of names:

**SOS2** (``method="sos2"``):

- ``{N}_lambda`` — variable, interpolation weights
- ``{N}_convex`` — constraint, ``sum(lambda) == 1`` (or ``== active``)
- ``{N}_link`` — constraint, equality link (stacked inputs when
  ``sign != "=="``, all tuples when ``sign="=="``)
- ``{N}_output_link`` — constraint, signed link on the first tuple
  *(only when* ``sign != "=="`` *)*

**Incremental** (``method="incremental"``):

- ``{N}_delta`` — variable, fill fractions :math:`\delta_i`
- ``{N}_order_binary`` — variable, per-segment binaries :math:`z_i`
- ``{N}_delta_bound`` — constraint, :math:`\delta_i \le z_i`
- ``{N}_fill_order`` — constraint, :math:`\delta_{i+1} \le \delta_i`
- ``{N}_binary_order`` — constraint, :math:`z_{i+1} \le \delta_i`
- ``{N}_active_bound`` — constraint, :math:`\delta_i \le active`
  *(only when* ``active`` *is given)*
- ``{N}_link`` / ``{N}_output_link`` — same split as SOS2

**LP** (``method="lp"``):

- ``{N}_chord`` — constraint, per-segment chord inequality
- ``{N}_domain_lo``, ``{N}_domain_hi`` — constraints, :math:`x_0 \le x \le x_n`
- *no auxiliary variables*

**Disjunctive** (``segments(...)`` input):

- ``{N}_segment_binary`` — variable, per-segment selectors :math:`z_k`
- ``{N}_select`` — constraint, ``sum(z_k) == 1`` (or ``== active``)
- ``{N}_lambda`` — variable, within-segment weights
- ``{N}_convex`` — constraint, per-segment :math:`\sum_i \lambda_{k,i} = z_k`
- ``{N}_link`` / ``{N}_output_link`` — same split as SOS2


See Also
--------

- :doc:`piecewise-linear-constraints-tutorial` — worked examples of the
  equality API (notebook)
- :doc:`piecewise-inequality-bounds-tutorial` — the ``sign`` parameter, the LP
  formulation and the first-tuple convention (notebook)
- :doc:`sos-constraints` — low-level SOS1/SOS2 constraint API
