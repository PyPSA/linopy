.. _piecewise-linear-constraints:

Piecewise Linear Constraints
============================

Piecewise linear (PWL) constraints approximate nonlinear functions as connected
linear pieces, allowing you to model cost curves, efficiency curves, or
production functions within a linear programming framework.

**Terminology used in this page:**

- **breakpoint** — an :math:`(x, y)` knot where the slope can change.
- **piece** — a linear part between two adjacent breakpoints on a single
  connected curve.  ``n`` breakpoints define ``n − 1`` pieces.
- **segment** — a *disjoint* operating region in the disjunctive
  formulation, built via the :func:`~linopy.segments` factory.  Within
  one segment the curve is itself piecewise-linear (made of pieces);
  between segments there are gaps.

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
        (fuel, [0, 20, 30, 35], "<="),  # bounded by the curve
        (power, [0, 10, 20, 30]),  # pinned to the curve
    )

Each ``(expression, breakpoints[, sign])`` tuple pairs a variable with its
breakpoint values, and optionally marks it as bounded by the curve (``"<="``
or ``">="``) instead of pinned to it.  All tuples share interpolation weights,
so at any feasible point every variable corresponds to the *same* point on
the piecewise curve.


API
---

``add_piecewise_formulation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    m.add_piecewise_formulation(
        (expr1, breakpoints1),  # pinned (sign defaults to "==")
        (expr2, breakpoints2, "<="),  # or with an explicit sign
        ...,
        method="auto",  # "auto", "sos2", "incremental", or "lp"
        active=None,  # binary variable to gate the constraint
        name=None,  # base name for generated variables/constraints
    )

Creates auxiliary variables and constraints that enforce either a joint
equality (all tuples on the curve, the default) or a one-sided bound
(at most one tuple bounded by the curve, the rest pinned).

``breakpoints`` and ``segments``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two factories with distinct geometric meaning:

- ``breakpoints()`` — values along a single **connected** curve.  Linear
  pieces between adjacent breakpoints are interpolated continuously.
- ``segments()`` — **disjoint** operating regions with gaps between them
  (e.g. forbidden zones).  Builds a 2-D array consumed by the
  *disjunctive* formulation, where exactly one region is active at a time.

.. code-block:: python

    linopy.breakpoints([0, 50, 100])  # connected
    linopy.breakpoints({"gen1": [0, 50], "gen2": [0, 80]}, dim="gen")  # per-entity
    linopy.breakpoints(slopes=[1.2, 1.4], x_points=[0, 30, 60], y0=0)  # from slopes
    linopy.segments([(0, 10), (50, 100)])  # two disjoint regions
    linopy.segments({"gen1": [(0, 10)], "gen2": [(0, 80)]}, dim="gen")


Per-tuple sign — equality vs inequality
----------------------------------------

By default each tuple's expression is **pinned** to the piecewise curve.
Pass a third tuple element (``"<="`` or ``">="``) to mark a single
expression as **bounded** by the curve — it can undershoot (``"<="``) or
overshoot (``">="``) the interpolated value, while every other tuple
stays pinned.

.. code-block:: python

    # Joint equality (default): both expressions on the curve.
    m.add_piecewise_formulation((y, y_pts), (x, x_pts))

    # Bounded above: y <= f(x), x pinned.
    m.add_piecewise_formulation((y, y_pts, "<="), (x, x_pts))

    # Bounded below: y >= f(x), x pinned.
    m.add_piecewise_formulation((y, y_pts, ">="), (x, x_pts))

    # 3-variable equality (CHP heat/power/fuel): all three on one curve.
    m.add_piecewise_formulation((power, p_pts), (fuel, f_pts), (heat, h_pts))

**Restrictions (current):**

- At most one tuple may carry a non-equality sign — a single bounded side.
- With **3 or more** tuples, all signs must be ``"=="``.

Multi-bounded and N≥3-inequality use cases aren't supported yet.  If you
have a concrete use case, please open an issue at
https://github.com/PyPSA/linopy/issues so we can scope it properly.

**Formulation.**  For methods that introduce shared interpolation
weights (SOS2 and incremental — see below), only the link constraint
between the weights and the bounded expression changes.  Pinned tuples
:math:`j` keep the equality, and the bounded tuple :math:`b` flips to
the requested sign:

.. math::

   &e_j = \sum_{i=0}^{n} \lambda_i \, B_{j,i}
   \quad \text{(pinned, } j \ne b \text{)}

   &e_b \ \text{sign}\ \sum_{i=0}^{n} \lambda_i \, B_{b,i}
   \quad \text{(bounded)}

Internally this shows up as a stacked ``*_link`` equality covering the
pinned tuples plus a separate signed ``*_output_link`` for the bounded
tuple.  The ``method="lp"`` path encodes the same one-sided semantics
without weights — see the LP section below.

**Geometry.**  For 2 variables with ``sign="<="`` on a concave curve
:math:`f`, the feasible region is the **hypograph** of :math:`f` on its
domain:

.. math::

   \{ (x, y) \ :\ x_0 \le x \le x_n,\ y \le f(x) \}.

For convex :math:`f` with ``sign=">="`` it is the **epigraph**.  Mismatched
sign + curvature (convex + ``"<="``, or concave + ``">="``) describes a
*non-convex* region — ``method="auto"`` falls back to SOS2/incremental
and ``method="lp"`` raises.

**Choice of bounded tuple.**  The bounded tuple should correspond to a
quantity with a mechanism for below-curve operation — typically a
controllable dissipation path: heat rejection via cooling tower (also
called *thermal curtailment*), electrical curtailment, or emissions
after post-treatment.  Marking a consumption-side variable such as fuel
intake as bounded yields a valid but **loose** formulation: the
characteristic curve fixes fuel draw at a given load, so ``"<="`` on
fuel admits operating points the plant cannot physically realise.  An
objective that rewards lower fuel may then find a non-physical optimum
— safe only when no such objective pressure exists.

**When is a one-sided bound wanted?**

For *continuous* curves, the main reason to reach for ``"<="`` / ``">="``
is to unlock the **LP chord formulation** — no SOS2, no binaries, just
pure LP.  On a convex/concave curve with a matching sign, the chord
inequalities are as tight as SOS2, so you get the same optimum with a
cheaper model.  Inequality formulations also tighten the LP relaxation
of SOS2/incremental, which can reduce branch-and-bound work even when
LP itself is not applicable.

For *disjunctive* curves (``segments(...)``), the per-tuple sign is a
first-class tool in its own right: disconnected operating regions with a
bounded output, always exact regardless of segment curvature (see the
disjunctive section below).

If the curvature doesn't match the sign (convex + ``"<="``, or concave +
``">="``), LP is not applicable — ``method="auto"`` falls back to
SOS2/incremental with the signed link, which gives a valid but much
more expensive model.  In that case prefer ``"=="`` unless you genuinely
need the one-sided semantics.  See the
:doc:`piecewise-inequality-bounds-tutorial` notebook for a full
walkthrough.

.. warning::

   With a bounded tuple and ``active=0``, the output is only forced to
   ``0`` on the signed side — the complementary bound still comes from
   the output variable's own lower/upper bound.  In the common case of
   non-negative outputs (fuel, cost, heat), set ``lower=0`` on that
   variable: combined with the ``y ≤ 0`` constraint from deactivation,
   this forces ``y = 0`` automatically.  See the docstring for the
   full recipe.


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
segment dimension.  A bounded tuple (``"<="`` / ``">="``) also works here.

N-variable linking
~~~~~~~~~~~~~~~~~~

Link any number of variables through shared breakpoints (joint equality):

.. code-block:: python

    m.add_piecewise_formulation(
        (power, [0, 30, 60, 100]),
        (fuel, [0, 40, 85, 160]),
        (heat, [0, 25, 55, 95]),
    )

All variables are symmetric here; every feasible point is the same
``λ``-weighted combination of breakpoints across all three.  With 3 or
more tuples, only ``"=="`` signs are accepted — bounding one expression
by a multi-input curve isn't supported yet; see the per-tuple sign
section above for the issue link.


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

At-a-glance comparison:

.. list-table::
   :header-rows: 1
   :widths: 26 18 18 18 20

   * - Property
     - ``lp``
     - ``sos2``
     - ``incremental``
     - Disjunctive
   * - Curve layout
     - Connected
     - Connected
     - Connected
     - Disconnected
   * - Supported per-tuple sign
     - one ``<=`` or ``>=`` (required)
     - all ``==`` or one ``<=``/``>=``
     - all ``==`` or one ``<=``/``>=``
     - all ``==`` or one ``<=``/``>=``
   * - Number of tuples
     - Exactly 2
     - ≥ 2 (3+ requires all ``==``)
     - ≥ 2 (3+ requires all ``==``)
     - ≥ 2 (3+ requires all ``==``)
   * - Breakpoint order
     - Strictly monotonic
     - Any
     - Strictly monotonic
     - Any (per segment)
   * - Curvature requirement
     - Concave (``<=``) or convex (``>=``)
     - None
     - None
     - None
   * - Auxiliary variables
     - **None**
     - Continuous + SOS2
     - Continuous + binary
     - Binary + SOS2
   * - ``active=`` supported
     - No
     - Yes
     - Yes
     - Yes
   * - Solver requirement
     - **Any LP solver**
     - SOS2-capable
     - MIP-capable
     - SOS2 + MIP

LP (chord-line) Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **2-variable inequality** on a **convex** or **concave** curve.  Adds one
chord inequality per piece plus a domain bound — no auxiliary variables and
no MIP relaxation:

.. math::

   &y \ \text{sign}\ m_k \cdot x + c_k
   \quad \forall\ \text{pieces } k

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
    m.add_piecewise_formulation((y, yp, "<="), (x, xp))

    # Or explicitly:
    m.add_piecewise_formulation((y, yp, "<="), (x, xp), method="lp")

**Not supported with** ``method="lp"``: all-equality, more than 2 tuples,
and ``active``.  ``method="auto"`` falls back to SOS2/incremental in all
three cases.

The underlying chord expressions are also exposed as a standalone helper,
``linopy.tangent_lines(x, x_pts, y_pts)``, which returns the per-piece
chord as a :class:`~linopy.expressions.LinearExpression` with no variables
created.  Use it directly if you want to compose the chord bound with other
constraints by hand, without the domain bound that ``method="lp"`` adds
automatically.

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
non-zero, so every expression is interpolated within the same piece.

With a bounded tuple, the pinned tuples still use the equality above; the
bounded tuple's link is replaced by a one-sided ``e_b \ \text{sign}\ \sum_i
\lambda_i B_{b,i}`` constraint.

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

With a bounded tuple the same split as SOS2 applies: pinned tuples use the
equality above; the bounded tuple's link uses the requested sign.

.. code-block:: python

    m.add_piecewise_formulation((power, xp), (fuel, yp), method="incremental")

**Limitation:** breakpoint sequences must be strictly monotonic.

Disjunctive (Disaggregated Convex Combination)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **disconnected segments** (gaps between operating regions).  Binary
indicators :math:`z_k` select exactly one segment; SOS2 applies within it:

.. math::

   &z_k \in \{0, 1\}, \quad \sum_{k} z_k = 1

   &\sum_{i} \lambda_{k,i} = z_k, \qquad
   e_j = \sum_{k} \sum_{i} \lambda_{k,i} \, B_{j,k,i}

No big-M constants are needed, giving a tight LP relaxation.

**Disjunctive + bounded tuple.**  A per-tuple ``"<="`` / ``">="`` works
here too, applied to the bounded tuple exactly as for the continuous
methods.  Because the disjunctive machinery already carries a
per-segment binary, there is **no curvature requirement** on the
segments — inequality is always exact on the hypograph (or epigraph) of
the active segment, whatever its slope pattern.  This makes disjunctive
plus a bounded tuple a first-class tool for "bounded output on
disconnected operating regions" that ``method="lp"`` cannot handle.


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

   With a bounded tuple, deactivation only pushes the signed bound to
   ``0`` — the complementary side comes from the output variable's own
   lower/upper bound.  Set ``lower=0`` on naturally non-negative outputs
   (fuel, cost, heat) to pin the output to zero on deactivation.  See
   the per-tuple sign section above for details.

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

Inspecting generated objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The returned :class:`PiecewiseFormulation` exposes ``.variables`` and
``.constraints`` as live views into the model — use them to introspect
exactly what was generated, rather than relying on documented name
conventions:

.. code-block:: python

    f = m.add_piecewise_formulation((y, y_pts, "<="), (x, x_pts))
    print(f)  # method, convexity, vars/cons summary

The comparison table above describes the *kind* of auxiliary objects each
method creates (continuous + SOS2, binary + SOS2, none, …); exact name
suffixes are an implementation detail and may evolve.


See Also
--------

- :doc:`piecewise-linear-constraints-tutorial` — worked examples of the
  equality API (notebook)
- :doc:`piecewise-inequality-bounds-tutorial` — per-tuple sign and the LP
  formulation (notebook)
- :doc:`sos-constraints` — low-level SOS1/SOS2 constraint API
