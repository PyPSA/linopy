.. _piecewise-linear-constraints:

Piecewise Linear Constraints
============================

Piecewise linear (PWL) constraints approximate nonlinear functions as
connected linear pieces, allowing you to model cost curves, efficiency
curves, or production functions within a linear programming framework.

Throughout this page: a **breakpoint** is a knot where the slope can
change; a **piece** is the linear part between adjacent breakpoints; a
**segment** is a disjoint operating region in the disjunctive
formulation.  Per-tuple breakpoint arrays are paired by index — the
``i``-th entries across all tuples together define one knot.

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

    # fuel >= f(power) on the same heat-rate curve as above.
    m.add_piecewise_formulation(
        (fuel, [0, 36, 84, 170], ">="),
        (power, [0, 30, 60, 100]),
    )

Over-fuelling is physically admissible but wasteful, so minimising
fuel pulls the operating point onto the curve.  ``method="auto"``
picks the cheapest correct formulation: pure LP (chord constraints)
here, since convex + ``">="`` is LP-applicable; SOS2/incremental
otherwise.

Each ``(expression, breakpoints[, sign])`` tuple pairs a variable with
its breakpoint values.  The optional sign (default ``"=="``) is ``"<="``
or ``">="`` to mark that expression as bounded by the curve.  With every
sign ``"=="``, all tuples land on the same point of the piecewise curve
— see *Per-tuple sign* below for the geometry of the inequality cases.


API
---

``add_piecewise_formulation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    m.add_piecewise_formulation(
        (expr1, breakpoints1),  # sign defaults to "==" (equality role)
        (expr2, breakpoints2, "<="),  # or with an explicit sign
        ...,
        method="auto",  # "auto", "sos2", "incremental", or "lp"
        active=None,  # binary variable to gate the constraint
        name=None,  # base name for generated variables/constraints
    )

Adds constraints — and, depending on the resolved method, auxiliary
variables — for either an all-equality joint (every tuple at the same
point on the curve, the default) or a one-sided bound on a single
tuple.  The pure-LP path adds chord and domain constraints only; SOS2,
incremental, and disjunctive also add interpolation weights and/or
binaries (see *Formulation Methods* below).


Breakpoint Construction
-----------------------

Each tuple's breakpoints come from :func:`~linopy.breakpoints` (a
single connected curve) or :func:`~linopy.segments` (disjoint
operating bands).  :class:`~linopy.Slopes` can stand in for
:func:`~linopy.breakpoints` when per-piece slopes are the natural
input — it resolves to a breakpoints array.

``breakpoints()`` — a connected curve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Values along a single **connected** piecewise curve — the default case
for efficiency curves, heat rates, and cost curves.

The simplest form passes a Python list directly in the tuple:

.. code-block:: python

    m.add_piecewise_formulation(
        (power, [0, 30, 60, 100]),
        (fuel, [0, 36, 84, 170]),
    )

Equivalent, but explicit about the DataArray construction:

.. code-block:: python

    m.add_piecewise_formulation(
        (power, linopy.breakpoints([0, 30, 60, 100])),
        (fuel, linopy.breakpoints([0, 36, 84, 170])),
    )

**Per-entity curves.**  Different generators can have different
curves.  Pass a dict to ``breakpoints()`` with entity names as keys:

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

Ragged lengths are NaN-padded automatically.  Breakpoints are auto-
broadcast over remaining dimensions (e.g. ``time``).

**Specifying by slopes.**  :class:`linopy.Slopes` resolves to a
breakpoint array from per-piece slopes plus an initial ``y0``,
instead of from absolute y-values — useful when slopes are the
natural input (e.g. marginal costs).  The x grid is borrowed from
the sibling tuple, so the y breakpoints don't have to be computed
by hand:

.. code-block:: python

    m.add_piecewise_formulation(
        (power, [0, 50, 100, 150]),
        (cost, linopy.Slopes([1.1, 1.5, 1.9], y0=0)),
    )
    # cost breakpoints resolve to: [0, 55, 130, 225]

For standalone resolution outside ``add_piecewise_formulation``, call
:meth:`linopy.Slopes.to_breakpoints` with an explicit x grid::

    bp = linopy.Slopes([1.1, 1.5, 1.9], y0=0).to_breakpoints([0, 50, 100, 150])

``segments()`` — disjoint operating bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For equipment with disconnected operating bands.  Each segment is one
band's ``(range, curve)``; a binary picks exactly one per operating
point, with continuous interpolation within the chosen band.

.. code-block:: python

    # Stepped pump with two speed bands.
    m.add_piecewise_formulation(
        (flow, linopy.segments([(5, 25), (40, 100)])),
        (power, linopy.segments([(1, 7), (15, 50)])),
    )

Bounded tuples (``"<="`` / ``">="``) are supported on disjunctive
curves too.

For a single on/off gate on one continuous curve, prefer ``active=...``
(see :ref:`piecewise-active`) — using a degenerate ``(0, 0)`` segment
to encode "off" mixes the disjunctive concept with on/off logic.

N-variable linking
~~~~~~~~~~~~~~~~~~

Independent of the building block used, any number of variables can be
linked through shared breakpoints (joint equality):

.. code-block:: python

    m.add_piecewise_formulation(
        (power, [0, 30, 60, 100]),
        (fuel, [0, 40, 85, 160]),
        (heat, [0, 25, 55, 95]),
    )

All variables are symmetric here; every feasible point is the same
``λ``-weighted combination of breakpoints across all three.  Sign
restrictions apply (see *Per-tuple sign* below) — for ``N ≥ 3`` tuples
all signs must be ``"=="``.


Per-tuple sign — equality vs inequality
----------------------------------------

Roles and restrictions
~~~~~~~~~~~~~~~~~~~~~~

Each tuple's optional third element is a sign:

- ``"=="`` (default) — **equality role**: the tuple enters as an
  equality.
- ``"<="`` / ``">="`` — **bounded**: the expression undershoots /
  overshoots the curve.

.. note::

   **Current restrictions.**

   - At most one tuple may carry a non-equality sign — a single bounded side.
   - With **3 or more** tuples, all signs must be ``"=="``.

   Multi-bounded and N≥3-inequality use cases aren't supported yet.
   If you have a concrete use case, please open an issue at
   https://github.com/PyPSA/linopy/issues so we can scope it properly.

Geometry
~~~~~~~~

What the formulation actually constrains depends on the tuple count and
signs:

- **All-equality (default).**  Shared interpolation weights put the
  joint :math:`(e_1, \ldots, e_N)` exactly on the curve.
- **One bounded + one equality (2 tuples).**  The joint :math:`(x, y)`
  lies in the **hypograph** (``"<="`` on a concave :math:`f`) or
  **epigraph** (``">="`` on a convex :math:`f`):

  .. math::

     \{ (x, y) \ :\ x_{\min} \le x \le x_{\max},\ y \le f(x) \}
     \qquad \text{(hypograph)}

  The equality axis is just confined to its breakpoint domain
  :math:`[x_{\min}, x_{\max}]` — a single coordinate can't locate a
  curve point.  Same projection in LP (enforced directly) and
  SOS2/incremental (enforced via the weight link).
- **Mismatched sign + curvature** (convex + ``"<="``, or concave +
  ``">="``) describes a *non-convex* region — ``method="auto"`` falls
  back to SOS2/incremental and ``method="lp"`` raises.

.. code-block:: python

    # All-equality: joint (x, y) on the curve.
    m.add_piecewise_formulation((y, y_pts), (x, x_pts))

    # Bounded: joint (x, y) in the hypograph — y ≤ f(x), x ∈ [x_min, x_max].
    m.add_piecewise_formulation((y, y_pts, "<="), (x, x_pts))

    # Bounded: joint (x, y) in the epigraph — y ≥ f(x), x ∈ [x_min, x_max].
    m.add_piecewise_formulation((y, y_pts, ">="), (x, x_pts))

    # 3-variable all-equality (CHP): joint (power, fuel, heat) on the curve.
    m.add_piecewise_formulation((power, p_pts), (fuel, f_pts), (heat, h_pts))

Choice of bounded tuple and sign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pick the sign matching the physically admissible direction for that
expression:

- ``"<="`` for a quantity with a controllable *dissipation path* — heat
  rejection via cooling tower (*thermal curtailment*), electrical
  curtailment, emissions after post-treatment — so undershooting the
  curve is realisable.
- ``">="`` for an *input* whose over-supply is admissible but wasteful —
  fuel, raw materials — so overshooting the curve is realisable
  (objective pressure then pulls the operating point onto the curve).

The wrong direction (``"<="`` on fuel, ``">="`` on a non-curtailable
output) yields a valid but **loose** formulation that admits operating
points the plant cannot physically realise; an objective rewarding the
wrong direction may then find a non-physical optimum — safe only when
no such objective pressure exists.

When is a one-sided bound wanted?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For *continuous* curves, the main reason to reach for ``"<="`` /
``">="`` is to unlock the **LP chord formulation** — no SOS2, no
binaries, just pure LP.  On a convex/concave curve with a matching
sign, the chord inequalities are as tight as SOS2, so you get the same
optimum with a cheaper model.  Inequality formulations also tighten
the LP relaxation of SOS2/incremental, which can reduce branch-and-
bound work even when LP itself is not applicable.

For *disjunctive* curves (``segments(...)``), the per-tuple sign is a
first-class tool in its own right: disconnected operating regions with
a bounded output, always exact regardless of segment curvature (see
the disjunctive section below).

If the curvature doesn't match the sign (convex + ``"<="``, or concave +
``">="``), LP is not applicable — ``method="auto"`` falls back to
SOS2/incremental with the signed link, which gives a valid but much
more expensive model.  In that case prefer ``"=="`` unless you
genuinely need the one-sided semantics.  See the
:doc:`piecewise-inequality-bounds-tutorial` notebook for a full
walkthrough.

Formulation Methods
-------------------

Pass ``method="auto"`` (the default) and linopy picks the cheapest correct
formulation based on ``sign``, curvature and breakpoint layout:

- **2-variable inequality on a convex/concave curve** → ``lp`` (chord lines,
  no auxiliary variables)
- **All breakpoints monotonic** → ``incremental``
- **Otherwise** → ``sos2``
- **Disjunctive (segments)** → SOS2 applied per segment with binary
  segment selection (the disjunctive formulation in the table below).

The resolved choice is exposed on the returned ``PiecewiseFormulation``
via ``.method`` (and ``.convexity`` when well-defined).  An
``INFO``-level log line explains the resolution whenever
``method="auto"`` is in play.

At-a-glance comparison:

.. list-table::
   :header-rows: 1
   :widths: 26 18 18 18 20

   * - Property
     - ``lp``
     - ``incremental``
     - ``sos2``
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
     - Strictly monotonic
     - Any
     - Any (per segment)
   * - Curvature requirement
     - Concave (``<=``) or convex (``>=``)
     - None
     - None
     - None
   * - Auxiliary variables
     - **None**
     - Continuous + binary
     - Continuous + SOS2
     - Continuous + binary + SOS2
   * - ``active=`` supported
     - No
     - Yes
     - Yes
     - Yes
   * - Solver requirement
     - **Any LP solver**
     - MIP-capable
     - SOS2-capable (or MIP via :ref:`Big-M reformulation <sos-reformulation>`)
     - SOS2 + MIP (or MIP via :ref:`Big-M reformulation <sos-reformulation>`)

.. note::

   Disjunctive formulations report ``method="sos2"`` (the underlying
   per-segment encoding uses SOS2), but the table treats them as a
   separate column because the per-segment binaries change the
   auxiliary-variable structure and solver requirements.

LP (chord-line) formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **2-variable inequality** on a **convex** or **concave** curve.
Adds one chord inequality per piece plus a domain bound — no auxiliary
variables and no MIP relaxation:

.. math::

   &y \ \text{sign}\ m_k \cdot x + c_k
   \quad \forall\ \text{pieces } k

   &x_{\min} \le x \le x_{\max}

where :math:`m_k = (y_{k+1} - y_k)/(x_{k+1} - x_k)` and
:math:`c_k = y_k - m_k\, x_k`.  The domain bound uses :math:`x_{\min}`
and :math:`x_{\max}` rather than the first/last breakpoint so that
descending x grids work too — strictly-monotonic breakpoints are
accepted in either order.  For concave :math:`f` with ``sign="<="``,
the intersection of all chord inequalities equals the hypograph of
:math:`f` on its domain.

The LP dispatch requires curvature and sign to match: ``sign="<="``
needs concave (or linear); ``sign=">="`` needs convex (or linear).  A
mismatch is *not* just a loose bound — it describes the wrong region
(see the :doc:`piecewise-inequality-bounds-tutorial`).
``method="auto"`` detects this and falls back; ``method="lp"`` raises.

.. code-block:: python

    # y <= f(x) on a concave f — auto picks LP
    m.add_piecewise_formulation((y, yp, "<="), (x, xp))

    # Or explicitly:
    m.add_piecewise_formulation((y, yp, "<="), (x, xp), method="lp")

**Not supported with** ``method="lp"``: all-equality, more than 2
tuples, and ``active``.  ``method="auto"`` falls back to
SOS2/incremental in all three cases.

Chord expressions as a building block
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The underlying chord expressions are also exposed as a standalone
helper, :func:`~linopy.tangent_lines`, which returns the per-piece
chord as a :class:`~linopy.expressions.LinearExpression` with no
variables created.  Use it directly when you want to compose the chord
bound with other constraints by hand, without the domain bound that
``method="lp"`` adds automatically:

.. code-block:: python

    chord = linopy.tangent_lines(x, x_pts, y_pts)
    m.add_constraints(y <= chord + slack)

Incremental (Delta) formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default MIP encoding when ``method="auto"`` is in play and breakpoints
are **strictly monotonic** — produces a tight MIP directly, without going
through an SOS2 layer.  Uses fill-fraction variables :math:`\delta_i` with
binary indicators :math:`z_i`:

.. math::

   &\delta_i \in [0, 1], \quad z_i \in \{0, 1\}

   &\delta_{i+1} \le \delta_i, \quad z_{i+1} \le \delta_i, \quad \delta_i \le z_i

   &e_j = B_{j,0} + \sum_{i=1}^{n} \delta_i \, (B_{j,i} - B_{j,i-1})

With a bounded tuple, the link to that tuple's expression flips to the
requested sign while the equality-signed tuples keep the equality above.

.. code-block:: python

    m.add_piecewise_formulation((power, xp), (fuel, yp), method="incremental")

**Limitation:** breakpoint sequences must be strictly monotonic.

SOS2 (Convex combination)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Fallback when breakpoints aren't strictly monotonic (the only case
``method="auto"`` does not pick incremental for a connected curve).
Introduces interpolation weights :math:`\lambda_i` with an SOS2
adjacency constraint:

.. math::

   &\sum_{i=0}^{n} \lambda_i = 1, \qquad
   \text{SOS2}(\lambda_0, \ldots, \lambda_n)

   &e_j = \sum_{i=0}^{n} \lambda_i \, B_{j,i}
   \quad \text{for each expression } j

The SOS2 constraint ensures at most two adjacent :math:`\lambda_i` are
non-zero, so every expression is interpolated within the same piece.
With a bounded tuple, the bounded link flips to the requested sign as
above.

.. note::

   Solvers with native SOS2 support handle the adjacency constraint via
   branch-and-bound.  Solvers without it see the Big-M reformulation
   linopy applies (controlled by ``reformulate_sos=`` on ``solve``) —
   see :ref:`sos-reformulation` for the reformulated MIP form, which is
   the model those solvers actually receive.  When breakpoints are
   monotonic, prefer ``method="incremental"`` (or just ``"auto"``): it
   builds a similar MIP encoding directly and does not depend on
   solver SOS2 support or the reformulation step.

.. code-block:: python

    m.add_piecewise_formulation((power, xp), (fuel, yp), method="sos2")

Disjunctive (Disaggregated convex combination)
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

.. _piecewise-active:

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

Not supported with ``method="lp"`` (gating needs a binary).  Use
``method="auto"``, or *Chord expressions as a building block* for
manual gating.

.. warning::

   With a bounded tuple and ``active=0``, the output is only forced to
   ``0`` on the signed side — the complementary bound still comes from
   the output variable's own lower/upper bound.  In the common case of
   non-negative outputs (fuel, cost, heat), set ``lower=0`` on that
   variable: combined with the ``y ≤ 0`` constraint from deactivation,
   this forces ``y = 0`` automatically.

Partial gates
^^^^^^^^^^^^^

``active`` must cover the formulation's full coordinate; a gate defined
over only a subset (or with masked entries) is rejected unless
``active_fill`` is set. ``active_fill`` gates the missing entries as
always-active (``1``) or always-off (``0``) — handy when one formulation
mixes committable and non-committable units sharing a single ``status``:

.. code-block:: python

    m.add_piecewise_formulation(
        (power, [30, 60, 100]), (fuel, [40, 90, 170]), active=status, active_fill=1
    )

``active_fill`` is transitional: under v1 semantics, pad ``active``
explicitly with ``active.reindex(coords).fillna(value)`` instead.

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


Tutorials
---------

.. toctree::
   :maxdepth: 1

   piecewise-linear-constraints-tutorial
   piecewise-inequality-bounds-tutorial

See Also
--------

- :doc:`sos-constraints` — low-level SOS1/SOS2 constraint API
