.. _piecewise-linear-constraints:

Piecewise Linear Constraints
============================

Piecewise linear (PWL) constraints approximate nonlinear functions as connected
linear segments, allowing you to model cost curves, efficiency curves, or
production functions within a linear programming framework.

Use :py:func:`~linopy.piecewise.piecewise` to describe the function and
:py:meth:`~linopy.model.Model.add_piecewise_constraints` to add it to a model.

.. contents::
   :local:
   :depth: 2

Quick Start
-----------

.. code-block:: python

    import linopy

    m = linopy.Model()
    x = m.add_variables(name="x", lower=0, upper=100)
    y = m.add_variables(name="y")

    # y equals a piecewise linear function of x
    x_pts = linopy.breakpoints([0, 30, 60, 100])
    y_pts = linopy.breakpoints([0, 36, 84, 170])

    m.add_piecewise_constraints(linopy.piecewise(x, x_pts, y_pts) == y)

The ``piecewise()`` call creates a lazy descriptor. Comparing it with a
variable (``==``, ``<=``, ``>=``) produces a
:class:`~linopy.piecewise.PiecewiseConstraintDescriptor` that
``add_piecewise_constraints`` knows how to process.

.. note::

   The ``piecewise(...)`` expression must appear on the **left** side of the
   comparison operator. Writing ``y == piecewise(...)`` will not work because
   the variable's ``__eq__`` method takes precedence over Python's reflected
   operator lookup.


Formulations
------------

SOS2 (Convex Combination)
~~~~~~~~~~~~~~~~~~~~~~~~~

Given breakpoints :math:`b_0, b_1, \ldots, b_n`, the SOS2 formulation
introduces interpolation variables :math:`\lambda_i` such that:

.. math::

   \lambda_i \in [0, 1], \quad
   \sum_{i=0}^{n} \lambda_i = 1, \quad
   x = \sum_{i=0}^{n} \lambda_i \, b_i

The SOS2 constraint ensures that **at most two adjacent** :math:`\lambda_i` can
be non-zero, so :math:`x` is interpolated within one segment.

.. note::

   SOS2 is a combinatorial constraint handled via branch-and-bound, similar to
   integer variables. Prefer the incremental method
   (``method="incremental"`` or ``method="auto"``) when breakpoints are
   monotonic.

Incremental (Delta) Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **strictly monotonic** breakpoints :math:`b_0 < b_1 < \cdots < b_n`, the
incremental formulation uses fill-fraction variables:

.. math::

   \delta_i \in [0, 1], \quad
   \delta_{i+1} \le \delta_i, \quad
   x = b_0 + \sum_{i=1}^{n} \delta_i \, (b_i - b_{i-1})

The filling-order constraints enforce that segment :math:`i+1` cannot be
partially filled unless segment :math:`i` is completely filled. Binary
indicator variables enforce integrality.

**Limitation:** Breakpoints must be strictly monotonic. For non-monotonic
curves, use SOS2.

LP (Tangent-Line) Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **inequality** constraints where the function is **convex** (for ``>=``)
or **concave** (for ``<=``), a pure LP formulation adds one tangent-line
constraint per segment — no SOS2 or binary variables needed.

.. math::

   y \le m_k \, x + c_k \quad \text{for each segment } k \text{ (concave case)}

Domain bounds :math:`x_{\min} \le x \le x_{\max}` are added automatically.

**Limitation:** Only valid for inequality constraints with the correct
convexity; not valid for equality constraints.

Disjunctive (Disaggregated Convex Combination)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **disconnected segments** (with gaps), the disjunctive formulation selects
exactly one segment via binary indicators and applies SOS2 within it. No big-M
constants are needed, giving a tight LP relaxation.

Given :math:`K` segments, each with breakpoints :math:`b_{k,0}, \ldots, b_{k,n_k}`:

.. math::

   y_k \in \{0, 1\}, \quad \sum_{k} y_k = 1

   \lambda_{k,i} \in [0, 1], \quad
   \sum_{i} \lambda_{k,i} = y_k, \quad
   x = \sum_{k} \sum_{i} \lambda_{k,i} \, b_{k,i}


.. _choosing-a-formulation:

Choosing a Formulation
~~~~~~~~~~~~~~~~~~~~~~

Pass ``method="auto"`` (the default) and linopy will pick the best
formulation automatically:

- **Equality + monotonic x** → incremental
- **Inequality + correct convexity** → LP
- Otherwise → SOS2
- Disjunctive (segments) → always SOS2 with binary selection

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 15 20

   * - Property
     - SOS2
     - Incremental
     - LP
     - Disjunctive
   * - Segments
     - Connected
     - Connected
     - Connected
     - Disconnected
   * - Constraint type
     - ``==``, ``<=``, ``>=``
     - ``==``, ``<=``, ``>=``
     - ``<=``, ``>=`` only
     - ``==``, ``<=``, ``>=``
   * - Breakpoint order
     - Any
     - Strictly monotonic
     - Strictly increasing
     - Any (per segment)
   * - Convexity requirement
     - None
     - None
     - Concave (≤) or convex (≥)
     - None
   * - Variable types
     - Continuous + SOS2
     - Continuous + binary
     - Continuous only
     - Binary + SOS2
   * - Solver support
     - SOS2-capable
     - MIP-capable
     - **Any LP solver**
     - SOS2 + MIP


Basic Usage
-----------

Equality constraint
~~~~~~~~~~~~~~~~~~~

Link ``y`` to a piecewise linear function of ``x``:

.. code-block:: python

    import linopy

    m = linopy.Model()
    x = m.add_variables(name="x", lower=0, upper=100)
    y = m.add_variables(name="y")

    x_pts = linopy.breakpoints([0, 30, 60, 100])
    y_pts = linopy.breakpoints([0, 36, 84, 170])

    m.add_piecewise_constraints(linopy.piecewise(x, x_pts, y_pts) == y)

Inequality constraints
~~~~~~~~~~~~~~~~~~~~~~

Use ``<=`` or ``>=`` to bound ``y`` by the piecewise function:

.. code-block:: python

    pw = linopy.piecewise(x, x_pts, y_pts)

    # y must be at most the piecewise function of x  (pw >= y  ↔  y <= pw)
    m.add_piecewise_constraints(pw >= y)

    # y must be at least the piecewise function of x  (pw <= y  ↔  y >= pw)
    m.add_piecewise_constraints(pw <= y)

Choosing a method
~~~~~~~~~~~~~~~~~

.. code-block:: python

    pw = linopy.piecewise(x, x_pts, y_pts)

    # Explicit SOS2
    m.add_piecewise_constraints(pw == y, method="sos2")

    # Explicit incremental (requires monotonic x_pts)
    m.add_piecewise_constraints(pw == y, method="incremental")

    # Explicit LP (requires inequality + correct convexity + increasing x_pts)
    m.add_piecewise_constraints(pw >= y, method="lp")

    # Auto-select best method (default)
    m.add_piecewise_constraints(pw == y, method="auto")

Disjunctive (disconnected segments)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`~linopy.piecewise.segments` to define breakpoints with gaps:

.. code-block:: python

    m = linopy.Model()
    x = m.add_variables(name="x", lower=0, upper=100)
    y = m.add_variables(name="y")

    # Two disconnected segments: [0,10] and [50,100]
    x_seg = linopy.segments([(0, 10), (50, 100)])
    y_seg = linopy.segments([(0, 15), (60, 130)])

    m.add_piecewise_constraints(linopy.piecewise(x, x_seg, y_seg) == y)

The disjunctive formulation is selected automatically when
``x_points`` / ``y_points`` have a segment dimension (created by
:func:`~linopy.piecewise.segments`).


Breakpoints Factory
-------------------

The :func:`~linopy.piecewise.breakpoints` factory creates DataArrays with
the correct ``_breakpoint`` dimension. It accepts several input types
(``BreaksLike``):

From a list
~~~~~~~~~~~

.. code-block:: python

    # 1D breakpoints (dims: [_breakpoint])
    bp = linopy.breakpoints([0, 50, 100])

From a pandas Series
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd

    bp = linopy.breakpoints(pd.Series([0, 50, 100]))

From a DataFrame (per-entity, requires ``dim``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # rows = entities, columns = breakpoints
    df = pd.DataFrame(
        {"bp0": [0, 0], "bp1": [50, 80], "bp2": [100, float("nan")]},
        index=["gen1", "gen2"],
    )
    bp = linopy.breakpoints(df, dim="generator")

From a dict (per-entity, ragged lengths allowed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # NaN-padded to the longest entry
    bp = linopy.breakpoints(
        {"gen1": [0, 50, 100], "gen2": [0, 80]},
        dim="generator",
    )

From a DataArray (pass-through)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import xarray as xr

    arr = xr.DataArray([0, 50, 100], dims=["_breakpoint"])
    bp = linopy.breakpoints(arr)  # returned as-is

Slopes mode
~~~~~~~~~~~

Compute y-breakpoints from segment slopes and an initial y-value:

.. code-block:: python

    y_pts = linopy.breakpoints(
        slopes=[1.2, 1.4, 1.7],
        x_points=[0, 30, 60, 100],
        y0=0,
    )
    # Equivalent to breakpoints([0, 36, 78, 146])


Segments Factory
----------------

The :func:`~linopy.piecewise.segments` factory creates DataArrays with both
``_segment`` and ``_breakpoint`` dimensions (``SegmentsLike``):

From a list of sequences
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # dims: [_segment, _breakpoint]
    seg = linopy.segments([(0, 10), (50, 100)])

From a dict (per-entity)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    seg = linopy.segments(
        {"gen1": [(0, 10), (50, 100)], "gen2": [(0, 80)]},
        dim="generator",
    )

From a DataFrame
~~~~~~~~~~~~~~~~

.. code-block:: python

    # rows = segments, columns = breakpoints
    seg = linopy.segments(pd.DataFrame([[0, 10], [50, 100]]))


Auto-broadcasting
-----------------

Breakpoints are automatically broadcast to match the dimensions of the
expressions. You don't need ``expand_dims`` when your variables have extra
dimensions (e.g. ``time``):

.. code-block:: python

    import pandas as pd
    import linopy

    m = linopy.Model()
    time = pd.Index([1, 2, 3], name="time")
    x = m.add_variables(name="x", lower=0, upper=100, coords=[time])
    y = m.add_variables(name="y", coords=[time])

    # 1D breakpoints auto-expand to match x's time dimension
    x_pts = linopy.breakpoints([0, 50, 100])
    y_pts = linopy.breakpoints([0, 70, 150])
    m.add_piecewise_constraints(linopy.piecewise(x, x_pts, y_pts) == y)


Method Signatures
-----------------

``piecewise``
~~~~~~~~~~~~~

.. code-block:: python

    linopy.piecewise(expr, x_points, y_points)

- ``expr`` -- ``Variable`` or ``LinearExpression``. The "x" side expression.
- ``x_points`` -- ``BreaksLike``. Breakpoint x-coordinates.
- ``y_points`` -- ``BreaksLike``. Breakpoint y-coordinates.

Returns a :class:`~linopy.piecewise.PiecewiseExpression` that supports
``==``, ``<=``, ``>=`` comparison with another expression.

``add_piecewise_constraints``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    Model.add_piecewise_constraints(
        descriptor,
        method="auto",
        name=None,
        skip_nan_check=False,
    )

- ``descriptor`` -- :class:`~linopy.piecewise.PiecewiseConstraintDescriptor`.
  Created by comparing a ``PiecewiseExpression`` with an expression, e.g.
  ``piecewise(x, x_pts, y_pts) == y``.
- ``method`` -- ``"auto"`` (default), ``"sos2"``, ``"incremental"``, or ``"lp"``.
- ``name`` -- ``str``, optional. Base name for generated variables/constraints.
- ``skip_nan_check`` -- ``bool``, default ``False``.

Returns a :class:`~linopy.constraints.Constraint`, but the returned object is
formulation-dependent: typically ``{name}_convex`` (SOS2), ``{name}_fill`` or
``{name}_y_link`` (incremental), and ``{name}_select`` (disjunctive). For
inequality constraints, the returned constraint is the core piecewise
formulation constraint, not ``{name}_ineq``.

``breakpoints``
~~~~~~~~~~~~~~~~

.. code-block:: python

    linopy.breakpoints(values, dim=None)
    linopy.breakpoints(slopes, x_points, y0, dim=None)

- ``values`` -- ``BreaksLike`` (list, Series, DataFrame, DataArray, or dict).
- ``slopes``, ``x_points``, ``y0`` -- for slopes mode (mutually exclusive with
  ``values``).
- ``dim`` -- ``str``, required when ``values`` or ``slopes`` is a DataFrame or dict.

``segments``
~~~~~~~~~~~~~

.. code-block:: python

    linopy.segments(values, dim=None)

- ``values`` -- ``SegmentsLike`` (list of sequences, DataFrame, DataArray, or
  dict).
- ``dim`` -- ``str``, required when ``values`` is a dict.


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
     - :math:`x = \sum_i \lambda_i \, x_i`.
   * - ``{name}_y_link``
     - Constraint
     - :math:`y = \sum_i \lambda_i \, y_i`.
   * - ``{name}_aux``
     - Variable
     - Auxiliary variable :math:`z` (inequality constraints only).
   * - ``{name}_ineq``
     - Constraint
     - :math:`y \le z` or :math:`y \ge z` (inequality only).

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
   * - ``{name}_inc_link``
     - Constraint
     - :math:`\delta_i \le y_i` (delta bounded by binary).
   * - ``{name}_fill``
     - Constraint
     - :math:`\delta_{i+1} \le \delta_i` (fill order, 3+ breakpoints).
   * - ``{name}_inc_order``
     - Constraint
     - :math:`y_{i+1} \le \delta_i` (binary ordering, 3+ breakpoints).
   * - ``{name}_x_link``
     - Constraint
     - :math:`x = x_0 + \sum_i \delta_i \, \Delta x_i`.
   * - ``{name}_y_link``
     - Constraint
     - :math:`y = y_0 + \sum_i \delta_i \, \Delta y_i`.
   * - ``{name}_aux``
     - Variable
     - Auxiliary variable :math:`z` (inequality constraints only).
   * - ``{name}_ineq``
     - Constraint
     - :math:`y \le z` or :math:`y \ge z` (inequality only).

**LP method:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Name
     - Type
     - Description
   * - ``{name}_lp``
     - Constraint
     - Tangent-line constraints (one per segment).
   * - ``{name}_lp_domain_lo``
     - Constraint
     - :math:`x \ge x_{\min}`.
   * - ``{name}_lp_domain_hi``
     - Constraint
     - :math:`x \le x_{\max}`.

**Disjunctive method:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Name
     - Type
     - Description
   * - ``{name}_binary``
     - Variable
     - Segment indicators :math:`y_k \in \{0, 1\}`.
   * - ``{name}_select``
     - Constraint
     - :math:`\sum_k y_k = 1`.
   * - ``{name}_lambda``
     - Variable
     - Per-segment interpolation weights (SOS2).
   * - ``{name}_convex``
     - Constraint
     - :math:`\sum_i \lambda_{k,i} = y_k`.
   * - ``{name}_x_link``
     - Constraint
     - :math:`x = \sum_k \sum_i \lambda_{k,i} \, x_{k,i}`.
   * - ``{name}_y_link``
     - Constraint
     - :math:`y = \sum_k \sum_i \lambda_{k,i} \, y_{k,i}`.
   * - ``{name}_aux``
     - Variable
     - Auxiliary variable :math:`z` (inequality constraints only).
   * - ``{name}_ineq``
     - Constraint
     - :math:`y \le z` or :math:`y \ge z` (inequality only).

See Also
--------

- :doc:`piecewise-linear-constraints-tutorial` -- Worked examples covering SOS2, incremental, LP, and disjunctive usage
- :doc:`sos-constraints` -- Low-level SOS1/SOS2 constraint API
- :doc:`creating-constraints` -- General constraint creation
- :doc:`user-guide` -- Overall linopy usage patterns
