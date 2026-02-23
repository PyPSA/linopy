.. _piecewise-linear-constraints:

Piecewise Linear Constraints
============================

Piecewise linear (PWL) constraints approximate nonlinear functions as connected
linear segments, allowing you to model cost curves, efficiency curves, or
production functions within a linear programming framework.

Linopy provides two methods:

- :py:meth:`~linopy.model.Model.add_piecewise_constraints` -- for
  **continuous** piecewise linear functions (segments connected end-to-end).
- :py:meth:`~linopy.model.Model.add_disjunctive_piecewise_constraints` -- for
  **disconnected** segments (with gaps between them).

.. contents::
   :local:
   :depth: 2

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

**Dict (multi-variable) case.** When multiple variables share the same lambdas,
breakpoints carry an extra *link* dimension :math:`v \in V` and linking becomes
:math:`x_v = \sum_i \lambda_i \, b_{v,i}` for all :math:`v`.

.. note::

   SOS2 is a combinatorial constraint handled via branch-and-bound, similar to
   integer variables. It cannot be reformulated as a pure LP. Prefer the
   incremental method (``method="incremental"`` or ``method="auto"``) when
   breakpoints are monotonic.

Incremental (Delta) Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **strictly monotonic** breakpoints :math:`b_0 < b_1 < \cdots < b_n`, the
incremental formulation is a **pure LP** (no SOS2 or binary variables):

.. math::

   \delta_i \in [0, 1], \quad
   \delta_{i+1} \le \delta_i, \quad
   x = b_0 + \sum_{i=1}^{n} \delta_i \, (b_i - b_{i-1})

The filling-order constraints enforce that segment :math:`i+1` cannot be
partially filled unless segment :math:`i` is completely filled.

**Limitation:** Breakpoints must be strictly monotonic for every linked
variable. In the dict case, each variable is checked independently -- e.g.
power increasing while fuel decreases is fine, but a curve that rises then
falls is not. For non-monotonic curves, use SOS2.

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

The incremental method is the fastest to solve (pure LP), but requires strictly
monotonic breakpoints. Pass ``method="auto"`` to use it automatically when
applicable, falling back to SOS2 otherwise.

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Property
     - SOS2
     - Incremental
     - Disjunctive
   * - Segments
     - Connected
     - Connected
     - Disconnected (gaps allowed)
   * - Breakpoint order
     - Any
     - Strictly monotonic
     - Any (per segment)
   * - Variable types
     - Continuous + SOS2
     - Continuous only (pure LP)
     - Binary + SOS2
   * - Solver support
     - Solvers with SOS2 support
     - **Any LP solver**
     - Solvers with SOS2 + MIP support

Basic Usage
-----------

Single variable
~~~~~~~~~~~~~~~

.. code-block:: python

    import linopy
    import xarray as xr

    m = linopy.Model()
    x = m.add_variables(name="x")

    breakpoints = xr.DataArray([0, 10, 50, 100], dims=["bp"])
    m.add_piecewise_constraints(x, breakpoints, dim="bp")

Dict of variables
~~~~~~~~~~~~~~~~~~

Link multiple variables through shared interpolation weights. For example, a
turbine where power input determines power output (via a nonlinear efficiency
factor):

.. code-block:: python

    m = linopy.Model()

    power_in = m.add_variables(name="power_in")
    power_out = m.add_variables(name="power_out")

    # At 50 MW input the turbine produces 47.5 MW output (95% eff),
    # at 100 MW input only 90 MW output (90% eff)
    breakpoints = xr.DataArray(
        [[0, 50, 100], [0, 47.5, 90]],
        coords={"var": ["power_in", "power_out"], "bp": [0, 1, 2]},
    )

    m.add_piecewise_constraints(
        {"power_in": power_in, "power_out": power_out},
        breakpoints,
        dim="bp",
    )

Incremental method
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    m.add_piecewise_constraints(x, breakpoints, dim="bp", method="incremental")

Pass ``method="auto"`` to automatically select incremental when breakpoints are
strictly monotonic, falling back to SOS2 otherwise.

Disjunctive (disconnected segments)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    m = linopy.Model()
    x = m.add_variables(name="x")

    # Two disconnected segments: [0, 10] and [50, 100]
    breakpoints = xr.DataArray(
        [[0, 10], [50, 100]],
        dims=["segment", "breakpoint"],
        coords={"segment": [0, 1], "breakpoint": [0, 1]},
    )

    m.add_disjunctive_piecewise_constraints(x, breakpoints)

Method Signatures
-----------------

``add_piecewise_constraints``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    Model.add_piecewise_constraints(
        expr,
        breakpoints,
        dim="breakpoint",
        mask=None,
        name=None,
        skip_nan_check=False,
        method="sos2",
    )

- ``expr`` -- ``Variable``, ``LinearExpression``, or ``dict`` of these.
- ``breakpoints`` -- ``xr.DataArray`` with breakpoint values. Must have ``dim``
  as a dimension. For the dict case, must also have a dimension whose
  coordinates match the dict keys.
- ``dim`` -- ``str``, default ``"breakpoint"``. Breakpoint-index dimension.
- ``mask`` -- ``xr.DataArray``, optional. Boolean mask for valid constraints.
- ``name`` -- ``str``, optional. Base name for generated variables/constraints.
- ``skip_nan_check`` -- ``bool``, default ``False``.
- ``method`` -- ``"sos2"`` (default), ``"incremental"``, or ``"auto"``.

``add_disjunctive_piecewise_constraints``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    Model.add_disjunctive_piecewise_constraints(
        expr,
        breakpoints,
        dim="breakpoint",
        segment_dim="segment",
        mask=None,
        name=None,
        skip_nan_check=False,
    )

Same as above, plus:

- ``segment_dim`` -- ``str``, default ``"segment"``. Dimension indexing
  segments. Use NaN in breakpoints to pad segments with fewer breakpoints.

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
   * - ``{name}_link``
     - Constraint
     - :math:`x = \sum_i \lambda_i \, b_i`.

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
   * - ``{name}_fill``
     - Constraint
     - :math:`\delta_{i+1} \le \delta_i` (only if 3+ breakpoints).
   * - ``{name}_link``
     - Constraint
     - :math:`x = b_0 + \sum_i \delta_i \, s_i`.

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
   * - ``{name}_link``
     - Constraint
     - :math:`x = \sum_k \sum_i \lambda_{k,i} \, b_{k,i}`.

See Also
--------

- :doc:`piecewise-linear-constraints-tutorial` -- Worked examples with all three formulations
- :doc:`sos-constraints` -- Low-level SOS1/SOS2 constraint API
- :doc:`creating-constraints` -- General constraint creation
- :doc:`user-guide` -- Overall linopy usage patterns
