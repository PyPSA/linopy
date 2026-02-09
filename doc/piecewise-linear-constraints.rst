.. _piecewise-linear-constraints:

Piecewise Linear Constraints
============================

Piecewise linear (PWL) constraints approximate nonlinear functions as connected
linear segments, allowing you to model cost curves, efficiency curves, or
production functions within a linear programming framework.

Linopy provides :py:meth:`~linopy.model.Model.add_piecewise_constraints` which
handles all the internal bookkeeping (lambda variables, SOS2 declarations,
convexity and linking constraints) so you can focus on the breakpoints that
define your function.

.. contents::
   :local:
   :depth: 2

Formulation
-----------

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
        link_dim="var",
        dim="bp",
    )

Method Signature
~~~~~~~~~~~~~~~~

.. code-block:: python

    Model.add_piecewise_constraints(
        expr,
        breakpoints,
        link_dim=None,
        dim="breakpoint",
        mask=None,
        name=None,
        skip_nan_check=False,
    )

- ``expr`` -- ``Variable``, ``LinearExpression``, or ``dict`` of these.
- ``breakpoints`` -- ``xr.DataArray`` with breakpoint values. Must have ``dim``
  as a dimension. For the dict case, must also have ``link_dim``.
- ``link_dim`` -- ``str``, optional. Dimension linking to different expressions.
- ``dim`` -- ``str``, default ``"breakpoint"``. Breakpoint-index dimension.
- ``mask`` -- ``xr.DataArray``, optional. Boolean mask for valid constraints.
- ``name`` -- ``str``, optional. Base name for generated variables/constraints.
- ``skip_nan_check`` -- ``bool``, default ``False``.

**Returns:** The convexity ``Constraint`` (:math:`\sum \lambda_i = 1`).

Generated Variables and Constraints
------------------------------------

Given base name ``name``, the following objects are created:

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

See Also
--------

- :doc:`sos-constraints` -- Low-level SOS1/SOS2 constraint API
- :doc:`creating-constraints` -- General constraint creation
- :doc:`user-guide` -- Overall linopy usage patterns
