.. _sos-constraints:

Special Ordered Sets (SOS) Constraints
=======================================

SOS constraints model situations where only one (SOS1) or two adjacent (SOS2) variables from an ordered set can be non-zero.

.. contents::
   :local:
   :depth: 2

Overview
--------

- **SOS1**: At most one variable can be non-zero (mutually exclusive choices)
- **SOS2**: At most two adjacent variables can be non-zero (piecewise linear functions)

Basic Usage
-----------

.. code-block:: python

    import linopy
    import pandas as pd

    m = linopy.Model()

    # SOS1: at most one option selected
    options = pd.Index([0, 1, 2], name="options")
    x = m.add_variables(coords=[options], name="x", lower=0, upper=1)
    m.add_sos_constraints(x, sos_type=1, sos_dim="options")

    # SOS2: at most two adjacent breakpoints active
    breakpoints = pd.Index([0.0, 1.0, 2.0], name="bp")
    lambdas = m.add_variables(coords=[breakpoints], name="lambdas", lower=0, upper=1)
    m.add_sos_constraints(lambdas, sos_type=2, sos_dim="bp")

**Requirements:**

- The SOS dimension must have numeric coordinates (used as ordering weights)
- Only one SOS constraint per variable

Multi-dimensional Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For multi-dimensional variables, the SOS constraint applies independently for each combination of non-SOS dimensions:

.. code-block:: python

    periods = pd.Index([0, 1], name="periods")
    modes = pd.Index([0, 1, 2], name="modes")
    x = m.add_variables(lower=0, upper=1, coords=[periods, modes], name="x")
    m.add_sos_constraints(x, sos_type=1, sos_dim="modes")
    # Result: at most one mode selected PER period

Solver Compatibility
--------------------

**Native SOS support:** Gurobi, CPLEX, CBC, SCIP, Xpress

**No SOS support (use reformulation):** HiGHS, GLPK, MOSEK

.. _sos-reformulation:

SOS Reformulation
-----------------

For solvers without native SOS support, linopy can reformulate SOS constraints as binary + linear constraints using the Big-M method.

.. code-block:: python

    # Automatic reformulation during solve
    m.solve(solver_name="highs", reformulate_sos=True)

    # Or reformulate manually
    m.reformulate_sos_constraints()
    m.solve(solver_name="highs")

Big-M Values
~~~~~~~~~~~~

Big-M values are derived from variable bounds by default. For infinite bounds, specify custom values:

.. code-block:: python

    # Finite bounds: Big-M = bounds (default)
    x = m.add_variables(lower=0, upper=100, coords=[idx], name="x")
    m.add_sos_constraints(x, sos_type=1, sos_dim="i")

    # Infinite bounds: specify Big-M explicitly
    x = m.add_variables(lower=0, upper=np.inf, coords=[idx], name="x")
    m.add_sos_constraints(x, sos_type=1, sos_dim="i", big_m=10)

    # Asymmetric Big-M
    m.add_sos_constraints(x, sos_type=1, sos_dim="i", big_m=(10, -5))

The reformulation uses the tighter of ``big_m`` and variable bounds.

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

**SOS1:** Binary indicators :math:`y_i`, linking constraints, cardinality :math:`\sum y_i \leq 1`

.. math::

    y_i \in \{0, 1\}, \quad x_i \leq U_i \cdot y_i, \quad x_i \geq L_i \cdot y_i, \quad \sum y_i \leq 1

**SOS2:** Segment indicators :math:`z_j` for :math:`j = 0, \ldots, n-2`, adjacency constraints

.. math::

    z_j \in \{0, 1\}, \quad x_0 \leq U_0 \cdot z_0, \quad x_i \leq U_i(z_{i-1} + z_i), \quad x_{n-1} \leq U_{n-1} \cdot z_{n-2}, \quad \sum z_j \leq 1

Example: Piecewise Linear with HiGHS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import linopy
    import pandas as pd

    m = linopy.Model()

    # Approximate f(x) = x² over [0, 3]
    bp = pd.Index([0.0, 1.0, 2.0, 3.0], name="bp")
    x_vals, y_vals = bp.to_numpy(), bp.to_numpy() ** 2

    lambdas = m.add_variables(lower=0, upper=1, coords=[bp], name="lambdas")
    m.add_sos_constraints(lambdas, sos_type=2, sos_dim="bp")

    x = m.add_variables(name="x", lower=0, upper=3)
    y = m.add_variables(name="y", lower=0, upper=9)

    m.add_constraints(lambdas.sum() == 1, name="convexity")
    m.add_constraints(x == (lambdas * x_vals).sum(), name="x_interp")
    m.add_constraints(y == (lambdas * y_vals).sum(), name="y_interp")
    m.add_constraints(x >= 1.5, name="x_min")
    m.add_objective(y)

    m.solve(solver_name="highs", reformulate_sos=True)

API Reference
-------------

.. py:method:: Model.add_sos_constraints(variable, sos_type, sos_dim, big_m=None)

   Add SOS constraint to a variable.

   :param variable: Variable to constrain
   :param sos_type: 1 (at most one non-zero) or 2 (at most two adjacent non-zero)
   :param sos_dim: Dimension for SOS ordering
   :param big_m: Custom Big-M: ``float`` (symmetric) or ``tuple[float, float]`` (upper, lower)

.. py:method:: Model.remove_sos_constraints(variable)

   Remove SOS constraints from a variable.

.. py:method:: Model.reformulate_sos_constraints(prefix="_sos_reform_")

   Convert SOS to binary + linear constraints. Returns list of reformulated variable names.

.. py:attribute:: Variables.sos

   All variables with SOS constraints.
