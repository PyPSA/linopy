.. _sos-constraints:

Special Ordered Sets (SOS) Constraints
=======================================

Special Ordered Sets (SOS) are a constraint type used in mixed-integer programming to model situations where only one or two variables from an ordered set can be non-zero. Linopy supports both SOS Type 1 and SOS Type 2 constraints.

.. contents::
   :local:
   :depth: 2

Overview
--------

SOS constraints are particularly useful for:

- **SOS1**: Modeling mutually exclusive choices (e.g., selecting one facility from multiple locations)
- **SOS2**: Piecewise linear approximations of nonlinear functions
- Improving branch-and-bound efficiency in mixed-integer programming

Types of SOS Constraints
-------------------------

SOS Type 1 (SOS1)
~~~~~~~~~~~~~~~~~~

In an SOS1 constraint, **at most one** variable in the ordered set can be non-zero.

**Example use cases:**
- Facility location problems (choose one location among many)
- Technology selection (choose one technology option)
- Mutually exclusive investment decisions

SOS Type 2 (SOS2)
~~~~~~~~~~~~~~~~~~

In an SOS2 constraint, **at most two adjacent** variables in the ordered set can be non-zero. The adjacency is determined by the ordering weights (coordinates) of the variables.

**Example use cases:**
- Piecewise linear approximation of nonlinear functions
- Portfolio optimization with discrete risk levels
- Production planning with discrete capacity levels

Basic Usage
-----------

Adding SOS Constraints
~~~~~~~~~~~~~~~~~~~~~~~

To add SOS constraints to variables in linopy:

.. code-block:: python

    import linopy
    import pandas as pd
    import xarray as xr

    # Create model
    m = linopy.Model()

    # Create variables with numeric coordinates
    coords = pd.Index([0, 1, 2], name="options")
    x = m.add_variables(coords=[coords], name="x", lower=0, upper=1)

    # Add SOS1 constraint
    m.add_sos_constraints(x, sos_type=1, sos_dim="options")

    # For SOS2 constraint
    breakpoints = pd.Index([0.0, 1.0, 2.0], name="breakpoints")
    lambdas = m.add_variables(coords=[breakpoints], name="lambdas", lower=0, upper=1)
    m.add_sos_constraints(lambdas, sos_type=2, sos_dim="breakpoints")

Method Signature
~~~~~~~~~~~~~~~~

.. code-block:: python

    Model.add_sos_constraints(variable, sos_type, sos_dim)

**Parameters:**

- ``variable`` : Variable
    The variable to which the SOS constraint should be applied
- ``sos_type`` : {1, 2}
    Type of SOS constraint (1 or 2)
- ``sos_dim`` : str
    Name of the dimension along which the SOS constraint applies

**Requirements:**

- The specified dimension must exist in the variable
- The coordinates for the SOS dimension must be numeric (used as weights for ordering)
- Only one SOS constraint can be applied per variable

Examples
--------

Example 1: Facility Location (SOS1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import linopy
    import pandas as pd
    import xarray as xr

    # Problem data
    locations = pd.Index([0, 1, 2, 3], name="locations")
    costs = xr.DataArray([100, 150, 120, 80], coords=[locations])
    benefits = xr.DataArray([200, 300, 250, 180], coords=[locations])

    # Create model
    m = linopy.Model()

    # Decision variables: build facility at location i
    build = m.add_variables(coords=[locations], name="build", lower=0, upper=1)

    # SOS1 constraint: at most one facility can be built
    m.add_sos_constraints(build, sos_type=1, sos_dim="locations")

    # Objective: maximize net benefit
    net_benefit = benefits - costs
    m.add_objective(-((net_benefit * build).sum()))

    # Solve
    m.solve(solver_name="gurobi")

    if m.status == "ok":
        solution = build.solution.to_pandas()
        selected_location = solution[solution > 0.5].index[0]
        print(f"Build facility at location {selected_location}")

Example 2: Piecewise Linear Approximation (SOS2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np

    # Approximate f(x) = x² over [0, 3] with breakpoints
    breakpoints = pd.Index([0, 1, 2, 3], name="breakpoints")

    x_vals = xr.DataArray(breakpoints.to_series())
    y_vals = x_vals**2

    # Create model
    m = linopy.Model()

    # SOS2 variables (interpolation weights)
    lambdas = m.add_variables(lower=0, upper=1, coords=[breakpoints], name="lambdas")
    m.add_sos_constraints(lambdas, sos_type=2, sos_dim="breakpoints")

    # Interpolated coordinates
    x = m.add_variables(name="x", lower=0, upper=3)
    y = m.add_variables(name="y", lower=0, upper=9)

    # Constraints
    m.add_constraints(lambdas.sum() == 1, name="convexity")
    m.add_constraints(x == lambdas @ x_vals, name="x_interpolation")
    m.add_constraints(y == lambdas @ y_vals, name="y_interpolation")
    m.add_constraints(x >= 1.5, name="x_minimum")

    # Objective: minimize approximated function value
    m.add_objective(y)

    # Solve
    m.solve(solver_name="gurobi")

Working with Multi-dimensional Variables
-----------------------------------------

SOS constraints are created for each dimension that is not sos_dim.

.. code-block:: python

    # Multi-period production planning
    periods = pd.Index(range(3), name="periods")
    modes = pd.Index([0, 1, 2], name="modes")

    # 2D variables: periods × modes
    period_modes = m.add_variables(
        lower=0, upper=1, coords=[periods, modes], name="use_mode"
    )

    # Adds SOS1 constraint for each period
    m.add_sos_constraints(period_modes, sos_type=1, sos_dim="modes")

Accessing SOS Variables
-----------------------

You can easily identify and access variables with SOS constraints:

.. code-block:: python

    # Get all variables with SOS constraints
    sos_variables = m.variables.sos
    print(f"SOS variables: {list(sos_variables.keys())}")

    # Check SOS properties of a variable
    for var_name in sos_variables:
        var = m.variables[var_name]
        sos_type = var.attrs["sos_type"]
        sos_dim = var.attrs["sos_dim"]
        print(f"{var_name}: SOS{sos_type} on dimension '{sos_dim}'")

Variable Representation
~~~~~~~~~~~~~~~~~~~~~~~

Variables with SOS constraints show their SOS information in string representations:

.. code-block:: python

    print(build)
    # Output: Variable (locations: 4) - sos1 on locations
    # -----------------------------------------------
    # [0]: build[0] ∈ [0, 1]
    # [1]: build[1] ∈ [0, 1]
    # [2]: build[2] ∈ [0, 1]
    # [3]: build[3] ∈ [0, 1]

LP File Export
--------------

The generated LP file will include a SOS section:

.. code-block:: text

    sos

    s0: S1 ::  x0:0  x1:1  x2:2
    s3: S2 ::  x3:0.0  x4:1.0  x5:2.0

Solver Compatibility
--------------------

SOS constraints are supported by most modern mixed-integer programming solvers through the LP file format:

**Supported solvers (via LP file):**

- Gurobi
- CPLEX
- COIN-OR CBC
- SCIP
- Xpress

**Direct API support:**

- Gurobi (via ``gurobipy``)

**Unsupported solvers:**

- HiGHS (does not support SOS constraints)
- GLPK
- MOSEK
- MindOpt

For these solvers, linopy provides automatic reformulation (see :ref:`sos-reformulation` below).

.. _sos-reformulation:

SOS Reformulation for Unsupported Solvers
-----------------------------------------

Linopy can automatically reformulate SOS constraints as binary + linear constraints
using the Big-M method. This allows you to use SOS constraints with solvers that
don't support them natively (HiGHS, GLPK, MOSEK, etc.).

Enabling Reformulation
~~~~~~~~~~~~~~~~~~~~~~

Pass ``reformulate_sos=True`` to the ``solve()`` method:

.. code-block:: python

    import linopy
    import pandas as pd

    m = linopy.Model()
    idx = pd.Index([0, 1, 2], name="i")
    x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
    m.add_sos_constraints(x, sos_type=1, sos_dim="i")
    m.add_objective(x.sum(), sense="max")

    # Now works with HiGHS!
    m.solve(solver_name="highs", reformulate_sos=True)

You can also reformulate manually before solving:

.. code-block:: python

    # Reformulate in place
    reformulated_vars = m.reformulate_sos_constraints()
    print(f"Reformulated: {reformulated_vars}")

    # Then solve with any solver
    m.solve(solver_name="highs")

Requirements
~~~~~~~~~~~~

**Finite bounds or custom Big-M required.** The reformulation uses the Big-M method.
By default, Big-M values are derived from variable bounds. If any SOS variable has
infinite bounds, you must either:

1. Set finite bounds on the variable, or
2. Specify custom Big-M values via the ``big_m`` parameter

.. code-block:: python

    # Option 1: Finite bounds (default Big-M = bounds)
    x = m.add_variables(lower=0, upper=100, coords=[idx], name="x")
    m.add_sos_constraints(x, sos_type=1, sos_dim="i")

    # Option 2: Custom Big-M (allows infinite bounds)
    x = m.add_variables(lower=0, upper=np.inf, coords=[idx], name="x")
    m.add_sos_constraints(x, sos_type=1, sos_dim="i", big_m=10)

Custom Big-M Values
~~~~~~~~~~~~~~~~~~~

The ``big_m`` parameter in ``add_sos_constraints()`` allows you to specify tighter
Big-M values than the variable bounds. This is useful when:

- Variable bounds are conservatively large (e.g., ``upper=1e6``)
- Other constraints in your model imply tighter effective bounds
- You have domain knowledge about the actual feasible range

**Syntax options:**

.. code-block:: python

    # Scalar: symmetric Big-M (M_upper = 10, M_lower = -10)
    m.add_sos_constraints(x, sos_type=1, sos_dim="i", big_m=10)

    # Tuple of scalars: asymmetric (M_upper = 10, M_lower = -5)
    m.add_sos_constraints(x, sos_type=1, sos_dim="i", big_m=(10, -5))

.. note::

   The reformulation uses the **tighter** of custom ``big_m`` and variable bounds:

   - ``M_upper = min(big_m_upper, var.upper)``
   - ``M_lower = max(big_m_lower, var.lower)``

   This ensures a loose ``big_m`` won't make the relaxation worse than using bounds alone.

**Why tighter Big-M matters:**

Tighter Big-M values lead to:

- Better LP relaxation bounds (closer to optimal integer solution)
- Fewer branch-and-bound nodes
- Faster solve times

.. code-block:: python

    # Example: Variable has large bounds but we know effective range is smaller
    x = m.add_variables(lower=0, upper=1000, coords=[idx], name="x")

    # Other constraints limit x to [0, 10] in practice
    m.add_constraints(x <= 10)

    # Use tight Big-M for better performance
    m.add_sos_constraints(x, sos_type=1, sos_dim="i", big_m=10)

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

The reformulation converts SOS constraints into Mixed-Integer Linear Programming
(MILP) constraints using binary indicator variables.

**SOS1 Reformulation**

Given variables :math:`x_i` for :math:`i \in I` with bounds :math:`L_i \leq x_i \leq U_i`,
the SOS1 constraint "at most one :math:`x_i` is non-zero" is reformulated as:

.. math::

    \text{Binary indicators:} \quad & y_i \in \{0, 1\} \quad \forall i \in I \\[0.5em]
    \text{Upper linking:} \quad & x_i \leq U_i \cdot y_i \quad \forall i \in I \text{ where } U_i > 0 \\[0.5em]
    \text{Lower linking:} \quad & x_i \geq L_i \cdot y_i \quad \forall i \in I \text{ where } L_i < 0 \\[0.5em]
    \text{Cardinality:} \quad & \sum_{i \in I} y_i \leq 1

**Interpretation:**

- :math:`y_i = 1` means variable :math:`x_i` is "selected" (allowed to be non-zero)
- :math:`y_i = 0` forces :math:`x_i = 0` via the linking constraints
- The cardinality constraint ensures at most one :math:`y_i = 1`

**Example:** For :math:`x \in [0, 10]`:

- If :math:`y = 0`: constraint :math:`x \leq 10 \cdot 0 = 0` forces :math:`x = 0`
- If :math:`y = 1`: constraint :math:`x \leq 10 \cdot 1 = 10` allows :math:`x \in [0, 10]`

**SOS2 Reformulation**

Given ordered variables :math:`x_0, x_1, \ldots, x_{n-1}` with bounds :math:`L_i \leq x_i \leq U_i`,
the SOS2 constraint "at most two adjacent :math:`x_i` are non-zero" is reformulated using
segment indicators:

.. math::

    \text{Segment indicators:} \quad & z_j \in \{0, 1\} \quad \forall j \in \{0, \ldots, n-2\} \\[0.5em]
    \text{First variable:} \quad & x_0 \leq U_0 \cdot z_0 \\[0.5em]
    \text{Middle variables:} \quad & x_i \leq U_i \cdot (z_{i-1} + z_i) \quad \forall i \in \{1, \ldots, n-2\} \\[0.5em]
    \text{Last variable:} \quad & x_{n-1} \leq U_{n-1} \cdot z_{n-2} \\[0.5em]
    \text{Cardinality:} \quad & \sum_{j=0}^{n-2} z_j \leq 1

(Similar constraints for lower bounds when :math:`L_i < 0`)

**Interpretation:**

- :math:`z_j = 1` means "segment :math:`j`" is active (between positions :math:`j` and :math:`j+1`)
- Variable :math:`x_i` can only be non-zero if an adjacent segment is active
- The cardinality constraint ensures at most one segment is active
- This guarantees at most two adjacent variables :math:`(x_j, x_{j+1})` are non-zero

**Example:** For :math:`n = 4` variables :math:`(x_0, x_1, x_2, x_3)`:

- If :math:`z_1 = 1` (segment 1 active): :math:`x_1` and :math:`x_2` can be non-zero
- Constraints force :math:`x_0 = 0` (needs :math:`z_0`) and :math:`x_3 = 0` (needs :math:`z_2`)

Auxiliary Variables and Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reformulation creates auxiliary variables and constraints with a ``_sos_reform_`` prefix:

.. code-block:: python

    m.reformulate_sos_constraints()

    # For SOS1 variable 'x':
    # - Binary indicators: _sos_reform_x_y
    # - Upper constraints: _sos_reform_x_upper
    # - Lower constraints: _sos_reform_x_lower (if L < 0)
    # - Cardinality:       _sos_reform_x_card

    # For SOS2 variable 'x':
    # - Segment indicators: _sos_reform_x_z
    # - Upper constraints:  _sos_reform_x_upper_first, _sos_reform_x_upper_mid_*, _sos_reform_x_upper_last
    # - Lower constraints:  _sos_reform_x_lower_first, _sos_reform_x_lower_mid_*, _sos_reform_x_lower_last
    # - Cardinality:        _sos_reform_x_card

You can use a custom prefix:

.. code-block:: python

    m.reformulate_sos_constraints(prefix="_my_sos_")

Multi-dimensional Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For multi-dimensional variables, the reformulation respects xarray broadcasting.
Constraints are created for each combination of non-SOS dimensions:

.. code-block:: python

    # 2D variable: periods × options
    periods = pd.Index([0, 1], name="periods")
    options = pd.Index([0, 1, 2], name="options")
    x = m.add_variables(lower=0, upper=1, coords=[periods, options], name="x")
    m.add_sos_constraints(x, sos_type=1, sos_dim="options")

    # After reformulation:
    # - Binary y has shape (periods: 2, options: 3)
    # - Cardinality constraint: sum over 'options' for each period
    # - Result: at most one option selected PER period

Edge Cases
~~~~~~~~~~

The reformulation handles several edge cases automatically:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Case
     - Handling
   * - Single-element SOS
     - Skipped (trivially satisfied)
   * - All-zero bounds (L=U=0)
     - Skipped (variable already fixed to 0)
   * - All-positive bounds
     - Only upper linking constraints created
   * - All-negative bounds
     - Only lower linking constraints created
   * - Mixed bounds
     - Both upper and lower constraints created

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Reformulation trade-offs:**

- **Pros:** Works with any MIP solver; explicit constraints can sometimes help the solver
- **Cons:** Adds binary variables and constraints; may be slower than native SOS support

**When to use native SOS:**

- Use native SOS (Gurobi, CPLEX) when available—solvers have specialized branching strategies
- Native SOS often provides better performance for piecewise linear problems

**When reformulation is useful:**

- When you must use a solver without SOS support (HiGHS, GLPK)
- For model portability across different solvers
- When debugging—explicit constraints are easier to inspect

Example: Piecewise Linear with HiGHS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import linopy
    import pandas as pd
    import numpy as np

    # Approximate f(x) = x² using piecewise linear with HiGHS
    m = linopy.Model()

    breakpoints = pd.Index([0.0, 1.0, 2.0, 3.0], name="bp")
    x_vals = breakpoints.to_numpy()
    y_vals = x_vals**2  # [0, 1, 4, 9]

    # SOS2 interpolation weights
    lambdas = m.add_variables(lower=0, upper=1, coords=[breakpoints], name="lambdas")
    m.add_sos_constraints(lambdas, sos_type=2, sos_dim="bp")

    # Interpolated point
    x = m.add_variables(name="x", lower=0, upper=3)
    y = m.add_variables(name="y", lower=0, upper=9)

    # Constraints
    m.add_constraints(lambdas.sum() == 1, name="convexity")
    m.add_constraints(x == (lambdas * x_vals).sum(), name="x_interp")
    m.add_constraints(y == (lambdas * y_vals).sum(), name="y_interp")
    m.add_constraints(x >= 1.5, name="x_min")

    # Minimize y (approximated x²)
    m.add_objective(y)

    # Solve with HiGHS using reformulation
    m.solve(solver_name="highs", reformulate_sos=True)

    print(f"x = {x.solution.item():.2f}")
    print(f"y ≈ x² = {y.solution.item():.2f}")
    # Output: x = 1.50, y ≈ x² = 2.50 (exact: 2.25)

Common Patterns
---------------

Piecewise Linear Cost Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def add_piecewise_cost(model, variable, breakpoints, costs):
        """Add piecewise linear cost function using SOS2."""
        n_segments = len(breakpoints)
        lambda_coords = pd.Index(range(n_segments), name="segments")

        lambdas = model.add_variables(
            coords=[lambda_coords], name="cost_lambdas", lower=0, upper=1
        )
        model.add_sos_constraints(lambdas, sos_type=2, sos_dim="segments")

        cost_var = model.add_variables(name="cost", lower=0)

        x_vals = xr.DataArray(breakpoints, coords=[lambda_coords])
        c_vals = xr.DataArray(costs, coords=[lambda_coords])

        model.add_constraints(lambdas.sum() == 1, name="cost_convexity")
        model.add_constraints(variable == (x_vals * lambdas).sum(), name="cost_x_def")
        model.add_constraints(cost_var == (c_vals * lambdas).sum(), name="cost_def")

        return cost_var

Mutually Exclusive Investments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def add_exclusive_investments(model, projects, costs, returns):
        """Add mutually exclusive investment decisions using SOS1."""
        project_coords = pd.Index(projects, name="projects")

        invest = model.add_variables(
            coords=[project_coords], name="invest", binary=True
        )
        model.add_sos_constraints(invest, sos_type=1, sos_dim="projects")

        total_cost = (invest * costs).sum()
        total_return = (invest * returns).sum()

        return invest, total_cost, total_return


See Also
--------

- :doc:`creating-variables`: Creating variables with coordinates
- :doc:`creating-constraints`: Adding regular constraints
- :doc:`user-guide`: General linopy usage patterns

API Reference
-------------

.. py:method:: Model.add_sos_constraints(variable, sos_type, sos_dim, big_m=None)

   Add an SOS1 or SOS2 constraint for one dimension of a variable.

   :param variable: Variable to constrain
   :type variable: Variable
   :param sos_type: Type of SOS constraint (1 or 2)
   :type sos_type: Literal[1, 2]
   :param sos_dim: Dimension along which to apply the SOS constraint
   :type sos_dim: str
   :param big_m: Custom Big-M value(s) for reformulation. Can be:

      - ``None`` (default): Use variable bounds
      - ``float``: Symmetric Big-M (upper = big_m, lower = -big_m)
      - ``tuple[float, float]``: Asymmetric (upper, lower)

   :type big_m: float | tuple[float, float] | None

.. py:method:: Model.remove_sos_constraints(variable)

   Remove SOS constraints from a variable.

   :param variable: Variable from which to remove SOS constraints
   :type variable: Variable

.. py:method:: Model.reformulate_sos_constraints(prefix="_sos_reform_")

   Reformulate SOS constraints as binary + linear constraints using the Big-M method.

   :param prefix: Prefix for auxiliary variable and constraint names
   :type prefix: str
   :returns: List of variable names that were reformulated
   :rtype: list[str]
   :raises ValueError: If any SOS variable has infinite bounds and no custom big_m

.. py:attribute:: Variables.sos

   Property returning all variables with SOS constraints.
