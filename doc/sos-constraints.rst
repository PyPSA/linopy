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
