.. _coming-from-other-tools:

Coming from other tools
=======================

This page is for users arriving from JuMP, Pyomo, or GAMS. It compares syntax on a common toy problem, reports linopy's performance against the alternatives, and shows a Pyomo-style "function over coordinates" pattern that linopy also supports.

If you are coming from GAMS specifically, see :doc:`transport-tutorial` — a full walk-through of the classic GAMS transport problem with linopy-equivalent code annotated alongside the original GAMS syntax.


Syntax cheatsheet
-----------------

To compare the API surface, let's formulate the same toy problem in each tool:

.. math::

    & \min \;\; \sum_{i,j} 2 x_{i,j} + \; y_{i,j} \\
    s.t. & \\
    & x_{i,j} - y_{i,j} \; \ge \; i-1 \qquad \forall \; i,j \in \{1,...,N\} \\
    & x_{i,j} + y_{i,j} \; \ge \; 0 \qquad \forall \; i,j \in \{1,...,N\}


**JuMP** (Julia):

.. code-block:: julia

    using JuMP

    function create_model(N)
        m = Model()
        @variable(m, x[1:N, 1:N])
        @variable(m, y[1:N, 1:N])
        @constraint(m, x - y .>= 0:(N-1))
        @constraint(m, x + y .>= 0)
        @objective(m, Min, 2 * sum(x) + sum(y))
        return m
    end

**linopy** (Python):

.. code-block:: python

    from linopy import Model
    from numpy import arange


    def create_model(N):
        m = Model()
        x = m.add_variables(coords=[arange(N), arange(N)])
        y = m.add_variables(coords=[arange(N), arange(N)])
        m.add_constraints(x - y >= arange(N))
        m.add_constraints(x + y >= 0)
        m.add_objective((2 * x).sum() + y.sum())
        return m

The linopy and JuMP formulations are close in spirit: both rely on broadcasting and array-style operations rather than explicit per-element loops.

**Pyomo** (Python):

.. code-block:: python

    from numpy import arange
    from pyomo.environ import ConcreteModel, Constraint, Objective, Set, Var


    def create_model(N):
        m = ConcreteModel()
        m.N = Set(initialize=arange(N))

        m.x = Var(m.N, m.N, bounds=(None, None))
        m.y = Var(m.N, m.N, bounds=(None, None))

        def bound1(m, i, j):
            return m.x[(i, j)] - m.y[(i, j)] >= i

        def bound2(m, i, j):
            return m.x[(i, j)] + m.y[(i, j)] >= 0

        def objective(m):
            return sum(2 * m.x[(i, j)] + m.y[(i, j)] for i in m.N for j in m.N)

        m.con1 = Constraint(m.N, m.N, rule=bound1)
        m.con2 = Constraint(m.N, m.N, rule=bound2)
        m.obj = Objective(rule=objective)
        return m

Pyomo builds constraints from element-wise rule functions instead of vectorised expressions.


Performance
-----------

linopy's performance scales well with problem size, comparable to `JuMP <https://jump.dev/>`_ on speed and ahead of it on memory efficiency for large models. Against `Pyomo <https://pyomo.org>`_, linopy typically delivers:

* a **speedup of 4–6×**
* a **memory reduction of roughly 50%**

The figure below shows memory usage and build time on the toy problem above with the `Gurobi <https://gurobi.com>`_ solver. The benchmark workflow is `available here <https://github.com/PyPSA/linopy/tree/benchmark/benchmark>`_.

.. image:: benchmark.png
    :width: 1500
    :alt: benchmark
    :align: center


For Pyomo users: scalar-style construction
------------------------------------------

If you are used to Pyomo's pattern of constructing constraints from a rule function indexed over a set, linopy supports the same shape via :meth:`linopy.Model.linexpr` and the :attr:`linopy.Variable.at` indexer.

.. code-block:: python

    import pandas as pd
    import linopy

    m = linopy.Model()
    coords = pd.RangeIndex(10), ["a", "b"]
    x = m.add_variables(0, 100, coords, name="x")

    # Pick a single scalar entry from a variable
    x.at[0, "a"]

A :class:`linopy.expressions.ScalarVariable` is light-weight and can be combined inside a function to build expressions, just like Pyomo:

.. code-block:: python

    def bound(m, i, j):
        if i % 2:
            return (i / 2) * x.at[i, j]
        else:
            return i * x.at[i, j]


    expr = m.linexpr(bound, coords)

The first argument of the rule function must be the model itself, even when it is unused.

The same shape works for :meth:`linopy.Model.add_constraints`. When given a function as the first argument, ``add_constraints`` expects non-empty ``coords`` and the function to return an ``AnonymousScalarConstraint``:

.. code-block:: python

    def bound(m, i, j):
        if i % 2:
            return (i / 2) * x.at[i, j] >= i
        else:
            return i * x.at[i, j] == 0.0


    con = m.add_constraints(bound, coords=coords)

.. note::
   In most cases, linopy's vectorised API is faster to write and faster to build than the rule-function shape. Prefer it where the underlying constraint structure is expressible as broadcast arithmetic; reach for ``linexpr(rule, coords)`` only when per-index logic genuinely varies.

For a full GAMS-style walk-through, see :doc:`transport-tutorial`.
