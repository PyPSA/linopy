
Syntax comparison
=================

In order to compare the syntax between different API's, let's initialize the following problem in the different API's:

.. math::

    & \min \;\; \sum_{i,j} 2 x_{i,j} + \; y_{i,j} \\
    s.t. & \\
    & x_{i,j} - y_{i,j} \; \ge \; i-1 \qquad \forall \; i,j \in \{1,...,N\} \\
    & x_{i,j} + y_{i,j} \; \ge \; 0 \qquad \forall \; i,j \in \{1,...,N\}





In ``JuMP`` the formulation translates to the following code:

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

The same model in ``linopy`` is initialized by

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

Note that the syntax is quite similar.

In ``Pyomo`` the code would look like

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

which is heavily based on the internal call of functions in order to define the constraints.
