Benchmarking
============


Linopy's performance scales well with the problem size. Its overall speed is comparable with the mighty `JuMP <https://jump.dev/>`_ package written in `Julia <https://julialang.org/>`_. It even outperforms `JuMP` for large models as we see in the following comparison. We initialize a very simple optimization model

.. math::

    & \min \;\; 2 x_{i,j} \; y_{i,j} \qquad \forall \; i,j \in \{1,...,N\} \\
    s.t. & \\
    & x_{i,j} - y_{i,j} \; \ge \; i \qquad \forall \; i,j \in \{1,...,N\}


In `JuMP` this translates to the following code:

 .. code-block:: julia

    using JuMP

    function create_model(N)
        m = Model()
        @variable(m, x[1:N, 1:N])
        @variable(m, y[1:N, 1:N])
        @constraint(m, [i=1:N, j=1:N], x[i, j] - y[i, j] >= i)
        @objective(m, Min, sum(2 * x[i, j] + y[i, j] for i in 1:N, j in 1:N))
        return m
    end

When running it with :math:`N=1000` (after a first compilation run), the initialization takes around 3.3 seconds.

 .. code-block:: julia

    @time create_model(1000)

 `> 3.328916 seconds (32.00 M allocations: 1.976 GiB, 37.59% gc time)`

When initializing the same problem with `linopy`, the code runs for

 .. code-block:: python

     from linopy import Model
     from numpy import arange


     def create_model(N):
         m = Model()
         coords = [arange(N), arange(N)]
         x = m.add_variables(coords=coords)
         y = m.add_variables(coords=coords)
         m.add_constraints(x - y >= arange(N))
         m.add_constraints(x + y >= 0)
         m.add_objective((2 * x).sum() + y.sum())
         return

 .. code-block:: python

     %time create_model(1000)

 | `> CPU times: user 86.1 ms, sys: 20.5 ms, total: 107 ms`
 | `> Wall time: 106 ms`



or `Pyomo <https://www.pyomo.org/>`_
