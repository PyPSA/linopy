Getting Started
===============

This guide will provide you with the necessary steps to get started with Linopy, from installation to creating your first model and beyond.

Before you start, make sure you have the following:

- Python 3.9 or later installed on your system.
- Basic knowledge of Python and linear programming.


Install Linopy
--------------

You can install Linopy using pip or conda. Here are the commands for each method:

.. code-block:: bash

   pip install linopy

or

.. code-block:: bash

   conda install -c conda-forge linopy


Install a solver
----------------

Linopy won't work without a solver. Currently, the following solvers are supported:

-  `Cbc <https://projects.coin-or.org/Cbc>`__ - open source, free, fast
-  `GLPK <https://www.gnu.org/software/glpk/>`__ - open source, free, not very fast
-  `HiGHS <https://www.maths.ed.ac.uk/hall/HiGHS/>`__ - open source, free, fast
-  `Gurobi <https://www.gurobi.com/>`__  - closed source, commercial, very fast
-  `Xpress <https://www.fico.com/en/products/fico-xpress-solver>`__ - closed source, commercial, very fast
-  `Cplex <https://www.ibm.com/de-de/analytics/cplex-optimizer>`__ - closed source, commercial, very fast
-  `MOSEK <https://www.mosek.com/>`__
-  `MindOpt <https://solver.damo.alibaba.com/doc/en/html/index.html>`__ -
-  `COPT <https://www.shanshu.ai/copt>`__ - closed source, commercial, very fast

For a subset of the solvers, Linopy provides a wrapper.

.. code:: bash

    pip install linopy[solvers]


We recommend to install the HiGHS solver if possible, which is free and open source but not yet available on all platforms.

.. code:: bash

    pip install highspy


For most of the other solvers, please click on the links to get further installation information.



If you're ready, let's dive in!
