Getting Started
===============

This guide will provide you with the necessary steps to get started with Linopy, from installation to creating your first model and beyond.

Before you start, make sure you have the following:

- Python 3.9 or later installed on your system.
- Basic knowledge of Python and linear programming.


Install Linopy
--------------

You can install Linopy using uv or conda. Here are the commands for each method:

.. code-block:: bash

   uv pip install linopy

or

.. code-block:: bash

   conda install -c conda-forge linopy


Install a solver
----------------

Linopy won't work without a solver. Currently, the following solvers are supported:

CPU-based solvers
~~~~~~~~~~~~~~~~~

-  `Cbc <https://projects.coin-or.org/Cbc>`__ - open source, free, fast
-  `GLPK <https://www.gnu.org/software/glpk/>`__ - open source, free, not very fast
-  `HiGHS <https://highs.dev/>`__ - open source, free, fast
-  `SCIP <https://www.scipopt.org/>`__ - open source (Apache-2.0), fast MIP solver
-  `Gurobi <https://www.gurobi.com/>`__  - closed source, commercial, very fast
-  `Xpress <https://www.fico.com/en/fico-xpress-trial-and-licensing-options>`__ - closed source, commercial, very fast (GPU acceleration available in v9.8+)
-  `Cplex <https://www.ibm.com/de-de/analytics/cplex-optimizer>`__ - closed source, commercial, very fast
-  `MOSEK <https://www.mosek.com/>`__ - closed source, commercial, strong on conic/QP
-  `MindOpt <https://solver.damo.alibaba.com/doc/en/html/index.html>`__ - closed source, commercial
-  `COPT <https://www.shanshu.ai/copt>`__ - closed source, commercial, very fast

The ``linopy[solvers]`` extra installs the Python clients for the
supported solvers (HiGHS, SCIP, Gurobi, CPLEX, MOSEK, MindOpt, COPT,
Xpress, Knitro). For the commercial ones a separate license is still
required:

.. code:: bash

    uv pip install "linopy[solvers]"


We recommend the HiGHS solver, which is free, open source, and fast
across a wide range of problem sizes. It is included in both the
``solvers`` and ``dev`` extras.


GPU-accelerated solvers
~~~~~~~~~~~~~~~~~~~~~~~

For large-scale optimization problems, GPU-accelerated solvers can provide significant performance improvements:

-  `cuPDLPx <https://github.com/MIT-Lu-Lab/cuPDLPx>`__ - open source, GPU-accelerated first-order solver

**Note:** GPU solvers require compatible NVIDIA GPU hardware and CUDA installation. See the :doc:`gpu-acceleration` guide for detailed setup instructions.

.. code:: bash

    uv pip install cupdlpx


For most of the other solvers, please click on the links to get further installation information.



If you're ready, let's dive in!
