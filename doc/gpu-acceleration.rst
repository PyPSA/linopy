========================
GPU-Accelerated Solving
========================

.. warning::

   This feature is **experimental** and not tested in CI due to the lack of GPU-enabled machines. Use with caution and please report any issues.

Linopy supports GPU-accelerated optimization solvers that can significantly speed up solving large-scale linear programming problems by leveraging the parallel processing capabilities of modern GPUs.

Supported GPU Solvers
=====================

cuOpt
-----

`NVIDIA cuOpt <https://docs.nvidia.com/cuopt/>`_ is NVIDIA's open-source (Apache 2.0) GPU-accelerated optimization solver. For linear and mixed-integer problems it offers a GPU barrier method, a GPU first-order method (PDLP) and a CPU dual simplex, run individually or concurrently.

To install it together with linopy, use the ``gpu`` extra:

.. code-block:: bash

    pip install "linopy[gpu]"

or

.. code-block:: bash

    uv pip install "linopy[gpu]"

The extra is **Linux-only** (there are no macOS or Windows wheels) and requires an **NVIDIA GPU of compute capability 7.0 or higher** with a **CUDA 12 driver, version 525.60.13 or newer**. It is deliberately not part of ``linopy[solvers]``: it pulls in ``cuopt-cu12`` and its CUDA dependencies, several GB unpacked.

.. note::

   The extra installs ``cuopt-cu12``, the build for CUDA 12 drivers. The ``cu13`` build requires a newer driver and linopy does not select between the two. Also note that the bare ``cuopt`` package on PyPI is an unrelated project — do not ``pip install cuopt``.

**Usage:**

.. code-block:: python

    m.solve("cuopt")

    # the same solve, without the file IO warning
    m.solve("cuopt", io_api="direct")

    # solver options are passed on as keyword arguments
    m.solve("cuopt", io_api="direct", method=1, time_limit=60.0)

**Features:**

- GPU-accelerated solving for large-scale linear programs
- Linear Programming (LP) and Mixed-Integer Programming (MIP)
- Convex quadratic objectives (QP), solved with the GPU barrier method
- Continuous, binary, integer and semi-continuous variables
- Inequality and equality constraints
- Duals for LP models; MIP gap and dual bound reported via ``model.solver.report``
- Open source (Apache 2.0 license)

**Limitations:**

- Only the direct API is supported. Passing ``io_api=None`` or a file ``io_api`` (``lp``, ``lp-polars``, ``mps``) still solves, but the model is built through the direct API — with an informational log message for the default ``io_api=None`` and a warning for an explicitly requested file ``io_api``. An empty temporary problem file is created and removed either way; ``io_api="direct"`` only skips the message.
- ``keep_files=True`` is not supported: it makes linopy ask the solver for a solution file, which cuOpt cannot write, so the solve raises ``NotImplementedError``
- Quadratic objectives combined with integer or semi-continuous variables (MIQP) are not supported and are rejected with a ``NotImplementedError`` before the solve. cuOpt does not refuse such a model itself — it returns an empty solution, which is indistinguishable from a failed solve — so linopy checks up front.
- A quadratic objective must be convex for minimisation (concave for maximisation). cuOpt detects a Hessian that is not positive semi-definite and reports ``NumericalError``, which linopy surfaces as the ``internal_solver_error`` termination condition rather than a wrong answer. The same applies to a quadratic objective on an unbounded model.
- Quadratic *constraints* are not supported
- SOS and indicator constraints are not supported
- No warm start and no basis files: a basis file is ignored with a warning, a warmstart file raises ``NotImplementedError``. cuOpt's own PDLP warm start requires ``method=1``, ``pdlp_solver_mode=1`` and ``presolve=0`` at the same time, and its payload is not a file, so it is not wired up.
- No solution files — cuOpt is called through its Python API and the solution is read from memory
- Reduced costs are not returned. linopy has no interface for them, and cuOpt's reduced-cost sign is unreliable for maximisation problems.
- Remote cuOpt execution (``CUOPT_REMOTE_HOST``) is not used. If cuOpt suggests setting that variable, it means no usable local GPU was found.
- Default tolerances are ``1e-4``, so objective values can differ from a simplex solver in the fourth or fifth significant digit

**Notes:**

- linopy defaults to ``method=3`` (Barrier) instead of cuOpt's own default ``method=0`` (Concurrent), which can crash the process on repeated solves of models with more than about 1300 variables. For very large sparse LPs, try ``method=1`` (PDLP).
- On a model with a quadratic objective, cuOpt always solves with the barrier method: it silently overrides ``method`` and ``crossover``, so passing either has no effect on a QP.
- Solver options are checked on the way in: an unrecognised option raises ``ValueError`` naming the offending parameter (cuOpt's own message does not say which one it rejected), and Python booleans are converted for cuOpt's integer-typed parameters, so ``presolve=False`` works as expected.
- A limit termination (``time_limit``, ``iteration_limit``, ``node_limit``, ``first_primal_feasible``) returns status ``ok`` with the matching termination condition — ``time_limit``, ``iteration_limit``, or ``suboptimal`` for an unproven incumbent. A limit *setting* never implies a limit *status*: cuOpt may still finish ``optimal`` within the limit. If the limit expires before any feasible point is found, linopy returns an empty solution rather than scattering cuOpt's empty primal; a termination status linopy does not recognise maps to the ``unknown`` condition.
- ``log_file`` (or linopy's ``log_fn``) writes cuOpt's log to a file, truncating it; if both are given, ``log_fn`` wins. ``log_to_console`` controls the console output.
- linopy imports only ``cuopt.linear_programming``. Importing ``cuopt.routing`` yourself installs a global ``sys.excepthook`` that writes an ``error_log.txt`` into the current working directory.
- ``Ctrl-C`` returns control to Python, but the GPU solve runs to completion in the background. ``time_limit`` is the only hard bound on solve time.
- On a machine without a usable GPU, cuOpt is simply absent from ``linopy.available_solvers``

For the complete list of cuOpt parameters, see the `cuOpt documentation <https://docs.nvidia.com/cuopt/>`_.

cuPDLPx
-------

`cuPDLPx <https://github.com/MIT-Lu-Lab/cuPDLPx>`_ is an open-source, GPU-accelerated first-order solver developed by MIT. It implements a Primal-Dual hybrid gradient (PDHG) method optimized for GPUs.

To install it, you have to have the `CUDA Toolkit <https://developer.nvidia.com/cuda/toolkit>`_ installed requiring NVIDIA GPUs on your computer. Then, install with

.. code-block:: bash

    # Install CUDA Toolkit first (if not already installed)
    # Follow instructions at: https://developer.nvidia.com/cuda-downloads

    # Install cuPDLPx
    uv pip install "cupdlpx>=0.1.2"

**Features:**

- GPU-accelerated solving for large-scale linear programs
- Open source (Apache 2.0 license)
- Direct API integration with linopy
- Designed for problems with millions of variables and constraints

**Limitations:**

- Currently supports only Linear Programming (LP)
- Does not support Mixed-Integer Programming (MIP) or Quadratic Programming (QP)
- Lower numerical precision compared to CPU solvers (typical tolerance: ~2.5e-4 vs 1e-5)
- File I/O not currently supported through cuPDLPx API

For a complete list of cuPDLPx parameters, see the `cuPDLPx documentation <https://github.com/MIT-Lu-Lab/cuPDLPx/tree/main/python#parameters>`_.

Xpress with GPU Acceleration
-----------------------------

`FICO Xpress <https://www.fico.com/en/fico-xpress-trial-and-licensing-options>`_ version 9.8 and later includes GPU acceleration support for certain operations.

**Features:**

- Commercial solver with GPU support
- Supports LP, MIP, and QP
- Full-precision solving

Prerequisites
=============

Hardware Requirements
---------------------

GPU solvers require:

- NVIDIA GPU with CUDA support (compute capability 6.0 or higher recommended; cuOpt requires compute capability 7.0 or higher — see above)
- Sufficient GPU memory for your problem size (varies by problem)
- PCIe 3.0 or higher for optimal data transfer

Software Requirements
---------------------

1. **CUDA Toolkit**: Most GPU solvers require CUDA 11.0 or later (cuOpt requires a CUDA 12 driver, version 525.60.13 or newer — see above)
2. **Compatible GPU drivers**: Match your CUDA version

Verifying Installation
======================

To verify that the GPU solvers are properly installed and detected, check that they appear in the lists below — cuOpt only reports itself as available if a usable GPU is found, so a missing entry points at the driver or the hardware rather than at the installation:

.. code-block:: python

    import linopy
    from linopy.solver_capabilities import (
        SolverFeature,
        get_available_solvers_with_feature,
    )

    # Check available solvers
    print("All available solvers:", linopy.available_solvers)

    # Check GPU-accelerated solvers
    gpu_solvers = get_available_solvers_with_feature(
        SolverFeature.GPU_ACCELERATION, linopy.available_solvers
    )
    print("GPU solvers:", gpu_solvers)


By default, GPU tests are skipped in the test suite to support CI environments without GPUs. To run GPU tests locally:

.. code-block:: bash

    # Run all tests including GPU tests
    pytest --run-gpu

    # Run only GPU tests
    pytest -m gpu --run-gpu

    # Run specific GPU test
    pytest test/test_optimization.py -k "cuopt or cupdlpx" --run-gpu


References
==========

- `cuOpt Documentation <https://docs.nvidia.com/cuopt/>`_
- `cuOpt Repository <https://github.com/NVIDIA/cuopt>`_
- `cuPDLPx Repository <https://github.com/MIT-Lu-Lab/cuPDLPx>`_
- `cuPDLPx Python Documentation <https://github.com/MIT-Lu-Lab/cuPDLPx/tree/main/python>`_
- `CUDA Installation Guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_
- `NVIDIA GPU Computing Resources <https://developer.nvidia.com/gpu-computing>`_
