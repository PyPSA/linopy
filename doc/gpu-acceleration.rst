========================
GPU-Accelerated Solving
========================

.. warning::

   This feature is **experimental** and not tested in CI due to the lack of GPU-enabled machines. Use with caution and please report any issues.

Linopy supports GPU-accelerated optimization solvers that can significantly speed up solving large-scale linear programming problems by leveraging the parallel processing capabilities of modern GPUs.

Supported GPU Solvers
=====================

cuPDLPx
-------

`cuPDLPx <https://github.com/MIT-Lu-Lab/cuPDLPx>`_ is an open-source, GPU-accelerated first-order solver developed by MIT. It implements a Primal-Dual hybrid gradient (PDHG) method optimized for GPUs.

To install it, you have to have the `CUDA Toolkit <https://developer.nvidia.com/cuda/toolkit>`_ installed requiring NVIDIA GPUs on your computer. Then, install with

.. code-block:: bash

    # Install CUDA Toolkit first (if not already installed)
    # Follow instructions at: https://developer.nvidia.com/cuda-downloads

    # Install cuPDLPx
    pip install cupdlpx>=0.1.2

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

- NVIDIA GPU with CUDA support (compute capability 6.0 or higher recommended)
- Sufficient GPU memory for your problem size (varies by problem)
- PCIe 3.0 or higher for optimal data transfer

Software Requirements
---------------------

1. **CUDA Toolkit**: Most GPU solvers require CUDA 11.0 or later
2. **Compatible GPU drivers**: Match your CUDA version

Verifying Installation
======================

To verify that the GPU solvers are properly installed and detected:

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
    pytest test/test_optimization.py -k cupdlpx --run-gpu


References
==========

- `cuPDLPx Repository <https://github.com/MIT-Lu-Lab/cuPDLPx>`_
- `cuPDLPx Python Documentation <https://github.com/MIT-Lu-Lab/cuPDLPx/tree/main/python>`_
- `CUDA Installation Guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_
- `NVIDIA GPU Computing Resources <https://developer.nvidia.com/gpu-computing>`_
