# Installation

This page covers installing linopy and a solver.

Before you start, make sure you have the following:

- Python 3.11 or later installed on your system.
- Basic knowledge of Python and linear programming.

## Install Linopy

You can install Linopy using uv or conda. Here are the commands for each method:

```bash
uv pip install linopy
```

or

```bash
conda install -c conda-forge linopy
```

## Optional dependencies

The base install pulls numpy, scipy, xarray, pandas, polars, tqdm, packaging
and bottleneck. Chunked (dask-backed) arrays, for example via
`Variable.chunk()`, need the `dask` extra:

```bash
uv pip install "linopy[dask]"
```

## Install a solver

Linopy won't work without a solver. Currently, the following solvers are supported:

### CPU-based solvers

- [Cbc](https://projects.coin-or.org/Cbc) - open source, free, fast
- [GLPK](https://www.gnu.org/software/glpk/) - open source, free, not very fast
- [HiGHS](https://highs.dev/) - open source, free, fast
- [SCIP](https://www.scipopt.org/) - open source (Apache-2.0), fast MIP solver
- [Gurobi](https://www.gurobi.com/) - closed source, commercial, very fast
- [Xpress](https://www.fico.com/en/fico-xpress-trial-and-licensing-options) - closed source, commercial, very fast (GPU acceleration available in v9.8+)
- [Knitro](https://www.artelys.com/solvers/knitro/) - closed source, commercial, strong on nonlinear problems
- [Cplex](https://www.ibm.com/de-de/analytics/cplex-optimizer) - closed source, commercial, very fast
- [MOSEK](https://www.mosek.com/) - closed source, commercial, strong on conic/QP
- [MindOpt](https://solver.damo.alibaba.com/doc/en/html/index.html) - closed source, commercial
- [COPT](https://www.shanshu.ai/copt) - closed source, commercial, very fast

The `linopy[solvers]` extra installs the Python clients for the
supported solvers (HiGHS, SCIP, Gurobi, CPLEX, MOSEK, MindOpt, COPT,
Xpress, Knitro). For the commercial ones a separate license is still
required:

```bash
uv pip install "linopy[solvers]"
```

We recommend the HiGHS solver, which is free, open source, and fast
across a wide range of problem sizes. It is included in both the
`solvers` and `dev` extras.

### GPU-accelerated solvers

For large-scale optimization problems, GPU-accelerated solvers can provide significant performance improvements:

- [cuOpt](https://docs.nvidia.com/cuopt/) - open source, GPU-accelerated LP and MIP solver by NVIDIA
- [cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx) - open source, GPU-accelerated first-order solver

!!! note

    GPU solvers require compatible NVIDIA GPU hardware and CUDA installation. See the [GPU-Accelerated Solving](../user-guide/gpu-acceleration.md) guide for detailed setup instructions.

The `linopy[gpu]` extra installs cuOpt. It is **Linux-only** and requires an
NVIDIA GPU of **compute capability 7.0 or higher** with a **CUDA 12 driver,
version 525.60.13 or newer**. It is not part of `linopy[solvers]`, because it
pulls in several GB of CUDA packages:

```bash
uv pip install "linopy[gpu]"
```

The extra installs `cuopt-cu12`, the build for CUDA 12 drivers; the bare
`cuopt` package on PyPI is an unrelated project.

cuPDLPx is installed on its own:

```bash
uv pip install cupdlpx
```

For most of the other solvers, please click on the links to get further installation information.

Once a solver is installed, continue with [Create a model](create-a-model.ipynb).
