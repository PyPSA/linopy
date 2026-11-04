# Overview

In Getting Started you [installed linopy](../home/installation.md), built
a [first scalar model](../home/create-a-model.ipynb), and saw
[N-D variables on coordinates](../home/create-a-model-with-coordinates.ipynb). The User
Guide reopens each of those pieces in depth and adds the rest of the
modelling surface.

Most pages are runnable Jupyter notebooks — read them top to bottom, or
use them as a reference once you know what you're looking for.

## Core building blocks

The four notebooks below cover the model object you'll interact with
most. Read them in order the first time; come back to them whenever
you're unsure what a particular operator or argument does.

- [Creating Variables](creating-variables.ipynb) — declaring decision variables, with bounds
    and coordinates. Continuous, integer, binary, and semi-continuous.
- [Creating Expressions](creating-expressions.ipynb) — combining variables into linear (and
    quadratic) expressions; arithmetic, broadcasting, `sum`,
    `groupby`, `rolling`, `where`.
- [Creating Constraints](creating-constraints.ipynb) — turning expressions into `≤` / `≥`
    / `==` constraints, held dense or sparse (CSR).
- [Coordinate Alignment](coordinate-alignment.ipynb) — how linopy lines up operands that live
    on different coordinates, and how to control it with `join`.

After these four you can build any LP/MIP/QP linopy supports.

If your data comes as tables, one row per key combination, read
[Long-form input data](long-form-inputs.ipynb). It pivots such frames with
`linopy.from_long` and sums over member-group tables with
`linopy.Relation`.

## Dense and sparse models

Every model holds its expressions in one of two representations, picked
with `Model(sparse=...)` at construction. The dense representation, the
default, pads each row to the widest row along `_term`. The sparse
representation stores each expression as a CSR matrix (compressed sparse
rows) and holds every constraint's left-hand side in CSR form whenever the
grid can hold its coordinates. Both share one API and give the same model.

```python
m = linopy.Model(sparse=True)
```

Sparse suits models that aggregate many entities into few rows, such as
generators into buses. Operations without a sparse path run on a dense
copy, emit a `PerformanceWarning` and leave the expression sparse. See
[Dense and sparse models](sparse-models.ipynb) for the operations that keep the sparse backing, the
ones that densify, and the `warn_on_densify` option.

## Numerical scaling

Large or very small coefficients can make mathematical programs harder
for solvers to process. Linopy therefore accepts optional positive,
finite `scaling` factors when adding variables, constraints, and the
objective.

For continuous and semi-continuous variables, `scaling=s` means the
solver-side column represents `s * x`. Bounds are multiplied by `s`,
and coefficients that reference `x` are divided by `s`. Binary and
integer variable scaling is stored and round-tripped, but their solver
columns remain ordinary discrete columns.

For constraints, `scaling=s` multiplies both the left-hand-side
coefficients and the right-hand side by `s`. Objective scaling is
row-like: `scaling=s` multiplies linear and quadratic objective
coefficients by `s`. Primal values, dual values, and objective values
are transformed back to the original user units after solving.

Equivalently, for constraint matrix `A`, right-hand side `b`,
linear objective coefficients `c`, variable scaling matrix `Sx`,
constraint scaling matrix `Sc`, and scalar objective scaling `so`,
linopy exports `y = Sx x` and
`A_solver = Sc A Sx^-1`, `b_solver = Sc b`, and
`c_solver = so c Sx^-1`. Quadratic objective coefficients receive the
inverse variable scaling once for each variable factor and the scalar
objective scaling once globally.

```python
x = m.add_variables(lower=0, scaling=1e3, name="x")
m.add_constraints(2 * x >= 10, scaling=10, name="demand")
m.add_objective(5 * x, scaling=100)
```

For a worked example that builds a badly-scaled model and applies each of the
three variants step by step, see the [Numerical Scaling](numerical-scaling.ipynb) tutorial.

## Working with an existing model

Once you've built a model, you'll often want to inspect it, change a
bound, swap a constraint, or copy it for what-if analysis.

- [Modifying Models](manipulating-models.ipynb) — modifying or removing variables and
    constraints in place; `Model.copy()`; `fix` / `relax` for
    variables.

## Building a model from a spec

Instead of calling `add_variables` / `add_constraints` directly,
you can declare a model as a [mathspec](https://mathspec.readthedocs.io) YAML program attached to
data, and let linopy build it.

- [Building models from mathspec programs](building-models-from-specs.ipynb) — `Model.from_spec`, attaching
    data to a spec, solving, and folding named expressions with
    `linopy.spec.evaluate`. Requires `pip install "linopy[spec]"`
    (Python \>= 3.12).
- [Spec data and results](spec-data.md) — the source contract and missing rows, storing the
    data in `model.parameters`, reading results, saving and rebuilding,
    typesetting and variants.

linopy supports a subset of mathspec. A constraint row that absence
leaves without terms (a `missing: absent` parameter, a `missing: neutral` parameter such as `0 == load`, or a `where:`-masked
variable inside a sum) is refused, where mathspec skips and reports it;
linopy is stricter here ([\#993](https://github.com/PyPSA/linopy/issues/993)). `store=True` stores
only relations of one key column and one value column, and raises
`SpecDataError` for any other.

## Where to go next

- **Solving** — [Using Solvers](using-solvers.ipynb),
    [Remote Solving with SSH](solve-on-remote.ipynb),
    [GPU-Accelerated Solving](gpu-acceleration.md) and
    [Numerical Scaling](numerical-scaling.ipynb).
- **Advanced formulations** — [SOS Constraints](sos-constraints.md),
    [Piecewise Linear Constraints](piecewise-linear-constraints.md), and
    [Testing with Linopy](testing-framework.ipynb) for asserting structural
    properties of a model.
- **Troubleshooting** — [diagnosing infeasible models](infeasible-model.ipynb)
    and [double logging in Gurobi](gurobi-double-logging.md).
- **Coming from other tools** — the [syntax comparison](syntax.md), the
    [benchmarks](benchmark.md) and [Migrating from Pyomo](migrating-from-pyomo.ipynb).
- **Examples** — end-to-end walkthroughs:
    [The Transport Problem](../examples/transport-tutorial.ipynb),
    [Power Portfolio Build-Out](../examples/energy-resource-build.ipynb) and
    [Sudoku Solver](../examples/sudoku.ipynb).
- **Reference** — the full [API reference](../api/index.md).
