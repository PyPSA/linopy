# Restricting a Model

[`linopy.Model.restrict()`][linopy.model.Model.restrict] cuts a subproblem out of a model. You select
labels along one or several dimensions, and pass fixed values for the
variables the selection leaves out. Linopy returns a window of the model:
the selected variables and constraints, with every term on a left-out
variable moved to the right-hand side.

This one operation serves several patterns:

- **Rolling-horizon optimization**: solve a long horizon one time slice at a
    time, carrying the end state into the next slice.
- **Spatial or sectoral decomposition**: solve one region or one sector with
    the exchange to its neighbours held fixed.
- **Fix-and-optimize heuristics**: reoptimize a part of the model while the
    rest stays at a known solution.
- **Decomposition subproblems**: condition a subproblem on master-problem
    values, as in Benders-type or ADMM-type schemes.

Three methods cover the workflow:

- [`linopy.Model.restrict()`][linopy.model.Model.restrict] returns a window of the model,
- [`linopy.restrict.WindowModel.solve()`][linopy.restrict.WindowModel.solve] solves it,
- [`linopy.restrict.WindowModel.advance()`][linopy.restrict.WindowModel.advance] moves it to a new selection.

A window is a row and column slice of the model's matrices. Creating or
moving it does not rebuild variables or constraints.

## Selecting a window

[`restrict()`][linopy.model.Model.restrict] follows [`xarray.Dataset.sel()`][xarray.Dataset.sel] semantics:
every container's labels are selected along the dimensions it holds, containers
without them stay whole. Labels, lists, slices and scalars work, and several
dimensions can be selected at once, e.g.
`m.restrict(snapshot=slice(t0, t1), bus=["north"])`. The parent model is
unchanged.

## Boundary values

A constraint in the window can reference a variable outside it.
[`restrict()`][linopy.model.Model.restrict] moves those terms to the right-hand side,
multiplied by their coefficient:

$$
A_{WW}\, x_W \sim b_W - A_{WB}\, x_B
$$

where $x_W$ are the window's variables, $x_B$ the variables
outside it, and $x_B$ on the right-hand side the values you pass in
`fix`. In words, each outside term adds `coefficient * fixed value` to the
right-hand side. Rows that lose every term are checked and dropped when they
are satisfied.

Without `fix`, a window that cuts a coupling raises
[`RestrictionError`][linopy.restrict.RestrictionError] naming the constraints. A window
with no outside terms needs no `fix`.

`fix` values are keyed by variable name, aligned with the variable by
coordinate label and broadcast over the dimensions they lack, so a
DataArray over the full model, a `pandas.Series` or a scalar works directly.
Every coordinate of the variable must be present, otherwise `restrict`
raises. Values for in-window variables are ignored. A `NaN` at an outside
label counts as missing.

A folded row counts as satisfied within a tolerance of `tol=1e-9`, relative to
the magnitude of its folded terms when that exceeds one. Pass a larger `tol`
when the fixed values carry solver noise.

## Dropping rows

`drop` leaves rows out of a window, keyed by constraint name, as boolean masks
aligned like `fix` where `True` drops the row; `True` alone drops the whole
constraint. Masks must be boolean; other dtypes, e.g. `NaN` or labels, raise.
Dropped rows are removed before folding, so they are never checked, and their
duals are `NaN`. In a rolling horizon with look-ahead, this keeps the
target of the next block from binding in the current window:

```python
lookahead = sns > block_end
window.advance(snapshot=window_sns, fix=fixed, drop={"soc_target": lookahead})
```

## Example: one region of a network

Two regions balance their demand with local generation and imports. The
exchange constraint couples the regions: what one imports, the other exports.

```python
import pandas as pd
import xarray as xr
import linopy

buses = pd.Index(["north", "south"], name="bus")
sns = pd.RangeIndex(24, name="snapshot")
demand = xr.DataArray([[30.0] * 24, [50.0] * 24], coords=[buses, sns])
cost = xr.DataArray([1.0, 2.0], coords=[buses])

m = linopy.Model()
gen = m.add_variables(0, 100, coords=[buses, sns], name="gen")
imports = m.add_variables(-20, 20, coords=[buses, sns], name="imports")
m.add_constraints(gen + imports == demand, name="balance")
m.add_constraints(imports.sum("bus") == 0, name="exchange")
m.add_objective((cost * gen).sum())

# solve the north with the south importing 15 in every snapshot
south_imports = xr.DataArray([0.0, 15.0], coords=[buses])
north = m.restrict(bus=["north"], fix={"imports": south_imports})
north.solve(solver_name="highs")
```

The `exchange` constraint has no `bus` dimension, so it stays whole. Its
term on the south's imports moves to the right-hand side, which forces the
north to export 15.

## Example: rolling horizon

A long-horizon problem is often solved one time slice at a time: solve a slice
(say, one day), carry the end state into the next slice, then move forward.

```python
import pandas as pd
import xarray as xr
import linopy

sns = pd.date_range("2026-01-01", periods=8760, freq="h", name="snapshot")
m = linopy.Model()
soc = m.add_variables(0, 100, coords=[sns], name="soc")
flow = m.add_variables(0, 1000, coords=[sns], name="flow")
# the balance references the previous snapshot
m.add_constraints(soc - 0.5 * soc.shift(snapshot=1) - flow <= 10, name="balance")
m.add_objective(flow.sum(), sense="min")

# initial state of charge over the full horizon
initial = xr.DataArray([50.0] * 8760, dims=["snapshot"])

window = m.restrict(snapshot=sns[:24], fix={"soc": initial})

for start in range(0, 8760, 24):
    if start:
        window.advance(snapshot=sns[start : start + 24], fix={"soc": initial})
    window.solve(solver_name="highs")
    if window.status != "ok":
        raise RuntimeError(
            f"window starting at {start} failed: {window.termination_condition}"
        )

    # containers are built when you read them
    solution = window.solution
```

In the first row of each window, the balance references the previous
snapshot's state of charge, which lies outside the window. That term moves to
the right-hand side with the value from `fix`.

!!! note

    Folding uses the constraint's *actual* coefficient on the outside term. Hand-written rolling-horizon code often uses a coefficient of 1 for the initial state of charge, instead of the loss coefficient used on the other rows. `restrict` reproduces a slightly different model.

## Working with a window

[`WindowModel`][linopy.restrict.WindowModel] hands the solver the window's matrices
directly:

- `matrices` holds `A`, `b`, `sense`,
    `c`, `lb`, `ub` and the labels of its rows and columns.
- [`solution`][linopy.restrict.WindowModel.solution] and
    [`dual`][linopy.restrict.WindowModel.dual] are dictionaries with one
    `DataArray` per variable or constraint, each with its own coordinates. They
    are built when you read them.
- `objective` is the last solve's value,
    `None` when the solve returned no solution.
- [`status`][linopy.restrict.WindowModel.status] and
    [`termination_condition`][linopy.restrict.WindowModel.termination_condition] report the outcome
    of the last solve, as on [`Model`][linopy.model.Model]. An infeasible window does
    not raise on `solve()`, but reading its values does, so check that
    `status` is `"ok"` in a loop. A status of `"ok"` does not imply an
    optimum: a MIP stopped by its time limit is `"ok"` with the termination
    condition `"time_limit"`. Its incumbent, if any, is readable.
- [`solve_fixed_lp()`][linopy.restrict.WindowModel.solve_fixed_lp] re-solves a
    MIP window as an LP with its discrete variables fixed at the last
    solution, so `dual` returns the duals of the fixed problem. A later
    `solve()` solves the MIP again.
- `log_fn` in `solve()` writes the solver log, e.g. one file per window.
- [`solver`][linopy.restrict.WindowModel.solver] and
    [`solver_model`][linopy.restrict.WindowModel.solver_model] expose the
    solver and its native model;
    [`close()`][linopy.restrict.WindowModel.close] disposes them, e.g. to
    release a license, and the next `solve()` creates a new one with the
    same solver and options.

Terms outside the window are dropped from the objective, so the window's value
differs from the full model's by a constant.

## Moving the window

[`advance()`][linopy.restrict.WindowModel.advance] slices the matrices for a new
selection along the same dimensions and moves the new boundary values to the
right-hand side. When the new window's matrix has the same sparsity pattern,
the solver updates its existing model instead of building a new one, so a
chain of equal-sized windows keeps that fast path. A changed pattern, e.g. a
coefficient that is zero in one window only, rebuilds the solver model, or
raises [`RebuildRequiredError`][linopy.persistent.errors.RebuildRequiredError] with
`disallow_rebuild=True`. `fix`, `drop` and `tol` default to the previous
window's; `drop={}` clears the masks.

Each solve starts from scratch: the basis left by the previous window is
discarded, because a window shifted in time usually makes it a poor start that
slows the solver down. Pass `warm_start=True` to `solve` to keep it, e.g.
when re-solving the same window with slightly changed fixed values. HiGHS and
Gurobi discard the basis in place; other solvers rebuild their model instead,
or raise [`RebuildRequiredError`][linopy.persistent.errors.RebuildRequiredError] with
`disallow_rebuild=True`.

## Limits

Every container holding a selected dimension must contain the selected labels;
otherwise `restrict` raises [`RestrictionError`][linopy.restrict.RestrictionError]. A
slice selects whatever labels each container holds within its bounds.

Indicators and quadratic objectives raise `NotImplementedError`.

Native SOS constraints also raise: they are not part of the matrices.
Reformulate them first with [`linopy.Model.apply_sos_reformulation()`][linopy.model.Model.apply_sos_reformulation]. A
cardinality row from the reformulation that spans the selected dimension
still cannot be windowed.

Piecewise formulations work; they expand into ordinary variables and constraints
when added.

A window is not a [`Model`][linopy.model.Model]: it cannot take new variables or
constraints.
