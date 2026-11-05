# Solvers

## Solvers

The stateful [`Solver`][linopy.solvers.Solver] instance owns the solver-side
model and exposes a two-step [`from_name()`][linopy.solvers.Solver.from_name] /
[`solve()`][linopy.solvers.Solver.solve] workflow. [`linopy.Model.solve()`][linopy.model.Model.solve] is a
thin wrapper around it.

::: linopy.solvers.Solver
    options:
      heading_level: 3
      members: false

### Construction

::: linopy.solvers.Solver.from_name
    options:
      heading_level: 4

::: linopy.solvers.Solver.from_model
    options:
      heading_level: 4

::: linopy.solvers.Solver.from_file
    options:
      heading_level: 4

### Existing LP/MPS files

Use `Solver.from_file` to solve an existing file without constructing a linopy `Model`. The input is read during `solve()` and remains owned by the caller.

```python
from linopy.solvers import Solver

solver = Solver.from_file("highs", "problem.mps", options={"time_limit": 60})
try:
    result = solver.solve()
    print(result.status)
    print(result.solution.objective)
finally:
    solver.close()
```

Backend restrictions on file contents and names still apply. Without a linopy `Model`, label-indexed primal and dual arrays may be empty. Variable values can be accessed through the native solver model, when available, or through a backend-supported solution file. Read native model values before calling `close()`.

### Solving

::: linopy.solvers.Solver.solve
    options:
      heading_level: 4

::: linopy.solvers.Solver.update_solver_model
    options:
      heading_level: 4

::: linopy.solvers.Solver.close
    options:
      heading_level: 4

### Post-solve state

::: linopy.solvers.Solver.status
    options:
      heading_level: 4

::: linopy.solvers.Solver.solution
    options:
      heading_level: 4

::: linopy.solvers.Solver.report
    options:
      heading_level: 4

::: linopy.solvers.Solver.solver_model
    options:
      heading_level: 4

### Capabilities

::: linopy.solvers.Solver.is_available
    options:
      heading_level: 4

::: linopy.solvers.Solver.license_status
    options:
      heading_level: 4

::: linopy.solvers.Solver.supports
    options:
      heading_level: 4

::: linopy.solvers.Solver.supported_features
    options:
      heading_level: 4

::: linopy.solvers.Solver.runtime_features
    options:
      heading_level: 4

### Discovery

::: linopy.solvers.available_solvers
    options:
      heading_level: 4

::: linopy.solvers.licensed_solvers
    options:
      heading_level: 4

::: linopy.solvers.SolverFeature
    options:
      heading_level: 4

::: linopy.solvers.LicenseStatus
    options:
      heading_level: 4

### Implementations

::: linopy.solvers.CBC
    options:
      heading_level: 4
      members: false

::: linopy.solvers.COPT
    options:
      heading_level: 4
      members: false

::: linopy.solvers.Cplex
    options:
      heading_level: 4
      members: false

::: linopy.solvers.GLPK
    options:
      heading_level: 4
      members: false

::: linopy.solvers.Gurobi
    options:
      heading_level: 4
      members: false

::: linopy.solvers.Highs
    options:
      heading_level: 4
      members: false

::: linopy.solvers.Knitro
    options:
      heading_level: 4
      members: false

::: linopy.solvers.MindOpt
    options:
      heading_level: 4
      members: false

::: linopy.solvers.Mosek
    options:
      heading_level: 4
      members: false

::: linopy.solvers.SCIP
    options:
      heading_level: 4
      members: false

::: linopy.solvers.Xpress
    options:
      heading_level: 4
      members: false

::: linopy.solvers.cuOpt
    options:
      heading_level: 4

::: linopy.solvers.cuPDLPx
    options:
      heading_level: 4

## Solver status and result types

Types returned by or compared against [`linopy.Model.status`][linopy.model.Model.status],
[`linopy.Model.termination_condition`][linopy.model.Model.termination_condition], and [`linopy.Model.solution`][linopy.model.Model.solution], plus
[`SolverReport`][linopy.constants.SolverReport] surfaced on
[Solver.report][linopy.solvers.Solver.report] and
[Result.report][linopy.constants.Result].

::: linopy.constants.SolverStatus
    options:
      heading_level: 3

::: linopy.constants.TerminationCondition
    options:
      heading_level: 3

::: linopy.constants.Status
    options:
      heading_level: 3

::: linopy.constants.Solution
    options:
      heading_level: 3

::: linopy.constants.SolverReport
    options:
      heading_level: 3

::: linopy.constants.Result
    options:
      heading_level: 3

## Remote solving

::: linopy.remote.RemoteHandler
    options:
      heading_level: 3
