# Model

Central container for an optimization problem. Most of linopy's
surface lives here.

::: linopy.model.Model
    options:
      heading_level: 2
      members: false

## Building a model

::: linopy.model.Model.add_variables
    options:
      heading_level: 3

::: linopy.model.Model.add_constraints
    options:
      heading_level: 3

::: linopy.model.Model.add_objective
    options:
      heading_level: 3

::: linopy.model.Model.add_expressions
    options:
      heading_level: 3

::: linopy.model.Model.add_sos_constraints
    options:
      heading_level: 3

::: linopy.model.Model.add_piecewise_formulation
    options:
      heading_level: 3

## Inspecting a model

The backing stores are internal; `Model.store_cls` and
`LinearExpression.is_sparse` are the public entry points.

::: linopy.model.Model.variables
    options:
      heading_level: 3

::: linopy.model.Model.constraints
    options:
      heading_level: 3

::: linopy.model.Model.expressions
    options:
      heading_level: 3

::: linopy.model.Model.objective
    options:
      heading_level: 3

::: linopy.model.Model.sense
    options:
      heading_level: 3

::: linopy.model.Model.type
    options:
      heading_level: 3

::: linopy.model.Model.is_linear
    options:
      heading_level: 3

::: linopy.model.Model.is_quadratic
    options:
      heading_level: 3

::: linopy.model.Model.sparse
    options:
      heading_level: 3

::: linopy.model.Model.store_cls
    options:
      heading_level: 3

## Modifying a model

::: linopy.model.Model.remove_variables
    options:
      heading_level: 3

::: linopy.model.Model.remove_constraints
    options:
      heading_level: 3

::: linopy.model.Model.remove_expressions
    options:
      heading_level: 3

::: linopy.model.Model.remove_objective
    options:
      heading_level: 3

::: linopy.model.Model.remove_sos_constraints
    options:
      heading_level: 3

::: linopy.model.Model.assign_coords
    options:
      heading_level: 3

::: linopy.model.Model.restrict
    options:
      heading_level: 3

::: linopy.model.Model.copy
    options:
      heading_level: 3

::: linopy.model.Model.apply_sos_reformulation
    options:
      heading_level: 3

::: linopy.model.Model.undo_sos_reformulation
    options:
      heading_level: 3

::: linopy.model.Model.reformulate_sos_constraints
    options:
      heading_level: 3

## Restricting a model

See [Restricting a Model](../user-guide/restrict.md) for a guide.

::: linopy.model.Model.restrict
    options:
      heading_level: 3

::: linopy.restrict.WindowModel
    options:
      heading_level: 3

## Solving

::: linopy.model.Model.solve
    options:
      heading_level: 3

## Post-solve access

::: linopy.model.Model.solution
    options:
      heading_level: 3

::: linopy.model.Model.dual
    options:
      heading_level: 3

::: linopy.model.Model.status
    options:
      heading_level: 3

::: linopy.model.Model.termination_condition
    options:
      heading_level: 3

## Diagnostics

::: linopy.model.Model.compute_infeasibilities
    options:
      heading_level: 3

::: linopy.model.Model.format_infeasibilities
    options:
      heading_level: 3

## IO

::: linopy.model.Model.to_file
    options:
      heading_level: 3

::: linopy.model.Model.to_netcdf
    options:
      heading_level: 3

::: linopy.io.read_netcdf
    options:
      heading_level: 3

## Building from specs

Build a model from a [mathspec](https://mathspec.readthedocs.io) YAML
program attached to data, and read its named expressions as functions of the
spec, the model and the data. Requires `pip install "linopy[spec]"` (Python \>= 3.12). See [Spec data and results](../user-guide/spec-data.md).

::: linopy.model.Model.from_spec
    options:
      heading_level: 3

::: linopy.spec.evaluate
    options:
      heading_level: 3

::: linopy.spec.SpecDataError
    options:
      heading_level: 3
