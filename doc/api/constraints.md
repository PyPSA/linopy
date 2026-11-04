# Constraints

## Constraint

A multi-dimensional constraint `lhs sign rhs` with labels. Its lhs holds a
dense store, or in a sparse model (`Model(sparse=True)`) a memory-efficient
CSR store; `to_sparse` and `to_dense` convert it. See the
[Creating Constraints](../user-guide/creating-constraints.ipynb) and [Dense and sparse models](../user-guide/sparse-models.ipynb) guides for usage.

::: linopy.constraints.Constraint
    options:
      heading_level: 3
      members: false

### Structure

::: linopy.constraints.Constraint.lhs
    options:
      heading_level: 4

::: linopy.constraints.Constraint.sign
    options:
      heading_level: 4

::: linopy.constraints.Constraint.rhs
    options:
      heading_level: 4

::: linopy.constraints.Constraint.scaling
    options:
      heading_level: 4

::: linopy.constraints.Constraint.coeffs
    options:
      heading_level: 4

::: linopy.constraints.Constraint.vars
    options:
      heading_level: 4

::: linopy.constraints.Constraint.ncons
    options:
      heading_level: 4

::: linopy.constraints.Constraint.nterm
    options:
      heading_level: 4

::: linopy.constraints.Constraint.labels
    options:
      heading_level: 4

### Modification

`Constraint.update` is the canonical mutation API.

::: linopy.constraints.Constraint.update
    options:
      heading_level: 4

::: linopy.constraints.Constraint.soften
    options:
      heading_level: 4

::: linopy.constraints.Constraint.sanitize_zeros
    options:
      heading_level: 4

::: linopy.constraints.Constraint.sanitize_infinities
    options:
      heading_level: 4

::: linopy.constraints.Constraint.sanitize_missings
    options:
      heading_level: 4

### Post-solve access

::: linopy.constraints.Constraint.dual
    options:
      heading_level: 4

### Conversion

::: linopy.constraints.Constraint.to_polars
    options:
      heading_level: 4

::: linopy.constraints.Constraint.to_sparse
    options:
      heading_level: 4

::: linopy.constraints.Constraint.to_dense
    options:
      heading_level: 4

::: linopy.constraints.Constraint.to_matrix
    options:
      heading_level: 4

### Construction

::: linopy.constraints.Constraint.from_dataset
    options:
      heading_level: 4

## Constraints

Container for the collection of constraints on a model. Accessed via
`model.constraints`.

::: linopy.constraints.Constraints
    options:
      heading_level: 3
      members: false

### Inventory

::: linopy.constraints.Constraints.inequalities
    options:
      heading_level: 4

::: linopy.constraints.Constraints.equalities
    options:
      heading_level: 4

### Aggregate access

::: linopy.constraints.Constraints.coeffs
    options:
      heading_level: 4

::: linopy.constraints.Constraints.vars
    options:
      heading_level: 4

::: linopy.constraints.Constraints.sign
    options:
      heading_level: 4

::: linopy.constraints.Constraints.rhs
    options:
      heading_level: 4

::: linopy.constraints.Constraints.scaling
    options:
      heading_level: 4

::: linopy.constraints.Constraints.dual
    options:
      heading_level: 4

### Conversion

::: linopy.constraints.Constraints.to_matrix
    options:
      heading_level: 4
