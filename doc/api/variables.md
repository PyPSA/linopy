# Variables

## Variable

Subclass of `xarray.DataArray` carrying labels for a multi-dimensional
decision variable.

::: linopy.variables.Variable
    options:
      heading_level: 3
      members: false

### Attributes

::: linopy.variables.Variable.lower
    options:
      heading_level: 4

::: linopy.variables.Variable.upper
    options:
      heading_level: 4

::: linopy.variables.Variable.scaling
    options:
      heading_level: 4

::: linopy.variables.Variable.type
    options:
      heading_level: 4

::: linopy.variables.Variable.solution
    options:
      heading_level: 4

### Modification

`Variable.update` is the canonical mutation API. The legacy `lower` /
`upper` setters still forward to `update` but emit a
`DeprecationWarning` and will be removed in a future release.

::: linopy.variables.Variable.update
    options:
      heading_level: 4

::: linopy.variables.Variable.fix
    options:
      heading_level: 4

::: linopy.variables.Variable.unfix
    options:
      heading_level: 4

::: linopy.variables.Variable.relax
    options:
      heading_level: 4

::: linopy.variables.Variable.unrelax
    options:
      heading_level: 4

### Operations

::: linopy.variables.Variable.sum
    options:
      heading_level: 4

::: linopy.variables.Variable.where
    options:
      heading_level: 4

### Conversion

::: linopy.variables.Variable.to_linexpr
    options:
      heading_level: 4

::: linopy.variables.Variable.to_polars
    options:
      heading_level: 4

## Variables

Container for the collection of variables on a model. Accessed via
`model.variables`.

::: linopy.variables.Variables
    options:
      heading_level: 3
      members: false

### Attributes

::: linopy.variables.Variables.lower
    options:
      heading_level: 4

::: linopy.variables.Variables.upper
    options:
      heading_level: 4

::: linopy.variables.Variables.scaling
    options:
      heading_level: 4

::: linopy.variables.Variables.solution
    options:
      heading_level: 4

### Modification

::: linopy.variables.Variables.fix
    options:
      heading_level: 4

::: linopy.variables.Variables.unfix
    options:
      heading_level: 4

::: linopy.variables.Variables.relax
    options:
      heading_level: 4

::: linopy.variables.Variables.unrelax
    options:
      heading_level: 4

### Inventory

::: linopy.variables.Variables.continuous
    options:
      heading_level: 4

::: linopy.variables.Variables.binaries
    options:
      heading_level: 4

::: linopy.variables.Variables.integers
    options:
      heading_level: 4

::: linopy.variables.Variables.semi_continuous
    options:
      heading_level: 4

::: linopy.variables.Variables.sos
    options:
      heading_level: 4
