# Expressions

## LinearExpression

Linear combination of variables. Arithmetic on `Variable` /
`LinearExpression` returns a `LinearExpression`.

::: linopy.expressions.LinearExpression
    options:
      heading_level: 3
      members: false

### Post-solve access

::: linopy.expressions.LinearExpression.solution
    options:
      heading_level: 4

### Operations

::: linopy.expressions.LinearExpression.sum
    options:
      heading_level: 4

::: linopy.expressions.LinearExpression.where
    options:
      heading_level: 4

::: linopy.expressions.LinearExpression.groupby
    options:
      heading_level: 4

::: linopy.expressions.LinearExpression.rolling
    options:
      heading_level: 4

::: linopy.expressions.LinearExpression.simplify
    options:
      heading_level: 4

::: linopy.expressions.LinearExpression.densify_terms
    options:
      heading_level: 4

### Structure

::: linopy.expressions.LinearExpression.name
    options:
      heading_level: 4

::: linopy.expressions.LinearExpression.vars
    options:
      heading_level: 4

::: linopy.expressions.LinearExpression.coeffs
    options:
      heading_level: 4

::: linopy.expressions.LinearExpression.const
    options:
      heading_level: 4

::: linopy.expressions.LinearExpression.nterm
    options:
      heading_level: 4

::: linopy.expressions.LinearExpression.has_terms
    options:
      heading_level: 4

::: linopy.expressions.LinearExpression.is_sparse
    options:
      heading_level: 4

### Conversion

::: linopy.expressions.LinearExpression.to_polars
    options:
      heading_level: 4

### Construction

::: linopy.expressions.LinearExpression.from_tuples
    options:
      heading_level: 4

::: linopy.expressions.LinearExpression.from_coo
    options:
      heading_level: 4

::: linopy.expressions.merge
    options:
      heading_level: 4

## QuadraticExpression

Quadratic combination of variables, returned when squared
`Variable` / `LinearExpression` arithmetic is performed.

::: linopy.expressions.QuadraticExpression
    options:
      heading_level: 3
      members: false

### Structure

::: linopy.expressions.QuadraticExpression.name
    options:
      heading_level: 4

::: linopy.expressions.QuadraticExpression.vars
    options:
      heading_level: 4

::: linopy.expressions.QuadraticExpression.coeffs
    options:
      heading_level: 4

::: linopy.expressions.QuadraticExpression.const
    options:
      heading_level: 4

::: linopy.expressions.QuadraticExpression.nterm
    options:
      heading_level: 4

::: linopy.expressions.QuadraticExpression.has_terms
    options:
      heading_level: 4

### Conversion

::: linopy.expressions.QuadraticExpression.to_matrix
    options:
      heading_level: 4

::: linopy.expressions.QuadraticExpression.to_polars
    options:
      heading_level: 4

### Post-solve access

::: linopy.expressions.QuadraticExpression.solution
    options:
      heading_level: 4

## Expressions

Container for the collection of named expressions on a model. Accessed via
`model.expressions`.

::: linopy.expressions.Expressions
    options:
      heading_level: 3
      members: false

### Modification

::: linopy.expressions.Expressions.add
    options:
      heading_level: 4

::: linopy.expressions.Expressions.remove
    options:
      heading_level: 4

### Post-solve access

::: linopy.expressions.Expressions.solution
    options:
      heading_level: 4
