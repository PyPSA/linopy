# API Reference

Reference for linopy's public API. Most workflows start at
[`Model`][linopy.model.Model] — [`Variable`][linopy.variables.Variable],
[`Constraint`][linopy.constraints.Constraint], and
[`Objective`][linopy.objective.Objective] are all built through
[Model.add_variables][linopy.model.Model.add_variables],
[Model.add_constraints][linopy.model.Model.add_constraints],
[Model.add_objective][linopy.model.Model.add_objective],
and accessed through the matching
[Model.variables][linopy.model.Model.variables],
[Model.constraints][linopy.model.Model.constraints], and
[Model.objective][linopy.model.Model.objective] accessors.
The supporting classes below cover those types in detail.

<div class="grid cards" markdown>

-   [**Model**](model.md)

    Model

-   [**Variables**](variables.md)

    Variable, Variables

-   [**Expressions**](expressions.md)

    LinearExpression, QuadraticExpression, Expressions

-   [**Constraints**](constraints.md)

    Constraint, Constraints

-   [**Objective**](objective.md)

    Objective

-   [**Piecewise**](piecewise.md)

    Piecewise

-   [**Solvers**](solvers.md)

    Solvers, Solver status and result types, Remote solving

-   [**Utilities**](utilities.md)

    Utilities

-   [**Testing**](testing.md)

    Testing

-   [**Warnings**](warnings.md)

    Warnings

</div>
