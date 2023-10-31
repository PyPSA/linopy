.. currentmodule:: linopy

#############
API reference
#############

This page provides an auto-generated summary of linopy's API.



Creating a model
================

.. autosummary::
    :toctree: generated/

    model.Model
    model.Model.add_variables
    model.Model.add_constraints
    model.Model.add_objective
    model.Model.linexpr
    model.Model.remove_constraints


Classes under the hook
======================

Variable
--------

``Variable`` is a subclass of ``xarray.DataArray`` and contains all labels referring to a multi-dimensional variable.

.. autosummary::
    :toctree: generated/

    variables.Variable
    variables.Variable.lower
    variables.Variable.upper
    variables.Variable.sum
    variables.Variable.where
    variables.Variable.sanitize
    variables.Variables
    variables.ScalarVariable

Variables
---------

``Variables`` is a container for multiple N-D labeled variables. It is automatically added to a ``Model`` instance when initialized.

.. autosummary::
    :toctree: generated/

    variables.Variables
    variables.Variables.add
    variables.Variables.remove
    variables.Variables.continuous
    variables.Variables.integers
    variables.Variables.binaries
    variables.Variables.integers
    variables.Variables.flat


LinearExpressions
-----------------

.. autosummary::
    :toctree: generated/

    expressions.LinearExpression
    expressions.LinearExpression.sum
    expressions.LinearExpression.where
    expressions.LinearExpression.groupby
    expressions.LinearExpression.rolling
    expressions.LinearExpression.from_tuples
    expressions.merge
    expressions.ScalarLinearExpression

Constraint
----------

``Constraint`` is a subclass of ``xarray.DataArray`` and contains all labels referring to a multi-dimensional constraint.

.. autosummary::
    :toctree: generated/

    constraints.Constraint
    constraints.Constraint.coeffs
    constraints.Constraint.vars
    constraints.Constraint.lhs
    constraints.Constraint.sign
    constraints.Constraint.rhs
    constraints.Constraint.flat


Constraints
-----------

.. autosummary::
    :toctree: generated/

    constraints.Constraints
    constraints.Constraints.add
    constraints.Constraints.remove
    constraints.Constraints.coefficientrange
    constraints.Constraints.inequalities
    constraints.Constraints.equalities
    constraints.Constraints.sanitize_missings
    constraints.Constraints.flat
    constraints.Constraints.to_matrix


IO functions
============

.. autosummary::
    :toctree: generated/

    model.Model.get_problem_file
    model.Model.get_solution_file
    model.Model.to_file
    model.Model.to_netcdf
    io.read_netcdf

Solvers
========

.. autosummary::
    :toctree: generated/

    solvers.run_cbc
    solvers.run_glpk
    solvers.run_highs
    solvers.run_cplex
    solvers.run_gurobi
    solvers.run_xpress
    solvers.run_mosek
    solvers.run_mindopt
    solvers.run_copt

Solving
========

.. autosummary::
    :toctree: generated/

    model.Model.solve
    constants.SolverStatus
    constants.TerminationCondition
    constants.Status
    constants.Solution
    constants.Result
