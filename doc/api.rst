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
    model.Model.add_sos_constraints
    model.Model.add_piecewise_formulation
    piecewise.PiecewiseFormulation
    piecewise.Slopes
    piecewise.breakpoints
    piecewise.segments
    piecewise.tangent_lines
    model.Model.linexpr
    model.Model.remove_constraints
    model.Model.reformulate_sos_constraints
    model.Model.compute_infeasibilities
    model.Model.format_infeasibilities
    model.Model.copy


Top-level helpers
=================

.. autosummary::
    :toctree: generated/

    align
    merge
    options
    EvolvingAPIWarning
    PerformanceWarning


Classes under the hood
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
    variables.Variable.to_linexpr
    variables.Variable.fix
    variables.Variable.unfix
    variables.Variable.relax
    variables.Variable.unrelax
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


QuadraticExpressions
--------------------

.. autosummary::
    :toctree: generated/

    expressions.QuadraticExpression


Objective
---------

.. autosummary::
    :toctree: generated/

    objective.Objective

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
    constraints.Constraint.freeze
    constraints.Constraint.mutable


CSRConstraint
-------------

``CSRConstraint`` is a memory-efficient, immutable constraint representation backed by a scipy CSR sparse matrix. See the :doc:`creating-constraints` guide for usage.

.. autosummary::
    :toctree: generated/

    constraints.CSRConstraint
    constraints.CSRConstraint.coeffs
    constraints.CSRConstraint.vars
    constraints.CSRConstraint.sign
    constraints.CSRConstraint.rhs
    constraints.CSRConstraint.ncons
    constraints.CSRConstraint.nterm
    constraints.CSRConstraint.freeze
    constraints.CSRConstraint.mutable


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

Solver utilities
=================

.. autosummary::
    :toctree: generated/

    solvers.available_solvers
    solvers.quadratic_solvers
    solvers.Solver


Solvers
=======

.. autosummary::
    :toctree: generated/

    solvers.CBC
    solvers.COPT
    solvers.Cplex
    solvers.GLPK
    solvers.Gurobi
    solvers.Highs
    solvers.Knitro
    solvers.MindOpt
    solvers.Mosek
    solvers.PIPS
    solvers.SCIP
    solvers.Xpress
    solvers.cuPDLPx


Remote solving
==============

.. autosummary::
    :toctree: generated/

    remote.RemoteHandler


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
