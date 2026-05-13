.. currentmodule:: linopy

#############
API reference
#############

Reference for linopy's public API. Top sections are task-oriented
(creating, inspecting, modifying, solving, IO); supporting classes
are grouped below. Each entry links to a dedicated page with the full
signature and docstring.

.. contents::
   :local:
   :depth: 2


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
   piecewise.breakpoints
   piecewise.segments
   piecewise.Slopes


Inspecting a model
==================

.. autosummary::
   :toctree: generated/

   model.Model.variables
   model.Model.constraints
   model.Model.objective
   model.Model.sense
   model.Model.type
   model.Model.is_linear
   model.Model.is_quadratic


Modifying a model
=================

.. autosummary::
   :toctree: generated/

   model.Model.remove_variables
   model.Model.remove_constraints
   model.Model.remove_objective
   model.Model.remove_sos_constraints
   model.Model.copy
   model.Model.reformulate_sos_constraints


Solving
=======

.. autosummary::
   :toctree: generated/

   model.Model.solve


Post-solve access
=================

.. autosummary::
   :toctree: generated/

   model.Model.solution
   model.Model.dual
   model.Model.status
   model.Model.termination_condition


Diagnostics
===========

.. autosummary::
   :toctree: generated/

   model.Model.compute_infeasibilities
   model.Model.format_infeasibilities


IO
==

.. autosummary::
   :toctree: generated/

   model.Model.to_file
   model.Model.to_netcdf
   model.Model.get_problem_file
   model.Model.get_solution_file
   io.read_netcdf


Top-level helpers
=================

.. autosummary::
   :toctree: generated/

   align
   options


Other classes and types
=======================

Variable
--------

``Variable`` is a subclass of ``xarray.DataArray`` and carries labels
for a multi-dimensional decision variable.

.. autosummary::
   :toctree: generated/

   variables.Variable

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   variables.Variable.lower
   variables.Variable.upper
   variables.Variable.type
   variables.Variable.solution

Modification
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   variables.Variable.fix
   variables.Variable.unfix
   variables.Variable.relax
   variables.Variable.unrelax

Operations
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   variables.Variable.sum
   variables.Variable.where
   variables.Variable.sanitize

Conversion
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   variables.Variable.to_linexpr
   variables.Variable.to_polars


Variables
---------

``Variables`` is a container for the collection of variables on a
model. Accessed via ``model.variables``.

.. autosummary::
   :toctree: generated/

   variables.Variables

Inventory by type
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   variables.Variables.continuous
   variables.Variables.binaries
   variables.Variables.integers
   variables.Variables.semi_continuous
   variables.Variables.sos

Aggregate access
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   variables.Variables.lower
   variables.Variables.upper
   variables.Variables.solution

Modification
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   variables.Variables.add
   variables.Variables.remove

Bulk modify
~~~~~~~~~~~

Container-wide analogues of :func:`Variable.fix`, etc.

.. autosummary::
   :toctree: generated/

   variables.Variables.fix
   variables.Variables.unfix
   variables.Variables.relax
   variables.Variables.unrelax


LinearExpression
----------------

Linear combination of variables. Arithmetic on ``Variable`` /
``LinearExpression`` returns a ``LinearExpression``.

.. autosummary::
   :toctree: generated/

   expressions.LinearExpression

Structure
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   expressions.LinearExpression.vars
   expressions.LinearExpression.coeffs
   expressions.LinearExpression.const
   expressions.LinearExpression.nterm

Construction
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   expressions.LinearExpression.from_tuples
   expressions.merge

Operations
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   expressions.LinearExpression.sum
   expressions.LinearExpression.where
   expressions.LinearExpression.groupby
   expressions.LinearExpression.rolling

Conversion
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   expressions.LinearExpression.to_constraint
   expressions.LinearExpression.to_quadexpr
   expressions.LinearExpression.to_polars

Post-solve access
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   expressions.LinearExpression.solution


QuadraticExpression
-------------------

Quadratic combination of variables, returned when squared
``Variable`` / ``LinearExpression`` arithmetic is performed.

.. autosummary::
   :toctree: generated/

   expressions.QuadraticExpression

Structure
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   expressions.QuadraticExpression.vars
   expressions.QuadraticExpression.coeffs
   expressions.QuadraticExpression.const
   expressions.QuadraticExpression.nterm

Conversion
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   expressions.QuadraticExpression.to_constraint
   expressions.QuadraticExpression.to_matrix
   expressions.QuadraticExpression.to_polars

Post-solve access
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   expressions.QuadraticExpression.solution


Constraint
----------

``Constraint`` is a subclass of ``xarray.DataArray`` and carries labels
for a multi-dimensional constraint.

.. autosummary::
   :toctree: generated/

   constraints.Constraint

Structure
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   constraints.Constraint.lhs
   constraints.Constraint.sign
   constraints.Constraint.rhs
   constraints.Constraint.coeffs
   constraints.Constraint.vars

Post-solve access
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   constraints.Constraint.dual

Conversion
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   constraints.Constraint.to_polars


CSRConstraint
-------------

Memory-efficient, immutable constraint representation backed by a scipy
CSR sparse matrix. Opt in via ``Model(freeze_constraints=True)`` or
``Model.add_constraints(..., freeze=True)``. See the
:doc:`creating-constraints` guide for usage.

.. autosummary::
   :toctree: generated/

   constraints.CSRConstraint

Structure
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   constraints.CSRConstraint.coeffs
   constraints.CSRConstraint.vars
   constraints.CSRConstraint.sign
   constraints.CSRConstraint.rhs
   constraints.CSRConstraint.ncons
   constraints.CSRConstraint.nterm

Post-solve access
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   constraints.CSRConstraint.dual

Conversion
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   constraints.CSRConstraint.to_polars


Constraints
-----------

Container for the collection of constraints on a model. Accessed via
``model.constraints``.

.. autosummary::
   :toctree: generated/

   constraints.Constraints

Inventory
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   constraints.Constraints.inequalities
   constraints.Constraints.equalities

Aggregate access
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   constraints.Constraints.coeffs
   constraints.Constraints.vars
   constraints.Constraints.sign
   constraints.Constraints.rhs
   constraints.Constraints.dual

Modification
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   constraints.Constraints.add
   constraints.Constraints.remove

Cleanup
~~~~~~~

.. autosummary::
   :toctree: generated/

   constraints.Constraints.sanitize_missings

Conversion
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   constraints.Constraints.to_matrix


Objective
---------

Wraps the objective expression on a model. Accessed via
``model.objective``.

.. autosummary::
   :toctree: generated/

   objective.Objective
   objective.Objective.expression
   objective.Objective.sense
   objective.Objective.value
   objective.Objective.is_linear
   objective.Objective.is_quadratic


Piecewise
---------

``PiecewiseFormulation`` is returned by
:func:`Model.add_piecewise_formulation` and exposes the resolved
formulation method together with the auxiliary variables/constraints
that were generated. :func:`tangent_lines` is a standalone helper for
composing chord-based bounds by hand, without going through
:func:`Model.add_piecewise_formulation`.

.. autosummary::
   :toctree: generated/

   piecewise.PiecewiseFormulation
   piecewise.PiecewiseFormulation.method
   piecewise.PiecewiseFormulation.convexity
   piecewise.PiecewiseFormulation.variables
   piecewise.PiecewiseFormulation.constraints
   piecewise.tangent_lines
   constants.PWL_METHOD
   constants.PWL_METHODS
   constants.PWL_CONVEXITY
   constants.PWL_CONVEXITIES


Solver interface
----------------

.. autosummary::
   :toctree: generated/

   solvers.available_solvers
   solvers.quadratic_solvers
   solvers.Solver


Solver implementations
----------------------

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
   solvers.SCIP
   solvers.Xpress
   solvers.cuPDLPx


Remote solving
--------------

.. autosummary::
   :toctree: generated/

   remote.RemoteHandler


Solver status and result types
------------------------------

Types returned by or compared against :attr:`Model.status`,
:attr:`Model.termination_condition`, and :attr:`Model.solution`.

.. autosummary::
   :toctree: generated/

   constants.SolverStatus
   constants.TerminationCondition
   constants.Status
   constants.Solution
   constants.Result


Warnings
--------

These warning classes can be silenced or filtered via
:func:`warnings.filterwarnings`.

.. autosummary::
   :toctree: generated/

   EvolvingAPIWarning
   PerformanceWarning
