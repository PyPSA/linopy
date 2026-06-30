.. currentmodule:: linopy

#############
API reference
#############

Reference for linopy's public API. Most workflows start at
:class:`~linopy.model.Model` — :class:`~linopy.variables.Variable`,
:class:`~linopy.constraints.Constraint`, and
:class:`~linopy.objective.Objective` are all built through
:meth:`Model.add_variables <linopy.model.Model.add_variables>`,
:meth:`Model.add_constraints <linopy.model.Model.add_constraints>`,
:meth:`Model.add_objective <linopy.model.Model.add_objective>`,
and accessed through the matching
:attr:`Model.variables <linopy.model.Model.variables>`,
:attr:`Model.constraints <linopy.model.Model.constraints>`, and
:attr:`Model.objective <linopy.model.Model.objective>` accessors.
The supporting classes below cover those types in detail.

.. contents::
   :local:
   :depth: 2


Model
=====

Central container for an optimization problem. Most of linopy's
surface lives here.

.. autosummary::
   :toctree: generated/

   model.Model

Building a model
----------------

.. autosummary::
   :toctree: generated/

   model.Model.add_variables
   model.Model.add_constraints
   model.Model.add_objective
   model.Model.add_sos_constraints
   model.Model.add_piecewise_formulation

Inspecting a model
------------------

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
-----------------

.. autosummary::
   :toctree: generated/

   model.Model.remove_variables
   model.Model.remove_constraints
   model.Model.remove_objective
   model.Model.remove_sos_constraints
   model.Model.copy
   model.Model.reformulate_sos_constraints

Solving
-------

.. autosummary::
   :toctree: generated/

   model.Model.solve

Post-solve access
-----------------

.. autosummary::
   :toctree: generated/

   model.Model.solution
   model.Model.dual
   model.Model.status
   model.Model.termination_condition

Diagnostics
-----------

.. autosummary::
   :toctree: generated/

   model.Model.compute_infeasibilities
   model.Model.format_infeasibilities

IO
--

.. autosummary::
   :toctree: generated/

   model.Model.to_file
   model.Model.to_netcdf
   io.read_netcdf


Variable
========

Subclass of ``xarray.DataArray`` carrying labels for a multi-dimensional
decision variable.

.. autosummary::
   :toctree: generated/

   variables.Variable

Attributes
----------

.. autosummary::
   :toctree: generated/

   variables.Variable.lower
   variables.Variable.upper
   variables.Variable.type
   variables.Variable.solution

Modification
------------

``Variable.update`` is the canonical mutation API. The legacy ``lower`` /
``upper`` setters still forward to ``update`` but emit a
``DeprecationWarning`` and will be removed in a future release.

.. autosummary::
   :toctree: generated/

   variables.Variable.update
   variables.Variable.fix
   variables.Variable.unfix
   variables.Variable.relax
   variables.Variable.unrelax

Operations
----------

.. autosummary::
   :toctree: generated/

   variables.Variable.sum
   variables.Variable.where

Conversion
----------

.. autosummary::
   :toctree: generated/

   variables.Variable.to_linexpr
   variables.Variable.to_polars


Variables
=========

Container for the collection of variables on a model. Accessed via
``model.variables``.

.. autosummary::
   :toctree: generated/

   variables.Variables

Attributes
----------

.. autosummary::
   :toctree: generated/

   variables.Variables.lower
   variables.Variables.upper
   variables.Variables.solution

Modification
------------

.. autosummary::
   :toctree: generated/

   variables.Variables.fix
   variables.Variables.unfix
   variables.Variables.relax
   variables.Variables.unrelax

Inventory
---------

.. autosummary::
   :toctree: generated/

   variables.Variables.continuous
   variables.Variables.binaries
   variables.Variables.integers
   variables.Variables.semi_continuous
   variables.Variables.sos


LinearExpression
================

Linear combination of variables. Arithmetic on ``Variable`` /
``LinearExpression`` returns a ``LinearExpression``.

.. autosummary::
   :toctree: generated/

   expressions.LinearExpression

Post-solve access
-----------------

.. autosummary::
   :toctree: generated/

   expressions.LinearExpression.solution

Operations
----------

.. autosummary::
   :toctree: generated/

   expressions.LinearExpression.sum
   expressions.LinearExpression.where
   expressions.LinearExpression.groupby
   expressions.LinearExpression.rolling

Structure
---------

.. autosummary::
   :toctree: generated/

   expressions.LinearExpression.vars
   expressions.LinearExpression.coeffs
   expressions.LinearExpression.const
   expressions.LinearExpression.nterm
   expressions.LinearExpression.has_terms

Conversion
----------

.. autosummary::
   :toctree: generated/

   expressions.LinearExpression.to_polars

Construction
------------

.. autosummary::
   :toctree: generated/

   expressions.LinearExpression.from_tuples
   expressions.merge


QuadraticExpression
===================

Quadratic combination of variables, returned when squared
``Variable`` / ``LinearExpression`` arithmetic is performed.

.. autosummary::
   :toctree: generated/

   expressions.QuadraticExpression

Structure
---------

.. autosummary::
   :toctree: generated/

   expressions.QuadraticExpression.vars
   expressions.QuadraticExpression.coeffs
   expressions.QuadraticExpression.const
   expressions.QuadraticExpression.nterm
   expressions.QuadraticExpression.has_terms

Conversion
----------

.. autosummary::
   :toctree: generated/

   expressions.QuadraticExpression.to_matrix
   expressions.QuadraticExpression.to_polars

Post-solve access
-----------------

.. autosummary::
   :toctree: generated/

   expressions.QuadraticExpression.solution


Constraint
==========

Subclass of ``xarray.DataArray`` carrying labels for a multi-dimensional
constraint.

.. autosummary::
   :toctree: generated/

   constraints.Constraint

Structure
---------

.. autosummary::
   :toctree: generated/

   constraints.Constraint.lhs
   constraints.Constraint.sign
   constraints.Constraint.rhs
   constraints.Constraint.coeffs
   constraints.Constraint.vars

Modification
------------

``Constraint.update`` is the canonical mutation API. The legacy ``lhs`` /
``sign`` / ``rhs`` / ``coeffs`` / ``vars`` setters still forward to
``update`` but emit a ``DeprecationWarning`` and will be removed in a
future release.

.. autosummary::
   :toctree: generated/

   constraints.Constraint.update

Post-solve access
-----------------

.. autosummary::
   :toctree: generated/

   constraints.Constraint.dual

Conversion
----------

.. autosummary::
   :toctree: generated/

   constraints.Constraint.to_polars


CSRConstraint
=============

Memory-efficient, immutable constraint representation backed by a scipy
CSR sparse matrix. Opt in via ``Model(freeze_constraints=True)`` or
``Model.add_constraints(..., freeze=True)``. See the
:doc:`creating-constraints` guide for usage.

.. autosummary::
   :toctree: generated/

   constraints.CSRConstraint

Structure
---------

.. autosummary::
   :toctree: generated/

   constraints.CSRConstraint.coeffs
   constraints.CSRConstraint.vars
   constraints.CSRConstraint.sign
   constraints.CSRConstraint.rhs
   constraints.CSRConstraint.ncons
   constraints.CSRConstraint.nterm

Post-solve access
-----------------

.. autosummary::
   :toctree: generated/

   constraints.CSRConstraint.dual

Conversion
----------

.. autosummary::
   :toctree: generated/

   constraints.CSRConstraint.to_polars


Constraints
===========

Container for the collection of constraints on a model. Accessed via
``model.constraints``.

.. autosummary::
   :toctree: generated/

   constraints.Constraints

Inventory
---------

.. autosummary::
   :toctree: generated/

   constraints.Constraints.inequalities
   constraints.Constraints.equalities

Aggregate access
----------------

.. autosummary::
   :toctree: generated/

   constraints.Constraints.coeffs
   constraints.Constraints.vars
   constraints.Constraints.sign
   constraints.Constraints.rhs
   constraints.Constraints.dual

Conversion
----------

.. autosummary::
   :toctree: generated/

   constraints.Constraints.to_matrix


Objective
=========

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
=========

Construction helpers
--------------------

.. autosummary::
   :toctree: generated/

   piecewise.breakpoints
   piecewise.segments
   piecewise.Slopes

PiecewiseFormulation
--------------------

Returned by :func:`Model.add_piecewise_formulation`.

.. autosummary::
   :toctree: generated/

   piecewise.PiecewiseFormulation
   piecewise.PiecewiseFormulation.method
   piecewise.PiecewiseFormulation.convexity
   piecewise.PiecewiseFormulation.variables
   piecewise.PiecewiseFormulation.constraints

Low-level helper
----------------

.. autosummary::
   :toctree: generated/

   piecewise.tangent_lines

Type aliases
------------

.. autosummary::
   :toctree: generated/

   constants.PWL_METHOD
   constants.PWL_METHODS
   constants.PWL_CONVEXITY
   constants.PWL_CONVEXITIES


Solvers
========

.. autosummary::
   :toctree: generated/

   solvers.available_solvers
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
==============

.. autosummary::
   :toctree: generated/

   remote.RemoteHandler


Solver status and result types
==============================

Types returned by or compared against :attr:`Model.status`,
:attr:`Model.termination_condition`, and :attr:`Model.solution`.

.. autosummary::
   :toctree: generated/

   constants.SolverStatus
   constants.TerminationCondition
   constants.Status
   constants.Solution
   constants.Result


Utilities
=========

.. autosummary::
   :toctree: generated/

   align
   options


Warnings
========

These warning classes can be silenced or filtered via
:func:`warnings.filterwarnings`.

.. autosummary::
   :toctree: generated/

   EvolvingAPIWarning
   PerformanceWarning
