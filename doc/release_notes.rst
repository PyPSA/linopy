Release Notes
=============

Upcoming Release
----------------

* LinearExpression has a new function `densify_terms` which reduces the `_term` axis to a minimal length while containing all non-zero coefficient values.
* When summing over one or multiple axes in a LinearExpression, terms with coefficient of zeros can now be dropped automatically.
* The export of LP files was restructured and is flat arrays under the hook to ensure performant export of long constraints.
* Dimensions of masks passed to `add_variables` and `add_constraints` now have to be a subset of the resulting labels dimensions.
* A new high-level function `merge` was added to concatenate multiple linear expressions.
* The ``Variable.where`` function now has -1 as default fill value.
* The return value of most Variable functions built on xarray functions now preserve the Variable type.
* The variable labels in linear expression which are added to a model are ensured to be stored as integers.
* A preliminary function to print out the subset of infeasible constraints was added (only available for Gurobi, based on https://www.gurobi.com/documentation/9.5/refman/py_model_computeiis.html)
* Constraints with only missing variables labels are now sanitized are receive a label -1.
* Binary variables now also have a non-nan lower and upper value due compatibility.
* Models can now be created using the `gurobipy` API, this can lead to faster total solving times.
* `.solve` has a new argument `io_api`. If set to 'direct' the io solving will be performed using the python API's. Currently only available for gurobi.
* The `Variable` class now has a `lower` and `upper` accessor, which allows to inspect and modify the lower and upper bounds of a assigned variable.
* The `Constraint` class now has a `lhs`, `vars`, `coeffs`, `rhs` and `sign` accessor, which allows to inspect and modify the left-hand-side, the signs and right-hand-side of a assigned constraint.
* Constraints can now be build combining linear expressions with right-hand-side via a `>=`, `<=` or a `==` operator. This creates an `AnonymousConstraint` which can be passed to `Model.add_constraints`.
* Add support of the HiGHS open source solver https://www.maths.ed.ac.uk/hall/HiGHS/ (https://github.com/PyPSA/linopy/pull/8, https://github.com/PyPSA/linopy/pull/17).


**Breaking changes**

* The low level IO function ``linopy.io.str_array_to_file`` was renamed to ``linopy.io.array_to_file``, the function ``linopy.io.join_str_arrays`` was removed.
* The `keep_coords` flag in ``LinearExpression.sum`` and ``Variable.sum`` was dropped.
* The `run_` functions in `linopy.solvers` now have a new set of arguments and keyword argument, in order to make solving io more flexible.
* `ncons` and `nvars` now count only non-missing constraints and variables.

Version 0.0.4
-------------


**Package Design**

The definitions of variables, constraints and linearexpression were moved to dedicated modules: ``linopy.variables``, ``linopy.constraints`` and ``linopy.expressions``.


**Internal/Data handling**

Most of the following changes are dedicated to data handling within the `Model` class. Users which rely on the internal structure have to expect some breaking changes.

* The model class now stores variables and constraints in dedicated (newly added) classes, ``Variables`` and ``Constraints``. The ``Variables`` class contains the ``xarray`` datasets `labels`, `lower` and `upper`. The ``Constraints`` class contains the datasets `labels`, `coeffs`, `vars`, `sign` and `rhs`. The two new class facilitate data access and helper functions.
* The "_term" dimension in the ``LinearExpression`` class is now stored without coordinates.
* As soon as a linear expression is added to a model the "_term" dimension is rename to "{constraintname}_term" in order align the model better with the contained arrays and to avoid unnecessary nans.
* Missing values in the ``Model.variables.labels`` and ``Model.constraints.labels`` arrays are now indicated by -1. This circumvents changing the type from `int` to `float`.
* ``LinearExpression`` now allows empty data as input.
* The `test_model_creation` script was refactored.


**New Features**

* The ``Variable`` class now has a accessor to get lower and upper bounds, ``get_lower_bound()`` and ``get_upper_bound()``.
* A new ``Constraint`` class was added which enables a better visual representation of the constraints. The class also has getter function to get coefficients, variables, signs and rhs constants. The new return type of the ``Model.add_constraints`` function is ``Constraint``.
* ``add_variables`` and ``add_constraints`` now accept a new argument ``mask``. The mask, which should be an boolean array, defines whether a variable/constraint is active (True) or should be ignored (False).
* A set of experimental eval functions was added. Now one can assign variable and constraints using string expressions. For further information see `linopy.Model.vareval`, `linopy.Model.lineval` and `linopy.Model.coneval`.
* ``Model`` has a new argument `force_dim_names`. When set to true assigned variables, constraints and data must always have custom dimension names, otherwise a ValueError is raised. These helps to avoid unintended broadcasting over dimension. Especially the use of pandas DataFrames and Series may become safer.
* A new binaries accessor ``Model.binaries`` was added.

Version 0.0.3
-------------

* Support assignment of variables and constraints without explicit names.
* Add support for xarray version > 0.16
* Add a documentation

Version 0.0.2
-------------

* Set up first runnable prototype.
