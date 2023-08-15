Release Notes
=============

.. Upcoming Release
.. ----------------


Version 0.2.1
-------------

**Bugfixes**

* When multiplying a `LinearExpression` with a constant value, the constant in the `LinearExpression` was not updated. This is fixed now.

**New Features**

* The `Variable` and the `LinearExpression` have a new method `cumsum`, which allows to compute the cumulative sum.

Version 0.2.1
-------------


* The documentation was revised and extended.
* A new function `print_labels` was added to the `Variables` and `Constraints` class. This function allows to print the variables/constraints from a list of labels.
* A new function `compute_infeasibilities` and `print_infeasibilities` was added to the `Model` class. This function allows to compute the infeasibilities of an infeasible model and print them out. The function only supports the `gurobi` solver so far.



Version 0.2.1
-------------


* Backwards compatibility for python 3.8.
* `Variable`, `LinearExpression` and `Constraint` now have a print function to easily print the objects with larger layouts, i.e. showing more terms and lines.


Version 0.2.0
-------------


**New Features**

* Linopy now supports quadratic programming. Therefore a new class `QuadraticExpression` was created, which can be assigned to the objective function. The `QuadraticExpression` class supports the same arithmetic operations as the `LinearExpression` and can be created by multiplying two `Variable` or `LinearExpression` objects. Note for the latter, the number of stacked terms must be equal to one (`expr.nterm == 1`).
* `LinearExpression`'s now support constant values. This allows defining linear expressions with numeric constant values, like `x + 5`.
* When defining constraints, it is not needed to separate variables from constants anymore. Thus, expressions  like `x <= y` or `5 * x + 10 >= y` are supported.
* The new default solver will now be the first element in `available_solvers`.
* The classes `Variable`, `LinearExpression` and `Constraint` now have a `loc` method.
* The classes `Variable`, `LinearExpression`, `Constraint`, `Variables` and `Constraints` now have a `flat` method, which returns a flattened `pandas.DataFrame` of the object in long-table format.
* It is now possible to access variables and constraints by a dot notation. For example, `model.variables.x` returns the variable `x` of the model.
* Variable assignment without explicit coordinates is now supported. In an internal step, integer coordinates are assigned to the dimensions without explicit coordinates.
* The `groupby` function now supports passing a `pandas.Dataframe` as `groupby` keys. These allows to group by multiple variables at once.
* The performance of the `groupby` function was strongly increased. In large operations a speedup of 10x was observed.
* New test functions `assert_varequal`, `assert_conequal` were added to the `testing` module.


**Deprecations**

* The class `AnonymousConstraint` is now deprecated in the favor of `Constraint`. The latter can now be assigned to a model or not.
* The `ravel` and `iter_ravel` method of the `Variables` and `Constraints` class is now deprecated in favor of the `flat` method.


**Breaking Changes**

* The `data` attribute of Variables and Constraints now returns a `xarray.Dataset` object instead of a `xarray.DataArray` object with the labels only.
* The deprecated `groupby_sum` function was removed in favor of the `groupby` method.
* The deprecated `rolling_sum` function was removed in favor of the `rolling` method.
* The deprecated `eval` module was removed in favor of the arithmetic operations on the classes `Variable`, `LinearExpression` and `Constraint`.
* The deprecated attribute `values` of the classes `Variable`, `LinearExpression` and `Constraint` was removed in favor of the `data` attribute.
* The deprecated `to_array` method of the classes `Variable` and `Constraint` was removed in favor of the `data` attribute.
* The deprecated `to_dataset` of the `LinearExpression` class was removed in favor of the `data` attribute.
* The function `get_lower_bound`, `get_upper_bound`, `get_variable_labels`, `get_variable_types`, `get_objective_coefficient`, `get_constraint_labels`, `get_constraint_sense`, `get_constraint_rhs`, `get_constraint_matrix` were removed in favor of the `matrices` accessor, i.e. `ub`, `lb`, `vlabels`, etc.
* The `LinearExpressionGroupby` class now takes a different set of arguments when initializing. These are `data: xr.Dataset`, `group: xr.DataArray`, `model: Any`, `kwargs: Mapping[str, Any]`.
* When grouping with a `xr.DataArray` / `pd.Series` / `pd.DataFrame` and summing afterwards, the keyword arguments like `squeeze`, `restore_coords` are ignored.


**Internal Changes**

* The internal data fields in `Variable` and `Constraint` are now always broadcasted to have aligned indexes. This allows for a more consistent handling of the objects.
* The inner structure of the `Variable`, `Variables`, `Constraint` and `Constraints` class has changed to a more stable design. All information of the `Variable` and the `Constraint` class is now stored in the `data` field. The `data` field is a `xarray.Dataset` object. The `Variables` and `Constraints` class "simple" containers for the `Variable` and `Constraint` objects, stored in dictionary under the `data` field. This design allows for a more flexible handling of individual variables and constraints.

**Other**

* License changed to MIT license.



Version 0.1.5
-------------


* Add `sel` functions to `Constraint` and `AnonymousConstraint` to allow for selection and inspection of constraints by coordinate.
* The printout of `Variables` and `Constraints` was refactored to a more concise layout.
* The solving termination condition "other" is now tagged as solving status "warning".

Version 0.1.4
-------------

* Fix representation of empty variables and linear expressions.
* The benchmark reported in [here](https://github.com/PyPSA/linopy/tree/master/benchmark) was updated to the latest version of linopy and adjusted to be fully reproducible.


Version 0.1.3
-------------

* **Hotfix** dual value retrieval for ``highs``.
* The MPS file writing was fixed for ``glpk`` solver. The MPS file writing is now tested against all solvers.


Version 0.1.2
-------------


* Fix display for constraint with single entry and no coordinates.


Version 0.1.1
-------------


* Printing out long LinearExpression is now accelerated in the `__repr__` function.
* Multiplication of LinearExpression's with pandas object was stabilized.
* A options handler was introduced that allows the user to change the maximum of printed lines and terms in the display of Variable's, LinearExpression's and Constraint's.
* If LinearExpression of exactly the same shape are joined together (in arithmetic operations), the coordinates of the first object is used to override the coordinates of the consecutive objects.


Version 0.1
-----------

This is the first major-minor release of linopy!  With this release, the package should more stable and consistent. The main changes are:

* The classes Variable, LinearExpression and Constraint now have a `__repr__` method. This allows for a better representation of the classes in the console.
* Linopy now defines and uses a fixed set of solver status and termination codes. This allows for a more consistent and reliable handling of solver results. The new codes are defined in the `linopy.constants` module. The implementation is inspired by https://github.com/0b11001111 and the implementation in this `PyPSA fork <https://github.com/0b11001111/PyPSA/blob/innoptem-lopf/pypsa/linear_program/solver.py>`_
* The automated summation of repeated variables in one constraint is now supported. Before the implementation for constraints like `x + x + x <= 5` was only working for solvers with a corresponding fallback computation. This is now fixed.
* Integer variables are now fully supported.
* Support exporting problems to MPS file via fast highspy MPS-writer (highspy installation required).
* The internal data structure of linopy classes were updated to a safer design. Instead of being defined as inherited xarray classes, the class `Variable`, `LinearExpression` and `Constraint` are now no inherited classes but contain the xarray objects in the `data` field. This allows the package to have more flexible function design and a reduced set of wrapped functions that are sensible to use in the optimization context.
* The class `Variable` and `LinearExpression` have new functions `groupby` and `rolling` imitating the corresponding xarray functions but with safe type inheritance and application of appended operations.
* Coefficients very close to zero (`< 1e-10`) are now automatically set to zero to avoid numerical issues with solvers.
* Coefficients of variables are no also allowed to be `np.nan`. These coefficients are ignored in the LP file writing.
* The classes Variable, LinearExpression, Constraint, ScalarVariable, ScalarLinearExpression and ScalarConstraint now require the model in the initialization (mostly internal code is affected).
* The `eval` module was removed in favor of arithmetic operations on the classes `Variable`, `LinearExpression` and `Constraint`.
* Solver options are now printed out in the console when solving a model.
* If a variable with indexes differing from the model internal indexes are assigned, linopy will raise a warning and align the variable to the model indexes.

Version 0.0.15
--------------

* Using the python `sum()` function over a `ScalarVariable` or a `ScalarLinearExpression` is now supported.
* Returning None type in `from_rule` assignment is now supported.
* Python 3.11 is now supported
* Xarray versions higher and lower `v2022.06.` are now supported.

Version 0.0.14
--------------

**New Features**

* Linopy now uses `highspy <https://pypi.org/project/highspy/>` as an interface to the HiGHS solver. This enables a direct and fast communication without needing to write an intermediate LP file.


Version 0.0.13
--------------

**New Features**

* The function `LinearExpression.from_tuples` now allows `ScalarVariable` as input.
* For compatibility reasons, the function `groupby_sum` now allows `pandas.Series` as input.

**Bug Fixes**

* Filtering out zeros is now an optional feature in the `solve` function. The previous behavior of filtering just before the LP file writing, lead to unexpected errors for constraints with only zero terms.


Version 0.0.12
--------------

**New Features**

* A new module was created to export basic mathematical quantities such as `lb`, `ub`, `c` vectors and the `A` matrix. Use it with the `matrices` accessor in `linopy.Model`.
* For `Constraints`` and `Variables`` a `ipython` autocompletion function for getting items was added.
* Inplace updates for constraints are now more flexible.
* AnonymousConstraint can now built from comparison operations of variables with constants, e.g. `x >= 5`.
* The `Model.add_constraints` function now support input of type `ScalarVariable`, `ScalarLinearExpression` and `ScalarConstraint`.
* Terms with zero coefficient are now filtered out before writing to file to avoid unnecessary overhead.
* The function `sanitize_zeros` was added to `Constraints`. Use this to filter out zero coefficient terms.

**Bug Fixes**

* Solving with `gurobi` and `io_api="direct"` lead to wrong objective assignment if the objective contained non-unique variables. This is fixed in this version.

Version 0.0.11
--------------

* Constraints and expressions can now be created using function that iterates over all combinations of given coordinates. This functionality mirrors the behavior of the Pyomo package. For complicated constraints which are hard to create with arrays of variables, it is easier (thus less efficient) to use an iterating function. For more information see the example notebook in the documentation.
* When getting the value of a variable, the value of the variable is returned as a `ScalarVariable`. This is useful for the above mentioned creation of expressions and constraints with iterating functions. This affect only the direct getter function, all other functions like `.sel` or `.isel` behave as known from Xarray.
* The docstring examples are now part of the Continuous Integration.
* Due to problems with indexing in the latest package version, the xarray dependency was set to `<=v2022.3.0`.

Version 0.0.10
--------------

* Improved type security when applying xarray functions on variables linear expressions and constraints.
* Correct block assignment for upcoming PIPS-IPM++ implementation.
* The function ``group_terms`` was renamed to ``groupby_sum``.
* A new function ``rolling_sum`` was introduced to compute rolling sums for variables and linear expressions.

Version 0.0.9
-------------

**New Features**

* Numpy ``__array_ufunc__`` was disabled in the `Variable`, `Constraint` and `LinearExpression` class in order to ensure persistence as the class when multiplying with `numpy` objects. As for pandas objects the issue https://github.com/pandas-dev/pandas/issues/45803 must be solved.
* The `Variable` class got a new accessor `sol` which points to the optimal values if the underlying model was optimized.
* The `Constraint` class got a new accessor `dual` which points to the dual values if tune underlying model was optimized and dual values are existent.
* When writing out the LP file, the handling of `nan` values is now checked in a more rigorous way. Before `linopy` was skipping and therefore ignoring constraints where the `rhs` was a `nan` value. As this behavior is not very save, such cases will raise an error now.
* Models can now be solved on a remote machine using a ssh tunnel. The implementation automatically stores the locally initialized model to a netcdf file on the server, runs the optimization and retrieves the results. See the example `Solve a model on a remote machine` in the documentation for further information.

**Bug Fixes**

* `linopy` is now continuously tested and working for Windows machines.

Version 0.0.8
-------------

**New Features**

* Writing out the LP was further sped up.
* The LP file writing for problems with "-0.0" coefficients was fixed.

**Breaking changes**

* the function ``as_str`` was replaced by ``int_to_str`` and ``float_to_str``.

Version 0.0.7
-------------

**New Features**

* Add ``get_name_by_label`` function to ``Variables`` and ``Constraints`` class. It retrieves the name of the variable/constraint containing the passed integer label. This is helpful for debugging.

**Bug Fixes**

* The `lhs` accessor for the ``Constraint`` class was fixed. This raised an error before as the `_term` dimension was not adjusted adequately.
* Variables and constraints which are fully masked are now skipped in the lp-file writing. This lead to a error before.

Version 0.0.6
-------------

* Hot fix: Assign ``linopy.__version__`` attribute
* Hot fix: Fix sign assignment in conversion from ``LinearExpression`` to ``AnonymousConstraint``.

Version 0.0.5
-------------

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
