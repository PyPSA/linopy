Release Notes
=============

.. Upcoming Version

* Fix compatibility for xpress versions below 9.6 (regression)
* Performance: Up to 50x faster ``repr()`` for variables/constraints via O(log n) label lookup and direct numpy indexing
* Performance: Up to 46x faster ``ncons`` property by replacing ``.flat.labels.unique()`` with direct counting

Version 0.5.8
--------------

* Replace pandas-based LP file writing with polars implementation for significantly improved performance on large models
* Consolidate "lp" and "lp-polars" io_api options - both now use the optimized polars backend
* Reduced memory usage and faster file I/O operations when exporting models to LP format
* Minor bugfix for multiplying variables with numpy type constants
* Harmonize dtypes before concatenation in lp file writing to avoid dtype mismatch errors. This error occurred when creating and storing models in netcdf format using windows machines and loading and solving them on linux machines.
* Add option to use polars series as constant input
* Fix expression merge to explicitly use outer join when combining expressions with disjoint coordinates for consistent behavior across xarray versions
* Adding xpress postsolve if necessary
* Handle ImportError in xpress import
* Fetch and display OETC worker error logs
* Fix windows permission error when dumping model file
* Performance improvements for xpress solver using C interface

Version 0.5.7
--------------

* Removed deprecated future warning for scalar get item operations
* Silenced version output from the HiGHS solver
* Mosek: Remove explicit use of Env, use global env instead
* Objectives can now be created from variables via `linopy.Model.add_objective`
* Added integration with OETC platform (refactored implementation)
* Add error message if highspy is not installed
* Fix MindOpt floating release issue
* Made merge expressions function infer class without triggering warnings
* Improved testing coverage
* Fix pypsa-eur environment path in CI

Version 0.5.6
--------------

* Improved variable/expression arithmetic methods so that they correctly handle types
* Gurobi: Pass dictionary as env argument `env={...}` through to gurobi env creation

**Breaking Changes**

* With this release, the package support for Python 3.9 was dropped and support for Python 3.10 was officially added.
* The selection of a single item in `__getitem__` now returns a `Variable` instead of a `ScalarVariable`.


Version 0.5.5
--------------

* Internally assign new data fields to expressions with a multiindexed-safe routine.

Version 0.5.4
--------------


**Bug Fixes**

* Remove default highs log file when `log_fn=None` and `io_api="direct"`. This caused `log_file` in
  `solver_options` to be ignored.
* Fix the parsing of solutions returned by the CBC solver when setting a MIP duality
  gap tolerance.
* Improve the mapping of termination conditions for the SCIP solver
* Treat GLPK's `integer undefined` status as not-OK
* Internally assign new data fields to `Variable` and `Constraint` with a multiindexed-safe routine. Before the
  assignment when using multi-indexed coordinates, an deprecation warning was raised. This is fixed now.


Version 0.5.3
--------------

**Bug Fixes**

* Fix the parsing of solutions returned by the CBC solver when solving from a file to not
  assume that variables start with `x`.
* Fix the retrieval of solutions from the SCIP solver, and do not turn off presolve.

**Minor Improvements**

* Support pickling models.

Version 0.5.2
--------------

**Bug Fixes**

* Fix the multiplication with of zero dimensional numpy arrays with linopy objects.
  This is mainly affecting operations where single numerical items from  pandas objects
  are selected and used for multiplication.

Version 0.5.1
--------------

**Deprecations**

* Renamed `expression.empty()` to `expression.empty` to align with the use of empty in
  pandas. A custom wrapper ensures that `expression.empty()` continues to work, but emits
  a DeprecationWarning.

**Features**

** Features **

* Added support for arithmetic operations with custom classes.
* Added `align` function as a wrapper around :func:`xr.align`.
* Avoid allocating a floating license for COPT during the initial solver check

**Bug fixes**

* Ensure compatibility with xarray >= v2025.03.00

Version 0.5.0
--------------

**Features**

* Multiplication of a linear expression by a constant value may now introduce new
  dimensions.
* Added method `unstack` to `LinearExpression`, `Variable` and `Constraint` to unstack
  a dimension.
* Added extra argument in io methods `explicit_coordinate_names` to allow for export of
  variables and constraints with explicit coordinate names.

**Bug fixes**

* The internal handling of `Solution` objects was improved for more consistency.
  Solution objects created from solver calls now preserve the exact index names from
  the input file.

Version 0.4.4
--------------

* **IMPORTANT BUGFIX**: The last slice of constraints was not correctly written to LP files in case the constraint size was not a multiple of the slice size. This is fixed now.
* Solution files that following a different naming scheme of variables and constraints using more than on initial letter in the prefix (e.g. `col123`, `row456`) are now supported.
* GLPK solver is always called with the `--freemps` option instead of the `--mps` when using the Solver API to solve an external MPS file. `--mps` is for the older fixed-column MPS format that is rarely used nowadays. Almost all fixed MPS files can be parsed by the free MPS format.

Version 0.4.3
--------------

* **Version 0.4.3 includes a major bug and can not be installed anymore.**
* When creating slices for variables and constraints (important for the `solve` function), the slicing is now fixed in case no dimension to slice is available.
* Added a pandas priority attribute. With this change, the operation with pandas objects is now prioritizing linopy objects over pandas objects. This is useful when the using linopy objects in arithmetic operations with pandas objects, e.g. `a * x` where `a` is a pandas Series/DataFrame and `x` is a linopy variable.
* The method :meth:`model.to_file <linopy.model.Model.to_file>` now includes a progress argument to enable or disable the progress bar while writing.

Version 0.4.2
--------------

* **Version 0.4.2 includes a major bug and can not be installed anymore.**
* Fix the file handler to properly close the file when reading the sense from a problem file.

Version 0.4.1
--------------

* Fix the `slice_size` argument in the `solve` function. The argument was not properly passed to the `to_file` function.
* Fix the slicing of constraints in case the term dimension is larger than the leading constraint coordinate dimension.

Version 0.4.0
--------------

* When writing out an LP file, large variables and constraints are now chunked to avoid memory issues. This is especially useful for large models with constraints with many terms. The chunk size can be set with the `slice_size` argument in the `solve` function.
* Constraints which of the form `<= infinity` and `>= -infinity` are now automatically filtered out when solving. The `solve` function now has a new argument `sanitize_infinities` to control this feature. Default is set to `True`.
* The representation of linopy objects with multiindexed coordinates was improved to be more readable.
* Grouping expressions is now supported on dimensions called "group" and dimensions that have the same name as the grouping object.
* Grouping dimensions which have multiindexed coordinates is now supported.

Version 0.3.15
--------------

* The group dimension when grouping by a pandas dataframe is now always `group`. This fixes the case that the dataframe contains a column named `name`.

Version 0.3.14
--------------

* Ensure compatibility with xarray >= v2024.07.0, which has drop the ``squeeze`` argument from the ``groupby`` function.

Version 0.3.13
--------------

* Follow-up release to properly fix all deprecations from multiindexed data assignments in xarray datasets.
* Fix typing relevant import for non-default highs dependency in vanilla installation

Version 0.3.12
--------------

* Support for warmstart in HiGHS using basis or solution files, including support for writing basis and solution files of a solved model.
* Linopy now uses mypy for type checking allowing for a more secure and stable code base.
* The creation of solution files with gurobi, scip and mindopt is now supported.

Version 0.3.11
--------------

* The writing and reading from netcdf files was fixed to correctly handle the model `parameters` field.

Version 0.3.10
--------------

* The classes `Variable`, `LinearExpression` and `Constraint` now have a new `getitem` method that allows selecting a subset of the object in the same way as `xarray` objects, i.e. by integer labels or boolean index. Example usage: `x[[1, 2]]` or `x[x.indexes["some_index"] > 5]`.

* The class `Constraint` now has a new method `.loc` to select a subset of the constraint by labels.

* Selecting a single variable with the `getitem` (`[]`) method now raises a `FutureWarning` that the return type will change to `Variable` instead of a `ScalarVariable` in the future. To get a `ScalarVariable` in the future, use the `at[]` method.

* A new module `examples` was added which contains example models. For example, you can call `m = linopy.examples.benchmark_model()`.

* A new memory-efficient and super fast LP file writing method was added which uses the `Polars package <https://github.com/pola-rs/polars>`_. It is still in experimental mode but seems to be very promising. Activate it with the `io_api="lp-polars"` argument in the `solve` function.


* The Constraint class now supports the methods `assign`, `assign_attrs`, `assign_coords`, `broadcast_like`, `chunk`, `drop_sel`, `drop_isel`, `expand_dims`, `sel`, `isel`, `shift`, `swap_dims`, `set_index`, `reindex`, `reindex_like`, `rename`, `rename_dims`, `roll`, `stack`. These methods allow to manipulation of a (anonymous) constraint more flexibly.

* The Variable, expressions and Constraint classes now have new methods `swap_dims` and `set_index`. The `swap_dims` method allows to swap the dimensions of the object. The `set_index` method allows to set a new index for the object. Both methods are useful for reshaping the object more flexibly.

Version 0.3.9
-------------


* The matrices accessor of the `Model` class now has a new function `dual` which returns the dual values of the constraints if the underlying model was optimized and dual values are existent.

* The Variables class now has a new function `get_solver_attribute` which parses solver-specific attributes of the variables. For now, this function only works for Gurobi `solver_model`s. For example, the function allows retrieving the variable fields `SAObjUp` or `RC`.

* The constraint assignment with a `LinearExpression` and a constant value when using the pattern `model.add_constraints(lhs_with_constant, sign, rhs)` was fixed. Before, the constant value was not added to the right-hand-side properly which led to the wrong constraint behavior. This is fixed now.

* ``nan`` s in constants is now handled more consistently. These are ignored when in the addition of expressions (effectively filled by zero). In a future version, this might change to align the propagation of ``nan`` s with tools like numpy/pandas/xarray.

* Up to now the `rhs` argument in the `add_constraints` function was not supporting an expression as an input type. This is now added.

* Linopy now supports python 3.12.

**Deprecations**

* The argument `dims` in the `.sum` function of variables and expressions was deprecated in favor of the `dim` argument. This aligns the argument name with the xarray convention.

Version 0.3.8
-------------

**New Features**

* The LinearExpression and QuadraticExpression class have a new attribute `solution` which returns the optimal values of the expression if the underlying model was optimized.

* It is now possible to access variables and constraints, that don't have python variable name format, as attributes from the corresponding containers. Therefore, a new formatting scheme was introduced which converts dashes and white spaces into underscores. For example, a variable was added to the model with the label "my-variable". This variable can now be accessed with `model.variables.my_variable`. In particular, the autocompletion function of the IPython console is aware of this new formatting scheme. This allows easy access to variables and constraints with long labels.

* Variables and LinearExpressions now have a new method `dot`, which allows computing the dot product of two objects. This multiplies objects and sums over common dimensions.

* The matmul operator `@`, which runs the `dot` operation, is now supported for Variables and LinearExpression.

**Bugfixes**

* The multiplication of two linear expression with non-zero constants led to wrong results of the cross terms. Given the multiplication `(v1 + c1)  * (v2 + c2)` with `v` being a variable and `c` a constant, the operation did not calculate the cross terms `v1 * c2 + v2 * c1`. This is fixed now.


Version 0.3.7
-------------

**New Features**

* A direct interface to the `Mosek` solver was added. With this change, a new conversion function `model.to_mosek` was added to convert a linopy model to a `mosek` model. The `solve` function now supports the `mosek` solver with `io_api="direct"`.

* It is now possible to create LinearExpression from a `pandas.DataFrame`, `pandas.Series`, a `numpy.array` or constant scalar values, e.g. `linopy.LinearExpression(df)`. This will create a LinearExpression with constants only and the coordinates of the DataFrame, Series or array as dimensions.

**Bugfixes**

* When grouping an expression or a variable by a `pandas.DataFrame` or a `xarray.DataArray`, the coordinates of the `groupby` object were not properly aligned. So in cases, when the `groupby` object was not indexed in the same way as the variable/expression, the `groupby` operation led to wrong results. This is fixed now.


Version 0.3.6
-------------

* The handling of `pandas` objects was improved. As `pandas` objects are fully aware of coordinates, their index and columns are now strictly taken into account. For example, when multiplying a `pandas.DataFrame` with a variable, linopy now checks the alignment of indexes and reindexes accordingly. Previously, if the axis shapes were the same, the indexes of the variable were inserted and the `pandas` indexes were effectively ignored. A warning has been added for cases where users should expect changes to the results with this version. **Important**: This does not apply to overwriting the coordinates when one expression is added to another, e.g. "x + df" still overwrites the index of "df" when the dimensional shapes are aligned.
* The `.mask` attribute of the `Constraint` class was fixed to return a proper boolean `xarray.DataArray` object.
* The printout of masked constraints was fixed.


Version 0.3.5
-------------

* The return type of ``coord_dims`` for expressions and constraints was changed from set to tuple to align with the xarray convention.
* The printout of transposed expressions and constraints was fixed.
* Variables and LinearExpressions now support the chaining operations `.add`, `.sub`, `.mul`, `.div`.
* Variables and LinearExpressions now have support for the power operator. For example, `x**2` is now supported.

Version 0.3.4
-------------

* Solver output of CBC and GLPK is sent to logging with level INFO instead of stdout
* Added support for QP problems with MOSEK and COPT.
* A warning was added when linopy is not able to add pass quadratic objective terms to the highs solver. This is the case when the "ipm" solver of highs is explicitly selected.


Version 0.3.3
-------------


* New solver interface for `SCIP <https://www.scipopt.org/>`. This solver is now supported by `linopy` and can be used with the `solve` function if the `pyscipopt` package is installed. The solver is available for free for general use. See the `SCIP website <https://www.scipopt.org/>` for more information.
* Linopy was refactored to use the new xarray API (>=2024.01) without the deprecation warnings.
* The set "quadratic_solvers" now only contains quadratic solvers which are installed and available to the user.
* The `solve` function now throws an error instead of a warning if the set value for ``io_api`` is not available for a solver.

Version 0.3.2
-------------

* The IO with NetCDF files was made more secure and fixed for some cases. In particular, variables and constraints with a dash in the name are now supported (as used by PyPSA). The object sense and value are now properly stored and retrieved from the netcdf file.
* The IO with NetCDF file now supports multiindexed coordinates.
* The representation of single indexed expressions and constraints with non-empty dimensions/coordinates was fixed, e.g. `x.loc[["a"]] > 0` where `x` has only one dimension. Therefore the representation now shows the coordinates.
* The creation of ``LinearExpression`` and ``Constraints`` was made robust against the case where the ``data`` argument is a ``xarray.DataArray`` with helper dimensions (like "_term" etc.) unintentionally assigned as coordinates.

Version 0.3.1
-------------


**New Features**

* Added solver interface for MOSEK.
* Support for MindOpt solver was added.
* Added solver interface for COPT by Cardinal Optimizer.
* Type consistency with fill values for constant values was improved, this prevent dtype warnings put out by xarray/numpy.

Version 0.3.0
-------------


**New Features**

* It is now possible to set the sense of the objective function to `minimize` or `maximize`. Therefore, a new class `Objective` was introduced which is used in `Model.objective`. It supports the same arithmetic operations as `LinearExpression` and `QuadraticExpression` and contains a `sense` attribute which can be set to `minimize` or `maximize`.
* The `fillna` function for variables was made more secure by raising a warning if the fill value is not of  variable-like type.
* The `where` and `fillna` functions for expressions were made more flexible: When passing a scalar value or a DataArray, the values are added as constants to the expression, where there were missing values before. If another expression is passed, the values are added to the expression, where there were missing values before.

**Breaking Changes**

* The `_fill_value` for LinearExpression and QuadraticExpression classes was changed to ``NaN`` for the constant array ("const"). This allows to use the `where` function for expressions with constant values in the argument `other`.
* The functions ``ravel`` and ``iter_ravel`` for Variables and Constraints were removed in favor of the ``flat`` function.
* The property ``non_helper_dims`` for Variables and Constraints was removed in favor of the ``coord_dims`` property.
* The function ``to_anonymous_constraint`` was removed in favor of the ``to_constraint`` function.
* The support for python 3.8 has been dropped.

Version 0.2.6
-------------

* The memory-efficiency of the IO to LP/MPS file was further improved. In particular, the function `to_dataframe` is now avoiding unnecessary data copies.
* The printout of time stamps was modified to be more readable, leaving out the display of seconds and below if not necessary.
* The gurobi environment is now enclosed in a context manager to avoid any unwanted use of a token.


Version 0.2.5
-------------


* The solution getter `model.solution` was falsely returning integer dtype in case of non-aligned indexes. This is fixed now.
* Highs is now in the set of default solvers when install `linopy` via pip.


Version 0.2.4
-------------


* The IO to LP/MPS file was made more memory-efficient. In particular, the memory excessive operation `to_dataframe` (see https://github.com/pydata/xarray/issues/6561) was replaced by an in-house implementation.


Version 0.2.3
-------------

**Bugfixes**

* When multiplying a `LinearExpression` with a constant value, the constant in the `LinearExpression` was not updated. This is fixed now.

**New Features**

* The `Variable` and the `LinearExpression` have a new method `cumsum`, which allows to compute the cumulative sum.


Version 0.2.2
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
