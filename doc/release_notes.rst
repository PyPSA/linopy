Release Notes
=============

Upcoming Release
----------------

* LinearExpression has a new function `densify_terms` which reduces the `_term` axis to a minimal length while containing all non-zero coefficient values.
* When summing over one or multiple axes in a LinearExpression, terms with coefficient of zeros can now be dropped automatically.
* The export of LP files was restructured and is now using Unicode dtype under the hook to ensure performant export of long constraints.

**Breaking changes**

* The low level IO function ``linopy.io.str_array_to_file`` was renamed to ``linopy.io.array_to_file``, the function ``linopy.io.join_str_arrays`` was removed.

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
