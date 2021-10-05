Release Notes
=============

Upcoming Release
----------------

* The "_term" dimension in the LinearExpression class is now stored without coordinates.
* As soon as a LinearExpression is added to a model the "_term" dimension is rename to "{constraintname}_term" in order align the model better with the contained arrays and to avoid unnecessary nans.
* `LinearExpression` now allows empty data as input.

Version 0.0.3
-------------

* Support assignment of variables and constraints without explicit names.
* Add support for xarray version > 0.16
* Add a documentation

Version 0.0.2
-------------

* Set up first runnable prototype.
