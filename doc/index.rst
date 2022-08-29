.. linopy documentation master file, created by
   sphinx-quickstart on Tue Jun 15 10:15:57 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: linopy

linopy: Linear optimization with N-D labeled variables
======================================================

|PyPI| |CI| |License: GPL v3|

**linopy** is an open-source python package that facilitates **linear or
mixed-integer optimisation** with **real world data**. It builds a
bridge between data analysis packages like
`xarray <https://github.com/pydata/xarray>`__ &
`pandas <https://pandas.pydata.org/>`__ and linear problem solvers like
`cbc <https://projects.coin-or.org/Cbc>`__,
`gurobi <https://www.gurobi.com/>`__ (see the full list below). The
project aims to make linear programming in python easy, highly-flexible
and performant.

Main features
-------------

**linopy** is heavily based on
`xarray <https://github.com/pydata/xarray>`__ which allows for many
flexible data-handling features:

-  Define (arrays of) contnuous or binary variables with
   **coordinates**, e.g. time, consumers, etc.
-  Apply **arithmetic operations** on the variables like adding,
   subtracting, multiplying with all the **broadcasting** potentials of
   xarray
-  Apply **arithmetic operations** on the **linear expressions**
   (combination of variables)
-  **Group terms** of a linear expression by coordinates
-  Get insight into the **clear and transparent data model**
-  **Modify** and **delete** assigned variables and constraints on the
   fly
-  Use **lazy operations** for large linear programs with
   `dask <https://dask.org/>`__
-  Choose from **different commercial and non-commercial solvers**
-  Fast **import and export** a linear model using xarray’s netcdf IO

Installation
------------

So far **linopy** is available on the PyPI repository

.. code:: bash

   pip install linopy

Supported solvers
-----------------

**linopy** supports the following solvers

-  `Cbc <https://projects.coin-or.org/Cbc>`__
-  `GLPK <https://www.gnu.org/software/glpk/>`__
-  `HiGHS <https://www.maths.ed.ac.uk/hall/HiGHS/>`__
-  `Gurobi <https://www.gurobi.com/>`__
-  `Xpress <https://www.fico.com/en/products/fico-xpress-solver>`__
-  `Cplex <https://www.ibm.com/de-de/analytics/cplex-optimizer>`__

Note that these do have to be installed by the user separately.

License
=======

Copyright 2021 Fabian Hofmann

This package is published under license GNU Public License GPLv3

.. |PyPI| image:: https://img.shields.io/pypi/v/linopy
   :target: https://pypi.org/project/linopy/
.. |CI| image:: https://github.com/FabianHofmann/linopy/actions/workflows/CI.yaml/badge.svg
   :target: https://github.com/FabianHofmann/linopy/actions/workflows/CI.yaml
.. |License: GPL v3| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   create-a-model
   solvers

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   creating-variables
   creating-constraints
   manipulating-models
   solve-on-remote


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Benchmarking

   benchmark
   syntax

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: References

   api
   release_notes
   contributing
