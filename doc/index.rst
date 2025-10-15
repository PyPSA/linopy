.. linopy documentation master file, created by
   sphinx-quickstart on Tue Jun 15 10:15:57 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: linopy

linopy: Linear optimization with N-D labeled variables
======================================================

|PyPI| |CI| |License|


Welcome to Linopy! This Python library is designed to make linear programming easy, flexible, and performant. Whether you're dealing with Linear, Integer, Mixed-Integer, or Quadratic Programming, Linopy is as a user-friendly interface to define variables and constraints. It serves as a bridge, connecting data analysis packages such like
`xarray <https://github.com/pydata/xarray>`__ &
`pandas <https://pandas.pydata.org/>`__ with problem solvers.


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
-  Support of various solvers
   - `Cbc <https://projects.coin-or.org/Cbc>`__
   - `GLPK <https://www.gnu.org/software/glpk/>`__
   - `HiGHS <https://www.maths.ed.ac.uk/hall/HiGHS/>`__
   - `MindOpt <https://solver.damo.alibaba.com/doc/en/html/index.html>`__
   - `Gurobi <https://www.gurobi.com/>`__
   - `Xpress <https://www.fico.com/en/products/fico-xpress-solver>`__
   - `Cplex <https://www.ibm.com/de-de/analytics/cplex-optimizer>`__
   - `MOSEK <https://www.mosek.com/>`__
   - `COPT <https://www.shanshu.ai/copt>`__



Citing Linopy
-------------

If you use Linopy in your research, please cite it as follows:


   Hofmann, F., (2023). Linopy: Linear optimization with n-dimensional labeled variables.
   Journal of Open Source Software, 8(84), 4823, https://doi.org/10.21105/joss.04823


A BibTeX entry for LaTeX users is

   @article{Hofmann2023,
   doi = {10.21105/joss.04823},
   url = {https://doi.org/10.21105/joss.04823},
   year = {2023}, publisher = {The Open Journal},
   volume = {8},
   number = {84},
   pages = {4823},
   author = {Fabian Hofmann},
   title = {Linopy: Linear optimization with n-dimensional labeled variables},
   journal = {Journal of Open Source Software} }



License
-------

Copyright 2021-2023 Fabian Hofmann

This package is published under MIT license.

.. |PyPI| image:: https://img.shields.io/pypi/v/linopy
   :target: https://pypi.org/project/linopy/
.. |CI| image:: https://github.com/FabianHofmann/linopy/actions/workflows/CI.yaml/badge.svg
   :target: https://github.com/FabianHofmann/linopy/actions/workflows/CI.yaml
.. |License| image:: https://img.shields.io/pypi/l/linopy.svg
   :target: https://mit-license.org/


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   prerequisites
   create-a-model
   create-a-model-with-coordinates

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   user-guide
   creating-variables
   creating-expressions
   creating-constraints
   manipulating-models
   testing-framework
   transport-tutorial
   infeasible-model
   solve-on-remote
   solve-on-oetc
   migrating-from-pyomo
   gurobi-double-logging


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
