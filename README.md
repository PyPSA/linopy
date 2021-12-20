# linopy: Linear optimization with N-D labeled variables

[![PyPI](https://img.shields.io/pypi/v/linopy)](https://pypi.org/project/linopy/) [![CI](https://github.com/FabianHofmann/linopy/actions/workflows/CI.yaml/badge.svg)](https://github.com/FabianHofmann/linopy/actions/workflows/CI.yaml) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![doc](https://readthedocs.org/projects/linopy/badge/?version=latest)](https://linopy.readthedocs.io/en/latest/) [![codecov](https://codecov.io/gh/PyPSA/linopy/branch/master/graph/badge.svg?token=TT4EYFCCZX)](https://codecov.io/gh/PyPSA/linopy)

**linopy** is an open-source python package that facilitates **linear or mixed-integer optimisation** with **real world data**. It builds a bridge between data analysis packages like [xarray](https://github.com/pydata/xarray) & [pandas](https://pandas.pydata.org/) and linear problem solvers like [cbc](https://projects.coin-or.org/Cbc), [gurobi](https://www.gurobi.com/) (see the full list below). The project aims to make linear programming in python easy, highly-flexible and performant.


## Main features

**linopy** is heavily based on [xarray](https://github.com/pydata/xarray) which allows for many flexible data-handling features:

* Define (arrays of) contnuous or binary variables with **coordinates**, e.g. time, consumers, etc.
* Apply **arithmetic operations** on the variables like adding, substracting, multiplying with all the  **broadcasting** potentials of xarray
* Apply **arithmetic operations** on the **linear expressions** (combination of variables)
* **Group terms** of a linear expression by coordinates
* Get insight into the **clear and transparent data model**
* **Modify** and **delete** assigned variables and constraints on the fly
* Use **lazy operations** for large linear programs  with [dask](https://dask.org/)
* Choose from **different commercial and non-commercial solvers**
* Fast **import and export** a linear model using xarray's netcdf IO


## Installation

So far **linopy** is available on the PyPI repository

```bash
pip install linopy
```

## Supported solvers

**linopy** supports the following solvers

* [Cbc](https://projects.coin-or.org/Cbc)
* [GLPK](https://www.gnu.org/software/glpk/)
* [HiGHS](https://www.maths.ed.ac.uk/hall/HiGHS/)
* [Gurobi](https://www.gurobi.com/)
* [Xpress](https://www.fico.com/en/products/fico-xpress-solver)
* [Cplex](https://www.ibm.com/de-de/analytics/cplex-optimizer)

Note that these do have to be installed by the user separately.


# License

Copyright 2021 Fabian Hofmann



This package is published under license GNU Public License GPLv3
