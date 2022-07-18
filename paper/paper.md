---
title: 'linopy: Linear optimization with N-D labeled variables'
tags:
  - python
  - linear optimization
authors:
  - name: Fabian Hofmann
    orcid: 0000-0002-6604-5450
    affiliation: "1" # (Multiple affiliations must be quoted)
  # - name: Tom Brown
  #   orcid: 0000-0001-5898-1911
  #   affiliation: "4"
  # - name: Jonas Hörsch
  #   orcid: 0000-0001-9438-767X
  #   affiliation: "5" # (Multiple affiliations must be quoted)
affiliations:
 - name: Institute of Energy Technology, Technical University of Berlin
   index: 1
#  - name: Climate Analytics gGmbH, Berlin
#    index: 5
date: 11 July 2022
bibliography: paper.bib


---

# Summary

`linopy` is an open-source package written in Python to facilitate linear and mixed-integer optimization with n-dimensional labeled input data. Using state-of-the-art data analysis packages, `linopy` enables a high-level algebraic syntax and memory-efficient, performant communication with open and proprietary solvers. While similar packages use object-oriented implementations of single variables and constraints, `linopy` stores and processes its data in an array-based data model. This allows the user to build large optimization models quickly in parallel and lays the foundation for features such as fast writing to NetCDF file, masking, solving on remote servers, and model scaling.

# Statement of need

The research community relies on a list of open-source and proprietary solvers such as `GLPK` [@GLPKGNUProject], `Gurobi` [@GurobiFastestSolver], `Xpress` [@FICOXpressSolver] etc. To ensure comparability and re-usability for a wide range of users, many research projects dealing with optimization programs aim to make their work compatibles with many solvers.
<!-- * There is therefore the general need, besides the solver specific interfaces like `Gurobipy` [@GurobipyPythonInterface] or the `Xpress` Python interface [@teamXpressFICOXpress]  -->
Fulfilling this need, JuMP[@dunningJuMPModelingLanguage2017] has recently gained a lot of attention due to a high-level optimization language, a fast implementation in Julia and direct communication with solvers. It builds an interface to a wide range of solvers allowing the user to choose the preferred or available solver.  Other interfaces like GAMS...

The equivalent packages in Python, namely Pyomo [@hartPyomoOptimizationModeling2017] and PuLP [@Pulp2022], are much less performant due to the lack of parallelized, low-level operations and slow communication / LP file writing.
Further, the Python equivalents do not make use of state-of-the-art data handling packages. In particular, the assignment of coordinates or indexes is often not supported or impacting the memory-efficiency due to use of memory-intensive object-oriented implementation where every single combination of coordinates is stored separately.
`linopy` tackles these issues together. By introducing an array-based data model for variables and constraints, it makes mathematical programming compatible with Python's advanced data handling packages like Numpy[@harrisArrayProgrammingNumPy2020], Pandas [@rebackPandasdevPandasPandas2022] or Xarray [@hoyerXarrayNDLabeled2017] while significantly increasing speed and flexibility.
The core data classes `Variable`, `LinearExpression` and `Constraint`  are subclasses of `xarray`'s `DataArray` and `Dataset` class containing n-dimensional arrays with unique labels referencing the optimization variables and coefficients.
...

# Basic Structure

Variables are defined together with a set of dimensions and their corresponding coordinates.
For example, creating a variable $x(d_1, d_2)$ defined on $d_1 \in \{1,N\}$ and $d_2 \in \{1, M\}$, would only require passing $d_1$ and $d_2$ to the variable initialization, with optional lower and upper bounds $l_x(d_1,d_2)$ and $u_x(d_1,d_2)$ being defined on (a subset of) $\{d_1, d_2\}$. The returned object is an $N \times M$ array of integer labels referencing to the optimization variables used by the solver.
...

# Related Research

`linopy` is used by several research projects and groups. The [PyPSA package](https://github.com/PyPSA/pypsa) [@brownPyPSAPythonPower2018] is an open-source software for (large scale) energy system modelling and optimization.

**Not yet:**
Together with the [PyPSA-Eur workflow](https://github.com/PyPSA/pypsa-eur) [@horschPyPSAEurOpenOptimisation2018] and the sector-coupled extension [PyPSA-Eur-sec](https://github.com/PyPSA/pypsa-eur-sec) [@brownPyPSAPythonPower2018], it uses `linopy` to solve large linear problems with up to $10^8$ variables.

# Availability

Stable versions of the `linopy` package are available for Linux, MacOS and Windows via
`pip` in the [Python Package Index (PyPI)](https://pypi.org/project/linopy/)
<!-- and for `conda` on [conda-forge](https://anaconda.org/conda-forge/linopy) [@AnacondaSoftwareDistribution2020]. -->
Development branches are available in the project's [GitHub repository](https://github.com/PyPSA/linopy) together with a documentation on [Read the Docs](https://linopy.readthedocs.io/en/master/).
The `linopy` package is released under [GPLv3](https://github.com/PyPSA/linopy/blob/master/LICENSES/GPL-3.0-or-later.txt) and welcomes contributions via the project's [GitHub repository](https://github.com/PyPSA/linopy).

# Acknowledgements

We thank all [contributors](https://github.com/PyPSA/linopy/graphs/contributors) who helped to develop `linopy`.
Fabian Hofmann is funded by the BreakthroughEnergy initiative.

# References
