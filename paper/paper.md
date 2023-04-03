---
title: 'Linopy: Linear optimization with n-dimensional labeled variables'
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
 - name: Department of Digital Transformation in Energy Systems, Technical University of Berlin
   index: 1
#  - name: Climate Analytics gGmbH, Berlin
#    index: 5
date: 5 September 2022
bibliography: paper.bib


---

# Summary

Linopy is an open-source package written in Python to build and process linear and mixed-integer optimization with n-dimensional labeled input data. Using state-of-the-art data analysis packages, Linopy enables a high-level algebraic syntax and memory-efficient, fast communication with open and proprietary solvers. While similar packages use object-oriented implementations of single variables and constraints, Linopy stores and processes its data in an array-based data model. This allows the user to build large optimization models quickly and lays the foundation for features such as fast writing to array-oriented scientific data formats, masking, automatic solving on remote servers and model scaling.

# Statement of need

Decades after its inception [@dantzigLinearProgrammingExtensions1963], mathematical optimization is nowadays of immense importance for business, industry  and governmental decision-making. Optimization is used to address various sorts of complex problems, such as challenges related to climate change, energy transitions, and food supply. Typically, an optimization problem, i.e. a mathematical program, consists of one objective function to be numerically minimized and a set of constraints that restrict the underlying variables to external conditions. Algebraic Modeling Languages (AML) aim at facilitating mathematical programming by allowing the user to formulate large scale, complex problems with a high-level syntax similar to the mathematical notation. The formulated problem is then passed to the solver of choice where a solution is calculated. AMLs provide the most user-friendly interface possible to various solvers, each with its own set of features.

Well established AMLs such as GAMS [@bussieckGeneralAlgebraicModeling2004] and AMPL [@fourerModelingLanguageMathematical1990] support a wide range of solvers, but are license-restricted and rely on closed-source code. In contrast, AMLs as JuMP [@dunningJuMPModelingLanguage2017], CVXPY [@diamondCVXPYPythonembeddedModeling2016], Pyomo [@hartPyomoOptimizationModeling2017], GEKKO [@bealGEKKOOptimizationSuite2018] and PuLP [@mitchelPulp2022] are open-source and have gained increasing attention throughout the recent years. While the Julia package JuMP is characterized by high-performance, in-memory communication with the solvers, the Python packages Pyomo, GEKKO and PuLP lack parallelized, low-level operations and communicate slower with the solver via intermediate files written to disk. An exception is CVXPY, which supports fast array-based operations and uses low-level wrappers to the solvers. However, it is common among Python AMLs not to make use of state-of-the-art data handling packages. In particular, the assignment of coordinates or indexes is often not supported or memory extensive due to use of an object-oriented implementation where every single combination of coordinates is stored separately.

Linopy is an open-source Python package representing a new kind of AML that tackles these issues together. By introducing an array-based data model for variables and constraints, Linopy makes mathematical programming compatible with Python's advanced data handling packages Numpy [@harrisArrayProgrammingNumPy2020], Pandas [@rebackPandasdevPandasPandas2022] and Xarray [@hoyerXarrayNDLabeled2017].

The approach follows the idea that a variable $x(d_1, d_2, ..., d_K)$ may be defined on an arbitrary number of $K\ge 0$ dimensions, each dimension spanning over a set of $N_i$ discrete coordinates of arbitrary data type (integer, string, date-time, etc.), i.e. $d_i \in \{c_{i,1},...,c_{i,N}\}$. The variable $x(d_1, d_2, ..., d_K)$ is then stored as an array of shape $N_1 \times N_2 \times ... \times N_K$ containing integer labels referencing the optimization variables used by the solver. Coordinates are automatically aligned when variables are used in linear expressions or when applying built-in functions, such as summing over specific dimension or grouping by user-defined labels. Note that if a variable should not be defined on the full set of coordinates given by $\{d_1, d_2, ..., d_K\}$, a boolean mask of the same shape may be used to select where the variable is defined and where not.

The array-based modelling approach does not only lead to more flexibility but also increases the overall performance. The following figure shows the [benchmark](https://github.com/PyPSA/linopy/tree/master/benchmark) against the AMLs JuMP, Pyomo, PuLP and CVXPY as well as the solver specific interface Gurobipy. The included AMLs packages are all open source and well-established and therefore suitable for comparison. Linopy outperforms all Python AMLs in memory efficiency and is close to CVXPY and Gurobipy in terms of speed while being faster than JuMP.

![Benchmark of Linopy against comparable packages. The producing Snakemake workflow is available [here](https://github.com/PyPSA/linopy/tree/master/benchmark). The software and hardware specifications are detailed [here](https://github.com/PyPSA/linopy/tree/master/benchmark#versions-specfications). The benchmark is based on a 1-dimensional knapsack problem and uses the Gurobi solver. The overhead is calculated from the difference of the whole solving process via the AML and the solving process on the solver side alone. Note that the benchmark is hardly dependent on the complexity of the problem. Thus, adding more terms to the constraints, setting different kind of index labels or changing it to a purely linear problem does hardly have an effect on the overhead. \label{fig:benchmark}](figures/benchmark-overhead.pdf)

Due to a strong alignment to the Xarray package, Linopy supports storing the optimization model as a NetCDF file [@rewNetCDFInterfaceScientific1990], which allows users to quickly share optimization problems with others. Using the [Paramiko package](https://paramiko.org), Linopy offers the user to send unsolved problems to a server and retrieve the solution after running the optimization remotely, which is particularly helpful if large computing resources are needed.

Linopy supports a list of well-established solvers, namely

* `GLPK` [@GLPKGNUProject]
* `CBC` [@forrestCoinorCbcRelease2022]
* `HiGHS` [@huangfuParallelizingDualRevised2018]
* `Gurobi` [@GurobiFastestSolver]
* `Xpress` [@FICOXpressSolver]
* `CPLEX` [@cplex2009v12]

while other solvers such as `PIPS-IPM++` [@rehfeldtMassivelyParallelInteriorpoint2022], the `SCIP` solver [@bestuzhevaSCIPOptimizationSuite2021] and `MOSEK` [@apsMOSEKOptimizerAPI2019] are planned to be integrated in future versions. Further, upcoming features target model coefficient scaling as for example presented and performed in [@morshedGeneralizedAffineScaling2020] and [@gokeGraphbasedFormulationModeling2021] as well as the integration of non-linear expressions.


# Related Research

Linopy is used by several research projects and groups, mostly related to energy system modelling. The energy system modelling tool [PyPSA package](https://github.com/PyPSA/pypsa) [@brownPyPSAPythonPower2018], which is used by [various institutions](https://pypsa.readthedocs.io/en/latest/users.html) and builds the core of the [PyPSA-Eur workflow](https://github.com/PyPSA/pypsa-eur) [@horschPyPSAEurOpenOptimisation2018],[@brownPyPSAPythonPower2018], uses Linopy as the primary optimization interface. The Fraunhofer Institute for Energy Economics and Energy System Technology is using Linopy in order to create an interface to GPU-based solvers. The German Aerospace Center uses Linopy for calculating stochastic optimization problems. Finally, a TU Berlin and Google Inc. cooperate on a [research project](https://github.com/PyPSA/247-cfe) that uses Linopy to analyze system-level impacts of 24/7 carbon-free electricity procurement in Europe.

# Availability

Stable versions of the Linopy package are available for Linux, MacOS and Windows via
`pip` in the [Python Package Index (PyPI)](https://pypi.org/project/linopy/).
<!-- and for `conda` on [conda-forge](https://anaconda.org/conda-forge/linopy) [@AnacondaSoftwareDistribution2020]. -->
Development branches are available in the project's [GitHub repository](https://github.com/PyPSA/linopy) together with a documentation on [Read the Docs](https://linopy.readthedocs.io/en/master/).
For continuous integration, Linopy uses automated tests on Github together with [Pre-Commit hooks](https://pre-commit.com/). The Linopy package is released under [GPLv3](https://github.com/PyPSA/linopy/blob/master/LICENSES/GPL-3.0-or-later.txt) and welcomes contributions via the project's [GitHub repository](https://github.com/PyPSA/linopy).

# Acknowledgements

We thank all [contributors](https://github.com/PyPSA/linopy/graphs/contributors) who helped to develop Linopy, in particular Jonas Hörsch who contributed important changes to the architecture of the package. Linopy was inspired by the [Nomopyomo](https://github.com/PyPSA/nomopyomo) prototype written by Tom Brown and its approach to store variables as integer labels.

Fabian Hofmann is funded by the [BreakthroughEnergy initiative](https://www.breakthroughenergy.org/).

# References
