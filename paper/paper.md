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

linopy is an open-source Python package that facilitates linear and mixed-integer optimization with n-dimensional, labeled input data. It uses state-of-the-art data analysis tools to enable a high-level algebraic syntax and a memory-efficient, performant communication with open and proprietary solvers. With

# Statement of need

# Convention

# Related Research

`linopy` is used by several research projects and groups. The [PyPSA package](https://github.com/PyPSA/pypsa) [@brownPyPSAPythonPower2018] is an open-source software for (large scale) energy system modelling and optimization. Together with the [PyPSA-Eur workflow](https://github.com/PyPSA/pypsa-eur) [@horschPyPSAEurOpenOptimisation2018] and the sector-coupled extension [PyPSA-Eur-sec](https://github.com/PyPSA/pypsa-eur-sec) [@brownPyPSAPythonPower2018], it uses `linopy` to solve large linear problems with up to $10^8$ variables.

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
