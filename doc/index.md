---
hide:

  - navigation
---

![linopy logo](assets/logo/logo.png){ width="320" }

# linopy: Linear optimization with N-D labeled variables

[![PyPI](https://img.shields.io/pypi/v/linopy)](https://pypi.org/project/linopy/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/linopy.svg)](https://anaconda.org/conda-forge/linopy)
[![Tests](https://github.com/PyPSA/linopy/actions/workflows/test.yml/badge.svg)](https://github.com/PyPSA/linopy/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/PyPSA/linopy/branch/master/graph/badge.svg)](https://codecov.io/gh/PyPSA/linopy)
[![License](https://img.shields.io/pypi/l/linopy.svg)](https://github.com/PyPSA/linopy/blob/master/LICENSE.txt)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.04823/status.svg)](https://doi.org/10.21105/joss.04823)

**linopy** is an open-source Python package for **linear, integer,
mixed-integer and quadratic optimization** with real-world data. It connects
data analysis packages like [xarray](https://github.com/pydata/xarray) and
[pandas](https://pandas.pydata.org/) with a wide range of open-source and
commercial solvers. Variables, constraints and expressions are
dimension-labelled N-dimensional arrays, so models are written the way the
data is shaped.

```python
import pandas as pd
import linopy

m = linopy.Model()
days = pd.Index(["Mon", "Tue", "Wed", "Thu", "Fri"], name="day")
apples = m.add_variables(lower=0, coords=[days], name="apples")
bananas = m.add_variables(lower=0, coords=[days], name="bananas")

m.add_constraints(3 * apples + 2 * bananas >= 8, name="daily_vitamins")
m.add_constraints((3 * apples + 2 * bananas).sum() >= 50, name="weekly_vitamins")
m.add_objective((2 * apples + bananas).sum())
m.solve()
```

## Quick Links

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Getting started**

    ---

    [:octicons-arrow-right-24: Installation](home/installation.md)

    [:octicons-arrow-right-24: Create a model](home/create-a-model.ipynb)

    [:octicons-arrow-right-24: Use coordinates](home/create-a-model-with-coordinates.ipynb)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Variables, expressions, constraints, solving and advanced formulations
    in depth.

    [:octicons-arrow-right-24: User Guide](user-guide/overview.md)

-   :material-new-box:{ .lg .middle } **Release Notes**

    ---

    New features, bug fixes and breaking changes, release by release.

    [:octicons-arrow-right-24: What's new](release-notes.md)

    [:octicons-arrow-right-24: Migrating to v1](user-guide/migrating-to-v1.md)

-   :material-code-braces:{ .lg .middle } **API Reference**

    ---

    Every public class and function, generated from the docstrings.

    [:octicons-arrow-right-24: API Reference](api/index.md)

-   :material-lightbulb-on:{ .lg .middle } **Examples**

    ---

    End-to-end models, from a transport problem to a power-plant build-out.

    [:octicons-arrow-right-24: Examples](examples/examples.md)

-   :material-scale-balance:{ .lg .middle } **Open Source, MIT**

    ---

    linopy is licensed under MIT and developed on
    [GitHub](https://github.com/PyPSA/linopy). Contributions are welcome.

    [:octicons-arrow-right-24: Contributing](contributing/contributing.md)

</div>

## Citing linopy

If you use linopy in your research, please cite the
[JOSS paper](https://doi.org/10.21105/joss.04823), see
[Citing linopy](home/citing.md).
