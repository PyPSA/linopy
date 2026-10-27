.. _spec-data:

Spec data and results
=====================

:doc:`building-models-from-specs` walks through a spec from YAML to a solved
model. This page is the reference for the parts around it: how linopy reads the
spec and its data, what a missing row means, where the data lives after the
build, and how to read results, rebuild, typeset and vary a model.

A spec-built model is a plain linopy model. Everything the spec adds beyond
the variables, constraints and objective is read as a function of the spec and
the model:

.. code-block:: python

    from pathlib import Path

    from linopy import Model
    from linopy.spec import evaluate, parameter

    spec = Path("dispatch.yaml")
    m = Model.from_spec(spec, sources)
    m.solve()

    evaluate(spec, m)  # every named expression, folded over the solution
    evaluate(spec, m, "spend")["spend"]  # one of them
    parameter(spec, m, "p_max")  # a parameter, filled as its missing: says

All three need ``pip install "linopy[spec]"`` (Python >= 3.12).

.. contents::
   :local:
   :depth: 1


Passing the spec
----------------

``Model.from_spec``, :func:`linopy.spec.evaluate`,
:func:`linopy.spec.parameter` and ``linopy.testing.synthetic_sources``
take the spec in any of four forms:

- a ``pathlib.Path`` to a YAML file,
- a ``str`` holding the YAML text itself,
- a mapping, the spec as a ``dict``,
- a ``mathspec.Spec``, such as the result of ``mathspec.to_spec``,
  ``mathspec.override`` or ``mathspec.merge``.

A ``str`` is always YAML text, never a file name. A one-line ``str`` ending in
``.yaml`` or ``.yml`` reads like a file name, and is refused with a
``ValueError`` pointing to ``pathlib.Path``:

.. code-block:: python

    Model.from_spec("dispatch.yaml", sources)  # ValueError
    Model.from_spec(Path("dispatch.yaml"), sources)  # reads the file


The source contract
-------------------

``sources`` is any ``collections.abc.Mapping`` keyed by the names the spec
declares, or an ``xarray.Dataset`` whose indexes are dimension sources and whose
data variables are parameters and relations.

linopy reads the keys once, off ``keys()``, and never iterates the mapping
beyond that. Each declared name is then read with one ``__getitem__``. A mapping
can therefore compute its values on access, from a database or a file, as in
the :ref:`SQLite example <spec-data-sqlite>` below.

**Dimensions.** A dimension's members come only from the source keyed by its
name, in the order the source gives them; linopy never sorts them. A dimension
source is a ``pandas.Index``, a list, a tuple, a 1-D numpy or pandas array
(including extension arrays such as a ``Categorical`` or ``Int64``), a
``pandas.Series`` or a 1-D ``DataArray``. A ``set`` or ``frozenset`` has no
order and is read sorted; one whose labels do not sort is refused. Each label
appears once.

**Parameters.** A parameter source is read for values, aligned onto the
dimension members. It can be

- a ``DataArray`` over the declared dimensions, in any order,
- a ``pandas.Series`` whose (Multi)Index levels are the declared dimensions,
- a ``dict`` keyed by label,
- a tidy ``DataFrame`` with one column (or index level) per dimension and the
  values in a column named after the parameter or ``value``; the
  parameter's name wins when both are present,
- a wide ``DataFrame`` for a parameter over two dimensions, its index and
  columns named after them (a square frame with unnamed axes is refused, as it
  does not say which axis is which),
- one number, for a parameter over no dimension or the same value everywhere.

**Relations.** A relation maps one key dimension into one value dimension. Its
source is a ``pandas.Series`` indexed by the key labels, a ``dict`` keyed by
them, or a 1-D ``DataArray`` over the key dimension. A ``NaN`` in a
``DataArray`` relation leaves that key label unmapped. Bare relations and
relations with several key or value columns raise ``SpecDataError``.

Missing rows
~~~~~~~~~~~~

A missing row is a member with no value. How it is spelt depends on the shape:

- In a **table** (a ``Series``, a ``dict`` or a tidy ``DataFrame``), a missing
  row is a row that is not there. A ``NaN`` in a table is refused: the row says
  the coordinate exists and denies it in the same breath. Drop the row instead.
- In a **grid** (a ``DataArray``, a ``Dataset`` data variable or a wide
  ``DataFrame``), every coordinate has a cell, so a ``NaN`` (``NaT`` for a
  datetime, ``NaN`` in an object array for a string) is a missing row.

A grid cannot hold a hole in an integer or boolean array, so an ``int`` or
``bool`` parameter with holes arrives as floats with ``NaN``: an ``int`` passes
when every present value is whole, a ``bool`` when every present value is 0 or
1. Without a hole, floats for an ``int`` or ``bool`` parameter are refused.

The parameter's ``missing:`` decides what a missing row means:

- ``refused`` (the default): every row a declaration reads must be there, or
  ``SpecDataError`` names the declaration, the parameter and the coordinate.
  A row is read by a variable's bounds and a constraint's sides on the rows
  their ``where:`` keeps, and by the objective and a named expression
  everywhere; a ``shift``, a lookup or a window reads the row it moves, and
  a ``cases:`` value only inside its region. A hole is not read where the
  term reading it is absent: a variable its ``where:`` masks, or an
  ``absent`` row of another parameter, takes the term with it, so
  ``w * x`` reads no ``w`` where ``x`` does not exist. A hole no
  declaration reads is left alone. A condition never demands a row: a
  missing row compares as false there, whatever the ``missing:``.
- ``neutral``: the row reads as the neutral value, ``0`` or ``False``.
- ``absent``: the row is absent, and the absence propagates through the
  arithmetic that reads it.

Label types
~~~~~~~~~~~

Labels must have the type of the dimension they belong to. An ``int``
dimension given the text ``"0"``, ``"1"`` is refused with a message saying the
labels differ in type, not spelling. For a datetime dimension the message
points to ``pd.to_datetime`` or ``parse_dates=``, since dates read from text
files and databases often arrive as strings.


.. _spec-data-sqlite:

A mapping over a database
-------------------------

Any ``Mapping`` works as a source, so a few lines turn a database into one,
each key read with one query when the build asks for it:

.. code-block:: python

    import sqlite3
    from collections.abc import Iterator, Mapping
    from pathlib import Path
    from typing import Any

    import pandas as pd

    from linopy import Model


    class Tables(Mapping[str, Any]):
        QUERIES = {
            "generator": "select name from generator",
            "snapshot": "select snapshot from load",
            "p_max": "select name as generator, p_max from generator",
            "cost": "select name as generator, cost from generator",
            "load": "select snapshot, load from load",
        }

        def __init__(self, con: sqlite3.Connection) -> None:
            self.con = con

        def __getitem__(self, key: str) -> Any:
            frame = pd.read_sql(self.QUERIES[key], self.con)
            return frame.iloc[:, 0].tolist() if frame.shape[1] == 1 else frame

        def __iter__(self) -> Iterator[str]:
            return iter(self.QUERIES)

        def __len__(self) -> int:
            return len(self.QUERIES)


    m = Model.from_spec(Path("dispatch.yaml"), Tables(sqlite3.connect("fleet.db")))

Dimensions come back as lists, parameters as tidy frames with the values in a
column named after the parameter. Declare a ``snapshot`` read from a date
column as a datetime dimension and parse it with ``parse_dates=`` in
``read_sql``.


What absence removes
--------------------

Absence alone never removes a whole variable or constraint, only individual
rows. To switch a feature off, say so with ``where:``, over a
``missing: absent`` parameter or a named condition:

.. code-block:: yaml

    parameters:
      e_max: { dims: [unit], missing: absent }
    variables:
      soc:
        dims: [snapshot, unit]
        where: e_max
        bounds: { lower: 0, upper: e_max }

- A variable exists exactly where its ``where:`` holds. A missing ``absent`` row
  in one of its bounds leaves that side open; a missing ``refused`` row is
  refused only where the variable exists. Wherever it does not, a parameter
  multiplying it reads no row, in a constraint, the objective or a named
  expression: a cost may be missing where the asset is off.
- A constraint row exists where its ``where:`` holds and absence does not
  propagate to the whole row. An ``absent`` row read outside a sum takes the
  row with it, so ``x <= c`` has no row where ``c`` is absent; write
  ``where: c`` on the constraint to make that visible in the spec.
- Inside a sum, an absent term is left out. A row the data leaves without any
  variable term, such as a sum over variables a ``where:`` masked away
  entirely, is refused rather than dropped: nothing is left to constrain, and
  the row would leave the problem without saying so. Mask it with a ``where:``
  on the constraint (`#993 <https://github.com/PyPSA/linopy/issues/993>`__).


Data in ``m.parameters``
------------------------

``Model.from_spec`` writes the spec's parameters and relations into
``m.parameters`` under their declared names, on the dimension members it
attached. Nothing else of the spec is kept on the model.

The parameters are raw: a missing row is a ``NaN`` hole, and an ``int`` or
``bool`` parameter with a hole is ``float64``. A relation is an array over its
key dimension, ``NaN`` where a label is unmapped.

- :func:`linopy.spec.parameter` returns a parameter filled as its ``missing:``
  says, at its declared dtype; only an ``absent`` or a ``refused`` hole stays
  ``NaN``. Whether a declaration reads a ``refused`` hole is for the build and
  ``evaluate`` to refuse.
- :func:`linopy.spec.evaluate` folds named expressions over ``m.solution``,
  the duals and ``m.parameters``, read afresh on every call. A body over data
  alone needs no solve; one that reads a variable before a solve raises
  ``AttributeError``, one that reads a ``refused`` hole ``SpecDataError``. An
  unknown name raises ``KeyError``.

Editing ``m.parameters`` changes what ``evaluate`` and ``parameter`` return,
but not the model: bounds, coefficients and right-hand sides were fixed when
the model was built. To change the data the model holds, build it again.

``Model.copy()`` carries ``m.parameters``; the copy is unsolved, so solve it
before evaluating an expression that reads a variable.


Saving and rebuilding
---------------------

``m.parameters`` is ordinary model data, so ``to_netcdf`` and ``read_netcdf``
carry it like any other parameter. Object parameters, such as a ``str``
parameter with a hole, keep their ``NaN`` holes on both the ``netcdf4`` and the
``scipy`` engine; an object parameter mixing types is refused on write.

The restored model is a plain linopy model with its solution and parameters,
so ``evaluate(spec, restored)`` works on it directly. The file holds the data,
not the spec: keep the YAML next to it. To rebuild the model, pass the YAML and
the restored parameters, a ``Dataset`` whose holes are missing rows:

.. code-block:: python

    from linopy import read_netcdf

    m.to_netcdf("dispatch.nc")
    rebuilt = Model.from_spec(
        Path("dispatch.yaml"), read_netcdf("dispatch.nc").parameters
    )


Typesetting
-----------

Typesetting is mathspec's own. ``mathspec.typeset`` renders a whole spec in
``"latex"``, ``"markdown"`` or ``"typst"``, and ``mathspec.typeset_line``
renders one declaration (a named expression, a constraint, a variable's
domain) as a bare line of math without delimiters:

.. code-block:: python

    import mathspec

    spec = mathspec.to_spec(Path("dispatch.yaml"))
    latex = mathspec.typeset(spec, "latex")
    line = mathspec.typeset_line(spec, "balance", "latex")

In Jupyter, render a line with ``IPython.display.Math(line)``. The markdown
format uses the math delimiters GitHub renders, which Jupyter does not.


Variants
--------

A variant of a model is a variant of its spec. ``mathspec.override`` lays
patches over a base spec, and ``mathspec.merge`` composes fragments as peers;
both return a ``mathspec.Spec``, which ``Model.from_spec`` takes as it is:

.. code-block:: python

    CO2_CAP = """
    parameters:
      emission: { dims: [generator] }
      co2_limit: { dims: [] }
    constraints:
      co2_cap:
        dims: []
        expression: sum(p * emission) <= co2_limit
    """
    capped = mathspec.override(Path("dispatch.yaml"), [CO2_CAP])
    m = Model.from_spec(capped, {**sources, "emission": emission, "co2_limit": 100.0})

Pass the same ``capped`` spec to ``evaluate`` and ``parameter`` for that model.
