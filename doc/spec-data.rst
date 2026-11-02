.. _spec-data:

Spec data and results
=====================

:doc:`building-models-from-specs` walks through a spec from YAML to a solved
model. This page is the reference for the parts around it: how linopy reads the
spec and its data, what a missing row means, how to store the data on the
model, and how to read results, rebuild, typeset and vary a model.

A spec-built model is a plain linopy model. Everything the spec adds beyond
the variables, constraints and objective is read as a function of the spec,
the model and the data:

.. code-block:: python

    from pathlib import Path

    from linopy import Model
    from linopy.spec import evaluate

    spec = Path("dispatch.yaml")
    m = Model.from_spec(spec, sources)
    m.solve()

    evaluate(
        spec, m, sources=sources
    )  # every named expression, folded over the solution
    evaluate(spec, m, "spend", sources=sources)["spend"]  # one of them

Both need ``pip install "linopy[spec]"`` (Python >= 3.12).

.. contents::
   :local:
   :depth: 1


Passing the spec
----------------

``Model.from_spec``, :func:`linopy.spec.evaluate` and
``linopy.testing.synthetic_sources`` take the spec in any of four forms:

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
declares, or an ``xarray.Dataset`` laid out the way ``store=True`` writes
``m.parameters``: its indexes are dimension sources, its data variables
parameters, and a relation of one key and one value column its value column
over the key's dimension.

Each kind of data takes exactly one shape for tables and one for grids, and
anything else is refused with ``SpecDataError`` naming the shape to pass,
rather than guessed at. A **table** is a ``pandas.DataFrame`` or a
``polars.DataFrame`` that carries its labels in columns, never in its index,
and has no column beyond those it is read for. A **grid** is an
``xarray.DataArray`` over the declared dimensions.

linopy reads the keys once, off ``keys()``, and never iterates the mapping
beyond that. Each declared name is then read with one ``__getitem__``. A mapping
can therefore compute its values on access, from a database or a file, as in
the :ref:`SQLite example <spec-data-sqlite>` below.

**Dimensions.** A dimension's members come only from the source keyed by its
name, in the order the source gives them; linopy never sorts them. A dimension
source is a ``pandas.Index``, a list, a tuple, a 1-D numpy or pandas array
(including extension arrays such as a ``Categorical`` or ``Int64``), or a
``pandas.Series`` or ``polars.Series``. A ``set`` has no order and is refused.
Each label appears once: a repeated label is refused, never deduplicated. To
take a dimension's labels from a table column, pass its unique values, which
keeps the order of first appearance:

.. code-block:: python

    sources["bus"] = generators["bus"].unique()  # pandas
    sources["bus"] = generators["bus"].unique(maintain_order=True)  # polars

A dimension is never derived from a relation or a parameter: without a source
under its name, it is refused.

**Parameters.** A parameter source is read for values, aligned onto the
dimension members. A parameter over dimensions is

- a table with exactly one column per dimension and the values in a column
  named after the parameter, or
- a ``DataArray`` over the declared dimensions, in any order.

A parameter over no dimension is one number (or a 0-d ``DataArray``).

.. code-block:: python

    # cap: {dims: [generator]}
    sources["cap"] = generators[["generator", "cap"]]
    # cost: {dims: [snapshot, generator]}
    sources["cost"] = pl.DataFrame(
        {"snapshot": [0, 0], "generator": ["g1", "g2"], "cost": [10.0, 20.0]}
    )

A ``Series``, a ``dict``, a frame with its labels in the index or with a
further column is refused; pass ``frame.reset_index()`` or select the
columns.

**Relations.** A relation is a table of rows with one column per column its
entry declares, key and value columns alike, each named after it. Its source is
a table with exactly those columns. A bare relation (key columns only), several
key columns and several value columns all arrive this way:

.. code-block:: python

    # cover: {key: {start: t, covered: t}, missing: absent}
    sources["cover"] = pd.DataFrame({"start": [0, 0, 1], "covered": [0, 1, 1]})
    # ends: {key: line, values: {bus0: bus, bus1: bus}}
    sources["ends"] = pd.DataFrame({"line": ["l0"], "bus0": ["b0"], "bus1": ["b1"]})

Each key tuple takes one row at most, a null in any column is refused, and every
label must be a member of its column's dimension. Under ``missing: refused``
(the default) every key tuple needs a row.

linopy holds a relation as its rows and never lays it out as a grid over its
key, so a bare relation over two columns of one long dimension stays as small
as its data. A ``sum(..., by=...)`` through it joins and groups row by row,
with mathspec's semantics: an unnamed key column is joined on and kept, and
``by=`` may name a list of columns. ``within=`` partitions by one value column
of a relation keyed by the stepped dimension alone.

.. note::

   A parameter, unlike a relation, is laid out on the full grid of its
   dimensions once a declaration reads it, so its memory follows the grid, not
   the rows of its table; ``store=True`` stores that grid. Reduce sparse data on
   a large grid before attaching it, e.g. pass ``sum(c, over=t)`` as its own
   parameter, summed with pandas.

Missing rows
~~~~~~~~~~~~

A missing row is a member with no value. How it is spelt depends on the shape:

- In a **table**, a missing row is a row that is not there. A ``NaN`` in a
  table is refused: the row says the coordinate exists and denies it in the
  same breath. Drop the row instead.
- In a **grid** (a ``DataArray`` or a ``Dataset`` data variable), every
  coordinate has a cell, so a ``NaN`` (``NaT`` for a datetime, ``NaN`` in an
  object array for a string) is a missing row.

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
            return frame.iloc[:, 0] if frame.shape[1] == 1 else frame

        def __iter__(self) -> Iterator[str]:
            return iter(self.QUERIES)

        def __len__(self) -> int:
            return len(self.QUERIES)


    m = Model.from_spec(Path("dispatch.yaml"), Tables(sqlite3.connect("fleet.db")))

Dimensions come back as one column, parameters as tables with the values in a
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
- linopy carries a product of two variables in the objective only. A
  constraint whose spec multiplies two variables builds wherever the data
  leaves no such product on a row its ``where:`` keeps: its coefficient is
  zero, or it is an absent term left out of a sum. The row then holds its
  linear terms only. A product the data keeps raises ``NotImplementedError`` naming
  the constraint and its first such rows.


Holes that operators create
---------------------------

Data the sources hold in full can still leave a hole in an expression: an
operator that reads along a dimension, or through a relation, finds nothing at
some rows. Such a hole is absence, and it behaves as above: outside a sum it
takes the row with it, also beside other terms, and inside a sum it is a term
left out. The model builds without an error or a warning, so a spec states
what each edge means. Take a ramp over an ordered dimension ``s``:

.. code-block:: yaml

    constraints:
      ramp:
        dims: [s]
        expression: p <= shift(p, along=s, offset=1)

``shift(p, along=s, offset=1)`` reads nothing at the first ``s``, so ``ramp``
has no row there; ``offset=2`` vacates the first two rows, ``offset=-1`` the
last one. ``p + q <= shift(p, along=s, offset=1)`` loses the same row: the
present ``p + q`` does not keep it. Pick what the edge means:

- ``edge=0``, or any number, reads that number at the vacated rows, so the row
  stays: ``shift(p, along=s, offset=1, edge=0)`` gives ``p[0] <= 0`` at the
  first ``s``.
- ``edge='wrap'`` reads from the other end, as on a cyclic horizon. With
  ``within=rel[col]`` each group of a relation closes on itself.
- ``where: position(s) > 0`` on the constraint keeps the drop and says so in
  the spec. The rows built are the same as without it. For ``offset=2`` write
  ``position(s) >= 2``, for ``offset=-1`` ``position(s) != -1``.

The other operators read past an edge in their own way:

- ``sum_back(p, along=s, window=3)`` is a sum, so it shortens its window at the
  start rather than dropping the row: the first ``s`` reads ``p[0]`` alone.
  ``edge='wrap'`` reaches back into the end, ``within=rel[col]`` stops at each
  group's edge.
- ``at(p, by=rel[col])`` reads nothing for a member the relation maps nowhere,
  and the row goes with it. ``where: rel`` on the constraint names the rows it
  keeps.
- Inside a sum a hole is a term left out, so
  ``sum(shift(q, along=s, offset=1), over=s)`` builds every row.

A row the data leaves present but without a variable term is the other case,
refused as described above. To see which rows a built constraint lacks, read
``m.constraints["ramp"].labels == -1``: a row absence dropped or a ``where:``
masked has no label.

Storing the data in ``m.parameters``
------------------------------------

``Model.from_spec`` reads the data straight from ``sources`` and, by default,
leaves ``m.parameters`` untouched. With ``store=True`` it writes the spec's
parameters and relations into ``m.parameters`` under their declared names, on
the dimension members it attached. Nothing else of the spec is kept on the
model.

The stored parameters are raw: a missing row is a ``NaN`` hole, and an ``int``
or ``bool`` parameter with a hole is ``float64``. A relation is an array over
its key dimension, ``NaN`` where a label is unmapped, so only a relation of one
key column and one value column is stored. With ``store=True``, a bare relation
or one with several key or value columns raises ``SpecDataError`` before
anything is written; build with ``store=False`` and keep the sources, to
rebuild from and to pass as ``evaluate(spec, m, sources=...)``.

Editing ``m.parameters`` does not change the model: bounds, coefficients and
right-hand sides were fixed when the model was built. ``evaluate(spec, m)``
reads the edit at once, over the solution of the unedited model; the model
itself takes it only when rebuilt from the stored parameters:

.. code-block:: python

    m = Model.from_spec(spec, sources, store=True)
    m.parameters["cost"].loc["wind"] = 10.0
    m = Model.from_spec(spec, m.parameters, store=True)

A key in ``m.parameters`` that the spec does not declare, such as data added by
hand, is ignored on a rebuild and by ``evaluate``.

``Model.copy()`` carries ``m.parameters``; the copy is unsolved, so solve it
before evaluating an expression that reads a variable.


Reading results
---------------

:func:`linopy.spec.evaluate` folds named expressions over ``m.solution`` and
the duals. It reads the data from ``sources=`` or, when none is given, from
``m.parameters`` as ``store=True`` wrote it; with neither, it raises
``KeyError``. The data goes through the same checks and ``missing:`` rules as
in the build. A body over data alone needs no solve; one that reads a variable
before a solve raises ``AttributeError``, one that reads a ``refused`` hole
``SpecDataError``. An unknown name raises ``KeyError``.


Saving and rebuilding
---------------------

``m.parameters`` is ordinary model data, so ``to_netcdf`` and ``read_netcdf``
carry the data a model was built with ``store=True`` like any other
parameter. Object parameters, such as a ``str``
parameter with a hole, keep their ``NaN`` holes on both the ``netcdf4`` and the
``scipy`` engine; an object parameter mixing types is refused on write.

The restored model is a plain linopy model with its solution and parameters,
so ``evaluate(spec, restored)`` works on it directly if it was built with
``store=True``; otherwise pass the data as ``evaluate(spec, restored, sources=...)``. The file holds the data,
not the spec: keep the YAML next to it. To rebuild the model, pass the YAML and
the restored parameters, a ``Dataset`` whose holes are missing rows, and store
them again:

.. code-block:: python

    from linopy import read_netcdf

    m.to_netcdf("dispatch.nc")
    rebuilt = Model.from_spec(
        Path("dispatch.yaml"), read_netcdf("dispatch.nc").parameters, store=True
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

Pass the same ``capped`` spec to ``evaluate`` for that model.
