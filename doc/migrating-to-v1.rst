Upgrading to linopy 1.0
=======================

.. note::

   linopy 1.0 removes the legacy arithmetic semantics. The strict "v1"
   convention is the only behaviour and there is no opt-out. This guide tells
   you *what* changes and how to fix it.

Why the convention changed
--------------------------

Legacy linopy silently guessed in three situations where a guess can quietly
change a model:

- a ``NaN`` in user-supplied data was filled — with a value that differed by
  operator (``0`` for ``+``/``*``, ``1`` for ``/``);
- a coordinate mismatch on a shared dimension was resolved by position or an
  inner join, which can drop terms and constraints without notice;
- an absent (masked, reindexed, or shifted-in) variable was treated as a
  variable fixed to zero.

The convention now replaces every silent guess with an explicit rule: it aligns
strictly by label and *raises* where legacy would have guessed, so a wrong model
surfaces as an error at build time instead of a wrong number at solve time. The
per-case bug catalogue is in `issue #714 <https://github.com/PyPSA/linopy/issues/714>`_.

linopy 1.0 removed legacy directly, without a release that made v1 the default.
``linopy.options["semantics"]`` is a deprecated no-op: ``"v1"`` emits a
``FutureWarning``, any other value raises, and the option is removed in 1.1.
:class:`linopy.LinopySemanticsWarning` is no longer emitted.

Who this affects
----------------

- **Downstream library maintainers** (PyPSA, PyPSA-Eur, Calliope, …) carry most
  of the work: fix the raises and release.
- **Direct linopy users** do the same for their own call sites.
- **End users of a downstream library** never touch linopy directly; an error
  from linopy after the upgrade is an upstream item for your maintainer.

How to upgrade
--------------

#. **Run your test suite** on linopy 1.0. Every difference from legacy either
   raises or changes a result as described in the table below.
#. **Fix each site** using the table.
#. **Drop** any ``linopy.options["semantics"] = "v1"`` line.

What changes, and how to fix it
-------------------------------

Every row is a legacy guess that is now an explicit rule.

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Situation
     - Now
     - Fix
   * - ``NaN`` in a user-supplied constant
     - Raises (linopy will not guess a fill).
     - ``.fillna(value)`` for a data error; ``mask=`` / ``.where(cond)`` /
       ``.reindex(...)`` on the *variable* for intended absence.
   * - Shared dimension with a **different label set**
     - Raises instead of an inner/positional join.
     - ``.sel`` / ``.reindex`` to a common index, ``.assign_coords`` to relabel,
       or an explicit ``join=`` on ``.add`` / ``.sub`` / ``.mul`` / ``.div`` /
       ``.le`` / ``.ge`` / ``.eq``.
   * - Shared dimension with the **same labels in a different order**
     - Raises (``join="exact"`` — no silent reindex).
     - ``.sortby(dim)`` / ``.reindex`` one side to match, or a reindexing
       ``join=`` (``"outer"`` / ``"inner"`` / ``"left"`` / ``"right"``).
   * - A reindexing ``join=`` that **creates labels** neither operand had
     - The join still returns the union of the coordinates; the created rows
       are filled per side — the linopy operand contributes the zero
       expression, the constant operand ``fill_value=`` (0 by default). A
       missing divisor now zeroes that row instead of leaving the term
       unscaled, and a missing numerator no longer comes out as
       ``1 / divisor``.
     - Pass ``fill_value=`` on ``.add`` / ``.sub`` / ``.mul`` / ``.div`` to
       choose the value, e.g. ``.div(other, join="outer", fill_value=1)`` to
       keep the term unscaled, or ``fill_value=linopy.ABSENT`` to keep the
       created rows absent so the constraint drops there.
   * - An **unlabeled** operand (numpy array, list, polars ``Series``)
     - Pairs with the linopy operand's dimensions by *size*; an ambiguous
       match (a square array, or two dimensions of equal length) or no size
       match raises rather than guessing.
     - Wrap it in a ``DataArray`` / ``Series`` / ``DataFrame`` with named
       dimensions so it aligns by label.
   * - A **masked / absent** variable in arithmetic
     - Absence propagates (the slot stays absent) instead of counting as ``0``.
     - Decide the intent: ``.fillna(0)`` to keep the old "treat as zero", or
       leave it to propagate and drop the term.
   * - **Conflicting auxiliary coordinates** on a shared dim — including a
       leftover *scalar* coord from positional indexing (``x.isel(time=0)``
       keeps ``time`` as a scalar coord, so a cyclic/boundary constraint like
       ``x.isel(time=0) == x.isel(time=-1)`` sees ``time`` = first vs last)
     - Raises instead of silently dropping one.
     - ``.drop_vars(name)`` / ``.assign_coords(name=...)``; for a leftover index
       coord, drop it at the source with ``.isel(..., drop=True)`` /
       ``.sel(..., drop=True)``.
   * - A first-class ``pd.MultiIndex`` **dimension** — and a per-*level* input
       onto it (e.g. per-``period`` bounds onto a ``(period, timestep)``
       ``snapshot``)
     - Rejected; linopy uses a flat dimension with the levels as aux coords, and a
       per-level input must be mapped onto the flat dimension explicitly (no
       implicit projection). Affects PyPSA multi-investment models.
     - ``.reset_index(dim)`` to flatten, then project the per-level input by
       its level aux coord.
   * - A multi-key ``groupby(...).sum()`` result
     - A flat ``group`` dimension with the keys as aux coords (not a stacked
       ``group`` MultiIndex).
     - Select on the key aux coords; convert an existing result with
       ``.reset_index("group")``.
   * - A ``groupby`` grouper (``Series`` / ``DataArray`` / ``DataFrame`` /
       coord name) with a differing label set or order
     - Raises — the grouper aligns to the grouped dimension by label, never
       by position.
     - ``.sortby`` / ``.reindex`` the grouper to the dimension's labels.

The full, normative rule list lives in the
:doc:`arithmetic convention <design/convention>`.
