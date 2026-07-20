Migrating to the v1 arithmetic convention
=========================================

.. note::

   v1 is **opt-in** in this release and legacy remains the default. Nothing
   changes until you set ``linopy.options["semantics"] = "v1"``. This guide is
   for deciding *when* to opt in and *what* to change when you do.

Why v1 exists
-------------

Legacy linopy silently guessed in three situations where a guess can quietly
change a model:

- a ``NaN`` in user-supplied data was filled — with a value that differed by
  operator (``0`` for ``+``/``*``, ``1`` for ``/``);
- a coordinate mismatch on a shared dimension was resolved by position or an
  inner join, which can drop terms and constraints without notice;
- an absent (masked, reindexed, or shifted-in) variable was treated as a
  variable fixed to zero.

The v1 convention replaces every silent guess with an explicit rule: it aligns
strictly by label and *raises* where legacy would have guessed, so a wrong model
surfaces as an error at build time instead of a wrong number at solve time. The
per-case bug catalogue is in `issue #714 <https://github.com/PyPSA/linopy/issues/714>`_.

The rollout
-----------

The transition happens over three steps so that no model changes behaviour
without warning first:

#. **Now — opt-in.** v1 is available via ``linopy.options["semantics"] = "v1"``.
   Legacy is the default. Under legacy, every operation whose result *would*
   change under v1 emits a :class:`linopy.LinopySemanticsWarning` that names the
   rule and the fix.
#. **A later minor release — default.** v1 becomes the default; legacy stays
   reachable via ``options["semantics"] = "legacy"`` for one more cycle.
#. **linopy 1.0 — legacy removed.** Only v1 remains.

Who this affects
----------------

- **Downstream library maintainers** (PyPSA, PyPSA-Eur, Calliope, …) carry most
  of the work: opt the library into v1, fix the raises, and release so it no
  longer warns under legacy.
- **Direct linopy users** do the same for their own call sites.
- **End users of a downstream library** never touch linopy directly. A
  ``LinopySemanticsWarning`` in your logs is an upstream item your maintainer
  will handle; you can :ref:`silence it <silencing-v1>` in the meantime.

How to migrate
--------------

The legacy warning already names the rule and the fix at every site, so the
recipe is short:

#. **Surface every site.** Run your existing test suite under legacy and turn
   the warning into an error so nothing is missed:

   .. code-block:: python

       import warnings
       from linopy import LinopySemanticsWarning

       warnings.filterwarnings("error", category=LinopySemanticsWarning)

#. **Fix each site** using the table below (the warning text points at the same
   fix in context).
#. **Opt in and run the suite** with ``linopy.options["semantics"] = "v1"``. The
   v1 raises catch anything the warnings missed.
#. **Release.** Set the option in your library's entry point, or simply drop it
   once the default flips to v1.

What changes, and how to fix it
-------------------------------

Every row is a legacy guess that becomes an explicit rule under v1. The
``LinopySemanticsWarning`` you see under legacy names the same fix.

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Situation
     - Under v1
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
   * - **Conflicting auxiliary coordinates** on a shared dim
     - Raises instead of silently dropping one.
     - ``.drop_vars(name)`` to remove the coord, or ``.assign_coords(name=...)``
       to relabel one side.
   * - A first-class ``pd.MultiIndex`` **dimension** — and a per-*level* input
       onto it (e.g. per-``period`` bounds onto a ``(period, timestep)``
       ``snapshot``)
     - Rejected; v1 uses a flat dimension with the levels as aux coords, and a
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
`arithmetic convention <https://github.com/PyPSA/linopy/blob/master/arithmetics-design/convention.md>`_.

.. _silencing-v1:

Silencing the warning
---------------------

If the warning comes from a library you do not maintain, silence it until the
upstream release lands:

.. code-block:: python

    import warnings
    import linopy

    warnings.filterwarnings("ignore", category=linopy.LinopySemanticsWarning)

Do this only when the warning is upstream — for your own call sites, fix it
instead, since the underlying result changes under v1.
