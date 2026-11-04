# Utilities

`options` holds these keys:

- `display_max_rows`, `display_max_terms`: size of the printed repr.
- `warn_on_densify`: the conversion notice, a `PerformanceWarning` when
    a sparse (CSR) backing is converted to dense or a dense operand is
    converted to sparse for a merge with a sparse one. `None` (default)
    warns in a sparse model, `True` always warns, `False` never warns.
    Only implicit conversions warn; `Constraint.to_dense()` and
    `Constraint.to_sparse()` never do. A pydata/sparse `@` operand that
    has to be densified warns under `None` in a dense model too; only
    `False` silences it. See [Dense and sparse models](../user-guide/sparse-models.ipynb).

::: linopy.align
    options:
      heading_level: 2

::: linopy.options
    options:
      heading_level: 2

::: linopy.ABSENT
    options:
      heading_level: 2

::: linopy.from_long
    options:
      heading_level: 2

::: linopy.Relation
    options:
      heading_level: 2
      members: false

::: linopy.Relation.group_of
    options:
      heading_level: 2

::: linopy.Relation.members
    options:
      heading_level: 2

::: linopy.Relation.groups
    options:
      heading_level: 2
