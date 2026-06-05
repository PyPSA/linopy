# Legacy removal checklist (for linopy 1.0)

The v1 convention ships alongside legacy from 0.x onward and replaces it
entirely at 1.0. This file enumerates everything to delete when that
release happens, in dependency order. Most edits are mechanical;
`grep "LEGACY: remove at 1.0"` finds every inline marker comment in the
source tree.

## Implementation

### `linopy/config.py`

- Drop `LEGACY_SEMANTICS`, `V1_SEMANTICS`, `VALID_SEMANTICS` constants.
- Drop `LEGACY_SEMANTICS_MESSAGE`.
- Drop the `LinopySemanticsWarning` class.
- Remove the `semantics` key from `options`. The option no longer exists;
  callers don't need to opt in.
- Remove the v1/legacy validation branch in `set_value`.

### `linopy/semantics.py`

- Delete `is_v1()` (always true). Inline `True` at the four import sites
  in `expressions.py`/`variables.py` or, better, delete the import and
  the now-dead `else` branches alongside it.
- Drop the legacy-warn branch from `check_user_nan_scalar` /
  `check_user_nan_array` — both become a single `raise ValueError(...)`.
- Delete `dim_coords_differ`: only the legacy `_align_constant` default
  path uses it.
- `merge_shared_user_coords_differ`, `conflicting_aux_coord`,
  `absorb_absence` stay — they're v1 enforcement helpers.

### `linopy/expressions.py`

- `_add_constant`: delete `_add_constant_legacy`; inline
  `_add_constant_v1` into the now-trivial dispatcher (or rename it to
  `_add_constant`).
- `_apply_constant_op`: same treatment with `_apply_constant_op_legacy`.
- `_align_constant`: delete the `else` branch under `if join is None:`
  that handles the legacy size-aware default (`other.sizes == self.const.sizes`
  positional + `reindex_like` left-join paths). The explicit-join code
  below stays.
- `to_constraint`: drop the `if is_v1(): ... return ...` wrapper and
  keep its body; delete the legacy auto-mask fallthrough that follows
  (the `rhs_nan_mask` plumbing plus the `rhs.reindex_like(...,
  fill_value=np.nan)` pad).
- `LinearExpression.isnull`: drop the legacy `(self.vars == -1).all(...)
  & self.const.isnull()` branch — `self.const.isnull()` is the v1 answer.
- `merge`:
  - Drop the `if differ: warn(...)` line and the `if aux_conflict:
    warn(...)` line — these are the §8 / §11 legacy warns. The raises
    above stay.
  - The `skipna = not is_v1()` simplifies to `skipna = False` (v1's
    propagation rule).
  - The trailing `if is_v1(): ds = absorb_absence(ds)` becomes
    unconditional.
- Drop the `LinopySemanticsWarning` / `LEGACY_SEMANTICS_MESSAGE` imports
  from `expressions.py`.

### `linopy/variables.py`

- `Variable.to_linexpr`: drop the `else` branch (legacy
  `reindex_like(fill_value=0).fillna(0)`); make the v1 `reindex_like(
  fill_value=NaN) → .where(~absent)` path unconditional. The
  `const = NaN`/`0` assign also becomes unconditional.
- Drop the `from linopy.semantics import is_v1` import.

### `linopy/alignment.py`

- `_enforce_implicit_projections`: drop the legacy `warn_legacy(...)`
  branch — the v1 raise for partial-level / coverage-gap projections
  becomes unconditional. The projection machinery itself
  (`_project_onto_multiindex_levels`, `_LevelProjection`) stays:
  full-coverage full-level projections remain legal under v1 (they are
  the same coordinate spelled differently, §8).
- `_dims_for_unlabeled_operand`: drop the legacy positional-pairing
  fallback (the `warn_legacy(...)` branches plus the `return
  list(candidates)`); the v1 size-pairing — the `is_v1()` block that
  raises on ambiguity / no-match — becomes the whole function. The
  `as_constant` / `_pair_axes_by_size` helpers stay (v1-clean).

### `linopy/piecewise.py` / `linopy/sos_reformulation.py`

Nothing to remove; these are v1-clean (the `drop=True` / `assign_coords`
fixes from Slice P are correct for both semantics).

## Tests

### `test/conftest.py`

- Drop the `LEGACY_SEMANTICS`/`V1_SEMANTICS`/`VALID_SEMANTICS` imports
  and the `LinopySemanticsWarning` import.
- Drop the `legacy` / `v1` marker registration in `pytest_configure`.
- Delete the autouse `semantics` fixture entirely (no more parameterization,
  no more warning suppression).

### `test/test_legacy_violations.py`

- Delete the file. Everything in it either documents legacy behaviour
  (gone) or tests v1 raises (covered by the per-module test files we
  add tests to alongside the implementation).

  Before deleting, move any v1 tests that don't have a per-§ home into
  the appropriate module:
  - `TestExactAlignmentConstant`, `TestExactAlignmentMerge`,
    `TestBroadcastNonSharedDim` → `test_linear_expression.py`.
  - `TestConstraintRHS` → `test_constraints.py`.
  - The rest are small enough to fold in alongside related tests.

### `test/test_convention.py`

- Delete the file (it tests the `options["semantics"]` framework, which
  is gone).

### Marker stripping

`grep -rn "@pytest.mark.legacy\|@pytest.mark.v1\|pytestmark = pytest.mark.legacy"
test/` finds every marker:

- `@pytest.mark.legacy` decorators — delete the decorator (the test
  body is documenting old behaviour; deleting the whole test is
  usually right). Spot-check before each delete; a few "legacy"
  marks turned out to gate on legacy auto-mask semantics and the
  test itself stays valid under v1.
- `@pytest.mark.v1` decorators — strip the decorator (the test stays).
- `pytestmark = pytest.mark.legacy` at module level — was only used
  while `piecewise.py` was non-v1-aware; removed in Slice P. Verify
  none remain.

## Documentation

### `arithmetics-design/goals.md`

- Drop the entire "Transitioning goals" section (the three transitioning
  goals + the schedule are about the legacy bridge, which is gone).
- Goal #4's mention of `LinopySemanticsWarning` (if any) goes.

### `arithmetics-design/convention.md`

- Drop the `## Legacy` framing if it exists. Each § already describes
  v1 directly; any "where today this does X" asides referencing legacy
  behaviour can go.
- Update the intro: "The strict ('v1') convention" → "The arithmetic
  convention" (drops the v1 framing now that there's only one).

### This file (`legacy-removal.md`)

- Delete after the 1.0 release ships.

## Order of operations

A safe sequence (each step compiles and tests pass):

1. Delete legacy test infrastructure (`test/conftest.py` fixture,
   `test/test_convention.py`, `test/test_legacy_violations.py` after
   moving v1 tests out).
2. Strip `@pytest.mark.legacy` decorators (the tests fail under v1
   anyway once the legacy paths are gone — delete or update each).
3. Delete legacy implementation branches in `expressions.py` /
   `variables.py`.
4. Delete `semantics.py` legacy bits (`is_v1`, the warn branches in
   `check_user_nan_*`, `dim_coords_differ`).
5. Delete `config.py` symbols (`LEGACY_SEMANTICS`, the warning class,
   the option key).
6. Update `arithmetics-design/goals.md` and `convention.md`.
7. Delete this file.
