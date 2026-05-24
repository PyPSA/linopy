# Docs plan — user-facing migration guide

Early-stage outline for the v1 migration docs. Not the guide itself —
the three pieces it needs to cover, written when someone picks it up.

## Three audiences, one migration

- **Downstream library maintainers** (PyPSA, pypsa-eur, calliope, …) —
  carry the bulk of the migration work: opt their codebases into v1,
  fix the raises, ship a release that no longer warns under legacy.
- **Direct users of linopy** — write linopy code themselves and need
  to know what changes for their own call sites.
- **End users of downstream libraries** — never touch linopy directly,
  but may see a `LinopySemanticsWarning` in CI logs and need a
  pointer to "this is upstream; your maintainer will handle it".

## Three things to cover

1. **Why v1 exists.** One paragraph: legacy silently mishandled NaN,
   coord mismatches, and absent variables. The bug catalogue in #714
   has the case-by-case detail.

2. **What's changing and when.** The rollout timeline:
   - v1 ships opt-in via `linopy.options['semantics'] = 'v1'`.
   - v1 becomes the default in a later minor release (date TBD).
   - Legacy removed at 1.0.

3. **How to migrate.** What downstream maintainers do to flip their
   codebase: opt in on a branch, run tests, fix the raises. The
   legacy warning text already names the rule and the fix per site,
   so the guide is mostly the high-level recipe plus a pointer to
   the spec (`arithmetics-design/convention.md`) for the rule list.
