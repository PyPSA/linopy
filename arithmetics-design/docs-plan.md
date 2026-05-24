# Docs plan — user-facing migration guide

This file collects bullet points for a later user-facing docs migration
guide (deferred from the v1 implementation PR). Not the guide itself —
a punch list for whoever writes it. Add to it as items come up.

## Audience and shape

Two audiences worth distinguishing:

- **Downstream library maintainers** (PyPSA, pypsa-eur, calliope, …) —
  need exhaustive coverage of every rule change, with examples drawn
  from their patterns.
- **End users of those libraries** — usually never see linopy directly;
  may hit a `LinopySemanticsWarning` in CI logs and need a one-page
  "what does this mean and what do I do" reference.

A short page for end-users (linked from each warning's docstring) plus
a longer section in the developer docs is probably the right split.

## Items to cover (rough bullets)

### Why v1

- One-paragraph summary: legacy silently mishandled NaN, mismatched
  coords, and absent variables, producing wrong answers without errors.
  The v1 convention closes those holes.
- Link the bug catalogue (#714) and the convention spec
  (`arithmetics-design/convention.md`).

### Timeline

- Legacy stays the default through the 0.x series.
- v1 is opt-in via `linopy.options['semantics'] = 'v1'`.
- v1 becomes the default in a future minor release (TBD).
- Legacy is removed at 1.0 — see `legacy-removal.md` for the
  maintainer-side checklist.

### What changes (the rule-by-rule cheat sheet)

One row per rule, three columns: "the operation", "legacy behaviour",
"v1 behaviour + how to migrate".

- §5 NaN in a user constant: legacy silently fills (0 for +/-/*, 1 for
  /); v1 raises. Migrate with `.fillna(value)` or by marking absence
  on the variable.
- §6 absent variable in arithmetic: legacy contributes 0; v1
  propagates absence. Migrate with `var.fillna(0)` to keep legacy
  behaviour.
- §8 coord mismatch on shared dim: legacy aligns by position when
  sizes match, otherwise left-joins; v1 raises. Migrate with `.sel`,
  `.reindex`, `.assign_coords`, `linopy.align`, or an explicit
  `join=` argument.
- §11 aux-coord conflict: legacy silently drops; v1 raises. Migrate
  with `.drop_vars`, `.assign_coords`, or `.isel(..., drop=True)`.
- §12 NaN in constraint RHS: legacy treats as "no constraint at this
  row"; v1 raises. Use `mask=` on the variable for explicit per-row
  masking instead.

Reference: the legacy warning text on each of these names the rule and
the fix — users who see the warning should be able to migrate without
opening this guide.

### How to migrate a codebase

- Opt in to v1 on a branch, run tests, fix raises one by one.
- Before opting in, run legacy with warnings-as-errors to surface every
  call site that will change under v1
  (`pytest -W error::LinopySemanticsWarning`).
- For PyPSA-style frameworks: search for `mask=` and `.fillna(...)`
  patterns, those are the most common touchpoints.

### Known limitations

- **Warning source-frame attribution.** On Python 3.12+ the warning's
  source frame points at the user's exact call (via stdlib
  `skip_file_prefixes`). On Python 3.11 it falls back to a static
  stacklevel that's correct for the most common case (`expr + var`
  merge chain) but may point one frame too far on shorter chains
  (`var.fillna(0)`). The warning *text* is identical on both versions
  — only the source-frame is approximate on 3.11.

### Related issues worth referencing

- #295 — aux-coord conflicts silently dropped (now §11).
- #586 / #550 / #708 — coord alignment by position (now §8).
- #627 — open question: should user NaN be read as absence? (Locked to
  "raise" for v1; flagged in the §5 section.)
- #707 — algebraic equivalence of `x - a <= 0` and `x <= a` (now §12).
- #711 — subset-constant associativity (now §8 + §6).
- #712 — absent-as-zero (now §6 / §1).
- #713 — silent NaN-fill (now §5).
- PyPSA #1683 — `0 * inf = NaN` constraint bounds; v1 surfaces this at
  construction.

### Things to actively defer / not mention

- The dead-term invariant (§2 storage rule) — internal to linopy, not
  user-facing.
- The `_v1` / `_legacy` method split in `expressions.py` — implementation
  detail.

## Items added after this file was written

(append here as new items come up during the rollout)
