# The v1 convention — design & transitioning goals

Goals for linopy's strict ("v1") convention. The bugs that motivate
it are catalogued in [#714]; the convention itself is in
[`convention.md`](convention.md).

## Design goals

The convention serves four goals, in priority order:

1. **No silent wrong answers.** Every bug in the catalogue ([#714]) returns a
   plausible result with no error. The overriding goal: a mismatch linopy
   cannot resolve unambiguously must raise, not get guessed. Where the library
   cannot decide, the caller does — with an explicit join, `.sel()`, or
   `fill_value=`.
2. **Preserve the algebraic laws.** Commutativity, associativity,
   distributivity, the identities. Optimization code builds expressions by
   rearranging terms, and the convention must keep that safe.
3. **Absence is first-class.** A variable can be genuinely absent at a slot —
   masked out, or shifted past the edge. The data model needs an explicit
   marker for that absence, kept distinct from a zero term, so absent-vs-zero
   is never a silent guess.
4. **Least surprise.** linopy is built on xarray and its users know xarray. The
   convention should behave the way xarray already taught them — align by
   label, broadcast non-shared dimensions, resolve mismatches with a named
   join — not invent linopy-specific rules. Auxiliary coordinates the user
   attached are the user's; linopy validates and carries them through,
   never silently dropped or rewritten.

## Transitioning goals

1. **Non-breaking.** Existing code keeps working — legacy stays available and
   unchanged until it is removed at linopy 1.0.
2. **Actionable warnings.** Warn every legacy user about behaviour changes —
   what changes under v1, and how to fix it — aiming for 100% coverage.
3. **No silent change.** Opting into v1 never silently changes a model — every
   difference is either raised, or was warned about in legacy mode.

**Schedule:**

1. Introduce v1 as opt-in — warn about behaviour changes on legacy, raise if
   opted into v1.
2. Make v1 the default, allow opt-out.
3. linopy 1.0 — drop the legacy convention entirely.

<!-- references -->
[#714]: https://github.com/PyPSA/linopy/issues/714
