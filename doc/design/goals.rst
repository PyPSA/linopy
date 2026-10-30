The arithmetic convention — design goals
========================================

Goals for linopy's strict arithmetic convention. The bugs that motivate
it are catalogued in `#714 <https://github.com/PyPSA/linopy/issues/714>`__; the convention itself is in
:doc:`convention`.

Design goals
------------

The convention serves four goals, in priority order:

1. **No silent wrong answers.** Every bug in the catalogue (`#714 <https://github.com/PyPSA/linopy/issues/714>`__) returns a
   plausible result with no error. The overriding goal: a mismatch linopy
   cannot resolve unambiguously must raise, not get guessed. Where the library
   cannot decide, the caller does — with an explicit join, ``.sel()``, or
   ``fill_value=``.
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
