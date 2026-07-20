# Open items for the v1 convention

The convention itself is **fully specified and implemented**:
[`convention.md`](convention.md) §1–§13 plus the object-scope and
coordinate-alignment intro rules all have a v1 implementation. The high-level
schedule lives in [`goals.md`](goals.md) (opt-in → default → 1.0); this file
tracks the concrete items per stage.

The one open **design** decision — [#744] MultiIndex storage — is **resolved**:
v1 disallows first-class `pd.MultiIndex` (a flat dim + auxiliary level coords),
implemented in [#803]. §8 alignment was subsequently tightened to full
`join="exact"` — a pure reorder raises rather than reindexing ([#831]) — and
grouper alignment made strict to match ([#827]). Everything left is rollout +
cleanup.

## Stage 1 — release v1 (opt-in)

Legacy stays the default. The transition surface must already be **complete**
here ([`goals.md`] step 1: *warn on legacy, raise on v1*).

- [x] **Resolve [#744] — MultiIndex storage — before #717 merges.** Resolved:
  v1 disallows first-class `pd.MultiIndex` (flat dim + auxiliary level coords).
  §11's stacked-MultiIndex rule and the storage of a `snapshot`-style
  `(period, timestep)` dim both depend on this, and §11 **ships in #717** — so
  the implementation ([#803]) must land in the **same release cut** as #717,
  else released §11 behaviour would change when it lands.
- [ ] Land [#717] (v1 semantics) → `master`
- [ ] Land [#803] (v1 MultiIndex drop) with it — same cut, so §11 ships final
- [ ] **Transition surface complete** — every behaviour-change site raises under
  v1 *and* warns under legacy (`warn_legacy`, naming the fix). This is the
  "no silent change" guarantee ([`goals.md`] transitioning goal #3): shipping v1
  with a gap would silently change any model that opts in. Includes #803's
  MultiIndex rejects and the multi-key-`groupby`-flat change as new fork sites.
  The reorder→raise change ([#831]) added its own fork sites: reordered constant
  operands now warn under legacy (they were silently reindexed) — done for
  `+`/`-`/`*`/`/`/rhs. A full audit (probe every §'s legacy→v1 divergence)
  found one remaining silent site: `groupby([names]).sum(observed=True)` minted
  the `group` MultiIndex without warning (only DataFrame groupers warned) — now
  fixed. No other silent fork site found.
- [x] **Land grouper strict alignment on the branch** — [#830]'s
  check-and-raise on a reordered/mismatched grouper is merged in. The
  **Groupers** rule in [`convention.md`] §13 documents the behaviour.
- [x] Changelog note — v1 opt-in via `options['semantics'] = 'v1'` (legacy
  remains the default), with the strict-alignment / absence / user-NaN /
  aux-coord / MultiIndex summary and a link to [`convention.md`].

## Stage 2 — make v1 the default (legacy opt-out)

- [x] Write the **migration guide** — `doc/migrating-to-v1.rst` (why v1, the
  opt-in→default→1.0 rollout, the three audiences, the surface-warnings-then-fix
  recipe, and a situation→v1→fix table). [`docs-plan.md`] was the outline.
- [ ] Flip the `options['semantics']` default to v1.
- [ ] Close [#714] once v1 ships as default.

## Stage 3 — linopy 1.0, remove legacy

- [ ] The **strip**: the concentrated MI machinery (the `alignment.py`
  level-projection subsystem, netcdf MI (de)serialize) *and* the scattered
  surface (`assign_multiindex_safe` ×39, `isinstance(MultiIndex)` guards, MI
  branches) — all kept live for legacy until now — plus every other
  `grep "LEGACY: remove at 1.0"` marker. Dependency-ordered checklist in
  [`legacy-removal.md`].
- [ ] Reframe [`convention.md`] / [`goals.md`] — drop the "v1"/legacy framing
  once there is only one convention. At this point [`convention.md`] also
  **graduates into the rendered Sphinx docs** (add `myst-parser`, into the
  toctree, internal `:doc:` cross-refs replacing the GitHub blob links in
  `doc/migrating-to-v1.rst` and `doc/release_notes.rst`). Until then it stays a
  design-record linked by URL — the rules are permanent but the transitional
  framing would only be re-done, so rendering now is wasted effort.

## Not blocking (follow-ups)

- [ ] Test cleanup — public-API assertions (`.indexes` / `.coords` / `.sizes` /
  `.coord_dims`) instead of internal `.data` / `.coeffs.coords`; assert the
  **full** error/warning text in v1-raise and legacy-warn tests; de-dup repeated
  index/setup fixtures.
- [x] Pin the **legacy** side of every v1/legacy result divergence.

## Decisions

- [x] **[#744] — MultiIndex storage** → v1 disallows MI (flat + aux coords),
  implemented in [#803]; must land in the same release cut as #717 (§11 depends
  on it).
- [x] **Legacy warnings live from the opt-in release** → yes; the transition
  surface is complete at stage 1, not deferred.
- [x] **§8 is full `join="exact"`** → a pure reorder (same labels, different
  order) raises, not reindexes ([#831]); the reindexing named joins still
  resolve it. Groupers align strictly the same way ([#827]/[#830]).
- [ ] **When v1 becomes the default** — pick the release; gated on the migration
  guide.

<!-- references -->
[#744]: https://github.com/PyPSA/linopy/issues/744
[#827]: https://github.com/PyPSA/linopy/issues/827
[#830]: https://github.com/PyPSA/linopy/pull/830
[#831]: https://github.com/PyPSA/linopy/pull/831
[#714]: https://github.com/PyPSA/linopy/issues/714
[#717]: https://github.com/PyPSA/linopy/pull/717
[#803]: https://github.com/PyPSA/linopy/pull/803
[`goals.md`]: goals.md
[`docs-plan.md`]: docs-plan.md
[`legacy-removal.md`]: legacy-removal.md
[`convention.md`]: convention.md
