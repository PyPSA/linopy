# MultiIndex feasibility for v1 (#744)

> Verification note (Claude Code, prompted by @FBumann), 2026-06-29. Tracks
> whether linopy can drop **first-class `pd.MultiIndex`** support in v1 in favour
> of a **flat dim + auxiliary level coords** model. Discussion home:
> [#744](https://github.com/PyPSA/linopy/issues/744). Equality checks are tracked
> in [`test/test_mi_feasibility.py`](../test/test_mi_feasibility.py) (runs under
> both `legacy` and `v1`).

**Question.** Can v1 drop the stacked `pd.MultiIndex` snapshot for a flat
`snapshot` dim carrying `period`/`timestep` as auxiliary level coords?

**"Feasible"** has a precise, testable meaning: every real MI use case has a
flat+aux form that builds an *equivalent model* — proven by an explicit equality
check, not asserted.

## Data model under test

| | today (C) | proposed (flat+aux) |
|---|---|---|
| snapshot dim | `MultiIndex[(period, timestep)]` | flat `snapshot` dim |
| level identity | MI levels | `period`/`timestep` **aux coords** on `snapshot` |
| entry conversion | — | `obj.reset_index("snapshot")` (canonical xarray, no custom logic) |
| alignment | tuple-identity | **positional** (one canonical snapshot order) |

The entry conversion is byte-identical across linopy's supported xarray range
(2024.2.0 floor → 2026.4.0); it post-dates xarray's explicit-indexes refactor
(~2022.06, below the floor), so no compat shim is needed.

## Feasibility matrix

Two axes, deliberately separate:
- **feasible** — can flat+aux build an *equivalent model*? ✅ verified (has a test) · ☐ open.
- **desirable** — is the flat+aux form at least as good, across three facets — *nicer* (ergonomics), *safer* (catches errors), *flexible* (coords stay manipulable)? 👍 better · ➖ parity · ⚠️ friction/cost · ☐ TBD.

The `entry`–`model` rows are tracked in `test/test_mi_feasibility.py`; PyPSA links
are pinned at **v1.2.4** (commit [`fb425cb`](https://github.com/PyPSA/PyPSA/tree/v1.2.4)).

| op | MI form | flat+aux form | feasible | desirable | PyPSA call site @ v1.2.4 |
|---|---|---|---|---|---|
| **entry** | `coords=[mi]` | `reset_index(dim)` | ✅ | 👍 deletes MI machinery | — |
| **select** | `sel(snapshot=(p, slice))` | `where(period == p)` | ✅ | ➖ parity | [`constraints.py` L1235‑1248](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1235-L1248) (KVL per-period) |
| **roll** | per-period `sel`-loop | `groupby("period").roll` | ✅ (#751) | 👍 deletes `.data` loop | [`constraints.py` L1694](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1694) (SOC) |
| **group** | level groupby | `groupby("period").sum()` | ✅ (#751) | 👍 flat works (#751); MI level-groupby broken upstream ([xarray#6836](https://github.com/pydata/xarray/issues/6836)) | — |
| **model** | MI per-period LP | flat+aux LP (solved `==`) | ✅ | ➖ parity | — |
| **soc** | `.data.sel().roll` + `FILL_VALUE` rebuild | previous-SOC via `groupby("period").roll` | ☐ | 👍 deletes `FILL_VALUE` hack | [roll L1694](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1694), [fill L1735‑1737](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1735-L1737), [store-energy L1875‑1908](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1875-L1908) |
| **stoch** | stochastic = 2nd MI (scenario) | two aux-coord groups (`period`+`scenario`) | ☐ | ☐ TBD | — |
| **output** | `solution`/`dual` MI-indexed | flat + level coords (A) or re-stacked (B) | ☐ | ⚠️ A/B trade-off | — |
| **n.snapshots** | `pd.MultiIndex` public API | flat dim + level coords | ☐ | ⚠️ PyPSA API migration | [`global_constraints.py` L267](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/global_constraints.py#L267) (`reindex_like(lhs.data)`) |

The per-op cells above are the *nicer* facet. Two more facets are
**representation-wide**, not per-op (both verified this session):

- **safer** — under v1 a conflicting level/aux coord *raises* (the general
  aux-coord-conflict rule, `linopy/semantics.py: enforce_aux_conflict`); legacy
  silently keeps one side ([#295](https://github.com/PyPSA/linopy/issues/295)).
  MI "avoids" the conflict only by locking the levels into the index.
- **flexible** — flat+aux level coords can be `drop_vars`/`rename`/`assign_coords`'d
  like any coord; on an MI the same reassignment raises *"cannot drop or update
  coordinate … would corrupt the index"*. Removing MI removes that rigidity.

The two axes tell different stories: flat+aux is **feasible almost everywhere**,
and **desirable for every build-time op** — nicer (deletes MI machinery and
PyPSA's `.data`/`FILL_VALUE` workarounds), safer, and more flexible. The only ⚠️
cost sits at the **public boundary** (`output`, `n.snapshots`), which is exactly
the A-vs-B question. So "is it nice?" and "is it possible?" point the same way for
the internals; the debate is purely about the user-facing API.

**Side finding (not a row):** `Variable.sel` can't MI-tuple-select
(`x.sel(snapshot=(p, slice))` → `InvalidIndexError`), which is why PyPSA drops to
`.data` ([#752](https://github.com/PyPSA/linopy/issues/752) §2). Under flat+aux it
becomes `where(period == p)` / `isel`, removing the internals reach. Pinned by
`test_variable_mi_tuple_sel_not_forwarded`.

The `entry`–`model` rows answer the **steady-state (linopy)** question; the open
rows are **PyPSA-owned** and answer the **transition** question — verified by
solution-equivalence on real networks (multi-period, stochastic, Monte-Carlo)
plus scoping the public `n.snapshots` change.

## Sub-decision: the snapshot dim coordinate

After `reset_index`, the dim is **coordinate-less** (xarray virtualizes `0..N-1`
on access). `timestep` can't be the coord (non-unique across periods); the unique
`(period, timestep)` pair *is* an MI. So:

| dim coordinate | alignment | `.sel(snapshot=int)` | `.sel` by level tuple |
|---|---|---|---|
| pure `reset_index` (none, virtual `0..N-1`) | positional | ✅ positional fallback | ❌ → `where(period==…)` |
| `+ assign_coords(RangeIndex)` (stored int index) | by label (`==` position here) | ✅ label lookup | ❌ → `where(period==…)` |

Integer `.sel(snapshot=k)` works **either way** — on the coordinate-less dim
xarray degrades to positional selection (the tell: out-of-range raises
`IndexError`, not a label `KeyError`); with a `RangeIndex` it's a label lookup.
Since the coord is `0..N-1`, label `==` position, so they're indistinguishable for
valid integers. What flat+aux genuinely loses is `.sel(snapshot=(2020, "t1"))` —
selection by the **level tuple** (needs the MI) — replaced by `where(period==…)`;
that capability is gone under *both* flat variants, so it is not what the
coordinate choice decides.

The only thing the coordinate choice decides is **alignment**: positional
(coordinate-less) vs label (`RangeIndex`). For snapshots this is a distinction
without a difference — one canonical `n.snapshots` order means positional and
label alignment coincide — and it matches the model linopy already uses for a
plain single-period datetime snapshot dim. The single line for §11 is just:
**snapshot alignment is positional, not tuple-identity.**

## Transition shape (PyPSA)

PyPSA's hard MI couplings are a short, enumerated set
([#752](https://github.com/PyPSA/linopy/issues/752)) — the SOC roll + `FILL_VALUE`
rebuild and the growth `reindex_like(lhs.data)` linked above. These are *reluctant
workarounds*: the code comment says *"internal xarray multi-index difficulties"*
(cf. pydata/xarray#6836). #751 fixed the level groupby, so migrating these sites
**deletes** the `.data`-reach rather than re-spelling it. The build-time sites
break under **A and B equally** (they `.sel` the *stored* MI mid-build), so they
migrate regardless; A vs B is decided only by the **public `n.snapshots`** question
(`output`/`n.snapshots` rows), pressure-tested against
[PyPSA#1484](https://github.com/PyPSA/PyPSA/issues/1484) (Monte-Carlo, which wants
an MI level per sampled dimension).

## Decision record (to fill once the open rows close)

> _Outcome, the chosen option (A — return flat / B — re-stack MI at the boundary),
> the `n.snapshots` migration scope, and the evidence rows that justify it. This
> note, once complete, is what closes #744._

- **Internal normalization to flat+aux:** _safe to adopt regardless of A/B_ —
  cheap, canonical, version-safe (`entry` row + version sweep). **[provisional]**
- **A vs B:** _TBD_ — gated on the `output`/`n.snapshots` rows and the PyPSA
  stochastic experiments.
