# MultiIndex feasibility for v1 (#744)

> Verification note (Claude Code, prompted by @FBumann), 2026-06-29. Tracks
> whether linopy can drop **first-class `pd.MultiIndex`** support in v1 in favour
> of a **flat dim + auxiliary level coords** model. Discussion home:
> [#744](https://github.com/PyPSA/linopy/issues/744). Equality checks are tracked
> in [`test/test_mi_feasibility.py`](../test/test_mi_feasibility.py) (runs under
> both `legacy` and `v1`).

## Where MultiIndex is used in PyPSA

**Working assumption, stated boldly so it can be falsified: inside the linopy model
PyPSA uses a `pd.MultiIndex` in _exactly one place_ — the `snapshot` dimension**
(multi-period `(period, timestep)`). Everything else that looks MultiIndex-ish is
*not* an in-model index — which is why only those rows below carry a ⊥ tag:

- **`stochastic` `(scenario, name)`** — `scenario` and `name` are separate N-D dims
  inside linopy; the MI is only an `xarray.stack` at the pandas output
  (`components/array.py` @ v1.2.4). Not an in-model MI.
- **`n.snapshots`** — a real MI, but it lives in PyPSA's *public API*, never handed
  to linopy as a model index.
- **`output` / `dual`** — MI appears only on the returned pandas object, rebuilt at
  the boundary by the caller.

So the whole linopy-side question collapses to **one axis, `snapshot`**, and the
rest of this note works from that premise. Other axes (contingencies, a future
Monte-Carlo #1484) are *assumed* to follow the same N-D pattern, not audited — if
PyPSA holds an in-model MultiIndex anywhere else, this assumption and the scope
below need revisiting. **Corrections welcome.**

**Question.** Can v1 drop the stacked `pd.MultiIndex` snapshot for a flat
`snapshot` dim carrying `period`/`timestep` as auxiliary level coords?

**"Feasible"** has a precise, testable meaning: every real MI use case has a
flat+aux form that builds an *equivalent model* — proven by an explicit equality
check, not asserted.

## Data model under test

| | today (MI) | proposed (flat+aux) |
|---|---|---|
| snapshot dim | `MultiIndex[(period, timestep)]` | flat `snapshot` dim |
| level identity | MI levels | `period`/`timestep` **aux coords** on `snapshot` |
| entry conversion | — | `obj.reset_index("snapshot")` (canonical xarray, no custom logic) |
| alignment | tuple-identity | **positional** (one canonical snapshot order) |

The entry conversion is byte-identical across linopy's supported xarray range
(2024.2.0 floor → 2026.4.0); it post-dates xarray's explicit-indexes refactor
(~2022.06, below the floor), so no compat shim is needed.

## Feasibility matrix

Two axes — **feasible** (can flat+aux build an *equivalent model*?) and
**desirable** (our *opinion*: is it at least as good — *nicer* / *safer* /
*flexible*?):

| bubble | feasible | desirable |
|---|---|---|
| 🟢 | works today (tested) | better |
| 🔵 | achievable (unimplemented) | — |
| ⚪ | — | parity |
| 🔴 | not feasible | worse |

The 🟢 rows are tracked in `test/test_mi_feasibility.py` (which also solves a full
LP both ways, `test_per_period_lp_equivalent`, so the per-op rewrites are shown to
compose); PyPSA links are pinned at **v1.2.4** (commit [`fb425cb`](https://github.com/PyPSA/PyPSA/tree/v1.2.4)).

**⊥** marks a row that is *not* an in-linopy MultiIndex: the MI lives only at the
PyPSA/pandas boundary, or — for `stochastic` — nowhere in the model. The six
unmarked rows are the linopy decision proper (the snapshot `(period, timestep)` MI
that actually sits inside linopy variables); the ⊥ rows are PyPSA-side or no-op.

| op | MI form | flat+aux form | feasible | desirable | PyPSA call site @ v1.2.4 |
|---|---|---|---|---|---|
| **entry** | `coords=[mi]` | `reset_index(dim)` | 🟢 | 🟢 deletes MI machinery | [`constraints.py` L1052](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1052) (`from_pandas_multiindex`) |
| **level select** | `sel(snapshot=(p, slice))` | `where(period == p)` | 🟢 | ⚪ parity | [`constraints.py` L1235‑1248](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1235-L1248) (KVL; the per-period loop is topology-driven — `cycle_matrix(period)` — not MI, so flat+aux only swaps the `sns.get_loc` slice for `where`, the loop stays) |
| **period roll** | `roll(1)` + `_period_start_mask` | `groupby("period").map(roll)` | 🟢 (#751) | 🟢 mask-free (boundary from grouping) | [`constraints.py` L1694](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1694) (SOC) |
| **level groupby** | `groupby(MI level)` ❌ broken ([xarray#6836](https://github.com/pydata/xarray/issues/6836)) | `groupby("period").sum()` | 🟢 (#751) | 🟢 not just *nicer* — MI is *broken*, not merely workaround-y; flat+aux groups by the aux coord and just works (#751) | — |
| **storage SOC** | `.data.sel().roll` + `FILL_VALUE` rebuild; `_period_start_mask` (shared w/ ramps) | previous-SOC via `groupby("period").roll`, then period-start: wrap (cyclic) · `.where` drops the term (non-cyclic) · `mask=` drops the row (ramp) | 🟢 | 🟢 deletes `FILL_VALUE` hack | [roll L1694](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1694), [fill L1735‑1737](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1735-L1737), [store-energy L1875‑1908](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1875-L1908); boundary mask [`common.py` L22](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/common.py#L22) → also ramps [`constraints.py` L838](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L838) |
| **stochastic** ⊥ *not an MI* | `scenario` is a clean dim *into* linopy; `(scenario, name)` MI only on the pandas round-trip | `scenario` dim unchanged; round-trip rebuild like `output` | 🔵 | ⚪ *not* an in-model MI — `scenario`/`name` are separate dims (N-D); the `(scenario, name)` MI is only `.stack`→pandas at output. Nothing to decompose; already the flat+aux target form, PyPSA's MI here is output-cosmetic | `(scenario,name)` *pandas cols* [`common.py` L78‑80](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/common.py#L78-L80); per-scenario loop [`optimize.py` L225‑229](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/optimize.py#L225-L229); [`isel(scenario=0)` L1092‑1094](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1092-L1094); feature [PyPSA#1154](https://github.com/PyPSA/PyPSA/pull/1154), [#1484](https://github.com/PyPSA/PyPSA/issues/1484) wants *more*; output `.stack(scenario,name)` [`array.py` L55‑64](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/components/array.py#L55-L64) |
| **output** ⊥ *boundary* | `solution`/`dual` MI-indexed | flat solution; caller re-stacks (or not) | 🟢 | 🟢 cheap boundary conversion (PyPSA's choice) | — |
| **snapshots param** | MI parked on `model.parameters`, rebuilt via `.to_index()` | flat param; `assign_solution` rebuilds `period`/`timestep` from aux | 🔵 | 🟢 removes the MI living *inside* a linopy object | store [`optimize.py` L689](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/optimize.py#L689); rebuild [L905](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/optimize.py#L905)/[L1114](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/optimize.py#L1114) |
| **n.snapshots** ⊥ *PyPSA API* | `pd.MultiIndex` public API | flat dim + level coords | 🔵 | 🔴 PyPSA API migration | [`global_constraints.py` L267](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/global_constraints.py#L267) (`reindex_like(lhs.data)`) |

**No row needs a linopy change to be feasible.** The `entry` conversion is a
user-side `reset_index` with today's linopy; linopy auto-accepting `coords=[mi]`
and decomposing it would be *optional* input sugar. Dropping MI is a simplification
linopy chooses, not a capability it must add.

**`level groupby` is the one row where flat+aux is *necessary*, not merely
preferable.** Elsewhere flat+aux is parity or deletes a workaround for something MI
still does; here grouping by an MI *level* is broken upstream (xarray#6836, outside
linopy's control) — there is no working native path at all. flat+aux groups by the
aux-coord *name*, which #751 routed onto the existing fast path. (The dense-`_term`
memory cost of groupby-sum, [#756](https://github.com/PyPSA/linopy/issues/756)/[#757](https://github.com/PyPSA/linopy/issues/757),
is real but representation-agnostic — both forms hit the same `unstack` — so it is
*not* what separates MI from flat+aux here.)

The per-op cells above are the *nicer* facet. Two more facets are
**representation-wide**, not per-op (both verified this session):

- **safer** — under v1 a conflicting level/aux coord *raises* (the general
  aux-coord-conflict rule, `linopy/semantics.py: enforce_aux_conflict`); legacy
  silently keeps one side ([#295](https://github.com/PyPSA/linopy/issues/295)).
  MI "avoids" the conflict only by locking the levels into the index.
- **flexible** — flat+aux level coords can be `drop_vars`/`rename`/`assign_coords`'d
  like any coord; on an MI the same reassignment raises *"cannot drop or update
  coordinate … would corrupt the index"*. Removing MI removes that rigidity.

The two axes tell different stories: flat+aux is **feasible almost everywhere**
and **desirable for every build-time op** — nicer (deletes MI machinery and
PyPSA's `.data`/`FILL_VALUE` workarounds), safer, and more flexible. The output
boundary is a **cheap conversion PyPSA owns** (`output` row, tested), and the one MI that lived *inside* a linopy object — `model.parameters.snapshots` (`snapshots param` row) — goes flat too. So the
conclusion for linopy is clean: **accept MI as input sugar, decompose on entry
(`reset_index`), and never reconstruct — flat in, flat out. linopy drops
MultiIndex from its mental model entirely.**

The remaining cost is **not inside linopy**, and it is essentially **one** thing:
whether PyPSA keeps `n.snapshots` as MI — a cheap boundary wrap it can do on its
own side, decoupled from this decision. The **stochastic / Monte-Carlo direction**
(`stochastic` row) turns out *not* to be a second in-model MI at all: `scenario`
and `name` are separate N-D dims inside linopy, and the `(scenario, name)` MI is
only an `xarray.stack` at the pandas output. So there is nothing to decompose —
it is already the flat+aux form. The one thing to watch is whether future
Monte-Carlo work ([#1484](https://github.com/PyPSA/PyPSA/issues/1484)) introduces an
*in-model* stacked index rather than more dims; nothing in v1.2.4 does.

**Side finding (not a row):** `Variable.sel` can't MI-tuple-select
(`x.sel(snapshot=(p, slice))` → `InvalidIndexError`), which is why PyPSA drops to
`.data` ([#752](https://github.com/PyPSA/linopy/issues/752) §2). Under flat+aux it
becomes `where(period == p)` / `isel`, removing the internals reach. Pinned by
`test_variable_mi_tuple_sel_not_forwarded`.

The 🟢 rows answer the **steady-state (linopy)** question; the 🔵
rows are **PyPSA-owned** and answer the **transition** question — verified by
solution-equivalence on real networks (multi-period, stochastic, Monte-Carlo)
plus scoping the public `n.snapshots` change.

## What `reset_index` changes (and doesn't)

Today `snapshot` is a **single dimension** whose *index* is a `MultiIndex(period,
timestep)`; `period`/`timestep` are **levels, never dimensions**. They surface as a
real dim only *momentarily* — `.sel(period=p)` collapses the 2-level MI to one level
and xarray renames `snapshot → timestep`; the two never coexist as independent dims.
That is exactly why per-period code must `.sel`-collapse, loop, or reach into `.data`
(the SOC/KVL/objective coupling). `reset_index("snapshot")` flips **only** the index
type (MultiIndex → flat), not the dim count — `('snapshot', 'name')` before and
after — demoting the levels to ordinary aux coords; collapse/loop/`.data` then becomes
direct `groupby`/`where`. That one line *is* the whole flat+aux transformation.

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

[#752](https://github.com/PyPSA/linopy/issues/752) catalogues PyPSA **reaching
into linopy internals** (term-storage: `.data`, `vars`/`coeffs`/`const`, `_term`,
the `FILL_VALUE` sentinel) — *not* MI per se. But several of those reaches are
**MI-driven workarounds**: the SOC `.data.sel().roll` + `FILL_VALUE` rebuild and
the growth `reindex_like(lhs.data)` linked above exist because MI groupby is broken
(the code comment says *"internal xarray multi-index difficulties"*, cf.
pydata/xarray#6836). #751 fixed the level groupby, so flat+aux **deletes** those
specific reaches rather than re-spelling them. These build-time sites `.sel` the
*stored* MI mid-build, so they migrate whether or not PyPSA re-stacks the output.
What's left to decide is only the **public `n.snapshots`** question (`n.snapshots`
row). Stochastic is *not* a second in-model MI — `scenario`/`name` are separate
N-D dims, the MI is pandas-output only — so the only watch item is whether
[PyPSA#1484](https://github.com/PyPSA/PyPSA/issues/1484) (Monte-Carlo) ever turns
that into an in-model stacked index; v1.2.4 does not.

## Decision record (to fill once the open rows close)

> _The `n.snapshots` scope (the one real open item) and the evidence rows that
> justify it. This note, once complete, is what closes #744._

- **linopy drops MultiIndex from its mental model** — accept MI as input sugar,
  decompose on entry (`reset_index`), never reconstruct (flat in, flat out).
  Internal normalization is safe to adopt now: cheap, canonical, version-safe
  (`entry` row + version sweep). **[provisional — internals verified]**
- **Output: linopy returns flat** — the re-stack is a cheap boundary conversion
  PyPSA owns (`output` row, tested); no MI adapter lives inside linopy.
- **`n.snapshots`** — PyPSA's independent, decoupled choice: keep MI (wrap at its
  own boundary) or flatten. _TBD, PyPSA-side._
- **Stochastic is N-D, not a second MI** (`stochastic` row) — `scenario`/`name` are
  separate dims; the `(scenario, name)` MI is only an output `.stack`. No in-model MI
  to remove. Watch only whether [#1484](https://github.com/PyPSA/PyPSA/issues/1484)
  (Monte-Carlo) introduces an in-model stacked index; v1.2.4 does not. _Low risk._
