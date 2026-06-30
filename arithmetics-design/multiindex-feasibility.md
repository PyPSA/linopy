# MultiIndex feasibility for v1 (#744)

> Verification note (Claude Code, prompted by @FBumann), 2026-06-29. Tracks whether
> linopy can drop **first-class `pd.MultiIndex`** support in v1 for a **flat dim +
> auxiliary level coords** model. Home: [#744](https://github.com/PyPSA/linopy/issues/744).
> Equality checks: [`test/test_mi_feasibility.py`](../test/test_mi_feasibility.py)
> (runs under both `legacy` and `v1`).

## Verdict

**Feasible — yes, with zero linopy changes.** linopy accepts an MI snapshot as
*input sugar*, `reset_index` on entry, and goes **flat in / flat out** — never
reconstructing it. PyPSA uses a MultiIndex on exactly one axis, `snapshot`
`(period, timestep)`, and all seven of its in-linopy uses have a flat+aux form that
builds the identical model, each tested under both semantics.

**One open item, and it is PyPSA's, not linopy's:** whether `n.snapshots` stays a
MultiIndex — a cheap boundary wrap PyPSA owns.

Everything below is evidence: the **matrix** of the seven in-linopy ops (settled,
tested now), then the two **PyPSA-side usages** (the transition, PyPSA-owned).

## Scope: one axis, `snapshot`

Inside the linopy model PyPSA uses a `pd.MultiIndex` only on the `snapshot`
dimension; it rides that axis through the whole lifecycle — in at `entry`, through
the per-period ops, out at `output` (`solution`/`dual` carry `snapshot`, so they are
MI-indexed too), parked on `snapshots param`. Two usages merely *look* MultiIndex-ish
and are handled PyPSA-side (their own table below): `stochastic` (`scenario`/`name`
are separate N-D dims; the MI is only an `xarray.stack` at the pandas output) and
`n.snapshots` (a real MI, but PyPSA's *public API*, never a linopy model index).

## Data model under test

| | today (MI) | proposed (flat+aux) |
|---|---|---|
| snapshot dim | `MultiIndex[(period, timestep)]` | flat `snapshot` dim |
| level identity | MI levels | `period`/`timestep` **aux coords** on `snapshot` |
| entry conversion | — | `obj.reset_index("snapshot")` (canonical xarray, no custom logic) |
| alignment | tuple-identity | **positional** (one canonical snapshot order) |

The entry conversion is byte-identical across linopy's supported xarray range
(2024.2.0 → 2026.4.0) — it post-dates the explicit-indexes refactor (~2022.06), so
no compat shim is needed.

## Feasibility matrix

Two axes, encoded separately so a glyph never means two things at once: **feasible**
— does flat+aux build an equivalent model? — and **desirable** — our opinion, is it
at least as good?

- **feasible** (glyph): ✅ tested · 🔲 achievable, untested · ❌ no
- **desirable** (word): better · parity · worse

All seven rows are the one `snapshot` MI inside linopy, and **all ✅** — tested under
both `legacy` and `v1`, with the build-time rewrites also shown to compose into an
identical LP (`test_per_period_lp_equivalent`). PyPSA links pinned at **v1.2.4**
([`fb425cb`](https://github.com/PyPSA/PyPSA/tree/v1.2.4)).

| op | MI form | flat+aux form | feasible | desirable | PyPSA call site @ v1.2.4 |
|---|---|---|---|---|---|
| **entry** | `coords=[mi]` | `reset_index(dim)` | ✅ | better — deletes MI machinery | [`constraints.py` L1052](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1052) (`from_pandas_multiindex`) |
| **level select** | `sel(snapshot=(p, slice))` | `where(period == p)` | ✅ | parity | [`constraints.py` L1235‑1248](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1235-L1248) (KVL; the per-period loop is topology-driven — `cycle_matrix(period)`, not MI — so flat+aux only swaps the `sns.get_loc` slice for `where`, the loop stays) |
| **period roll** | `roll(1)` + `_period_start_mask` | `groupby("period").map(roll)` | ✅ (#751) | better — mask-free (boundary from grouping) | [`constraints.py` L1694](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1694) (SOC) |
| **level groupby** | `groupby(MI level)` ❌ broken ([xarray#6836](https://github.com/pydata/xarray/issues/6836)) | `groupby("period").sum()` | ✅ (#751) | better — **necessary**: MI is *broken*, not just workaround-y; flat+aux groups by the aux coord and works (#751) | — |
| **storage SOC** | `.data.sel().roll` + `FILL_VALUE` rebuild; `_period_start_mask` (shared w/ ramps) | previous-SOC via `groupby("period").roll`, then period-start: wrap (cyclic) · `.where` term (non-cyclic) · `mask=` row (ramp) | ✅ | better — deletes `FILL_VALUE` hack | [roll L1694](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1694), [fill L1735‑1737](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1735-L1737), [store-energy L1875‑1908](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1875-L1908); boundary mask [`common.py` L22](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/common.py#L22) → also ramps [`constraints.py` L838](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L838) |
| **output** | `solution`/`dual` MI-indexed | flat solution; caller re-stacks (or not) | ✅ | better — cheap boundary conversion (PyPSA's choice) | — |
| **snapshots param** | MI parked on `model.parameters`, rebuilt via `.to_index()` | flat param; `assign_solution` rebuilds `period`/`timestep` from aux | ✅ | better — removes the MI living *inside* a linopy object | store [`optimize.py` L689](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/optimize.py#L689); rebuild [L905](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/optimize.py#L905)/[L1114](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/optimize.py#L1114) |

**No row needs a linopy change.** `entry` is a user-side `reset_index` with today's
linopy; auto-accepting `coords=[mi]` would be *optional* input sugar. Dropping MI is
a simplification linopy chooses, not a capability it must add.

**`level groupby` is the one *necessary* row, not merely nicer.** Grouping by an MI
*level* is broken upstream (xarray#6836, outside linopy's control) — no working
native path — while flat+aux groups by the aux-coord name, which #751 put on the
fast path. (The dense-`_term` memory cost of groupby-sum,
[#756](https://github.com/PyPSA/linopy/issues/756)/[#757](https://github.com/PyPSA/linopy/issues/757),
is representation-agnostic — both forms hit the same `unstack` — so it is *not* what
separates them.)

Beyond the per-op *nicer* facet, two are **representation-wide**:

- **safer** — under v1 a conflicting level/aux coord *raises* (`enforce_aux_conflict`);
  legacy silently keeps one side ([#295](https://github.com/PyPSA/linopy/issues/295)).
  MI "avoids" the conflict only by locking levels into the index.
- **flexible** — flat+aux level coords `drop_vars`/`rename`/`assign_coords` like any
  coord; on an MI the same raises *"would corrupt the index"*.

## Observed PyPSA MultiIndex usages — not linopy's to solve

Two MultiIndex usages appear in PyPSA but **need no in-linopy solution** — cheap for
PyPSA to handle at its boundary. They fail the *in-linopy MI* test on **opposite**
axes: `stochastic` is inside linopy but isn't an MI; `n.snapshots` is an MI but isn't
inside linopy.

| PyPSA MI usage | inside linopy? | an MI? | why linopy needn't solve it | call site @ v1.2.4 |
|---|---|---|---|---|
| **stochastic** `(scenario, name)` | **yes** — `scenario` is a real dim (N-D) | **no** — only an `xarray.stack` at the pandas output | already the flat+aux shape in the model; the MI is output-cosmetic, rebuilt at the boundary like `output`. Watch only whether [#1484](https://github.com/PyPSA/PyPSA/issues/1484) ever makes it an *in-model* stacked index (v1.2.4 does not) | pandas cols [`common.py` L78‑80](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/common.py#L78-L80); per-scenario loop [`optimize.py` L225‑229](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/optimize.py#L225-L229); [`isel(scenario=0)` L1092‑1094](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1092-L1094); output `.stack` [`array.py` L55‑64](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/components/array.py#L55-L64); [#1154](https://github.com/PyPSA/PyPSA/pull/1154) |
| **n.snapshots** | **no** — a PyPSA `Network` attribute | **yes** — but the MI lives in PyPSA | linopy only `reindex_like`s against it; the MI never enters the model. Keep MI (wrap at its boundary) or flatten — PyPSA's call | [`global_constraints.py` L267](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/global_constraints.py#L267) (`reindex_like(lhs.data)`) |

**Side finding (not a row):** `Variable.sel` can't MI-tuple-select
(`x.sel(snapshot=(p, slice))` → `InvalidIndexError`), which is why PyPSA drops to
`.data` ([#752](https://github.com/PyPSA/linopy/issues/752) §2). Under flat+aux it
becomes `where(period == p)` / `isel`. Pinned by
`test_variable_mi_tuple_sel_not_forwarded`.

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

After `reset_index` the dim is **coordinate-less** (xarray virtualizes `0..N-1`).
`timestep` can't be the coord (non-unique across periods); the unique `(period,
timestep)` pair *is* an MI. The choice is only positional vs label alignment:

| dim coordinate | alignment | `.sel(snapshot=int)` | `.sel` by level tuple |
|---|---|---|---|
| pure `reset_index` (virtual `0..N-1`) | positional | ✅ positional fallback | ❌ → `where(period==…)` |
| `+ assign_coords(RangeIndex)` | by label (`==` position) | ✅ label lookup | ❌ → `where(period==…)` |

Integer `.sel` works either way (label `==` position for `0..N-1`). What flat+aux
loses is selection by the **level tuple** `.sel(snapshot=(2020, "t1"))` — gone under
*both* variants, so it is not what the coordinate decides. With one canonical
`n.snapshots` order, positional and label alignment coincide, matching what linopy
already does for a plain datetime snapshot. The single rule for §11: **snapshot
alignment is positional, not tuple-identity.**

## Transition shape (PyPSA)

[#752](https://github.com/PyPSA/linopy/issues/752) catalogues PyPSA **reaching into
linopy internals** (`.data`, `_term`, the `FILL_VALUE` sentinel) — *not* MI per se.
But several reaches are **MI-driven workarounds** (SOC `.data.sel().roll` +
`FILL_VALUE`, growth `reindex_like(lhs.data)`) that exist because MI groupby is broken
(code comment: *"internal xarray multi-index difficulties"*, xarray#6836). #751 fixed
the level groupby, so flat+aux **deletes** those reaches rather than re-spelling them
— and they `.sel` the *stored* MI mid-build, so they migrate regardless of how PyPSA
presents the output.

## Decision record (to fill once the open item closes)

> _The `n.snapshots` scope — the one real open item — and the evidence rows that
> justify it. This note, once complete, is what closes #744._

- **linopy drops MultiIndex from its mental model** — accept MI as input sugar,
  decompose on entry (`reset_index`), never reconstruct (flat in, flat out). Safe to
  adopt now: cheap, canonical, version-safe. **[provisional — internals verified]**
- **Output: linopy returns flat** — the re-stack is a cheap boundary conversion PyPSA
  owns (`output` row, tested); no MI adapter lives inside linopy.
- **`n.snapshots`** — PyPSA's independent, decoupled choice: keep MI (wrap at its own
  boundary) or flatten. _TBD, PyPSA-side._
