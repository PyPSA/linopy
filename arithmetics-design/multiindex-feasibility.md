# MultiIndex feasibility for v1 (#744)

> Verification note (Claude Code, prompted by @FBumann), 2026-06-29. Can linopy v1
> drop first-class **`pd.MultiIndex`** for a **flat dim + aux level coords** model?
> Discussion: [#744](https://github.com/PyPSA/linopy/issues/744). Equality checks:
> [`test/test_mi_feasibility.py`](../test/test_mi_feasibility.py) (both `legacy` and
> `v1`). PyPSA refs pinned at **v1.2.4** ([`fb425cb`](https://github.com/PyPSA/PyPSA/tree/v1.2.4)).

## Verdict

**Feasible, and PyPSA stays stable.** Can linopy be simplified without destabilising
PyPSA, its primary consumer? Yes. linopy goes **flat in, flat out**: the model is flat
throughout and the solution comes back flat. An MI `snapshot` `(period, timestep)` is
flattened on entry (`reset_index`); PyPSA re-applies its own `n.snapshots` index when
mapping that solution onto components — as `assign_solution` already does — so PyPSA's
*results* stay MI-indexed exactly as today. For the **snapshot** axis MI is boundary
sugar PyPSA owns, never inside linopy's model; all **seven ways PyPSA puts it into
linopy's model** have a flat+aux form building the *identical* model, tested under both
semantics. The change is mostly **subtraction**:

- **less to maintain** — ~300 lines of first-class-MI machinery deleted, half one
  cluster (see *The payoff*).
- **xarray-native internals** — linopy stops knowing `pd.MultiIndex` exists or covering
  its quirks: the `isinstance(MI)` guards, the **39-site** *"would corrupt the index"*
  workaround (#303), the ~40 quirk-comments, the 169 lines of MI edge-case tests — a
  whole defensive surface gone, along with the latent risk of the MI edge cases it
  never fully covered. The model is just dims + ordinary aux coords.
- **unblocks work** — the MI-level groupby broken upstream (xarray#6836) works flat
  (#751); the MI coupling forcing PyPSA into linopy internals ([#752](https://github.com/PyPSA/linopy/issues/752)) goes.

The choices this forces are tabled in *Design decisions* — most settled pending
adoption. **Two are genuinely open, the second the harder:**

1. **`n.snapshots`** — PyPSA's own decoupled call: keep the MI (a cheap boundary wrap)
   or flatten; linopy works either way.
2. **multi-key `groupby`** — the *one* place linopy itself mints an MI (not PyPSA): a
   multi-key grouper returns a stacked `group` MultiIndex, consumed *downstream* (see
   *Second MI surface*). The same flat+aux fix, linopy-owned — but on a **public API**
   with **external** consumers, so it needs a deprecation path, not a boundary fix.

*Proof set, not universal: a PyPSA-Eur spot-check already surfaced a real second case
(multi-key `groupby`, below), so the generalization is not free — other forks/plugins
are unswept. For the **snapshot** axis specifically, after `reset_index` everything is
ordinary xarray and the only MI-specific capability lost is `.sel` by level tuple (→
`where`); MI snapshots aren't promoted, so few lean on them. More counterexamples
welcome — audit your own multi-key `groupby` + `.sel(group=…)`.*

## The evidence

PyPSA's observed MI uses of linopy, split by whether the MI **enters the model**.
The **snapshot** surface is the matrix below (the second surface, multi-key `groupby`,
has its own section). All ✅ (tested under `legacy`+`v1`; build-time rewrites
also compose into an identical LP, `test_per_period_lp_equivalent`). Glyph =
**feasible** (✅ tested · 🔲 achievable · ❌ no); word = **desirable** (better · parity
· worse).

| op | MI form | flat+aux form | feasible | desirable | PyPSA call site @ v1.2.4 |
|---|---|---|---|---|---|
| **entry** | `coords=[mi]` | `reset_index(dim)` | ✅ | better — deletes MI machinery | [`constraints.py` L1052](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1052) (`from_pandas_multiindex`) |
| **level select** | `sel(snapshot=(p, slice))` | `where(period == p)` | ✅ | parity | [`constraints.py` L1235‑1248](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1235-L1248) (KVL; the per-period loop is topology-driven — `cycle_matrix(period)`, not MI — so flat+aux only swaps the `sns.get_loc` slice for `where`, the loop stays) |
| **period roll** | `roll(1)` + `_period_start_mask` | `groupby("period").map(roll)` | ✅ (#751) | better — mask-free (boundary from grouping) | [`constraints.py` L1694](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1694) (SOC) |
| **level groupby** | `groupby(MI level)` ❌ broken ([xarray#6836](https://github.com/pydata/xarray/issues/6836)) | `groupby("period").sum()` | ✅ (#751) | better — **necessary**: MI is *broken*, not just workaround-y; flat+aux groups by the aux coord and works (#751) | — |
| **storage SOC** | `.data.sel().roll` + `FILL_VALUE` rebuild; `_period_start_mask` (shared w/ ramps) | previous-SOC via `groupby("period").roll`, then period-start: wrap (cyclic) · `.where` term (non-cyclic) · `mask=` row (ramp) | ✅ | better — deletes `FILL_VALUE` hack | [roll L1694](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1694), [fill L1735‑1737](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1735-L1737), [store-energy L1875‑1908](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1875-L1908); boundary mask [`common.py` L22](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/common.py#L22) → also ramps [`constraints.py` L838](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L838) |
| **output** | `solution`/`dual` MI-indexed | flat solution; caller re-stacks (or not) | ✅ | better — cheap boundary conversion (PyPSA's choice) | — |
| **snapshots param** | MI parked on `model.parameters`, rebuilt via `.to_index()` | flat param; `assign_solution` rebuilds `period`/`timestep` from aux | ✅ | better — removes the MI living *inside* a linopy object | store [`optimize.py` L689](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/optimize.py#L689); rebuild [L905](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/optimize.py#L905)/[L1114](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/optimize.py#L1114) |

No row needs a new linopy *capability* — the rewrites use ops linopy already has.
Entry is one `reset_index`; **who** runs it — linopy (accept an MI as input sugar) or
the caller (require flat input) — is a boundary policy decided in the *Decision
record*, not a capability gap. `level groupby`
is the one *necessary* row — an MI level can't be grouped (broken upstream,
xarray#6836; the dense-`_term` cost is representation-agnostic, not the differentiator),
flat+aux just works (#751); the rest are nicer or parity. Representation-wide, flat+aux
is also **safer** (v1 *raises* on a conflicting aux coord via `enforce_aux_conflict`;
MI only hides it, [#295](https://github.com/PyPSA/linopy/issues/295)) and **flexible**
(level coords `drop_vars`/`rename` freely; an MI raises *"would corrupt the index"*).

**Boundary uses** — two MI usages need no in-linopy solution, failing the in-model test
on opposite axes: `stochastic` is inside linopy but not an MI; `n.snapshots` is an MI
but not inside linopy.

| PyPSA MI usage | inside linopy? | an MI? | why linopy needn't solve it | call site @ v1.2.4 |
|---|---|---|---|---|
| **stochastic** `(scenario, name)` | **yes** — `scenario` is a real dim (N-D) | **no** — only an `xarray.stack` at the pandas output | already the flat+aux shape in the model; the MI is output-cosmetic, rebuilt at the boundary like `output`. Watch only whether [#1484](https://github.com/PyPSA/PyPSA/issues/1484) ever makes it an *in-model* stacked index (v1.2.4 does not) | pandas cols [`common.py` L78‑80](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/common.py#L78-L80); per-scenario loop [`optimize.py` L225‑229](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/optimize.py#L225-L229); [`isel(scenario=0)` L1092‑1094](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/constraints.py#L1092-L1094); output `.stack` [`array.py` L55‑64](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/components/array.py#L55-L64); [#1154](https://github.com/PyPSA/PyPSA/pull/1154) |
| **n.snapshots** | **no** — a PyPSA `Network` attribute | **yes** — but the MI lives in PyPSA | linopy only `reindex_like`s against it; the MI never enters the model. Keep MI (wrap at its boundary) or flatten — PyPSA's call | [`global_constraints.py` L267](https://github.com/PyPSA/PyPSA/blob/v1.2.4/pypsa/optimization/global_constraints.py#L267) (`reindex_like(lhs.data)`) |

(`Variable.sel` can't MI-tuple-select → `InvalidIndexError`, why PyPSA drops to `.data`
[#752](https://github.com/PyPSA/linopy/issues/752) §2; flat+aux makes it `where`/`isel`.
Pinned by `test_variable_mi_tuple_sel_not_forwarded`.)

## Second MI surface: multi-key `groupby`

The matrix above is the snapshot MI — passed *in* by PyPSA, solved at the entry
boundary. There is a **second** in-linopy MI, and it is **linopy's own**: a multi-key
`groupby` mints a stacked `group` MultiIndex. Verified here under both semantics —
`p.groupby(<country×carrier DataFrame>).sum()` → the `group` dim is a `pd.MultiIndex`,
and `.sel(group=("DE","wind"))` works.

PyPSA-Eur's `add_CCL_constraints` consumes it richly (not just `.sel`):
`grouper = concat([country, carrier]); lhs = p_nom.groupby(grouper).sum()`
([`solve_network.py` L1064](https://github.com/PyPSA/pypsa-eur/blob/master/scripts/solve_network.py)),
then `minimum.indexes["group"].intersection(lhs.indexes["group"])` and
`lhs.sel(group=index) >= minimum.loc[index]` (L1080‑1095). (`add_EQ`/`add_BAU` group by
a *single* key — flat `group` dim, no MI.)

It is the **same flat+aux change, owned by linopy**: `groupby` returns a flat `group`
dim with `country`/`carrier` as aux coords — uniform with every other dim, deleting the
last place linopy mints an MI (✅ desirable: safer / flexible / one representation; ⚪
parity on capability). But unlike snapshot it is a **public-API** change with
**external** consumers (PyPSA-Eur/-Earth/-DE), so the downstream `.sel(group=tuple)` /
`.intersection` / `.loc[MI]` → aux-coord selection migration needs a **deprecation
path**, and the full blast radius (forks, plugins) is **not yet scoped**. The harder of
the two open items.

## The payoff

Adopting flat+aux deletes first-class-MI machinery on **two layers**. The
*concentrated* one is ~300 lines — ~half the `alignment.py` *level-projection*
subsystem, whose only job is to make a single MI level align like a dimension (dead
once levels are aux coords):

| strip | what it does today | scale |
|---|---|---|
| **`alignment.py` level-projection** — `_project_onto_multiindex_levels`, `_enforce_implicit_projections`, `_LevelProjection`, the `projections` plumbing through `broadcast_to_coords`, plus the MI branches in `_expand_missing_dims`/`validate_alignment` and `_as_multiindex` | align a single-level operand against a full MI dim | ~150 lines |
| **netcdf MI (de)serialization** — `io.py` flatten-on-write + reconstruct (`{dim}_multiindex` attr); `common.py` MI level/code (de)serialize | flatten MI to store, rebuild MI on read | ~50 lines (+ read-only shim for old `.nc`) |
| **`assign_multiindex_safe`** (#303) — the corruption workaround | rebuild Datasets to dodge the *"would corrupt the index"* warning on assign — which fires *only* for the snapshot MI (internal stacks are `create_index=False`, so no other MI exists) | **39 call sites** → plain `.assign()` |
| **scattered guards / level-ops** in alignment & coords | skip-logic that only fires when a dim is an MI | 6 `isinstance(MI)` + 17 |

The *diffuse* layer is the **cognitive tax** that doesn't show up as deletable lines:
~40 quirk-comments explaining MI *"difficulties"*, and **169 lines of MI edge-case
tests across 10 files** — all moot once a snapshot is a flat dim. And that surface
almost certainly doesn't cover MI's *full* edge behaviour (the #303 corruption is one
gap that already bit), so every uncovered MI corner is **latent risk** flat+aux simply
retires. (This is why the first sweep undercounted: it scored deletable clusters and conservatively kept
`assign_multiindex_safe` — but with the snapshot MI gone it is the sole consumer of
the #303 workaround, so it goes too.)

Stays: the internal `_term`/`_factor`/groupby stacking (`create_index=False`, not MIs)
and the §11 aux-conflict logic — which gets *more* central, aux coords being the new
home for the levels.

> _Strip from a read-only sweep of the v1 tree; counts order-of-magnitude, not yet
> executed._

## Design decisions

What adopting flat+aux actually decides — **recommendation** in the last column.

| decision | options | recommendation |
|---|---|---|
| **Internal model** | first-class MI · **flat dim + aux coords** | flat+aux — feasible, tested, mostly subtraction. **[provisional]** |
| **MI on input** | reject · accept-with-warning · ~~silent accept~~ | silent flatten is **ruled out** (MI-in/flat-out surprises): reject, or accept *and warn*. _Undecided._ |
| **Snapshot alignment** | tuple-identity · **positional** | positional — one canonical `n.snapshots` order, matching a plain datetime snapshot (§11). |
| **Output** | reconstruct MI · **return flat** | flat — re-stack is the caller's cheap boundary step (`output` row, tested). |
| **`n.snapshots`** *(PyPSA-side)* | keep MI · flatten | PyPSA's decoupled call; linopy works either way. _TBD._ |
| **multi-key `groupby` output** | stacked `group` MI · **flat `group` + key aux coords** | flat+aux (uniform, deletes the last MI linopy mints) — but a **public-API** change; needs a deprecation path for external `.sel(group=…)` consumers. _Open; blast radius unscoped._ |

## Appendix

### `reset_index` is the whole transform

`snapshot` is one dim whose *index* is a `MultiIndex(period, timestep)`;
`period`/`timestep` are **levels, never dims** — they become a real dim only when
`.sel(period=p)` collapses the MI (xarray renames `snapshot → timestep`), which is why
per-period code must `.sel`-collapse, loop, or reach into `.data`. `reset_index` flips
**only** the index type (MI → flat), not the dim count (`('snapshot','name')` before
and after), demoting levels to aux coords — collapse/loop/`.data` becomes direct
`groupby`/`where`. Canonical xarray, byte-identical across the supported range
(2024.2.0 → 2026.4.0; post-dates the ~2022.06 explicit-indexes refactor, no shim).

| | today (MI) | flat+aux |
|---|---|---|
| `snapshot` | one dim, index = `MultiIndex(period, timestep)` | one flat dim |
| levels | MI levels (must `.sel`/collapse to use) | `period`/`timestep` aux coords |
| per-period op | `.sel(period=p)` / loop / `.data` | `groupby("period")` / `where` |
| alignment | tuple-identity | positional (one canonical order) |

### Snapshot dim coordinate

After `reset_index` the dim is **coordinate-less** (`0..N-1` virtual). The only choice
is positional vs label alignment — both lose level-tuple `.sel`:

| dim coordinate | alignment | `.sel(int)` | `.sel(level tuple)` |
|---|---|---|---|
| pure `reset_index` (virtual `0..N-1`) | positional | ✅ positional | ❌ → `where` |
| `+ assign_coords(RangeIndex)` | by label (`==` position) | ✅ label | ❌ → `where` |

With one canonical `n.snapshots` order, positional `==` label, matching a plain
datetime snapshot. **§11 rule: snapshot alignment is positional, not tuple-identity.**

### Transition shape (PyPSA)

[#752](https://github.com/PyPSA/linopy/issues/752) catalogues PyPSA **reaching into
linopy internals** (`.data`, `_term`, `FILL_VALUE`) — *not* MI per se. But several are
**MI-driven workarounds** (SOC `.data.sel().roll` + `FILL_VALUE`; growth
`reindex_like(lhs.data)`) that exist because MI groupby is broken (*"internal xarray
multi-index difficulties"*, xarray#6836). #751 fixed it, so flat+aux **deletes** those
reaches; they `.sel` the *stored* MI mid-build, so they migrate regardless of how PyPSA
presents output.
