# MultiIndex feasibility for v1 (#744)

> Verification note (Claude Code, prompted by @FBumann), 2026-06-29. Can linopy v1
> drop first-class **`pd.MultiIndex`** for a **flat dim + aux level coords** model?
> Discussion: [#744](https://github.com/PyPSA/linopy/issues/744). Equality checks:
> [`test/test_mi_feasibility.py`](../test/test_mi_feasibility.py) (both `legacy` and
> `v1`). PyPSA refs pinned at **v1.2.4** ([`fb425cb`](https://github.com/PyPSA/PyPSA/tree/v1.2.4)).

## Verdict

**Feasible, and PyPSA stays stable.** Can linopy be simplified without destabilising
PyPSA, its primary consumer? Yes. linopy goes **flat in, flat out**: an MI `snapshot`
`(period, timestep)` is accepted only as input sugar (`reset_index`'d on entry), the
model is flat throughout, and the solution comes back flat. PyPSA re-applies its own
`n.snapshots` index when mapping that solution onto components — as `assign_solution`
already does — so PyPSA's *results* stay MI-indexed exactly as today. MI is **boundary
sugar PyPSA owns**, never inside linopy's model. All **seven ways PyPSA puts that MI
into linopy's model** have a flat+aux form building the *identical* model, tested under
both semantics. The change is mostly **subtraction**:

- **less to maintain** — ~300 lines of first-class-MI machinery deleted, half one
  cluster (see *The payoff*).
- **xarray-native internals** — linopy stops knowing `pd.MultiIndex` exists or covering
  its quirks; the model is just dims + ordinary aux coords.
- **unblocks work** — the MI-level groupby broken upstream (xarray#6836) works flat
  (#751); the MI coupling forcing PyPSA into linopy internals ([#752](https://github.com/PyPSA/linopy/issues/752)) goes.

Small additive work: positional snapshot alignment; shrink MI-input to
accept-then-`reset_index`. **One open item, PyPSA's own decoupled call:** whether
`n.snapshots` stays an MI (a cheap boundary wrap) — linopy works either way.

*Proof set, not universal: PyPSA is the one consumer audited. But after `reset_index`
(general to any MI) everything is ordinary xarray, so the only MI-specific capability
lost for any user is `.sel` by level tuple (→ `where`). Counterexamples welcome.*

## The evidence

PyPSA's observed MI uses of linopy, split by whether the MI **enters the model**.
In-model uses are the matrix — all ✅ (tested under `legacy`+`v1`; build-time rewrites
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

## The payoff

Adopting flat+aux **deletes** ~300 lines of first-class-MI machinery — concentrated,
not diffuse: ~half is the `alignment.py` *level-projection* subsystem, whose only job
is to make a single MI level align like a dimension (dead once levels are aux coords).

| strip | what it does today | ~lines |
|---|---|---|
| **`alignment.py` level-projection** — `_project_onto_multiindex_levels`, `_enforce_implicit_projections`, `_LevelProjection`, the `projections` plumbing through `broadcast_to_coords`, plus the MI branches in `_expand_missing_dims`/`validate_alignment` and `_as_multiindex` | align a single-level operand against a full MI dim | ~150 |
| **netcdf MI (de)serialization** — `io.py` flatten-on-write + reconstruct (`{dim}_multiindex` attr); `common.py` MI level/code (de)serialize | flatten MI to store, rebuild MI on read | ~50 (keep a read-only shim for old `.nc`) |
| **scattered MI guards/branches** in alignment & coords | skip-logic that only fires when a dim is an MI | remainder |

Stays: `assign_multiindex_safe` and internal `_term`/`_factor`/groupby stacking
(general helpers over linopy's own indexes, unrelated to snapshot-MI), and the §11
aux-conflict logic — which gets *more* central, aux coords being the new home for the
levels.

> _Strip from a read-only sweep of the v1 tree; counts order-of-magnitude, not yet
> executed._

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

### Decision record (fill once `n.snapshots` closes)

- **linopy drops MI from its model** — flat dim + aux coords throughout, `reset_index`
  on entry, never reconstruct (flat in, flat out). Safe now: cheap, canonical,
  version-safe. **[provisional]**
- **MI on input — sugar or rejected?** Accept `coords=[mi]` and auto-`reset_index`
  (backwards-compatible; a thin input shim survives) *vs* reject MI so callers flatten
  first (purest; breaks existing MI-passing callers). _Recommend accept-as-sugar, with
  an optional deprecation path to flat-only._
- **Output returns flat** — the re-stack is a cheap boundary conversion PyPSA owns
  (`output` row, tested).
- **`n.snapshots`** — PyPSA's decoupled choice: keep MI or flatten. _TBD, PyPSA-side._
