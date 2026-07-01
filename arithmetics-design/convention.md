# The v1 convention

The strict ("v1") convention for linopy. Goals and rollout plan:
[`goals.md`](goals.md). The bugs it fixes are catalogued in [#714].

An object-scope statement, then thirteen sections in three groups: absence
(§1–§7), coordinate alignment (§8–§11), then constraints and reductions
(§12–§13).

## Object scope

The convention governs every operation a linopy object takes part in,
whatever the other operand is — a DataArray, a pandas Series or DataFrame, a
numpy array, a list, or a scalar. A non-linopy operand is converted to a
labelled array (`as_dataarray` in `linopy.common`) and from there behaves
exactly like the constant-only expression holding the same values and
coordinates: `x + arr` builds what `x + arr_expr` builds, for every operator
and in either operand position. No rule below depends on what type an
operand arrived as. How an operand *gets* its labels — its own coordinates,
or pairing by size when it has none — is the alignment group's first rule.

(The lone exception is type-decided: an expression is never a valid divisor,
so `x / arr` works where `x / arr_expr` raises `TypeError`.)

## Absence

Absence — a labelled slot the model does not cover — is the richer half of the
convention. The sections below say what it is (§1–§3), how it arises (§4–§5),
and how it flows through arithmetic and is resolved (§6–§7).

### §1. Absence is a first-class state

A *slot* — one labelled position — is either present or *absent*. An absent
slot is one the model does not cover. Absence is a state in its own right,
never a stand-in for a number: an absent variable is not a variable fixed to
zero ([#712]).

### §2. Encoding absence

The *marker* is how an absent slot is stored: `NaN` in floating-point fields
(`coeffs`, `const`, numeric constants), and `-1` in integer label fields (a
variable's `labels`, an expression's `vars`, which cannot hold a NaN). The two
encodings are one concept — an absent slot, whatever the dtype.

Within a single slot, the markers move together: `const.isnull()` at a slot
implies *every* term at that slot has `coeffs = NaN` and `vars = -1`. Operators
that introduce absence at a slot also absorb any live terms there, so the
storage never carries a half-absent row. A term at a *present* slot may still
carry `vars = -1` after `fillna(value)` revives the slot — that's a *dead
term*, inert at the solver layer, and only meaningful as storage book-keeping.

### §3. Testing absence

`isnull()` is the one predicate for absence. It reads the marker — `NaN` or
`-1`, whichever the field uses — and reports absence slot by slot. Every rule
that speaks of an "absent slot" means exactly what `isnull()` reports; the
caller never inspects the raw marker.

### §4. Creating absence

Absence enters a model only through named operations: `mask=` at construction
marks slots absent up front; `.where(cond)` masks slots in place, keeping
shape; `.reindex()`, `.reindex_like()`, `.shift()`, and `.unstack()`
restructure a coordinate and leave the new positions absent. Operations that
merely move or select existing data — `.roll()`, `.sel()`, `.isel()` — never
introduce it.

### §5. User-supplied NaN raises

A NaN in a user-supplied constant raises `ValueError`. linopy trusts NaN only
from its own structural operations (§4), which genuinely mark absence. A NaN in
user data is ambiguous — a deliberate "absent", or a data error — so linopy
refuses to guess and asks the caller to resolve it with `fillna()`. This
replaces today's silent per-operator fills, which guessed a different value for
every operator ([#713]). To mark slots absent, use the mechanisms of §4 — a
bare NaN in a constant is not one of them.

The alternative — reading user NaN as "absent" instead of raising — was
discussed in [#627] and closed: ambiguous overload of a numeric value
defeats goal #1, since a data-error NaN is silently re-labelled as
intentional absence.

### §6. Absence propagates through every operator

Every operator carries absence through unchanged: a slot absent in any operand
is absent in the result. `shifted * 3` is absent; `shifted + 5` is absent;
`x + shifted` is absent wherever `shifted` is — even though `x` itself is fine
there.

linopy never fills an absent slot on the user's behalf, because the right fill
depends on intent it cannot see: 0 for a sum, 1 for a product, or "leave this
out" entirely. Because every operator propagates the same way, the algebraic
laws of §10 carry over to absent slots untouched — absence absorbs, so every
grouping of an expression agrees. And `shifted * 3` staying absent, rather than
collapsing to `0`, is what preserves the absent-vs-zero distinction of §1.

### §7. Resolving absence

Because §6 never fills, turning an absent slot into a value is the caller's
explicit act, never linopy's. `fillna(value)` fills an expression's absent
slots; `.fillna(...)` fills a constant before it enters the arithmetic;
`fill_value=` on a named method fills as part of the call. Filling at the call
site documents the intent: `x + y.shift(time=1).fillna(0)` says "treat the
missing earlier step as zero" exactly where it matters.

## Coordinate alignment

linopy's operands are xarray objects, so the convention starts from xarray's
alignment model (goal 4): coordinates align by *label*, never by position;
non-shared dimensions broadcast; a mismatch on a shared dimension is resolved
by an explicit *join*.

Operands that carry coordinates — a DataArray, a pandas Series or DataFrame —
align by them, under the rules below. *Unlabeled* operands — numpy arrays,
lists, polars Series — carry no labels to align by, so they pair with the
linopy operand's dimensions by size: each axis adopts the dimension (and the
coordinates) whose length matches, and the rules below apply from there. The
pairing must be determined by the sizes alone. A length-4 array meeting a
variable with dims `(a: 4, time: 5)` pairs with `a`; meeting a variable with
dims `(a: 4, b: 4)` it could pair with either, so the operation raises — as
it does when no dimension matches. The same goes for a 4×4 array against
`(a: 4, b: 4)`: sizes cannot tell `(a, b)` from `(b, a)`. To name the
dimensions, wrap the array in a DataArray.

A scalar broadcasts over every dimension and so needs no pairing. A 0-d
array is treated as a scalar; a Python `list` is read as a numpy array
(it carries values, not labels). Implemented in `linopy.alignment`
([#736]).

### §8. Shared dimensions must carry the same labels

If two operands share a dimension, their coordinate labels must be the same
*set*, or the operator raises `ValueError`. Order is immaterial: the same
labels in a different order are the same coordinate and align by label (a
reindex), following "by label, never by position" above — only a difference in
the label set raises.

This is close to xarray's `arithmetic_join="exact"` — deliberately stricter
than xarray's own default (`inner`) — but order-independent, where xarray's
`exact` would reject a pure reorder. An inner join silently drops the
non-overlapping labels, and in an optimization model a dropped coordinate is a
dropped term or constraint: a silent wrong answer. Matching on the label set
surfaces a real mismatch where it happens. (The [pyoframe] library uses the
same model.)

Because the rule is identical for every operator, the operator-alignment split
([#708]) — `*` aligning by label while `+`, `-`, `/` go by position —
disappears.

### §9. Non-shared dimensions broadcast freely

A dimension present in only one operand broadcasts over the other, with no
restriction — for both expressions and constants. Only *shared* dimensions are
subject to §8.

### §10. Mismatches resolve via an explicit join

When coordinates genuinely differ, §8 raises — and the caller says how to
resolve it. Several primitives bring operands into agreement:

- `.sel()` / `.isel()` cut operands down to a shared subset — often the
  clearest fix.
- The named methods — `.add` `.sub` `.mul` `.div` `.le` `.ge` `.eq` — take a
  `join=` argument: `exact`, `inner`, `outer`, `left`, `right`, or `override`.
  `override` is the old positional behavior — still available, but now opt-in
  and named rather than triggered by a size coincidence. It still requires the
  shared dimensions to match in size: a genuine size mismatch raises rather
  than relabelling mismatched data, so reach for a label join (`inner` /
  `outer` / `left` / `right`) when the sizes really differ.
- `.reindex()` / `.reindex_like()` conform an operand to a target index
  (extending past the original creates absent positions — §4).
- `.assign_coords()` relabels an operand outright. Unlike `join="override"`,
  which checks the shared dims match in size, this is an *unguarded* relabel:
  it renames positions whether or not they correspond, so the "made explicit"
  here is the caller's responsibility, not a safety check.
- `linopy.align()` pre-aligns several operands at once.

Because no operator silently drops coordinates, the associativity break
([#711]) cannot occur: the operation that used to drop coordinates now raises.
Every standard algebraic law — commutativity, associativity, distributivity,
the identities — holds for same-coordinate operands.

### §11. Auxiliary-coordinate conflicts raise

Auxiliary (non-dimension) coordinates are user-attached metadata: a coord
defined on some dimension but not itself a dimension, like a `B(A)` group
label on dimension `A`. linopy *validates* them (the conflict-raise rule
below) and *propagates* them through arithmetic unchanged, but never
*computes* with them — they describe the data, they don't enter the math.

When two operands carry an aux coord with the same name and values agree,
the coord propagates to the result. When only one operand carries the
coord, it propagates from that operand unchanged — asymmetric presence is
not a conflict. When the values *do* disagree (same name on both sides,
different values), the operator raises — `xarray` silently drops the
conflict, which is the [#295] bug. The caller resolves it explicitly with
`.drop_vars(name)` (remove the coord) or `.assign_coords(name=...)`
(relabel one side).

**Stacked MultiIndex dimensions.** A stacked MultiIndex dim (e.g. PyPSA's
`(period, timestep)` `snapshot`) stores its *levels* as auxiliary
coordinates — `period` and `timestep` are non-dimension coords on
`snapshot` — and its elements are *level combinations* (one tuple per
position). An operand whose *dimension* names one of those levels — a
per-`period` weighting meeting a `snapshot`-indexed expression — is a
same-name conflict between a dimension and an auxiliary coordinate, and it
raises like any other conflict of this section. There is no implicit
projection; write it explicitly by selecting with the dimension's level
values:

    weights.sel(period=expr.indexes["snapshot"].get_level_values("period"))

An input that reconstructs the *entire* MultiIndex (all levels, every
combination) is not a conflict — it is the same coordinate spelled
differently, and aligns by tuple under §8, in any order. Order is
immaterial here exactly as for a plain dimension: the same tuples in a
different order are reordered to match, not rejected.

(Legacy projects implicitly and warns — scenario B of the [#732]/[#737]
discussion; the implicit projection is removed at 1.0.)

## Constraints and reductions

Two kinds of operation build on the rules above without being binary operators:
the comparisons that form constraints, and the reductions that collapse a
dimension.

### §12. Constraints follow the same rules

A constraint is built by comparing two sides with `<=`, `>=`, or `==` — and a
comparison is an operator like any other. It aligns its sides by §8 and carries
absence by §6, exactly as `+`, `-`, `*`, and `/` do. So algebraically equal
forms build the same constraint: `x - a <= 0` and `x <= a` agree, where today
they do not ([#707]).

Each slot becomes one constraint row. An absent slot yields no row — absence
propagated into a comparison drops the constraint there, the same outcome as
masking it.

### §13. Reductions skip absent slots

Reductions collapse a dimension rather than combining two operands, so the
NaN propagation of §6 does not apply: they *skip* absent slots instead. `sum`
(including `groupby.sum`) adds only the present terms, and the sum of none is
the zero expression. The objective totals its terms the way `sum` does.

Further reductions (`mean`, `resample`, `coarsen`) are not in linopy yet; they
are added under v1 only ([#703]) — as new operations with no legacy behaviour —
and follow this same skip-absent rule.

<!-- references -->
[pyoframe]: https://github.com/Bravos-Power/pyoframe
[#732]: https://github.com/PyPSA/linopy/pull/732
[#737]: https://github.com/PyPSA/linopy/pull/737
[#736]: https://github.com/PyPSA/linopy/issues/736
[#714]: https://github.com/PyPSA/linopy/issues/714
[#703]: https://github.com/PyPSA/linopy/issues/703
[#713]: https://github.com/PyPSA/linopy/issues/713
[#712]: https://github.com/PyPSA/linopy/issues/712
[#711]: https://github.com/PyPSA/linopy/issues/711
[#708]: https://github.com/PyPSA/linopy/issues/708
[#707]: https://github.com/PyPSA/linopy/issues/707
[#627]: https://github.com/PyPSA/linopy/issues/627
[#295]: https://github.com/PyPSA/linopy/issues/295
