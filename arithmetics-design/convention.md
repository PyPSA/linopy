# The v1 convention

The strict ("v1") convention for linopy. Goals and rollout plan:
[`goals.md`](goals.md). The bugs it fixes are catalogued in [#714].

Thirteen sections in three groups: absence (§1–§7), coordinate alignment
(§8–§11), then constraints and reductions (§12–§13).

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

**Open question:** whether user NaN should instead be read as "absent" — [#627].

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

**Open question:** how should v1 align *unlabeled* data — a raw numpy array
carries no labels to match on. Still open.

### §8. Shared dimensions must match exactly

If two operands share a dimension, their coordinate labels must be identical,
or the operator raises `ValueError`.

This is xarray's model with `arithmetic_join="exact"` — deliberately stricter
than xarray's own default (`inner`). An inner join silently drops the
non-overlapping labels, and in an optimization model a dropped coordinate is a
dropped term or constraint: a silent wrong answer. An exact match surfaces the
mismatch where it happens. (The [pyoframe] library uses the same model.)

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
  and named rather than triggered by a size coincidence.
- `.reindex()` / `.reindex_like()` conform an operand to a target index
  (extending past the original creates absent positions — §4).
- `.assign_coords()` relabels an operand outright (positional alignment, made
  explicit).
- `linopy.align()` pre-aligns several operands at once.

Because no operator silently drops coordinates, the associativity break
([#711]) cannot occur: the operation that used to drop coordinates now raises.
Every standard algebraic law — commutativity, associativity, distributivity,
the identities — holds for same-coordinate operands.

### §11. Auxiliary-coordinate conflicts raise

Non-dimension (auxiliary) coordinates propagate when operands agree on them. A
conflict raises, rather than silently keeping one side ([#295]).

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

Reductions — `sum`, `mean`, and the `groupby` / `resample` / `coarsen`
aggregations — collapse a dimension rather than combining two operands, so the
propagation of §6 does not apply: they *skip* absent slots instead. `sum` adds
the present terms, and the sum of none is the zero expression. `mean` divides
by the count of *present* slots, not all of them — dividing by all would treat
an absent slot as a zero term, which §1 forbids. The objective totals its
terms the way `sum` does.

<!-- references -->
[pyoframe]: https://github.com/Bravos-Power/pyoframe
[#714]: https://github.com/PyPSA/linopy/issues/714
[#713]: https://github.com/PyPSA/linopy/issues/713
[#712]: https://github.com/PyPSA/linopy/issues/712
[#711]: https://github.com/PyPSA/linopy/issues/711
[#708]: https://github.com/PyPSA/linopy/issues/708
[#707]: https://github.com/PyPSA/linopy/issues/707
[#627]: https://github.com/PyPSA/linopy/issues/627
[#295]: https://github.com/PyPSA/linopy/issues/295
