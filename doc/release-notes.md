# Release Notes

## Version 1.0.0

linopy 1.0 makes the strict arithmetic convention the only behaviour and puts dense and sparse models on one storage layer. It adds rolling-horizon windows, long-form inputs and model building from mathspec programs. Read **Breaking changes** before upgrading.

!!! important

    **Upgrading from 0.10.** Run your test suite on 1.0, set `linopy.options["warn_on_absorbed_terms"] = True` to find terms that silently drop, fix each site and remove any `linopy.options["semantics"]` line. [Upgrading to linopy 1.0](user-guide/migrating-to-v1.md) lists every change with its fix. Netcdf files from linopy \<1.0 do not load; rebuild and save the model again.

**Rolling-horizon windows**

- [`Model.restrict()`][linopy.model.Model.restrict] cuts a subproblem out of a model, e.g. a time slice. `m.restrict(snapshot=window, fix=...)` returns a [`WindowModel`][linopy.restrict.WindowModel]; terms on variables outside the window move to the right-hand side at their `fix` values. `window.advance(...)` moves the window and `window.solve()` updates the solver model in place where possible. See [Restricting a Model](user-guide/restrict.md). ([\#768](https://github.com/PyPSA/linopy/issues/768))
- For production runs, `drop={name: mask}` switches constraint rows off in one window, `solve(log_fn=...)` writes a log per window and `solve_fixed_lp()` re-solves a MIP window with fixed integers to get duals. `to_file()` and `compute_infeasibilities()` debug a window with the parent's names. ([\#1130](https://github.com/PyPSA/linopy/issues/1130), [\#1131](https://github.com/PyPSA/linopy/issues/1131), [\#1132](https://github.com/PyPSA/linopy/issues/1132))

**Long-form inputs**

- `linopy.from_long(frame, coords, default=None)` turns a long-form pandas or polars frame (key columns plus one value column) into a `DataArray` to pass wherever linopy takes a constant: bounds, coefficients, right-hand sides, penalties. Missing key combinations raise unless `default` fills them, and so does malformed input. ([\#1049](https://github.com/PyPSA/linopy/issues/1049))
- `expr.groupby(linopy.Relation(frame, member, group)).sum()` sums over a many-to-many relation without an incidence matrix, e.g. `Relation(ports, ["link", "port"], "bus")` adds each link port to its bus. `Relation.group_of(coords)` maps member cells to their group, e.g. to read nodal prices per generator. The sum stays sparse on a sparse expression. ([\#1049](https://github.com/PyPSA/linopy/issues/1049))
- Both emit [`linopy.EvolvingAPIWarning`][linopy.EvolvingAPIWarning] once per session, as their signature may change in minor releases. See [Long-form inputs](user-guide/long-form-inputs.ipynb).

**Models from mathspec programs**

- `Model.from_spec(spec, sources)` builds a model from a [mathspec](https://github.com/energy-models/mathspec) program and its data. The spec is a `pathlib.Path`, YAML text, a mapping or a `mathspec.Spec`; the sources are a mapping keyed by declared name or an `xarray.Dataset`. Requires mathspec from its `main` branch, pinned to a commit after 0.4, and Python \>=3.12 (`pip install "linopy[spec]"`). Emits [`linopy.EvolvingAPIWarning`][linopy.EvolvingAPIWarning] once per session. See [Building models from mathspec programs](user-guide/building-models-from-specs.ipynb). ([\#1063](https://github.com/PyPSA/linopy/issues/1063), [\#1071](https://github.com/PyPSA/linopy/issues/1071))
- Each kind of spec data takes exactly one shape: a dimension is a flat sequence of unique labels, a parameter a frame with one column per dimension plus a value column (or a `DataArray`), a relation a frame with exactly its declared columns. Anything else raises a `SpecDataError` that names the shape to pass. A missing row raises only where a declaration reads it. See [Spec data and results](user-guide/spec-data.md).
- `from_spec(..., store=True)` keeps the spec's parameters and relations in `model.parameters`, so the model rebuilds from a netcdf file. `linopy.spec.evaluate(spec, model, *names)` evaluates named expressions over the solution. `linopy.testing.synthetic_sources` makes test data for a spec.
- A missing row takes the value that `missing:` sets, e.g. `missing: 0`, in bounds, expressions and conditions alike. A parameter's `where:` limits its coordinates, and a dimension's `one_of:` its labels. Built objects follow the name order of their dimensions, as mathspec sorts every `dims:` list. ([\#1137](https://github.com/PyPSA/linopy/pull/1137))
- Limits: a constraint row that absence leaves without terms raises, where mathspec skips it ([\#993](https://github.com/PyPSA/linopy/issues/993)). `store=True` stores only relations with one key and one value column. A product of two variables in a constraint is accepted only where the data leaves it no live row.

**Other features**

- `Solver.from_file(name, problem_fn, options=...)` solves an existing LP/MPS file without a linopy model and leaves the file untouched. ([\#996](https://github.com/PyPSA/linopy/pull/996))
- `LinearExpression.from_coo(model, coords, rows, vars, coeffs)` builds an expression from COO triplets (cell position, variable label, coefficient) on a coordinate grid.
- `@`/`dot` contract a `DataArray` backed by a [pydata/sparse](https://sparse.pydata.org) array without densifying. Where no sparse path exists, e.g. for a `QuadraticExpression`, it densifies with a `PerformanceWarning`; `warn_on_densify=False` silences it.
- `linopy.options["warn_on_absorbed_terms"] = True` emits an `AbsorbedTermsWarning` wherever an absent operand drops the present terms of another, e.g. in `p - r.reindex(unit=units)`. ([\#1135](https://github.com/PyPSA/linopy/issues/1135))
- `linopy.testing.assert_equivalent` compares two linear expressions independently of their backing (dense or sparse) and term layout. ([\#1044](https://github.com/PyPSA/linopy/issues/1044))
- The `Variable` methods that wrap xarray, e.g. `sel`, `isel` and `shift`, carry typed signatures and return `Variable` for type checkers and IDEs. ([\#1029](https://github.com/PyPSA/linopy/pull/1029))

**Breaking changes**

*Arithmetic convention*

- The legacy arithmetic semantics are removed. The [v1 convention](user-guide/design/convention.md), opt-in since 0.10, is the only behaviour: shared dimensions align exactly by label, absence propagates instead of counting as zero, and ambiguous unlabeled operands and conflicting auxiliary coordinates raise. Code that relied on legacy behaviour now raises or gives different results.
- `linopy.options["semantics"]` is deprecated: `"v1"` is a no-op with a `FutureWarning`, `"legacy"` raises. `LinopySemanticsWarning` is no longer emitted but stays importable for warning filters. Both are removed in 1.1.
- NaN is never read as data. `Model(auto_mask=...)` and `Model.auto_mask` are removed; NaN in coefficients, right-hand sides or variable bounds (`add_variables`, `Variable.update`, `fix`) raises. Pass `mask=` to exclude positions.
- A constant operation that yields NaN in a live cell, e.g. `0 * inf` or `0 / 0`, raises a `ValueError`. A cell whose constant becomes NaN, e.g. by `where`, `reindex` or a join with `fill_value=linopy.ABSENT`, drops its terms, so its variables no longer reach `sum()`, a constraint or the objective. Assigning a `const` with NaN or partial coverage raises. ([\#1020](https://github.com/PyPSA/linopy/issues/1020), [\#1045](https://github.com/PyPSA/linopy/issues/1045))
- `MultiIndex` coordinates are rejected everywhere, also in `Model.parameters`, constant operands and `assign_coords`, with a `reset_index` hint. `broadcast_to_coords` no longer projects inputs onto `MultiIndex` levels and drops `warn_reorder`.
- `expand_dims` puts the new dimension first. ([\#1078](https://github.com/PyPSA/linopy/pull/1078))

*Expression storage and sparse models*

- A `LinearExpression` holds its terms in a store from the new `linopy.store` module: a `DenseStore` or a `CSRStore`, renamed from `linopy.csr.CSRLinearExpression` without an alias. `Model.store_cls` names the store a model builds on; `Variable.to_linexpr` returns an expression on it. ([\#1020](https://github.com/PyPSA/linopy/issues/1020), [\#1045](https://github.com/PyPSA/linopy/issues/1045))
- An operation without a sparse path, including reading `.data`, works on a dense copy and leaves the expression sparse. The densify notice now reads `Sparse (CSR) backing densified: op(case)`, so update warning filters that match the old texts. Merging a dense with a sparse operand converts the dense one and emits `Dense backing sparsified: merge with a sparse operand`. Explicit conversions emit no notice. ([\#1020](https://github.com/PyPSA/linopy/issues/1020), [\#1045](https://github.com/PyPSA/linopy/issues/1045), [\#1047](https://github.com/PyPSA/linopy/issues/1047))
- `LinearExpression.where` and `QuadraticExpression.where` take `drop` as their only keyword argument.

*Constraints*

- `Constraint` holds its left-hand side as a `LinearExpression` next to the fields `sign`, `rhs`, `labels`, `scaling`, `dual`, `binary_var`, `binary_val` and `blocks`. `Constraint.data` is rebuilt on each access, so writing into it has no effect; use `Constraint.update(...)`. ([\#1046](https://github.com/PyPSA/linopy/issues/1046))
- The constructor takes these fields instead of a dataset: `Constraint(lhs, sign, rhs, model, ...)`. Build from a dataset with `Constraint.from_dataset(ds, model, name)`; `skip_broadcast` is removed. A field whose dimensions or coordinates differ from the lhs raises instead of broadcasting, and `assign` and the other wrapped xarray methods reject foreign data variables. ([\#1046](https://github.com/PyPSA/linopy/issues/1046))
- `CSRConstraint`, `ConstraintBase` and the freeze API (`add_constraints(freeze=...)`, `Constraint.freeze()`, `CSRConstraint.mutable()`, ...) are removed. A constraint in a sparse model is a `Constraint` on a `CSRStore` with the full API; replace `isinstance(c, CSRConstraint)` with `c.lhs.is_sparse`. `Constraint.to_sparse()` and `to_dense()` convert in place. ([\#1033](https://github.com/PyPSA/linopy/issues/1033), [\#1047](https://github.com/PyPSA/linopy/issues/1047))
- A sparse constraint keeps `rhs`, `sign`, `scaling` and `dual` on its full grid like a dense one; masked rows have label `-1`. ([\#1047](https://github.com/PyPSA/linopy/issues/1047))
- `Constraint.update` raises when a new `lhs` or `rhs` adds dimensions, and `update(variables=...)` accepts only a `Variable`. After `update(coeffs=...)` or `update(variables=...)`, `coeffs` and `vars` keep the `_term` dimension. `sanitize_zeros` removes near-zero terms, so `_term` can shrink. `set_blocks` stores the `blocks` field. ([\#1046](https://github.com/PyPSA/linopy/issues/1046), [\#1047](https://github.com/PyPSA/linopy/issues/1047))
- `Constraint.active_labels`, `to_matrix` and `to_polars` raise on a constraint without labels.

*Saved models*

- Netcdf files use one layout for every expression, constraint and objective, and each reads back on the store it was written with. `read_netcdf` refuses files from linopy \<1.0. ([\#1048](https://github.com/PyPSA/linopy/issues/1048))

*Removed APIs*

- Deprecated sparse opt-ins: `Model(freeze_constraints=...)`, `Model.freeze_constraints`, `groupby(...).sum(sparse=...)` and `linopy.options["sparse_groupby"]`. Use `Model(sparse=True)` and `Model.sparse`.
- The OETC remote handler (`OetcHandler`, `OetcSettings`, `OetcCredentials`) and the `oetc` extra. Use `linopy.remote.RemoteHandler` (SSH). ([\#1086](https://github.com/PyPSA/linopy/issues/1086))
- Long-deprecated shims:
    - `sum(dims=...)`: use `dim`. Unknown keywords raise a `TypeError`.
    - `merge(a, b)`: pass a list; `dim`, `cls`, `join` and `fill_value` are keyword-only. Pass a `Variable` as `var.to_linexpr()`.
    - `LinearExpression.from_tuples` with a `(const,)` tuple: pass the constant.
    - `.empty()`: `empty` is a `bool` property.
    - `Variable.sol`: use `Variable.solution`.
    - The `Variable.lower`/`upper` and `Constraint.coeffs`/`vars`/`sign`/`rhs`/`lhs` setters: use `update(...)`.
    - `Constraint.flat`: use `Constraint.to_polars()`.
    - `print_labels`, `Model.print_infeasibilities` and `Model.compute_set_of_infeasible_constraints`: use `format_labels`, `Model.compute_infeasibilities` and `Model.format_infeasibilities`.
    - `assign_multiindex_safe` on variables, expressions and constraints: use `assign`.
    - `LinearExpressionGroupby.map(shortcut=...)`, `Variables.dataset_names`, `Constraints.dataset_names`, `LinearExpression.sanitize` and `SolverReport.barrier_iterations`.

*Dependencies*

- `dask` is optional: install `linopy[dask]` for chunked arrays. `toolz`, `numexpr` and `deprecation` are dropped; `pandas` is declared explicitly. ([\#1032](https://github.com/PyPSA/linopy/pull/1032))

*Other*

- Setting `Objective.expression`, also through `add_objective`, stores a shallow copy and leaves the caller's expression untouched.
- `Model.to_file` and `Model.solve` raise when the file extension does not match `io_api`. Before, `io_api="mps"` with a `.lp` path wrote an LP file.
- `linopy.testing.assert_conequal(strict=False)` compares left-hand sides with `assert_equivalent`. ([\#1047](https://github.com/PyPSA/linopy/issues/1047))
- Low-level API: `Solver.apply_update` takes `(diff, n_vars, n_cons)`. `linopy.semantics` drops `is_v1`, `warn_legacy`, `legacy_warnings_muted` and `warn_legacy_absence`, and `check_user_nan` always raises. `linopy.common.assign_multiindex_safe` and `assign_coords_multiindex_safe` are renamed to `assign_fields` and `reassign_coords`; `set_int_index`, `infer_schema_polars` and `require_constant` are removed. `CSRStore` and `Grid` methods use pandas-style verbs, e.g. `summed` is `sum` and `reindexed` is `reindex`. ([\#1018](https://github.com/PyPSA/linopy/issues/1018))

**Deprecations**

- `add_piecewise_formulation(active_fill=...)` emits a `FutureWarning` and is removed in 1.1. Pad a partial gate instead, e.g. `active=status.reindex(gen=gens).fillna(1)`.

**Sparse models**

- More operations keep an expression sparse: arithmetic on variables (`2 * x`, `x + y`), merges of operands over a subset or superset of dimensions, `assign_coords` and `groupby` on auxiliary coordinates of the grid, vectorised `sel`, `isel`, `[]` and `where` over a new labelled dimension, `densify_terms`, `Constraint.update(rhs=...)`, `Model.assign_coords`, `add_expressions`, `Model.copy` and netcdf writing. ([\#1020](https://github.com/PyPSA/linopy/issues/1020), [\#1049](https://github.com/PyPSA/linopy/issues/1049))
- Variable operations without a sparse path, e.g. quadratic products, `rolling` and `cumsum`, build the dense expression directly, without a `PerformanceWarning`.
- Sparse merges and sums are faster: rows are sorted and deduplicated once, when a constraint is added. Like a dense expression, a sparse one may now hold the same variable more than once, and `nterm` counts the stored terms.
- A conformance suite checks that every operation on a sparse expression stays sparse or names its densify fallback, and that both backends agree. ([\#1044](https://github.com/PyPSA/linopy/issues/1044))

**Performance**

- Dense model build skips xarray alignment and copies when operands already share their coordinates: PyPSA's `create_model` builds about twice as fast.
- Gurobi direct API: the model loads in one `loadModel` call (gurobipy \>=12) and the solution and duals are read in bulk. A 1M-variable LP builds in 0.43 s instead of 1.8 s, with 150 MB instead of 655 MB peak memory.
- HiGHS direct API: the model passes in one `passModel` call.
- `Model.assign_coords` relabels by position: about 9 ms instead of 810 ms for 30 containers on a 4380-label tz-aware axis.
- `.sel` skips indexers that select a whole dimension in order, which cost about 10% of PyPSA's sparse build.
- `Model.matrices` and `Constraint.to_polars` allocate less. ([\#1058](https://github.com/PyPSA/linopy/issues/1058))

**Bug fixes**

*Model building*

- `add_constraints` no longer raises on invalid infinite right-hand sides at positions excluded by `mask`. ([\#1116](https://github.com/PyPSA/linopy/issues/1116))
- `add_constraints` rejects dimensions named after reserved fields (`sign`, `rhs`, ...), like `add_variables`.
- Multiplying a constant or one-term expression by a multi-term expression no longer raises a `TypeError`. ([\#1055](https://github.com/PyPSA/linopy/issues/1055))
- `QuadraticExpression.groupby(...).map` returns a `QuadraticExpression`. ([\#1064](https://github.com/PyPSA/linopy/issues/1064))
- `Variable.fix` accepts NaN or non-binary fix values at masked positions of a binary variable. `Variables.fix` validates all variables before fixing any.
- The repr of a long expression shows the last terms of a `QuadraticExpression` after the `...` and prints the constant once. ([\#1023](https://github.com/PyPSA/linopy/pull/1023))

*I/O*

- `to_netcdf` handles timezone-aware datetimes in indexes, coordinates and `Model.parameters`. ([\#1021](https://github.com/PyPSA/linopy/pull/1021))
- Object-dtype parameters, e.g. strings with `NaN` holes, round-trip through netcdf with their holes. Mixed-type object arrays raise a `TypeError` on write.

*Solvers*

- With `add_objective(..., scaling=...)`, the solve log reports objective and dual bound in model units. ([\#1077](https://github.com/PyPSA/linopy/issues/1077))
- CPLEX: "optimal with unscaled infeasibilities" is reported as `imprecise` and its solution is imported. ([\#368](https://github.com/PyPSA/linopy/issues/368))
- HiGHS: a solve that stops at a limit without a solution no longer returns all-zero values, and a MIP no longer returns all-zero duals. `log_fn` and solver options also apply to a re-solve of a direct-API model. ([\#1130](https://github.com/PyPSA/linopy/issues/1130))
- Mosek: `log_fn` no longer disables the console log. ([\#948](https://github.com/PyPSA/linopy/issues/948))
- SCIP no longer queries duals for MILPs, works again with PySCIPOpt \<5 and applies solver options before reading the problem file. ([\#1004](https://github.com/PyPSA/linopy/pull/1004))

**Documentation**

- The documentation moved from Sphinx to [MkDocs](https://www.mkdocs.org/) with the Material theme and the layout of the [PyPSA documentation](https://docs.pypsa.org). The example notebooks moved from `examples/` into `doc/`, and redirects keep the old links working.

**Internal**

- `Variables` and `Constraints` share a `LabelContainer` base, and the piecewise formulations share one link builder.
- `linopy.matrices.Matrices` is a frozen dataclass of a model's flat arrays with `take(rows, cols)`. The direct solver builders consume it and no longer mutate the model.

## Version 0.10.1

**Bug fixes**

- `read_netcdf` no longer drops parameters, parameter attributes or `MultiIndex` dimensions whose name contains `-`. ([\#1025](https://github.com/PyPSA/linopy/pull/1025))
- `to_netcdf`/`read_netcdf` keep the auxiliary (non-index) coordinates of dense variables, constraints, expressions and `Model.parameters`, such as the levels on a flat dimension that v1 recommends instead of a `MultiIndex`, and scalar coordinates left by a scalar selection. ([\#1024](https://github.com/PyPSA/linopy/pull/1024))
- `licensed_solvers` and `check_solver_licenses()` no longer report Xpress as licensed when its licence is rejected, e.g. a licence that only covers an older Xpress release. Xpress now probes its licence with `xpress.init()`. ([\#994](https://github.com/PyPSA/linopy/issues/994))

## Version 0.10.0

This release adds an opt-in strict arithmetic convention (v1), an end-to-end sparse model path, numerical scaling, constraint softening and the GPU solver NVIDIA cuOpt. Read **Breaking changes** before upgrading.

!!! important

    **Switch to v1 now.** linopy 1.0 will make the v1 semantics the only behaviour and remove the legacy path. Opt in with `linopy.options["semantics"] = "v1"`, fix every `LinopySemanticsWarning` your model emits, and follow [Upgrading to linopy 1.0](user-guide/migrating-to-v1.md).

**Strict "v1" arithmetic (opt-in)**

Enable it with `linopy.options["semantics"] = "v1"`. Legacy stays the default in 0.10 but is removed in 1.0, so we recommend switching now. Under legacy, every operation whose result changes under v1 emits a `LinopySemanticsWarning` that names the fix, so a model can migrate step by step. The warnings fire exactly where the two results diverge, once per operation, and point at the user's code. The full rules are in [the arithmetic convention](user-guide/design/convention.md). Under v1:

- **Exact alignment.** Shared dimensions align by label with `join="exact"`. Differing labels or a pure reorder raise. Resolve with `.sel` / `.reindex` / `.assign_coords`, or pass `join=` to `.add` / `.sub` / `.mul` / `.div` / `.le` / `.ge` / `.eq`.
- **Unlabeled operands.** Numpy arrays, lists and polars `Series` pair with dimensions by size. An ambiguous match (square array, equal-length dimensions) or no match raises. Wrap the operand in a `DataArray` to name its dimensions. (<https://github.com/PyPSA/linopy/issues/736>)
- **NaN constants raise** instead of being filled with a value that differed per operator.
- **Absence propagates.** Masked, reindexed and shifted-in slots stay absent through every operator instead of collapsing to zero.
- **Explicit fill on joins.** A reindexing `join=` fills the positions it creates per side: the linopy operand with the zero expression, the constant with the new `fill_value=` of `.add` / `.sub` / `.mul` / `.div`. The default means "does not apply": a missing divisor zeroes the term, and a missing numerator no longer gives `1 / divisor`. `fill_value=linopy.ABSENT` keeps all created positions absent, also in `linopy.merge`. (<https://github.com/PyPSA/linopy/issues/890>)
- **Conflicting auxiliary coordinates raise** instead of being dropped silently.
- **Aligned quadratic products.** `x * y` or `expr * expr` over shared dimensions with differing or reordered labels raises instead of an outer join by label. Align the factors first with `.sel` / `.reindex` / `linopy.align`, as the product takes no `join=`.
- **No MultiIndex dimensions.** Use a flat dimension with the levels as auxiliary coordinates.
- **Label-aligned groupby.** A grouper must match the grouped dimension's labels in set and order. A multi-key grouper gives a flat `group` dimension with the keys as auxiliary coordinates. (<https://github.com/PyPSA/linopy/issues/827>)

**Sparse models**

- **One switch.** `Model(sparse=True)` turns on the sparse (CSR, compressed sparse row) path for a model. `groupby(...).sum()` and `@`/`dot` against a constant return CSR-backed expressions, and `add_constraints` freezes every constraint unless `freeze=False` is passed. The key requires v1 and rejects `chunk`. `Model.copy` and the netcdf round trip keep it. ([\#976](https://github.com/PyPSA/linopy/issues/976))

- **Sparse groupby-sum.** It builds from the ungrouped terms, so no padded term rectangle exists and build memory no longer depends on the largest group: 6-9x less peak memory on the hub-bus case. Multi-key groupers work; `observed=True` keeps only observed key combinations. ([\#745](https://github.com/PyPSA/linopy/issues/745), [\#870](https://github.com/PyPSA/linopy/pull/870), [\#940](https://github.com/PyPSA/linopy/pull/940))

- **Stays sparse.** A CSR-backed expression is no longer densified by:

    - metadata reads: `repr()`, `shape`, `sizes`, `dims`, `coords`, `indexes`, `isnull()`, `const`, `has_terms` and `drop_vars` of auxiliary coordinates ([\#962](https://github.com/PyPSA/linopy/issues/962), [\#1003](https://github.com/PyPSA/linopy/pull/1003))
    - arithmetic with a non-scalar constant, aligned as on the dense path ([\#965](https://github.com/PyPSA/linopy/issues/965))
    - `shift`, `roll` and `diff`; `shift` with a `fill_value` still densifies ([\#1011](https://github.com/PyPSA/linopy/issues/1011))
    - `linopy.merge(..., dim=<grid dimension>)`, with the coordinates of the dense `xr.concat` ([\#1011](https://github.com/PyPSA/linopy/issues/1011))
    - `sum(dim=...)` and a further `groupby(...).sum()` ([\#964](https://github.com/PyPSA/linopy/issues/964))
    - `where`, `sel`, `isel`, `loc` and `[]`, with the dense semantics; masked cells become absent ([\#966](https://github.com/PyPSA/linopy/issues/966))
    - `linopy.merge` of differing grids with auxiliary coordinates ([\#966](https://github.com/PyPSA/linopy/issues/966))
    - `@`/`dot` against a constant, which returns the compact canonical form ([\#748](https://github.com/PyPSA/linopy/issues/748), [\#756](https://github.com/PyPSA/linopy/issues/756), [\#925](https://github.com/PyPSA/linopy/issues/925))
    - `add_constraints(..., mask=..., freeze=True)` and an expression on the right-hand side ([\#970](https://github.com/PyPSA/linopy/issues/970))
    - `flat`, `to_polars` and `solution` ([\#968](https://github.com/PyPSA/linopy/issues/968))
    - use as linear objective, through `matrices.c`, LP/MPS, netcdf, `Model.copy`, persistent snapshots and direct solver APIs ([\#967](https://github.com/PyPSA/linopy/issues/967))

    An operand or indexer that adds dimensions still densifies.

- **Visibility.** `LinearExpression.is_sparse` tells if an expression is CSR-backed, and its repr reads `LinearExpression (sparse)`. `linopy.options["warn_on_densify"]` emits a `PerformanceWarning` with the reason whenever a sparse backing is dropped. It is on by default in a sparse model, where it also flags a merge of operands over different dimensions. Explicit conversions such as `to_dense()` or `freeze=False` warn only when the option is `True`; `False` silences it. ([\#969](https://github.com/PyPSA/linopy/issues/969))

- **Guide.** The new [Dense and sparse models](user-guide/sparse-models.ipynb) guide covers how to opt in, which operations stay sparse and which densify. ([\#971](https://github.com/PyPSA/linopy/issues/971))

**Numerical scaling**

- Variables, constraints and the objective accept a `scaling` factor that moves the problem into better-behaved units for the solver. Variable scaling acts on columns, constraint and objective scaling on rows.
- Primal values, duals and the objective are mapped back to the original units after the solve. Solver tolerances apply to the scaled problem.
- See the [Numerical Scaling](user-guide/numerical-scaling.ipynb) tutorial and the *Numerical scaling* section of the [Overview](user-guide/overview.md).

**Constraint softening**

- `Constraint.soften(penalty, max_violation=None, name=None)` adds a slack variable to the `lhs` (a positive/negative pair for equality constraints) and a penalty term to the objective. It returns a `Slack` named tuple; the `slack` property gives it back later.
- `add_constraints(..., penalty=...)` softens the constraint on creation, also with `freeze=True`.
- A frozen `CSRConstraint` is softened natively by appending the slack terms to its sparse rows. `to_dense()`/`mutable()`, `Constraint.freeze()` and netcdf keep the slack. ([\#975](https://github.com/PyPSA/linopy/issues/975))
- Detached copies from `.mutable()`, `.sel()` or `.isel()` cannot be softened.

**New solver: NVIDIA cuOpt**

- GPU-accelerated [NVIDIA cuOpt](https://docs.nvidia.com/cuopt/) for linear, mixed-integer and convex quadratic problems, via `model.solve("cuopt", io_api="direct")`.
- Install with `pip install "linopy[gpu]"`. Requires Linux, an NVIDIA GPU of compute capability 7.0+ and a CUDA 12 driver (525.60.13+). See [GPU-Accelerated Solving](user-guide/gpu-acceleration.md) for supported problem classes and limitations.

**Other features**

- [`linopy.Model.assign_coords()`][linopy.model.Model.assign_coords] replaces coordinate values across a model (variables, constraints, expressions, parameters) without changing its shape, e.g. `m.assign_coords(snapshot=new_snapshots)`. Containers on a subset of the dimension are mapped by label. Typical use: advancing the window in rolling-horizon runs with the persistent solver interface. (<https://github.com/PyPSA/linopy/issues/767>)
- `add_piecewise_formulation(..., mask=...)` declares which breakpoint slots are real, for **ragged** curves with different breakpoint counts per entity. Under v1 the mask is required (`mask=x_pts.notnull()`) instead of reading NaN padding. (<https://github.com/PyPSA/linopy/issues/884>)
- `Model.compute_infeasibilities` supports HiGHS, via `Highs.getIis`. (<https://github.com/PyPSA/linopy/issues/956>)
- `Model.to_netcdf(compression=...)` applies one encoding to every array, so compression no longer needs linopy's internal variable names. `True` (default) means `{"zlib": True, "complevel": 3}`, `False` turns it off, and explicit `encoding` entries win. ([\#1007](https://github.com/PyPSA/linopy/pull/1007))

**Performance**

*Model build*

- `@`/`dot` against a constant with zeros drops the zero terms instead of keeping one term per contracted member: 852 to 3 terms on PyPSA's KVL (Kirchhoff Voltage Law) constraint. ([\#748](https://github.com/PyPSA/linopy/issues/748))
- Sparse `@`/`dot` runs each chunk on only the variables it uses, which removes seconds of overhead on models with millions of variables. Chunks are sized by nonzeros. ([\#990](https://github.com/PyPSA/linopy/pull/990))
- Sparse `shift` on an uneven `groupby` result (2000 x 720 cells, 800 terms wide) takes 14 ms and 54 MB instead of 1 s and 4.4 GB. ([\#1011](https://github.com/PyPSA/linopy/issues/1011))
- `densify_terms` (`sum(drop_zeros=True)`) is vectorised: 127 s to 3 ms on a (2000 x 60) expression.
- An N-way `linopy.merge` of CSR-backed expressions runs in one pass and scales with the nonzeros, not quadratically with the operand count: about 30% faster for three operands. A sparse `reindex` that only reorders cells gathers the rows directly, about 4x faster. ([\#1010](https://github.com/PyPSA/linopy/issues/1010))
- `add_constraints` on a sparse model no longer copies the lhs matrix or expands the mask and scaling over the full grid: about half the peak memory. ([\#977](https://github.com/PyPSA/linopy/issues/977))
- Freezing a dense constraint builds the CSR matrix directly and sorts only rows with more than one term: 465 to 185 ms and 1.74 to 0.25 GB on a 2M-variable nodal balance. ([\#1009](https://github.com/PyPSA/linopy/issues/1009))

*Matrices, export and solvers*

- `Model.matrices` assembles constraint blocks in one pass into preallocated CSR buffers, without full-grid scaling lookups or `eliminate_zeros`: 45% faster with 200 MB less peak memory on an 8M-row model, with identical output. ([\#1008](https://github.com/PyPSA/linopy/issues/1008))
- Duals of frozen constraints are read back on the active rows only. A masked freeze gathers rows once, and mixed-sign senses convert vectorised.
- The LP writer writes constraints about 2x faster with byte-identical output.
- The MOSEK direct API builds bounds and names vectorised.
- Direct solver APIs no longer set variable and constraint names by default, which made `to_highspy` about 2x faster (see **Breaking changes**). ([\#978](https://github.com/PyPSA/linopy/issues/978))
- Persistent snapshots of tz-aware `DatetimeIndex` coordinates store UTC-ns arrays plus the timezone: capture ~24x and warm-start diffs ~33x faster. ([\#960](https://github.com/PyPSA/linopy/pull/960))

**Bug fixes**

*Copy and I/O*

- `Model.copy` (and `copy.copy`/`copy.deepcopy`) no longer turns a quadratic objective into a linear one. `assert_model_equal` now compares the objective type. ([\#903](https://github.com/PyPSA/linopy/issues/903))
- `to_netcdf`/`read_netcdf` keep a quadratic objective quadratic. ([\#903](https://github.com/PyPSA/linopy/issues/903))
- `read_netcdf` restores containers in insertion order, so a round trip no longer permutes `matrices.A`. `assert_model_equal` now compares the order. ([\#934](https://github.com/PyPSA/linopy/issues/934))
- `read_netcdf` no longer fails when a frozen constraint has a dimension or level name with `-`.
- `Model.copy` keeps frozen constraints frozen; `deep=True` copies their arrays and a CSR-backed objective. ([\#981](https://github.com/PyPSA/linopy/issues/981))

Files from earlier versions load as before.

*Frozen constraints*

- Adding or removing variables after `freeze=True` no longer breaks `model.matrices` with `inconsistent shapes`. ([\#926](https://github.com/PyPSA/linopy/issues/926))
- A `CSRConstraint` from a CSR-backed expression keeps its auxiliary coordinates through `to_dense()`, `freeze()` and netcdf. ([\#941](https://github.com/PyPSA/linopy/issues/941))
- Unsupported `CSRConstraint` operations (`loc`, `update`, `from_rule`, setting `coeffs`/`vars`/`sign`) raise an `AttributeError` that points to `.mutable()`. ([\#963](https://github.com/PyPSA/linopy/issues/963))
- Repeated matrix builds on an unchanged model reuse the cached column mapping again, so persistent snapshot diffs skip untouched frozen constraints. ([\#933](https://github.com/PyPSA/linopy/issues/933))

*Expressions*

- `sum()` no longer raises when another dimension has size 0. (<https://github.com/PyPSA/linopy/issues/906>)
- A multi-key `groupby` returns groups sorted by key, not in an order that changed with `PYTHONHASHSEED`.
- `linopy.merge` with `join="outer"`/`"left"`/`"right"` no longer raises a `MergeError` when only one operand has an auxiliary coordinate on a joined dimension. The same holds for `.add` / `.sub` / `.mul` / `.div` with a reindexing `join=`.
- `densify_terms` works on expressions without coordinate dimensions and on `QuadraticExpression`.

*Solvers and options*

- `Variable.get_solver_attribute` for Gurobi maps values by position. It returned values of the wrong variables when names were off and a variable was masked. ([\#978](https://github.com/PyPSA/linopy/issues/978))
- `Solver.close()` drops the solver model before its environment. After a file-based COPT or MindOpt solve, `model.solver_model` is `None` instead of a handle into freed memory. ([\#899](https://github.com/PyPSA/linopy/pull/899))
- The `linopy.options` context manager restores the values active on entry, not the defaults.

**Breaking changes**

- **No automatic solver release.** Garbage collection no longer closes the solver, as the finalizer could abort the interpreter. Call `model.solver.close()`, set `model.solver = None` or use `contextlib.closing(model.solver)`. A new `solve()` still closes the previous solver. ([\#899](https://github.com/PyPSA/linopy/pull/899))
- **No solver names by default.** `Model.set_names_in_solver_io` defaults to `False` for `to_highspy`, `to_gurobipy`, `to_mosek` and `to_xpress`. `solve()` still sets names with `warmstart_fn` or `basis_fn`. Pass `set_names=True` or `Model(set_names_in_solver_io=True)` to keep them. ([\#978](https://github.com/PyPSA/linopy/issues/978))
- **Compressed netcdf by default.** `Model.to_netcdf` compresses with zlib level 3; pass `compression=False` for the old output. ([\#1007](https://github.com/PyPSA/linopy/pull/1007))
- **Renamed sparse internals.** `linopy.sparse_expression` is now `linopy.csr`, `CSRExpression` is `CSRLinearExpression` and `CSRConstraint.from_mutable` is `CSRConstraint.from_dense`.
- **Other API changes.** `LinearExpressionGroupby(data=...)` is now `LinearExpressionGroupby(obj=...)`. The objective name lives on `Objective.name`/`Objective.attrs`, not in the expression's attributes ([\#967](https://github.com/PyPSA/linopy/issues/967)). `matrices.clabels` and `matrices.indicator_binvar` are `int32` by default.

**Deprecations**

- In favour of `Model(sparse=True)`, with a `FutureWarning`: `Model(freeze_constraints=...)`, the `Model.freeze_constraints` setter, `groupby(...).sum(sparse=...)` and `linopy.options["sparse_groupby"]`. They keep working until linopy 1.0 removes them together with the legacy semantics; only `@` ignores `sparse_groupby` already. ([\#976](https://github.com/PyPSA/linopy/issues/976))

**Documentation**

- The example notebooks use v1 semantics. The coordinate-alignment and expression tutorials teach strict label-based alignment and explicit `join=`.

## Version 0.9.1

**Features**

*Named expressions*

- You can now store expressions in the model and get them back by name. `Model.add_expressions` registers a `LinearExpression` or `QuadraticExpression` under a name (auto-named `expr0`, `expr1`, ... if you don't pass one). Registered expressions live in `Model.expressions`, a container that works like `Model.variables` and `Model.constraints`, and are removed with `Model.remove_expressions`. They survive `Model.to_netcdf`/`linopy.read_netcdf`, `Model.copy`, `copy.copy`, `copy.deepcopy` and pickling. ([\#882](https://github.com/PyPSA/linopy/pull/882))

*SOS constraints*

- `Model.add_sos_constraints` now works with any coordinates on the SOS dimension, not only numeric ones. ([\#893](https://github.com/PyPSA/linopy/pull/893))

**Performance**

- `Model.remove_variables` is 2-5x faster. The masked labels are filtered once per removal instead of once per constraint group, membership is tested against the contiguous label range rather than by a sort-based `isin`, and CSR-backed constraints are matched on term positions instead of gathering their labels. ([\#895](https://github.com/PyPSA/linopy/pull/895))
- `LinearExpression.reindex` keeps a sparse (CSR-backed) expression sparse for plain label changes — reorder, add, or drop coordinates — instead of expanding to the dense rectangle. New coordinates become absent cells and the result matches the dense reindex, so a `groupby(sparse=True) → reindex → merge` chain stays sparse and the build peak stays low (v1 only; other arguments fall back to dense). ([\#932](https://github.com/PyPSA/linopy/issues/932))
- `linopy.merge` (and `+` / `-` / `.add` / `.sub` with an explicit `join=`) keeps sparse (CSR-backed) expressions sparse when the operands live on different label subsets of the same dimensions, e.g. a nodal balance summing grouped generator, line and load terms. The CSR expressions are aligned row-wise onto the joined grid (`outer` / `inner` / `left` / `right` / `override`) instead of falling back to the dense rectangle; the positions the join creates carry the same fill as the dense path (zero, or absent with `fill_value=linopy.ABSENT`) and the result equals the dense one. Dense operands on a different grid are converted on the fly. ([\#749](https://github.com/PyPSA/linopy/issues/749))

**Bug fixes**

- An SOS set is now ordered by declaration, not by the values of its coordinates. Labels are names, but they were handed to the solver as SOS weights, so element-for-element identical models could reach different optima — silently changing who is adjacent in a `sos_type=2` set — just because their coordinates were named differently. **Behaviour change:** ascending coordinates are unaffected (this covers every piecewise formulation), descending ones now run in reverse with the same adjacency, and only a `sos_type=2` set whose numeric coordinates neither ascend nor descend changes meaning — that case now warns, and sorting the index restores the old order. ([\#892](https://github.com/PyPSA/linopy/issues/892), [\#893](https://github.com/PyPSA/linopy/pull/893))
- `Model.remove_variables` now ignores the removed variable's masked entries. A masked variable has no label, stored as `-1` — the same marker a constraint uses for an unused term slot — so it looked used by every constraint with a free slot, and models built with `mask=` could silently lose constraints and solve to a wrong optimum. ([\#883](https://github.com/PyPSA/linopy/issues/883), [\#886](https://github.com/PyPSA/linopy/pull/886), [\#895](https://github.com/PyPSA/linopy/pull/895))
- `Model.remove_variables` no longer raises an `IndexError` on a quadratic objective. A quadratic term is dropped when any of its factors uses the removed variable. ([\#883](https://github.com/PyPSA/linopy/issues/883), [\#895](https://github.com/PyPSA/linopy/pull/895))
- The Xpress interface no longer calls API functions deprecated in Xpress 9.8. ([\#880](https://github.com/PyPSA/linopy/pull/880))
- The `docs` extra no longer pins `numpy<2`. ([\#869](https://github.com/PyPSA/linopy/pull/869))
- Using timezone-aware `DatetimeIndex` coordinates in a model no longer raises a `ValueError`. The index is now retrieved directly via `DataArray.to_index()` instead of rebuilding it with `pd.Index()`, which dropped the timezone info and failed during alignment. ([\#898](https://github.com/PyPSA/linopy/issues/898))

## Version v0.9.0

**Features**

*In-place solver updates (persistent re-solve)*

- A built solver can now be re-solved against a mutated `Model` without a full rebuild. Construct with `Solver.from_name(..., track_updates=True)` and re-call `solver.solve(model)` after edits — the diff against the previous build is applied in place when the backend supports it, falling back to a rebuild otherwise. Supported on HiGHS, Gurobi, Xpress, and Mosek (`io_api="direct"`). ([\#718](https://github.com/PyPSA/linopy/pull/718))
- Pass `disallow_rebuild=True` to `solve(model, ...)` to guarantee an in-place update or raise `RebuildRequiredError`. Inspect `solver._last_rebuild_reason` (a `RebuildReason`, or `None` after an in-place update) to understand why a rebuild was triggered. ([\#718](https://github.com/PyPSA/linopy/pull/718))
- New `linopy.persistent` module exposes `ModelSnapshot`, `ModelDiff`, and `RebuildReason` for users who want to introspect or build the diff themselves. `ModelDiff.from_snapshot` / `from_models` return the `RebuildReason` directly when the change cannot be applied in place. ([\#718](https://github.com/PyPSA/linopy/pull/718))

*Improved IO*

- `Model.to_netcdf` now records the writing linopy version in the `_linopy_version` dataset attribute. Files written by older versions (without the attribute) continue to read unchanged. ([\#780](https://github.com/PyPSA/linopy/pull/780))

*Other*

- Default internal integer labels to `int32`, cutting memory ~25% and speeding up model build 10-35%. Models exceeding the int32 maximum (~2.1 billion labels) widen to `int64` automatically with a `UserWarning`; pass `Model(dtypes={"labels": np.int64})` upfront to avoid the mid-build upcast (exposed read-only via `Model.dtypes`). ([\#566](https://github.com/PyPSA/linopy/pull/566))
- `add_variables(binary=True, ...)` now accepts `lower`/`upper` bounds, as long as they are 0 or 1. Previously binary bounds could only be set via the `.lower`/`.upper` setters after creation. ([\#776](https://github.com/PyPSA/linopy/issues/776))
- `add_piecewise_formulation` gained an `active_fill` parameter that gates a partial `active` (defined over a subset of the indexed dimension, or masked) as always-active (`1`) or always-off (`0`); without it, a partial `active` — which was previously zeroed silently — now raises. Useful when one formulation mixes gated and ungated entities (e.g. committable and non-committable units sharing a `status`). `active_fill` is transitional and will be removed once v1 semantics make `active.reindex(coords).fillna(value)` sufficient. ([\#796](https://github.com/PyPSA/linopy/issues/796))

**Performance**

- `LinearExpression.groupby(...).sum()` now scatters terms directly into the padded result arrays via `xarray.apply_ufunc`, avoiding intermediate copies and speeding up the grouping. A single kernel covers both numpy and chunked (dask) data, the latter staying lazy. On representative models this lowers build and export peak memory by up to ~3x. The kernel emits the grouped result in its final axis order in one contiguous allocation; on dask inputs the reduction now runs over a single chunk (it no longer parallelises over the surviving dimensions). ([\#802](https://github.com/PyPSA/linopy/pull/802))

**Deprecations**

- Mutation via assignment to `Variable.lower` / `Variable.upper` / `Constraint.coeffs` / `Constraint.vars` / `Constraint.lhs` / `Constraint.sign` / `Constraint.rhs` is deprecated and emits a `DeprecationWarning`. Use `Variable.update(...)` / `Constraint.update(...)` instead — the canonical mutation API with one validation path and one place that flips the persistent-solver dirty flag. Read access to these properties is unchanged. The setters will be removed in a future release. ([\#718](https://github.com/PyPSA/linopy/pull/718))
- Passing a raw `DataArray` of integer labels to `Constraint.vars = ...` setter is deprecated and emits a `FutureWarning`. Pass a `Variable` to `Constraint.update()` instead — it is the supported input. The `DataArray` path will be removed in a future release. ([\#718](https://github.com/PyPSA/linopy/pull/718))

**Bug fixes**

- Fix GLPK objective parsing. ([\#818](https://github.com/PyPSA/linopy/pull/818))
- LP file export now honors bounds tightened below `[0, 1]` on a binary variable via the `.lower`/`.upper` setters after creation (e.g. `upper = 0`). Previously such bounds were written only by `io_api="direct"` and dropped by `io_api="lp"`. ([\#776](https://github.com/PyPSA/linopy/issues/776))
- Freezing an empty constraint group (e.g. an empty `isel` slice) no longer raises `ValueError: cannot reshape array of size 0`. `Model(freeze_constraints=True)` and `Constraint.freeze()` now round-trip zero-row constraints losslessly. ([\#783](https://github.com/PyPSA/linopy/pull/783))
- `Variable.where` no longer raises `ValueError: exact match required for all data variable names` once a solution is attached (after `Model.solve`) or the variable is fixed. The fill value now covers auxiliary data variables (`solution`, stashed bounds) instead of only `labels`/`lower`/`upper`. ([\#790](https://github.com/PyPSA/linopy/pull/790))
- `LinearExpression.groupby(...).sum()` with a multi-dimensional `DataArray` grouper now reduces over all of the grouper's dimensions on the default (fast) path, instead of leaking one of them into the result. ([\#824](https://github.com/PyPSA/linopy/pull/824))
- `LinearExpression.groupby(...).sum()` now keeps the grouped dimension's position, replacing it in place like xarray's native groupby-reduce, instead of moving the group dimension to the trailing position (regressed in 0.8.0). Both the default and `use_fallback=True` paths are fixed. ([\#860](https://github.com/PyPSA/linopy/pull/860))
- `LinearExpression.groupby(...).sum()` now raises when a grouper's labels are reordered or a different set relative to the expression, instead of silently regrouping by position. Reorder the grouper to match the expression's coordinates before grouping. ([\#827](https://github.com/PyPSA/linopy/issues/827))
- `linopy.testing.assert_linequal` now aligns dimension order before comparing, so mathematically identical expressions built in different orders (e.g. `x + y` versus `y + x`, which inherit different dimension orders from xarray broadcasting) are correctly treated as equal. Genuinely different expressions still fail. ([\#801](https://github.com/PyPSA/linopy/pull/801))
- Summing an expression over a dimension that carries an auxiliary (non-dimension) coordinate no longer leaks that coordinate onto the internal term dimension, where it broke later arithmetic with a `CoordinateValidationError`. Auxiliary coordinates on the remaining dimensions still propagate. ([\#295](https://github.com/PyPSA/linopy/issues/295))
- `Solver.close()` (also triggered by `model.solver = None` and the next `solve()` call) now explicitly disposes the `gurobipy` model before the environment. Previously the model was only dereferenced, so a user-held `model.solver_model` reference silently kept the Gurobi license acquired after `close()`. ([\#459](https://github.com/PyPSA/linopy/issues/459))

## Version 0.8.0

**Features**

*Constraints — CSR-backed storage*

- Add `CSRConstraint`: a memory-efficient immutable constraint representation backed by scipy CSR sparse matrices. Up to 90% memory savings for constraints with many terms and 30–120× faster matrix generation for direct solver APIs.
- Opt in globally via `Model(freeze_constraints=True)` or per-call via `model.add_constraints(..., freeze=True)`.
- Lossless conversion both ways with `Constraint.freeze()` / `CSRConstraint.mutable()`.

*Inspect the solver after solving*

- After `model.solve()`, the solver object stays available on `model.solver`. You can inspect it, reuse it, or release the underlying solver (and its license) by calling `model.solver.close()` or assigning `model.solver = None`. It is also released automatically when the model is garbage-collected.
- New `SolverReport` on the result (`result.report`) reports runtime, MIP gap, dual (best) bound, and iteration counts. It is shown in `repr(result)` and currently populated by CBC, HiGHS, Gurobi, Knitro, and cuPDLPx.

*Dualize LP models*

- New `Model.dualize()` constructs the LP dual as a standalone model, lifting finite variable bounds into explicit constraints so they are reflected in the dual. Dual variables are named after the primal constraints. Works for linear problems with linear objective and constraints. ([\#626](https://github.com/PyPSA/linopy/pull/626))

*A new way to call a solver (advanced)*

Most users should keep calling `model.solve(...)`. If you want more control, you can now build the solver yourself and run it in two steps:

```python
solver = Solver.from_name("gurobi", model, io_api="direct", options=...)
result = solver.solve()
model.assign_result(result)  # write the solution back
```

`Solver` is now a dataclass, so writing a new solver backend is simpler — subclasses just override the hooks they need (`_build_direct`, `_run_direct`, `_run_file`).

*Querying solver capabilities*

- Ask a solver class what it can do via `Gurobi.supports(SolverFeature.MIP)` (or any other `SolverFeature`). `SolverFeature` is importable from `linopy`.
- `linopy.solver_capabilities` still works (re-exports `SolverFeature` and `solver_supports`), but the new `SolverClass.supports(...)` API is preferred.

*Knowing which solvers you can actually use*

- `linopy.available_solvers` no longer tries to acquire licenses at import time, so importing linopy is faster and doesn't grab a license from solvers like Gurobi or Mosek. **Note:** membership now means "the package is installed", not "I have a working license" (see Breaking Changes). Call `available_solvers.refresh()` to re-scan. Same for `quadratic_solvers`.
- New `linopy.licensed_solvers`: the subset of installed solvers that currently pass a license check. Handy in tests and for picking a solver at runtime.
- New helpers for explicit license checks: `linopy.solvers.check_solver_licenses("gurobi", "mosek")`, `Gurobi.license_status()`, `Gurobi.is_available()`. They return a `LicenseStatus` dataclass (`name`, `ok`, `message`).

*Constraints — indicator constraints*

- Add indicator constraints for solvers that support them. They are part of the unified constraints container: `model.add_indicator_constraints` returns a `Constraint` and the constraint is stored in `model.constraints` (filterable via `model.constraints.indicator` / `model.constraints.regular`), so it round-trips through netCDF and `model.copy()`.

*Compact multi-key grouping*

- `LinearExpressionGroupby.sum` gains a pandas-style `observed` parameter for grouping by a list of coordinate names: `expr.groupby(["period", "season"]).sum(observed=True)` keeps the result stacked over only the observed key combinations (a `MultiIndex` `group` dimension) instead of unstacking into one dimension per key, which materialises the dense cartesian grid. The default `observed=False` mirrors xarray. When the grid would be mostly fill values, a `UserWarning` points to `observed=True`.
- The CSR-backed sparse groupby-sum (`sum(sparse=True)` or `linopy.options["sparse_groupby"]`, v1 semantics) now accepts multi-key groupers: a list of coordinate names or a pandas `DataFrame`. With `observed=True` (always for a `DataFrame`) the CSR result stays compact over the observed key combinations only, with neither the cartesian grid nor the group-size term padding; with `observed=False` it keeps one dimension per key, absent combinations being empty cells ([\#757](https://github.com/PyPSA/linopy/issues/757)).

*Other additions*

- Add `BaseExpression.has_terms` property: boolean array, true at slots with at least one live term ([\#741](https://github.com/PyPSA/linopy/issues/741)).
- Add `BaseExpression.variable_names` property, and documentation for `LinearExpression.where` with `drop=True`.

**Performance**

- ~10× faster direct solver communication (`io_api="direct"`), thanks to the new CSR-based matrix construction. Conversion helpers like `to_highspy` benefit too.
- Writing the solution back to the model after solving is faster: it no longer rebuilds the constraint matrix, and now uses positional (rather than label-based) indexing — roughly 2× faster overall.
- Xpress now supports `io_api="direct"`: the linopy model is loaded via the native `loadproblem` array API instead of being serialised through an LP/MPS file, with SOS constraints attached in-place. Adds `model.to_xpress()` matching the existing `to_gurobipy` / `to_highspy` / `to_mosek` helpers.

**Deprecations**

- `Solver.solve_problem`, `Solver.solve_problem_from_model`, and `Solver.solve_problem_from_file` still work but emit a `DeprecationWarning`. Use `Solver.from_name(...).solve()` (or simply `model.solve(...)`) instead. They will be removed in a future release.
- **Implicit MultiIndex-level projection is deprecated.** Passing an input indexed by a *level* of a stacked-`MultiIndex` dimension (e.g. per-`period` bounds onto a `(period, timestep)` `snapshot` index) emits an `EvolvingAPIWarning` — in arithmetic and in `add_variables` / `add_constraints` — and will raise under the upcoming v1 convention. Project the input onto the dimension explicitly (select with the dimension's level values) to keep current behavior. Affects PyPSA multi-investment models. See the level-projection entry under Bug Fixes for the new alignment behavior.

**Bug Fixes**

- `add_piecewise_formulation(mask=...)` no longer pins fully masked entities to zero: the `pwl_convex`, `pwl_link` (and disjunctive selection) rows get label `-1` for entities without any present breakpoint, so their variables stay free. ([\#1090](https://github.com/PyPSA/linopy/issues/1090))
- **⚠ Behavior change:** `add_variables` / `add_constraints`: extends 0.7.0's coords-as-truth rule to `lower`, `upper` and `mask` for every bound type and dim order. Pandas `Series` / `DataFrame` bounds or masks missing a dimension are now broadcast to `coords` instead of being silently dropped ([\#709](https://github.com/PyPSA/linopy/issues/709)) — a model that previously *ignored* such a partial bound now *applies* it, silently, with no error — review partial pandas bounds/masks when upgrading. The variable's dimension order always follows `coords` ([\#706](https://github.com/PyPSA/linopy/issues/706)); bare-tuple coord entries (`coords=[(0, 1, 2)]`) now behave like lists. Mismatched values or extra dims raise `ValueError` with a labelled message; sparse-coord masks (formerly a v0.6.3 `FutureWarning`, [\#580](https://github.com/PyPSA/linopy/issues/580)) raise `ValueError`, and masks with dims not in the data raise `ValueError` instead of `AssertionError`.
- **⚠ Behavior change:** Fix Mosek interface to inspect both the basic and IPM solutions and pick the one with the better status, so that an optimal crossover solution is not discarded when IPM terminates with a (near-)Farkas certificate. Mosek may now return a different (better-status) solution than 0.7.0 for the same model.
- Pandas inputs whose index names *levels* of a stacked-`MultiIndex` `coords` dimension are now projected onto that dimension: a level subset broadcasts across the others, the full set aligns element-wise. In `add_variables` / `add_constraints` the input must provide a value for every level combination of the MultiIndex or a `ValueError` is raised (the error lists the missing combinations); aligning the full level set with full coverage stays silent. Strict validation also rejects a `MultiIndex` input with *unnamed* levels whose combinations don't match `coords` (previously a silent bypass, as such inputs can't be projected by level name). Implicit level projections are deprecated (see Deprecations).
- `LinearExpression.groupby` now accepts a **non-dimension** coordinate as the key -- by name (`expr.groupby("period").sum()`, where `period` labels another dimension) or as the coordinate `DataArray` -- which previously raised `ValueError: ... already exists`. Grouping by a dimension or a `MultiIndex` level already worked ([\#750](https://github.com/PyPSA/linopy/issues/750)).
- `LinearExpression.groupby` with a **list of coordinate names** (`expr.groupby(["period", "season"]).sum()`) now takes the fast reindex path instead of silently falling back to the slow xarray implementation, returning one dimension per key as before ([\#753](https://github.com/PyPSA/linopy/issues/753)). See `observed` under Features to keep the result compact instead.
- SOS constraints on masked variables no longer cause solver-specific failures (Gurobi `IndexError`, Xpress `?404 Invalid column number`, LP parse errors, silent set corruption). `Model.solve()` and `Model.to_file()` now raise a clear `NotImplementedError` referring users to [\#688](https://github.com/PyPSA/linopy/issues/688); pass `reformulate_sos=True` as a workaround.
- `Model.solve(..., reformulate_sos=True)` now actually reformulates SOS constraints even when the solver supports them natively. Previously it was silently ignored with a warning.
- `add_piecewise_formulation` now produces a reproducible dimension order in the broadcast breakpoint array. The previous set-based expansion gave a hash-randomized order that varied between processes.
- `Variable.fix(value)` now places `value` correctly fix binary variables and correctly work on variables with named dimensions; previously array values could be misaligned.

**Breaking Changes**

- `result.solution.primal` and `result.solution.dual` are now `numpy` arrays indexed by linopy's integer labels (with `NaN` for slots without a value), instead of pandas Series keyed by variable/constraint name. If you accessed them by name, use `model.variables[name].solution` (or `model.constraints[name].dual`) instead.
- `Model.solver_model` and `Model.solver_name` are now read-only properties that delegate to `model.solver`. You can't reassign them (only `= None` is allowed, which closes the solver), and `solver_name` is `None` before the first solve.
- `available_solvers` now lists all *installed* solvers, even ones without a working license. If you used it to decide "can I actually solve with X?", switch to `linopy.licensed_solvers` or `SolverClass.license_status()`.
- Drop Python 3.10 support. Minimum supported version is now Python 3.11.
- `add_variables` / `add_constraints`: the v0.6.3 `mask` deprecations ([\#580](https://github.com/PyPSA/linopy/issues/580)) are now hard `ValueError`s; an unnamed `pd.MultiIndex` in sequence-form `coords` raises `TypeError` unless paired with `dims=[i]`. See Bug Fixes above.
- Sequence-form `coords` entries can no longer be `xarray.DataArray` objects — they raise `TypeError`. Pass the underlying index instead: `variable.indexes[dim]` (a `pd.Index`).

**Internal**

- New module `linopy.alignment` owns conversion, broadcasting, and alignment of user input against coordinates (moved out of `linopy.common`): `as_dataarray` (convert only), `broadcast_to_coords` (convert and broadcast against `coords`; `strict=True` by default raises on any mismatch, naming `label` in the error), `validate_alignment`, and `align`.
- Each `Solver` subclass now overrides at most three hooks: `_build_direct` (build the native model), `_run_direct` (run it), and `_run_file` (run the solver on an LP/MPS file). File-only solvers (CBC, GLPK, CPLEX, SCIP, Knitro, COPT, MindOpt) only override `_run_file`.
- New `ConstraintLabelIndex` cached on `Model.constraints` (mirrors the existing `Variables.label_index`); `ConstraintBase` gains `active_labels()` and a `range` property; `CSRConstraint` exposes `coords`.
- `linopy.common` gains `values_to_lookup_array`; the legacy pandas-based helpers `series_to_lookup_array` and `lookup_vals` are removed.
- `model.to_gurobipy()` / `model.to_highspy()` / `to_cupdlpx(model)` (and similar) all return the underlying solver model as before; internally they now go through `Solver.from_model(model, io_api="direct")`. No user-visible change.
- Adopt Python 3.11 type-syntax: the status enums (`ModelStatus`, `SolverStatus`, `TerminationCondition`) are now `StrEnum`, and classmethods plus the expression base class use `Self` instead of string forward-references and a self-typed `TypeVar`. No user-visible change — `Model.solve()` still returns `(status, termination_condition)` as plain strings.
- `Variable.fix()` now fixes a variable by collapsing its bounds (`lower = upper = value`) instead of adding a `__fix__` equality constraint; `unfix()` restores the original bounds ([\#769](https://github.com/PyPSA/linopy/issues/769)). A fix outside the current bounds now warns and overrides instead of raising, and its shadow price appears as the variable's reduced cost rather than a constraint dual.

## Version 0.7.0

**Features**

*Piecewise linear constraints (new)*

- `Model.add_piecewise_formulation((power, x_pts), (fuel, y_pts))` adds piecewise constraints with SOS2, incremental, disjunctive, or pure-LP formulations and automatic method dispatch. Supports N-variable linking (e.g. CHP) and per-entity breakpoints; emits [`linopy.EvolvingAPIWarning`][linopy.EvolvingAPIWarning] while the API stabilises.
- One-sided bounds: append `"<="` / `">="` to a tuple, e.g. `(fuel, y_pts, "<=")`. On matching convex/concave curves this dispatches to a pure-LP chord formulation.
- Unit-commitment gating via `active`: when zero, deactivates the piecewise relation.
- `PiecewiseFormulation` exposes `.method` / `.convexity` (persisted across netCDF round-trip).
- Construction helpers: `linopy.breakpoints()`, `linopy.segments()`, `linopy.Slopes` for per-piece slopes, and `tangent_lines()`.

*Variables*

- `fix()` / `unfix()` / `fixed` for fixing variables to values via equality constraints (rounds integers/binaries).
- `relax()` / `unrelax()` / `relaxed` for LP relaxation; supports partial relaxation (e.g. `m.variables.integers.relax()`).
- Semi-continuous variables on solvers that support them.

*Model*

- `Model.copy()` for a deep copy of a model, optionally including the solution; supports the `copy` protocol.
- SOS1 / SOS2 reformulations for solvers without native SOS, applied automatically by `Model.solve()` when needed.
- `format_labels()` / `format_infeasibilities()` return strings instead of printing; deprecates the `print_*` siblings.

*Expressions*

- Coordinate alignment between subset/superset operands: missing coords fill with 0 in arithmetic and NaN in comparisons. Fixes `subset + var` reverse-addition and result coords expanding past the variable's space.

*Solvers*

- OETC: `Model.solve()` forwards solver options to the handler; `OetcSettings.from_env()` reads `OETC_*`.
- SCIP supports quadratic problems on Windows.

**Performance**

- Faster solution unpacking in `Model.solve()`.

**Bug Fixes**

- `Model.solve()` raises a clear `ValueError` when no objective is set.
- `add_variables` no longer ignores `coords` when `lower` / `upper` are DataArrays, and handles MultiIndex coords correctly with scalar bounds.
- `Model.to_netcdf` no longer fails on the scipy netCDF backend when variables or constraints have MultiIndex coords; level names are now serialised as a JSON string (the legacy list form remains readable).
- CPLEX no longer errors on quality attributes that aren't always available.

**Breaking Changes**

- `google-cloud-storage` and `requests` are now optional. Install `linopy[oetc]` to keep the previous behaviour.

## Version 0.6.7

- Fix Xpress IIS label mapping for masked constraints and add a regression test for matching infeasible coordinates.
- Fix `Model.compute_infeasibilities` returning a flattened, deduplicated union of all IIS when Xpress found more than one. The Xpress path now computes a single IIS (via `firstIIS`), matching the Gurobi path.
- Use `xarray.Dataset.copy` instead of constructor for compatibility with the latest xarray version.
- Blacklist highspy 1.14.0 which produces wrong results due to broken presolve and crashes on Windows ([HiGHS#2964](https://github.com/ERGO-Code/HiGHS/issues/2964)).

## Version 0.6.6

- Free the knitro context and compute necessary quantities within linopy. Knitro context is not exposed anymore.

## Version 0.6.5

- Expose the knitro context to allow for more flexible use of the knitro python API.

## Version 0.6.4

- Add support for the knitro solver via the knitro python API

## Version 0.6.3

**Fix Regression**

- Reinsert broadcasting logic of mask object to be fully compatible with performance improvements in version 0.6.2 using np.where instead of xr.where.

## Version 0.6.2

**Features**

- Add `auto_mask` parameter to `Model` class that automatically masks variables and constraints where bounds, coefficients, or RHS values contain NaN. This eliminates the need to manually create mask arrays when working with sparse or incomplete data.

**Performance**

- Speed up LP file writing by 2-2.7x on large models through Polars streaming engine, join-based constraint assembly, and reduced per-constraint overhead

**Bug Fixes**

- Fix multiplication of constant-only `LinearExpression` with other expressions
- Fix docs and Gurobi license handling

## Version 0.6.1

- Avoid Gurobi initialization on linopy import.
- Fix LP file writing for negative zero (-0.0) values that produced invalid syntax like "+-0.0" rejected by Gurobi

## Version 0.6.0

**Features**

- Add `mock_solve` option to `Model.solve()` for quick testing without actual solving
- Add support for SOS1 and SOS2 (Special Ordered Sets) constraints via `Model.add_sos_constraints()` and `Model.remove_sos_constraints()`
- Add `simplify` method to `LinearExpression` to combine duplicate terms
- Add convenience function to create `LinearExpression` from constant
- Add support for GPU-accelerated solver [cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx)
- Add solver features registry for introspection of solver capabilities

**Performance**

- Up to 50x faster `repr()` for variables/constraints via O(log n) label lookup and direct numpy indexing
- Up to 46x faster `ncons` property by replacing `.flat.labels.unique()` with direct counting

**Bug Fixes**

- Fix HiGHS solver to properly stop on Ctrl-C keyboard interrupt
- Fix CBC solver to correctly parse negative objective values
- Fix Xpress compatibility for versions below 9.6 (regression from namespace change)
- Fix Xpress `getDual()` fallback for older versions
- Fix missing dependency for jupyter notebook example in documentation

**Solver Updates**

- Add Xpress 9.8+ API support with full backward compatibility to 9.6+

## Version 0.5.8

- Replace pandas-based LP file writing with polars implementation for significantly improved performance on large models
- Consolidate "lp" and "lp-polars" io_api options - both now use the optimized polars backend
- Reduced memory usage and faster file I/O operations when exporting models to LP format
- Minor bugfix for multiplying variables with numpy type constants
- Harmonize dtypes before concatenation in lp file writing to avoid dtype mismatch errors. This error occurred when creating and storing models in netcdf format using windows machines and loading and solving them on linux machines.
- Add option to use polars series as constant input
- Fix expression merge to explicitly use outer join when combining expressions with disjoint coordinates for consistent behavior across xarray versions
- Adding xpress postsolve if necessary
- Handle ImportError in xpress import
- Fetch and display OETC worker error logs
- Fix windows permission error when dumping model file
- Performance improvements for xpress solver using C interface

## Version 0.5.7

- Removed deprecated future warning for scalar get item operations
- Silenced version output from the HiGHS solver
- Mosek: Remove explicit use of Env, use global env instead
- Objectives can now be created from variables via linopy.Model.add_objective
- Added integration with OETC platform (refactored implementation)
- Add error message if highspy is not installed
- Fix MindOpt floating release issue
- Made merge expressions function infer class without triggering warnings
- Improved testing coverage
- Fix pypsa-eur environment path in CI

## Version 0.5.6

- Improved variable/expression arithmetic methods so that they correctly handle types
- Gurobi: Pass dictionary as env argument env={...} through to gurobi env creation

**Breaking Changes**

- With this release, the package support for Python 3.9 was dropped and support for Python 3.10 was officially added.
- The selection of a single item in \_\_getitem\_\_ now returns a Variable instead of a ScalarVariable.

## Version 0.5.5

- Internally assign new data fields to expressions with a multiindexed-safe routine.

## Version 0.5.4

**Bug Fixes**

- Remove default highs log file when log_fn=None and io_api="direct". This caused log_file in
    solver_options to be ignored.
- Fix the parsing of solutions returned by the CBC solver when setting a MIP duality
    gap tolerance.
- Improve the mapping of termination conditions for the SCIP solver
- Treat GLPK's integer undefined status as not-OK
- Internally assign new data fields to Variable and Constraint with a multiindexed-safe routine. Before the
    assignment when using multi-indexed coordinates, an deprecation warning was raised. This is fixed now.

## Version 0.5.3

**Bug Fixes**

- Fix the parsing of solutions returned by the CBC solver when solving from a file to not
    assume that variables start with x.
- Fix the retrieval of solutions from the SCIP solver, and do not turn off presolve.

**Minor Improvements**

- Support pickling models.

## Version 0.5.2

**Bug Fixes**

- Fix the multiplication with of zero dimensional numpy arrays with linopy objects.
    This is mainly affecting operations where single numerical items from pandas objects
    are selected and used for multiplication.

## Version 0.5.1

**Deprecations**

- Renamed expression.empty() to expression.empty to align with the use of empty in
    pandas. A custom wrapper ensures that expression.empty() continues to work, but emits
    a DeprecationWarning.

**Features**

\*\* Features \*\*

- Added support for arithmetic operations with custom classes.
- Added align function as a wrapper around [`xarray.align()`][xarray.align].
- Avoid allocating a floating license for COPT during the initial solver check

**Bug fixes**

- Ensure compatibility with xarray \>= v2025.03.00

## Version 0.5.0

**Features**

- Multiplication of a linear expression by a constant value may now introduce new
    dimensions.
- Added method unstack to LinearExpression, Variable and Constraint to unstack
    a dimension.
- Added extra argument in io methods explicit_coordinate_names to allow for export of
    variables and constraints with explicit coordinate names.

**Bug fixes**

- The internal handling of Solution objects was improved for more consistency.
    Solution objects created from solver calls now preserve the exact index names from
    the input file.

## Version 0.4.4

- **IMPORTANT BUGFIX**: The last slice of constraints was not correctly written to LP files in case the constraint size was not a multiple of the slice size. This is fixed now.
- Solution files that following a different naming scheme of variables and constraints using more than on initial letter in the prefix (e.g. col123, row456) are now supported.
- GLPK solver is always called with the --freemps option instead of the --mps when using the Solver API to solve an external MPS file. --mps is for the older fixed-column MPS format that is rarely used nowadays. Almost all fixed MPS files can be parsed by the free MPS format.

## Version 0.4.3

- **Version 0.4.3 includes a major bug and can not be installed anymore.**
- When creating slices for variables and constraints (important for the solve function), the slicing is now fixed in case no dimension to slice is available.
- Added a pandas priority attribute. With this change, the operation with pandas objects is now prioritizing linopy objects over pandas objects. This is useful when the using linopy objects in arithmetic operations with pandas objects, e.g. a \* x where a is a pandas Series/DataFrame and x is a linopy variable.
- The method [model.to_file][linopy.model.Model.to_file] now includes a progress argument to enable or disable the progress bar while writing.

## Version 0.4.2

- **Version 0.4.2 includes a major bug and can not be installed anymore.**
- Fix the file handler to properly close the file when reading the sense from a problem file.

## Version 0.4.1

- Fix the slice_size argument in the solve function. The argument was not properly passed to the to_file function.
- Fix the slicing of constraints in case the term dimension is larger than the leading constraint coordinate dimension.

## Version 0.4.0

- When writing out an LP file, large variables and constraints are now chunked to avoid memory issues. This is especially useful for large models with constraints with many terms. The chunk size can be set with the slice_size argument in the solve function.
- Constraints which of the form \<= infinity and \>= -infinity are now automatically filtered out when solving. The solve function now has a new argument sanitize_infinities to control this feature. Default is set to True.
- The representation of linopy objects with multiindexed coordinates was improved to be more readable.
- Grouping expressions is now supported on dimensions called "group" and dimensions that have the same name as the grouping object.
- Grouping dimensions which have multiindexed coordinates is now supported.

## Version 0.3.15

- The group dimension when grouping by a pandas dataframe is now always group. This fixes the case that the dataframe contains a column named name.

## Version 0.3.14

- Ensure compatibility with xarray \>= v2024.07.0, which has drop the `squeeze` argument from the `groupby` function.

## Version 0.3.13

- Follow-up release to properly fix all deprecations from multiindexed data assignments in xarray datasets.
- Fix typing relevant import for non-default highs dependency in vanilla installation

## Version 0.3.12

- Support for warmstart in HiGHS using basis or solution files, including support for writing basis and solution files of a solved model.
- Linopy now uses mypy for type checking allowing for a more secure and stable code base.
- The creation of solution files with gurobi, scip and mindopt is now supported.

## Version 0.3.11

- The writing and reading from netcdf files was fixed to correctly handle the model parameters field.

## Version 0.3.10

- The classes Variable, LinearExpression and Constraint now have a new getitem method that allows selecting a subset of the object in the same way as xarray objects, i.e. by integer labels or boolean index. Example usage: x\[\[1, 2\]\] or x\[x.indexes\["some_index"\] \> 5\].
- The class Constraint now has a new method .loc to select a subset of the constraint by labels.
- Selecting a single variable with the getitem (\[\]) method now raises a FutureWarning that the return type will change to Variable instead of a ScalarVariable in the future. To get a ScalarVariable in the future, use the at\[\] method.
- A new module examples was added which contains example models. For example, you can call m = linopy.examples.benchmark_model().
- A new memory-efficient and super fast LP file writing method was added which uses the [Polars package](https://github.com/pola-rs/polars). It is still in experimental mode but seems to be very promising. Activate it with the io_api="lp-polars" argument in the solve function.
- The Constraint class now supports the methods assign, assign_attrs, assign_coords, broadcast_like, chunk, drop_sel, drop_isel, expand_dims, sel, isel, shift, swap_dims, set_index, reindex, reindex_like, rename, rename_dims, roll, stack. These methods allow to manipulation of a (anonymous) constraint more flexibly.
- The Variable, expressions and Constraint classes now have new methods swap_dims and set_index. The swap_dims method allows to swap the dimensions of the object. The set_index method allows to set a new index for the object. Both methods are useful for reshaping the object more flexibly.

## Version 0.3.9

- The matrices accessor of the Model class now has a new function dual which returns the dual values of the constraints if the underlying model was optimized and dual values are existent.
- The Variables class now has a new function get_solver_attribute which parses solver-specific attributes of the variables. For now, this function only works for Gurobi solver_model\`s. For example, the function allows retrieving the variable fields \`SAObjUp or RC.
- The constraint assignment with a LinearExpression and a constant value when using the pattern model.add_constraints(lhs_with_constant, sign, rhs) was fixed. Before, the constant value was not added to the right-hand-side properly which led to the wrong constraint behavior. This is fixed now.
- `nan` s in constants is now handled more consistently. These are ignored when in the addition of expressions (effectively filled by zero). In a future version, this might change to align the propagation of `nan` s with tools like numpy/pandas/xarray.
- Up to now the rhs argument in the add_constraints function was not supporting an expression as an input type. This is now added.
- Linopy now supports python 3.12.

**Deprecations**

- The argument dims in the .sum function of variables and expressions was deprecated in favor of the dim argument. This aligns the argument name with the xarray convention.

## Version 0.3.8

**New Features**

- The LinearExpression and QuadraticExpression class have a new attribute solution which returns the optimal values of the expression if the underlying model was optimized.
- It is now possible to access variables and constraints, that don't have python variable name format, as attributes from the corresponding containers. Therefore, a new formatting scheme was introduced which converts dashes and white spaces into underscores. For example, a variable was added to the model with the label "my-variable". This variable can now be accessed with model.variables.my_variable. In particular, the autocompletion function of the IPython console is aware of this new formatting scheme. This allows easy access to variables and constraints with long labels.
- Variables and LinearExpressions now have a new method dot, which allows computing the dot product of two objects. This multiplies objects and sums over common dimensions.
- The matmul operator @, which runs the dot operation, is now supported for Variables and LinearExpression.

**Bugfixes**

- The multiplication of two linear expression with non-zero constants led to wrong results of the cross terms. Given the multiplication (v1 + c1) \* (v2 + c2) with v being a variable and c a constant, the operation did not calculate the cross terms v1 \* c2 + v2 \* c1. This is fixed now.

## Version 0.3.7

**New Features**

- A direct interface to the Mosek solver was added. With this change, a new conversion function model.to_mosek was added to convert a linopy model to a mosek model. The solve function now supports the mosek solver with io_api="direct".
- It is now possible to create LinearExpression from a pandas.DataFrame, pandas.Series, a numpy.array or constant scalar values, e.g. linopy.LinearExpression(df). This will create a LinearExpression with constants only and the coordinates of the DataFrame, Series or array as dimensions.

**Bugfixes**

- When grouping an expression or a variable by a pandas.DataFrame or a xarray.DataArray, the coordinates of the groupby object were not properly aligned. So in cases, when the groupby object was not indexed in the same way as the variable/expression, the groupby operation led to wrong results. This is fixed now.

## Version 0.3.6

- The handling of pandas objects was improved. As pandas objects are fully aware of coordinates, their index and columns are now strictly taken into account. For example, when multiplying a pandas.DataFrame with a variable, linopy now checks the alignment of indexes and reindexes accordingly. Previously, if the axis shapes were the same, the indexes of the variable were inserted and the pandas indexes were effectively ignored. A warning has been added for cases where users should expect changes to the results with this version. **Important**: This does not apply to overwriting the coordinates when one expression is added to another, e.g. "x + df" still overwrites the index of "df" when the dimensional shapes are aligned.
- The .mask attribute of the Constraint class was fixed to return a proper boolean xarray.DataArray object.
- The printout of masked constraints was fixed.

## Version 0.3.5

- The return type of `coord_dims` for expressions and constraints was changed from set to tuple to align with the xarray convention.
- The printout of transposed expressions and constraints was fixed.
- Variables and LinearExpressions now support the chaining operations .add, .sub, .mul, .div.
- Variables and LinearExpressions now have support for the power operator. For example, x\*\*2 is now supported.

## Version 0.3.4

- Solver output of CBC and GLPK is sent to logging with level INFO instead of stdout
- Added support for QP problems with MOSEK and COPT.
- A warning was added when linopy is not able to add pass quadratic objective terms to the highs solver. This is the case when the "ipm" solver of highs is explicitly selected.

## Version 0.3.3

- New solver interface for SCIP \<https://www.scipopt.org/\>. This solver is now supported by linopy and can be used with the solve function if the pyscipopt package is installed. The solver is available for free for general use. See the SCIP website \<https://www.scipopt.org/\> for more information.
- Linopy was refactored to use the new xarray API (\>=2024.01) without the deprecation warnings.
- The set "quadratic_solvers" now only contains quadratic solvers which are installed and available to the user.
- The solve function now throws an error instead of a warning if the set value for `io_api` is not available for a solver.

## Version 0.3.2

- The IO with NetCDF files was made more secure and fixed for some cases. In particular, variables and constraints with a dash in the name are now supported (as used by PyPSA). The object sense and value are now properly stored and retrieved from the netcdf file.
- The IO with NetCDF file now supports multiindexed coordinates.
- The representation of single indexed expressions and constraints with non-empty dimensions/coordinates was fixed, e.g. x.loc\[\["a"\]\] \> 0 where x has only one dimension. Therefore the representation now shows the coordinates.
- The creation of `LinearExpression` and `Constraints` was made robust against the case where the `data` argument is a `xarray.DataArray` with helper dimensions (like "\_term" etc.) unintentionally assigned as coordinates.

## Version 0.3.1

**New Features**

- Added solver interface for MOSEK.
- Support for MindOpt solver was added.
- Added solver interface for COPT by Cardinal Optimizer.
- Type consistency with fill values for constant values was improved, this prevent dtype warnings put out by xarray/numpy.

## Version 0.3.0

**New Features**

- It is now possible to set the sense of the objective function to minimize or maximize. Therefore, a new class Objective was introduced which is used in Model.objective. It supports the same arithmetic operations as LinearExpression and QuadraticExpression and contains a sense attribute which can be set to minimize or maximize.
- The fillna function for variables was made more secure by raising a warning if the fill value is not of variable-like type.
- The where and fillna functions for expressions were made more flexible: When passing a scalar value or a DataArray, the values are added as constants to the expression, where there were missing values before. If another expression is passed, the values are added to the expression, where there were missing values before.

**Breaking Changes**

- The \_fill_value for LinearExpression and QuadraticExpression classes was changed to `NaN` for the constant array ("const"). This allows to use the where function for expressions with constant values in the argument other.
- The functions `ravel` and `iter_ravel` for Variables and Constraints were removed in favor of the `flat` function.
- The property `non_helper_dims` for Variables and Constraints was removed in favor of the `coord_dims` property.
- The function `to_anonymous_constraint` was removed in favor of the `to_constraint` function.
- The support for python 3.8 has been dropped.

## Version 0.2.6

- The memory-efficiency of the IO to LP/MPS file was further improved. In particular, the function to_dataframe is now avoiding unnecessary data copies.
- The printout of time stamps was modified to be more readable, leaving out the display of seconds and below if not necessary.
- The gurobi environment is now enclosed in a context manager to avoid any unwanted use of a token.

## Version 0.2.5

- The solution getter model.solution was falsely returning integer dtype in case of non-aligned indexes. This is fixed now.
- Highs is now in the set of default solvers when install linopy via pip.

## Version 0.2.4

- The IO to LP/MPS file was made more memory-efficient. In particular, the memory excessive operation to_dataframe (see <https://github.com/pydata/xarray/issues/6561>) was replaced by an in-house implementation.

## Version 0.2.3

**Bugfixes**

- When multiplying a LinearExpression with a constant value, the constant in the LinearExpression was not updated. This is fixed now.

**New Features**

- The Variable and the LinearExpression have a new method cumsum, which allows to compute the cumulative sum.

## Version 0.2.2

- The documentation was revised and extended.
- A new function print_labels was added to the Variables and Constraints class. This function allows to print the variables/constraints from a list of labels.
- A new function compute_infeasibilities and print_infeasibilities was added to the Model class. This function allows to compute the infeasibilities of an infeasible model and print them out. The function only supports the gurobi solver so far.

## Version 0.2.1

- Backwards compatibility for python 3.8.
- Variable, LinearExpression and Constraint now have a print function to easily print the objects with larger layouts, i.e. showing more terms and lines.

## Version 0.2.0

**New Features**

- Linopy now supports quadratic programming. Therefore a new class QuadraticExpression was created, which can be assigned to the objective function. The QuadraticExpression class supports the same arithmetic operations as the LinearExpression and can be created by multiplying two Variable or LinearExpression objects. Note for the latter, the number of stacked terms must be equal to one (expr.nterm == 1).
- LinearExpression's now support constant values. This allows defining linear expressions with numeric constant values, like x + 5.
- When defining constraints, it is not needed to separate variables from constants anymore. Thus, expressions like x \<= y or 5 \* x + 10 \>= y are supported.
- The new default solver will now be the first element in available_solvers.
- The classes Variable, LinearExpression and Constraint now have a loc method.
- The classes Variable, LinearExpression, Constraint, Variables and Constraints now have a flat method, which returns a flattened pandas.DataFrame of the object in long-table format.
- It is now possible to access variables and constraints by a dot notation. For example, model.variables.x returns the variable x of the model.
- Variable assignment without explicit coordinates is now supported. In an internal step, integer coordinates are assigned to the dimensions without explicit coordinates.
- The groupby function now supports passing a pandas.Dataframe as groupby keys. These allows to group by multiple variables at once.
- The performance of the groupby function was strongly increased. In large operations a speedup of 10x was observed.
- New test functions assert_varequal, assert_conequal were added to the testing module.

**Deprecations**

- The class AnonymousConstraint is now deprecated in the favor of Constraint. The latter can now be assigned to a model or not.
- The ravel and iter_ravel method of the Variables and Constraints class is now deprecated in favor of the flat method.

**Breaking Changes**

- The data attribute of Variables and Constraints now returns a xarray.Dataset object instead of a xarray.DataArray object with the labels only.
- The deprecated groupby_sum function was removed in favor of the groupby method.
- The deprecated rolling_sum function was removed in favor of the rolling method.
- The deprecated eval module was removed in favor of the arithmetic operations on the classes Variable, LinearExpression and Constraint.
- The deprecated attribute values of the classes Variable, LinearExpression and Constraint was removed in favor of the data attribute.
- The deprecated to_array method of the classes Variable and Constraint was removed in favor of the data attribute.
- The deprecated to_dataset of the LinearExpression class was removed in favor of the data attribute.
- The function get_lower_bound, get_upper_bound, get_variable_labels, get_variable_types, get_objective_coefficient, get_constraint_labels, get_constraint_sense, get_constraint_rhs, get_constraint_matrix were removed in favor of the matrices accessor, i.e. ub, lb, vlabels, etc.
- The LinearExpressionGroupby class now takes a different set of arguments when initializing. These are data: xr.Dataset, group: xr.DataArray, model: Any, kwargs: Mapping\[str, Any\].
- When grouping with a xr.DataArray / pd.Series / pd.DataFrame and summing afterwards, the keyword arguments like squeeze, restore_coords are ignored.

**Internal Changes**

- The internal data fields in Variable and Constraint are now always broadcasted to have aligned indexes. This allows for a more consistent handling of the objects.
- The inner structure of the Variable, Variables, Constraint and Constraints class has changed to a more stable design. All information of the Variable and the Constraint class is now stored in the data field. The data field is a xarray.Dataset object. The Variables and Constraints class "simple" containers for the Variable and Constraint objects, stored in dictionary under the data field. This design allows for a more flexible handling of individual variables and constraints.

**Other**

- License changed to MIT license.

## Version 0.1.5

- Add sel functions to Constraint and AnonymousConstraint to allow for selection and inspection of constraints by coordinate.
- The printout of Variables and Constraints was refactored to a more concise layout.
- The solving termination condition "other" is now tagged as solving status "warning".

## Version 0.1.4

- Fix representation of empty variables and linear expressions.
- The benchmark reported [here](https://github.com/PyPSA/linopy/tree/master/benchmark) was updated to the latest version of linopy and adjusted to be fully reproducible.

## Version 0.1.3

- **Hotfix** dual value retrieval for `highs`.
- The MPS file writing was fixed for `glpk` solver. The MPS file writing is now tested against all solvers.

## Version 0.1.2

- Fix display for constraint with single entry and no coordinates.

## Version 0.1.1

- Printing out long LinearExpression is now accelerated in the \_\_repr\_\_ function.
- Multiplication of LinearExpression's with pandas object was stabilized.
- A options handler was introduced that allows the user to change the maximum of printed lines and terms in the display of Variable's, LinearExpression's and Constraint's.
- If LinearExpression of exactly the same shape are joined together (in arithmetic operations), the coordinates of the first object is used to override the coordinates of the consecutive objects.

## Version 0.1

This is the first major-minor release of linopy! With this release, the package should more stable and consistent. The main changes are:

- The classes Variable, LinearExpression and Constraint now have a \_\_repr\_\_ method. This allows for a better representation of the classes in the console.
- Linopy now defines and uses a fixed set of solver status and termination codes. This allows for a more consistent and reliable handling of solver results. The new codes are defined in the linopy.constants module. The implementation is inspired by <https://github.com/0b11001111> and the implementation in this [PyPSA fork](https://github.com/0b11001111/PyPSA/blob/innoptem-lopf/pypsa/linear_program/solver.py)
- The automated summation of repeated variables in one constraint is now supported. Before the implementation for constraints like x + x + x \<= 5 was only working for solvers with a corresponding fallback computation. This is now fixed.
- Integer variables are now fully supported.
- Support exporting problems to MPS file via fast highspy MPS-writer (highspy installation required).
- The internal data structure of linopy classes were updated to a safer design. Instead of being defined as inherited xarray classes, the class Variable, LinearExpression and Constraint are now no inherited classes but contain the xarray objects in the data field. This allows the package to have more flexible function design and a reduced set of wrapped functions that are sensible to use in the optimization context.
- The class Variable and LinearExpression have new functions groupby and rolling imitating the corresponding xarray functions but with safe type inheritance and application of appended operations.
- Coefficients very close to zero (\< 1e-10) are now automatically set to zero to avoid numerical issues with solvers.
- Coefficients of variables are no also allowed to be np.nan. These coefficients are ignored in the LP file writing.
- The classes Variable, LinearExpression, Constraint, ScalarVariable, ScalarLinearExpression and ScalarConstraint now require the model in the initialization (mostly internal code is affected).
- The eval module was removed in favor of arithmetic operations on the classes Variable, LinearExpression and Constraint.
- Solver options are now printed out in the console when solving a model.
- If a variable with indexes differing from the model internal indexes are assigned, linopy will raise a warning and align the variable to the model indexes.

## Version 0.0.15

- Using the python sum() function over a ScalarVariable or a ScalarLinearExpression is now supported.
- Returning None type in from_rule assignment is now supported.
- Python 3.11 is now supported
- Xarray versions higher and lower v2022.06. are now supported.

## Version 0.0.14

**New Features**

- Linopy now uses highspy \<https://pypi.org/project/highspy/\> as an interface to the HiGHS solver. This enables a direct and fast communication without needing to write an intermediate LP file.

## Version 0.0.13

**New Features**

- The function LinearExpression.from_tuples now allows ScalarVariable as input.
- For compatibility reasons, the function groupby_sum now allows pandas.Series as input.

**Bug Fixes**

- Filtering out zeros is now an optional feature in the solve function. The previous behavior of filtering just before the LP file writing, lead to unexpected errors for constraints with only zero terms.

## Version 0.0.12

**New Features**

- A new module was created to export basic mathematical quantities such as lb, ub, c vectors and the A matrix. Use it with the matrices accessor in linopy.Model.
- For Constraints and \`Variables a \`ipython autocompletion function for getting items was added.
- Inplace updates for constraints are now more flexible.
- AnonymousConstraint can now built from comparison operations of variables with constants, e.g. x \>= 5.
- The Model.add_constraints function now support input of type ScalarVariable, ScalarLinearExpression and ScalarConstraint.
- Terms with zero coefficient are now filtered out before writing to file to avoid unnecessary overhead.
- The function sanitize_zeros was added to Constraints. Use this to filter out zero coefficient terms.

**Bug Fixes**

- Solving with gurobi and io_api="direct" lead to wrong objective assignment if the objective contained non-unique variables. This is fixed in this version.

## Version 0.0.11

- Constraints and expressions can now be created using function that iterates over all combinations of given coordinates. This functionality mirrors the behavior of the Pyomo package. For complicated constraints which are hard to create with arrays of variables, it is easier (thus less efficient) to use an iterating function. For more information see the example notebook in the documentation.
- When getting the value of a variable, the value of the variable is returned as a ScalarVariable. This is useful for the above mentioned creation of expressions and constraints with iterating functions. This affect only the direct getter function, all other functions like .sel or .isel behave as known from Xarray.
- The docstring examples are now part of the Continuous Integration.
- Due to problems with indexing in the latest package version, the xarray dependency was set to \<=v2022.3.0.

## Version 0.0.10

- Improved type security when applying xarray functions on variables linear expressions and constraints.
- Correct block assignment for upcoming PIPS-IPM++ implementation.
- The function `group_terms` was renamed to `groupby_sum`.
- A new function `rolling_sum` was introduced to compute rolling sums for variables and linear expressions.

## Version 0.0.9

**New Features**

- Numpy `__array_ufunc__` was disabled in the Variable, Constraint and LinearExpression class in order to ensure persistence as the class when multiplying with numpy objects. As for pandas objects the issue <https://github.com/pandas-dev/pandas/issues/45803> must be solved.
- The Variable class got a new accessor sol which points to the optimal values if the underlying model was optimized.
- The Constraint class got a new accessor dual which points to the dual values if tune underlying model was optimized and dual values are existent.
- When writing out the LP file, the handling of nan values is now checked in a more rigorous way. Before linopy was skipping and therefore ignoring constraints where the rhs was a nan value. As this behavior is not very save, such cases will raise an error now.
- Models can now be solved on a remote machine using a ssh tunnel. The implementation automatically stores the locally initialized model to a netcdf file on the server, runs the optimization and retrieves the results. See the example Solve a model on a remote machine in the documentation for further information.

**Bug Fixes**

- linopy is now continuously tested and working for Windows machines.

## Version 0.0.8

**New Features**

- Writing out the LP was further sped up.
- The LP file writing for problems with "-0.0" coefficients was fixed.

**Breaking changes**

- the function `as_str` was replaced by `int_to_str` and `float_to_str`.

## Version 0.0.7

**New Features**

- Add `get_name_by_label` function to `Variables` and `Constraints` class. It retrieves the name of the variable/constraint containing the passed integer label. This is helpful for debugging.

**Bug Fixes**

- The lhs accessor for the `Constraint` class was fixed. This raised an error before as the \_term dimension was not adjusted adequately.
- Variables and constraints which are fully masked are now skipped in the lp-file writing. This lead to a error before.

## Version 0.0.6

- Hot fix: Assign `linopy.__version__` attribute
- Hot fix: Fix sign assignment in conversion from `LinearExpression` to `AnonymousConstraint`.

## Version 0.0.5

- LinearExpression has a new function densify_terms which reduces the \_term axis to a minimal length while containing all non-zero coefficient values.
- When summing over one or multiple axes in a LinearExpression, terms with coefficient of zeros can now be dropped automatically.
- The export of LP files was restructured and is flat arrays under the hook to ensure performant export of long constraints.
- Dimensions of masks passed to add_variables and add_constraints now have to be a subset of the resulting labels dimensions.
- A new high-level function merge was added to concatenate multiple linear expressions.
- The `Variable.where` function now has -1 as default fill value.
- The return value of most Variable functions built on xarray functions now preserve the Variable type.
- The variable labels in linear expression which are added to a model are ensured to be stored as integers.
- A preliminary function to print out the subset of infeasible constraints was added (only available for Gurobi, based on <https://www.gurobi.com/documentation/9.5/refman/py_model_computeiis.html>)
- Constraints with only missing variables labels are now sanitized are receive a label -1.
- Binary variables now also have a non-nan lower and upper value due compatibility.
- Models can now be created using the gurobipy API, this can lead to faster total solving times.
- .solve has a new argument io_api. If set to 'direct' the io solving will be performed using the python API's. Currently only available for gurobi.
- The Variable class now has a lower and upper accessor, which allows to inspect and modify the lower and upper bounds of a assigned variable.
- The Constraint class now has a lhs, vars, coeffs, rhs and sign accessor, which allows to inspect and modify the left-hand-side, the signs and right-hand-side of a assigned constraint.
- Constraints can now be build combining linear expressions with right-hand-side via a \>=, \<= or a == operator. This creates an AnonymousConstraint which can be passed to Model.add_constraints.
- Add support of the HiGHS open source solver <https://highs.dev/> (<https://github.com/PyPSA/linopy/pull/8>, <https://github.com/PyPSA/linopy/pull/17>).

**Breaking changes**

- The low level IO function `linopy.io.str_array_to_file` was renamed to `linopy.io.array_to_file`, the function `linopy.io.join_str_arrays` was removed.
- The keep_coords flag in `LinearExpression.sum` and `Variable.sum` was dropped.
- The run\_ functions in linopy.solvers now have a new set of arguments and keyword argument, in order to make solving io more flexible.
- ncons and nvars now count only non-missing constraints and variables.

## Version 0.0.4

**Package Design**

The definitions of variables, constraints and linearexpression were moved to dedicated modules: `linopy.variables`, `linopy.constraints` and `linopy.expressions`.

**Internal/Data handling**

Most of the following changes are dedicated to data handling within the Model class. Users which rely on the internal structure have to expect some breaking changes.

- The model class now stores variables and constraints in dedicated (newly added) classes, `Variables` and `Constraints`. The `Variables` class contains the `xarray` datasets labels, lower and upper. The `Constraints` class contains the datasets labels, coeffs, vars, sign and rhs. The two new class facilitate data access and helper functions.
- The "\_term" dimension in the `LinearExpression` class is now stored without coordinates.
- As soon as a linear expression is added to a model the "\_term" dimension is rename to "{constraintname}\_term" in order align the model better with the contained arrays and to avoid unnecessary nans.
- Missing values in the `Model.variables.labels` and `Model.constraints.labels` arrays are now indicated by -1. This circumvents changing the type from int to float.
- `LinearExpression` now allows empty data as input.
- The test_model_creation script was refactored.

**New Features**

- The `Variable` class now has a accessor to get lower and upper bounds, `get_lower_bound()` and `get_upper_bound()`.
- A new `Constraint` class was added which enables a better visual representation of the constraints. The class also has getter function to get coefficients, variables, signs and rhs constants. The new return type of the `Model.add_constraints` function is `Constraint`.
- `add_variables` and `add_constraints` now accept a new argument `mask`. The mask, which should be an boolean array, defines whether a variable/constraint is active (True) or should be ignored (False).
- A set of experimental eval functions was added. Now one can assign variable and constraints using string expressions. For further information see linopy.Model.vareval, linopy.Model.lineval and linopy.Model.coneval.
- `Model` has a new argument force_dim_names. When set to true assigned variables, constraints and data must always have custom dimension names, otherwise a ValueError is raised. These helps to avoid unintended broadcasting over dimension. Especially the use of pandas DataFrames and Series may become safer.
- A new binaries accessor `Model.binaries` was added.

## Version 0.0.3

- Support assignment of variables and constraints without explicit names.
- Add support for xarray version \> 0.16
- Add a documentation

## Version 0.0.2

- Set up first runnable prototype.
