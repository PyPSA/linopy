# Quadratic Constraints Implementation Status

**Date:** 2024-11-25
**Status:** Phase 1 Complete ✅

## Related Documents

- **Original Plan:** `dev-scripts/quadratic_constraints_plan.md` - Detailed design document
- **Remaining Tasks:** `dev-scripts/quadratic_constraints_remaining_tasks.md` - What's left to do

---

## Summary

Quadratic constraints (QCP/QCQP) have been successfully added to linopy. Users can now create and solve optimization problems with constraints of the form:

```
x'Qx + a'x ≤ b   (or ≥, =)
```

## Usage Example

```python
import linopy

m = linopy.Model()

# Variables
x = m.add_variables(lower=0, name='x')
y = m.add_variables(lower=0, name='y')

# Linear constraint (existing)
m.add_constraints(x + y <= 10, name='budget')

# Quadratic constraint (NEW!)
m.add_quadratic_constraints(
    x*x + 2*x*y + y*y + 3*x + 4*y,
    "<=",
    100,
    name='quadratic_budget'
)

# Objective
m.add_objective(x + 2*y)

# Solve (with Gurobi, MOSEK, CPLEX, etc.)
m.solve(solver_name='gurobi')
```

## What Was Implemented

### Core Components

| Component | File | Status |
|-----------|------|--------|
| `QTERM_DIM` constant | `constants.py` | ✅ |
| `QuadraticConstraint` class | `constraints.py` | ✅ |
| `QuadraticConstraints` container | `constraints.py` | ✅ |
| `QuadraticExpression.to_constraint()` | `expressions.py` | ✅ |
| `Model.add_quadratic_constraints()` | `model.py` | ✅ |
| `Model.quadratic_constraints` property | `model.py` | ✅ |
| `Model.has_quadratic_constraints` | `model.py` | ✅ |
| Model type detection (QCLP/QCQP) | `model.py` | ✅ |
| Solver validation | `model.py` | ✅ |
| `QUADRATIC_CONSTRAINT_SOLVERS` list | `solvers.py` | ✅ |
| LP file export | `io.py` | ✅ |
| Gurobi direct export | `io.py` | ✅ |
| Unit tests | `test/test_quadratic_constraint.py` | ✅ |

### Supported Solvers

| Solver | Support | Notes |
|--------|---------|-------|
| Gurobi | ✅ | Full support via `addQConstr()` |
| CPLEX | ✅ | Via LP file |
| MOSEK | ✅ | Via LP file (direct API pending) |
| Xpress | ✅ | Via LP file |
| COPT | ✅ | Via LP file |
| SCIP | ✅ | Via LP file |
| HiGHS | ❌ | Does not support QC - validation error raised |

### Model Type Strings

The `Model.type` property now returns:

| Type | Meaning |
|------|---------|
| `LP` | Linear constraints, linear objective |
| `QP` | Linear constraints, quadratic objective |
| `QCLP` | Quadratic constraints, linear objective |
| `QCQP` | Quadratic constraints, quadratic objective |
| `MILP` | Mixed-integer linear |
| `MIQP` | Mixed-integer quadratic objective |
| `MIQCLP` | Mixed-integer with quadratic constraints |
| `MIQCQP` | Mixed-integer QC with quadratic objective |

---

## What's NOT Yet Implemented

See `dev-scripts/quadratic_constraints_remaining_tasks.md` for full details.

### High Priority
1. Matrix accessor (`Qc` property in `matrices.py`)
2. MOSEK direct API support
3. HiGHS explicit validation in `to_highspy()`
4. Solution/dual retrieval for quadratic constraints

### Medium Priority
5. Multi-dimensional quadratic constraints (with coordinates)
6. netCDF serialization
7. Constraint modification methods

### Low Priority
8. Convexity checking (optional)
9. Documentation

---

## Testing

All 23 unit tests pass:

```bash
pytest test/test_quadratic_constraint.py -v
# 23 passed
```

Existing quadratic expression tests updated:
```bash
pytest test/test_quadratic_expression.py -v
# 32 passed (test_quadratic_to_constraint updated)
```

---

## Design Decisions Made

1. **Separate method**: Using `add_quadratic_constraints()` instead of overloading `add_constraints()` - clearer API, can add auto-detection later

2. **Separate container**: `Model.quadratic_constraints` is separate from `Model.constraints` - cleaner code, explicit handling

3. **Defer convexity**: Convexity checking deferred to solver - avoids false positives, solvers handle it better

4. **LP format**: Using Gurobi-style LP format with `[ ]` brackets for quadratic terms

---

## Files Modified

```
linopy/
├── constants.py      # +QTERM_DIM
├── constraints.py    # +QuadraticConstraint, +QuadraticConstraints (~500 lines)
├── expressions.py    # +to_constraint() for QuadraticExpression
├── model.py          # +add_quadratic_constraints(), properties, validation
├── solvers.py        # +QUADRATIC_CONSTRAINT_SOLVERS
└── io.py             # +quadratic_constraints_to_file(), updated to_gurobipy()

test/
├── test_quadratic_constraint.py  # NEW - 23 tests
└── test_quadratic_expression.py  # Updated 1 test
```

---

## Next Steps for Code Agent

1. Pick a task from `quadratic_constraints_remaining_tasks.md`
2. Tasks are independent - any order works
3. Recommended first: Matrix Accessor (enables other features)
4. Run `pytest test/test_quadratic_constraint.py` to verify nothing breaks
