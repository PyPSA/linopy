
<!-- blackdoc:off -->

# Quadratic Constraints and Expressions in Linopy

## Design Document and Implementation Plan

### Executive Summary

This document outlines a plan to extend linopy with support for **quadratic constraints** (QCP/QCQP). Currently, linopy supports:
- Linear constraints (`Constraint` class)
- Linear expressions (`LinearExpression` class)
- Quadratic expressions (`QuadraticExpression` class) — **only for objectives**

The goal is to enable quadratic constraints of the form:

```
x'Qx + a'x ≤ b   (or ≥, =)
```

This feature would make linopy suitable for a broader class of optimization problems including convex QCPs, non-convex QCQPs (where supported by solvers), and Second-Order Cone Programs (SOCPs).

---

## 1. Current Architecture Analysis

### 1.1 Linear Expression (`expressions.py`)

The `LinearExpression` class stores:
```python
Dataset {
    'coeffs': DataArray[float]  # shape: (..., _term)
    'vars': DataArray[int]      # shape: (..., _term), variable labels
    'const': DataArray[float]   # shape: (...)
}
```

Key method for constraint creation (`expressions.py:843-866`):
```python
def to_constraint(self, sign: SignLike, rhs: ConstantLike) -> Constraint:
    all_to_lhs = (self - rhs).data
    data = assign_multiindex_safe(
        all_to_lhs[["coeffs", "vars"]],
        sign=sign,
        rhs=-all_to_lhs.const
    )
    return constraints.Constraint(data, model=self.model)
```

### 1.2 Quadratic Expression (`expressions.py`)

The `QuadraticExpression` class adds a `_factor` dimension (size=2) for storing two variables per quadratic term:
```python
Dataset {
    'coeffs': DataArray[float]  # shape: (..., _factor, _term)
    'vars': DataArray[int]      # shape: (..., _factor, _term)
    'const': DataArray[float]   # shape: (...)
}
```

Currently, `QuadraticExpression.to_constraint()` raises `NotImplementedError` (`expressions.py:1805-1808`).

### 1.3 Linear Constraint (`constraints.py`)

The `Constraint` class stores:
```python
Dataset {
    'coeffs': DataArray[float]  # LHS coefficients
    'vars': DataArray[int]      # Variable labels
    'sign': DataArray[str]      # '=', '<=', '>='
    'rhs': DataArray[float]     # Right-hand side
    'labels': DataArray[int]    # Constraint labels (-1 if masked)
    'dual': DataArray[float]    # [OPTIONAL] Dual values
}
```

### 1.4 Solver Support for Quadratic Constraints

| Solver | QCP Support | QCQP Support | Non-Convex Support |
|--------|-------------|--------------|-------------------|
| Gurobi | ✅ Yes | ✅ Yes | ✅ Yes (v9.0+) |
| CPLEX | ✅ Yes | ✅ Yes | ⚠️ Limited |
| MOSEK | ✅ Yes | ✅ Yes | ❌ Convex only |
| Xpress | ✅ Yes | ✅ Yes | ⚠️ Limited |
| COPT | ✅ Yes | ✅ Yes | ⚠️ Limited |
| SCIP | ✅ Yes | ✅ Yes | ✅ Yes |
| HiGHS | ❌ No | ❌ No | ❌ No |
| GLPK | ❌ No | ❌ No | ❌ No |
| CBC | ❌ No | ❌ No | ❌ No |

**Key Insight**: HiGHS (a common default solver) does NOT support quadratic constraints. This has implications for default behavior and error handling.

---

## 2. Proposed Design

### 2.1 New Class: `QuadraticConstraint`

Create a new `QuadraticConstraint` class parallel to `Constraint`:

```python
class QuadraticConstraint:
    """
    A quadratic constraint of the form: x'Qx + a'x ≤ b (or ≥, =)

    Dataset structure:
    {
        'quad_coeffs': DataArray[float]   # shape: (..., _factor, _qterm)
        'quad_vars': DataArray[int]       # shape: (..., _factor, _qterm)
        'lin_coeffs': DataArray[float]    # shape: (..., _term)
        'lin_vars': DataArray[int]        # shape: (..., _term)
        'sign': DataArray[str]            # '=', '<=', '>='
        'rhs': DataArray[float]           # Right-hand side constant
        'labels': DataArray[int]          # Constraint labels
        'dual': DataArray[float]          # [OPTIONAL] Dual values (only for convex)
    }
    """
```

**Design Rationale**:
- Separate `quad_*` and `lin_*` arrays to allow efficient handling of purely linear terms
- Use `_qterm` dimension (distinct from `_term`) for quadratic terms
- Maintain API consistency with `Constraint` class

### 2.2 Container Class: `QuadraticConstraints`

Add a container class analogous to `Constraints`:

```python
class QuadraticConstraints:
    """
    Container for multiple QuadraticConstraint objects.
    Provides dict-like access and aggregation properties.
    """
```

### 2.3 Model Integration

Extend the `Model` class:

```python
class Model:
    def __init__(self, ...):
        self.constraints = Constraints()      # Linear constraints
        self.quadratic_constraints = QuadraticConstraints()  # NEW

    def add_quadratic_constraints(
        self,
        lhs: QuadraticExpression | Callable,
        sign: SignLike,
        rhs: ConstantLike,
        name: str | None = None,
        coords: CoordsLike | None = None,
        mask: MaskLike | None = None,
    ) -> QuadraticConstraint:
        """Add quadratic constraint(s) to the model."""

    @property
    def has_quadratic_constraints(self) -> bool:
        """Return True if model has any quadratic constraints."""

    @property
    def type(self) -> str:
        """Return problem type: 'LP', 'QP', 'MILP', 'MIQP', 'QCP', 'QCQP', etc."""
```

### 2.4 Expression API Changes

Implement `QuadraticExpression.to_constraint()`:

```python
def to_constraint(self, sign: SignLike, rhs: ConstantLike) -> QuadraticConstraint:
    """
    Convert quadratic expression to a quadratic constraint.

    Parameters
    ----------
    sign : str
        Constraint sense: '<=', '>=', or '='
    rhs : float or array-like
        Right-hand side constant

    Returns
    -------
    QuadraticConstraint
    """
```

Enable comparison operators on `QuadraticExpression`:
```python
# These would create QuadraticConstraint objects
quad_expr <= 10  # Works (returns QuadraticConstraint)
quad_expr >= 5   # Works
quad_expr == 0   # Works
```

---

## 3. Implementation Details

### 3.1 Data Storage for Quadratic Constraints

**Option A: Unified Storage** (simpler, less efficient)
```python
# Store everything with _factor dimension, linear terms have vars[_factor=1] = -1
Dataset {
    'coeffs': DataArray[float]  # shape: (..., _factor, _term)
    'vars': DataArray[int]      # shape: (..., _factor, _term)
    'sign': DataArray[str]
    'rhs': DataArray[float]
    'labels': DataArray[int]
}
```

**Option B: Split Storage** (recommended, more efficient)
```python
# Separate linear and quadratic terms
Dataset {
    'quad_coeffs': DataArray[float]  # shape: (..., _factor, _qterm)
    'quad_vars': DataArray[int]      # shape: (..., _factor, _qterm)
    'lin_coeffs': DataArray[float]   # shape: (..., _term)
    'lin_vars': DataArray[int]       # shape: (..., _term)
    'sign': DataArray[str]
    'rhs': DataArray[float]
    'labels': DataArray[int]
}
```

**Recommendation**: Option B provides clearer separation, easier debugging, and more efficient matrix construction for solvers that handle linear and quadratic parts separately.

### 3.2 Matrix Representation

Add to `MatrixAccessor`:

```python
@property
def Qc(self) -> list[tuple[csc_matrix, ndarray, float, str]]:
    """
    List of quadratic constraint matrices.

    Returns list of tuples: (Q_i, a_i, b_i, sense_i)
    where constraint i is: x'Q_i x + a_i'x {sense_i} b_i
    """

@property
def qc_labels(self) -> ndarray:
    """Labels of quadratic constraints."""
```

### 3.3 Solver Export Functions

#### LP File Format

The LP file format supports quadratic constraints in the `QCROWS` section:

```
Subject To
 c1: x + y <= 10

QCROWS
 qc1: [ x^2 + 2 x * y + y^2 ] + x + y <= 5
End
```

Add function:
```python
def quadratic_constraints_to_file(
    m: Model,
    f: BufferedWriter,
    progress: bool = False,
    explicit_coordinate_names: bool = False,
) -> None:
    """Write quadratic constraints to LP file."""
```

#### Direct API Export

**Gurobi** (`addQConstr` or matrix interface):
```python
def to_gurobipy(m: Model, env=None, ...):
    # ... existing code ...

    # Add quadratic constraints
    for qc in m.quadratic_constraints:
        model.addQConstr(Q, sense, rhs, name)
```

**MOSEK** (`putqconk`):
```python
def to_mosek(m: Model, task=None, ...):
    # ... existing code ...

    # Add quadratic constraints
    for k, (Q, a, b, sense) in enumerate(M.Qc):
        task.putqconk(k, Q.row, Q.col, Q.data)
        task.putarow(k, a.nonzero()[0], a[a.nonzero()])
```

### 3.4 Solution Handling

Quadratic constraints may have dual values (for convex problems):

```python
class QuadraticConstraint:
    @property
    def dual(self) -> DataArray | None:
        """
        Dual values for the quadratic constraint.

        Note: Only available for convex quadratic constraints
        and when the solver provides them.
        """
```

---

## 4. API Design Considerations

### 4.1 Consistency with Existing API

The API should feel natural to existing linopy users:

```python
import linopy as lp

m = lp.Model()
x = m.add_variables(coords=[range(3)], name='x')
y = m.add_variables(name='y')

# Linear constraint (existing)
m.add_constraints(x.sum() <= 10, name='linear_budget')

# Quadratic constraint (new - Option A: via add_constraints)
m.add_constraints(x @ x + y <= 5, name='quad_con')

# Quadratic constraint (new - Option B: via add_quadratic_constraints)
m.add_quadratic_constraints(x @ x + y <= 5, name='quad_con')
```

**Question for discussion**: Should quadratic constraints be added via:
- **Option A**: Same `add_constraints()` method (auto-detect based on expression type)
- **Option B**: Separate `add_quadratic_constraints()` method

**Recommendation**: Start with **Option B** for clarity, with Option A as a future enhancement. This makes the API explicit about what type of constraint is being added.

### 4.2 Operator Overloading

Enable natural syntax on `QuadraticExpression`:

```python
# All should return QuadraticConstraint
x * x <= 10
(x @ x) + y >= 5
2 * x * y == 0
```

### 4.3 Error Handling

```python
# Clear error for unsupported solvers
m.solve(solver='highs')
# Raises: "Solver 'highs' does not support quadratic constraints.
#          Use one of: ['gurobi', 'cplex', 'mosek', 'xpress', 'copt', 'scip']"

# Warning for non-convex constraints with convex-only solvers
m.solve(solver='mosek')
# Warning: "MOSEK requires convex quadratic constraints.
#           Non-convex constraints may cause solver failure."
```

---

## 5. File Structure Changes

### 5.1 New Files

None required - extend existing modules.

### 5.2 Modified Files

| File | Changes |
|------|---------|
| `expressions.py` | Implement `QuadraticExpression.to_constraint()` |
| `constraints.py` | Add `QuadraticConstraint` and `QuadraticConstraints` classes |
| `model.py` | Add `add_quadratic_constraints()`, `quadratic_constraints` property |
| `io.py` | Add LP file export for quadratic constraints |
| `solvers.py` | Add `QUADRATIC_CONSTRAINT_SOLVERS` list |
| `matrices.py` | Add `Qc` property for quadratic constraint matrices |
| `constants.py` | Add any new constants (e.g., `QTERM_DIM = "_qterm"`) |

### 5.3 New Test Files

- `test/test_quadratic_constraint.py` — Unit tests for QuadraticConstraint class
- `test/test_quadratic_optimization.py` — Integration tests with solvers

---

## 6. Implementation Phases

### Phase 1: Core Data Structures (Week 1-2)

1. Add `QTERM_DIM` constant
2. Implement `QuadraticConstraint` class with basic functionality:
   - `__init__`, `__repr__`, `__getitem__`
   - Properties: `labels`, `sign`, `rhs`, `lhs`, `mask`, etc.
   - Methods: `to_polars()`, `flat`
3. Implement `QuadraticConstraints` container
4. Add `QuadraticExpression.to_constraint()` method

### Phase 2: Model Integration (Week 2-3)

1. Add `Model.quadratic_constraints` property
2. Implement `Model.add_quadratic_constraints()` method
3. Update `Model.type` property for QCP/QCQP detection
4. Add `Model.has_quadratic_constraints` property
5. Update constraint label management

### Phase 3: Solver Export (Week 3-4)

1. Extend LP file writer with `QCROWS` section
2. Update `to_gurobipy()` for quadratic constraints
3. Update `to_mosek()` for quadratic constraints
4. Update other direct-API solvers (CPLEX, Xpress, COPT)
5. Add solver compatibility checks

### Phase 4: Solution Handling (Week 4-5)

1. Parse quadratic constraint duals from solver results
2. Map duals back to constraint coordinates
3. Add `QuadraticConstraint.dual` property

### Phase 5: Testing & Documentation (Week 5-6)

1. Comprehensive unit tests
2. Integration tests with each supported solver
3. Update documentation and examples
4. Add tutorial notebook

---

## 7. Code Examples

### 7.1 Basic Usage

```python
import linopy as lp

# Create model
m = lp.Model()

# Add variables
x = m.add_variables(lower=0, upper=10, name='x')
y = m.add_variables(lower=0, upper=10, name='y')

# Quadratic objective (already supported)
m.add_objective(x**2 + y**2 + x + y)

# Linear constraint (already supported)
m.add_constraints(x + y >= 1, name='sum_bound')

# NEW: Quadratic constraint
m.add_quadratic_constraints(x**2 + y**2 <= 25, name='circle')

# Solve with quadratic-capable solver
m.solve(solver='gurobi')

print(f"x = {x.solution.values}")
print(f"y = {y.solution.values}")
```

### 7.2 Multi-dimensional Quadratic Constraints

```python
import linopy as lp
import pandas as pd

m = lp.Model()

# Index set
times = pd.Index(range(24), name='time')

# Variables
power = m.add_variables(coords=[times], name='power')
reserve = m.add_variables(coords=[times], name='reserve')

# Quadratic constraint at each time step
# power²[t] + reserve²[t] <= capacity[t]
capacity = [100] * 24
qc = m.add_quadratic_constraints(
    power**2 + reserve**2 <= capacity,
    name='capacity_limit'
)

print(qc)
# QuadraticConstraint `capacity_limit` [time: 24]:
# -------------------------------------------
# 0: power[0]² + reserve[0]² <= 100
# 1: power[1]² + reserve[1]² <= 100
# ...
```

### 7.3 Rule-based Quadratic Constraints

```python
def capacity_rule(m, t):
    """Quadratic capacity constraint at time t."""
    return m['power'][t]**2 + m['reserve'][t]**2 <= capacity[t]

m.add_quadratic_constraints(capacity_rule, coords=[times], name='capacity')
```

---

## 8. Design Decisions (Resolved)

### Q1: Unified vs Separate `add_constraints` method?

**Decision**: Use separate `add_quadratic_constraints()` method.

```python
m.add_quadratic_constraints(x**2 <= 10)  # Quadratic
m.add_constraints(x <= 10)               # Linear
```

**Rationale**: Explicit API is clearer. Auto-detection can be added later by routing `add_constraints()` to `add_quadratic_constraints()` when a `QuadraticExpression` is detected.

### Q2: Storage location for quadratic constraints?

**Decision**: Separate containers.

```python
m.constraints           # Linear only
m.quadratic_constraints # Quadratic only
```

**Rationale**: Simpler implementation, matches current pattern for variables/constraints, avoids complexity of mixed container.

### Q3: How to handle mixed linear+quadratic in same named constraint group?

**Decision**: Each named constraint should be uniformly linear or quadratic. Mixed cases require two separate constraints.

### Q4: Convexity checking?

**Decision**: Defer to solver.

**Rationale**: Avoids computational overhead and complex eigenvalue analysis. Solvers like Gurobi and MOSEK provide clear error messages for non-convex constraints. We can add clear documentation about convexity requirements.

---

## 9. References

### Solver Documentation

- [Gurobi Quadratic Constraints](https://docs.gurobi.com/projects/optimizer/en/current/concepts/modeling/constraints.html)
- [MOSEK Conic Constraints](https://docs.mosek.com/latest/pythonapi/tutorial-cqo-shared.html)
- [CPLEX Quadratic Constraints](https://www.ibm.com/docs/en/icos/latest?topic=programming-adding-quadratic-constraints)

### Related Issues/PRs

- Current `NotImplementedError` in `expressions.py:1805-1808`
- Test showing limitation: `test/test_quadratic_expression.py:331`

### Mathematical Background

A quadratically constrained quadratic program (QCQP) has the form:

```
minimize    (1/2)x'Q₀x + c'x
subject to  (1/2)x'Qᵢx + aᵢ'x ≤ bᵢ    for i = 1,...,m
            Ax = b
            l ≤ x ≤ u
```

Where:
- Q₀ is the objective quadratic matrix (may be zero for QCP)
- Qᵢ are constraint quadratic matrices
- aᵢ are constraint linear coefficient vectors
- bᵢ are constraint right-hand sides

For **convex** QCPs, all Qᵢ must be positive semi-definite (PSD).

---

## 10. Summary

Adding quadratic constraint support to linopy is a significant but feasible enhancement. The key design decisions are:

1. **New class**: `QuadraticConstraint` parallel to `Constraint`
2. **Split storage**: Separate `quad_*` and `lin_*` arrays for efficiency
3. **Explicit API**: `add_quadratic_constraints()` method
4. **Solver filtering**: Clear error messages for unsupported solvers
5. **Phased implementation**: Core → Model → Export → Tests

This enhancement would expand linopy's capabilities to cover:
- Convex QCPs (portfolio optimization, geometric programming)
- QCQPs (facility location, engineering design)
- SOCPs via quadratic constraint reformulation

---

*Document Version: 1.0*
*Date: 2025-01-25*
*Status: Draft for Discussion*
