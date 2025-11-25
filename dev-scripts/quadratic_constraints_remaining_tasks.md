# Quadratic Constraints - Remaining Tasks

## Implementation Status

### ✅ Completed (Phase 1 - Core Implementation)

1. **Core Data Structures**
   - `QTERM_DIM` constant in `constants.py`
   - `QuadraticConstraint` class in `constraints.py`
   - `QuadraticConstraints` container class
   - `QuadraticExpression.to_constraint()` method

2. **Model Integration**
   - `Model.quadratic_constraints` property
   - `Model.add_quadratic_constraints()` method
   - `Model.has_quadratic_constraints` property
   - `Model.type` property updated for QCP/QCQP detection

3. **Solver Support**
   - `QUADRATIC_CONSTRAINT_SOLVERS` list in `solvers.py`
   - Solver validation in `Model.solve()` - rejects unsupported solvers

4. **Export Functionality**
   - LP file export via `quadratic_constraints_to_file()` in `io.py`
   - Gurobi direct API export via updated `to_gurobipy()`

5. **Tests**
   - `test/test_quadratic_constraint.py` - 23 unit tests
   - Updated `test_quadratic_to_constraint` in `test_quadratic_expression.py`

---

## 🔲 Remaining Tasks (Phase 2)

### High Priority

#### 1. Matrix Accessor for Quadratic Constraints
**File:** `linopy/matrices.py`

Add `Qc` property to `MatrixAccessor` class for quadratic constraint matrices.

```python
@property
def Qc(self) -> list[scipy.sparse.csc_matrix]:
    """Return list of Q matrices for quadratic constraints."""
    # Each quadratic constraint has its own Q matrix
    pass

@property
def qc_linear(self) -> scipy.sparse.csc_matrix:
    """Return linear coefficients for quadratic constraints."""
    pass

@property
def qc_sense(self) -> np.ndarray:
    """Return sense array for quadratic constraints."""
    pass

@property
def qc_rhs(self) -> np.ndarray:
    """Return RHS values for quadratic constraints."""
    pass
```

#### 2. MOSEK Direct API Support
**File:** `linopy/io.py` - `to_mosek()` function

Add quadratic constraint support to MOSEK export:
```python
# After linear constraints section
if len(m.quadratic_constraints):
    for name in m.quadratic_constraints:
        qcon = m.quadratic_constraints[name]
        # Use task.appendcone() or task.putqconk() for quadratic constraints
```

Reference: [MOSEK Python API - Quadratic Constraints](https://docs.mosek.com/latest/pythonapi/tutorial-qcqo.html)

#### 3. HiGHSpy Validation
**File:** `linopy/io.py` - `to_highspy()` function

Add explicit error if model has quadratic constraints:
```python
if len(m.quadratic_constraints):
    raise ValueError(
        "HiGHS does not support quadratic constraints. "
        "Use a solver that supports QCP: gurobi, cplex, mosek, xpress, copt, scip"
    )
```

#### 4. Solution Retrieval for Quadratic Constraints
**Files:** `linopy/solvers.py`, `linopy/constraints.py`

- Add dual value retrieval for quadratic constraints (where supported)
- Store duals in `QuadraticConstraint.dual` property
- Update solver result parsing

### Medium Priority

#### 5. Multi-dimensional Quadratic Constraints
**File:** `linopy/constraints.py`, `linopy/model.py`

Currently, quadratic constraints are primarily scalar. Add support for:
- Broadcasting over coordinates (like linear constraints)
- `iterate_slices()` support for memory-efficient processing
- Coordinate-based indexing

Example API:
```python
# Should work with coordinates
m.add_quadratic_constraints(
    x * x + y * y,  # where x, y have dims=['time', 'node']
    "<=",
    100,
    name="qc"
)
```

#### 6. Constraint Modification Methods
**File:** `linopy/constraints.py`

Add methods to `QuadraticConstraint`:
```python
def modify_rhs(self, new_rhs: ConstantLike) -> None:
    """Modify the right-hand side of the constraint."""

def modify_coeffs(self, new_coeffs: xr.DataArray) -> None:
    """Modify coefficients of the constraint."""
```

#### 7. netCDF Serialization
**File:** `linopy/io.py` - `to_netcdf()` and `read_netcdf()`

Add quadratic constraints to model serialization:
```python
# In to_netcdf()
qcons = [
    with_prefix(qcon.data, f"quadratic_constraints-{name}")
    for name, qcon in m.quadratic_constraints.items()
]

# In read_netcdf()
# Parse quadratic_constraints-* prefixed datasets
```

### Low Priority

#### 8. Convexity Checking (Optional)
**File:** `linopy/constraints.py` or new `linopy/analysis.py`

Add optional convexity verification:
```python
def check_convexity(self) -> bool:
    """
    Check if quadratic constraint is convex.

    A quadratic constraint x'Qx + a'x <= b is convex if Q is
    positive semidefinite.
    """
    # Extract Q matrix
    # Check eigenvalues or use Cholesky decomposition
    pass
```

#### 9. Constraint Printing Improvements
**File:** `linopy/constraints.py`

Enhance `_format_single_constraint()` for better display:
- Handle large constraints with truncation
- Add option for matrix form display
- Support LaTeX output

#### 10. Documentation
**Files:** `doc/` directory

- Add quadratic constraints section to user guide
- Document supported solvers and their limitations
- Add examples for common QCP formulations (portfolio optimization, etc.)

---

## Testing Tasks

### Unit Tests to Add

1. **Multi-dimensional constraints** - Test with coordinates
2. **Edge cases** - Empty constraints, single term, all linear terms
3. **Numerical precision** - Very small/large coefficients
4. **Memory efficiency** - Large constraint sets with `iterate_slices`

### Integration Tests

1. **Solver round-trip** - Create model, solve, verify solution
2. **File format round-trip** - Write LP, read back, compare
3. **Cross-solver consistency** - Same problem, multiple solvers

### Solver-Specific Tests

```python
@pytest.mark.parametrize("solver", ["gurobi", "mosek", "cplex"])
def test_qcp_solve(solver):
    """Test solving QCP with different solvers."""
    if solver not in available_solvers:
        pytest.skip(f"{solver} not available")
    # ... test code
```

---

## Code Quality Tasks

1. **Type hints** - Ensure all new functions have complete type annotations
2. **Docstrings** - Add NumPy-style docstrings to all public methods
3. **Linting** - Run `ruff check --fix` on all modified files
4. **MyPy** - Fix any type errors in new code

---

## Architecture Notes

### Data Structure

```
QuadraticConstraint.data (xarray.Dataset):
├── quad_coeffs: (_qterm, _factor) float64  # Quadratic term coefficients
├── quad_vars: (_qterm, _factor) int64      # Variable indices for quad terms
├── lin_coeffs: (_term) float64             # Linear term coefficients
├── lin_vars: (_term) int64                 # Variable indices for linear terms
├── sign: str                               # "<=", ">=", or "="
├── rhs: float64                            # Right-hand side value
└── labels: int64                           # Constraint label/index
```

### LP File Format

```
qc0:
+3.0 x0
+4.0 x1
+ [
+1.0 x0 ^ 2
+2.0 x0 * x1
+1.0 x1 ^ 2
]
<= +100.0
```

### Solver Compatibility Matrix

| Solver | QCP Support | API Method |
|--------|-------------|------------|
| Gurobi | ✅ | `addQConstr()` |
| CPLEX  | ✅ | `add_quadratic_constraint()` |
| MOSEK  | ✅ | `putqconk()` |
| Xpress | ✅ | `addConstraint()` with quadratic |
| COPT   | ✅ | `addQConstr()` |
| SCIP   | ✅ | LP file import |
| HiGHS  | ❌ | Not supported |

---

## Suggested Task Order for Next Agent

1. **Matrix Accessor** (`matrices.py`) - Enables programmatic access to constraint data
2. **MOSEK support** (`io.py`) - Important solver for QCP
3. **Multi-dimensional constraints** - Core functionality improvement
4. **netCDF serialization** - Model persistence
5. **Documentation** - User-facing docs

Each task is relatively independent and can be completed in a single session.
