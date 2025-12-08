# Solver Capabilities Registry - Phase 1 Planning Document

## Overview

This document outlines the design and implementation plan for a centralized solver capabilities registry within linopy. The goal is to replace scattered hardcoded solver feature checks with a single source of truth.

## Current State

### Existing Feature Lists (in `solvers.py`)

```python
QUADRATIC_SOLVERS = ["gurobi", "xpress", "cplex", "highs", "scip", "mosek", "copt", "mindopt"]
NO_SOLUTION_FILE_SOLVERS = ["xpress", "gurobi", "highs", "mosek", "scip", "copt", "mindopt"]

# Derived (filtered by availability)
quadratic_solvers = [s for s in QUADRATIC_SOLVERS if s in available_solvers]
```

### Scattered Hardcoded Checks

| Location | Check | Purpose |
|----------|-------|---------|
| `model.py:1199` | `solver_name in NO_SOLUTION_FILE_SOLVERS` | Skip solution file creation |
| `model.py:1211` | `solver_name not in quadratic_solvers` | Validate quadratic objective support |
| `model.py:1234` | `solver_name in ["glpk", "cbc"]` | Disable LP file names |
| `model.py:1342` | `solver_name in ["gurobi", "xpress"]` | IIS computation |
| `io.py:508` | `"highs" not in available_solvers` | MPS export requires HiGHS |
| `variables.py:852` | `solver_name != "gurobi"` | Solver attribute access |

---

## Proposed Design

### 1. Feature Enumeration

```python
# linopy/solver_capabilities.py

from enum import Enum, auto

class SolverFeature(Enum):
    """Enumeration of all solver capabilities tracked by linopy."""

    # Objective function support
    QUADRATIC_OBJECTIVE = auto()

    # I/O capabilities
    LP_FILE_NAMES = auto()  # Support for named variables/constraints in LP files
    SOLUTION_FILE_NOT_NEEDED = auto()

    # Advanced features
    IIS_COMPUTATION = auto()

    # Solver-specific
    SOLVER_ATTRIBUTE_ACCESS = auto()  # Direct access to solver variable attributes
```

Note: This is a minimal set for the proof of concept. Additional features can be added as needed:
- `QUADRATIC_CONSTRAINTS`, `NONCONVEX_QUADRATIC_CONSTRAINTS` (when QC support is added)
- `INTEGER_VARIABLES`, `BINARY_VARIABLES` (currently all solvers support these)
- `DIRECT_API`, `LP_FILE_READ`, `MPS_FILE_READ` (I/O API tracking)
- `WARMSTART`, `BASIS_IO` (advanced optimization features)

### 2. Solver Registry

```python
from dataclasses import dataclass
from typing import FrozenSet

@dataclass(frozen=True)
class SolverInfo:
    """Information about a solver's capabilities."""
    name: str
    features: FrozenSet[SolverFeature]
    display_name: str = ""

    def __post_init__(self):
        if not self.display_name:
            object.__setattr__(self, 'display_name', self.name.upper())

    def supports(self, feature: SolverFeature) -> bool:
        """Check if this solver supports a given feature."""
        return feature in self.features


# Define all solver capabilities (proof of concept - minimal feature set)
SOLVER_REGISTRY: dict[str, SolverInfo] = {
    "gurobi": SolverInfo(
        name="gurobi",
        display_name="Gurobi",
        features=frozenset({
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            SolverFeature.IIS_COMPUTATION,
            SolverFeature.SOLVER_ATTRIBUTE_ACCESS,
        }),
    ),
    "highs": SolverInfo(
        name="highs",
        display_name="HiGHS",
        features=frozenset({
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }),
    ),
    "glpk": SolverInfo(
        name="glpk",
        display_name="GLPK",
        features=frozenset({
            # Note: LP_FILE_NAMES intentionally NOT included
        }),
    ),
    "cbc": SolverInfo(
        name="cbc",
        display_name="CBC",
        features=frozenset({
            # Note: LP_FILE_NAMES intentionally NOT included
        }),
    ),
    "cplex": SolverInfo(
        name="cplex",
        display_name="CPLEX",
        features=frozenset({
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
        }),
    ),
    "xpress": SolverInfo(
        name="xpress",
        display_name="FICO Xpress",
        features=frozenset({
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            SolverFeature.IIS_COMPUTATION,
        }),
    ),
    "scip": SolverInfo(
        name="scip",
        display_name="SCIP",
        features=frozenset({
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }),
    ),
    "mosek": SolverInfo(
        name="mosek",
        display_name="MOSEK",
        features=frozenset({
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }),
    ),
    "copt": SolverInfo(
        name="copt",
        display_name="COPT",
        features=frozenset({
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }),
    ),
    "mindopt": SolverInfo(
        name="mindopt",
        display_name="MindOpt",
        features=frozenset({
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }),
    ),
}
```

### 3. Helper Functions

```python
def solver_supports(solver_name: str, feature: SolverFeature) -> bool:
    """
    Check if a solver supports a given feature.

    Parameters
    ----------
    solver_name : str
        Name of the solver (e.g., "gurobi", "highs")
    feature : SolverFeature
        The feature to check for

    Returns
    -------
    bool
        True if the solver supports the feature, False otherwise

    Raises
    ------
    KeyError
        If the solver is not in the registry
    """
    if solver_name not in SOLVER_REGISTRY:
        raise KeyError(f"Unknown solver: {solver_name}")
    return SOLVER_REGISTRY[solver_name].supports(feature)


def get_solvers_with_feature(feature: SolverFeature) -> list[str]:
    """
    Get all solvers that support a given feature.

    Parameters
    ----------
    feature : SolverFeature
        The feature to filter by

    Returns
    -------
    list[str]
        List of solver names supporting the feature
    """
    return [name for name, info in SOLVER_REGISTRY.items() if info.supports(feature)]


def get_available_solvers_with_feature(feature: SolverFeature) -> list[str]:
    """
    Get installed solvers that support a given feature.

    Parameters
    ----------
    feature : SolverFeature
        The feature to filter by

    Returns
    -------
    list[str]
        List of installed solver names supporting the feature
    """
    from linopy.solvers import available_solvers
    return [s for s in get_solvers_with_feature(feature) if s in available_solvers]
```

### 4. Backward Compatibility Layer

To maintain backward compatibility, we'll keep the existing module-level lists but generate them from the registry:

```python
# These are generated from the registry for backward compatibility
QUADRATIC_SOLVERS = get_solvers_with_feature(SolverFeature.QUADRATIC_OBJECTIVE)
NO_SOLUTION_FILE_SOLVERS = get_solvers_with_feature(SolverFeature.SOLUTION_FILE_NOT_NEEDED)

# Derived (filtered by availability) - also generated from registry
quadratic_solvers = get_available_solvers_with_feature(SolverFeature.QUADRATIC_OBJECTIVE)
```

No deprecation warnings - these lists remain fully supported and are simply generated from the registry now.

---

## Migration Plan

### Step 1: Create the New Module

Create `linopy/solver_capabilities.py` with:
- `SolverFeature` enum
- `SolverInfo` dataclass
- `SOLVER_REGISTRY` dictionary
- Helper functions

### Step 2: Update `solvers.py`

- Import the new module
- Generate existing lists from registry (backward compat):
  - `QUADRATIC_SOLVERS`
  - `NO_SOLUTION_FILE_SOLVERS`
  - `quadratic_solvers` (filtered by availability)

### Step 3: Update `model.py`

Replace:
```python
# Before (model.py:1199)
if solver_name in NO_SOLUTION_FILE_SOLVERS:

# Before (model.py:1211)
if solver_name not in quadratic_solvers:

# Before (model.py:1234)
if solver_name in ["glpk", "cbc"]:

# Before (model.py:1342)
if solver_name in ["gurobi", "xpress"]:
```

With:
```python
# After
if solver_supports(solver_name, SolverFeature.SOLUTION_FILE_NOT_NEEDED):
if not solver_supports(solver_name, SolverFeature.QUADRATIC_OBJECTIVE):
if not solver_supports(solver_name, SolverFeature.LP_FILE_NAMES):
if solver_supports(solver_name, SolverFeature.IIS_COMPUTATION):
```

**Note**: Platform-specific bugs (e.g., SCIP quadratic on Windows) are handled by adjusting the registry at import time in `solver_capabilities.py`, not by runtime list modification.

### Step 4: Update `variables.py`

Replace:
```python
# Before (variables.py:852)
if self.model.solver_name != "gurobi":
```

With:
```python
# After
if not solver_supports(self.model.solver_name, SolverFeature.SOLVER_ATTRIBUTE_ACCESS):
```

### Step 5: Update Tests (optional for POC)

- Update test parametrization to use `get_available_solvers_with_feature()`
- Replace hardcoded solver lists in test files

Note: `io.py` doesn't need changes - the check at line 508 (`"highs" not in available_solvers`) is about package availability, not solver features.

---

## File Changes Summary

| File | Changes |
|------|---------|
| `linopy/solver_capabilities.py` | **NEW** - Core registry module (~170 lines) |
| `linopy/solvers.py` | Import registry, generate compat lists (~10 lines changed) |
| `linopy/model.py` | Replace 4 hardcoded checks |
| `linopy/variables.py` | Replace 1 hardcoded check |
| `test/test_optimization.py` | Remove Windows SCIP workaround (handled in registry) |
| `linopy/__init__.py` | Export `SolverFeature`, `solver_supports` (optional) |

---

## API Design Decisions

### Decision 1: Enum vs String Features

**Chosen: Enum**

Pros:
- Type safety and IDE autocomplete
- Prevents typos in feature names
- Clear documentation of all available features

Cons:
- Slightly more verbose
- Requires import

### Decision 2: Registry Structure

**Chosen: Dict[str, SolverInfo]**

Alternative considered: Class-based registry with registration decorator

The dict approach is simpler and sufficient for our needs. We can always migrate to a more sophisticated approach in Phase 2 if needed.

### Decision 3: Immutable SolverInfo

**Chosen: frozen dataclass with frozenset**

This prevents accidental modification of solver capabilities at runtime.

---

## Testing Strategy

1. **Unit tests for the registry module**
   - Test `solver_supports()` with valid/invalid solvers
   - Test `get_solvers_with_feature()` returns correct solvers
   - Test that all solvers in registry are valid

2. **Integration tests**
   - Ensure existing tests still pass after migration
   - Test that backward compatibility lists match expected values

3. **Validation**
   - Add a test that validates registry entries against actual solver behavior (where possible)

---

## Future Considerations (Phase 2)

If we decide to extract this to a separate package:

1. **Version-aware features**: `SolverFeature.QUADRATIC_OBJECTIVE` could have version requirements
2. **Runtime detection**: Auto-detect solver versions and adjust capabilities
3. **Custom solver registration**: Allow users to register proprietary solvers
4. **Solver metadata**: License type, homepage, documentation links
5. **Feature dependencies**: Some features may require others

---

## Design Decisions (Resolved)

1. **Feature granularity**: Start with current features as proof of concept. More granular features (SOS constraints, indicator constraints, semi-continuous variables) can be added later.

2. **Runtime extensibility**: No. The registry is fixed at module load time. Users cannot register custom solvers.

3. **Validation timing**: Keep validation at solve time only. The framework is solver-agnostic - solver choice happens at solve time, not when building the model.

4. **Backward compatibility**: Generate old lists directly from registry without deprecation warnings. This simplifies the transition and maintains full compatibility.

---

## Implementation Order

1. Create `solver_capabilities.py` with full registry
2. Update `solvers.py` to generate lists from registry
3. Update `model.py` (4 checks)
4. Update `variables.py` (1 check)
5. Run tests and verify everything works

Estimated scope: ~150 lines of new code, ~20 lines modified across existing files.
