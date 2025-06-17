# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=./ --cov-report=xml linopy --doctest-modules test

# Run a specific test file
pytest test/test_model.py

# Run a specific test function
pytest test/test_model.py::test_model_creation
```

### Linting and Type Checking
```bash
# Run linter (ruff)
ruff check .
ruff check --fix .  # Auto-fix issues

# Run formatter
ruff format .

# Run type checker
mypy .

# Run all pre-commit hooks
pre-commit run --all-files
```

### Development Setup
```bash
# Create virtual environment and install development dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install uv
uv pip install -e .[dev,solvers]
```

## High-Level Architecture

linopy is a linear optimization library built on top of xarray, providing N-dimensional labeled arrays for variables and constraints. The architecture follows these key principles:

### Core Components

1. **Model** (`model.py`): Central container for optimization problems
   - Manages variables, constraints, and objective
   - Handles solver integration through abstract interfaces
   - Supports chunked operations for memory efficiency
   - Provides matrix representations for solver APIs

2. **Variables** (`variables.py`): Multi-dimensional decision variables
   - Built on xarray.Dataset with labels, lower, and upper bounds
   - Arithmetic operations automatically create LinearExpressions
   - Support for continuous and binary variables
   - Container class (Variables) manages collections with dict-like access

3. **Constraints** (`constraints.py`): Linear inequality/equality constraints
   - Store coefficients, variable references, signs, and RHS values
   - Support ≤, ≥, and = constraints
   - Container class (Constraints) provides organized access

4. **Expressions** (`expressions.py`): Linear combinations of variables
   - LinearExpression: coeffs × vars + const
   - QuadraticExpression: for non-linear optimization
   - Support full arithmetic operations with automatic broadcasting
   - Special `_term` dimension for handling multiple terms

5. **Solvers** (`solvers.py`): Abstract interface with multiple implementations
   - File-based solvers: Write LP/MPS files, call solver, parse results
   - Direct API solvers: Use Python bindings (e.g., gurobipy)
   - Automatic solver detection based on installed packages

### Data Flow Pattern

1. User creates Model and adds Variables with coordinates (dimensions)
2. Variables combined into LinearExpressions through arithmetic
3. Expressions used to create Constraints and Objective
4. Model.solve() converts to solver format and retrieves solution
5. Solution stored back in xarray format with original dimensions

### Key Design Patterns

- **xarray Integration**: All data structures use xarray for dimension handling
- **Lazy Evaluation**: Expressions built symbolically before solving
- **Broadcasting**: Operations automatically align dimensions
- **Solver Abstraction**: Clean separation between model and solver specifics
- **Memory Efficiency**: Support for dask arrays and chunked operations

When modifying the codebase, maintain consistency with these patterns and ensure new features integrate naturally with the xarray-based architecture.

## Working with the Github Repository

* The main branch is `master`.
* Always create a feature branch for new features or bug fixes.
* Use the github cli (gh) to interact with the Github repository.

### GitHub Claude Code Integration

This repository includes Claude Code GitHub Actions for automated assistance:

1. **Automated PR Reviews** (`claude-code-review.yml`):
   - Automatically reviews PRs only when first created (opened)
   - Subsequent reviews require manual `@claude` mention
   - Focuses on Python best practices, xarray patterns, and optimization correctness
   - Can run tests and linting as part of the review
   - **Skip initial review by**: Adding `[skip-review]` or `[WIP]` to PR title, or using draft PRs

2. **Manual Claude Assistance** (`claude.yml`):
   - Trigger by mentioning `@claude` in any:
     - Issue comments
     - Pull request comments
     - Pull request reviews
     - New issue body or title
   - Claude can help with bug fixes, feature implementation, code explanations, etc.

**Note**: Both workflows require the `ANTHROPIC_API_KEY` secret to be configured in the repository settings.


## Development Guidelines

1. Always write tests for new features or bug fixes.
2. Always run the tests after making changes and ensure they pass.
3. Always use ruff for linting and formatting, run `ruff check --fix .` to auto-fix issues.
4. Use type hints and mypy for type checking.
5. Always write tests into the `test` directory, following the naming convention `test_*.py`.
6. Always write temporary and non git-tracked code in the `dev-scripts` directory.
