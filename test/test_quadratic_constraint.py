#!/usr/bin/env python3
"""
Tests for quadratic constraints.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

import linopy
from linopy import Model
from linopy.constraints import QuadraticConstraint


@pytest.fixture
def m() -> Model:
    m = Model()
    m.add_variables(lower=0, name="x")
    m.add_variables(lower=0, name="y")
    return m


@pytest.fixture
def x(m: Model) -> linopy.Variable:
    return m.variables["x"]


@pytest.fixture
def y(m: Model) -> linopy.Variable:
    return m.variables["y"]


class TestQuadraticConstraintCreation:
    """Tests for quadratic constraint creation."""

    def test_create_simple_quadratic_constraint(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test creating a simple quadratic constraint."""
        qexpr = x * x + y * y
        qcon = m.add_quadratic_constraints(qexpr, "<=", 100, name="qc1")

        assert isinstance(qcon, QuadraticConstraint)
        assert qcon.name == "qc1"
        assert str(qcon.sign.values) == "<="
        assert float(qcon.rhs.values) == 100.0

    def test_create_mixed_quadratic_constraint(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test creating a quadratic constraint with both linear and quadratic terms."""
        qexpr = x * x + 2 * x * y + y * y + 3 * x + 4 * y
        qcon = m.add_quadratic_constraints(qexpr, "<=", 100, name="mixed")

        assert isinstance(qcon, QuadraticConstraint)
        # Check repr works
        repr_str = repr(qcon)
        assert "x²" in repr_str or "x^2" in repr_str or "x·x" in repr_str

    def test_create_cross_product_constraint(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test creating a constraint with cross product term."""
        qexpr = x * y
        qcon = m.add_quadratic_constraints(qexpr, "<=", 10, name="cross")

        assert isinstance(qcon, QuadraticConstraint)

    def test_create_constraint_with_different_signs(
        self, m: Model, x: linopy.Variable
    ) -> None:
        """Test creating constraints with different comparison operators."""
        qexpr = x * x

        qcon_le = m.add_quadratic_constraints(qexpr, "<=", 100, name="qc_le")
        assert str(qcon_le.sign.values) == "<="

        qcon_ge = m.add_quadratic_constraints(qexpr, ">=", 1, name="qc_ge")
        assert str(qcon_ge.sign.values) == ">="

        qcon_eq = m.add_quadratic_constraints(qexpr, "==", 50, name="qc_eq")
        assert str(qcon_eq.sign.values) == "="

    def test_create_constraint_with_negative_rhs(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test creating a constraint where terms move to create negative RHS."""
        qexpr = x * x - 100
        qcon = m.add_quadratic_constraints(qexpr, "<=", 0, name="neg_rhs")

        assert isinstance(qcon, QuadraticConstraint)


class TestQuadraticConstraintsContainer:
    """Tests for the QuadraticConstraints container class."""

    def test_empty_quadratic_constraints(self) -> None:
        """Test that a new model has empty quadratic constraints."""
        m = Model()
        assert len(m.quadratic_constraints) == 0
        assert list(m.quadratic_constraints) == []

    def test_add_single_constraint(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test adding a single quadratic constraint."""
        qexpr = x * x + y * y
        m.add_quadratic_constraints(qexpr, "<=", 100, name="qc1")

        assert len(m.quadratic_constraints) == 1
        assert "qc1" in m.quadratic_constraints
        assert m.quadratic_constraints["qc1"].name == "qc1"

    def test_add_multiple_constraints(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test adding multiple quadratic constraints."""
        m.add_quadratic_constraints(x * x, "<=", 100, name="qc1")
        m.add_quadratic_constraints(y * y, "<=", 50, name="qc2")
        m.add_quadratic_constraints(x * y, "<=", 25, name="qc3")

        assert len(m.quadratic_constraints) == 3
        assert set(m.quadratic_constraints) == {"qc1", "qc2", "qc3"}

    def test_remove_constraint(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test removing a quadratic constraint."""
        m.add_quadratic_constraints(x * x, "<=", 100, name="qc1")
        m.add_quadratic_constraints(y * y, "<=", 50, name="qc2")

        assert len(m.quadratic_constraints) == 2

        m.quadratic_constraints.remove("qc1")

        assert len(m.quadratic_constraints) == 1
        assert "qc1" not in m.quadratic_constraints
        assert "qc2" in m.quadratic_constraints


class TestModelTypeDetection:
    """Tests for model type detection with quadratic constraints."""

    def test_model_type_qclp(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test that adding quadratic constraints changes model type to QCLP."""
        # Add linear objective
        m.add_objective(x + y)

        # Add quadratic constraint
        m.add_quadratic_constraints(x * x + y * y, "<=", 100, name="qc1")

        assert m.type == "QCLP"
        assert m.has_quadratic_constraints

    def test_model_type_qcqp(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test QCQP type detection with quadratic objective and constraints."""
        # Add quadratic objective
        m.add_objective(x * x + y * y)

        # Add quadratic constraint
        m.add_quadratic_constraints(x * y, "<=", 10, name="qc1")

        assert m.type == "QCQP"

    def test_model_type_miqclp(self, x: linopy.Variable, y: linopy.Variable) -> None:
        """Test MIQCLP type with integers and quadratic constraints."""
        m = Model()
        x = m.add_variables(lower=0, upper=10, integer=True, name="x")
        y = m.add_variables(lower=0, name="y")

        m.add_objective(x + y)
        m.add_quadratic_constraints(y * y, "<=", 100, name="qc1")

        assert "MI" in m.type
        assert "QC" in m.type


class TestQuadraticConstraintProperties:
    """Tests for QuadraticConstraint properties and methods."""

    def test_flat_property(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test the flat property returns a DataFrame."""
        qexpr = x * x + 2 * x * y + y * y
        qcon = m.add_quadratic_constraints(qexpr, "<=", 100, name="qc1")

        flat_df = qcon.flat
        assert isinstance(flat_df, pd.DataFrame)
        assert "coeffs" in flat_df.columns

    def test_to_polars(self, m: Model, x: linopy.Variable, y: linopy.Variable) -> None:
        """Test the to_polars method."""
        qexpr = x * x + y * y
        qcon = m.add_quadratic_constraints(qexpr, "<=", 100, name="qc1")

        df = qcon.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert "coeffs" in df.columns
        assert "is_quadratic" in df.columns

    def test_ncons_property(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test the ncons property of QuadraticConstraints container."""
        m.add_quadratic_constraints(x * x, "<=", 100, name="qc1")
        m.add_quadratic_constraints(y * y, "<=", 50, name="qc2")

        assert m.quadratic_constraints.ncons == 2

    def test_labels_property(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test the labels property."""
        qcon = m.add_quadratic_constraints(x * x + y * y, "<=", 100, name="qc1")

        labels = qcon.labels
        assert labels is not None


class TestLPFileExport:
    """Tests for LP file export with quadratic constraints."""

    def test_lp_file_with_quadratic_constraint(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test that quadratic constraints are written to LP files."""
        m.add_objective(x + y)
        m.add_constraints(x + y <= 10, name="linear_c")
        m.add_quadratic_constraints(x * x + y * y, "<=", 100, name="qc1")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lp", delete=False) as f:
            fn = Path(f.name)

        m.to_file(fn, progress=False)
        content = fn.read_text()

        # Check that quadratic constraint is in the file
        assert "qc0" in content or "qc1" in content
        assert "^ 2" in content  # Squared term
        assert "<=" in content

        # Clean up
        fn.unlink()

    def test_lp_file_with_mixed_constraint(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test LP export with mixed linear/quadratic constraint."""
        m.add_objective(x + y)
        qexpr = x * x + 2 * x * y + y * y + 3 * x + 4 * y
        m.add_quadratic_constraints(qexpr, "<=", 100, name="mixed")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lp", delete=False) as f:
            fn = Path(f.name)

        m.to_file(fn, progress=False)
        content = fn.read_text()

        # Check for both linear and quadratic terms
        assert "[" in content  # Opening bracket for quadratic section
        assert "]" in content  # Closing bracket
        assert "^ 2" in content or "* x" in content  # Quadratic terms

        # Clean up
        fn.unlink()

    def test_lp_file_with_multidimensional_constraint(self) -> None:
        """Test LP export with multi-dimensional quadratic constraints."""
        m = Model()
        x = m.add_variables(lower=0, coords=[range(3)], name="x")
        y = m.add_variables(lower=0, coords=[range(3)], name="y")

        m.add_objective((x + y).sum())
        m.add_quadratic_constraints(x * x + y * y, "<=", 25, name="circles")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lp", delete=False) as f:
            fn = Path(f.name)

        m.to_file(fn, progress=False)
        content = fn.read_text()

        # Should have 3 quadratic constraints (qc0, qc1, qc2)
        assert "qc0:" in content
        assert "qc1:" in content
        assert "qc2:" in content

        # Clean up
        fn.unlink()


class TestSolverValidation:
    """Tests for solver validation with quadratic constraints."""

    def test_highs_rejects_quadratic_constraints(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test that HiGHS raises an error for quadratic constraints."""
        if "highs" not in linopy.available_solvers:
            pytest.skip("HiGHS not available")

        m.add_objective(x + y)  # Linear objective
        m.add_quadratic_constraints(x * x + y * y, "<=", 100, name="qc1")

        # HiGHS supports QP (quadratic objective) but not QCP (quadratic constraints)
        from linopy.solvers import quadratic_constraint_solvers

        if "highs" not in quadratic_constraint_solvers:
            with pytest.raises(ValueError, match="does not support quadratic constraints"):
                m.solve(solver_name="highs")

    def test_highs_accepts_quadratic_objective(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test that HiGHS accepts quadratic objectives (but not QC)."""
        if "highs" not in linopy.available_solvers:
            pytest.skip("HiGHS not available")

        # Quadratic objective, no quadratic constraints
        m.add_objective(x * x + y * y)
        m.add_constraints(x + y >= 1, name="c1")

        # This should work - HiGHS supports QP
        status, _ = m.solve(solver_name="highs")
        assert status == "ok"

    def test_supported_solver_accepts_quadratic_constraints(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test that supported solvers accept quadratic constraints."""
        from linopy.solvers import quadratic_constraint_solvers

        # Find a solver that supports QC
        available_qc_solvers = [
            s for s in quadratic_constraint_solvers if s in linopy.available_solvers
        ]
        if not available_qc_solvers:
            pytest.skip("No QC-supporting solver available")

        solver = available_qc_solvers[0]

        m.add_objective(x + y, sense="max")
        m.add_constraints(x + y <= 10, name="budget")
        m.add_quadratic_constraints(x * x + y * y, "<=", 25, name="circle")

        # Should succeed
        status, _ = m.solve(solver_name=solver)
        assert status == "ok"

    def test_is_quadratic_with_qc_only(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test that is_quadratic is True when only QC are present."""
        m.add_objective(x + y)  # Linear objective
        m.add_quadratic_constraints(x * x, "<=", 10, name="qc")

        assert m.has_quadratic_constraints is True
        assert m.objective.is_quadratic is False
        assert m.is_quadratic is True  # True because of QC

    def test_is_quadratic_with_quadratic_objective_only(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test that is_quadratic is True when only quadratic objective."""
        m.add_objective(x * x + y * y)  # Quadratic objective
        m.add_constraints(x + y <= 10, name="c")  # Linear constraint

        assert m.has_quadratic_constraints is False
        assert m.objective.is_quadratic is True
        assert m.is_quadratic is True  # True because of objective


class TestQuadraticConstraintRepr:
    """Tests for QuadraticConstraint string representations."""

    def test_repr_simple(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test repr of simple quadratic constraint."""
        qcon = m.add_quadratic_constraints(x * x, "<=", 100, name="simple")
        repr_str = repr(qcon)

        assert "QuadraticConstraint" in repr_str
        assert "simple" in repr_str
        assert "100" in repr_str

    def test_repr_with_cross_term(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test repr includes cross product terms."""
        qcon = m.add_quadratic_constraints(x * y, "<=", 50, name="cross")
        repr_str = repr(qcon)

        assert "QuadraticConstraint" in repr_str
        # Should show cross product (x·y or similar)

    def test_quadratic_constraints_container_repr(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test repr of QuadraticConstraints container."""
        m.add_quadratic_constraints(x * x, "<=", 100, name="qc1")
        m.add_quadratic_constraints(y * y, "<=", 50, name="qc2")

        repr_str = repr(m.quadratic_constraints)
        assert "QuadraticConstraints" in repr_str
        assert "qc1" in repr_str or "2" in repr_str  # Either name or count

    def test_empty_container_repr(self) -> None:
        """Test repr of empty QuadraticConstraints."""
        m = Model()
        repr_str = repr(m.quadratic_constraints)
        assert "QuadraticConstraints" in repr_str


class TestMatrixAccessor:
    """Tests for matrix accessor with quadratic constraints."""

    def test_qclabels(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test qclabels property."""
        m.add_objective(x + y)
        m.add_quadratic_constraints(x * x, "<=", 25, name="qc1")
        m.add_quadratic_constraints(y * y, ">=", 10, name="qc2")

        labels = m.matrices.qclabels
        assert len(labels) == 2
        assert labels[0] == 0
        assert labels[1] == 1

    def test_qc_sense(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test qc_sense property."""
        m.add_objective(x + y)
        m.add_quadratic_constraints(x * x, "<=", 25, name="qc1")
        m.add_quadratic_constraints(y * y, ">=", 10, name="qc2")

        senses = m.matrices.qc_sense
        assert len(senses) == 2
        assert senses[0] == "<="
        assert senses[1] == ">="

    def test_qc_rhs(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test qc_rhs property."""
        m.add_objective(x + y)
        m.add_quadratic_constraints(x * x, "<=", 25, name="qc1")
        m.add_quadratic_constraints(y * y, ">=", 10, name="qc2")

        rhs = m.matrices.qc_rhs
        assert len(rhs) == 2
        assert rhs[0] == 25.0
        assert rhs[1] == 10.0

    def test_Qc_matrices(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test Qc property returns Q matrices."""
        m.add_objective(x + y)
        m.add_quadratic_constraints(x * x + y * y, "<=", 25, name="circle")

        Qc = m.matrices.Qc
        assert len(Qc) == 1
        Q = Qc[0].toarray()

        # Should be symmetric with doubled diagonal
        assert Q[0, 0] == 2.0  # x^2 coefficient doubled
        assert Q[1, 1] == 2.0  # y^2 coefficient doubled

    def test_Qc_cross_terms(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test Qc with cross product terms."""
        m.add_objective(x + y)
        m.add_quadratic_constraints(x * y, "<=", 10, name="cross")

        Qc = m.matrices.Qc
        Q = Qc[0].toarray()

        # Cross term should be symmetric
        assert Q[0, 1] == 1.0
        assert Q[1, 0] == 1.0
        assert Q[0, 0] == 0.0  # No x^2 term
        assert Q[1, 1] == 0.0  # No y^2 term

    def test_qc_linear(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test qc_linear property."""
        m.add_objective(x + y)
        m.add_quadratic_constraints(x * x + 3 * x + 4 * y, "<=", 25, name="mixed")

        A = m.matrices.qc_linear
        assert A is not None
        assert A.shape == (1, 2)  # 1 constraint, 2 variables

        A_dense = A.toarray()
        assert A_dense[0, 0] == 3.0  # coefficient of x
        assert A_dense[0, 1] == 4.0  # coefficient of y

    def test_empty_quadratic_constraints(self, m: Model) -> None:
        """Test matrix accessors with no quadratic constraints."""
        m.add_objective(m.variables["x"])

        assert len(m.matrices.qclabels) == 0
        assert len(m.matrices.qc_sense) == 0
        assert len(m.matrices.qc_rhs) == 0
        assert len(m.matrices.Qc) == 0
        assert m.matrices.qc_linear is None


class TestNetCDFSerialization:
    """Tests for netCDF serialization of quadratic constraints."""

    def test_netcdf_roundtrip(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test saving and loading a model with quadratic constraints."""
        m.add_objective(x + y)
        m.add_constraints(x + y <= 10, name="linear")
        m.add_quadratic_constraints(x * x + y * y, "<=", 25, name="circle")
        m.add_quadratic_constraints(x * y, "<=", 10, name="mixed")

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            fn = Path(f.name)

        m.to_netcdf(fn)
        m2 = linopy.read_netcdf(fn)

        # Check quadratic constraints were loaded
        assert len(m2.quadratic_constraints) == 2
        assert "circle" in m2.quadratic_constraints
        assert "mixed" in m2.quadratic_constraints
        assert m2.type == "QCLP"

        # Check constraint properties preserved
        assert float(m2.quadratic_constraints["circle"].rhs.values) == 25.0
        assert str(m2.quadratic_constraints["circle"].sign.values) == "<="

        fn.unlink()

    def test_netcdf_roundtrip_multidimensional(self) -> None:
        """Test netCDF roundtrip with multi-dimensional quadratic constraints."""
        m = Model()
        x = m.add_variables(lower=0, coords=[range(3)], name="x")
        y = m.add_variables(lower=0, coords=[range(3)], name="y")

        m.add_objective((x + y).sum())
        m.add_quadratic_constraints(x * x + y * y, "<=", 25, name="circles")

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            fn = Path(f.name)

        m.to_netcdf(fn)
        m2 = linopy.read_netcdf(fn)

        # Check constraint shape preserved
        assert m2.quadratic_constraints["circles"].shape == (3,)
        assert len(m2.quadratic_constraints["circles"].labels.values.ravel()) == 3

        fn.unlink()


class TestMOSEKExport:
    """Tests for MOSEK direct API export with quadratic constraints."""

    def test_to_mosek_with_quadratic_constraints(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test that to_mosek works with quadratic constraints."""
        if "mosek" not in linopy.available_solvers:
            pytest.skip("MOSEK not available")

        m.add_constraints(x + y <= 8, name="budget")
        m.add_quadratic_constraints(x * x + y * y, "<=", 25, name="circle")
        m.add_objective(x + 2 * y, sense="max")

        from linopy.io import to_mosek

        task = to_mosek(m)
        # If we got here without error, the export worked
        assert task is not None

    def test_to_mosek_multidimensional(self) -> None:
        """Test MOSEK export with multi-dimensional quadratic constraints."""
        if "mosek" not in linopy.available_solvers:
            pytest.skip("MOSEK not available")

        m = Model()
        x = m.add_variables(lower=0, coords=[range(3)], name="x")
        y = m.add_variables(lower=0, coords=[range(3)], name="y")

        m.add_constraints(x + y <= 8, name="budget")
        m.add_quadratic_constraints(x * x + y * y, "<=", 25, name="circles")
        m.add_objective((x + 2 * y).sum(), sense="max")

        from linopy.io import to_mosek

        task = to_mosek(m)
        assert task is not None


class TestDualValues:
    """Tests for dual value retrieval for quadratic constraints."""

    def test_qc_dual_with_gurobi(
        self, m: Model, x: linopy.Variable, y: linopy.Variable
    ) -> None:
        """Test that dual values can be retrieved for convex QC with Gurobi."""
        if "gurobi" not in linopy.available_solvers:
            pytest.skip("Gurobi not available")

        m.add_constraints(x + y <= 8, name="budget")
        m.add_quadratic_constraints(x * x + y * y, "<=", 25, name="circle")
        m.add_objective(x + 2 * y, sense="max")

        # Solve with QCPDual enabled
        m.solve(solver_name="gurobi", QCPDual=1)

        # Check dual values exist
        dual = m.quadratic_constraints["circle"].dual
        assert dual is not None
        assert not dual.isnull().all()
        # Dual should be positive for binding <= constraint
        assert float(dual.values) > 0

    def test_qc_dual_multidimensional(self) -> None:
        """Test dual values for multi-dimensional quadratic constraints."""
        if "gurobi" not in linopy.available_solvers:
            pytest.skip("Gurobi not available")

        m = Model()
        x = m.add_variables(lower=0, coords=[range(3)], name="x")
        y = m.add_variables(lower=0, coords=[range(3)], name="y")

        m.add_constraints(x + y <= 8, name="budget")
        m.add_quadratic_constraints(x * x + y * y, "<=", 25, name="circles")
        m.add_objective((x + 2 * y).sum(), sense="max")

        m.solve(solver_name="gurobi", QCPDual=1)

        dual = m.quadratic_constraints["circles"].dual
        assert dual.shape == (3,)
        assert not dual.isnull().all()
