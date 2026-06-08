"""Tests for indicator constraint support."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import linopy
from linopy import Model, available_solvers
from linopy.constraints import Constraint, CSRConstraint
from linopy.variables import Variable

requires_gurobi = pytest.mark.skipif(
    "gurobi" not in available_solvers, reason="Gurobi not installed"
)


@pytest.fixture
def mbx() -> tuple[Model, Variable, Variable]:
    """Model with a scalar binary ``b`` and continuous ``x`` in [0, 10]."""
    m = Model()
    b = m.add_variables(name="b", binary=True)
    x = m.add_variables(lower=0, upper=10, name="x")
    return m, b, x


@pytest.fixture
def coords_mbx() -> tuple[Model, Variable, Variable, pd.RangeIndex]:
    """Model with index-aligned binary ``b`` and continuous ``x`` over a length-3 index."""
    m = Model()
    idx = pd.RangeIndex(3, name="i")
    b = m.add_variables(coords=[idx], name="b", binary=True)
    x = m.add_variables(coords=[idx], lower=0, upper=10, name="x")
    return m, b, x, idx


class TestConstruction:
    """Creating indicator constraints and validating their arguments."""

    def test_basic_fields(self, mbx: tuple[Model, Variable, Variable]) -> None:
        """Indicator constraint is created with correct fields."""
        m, b, x = mbx
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

        assert "ic0" in m.indicator_constraints
        ic = m.indicator_constraints["ic0"]
        for field in (
            "coeffs",
            "vars",
            "sign",
            "rhs",
            "binary_var",
            "binary_val",
            "labels",
        ):
            assert field in ic

    def test_auto_name(self, mbx: tuple[Model, Variable, Variable]) -> None:
        """Omitting ``name`` auto-generates an ``indcon`` name."""
        m, b, x = mbx
        con = m.add_indicator_constraints(b, 1, x, "<=", 5)
        assert con.name.startswith("indcon")
        assert con.name in m.indicator_constraints

    @pytest.mark.parametrize(
        ("build_lhs", "kwargs"),
        [
            (lambda x: x <= 5, {}),
            (lambda x: 2 * x, {"sign": "<=", "rhs": 5}),
            (lambda x: 1 * x.at[()] <= 5, {}),
        ],
        ids=["constraint", "linexpr", "scalarcon"],
    )
    def test_valid_lhs(
        self, mbx: tuple[Model, Variable, Variable], build_lhs: object, kwargs: dict
    ) -> None:
        """Each accepted lhs form (constraint, expression, scalar) builds an indicator."""
        m, b, x = mbx
        ic = m.add_indicator_constraints(b, 1, build_lhs(x), name="ic0", **kwargs)  # type: ignore[operator]
        assert ic.is_indicator
        assert "ic0" in m.indicator_constraints

    @pytest.mark.parametrize(
        ("build_lhs", "kwargs", "exc", "match"),
        [
            (lambda x: x <= 5, {"sign": "<=", "rhs": 5}, ValueError, "must be None"),
            (
                lambda x: 2 * x,
                {},
                ValueError,
                "are required when",
            ),
            (lambda x: x, {}, ValueError, "are required when"),
            (
                lambda x: 1 * x.at[()] <= 5,
                {"sign": "<=", "rhs": 5},
                ValueError,
                "must be None",
            ),
            (lambda x: "not-a-constraint", {}, TypeError, "must be a LinearExpression"),
        ],
        ids=[
            "con+sign",
            "linexpr-no-sign",
            "var-no-sign",
            "scalarcon+sign",
            "bad-type",
        ],
    )
    def test_invalid_lhs(
        self,
        mbx: tuple[Model, Variable, Variable],
        build_lhs: object,
        kwargs: dict,
        exc: type[Exception],
        match: str,
    ) -> None:
        """Invalid lhs / sign / rhs combinations raise with a clear message."""
        m, b, x = mbx
        with pytest.raises(exc, match=match):
            m.add_indicator_constraints(b, 1, build_lhs(x), name="ic0", **kwargs)  # type: ignore[operator]

    def test_non_binary_var_raises(self, mbx: tuple[Model, Variable, Variable]) -> None:
        """Non-binary variable raises ValueError."""
        m, _, x = mbx
        y = m.add_variables(lower=0, upper=1, name="y")
        with pytest.raises(ValueError, match="must be binary"):
            m.add_indicator_constraints(y, 1, x, "<=", 5)

    def test_bad_binary_val_raises(self, mbx: tuple[Model, Variable, Variable]) -> None:
        """Invalid binary_val raises ValueError."""
        m, b, x = mbx
        with pytest.raises(ValueError, match="must be 0 or 1"):
            m.add_indicator_constraints(b, 2, x, "<=", 5)

    def test_duplicate_name_raises(self, mbx: tuple[Model, Variable, Variable]) -> None:
        """Duplicate name raises ValueError."""
        m, b, x = mbx
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")
        with pytest.raises(ValueError, match="already assigned"):
            m.add_indicator_constraints(b, 0, x, ">=", 0, name="ic0")


class TestContainer:
    """How indicator constraints live in (and stay separate within) the container."""

    def test_separate_from_regular(self, mbx: tuple[Model, Variable, Variable]) -> None:
        """Indicator constraints are separate from regular constraints."""
        m, b, x = mbx
        m.add_constraints(x >= 0, name="regular")
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

        assert "regular" in m.constraints
        assert "regular" in m.constraints.regular
        assert "ic0" in m.constraints
        assert "ic0" in m.constraints.indicator
        assert "ic0" not in m.constraints.regular
        assert "ic0" in m.indicator_constraints

    def test_in_unified_container(self, mbx: tuple[Model, Variable, Variable]) -> None:
        """Indicator constraint lives in the unified container, not in regular."""
        m, b, x = mbx
        m.add_constraints(x >= 0, name="regular")
        ic = m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

        assert isinstance(ic, Constraint)
        assert ic.is_indicator
        assert "ic0" in m.constraints
        assert "ic0" in m.constraints.indicator
        assert "ic0" not in m.constraints.regular
        assert "ic0" in m.indicator_constraints
        assert "regular" in m.constraints.regular
        assert "regular" not in m.constraints.indicator

    def test_remove(self, mbx: tuple[Model, Variable, Variable]) -> None:
        """Indicator constraints can be removed."""
        m, b, x = mbx
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")
        assert "ic0" in m.indicator_constraints
        m.remove_indicator_constraints("ic0")
        assert "ic0" not in m.indicator_constraints

    def test_regular_constraint_has_no_indicator_fields(
        self, mbx: tuple[Model, Variable, Variable]
    ) -> None:
        """Regular constraints report no indicator metadata, frozen or not."""
        m, _, x = mbx
        m.add_constraints(x <= 5, name="c")
        con = m.constraints["c"]
        assert not con.is_indicator
        assert con.binary_var is None
        assert con.binary_val is None

        frozen = con.freeze()
        assert not frozen.is_indicator
        assert frozen.binary_var is None
        assert frozen.binary_val is None

    def test_with_coords(
        self, coords_mbx: tuple[Model, Variable, Variable, pd.RangeIndex]
    ) -> None:
        """Indicator constraints work with multi-dimensional coords."""
        m, b, x, idx = coords_mbx
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")
        assert m.indicator_constraints["ic0"].labels.size == idx.size


class TestPersistence:
    """Indicator metadata survives freeze, copy, and netCDF round-trips."""

    def test_freeze_roundtrip(self, mbx: tuple[Model, Variable, Variable]) -> None:
        """is_indicator and binary fields survive freeze and freeze->mutable."""
        m, b, x = mbx
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

        frozen = m.constraints["ic0"].freeze()
        assert isinstance(frozen, CSRConstraint)
        assert frozen.is_indicator
        assert frozen.binary_var is not None
        assert frozen.binary_val == 1

        mutable = frozen.mutable()
        assert isinstance(mutable, Constraint)
        assert mutable.is_indicator
        assert mutable.binary_var is not None
        assert np.all(mutable.binary_val == 1)

    def test_copy_preserves(self, mbx: tuple[Model, Variable, Variable]) -> None:
        """Model.copy preserves is_indicator and the binary fields."""
        m, b, x = mbx
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

        ic = m.copy().constraints["ic0"]
        assert ic.is_indicator
        assert ic.binary_var is not None
        assert np.all(ic.binary_val == 1)

    @pytest.mark.parametrize("freeze_constraints", [False, True])
    def test_netcdf_roundtrip(self, tmp_path: Path, freeze_constraints: bool) -> None:
        """is_indicator and binary fields survive a netCDF round-trip."""
        m = Model(freeze_constraints=freeze_constraints)
        b = m.add_variables(name="b", binary=True)
        x = m.add_variables(lower=0, upper=10, name="x")
        m.add_constraints(x >= 0, name="regular")
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

        fn = tmp_path / "model.nc"
        m.to_netcdf(fn)
        ic = linopy.read_netcdf(fn).constraints["ic0"]

        assert ic.is_indicator
        assert ic.binary_var is not None
        assert np.all(ic.binary_val == 1)

    def test_array_binval_roundtrip(self, tmp_path: Path) -> None:
        """A coords-based indicator has per-element binary_val that round-trips."""
        m = Model(freeze_constraints=True)
        idx = pd.RangeIndex(3, name="i")
        b = m.add_variables(coords=[idx], name="b", binary=True)
        x = m.add_variables(coords=[idx], lower=0, upper=10, name="x")
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

        frozen = m.constraints["ic0"]
        assert isinstance(frozen, CSRConstraint)
        assert isinstance(frozen.binary_val, np.ndarray) and frozen.binary_val.ndim == 1
        assert "binary_val" in frozen.data

        fn = tmp_path / "model.nc"
        m.to_netcdf(fn)
        ic = linopy.read_netcdf(fn).constraints["ic0"]
        assert ic.is_indicator
        assert np.all(ic.binary_val == 1)


class TestMatrices:
    """Indicator rows are split out of the regular constraint matrix."""

    def test_matrix_split(
        self, coords_mbx: tuple[Model, Variable, Variable, pd.RangeIndex]
    ) -> None:
        """Regular A excludes indicator rows; indicator arrays carry them."""
        m, b, x, idx = coords_mbx
        m.add_constraints(x >= 0, name="regular")
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

        n = idx.size
        assert m.matrices.A is not None
        assert m.matrices.indicator_A is not None
        assert m.matrices.A.shape[0] == n
        assert m.matrices.indicator_A.shape[0] == n
        assert len(m.matrices.clabels) == n
        assert m.ncons == n

        np.testing.assert_array_equal(m.matrices.indicator_binval, np.full(n, 1))
        np.testing.assert_array_equal(m.matrices.indicator_b, np.full(n, 5.0))
        np.testing.assert_array_equal(m.matrices.indicator_sense, np.full(n, "<"))
        assert m.matrices.indicator_binvar.shape == (n,)

    def test_to_matrix_skips_indicator(
        self, coords_mbx: tuple[Model, Variable, Variable, pd.RangeIndex]
    ) -> None:
        """Constraints.to_matrix drops indicator rows; an indicator-only set raises."""
        m, b, x, idx = coords_mbx
        m.add_constraints(x <= 5, name="regular")
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

        A, labels = m.constraints.to_matrix()
        assert A.shape[0] == idx.size

        with pytest.raises(ValueError, match="No constraints available"):
            m.constraints.indicator.to_matrix()


class TestLPFile:
    """LP export of indicator constraints."""

    def test_with_regular_constraints(
        self, mbx: tuple[Model, Variable, Variable], tmp_path: Path
    ) -> None:
        """LP file contains general constraints section."""
        m, b, x = mbx
        m.add_constraints(x >= 0, name="dummy")
        m.add_objective(x)
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

        fn = tmp_path / "test.lp"
        m.to_file(fn)
        content = fn.read_text()
        assert "= 1 ->" in content
        label = int(m.indicator_constraints["ic0"].labels.item())
        assert f"ic{label}:" in content

    def test_indicator_only(
        self, mbx: tuple[Model, Variable, Variable], tmp_path: Path
    ) -> None:
        """An LP file with no regular constraints still writes the indicator section."""
        m, b, x = mbx
        m.add_objective(x)
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

        fn = tmp_path / "test.lp"
        m.to_file(fn)
        content = fn.read_text()
        assert "s.t." in content
        assert "= 1 ->" in content


class TestSolve:
    """Solving models with indicator constraints."""

    @requires_gurobi
    @pytest.mark.parametrize(
        ("io_api", "trigger", "expected"),
        [
            (None, 1, 5),
            (None, 0, 10),
            ("direct", 1, 5),
            ("direct", 0, 10),
        ],
        ids=["lp-active", "lp-inactive", "direct-active", "direct-inactive"],
    )
    def test_gurobi_enforces_at_trigger(
        self,
        mbx: tuple[Model, Variable, Variable],
        io_api: str | None,
        trigger: int,
        expected: float,
    ) -> None:
        """
        The indicator is enforced (x<=5) only when b matches the trigger value.

        With b fixed to 1, an active trigger caps x at 5; an inactive one leaves
        x free to its upper bound of 10.
        """
        m, b, x = mbx
        m.add_constraints(b >= 1, name="fix_b")
        m.add_indicator_constraints(b, trigger, x, "<=", 5, name="ic0")
        m.add_objective(x, sense="max")
        m.solve(solver_name="gurobi", io_api=io_api)
        assert m.objective.value is not None
        assert np.isclose(m.objective.value, expected, atol=1e-6)

    @requires_gurobi
    def test_gurobi_multiple(self) -> None:
        """Multiple indicators combine: x <= min(10, 5) = 5."""
        m = Model()
        b1 = m.add_variables(name="b1", binary=True)
        b2 = m.add_variables(name="b2", binary=True)
        x = m.add_variables(lower=0, upper=20, name="x")
        m.add_constraints(b1 >= 1, name="fix_b1")
        m.add_constraints(b2 >= 1, name="fix_b2")
        m.add_indicator_constraints(b1, 1, x, "<=", 10, name="ic1")
        m.add_indicator_constraints(b2, 1, x, "<=", 5, name="ic2")
        m.add_objective(x, sense="max")
        m.solve(solver_name="gurobi")
        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 5, atol=1e-6)

    def test_unsupported_solver_raises(
        self, mbx: tuple[Model, Variable, Variable]
    ) -> None:
        """Solvers without indicator support raise ValueError."""
        m, b, x = mbx
        m.add_constraints(x >= 0, name="dummy")
        m.add_objective(x)
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

        for solver in ["glpk", "highs", "mosek", "mindopt"]:
            if solver in available_solvers:
                with pytest.raises(
                    ValueError, match="does not support indicator constraints"
                ):
                    m.solve(solver_name=solver)

    @requires_gurobi
    def test_mip_has_no_duals(self, mbx: tuple[Model, Variable, Variable]) -> None:
        """Indicator rows are skipped when collecting duals; MIPs expose none."""
        m, b, x = mbx
        m.add_constraints(b >= 1, name="fix_b")
        m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")
        m.add_objective(x, sense="max")
        m.solve(solver_name="gurobi")
        with pytest.raises(AttributeError, match="dual"):
            _ = m.matrices.dual
