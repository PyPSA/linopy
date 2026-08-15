"""
Which order an SOS set runs in: the declared one, or the one its labels imply.

Where the labels ascend the two coincide — the common case, and every piecewise
model. Where they do not, the declared order wins.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pytest

from linopy import Model, available_solvers

#: Gains that separate the two orders: the first two members declared are worth
#: 1.0 each, the last 0.1. Declaration order can take the first two and scores
#: 2.0; label order cannot, and settles for 1.1.
GAINS = [1.0, 1.0, 0.1]

needs_highs = pytest.mark.skipif(
    "highs" not in available_solvers, reason="HiGHS not installed"
)


@pytest.fixture
def sos_model() -> Callable[..., Model]:
    """A three-member SOS set over ``labels``, maximising ``GAINS``."""

    def build(labels: list, sos_type: Literal[1, 2] = 2, **kwargs: Any) -> Model:
        m = Model()
        index = pd.Index(labels, name="size")
        x = m.add_variables(lower=0, upper=1, coords=[index], name="x")
        m.add_sos_constraints(x, sos_type=sos_type, sos_dim="size", **kwargs)
        m.add_objective((x * pd.Series(GAINS, index=index)).sum(), sense="max")
        return m

    return build


@pytest.fixture
def optimum(sos_model: Callable[..., Model]) -> Callable[..., float]:
    """The objective such a set reaches."""

    def solve(labels: list, sos_type: Literal[1, 2] = 2, **kwargs: Any) -> float:
        m = sos_model(labels, sos_type=sos_type, **kwargs)
        m.solve(solver_name="highs", reformulate_sos=True, output_flag=False)
        assert m.objective.value is not None
        return float(m.objective.value)

    return solve


@pytest.fixture
def lp_sos_section(tmp_path: Path) -> Callable[[Model], str]:
    """The ``sos`` section a model writes into an LP file."""

    def read(m: Model) -> str:
        fn = tmp_path / "sos.lp"
        m.to_file(fn, io_api="lp")
        return fn.read_text().split("\nsos\n")[1]

    return read


def test_ascending_labels_are_written_as_weights_verbatim(
    sos_model: Callable[..., Model], lp_sos_section: Callable[[Model], str]
) -> None:
    section = lp_sos_section(sos_model([0.0, 1.5, 3.5]))

    assert "1.5" in section
    assert "3.5" in section


@needs_highs
def test_ascending_labels_make_the_first_two_members_adjacent(
    optimum: Callable[..., float],
) -> None:
    assert optimum([0, 1, 2]) == pytest.approx(2.0)


@needs_highs
@pytest.mark.parametrize(
    "labels",
    [
        pytest.param([0, 1, 2], id="ascending"),
        pytest.param([30, 10, 20], id="unordered"),
        pytest.param([2, 1, 0], id="descending"),
    ],
)
def test_sos1_optimum_is_independent_of_label_order(
    optimum: Callable[..., float], labels: list
) -> None:
    assert optimum(labels, sos_type=1) == pytest.approx(1.0)


@needs_highs
def test_descending_labels_reach_the_same_optimum(
    optimum: Callable[..., float],
) -> None:
    """Reversing a set leaves which of its members are adjacent unchanged."""
    assert optimum([2, 1, 0]) == pytest.approx(optimum([0, 1, 2]))


@pytest.mark.parametrize(
    "labels",
    [pytest.param([0, 1, 2], id="ascending"), pytest.param([2, 1, 0], id="descending")],
)
def test_monotonic_labels_do_not_warn(
    sos_model: Callable[..., Model], recwarn: pytest.WarningsRecorder, labels: list
) -> None:
    sos_model(labels)

    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize(
    "sos_type", [pytest.param(1, id="sos1"), pytest.param(2, id="sos2")]
)
def test_string_labels_are_accepted(
    sos_model: Callable[..., Model], sos_type: Literal[1, 2]
) -> None:
    m = sos_model(["s1", "s2", "s3"], sos_type=sos_type)

    assert list(m.variables.sos) == ["x"]


@needs_highs
@pytest.mark.parametrize(
    "labels",
    [
        pytest.param([30, 10, 20], id="numeric-out-of-order"),
        pytest.param(["s1", "s2", "s3"], id="string"),
    ],
)
def test_declaration_order_governs_adjacency(
    optimum: Callable[..., float], labels: list
) -> None:
    assert optimum(labels) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param([30, 10, 20], id="numeric-out-of-order"),
        pytest.param(["s1", "s2", "s3"], id="string"),
    ],
)
def test_weights_are_positions_where_labels_do_not_ascend(
    sos_model: Callable[..., Model],
    lp_sos_section: Callable[[Model], str],
    labels: list,
) -> None:
    section = lp_sos_section(sos_model(labels))

    assert "x0:0" in section
    assert "x1:1" in section
    assert "x2:2" in section


def test_labels_that_regroup_the_set_warn(sos_model: Callable[..., Model]) -> None:
    with pytest.warns(UserWarning, match="order"):
        sos_model([30, 10, 20])
