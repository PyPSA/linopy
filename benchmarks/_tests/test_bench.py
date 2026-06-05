"""
Tests for the ad-hoc ``bench`` helper.

The contract under test is the *seam*: a ``bench`` result must round-trip
into ``snapshot.load_long_df`` exactly like a real snapshot, and its
in-process ``to_df`` must line up column-for-column with the loaded frame.
These are the only non-obvious behaviours — the timing math itself is not
asserted beyond "finite and positive", since wall-clock values aren't
reproducible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import linopy
from benchmarks import REGISTRY, bench
from benchmarks.phases import touch_matrices
from benchmarks.snapshot import load_long_df


def _tiny() -> int:
    return sum(range(1000))


def _alloc() -> int:
    # Allocate ~16 MB so the memray peak is unambiguously above zero;
    # ``_tiny`` allocates nothing measurable.
    data = [0] * 2_000_000
    return len(data)


def test_timing_snapshot_round_trips_into_loader(tmp_path: Path) -> None:
    """A synthesized id parses back into the (phase, spec, size) columns."""
    snap = tmp_path / "t.json"
    bench.time(_tiny, rounds=3).to_snapshot(snap, spec="basic", size=100, phase="build")

    df, unit = load_long_df([snap])
    assert unit == "s"
    assert len(df) == 1
    row = df.iloc[0]
    assert (row["phase"], row["spec"], row["size"]) == ("build", "basic", 100)
    assert row["value"] > 0


def test_compare_writes_n_entries(tmp_path: Path) -> None:
    """``compare`` collects N cases into one snapshot → N loadable rows."""
    snap = tmp_path / "cmp.json"
    rs = bench.compare({"a": _tiny, "b": _tiny, "c": _tiny}, kind="time", rounds=2)
    rs.to_snapshot(snap)

    df, unit = load_long_df([snap])
    assert unit == "s"
    assert len(df) == 3
    assert set(df["test_id"]) == {"a", "b", "c"}


def test_to_df_columns_match_loader(tmp_path: Path) -> None:
    """In-process ``to_df`` shares the loader's exact column set/order."""
    snap = tmp_path / "t.json"
    result = bench.time(_tiny, rounds=2)
    result.to_snapshot(snap, spec="basic", size=10, phase="build")

    loaded, _ = load_long_df([snap])
    assert list(result.to_df().columns) == list(loaded.columns)


def test_memory_path_round_trips(tmp_path: Path) -> None:
    """Memory results carry MiB and round-trip through the loader."""
    pytest.importorskip("memray")
    snap = tmp_path / "m.json"
    result = bench.memory(_alloc)
    assert result.peak_mib > 0
    result.to_snapshot(snap, spec="basic", size=10, phase="build")

    df, unit = load_long_df([snap])
    assert unit == "MiB"
    assert df.iloc[0]["value"] > 0


def test_phase_verb_on_custom_model() -> None:
    """The headline use case: a phase verb timed on a hand-built model."""
    m = linopy.Model()
    x = m.add_variables(lower=0, name="x")
    m.add_constraints(x >= 1)
    m.add_objective(x)

    result = bench.time(touch_matrices, m, rounds=2)
    assert result.stats["min"] > 0
    assert result.stats["rounds"] == 2


def test_registry_builder_times() -> None:
    """A registry builder is a plain callable — no special-casing needed."""
    result = bench.time(REGISTRY["basic"].build, 50, rounds=2)
    assert result.stats["min"] > 0


def test_partial_id_spec_rejected(tmp_path: Path) -> None:
    """A half-given (spec/size/phase) id is ambiguous and must error."""
    result = bench.time(_tiny, rounds=1)
    with pytest.raises(ValueError, match="given together"):
        result.to_snapshot(tmp_path / "x.json", spec="basic")
