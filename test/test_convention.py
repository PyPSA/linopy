"""Tests for the v1 semantics option and the test harness."""

from __future__ import annotations

import pytest

import linopy


class TestSemanticsOption:
    def test_default_is_legacy(self) -> None:
        linopy.options.reset()
        assert linopy.options["semantics"] == "legacy"

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid semantics"):
            linopy.options["semantics"] = "exact"


class TestHarness:
    """The autouse ``semantics`` fixture and the legacy / v1 markers."""

    @pytest.mark.legacy
    def test_legacy_marker(self) -> None:
        assert linopy.options["semantics"] == "legacy"

    @pytest.mark.v1
    def test_v1_marker(self) -> None:
        assert linopy.options["semantics"] == "v1"
