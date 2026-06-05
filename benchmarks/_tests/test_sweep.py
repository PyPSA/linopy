"""Unit tests for sweep helpers (no venvs spun up)."""

from __future__ import annotations

import pytest

from benchmarks.sweep import _snapshot_label


@pytest.mark.parametrize(
    "spec,expected",
    [
        # plain releases pass through unchanged
        ("0.6.1", "0.6.1"),
        ("0.5.0a1", "0.5.0a1"),
        # git spec pinned to a sha -> the sha (clean, reproducible filename)
        ("git+https://github.com/PyPSA/linopy.git@2993b95", "2993b95"),
        # git spec on a branch -> the branch name
        ("git+https://github.com/PyPSA/linopy.git@main", "main"),
        # PEP 508 local file url -> sanitised (no slashes survive)
        ("linopy @ file:///home/me/linopy", "file-home-me-linopy"),
    ],
)
def test_snapshot_label(spec: str, expected: str) -> None:
    label = _snapshot_label(spec)
    assert label == expected
    # whatever the input, the label must be a safe single path segment.
    assert "/" not in label and " " not in label and label


def test_snapshot_label_never_empty() -> None:
    # a spec that sanitises to nothing still yields a usable stub.
    assert _snapshot_label("@@@") == "spec"
