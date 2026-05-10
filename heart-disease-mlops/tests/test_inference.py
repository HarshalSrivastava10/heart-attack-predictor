from __future__ import annotations

import pytest

from heart_mlops.inference import risk_level_from_probability


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.0, "no risk"),
        (0.24, "no risk"),
        (0.25, "low risk"),
        (0.49, "low risk"),
        (0.5, "medium risk"),
        (0.74, "medium risk"),
        (0.75, "high risk"),
        (1.0, "high risk"),
    ],
)
def test_risk_level_from_probability(p: float, expected: str) -> None:
    assert risk_level_from_probability(p) == expected
