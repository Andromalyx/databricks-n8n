"""Statistical audit suite: uniformity, normality, independence, bias/drift.

Every test returns a `TestResult` so the reporting layer can render a
consistent table instead of parsing bespoke return shapes per module.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    alpha: float
    interpretation: str

    @property
    def significant(self) -> bool:
        """True => reject H0 (fair/independent/normal) at `alpha`."""
        return self.p_value < self.alpha

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        flag = "REJECT H0" if self.significant else "fail to reject H0"
        return f"{self.name}: stat={self.statistic:.4f}, p={self.p_value:.4g} -> {flag}"
