"""Synthetic Vietlott 6/45 draw generators.

Used two ways in this project:
1. As a stand-in dataset so the full pipeline runs end-to-end before real
   history is available.
2. As the null-distribution engine for the stats/ modules: instead of
   deriving exact formulas for order statistics of a without-replacement
   draw (messy combinatorics), we simulate many fair draws and compare the
   real data's statistics to that empirical distribution.

`simulate_biased_draws` / `simulate_drifting_draws` exist purely to sanity-check
that our detectors (stats/*.py) actually have power to catch a rigged process
-- if they can't flag an obviously biased synthetic series, they're useless
on real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_POOL_SIZE = 45
DEFAULT_DRAW_SIZE = 6


def _draw_dates(n_draws: int, start: str = "2020-01-01", freq: str = "3D") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n_draws, freq=freq)


def _to_frame(draws: np.ndarray, start: str = "2020-01-01", freq: str = "3D") -> pd.DataFrame:
    draws = np.sort(draws, axis=1)  # Vietlott publishes draws sorted ascending
    n_draws, draw_size = draws.shape
    cols = {f"n{i + 1}": draws[:, i] for i in range(draw_size)}
    df = pd.DataFrame(cols)
    df.insert(0, "draw_id", np.arange(1, n_draws + 1))
    df.insert(1, "draw_date", _draw_dates(n_draws, start=start, freq=freq))
    return df


def simulate_fair_draws(
    n_draws: int,
    pool_size: int = DEFAULT_POOL_SIZE,
    draw_size: int = DEFAULT_DRAW_SIZE,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate `n_draws` independent, uniform, without-replacement draws.

    This is the null model H0 that every test in stats/ checks real data
    against: each of the C(pool_size, draw_size) combinations equally likely,
    draws independent of each other.
    """
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(pool_size, size=draw_size, replace=False) + 1 for _ in range(n_draws)])
    return _to_frame(draws)


def simulate_biased_draws(
    n_draws: int,
    weights: np.ndarray | None = None,
    pool_size: int = DEFAULT_POOL_SIZE,
    draw_size: int = DEFAULT_DRAW_SIZE,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate draws where numbers are sampled with unequal probability.

    `weights` defaults to a mild bias favoring low numbers (a common
    real-world failure mode: worn/lighter balls, or a manual "quick pick"
    habit bleeding into a supposedly mechanical draw). Pass your own
    `weights` (length `pool_size`, need not be normalized) to test other
    bias shapes.
    """
    rng = np.random.default_rng(seed)
    if weights is None:
        weights = np.linspace(1.3, 0.7, pool_size)  # low numbers ~2x more likely than high
    weights = np.asarray(weights, dtype=float)
    probs = weights / weights.sum()

    draws = np.empty((n_draws, draw_size), dtype=int)
    for i in range(n_draws):
        draws[i] = rng.choice(pool_size, size=draw_size, replace=False, p=probs) + 1
    return _to_frame(draws)


def simulate_drifting_draws(
    n_draws: int,
    pool_size: int = DEFAULT_POOL_SIZE,
    draw_size: int = DEFAULT_DRAW_SIZE,
    drift_start_fraction: float = 0.5,
    seed: int | None = None,
) -> pd.DataFrame:
    """Fair draws for the first half of history, biased for the second half.

    Exercises the split-half drift test in stats/bias_detection.py, which is
    designed to catch exactly this signature (e.g. equipment swapped or
    tampered with partway through a lottery's operating history).
    """
    rng = np.random.default_rng(seed)
    split = int(n_draws * drift_start_fraction)
    fair = simulate_fair_draws(split, pool_size, draw_size, seed=rng.integers(1 << 31))
    biased = simulate_biased_draws(n_draws - split, pool_size=pool_size, draw_size=draw_size, seed=rng.integers(1 << 31))
    combined = pd.concat([fair, biased], ignore_index=True)
    combined["draw_id"] = np.arange(1, len(combined) + 1)
    combined["draw_date"] = _draw_dates(len(combined))
    return combined
