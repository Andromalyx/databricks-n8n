"""Is the distribution of per-draw statistics (sum, mean, spread) "normal"?

Important caveat baked into this module: the sum of 6 distinct values drawn
without replacement from {1..45} is NOT exactly Gaussian -- it has a known,
slightly non-normal, bounded, symmetric distribution fixed entirely by
combinatorics. Standard normality tests (Shapiro-Wilk, D'Agostino, Anderson-
Darling) answer "does this look Gaussian", which is a reasonable sanity
check but the wrong ground truth. The rigorous question -- "does this match
what a FAIR draw actually produces" -- is answered by `ks_test_vs_fair_null`,
which compares observed statistics against a Monte Carlo simulated null
built from `wealth_prediction.data.synthetic.simulate_fair_draws`. Treat the
classic-normality-test results as descriptive context, and the Monte Carlo
KS test as the actual hypothesis test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from wealth_prediction.config import GameConfig
from wealth_prediction.data.loader import numbers_matrix
from wealth_prediction.data.synthetic import simulate_fair_draws
from wealth_prediction.stats import TestResult


def draw_statistics(df: pd.DataFrame, game: GameConfig) -> pd.DataFrame:
    """Per-draw summary stats commonly checked by lottery "system" players."""
    m = numbers_matrix(df, game)
    return pd.DataFrame(
        {
            "draw_id": df["draw_id"].to_numpy(),
            "sum": m.sum(axis=1),
            "mean": m.mean(axis=1),
            "std": m.std(axis=1),
            "range": m.max(axis=1) - m.min(axis=1),
            "odd_count": (m % 2 == 1).sum(axis=1),
        }
    )


def shapiro_wilk_test(values: np.ndarray, alpha: float = 0.05) -> TestResult:
    if len(values) > 5000:
        values = np.random.default_rng(0).choice(values, size=5000, replace=False)  # Shapiro is unreliable/slow above n=5000
    stat, p = stats.shapiro(values)
    return TestResult("Shapiro-Wilk (vs. Gaussian)", stat, p, alpha, "H0: values are drawn from a normal distribution.")


def dagostino_k2_test(values: np.ndarray, alpha: float = 0.05) -> TestResult:
    stat, p = stats.normaltest(values)
    return TestResult(
        "D'Agostino-Pearson K^2 (vs. Gaussian)", stat, p, alpha,
        "H0: skewness and kurtosis match a normal distribution.",
    )


def ks_test_vs_gaussian(values: np.ndarray, alpha: float = 0.05) -> TestResult:
    mu, sigma = np.mean(values), np.std(values, ddof=1)
    stat, p = stats.kstest(values, "norm", args=(mu, sigma))
    return TestResult(
        "Kolmogorov-Smirnov (vs. fitted Gaussian)", stat, p, alpha,
        f"H0: values ~ N({mu:.2f}, {sigma:.2f}^2).",
    )


def ks_test_vs_fair_null(
    observed: np.ndarray, game: GameConfig, n_sims: int = 20000, seed: int | None = None, alpha: float = 0.05
) -> TestResult:
    """The test that actually matters: two-sample KS between the observed
    per-draw sums and `n_sims` independent simulated fair draws.
    """
    rng = np.random.default_rng(seed)
    sim = simulate_fair_draws(n_sims, game.pool_size, game.draw_size, seed=int(rng.integers(1 << 31)))
    null_values = draw_statistics(sim, game)["sum"].to_numpy()
    stat, p = stats.ks_2samp(observed, null_values)
    return TestResult(
        "KS 2-sample (observed sums vs. Monte Carlo fair-draw sums)", stat, p, alpha,
        f"H0: observed draw sums are drawn from the same distribution as {n_sims} simulated fair draws.",
    )


def run_normality_suite(df: pd.DataFrame, game: GameConfig, n_sims: int = 20000, seed: int | None = None, alpha: float = 0.05) -> dict:
    stat_df = draw_statistics(df, game)
    sums = stat_df["sum"].to_numpy()
    return {
        "draw_statistics": stat_df,
        "shapiro": shapiro_wilk_test(sums, alpha),
        "dagostino_k2": dagostino_k2_test(sums, alpha),
        "ks_gaussian": ks_test_vs_gaussian(sums, alpha),
        "ks_fair_null": ks_test_vs_fair_null(sums, game, n_sims=n_sims, seed=seed, alpha=alpha),
    }
