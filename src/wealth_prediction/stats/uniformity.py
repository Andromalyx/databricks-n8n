"""Is every number 1..pool_size drawn equally often (marginal uniformity)?

Under H0 (fair draw), each number's marginal draw probability is
draw_size / pool_size regardless of the other numbers -- this holds exactly
even though draws are sampled without replacement, so a plain chi-square
goodness-of-fit test against a flat expected count is valid here (no need
for Monte Carlo, unlike the order-statistic tests in normality.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from wealth_prediction.config import GameConfig
from wealth_prediction.data.loader import to_long_format
from wealth_prediction.stats import TestResult


def number_frequencies(df: pd.DataFrame, game: GameConfig) -> pd.Series:
    """Observed draw count for each number 1..pool_size (zeros included)."""
    long_df = to_long_format(df, game)
    counts = long_df["number"].value_counts().reindex(range(1, game.pool_size + 1), fill_value=0)
    return counts.sort_index()


def chi_square_uniformity_test(df: pd.DataFrame, game: GameConfig, alpha: float = 0.05) -> TestResult:
    """Omnibus test: are all pool_size numbers equally likely overall?"""
    observed = number_frequencies(df, game).to_numpy()
    n_draws = len(df)
    expected_per_number = n_draws * game.draw_size / game.pool_size
    expected = np.full(game.pool_size, expected_per_number)

    if expected_per_number < 5:
        # Chi-square's asymptotic p-value is unreliable with low expected counts;
        # still compute it, but the caller should treat the p-value as indicative only.
        pass

    chi2, p = stats.chisquare(observed, expected)
    return TestResult(
        name="Chi-square goodness-of-fit (number frequency vs. uniform)",
        statistic=chi2,
        p_value=p,
        alpha=alpha,
        interpretation=(
            f"H0: all {game.pool_size} numbers equally likely across {n_draws} draws "
            f"(expected {expected_per_number:.1f} appearances each)."
        ),
    )


def per_number_significance(df: pd.DataFrame, game: GameConfig, alpha: float = 0.05) -> pd.DataFrame:
    """Per-number two-sided binomial test, Benjamini-Hochberg corrected across all `pool_size` tests.

    Flags individual "hot"/"cold" numbers, i.e. candidates for physical ball
    bias, while controlling the false-discovery rate that would otherwise
    inflate with pool_size=45 simultaneous tests.
    """
    n_draws = len(df)
    p_expected = game.draw_size / game.pool_size
    counts = number_frequencies(df, game)

    raw_pvalues = [
        stats.binomtest(int(c), n_draws, p_expected, alternative="two-sided").pvalue
        for c in counts
    ]
    _, p_adj, _, _ = multipletests(raw_pvalues, alpha=alpha, method="fdr_bh")

    result = pd.DataFrame(
        {
            "number": counts.index,
            "observed": counts.to_numpy(),
            "expected": n_draws * p_expected,
            "p_value": raw_pvalues,
            "p_adjusted": p_adj,
        }
    )
    result["flagged"] = result["p_adjusted"] < alpha
    return result.sort_values("p_adjusted").reset_index(drop=True)
