"""Are draws independent of each other (no serial structure over time)?

A fair mechanical/RNG draw has no memory: this draw's sum, parity, or
co-occurring numbers should carry zero information about the next one.
These tests probe that from four angles: runs (streakiness), autocorrelation
(linear serial dependence), pairwise co-occurrence (do certain number pairs
show up together more than chance), and inter-appearance gaps (does each
number's "time between hits" match the geometric distribution implied by
independent Bernoulli trials).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.proportion import proportions_ztest

from wealth_prediction.config import GameConfig
from wealth_prediction.data.loader import numbers_matrix, to_long_format
from wealth_prediction.stats import TestResult


def runs_test(binary_sequence: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Wald-Wolfowitz runs test on a 0/1 sequence (e.g. odd/even sum per draw).

    Too few runs => streakiness (positive serial correlation); too many =>
    unnatural alternation. Implemented manually since scipy/statsmodels don't
    ship a maintained public runs test.
    """
    x = np.asarray(binary_sequence).astype(int)
    n1, n0 = int(x.sum()), int(len(x) - x.sum())
    n = n1 + n0
    n_runs = 1 + int(np.sum(x[1:] != x[:-1]))

    mean_runs = 2 * n1 * n0 / n + 1
    var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1))
    z = (n_runs - mean_runs) / np.sqrt(var_runs) if var_runs > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return TestResult(
        "Wald-Wolfowitz runs test (parity sequence)", z, p, alpha,
        f"H0: {n_runs} observed runs is consistent with a random 0/1 sequence "
        f"(expected {mean_runs:.1f}).",
    )


def ljung_box_autocorrelation(values: np.ndarray, lags: int = 10, alpha: float = 0.05) -> TestResult:
    """Combined Ljung-Box test for autocorrelation across the first `lags` lags of a time series."""
    result = acorr_ljungbox(values, lags=[lags], return_df=True)
    stat = float(result["lb_stat"].iloc[0])
    p = float(result["lb_pvalue"].iloc[0])
    return TestResult(
        f"Ljung-Box (autocorrelation, lags=1..{lags})", stat, p, alpha,
        "H0: no autocorrelation in the series up to the given lag.",
    )


def pairwise_cooccurrence_test(df: pd.DataFrame, game: GameConfig, alpha: float = 0.05) -> TestResult:
    """Chi-square goodness-of-fit over all C(pool_size, 2) number pairs.

    By symmetry of sampling without replacement, P(both i and j drawn in the
    same draw) = draw_size*(draw_size-1) / (pool_size*(pool_size-1)) for every
    pair (i, j) under H0 -- no combinatorial approximation or simulation needed.
    """
    n_draws = len(df)
    p_pair = game.draw_size * (game.draw_size - 1) / (game.pool_size * (game.pool_size - 1))
    expected = n_draws * p_pair

    long_df = to_long_format(df, game)
    draw_to_numbers = long_df.groupby("draw_id")["number"].apply(set)

    counts = np.zeros((game.pool_size + 1, game.pool_size + 1), dtype=int)
    for numbers in draw_to_numbers:
        for i, j in itertools.combinations(sorted(numbers), 2):
            counts[i, j] += 1

    pairs = list(itertools.combinations(range(1, game.pool_size + 1), 2))
    observed = np.array([counts[i, j] for i, j in pairs])
    chi2, p = stats.chisquare(observed, np.full(len(pairs), expected))

    return TestResult(
        "Chi-square goodness-of-fit (pairwise co-occurrence)", chi2, p, alpha,
        f"H0: every one of {len(pairs)} number pairs co-occurs at rate "
        f"{p_pair:.5f} (expected {expected:.2f} times each over {n_draws} draws).",
    )


def repeat_rate_test(df: pd.DataFrame, game: GameConfig, alpha: float = 0.05) -> TestResult:
    """Does a number that appeared in draw t-1 repeat in draw t more or less often than one that didn't?

    Directly tests the "hot/cold is tied to overlap with the immediately
    preceding draw" idea in its simplest form: under independence,
    P(appear in draw t | was in draw t-1) should equal
    P(appear in draw t | wasn't in draw t-1), and both should equal
    draw_size/pool_size. This is a plainer, single-number version of what
    `modeling.features`'s markov_score/cooccurrence_score already encode as
    model inputs -- useful as a standalone sanity check independent of any
    trained model.
    """
    matrix = numbers_matrix(df, game)
    n_draws, pool_size = len(matrix), game.pool_size
    appeared = np.zeros((n_draws, pool_size), dtype=bool)
    for t in range(n_draws):
        appeared[t, matrix[t] - 1] = True

    was_in_previous = appeared[:-1].ravel()
    current = appeared[1:].ravel()

    n1, x1 = int(was_in_previous.sum()), int(current[was_in_previous].sum())
    n2, x2 = int((~was_in_previous).sum()), int(current[~was_in_previous].sum())

    stat, p_value = proportions_ztest([x1, x2], [n1, n2])

    return TestResult(
        "Two-proportion z-test (repeat rate vs. previous draw)", stat, p_value, alpha,
        f"H0: P(appear | was in previous draw)={x1 / n1:.4f} equals "
        f"P(appear | wasn't in previous draw)={x2 / n2:.4f}.",
    )


def gap_distribution_test(df: pd.DataFrame, game: GameConfig, alpha: float = 0.05, n_bins: int = 10) -> TestResult:
    """Chi-square goodness-of-fit of inter-appearance gaps against Geometric(p).

    Under independence, the number of draws between consecutive appearances
    of a given number follows Geometric(p) with p = draw_size / pool_size.
    Gaps are pooled across all numbers to get enough samples for the test.
    """
    p = game.draw_size / game.pool_size
    long_df = to_long_format(df, game)

    gaps: list[int] = []
    for _, group in long_df.groupby("number"):
        draw_ids = np.sort(group["draw_id"].to_numpy())
        gaps.extend(np.diff(draw_ids) - 1)  # number of draws skipped between hits
    gaps = np.array(gaps)

    max_gap = int(np.quantile(gaps, 0.99)) or 1
    edges = np.unique(np.linspace(0, max_gap, n_bins, dtype=int))
    boundaries = np.concatenate([edges, [np.inf]])  # len(edges)+1 boundaries -> len(edges) bins
    observed, _ = np.histogram(gaps, bins=boundaries)

    # gap follows a 0-indexed geometric distribution (gap=0 means "hit again next
    # draw"): P(gap < b) = P(number of trials-to-hit <= b) = geom.cdf(b, p).
    cdf_at_boundaries = np.where(np.isinf(boundaries), 1.0, stats.geom.cdf(boundaries, p))
    bin_probs = np.diff(cdf_at_boundaries)
    expected = bin_probs * len(gaps)

    # Merge tiny-expected-count bins into neighbors so chi-square stays valid.
    observed, expected = _merge_sparse_bins(observed, expected)
    chi2, pval = stats.chisquare(observed, expected)

    return TestResult(
        "Chi-square goodness-of-fit (inter-appearance gaps vs. Geometric)", chi2, pval, alpha,
        f"H0: gaps between repeat appearances of a number follow Geometric(p={p:.4f}).",
    )


def _merge_sparse_bins(observed: np.ndarray, expected: np.ndarray, min_expected: float = 5.0):
    obs, exp = list(observed), list(expected)
    i = 0
    while i < len(exp) - 1:
        if exp[i] < min_expected:
            exp[i + 1] += exp[i]
            obs[i + 1] += obs[i]
            del exp[i], obs[i]
        else:
            i += 1
    return np.array(obs), np.array(exp)
