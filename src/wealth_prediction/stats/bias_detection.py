"""Detect drift, hot/cold numbers, and (optionally) revenue-linked bias.

This module is the closest thing to testing the user's "does the operator
avoid the winner to optimize revenue" hypothesis. Read the caveat in
README.md before over-interpreting results: a licensed, audited lottery is
*expected* to pass every test here, and a rejection should be treated as
"investigate further" (more data, check for a data-collection artifact),
not as proof of fraud.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from wealth_prediction.config import GameConfig
from wealth_prediction.stats import TestResult
from wealth_prediction.stats.uniformity import number_frequencies


def hot_cold_numbers(df: pd.DataFrame, game: GameConfig, ci: float = 0.95) -> pd.DataFrame:
    """Observed appearance rate per number with a Wilson confidence interval.

    Ranking by point estimate alone is misleading with a few hundred draws;
    the CI shows whether "hot" numbers are actually distinguishable from the
    baseline rate draw_size/pool_size, or just noise.
    """
    n_draws = len(df)
    counts = number_frequencies(df, game)
    baseline = game.draw_size / game.pool_size

    low, high = proportion_confint(counts, n_draws, alpha=1 - ci, method="wilson")
    out = pd.DataFrame(
        {
            "number": counts.index,
            "observed_rate": counts.to_numpy() / n_draws,
            "baseline_rate": baseline,
            "ci_low": low,
            "ci_high": high,
        }
    )
    out["distinguishable_from_baseline"] = (out["ci_low"] > baseline) | (out["ci_high"] < baseline)
    return out.sort_values("observed_rate", ascending=False).reset_index(drop=True)


def split_half_drift_test(df: pd.DataFrame, game: GameConfig, alpha: float = 0.05) -> TestResult:
    """Chi-square test of homogeneity: is the per-number frequency profile
    stable between the first and second half of the recorded history?

    A significant result suggests something changed mid-history -- e.g. a
    ball set replaced, RNG reseeded/reconfigured, or a data-source splice --
    and is the standard first check real lottery-integrity audits run.
    """
    mid = len(df) // 2
    first, second = df.iloc[:mid], df.iloc[mid:]

    freq_first = number_frequencies(first, game)
    freq_second = number_frequencies(second, game)
    table = np.vstack([freq_first.to_numpy(), freq_second.to_numpy()])

    chi2, p, dof, _ = stats.chi2_contingency(table)
    return TestResult(
        "Chi-square homogeneity (first half vs. second half of history)", chi2, p, alpha,
        f"H0: number-frequency profile is stable across draws 1..{mid} vs. {mid + 1}..{len(df)}.",
    )


@dataclass
class RevenueCorrelationResult:
    test: TestResult
    n_matched: int


def revenue_correlation_test(
    draw_stats: pd.DataFrame, revenue: pd.DataFrame, value_col: str, alpha: float = 0.05
) -> RevenueCorrelationResult:
    """Spearman correlation between a per-draw statistic and an external
    revenue/jackpot/sales series, joined on draw_id.

    Optional: only meaningful if you have real sales or jackpot-rollover data
    (Vietlott doesn't publish this alongside draw results). A significant
    correlation between, say, jackpot size and draw sum/spread would be the
    kind of pattern a "revenue-optimizing" draw process would leave behind;
    a null result here just means this particular linkage wasn't detected,
    not that no bias exists.
    """
    merged = draw_stats.merge(revenue[["draw_id", value_col]], on="draw_id", how="inner")
    if len(merged) < 10:
        raise ValueError(f"Only {len(merged)} matching draw_ids between draw_stats and revenue data; need >= 10.")

    stat, p = stats.spearmanr(merged["sum"], merged[value_col])
    result = TestResult(
        f"Spearman correlation (draw sum vs. {value_col})", stat, p, alpha,
        f"H0: no monotonic association between per-draw sum and {value_col} (n={len(merged)}).",
    )
    return RevenueCorrelationResult(test=result, n_matched=len(merged))


def randomness_scorecard(results: list[TestResult]) -> dict:
    """Aggregate a batch of independent-ish TestResults into one omnibus verdict.

    Uses Fisher's method to combine p-values into a single omnibus statistic,
    and separately reports the naive "how many rejected at alpha" tally so a
    reader can see both the combined signal and how it's distributed --
    with ~20 tests at alpha=0.05 you expect about one spurious rejection by
    chance alone, so a single flagged test out of many is not, by itself, evidence of bias.
    """
    pvalues = np.array([r.p_value for r in results])
    alpha = results[0].alpha if results else 0.05
    combined_stat, combined_p = stats.combine_pvalues(pvalues, method="fisher")

    n_significant = int(sum(r.significant for r in results))
    return {
        "n_tests": len(results),
        "n_significant": n_significant,
        "expected_false_positives_at_alpha": round(len(results) * alpha, 2),
        "fisher_combined_statistic": combined_stat,
        "fisher_combined_p_value": combined_p,
        "omnibus_reject_h0_fair_and_independent": combined_p < alpha,
        "results": results,
    }
