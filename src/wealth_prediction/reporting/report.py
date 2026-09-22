"""Render the statistical audit into figures + a markdown summary report."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs in CLI/CI contexts with no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from wealth_prediction.config import GameConfig
from wealth_prediction.stats import TestResult

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
PALETTE = {"bar": "#4C72B0", "highlight": "#C44E52", "baseline": "#55A868"}


def plot_number_frequencies(freq_counts: pd.Series, game: GameConfig, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(freq_counts.index, freq_counts.to_numpy(), color=PALETTE["bar"])
    expected = freq_counts.sum() / game.pool_size
    ax.axhline(expected, color=PALETTE["highlight"], linestyle="--", label=f"expected ({expected:.1f})")
    ax.set_xlabel("Number")
    ax.set_ylabel("Times drawn")
    ax.set_title(f"{game.name}: draw frequency per number")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sum_distribution(observed_sums: np.ndarray, null_sums: np.ndarray, game: GameConfig, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(null_sums, stat="density", color=PALETTE["baseline"], label="Monte Carlo fair-draw null", ax=ax, alpha=0.5, bins=40)
    sns.histplot(observed_sums, stat="density", color=PALETTE["highlight"], label="Observed draws", ax=ax, alpha=0.5, bins=40)
    ax.set_xlabel("Sum of 6 drawn numbers")
    ax.set_title(f"{game.name}: observed vs. simulated-fair draw sums")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_hot_cold(hot_cold_df: pd.DataFrame, out_path: Path, top_n: int = 10) -> Path:
    ranked = pd.concat([hot_cold_df.head(top_n), hot_cold_df.tail(top_n)])
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE["highlight"] if v else PALETTE["bar"] for v in ranked["distinguishable_from_baseline"]]
    ax.barh(ranked["number"].astype(str), ranked["observed_rate"], color=colors)
    ax.axvline(hot_cold_df["baseline_rate"].iloc[0], color="black", linestyle="--", label="theoretical baseline")
    ax.set_xlabel("Observed appearance rate")
    ax.set_title("Hottest & coldest numbers (red = outside baseline Wilson CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _results_table(results: list[TestResult]) -> str:
    lines = ["| Test | Statistic | p-value | Significant @ alpha |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.name} | {r.statistic:.4f} | {r.p_value:.4g} | {'YES' if r.significant else 'no'} |")
    return "\n".join(lines)


def _bundle_table(bundle: pd.DataFrame) -> str:
    lines = ["| Number | Predicted probability | Baseline | Lift |", "|---|---|---|---|"]
    for _, row in bundle.iterrows():
        lines.append(
            f"| {int(row['number'])} | {row['predicted_probability']:.4f} | "
            f"{row['baseline_probability']:.4f} | {row['lift_over_baseline']:.2f}x |"
        )
    return "\n".join(lines)


def generate_markdown_report(
    game: GameConfig,
    n_draws: int,
    all_results: list[TestResult],
    scorecard: dict,
    model_summary: str | None,
    bundle: pd.DataFrame | None,
    model_beats_baseline: bool,
    figure_paths: dict[str, Path],
    out_path: Path,
    missing_draw_id_runs: list[tuple[int, int]] | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {game.name}: randomness & bias audit",
        "",
        f"Draws analyzed: **{n_draws}**",
        "",
    ]

    if missing_draw_id_runs:
        total_missing = sum(end - start + 1 for start, end in missing_draw_id_runs)
        largest = max(end - start + 1 for start, end in missing_draw_id_runs)
        lines += [
            "## ⚠ Data quality warning",
            "",
            f"**{total_missing} draw_id(s) missing** from the source history, in "
            f"{len(missing_draw_id_runs)} run(s) (largest: {largest} consecutive draws): "
            f"{missing_draw_id_runs}.",
            "",
            "Gap-based tests below (inter-appearance gaps, gap-hazard trend) compute "
            "\"draws since last appearance\" from draw_id arithmetic, so any inter-"
            "appearance interval that spans one of these holes gets an artificially "
            "inflated gap. A significant gap-distribution result alongside otherwise-"
            "clean tests is a strong hint to check here before concluding the game is "
            "biased -- re-run on a gap-free sub-range to confirm before trusting it.",
            "",
        ]

    lines += [
        "## Statistical test results",
        "",
        _results_table(all_results),
        "",
        "## Omnibus verdict",
        "",
        f"- Tests run: {scorecard['n_tests']}",
        f"- Significant at alpha (uncorrected): {scorecard['n_significant']} "
        f"(expected ~{scorecard['expected_false_positives_at_alpha']} by chance alone)",
        f"- Fisher combined p-value: {scorecard['fisher_combined_p_value']:.4g}",
        f"- **Reject fair-and-independent null overall: "
        f"{'YES' if scorecard['omnibus_reject_h0_fair_and_independent'] else 'NO'}**",
        "",
    ]
    if model_summary:
        lines += ["## ML model vs. random baseline", "", model_summary, ""]

    if bundle is not None and not bundle.empty:
        lines += [
            "## Suggested number bundle (next draw)",
            "",
            "**Not a prediction.** These are the numbers the model currently ranks highest, "
            f"including Markov-transition and pairwise co-occurrence features. "
            f"Model beats random baseline (see above): "
            f"**{'YES' if model_beats_baseline else 'NO'}**"
            + ("" if model_beats_baseline else " -- treat this ranking as noise, not a signal."),
            "",
            _bundle_table(bundle),
            "",
        ]

    if figure_paths:
        lines += ["## Figures", ""]
        for name, path in figure_paths.items():
            lines.append(f"![{name}]({path.name})")
        lines.append("")

    lines += [
        "## How to read this",
        "",
        "- A single rejected test out of many is not evidence of bias -- see the false-positive baseline above.",
        "- The Fisher combined p-value is the primary omnibus signal; individual tests are diagnostic detail.",
        "- If a fair lottery, no model can beat the constant baseline out-of-sample by construction; "
        "a confirmed 'no edge' result is the expected, reassuring outcome, not a failure of the analysis.",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report to %s", out_path)
    return out_path
