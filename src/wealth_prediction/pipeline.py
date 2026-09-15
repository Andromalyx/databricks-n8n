"""End-to-end CLI: load draws -> run stats suite -> evaluate model -> write report.

Usage:
    python -m wealth_prediction.pipeline --source synthetic --n-draws 500
    python -m wealth_prediction.pipeline --source csv --data data/raw/draws.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from wealth_prediction.config import load_config
from wealth_prediction.data import loader, synthetic
from wealth_prediction.modeling.predictor import evaluate_model_vs_baseline
from wealth_prediction.reporting.report import (
    generate_markdown_report,
    plot_hot_cold,
    plot_number_frequencies,
    plot_sum_distribution,
)
from wealth_prediction.stats import bias_detection, independence, normality, uniformity
from wealth_prediction.stats.normality import draw_statistics

logger = logging.getLogger(__name__)


def run_audit(df, config, out_dir: Path, run_model: bool = True) -> Path:
    game, analysis = config.game, config.analysis
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running uniformity tests...")
    chi2_uniform = uniformity.chi_square_uniformity_test(df, game, analysis.significance_level)
    per_number = uniformity.per_number_significance(df, game, analysis.significance_level)

    logger.info("Running normality suite (this simulates %d Monte Carlo draws)...", analysis.monte_carlo_sims)
    norm_results = normality.run_normality_suite(
        df, game, n_sims=analysis.monte_carlo_sims, seed=analysis.random_seed, alpha=analysis.significance_level
    )

    logger.info("Running independence tests...")
    stat_df = norm_results["draw_statistics"]
    runs = independence.runs_test((stat_df["sum"] % 2).to_numpy(), analysis.significance_level)
    ljung_box = independence.ljung_box_autocorrelation(stat_df["sum"].to_numpy(), alpha=analysis.significance_level)
    cooccurrence = independence.pairwise_cooccurrence_test(df, game, analysis.significance_level)
    gap_test = independence.gap_distribution_test(df, game, analysis.significance_level)

    logger.info("Running bias/drift detection...")
    hot_cold = bias_detection.hot_cold_numbers(df, game)
    drift = bias_detection.split_half_drift_test(df, game, analysis.significance_level) if analysis.split_half_drift_test else None

    all_results = [
        chi2_uniform,
        norm_results["shapiro"],
        norm_results["dagostino_k2"],
        norm_results["ks_gaussian"],
        norm_results["ks_fair_null"],
        runs,
        ljung_box,
        cooccurrence,
        gap_test,
    ]
    if drift:
        all_results.append(drift)
    scorecard = bias_detection.randomness_scorecard(all_results)

    model_summary = None
    if run_model:
        try:
            logger.info("Evaluating ML model vs. random baseline (walk-forward)...")
            eval_result = evaluate_model_vs_baseline(df, game, config.modeling, alpha=analysis.significance_level)
            model_summary = eval_result.summary()
        except ValueError as exc:
            logger.warning("Skipping model evaluation: %s", exc)

    freq_counts = uniformity.number_frequencies(df, game)
    null_sums = synthetic.simulate_fair_draws(
        analysis.monte_carlo_sims, game.pool_size, game.draw_size, seed=analysis.random_seed
    )
    null_sum_values = draw_statistics(null_sums, game)["sum"].to_numpy()

    figures = {
        "Number frequency": plot_number_frequencies(freq_counts, game, fig_dir / "number_frequency.png"),
        "Sum distribution vs. fair null": plot_sum_distribution(
            stat_df["sum"].to_numpy(), null_sum_values, game, fig_dir / "sum_distribution.png"
        ),
        "Hot & cold numbers": plot_hot_cold(hot_cold, fig_dir / "hot_cold.png"),
    }

    per_number.to_csv(out_dir / "per_number_significance.csv", index=False)
    hot_cold.to_csv(out_dir / "hot_cold_numbers.csv", index=False)

    report_path = generate_markdown_report(
        game=game,
        n_draws=len(df),
        all_results=all_results,
        scorecard=scorecard,
        model_summary=model_summary,
        figure_paths={k: v.relative_to(out_dir) for k, v in figures.items()},
        out_path=out_dir / "report.md",
    )
    return report_path


def _load_data(args, config):
    if args.source == "csv":
        data_path = Path(args.data) if args.data else config.data.raw_path
        return loader.load_csv(data_path, config.game)
    if args.source == "synthetic":
        return synthetic.simulate_fair_draws(
            args.n_draws, config.game.pool_size, config.game.draw_size, seed=config.analysis.random_seed
        )
    if args.source == "synthetic-biased":
        return synthetic.simulate_biased_draws(
            args.n_draws, pool_size=config.game.pool_size, draw_size=config.game.draw_size, seed=config.analysis.random_seed
        )
    raise ValueError(f"Unknown source: {args.source}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["csv", "synthetic", "synthetic-biased"], default="synthetic")
    parser.add_argument("--data", default=None, help="CSV path when --source csv")
    parser.add_argument("--n-draws", type=int, default=500, help="Draw count for synthetic sources")
    parser.add_argument("--out-dir", default="reports", help="Output directory for report + figures")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--no-model", action="store_true", help="Skip the ML-vs-baseline evaluation")
    args = parser.parse_args()

    config = load_config(args.config)
    df = _load_data(args, config)
    report_path = run_audit(df, config, Path(args.out_dir), run_model=not args.no_model)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
