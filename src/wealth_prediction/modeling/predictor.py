"""Honest test of "is there an exploitable pattern": model vs. random baseline.

If Vietlott 6/45 is fair, NO feature set can out-predict the constant
baseline P(number appears) = draw_size/pool_size out-of-sample, because that
constant already is the true generating probability. So this module isn't
trying to build a numbers-picking system -- it's a statistical instrument:
train a real classifier on real features, evaluate it walk-forward
(chronological split, never shuffled), and formally test whether its
predictions beat the trivial baseline by more than sampling noise would
produce. A confirmed "no" is itself the useful, expected result for a fair
lottery; a reproducible, significant "yes" would be the actual finding worth
investigating further.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from wealth_prediction.config import GameConfig, ModelingConfig
from wealth_prediction.modeling.features import FEATURE_COLUMNS, build_panel, next_draw_features

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    model_log_loss: float
    baseline_log_loss: float
    model_brier: float
    baseline_brier: float
    n_test_samples: int
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    alpha: float

    @property
    def model_beats_baseline(self) -> bool:
        """True only if the model is BOTH numerically better AND the
        improvement is statistically significant -- guards against reporting
        noise as an "edge".
        """
        return self.model_log_loss < self.baseline_log_loss and self.wilcoxon_p_value < self.alpha

    def summary(self) -> str:
        verdict = "MODEL BEATS BASELINE (investigate further)" if self.model_beats_baseline else "no exploitable edge detected"
        return (
            f"log-loss: model={self.model_log_loss:.4f} vs baseline={self.baseline_log_loss:.4f}; "
            f"Brier: model={self.model_brier:.4f} vs baseline={self.baseline_brier:.4f}; "
            f"Wilcoxon p={self.wilcoxon_p_value:.4g} (n={self.n_test_samples}) -> {verdict}"
        )


def _chronological_split(panel: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    draw_ids = np.sort(panel["draw_id"].unique())
    split_idx = int(len(draw_ids) * (1 - test_fraction))
    if split_idx < 1 or split_idx >= len(draw_ids):
        raise ValueError(f"test_fraction={test_fraction} leaves no data on one side of {len(draw_ids)} draws.")
    cutoff = draw_ids[split_idx]
    return panel[panel["draw_id"] < cutoff], panel[panel["draw_id"] >= cutoff]


def evaluate_model_vs_baseline(
    df: pd.DataFrame,
    game: GameConfig,
    modeling: ModelingConfig,
    model=None,
    alpha: float = 0.05,
) -> EvaluationResult:
    """Train `model` (default: GradientBoostingClassifier) on a chronological
    train split, evaluate log-loss/Brier on the held-out future draws against
    the constant baseline, and Wilcoxon-test whether the per-sample log-loss
    improvement is significant.
    """
    if len(df) < 30:
        raise ValueError(f"Need at least 30 draws for a meaningful walk-forward split, got {len(df)}.")

    panel = build_panel(df, game, history_window=modeling.history_window)
    train, test = _chronological_split(panel, modeling.test_fraction)

    model = model or GradientBoostingClassifier(random_state=0)
    model.fit(train[FEATURE_COLUMNS], train["appeared"])

    y_test = test["appeared"].to_numpy()
    p_model = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    p_model = np.clip(p_model, 1e-6, 1 - 1e-6)  # avoid -inf log-loss on a perfectly confident wrong call

    baseline_rate = game.draw_size / game.pool_size
    p_baseline = np.full_like(p_model, baseline_rate)

    model_ll = log_loss(y_test, p_model, labels=[0, 1])
    baseline_ll = log_loss(y_test, p_baseline, labels=[0, 1])
    model_brier = brier_score_loss(y_test, p_model)
    baseline_brier = brier_score_loss(y_test, p_baseline)

    per_sample_model_ll = -(y_test * np.log(p_model) + (1 - y_test) * np.log(1 - p_model))
    per_sample_baseline_ll = -(y_test * np.log(p_baseline) + (1 - y_test) * np.log(1 - p_baseline))
    diff = per_sample_baseline_ll - per_sample_model_ll  # positive => model better on that sample
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diff)
    except ValueError:
        # All-zero differences (identical predictions): no evidence of any edge.
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0

    result = EvaluationResult(
        model_log_loss=model_ll,
        baseline_log_loss=baseline_ll,
        model_brier=model_brier,
        baseline_brier=baseline_brier,
        n_test_samples=len(y_test),
        wilcoxon_statistic=wilcoxon_stat,
        wilcoxon_p_value=wilcoxon_p,
        alpha=alpha,
    )
    logger.info(result.summary())
    return result


def backtest_bundle(
    df: pd.DataFrame,
    game: GameConfig,
    modeling: ModelingConfig,
    bundle_size: int | None = None,
    model=None,
) -> pd.DataFrame:
    """Walk-forward backtest: for each held-out test draw, take the model's
    top `bundle_size` numbers -- trained only on data strictly before the
    test period, same split as `evaluate_model_vs_baseline` -- and check how
    many were actually drawn.

    One row per test draw: predicted numbers with their model probabilities,
    the actual draw, which predicted numbers hit, and the count. Compare the
    result with `summarize_bundle_backtest`, which checks it against the
    exact chance-level distribution rather than eyeballing the numbers.
    """
    bundle_size = bundle_size or game.draw_size
    if len(df) < 30:
        raise ValueError(f"Need at least 30 draws for a meaningful walk-forward split, got {len(df)}.")

    panel = build_panel(df, game, history_window=modeling.history_window)
    train, test = _chronological_split(panel, modeling.test_fraction)

    model = model or GradientBoostingClassifier(random_state=0)
    model.fit(train[FEATURE_COLUMNS], train["appeared"])

    test = test.copy()
    test["predicted_probability"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    draw_lookup = df.set_index("draw_id")
    number_cols = [f"n{i + 1}" for i in range(game.draw_size)]

    rows = []
    for draw_id, group in test.groupby("draw_id"):
        top = group.nlargest(bundle_size, "predicted_probability")
        actual_numbers = sorted(int(n) for n in draw_lookup.loc[draw_id, number_cols])
        actual_set = set(actual_numbers)
        predicted_numbers = top["number"].tolist()
        hits = [n for n in predicted_numbers if n in actual_set]

        rows.append(
            {
                "draw_id": draw_id,
                "draw_date": draw_lookup.loc[draw_id, "draw_date"],
                "predicted_numbers": predicted_numbers,
                "predicted_probabilities": [round(p, 4) for p in top["predicted_probability"]],
                "actual_numbers": actual_numbers,
                "hits": hits,
                "hit_probabilities": [
                    round(p, 4)
                    for n, p in zip(predicted_numbers, top["predicted_probability"])
                    if n in actual_set
                ],
                "n_correct": len(hits),
            }
        )

    return pd.DataFrame(rows)


def summarize_bundle_backtest(results: pd.DataFrame, game: GameConfig, bundle_size: int, alpha: float = 0.05) -> dict:
    """Aggregate a `backtest_bundle` run against the exact chance-level null.

    Picking `bundle_size` numbers uniformly at random (no model at all) and
    checking overlap with `draw_size` actual winners out of `pool_size` gives
    Hypergeom(pool_size, draw_size, bundle_size) hits per draw -- that's the
    correct baseline to compare against, not "0 out of 6" intuition. The
    per-draw counts are summed and compared to their combined mean/variance
    under that null via a normal-approximation z-test (accurate once there
    are a few dozen test draws).
    """
    n_draws = len(results)
    expected_mean_per_draw = bundle_size * game.draw_size / game.pool_size
    expected_var_per_draw = stats.hypergeom.var(game.pool_size, game.draw_size, bundle_size)

    total_observed = int(results["n_correct"].sum())
    total_expected = n_draws * expected_mean_per_draw
    total_var = n_draws * expected_var_per_draw
    z = (total_observed - total_expected) / np.sqrt(total_var) if total_var > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "n_test_draws": n_draws,
        "bundle_size": bundle_size,
        "total_correct": total_observed,
        "total_possible": n_draws * bundle_size,
        "observed_avg_correct_per_draw": results["n_correct"].mean(),
        "chance_avg_correct_per_draw": expected_mean_per_draw,
        "z_score": z,
        "p_value": p_value,
        "beats_chance": bool(results["n_correct"].mean() > expected_mean_per_draw and p_value < alpha),
        "match_distribution": results["n_correct"].value_counts().sort_index().to_dict(),
    }


def predict_next_draw_scores(
    df: pd.DataFrame, game: GameConfig, modeling: ModelingConfig, model=None
) -> pd.DataFrame:
    """Every number's estimated appearance probability for the next, not-yet-drawn draw.

    Trains on the *complete* history (no held-out split -- there's nothing to
    hold out against, since this draw hasn't happened) and scores
    `next_draw_features`. This is a shortlist tool, not a prediction: read
    `evaluate_model_vs_baseline`'s result first. If that came back
    `model_beats_baseline=False` (the expected outcome for a fair lottery),
    the ranking below is statistically indistinguishable from ranking 45
    numbers by coin flips, dressed up in real feature names.
    """
    panel = build_panel(df, game, history_window=modeling.history_window)
    model = model or GradientBoostingClassifier(random_state=0)
    model.fit(panel[FEATURE_COLUMNS], panel["appeared"])

    next_features = next_draw_features(df, game, history_window=modeling.history_window)
    proba = model.predict_proba(next_features[FEATURE_COLUMNS])[:, 1]

    baseline_rate = game.draw_size / game.pool_size
    out = next_features[["number"]].copy()
    out["predicted_probability"] = proba
    out["baseline_probability"] = baseline_rate
    out["lift_over_baseline"] = out["predicted_probability"] / baseline_rate
    return out.sort_values("predicted_probability", ascending=False).reset_index(drop=True)


def recommend_number_bundle(
    df: pd.DataFrame, game: GameConfig, modeling: ModelingConfig, bundle_size: int = 10, model=None
) -> pd.DataFrame:
    """Top `bundle_size` numbers by predicted next-draw probability -- a shortlist, not a ticket.

    Deliberately returns more numbers than `game.draw_size`: the point is to
    surface "these look mildly more likely given recent history", not to
    claim which exact 6 will be drawn. Always report this alongside
    `evaluate_model_vs_baseline`'s verdict (see its docstring) so the reader
    knows whether the ranking has any statistical backing at all.
    """
    return predict_next_draw_scores(df, game, modeling, model=model).head(bundle_size)


def default_models() -> dict[str, object]:
    """A couple of reasonable, cheap baselines to compare -- gradient boosting
    for nonlinear structure, logistic regression as a simpler linear check.
    """
    return {
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
        "logistic_regression": LogisticRegression(max_iter=1000),
    }
